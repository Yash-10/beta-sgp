import numpy as np
import sys
import cv2
import glob
import matplotlib.pyplot as plt

# import pandas as pd

from itertools import repeat
from timeit import default_timer as timer
from joblib import Parallel, delayed

from astropy.io import fits
from astropy.table import Table
from astropy.convolution import Gaussian2DKernel
from astropy.nddata import Cutout2D
from astropy.stats import gaussian_fwhm_to_sigma

from reproject.mosaicking import reproject_and_coadd
from reproject import reproject_exact

from sep import extract, Background
from sgp import sgp, DEFAULT_PARAMS, calculate_flux

######## Parameters #######
nsigma = 2.
npixels = 5
mosaick = False  # Whether to mosaick the restored subdivisions.
###########################

# Joblib setup
from joblib import Memory
cachedir = './gc_cache'
memory = Memory(cachedir, verbose=0)

def decide_star_cutout_size(data, bkg, subdiv_number, nsigma=2.):
    """
    data must be background-subtracted, But still need to pass bkg object.
    subdiv_number: tuple of (x, y)
        - Out of the 16 subdivisions, in which subdivision is the current star in?
        - For subdivisions closer to center, offset must be very low - 1 or even 0.
        - eg: (2, 2).

    Notes
    -----
    - We do not need a pitch-perfect deblending procedure since it is only used to detect
    any presence of multiple sources. Ultimately, the unwanted source would be masked before
    the main source is restored.
    - This source extractor also deblends sources if there more than one.
    - This is in well agreement with `make_source_mask` from photutils.
    - Moreover, source extractor is upto ~5-10 times faster than photutils since it is written in C.

    Returns
    -------
    (a) Optimal star stamp size, (b) Mask denoting background and other surrounding source detections, if any.

    """
    objects, segmap = extract(data, thresh=nsigma, err=bkg.globalrms, segmentation_map=True)
    segmap = segmap.astype(float, copy=False)

    unique, counts = np.unique(segmap, return_counts=True)
    if len(counts) > 2:
        index = np.where(unique == 0.)
        _u = np.delete(unique, index)  # _u = unique[unique != 0.]
        _c = np.delete(counts, index)
        dominant_label = _u[_c.argmax()]

        # Create mask: True for (i) surrounding sources (if any), and (ii) background. False for concerned source.
        mask = np.copy(segmap)
        mask = mask.astype(bool, copy=False)
        mask[(mask != 0.) & (mask != dominant_label)] = True
        mask[(mask == 0.) | (mask == dominant_label)] = False
    else:
        dominant_label = 1.  # Since 0 is assigned for pixels with no source, 1 would have all source pixels.
        mask = None

    i, j = np.where(segmap == dominant_label)
    y_extent = max(i) - min(i)
    x_extent = max(j) - min(j)
    approx_size = max(x_extent, y_extent)
    if subdiv_number[0] in [2, 3] and subdiv_number[1] in [2, 3]:  # Towards the center of cluster.
        _offset = 1
    else:
        _offset = 3

    # # Calculate flux
    # # data must be background-subtracted.
    # # TODO: Ensure flux only for main source if there are multiple sources.
    # print("objects")
    # print(objects)
    # flux, fluxerr, flag = sum_circle(
    #     data, objects['x'], objects['y'], approx_size, segmap=segmap, err=bkg.globalrms, gain=1.22
    # )

    return max(x_extent, y_extent) + _offset, mask, _offset  # Give 2 pixel offset.

def apply_mask(array, data, size=40):
    """
    array: Star coordinate array of shape (nstars, 2).
    data: 2D Image array.

    """
    hsize = (size - 1) / 2
    if array.ndim == 1:
        x = array[0]
        y = array[1]
    else:
        x = array[:, 0]
        y = array[:, 1]
    # Don't select stars too close to the edge.
    mask = ((x > hsize) & (x < (data.shape[1] - 1 - hsize)) &
            (y > hsize) & (y < (data.shape[0] - 1 - hsize)))
    return array[mask]

@memory.cache
def do_on_ys(image, ix, iy):
    for iy in range(1, 5):
        _base_name = '_'.join((image.split('.')[0] + 'r', str(ix), str(iy)))
        cut_image = '.'.join((_base_name, 'fits'))
        final_name = '_'.join((_base_name, str(ix), str(iy)))   # Need this just to match naming convention.
        cut_coord_list = '.'.join((final_name, 'coo'))

        try:  # For some reason, some subdivisions cannot be extracted (needs investigation).
            data = fits.getdata(cut_image)
        except:  # If "r" suffixed images cannot be extracted, used the non-resampled version.
            mosaick = False  # Overwrite mosaick value since if except, then subdivisions are not actually resampled.
            _base_name = '_'.join((image.split('.')[0], str(ix), str(iy)))
            cut_image = '.'.join((_base_name, 'fits'))
            data = fits.getdata(cut_image)
            cut_coord_list = '.'.join((_base_name, 'coo'))

        stars = np.loadtxt(cut_coord_list, skiprows=3, usecols=[0, 1])
        stars = apply_mask(stars, data)  # Exclude stars very close to the edge.

        # # Generate a astropy table of star coordinates. # TODOL Remove below three lines, not needed.
        # t = Table()
        # t['x'] = stars[:, 0]
        # t['y'] = stars[:, 1]

        for xc, yc in stars:
            check_star_cutout = Cutout2D(data, position=(xc, yc), size=40, copy=True, mode='partial', fill_value=sys.float_info.epsilon)  # 40 is a safe size choice.
            # Estimate background on this check stamp
            d = np.ascontiguousarray(check_star_cutout.data)
            d = d.byteswap().newbyteorder()
            del check_star_cutout
            bkg = Background(d, bw=8, bh=8, fw=3, fh=3)  # 8 = 40 / 5
            bkg.subfrom(d)

            try:  # In very rare cases, no source is detected where finding size becomes difficult. If this happens, we simply move on to the next star.
                size, other_mask, offset = decide_star_cutout_size(d, bkg, (ix, iy), nsigma=2.)
            except ValueError:
                continue

            # Interpolate to match shape of recon_img.
            interpolated_bkg = cv2.resize(bkg.back(), dsize=(size, size), interpolation=cv2.INTER_CUBIC)

            cutout = Cutout2D(data, position=(xc, yc), size=size, mode='partial', fill_value=bkg.globalback)

            ground_truth_star_stamp_name = '_'.join((_best_base_name, str(ix), str(iy))) + '.fits'
            ground_truth_star_stamp = fits.getdata(ground_truth_star_stamp_name)
            best_cut_image = Cutout2D(ground_truth_star_stamp, (xc, yc), size=size).data

            flux_before = calculate_flux(
                cutout.data, bkg.globalback, offset, size=size
            )

            if np.any(cutout.data > 1e10):
                continue

            max_projs, gamma, beta, alpha_min, alpha_max, alpha, M_alpha, tau, M = DEFAULT_PARAMS
            try:
                recon_img, rel_klds, rel_recon_errors, num_iters, extract_coord, execution_time, best_section = sgp(
                    cutout.data, psf, bkg.globalback, gamma=gamma, beta=beta, alpha_min=alpha_min, alpha_max=alpha_max,
                    alpha=alpha, M_alpha=M_alpha, tau=tau, M=M, proj_type=1, best_img=None, best_coord=(xc, yc), best_cutout=best_cut_image,
                    max_projs=max_projs, size=size, init_recon=2, stop_criterion=2, current_xy=(xc, yc), save=False,
                    filename=image, verbose=True, clip_X_upp_bound=False, diapl=False, to_search_for_best_stamp=False, offset=offset
                )
            except ValueError:
                continue

            recon_bkg = Background(recon_img, bw=8, bh=8, fw=3, fh=3)  # 8 = 40 / 5
            recon_bkg.subfrom(recon_img)
            recon_img += interpolated_bkg

            flux_after = calculate_flux(recon_img, interpolated_bkg, offset, size=size)

            print(f'Flux before: {flux_before}')
            print(f'Flux after: {flux_after}')

            cutout.data[...] = recon_img  # Trick to modify `cutout.data` inplace.

        restored_cutout_filename = '_'.join(('restored', cut_image))
        fits.writeto(restored_cutout_filename, data, overwrite=True)

if __name__ == "__main__":
    with open("candidate_defect_images.txt", "r") as f:
        defect_images = f.read().splitlines()

    _best_base_name = 'ccfbtf170075' + 'r'
    best_cut_image = '.'.join((_best_base_name, 'fits'))
    best_cut_coord_list = '.'.join((_best_base_name, 'coo'))
    
    fluxes = []
    count = 0

    start = timer()
    for image in sorted(defect_images)[15:30]:
        psfs = glob.glob(f"_PSF_BINS/psf_{image.split('.')[0]}_*_*.fits")
        psf = np.mean([fits.getdata(p) for p in psfs], axis=0)

        if image not in defect_images:
            continue
        for ix in range(1, 5):
            Parallel(n_jobs=2, verbose=10)(delayed(do_on_ys)(image, ix, iy) for iy in range(1, 5))

        # Mosaick
        if mosaick:
            arr, footprint = reproject_and_coadd(
                [fits.open(f)[0] for f in glob.glob('restored' + image.split('.')[0] + 'r' + '_*.fits')],
                output_projection=fits.open('calibrated_ccfbxi300024.fits')[0].header, reproject_function=reproject_exact
            )
            # Save restored GC image
            fits.writeto(f'mosaicked_{image}', arr)

    end = timer() - start
    print(f"Time taken = {end}s")
