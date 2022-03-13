import numpy as np
import glob
import matplotlib.pyplot as plt

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
cachedir = 'gc_cache/'
memory = Memory(cachedir, verbose=0)

def decide_star_cutout_size(data, bkg, nsigma=2.):
    """
    data must be background-subtracted, But still need to pass bkg object.

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
    _offset = 2

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
            # TODO: Note, using such subdivisions for mosaicking would not make sense.
            _base_name = '_'.join((image.split('.')[0], str(ix), str(iy)))
            cut_image = '.'.join((_base_name, 'fits'))
            data = fits.getdata(cut_image)
            cut_coord_list = '.'.join((_base_name, 'coo'))
            print(cut_coord_list)

        stars = np.loadtxt(cut_coord_list, skiprows=3, usecols=[0, 1])
        stars = apply_mask(stars, data)  # Exclude stars very close to the edge.

        # # Generate a astropy table of star coordinates. # TODOL Remove below three lines, not needed.
        # t = Table()
        # t['x'] = stars[:, 0]
        # t['y'] = stars[:, 1]

        for xc, yc in stars:
            check_star_cutout = Cutout2D(data, position=(xc, yc), size=40)  # 40 is a safe size choice.
            # Estimate background on this check stamp
            d = np.ascontiguousarray(check_star_cutout.data)
            d = d.byteswap().newbyteorder()
            del check_star_cutout
            bkg = Background(d, bw=8, bh=8, fw=3, fh=3)  # 8 = 40 / 5
            bkg.subfrom(d)

            try:  # In very rare cases, no source is detected where finding size becomes difficult. If this happens, we simply move on to the next star.
                size, other_mask, offset = decide_star_cutout_size(d, bkg, nsigma=2.)
            except ValueError:
                continue

            # TODO: Update: Not really. Remove this todo. The stamp could still have multiple detections at this point. Do smth to mask them.
            cutout = Cutout2D(data, position=(xc, yc), size=size)

            ground_truth_star_stamp_name = '_'.join((_best_base_name, str(ix), str(iy))) + '.fits'
            ground_truth_star_stamp = fits.getdata(ground_truth_star_stamp_name)
            best_cut_image = Cutout2D(ground_truth_star_stamp, (xc, yc), size=size).data

            flux_before = calculate_flux(
                cutout.data-bkg.globalback, offset, mask=other_mask, size=size
            )

            if np.any(cutout.data > 1e10):
                continue

            max_projs, gamma, beta, alpha_min, alpha_max, alpha, M_alpha, tau, M = DEFAULT_PARAMS
            try:
                recon_img, rel_klds, rel_recon_errors, num_iters, extract_coord, execution_time, best_section, flux_before = sgp(
                    cutout.data, psf, bkg.globalback, gamma=gamma, beta=beta, alpha_min=alpha_min, alpha_max=alpha_max,
                    alpha=alpha, M_alpha=M_alpha, tau=tau, M=M, proj_type=1, best_img=None, best_coord=(xc, yc), best_cutout=best_cut_image,
                    max_projs=max_projs, size=size, init_recon=2, stop_criterion=2, current_xy=(xc, yc), save=False,
                    filename=image, verbose=True, clip_X_upp_bound=False, diapl=False, to_search_for_best_stamp=False, flux_stamp=flux_before
                )
            except ValueError:
                continue

            # fig, ax = plt.subplots(1, 2)
            # ax[0].imshow(cutout.data)
            # ax[1].imshow(recon_img)
            # plt.show()

            ##recon_bkg = Background(recon_img, bw=8, bh=8, fw=3, fh=3)  # 8 = 40 / 5
            ##recon_bkg.subfrom(recon_img)
            ##_, recon_other_mask = decide_star_cutout_size(recon_img, recon_bkg, nsigma=2.)
            ##flux_after = calculate_flux(recon_img, mask=None, size=size, nsigma=2.)

            # Calculating restored star's flux cannot be calculated as of now since we do not have a good bkg estimate for the restored star
            print(f'Flux before: {flux_before}')

            # TODO: Can I add a varying bkg map still? - would be better...
            cutout.data[...] = recon_img + bkg.globalback  # Trick to modify `cutout.data` inplace.

        restored_cutout_filename = '_'.join(('restored', cut_image))
        fits.writeto(restored_cutout_filename, data)

def worker(args_batch):
    image, ix, iy = args_batch
    do_on_ys(image, ix, iy)

if __name__ == "__main__":
    with open("candidate_defect_images.txt", "r") as f:
        defect_images = f.read().splitlines()

    _best_base_name = 'ccfbtf170075' + 'r'
    best_cut_image = '.'.join((_best_base_name, 'fits'))
    best_cut_coord_list = '.'.join((_best_base_name, 'coo'))

    # Calculate mean PSF (mean of PSFs over a single frame) as a prior for the SGP method.
    psfs = sorted(glob.glob("/home/yash/DIAPL/work/_PSF_BINS/psf_cc*.fits"))
    mean_psfs = []
    for i in range(0, len(psfs), 4):
        data_psfs = [fits.getdata(psfs[n]) for n in range(i, i+4)]
        mean_psf = np.mean(data_psfs, axis=0)
        mean_psfs.append(mean_psf)

    start = timer()
    for image, psf in zip(sorted(defect_images), mean_psfs):
        if image not in defect_images:
            continue
        for ix in range(1, 5):
            arguments = [[image, ix, 1], [image, ix, 2], [image, ix, 3], [image, ix, 4]]
            with Parallel(n_jobs=4, verbose=10) as parallel:
                funcs = repeat(do_on_ys, 4)  # functools.partial seems not pickle-able
                args_batches = np.array_split(arguments, 4, axis=0)
                print(*args_batches)
                jobs = args_batches
                parallel(delayed(worker)(*job) for job in jobs)
            sleep(1)

        # Mosaick
        if mosaick:
            arr, footprint = reproject_and_coadd(
                [fits.open(f)[0] for f in glob.glob('restored' + image.split('.')[0] + 'r' + '_*.fits')],
                output_projection=fits.open('calibrated_ccfbxi300024.fits')[0].header, reproject_function=reproject_exact
            )
            # Save restored GC image
            fits.writeto(f'mosaicked_{image}', arr)
        break
    end = timer() - start
    print(f"Time taken = {end}s")
