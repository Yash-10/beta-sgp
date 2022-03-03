import numpy as np
import glob
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.table import Table
from astropy.convolution import Gaussian2DKernel
from astropy.nddata import Cutout2D
from astropy.stats import gaussian_fwhm_to_sigma

from sep import extract, Background
from sgp import sgp, DEFAULT_PARAMS, calculate_flux

######## Parameters #######
nsigma = 2.
filter_fwhm = 1.5
filter_size = 11
npixels = 5
###########################

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

    """
    _, segmap = extract(data, thresh=nsigma, err=bkg.globalrms, segmentation_map=True)
    segmap = segmap.astype(float, copy=False)

    unique, counts = np.unique(segmap, return_counts=True)
    if len(counts) > 2:
        index = np.where(unique == 0.)
        _u = np.delete(unique, index)  # _u = unique[unique != 0.]
        _c = np.delete(counts, index)
        dominant_label = _u[_c.argmax()]
    else:
        dominant_label = 1.  # Since 0 is assigned for pixels with no source, 1 would have all source pixels.
    # TODO: Mask unwanted sources and return the mask also.
    i, j = np.where(segmap == dominant_label)
    y_extent = max(i) - min(i)
    x_extent = max(j) - min(j)
    return max(x_extent, y_extent) + 2  # Give 1 pixel offset.

# @jit
def apply_mask(array, data, size=35):
    """
    array: Star coordinate array of shape (nstars, 2).
    data: 2D Image array.

    """
    hsize = (size - 1) / 2
    x = array[:, 0]
    y = array[:, 1]
    # Don't select stars too close to the edge.
    mask = ((x > hsize) & (x < (data.shape[1] - 1 - hsize)) &
            (y > hsize) & (y < (data.shape[0] - 1 - hsize)))
    return array[mask]

f = open("candidate_defect_images.txt", "r")
defect_images = []
for line in f.readlines():
    defect_images.append(line.rstrip())

# Calculate mean PSF (mean of PSFs over a single frame) as a prior for the SGP method.
psfs = sorted(glob.glob("/home/yash/DIAPL/work/_PSF_BINS/psf_cc*.fits"))
mean_psfs = []
for i in range(0, len(psfs), 4):
    data_psfs = [fits.getdata(psfs[n]) for n in range(i, i+4)]
    mean_psf = np.mean(data_psfs, axis=0)
    mean_psfs.append(mean_psf)

for image, psf in zip(sorted(glob.glob('cc*.fits')), mean_psfs):
    if image not in defect_images:
        continue

    data = fits.getdata(image)
    coord_list = '.'.join((image.split('.')[0], 'coo'))

    stars = np.loadtxt(coord_list, skiprows=3, usecols=[0, 1])
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

        size = decide_star_cutout_size(d, bkg, nsigma=2.)
        # TODO: The stamp could still have multiple detections at this point. Do smth to mask them.
        cutout = Cutout2D(data, position=(xc, yc), size=size)
        lala = Cutout2D(data, position=(xc, yc), size=40)

        max_projs, gamma, beta, alpha_min, alpha_max, alpha, M_alpha, tau, M = DEFAULT_PARAMS
        recon_img, rel_klds, rel_recon_errors, num_iters, extract_coord, execution_time, best_section = sgp(
            cutout.data, psf, bkg.globalback, gamma=gamma, beta=beta, alpha_min=alpha_min, alpha_max=alpha_max,
            alpha=alpha, M_alpha=M_alpha, tau=tau, M=M, proj_type=1, best_img=None, best_coord=None,
            max_projs=max_projs, size=size, init_recon=2, stop_criterion=1, current_xy=(xc, yc), save=False,
            filename=image, verbose=True, clip_X_upp_bound=False, diapl=False
        )

        # TODO: Check flux
        flux_before = calculate_flux(cutout.data, size=size, nsigma=2.)
        flux_after = calculate_flux(recon_img, size=size, nsigma=2.)

        print(f'\n\nFlux before: {flux_before}, Flux after: {flux_after}\n\n')

        # TODO: Can I add a varying bkg map still? - would be better...
        # cutout.data[...] = recon_img + bkg.globalback  # Trick to modify `cutout.data` inplace.

        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(cutout.data, origin='lower')
        ax[1].imshow(recon_img, origin='lower')
        ax[2].imshow(lala.data, origin='lower')
        plt.show()

    fits.writeto('ccc.fits', data)
    break
