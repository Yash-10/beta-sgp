import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from math import floor

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
from astropy.wcs import WCS
from astropy.wcs.utils import skycoord_to_pixel
from photutils.background import MedianBackground
from photutils.centroids import centroid_2dg

from sklearn.preprocessing import KernelCenterer
from sep import extract, Background

from radprof_ellipticity import radial_profile
from sgp import sgp, calculate_flux, DEFAULT_PARAMS
from sgp_validation import validate_single

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
    if isinstance(bkg, Background):
        objects, segmap = extract(data, thresh=nsigma, err=bkg.globalrms, segmentation_map=True)
    else:
        objects, segmap = extract(data, thresh=nsigma, segmentation_map=True)
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
    _offset = 7

    return max(x_extent, y_extent) + _offset, mask, _offset  # Give pixel offset, if needed.

def get_which_section(x, y):
    # See https://math.stackexchange.com/questions/528501/how-to-determine-which-cell-in-a-grid-a-point-belongs-to
    N = 4
    return floor(x * N / 2048) + 1, floor(y * N / 2048) + 1

def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

if __name__ == "__main__":
    ra, dec = '16h41m23.55s', '+36d30m17.3s'  # Other options: ('16h41m32.53s', '+36d24m42.6s')
    coord = SkyCoord(ra=ra, dec=dec)
    imagename = 'cal_ccfbvc170120.fits'
    hdul = fits.open(imagename)
    w = WCS(hdul[0].header)
    image = hdul[0].data

    # Get PSF
    x, y = skycoord_to_pixel(coord, wcs=w)
    sec_x, sec_y = get_which_section(x, y)
    psfname = f"/home/yash/DIAPL/work/psf{imagename.split('_')[1].split('.')[0]}_{sec_x}_{sec_y}_img.fits"
    psf = fits.getdata(psfname)
    # Center the PSF matrix
    psf = KernelCenterer().fit_transform(psf)
    psf = np.abs(psf)

    check_star_cutout = Cutout2D(image, coord, wcs=w, size=45, copy=True)  # 40 is a safe size choice.
    # Estimate background on this check stamp
    d = np.ascontiguousarray(check_star_cutout.data)
    d = d.byteswap().newbyteorder()
    del check_star_cutout
    bkg = Background(d, bw=8, bh=8, fw=3, fh=3)  # 8 = 40 / 5
    bkg.subfrom(d)

    size, other_mask, offset = decide_star_cutout_size(d, bkg, nsigma=2.)
    print(f"Optimal star stamp size: {size}")
    stamp = Cutout2D(image, coord, wcs=w, size=size)

    best_img = 'cal_ccfbtf170075.fits'
    hdul_best = fits.open(best_img)
    w_best = WCS(hdul_best[0].header)
    best_cutout = Cutout2D(hdul_best[0].data, coord, wcs=w_best, size=size).data
    xbest, ybest = skycoord_to_pixel(coord, wcs=w_best)

    flux_before = calculate_flux(
        stamp.data, bkg.globalback, 0, size=size
    )

    circ_mask = create_circular_mask(size, size, center=None, radius=size/2)
    circ_mask = ~circ_mask
    circ_mask = circ_mask.astype(int, copy=False)
    dd = np.multiply(circ_mask, stamp.data)
    plt.imshow(dd)
    plt.show()

    # params = validate_single(stamp.data, psf, bkg.globalback, x, y, search_type='fine', flux_criteria=1, size=size, best_cutout=best_cutout, xbest=xbest, ybest=ybest)
    params = None
    if params is None:  # If no optimal parameter that satisfied all conditions, then use default.
        print("\n\nNo best parameter found that matches all conditions. Falling to default params\n\n")
        max_projs, gamma, beta, alpha_min, alpha_max, alpha, M_alpha, tau, M = DEFAULT_PARAMS
    else:
        max_projs, gamma, beta, alpha_min, alpha_max, alpha, M_alpha, tau, M = params
        print(f"\n\nOptimal parameters: (max_projs, gamma, beta, alpha_min, alpha_max, alpha, M_alpha, tau, M) = {params}\n\n")

    # optimal_params = (100, 0.001, 0.1, 1e-06, 1000000.0, 100.0, 3, 0.5, 1)  # Found using validation.
    # max_projs, gamma, beta, alpha_min, alpha_max, alpha, M_alpha, tau, M = optimal_params
    recon_img, rel_klds, rel_recon_errors, num_iters, extract_coord, execution_time, best_section = sgp(
        stamp.data, psf, bkg.globalback, gamma=gamma, beta=beta, alpha_min=alpha_min, alpha_max=alpha_max,
        alpha=alpha, M_alpha=M_alpha, tau=tau, M=M, proj_type=1, best_img=None, best_coord=(xbest, ybest), best_cutout=best_cutout,
        max_projs=max_projs, size=size, init_recon=2, stop_criterion=2, current_xy=(x, y), save=False,
        filename=image, verbose=True, clip_X_upp_bound=False, diapl=False, to_search_for_best_stamp=False, offset=offset
    )
    bkg_after = MedianBackground().calc_background(recon_img)
    flux_after = calculate_flux(
        recon_img, bkg_after, 0, size=size
    )

    reconx, recony = centroid_2dg(recon_img-bkg_after)
    rsize, rother_mask, roffset = decide_star_cutout_size(recon_img, bkg_after, nsigma=2.)
    print(f"Approx stamp size around restored star: {rsize}")
    recon_circ_mask = create_circular_mask(size, size, center=(reconx, recony), radius=(rsize-offset)/2)
    recon_circ_mask = recon_circ_mask.astype(int, copy=False)
    recon_dd = np.multiply(recon_circ_mask, recon_img)

    plt.imshow(dd)
    plt.title("dd")
    plt.show()
    plt.imshow(recon_dd)
    plt.show()
    plt.imshow(dd+recon_dd)
    plt.show()

    show = recon_dd+dd
    show[show==0.] = bkg.globalback

    # recon_img -= bkg_after
    # recon_img += bkg.globalback

    print(f"Flux before: {flux_before}")
    print(f"Flux after: {flux_after}")

    fig, ax = plt.subplots(1, 3)
    ax[0].imshow(stamp.data, origin='lower')
    ax[0].set_title(f"Original: {x, y} - Flux: {flux_before}")
    ax[1].imshow(best_cutout, origin='lower')
    ax[1].set_title(f"Best cutout: {xbest, ybest}")
    ax[2].imshow(recon_img, origin='lower')
    ax[2].set_title(f"Restored stamp - Flux: {flux_after}")
    plt.show()

    stamp.data[...] = show
    fits.writeto("restored_cal_ccfbvc170120.fits", image, header=hdul[0].header, overwrite=True)

    # df = pd.read_csv('variable_stars.txt', header=None, sep=' ')
    # df.columns = ['id', 'ra', 'dec']
    # for pair in df.iterrows():
    #     ra = pair[1]['ra']
    #     dec = pair[1]['dec']
    #     coord = SkyCoord(ra=ra, dec=dec)
    #     hdul = fits.open('cal_ccfbvc170120.fits')
    #     w = WCS(hdul[0].header)
    #     image = hdul[0].data
    #     size = 25
    #     try:
    #         stamp = Cutout2D(image, coord, size, wcs=w).data
    #     except:
    #         continue

    #     plt.imshow(stamp)

    #     x, y = skycoord_to_pixel(coord, wcs=w)
    #     plt.title(f'X: {x}, Y: {y}\nra: {ra}, dec: {dec}')
    #     plt.show()
