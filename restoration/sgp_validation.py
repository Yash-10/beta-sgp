import itertools
import logging
import glob
import random

import numpy as np

from photutils.background import MedianBackground
from astropy.nddata import Cutout2D
from astropy.io import fits

from radprof_ellipticity import calculate_ellipticity_fwhm

logging.basicConfig(filename='sgp_val.log', level=logging.INFO,
    format='%(asctime)s:%(funcName)s:%(message)s'
)

def setup(mode='test'):
    """
    Some setup required for validation.
    mode: str
        'test' if one aims to test SGP. For final restoration of the whole GC image, use 'final'.

    """
    if mode == 'test':
        coord_files = sorted(glob.glob("cc*c.coo"))
        # Note: There are some images with negative pixel values.
        science_imgs = sorted(glob.glob("cc*[!m]c.fits"))
        psfs = sorted(glob.glob("_PSF_BINS/psf_cc*.fits"))

        # Calculate mean PSF (mean of PSFs over a single frame) as a prior for the SGP method.
        mean_psfs = []
        for i in range(0, len(psfs), 4):
            data_psfs = [fits.getdata(psfs[n]) for n in range(i, i+4)]
            mean_psf = np.mean(data_psfs, axis=0)
            mean_psfs.append(mean_psf)

        with open("test_images.txt", "r") as f:
            elliptical_images = sorted([line.strip() for line in f.readlines()])

        elliptical_indices = [science_imgs.index(elliptical_images[i]) for i in range(len(elliptical_images))]
        elliptical_coord_files = sorted([coord_files[elliptical_indices[i]] for i in range(len(elliptical_indices))])
        elliptical_psfs = [mean_psfs[elliptical_indices[i]] for i in range(len(elliptical_indices))]

        # Get best image.
        best_coord_file = "ccfbtf170075c.coo"

        best_sci_img = fits.getdata("ccfbtf170075c.fits")
        best_psfs = [
                        "_PSF_BINS/psf_ccfbtf170075_1_1_img.fits", "_PSF_BINS/psf_ccfbtf170075_1_2_img.fits",
                        "_PSF_BINS/psf_ccfbtf170075_2_1_img.fits", "_PSF_BINS/psf_ccfbtf170075_2_2_img.fits"
                    ]
        best_psf = np.mean([fits.getdata(psfname) for psfname in best_psfs], axis=0)

        return best_sci_img, best_coord_file, best_psf
    elif mode == 'final':
        base_dir = 'working'
        best_sci_name = f'{base_dir}/cwcs_ccfbtf170075.fits'
        best_sci_img = fits.getdata(best_sci_name)
        best_coord_file = 'best_coord_photutils.csv'

        best_psfs = [
                        "/home/yash/DIAPL/work/_PSF_BINS/psf_ccfbtf170075_1_1_img.fits", "/home/yash/DIAPL/work/_PSF_BINS/psf_ccfbtf170075_1_2_img.fits",
                        "/home/yash/DIAPL/work/_PSF_BINS/psf_ccfbtf170075_2_1_img.fits", "/home/yash/DIAPL/work/_PSF_BINS/psf_ccfbtf170075_2_2_img.fits"
                    ]
        best_psf = np.mean([fits.getdata(psfname) for psfname in best_psfs], axis=0)

        return best_sci_img, best_coord_file, best_psf

def validate_single(data, psf, bkg, x, y, search_type='coarse', flux_criteria=1, size=25, mode='test', do_setup=False, best_cutout=None, xbest=None, ybest=None):
    """
    search_type: Whether to do an extensive grid search over SGP parameters or a coarser search,
    else use 'fine', defaults to 'coarse'.
        For better performance (in terms of speed), keep the 'coarse' option.
    
    flux_criteria: int
        0: Use 1% of median flux of the whole dataset, used in SGP.
        1: Use 10% of only the current image flux, used in `physical_validation.py`.

    In this module, it is a helper function for `validate`. However, it can be used in itself.

    It runs SGP and calculates the optimal set of parameters, based on the following metrics:
        1. Flux conservation,
        2. Relative Reconstruction Error (RRE),
        3. Ellipticity.

    First, flux condition (1) must be satisfied. If yes, only then (2) and (3) are checked.

    """
    MEDIAN_FLUX = 61169.92578125
    from sgp import sgp, calculate_flux, calculate_bkg

    if do_setup:
        best_sci_img, best_coord_file, best_psf = setup(mode=mode)  # Hack.

    optimal_params_set = []
    optimal_projs = -1
    min_flux_diff = np.Inf
    least_err = np.Inf
    best_params = None

    #######################################
    ### GRID search over SGP parameters ###
    #######################################
    if search_type == 'coarse':
        param_grid = {
            "max_projs": [100, 500],
            "gamma": [1e-4],
            "beta": [0.4],
            "alpha_min": [1e-2],
            "alpha_max": [1e2, 1e4],
            "alpha": [1e-2, 1, 1e2],
            "M_alpha": [3],
            "tau": [0.5],
            "M": [1, 3]
        }
    elif search_type == 'fine':
        # 1. Finer search (computationally intensive)
        param_grid = {
            "max_projs": [100, 500, 700, 900, 1500, 3000],
            "gamma": [1e-3, 1e-5, 1e-7],
            "beta": [0.1, 0.5],
            "alpha_min": [1e-6, 1e-4, 1e-2, 1e-1],
            "alpha_max": [1e6, 1e4, 1e2, 1e1],
            "alpha": [1e-6, 1e-4, 1e-2, 1e0, 1e2, 1e4, 1e6],
            "M_alpha": [3],
            "tau": [0.5],
            "M": [1, 3]
        }

    keys, values = zip(*param_grid.items())

    i = 0
    for v in itertools.product(*values):
        sgp_hyperparameters = dict(zip(keys, v))
        max_projs, gamma, beta, alpha_min, alpha_max, alpha, M_alpha, tau, M = sgp_hyperparameters.values()
        # Safe to keep in a try-except block since some parameters could yield very bad results, and we
        # do not want to terminate due to one bad choice during validation.
        try:
            recon_img, rel_klds, rel_recon_errors, num_iters, extract_coord, execution_time, best_section = sgp(
                data, psf, bkg, gamma=gamma, beta=beta, alpha_min=alpha_min, alpha_max=alpha_max,
                alpha=alpha, M_alpha=M_alpha, tau=tau, M=M, proj_type=1, best_img=None, best_coord=(xbest, ybest), best_cutout=best_cutout,
                max_projs=max_projs, size=size, init_recon=2, stop_criterion=2, current_xy=(x, y), save=False,
                filename=None, verbose=True, clip_X_upp_bound=False, diapl=False, to_search_for_best_stamp=False, offset=0
            )
        except TypeError:
           print("Error in validation")
           continue

        bkg_before = calculate_bkg(data)
        flux_before = calculate_flux(
            data, bkg_before, None, size=size
        )
        try:
            bkg_after = calculate_bkg(recon_img)
            flux_after = calculate_flux(
                recon_img, bkg_after, None, size=size
            )
        except TypeError:
            continue

        flux_diff = abs(flux_after - flux_before)

        if flux_criteria == 1:
            flux_thresh = 0.01 * flux_before
        else:
            flux_thresh = 0.01 * MEDIAN_FLUX

        if flux_after < flux_before + flux_thresh and flux_before - flux_thresh < flux_after:
            if flux_diff < min_flux_diff:
                optimal_projs = max_projs
                min_flux_diff = flux_diff
            else:
                optimal_projs = 500  # Some default value.

            recon_ecc, recon_fwhm = calculate_ellipticity_fwhm(recon_img, use_moments=True)
            data_ecc, data_fwhm = calculate_ellipticity_fwhm(data, use_moments=True)
            least_recon_error_current = rel_recon_errors[-2]  # Second-last error would be the least.
            # Reason for recon_fwhm > 3 is because we do not want stars to be reconstructed as very small, i.e. point sources.
            if ((recon_fwhm.value >= 3) and (recon_fwhm.value < data_fwhm.value and recon_ecc.value < data_ecc.value)) and (least_recon_error_current < least_err):
                best_params = (optimal_projs, gamma, beta, alpha_min, alpha_max, alpha, M_alpha, tau, M)
                least_ellipticity = recon_ecc
                least_err = least_recon_error_current
        else:
            continue

    return best_params


def validate(elliptical_coord_files, elliptical_images, elliptical_psfs):
    for coord_list, science_img, psf in zip(elliptical_coord_files, elliptical_images, elliptical_psfs):
        image = fits.getdata(science_img)

        # Calculate background level.
        arr = np.loadtxt(coord_list, skiprows=3, usecols=[0, 1])

        # Select random `num_val` stars for validation.
        num_rows = arr.shape[0]
        rand_indices = np.random.choice(num_rows, size=10, replace=False)
        select_arr = arr[rand_indices, :]

        size = 25

        for x, y in select_arr:
            # Extract star stamps.
            cutout = Cutout2D(image, (x, y), size)
            # Background estimate.
            bkg = MedianBackground()
            bkg = bkg.calc_background(cutout.data)

            optimal_params = validate_single(cutout.data, psf, bkg, x, y)
            logging.info(f"========\nOptimal parameters for {science_img} at coordinates x = {x} and y = {y}:\n {optimal_params}")


if __name__ == "__main__":
    # Grid search
    gammas = [1, 1e-2, 1e-4, 1e-6]
    betas = [0.1, 0.4, 0.6, 0.8]
    alpha_mins = [1e-8, 1e-6, 1e-4, 1e-2]
    alpha_maxs = [1, 1e2, 1e4, 1e6]
    alphas = [1e-2, 1e-1, 1, 1e2]
    M_alphas = [1, 3, 5, 7]
    taus = [0.1, 0.3, 0.5, 0.7]
    Ms = [1, 3, 5, 7]

    # List of params
    params = [gammas, betas, alpha_mins, alpha_maxs, alphas, M_alphas, taus, Ms]

    num_val = 20  # Number of stars to select from each image for validation.

    coord_files = sorted(glob.glob("cc*c.coo"))
    # Note: There are some images with negative pixel values.
    science_imgs = sorted(glob.glob("cc*[!m]c.fits"))
    psfs = sorted(glob.glob("_PSF_BINS/psf_cc*.fits"))

    # Calculate mean PSF (mean of PSFs over a single frame) as a prior for the SGP method.
    mean_psfs = []
    for i in range(0, len(psfs), 4):
        data_psfs = [fits.getdata(psfs[n]) for n in range(i, i+4)]
        mean_psf = np.mean(data_psfs, axis=0)
        mean_psfs.append(mean_psf)

    with open("test_images.txt", "r") as f:
        elliptical_images = sorted([line.strip() for line in f.readlines()])

    elliptical_indices = [science_imgs.index(elliptical_images[i]) for i in range(len(elliptical_images))]
    elliptical_coord_files = sorted([coord_files[elliptical_indices[i]] for i in range(len(elliptical_indices))])
    elliptical_psfs = [mean_psfs[elliptical_indices[i]] for i in range(len(elliptical_indices))]

    # Get best image.
    best_coord_file = "ccfbtf170075c.coo"

    best_sci_img = fits.getdata("ccfbtf170075c.fits")
    best_psfs = [
                    "_PSF_BINS/psf_ccfbtf170075_1_1_img.fits", "_PSF_BINS/psf_ccfbtf170075_1_2_img.fits",
                    "_PSF_BINS/psf_ccfbtf170075_2_1_img.fits", "_PSF_BINS/psf_ccfbtf170075_2_2_img.fits"
                ]
    best_psf = np.mean([fits.getdata(psfname) for psfname in best_psfs], axis=0)

    validate(elliptical_coord_files, elliptical_images, elliptical_psfs)