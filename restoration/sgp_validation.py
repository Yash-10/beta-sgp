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

def setup():
    """
    Some setup required for validation.

    """
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

    with open("trail_images.txt", "r") as f:
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

def validate_single(data, psf, bkg, x, y, search_type='coarse', flux_criteria=0, size=25):
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
    from sgp import sgp, calculate_flux

    best_sci_img, best_coord_file, best_psf = setup()  # Hack.

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
            "max_projs": [100, 500, 1000, 2000],
            "gamma": [1e-4, 1e-1],
            "beta": [0.4],
            "alpha_min": [1e-2, 1e-4, 1e-6],
            "alpha_max": [1e2, 1e4, 1e6],
            "alpha": [1e-2, 1, 1e2, 1e4],
            "M_alpha": [3],
            "tau": [0.5],
            "M": [1, 3]
        }
    elif search_type == 'fine':
        # 1. Finer search (computationally intensive)
        param_grid = {
            "max_projs": [100, 500, 700, 900, 1000, 1500, 3000, 5000, 10000, 50000],
            "gamma": [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
            "beta": [0.4],
            "alpha_min": [1e-6, 1e-4, 1e-2, 1e-1],
            "alpha_max": [1e6, 1e4, 1e2, 1e1],
            "alpha": [1e-2, 1e-1, 1e0, 1.3, 1e1, 1e2, 1e3],
            "M_alpha": [3],
            "tau": [0.5],
            "M": [1, 3, 7]
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
                data, psf, bkg, init_recon=2, proj_type=1, stop_criterion=2, MAXIT=150,
                gamma=sgp_hyperparameters["gamma"], beta=sgp_hyperparameters["beta"], alpha=sgp_hyperparameters["alpha"],
                alpha_min=sgp_hyperparameters["alpha_min"], alpha_max=sgp_hyperparameters["alpha_max"],
                M_alpha=sgp_hyperparameters["M_alpha"], tau=sgp_hyperparameters["tau"], M=sgp_hyperparameters["M"],
                max_projs=sgp_hyperparameters["max_projs"], size=size, best_img=best_sci_img, best_coord=best_coord_file, current_xy=(x, y),
                verbose=True, clip_X_upp_bound=False, save=False
            )
        except TypeError:
           print("Error in validation")
           continue

        flux_before = calculate_flux(data, size=size)
        try:
            flux_after = calculate_flux(recon_img, size=size)
        except TypeError:
            continue

        flux_diff = abs(flux_after - flux_before)

        if flux_criteria == 1:
            flux_thresh = 0.1 * flux_before
        else:
            flux_thresh = 0.01 * MEDIAN_FLUX

        if flux_after < flux_before + flux_thresh and flux_after > flux_before - flux_thresh:
            if flux_diff < min_flux_diff:
                optimal_projs = max_projs
                min_flux_diff = flux_diff
            else:
                optimal_projs = 500  # Some arbitrary default value.

            recon_ecc, recon_fwhm = calculate_ellipticity_fwhm(recon_img, use_moments=True)
            data_ecc, data_fwhm = calculate_ellipticity_fwhm(data, use_moments=True)
            least_recon_error_current = rel_recon_errors[-2]  # Second-last error would be the least.
            if recon_ecc < data_ecc and least_recon_error_current < least_err:
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