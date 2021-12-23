import os
import errno
import glob
import numpy as np
import pandas as pd
import sys
from timeit import default_timer as timer

import astropy.units as u
from astropy.convolution import convolve, convolve_fft
from astropy.io import fits

from photutils.background import Background2D, MedianBackground
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from photutils.segmentation import detect_threshold, detect_sources

import matplotlib.pyplot as plt

from photutils.centroids import centroid_2dg
from photutils.segmentation import SourceCatalog

from sgp_validation import validate_single
from flux_conserve_proj import projectDF
from radprof_ellipticity import radial_profile, calculate_ellipticity_fwhm
from sgp import sgp, calculate_flux
from richardson_lucy import rl

if __name__ == "__main__":
    DEFAULT_PARAMS = (1000, 1e-4, 0.4, 1.3, 1e-1, 1e1, 3, 0.5, 1)

    plt.rcParams.update({'font.size': 14})
    ##### Some user parameters ####
    save = True
    verbose = True
    plot = True

    # List to store parameters and results.
    compare_list = []

    # Median flux of all star stamps from all images.
    MEDIAN_FLUX = 61169.92578125  # Calculate using all stamps from the dataset.
    # Count variables for success/failure for flux criterion.
    success = 0
    failure = 0

    ########## Get the ideal star stamp ##########
    best_coord_file = "ccfbtf170075c.coo"

    best_sci_name = "ccfbtf170075c.fits"
    best_sci_img = fits.getdata(best_sci_name)
    best_psfs = [
                    "_PSF_BINS/psf_ccfbtf170075_1_1_img.fits", "_PSF_BINS/psf_ccfbtf170075_1_2_img.fits",
                    "_PSF_BINS/psf_ccfbtf170075_2_1_img.fits", "_PSF_BINS/psf_ccfbtf170075_2_2_img.fits"
                ]
    best_psf = np.mean([fits.getdata(psfname) for psfname in best_psfs], axis=0)  # Take absolute value of PSF before mean is taken?

    # Best star stamp
    arr = np.loadtxt(best_coord_file, skiprows=3, usecols=[0, 1])
    size = 25
    for x, y in arr:
        # Extract star stamp.
        cutout = Cutout2D(best_sci_img, (x, y), size, mode='partial', fill_value=sys.float_info.epsilon)
        best = cutout.data
        break
    
    best_center = centroid_2dg(best)
    best_profile = radial_profile(best, best_center)

    ### Get all files ###
    coord_files = sorted(glob.glob("cc*c.coo"))
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
    
    elliptical_images = elliptical_images[:20]

    elliptical_indices = [science_imgs.index(elliptical_images[i]) for i in range(len(elliptical_images))]
    elliptical_coord_files = sorted([coord_files[elliptical_indices[i]] for i in range(len(elliptical_indices))])
    elliptical_psfs = [mean_psfs[elliptical_indices[i]] for i in range(len(elliptical_indices))]

    elliptical_coord_files = elliptical_coord_files
    elliptical_psfs = elliptical_psfs

    figs = []
    original_radprofs = []
    c = 1
    for coord_list, science_img, psf in zip(elliptical_coord_files, elliptical_images, elliptical_psfs):
        image = fits.getdata(science_img)

        # Calculate background level.
        arr = np.loadtxt(coord_list, skiprows=3, usecols=[0, 1])

        if arr.size == 2:
            arr = np.expand_dims(arr, axis=0)
        for x, y in arr:
            # Extract star stamps.
            cutout = Cutout2D(image, (x, y), size, mode='partial', fill_value=sys.float_info.epsilon)
            # Background estimate.
            bkg = MedianBackground()
            bkg = bkg.calc_background(cutout.data)

            #SGP to get the reconstrcted image.

            #################
            ## VALIDATION ###
            #################

            params = validate_single(cutout.data, psf, bkg, x, y, search_type='coarse', flux_criteria=0, size=size)
            if params is None:  # If no optimal parameter that satisfied all conditions, then use default.
                print("\n\nNo best parameter found that matches all conditions. Falling to default params\n\n")
                max_projs, gamma, beta, alpha_min, alpha_max, alpha, M_alpha, tau, M = DEFAULT_PARAMS
            else:
                max_projs, gamma, beta, alpha_min, alpha_max, alpha, M_alpha, tau, M = params
                print(f"\n\nOptimal parameters: (max_projs, gamma, beta, alpha_min, alpha_max, alpha, M_alpha, tau, M) = {params}\n\n")

            ###########
            ### SGP ###
            ###########
            try:
                sgp_recon_img, sgp_rel_klds, sgp_rel_recon_errors, sgp_num_iters, sgp_extract_coord, sgp_execution_time, sgp_best_section = sgp(
                    cutout.data, psf, bkg, gamma=gamma, beta=beta, alpha_min=alpha_min, alpha_max=alpha_max,
                    alpha=alpha, M_alpha=M_alpha, tau=tau, M=M, proj_type=1, best_img=best_sci_img, best_coord=best_coord_file,
                    max_projs=max_projs, size=size, init_recon=2, stop_criterion=2, current_xy=(x, y), save=False,
                    filename=science_img, verbose=True, clip_X_upp_bound=False
                )
            except TypeError:
                continue

            # RL to get the reconstrcted image.
            rl_recon_img, rl_rel_recon_errors, rl_num_iters, rl_execution_time, rl_extract_coord = rl(
                cutout.data, psf, bkg, max_iter=1000, best_img=best_sci_img, best_coord=best_coord_file, current_xy=(x, y),
                flux_conserve=False, save=False, filename=science_img
            )

            try:
                flux_before = calculate_flux(cutout.data, size=size)
                sgp_flux_after = calculate_flux(sgp_recon_img.reshape(size, size), size=size)
                rl_flux_after = calculate_flux(rl_recon_img.reshape(size, size), size=size)
            except TypeError:
                continue

            # We calculate ellipticity and fwhm from the `radprof_ellipticity` module.
            before_ecc, before_fwhm = calculate_ellipticity_fwhm(cutout.data, use_moments=True)
            sgp_after_ecc, sgp_after_fwhm = calculate_ellipticity_fwhm(sgp_recon_img, use_moments=True)
            rl_after_ecc, rl_after_fwhm = calculate_ellipticity_fwhm(rl_recon_img, use_moments=True)

            before_center = centroid_2dg(cutout.data)

            sgp_after_center = centroid_2dg(sgp_recon_img)
            sgp_centroid_err = (before_center[0]-sgp_after_center[0], before_center[1]-sgp_after_center[1])
            sgp_l1_centroid_err = np.linalg.norm(before_center-sgp_after_center, ord=1)

            rl_after_center = centroid_2dg(rl_recon_img)
            rl_centroid_err = (before_center[0]-rl_after_center[0], before_center[1]-rl_after_center[1])
            rl_l1_centroid_err = np.linalg.norm(before_center-rl_after_center, ord=1)

            ######################
            ### Radial Profile ###
            ######################

            orig_center = centroid_2dg(cutout.data)
            original_radprof = radial_profile(cutout.data, orig_center)[:17]
            
            rl_recon_radprof = radial_profile(rl_recon_img, rl_after_center)
            sgp_recon_radprof = radial_profile(sgp_recon_img, sgp_after_center)

            # Update final needed parameters list.
            star_coord = (x, y)
            compare_list.append(
                [science_img, rl_num_iters, rl_execution_time, sgp_num_iters, sgp_execution_time, rl_after_ecc, rl_after_fwhm, sgp_after_ecc, sgp_after_fwhm, original_radprof, rl_recon_radprof, sgp_recon_radprof, rl_centroid_err, sgp_centroid_err, rl_l1_centroid_err, sgp_l1_centroid_err]
            )
            break

    if save:
        compare = np.array(compare_list)
        df = pd.DataFrame(compare)
        df.to_csv("compare_metrics.csv")