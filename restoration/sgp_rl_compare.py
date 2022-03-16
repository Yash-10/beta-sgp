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
from sep import Background
from sgp import sgp, calculate_flux, apply_mask, DEFAULT_PARAMS
from rl import rl

if __name__ == "__main__":
    with open("candidate_defect_images.txt", "r") as f:
        defect_images = f.read().splitlines()

    plot = False
    verbose = True
    save = True
    compare_list = []
    success = 0
    failure = 0
    size = 25
    offset = None
    original_radprofs = []
    count = 0
    MEDIAN_FLUX = 61169.92578125  # Calculate using all stamps from the dataset.

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
        for ix, iy in zip([1, 2], [2, 1]):
            _base_name = '_'.join((image.split('.')[0] + 'r', str(ix), str(iy)))
            cut_image = '.'.join((_base_name, 'fits'))
            final_name = '_'.join((_base_name, str(ix), str(iy)))   # Need this just to match naming convention.
            cut_coord_list = '.'.join((final_name, 'coo'))

            try:  # For some reason, some subdivisions cannot be extracted (needs investigation).
                data = fits.getdata(cut_image)
            except:  # If "r" suffixed images cannot be extracted, used the non-resampled version.
                _base_name = '_'.join((image.split('.')[0], str(ix), str(iy)))
                cut_image = '.'.join((_base_name, 'fits'))
                data = fits.getdata(cut_image)
                cut_coord_list = '.'.join((_base_name, 'coo'))

            try:
                stars = np.loadtxt(cut_coord_list, skiprows=3, usecols=[0, 1])
            except OSError:
                continue
            stars = apply_mask(stars, data, size=25)  # Exclude stars very close to the edge.

            if stars.size == 2:
                stars = np.expand_dims(stars, axis=0)
            for xc, yc in stars:
                check_star_cutout = Cutout2D(data, position=(xc, yc), size=40)  # 40 is a safe size choice.
                # Estimate background on this check stamp
                d = np.ascontiguousarray(check_star_cutout.data)
                d = d.byteswap().newbyteorder()
                del check_star_cutout
                bkg = Background(d, bw=8, bh=8, fw=3, fh=3)  # 8 = 40 / 5
                bkg.subfrom(d)

                cutout = Cutout2D(data, position=(xc, yc), size=size)

                ground_truth_star_stamp_name = '_'.join((_best_base_name, str(ix), str(iy))) + '.fits'
                ground_truth_star_stamp = fits.getdata(ground_truth_star_stamp_name)
                best_cut_image = Cutout2D(ground_truth_star_stamp, (xc, yc), size=size).data

                flux_before = calculate_flux(
                    cutout.data, bkg.globalback, offset, size=size
                )

                if np.any(cutout.data > 1e10):
                    continue

                ###########
                ### SGP ###
                ###########
                max_projs, gamma, beta, alpha_min, alpha_max, alpha, M_alpha, tau, M = DEFAULT_PARAMS
                try:
                    sgp_recon_img, sgp_rel_klds, sgp_rel_recon_errors, sgp_num_iters, sgp_extract_coord, sgp_execution_time, sgp_best_section = sgp(
                        cutout.data, psf, bkg.globalback, gamma=gamma, beta=beta, alpha_min=alpha_min, alpha_max=alpha_max,
                        alpha=alpha, M_alpha=M_alpha, tau=tau, M=M, proj_type=1, best_img=None, best_coord=(xc, yc), best_cutout=best_cut_image,
                        max_projs=max_projs, size=size, init_recon=2, stop_criterion=2, current_xy=(xc, yc), save=False,
                        filename=image, verbose=True, clip_X_upp_bound=False, diapl=False, to_search_for_best_stamp=False, offset=offset
                    )
                except TypeError:
                    continue

                # RL to get the reconstrcted image.
                rl_recon_img, rl_rel_recon_errors, rl_num_iters, rl_execution_time, rl_extract_coord = rl(
                        cutout.data, psf, bkg.globalback, max_iter=1500, normalize_kernel=True,
                        best_stamp=best_cut_image, current_xy=(xc, yc)
                    )

                sgp_bkg_after = MedianBackground().calc_background(sgp_recon_img)
                sgp_flux_after = calculate_flux(
                    sgp_recon_img, sgp_bkg_after, offset, size=size
                )
                rl_bkg_after = MedianBackground().calc_background(rl_recon_img)
                rl_flux_after = calculate_flux(rl_recon_img.reshape(size, size), rl_bkg_after, offset, size=size)

                # We calculate ellipticity and fwhm from the `radprof_ellipticity` module.
                before_ecc, before_fwhm = calculate_ellipticity_fwhm(cutout.data, use_moments=True)
                sgp_after_ecc, sgp_after_fwhm = calculate_ellipticity_fwhm(sgp_recon_img, use_moments=True)
                rl_after_ecc, rl_after_fwhm = calculate_ellipticity_fwhm(rl_recon_img, use_moments=True)

                before_center = centroid_2dg(cutout.data)

                sgp_after_center = centroid_2dg(sgp_recon_img)
                sgp_centroid_err = (before_center[0]-sgp_after_center[0], before_center[1]-sgp_after_center[1])
                sgp_l1_centroid_err = abs(before_center[0]-sgp_after_center[0]) + abs(before_center[1]-sgp_after_center[1])

                rl_after_center = centroid_2dg(rl_recon_img)
                rl_centroid_err = (before_center[0]-rl_after_center[0], before_center[1]-rl_after_center[1])
                rl_l1_centroid_err = abs(before_center[0]-rl_after_center[0]) + abs(before_center[1]-rl_after_center[1])

                sgp_after_fwhm = np.round(sgp_after_fwhm.value, 3)
                rl_after_fwhm = np.round(rl_after_fwhm.value, 3)
                sgp_l1_centroid_err = np.round(sgp_l1_centroid_err, 3)
                rl_l1_centroid_err = np.round(rl_l1_centroid_err, 3)
                sgp_execution_time = np.round(sgp_execution_time, 3)
                rl_execution_time = np.round(rl_execution_time, 3)

                ######################
                ### Radial Profile ###
                ######################

                orig_center = centroid_2dg(cutout.data)
                original_radprof = radial_profile(cutout.data, orig_center)[:17]

                # rl_recon_radprof = radial_profile(rl_recon_img, rl_after_center)
                # sgp_recon_radprof = radial_profile(sgp_recon_img, sgp_after_center)

                # Update final needed parameters list.
                star_coord = (xc, yc)
                compare_list.append(
                    [image, rl_num_iters, rl_execution_time, sgp_num_iters, sgp_execution_time, rl_after_fwhm, sgp_after_fwhm, rl_l1_centroid_err, sgp_l1_centroid_err]
                )
                count += 1
                break
            break
        
        if count == 30:
            if save:
                compare = np.array(compare_list)
                df = pd.DataFrame(compare)
                df.columns = [
                    "image", "rl num_iters", "rl exec_time (s)", "fc-sgp num_iters", "fc-sgp exec_time (s)", "rl fwhm (pix)", "fc-sgp fwhm (pix)", "rl l1 centroid_err", "fc-sgp l1 centroid_err"
                ]
                # df["rl exec_time (s)"] = np.round(df["rl exec_time (s)"], 3)
                # df["fc-sgp exec_time (s)"] = np.round(df["fc-sgp exec_time (s)"], 3)
                # df["rl fwhm (pix)"] = np.round(df["rl fwhm (pix)"], 3)
                # df["fc-sgp fwhm (pix)"] = np.round(df["fc-sgp fwhm (pix)"], 3)
                # df["rl l1 centroid_err"] = np.round(df["rl l1 centroid_err"], 3)
                # df["fc-sgp l1 centroid_err"] = np.round(df["fc-sgp l1 centroid_err"], 3)
                df.to_csv("sgp_rl_compare_metrics.csv")
            sys.exit()

    if save:
        compare = np.array(compare_list)
        df = pd.DataFrame(compare)
        df.columns = [
            "image", "rl num_iters", "rl exec_time (s)", "fc-sgp num_iters", "fc-sgp exec_time (s)", "rl fwhm (pix)", "fc-sgp fwhm (pix)", "rl l1 centroid_err", "fc-sgp l1 centroid_err"
        ]
        df = np.round(df, decimals=3)
        df.to_csv("sgp_rl_compare_metrics.csv")
