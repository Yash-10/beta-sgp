import os
import sys
import glob
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from numpy import unravel_index

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.convolution import convolve  # Differs from `scipy.signal.convolve` by handling NaN values differently.
from astropy.nddata import Cutout2D

from sep import extract, Background
from sgp import calculate_flux, DEFAULT_PARAMS

from photutils.background import (
    MeanBackground, MedianBackground
)
from photutils.centroids import centroid_2dg
from radprof_ellipticity import radial_profile, calculate_ellipticity_fwhm

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

# Notes
# 1. By default, SExtractor is used, and `sigma_clip` is passed as None, i.e. no sigma clipping.

def rl(
    image, psf, bkg_estimate, max_iter=1500, normalize_kernel=True,
    best_stamp=None, best_coord=None, current_xy=None
):
    """
    image: Observed image.
    psf: PSF estimate for the image.
    bkg_estimate: Background estimate.

    """
    if best_stamp is not None:
        extract_coord = None
    rel_recon_errors = []  # List to store relative construction errors.

    min_val = min(image[image > 0])
    image[image <= 0] = min_val * sys.float_info.epsilon * sys.float_info.epsilon
    min_psf = min(psf[psf > 0])
    psf[psf <= 0] = min_psf * sys.float_info.epsilon * sys.float_info.epsilon

    start = timer()
    image = image.astype(float, copy=False)
    psf = psf.astype(float, copy=False)
    # deconv_img = np.full(image.shape, 0.5, dtype=image.dtype)
    deconv_img = image

    prev_rel_err = np.Inf  # Initialize previous relative reconstruction error.
    prev_recon_img = None  # Store previous reconstructed image. Needed only for `stop_criterion=2`
    
    loop = True
    iter_ = 1
    while loop:
        prev_recon_img = deconv_img

        # boundary="extend" and normalize_kernel=True for astropy convolution.
        conv = convolve(deconv_img, psf, normalize_kernel=normalize_kernel) + bkg_estimate
        relative = image / conv  # Handle near-zero values?
        deconv_img *= convolve(relative, np.flip(psf), normalize_kernel=normalize_kernel)

        rel_err = np.linalg.norm(deconv_img.ravel() - best_stamp.ravel()) / np.linalg.norm(best_stamp.ravel())
        rel_recon_errors.append(rel_err)
        print(f"Iteration no.: {iter_-1}, rel recon err: {rel_err}")

        if rel_err <= prev_rel_err:
            prev_rel_err = rel_err
        else:
            loop = False
            deconv_img = prev_recon_img

        iter_ += 1
        if iter_-1 > max_iter:
            loop = False
    
    end = timer()

    return deconv_img, rel_recon_errors[:-1], iter_-1, end-start, extract_coord  # Don't select the last error since it went up and termination occured before that.

def rl_mul_relax(image, psf, bkg_estimate, max_iter=1500, alpha=1.5,
    best_stamp=None, current_xy=None, normalize_kernel=True
):
    """
    image: Observed image.
    psf: PSF estimate for the image.
    bkg_estimate: Background estimate.

    Notes
    -----
    Add a multiplicative relaxation - acceleration.
    alpha: Multiplicative relaxation parameter. `alpha` must be > 1. Convergence proved for `alpha < 2`.

    TODO:
    - Add grid/random search for alpha in the interval (1, 2].

    """
    rel_recon_errors = []  # List to store relative construction errors.

    min_val = min(image[image > 0])
    image[image <= 0] = min_val * sys.float_info.epsilon * sys.float_info.epsilon
    min_psf = min(psf[psf > 0])
    psf[psf <= 0] = min_psf * sys.float_info.epsilon * sys.float_info.epsilon

    start = timer()
    image = image.astype(float, copy=False)
    psf = psf.astype(float, copy=False)
    # deconv_img = np.full(image.shape, 0.5, dtype=image.dtype)
    deconv_img = image

    prev_rel_err = np.Inf  # Initialize previous relative reconstruction error.
    prev_recon_img = None  # Store previous reconstructed image. Needed only for `stop_criterion=2`
    
    loop = True
    iter_ = 1
    while loop:
        prev_recon_img = deconv_img

        # boundary="extend" and normalize_kernel=True for astropy convolution.
        conv = convolve(deconv_img, psf, normalize_kernel=normalize_kernel) + bkg_estimate
        relative = image / conv  # Handle near-zero values?
        deconv_img *= convolve(relative, np.flip(psf), normalize_kernel=normalize_kernel) ** alpha

        rel_err = np.linalg.norm(deconv_img.ravel() - best_stamp.ravel()) / np.linalg.norm(best_stamp.ravel())
        rel_recon_errors.append(rel_err)
        print(f"Iteration no.: {iter_-1}, rel recon err: {rel_err}")

        if rel_err <= prev_rel_err:
            prev_rel_err = rel_err
        else:
            loop = False
            deconv_img = prev_recon_img

        iter_ += 1
        if iter_-1 > max_iter:
            loop = False
    
    end = timer()

    return deconv_img, rel_recon_errors[:-1], iter_-1, end-start, extract_coord  # Don't select the last error since it went up and termination occured before that.

if __name__ == "__main__":
    with open("candidate_defect_images.txt", "r") as f:
        defect_images = f.read().splitlines()
    
    os.mkdir("RL_original_images")

    plot = False
    verbose = True
    save = True
    final_params_list = []
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

    # Calculate mean PSF (mean of PSFs over a single frame) as a prior for the RL method.
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

                fits.writeto(f"RL_original_images/{image}_original_{xc}_{yc}.fits", cutout.data)

                max_projs, gamma, beta, alpha_min, alpha_max, alpha, M_alpha, tau, M = DEFAULT_PARAMS
                try:
                    recon_img, rel_recon_errors, num_iters, execution_time, extract_coord = rl(
                        cutout.data, psf, bkg.globalback, max_iter=1500, normalize_kernel=True,
                        best_stamp=best_cut_image, current_xy=(xc, yc)
                    )
                except ValueError:
                    continue

                # TODO: Revisit this: Calculating bkg on recon stamp seems difficult.
                bkg_after = MedianBackground().calc_background(recon_img)
                flux_after = calculate_flux(
                    recon_img, bkg_after, offset, size=size
                )

                # We calculate ellipticity and fwhm from the `radprof_ellipticity` module.
                before_ecc, before_fwhm = calculate_ellipticity_fwhm(cutout.data, use_moments=True)
                after_ecc, after_fwhm = calculate_ellipticity_fwhm(recon_img, use_moments=True)

                before_center = centroid_2dg(cutout.data)
                after_center = centroid_2dg(recon_img)
                centroid_err = (before_center[0]-after_center[0], before_center[1]-after_center[1])
                l2_centroid_err = np.linalg.norm(before_center-after_center)

                if verbose:
                    print("\n\n")
                    print(f"No. of iterations: {num_iters}")
                    print(f"Execution time: {execution_time}s")
                    print(f"Flux (before): {flux_before}")
                    print(f"Flux (after): {flux_after}")
                    print(f"Ideal stamp for relative reconstruction error from {ground_truth_star_stamp_name} at x, y = {extract_coord}")
                    print(f"Centroid error (before-after) = {centroid_err}")
                    print("\n\n")

                if plot:
                    fig, ax = plt.subplots(2, 2)
                    fig.suptitle("RL")

                    ax[0, 0].imshow(cutout.data, origin="lower")
                    ax[0, 0].set_title("Original", loc="center")
                    ax[0, 1].imshow(recon_img.reshape(size, size), origin="lower")
                    ax[0, 1].set_title("Reconstructed", loc="center")
                    # ax[1, 0].plot(rel_klds[:-1])  # Don't select the last value since from that value, the error rises - for plotting reasons.
                    ax[1, 0].set_xticks(range(0, num_iters))
                    ax[1, 0].set_title("Relative KL divergence", loc="center")
                    ax[1, 1].plot(rel_recon_errors[:-1])
                    ax[1, 1].set_xticks(range(0, num_iters))
                    ax[1, 1].set_title("Relative reconstruction error", loc="center")
                    ax[1, 0].set_xlabel("Iteration no.")
                    ax[1, 1].set_xlabel("Iteration no.")

                    # From https://stackoverflow.com/questions/12998430/remove-xticks-in-a-matplotlib-plot
                    ax[0, 0].tick_params(
                        axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=False
                    ) # labels along the bottom edge are off
                    ax[0, 1].tick_params(
                        axis='x',          # changes apply to the x-axis
                        which='both',      # both major and minor ticks are affected
                        bottom=False,      # ticks along the bottom edge are off
                        top=False,         # ticks along the top edge are off
                        labelbottom=False
                    )
                    plt.show()

                ## Success/Failure based on flux criterion ##
                flux_thresh = 0.05 * MEDIAN_FLUX
                if flux_after < flux_before + flux_thresh and flux_after > flux_before - flux_thresh:
                    success += 1
                    flag = 1  # Flag to denote if reconstruction is under the flux limit.
                else:
                    failure += 1
                    flag = 0

                ######################
                ### Radial Profile ###
                ######################

                orig_center = centroid_2dg(cutout.data)
                original_radprof = radial_profile(cutout.data, orig_center)[:17]
                original_radprofs.append(original_radprof)

                print(f"Success till now: {success}")
                print(f"Failure till now: {failure}")

                # Update final needed parameters list.
                star_coord = (xc, yc)
                final_params_list.append(
                    [image, num_iters, execution_time, DEFAULT_PARAMS, star_coord, rel_recon_errors, np.round(flux_before, 3), np.round(flux_after, 3), centroid_err, l2_centroid_err, before_ecc, after_ecc, before_fwhm, after_fwhm, flag]
                )
                count += 1
                break
            break

        if count == 30:
            print(f"Success count: {success}")
            print(f"Failure count: {failure}")

            if save:
                np.save("original_radprofs.npy", np.array(original_radprofs))
                final_params = np.array(final_params_list)
                df = pd.DataFrame(final_params)
                df.to_csv("rl_params_and_metrics.csv")
            sys.exit()

    print(f"Success count: {success}")
    print(f"Failure count: {failure}")

    if save:
        np.save("original_radprofs.npy", np.array(original_radprofs))
        final_params = np.array(final_params_list)
        df = pd.DataFrame(final_params)
        df.to_csv("rl_params_and_metrics.csv")
