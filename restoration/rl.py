import os
import sys
import glob
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from numpy import unravel_index

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.convolution import convolve, convolve_fft  # Differs from `scipy.signal.convolve` by handling NaN values differently.
from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats

from sgp import source_info, calculate_bkg
from sklearn.preprocessing import KernelCenterer

from photutils.segmentation import detect_threshold, detect_sources, make_source_mask, SegmentationImage

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

def rl(
    image, psf, bkg_estimate, max_iter=1500, normalize_kernel=True,
    best_stamp=None, best_coord=None, current_xy=None
):
    """
    image: Observed image.
    psf: PSF estimate for the image.
    bkg_estimate: Background estimate.

    """
    # eps = 1e-12  # Use eps to prevent zero division - taken from scikit-image `richardson_lucy` source code.
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
    deconv_img = np.full(image.shape, 0.5, dtype=float)
    # deconv_img = image.copy()  # Does not work well for RL

    prev_rel_err = np.Inf  # Initialize previous relative reconstruction error.
    prev_recon_img = None  # Store previous reconstructed image. Needed only for `stop_criterion=2`
    
    loop = True
    iter_ = 1
    while loop:
        prev_recon_img = deconv_img

        # Note: The only mode possible in convolve_fft is mode = 'same', which is the default.
        conv = convolve_fft(deconv_img, psf, normalize_kernel=normalize_kernel) + bkg_estimate
        relative = image / conv
        deconv_img *= convolve_fft(relative, np.flip(psf), normalize_kernel=normalize_kernel)

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
        conv = convolve_fft(deconv_img, psf, normalize_kernel=normalize_kernel) + bkg_estimate
        relative = image / conv  # Handle near-zero values?
        deconv_img *= convolve_fft(relative, np.flip(psf), normalize_kernel=normalize_kernel) ** alpha

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
    with open('defect_images.txt') as f:
        defect_images = f.read().splitlines()

    os.mkdir("RL_reconstructed_images")

    plot = False
    verbose = True
    save = True
    final_params_list = []
    success = 0
    failure = 0
    size = 30
    approx_size = 30
    offset = None
    original_radprofs = []
    count = 0
    MEDIAN_FLUX = 61169.92578125  # Calculate using all stamps from the sample. TODO: Change this value.

    # _best_base_name = 'ccfbtf170075' + 'r'
    # best_cut_image = '.'.join((_best_base_name, 'fits'))
    # best_cut_coord_list = '.'.join((_best_base_name, 'coo'))

    df = pd.read_csv('coords_to_use.csv')

    # os.mkdir("COMPARE_IMAGES/")

    for image in sorted(defect_images):
        dfnames = df['name'].tolist()
        dfnames = [fg.strip() for fg in dfnames]
        if image not in dfnames:
            continue

        data = fits.getdata(image.split('.')[0] + 'r' + '1_2.fits')

        stars = df[df['name'] == image][[' x', ' y']].to_numpy()
        for xc, yc in stars:
            print(f'Coordindates: x: {xc}, y: {yc}')
            _check_cutout = Cutout2D(data, (xc, yc), size=60, mode='partial', fill_value=0.0, copy=True).data
            cutout = Cutout2D(data, (xc, yc), size=size, mode='partial', fill_value=0.0).data

            # Estimate background on check stamp.
            bkg, _ = calculate_bkg(_check_cutout)

            mask = make_source_mask(cutout, nsigma=2, npixels=5, dilate_size=5)

            prop_table_before = source_info(cutout, bkg, mask, approx_size).to_table()
            flux_before = prop_table_before['segment_flux'].value[0]
            flux_before_err = prop_table_before['segment_fluxerr'].value[0]
            psf = fits.getdata(f'../work/psf{image.split(".")[0]}_{str(1)}_{str(2)}_img.fits')
            # Center the PSF matrix
            psf = KernelCenterer().fit_transform(psf)
            psf = np.abs(psf)

            ref_imagename = f'cal_ccfbtf170075r{str(1)}_{str(2)}.fits'
            best_cutout = Cutout2D(fits.getdata(ref_imagename), (xc, yc), size=size, mode='partial', fill_value=0.0).data

            recon_img, rel_recon_errors, num_iters, execution_time, extract_coord = rl(
                cutout, psf, bkg, max_iter=10, normalize_kernel=True,
                best_stamp=best_cutout, current_xy=(xc, yc)
            )
            fits.writeto(f"RL_reconstructed_images/{image}_{xc}_{yc}_RL_recon_{num_iters}.fits", recon_img)

            bkg_after, mask_after = calculate_bkg(recon_img)

            prop_table_after = source_info(recon_img, bkg_after, mask_after, approx_size).to_table()
            flux_after = prop_table_after['segment_flux'].value[0]
            flux_after_err = prop_table_after['segment_fluxerr'].value[0]

            print(f"Flux before: {flux_before} +- {flux_before_err}")
            print(f"Flux after: {flux_after} +- {flux_after_err}")

            # We calculate ellipticity and fwhm from the `radprof_ellipticity` module.
            before_ecc, before_fwhm = calculate_ellipticity_fwhm(cutout-bkg, bkg, use_moments=True)
            after_ecc, after_fwhm = calculate_ellipticity_fwhm(recon_img-bkg_after, bkg_after, use_moments=True)

            before_center = centroid_2dg(cutout-bkg)
            after_center = centroid_2dg(recon_img-bkg_after)
            l1_centroid_err = abs(before_center[0]-after_center[0]) + abs(before_center[1]-after_center[1])

            if verbose:
                print("\n\n")
                print(f"No. of iterations: {num_iters}")
                print(f"Execution time: {execution_time}s")
                print(f"Flux (before): {flux_before}")
                print(f"Flux (after): {flux_after}")
                print(f"Ideal stamp for relative reconstruction error at x, y = {extract_coord}")
                print(f"Centroid error (before-after) = {l1_centroid_err}")
                print("\n\n")

            if plot:
                fig, ax = plt.subplots(2, 2)
                fig.suptitle("RL")

                ax[0, 0].imshow(cutout, origin="lower")
                ax[0, 0].set_title("Original", loc="center")
                ax[0, 1].imshow(recon_img.reshape(size, size), origin="lower")
                ax[0, 1].set_title("Reconstructed", loc="center")
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

            orig_center = centroid_2dg(cutout)
            original_radprof = radial_profile(cutout, orig_center)[:17]
            original_radprofs.append(original_radprof)

            print(f"Success till now: {success}")
            print(f"Failure till now: {failure}")

            execution_time = np.round(execution_time, 3)
            l1_centroid_err = np.round(l1_centroid_err, 3)

            # Update final needed parameters list.
            star_coord = (xc, yc)
            final_params_list.append(
                [image, num_iters, execution_time, star_coord, rel_recon_errors, np.round(flux_before, 3), np.round(flux_after, 3), l1_centroid_err, before_ecc, after_ecc, np.round(before_fwhm.value, 3), np.round(after_fwhm.value, 3), flag]
            )
            count += 1

            # fits.writeto(f"RL_original_images/{image}_{xc}_{yc}_SGP_orig.fits", cutout)

    if count == 30:
        print(f"Success count: {success}")
        print(f"Failure count: {failure}")

        if save:
            np.save("rl_original_radprofs.npy", np.array(original_radprofs))
            final_params = np.array(final_params_list)
            df = pd.DataFrame(final_params)
            df.columns = ["image", "num_iters", "execution_time", "star_coord", "rel_recon_errors", "flux_before", "flux_after", "l1_centroid_err", "before_ecc", "after_ecc", "before_fwhm (pix)", "after_fwhm (pix)", "flag"]
            df.to_csv("rl_params_and_metrics.csv")
        sys.exit()

    print(f"Success count: {success}")
    print(f"Failure count: {failure}")

    if save:
        np.save("rl_original_radprofs.npy", np.array(original_radprofs))
        final_params = np.array(final_params_list)
        df = pd.DataFrame(final_params)
        df.columns = ["image", "num_iters", "execution_time", "star_coord", "rel_recon_errors", "flux_before", "flux_after", "l1_centroid_err", "before_ecc", "after_ecc", "before_fwhm (pix)", "after_fwhm (pix)", "flag"]
        df.to_csv("rl_params_and_metrics.csv")
