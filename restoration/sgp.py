import os
import sys
import errno
import glob
import logging
import numpy as np
import pandas as pd
from functools import partial
# Use default_timer instead of timeit.timeit: Reasons here: https://stackoverflow.com/a/25823885
from timeit import default_timer as timer
import argparse

import astropy.units as u
from astropy.io import fits

from photutils.background import Background2D, MedianBackground, MeanBackground, StdBackgroundRMS

from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats, SigmaClip, gaussian_fwhm_to_sigma
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve, convolve_fft
from photutils.segmentation import detect_threshold, detect_sources, make_source_mask, SegmentationImage

import matplotlib.pyplot as plt

from photutils.centroids import centroid_2dg
from sklearn.preprocessing import KernelCenterer

from flux_conserve_proj import projectDF
from utils import calculate_bkg, source_info, preprocess

DEFAULT_PARAMS = (1000, 1e-4, 0.4, 1e-5, 1e5, 1e1, 3, 0.5, 1)


def sgp(
    gn, psf, bkg, init_recon=0, proj_type=0, stop_criterion=0, MAXIT=27,
    gamma=1e-4, beta=0.4, alpha=1.3, alpha_min=1e-5, alpha_max=1e5, M_alpha=3,
    tau=0.5, M=1, max_projs=1000, clip_X_upp_bound=True, save=True,
    verbose=True, flux=None, ccd_sat_level=None, scale_data=True
):
    """Perform the SGP algorithm on a single star stamp.

    Args:
        gn (_type_): Observed star cutout image.
        psf (_type_): PSF matrix.
        bkg (_type_): Background level around the star cutout.
        init_recon (int, optional): Initialization for the reconstructed image.
            Either 0, 1, 2, or 3. Defaults to 0.
        proj_type (int, optional): Type of projection during the iteration.
            Either 0 or 1. Defaults to 0.
        stop_criterion (int, optional): Choice of rule to stop iteration.
            Either 1, 2, 3, or 4. Defaults to 0.
        MAXIT (int, optional): Maximum no. of iterations. Defaults to 500.
        gamma (_type_, optional): Linesearch penalty parameter. Defaults to 1e-4.
        beta (float, optional): Linesearch back-tracking/scaling parameter (used in the backtracking loop). Defaults to 0.4.
        alpha (float, optional): Initial value for alpha, the step length. Defaults to 1.3.
            This value is updated during the iterations.
        alpha_min (_type_, optional): alpha lower bound for Barzilai-Borwein' steplength. Defaults to 1e-5.
        alpha_max (_type_, optional): alpha upper bound for Barzilai-Borwein' steplength. Defaults to 1e5.
        M_alpha (int, optional): Memory length for `alphabb2`. Defaults to 3.
        tau (float, optional): Alternating parameter.. Defaults to 0.5.
        M (int, optional): Non-monotone linear search memory (M = 1 means monotone search). Defaults to 1.
        max_projs (int, optional): Maximum no. of iterations for the flux conservation procedure. Defaults to 1000.
        clip_X_upp_bound (bool, optional): Clip the elements of the diagonal scaling matrix at the upper bound. Defaults to False.
        save (bool, optional): Whether to save the reconstructed image or not. Defaults to True.
        verbose (bool, optional): Controls screen verbosity. Defaults to True.
        flux (_type_, optional): Precomputed flux of the object inside `gn`. Defaults to None.
        ccd_sat_level (float, optional): Saturating pixel value (i.e. counts) for the CCD used. Defaults to 65000.0.
        scale_data (bool, optional): Whether to scale `gn`, 'bkg`, and `x` (the reconstructed image) before applying SGP. Defaults to False.

    Raises:
        OSError: _description_

    Returns:
        _type_: _description_

    Notes:
        == Porting from MATLAB to Numpy ==
        1. np.dot(a, b) is same as dot(a, b) ONLY for 1-D arrays.
        2. np.multiply(a, b) (or a * b) is same as a .* b
        3. If C = [1, 2, 3], then C[0:2] is same as C(1:2).
            In general, array[i:k] in NumPy is same as array(i+1:k) in MATLAB.
        4. x.T in Numpy is the same as x' in matlab, where x is a two-dimensional array.

        * Instead of using Afunction, here a slightly different approach is used as compared to SGP-dec.
        * See here: https://numpy.org/doc/stable/user/numpy-for-matlab-users.html for more details.

    """
    # Check normalization condition of PSF.
    checkPSF = np.abs(np.sum(psf.flatten()) - 1.)
    tolCheckPSF = 1e4 * np.finfo(float).eps
    if checkPSF > tolCheckPSF:
        raise ValueError("PSF is not properly normalized! Ensure that each column of the PSF matrix sums up to one.")

    logging.basicConfig(filename='sgp.log', level=logging.INFO, force=True)

    _shape = gn.shape

    TF = np.fft.fftn(np.fft.fftshift(psf))
    CTF = np.conj(TF)
    def afunction(x, TF, dimensions):
        x = np.reshape(x, dimensions)
        out = np.real(np.fft.ifftn(
            np.multiply(TF, np.fft.fftn(x))
        ))
        out = out.flatten()
        return out

    A = partial(afunction, TF=TF, dimensions=psf.shape)
    AT = partial(afunction, TF=CTF, dimensions=psf.shape)

    # def A(psf, x):
    #     """Describes the PSF function.

    #     Args:
    #         psf (numpy.ndarray): PSF matrix.
    #         x (numpy.ndarray): Image with which PSF needs to be convolved.

    #     Returns:
    #         numpy.ndarray: Convoluted version of image `x`.

    #     Note
    #     ----
    #     It uses the FFT version of the convolution to speed up the convolution process.

    #     """
    #     x = x.reshape(_shape)
    #     conv = convolve_fft(x, psf, normalize_kernel=True, normalization_zero_tol=1e-4).ravel()
    #     return conv

    # def AT(psf, x):
    #     """Describes the transposed PSF function.

    #     Args:
    #         psf (numpy.ndarray): PSF matrix.
    #         x (numpy.ndarray): Image with which PSF needs to be convolved.

    #     Returns:
    #         numpy.ndarray: Transpose-convoluted version of image `x`.

    #     Note
    #     ----
    #     It uses the FFT version of the convolution to speed up the convolution process.

    #     """
    #     x = x.reshape(_shape)
    #     conv = convolve_fft(x, psf.conj().T, normalize_kernel=True, normalization_zero_tol=1e-4).ravel()
    #     return conv

    # A = partial(A, psf=psf)
    # AT = partial(AT, psf=psf)

    t0 = timer()  # Start clock timer.

    # Initialization of reconstructed image.
    if init_recon == 0:
        x = np.zeros_like(gn)
    elif init_recon == 1:
        x = np.random.randn(*gn.shape)
    elif init_recon == 2:
        x = gn.copy()
    elif init_recon == 3:
        if flux is None:
            x = np.sum(gn.flatten() - bkg) / gn.size * np.ones_like(gn)
        else:
            x = flux / gn.size * np.ones_like(gn)

    # Treat images as vectors.
    gn = gn.flatten()
    x = x.flatten()

    # Stop criterion settings.
    if stop_criterion == 1:
        tol = []
    elif stop_criterion == 2 or stop_criterion == 3:
        tol = 1e-4
    elif stop_criterion == 4:
        tol = 1 + 1 / np.mean(gn)

    # Scaling
    if scale_data:
        scaling = np.max(gn)
        gn = gn / scaling
        bkg = bkg / scaling
        x = x / scaling
    else:
        scaling = 1.   # Scaling can have adverse effects on the flux in the final scaled output image, hence we do not scale.

    # Change null pixels of observed image.
    vmin = np.min(gn[gn > 0])
    eps = np.finfo(float).eps
    gn[gn <= 0] = vmin * eps * eps

    # Computations needed only once.
    N = gn.size
    if flux is None:
        flux = np.sum(gn) - N * bkg
    else:  # If flux is already provided, we need to scale it. This option is recommended.
        flux /= scaling  # Input a precomputed flux: this could be more accurate in some situations.

    iter_ = 1
    Valpha = alpha_max * np.ones(M_alpha)
    Fold = -1e30 * np.ones(M)
    Discr_coeff = 2 / N * scaling
    ONE = np.ones(N)

    # Projection type.
    pflag = proj_type  # Default: 0.

    # Setup directory to store reconstructed images.
    if save:
        dirname = "SGP_reconstructed_images/"
        try:
            os.mkdir(dirname)
            fits.writeto(f'{dirname}/orig.fits', gn.reshape(_shape))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise OSError("Directory already exists!")
            pass

    discr = np.zeros(MAXIT + 1)
    times = np.zeros(MAXIT + 1)
    times[0] = 0

    # Start of SGP.
    # Project the initial point.
    if pflag == 0:
        x[x < 0] = 0
    elif pflag == 1:
        # Here an identity matrix is used for projecting the initial point, which means it is not a "scaled" gradient projection as of now.
        # Instead, it is a simple gradient projection without scaling (see equation 2 in https://kd.nsfc.gov.cn/paperDownload/ZD6608905.pdf, for example).
        x = projectDF(flux, x, np.ones_like(x), scaling, ccd_sat_level=ccd_sat_level)

    # Compute objecive function value.
    x_tf = A(x=x)
    den = x_tf + bkg
    temp = np.divide(gn, den)
    g = ONE - AT(x=temp)
    # KL divergence.
    fv = np.sum(np.multiply(gn, np.log(temp))) + np.sum(x_tf) - flux

    # Bounds for scaling matrix.
    y = np.multiply((flux / (flux + N * bkg)), AT(x=gn))
    X_low_bound = np.min(y[y > 0])
    X_upp_bound = np.max(y)
    if X_upp_bound / X_low_bound < 50:
        X_low_bound = X_low_bound / 10
        X_upp_bound = X_upp_bound * 10

    # Discrepancy.
    discr[0] = Discr_coeff * fv

    # Scaling matrix.
    if init_recon == 0:
        X = np.ones_like(x)
    else:
        X = x.copy()
        # Bounds
        X[X < X_low_bound] = X_low_bound
        if clip_X_upp_bound:
            X[X > X_upp_bound] = X_upp_bound

    if pflag == 1:
        D = np.divide(1, X)

    # Setup tolerance for main SGP iterations.
    if verbose:
        if stop_criterion == 2:
            logging.info(f'it {iter_-1} || x_k - x_(k-1) ||^2 / || x_k ||^2 0 \n')
            tol = tol * tol
        elif stop_criterion == 3:
            logging.info(f'it {iter_-1} | f_k - f_(k-1) | / | f_k | 0 \n')
        elif stop_criterion == 4:
            logging.info(f'it {iter_-1} D_k {discr[0]} \n')

    # Main loop.
    loop = True
    while loop:
        # Store alpha and obj func values.
        Valpha[0:M_alpha-1] = Valpha[1:M_alpha]
        Fold[0:M-1] = Fold[1:M]
        Fold[M-1] = fv

        # Compute descent direction.
        y = x - alpha * np.multiply(X, g)

        if pflag == 0:
            y[y < 0] = 0
        elif pflag == 1:
            y = projectDF(flux, np.multiply(y, D), D, scaling, ccd_sat_level=ccd_sat_level)

        d = y - x

        # Backtracking loop for linearsearch.
        gd = np.dot(d, g)
        lam = 1  # `lam = 1` is a common choice, so we use it.

        fcontinue = 1
        d_tf = A(x=d)
        fr = max(Fold)

        while fcontinue:
            xplus = x + lam * d
            x_tf_try = x_tf + lam*d_tf
            den = x_tf_try + bkg

            temp = np.divide(gn, den)
            fv = np.sum(np.multiply(gn, np.log(temp))) + np.sum(x_tf_try) - flux

            if fv <= fr + gamma * lam * gd or lam < 1e-12:
                x = xplus.copy()
                xplus = None  # clear the variable.
                sk = lam*d
                x_tf = x_tf_try
                x_tf_try = None  # clear
                gtemp = ONE - AT(x=temp)

                yk = gtemp - g
                g = gtemp.copy()
                gtemp = None  # clear
                fcontinue = 0
            else:
                lam = lam * beta

        if fv >= fr and verbose:
            logging.warning("\tWarning, fv >= fr")

        # Update the scaling matrix and steplength
        X = x.copy()
        X[X < X_low_bound] = X_low_bound
        if clip_X_upp_bound:
            X[X > X_upp_bound] = X_upp_bound

        # Since if `clip_X_upp_bound` is set to false, we do not clip at the upper bound, we need to ensure the upper bound still applies.
        # assert all(np.isfinite(X)), "The scaling matrix violates either the lower or upper bound!"

        D = np.divide(1, X)
        sk2 = np.multiply(sk, D)
        yk2 = np.multiply(yk, X)
        bk = np.dot(sk2, yk)
        ck = np.dot(yk2, sk)
        if bk <= 0:
            alpha1 = min(10*alpha, alpha_max)
        else:
            alpha1bb = np.sum(np.dot(sk2, sk2)) / bk
            alpha1 = min(alpha_max, max(alpha_min, alpha1bb))
        if ck <= 0:
            alpha2 = min(10*alpha, alpha_max)
        else:
            alpha2bb = ck / np.sum(np.dot(yk2, yk2))
            alpha2 = min(alpha_max, max(alpha_min, alpha2bb))

        Valpha[M_alpha-1] = alpha2

        if iter_ <= 20:
            alpha = min(Valpha)
        elif alpha2/alpha1 < tau:
            alpha = min(Valpha)
            tau = tau * 0.9
        else:
            alpha = alpha1
            tau = tau * 1.1

        # Note: At this point, the original matlab code does: alpha = double(single(alpha)), which we do not do here. Don't know what's the meaning of that.

        iter_ += 1
        times[iter_-1] = timer() - t0
        discr[iter_-1] = Discr_coeff * fv

        # Stop criterion.
        if stop_criterion == 1:
            logging.info(f'it {iter_-1} of  {MAXIT}\n')
        elif stop_criterion == 2:
            normstep = np.dot(sk, sk) / np.dot(x, x)
            loop = normstep > tol
            logging.info(f'it {iter_-1} || x_k - x_(k-1) ||^2 / || x_k ||^2 {normstep} tol {tol}\n')
        elif stop_criterion == 3:
            reldecrease = abs(fv-Fold[M-1]) / abs(fv)
            loop = reldecrease > tol
            logging.info(f'it {iter_-1} | f_k - f_(k-1) | / | f_k | {reldecrease} tol {tol}\n')
        elif stop_criterion == 4:
            loop = discr[iter_-1] > tol
            logging.info(f'it {iter_-1} D_k {discr[iter_-1]} tol {tol}\n')

        if iter_ > MAXIT:
            loop = False

        if save:
            filename = f'SGP_reconstructed_images/rec_{iter_-1}.fits'
            fits.writeto(filename, x.reshape(_shape), overwrite=True)
            # Residual image.
            res = np.divide(x - gn, np.sqrt(x))
            filename = f'SGP_reconstructed_images/res_{iter_-1}.fits'
            fits.writeto(filename, res.reshape(_shape), overwrite=True)

    # Since calculations were done on scaled flattened images, reshape them to a 2D matrix and scale them.
    x = x.reshape(_shape)
    x = x * scaling

    discr = discr[0:iter_]
    times = times[0:iter_]
    iter_ = iter_ - 1

    return x, iter_, discr, times


if __name__ == "__main__":
    with open('defect_images.txt') as f:
        defect_images = f.read().splitlines()

    os.mkdir("FC_SGP_original_images")

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

    for image in sorted(defect_images):
        dfnames = df['name'].tolist()
        dfnames = [fg.strip() for fg in dfnames]
        if image not in dfnames:
            continue

        data = fits.getdata(image.split('.')[0] + 'r' + '1_2.fits')

        stars = df[df['name'] == image][[' x', ' y']].to_numpy()
        print(stars)
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
            best_bkg, _ = calculate_bkg(best_cutout)
            best_centroid = centroid_2dg(best_cutout-best_bkg)
            print(f'Centroid of ground-truth star: {best_centroid}')

            # Uncomment below lines if you want to use validation.
            # params = validate_single(
            #     cutout, psf, bkg, x, y, size=size,
            #     best_cutout=best_cutout, xbest=x, ybest=y
            # )
            params = DEFAULT_PARAMS

            if params is None:  # If no optimal parameter that satisfied all conditions, then use default.
                print("\n\nNo best parameter found that matches all conditions. Falling to default params\n\n")
                max_projs, gamma, beta, alpha_min, alpha_max, alpha, M_alpha, tau, M = DEFAULT_PARAMS
            else:
                max_projs, gamma, beta, alpha_min, alpha_max, alpha, M_alpha, tau, M = params
                print(f"\n\nOptimal parameters: (max_projs, gamma, beta, alpha_min, alpha_max, alpha, M_alpha, tau, M) = {params}\n\n")

            recon_img, rel_klds, rel_recon_errors, num_iters, extract_coord, execution_time, best_section = sgp(
                cutout, psf, bkg, gamma=gamma, beta=beta, alpha_min=alpha_min, alpha_max=alpha_max,
                alpha=alpha, M_alpha=M_alpha, tau=tau, M=M, proj_type=1, best_img=None, best_coord=(xc, yc), best_cutout=best_cutout,
                max_projs=max_projs, size=size, init_recon=2, stop_criterion=2, current_xy=(xc, yc), save=True,
                filename=image, verbose=True, clip_X_upp_bound=False, diapl=False, to_search_for_best_stamp=False, offset=offset,
                flux=flux_before
            )

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
                print("\n\n")

            if plot:
                fig, ax = plt.subplots(2, 2)
                fig.suptitle("FC_SGP")

                ax[0, 0].imshow(cutout, origin="lower")
                ax[0, 0].set_title("Original", loc="center")
                ax[0, 1].imshow(recon_img.reshape(size, size), origin="lower")
                ax[0, 1].set_title("Reconstructed", loc="center")
                ax[1, 0].plot(rel_klds[:-1])  # Don't select the last value since from that value, the error rises - for plotting reasons.
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
            flux_thresh = 0.05 * flux_before
            if flux_after < flux_before + flux_thresh and flux_after > flux_before - flux_thresh:
                success += 1
                flag = 1  # Flag to denote if reconstruction is under the flux limit.
            else:
                failure += 1
                flag = 0

            print(f"Success till now: {success}")
            print(f"Failure till now: {failure}")

            execution_time = np.round(execution_time, 3)
            l1_centroid_err = np.round(l1_centroid_err, 3)

            # Update final needed parameters list.
            star_coord = (xc, yc)
            final_params_list.append(
                [image, num_iters, execution_time, star_coord, rel_klds, rel_recon_errors, np.round(flux_before, 3), np.round(flux_after, 3), bkg, bkg_after, l1_centroid_err, before_ecc, after_ecc, np.round(before_fwhm.value, 3), np.round(after_fwhm.value, 3), flag]
            )
            count += 1

            fits.writeto(f"FC_SGP_original_images/{image}_{xc}_{yc}_SGP_orig.fits", cutout)

    if count == 30:
        print(f"Success count: {success}")
        print(f"Failure count: {failure}")

        if save:
            final_params = np.array(final_params_list)
            df = pd.DataFrame(final_params)
            df.columns = ["image", "num_iters", "execution_time", "star_coord", "rel_klds", "rel_recon_errors", "flux_before", "flux_after", "bkg_before", "bkg_after", "l1_centroid_err", "before_ecc", "after_ecc", "before_fwhm (pix)", "after_fwhm (pix)", "flag"]
            df.to_csv("fc_sgp_params_and_metrics.csv")
        sys.exit()

    print(f"Success count: {success}")
    print(f"Failure count: {failure}")

    if save:
        final_params = np.array(final_params_list)
        df = pd.DataFrame(final_params)
        df.columns = ["image", "num_iters", "execution_time", "star_coord", "rel_klds", "rel_recon_errors", "flux_before", "flux_after", "bkg_before", "bkg_after", "l1_centroid_err", "before_ecc", "after_ecc", "before_fwhm (pix)", "after_fwhm (pix)", "flag"]
        df.to_csv("fc_sgp_params_and_metrics.csv")
