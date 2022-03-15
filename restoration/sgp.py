import os
import errno
import glob
import numpy as np
import pandas as pd
import sys
# Use default_timer instead of timeit.timeit: Reasons here: https://stackoverflow.com/a/25823885
from timeit import default_timer as timer

import astropy.units as u
from astropy.convolution import convolve, convolve_fft, Gaussian2DKernel
from astropy.io import fits

from photutils.background import Background2D, MedianBackground, MeanBackground
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from photutils.segmentation import detect_threshold, detect_sources

import matplotlib.pyplot as plt

from photutils.centroids import centroid_2dg
from photutils.segmentation import SourceCatalog

from sep import extract, Background
from sgp_validation import validate_single
from flux_conserve_proj import projectDF
from radprof_ellipticity import radial_profile, calculate_ellipticity_fwhm


DEFAULT_PARAMS = (1000, 1e-4, 0.4, 1.3, 1e-1, 1e1, 3, 0.5, 1)

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

def relative_recon_error(ground_truth, image, scaling, l2=True):
    """Calculates the relative reconstruction error between `image`,
    the restored image at the kth iteration and the ground truth image.

    ground_truth: 2D array
        Image array representing the ground truth.
    image: 2D array
        Partially restored image under consideration.
    l2: bool
        Whether to use

    """
    ground_truth /= scaling
    if l2:
        rel_err = np.linalg.norm(image - ground_truth) / np.linalg.norm(ground_truth)
    else:
        rel_err = np.linalg.norm(image - ground_truth, ord=1) / np.linalg.norm(ground_truth, ord=1)
    return rel_err

# def calc_properties(stamp, bkg, size=25):
#     # Reshape
#     stamp = stamp.reshape(size, size)
#     stamp -= bkg  # Background-subtracted data.

#     thresh = detect_threshold(stamp, nsigma=2)
#     segment_image = detect_sources(stamp, thresh, npixels=5)
#     sp = SourceCatalog(stamp, segment_img=segment_image, localbkg_width=16)  # `localbkg_width` allows subtracting bkg level.
#     return sp.min_value[0], sp.max_value[0]  # Index by zero since when this method is used, we would already be sure that only one source is present.

def calculate_flux(stamp, bkg, offset=None, size=25):
    """
    stamp: 2D array.

    """
    # Reshape
    stamp = stamp.reshape(size, size)
    if offset is not None:
        stamp = stamp[offset:size-offset, offset:size-offset]
    N = stamp.size

    return stamp.sum() - N * bkg  # Since stamp is background subtracted, we sum over all pixels only.

# def calculate_psnr(img, truth, max_value=1):
#     """"Calculating peak signal-to-noise ratio (PSNR) between two images.
    
#     Note: This metric is not used anymore.

#     """
#     img = img.flatten()
#     mse = np.mean((np.array(img/img.max(), dtype=np.float32) - np.array(truth/truth.max(), dtype=np.float32)) ** 2)
#     if mse == 0:
#         return 100
#     return 20 * np.log10(max_value / (np.sqrt(mse)))

def sgp(
    image, psf, bkg, init_recon=0, proj_type=0, stop_criterion=0, MAXIT=500,
    gamma=1e-4, beta=0.4, alpha=1.3, alpha_min=1e-5, alpha_max=1e5, M_alpha=3, tau=0.5, M=1,
    max_projs=1000, size=25, clip_X_upp_bound=False, best_img=None, best_coord=None, best_cutout=None, current_xy=None, save=True,
    filename=None, verbose=True, diapl=True, to_search_for_best_stamp=True, offset=None, flux=None
):
    """Perform the SGP algorithm on a single star stamp.

    Parameters
    ----------
    image: Observed star cutout image.
    psf: PSF.
    bkg: Background level around the star cutout (float).
    init_recon: 2D matrix to initialize the reconstructed image.
        - 0: All entries equal to 1: np.ones_like(image), default.
        - 1: Random entries: np.random.randn(image.shape).
        - 2: Same as observed image.
        - 3: Initialize with: np.ones(gn.size) * sum(gn.ravel() - bkg) / gn.size
    proj_type: Type of projection during the iteration:
        - 0: Non-negativity constraint only.
        - 1: Non-negativity + flux conservation. (flux_observed_img = sum(observed_img - bkg))
    stop_criterion: Choice of rule to stop iteration.
        - 0: iter > MAXIT, where `iter` denotes the current iteration number.
            Maxout.
        - 1: Use KL divergence condition plus a maxout condition to terminate if the objective function cannot be minimized within `MAXIT` no. of iterations.
            abs(KL(k) - KL(k-1)) <= tol * KL(k) OR iter > MAXIT.
        - 2: Relative reconstruction error (RRE) with respect to a pre-defined best star cutout image (from original science image). Here, the pre-defined cutout is taken from the `ccfbtf170075c.fits` image.
            Stop iteration when the current RRE becomes greater than the previous iteration's RRE.
    MAXIT: Maximum no. of iterations - for Maxout condition.

    gamma: Used for sufficient decrease: Linesearch penalty parameter.
    beta: Linesearch back-tracking/scaling parameter (used in the backtracking loop).
    alpha: Initial value for alpha, the step length.
        - This value is updated during the iterations.
    alpha_min: alpha lower bound for Barzilai-Borwein' steplength.
    alpha_max: alpha upper bound for Barzilai-Borwein' steplength.
    M_alpha: Memory length for `alphabb2`.
    tau: Alternating parameter.
    M: Non-monotone linear search memory (M = 1 means monotone search).
    max_projs: Maximum no. of iterations for the flux conservation procedure, see `flux_conserve_proj.py`.
    size: Size of the stamp to extract, defaults to 25.
    clip_X_upp_bound: Clip the scaling matrix at the upper bound, defaults to False
        Using False gave significantly better results.
    best_img: A star stamp extracted from the lowest FWHM science image.
        - It is used to calculate the relative construction error.
        - This stamp is considered to be an ideal star image.
    best_coord: Coordinate list file for `best_img`.
        - It is used to extract the relevant cutout from the lowest FWHM image.
        - The coordinates are estimated using the DIAPL package.
    best_cutout: Need to specify if best_img is None.
    to_search_for_best_stamp: Whether to search for ground-truth stamp.
    current_xy: tuple, x and y coordinates of the star cutout with respect to the original image.
    offset: Offset used while deciding star stamp size.
    save: To save the reconstructed image or not.
    
    Notes
    -----
    - The scaling matrix used here is not diagonal as of now.

    Porting frm MATLAB to Python:
    -----------------------------
    1. np.dot(a, b) is same as dot(a, b) ONLY for 1-D arrays.
    2. np.multiply(a, b) (or a * b) is same as a .* b
    3. If C = [1, 2, 3], then C[0:2] is same as C(1:2).
        In general, array[i:k] in NumPy is same as array(i+1:k) in MATLAB.

    """
    def A(psf, x):
        """Function representing the PSF matrix.

        Returns a convoluted image; partially reconstructed image convolved with PSF.

        Note:
        -----
        - It uses the FFT version of the convolution to speed up the convolution process.

        """
        x = x.reshape(image.shape)
        conv = convolve_fft(x, psf, normalize_kernel=True).ravel()
        return conv

    def AT(psf, x):
        """Function representing the transposed PSF matrix.

        Returns a convoluted image; partially reconstructed image convolved with the transpose of PSF.

        Note:
        -----
        - It uses the FFT version of the convolution to speed up the convolution process.

        """
        x = x.reshape(image.shape)
        conv = convolve_fft(x, psf.T, normalize_kernel=True).ravel()
        return conv

    ##### Closure end #####
    t0 = timer()  # Start clock timer.
    
    # image[image<0] = 0
    _shape = image.shape
    gn = image.flatten()
    # Change null pixels of the observed image.
    min_val = min(gn[gn > 0])
    gn[gn <= 0] = min_val * sys.float_info.epsilon * sys.float_info.epsilon

    assert np.all(gn >= 0), "Each pixel in the observed image must be non-negative!"

    rel_recon_errors = []  # List to store relative construction errors.
    rel_klds = []
    prev_rel_err = np.Inf  # Initialize previous relative reconstruction error.
    min_flux_diff = np.Inf

    if stop_criterion == 2:
        prev_recon_img = None  # Store previous reconstructed image.

    #############################################
    ###### Ensure PSF values are positive. ######
    #############################################
    min_psf = min(psf[psf > 0])
    psf[psf <= 0] = min_psf * sys.float_info.epsilon * sys.float_info.epsilon
    # Normalize PSF such that columns sum to one.
    psf /= psf.sum(axis=0, keepdims=1)

    #############################################
    ### Initialization of reconstructed image ###
    #############################################
    if init_recon == 0:
        x = np.ones_like(image)
    elif init_recon == 1:
        x = np.random.randn(*_shape)
    elif init_recon == 2:
        x = gn.reshape(_shape)
    elif init_recon == 3:
        x = calculate_flux(gn, bkg, offset, size=size) / image.size * np.ones(_shape)
        # gaussian_kernel = Gaussian2DKernel(10, x_size=11, y_size=11)
        # print("Convolving constant image to prevent edge effects...")
        # x = A(gaussian_kernel, x)
        # plt.imshow(x.reshape(_shape))
        # plt.show()

    # Treat image as vector
    x = x.flatten()

    # Stop criterion
    if stop_criterion == 0:
        tol = None  # No need of tolerance.
    elif stop_criterion == 1 or stop_criterion == 2 or stop_criterion == 3:
        tol = 1e-4

    # # Scale all image values.
    scaling = np.max(gn)
    gn /= scaling
    bkg /= scaling
    x /= scaling

    # Computations needed only once.
    N = gn.size
    flux = calculate_flux(gn, bkg, offset, size=size)

    iter_ = 1
    Valpha = alpha_max * np.ones(M_alpha)
    Fold = -1e30 * np.ones(M)
    Discr_coeff = 2 / N
    ONE = np.ones(N)

    ##############################################
    ### Set flag based on the projection type. ###
    ##############################################
    pflag = proj_type  # Default: 0.

    #####################################################
    ### Setup directory to store reconstructed images ###
    #####################################################
    if save:
        dirname = "SGP_reconstructed_images/"
        try:
            os.mkdir(dirname)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise OSError("Directory already exists!")
            pass

    discr = np.zeros(MAXIT + 1)
    times = np.zeros(MAXIT + 1)
    times[0] = 0

    #####################
    ### START of SGP. ###
    #####################

    # Project the initial point.
    if pflag == 0:
        x[x < 0] = 0
    elif pflag == 1:
        x = projectDF(flux, x, np.ones_like(x))

    #######################################
    ### Compute objecive function value ###
    #######################################
    x_tf = A(psf, x)
    den = x_tf + bkg
    temp = np.divide(gn, den)
    g = ONE - AT(psf, temp)
    # KL divergence.
    fv = np.sum(np.multiply(gn, np.log(temp))) + np.sum(x_tf) - flux

    # Bounds for scaling matrix.
    y = np.multiply((flux / (flux + N * bkg)), AT(psf, gn))
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
        X = x
        # Bounds
        X[X < X_low_bound] = X_low_bound
        if clip_X_upp_bound:
            X[X > X_upp_bound] = X_upp_bound

    if pflag == 1:
        D = np.divide(1, X)

    ###############################################
    ### Setup tolerance for main SGP iterations ###
    ###############################################
    if verbose:
        if stop_criterion == 1:  # KLD
            print("\n=== Stop criterion: Relative KL divergence ===\n")
        elif stop_criterion == 2:  # RRE
            print("\n=== Stop criterion: Relative Reconstruction error ===\n")

    #################
    ### Main loop ###
    #################
    loop = True
    while loop:
        prev_recon_img = x

        # Store alpha and obj func values
        Valpha[0:M_alpha-1] = Valpha[1:M_alpha]
        Fold[0:M-1] = Fold[1:M]
        Fold[M-1] = fv

        # Compute descent direction.
        y = x - alpha * np.multiply(X, g)

        if pflag == 0:
            y[y < 0] = 0
        elif pflag == 1:
            y = projectDF(flux, np.multiply(y, D), D)
        d = y - x

        # Backtracking loop for linearsearch.
        gd = np.dot(d, g)
        lam = 1  # `lam = 1` is a common choice. `lam = 2` also works better here.

        fcontinue = 1
        d_tf = A(psf, d)
        fr = max(Fold)

        while fcontinue:
            xplus = x + lam * d
            x_tf_try = x_tf + lam*d_tf
            den = x_tf_try + bkg

            temp = np.divide(gn, den)
            fv = np.sum(np.multiply(gn, np.log(temp))) + np.sum(x_tf_try) - flux

            if fv <= fr + gamma * lam * gd or lam < 1e-12:
                x = xplus
                xplus = None  # clear the variable.
                sk = lam*d
                x_tf = x_tf_try
                x_tf_try = None  # clear
                gtemp = ONE - AT(psf, temp)

                yk = gtemp - g
                g = gtemp
                gtemp = None  # clear
                fcontinue = 0
            else:
                lam = lam * beta

        if verbose:
            if fv >= fr:
                print("\tWarning, fv >= fr")

        # Update the scaling matrix and steplength
        X = x
        X[X < X_low_bound] = X_low_bound
        if clip_X_upp_bound:
            X[X > X_upp_bound] = X_upp_bound

        # Since if `clip_X_upp_bound` is set to false, we do not clip at the upper bound, we need to ensure the upper bound still applies.
        assert all(np.isfinite(X)), "The scaling matrix violates either the lower or upper bound!"

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

        # Increment iteration counter.
        iter_ += 1

        times[iter_ - 1] = timer() - t0
        discr[iter_-2] = Discr_coeff * fv

        ######################
        ### STOP criterion ###
        ######################
        #########################
        #### 1. KL divergence ###
        #########################

        # We calculate KLD and RRE irrespective of whether it used as a stopping criterion or not - for analysis.
        reldecrease = abs(fv-Fold[M-1]) / abs(fv)
        if verbose:
            print(f"Iter: {iter_-1} / {MAXIT}, reldecrease: {reldecrease}, tol: {tol}")
        rel_klds.append(reldecrease)

        #########################################
        #### 2. Relative Reconstruction Error ###
        #########################################

        # Need to provide the image "best_img" as input.
        # Equation: || xk - x || / || x ||, where ||*|| is the L1/L2-norm.
        if stop_criterion == 2 or stop_criterion == 3:
            if to_search_for_best_stamp:
                best_section, extract_coord = search_for_best_stamp(best_img, best_coord, current_xy, size=size, diapl=False)
            else:
                best_section, extract_coord = best_cutout.ravel(), best_coord

            # Note: We have not scaled `best_section` so the error decrement might look small.
            rel_err = relative_recon_error(best_section, x, scaling)

            if verbose:
                print(f"Iter: {iter_-1}, rel_err: {rel_err}")
            rel_recon_errors.append(rel_err)
        else:
            extract_coord = None
            best_section = None

        if stop_criterion == 1:  # KLD
            loop = reldecrease > tol
        elif stop_criterion == 2:  # RRE
            if rel_err < prev_rel_err:
                prev_rel_err = rel_err
            else:
                x = prev_recon_img  # Return the previous iteration reconstructed image, since at the current iteration, we are one step ahead.
                loop = False
        elif stop_criterion == 3:  # RRE + flux
            ## Might not work for some images.
            flux_diff_now = abs(calculate_flux(x, bkg, offset, size=size) - flux)
            if rel_err < prev_rel_err and flux_diff_now < min_flux_diff:
                prev_rel_err = rel_err
                min_flux_diff = flux_diff_now
            else:
                x = prev_recon_img
                loop = False

        # Apply Maxout condition.
        if iter_ > MAXIT:
            loop = False

    end = timer()
    if verbose:
        print(f"Execution time: {end-t0}s")

    # Since calculations were done on scaled flattened images, reshape them to a 2D matrix and scale them.
    final_img = x.reshape(_shape)
    final_img = final_img * scaling

    if save:
        try:
            fits.writeto(
                os.path.join(
                    dirname, f"{filename}_{current_xy[0]}_{current_xy[1]}_SGP_recon_{iter_-1}.fits"
                ), final_img
            )
        except OSError:
           print("Could not save restored image. File already exists!")

    return final_img, rel_klds, rel_recon_errors, iter_-1, extract_coord, end-t0, best_section


if __name__ == "__main__":
    with open("candidate_defect_images.txt", "r") as f:
        defect_images = f.read().splitlines()
    
    os.mkdir("SGP_original_images")

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

                fits.writeto(f"SGP_original_images/{image}_original_{xc}_{yc}.fits", cutout.data)

                max_projs, gamma, beta, alpha_min, alpha_max, alpha, M_alpha, tau, M = DEFAULT_PARAMS
                try:
                    recon_img, rel_klds, rel_recon_errors, num_iters, extract_coord, execution_time, best_section = sgp(
                        cutout.data, psf, bkg.globalback, gamma=gamma, beta=beta, alpha_min=alpha_min, alpha_max=alpha_max,
                        alpha=alpha, M_alpha=M_alpha, tau=tau, M=M, proj_type=1, best_img=None, best_coord=(xc, yc), best_cutout=best_cut_image,
                        max_projs=max_projs, size=size, init_recon=2, stop_criterion=2, current_xy=(xc, yc), save=save,
                        filename=image, verbose=True, clip_X_upp_bound=False, diapl=False, to_search_for_best_stamp=False, offset=offset
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
                    fig.suptitle("SGP")

                    ax[0, 0].imshow(cutout.data, origin="lower")
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
                    [image, num_iters, execution_time, DEFAULT_PARAMS, star_coord, rel_klds, rel_recon_errors, np.round(flux_before, 3), np.round(flux_after, 3), centroid_err, l2_centroid_err, before_ecc, after_ecc, before_fwhm, after_fwhm, flag]
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
                df.to_csv("sgp_params_and_metrics.csv")
            sys.exit()

    print(f"Success count: {success}")
    print(f"Failure count: {failure}")

    if save:
        np.save("original_radprofs.npy", np.array(original_radprofs))
        final_params = np.array(final_params_list)
        df = pd.DataFrame(final_params)
        df.to_csv("sgp_params_and_metrics.csv")
