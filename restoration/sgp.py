import math
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
from astropy.wcs import WCS
from astropy.wcs.utils import pixel_to_skycoord

from photutils.background import Background2D, MedianBackground, MeanBackground, StdBackgroundRMS

from astropy.nddata import Cutout2D
from astropy.stats import sigma_clipped_stats, SigmaClip, gaussian_fwhm_to_sigma
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
from astropy.convolution import convolve, convolve_fft
from photutils.segmentation import detect_threshold, detect_sources, make_source_mask, SegmentationImage

import matplotlib.pyplot as plt

from flux_conserve_proj import projectDF
from utils import source_info, scale_psf, artificial_sky_background, create_subdivisions, reconstruct_full_image_from_patches


DEFAULT_PARAMS = (1000, 1e-4, 0.4, 1e-5, 1e5, 1e1, 3, 0.5, 1)
DEFAULT_COLUMNS = ['label', 'xcentroid', 'ycentroid', 'sky_centroid',
                   'bbox_xmin', 'bbox_xmax', 'bbox_ymin', 'bbox_ymax',
                   'area', 'semimajor_sigma', 'semiminor_sigma',
                   'orientation', 'eccentricity', 'min_value', 'max_value',
                   'local_background', 'segment_flux', 'segment_fluxerr', 'ellipticity', 'fwhm']

def sgp(
    gn, psf, bkg, init_recon=0, proj_type=0, stop_criterion=0, MAXIT=500,
    gamma=1e-4, beta=0.4, alpha=1.3, alpha_min=1e-5, alpha_max=1e5, M_alpha=3,
    tau=0.5, M=1, max_projs=1000, save=False, obj=None, verbose=True, flux=None,
    ccd_sat_level=None, scale_data=True, errflag=False, tol_convergence=1e-4,
    use_original_SGP_Afunction=True
):
    """Perform the SGP algorithm.

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
        save (bool, optional): Whether to save the reconstructed image or not. Defaults to True.
        verbose (bool, optional): Controls screen verbosity. Defaults to True.
        flux (_type_, optional): Precomputed flux of the object inside `gn`. Defaults to None.
        ccd_sat_level (float, optional): Saturating pixel value (i.e. counts) for the CCD used. Defaults to 65000.0.
        scale_data (bool, optional): Whether to scale `gn`, 'bkg`, and `x` (the reconstructed image) before applying SGP. Defaults to False.
        use_original_SGP_Afunction: If True, works only when image and PSF size are same. Set to False if sizes are different.
        tol_convergence: only used if stop_criterion == 2 or 3.

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
        4. x.conj().T in Numpy is the same as x' in matlab, where x is a two-dimensional array.

        * Afunction implementation is provided (similar to SGP-dec). A slightly different approach compared to SGP-dec is also commented out in the code.
        * See here: https://numpy.org/doc/stable/user/numpy-for-matlab-users.html for more details.

    """
    # Check normalization condition of PSF.
    checkPSF = np.abs(np.sum(psf.flatten()) - 1.)
    tolCheckPSF = 1e4 * np.finfo(float).eps
    if checkPSF > tolCheckPSF:
        errmsg = f"\n\tsum(psf) - 1. = {checkPSF}, tolerance = {tolCheckPSF}"
        raise ValueError(f'PSF is not normalized! Provide a normalized PSF! {errmsg}')

    logging.basicConfig(filename='sgp.log', level=logging.INFO, force=True)

    _shape = gn.shape

    if use_original_SGP_Afunction:
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
    else:
        def A(psf, x):
            """Describes the PSF function.

            Args:
                psf (numpy.ndarray): PSF matrix.
                x (numpy.ndarray): Image with which PSF needs to be convolved.

            Returns:
                numpy.ndarray: Convoluted version of image `x`.

            Note
            ----
            It uses the FFT version of the convolution to speed up the convolution process.

            """
            x = x.reshape(_shape)
            conv = convolve_fft(x, psf, normalize_kernel=True, normalization_zero_tol=1e-4).ravel()
            return conv

        def AT(psf, x):
            """Describes the transposed PSF function.

            Args:
                psf (numpy.ndarray): PSF matrix.
                x (numpy.ndarray): Image with which PSF needs to be convolved.

            Returns:
                numpy.ndarray: Transpose-convoluted version of image `x`.

            Note
            ----
            It uses the FFT version of the convolution to speed up the convolution process.

            """
            x = x.reshape(_shape)
            conv = convolve_fft(x, psf.conj().T, normalize_kernel=True, normalization_zero_tol=1e-4).ravel()
            return conv

        A = partial(A, psf=psf)
        AT = partial(AT, psf=psf)

    t0 = timer()  # Start clock timer.

    # Initialization of reconstructed image.
    if init_recon == 0:
        x = np.zeros_like(gn)
    elif init_recon == 1:
        np.random.seed(42)
        x = np.random.randn(*gn.shape)
    elif init_recon == 2:
        x = gn.copy()
    elif init_recon == 3:
        if flux is None:
            x = np.sum(gn - bkg) / gn.size * np.ones_like(gn)
        else:
            x = flux / gn.size * np.ones_like(gn)

    # Treat images as vectors.
    gn = gn.flatten()
    x = x.flatten()
    bkg = bkg.flatten()

    # Stop criterion settings.
    if stop_criterion == 1:
        tol = []
    elif stop_criterion == 2 or stop_criterion == 3:
        tol = tol_convergence
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
        flux = np.sum(gn - bkg)
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

    if errflag and obj is None:
        raise ValueError("errflag was set to True but no ground-truth was passed.")

    if errflag:
        err = np.zeros(MAXIT + 1)
        obj = obj.flatten()
        obj = obj / scaling
        obj_sum = np.sum(obj * obj)

    # Start of SGP.
    # Project the initial point.
    if pflag == 0:
        x[x < 0] = 0
    elif pflag == 1:
        # Here an identity matrix is used for projecting the initial point, which means it is not a "scaled" gradient projection as of now.
        # Instead, it is a simple gradient projection without scaling (see equation 2 in https://kd.nsfc.gov.cn/paperDownload/ZD6608905.pdf, for example).
        x = projectDF(flux, x, np.ones_like(x), scaling, ccd_sat_level=ccd_sat_level, max_projs=max_projs)

    if errflag:
        e = x - obj
        err[0] = np.sqrt(np.sum(e * e) / obj_sum)

    # Compute objecive function value.
    x_tf = A(x=x)
    den = x_tf + bkg
    temp = np.divide(gn, den)
    g = ONE - AT(x=temp)
    # KL divergence.
    fv = np.sum(np.multiply(gn, np.log(temp))) + np.sum(x_tf) - flux

    # Bounds for scaling matrix.
    y = np.multiply((flux / (flux + bkg)), AT(x=gn))
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
        prev_x = x.copy()

        # Store alpha and obj func values.
        Valpha[0:M_alpha-1] = Valpha[1:M_alpha]
        Fold[0:M-1] = Fold[1:M]
        Fold[M-1] = fv

        # Compute descent direction.
        y = x - alpha * np.multiply(X, g)

        if pflag == 0:
            y[y < 0] = 0
        elif pflag == 1:
            y = projectDF(flux, np.multiply(y, D), D, scaling, ccd_sat_level=ccd_sat_level, max_projs=max_projs)

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
        X[X > X_upp_bound] = X_upp_bound

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

        if errflag:
            e = x - obj
            err[iter_] = np.sqrt(np.sum(e * e) / obj_sum)

        # Stop criterion.
        if stop_criterion == 1:
            logging.info(f'it {iter_-1} of  {MAXIT}\n')
        elif stop_criterion == 2:
            normstep = np.dot(sk, sk) / np.dot(x, x)
            loop = normstep > tol
            logging.info(f'it {iter_-1} || x_k - x_(k-1) ||^2 / || x_k ||^2 {normstep} tol {tol}\n')
        elif stop_criterion == 3:
            reldecrease = (Fold[M-1]-fv) / fv
            loop = reldecrease > tol and reldecrease >= 0
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

        if not loop:
            x = prev_x

    # Since calculations were done on scaled flattened images, reshape them to a 2D matrix and scale them.
    x = x.reshape(_shape)
    x = x * scaling

    if errflag:
        err = err[0:iter_]

    discr = discr[0:iter_]
    times = times[0:iter_]
    iter_ = iter_ - 1

    return x, iter_, discr, times, err if errflag else None


def betaDiv(y, x, betaParam):
    """_summary_
    Args:
        y (_type_): _description_
        x (_type_): _description_
        betaParam (_type_): _description_
    Returns:
        _type_: _description_
    Note:
    - See an example implementation here: https://pytorch-nmf.readthedocs.io/en/stable/modules/metrics.html#torchnmf.metrics.beta_div
    """
    if betaParam == 0:
        return np.sum(x / y) - np.sum(np.log(x / y)) - x.size  # or y.size can also be used
    elif betaParam == 1:
        return np.sum(np.multiply(x, np.log(np.divide(x, y)))) - np.sum(x) + np.sum(y)
    else:
        scal = 1 / (betaParam * (betaParam - 1))
        return np.sum(scal*x**betaParam) + np.sum(scal*(betaParam-1)*y**betaParam) - np.sum(scal*betaParam*x*y**(betaParam-1))


# @njit
def betaDivDeriv(y, x, betaParam):  # Verified that the derivative is correct using pytorch backward() followed by .grad attribute checking.
    """ To get the derivative equation:
            from sympy import diff, symbols
            x, y, beta = symbols('x y beta')
            diff(betaDiv(y, x, beta), beta)
            then copy and paste the expression to return.
    Args:
        y (_type_): _description_
        x (_type_): _description_
        beta (_type_): _description_
    Returns:
        _type_: _description_
    Note:
    Comparing with PyTorch grad calculation:
    ```
    In [27]: x=torch.tensor([1,2, 4.5, 7.9, 1.5], requires_grad=True)
    In [28]: y=torch.tensor([9.3,2.5, 4.5, 7.9, 1.5], requires_grad=True)
    In [29]: f=betaDiv(y, x, betaParam)
    In [30]: betaParam=torch.tensor(1.5, requires_grad=True)
    In [31]: f=betaDiv(y, x, betaParam)
    In [32]: f.backward()
    In [33]: betaParam.grad
    Out[33]: tensor(24.6697)
    In [34]: betaDivDeriv(y, x, betaParam).sum()
    Out[34]: tensor(24.6697, grad_fn=<SumBackward0>)
    ```
    Indeed, both are same.
    """
    # t1 = (-(2*beta-1) / (beta**2-beta)**2) * (x**beta+(beta-1)*y**beta-beta*x*y**(beta-1))
    # t2 = (1 / (beta *(beta-1))) * (beta*x**(beta-1)+(beta-1)*beta*y**(beta-1)+y**beta-x*(y**(beta-1)+beta*(beta-1)*y**(beta-2)))
    # return t1+t2
    if betaParam == 0 or betaParam == 1:  # Special cases.
        return 0
    return -x*y**(betaParam - 1)*np.log(y)/(betaParam - 1) + x*y**(betaParam - 1)/(betaParam - 1)**2 + x**betaParam*np.log(x)/(betaParam*(betaParam - 1)) - x**betaParam/(betaParam*(betaParam - 1)**2) + y**betaParam*np.log(y)/betaParam - x**betaParam/(betaParam**2*(betaParam - 1)) - y**betaParam/betaParam**2


def betaDivDerivwrtY(AT, den_arg, gn_arg, betaParam):  # Verified and compared with the special case of KL divergence.
    return den_arg**(betaParam-1) - AT(x=gn_arg*den_arg**(betaParam-2))


def lr_schedule(init_lr, k, epoch):
    return init_lr * math.exp(-k * epoch)


def sgp_betaDiv(
    gn, psf, bkg, init_recon=0, proj_type=0, stop_criterion=0, MAXIT=500,
    gamma=1e-4, beta=0.4, alpha=1.3, alpha_min=1e-5, alpha_max=1e5, M_alpha=3,
    tau=0.5, M=1, max_projs=1000, save=False, obj=None, verbose=True, flux=None,
    ccd_sat_level=None, scale_data=True, errflag=False, adapt_beta=True,
    betaParam=1.005, lr=1e-3, lr_exp_param=0.1, schedule_lr=False, tol_convergence=1e-4,
    use_original_SGP_Afunction=True
):
    # TODO: errflag and obj are just added. -- add implementation for them as well.
    """Performs the SGP algorithm.
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
        save (bool, optional): Whether to save the reconstructed image or not. Defaults to True.
        verbose (bool, optional): Controls screen verbosity. Defaults to True.
        flux (_type_, optional): Precomputed flux of the object inside `gn`. Defaults to None.
        ccd_sat_level (float, optional): Saturating pixel value (i.e. counts) for the CCD used. Defaults to 65000.0.
        scale_data (bool, optional): Whether to scale `gn`, 'bkg`, and `x` (the reconstructed image) before applying SGP. Defaults to False.
        use_original_SGP_Afunction: If True, works only when image and PSF size are same. Set to False if sizes are different.
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
        4. x.conj().T in Numpy is the same as x' in matlab, where x is a two-dimensional array.
        * Afunction implementation is provided (similar to SGP-dec). A slightly different approach compared to SGP-dec is also commented out in the code.
        * See here: https://numpy.org/doc/stable/user/numpy-for-matlab-users.html for more details.
    """
    # Check normalization condition of PSF.
    checkPSF = np.abs(np.sum(psf.flatten()) - 1.)
    tolCheckPSF = 1e4 * np.finfo(float).eps
    if checkPSF > tolCheckPSF:
        errmsg = f"\n\tsum(psf) - 1. = {checkPSF}, tolerance = {tolCheckPSF}"
        raise ValueError(f'PSF is not normalized! Provide a normalized PSF! {errmsg}')

    logging.basicConfig(filename='sgp.log', level=logging.INFO, force=True)

    _shape = gn.shape
    if schedule_lr:
        init_lr = lr
    
    if use_original_SGP_Afunction:
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
    else:
        def A(psf, x):
            """Describes the PSF function.
            Args:
                psf (numpy.ndarray): PSF matrix.
                x (numpy.ndarray): Image with which PSF needs to be convolved.
            Returns:
                numpy.ndarray: Convoluted version of image `x`.
            Note
            ----
            It uses the FFT version of the convolution to speed up the convolution process.
            """
            x = x.reshape(_shape)
            conv = convolve_fft(x, psf, normalize_kernel=True, normalization_zero_tol=1e-4).ravel()
            return conv

        def AT(psf, x):
            """Describes the transposed PSF function.
            Args:
                psf (numpy.ndarray): PSF matrix.
                x (numpy.ndarray): Image with which PSF needs to be convolved.
            Returns:
                numpy.ndarray: Transpose-convoluted version of image `x`.
            Note
            ----
            It uses the FFT version of the convolution to speed up the convolution process.
            """
            x = x.reshape(_shape)
            conv = convolve_fft(x, psf.conj().T, normalize_kernel=True, normalization_zero_tol=1e-4).ravel()
            return conv

        A = partial(A, psf=psf)
        AT = partial(AT, psf=psf)

    t0 = timer()  # Start clock timer.

    # Initialization of reconstructed image.
    if init_recon == 0:
        x = np.zeros_like(gn)
    elif init_recon == 1:
        np.random.seed(42)
        x = np.random.randn(*gn.shape)
    elif init_recon == 2:
        x = gn.copy()
    elif init_recon == 3:
        if flux is None:
            x = np.sum(gn - bkg) / gn.size * np.ones_like(gn)
        else:
            x = flux / gn.size * np.ones_like(gn)

    # Treat images as vectors.
    gn = gn.flatten()
    x = x.flatten()
    bkg = bkg.flatten()

    # Stop criterion settings.
    if stop_criterion == 1:
        tol = []
    elif stop_criterion == 2 or stop_criterion == 3:
        tol = tol_convergence
    elif stop_criterion == 4:
        tol = 1 + 1 / np.mean(gn)

    # Scaling
    # !! Note that for stop_criterion=3 where the KL divergence is calculated, it is important to scale the data since KL divergence needs probabilities. In future, it might be helpful to forcefully scale data if stop_criterion=3
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
        flux = np.sum(gn - bkg)
    else:  # If flux is already provided, we need to scale it. This option is recommended.
        flux /= scaling  # Input a precomputed flux: this could be more accurate in some situations.

    iter_ = 1
    Valpha = alpha_max * np.ones(M_alpha)
    Fold = -1e30 * np.ones(M)
    Discr_coeff = 2 / N * scaling
    # ONE = np.ones(N)

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
        x = projectDF(flux, x, np.ones_like(x), scaling, ccd_sat_level=ccd_sat_level, max_projs=max_projs)

    # Compute objecive function value.
    x_tf = A(x=x)
    den = x_tf + bkg
    temp = np.divide(gn, den)
    g = betaDivDerivwrtY(AT, den, gn, betaParam)

    # Divergence.
    # Note: beta div with betaParam=1 and KL divergence equation will match only if flux=None, else slight differences will be there.
    fv = betaDiv(den, gn, betaParam)

    # Bounds for scaling matrix.
    y = np.multiply((flux / (flux + bkg)), AT(x=gn))
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
    epoch = 0
    betadivs = []
    while loop:
        epoch += 1
        prev_x = x.copy()

        # Store alpha and obj func values.
        Valpha[0:M_alpha-1] = Valpha[1:M_alpha]
        Fold[0:M-1] = Fold[1:M]
        Fold[M-1] = fv

        # Compute descent direction.
        y = x - alpha * np.multiply(X, g)

        if pflag == 0:
            y[y < 0] = 0
        elif pflag == 1:
            y = projectDF(flux, np.multiply(y, D), D, scaling, ccd_sat_level=ccd_sat_level, max_projs=max_projs)

        d = y - x

        # Backtracking loop for linearsearch.
        gd = np.dot(d, g)
        # In every epoch, lambda is set to 1. If we do not, then lambda will be multiplied many times with beta, which is less than 1, causing lambda to reach zero. This would mean there are no updates on x in later stages.
        lam = 1  # `lam = 1` is a common choice, so we use it.

        fcontinue = 1
        d_tf = A(x=d)
        fr = max(Fold)

        while fcontinue:
            xplus = x + lam * d
            x_tf_try = x_tf + lam*d_tf
            den = x_tf_try + bkg

            # temp = np.divide(gn, den)
            fv = betaDiv(den, gn, betaParam)

            if fv <= fr + gamma * lam * gd or lam < 1e-12:
                x = xplus.copy()
                xplus = None  # clear the variable.
                sk = lam*d
                x_tf = x_tf_try
                x_tf_try = None  # clear
                gtemp = betaDivDerivwrtY(AT, den, gn, betaParam)

                yk = gtemp - g
                g = gtemp.copy()
                gtemp = None  # clear
                fcontinue = 0
            else:
                lam = lam * beta
                if adapt_beta:
                    bgrad = betaDivDeriv(den, gn, betaParam).mean()
                    betaParam = betaParam - lr * bgrad
                    # betaParam = np.mean(np.full(_shape, betaParam).ravel() - lr * bgrad)  # This is another option to update betaParam.

        if fv >= fr and verbose:
            logging.warning("\tWarning, fv >= fr")

        # Update the scaling matrix and steplength
        X = x.copy()
        X[X < X_low_bound] = X_low_bound
        X[X > X_upp_bound] = X_upp_bound

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

        if schedule_lr:
            # Update learning rate.
            lr = lr_schedule(init_lr, lr_exp_param, epoch)
            # print(f'Learning rate at epoch {epoch}: {lr}')

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
            reldecrease = (Fold[M-1]-fv) / fv
            loop = reldecrease > tol and reldecrease >= 0
            betadivs.append(fv)
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

        if not loop:
            x = prev_x

        if epoch == MAXIT:
          break

    # Since calculations were done on scaled flattened images, reshape them to a 2D matrix and scale them.
    x = x.reshape(_shape)
    x = x * scaling

    discr = discr[0:iter_]
    times = times[0:iter_]
    iter_ = iter_ - 1

    print(f'Beta parameter in beta-divergence (final value): {betaParam}')
    print(f'No. of iterations: {iter_}')

    return x, iter_, discr, times, None


def print_options(opt):
    print('\n')
    print("------------ Options ------------")
    for arg in vars(opt):
        print(f'{arg}:\t\t{getattr(opt, arg)}')
    print("------------ End ----------------")
    print('\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='sets data path for data used for SGP')
    parser.add_argument('--data_path_sciimg', type=str, help='data path that contains the science image.', required=True)
    parser.add_argument('--data_path_psf', type=str, help='data path that contains the PSF associated with the supplied science image.', required=True)
    # Important note: using scale_psf when not using --degrade_image would mean manipulating the PSF - this will rarely help since it changes the science.
    parser.add_argument('--scale_psf', action='store_true', help='Whether to scale FWHM of PSF. A 2D Gaussian PSF is returned and hence the scaled PSF might not have properties exactly the same as input PSF.')
    parser.add_argument('--psf_scale_fwhm', type=float, default=1.2, help='Only used if --scale_psf is specified. It specifies the FWHM of the 2D Gaussian kernel used to generate the scaled PSF.')
    parser.add_argument('--init_recon', type=int, default=2, help='How to initialize the reconstructed image.')
    parser.add_argument('--stop_criterion', type=int, default=3, help='How to decide when to stop SGP iterations.')
    # parser.add_argument('--save_images', action='store_true', help='if specified, store images as FITS files.')
    parser.add_argument('--flip_image', action='store_true', help='if specified, horizontally flips the input image before passing it to SGP.')
    parser.add_argument('--add_bkg_to_deconvolved', action='store_true', help='if specificed, adds an artificial background to the deconvolved image before detection with the aim to remove any spurious sources.')
    parser.add_argument('--box_height', type=int, default=64, help='height of box for estimating background in the input image given by `data_path_sciimg`, only used if not specified --use_subdiv')
    parser.add_argument('--box_width', type=int, default=64, help='width of box for estimating background in the input image given by `data_path_sciimg`, only used if not specified --use_subdiv')
    parser.add_argument('--use_subdiv', action='store_true', help='If specified, creates subdivisions, deconvolves each of them, and then mosaics them to create a single final deconvolved image.')
    parser.add_argument('--subdivision_size', type=int, default=100, help='subdivision size, only considered if --use_subdiv is specified.')
    parser.add_argument('--subdiv_overlap', type=int, default=10, help='overlap to use while extracting the subdivisions, only considered if --use_subdiv is specified.')
    parser.add_argument('--sextractor_config_file_name', type=str, help='(Note: This must invariably be in the sgp_reconstruction_results/ directory, but pass only the filename to this argument and not the entire path) Name of the sextractor config file for original image. The config file for the deconvolved images is set based on the original config file. The Only used if use_sextractor is True')
    parser.add_argument('--use_sextractor', action='store_true', help='Whether to use the original SExtractor for extracting source information.')
    parser.add_argument('--use_beta_div', action='store_true', help='Whether to use beta divergence inside SGP instead of the KL divergence.')
    parser.add_argument('--initial_beta', type=float, default=1.005, help='The initial value of beta to start with.')
    parser.add_argument('--initial_lr', type=float, default=1e-3, help='The initial learning rate to use for updating beta.')
    parser.add_argument('--tol_convergence', type=float, default=1e-4, help='The tolerance level to use for terminating the SGP iterations.')
    parser.add_argument('--gain', type=float, default=None, help='CCD gain')
    parser.add_argument('--saturate', type=float, default=None, help='CCD saturating pixel value.')
    # parser.add_argument('--plot', action='store_true', help='Whether to show any plots while running this script')

    opt = parser.parse_args()
    print_options(opt)

    psf_hdul = fits.open(opt.data_path_psf)
    fwhm = psf_hdul[0].header['FWHM']  # in pix.
    psf = psf_hdul[0].data
    if opt.scale_psf:
        # Scale PSF.
        psf = scale_psf(psf, gaussian_fwhm=opt.psf_scale_fwhm, size=psf.shape)

    hdul = fits.open(opt.data_path_sciimg)
    image_header = hdul[0].header
    # readout_noise = image_header['READNOI']  # in e-
    if opt.gain is None:
        gain = image_header['GAIN']  # in e-/ADU
    else:
        gain = opt.gain
    if opt.saturate is None:
        ccd_sat_level = image_header['SATURATE']
    else:
        ccd_sat_level = opt.saturate

    # Get WCS
    wcs = WCS(hdul[0].header)
    image = hdul[0].data
    if opt.flip_image:
        image = np.fliplr(image)
        psf = np.fliplr(psf)

    # Statistics of images.
    dirname = 'sgp_reconstruction_results'
    if not os.path.exists(dirname):
        os.mkdir(dirname)
    basename = opt.data_path_sciimg.split('/')[-1]

    if opt.use_subdiv:
        subdivs = create_subdivisions(
            image, subdiv_shape=(opt.subdivision_size, opt.subdivision_size),
            overlap=opt.subdiv_overlap, wcs=wcs
        )

        orig_fluxes = []
        deconv_fluxes = []
        deconv_objects_count = 0
        orig_objects_count = 0
        orig_objects = []
        deconv_objects = []
        execution_times = []

        for i, subdiv in enumerate(subdivs):
            assert subdiv.data.shape == (opt.subdivision_size, opt.subdivision_size)
            if opt.use_sextractor:
                fits.writeto(f'{dirname}/subdiv_{i}_temp.fits', subdiv.data, overwrite=True)

            objects, orig_fluxes_subdiv, orig_bkg, orig_bkg_rms, fig = source_info(
                subdiv.data, opt.subdivision_size, opt.subdivision_size,
                min_area=5, threshold=3, gain=gain, plot_positions_indicator=False,  # maskthresh=ccd_sat_level
                use_sextractor=opt.use_sextractor, image_name=f'subdiv_{i}_temp.fits',
                defaultFile=opt.sextractor_config_file_name
            )

            x_in_nonsubdivided = []
            y_in_nonsubdivided = []
            for obj in objects.iterrows():
                _x, _y = subdiv.to_original_position((obj[1]['X_IMAGE_DBL'], obj[1]['Y_IMAGE_DBL']))
                x_in_nonsubdivided.append(_x)
                y_in_nonsubdivided.append(_y)
            
            objects['X_IMAGE_DBL'] = x_in_nonsubdivided
            objects['Y_IMAGE_DBL'] = y_in_nonsubdivided

            orig_objects.append(np.expand_dims(objects, 1))

            if fig is not None:
                fig.savefig(f'{dirname}/orig_{opt.data_path_sciimg.split("/")[-1]}_{i}_positions.png', bbox_inches='tight')
            print(f'No. of objects [subdivision {i}] (original): {len(objects)}')
            if opt.use_beta_div:
                deconvolved, iterations, _, exec_times, errs = sgp_betaDiv(
                    subdiv.data, psf, orig_bkg, init_recon=opt.init_recon, proj_type=1,
                    stop_criterion=opt.stop_criterion, flux=np.sum(orig_fluxes_subdiv), scale_data=True,
                    save=False, ccd_sat_level=ccd_sat_level, errflag=False, obj=None, betaParam=opt.initial_beta,
                    lr=opt.initial_lr, lr_exp_param=0.1, schedule_lr=True, tol_convergence=opt.tol_convergence
                )
            else:
                deconvolved, iterations, _, exec_times, errs = sgp(
                    subdiv.data, psf, orig_bkg, init_recon=opt.init_recon, proj_type=1,
                    stop_criterion=opt.stop_criterion, flux=np.sum(orig_fluxes_subdiv), scale_data=True,
                    save=False, ccd_sat_level=ccd_sat_level, errflag=False, obj=None, tol_convergence=opt.tol_convergence
                )

            deconvolved = deconvolved.byteswap().newbyteorder()
            if opt.use_sextractor:
                fits.writeto(f'{dirname}/subdiv_deconvolved_{i}_temp.fits', deconvolved, overwrite=True)
            deconv_objects_subdiv, deconv_fluxes_subdiv, deconv_bkg, deconv_bkg_rms, fig = source_info(
                deconvolved, deconvolved.shape[1], deconvolved.shape[0],
                min_area=1, threshold=3, gain=gain, plot_positions_indicator=False,  # maskthresh=ccd_sat_level
                use_sextractor=opt.use_sextractor, image_name=f'subdiv_deconvolved_{i}_temp.fits',
                defaultFile=opt.sextractor_config_file_name.replace('orig_', 'deconv_')
            )

            x_in_nonsubdivided = []
            y_in_nonsubdivided = []
            xpeak_in_nonsubdivided = []
            ypeak_in_nonsubdivided = []
            ra_in_nonsubdivided = []
            dec_in_nonsubdivided = []

            for obj in deconv_objects_subdiv.iterrows():
                _x, _y = subdiv.to_original_position((obj[1]['X_IMAGE_DBL'], obj[1]['Y_IMAGE_DBL']))
                _xpeak, _ypeak = subdiv.to_original_position((obj[1]['XPEAK_IMAGE'], obj[1]['YPEAK_IMAGE']))
                celestial_coords = pixel_to_skycoord(_x, _y, wcs)
                celestial_coords_peak = pixel_to_skycoord(_xpeak, _ypeak, wcs)
                x_in_nonsubdivided.append(_x)
                y_in_nonsubdivided.append(_y)
                xpeak_in_nonsubdivided.append(_xpeak)
                ypeak_in_nonsubdivided.append(_ypeak)
                ra_in_nonsubdivided.append((celestial_coords.ra * u.deg).value)
                dec_in_nonsubdivided.append((celestial_coords.dec * u.deg).value)

            deconv_objects_subdiv['X_IMAGE_DBL'] = x_in_nonsubdivided
            deconv_objects_subdiv['Y_IMAGE_DBL'] = y_in_nonsubdivided
            deconv_objects_subdiv['X_IMAGE'] = x_in_nonsubdivided
            deconv_objects_subdiv['Y_IMAGE'] = y_in_nonsubdivided
            deconv_objects_subdiv['X_WORLD'] = ra_in_nonsubdivided
            deconv_objects_subdiv['Y_WORLD'] = dec_in_nonsubdivided
            deconv_objects_subdiv['XPEAK_IMAGE'] = xpeak_in_nonsubdivided
            deconv_objects_subdiv['YPEAK_IMAGE'] = ypeak_in_nonsubdivided

            deconv_objects.append(np.expand_dims(deconv_objects_subdiv, 1))
            deconv_fluxes.append(np.sum(deconv_fluxes_subdiv))
            print(f'No. of objects [subdivision {i}] (deconvolved): {len(deconv_objects_subdiv)}')
            deconv_objects_count += len(deconv_objects_subdiv)
            orig_objects_count += len(objects)
            orig_fluxes.append(np.sum(orig_fluxes_subdiv))

            print(f'iterations: {iterations}')

            if i < 10:
                fits.writeto(f'{dirname}/temp_deconvolved_image_0{i}.fits', deconvolved, header=subdiv.wcs.to_header(), overwrite=True)
                fits.writeto(f'{dirname}/temp_deconvolved_bkg_0{i}.fits', deconv_bkg, header=subdiv.wcs.to_header(), overwrite=True)
                fits.writeto(f'{dirname}/temp_deconvolved_bkgrms_0{i}.fits', deconv_bkg_rms, header=subdiv.wcs.to_header(), overwrite=True)
            else:
                fits.writeto(f'{dirname}/temp_deconvolved_image_{i}.fits', deconvolved, header=subdiv.wcs.to_header(), overwrite=True)
                fits.writeto(f'{dirname}/temp_deconvolved_bkg_{i}.fits', deconv_bkg, header=subdiv.wcs.to_header(), overwrite=True)
                fits.writeto(f'{dirname}/temp_deconvolved_bkgrms_{i}.fits', deconv_bkg_rms, header=subdiv.wcs.to_header(), overwrite=True)
            execution_times.append(exec_times[-1])

            # fig,ax=plt.subplots(1,3)
            # ax[0].imshow(subdiv.data)
            # ax[1].imshow(deconvolved)
            # ax[2].imshow(subdiv.data - deconvolved)
            # plt.show()

        # Reconstruct the subdivisions into a single image.
        t0_recon = timer()
        deconvolved, _ = reconstruct_full_image_from_patches(hdul[0].header, string_key="image")
        deconvolved_bkg, _ = reconstruct_full_image_from_patches(hdul[0].header, string_key="bkg")
        deconvolved_bkg_rms, _ = reconstruct_full_image_from_patches(hdul[0].header, string_key="bkgrms")
        t_recon = timer() - t0_recon
        print(f'Execution time [all subdivisions] + mosaicking: {np.sum(execution_times) + t_recon} seconds.')

        # Stack all the subdivision objects into a single array.
        deconv_objects = np.squeeze(np.vstack(deconv_objects))
        orig_objects = np.squeeze(np.vstack(orig_objects))
    else:
        fits.writeto(os.path.join(dirname, f'orig_{basename}'), image, overwrite=True, header=fits.getheader(opt.data_path_sciimg))

        # mask = np.ma.array(image, mask=image > ccd_sat_level).mask
        orig_objects, orig_fluxes, orig_bkg, orig_bkg_rms, fig = source_info(
            image, opt.box_width, opt.box_height, min_area=5, threshold=3, gain=gain, plot_positions_indicator=False,  #  maskthresh=ccd_sat_level
            use_sextractor=opt.use_sextractor, image_name=f'orig_{basename}', defaultFile=opt.sextractor_config_file_name
        )

        if fig is not None:
            fig.savefig(f'{dirname}/orig_{opt.data_path_sciimg.split("/")[-1]}_positions.png', bbox_inches='tight')
        print(f'No. of objects (original): {len(orig_objects)}')

        if opt.use_beta_div:
            deconvolved, iterations, _, exec_times, errs = sgp_betaDiv(
                image, psf, orig_bkg, init_recon=opt.init_recon, proj_type=1,
                stop_criterion=opt.stop_criterion, flux=np.sum(orig_fluxes), scale_data=True,
                save=False, ccd_sat_level=ccd_sat_level, errflag=False, obj=None, betaParam=opt.initial_beta,
                lr=opt.initial_lr, lr_exp_param=0.1, schedule_lr=True, tol_convergence=opt.tol_convergence
            )
        else:
            deconvolved, iterations, _, exec_times, errs = sgp(
                image, psf, orig_bkg, init_recon=opt.init_recon, proj_type=1,
                stop_criterion=opt.stop_criterion, flux=np.sum(orig_fluxes), scale_data=True,
                save=False, ccd_sat_level=ccd_sat_level, errflag=False, obj=None, tol_convergence=opt.tol_convergence
            )
        print(f'Execution time: {exec_times[-1]} seconds.')

    if opt.add_bkg_to_deconvolved:
        deconvolved += artificial_sky_background(deconvolved, min(deconvolved[deconvolved > 0])*5, gain=gain)

    ## The below two lines are only very needed for very particular cases. But here we keep it general for all cases.
    # import sep
    # sep.set_sub_object_limit(3000)
    # mask = np.ma.array(deconvolved, mask=deconvolved > ccd_sat_level).mask
    if not opt.use_subdiv:
        deconvolved = deconvolved.byteswap().newbyteorder()
        # First write deconvolved image, then update its header, then again write deconvolved image after header update.
        fits.writeto(os.path.join(dirname, f'deconvolved_{basename}'), deconvolved, overwrite=True)
        deconv_header = fits.open(os.path.join(dirname, f'deconvolved_{basename}'))[0].header
        for item in wcs.to_header().items():
            deconv_header.append(item)
        fits.writeto(os.path.join(dirname, f'deconvolved_{basename}'), deconvolved, overwrite=True, header=deconv_header)
        deconv_objects, deconv_fluxes, deconvolved_bkg, deconvolved_bkg_rms, fig = source_info(
            deconvolved, opt.box_width, opt.box_height, min_area=1, threshold=3, gain=gain, plot_positions_indicator=False,  # maskthresh=ccd_sat_level
            use_sextractor=opt.use_sextractor, image_name=f'deconvolved_{basename}',
            defaultFile=None if not opt.use_sextractor else opt.sextractor_config_file_name.replace('orig_', 'deconv_')
        )
        if fig is not None:
            fig.savefig(f'{dirname}/deconvolved_{opt.data_path_sciimg.split("/")[-1]}_positions.png', bbox_inches='tight')
        print(f'No. of objects (deconvolved): {len(deconv_objects)}')
        fits.writeto(os.path.join(dirname, f'deconv_bkg_{basename}'), deconvolved_bkg, overwrite=True)
        # TODO: Make sure below line no err is raised.
        fits.writeto(os.path.join(dirname, f'deconv_bkgrms_{basename}'), deconvolved_bkg_rms, overwrite=True)

    if opt.use_sextractor and opt.use_subdiv:
        columns = [
            'NUMBER', 'FLUX_ISO', 'FLUXERR_ISO', 'MAG_ISO', 'MAGERR_ISO', 'BACKGROUND',
            'XPEAK_IMAGE', 'YPEAK_IMAGE', 'X_IMAGE', 'Y_IMAGE', 'X_IMAGE_DBL', 'Y_IMAGE_DBL',
            'X_WORLD', 'Y_WORLD', 'A_IMAGE', 'B_IMAGE', 'THETA_IMAGE', 'ERRA_IMAGE', 'ERRB_IMAGE',
            'ERRTHETA_IMAGE', 'MU_THRESHOLD', 'FLAGS', 'FWHM_IMAGE', 'ELONGATION', 'ELLIPTICITY', 'CLASS_STAR'
        ]
        pd.DataFrame(data=orig_objects, columns=columns).to_csv(f'{os.path.join(dirname, f"orig_{basename}")}_scat_sextractor.csv')
        pd.DataFrame(data=deconv_objects, columns=columns).to_csv(f'{os.path.join(dirname, f"deconv_{basename}")}_scat_sextractor.csv')
    else:
        columns = [
            'thresh', 'npix', 'tnpix', 'xmin', 'xmax', 'ymin', 'ymax', 'x', 'y', 'x2', 'y2', 'xy',
            'errx2', 'erry2', 'errxy', 'a', 'b', 'theta', 'cxx', 'cyy', 'cxy', 'cflux', 'flux',
            'cpeak', 'peak', 'xcpeak', 'ycpeak', 'xpeak', 'ypeak', 'flag'
        ]
        pd.DataFrame(data=orig_objects, columns=columns).to_csv(f'{os.path.join(dirname, f"orig_{basename}")}_scat.csv')
        pd.DataFrame(data=deconv_objects, columns=columns).to_csv(f'{os.path.join(dirname, f"deconv_{basename}")}_scat.csv')

    print(f'Total flux (before): {np.sum(orig_fluxes)}')
    print(f'Total flux (after): {np.sum(deconv_fluxes)}')

    # Save images and results.
    if opt.use_subdiv:
        fits.writeto(os.path.join(dirname, f'subdiv_orig_{basename}'), image, overwrite=True, header=fits.getheader(opt.data_path_sciimg))

        # First write deconvolved image, then update its header, then again write deconvolved image after header update.
        fits.writeto(os.path.join(dirname, f'subdiv_deconvolved_{basename}'), deconvolved, overwrite=True)
        deconv_header = fits.open(os.path.join(dirname, f'subdiv_deconvolved_{basename}'))[0].header
        for item in wcs.to_header().items():
            deconv_header.append(item)
        fits.writeto(os.path.join(dirname, f'subdiv_deconvolved_{basename}'), deconvolved, overwrite=True, header=deconv_header)

        fits.writeto(os.path.join(dirname, f'deconv_bkg_{basename}'), deconvolved_bkg, overwrite=True)
        fits.writeto(os.path.join(dirname, f'deconv_bkgrms_{basename}'), deconvolved_bkg_rms, overwrite=True)

        # Note: If we use the subdivision approach, then the below type of background files are not needed anymore.
        for img in glob.glob('*.fits_scat_sextractor_bkg.fits'):
            os.remove(img)

    # else:
    #     fits.writeto(os.path.join(dirname, f'orig_{basename}'), image, overwrite=True, header=fits.getheader(opt.data_path_sciimg))
    #     fits.writeto(os.path.join(dirname, f'deconvolved_{basename}'), deconvolved, overwrite=True)
    #     fits.writeto(os.path.join(dirname, f'deconv_bkg_{basename}'), deconvolved_bkg, overwrite=True)
    #     fits.writeto(os.path.join(dirname, f'deconv_rms_{basename}'), deconvolved_bkg.rms(), overwrite=True)

    # fits.writeto(os.path.join(dirname, f'orig_bkg_{basename}'), orig_bkg, overwrite=True)
    # fits.writeto(os.path.join(dirname, f'orig_rms_{basename}'), orig_bkg_rms, overwrite=True)

    # Remove temporary deconvolved images.
    for img in glob.glob(f'{dirname}/temp_deconvolved_*.fits'):
        os.remove(img)
    if opt.use_sextractor:
        for img in glob.glob(f'{dirname}/subdiv_*_temp.fits'):
            os.remove(img)

    exec_times_file = f"{dirname}/execution_times.txt"
    if os.path.exists(exec_times_file) and os.stat(exec_times_file).st_size == 0:
        with open(exec_times_file, "w") as f:
            f.write(f'{opt.data_path_sciimg},{exec_times[-1]},{image.shape[1]},{image.shape[0]},{len(orig_objects)}\n')
    else:
        with open(exec_times_file, "a") as f:
            f.write(f'{opt.data_path_sciimg},{exec_times[-1]},{image.shape[1]},{image.shape[0]},{len(orig_objects)}\n')

    #if opt.plot:
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from astropy.visualization import ImageNormalize, LogStretch, ZScaleInterval

    norm = ImageNormalize(stretch=LogStretch(), interval=ZScaleInterval())
    fig, ax = plt.subplots(1, 2, figsize=(20, 15))
    im0 = ax[0].imshow(image, origin='lower', norm=norm)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im0, cax=cax, orientation='vertical')
    ax[0].set_title('(a) Original image (from ZTF)', fontsize=12)

    im2 = ax[1].imshow(deconvolved, origin='lower', norm=norm)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes('right', size='5%', pad=0.05)
    fig.colorbar(im2, cax=cax, orientation='vertical')
    ax[1].set_title('Result of deconvolution', fontsize=12)

    plt.show()
