import os
import errno
import glob
import logging
import numpy as np
import pandas as pd
import sys
from math import floor
from functools import partial
# Use default_timer instead of timeit.timeit: Reasons here: https://stackoverflow.com/a/25823885
from timeit import default_timer as timer

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

# from sgp_validation import validate_single
from flux_conserve_proj import projectDF
from utils import (
    decide_star_cutout_size, calculate_flux, get_bkg_and_rms, source_info, get_stars
)
# from afunction import afunction

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

DEFAULT_PARAMS = (1000, 1e-4, 0.4, 1e-5, 1e5, 1e1, 3, 0.5, 1)
max_projs, gamma, beta, alpha_min, alpha_max, alpha, M_alpha, tau, M = DEFAULT_PARAMS
DEFAULT_COLUMNS = ['label', 'xcentroid', 'ycentroid', 'sky_centroid',
                   'bbox_xmin', 'bbox_xmax', 'bbox_ymin', 'bbox_ymax',
                   'area', 'semimajor_sigma', 'semiminor_sigma',
                   'orientation', 'eccentricity', 'min_value', 'max_value',
                   'local_background', 'segment_flux', 'segment_fluxerr', 'ellipticity', 'fwhm']

# from numba import njit


# @njit
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
            from sympy import diff
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


# def lr_schedule(init_lr, decay_rate, epoch):
#     return init_lr * (1 - decay_rate / 100) ** epoch

import math
def lr_schedule(init_lr, k, epoch):
    return init_lr * math.exp(-k * epoch)


def sgp(
    gn, psf, bkg, init_recon=0, proj_type=0, stop_criterion=0, MAXIT=500,
    gamma=1e-4, beta=0.4, alpha=1.3, alpha_min=1e-5, alpha_max=1e5, M_alpha=3,
    tau=0.5, M=1, max_projs=1000, clip_X_upp_bound=True, save=False,
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

    # TF = np.fft.fftn(np.fft.fftshift(psf))
    # CTF = np.conj(TF)
    # def afunction(x, TF, dimensions):
    #     x = np.reshape(x, dimensions)
    #     out = np.real(np.fft.ifftn(
    #         np.multiply(TF, np.fft.fftn(x))
    #     ))
    #     out = out.flatten()
    #     return out

    # A = partial(afunction, TF=TF, dimensions=psf.shape)
    # AT = partial(afunction, TF=CTF, dimensions=psf.shape)

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
        x = projectDF(flux, x, np.ones_like(x), scaling, ccd_sat_level=ccd_sat_level, max_projs=max_projs)

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
            print(fv)

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

def sgp_betaDiv(
    gn, psf, bkg, init_recon=0, proj_type=0, stop_criterion=0, MAXIT=500,
    gamma=1e-4, beta=0.4, alpha=1.3, alpha_min=1e-5, alpha_max=1e5, M_alpha=3,
    tau=0.5, M=1, max_projs=1000, clip_X_upp_bound=True, save=False,
    verbose=True, flux=None, ccd_sat_level=None, scale_data=True,
    betaParam=1., lr=1e-3, lr_exp_param=0.1, schedule_lr=False
):
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

    # TF = np.fft.fftn(np.fft.fftshift(psf))
    # CTF = np.conj(TF)
    # def afunction(x, TF, dimensions):
    #     x = np.reshape(x, dimensions)
    #     out = np.real(np.fft.ifftn(
    #         np.multiply(TF, np.fft.fftn(x))
    #     ))
    #     out = out.flatten()
    #     return out

    # A = partial(afunction, TF=TF, dimensions=psf.shape)
    # AT = partial(afunction, TF=CTF, dimensions=psf.shape)

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
    epoch = 0
    # for epoch in range(epochs):
    while loop:
        epoch += 1
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
            # print(fv)

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
                bgrad = betaDivDeriv(den, gn, betaParam)
                betaParam = betaParam - lr * np.sum(bgrad)
                # betaParam = np.mean(np.full(_shape, betaParam).ravel() - lr * bgrad)  # This is another option to update betaParam.

        # bgrad = betaDivDeriv(den, gn, betaParam)
        # betaParam = betaParam - lr * np.sum(bgrad)

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

        # bgrad = betaDivDeriv(den, gn, betaParam)
        # print(len(bgrads))
        # betaParam = betaParam - lr * np.mean(bgrads)

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
            reldecrease = abs(fv-Fold[M-1]) / abs(fv)
            loop = reldecrease > tol
            # if reldecrease <= tol:
            #     break
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

        if epoch == 10:
          break

    # Since calculations were done on scaled flattened images, reshape them to a 2D matrix and scale them.
    x = x.reshape(_shape)
    x = x * scaling

    discr = discr[0:iter_]
    times = times[0:iter_]
    iter_ = iter_ - 1

    print(f'beta parameter in beta-divergence (final value): {betaParam}')
    print(f'iterations: {iter_}')

    return x, iter_, discr, times


if __name__ == "__main__":
    with open('defect_images.txt') as f:
        defect_images = f.read().splitlines()

    plot = True
    useBetaDiv = True
    use_photutils_for_flux = False
    verbose = True
    save = False
    final_params_list = []
    success = 0
    failure = 0
    size = 30
    localbkg_width = 6
    offset = None
    num_stars_to_select = 6  # No. of stars to select from each image.
    
    os.mkdir('sgp_reconstruction_results')
    os.mkdir('sgp_reconstruction_results/kldiv')
    os.mkdir('sgp_reconstruction_results/betadiv')

    for image in sorted(defect_images):
        data = fits.getdata(image.split('.')[0] + 'r' + '1_2.fits')
        stars_tbl = get_stars(data)

        # Deal with some specific cases.
        if image == 'ccfbwe010079.fits':
            mask = stars_tbl['x'] == 16.26472594704304
            stars_tbl = stars_tbl[~mask]
        elif image == 'ccfbwj010030.fits':
            mask = stars_tbl['x'] == 50.35448093535504
            stars_tbl = stars_tbl[~mask]
            stars_tbl = stars_tbl[stars_tbl['x'] > 55.]
        elif image == 'ccfbwd010113.fits':
            mask = stars_tbl['x'] == 16.29232775399927
            stars_tbl = stars_tbl[~mask]

        np.random.seed(42)
        if len(stars_tbl) < num_stars_to_select:
            star_inds = np.random.choice(np.arange(0, len(stars_tbl)), size=len(stars_tbl), replace=False)
        else:
            star_inds = np.random.choice(np.arange(0, len(stars_tbl)), size=num_stars_to_select, replace=False)

        for i in star_inds:
            xc, yc = stars_tbl[i]['x'], stars_tbl[i]['y']
            print(f'Image: {image}, coordindates: x: {xc}, y: {yc}')

            # _check_cutout = Cutout2D(data, (xc, yc), size=size+30, mode='partial', fill_value=0.0, copy=True).data  # Extract slightly larger stamp for more accurate background estimation.
            cutout = Cutout2D(data, (xc, yc), size=size, mode='strict').data
            assert cutout.shape == (30, 30)

            # Estimate background on check stamp.
            # bkg, _ = get_bkg_and_rms(_check_cutout, nsigma=3.)
            # mask = make_source_mask(cutout, nsigma=2, npixels=5, dilate_size=5)

            try:
                ptb, bkg_before = source_info(cutout, localbkg_width)
                ptb = ptb.to_table(columns=DEFAULT_COLUMNS)
                before_table = ptb[np.where(ptb['area'] == ptb['area'].max())]
            except:
               continue

            if use_photutils_for_flux:
                flux_before = before_table['segment_flux'].value[0]
                flux_before_err = before_table['segment_fluxerr'].value[0]  # Note: we are not reporting error in the current implementation.
            else:
                flux_before = np.sum(cutout - bkg_before.background_median)
                flux_before_err =  None

            # Get PSF matrix.
            # TODO: Decide whether to use this or photutils ePSF method for PSF modelling.
            psf = fits.getdata(f'./psf{image.split(".")[0]}_{str(1)}_{str(2)}_img.fits')
            # Center the PSF matrix
            psf = KernelCenterer().fit_transform(psf)
            psf = np.abs(psf)
            # psf[psf<=0.] = 1e-12
            psf = psf/psf.sum()

            # Uncomment below lines if you want to use validation.
            # params = validate_single(
            #     cutout, psf, bkg, x, y, size=size,
            #     best_cutout=best_cutout, xbest=x, ybest=y
            # )
            params = DEFAULT_PARAMS
            max_projs, gamma, beta, alpha_min, alpha_max, alpha, M_alpha, tau, M = params
            print(f"\n\nParameters used: (max_projs, gamma, beta, alpha_min, alpha_max, alpha, M_alpha, tau, M) = {params}\n\n")

            if useBetaDiv:
                min_ell, min_fwhm, bestBeta = np.Inf, np.Inf, None
                for _ in range(30):
                    betaParam = np.random.uniform(low=0.95, high=1.05)
                    try:
                        recon_img, num_iters, _, execution_times = sgp_betaDiv(
                            cutout, psf, bkg_before.background_median, gamma=gamma, beta=beta, alpha_min=alpha_min,
                            alpha_max=alpha_max, alpha=alpha, M_alpha=M_alpha, tau=tau, M=M, proj_type=1,
                            max_projs=max_projs, init_recon=2, stop_criterion=3, save=True, verbose=True,
                            clip_X_upp_bound=True, flux=None, ccd_sat_level=65000, scale_data=True,
                            betaParam=betaParam, lr=1e-3, lr_exp_param=0.1, schedule_lr=True
                        )
                    except:
                        continue
                    pta, bkg_after = source_info(recon_img, localbkg_width)
                    pta = pta.to_table(columns=DEFAULT_COLUMNS)
                    after_table = pta[np.where(pta['area'] == pta['area'].max())]
                    after_ecc, after_fwhm = after_table['ellipticity'].value[0], after_table['fwhm'].value[0]

                    if after_fwhm < min_fwhm and after_ecc < min_ell and after_fwhm >= 2:  # one more condition can be added if needed: after_fwhm >= 2.
                        min_fwhm = after_fwhm
                        min_ell = after_ecc
                        bestBeta = betaParam

                if bestBeta is None:
                    bestBeta = 1.01
                print(f'best beta: {bestBeta}')

                # Run with best beta.
                recon_img, num_iters, _, execution_times = sgp_betaDiv(
                    cutout, psf, bkg_before.background_median, gamma=gamma, beta=beta, alpha_min=alpha_min,
                    alpha_max=alpha_max, alpha=alpha, M_alpha=M_alpha, tau=tau, M=M, proj_type=1,
                    max_projs=max_projs, init_recon=2, stop_criterion=3, save=True, verbose=True,
                    clip_X_upp_bound=True, flux=None, ccd_sat_level=65000, scale_data=True,
                    betaParam=bestBeta, lr=1e-3, lr_exp_param=0.1, schedule_lr=True
                )
                fits.writeto(f'sgp_reconstruction_results/betadiv/deconv_{image}_{xc}_{yc}_beta{bestBeta}', recon_img, overwrite=True)
            else:
                recon_img, num_iters, _, execution_times = sgp(
                    cutout, psf, bkg_before.background_median, gamma=gamma, beta=beta, alpha_min=alpha_min,
                    alpha_max=alpha_max, alpha=alpha, M_alpha=M_alpha, tau=tau, M=M, proj_type=1,
                    max_projs=max_projs, init_recon=2, stop_criterion=3, save=True, verbose=True,
                    clip_X_upp_bound=True, flux=None, ccd_sat_level=65000, scale_data=True
                )
                fits.writeto(f'sgp_reconstruction_results/kldiv/deconv_{image}_{xc}_{yc}', recon_img, overwrite=True)

            try:
                pta, bkg_after = source_info(recon_img, localbkg_width)
                pta = pta.to_table(columns=DEFAULT_COLUMNS)
                after_table = pta[np.where(pta['area'] == pta['area'].max())]
            except:
                continue

            if use_photutils_for_flux:
                flux_after = after_table['segment_flux'].value[0]
                flux_after_err = after_table['segment_fluxerr'].value[0]
            else:
                flux_after = np.sum(recon_img - bkg_after.background_median)
                flux_after_err = None

            print(f"Flux before: {flux_before} +- {flux_before_err}")
            print(f"Flux after: {flux_after} +- {flux_after_err}")

            before_ecc, before_fwhm = before_table['ellipticity'].value[0], before_table['fwhm'].value[0]
            after_ecc, after_fwhm = after_table['ellipticity'].value[0], after_table['fwhm'].value[0]

            before_center = (before_table['xcentroid'].value[0], before_table['ycentroid'].value[0])
            after_center = (after_table['xcentroid'].value[0], after_table['ycentroid'].value[0])
            l1_centroid_err = abs(before_center[0]-after_center[0]) + abs(before_center[1]-after_center[1])

            if verbose:
                print("\n\n")
                print(f"No. of iterations: {num_iters}")
                print(f"Execution time: {execution_times[-1]}s")
                print(f"Flux (before): {flux_before}")
                print(f"Flux (after): {flux_after}")
                print("\n\n")

            if plot:
                fig, ax = plt.subplots(1, 2)
                fig.suptitle("SGP")

                ax[0].imshow(cutout, origin="lower")
                ax[0].set_title("Original", loc="center")
                ax[1].imshow(recon_img.reshape(size, size), origin="lower")
                ax[1].set_title("Reconstructed", loc="center")

                # From https://stackoverflow.com/questions/12998430/remove-xticks-in-a-matplotlib-plot
                ax[0].tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False
                ) # labels along the bottom edge are off
                ax[1].tick_params(
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

            execution_time = np.round(np.sum(execution_times), 4)
            l1_centroid_err = np.round(l1_centroid_err, 4)

            # Update final needed parameters list.
            star_coord = (xc, yc)
            final_params_list.append(
                [image, num_iters, execution_time, star_coord, np.round(flux_before, 4), np.round(flux_after, 4), bkg_before.background_median, bkg_after.background_median, l1_centroid_err, before_ecc, after_ecc, np.round(before_fwhm, 4), np.round(after_fwhm, 4), flag, bestBeta if useBetaDiv else 1.0]
            )

    print(f"Success count: {success}")
    print(f"Failure count: {failure}")

    final_params = np.array(final_params_list)
    df = pd.DataFrame(final_params)
    df.columns = ["image", "num_iters", "execution_time", "star_coord", "flux_before", "flux_after", "bkg_before", "bkg_after", "l1_centroid_err", "before_ecc", "after_ecc", "before_fwhm (pix)", "after_fwhm (pix)", "flag", "bestBeta"]
    df.to_csv("fc_sgp_params_and_metrics_useBetaDiv.csv")

