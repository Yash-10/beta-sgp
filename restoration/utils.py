from math import floor
import numpy as np

from astropy.convolution import convolve, Gaussian2DKernel
from astropy.stats import sigma_clipped_stats, gaussian_fwhm_to_sigma, SigmaClip
from photutils.background import StdBackgroundRMS
from photutils.segmentation import make_source_mask, SegmentationImage, SourceCatalog
from photutils.utils import calc_total_error

from photutils.datasets import apply_poisson_noise


def preprocess(image, psf, bkg, readout_noise):
    # Convolve.
    final = convolve(image, psf, normalize_kernel=True, normalization_zero_tol=1e-4)
    # Add background.
    # final += bkg
    # # Now perturb the image with Poisson noise and readout noise.
    # final = apply_poisson_noise(final, seed=42)
    # Compensate for readout noise.
    # final += readout_noise ** 2
    # bkg += readout_noise ** 2
    return final

def calculate_bkg(data):
    """Calculate background level of a 2D array, `data`.

    Args:
        data (numpy.ndarray): Two-dimensional array representing the image for which background needs to be calculated.

    Returns:
        float, numpy.ndarray: Background level, source (boolean) mask defining source pixels.

    Note
    ----
    1. This function requires photutils<1.5.0 (eg: photutils==1.4.0) since `make_source_mask` is deprecated in recent versions.

    """
    mask = make_source_mask(data, nsigma=2, npixels=5, dilate_size=5)
    mean, median, std = sigma_clipped_stats(data, sigma=3.0, mask=mask)
    return median, mask

def source_info(data, bkg, segment_image, localbkg_width, gain=1.):
    """Returns a catalog for source measurements and properties.

    Args:
        data (numpy.ndarray): 2D Image.
        bkg (float, numpy.ndarray): Background level.
        segment_image (numpy.ndarray): Source (boolean) mask
        localbkg_width (float): local background width to be used in `SourceCatalog`.

    Returns:
        photutils.segmentation.SourceCatalog: A source catalog object.

    Note
    ----
    `data` must NOT be background-subtracted.

    """
    segment_image = segment_image.astype(int, copy=False)
    segment_image = SegmentationImage(segment_image)
    data_bkg_subtracted = data - bkg

    sigma = 3.0 * gaussian_fwhm_to_sigma  # FWHM = 3.
    kernel = Gaussian2DKernel(sigma, x_size=3, y_size=3)
    convolved_data = convolve(data_bkg_subtracted, kernel, normalize_kernel=True)

    sigma_clip = SigmaClip(sigma=3.0)
    bkgrms = StdBackgroundRMS(sigma_clip)
    bkgrms_value = bkgrms.calc_background_rms(data)

    effective_gain = gain  # Need to verify the value.
    error = calc_total_error(data, bkgrms_value, effective_gain)

    scat = SourceCatalog(
        data_bkg_subtracted, segment_image, convolved_data=convolved_data, error=error, localbkg_width=localbkg_width
    )
    return scat
