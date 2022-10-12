from math import floor
import numpy as np

from astropy.modeling import models, fitting
from astropy.stats import gaussian_fwhm_to_sigma

from astropy.convolution import convolve, Gaussian2DKernel, Moffat2DKernel
from astropy.stats import sigma_clipped_stats, gaussian_fwhm_to_sigma, SigmaClip
from photutils.background import StdBackgroundRMS
from photutils.segmentation import make_source_mask, SegmentationImage, SourceCatalog, SourceFinder

from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel


def calculate_flux(image, bkg):
    """Calculates flux, defined as sum(pixels) - N * bkg, where N is no. of pixels.

    Args:
        image (numpy.ndarray): 1D vector representing the flattened image.
        bkg (float, numpy.ndarray): Background level. Either a single float value or a 1D vector.

    Returns:
        float: Flux.
    """
    N = image.size
    if isinstance(bkg, float):
        return image.sum() - N * bkg
    else:  # If background is also an array with different backgrounds for each pixel: this would be better to use in cases where the background inside the image varies a lot.
        return (image - bkg).sum()


def source_info(data, localbkg_width, n_pixels=5):
    """Returns a catalog for source measurements and properties.

    Args:
        data (numpy.ndarray): 2D Image.
        bkg (float, numpy.ndarray): Background level.
        segment_image (numpy.ndarray): Source (boolean) mask.
        localbkg_width (float): local background width to be used in `SourceCatalog`.
    Returns:
        photutils.segmentation.SourceCatalog: A source catalog object.

    Note
    ----
    `data` must NOT be background-subtracted.
    See https://photutils.readthedocs.io/en/stable/segmentation.html

    """
    finder = SourceFinder(npixels=n_pixels, progress_bar=False, deblend=True, nproc=1)  # Can pass nproc=None to use all CPUs in a machine.
    bkg_estimator = MedianBackground()
    bkg = Background2D(data, (5, 5), filter_size=(3, 3), bkg_estimator=bkg_estimator)
    data_bkg_subtracted = data - bkg.background  # subtract the background
    threshold = 1.5 * bkg.background_rms
    kernel = make_2dgaussian_kernel(1.2, size=3)  # FWHM = 1.2
    convolved_data = convolve(data_bkg_subtracted, kernel)
    segment_map = finder(convolved_data, threshold)

    scat = SourceCatalog(
        data_bkg_subtracted, segment_map, background=bkg.background, convolved_data=convolved_data, localbkg_width=localbkg_width
    )
    return scat, bkg


def get_which_section(x, y):
    # See https://math.stackexchange.com/questions/528501/how-to-determine-which-cell-in-a-grid-a-point-belongs-to
    N = 4
    return floor(x * N / 2048) + 1, floor(y * N / 2048) + 1


def relative_recon_error(ground_truth, image, scaling, l2=True):
    """Calculates the relative reconstruction error between `image`,
    the restored image at the kth iteration and the ground truth image.

    ground_truth: 2D array
        Image array representing the ground truth.
    image: 2D array
        Partially restored image under consideration.
    l2: bool
        Whether to use

    Idea: Any novel error metric other than simple norm distances?

    """
    ground_truth /= scaling
    if l2:
        rel_err = np.linalg.norm(image - ground_truth) / np.linalg.norm(ground_truth)
    else:
        rel_err = np.linalg.norm(image - ground_truth, ord=1) / np.linalg.norm(ground_truth, ord=1)

    return rel_err


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


def decide_star_cutout_size(data):
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
    segmap = source_info(data, localbkg_width=10, n_pixels=5)[0].segment[0]  # The last zero index is to select the first source, which is presumably the middle one, which is what we want.

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

    return approx_size + _offset, mask, _offset  # Give 2 pixel offset.


from astropy.io import fits
from photutils.background import MeanBackground, MedianBackground, StdBackgroundRMS
from photutils.detection import find_peaks
from astropy.stats import SigmaClip
from astropy.stats import sigma_clipped_stats
from astropy.nddata import NDData
from astropy.table import Table
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.psf import extract_stars
from photutils.segmentation import detect_threshold


def get_bkg_and_rms(data, nsigma):
    medbkg = MedianBackground()
    bkg = medbkg.calc_background(data)
    sigma_clip = SigmaClip(sigma=nsigma)
    bkgrms = StdBackgroundRMS(sigma_clip)
    rms_bkg = bkgrms.calc_background_rms(data)
    return bkg, rms_bkg


def get_stars(data, size=30):
    # Subtract background from data.
    nsigma = 3.
    bkg, _ = get_bkg_and_rms(data, nsigma=nsigma)
    _, _, std = sigma_clipped_stats(data, sigma=3.0)
    # # data_bkg_subtracted = data - bkg

    # thresh = detect_threshold(data_bkg_subtracted, nsigma=2, background=0.)
    # peaks_tbl = find_peaks(data_bkg_subtracted, threshold=thresh)

    hsize = (size - 1) / 2

    iraffind = IRAFStarFinder(fwhm=5.0, threshold=5.*std)
    stars_tbl = iraffind(data - bkg)  # Note: input to iraffind must be bkg-subtracted.

    x = stars_tbl['xcentroid']
    y = stars_tbl['ycentroid']

    mask = ((x > hsize) & (x < (data.shape[1] - 1 - hsize)) &
            (y > hsize) & (y < (data.shape[0] - 1 - hsize)))

    stars_table = Table()

    stars_table['x'] = x[mask]
    stars_table['y'] = y[mask]

    return stars_table
