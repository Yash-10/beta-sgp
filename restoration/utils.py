import os
import sys
import subprocess
import numpy as np
import pandas as pd

import sep
import glob
from astropy.io import fits
from astropy import modeling
# from astropy.modeling import models, fitting
from astropy.stats import gaussian_fwhm_to_sigma

from astropy.convolution import convolve, Gaussian2DKernel, Moffat2DKernel
from astropy.stats import sigma_clipped_stats, gaussian_fwhm_to_sigma, SigmaClip
from photutils.segmentation import SourceFinder

from photutils.background import Background2D, MedianBackground
from astropy.convolution import convolve
from photutils.segmentation import make_2dgaussian_kernel
from matplotlib.patches import Ellipse

from astropy.nddata import Cutout2D

from astropy.stats import sigma_clipped_stats, SigmaClip
from photutils.segmentation import detect_threshold

import ndpatch
from reproject.mosaicking import reproject_and_coadd
from reproject import reproject_exact

# from photutils.datasets import apply_poisson_noise
from photutils.utils import calc_total_error
from photutils.detection import find_peaks

from photutils.segmentation import make_source_mask, SegmentationImage, SourceCatalog, SourceFinder

from scipy.stats import wasserstein_distance

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('paper', font_scale = 2)


def degrade(image, psf):
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


# def get_quick_segmap(data, n_pixels=5):
#     """Returns a segmentation map using simple statistics. Not meant to be used for analyses.

#     Args:
#         data (_type_): _description_
#         n_pixels (int, optional): _description_. Defaults to 5.

#     Returns:
#         _type_: _description_
#     """
#     finder = SourceFinder(npixels=n_pixels, progress_bar=False, deblend=True, nproc=1)  # Can pass nproc=None to use all CPUs in a machine.

#     _, median, std = sigma_clipped_stats(data)
#     threshold = 1.5 * std

#     data_bkg_subtracted = data - median  # subtract the background
#     kernel = make_2dgaussian_kernel(1.2, size=5)  # FWHM = 1.2
#     convolved_data = convolve(data_bkg_subtracted, kernel)
#     segment_map = finder(convolved_data, threshold)

#     return segment_map

def radial_profile(data, center):
    """From https://stackoverflow.com/a/34979185. 
    `data` must be background subtracted for more accurate profiles.
    """
    x, y = np.indices((data.shape))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile.tolist()

# def radial_profile(image, center=None, stddev=False, returnradii=False, return_nr=False, 
#         binsize=0.5, weights=None, steps=False, interpnan=False, left=None, right=None,
#         mask=None ):
#     """
#     Calculate the azimuthally averaged radial profile.
#     image - The 2D image
#     center - The [x,y] pixel coordinates used as the center. The default is 
#              None, which then uses the center of the image (including 
#              fractional pixels).
#     stddev - if specified, return the azimuthal standard deviation instead of the average
#     returnradii - if specified, return (radii_array,radial_profile)
#     return_nr   - if specified, return number of pixels per radius *and* radius
#     binsize - size of the averaging bin.  Can lead to strange results if
#         non-binsize factors are used to specify the center and the binsize is
#         too large
#     weights - can do a weighted average instead of a simple average if this keyword parameter
#         is set.  weights.shape must = image.shape.  weighted stddev is undefined, so don't
#         set weights and stddev.
#     steps - if specified, will return a double-length bin array and radial
#         profile so you can plot a step-form radial profile (which more accurately
#         represents what's going on)
#     interpnan - Interpolate over NAN values, i.e. bins where there is no data?
#         left,right - passed to interpnan; they set the extrapolated values
#     mask - can supply a mask (boolean array same size as image with True for OK and False for not)
#         to average over only select data.
#     If a bin contains NO DATA, it will have a NAN value because of the
#     divide-by-sum-of-weights component.  I think this is a useful way to denote
#     lack of data, but users let me know if an alternative is prefered...
    
#     """
#     # Calculate the indices from the image
#     y, x = np.indices(image.shape)

#     if center is None:
#         center = np.array([(x.max()-x.min())/2.0, (y.max()-y.min())/2.0])

#     r = np.hypot(x - center[0], y - center[1])

#     if weights is None:
#         weights = np.ones(image.shape)
#     elif stddev:
#         raise ValueError("Weighted standard deviation is not defined.")

#     if mask is None:
#         mask = np.ones(image.shape,dtype='bool')
#     # obsolete elif len(mask.shape) > 1:
#     # obsolete     mask = mask.ravel()

#     # the 'bins' as initially defined are lower/upper bounds for each bin
#     # so that values will be in [lower,upper)  
#     nbins = int(np.round(r.max() / binsize)+1)
#     maxbin = nbins * binsize
#     bins = np.linspace(0,maxbin,nbins+1)
#     # but we're probably more interested in the bin centers than their left or right sides...
#     bin_centers = (bins[1:]+bins[:-1])/2.0

#     # how many per bin (i.e., histogram)?
#     # there are never any in bin 0, because the lowest index returned by digitize is 1
#     #nr = np.bincount(whichbin)[1:]
#     nr = np.histogram(r, bins, weights=mask.astype('int'))[0]

#     # recall that bins are from 1 to nbins (which is expressed in array terms by arange(nbins)+1 or range(1,nbins+1) )
#     # radial_prof.shape = bin_centers.shape
#     if stddev:
#         # Find out which radial bin each point in the map belongs to
#         whichbin = np.digitize(r.flat,bins)
#         # This method is still very slow; is there a trick to do this with histograms? 
#         radial_prof = np.array([image.flat[mask.flat*(whichbin==b)].std() for b in range(1,nbins+1)])
#     else: 
#         radial_prof = np.histogram(r, bins, weights=(image*weights*mask))[0] / np.histogram(r, bins, weights=(mask*weights))[0]

#     if interpnan:
#         radial_prof = np.interp(bin_centers,bin_centers[radial_prof==radial_prof],radial_prof[radial_prof==radial_prof],left=left,right=right)

#     if steps:
#         xarr = np.array(zip(bins[:-1],bins[1:])).ravel() 
#         yarr = np.array(zip(radial_prof,radial_prof)).ravel() 
#         return xarr,yarr
#     elif returnradii: 
#         return bin_centers,radial_prof
#     elif return_nr:
#         return nr,bin_centers,radial_prof
#     else:
#         return radial_prof


def fit_radprof(radprof, table):
    fitter = modeling.fitting.LevMarLSQFitter()
    # Initialize the gaussian using the amplitude of the observed radial profile. Didn't use mean/std since it gave unreliable fits.
    model = modeling.models.Gaussian1D(
        amplitude=0.8*np.max(radprof), mean=0., stddev=gaussian_fwhm_to_sigma * table['fwhm'].value[0]
    )
    alpha = 1
    gamma = table['fwhm'].value[0] / (2 * np.sqrt(2 ** (1/alpha) - 1))
    # model = modeling.models.Moffat1D(
    #     amplitude=0.8*max(radprof), alpha=alpha, gamma=gamma
    # )
    x = np.arange(0, len(radprof))
    fitted_model = fitter(model, x, radprof)
    fitted_data = fitted_model(x)

    # Fit errors
    try:
        param_cov_matrix = fitter.fit_info["param_cov"]
        param_errs = np.absolute(param_cov_matrix.diagonal()) ** 0.5
    except:
        param_errs = np.nan

    return fitted_data, param_errs

# def estimate_quick_FWHM(image):
#     bkg = Background2D(image, box_size=50)
#     threshold = detect_threshold(image, nsigma=3.0, background=bkg.background)
#     peaks = find_peaks(image, threshold=threshold, box_size=5)
#     print(peaks)
#     for row in peaks:
#         center = (row['x_peak'], row['y_peak'])
#         cutout = Cutout2D(image, center, size=50).data
#         plt.imshow(cutout)
#         plt.show()
#         radprof = radial_profile(cutout, center)
#         print(radprof)
#         # print(np.where(np.array(radprof) <= radprof[0]/2))


def source_info(data, box_size=(5, 5), n_pixels=5, sigma_threshold=1.5):
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
    bkg = Background2D(data, box_size, filter_size=(3, 3), bkg_estimator=bkg_estimator)
    data_bkg_subtracted = data - bkg.background  # subtract the background
    threshold = sigma_threshold * bkg.background_rms
    kernel = make_2dgaussian_kernel(1.2, size=3)  # FWHM = 1.2
    convolved_data = convolve(data_bkg_subtracted, kernel)
    segment_map = finder(convolved_data, threshold)

    scat = SourceCatalog(
        data_bkg_subtracted, segment_map, background=bkg.background, convolved_data=convolved_data, localbkg_width=5
    )
    return scat, bkg

def scale_psf(psf, gaussian_fwhm=1.2, size=None):  # For example, seeing_fwhm=2.53319/1.012
    """Returns a 2D Gaussian kernel by scaling the FWHM of psf.

    Args:
        psf (numpy.ndarray): The input PSF.
        gaussian_fwhm (float): FWHM (in pix) of the Gaussian kernel used to convolve the PSF.
        size (tuple): X and Y size of the Gaussian kernel used to convolve the PSF, defaults to None.

    Returns:
        scaled_psf: The scaled version of the PSF.

     Notes:
        * See this talk paper: https://reu.physics.ucsb.edu/sites/default/files/sitefiles/REUPapersTalks/2021-REUPapersTalks/Beck-Dacus-UCSB-REU-paper.pdf

    """
    if size is None:
        size = psf.shape

    kernel = make_2dgaussian_kernel(gaussian_fwhm, size=size)
    scaled_psf = convolve(psf, kernel)

    # Ensure normalization of PSF.
    scaled_psf /= scaled_psf.sum()
    return scaled_psf

# Wasserstein distance.
# Code taken from https://renkulab.io/gitlab/nathanael.perraudin/darkmattergan/-/blob/master/cosmotools/metric/evaluation.py
def wasserstein_distance_norm(p, q):
    """Computes 1-Wasserstein distance between standardized p and q arrays.

    Notes
    -----
    - p denotes ideal radial profile (Gaussian) and q denotes the radial profile of restored star stamp.
    - p and q are standardized using mean and standard deviation of p.

    Returns:
        float: 1-Wasserstein distance between two sets of radial profile.

    """
    # mu, sig = p.mean(), p.std()
    # p_norm = (p.flatten() - mu)/sig
    # q_norm = (q.flatten() - mu)/sig
    return wasserstein_distance(p, q)

def plot_positions(data_sub, objects):
    # plot background-subtracted image
    fig, ax = plt.subplots(figsize=(10, 8))
    m, s = np.mean(data_sub), np.std(data_sub)
    im = ax.imshow(data_sub, interpolation='nearest', cmap='gray',
                vmin=m-s, vmax=m+s, origin='lower')

    # plot an ellipse for each object
    for i in range(len(objects)):
        e = Ellipse(xy=(objects['x'][i], objects['y'][i]),
                    width=6*objects['a'][i],
                    height=6*objects['b'][i],
                    angle=objects['theta'][i] * 180. / np.pi)
        e.set_facecolor('none')
        e.set_edgecolor('red')
        ax.add_artist(e)

    return fig


def validation_source(image, coord, bkgmap, rmsmap, size=100):
    """This validation is mainly designed for source detection on deconvolved images.
    We sometimes observed spurious sources to be detected. This function can help guide.

    Args:
        image (_type_): image, must not be background-subtracted.
        coord (_type_): (x, y) position of the source.
        bkgmap (_type_): 2D background.
        rmsmap (_type_): 2D background RMS.

    """
    source_cutout = Cutout2D(image, coord, size=size, mode='partial', fill_value=0.0).data
    bkg = np.median(Cutout2D(bkgmap, coord, size=size, mode='partial', fill_value=0.0).data)
    rms = np.mean(Cutout2D(rmsmap, coord, size=size, mode='partial', fill_value=0.0).data)
    source_pixs = np.sort(source_cutout.flatten())[-3:].mean()

    return source_pixs > bkg + 3 * rms


def calculate_slice_bboxes(
    image_height: int,
    image_width: int,
    slice_height: int = 512,
    slice_width: int = 512,
    overlap_height_ratio: float = 0.2,
    overlap_width_ratio: float = 0.2,
):
    """
    Given the height and width of an image, calculates how to divide the image into
    overlapping slices according to the height and width provided. These slices are returned
    as bounding boxes in xyxy format.
    :param image_height: Height of the original image.
    :param image_width: Width of the original image.
    :param slice_height: Height of each slice
    :param slice_width: Width of each slice
    :param overlap_height_ratio: Fractional overlap in height of each slice (e.g. an overlap of 0.2 for a slice of size 100 yields an overlap of 20 pixels)
    :param overlap_width_ratio: Fractional overlap in width of each slice (e.g. an overlap of 0.2 for a slice of size 100 yields an overlap of 20 pixels)
    :return: a list of bounding boxes in xyxy format

    Credit: https://towardsdatascience.com/slicing-images-into-overlapping-patches-at-runtime-911fa38618d7

    """

    slice_bboxes = []
    y_max = y_min = 0
    y_overlap = int(overlap_height_ratio * slice_height)
    x_overlap = int(overlap_width_ratio * slice_width)
    while y_max < image_height:
        x_min = x_max = 0
        y_max = y_min + slice_height
        while x_max < image_width:
            x_max = x_min + slice_width
            if y_max > image_height or x_max > image_width:
                xmax = min(image_width, x_max)
                ymax = min(image_height, y_max)
                xmin = max(0, xmax - slice_width)
                ymin = max(0, ymax - slice_height)
                slice_bboxes.append([xmin, ymin, xmax, ymax])
            else:
                slice_bboxes.append([x_min, y_min, x_max, y_max])
            x_min = x_max - x_overlap
        y_min = y_max - y_overlap
    return slice_bboxes


def create_subdivisions(image, subdiv_shape=(100, 100), overlap=10, wcs=None):
    # Note: All shapes assume first entry is the height and second entry is width.
    # indices = ndpatch.get_patches_indices(image.shape, subdiv_shape, overlap)
    sliceXYXY = calculate_slice_bboxes(
        image.shape[0], image.shape[1], subdiv_shape[0], subdiv_shape[1],
        overlap/subdiv_shape[0], overlap/subdiv_shape[1]
    )
    subdivs = []
    for s in sliceXYXY:
        cutout = Cutout2D(image, ((s[0]+s[2])/2, (s[1]+s[3])/ 2), size=subdiv_shape, wcs=wcs)
        subdivs.append(cutout)
    return subdivs


def reconstruct_full_image_from_patches(output_projection_header, string_key='image', dirname='sgp_reconstruction_results'):  # string_key can be, for e.g., 'image', 'bkg', or 'bkgrms'.
    arr, footprint = reproject_and_coadd(
        [fits.open(f)[0] for f in sorted(glob.glob(f'{dirname}/temp_deconvolved_{string_key}*.fits'), key=lambda x: x[17:19])],
        output_projection=output_projection_header, reproject_function=reproject_exact, match_background=True
    )
    return arr, footprint


def artificial_sky_background(image, sky_counts, gain=1):
    """
    Generate sky background, optionally including a gradient across the image (because
    some times Moons happen).

    Parameters
    ----------

    image : numpy array
        Image whose shape the cosmic array should match.
    sky_counts : float
        The target value for the number of counts (as opposed to electrons or 
        photons) from the sky.
    gain : float, optional
        Gain of the camera, in units of electrons/ADU.

    This function is taken from https://www.astropy.org/ccd-reduction-and-photometry-guide/v/dev/notebooks/01-03-Construction-of-an-artificial-but-realistic-image.html

    """
    # Set up the random number generator, allowing a seed to be set from the environment
    seed = os.getenv('GUIDE_RANDOM_SEED', None)

    if seed is not None:
        seed = int(seed)

    # This is the generator to use for any image component which changes in each image, e.g. read noise
    # or Poisson error
    noise_rng = np.random.default_rng(seed)

    sky_im = noise_rng.poisson(sky_counts * gain, size=image.shape) / gain

    return sky_im
