import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.modeling.functional_models import Moffat1D
from astropy import modeling
from astropy.nddata import Cutout2D

from photutils.aperture import CircularAperture
from photutils.aperture import aperture_photometry
from photutils.background import MedianBackground
from photutils.centroids import centroid_2dg, centroid_com
from photutils.morphology import data_properties  # To determine ellipticity.
from astropy.stats import gaussian_fwhm_to_sigma

from scipy.stats import entropy, wasserstein_distance
from scipy.spatial import distance

def radial_profile(data, center):
    """From https://stackoverflow.com/a/34979185"""
    x, y = np.indices((data.shape))
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile.tolist()

def calculate_ellipticity_fwhm(data, bkg, use_moments=True):
    """
    data: 2D numpy array
        Background subtracted data. Bkg is subtracted and the resulting image is used for analysis.
    use_moments: bool
        Whether to use image moments to estimate ellipticity, defaults to False.
        If False, ellipticity is estimated on a gaussian fitted on the source, currently not supported.

    """
    if use_moments:
        # Use background-subtracted data, as per photutils docs: https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.SourceCatalog.html
        properties = data_properties(data, background=bkg)
        ellipticity, fwhm = properties.ellipticity, properties.fwhm
        return ellipticity, fwhm
    else:  # Use a more sophisticated model fitting.
        raise NotImplementedError("This is not yet supported, instead use `use_moments=True`")
        # # from photutils.detection import detect_sources  # Used to extract only the source region from the stamp.
        # from astropy.modeling import models, fitting  # For fitting model.
        # from astropy.convolution import Gaussian2DKernel

        # p_init = Gaussian2DKernel(x_stddev=2.)
        # fit_p = fitting.LevMarLSQFitter()
        # p = fit_p(p_init, data_)

def js_divergence(p, q):
    """
    Modified version of the KL divergence called Jensen–Shannon divergence. Advantage: It is symmetric.

    """
    p = np.array(p)
    q = np.array(q)
    p_copy = p[p >= sys.float_info.epsilon]  # Select values only above machine epsilon to prevent JS divergence calculation to blow-up.
    q_copy = q[:len(p_copy)]

    m = 0.5 * (p_copy + q_copy)
    kld_pm = entropy(p_copy, m)
    kld_qm = entropy(q_copy, m)
    return 0.5 * kld_pm + 0.5 * kld_qm
    # return np.sum(np.where(p_copy != 0, p_copy * np.log(p_copy / q_copy), 0)) ==> KLD from scratch.

def get_stddev_from_fwhm(fwhm):
    return fwhm / (2 * np.sqrt(2 * np.log(2)))

if __name__ == "__main__":
    save = True
    # Median flux of all star stamps from all images.
    MEDIAN_FLUX = 61169.92578125

    data = pd.read_csv('sgp_params_and_metrics.csv')

    radprof_params_list = []
    for oimage, rimage in zip(sorted(glob.glob("SGP_original_images/*.fits")), sorted(glob.glob("SGP_reconstructed_images/*.fits"))):  # Select all reconstructed images.
        rimg = fits.getdata(rimage)
        oimg = fits.getdata(oimage)

        img_data = data[data['image'] == rimage.split('/')[1].split('_')[0]].iloc[0]
        if img_data.empty:
            raise ValueError("Empty dataframe! No match found")
        
        #### Reconstructed image radial profile ###
        recon_center = centroid_com(rimg - img_data['bkg_after'])
        recon_radprof = radial_profile(rimg - img_data['bkg_after'], recon_center)[:21]

        ### Original radial profile ###
        orig_center = centroid_com(oimg - img_data['bkg_before'])
        original_radprof = radial_profile(oimg - img_data['bkg_before'], orig_center)[:21]

        ########### Check fitting gaussian ##########
        fitter = modeling.fitting.LevMarLSQFitter()
        model_stddev = np.array(original_radprof).std()
        # Initialize the gaussian using the amplitude of the observed radial profile. Didn't use mean/std since it gave unreliable fits.
        model = modeling.models.Gaussian1D(amplitude=0.8*np.max(original_radprof), mean=0., stddev=gaussian_fwhm_to_sigma * img_data['after_fwhm (pix)'])  # stddev=gaussian_sigma_to_fwhm(img_data['after_fwhm (pix)'])
        x = np.arange(0, 21)
        fitted_model = fitter(model, x, original_radprof)
        orig_fitted_data = list(fitted_model(x))

        fitter = modeling.fitting.LevMarLSQFitter()
        model_stddev = np.array(recon_radprof).std()
        # Initialize the gaussian using the amplitude of the observed radial profile. Didn't use mean/std since it gave unreliable fits.
        model = modeling.models.Gaussian1D(amplitude=0.8*np.max(recon_radprof), mean=0., stddev=gaussian_fwhm_to_sigma * img_data['after_fwhm (pix)'])  # stddev=gaussian_sigma_to_fwhm(img_data['after_fwhm (pix)'])
        x = np.arange(0, 21)
        fitted_model = fitter(model, x, recon_radprof)
        fitted_data = list(fitted_model(x))

        # Calculate statistical distance between actual and fitted profile.
        w_before = wasserstein_distance(original_radprof, orig_fitted_data)
        w_after = wasserstein_distance(recon_radprof, fitted_data)

        # print(len(original_radprof))
        # print(len(recon_radprof))
        # print(len(fitted_data))
        # print(len(orig_fitted_data))

        # Fit errors
        try:
            param_cov_matrix = fitter.fit_info["param_cov"]
            param_errs = np.absolute(param_cov_matrix.diagonal()) ** 0.5
        except:
            param_errs = np.nan

        radprof_params_list.append(
            [
                oimage, original_radprof, orig_fitted_data, recon_radprof, fitted_data, w_before, w_after, param_errs
            ]
        )

    # final_radprof_params = np.array(radprof_params_list)

    if save:
        df = pd.DataFrame(radprof_params_list)
        df.to_csv("sgp_radprof_params_and_metrics.csv")
