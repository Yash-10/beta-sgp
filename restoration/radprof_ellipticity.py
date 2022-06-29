import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy import modeling
from astropy.nddata import Cutout2D

from photutils.aperture import CircularAperture
from photutils.aperture import aperture_photometry
from photutils.background import MedianBackground
from photutils.centroids import centroid_2dg
from photutils.morphology import data_properties  # To determine ellipticity.

from scipy.stats import entropy

def radial_profile(data, positions, length=20):
    """
    data: 2D numpy array whose radial profile needs to be calculated (must be background subtracted).
    positions: Pixel coordinate location in `data` termed as center.
    length: How many points to calculate for the radial profile, defaults to 20.

    Notes
    -----
    - It uses difference between consecutive aperture sums to get the profile.

    """
    radii = np.arange(1, length+1)
    apertures = [CircularAperture(positions, r=r) for r in radii]
    phot_table = aperture_photometry(data, apertures)
    aperture_sums = [phot_table[f"aperture_sum_{i-1}"] for i in range(1, length+1)]  # i-1 because aperture_sum_<i> starts from i = 0.
    aperture_consecutive_diffs = [aperture_sums[0]] + [t - s for s, t in zip(aperture_sums, aperture_sums[1:])]
    aperture_consecutive_diffs = [aperture_consecutive_diffs[i].data[0] for i in range(len(aperture_consecutive_diffs))]

    return aperture_consecutive_diffs

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

if __name__ == "__main__":
    save = True
    # Median flux of all star stamps from all images.
    MEDIAN_FLUX = 61169.92578125

    data = pd.read_csv('fc_sgp_params_and_metrics.csv')

    radprof_params_list = []
    for oimage, rimage in zip(sorted(glob.glob("FC_SGP_original_images/*.fits")), sorted(glob.glob("FC_SGP_reconstructed_images/*.fits"))):  # Select all reconstructed images.
        rimg = fits.getdata(rimage)
        oimg = fits.getdata(oimage)

        img_data = data[data['image'] == rimage.split('/')[1].split('_')[0]].iloc[0]
        if img_data.empty:
            raise ValueError("Empty dataframe! No match found")
        
        #### Reconstructed image radial profile ###
        recon_center = centroid_2dg(rimg - img_data['bkg_after'])
        recon_radprof = radial_profile(rimg - img_data['bkg_after'], recon_center)

        ### Original radial profile ###
        orig_center = centroid_2dg(oimg - img_data['bkg_before'])
        original_radprof = radial_profile(oimg - img_data['bkg_before'], orig_center)

        ########### Check fitting gaussian ##########
        fitter = modeling.fitting.LevMarLSQFitter()
        model_mean = np.array(original_radprof).mean()
        model_stddev = np.array(original_radprof).std()
        # Initialize the gaussian using the amplitude of the observed radial profile. Didn't use mean/std since it gave unreliable fits.
        model = modeling.models.Gaussian1D(amplitude=np.array(original_radprof).max())
        x = np.arange(0, 20)
        fitted_model = fitter(model, x, original_radprof)
        orig_fitted_data = list(fitted_model(x))

        fitter = modeling.fitting.LevMarLSQFitter()
        model_mean = np.array(recon_radprof).mean()
        model_stddev = np.array(recon_radprof).std()
        # Initialize the gaussian using the amplitude of the observed radial profile. Didn't use mean/std since it gave unreliable fits.
        model = modeling.models.Gaussian1D(amplitude=np.array(recon_radprof).max())
        x = np.arange(0, 20)
        fitted_model = fitter(model, x, recon_radprof)
        fitted_data = list(fitted_model(x))

        # Calculate statistical distance between actual and fitted profile.
        jsd_before = js_divergence(original_radprof, orig_fitted_data)
        jsd_after = js_divergence(recon_radprof, fitted_data)
        # jsd /= np.linalg.norm(fitted_data)

        # Fit errors
        try:
            param_cov_matrix = fitter.fit_info["param_cov"]
            param_errs = np.absolute(param_cov_matrix.diagonal()) ** 0.5
        except:
            param_errs = np.nan

        radprof_params_list.append(
            [
                oimage, original_radprof, orig_fitted_data, recon_radprof, fitted_data, jsd_before, jsd_after, param_errs
            ]
        )

    final_radprof_params = np.array(radprof_params_list)

    if save:
        df = pd.DataFrame(final_radprof_params)
        df.to_csv("fcsgp_radprof_params_and_metrics.csv")
