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
    data: 2D numpy array whose radial profile needs to be calculated.
    positions: Pixel coordinate location in `data` termed as center.
    length: How many points to calculate for the radial profile, defaults to 20.

    Notes
    -----
    - It uses difference between consecutive aperture sums to get the profile.

    """
    bkg = MedianBackground()
    bkg = bkg.calc_background(data)

    radii = np.arange(1, length+1)
    apertures = [CircularAperture(positions, r=r) for r in radii]
    phot_table = aperture_photometry(data - bkg, apertures)
    aperture_sums = [phot_table[f"aperture_sum_{i-1}"] for i in range(1, length+1)]  # i-1 because aperture_sum_<i> starts from i = 0.
    aperture_consecutive_diffs = [aperture_sums[0]] + [t - s for s, t in zip(aperture_sums, aperture_sums[1:])]
    aperture_consecutive_diffs = [aperture_consecutive_diffs[i].data[0] for i in range(len(aperture_consecutive_diffs))]

    return aperture_consecutive_diffs

def calculate_ellipticity_fwhm(data, use_moments=True):
    """
    data: 2D numpy array
        Non-background subtracted data. Bkg is subtracted and the resulting image is used for analysis.
    use_moments: bool
        Whether to use image moments to estimate ellipticity, defaults to False.
        If False, ellipticity is estimated on a gaussian fitted on the source, currently not supported.

    """
    bkg = MedianBackground()
    bkg = bkg.calc_background(data)

    if use_moments:
        # Use background-subtracted data, as per photutils docs: https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.SourceCatalog.html
        properties = data_properties(data - bkg, background=bkg)
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
    # Median flux of all star stamps from all images.
    MEDIAN_FLUX = 61169.92578125

    ########## Sample reconstructed test images ##########
    figs = []

    c = 1
    radprof_params_list = []
    for image in sorted(glob.glob("SGP_reconstructed_images/*.fits")):  # Select all reconstructed images.
        #### Reconstructed image radial profile ###
        # Flux.
        bkg = MedianBackground()
        img = fits.getdata(image)
        bkg = bkg.calc_background(img)
        flux_recon = np.sum(img) - (img.size * bkg)

        arr = fits.getdata(image)
        center = centroid_2dg(arr)
        radprof = radial_profile(arr, center)

        ### Original radial profile ###
        orig_filename = image.split("/")[1].split("_")[0]
        orig_full_img = fits.getdata(orig_filename)
        coord = float(image.split("/")[1].split("_")[1]), float(image.split("/")[1].split("_")[2])
        orig_section = Cutout2D(orig_full_img, coord, 25).data
        orig_center = centroid_2dg(orig_section)
        original_radprof = radial_profile(orig_section, orig_center)

        # Flux.
        bkg = MedianBackground()
        bkg = bkg.calc_background(orig_section)
        flux_orig = np.sum(orig_section) - (orig_section.size * bkg)

        flux_thresh = 0.01 * MEDIAN_FLUX
        if flux_recon < flux_orig + flux_thresh and flux_recon > flux_orig - flux_thresh:
            label = 1  # Success
        else:
            label = 0  # Failure

        ########### Check fitting gaussian ##########
        fitter = modeling.fitting.LevMarLSQFitter()
        model_mean = np.array(radprof).mean()
        model_stddev = np.array(radprof).std()
        # Initialize the gaussian using the amplitude of the observed radial profile. Didn't use mean/std since it gave unreliable fits.
        model = modeling.models.Gaussian1D(amplitude=np.array(radprof).max())
        print(f"Model: {model.outputs}")
        x = np.arange(0, 20)
        fitted_model = fitter(model, x, radprof)
        fitted_data = fitted_model(x)

        # Plot fitted and actual radial profile.
        fig, ax = plt.subplots(1, 1)
        ax.plot(np.arange(0, 20), fitted_data, "--", label="gaussian fit", alpha=0.8)
        ax.plot(np.arange(0, 20), radprof, label="reconstructed", alpha=0.8)
        ax.plot(np.arange(0, 20), original_radprof, label="original", alpha=0.8)

        # Calculate statistical distance between actual and fitted profile.
        jsd = js_divergence(radprof, fitted_data)
        # jsd /= np.linalg.norm(fitted_data)

        if label == 1:
            ax.set_title(f"Radial profile: {image}\nJSD: {jsd}", color="green", loc="center")
        elif label == 0:
            ax.set_title(f"Radial profile: {image}\nJSD: {jsd}", color="red", loc="center")
        ax.set_xlabel("Distance from centroid (pixels)")
        ax.legend()

        figs.append(fig)

        # Fit errors
        param_cov_matrix = fitter.fit_info["param_cov"]
        param_errs = np.absolute(param_cov_matrix.diagonal()) ** 0.5
        # print(param_errs)

        plt.show()
        plt.savefig(f"radprof_{c}.png", bbox_inches="tight")
        c += 1

        radprof_params_list.append([jsd, param_errs])

    final_radprof_params = np.array(radprof_params_list)

    df = pd.DataFrame(final_radprof_params)
    df.to_csv("radprof_params_and_metrics.csv")