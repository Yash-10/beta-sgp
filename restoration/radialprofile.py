import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from astropy.io import fits
# from astropy.modeling.functional_models import Moffat1D
from astropy import modeling

from photutils.centroids import centroid_2dg, centroid_com
from astropy.stats import gaussian_fwhm_to_sigma

from scipy.stats import entropy, wasserstein_distance

from utils import source_info


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


# def get_stddev_from_fwhm(fwhm):
#     return fwhm / (2 * np.sqrt(2 * np.log(2)))

def fit_radprof(radprof, table):
    fitter = modeling.fitting.LevMarLSQFitter()
    # Initialize the gaussian using the amplitude of the observed radial profile. Didn't use mean/std since it gave unreliable fits.
    model = modeling.models.Gaussian1D(
        amplitude=0.8*np.max(oi), mean=0., stddev=gaussian_fwhm_to_sigma * table['fwhm'].value[0]
    )
    x = np.arange(0, 21)
    fitted_model = fitter(model, x, radprof)
    fitted_data = list(fitted_model(x))

    # Fit errors
    try:
        param_cov_matrix = fitter.fit_info["param_cov"]
        param_errs = np.absolute(param_cov_matrix.diagonal()) ** 0.5
    except:
        param_errs = np.nan

    return fitted_data, param_errs


if __name__ == "__main__":
    from sgp import DEFAULT_COLUMNS

    orig_imgs = sorted(glob.glob('../results/content/sgp_experiments/sgp_reconstruction_results/orig_cc*.fits*'))
    kl_imgs = sorted(glob.glob('../results/content/sgp_experiments/sgp_reconstruction_results/kldiv/deconv_*.fits*'))
    beta_imgs = sorted(glob.glob('../results/content/sgp_experiments/sgp_reconstruction_results/betadiv/deconv_*.fits*'))

    radprof_params_list = []
    for o, k, b in zip(orig_imgs, kl_imgs, beta_imgs):
        oi = fits.getdata(o)
        ki = fits.getdata(k)
        bi = fits.getdata(b)

        to, bkg_o = source_info(oi, localbkg_width=6)
        tk, bkg_k = source_info(ki, localbkg_width=6)
        tb, bkg_b = source_info(bi, localbkg_width=6)
        to = to.to_table(columns=DEFAULT_COLUMNS)
        tk = tk.to_table(columns=DEFAULT_COLUMNS)
        tb = tb.to_table(columns=DEFAULT_COLUMNS)

        # Reconstructed image radial profile
        oi_radprof = radial_profile(oi - bkg_o.background_median, (to['xcentroid'].value[0], to['ycentroid'].value[0]))[:21]
        ki_radprof = radial_profile(ki - bkg_k.background_median, (tk['xcentroid'].value[0], tk['ycentroid'].value[0]))[:21]
        bi_radprof = radial_profile(bi - bkg_b.background_median, (tb['xcentroid'].value[0], tb['ycentroid'].value[0]))[:21]

        # Check fitting gaussian
        fitted_o, o_errs = fit_radprof(oi_radprof, to)
        fitted_k, k_errs = fit_radprof(ki_radprof, tk)
        fitted_b, b_errs = fit_radprof(bi_radprof, tb)

        # Calculate statistical distance between actual and fitted profile.
        w_o = wasserstein_distance(oi_radprof, fitted_o)
        w_k = wasserstein_distance(ki_radprof, fitted_k)
        w_b = wasserstein_distance(bi_radprof, fitted_b)

        radprof_params_list.append(
            [
                o, oi_radprof, fitted_o, o_errs, ki_radprof, fitted_k, k_errs, bi_radprof, fitted_b, b_errs, w_o, w_k, w_b
            ]
        )

    df = pd.DataFrame(radprof_params_list)
    df.columns = [
        'image_name', 'orig_radprof', 'fitted_radprof', 'o_errs', 'kldiv_radprof', 'fitted_kldiv', 'k_errs',
        'betadiv_radprof', 'fitted_betadiv', 'b_errs', 'wasserstein_orig', 'wasserstein_kldiv', 'wasserstein_beta'
    ]
    df.to_csv("radprof_params_and_metrics.csv")
