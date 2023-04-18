import random
import numpy as np
import pandas as pd
import glob
from sgp import sgp, sgp_betaDiv, DEFAULT_COLUMNS
from utils import source_info, radial_profile, fit_radprof, wasserstein_distance_norm

from astropy.nddata import Cutout2D
from astropy.io import fits

# We select all subdivisions. Here it is fine to select even the ones closer to the center.
image_list = glob.glob('M13_raw_images/ccfb*[!m]c*_*.fits')

random.seed(42)
# image_list_considered = random.sample(image_list, 100)

################## Set SGP hyperparameters ##################
from sgp import DEFAULT_PARAMS
max_projs, gamma, beta, alpha_min, alpha_max, alpha, M_alpha, tau, M = DEFAULT_PARAMS
tol_convergence = 1e-5  # Following the suggestions from the original SGP paper to reduce tolerance to prevent suppression of
USE_BETADIV = True
CROWDED = False
#############################################################

image_list_considered = ['M13_raw_images/ccfbvc310082c3_3.fits' if CROWDED else 'M13_raw_images/ccfbvc310081c1_5.fits']

# Create lists for storing results
FLUX_FRACTIONAL_DIFFERENCE, FWHM_RATIO, ELLIPTICITY_RATIO, NUM_ITERS, EXEC_TIME, ORIG_FLUX, RESTORED_FLUX = [], [], [], [], [], [], []

# Loop over all subdivisions
for image in image_list_considered:
    subdivision = image.split('/')[1].split('.fits')[0][-3:]
    star_coord_list = image.replace('.fits', '.coo')

    # If coordinate list cannot be found, continue.
    try:
        coord_list = pd.read_csv(star_coord_list, skiprows=3, header=None, delim_whitespace=True)
    except:
        continue
    coord_list.columns = [
        'x', 'y', 'approx_flux', 'local_bkg_level', 'num_saturated_pixels_in_aperture'
    ]
    img = fits.getdata(image)
    # if CROWDED:
    # #     img = img[55:355,55:355]
    if not CROWDED:
        img = img[:375, 75:]

    if CROWDED:
        psf = 'M13_raw_images/' + 'psf' + image.split('/')[1]
    else:
        psf = 'M13_raw_images/psfccfbvc310081_1_5_img.fits'

    psf = psf.replace(f'c{subdivision}', f'_{subdivision}_img')

    # If PSF cannot be found, continue.
    try:
        psf = fits.getdata(psf)
    except:
        continue

    orig_scat, orig_bkg = source_info(img, box_size=60, n_pixels=5)  # Around 7 times typical FWHM.
    orig_scat = orig_scat.to_table(columns=DEFAULT_COLUMNS)
    if CROWDED:
        orig_scat.to_pandas().to_csv('CROWDED_SUBDIV_ORIGCAT.csv')
    else:
        orig_scat.to_pandas().to_csv('SUBDIV_ORIGCAT.csv')

    if USE_BETADIV:
        rands = []
        seeds = [0, 42, 951, 93, 810]
        for i in range(5):
            np.random.seed(seeds[i])
            rands.append(
                np.random.normal(loc=1, scale=0.05)
            )
            # rands = [0.999, 0.99, 1.001, 1.01, 1.1]
        min_fwhm_ratio = np.Inf
        min_ellipticity_ratio = np.Inf
        min_wd_radial_profile_distance = np.Inf
        min_flux_fractional_difference = np.Inf
        best_beta_init = None
        for rand in rands:
            recon_img, num_iters, _, times, _= sgp_betaDiv(
                img, psf, orig_bkg.background, gamma=gamma, beta=beta, alpha_min=alpha_min,  # bkg=row['local_bkg_level']
                alpha_max=alpha_max, alpha=alpha, M_alpha=M_alpha, tau=tau, M=M, proj_type=1,
                max_projs=max_projs, init_recon=2, stop_criterion=3, save=False, verbose=True,
                flux=orig_scat['segment_flux'].value.sum(), ccd_sat_level=65000, scale_data=True,
                betaParam=rand, lr=1e-3, lr_exp_param=0.1, schedule_lr=True,
                adapt_beta=False, use_original_SGP_Afunction=False, tol_convergence=tol_convergence
            )
            _restored_scat, _ = source_info(recon_img, box_size=60, n_pixels=1)
            _restored_scat = _restored_scat.to_table(columns=DEFAULT_COLUMNS)
            flux_fractional_difference = 1 - (_restored_scat['segment_flux'].value.sum() / orig_scat['segment_flux'].value.sum())
            fwhm_ratio = np.median(_restored_scat['fwhm'].value) / np.median(orig_scat['fwhm'].value)
            ellipticity_ratio = np.median(_restored_scat['ellipticity'].value) / np.median(orig_scat['ellipticity'].value)
            if flux_fractional_difference < min_flux_fractional_difference:
                min_flux_fractional_difference = flux_fractional_difference
                best_beta_init = rand
        recon_img, num_iters, _, times, _= sgp_betaDiv(
            img, psf, orig_bkg.background, gamma=gamma, beta=beta, alpha_min=alpha_min,  # bkg=row['local_bkg_level']
            alpha_max=alpha_max, alpha=alpha, M_alpha=M_alpha, tau=tau, M=M, proj_type=1,
            max_projs=max_projs, init_recon=2, stop_criterion=3, save=False, verbose=True,
            flux=orig_scat['segment_flux'].value.sum(), ccd_sat_level=65000, scale_data=True,
            betaParam=best_beta_init, lr=1e-3, lr_exp_param=0.1, schedule_lr=True,
            adapt_beta=False, use_original_SGP_Afunction=False, tol_convergence=tol_convergence
        )
    else:
        recon_img, num_iters, _, times, _= sgp(
            img, psf, orig_bkg.background, gamma=gamma, beta=beta, alpha_min=alpha_min,  # bkg=row['local_bkg_level']
            alpha_max=alpha_max, alpha=alpha, M_alpha=M_alpha, tau=tau, M=M, proj_type=1,
            max_projs=max_projs, init_recon=2, stop_criterion=3, save=False, verbose=True,
            flux=orig_scat['segment_flux'].value.sum(), ccd_sat_level=65000, scale_data=True,
            use_original_SGP_Afunction=False, tol_convergence=tol_convergence
        )

    if USE_BETADIV:
        restored_scat, restored_bkg = source_info(recon_img, box_size=60, n_pixels=1)
        restored_scat = restored_scat.to_table(columns=DEFAULT_COLUMNS)
    else:
        restored_scat, restored_bkg = source_info(recon_img, box_size=60, n_pixels=1)
        restored_scat = restored_scat.to_table(columns=DEFAULT_COLUMNS)

    # Results
    ORIG_FLUX.append(
        orig_scat['segment_flux'].value
    )
    RESTORED_FLUX.append(
        restored_scat['segment_flux'].value
    )
    # FLUX_FRACTIONAL_DIFFERENCE.append(
    #     1 - (restored_scat['segment_flux'].value / orig_scat['segment_flux'].value)
    # )
    # FWHM_RATIO.append(
    #     restored_scat['fwhm'].value / orig_scat['fwhm'].value
    # )
    # ELLIPTICITY_RATIO.append(
    #     restored_scat['ellipticity'].value / orig_scat['ellipticity'].value
    # )
    NUM_ITERS.append(
        num_iters
    )
    EXEC_TIME.append(
        times[-1]
    )

if USE_BETADIV:
    if CROWDED:
        # np.save('SUBDIV_FLUX_FRACTIONAL_DIFFERENCE_BETA.npy', FLUX_FRACTIONAL_DIFFERENCE)
        # np.save('SUBDIV_FWHM_RATIO_BETA.npy', FWHM_RATIO)
        # np.save('SUBDIV_ELLIPTICITY_RATIO_BETA.npy', ELLIPTICITY_RATIO)
        np.save('CROWDED_SUBDIV_NUM_ITERS_BETA.npy', NUM_ITERS)
        np.save('CROWDED_SUBDIV_EXEC_TIME_BETA.npy', EXEC_TIME)
        np.save('CROWDED_SUBDIV_ORIG_FLUX_BETA.npy', ORIG_FLUX)
        np.save('CROWDED_SUBDIV_RESTORED_FLUX_BETA.npy', RESTORED_FLUX)
        fits.writeto('CROWDED_SUBDIV_ORIGIMG_BETA.fits', img, overwrite=True)
        fits.writeto('CROWDED_SUBDIV_RESTOREDIMG_BETA.fits', recon_img, overwrite=True)
        restored_scat.to_pandas().to_csv('CROWDED_SUBDIV_RESTORED_BETA.csv')
        np.save('CROWDED_SUBDIV_BEST_BETA_INIT.npy', best_beta_init)
    else:
        # np.save('SUBDIV_FLUX_FRACTIONAL_DIFFERENCE_BETA.npy', FLUX_FRACTIONAL_DIFFERENCE)
        # np.save('SUBDIV_FWHM_RATIO_BETA.npy', FWHM_RATIO)
        # np.save('SUBDIV_ELLIPTICITY_RATIO_BETA.npy', ELLIPTICITY_RATIO)
        np.save('SUBDIV_NUM_ITERS_BETA.npy', NUM_ITERS)
        np.save('SUBDIV_EXEC_TIME_BETA.npy', EXEC_TIME)
        np.save('SUBDIV_ORIG_FLUX_BETA.npy', ORIG_FLUX)
        np.save('SUBDIV_RESTORED_FLUX_BETA.npy', RESTORED_FLUX)
        fits.writeto('SUBDIV_ORIGIMG_BETA.fits', img, overwrite=True)
        fits.writeto('SUBDIV_RESTOREDIMG_BETA.fits', recon_img, overwrite=True)
        restored_scat.to_pandas().to_csv('SUBDIV_RESTORED_BETA.csv')
        np.save('SUBDIV_BEST_BETA_INIT.npy', best_beta_init)
else:
    if CROWDED:
        # np.save('SUBDIV_FLUX_FRACTIONAL_DIFFERENCE.npy', FLUX_FRACTIONAL_DIFFERENCE)
        # np.save('SUBDIV_FWHM_RATIO.npy', FWHM_RATIO)
        # np.save('SUBDIV_ELLIPTICITY_RATIO.npy', ELLIPTICITY_RATIO)
        np.save('CROWDED_SUBDIV_NUM_ITERS.npy', NUM_ITERS)
        np.save('CROWDED_SUBDIV_EXEC_TIME.npy', EXEC_TIME)
        np.save('CROWDED_SUBDIV_ORIG_FLUX.npy', ORIG_FLUX)
        np.save('CROWDED_SUBDIV_RESTORED_FLUX.npy', RESTORED_FLUX)
        fits.writeto('CROWDED_SUBDIV_ORIGIMG.fits', img, overwrite=True)
        fits.writeto('CROWDED_SUBDIV_RESTOREDIMG.fits', recon_img, overwrite=True)
        restored_scat.to_pandas().to_csv('CROWDED_SUBDIV_RESTORED.csv')
    else:
        # np.save('SUBDIV_FLUX_FRACTIONAL_DIFFERENCE.npy', FLUX_FRACTIONAL_DIFFERENCE)
        # np.save('SUBDIV_FWHM_RATIO.npy', FWHM_RATIO)
        # np.save('SUBDIV_ELLIPTICITY_RATIO.npy', ELLIPTICITY_RATIO)
        np.save('SUBDIV_NUM_ITERS.npy', NUM_ITERS)
        np.save('SUBDIV_EXEC_TIME.npy', EXEC_TIME)
        np.save('SUBDIV_ORIG_FLUX.npy', ORIG_FLUX)
        np.save('SUBDIV_RESTORED_FLUX.npy', RESTORED_FLUX)
        fits.writeto('SUBDIV_ORIGIMG.fits', img, overwrite=True)
        fits.writeto('SUBDIV_RESTOREDIMG.fits', recon_img, overwrite=True)
        restored_scat.to_pandas().to_csv('SUBDIV_RESTORED.csv')
