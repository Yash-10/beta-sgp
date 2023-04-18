import random
import numpy as np
import pandas as pd
import glob
from sgp import sgp, sgp_betaDiv, DEFAULT_COLUMNS
from utils import source_info, radial_profile, fit_radprof, wasserstein_distance_norm

from astropy.nddata import Cutout2D
from astropy.io import fits


# Aim is to select subdivisions not very close to the center to ensure a single star stamp contains more or less a single star.
image_list = glob.glob('M13_raw_images/ccfb*[!m]c1_*.fits') + glob.glob('M13_raw_images/ccfb*[!m]c4_*.fits') + glob.glob('M13_raw_images/ccfb*[!m]c2_1.fits') + \
    glob.glob('M13_raw_images/ccfb*[!m]c2_2.fits') + glob.glob('M13_raw_images/ccfb*[!m]c2_4.fits') + glob.glob('M13_raw_images/ccfb*[!m]c2_5.fits') + \
    glob.glob('M13_raw_images/ccfb*[!m]c3_1.fits') + glob.glob('M13_raw_images/ccfb*[!m]c3_2.fits') + glob.glob('M13_raw_images/ccfb*[!m]c3_4.fits') + \
    glob.glob('M13_raw_images/ccfb*[!m]c3_5.fits')

random.seed(42)
image_list_considered = random.sample(image_list, 100)

################## Set SGP hyperparameters ##################
from sgp import DEFAULT_PARAMS
max_projs, gamma, beta, alpha_min, alpha_max, alpha, M_alpha, tau, M = DEFAULT_PARAMS
CUTOUT_SIZE = 31

USE_BETADIV = True
#############################################################

# Create lists for storing results
FLUX_FRACTIONAL_DIFFERENCE, FWHM_RATIO, ELLIPTICITY_RATIO, WD_RADIAL_PROFILE_DISTANCE, NUM_ITERS, EXEC_TIME, ORIG_FLUX, RESTORED_FLUX = [], [], [], [], [], [], [], []

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

    psf = 'M13_raw_images/' + 'psf' + image.split('/')[1]
    psf = psf.replace(f'c{subdivision}', f'_{subdivision}_img')

    # If PSF cannot be found, continue.
    try:
        psf = fits.getdata(psf)
    except:
        continue

    for coord_data in coord_list.iterrows():
        row = coord_data[1]
        cutout = Cutout2D(img, (row['x'], row['y']), size=CUTOUT_SIZE)

        if cutout.data.shape != (31, 31):
            continue

        orig_scat, orig_bkg = source_info(cutout.data, localbkg_width=5)
        orig_scat = orig_scat.to_table(columns=DEFAULT_COLUMNS)
        if len(orig_scat) != 1:
            continue

        if USE_BETADIV:
            rands = []
            seeds = [0, 42, 951, 93, 810]
            for i in range(5):
                np.random.seed(seeds[i])
                rands.append(
                    np.random.normal(loc=1, scale=0.05)
                )
            min_fwhm_ratio = np.Inf
            min_ellipticity_ratio = np.Inf
            min_wd_radial_profile_distance = np.Inf
            min_flux_fractional_difference = np.Inf
            best_beta_init = None
            for rand in rands:
                recon_img, num_iters, _, times, _= sgp_betaDiv(
                    cutout.data, psf, orig_bkg.background_median, gamma=gamma, beta=beta, alpha_min=alpha_min,  # bkg=row['local_bkg_level']
                    alpha_max=alpha_max, alpha=alpha, M_alpha=M_alpha, tau=tau, M=M, proj_type=1,
                    max_projs=max_projs, init_recon=2, stop_criterion=3, save=False, verbose=True,
                    flux=orig_scat['segment_flux'].value[0], ccd_sat_level=65000, scale_data=True,
                    betaParam=rand, lr=1e-3, lr_exp_param=0.1, schedule_lr=True,
                    adapt_beta=True
                )
                _restored_scat, _ = source_info(recon_img, localbkg_width=5)
                _restored_scat = _restored_scat.to_table(columns=DEFAULT_COLUMNS)
                flux_fractional_difference = 1 - (_restored_scat['segment_flux'].value[0] / orig_scat['segment_flux'].value[0])
                fwhm_ratio = _restored_scat['fwhm'].value[0] / orig_scat['fwhm'].value[0]
                ellipticity_ratio = _restored_scat['ellipticity'].value[0] / orig_scat['ellipticity'].value[0]
                if flux_fractional_difference < min_flux_fractional_difference:
                    min_flux_fractional_difference = flux_fractional_difference
                    best_beta_init = rand
            recon_img, num_iters, _, times, _= sgp_betaDiv(
                cutout.data, psf, orig_bkg.background_median, gamma=gamma, beta=beta, alpha_min=alpha_min,  # bkg=row['local_bkg_level']
                alpha_max=alpha_max, alpha=alpha, M_alpha=M_alpha, tau=tau, M=M, proj_type=1,
                max_projs=max_projs, init_recon=2, stop_criterion=3, save=False, verbose=True,
                flux=orig_scat['segment_flux'].value[0], ccd_sat_level=65000, scale_data=True,
                betaParam=best_beta_init, lr=1e-3, lr_exp_param=0.1, schedule_lr=True,
                adapt_beta=True
            )
        else:
            recon_img, num_iters, _, times, _= sgp(
                cutout.data, psf, orig_bkg.background_median, gamma=gamma, beta=beta, alpha_min=alpha_min,  # bkg=row['local_bkg_level']
                alpha_max=alpha_max, alpha=alpha, M_alpha=M_alpha, tau=tau, M=M, proj_type=1,
                max_projs=max_projs, init_recon=2, stop_criterion=3, save=False, verbose=True,
                flux=orig_scat['segment_flux'].value[0], ccd_sat_level=65000, scale_data=True
            )

        restored_scat, restored_bkg = source_info(recon_img, localbkg_width=5)
        restored_scat = restored_scat.to_table(columns=DEFAULT_COLUMNS)

        # Get radial profile of stars.
        orig_radprof = radial_profile(cutout.data-orig_bkg.background, center=(orig_scat['xcentroid'].value[0], orig_scat['ycentroid'].value[0]))
        restored_radprof = radial_profile(recon_img-restored_bkg.background, center=(restored_scat['xcentroid'].value[0], restored_scat['ycentroid'].value[0]))

        fitted_orig, _ = fit_radprof(orig_radprof, orig_scat)
        fitted_restored, _ = fit_radprof(restored_radprof, restored_scat)

        # Results
        ORIG_FLUX.append(
            orig_scat['segment_flux'].value[0]
        )
        RESTORED_FLUX.append(
            restored_scat['segment_flux'].value[0]
        )
        FLUX_FRACTIONAL_DIFFERENCE.append(
            1 - (restored_scat['segment_flux'].value[0] / orig_scat['segment_flux'].value[0])
        )
        FWHM_RATIO.append(
            restored_scat['fwhm'].value[0] / orig_scat['fwhm'].value[0]
        )
        ELLIPTICITY_RATIO.append(
            restored_scat['ellipticity'].value[0] / orig_scat['ellipticity'].value[0]
        )
        WD_RADIAL_PROFILE_DISTANCE.append(
            wasserstein_distance_norm(fitted_orig, fitted_restored)
        )
        NUM_ITERS.append(
            num_iters
        )
        EXEC_TIME.append(
            times[-1]
        )

if USE_BETADIV:
    np.save('FLUX_FRACTIONAL_DIFFERENCE_BETA.npy', FLUX_FRACTIONAL_DIFFERENCE)
    np.save('FWHM_RATIO_BETA.npy', FWHM_RATIO)
    np.save('ELLIPTICITY_RATIO_BETA.npy', ELLIPTICITY_RATIO)
    np.save('WD_RADIAL_PROFILE_DISTANCE_BETA.npy', WD_RADIAL_PROFILE_DISTANCE)
    np.save('NUM_ITERS_BETA.npy', NUM_ITERS)
    np.save('EXEC_TIME_BETA.npy', EXEC_TIME)
    np.save('ORIG_FLUX_BETA.npy', ORIG_FLUX)
    np.save('RESTORED_FLUX_BETA.npy', RESTORED_FLUX)
else:
    np.save('FLUX_FRACTIONAL_DIFFERENCE.npy', FLUX_FRACTIONAL_DIFFERENCE)
    np.save('FWHM_RATIO.npy', FWHM_RATIO)
    np.save('ELLIPTICITY_RATIO.npy', ELLIPTICITY_RATIO)
    np.save('WD_RADIAL_PROFILE_DISTANCE.npy', WD_RADIAL_PROFILE_DISTANCE)
    np.save('NUM_ITERS.npy', NUM_ITERS)
    np.save('EXEC_TIME.npy', EXEC_TIME)
    np.save('ORIG_FLUX.npy', ORIG_FLUX)
    np.save('RESTORED_FLUX.npy', RESTORED_FLUX)
