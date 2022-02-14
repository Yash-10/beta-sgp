#############################
#### Getting best_coord ####
#Right now it seems a bit of a hackery, run the below to get detected coordinates:

"""
import glob
import numpy as np

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import SigmaClip
from astropy.wcs import WCS

from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.background import MeanBackground, MedianBackground, StdBackgroundRMS
from astropy.stats import sigma_clipped_stats

medbkg = MedianBackground()
bkg = medbkg.calc_background(best_sci_img)
nsigma = 3.0
sigma_clip = SigmaClip(sigma=nsigma)
bkgrms = StdBackgroundRMS(sigma_clip)
rms_bkg = bkgrms.calc_background_rms(best_sci_img)
# For our purposes, Iraf star finder works better than Daophot on same parameters.
iraf_find = IRAFStarFinder(fwhm=5.0, threshold=bkg+nsigma*rms_bkg)
sources = iraf_find(best_sci_img - bkg)
best_coord = sources[['xcentroid', 'ycentroid']].to_pandas()
best_coord.to_csv('best_coord_photutils.csv')
"""

##############################

import os
import sys
from timeit import default_timer as timer
import glob
import numpy as np

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.stats import SigmaClip
from astropy.wcs import WCS
from astropy.convolution import Kernel2D
from astropy.modeling import models, fitting

import photutils
from photutils.detection import DAOStarFinder, IRAFStarFinder
from photutils.background import MeanBackground, MedianBackground, StdBackgroundRMS, Background2D
from astropy.stats import sigma_clipped_stats
from photutils.segmentation import make_source_mask, detect_sources

from reproject.mosaicking import reproject_and_coadd
from reproject import reproject_exact, reproject_interp

# Important: `sgp.py` in this directory must be an exact copy of sgp module from DIAPL/work.
# This is not a good behaviorial choice since import from DIAPL/work should have been allowed.
# But packaging now would not seem good...
from sgp import sgp, DEFAULT_PARAMS, calculate_flux
from sgp_validation import validate_single

import matplotlib.pyplot as plt

start = timer()

plot = False
save = True
base_dir = 'working'
# star_cutout_size = 25
psfs = sorted(glob.glob("/home/yash/DIAPL/work/_PSF_BINS/psf_cc*.fits"))

# Setup best science image for RRE
best_sci_name = f'{base_dir}/cwcs_ccfbtf170075.fits'
best_sci_img = fits.getdata(best_sci_name)
best_coord_file = 'best_coord_photutils.csv'

# Calculate mean PSF (mean of PSFs over a single frame) as a prior for the SGP method.
mean_psfs = []
for i in range(0, len(psfs), 4):
    data_psfs = [fits.getdata(psfs[n]) for n in range(i, i+4)]
    mean_psf = np.mean(data_psfs, axis=0)
    mean_psfs.append((psfs[i].split('/')[6].split('_')[1], mean_psf))

def get_subdiv_number(current_x, current_y, x, y):
    return np.where(x==current_x)[0][0] + 1, np.where(y==current_y)[0][0] + 1

def get_bkg_and_rms(data, nsigma):
    medbkg = MedianBackground()
    bkg = medbkg.calc_background(data)
    sigma_clip = SigmaClip(sigma=nsigma)
    bkgrms = StdBackgroundRMS(sigma_clip)
    rms_bkg = bkgrms.calc_background_rms(data)
    return bkg, rms_bkg

def fit_gaussian_2d(data):
    """Fits a 2D Gaussian to a star cutout and returns the model data."""
    fit_w = fitting.LevMarLSQFitter()

    y0, x0 = np.unravel_index(np.argmax(data), data.shape)
    sigma = np.std(data)
    amp = np.max(data)

    w = models.Gaussian2D(amp, x0, y0, sigma, sigma)
    print(w)
    yi, xi = np.indices(data.shape)
    g = fit_w(w, xi, yi, data)
    model_data = g(xi, yi)
    return model_data

def decide_star_cutout_size(data, nsigma=3.):
    star_mask = make_source_mask(data, nsigma=nsigma, npixels=4, dilate_size=1, sigclip_sigma=2., sigclip_iters=15)
    star_mask = star_mask.astype(float)
    i, j = np.where(star_mask==1.)
    y_extent = max(i) - min(i)
    x_extent = max(j) - min(j)
    return max(x_extent, y_extent) + 3

# # Extract WCS from calibrated image
# hdul = fits.open('calibrated_ccfbxi300024.fits')
# wcs = WCS(hdul[0].header)
# assert wcs.has_celestial  # This must be true since that's why we are copying WCS from this image to all other images.

# for fits_image in glob.glob("cc*[!m].fits"):
#     with fits.open(fits_image) as hdul:  # while `mode='update'` does this inplace, we create a new fits file to prevent overwriting the original image.
#         hdul[0].header.extend(wcs.to_header())
#         del hdul[0].header['DEC']
#         hdul.writeto(f'{base_dir}/cwcs_{fits_image}')

# # Test
# for image in glob.glob(f'{base_dir}/cwcs_*[!m].fits'):
#     assert WCS(header=fits.open(image)[0].header).has_celestial, f'WCS from {image} does not have a celestial component'

# Cut subdivisions from the newly created images.
x_centers = np.arange(128, 2048, 256)
y_centers = np.arange(128, 2048, 256)

# TODO: Can use https://photutils.readthedocs.io/en/stable/api/photutils.segmentation.deblend_sources.html#photutils.segmentation.deblend_sources

for image in ["working/cwcs_ccfbta060113.fits"]: # TODO: STart from remianing files, not from start again.
    corresponding_raw_image = image.split('/')[1].split('_')[1]
    _hdu = fits.open(image)[0]
    wcs = WCS(header=_hdu.header)
    total_stars_restored = 0
    for x in x_centers:
        for y in y_centers:
            print(f'Current X, Y pair: {x}, {y}')
            # TODO: Note, the cutouts currently have no overlap which was done in DIAPL, better to add it in the future.
            cutout = Cutout2D(_hdu.data, position=(x, y), size=256, wcs=wcs, mode='partial', fill_value=sys.float_info.epsilon)
            subdivx, subdivy = get_subdiv_number(x, y, x_centers, y_centers)
            if plot:
                plt.imshow(cutout.data)
                # Get subdivison number
                plt.title(f'{subdivx}_{subdivy}')
                plt.show()

            nsigma = 3.0

            # TODO: Question: Why use photutils when it cannot handle dense overlap whereas DIAPL can?
            bkg, rms_bkg = get_bkg_and_rms(cutout.data, nsigma=nsigma)

            # For our purposes, Iraf star finder works better than Daophot on same parameters.
            iraf_find = IRAFStarFinder(fwhm=4.0, threshold=bkg+nsigma*rms_bkg)
            sources = iraf_find(cutout.data - bkg)

            print('---SOURCES---')
            print(sources)
            print('---------------')

            if sources is None:  # No sources in this subdivision
                print('===== NO SOURCES =====')
                # If no sources, then use old subdivision as the reconstructed subdivision since finally we want to create the whole frame.
                fits.writeto(f'restored_{corresponding_raw_image}_{subdivx}_{subdivy}.fits', cutout.data, header=cutout.wcs.to_header())
                continue  # This is a rare case and mostly happens for subdivs near the edges, else we are sure to find atleast one star.

            # On this subdivision, we estimate the background level by masking followed by sigma-clipping.
            # Currelty we assume background to not vary much in this subdivision, else 2d bkg maps are preferred.
            mask = make_source_mask(cutout.data, nsigma=nsigma, npixels=5, dilate_size=1)
            mean, median, std = sigma_clipped_stats(cutout.data, sigma=3.0, mask=mask)  # median is the bkg level.
            bkg_level = mean

            #### sigma_clip = SigmaClip(sigma=3.)
            #### bkg_estimator = MedianBackground()
            #### bkg_2d = Background2D(cutout.data, box_size=(1, 1), filter_size=(1, 1), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
            #### bkg_level = bkg_2d.background

            for xcentroid, ycentroid, orig_star_fwhm in zip(sources['xcentroid'], sources['ycentroid'], sources['fwhm']):
                # TODO: Instead of hardcoding a size of 25, measure the approximate size of star and use it?
                # TODO: Before restoration remove original source and add restored source.

                # 45 is a safe size - we assume all stars are less than or equal to this size - a good assumption for this dataset.
                # We set copy=True since this cutout is only used for good cutout size estimation.
                check_star_cutout = Cutout2D(cutout.data, position=(xcentroid, ycentroid), size=(45, 45), wcs=cutout.wcs, mode='partial', fill_value=sys.float_info.epsilon, copy=True)
                star_cutout_size = decide_star_cutout_size(check_star_cutout.data, nsigma=nsigma)

                star_cutout = Cutout2D(cutout.data, position=(xcentroid, ycentroid), size=(star_cutout_size, star_cutout_size), wcs=cutout.wcs, mode='partial', fill_value=sys.float_info.epsilon)

                # only_star_region_in_cutout = np.multiply(star_cutout, mask_cutout)

                # threshold = photutils.detect_threshold(star_cutout.data, 3)
                # npixels = 5  # minimum number of connected pixels
                # segm = photutils.detect_sources(star_cutout.data, threshold, npixels)
                # if segm is None:
                #     continue
                # segm_ = segm.data.astype(float)
                # star_cutout.data[segm_==1.0] = 0.0

                print(f'\n\nFWHM: {orig_star_fwhm}\n\n')

                if plot:
                    plt.imshow(star_cutout.data)
                    plt.title(f'Position in subdivision: {xcentroid}, {ycentroid}')
                    plt.show()

                # Apply SGP on star stamps

                # Get the PSF model
                for v in range(len(mean_psfs)):
                    if mean_psfs[v][0] == image.split('_')[1].split('.')[0]:
                        psf = mean_psfs[v][1]
                        break

                ## VALIDATION ##
                max_projs, gamma, beta, alpha_min, alpha_max, alpha, M_alpha, tau, M = DEFAULT_PARAMS
                params = validate_single(star_cutout.data, psf, bkg, xcentroid, ycentroid, search_type='coarse', flux_criteria=0, size=star_cutout_size, mode='final')
                if params is None:  # If no optimal parameter that satisfied all conditions, then use default.
                    print("\n\nNo best parameter found that matches all conditions. Falling to default params\n\n")
                # max_projs, gamma, beta, alpha_min, alpha_max, alpha, M_alpha, tau, M = DEFAULT_PARAMS
                else:
                    max_projs, gamma, beta, alpha_min, alpha_max, alpha, M_alpha, tau, M = params
                    print(f"\n\nOptimal parameters: (max_projs, gamma, beta, alpha_min, alpha_max, alpha, M_alpha, tau, M) = {params}\n\n")

                # TODO: Note: The same ground-truth image seems to be used for all stamps.

                try:
                    recon_img, rel_klds, rel_recon_errors, num_iters, extract_coord, execution_time, best_section = sgp(
                        star_cutout.data, psf, bkg, gamma=gamma, beta=beta, alpha_min=alpha_min, alpha_max=alpha_max,
                        alpha=alpha, M_alpha=M_alpha, tau=tau, M=M, proj_type=1, best_img=best_sci_img, best_coord=best_coord_file,
                        max_projs=max_projs, size=star_cutout_size, init_recon=2, stop_criterion=2, current_xy=(xcentroid, ycentroid), save=False,
                        filename=image, verbose=True, clip_X_upp_bound=False, diapl=False
                    )
                except:
                    continue
                
                # fig, ax = plt.subplots(1, 2)
                # ax[0].imshow(star_cutout.data)
                # ax[1].imshow(recon_img)
                # plt.show()

                # # Print flux values to ensure they do not differ significantly.
                # try:  # If no source was detected, no point in running this.
                #     print(f'Flux before: {calculate_flux(star_cutout.data, size=star_cutout_size)}, Flux after: {calculate_flux(recon_img, size=star_cutout_size)}')
                # except:
                #     pass
                # cut = Cutout2D(star_cutout.data, position=(xcentroid, ycentroid), size=(star_cutout_size, star_cutout_size), wcs=_wcs, mode='partial', fill_value=sys.float_info.epsilon)

                # For putting the restored star onto the original star, follow these steps
                # TODO: Is kernel needed?
                # kernel = Kernel2D(array=psf)
                star_mask = make_source_mask(star_cutout.data, nsigma=nsigma, npixels=4, dilate_size=1, sigclip_sigma=2., sigclip_iters=15)

                # TODO: 20 size gave good restoration - adapt and switch between 25 and 20????

                star_mask = star_mask.astype(np.float64)

                mask_recon = make_source_mask(recon_img, nsigma=nsigma, npixels=4, dilate_size=11)
                mean, median, std = sigma_clipped_stats(recon_img, sigma=nsigma, mask=mask_recon)  # median is the bkg level.
                recon_bkg_level = mean

                recon_img -= recon_bkg_level
                recon_img += bkg_level

                # fig, ax = plt.subplots(1, 2)
                # ax[0].imshow(star_cutout.data)
                # ax[1].imshow(recon_img)
                # plt.show()

                recon_star_mask = make_source_mask(recon_img, nsigma=nsigma, npixels=4, dilate_size=1)
                recon_source = np.multiply(recon_img, recon_star_mask)
                star_mask = star_mask.astype(np.float64)
                np.putmask(star_cutout.data, star_mask==1., recon_source.astype(np.float32))
                star_cutout.data.astype(np.float64, copy=False)

                np.putmask(star_cutout.data, np.logical_and(star_mask==1., recon_star_mask==0.), bkg_level)

                # Print flux values to ensure they do not differ significantly.
                try:  # If no source was detected, no point in running this.
                    print(f'\n\nFlux before: {calculate_flux(star_cutout.data, size=star_cutout_size, nsigma=nsigma)}, Flux after: {calculate_flux(recon_img, size=star_cutout_size, nsigma=nsigma)}\n\n')
                except:
                    pass

                # plt.imshow(star_cutout.data)
                # plt.show()

                # plt.imshow(star_cutout.data)
                # plt.show()

                # if plot:
                #     fig, ax = plt.subplots(1, 2)
                #     ax[0].imshow(star_cutout.data)
                #     ax[0].set_title('Original')
                #     ax[1].imshow(recon_img)
                #     ax[1].set_title('Reconstructed')
                #     plt.show()

            # All possible stars of a subdivision are restored
            # See https://docs.astropy.org/en/stable/nddata/utils.html#saving-a-2d-cutout-to-a-fits-file-with-an-updated-wcs
            # _hdu.data = star_cutout.data
            # _hdu.header.update(star_cutout.wcs.to_header())
            # restored_cutout_filename = f'restored_{corresponding_raw_image}_{subdivx}_{subdivy}.fits'
            # _hdu.writeto(restored_cutout_filename)

            total_stars_restored += 1

            if save:
                restored_cutout_filename = f'restored_{corresponding_raw_image}_{subdivx}_{subdivy}.fits'
                fits.writeto(restored_cutout_filename, cutout.data, header=cutout.wcs.to_header())

    print(f'\n\nTtotal number of stars restored: {total_stars_restored}\n\n')
    
    if save:
        arr, footprint = reproject_and_coadd(
            [fits.open(f)[0] for f in glob.glob(f'restored_{corresponding_raw_image}_*.fits')],
            output_projection=fits.open(f'{base_dir}/cwcs_{corresponding_raw_image}')[0].header, reproject_function=reproject_exact
        )
        fits.writeto(f'mosaicked_{corresponding_raw_image}.fits', arr)

        # Remove unecessary files
        for i in  glob.glob(f'restored_{corresponding_raw_image}_*.fits'):
            os.remove(i)

print(f'Completed in {timer() - start}s')
