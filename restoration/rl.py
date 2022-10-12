import os
import sys
import glob
from timeit import default_timer as timer

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.convolution import convolve_fft  # Differs from `scipy.signal.convolve` by handling NaN values differently.
from astropy.nddata import Cutout2D

from utils import source_info
from sklearn.preprocessing import KernelCenterer

from utils import (
    decide_star_cutout_size, calculate_flux, get_bkg_and_rms, source_info, get_stars
)

DEFAULT_COLUMNS = ['label', 'xcentroid', 'ycentroid', 'sky_centroid',
                   'bbox_xmin', 'bbox_xmax', 'bbox_ymin', 'bbox_ymax',
                   'area', 'semimajor_sigma', 'semiminor_sigma',
                   'orientation', 'eccentricity', 'min_value', 'max_value',
                   'local_background', 'segment_flux', 'segment_fluxerr', 'ellipticity', 'fwhm']

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


def rl(
    image, psf, bkg, max_iter=500, normalize_kernel=True, tol=1e-4, alpha=1.
):
    """
    image: Observed image.
    psf: PSF estimate for the image.
    bkg_estimate: Background estimate.
    alpha: Multiplicative relaxation factor, defaults to 1.

    """
    # eps = 1e-12  # Use eps to prevent zero division - taken from scikit-image `richardson_lucy` source code.

    min_val = min(image[image > 0])
    image[image <= 0] = min_val * np.finfo(float).eps * np.finfo(float).eps

    flux = np.sum(image) - image.size * bkg

    start = timer()
    image = image.astype(float, copy=False)
    psf = psf.astype(float, copy=False)
    deconv_img = np.full(image.shape, 0.5, dtype=float)
    # deconv_img = np.full(image.shape, flux/image.size, dtype=float)
    # deconv_img = image.copy()  # Does not work well for RL

    prev_recon_img = None  # Store previous reconstructed image.

    loop = True
    iter_ = 1
    prev_fv = -np.Inf
    while loop:
        prev_recon_img = deconv_img

        # Note: The only mode possible in convolve_fft is mode = 'same', which is the default.
        c = convolve_fft(deconv_img, psf, normalize_kernel=normalize_kernel)
        conv = c + bkg
        relative = image / conv
        deconv_img *= convolve_fft(relative, np.flip(psf), normalize_kernel=normalize_kernel) ** alpha

        fv = np.sum(np.multiply(image, np.log(relative))) + np.sum(c) - flux

        print(f"Iteration no.: {iter_-1}")
        if abs(fv - prev_fv) / abs(fv) <= tol:
            loop = False
            deconv_img = prev_recon_img
        else:
            prev_fv = fv

        iter_ += 1
        if iter_ - 1 > max_iter:
            loop = False

    end = timer()

    return deconv_img, iter_-1, end-start  # Don't select the last error since it went up and termination occured before that.


if __name__ == "__main__":
    with open('defect_images.txt') as f:
        defect_images = f.read().splitlines()

    plot = False
    use_photutils_for_flux = False
    verbose = True
    save = False
    final_params_list = []
    success = 0
    failure = 0
    size = 30
    localbkg_width = 6
    offset = None
    num_stars_to_select = 6  # No. of stars to select from each image.

    os.mkdir('rl_reconstruction_results')
    os.mkdir('rl_reconstruction_results/kldiv')

    for image in sorted(defect_images):
        data = fits.getdata(image.split('.')[0] + 'r' + '1_2.fits')
        stars_tbl = get_stars(data)

        # Deal with some specific cases.
        if image == 'ccfbwe010079.fits':
            mask = stars_tbl['x'] == 16.26472594704304
            stars_tbl = stars_tbl[~mask]
        elif image == 'ccfbwj010030.fits':
            mask = stars_tbl['x'] == 50.35448093535504
            stars_tbl = stars_tbl[~mask]
            stars_tbl = stars_tbl[stars_tbl['x'] > 55.]
        elif image == 'ccfbwd010113.fits':
            mask = stars_tbl['x'] == 16.29232775399927
            stars_tbl = stars_tbl[~mask]

        np.random.seed(42)
        if len(stars_tbl) < num_stars_to_select:
            star_inds = np.random.choice(np.arange(0, len(stars_tbl)), size=len(stars_tbl), replace=False)
        else:
            star_inds = np.random.choice(np.arange(0, len(stars_tbl)), size=num_stars_to_select, replace=False)

        for i in star_inds:
            xc, yc = stars_tbl[i]['x'], stars_tbl[i]['y']
            print(f'Image: {image}, coordindates: x: {xc}, y: {yc}')

            # _check_cutout = Cutout2D(data, (xc, yc), size=size+30, mode='partial', fill_value=0.0, copy=True).data  # Extract slightly larger stamp for more accurate background estimation.
            cutout = Cutout2D(data, (xc, yc), size=size, mode='strict').data
            assert cutout.shape == (30, 30)

            # Estimate background on check stamp.
            # bkg, _ = get_bkg_and_rms(_check_cutout, nsigma=3.)
            # mask = make_source_mask(cutout, nsigma=2, npixels=5, dilate_size=5)

            #try:
            ptb, bkg_before = source_info(cutout, localbkg_width)
            ptb = ptb.to_table(columns=DEFAULT_COLUMNS)
            before_table = ptb[np.where(ptb['area'] == ptb['area'].max())]
            #except:
            #   continue

            if use_photutils_for_flux:
                flux_before = before_table['segment_flux'].value[0]
                flux_before_err = before_table['segment_fluxerr'].value[0]  # Note: we are not reporting error in the current implementation.
            else:
                flux_before = np.sum(cutout - bkg_before.background_median)
                flux_before_err =  None

            # Get PSF matrix.
            # TODO: Decide whether to use this or photutils ePSF method for PSF modelling.
            psf = fits.getdata(f'../work/psf{image.split(".")[0]}_{str(1)}_{str(2)}_img.fits')
            # Center the PSF matrix
            psf = KernelCenterer().fit_transform(psf)
            psf = np.abs(psf)
            # psf[psf<=0.] = 1e-12
            psf = psf/psf.sum()

            recon_img, num_iters, execution_time = rl(
                cutout, psf, bkg_before.background_median, max_iter=500, tol=1e-4
            )
            fits.writeto(f'rl_reconstruction_results/kldiv/deconv_{image}_{xc}_{yc}', recon_img, overwrite=True)

            try:
                pta, bkg_after = source_info(recon_img, localbkg_width)
                pta = pta.to_table(columns=DEFAULT_COLUMNS)
                after_table = pta[np.where(pta['area'] == pta['area'].max())]
            except:
                continue

            if use_photutils_for_flux:
                flux_after = after_table['segment_flux'].value[0]
                flux_after_err = after_table['segment_fluxerr'].value[0]
            else:
                flux_after = np.sum(recon_img - bkg_after.background_median)
                flux_after_err = None

            print(f"Flux before: {flux_before} +- {flux_before_err}")
            print(f"Flux after: {flux_after} +- {flux_after_err}")

            before_ecc, before_fwhm = before_table['ellipticity'].value[0], before_table['fwhm'].value[0]
            after_ecc, after_fwhm = after_table['ellipticity'].value[0], after_table['fwhm'].value[0]

            before_center = (before_table['xcentroid'].value[0], before_table['ycentroid'].value[0])
            after_center = (after_table['xcentroid'].value[0], after_table['ycentroid'].value[0])
            l1_centroid_err = abs(before_center[0]-after_center[0]) + abs(before_center[1]-after_center[1])

            if verbose:
                print("\n\n")
                print(f"No. of iterations: {num_iters}")
                print(f"Execution time: {execution_time}s")
                print(f"Flux (before): {flux_before}")
                print(f"Flux (after): {flux_after}")
                print("\n\n")

            if plot:
                fig, ax = plt.subplots(1, 2)
                fig.suptitle("RL")

                ax[0].imshow(cutout, origin="lower")
                ax[0].set_title("Original", loc="center")
                ax[1].imshow(recon_img.reshape(size, size), origin="lower")
                ax[1].set_title("Reconstructed", loc="center")

                # From https://stackoverflow.com/questions/12998430/remove-xticks-in-a-matplotlib-plot
                ax[0].tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False
                ) # labels along the bottom edge are off
                ax[1].tick_params(
                    axis='x',          # changes apply to the x-axis
                    which='both',      # both major and minor ticks are affected
                    bottom=False,      # ticks along the bottom edge are off
                    top=False,         # ticks along the top edge are off
                    labelbottom=False
                )
                plt.show()

            ## Success/Failure based on flux criterion ##
            flux_thresh = 0.05 * flux_before
            if flux_after < flux_before + flux_thresh and flux_after > flux_before - flux_thresh:
                success += 1
                flag = 1  # Flag to denote if reconstruction is under the flux limit.
            else:
                failure += 1
                flag = 0

            print(f"Success till now: {success}")
            print(f"Failure till now: {failure}")

            execution_time = np.round(execution_time, 4)
            l1_centroid_err = np.round(l1_centroid_err, 4)

            # Update final needed parameters list.
            star_coord = (xc, yc)
            final_params_list.append(
                [image, num_iters, execution_time, star_coord, np.round(flux_before, 4), np.round(flux_after, 4), bkg_before.background_median, bkg_after.background_median, l1_centroid_err, before_ecc, after_ecc, np.round(before_fwhm, 4), np.round(after_fwhm, 4), flag]
            )

    print(f"Success count: {success}")
    print(f"Failure count: {failure}")

    final_params = np.array(final_params_list)
    df = pd.DataFrame(final_params)
    df.columns = ["image", "num_iters", "execution_time", "star_coord", "flux_before", "flux_after", "bkg_before", "bkg_after", "l1_centroid_err", "before_ecc", "after_ecc", "before_fwhm (pix)", "after_fwhm (pix)", "flag"]
    df.to_csv("rl_params_and_metrics.csv")
