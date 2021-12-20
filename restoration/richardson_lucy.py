import os
import sys
import logging
import glob
import pandas as pd
from timeit import default_timer as timer

import numpy as np

import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.convolution import convolve  # Differs from `scipy.signal.convolve` by handling NaN values differently.
from astropy.nddata import Cutout2D
from sgp import calculate_flux
from radprof_ellipticity import calculate_ellipticity_fwhm

from photutils.background import MedianBackground
from photutils.centroids import centroid_2dg


logging.basicConfig(filename='rl_restore.log', level=logging.INFO,
    format='%(asctime)s:%(funcName)s:%(message)s'
)

# def calculate_psnr(img1, img2, max_value=1):
#     """"Calculating peak signal-to-noise ratio (PSNR) between two images.
    
#     Note
#     ----
#     No longer used, kept for storage purposes.

#     """
#     print(img1.max())
#     print(img2.max())
#     mse = np.mean((np.array(img1/img1.max(), dtype=np.float32) - np.array(img2/img2.max(), dtype=np.float32)) ** 2)
#     if mse == 0:
#         return 100
#     return 20 * np.log10(max_value / (np.sqrt(mse)))

def rl(
    image, psf, bkg_estimate, max_iter=1500, normalize_kernel=True,
    best_img=None, best_coord=None, current_xy=None, flux_conserve=False,
    filename=None, save=False
):
    """Performs richardson-lucy deconvolution using the modification by Snyder 1990.

    Equation: $f^{(k+1)} = f^{(k)} \circ A^{T} \dfrac{g}{Af^{(k)} + b}$.

    image: 2D array
        Observed star stamp.
    psf: 2D array
        PSF estimate.
    bkg_estimate: float
        Background estimate.

    """
    rel_recon_errors = []  # List to store relative construction errors.
    rel_klds = []

    min_psf = min(psf[psf > 0])
    psf[psf <= 0] = min_psf * sys.float_info.epsilon * sys.float_info.epsilon

    start = timer()
    image = image.astype(float, copy=False)
    psf = psf.astype(float, copy=False)
    # Initialize reconstructed image.
    deconv_img = np.full(image.shape, 0.5, dtype=image.dtype)

    prev_rel_err = np.Inf  # Initialize previous relative reconstruction error.
    prev_recon_img = None  # Store previous reconstructed image. Needed only for `stop_criterion=2`

    # Search for nearest star in best science image.
    arr = np.loadtxt(best_coord, skiprows=3, usecols=[0, 1])
    rows = np.where((current_xy[0]-100 < arr[:, 0]) & (arr[:, 0] < current_xy[0]+100))
    new_rows = np.where((current_xy[1]-100 < arr[:, 1]) & (arr[:, 1] < current_xy[1]+100))
    
    intersect = np.intersect1d(rows, new_rows)
    if intersect.size != 0:
        extract_coord = arr[np.intersect1d(rows, new_rows)[0]]
    else:
        extract_coord = arr[rows[0][0]]

    best_img = best_img.astype(float, copy=False)
    best_section = Cutout2D(best_img, extract_coord, 25).data

    R = np.array([[0, 0.25, 0], [0.25, 0, 0.25], [0, 0.25, 0]])
    lambda_ = 0.5
    
    loop = True
    iter_ = 1
    while loop:
        prev_recon_img = deconv_img

        # boundary="extend" and normalize_kernel=True for astropy convolution.
        conv = convolve(deconv_img, psf, normalize_kernel=normalize_kernel) + bkg_estimate  # Important to add bkg_estimate, else gives severe boundary artifacts and a blurred output.
        relative = image / conv  # Handle near-zero values?
        if not flux_conserve:
            deconv_img = (1 - lambda_) * deconv_img * convolve(relative, np.flip(psf), normalize_kernel=normalize_kernel) + lambda_ * convolve(deconv_img, R, normalize_kernel=normalize_kernel)
        else:
            deconv_img *= convolve(relative, np.flip(psf), normalize_kernel=normalize_kernel)

        rel_err = np.linalg.norm(deconv_img.ravel() - best_section.ravel()) / np.linalg.norm(best_section.ravel())
        rel_recon_errors.append(rel_err)
        logging.info(f"Iteration no.: {iter_-1}, rel recon err: {rel_err}")

        if rel_err <= prev_rel_err:
            prev_rel_err = rel_err
        else:
            loop = False
            deconv_img = prev_recon_img

        iter_ += 1
        if iter_-1 > max_iter:
            loop = False

    end = timer()

    if save:
        fits.writeto(
        os.path.join(
                dirname, f"{filename}_{current_xy[0]}_{current_xy[1]}_RL_recon_{iter_-1}.fits"
            ), deconv_img
        )

    return deconv_img, rel_recon_errors[:-1], iter_-1, end-start, extract_coord  # Don't select the last error since it went up and termination occured before that.

def rl_mul_relax(image, psf, bkg_estimate, max_iter=1500, alpha=1.5,
    best_img=None, best_coord=None, current_xy=None, normalize_kernel=True
):
    """Performs richardson-lucy deconvolution with a multiplicative relaxation.

    image: 2D array
        Observed star stamp.
    psf: 2D array
        PSF estimate.
    bkg_estimate: float
        Background estimate
    alpha: float
        Multiplicative relaxation parameter.

    Notes
    -----
    - The multiplicative relaxation serves as an accelerator for RL.
    - `alpha` must be > 1. Convergence proved for `alpha < 2`.

    """
    def AT(psf, x):
        """Function representing the transposed PSF matrix.
        
        Returns a convoluted image; partially reconstructed image convolved with the transpose of PSF.

        """
        x = x.reshape(image.shape)
        conv = convolve(x, psf.T, normalize_kernel=True).ravel()
        return conv

    rel_recon_errors = []  # List to store relative construction errors.
    rel_klds = []

    psf[psf <= 0] = sys.float_info.epsilon

    start = timer()
    image = image.astype(float, copy=False)
    psf = psf.astype(float, copy=False)
    deconv_img = np.full(image.shape, 0.5, dtype=image.dtype)

    bkg = MedianBackground()
    bkg = bkg.calc_background(image)
    image -= bkg

    prev_rel_err = np.Inf  # Initialize previous relative reconstruction error.
    prev_recon_img = None  # Store previous reconstructed image. Needed only for `stop_criterion=2`

    # Search for nearest star in best science image.
    arr = np.loadtxt(best_coord, skiprows=3, usecols=[0, 1])
    rows = np.where((current_xy[0]-100 < arr[:, 0]) & (arr[:, 0] < current_xy[0]+100))
    new_rows = np.where((current_xy[1]-100 < arr[:, 1]) & (arr[:, 1] < current_xy[1]+100))
    
    intersect = np.intersect1d(rows, new_rows)
    if intersect.size != 0:
        extract_coord = arr[np.intersect1d(rows, new_rows)[0]]
    else:
        extract_coord = arr[rows[0][0]]

    best_img = best_img.astype(float, copy=False)
    best_section = Cutout2D(best_img, extract_coord, 25).data
    
    loop = True
    iter_ = 1
    while loop:
        prev_recon_img = deconv_img

        # boundary="extend" and normalize_kernel=True for astropy convolution.
        conv = convolve(deconv_img, psf, normalize_kernel=normalize_kernel) + bkg_estimate
        bkg = MedianBackground()
        bkg = bkg.calc_background(image)
        image -= bkg
        relative = image / conv  # Handle near-zero values?
        deconv_img *= convolve(relative, np.flip(psf), normalize_kernel=normalize_kernel) ** alpha

        rel_err = np.linalg.norm(deconv_img.ravel() - best_section.ravel()) / np.linalg.norm(best_section.ravel())
        rel_recon_errors.append(rel_err)
        logging.info(f"Iteration no.: {iter_-1}, rel recon err: {rel_err}")

        if rel_err <= prev_rel_err:
            prev_rel_err = rel_err
        else:
            loop = False
            deconv_img = prev_recon_img

        iter_ += 1
        if iter_-1 > max_iter:
            loop = False
    
    end = timer()

    return deconv_img, rel_recon_errors[:-1], iter_-1, end-start, extract_coord  # Don't select the last error since it went up and termination occured before that.

if __name__ == "__main__":
    # User options
    save = False
    plot = True
    verbose = True

    if save:
        dirname = "RL_reconstructed_images/"
        try:
            os.mkdir(dirname)
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise OSError("Directory already exists!")
            pass

    final_params_list = []

    # Median flux.
    MEDIAN_FLUX = 61169.92578125  # Calculate using all stamps from the dataset.
    success = 0
    failure = 0

    best_coord_file = "ccfbtf170075c.coo"

    best_sci_name = "ccfbtf170075c.fits"
    best_sci_img = fits.getdata(best_sci_name)
    best_psfs = [
                    "_PSF_BINS/psf_ccfbtf170075_1_1_img.fits", "_PSF_BINS/psf_ccfbtf170075_1_2_img.fits",
                    "_PSF_BINS/psf_ccfbtf170075_2_1_img.fits", "_PSF_BINS/psf_ccfbtf170075_2_2_img.fits"
                ]
    best_psf = np.mean([fits.getdata(psfname) for psfname in best_psfs], axis=0)

    # Best star stamp
    arr = np.loadtxt(best_coord_file, skiprows=3, usecols=[0, 1])
    size = 25
    for x, y in arr:
        # Extract star stamp.
        cutout = Cutout2D(best_sci_img, (x, y), size)
        best = cutout.data
        break  # Select only one stamp and exit.

    coord_files = sorted(glob.glob("cc*c.coo"))
    science_imgs = sorted(glob.glob("cc*[!m]c.fits"))
    psfs = sorted(glob.glob("_PSF_BINS/psf_cc*.fits"))

    # Calculate mean PSF (mean of PSFs over a single frame) as a prior for the RL method.
    mean_psfs = []
    for i in range(0, len(psfs), 4):
        data_psfs = [fits.getdata(psfs[n]) for n in range(i, i+4)]
        mean_psf = np.mean(data_psfs, axis=0)
        mean_psfs.append(mean_psf)

    with open("test_images.txt", "r") as f:
        elliptical_images = sorted([line.strip() for line in f.readlines()])

    elliptical_indices = [science_imgs.index(elliptical_images[i]) for i in range(len(elliptical_images))]
    elliptical_coord_files = sorted([coord_files[elliptical_indices[i]] for i in range(len(elliptical_indices))])
    elliptical_psfs = [mean_psfs[elliptical_indices[i]] for i in range(len(elliptical_indices))]

    if plot:
        figs = []
        # Plotting parameters
        from matplotlib import rc
        rc('font',**{'family':'sans-serif','sans-serif':['DejaVu Sans'],'size':10})
        rc('mathtext',**{'default':'regular'})

    for coord_list, science_img, psf in zip(elliptical_coord_files, elliptical_images, elliptical_psfs):
        image = fits.getdata(science_img)

        arr = np.loadtxt(coord_list, skiprows=3, usecols=[0, 1])
        size = 25

        if arr.size == 2:
            arr = np.expand_dims(arr, axis=0)
        for x, y in arr:
            # Extract star stamps.
            cutout = Cutout2D(image, (x, y), size, mode='partial', fill_value=sys.float_info.epsilon)

            # Background estimate.
            bkg = MedianBackground()
            bkg = bkg.calc_background(cutout.data)

            # RL to get the reconstrcted image.
            recon_img, rel_recon_errors, num_iters, execution_time, extract_coord = rl(
                cutout.data, psf, bkg, max_iter=1000, best_img=best_sci_img, best_coord=best_coord_file, current_xy=(x, y),
                flux_conserve=False, save=save, filename=science_img
            )

            try:
                flux_before = calculate_flux(cutout.data, size=size)
                flux_after = calculate_flux(recon_img.reshape(size, size), size=size)

                flux_thresh = 0.01 * MEDIAN_FLUX  # 1% threshold.
                if flux_after < flux_before + flux_thresh and flux_after > flux_before - flux_thresh:
                    success += 1
                    flag = 1
                else:
                    failure += 1
                    flag = 0
            except:
                flag = 0
                failure += 1

            before_center = centroid_2dg(cutout.data)
            after_center = centroid_2dg(recon_img)
            centroid_err = (before_center[0]-after_center[0], before_center[1]-after_center[1])
            l1_centroid_err = np.linalg.norm(before_center-after_center)

            # We calculate ellipticity and fwhm from the `radprof_ellipticity` module.
            before_ecc, before_fwhm = calculate_ellipticity_fwhm(cutout.data, use_moments=True)
            after_ecc, after_fwhm = calculate_ellipticity_fwhm(recon_img, use_moments=True)

            if verbose:
                logging.info("\n\n")
                logging.info(f"No. of iterations: {num_iters}")
                logging.info(f"Execution time: {execution_time}s")
                logging.info(f"Flux (before): {flux_before}")
                logging.info(f"Flux (after): {flux_after}")
                logging.info(f"Ideal stamp for relative reconstruction error from {best_sci_name} at x, y = {extract_coord}")
                logging.info(f"Centroid (before): {before_center}")
                logging.info(f"Centroid (after): {after_center}")
                logging.info(f"Centroid error (before-after) = {centroid_err}")
                logging.info("\n\n")
            if plot:
                fig, ax = plt.subplots(1, 3)
                fig.suptitle("RL")
                ax[0].imshow(cutout.data, origin='lower')
                ax[0].set_title("Original")
                ax[1].imshow(recon_img[:size*size].reshape(size, size), origin='lower')
                ax[1].set_title("Reconstructed")
                ax[2].plot(rel_recon_errors, c="black")
                ax[2].set_title("Relative reconstruction error")
                ax[2].set_xlabel("Iterations")
                ax[2].set_ylabel("Error")

                plt.tight_layout()  # Avoid overlapping labels/titles.
                plt.show()
                figs.append(fig)
            
            final_params_list.append(
                [science_img, extract_coord, rel_recon_errors, np.round(flux_before, 2), np.round(flux_after, 2), num_iters, execution_time, centroid_err, l1_centroid_err, before_ecc, after_ecc, before_fwhm, after_fwhm, flag]
            )
            
            break

    final_params = np.array(final_params_list)
    df = pd.DataFrame(final_params)
    df.to_csv("rl_params_and_metrics.csv")

    if verbose:
        logging.debug("\n")
        logging.debug(f"Success count: {success}")
        logging.debug(f"Failure count: {failure}")

    ### END ###