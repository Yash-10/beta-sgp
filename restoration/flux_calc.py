#######################################################################

#Script to calculate flux values of all stars above the 3-sigma level.
#It is used to devise the median flux value used as a condition post SGP

#######################################################################

import sys
import glob
import numpy as np
import pandas as pd

from astropy.io import fits

from photutils.background import Background2D, MMMBackground, ModeEstimatorBackground, MeanBackground, MedianBackground
from astropy.nddata import Cutout2D

import matplotlib.pyplot as plt
import seaborn as sns

from sgp import calculate_flux

coord_files = sorted(glob.glob("cc*c.coo"))
science_imgs = sorted(glob.glob("cc*[!m]c.fits"))

with open("test_images.txt", "r") as f:
    elliptical_images = sorted([line.strip() for line in f.readlines()])

elliptical_indices = [science_imgs.index(elliptical_images[i]) for i in range(len(elliptical_images))]
elliptical_coord_files = sorted([coord_files[elliptical_indices[i]] for i in range(len(elliptical_indices))])

fluxes = []
no_3sigma = 0
for coord_list, science_img in zip(elliptical_coord_files, elliptical_images):
    arr = np.loadtxt(coord_list, skiprows=3, usecols=[0, 1])
    size = 25
    print(f"Calculating on image: {science_img}")
    for x, y in arr:
        image = fits.getdata(science_img)
        cutout = Cutout2D(image, (x, y), size, mode='partial', fill_value=sys.float_info.epsilon)
        try:
            flux = calculate_flux(cutout.data, size=25)
            fluxes.append(flux)
        except TypeError:  # If the current potential star is not above 3-sigma, leave it.
            print("The current star stamp was not above the 3-sigma level. Continuing...")
            no_3sigma += 1
            pass

fluxes = np.array(fluxes)
np.savetxt("fluxes.txt", fluxes)
print(f"Total {no_3sigma} stars were not above the 3-sigma level")