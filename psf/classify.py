import os
import sys
import numpy as np
import glob
import matplotlib.pyplot as plt
plt.style.use("seaborn")

from astropy.io import fits
from skimage.metrics import structural_similarity
from turbustat.statistics import PowerSpectrum


def ps_from_file_list(file_list):
    mean_psf = np.mean([fits.getdata(f) for f in file_list], axis=0)
    pspec = PowerSpectrum(mean_psf, header=hdr)  
    pspec.run(verbose=False)
    return pspec.slope

"""
defect - Either trailed or distorted.
narrow - Good PSFs
"""
classes = ["defect", "narrow"]

l = []
for file_ in sorted(glob.glob("psf*_img.fits")):
    l.append(file_)

psfs = []
names = []
for i in range(0, len(l), 16):
    p = [fits.getdata(psf) for psf in l[i:i+16]]
    mean_psf = np.mean(p, axis=0)
    names.append(l[i].split('_')[0][3:]+'.fits')
    psfs.append(mean_psf/mean_psf.sum())

hdr = fits.header.Header()  # Dummy header.
defect = []
good = []
defectnames = []
goodnames = []

################
## Hypothesis ##
################
best_seeing_images = [
    'ccfbtf170074.fits',
    'ccfbxc220072.fits',
    'ccfbxc220074.fits',
    'ccfbuh230025.fits',
    'ccfbxh060058.fits',
    'ccfbuh230026.fits',
    'ccfbuk060027.fits',
    'ccfbxc220073.fits',
    'ccfbtf170076.fits',
    'ccfbtf170075.fits'
]
best_seeing_ps_slopes = []
for im in best_seeing_images:
    best_seeing_ps_slope = ps_from_file_list(sorted(glob.glob(f"psf*{im.split('.')[0]}*_img.fits")))
    best_seeing_ps_slopes.append(best_seeing_ps_slope)

best_defect_images = [
    "ccfbue110098.fits",
    "ccfbue110099.fits",
    "ccfbvc310079.fits",
    "ccfbwd010113.fits",
    "ccfbwe010079.fits",
    "ccfbwi110036.fits",

    "ccfbvc310078.fits",
    "ccfbvc310080.fits",
    "ccfbvc310081.fits",
    "ccfbvc310082.fits",
    "ccfbvc170119.fits",
    "ccfbvc170120.fits",
    "ccfbvc170121.fits",
    "ccfbvc170118.fits"
]
best_defect_ps_slopes = []
for im in best_defect_images:
    best_defect_ps_slope = ps_from_file_list(sorted(glob.glob(f"psf*{im.split('.')[0]}*_img.fits")))
    best_defect_ps_slopes.append(best_defect_ps_slope)

for name, psf in zip(names, psfs):
    pspec = PowerSpectrum(psf, header=hdr)  
    pspec.run(verbose=False)

    if pspec.slope > np.mean(best_defect_ps_slopes) or (pspec.slope <= np.mean(best_defect_ps_slopes) and pspec.slope > np.mean(best_seeing_ps_slopes) + np.std(best_seeing_ps_slopes)):
        best_psf = np.mean([fits.getdata(f) for f in sorted(glob.glob(f"psf*ccfbtf170075*_img.fits"))], axis=0)
        ssim = structural_similarity(psf, best_psf)
        if ssim < 0.95:
            defect.append(psf)
            defectnames.append(name)
    elif pspec.slope < np.mean(best_seeing_ps_slopes) or pspec.slope < np.mean(best_defect_ps_slopes) - np.std(best_defect_ps_slopes):
        good.append(psf)
        goodnames.append(name)

defectnames = defectnames + best_defect_images
goodnames = goodnames + best_seeing_images

with open("good_images.txt", 'w') as f:
    for psf in goodnames:
        f.write(f"{psf}\n")

with open("defect_images.txt", 'w') as f:
    for psf in defectnames:
        f.write(f"{psf}\n")
