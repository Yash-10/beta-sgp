###############

# References: [1] https://matplotlib.org/stable/gallery/subplots_axes_and_figures/zoom_inset_axes.html

# Note: The ``.fits`` files must be in the current directory.

###############

import numpy
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.visualization import ImageNormalize, ZScaleInterval

distorted = 'ccfbvc170119r3_1.fits'
sdistorted = 's_' + distorted
trail = 'ccfbwi110033r3_1.fits'
strail = 's_' + trail
ideal = 'ccfbtf170075r3_1.fits'
sideal = 's_' + ideal

SIZE = 150

# Note: (x, y) for each of the three examples are the same => All show the same region of the sky.

################### 1. Distort ##################
distort_x_y = (156.76, 374.92)
image = fits.getdata(distorted)
cutout = Cutout2D(image, distort_x_y, size=SIZE, mode='partial')
# Use Z-scaling for better visualization.
norm = ImageNormalize(cutout.data, interval=ZScaleInterval())

# print(cutout.to_original_position(distort_x_y))

# plt.imshow(cutout.data, origin='lower', cmap='gray', norm=norm)
# plt.show()

fig, ax = plt.subplots(figsize=[5, 4])
ax.imshow(image, origin="lower", cmap="gray", norm=norm)

# inset axes....
axins = ax.inset_axes([-0.6, 0.3, 0.47, 0.47])
axins.imshow(image, origin="lower", cmap="gray", norm=norm)
# sub region of the original image
x1, x2, y1, y2 = distort_x_y[0]-SIZE/2, distort_x_y[0]+SIZE/2, distort_x_y[1]-SIZE/2, distort_x_y[1]+SIZE/2
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels([])
axins.set_yticklabels([])

ax.indicate_inset_zoom(axins, edgecolor="springgreen")

plt.savefig("ccfbvc170119_inset.png", bbox_inches="tight", dpi=500)
plt.close()

plt.imshow(image, norm=ImageNormalize(image, interval=ZScaleInterval()), cmap='gray', origin='lower')
plt.savefig("ccfbvc170119_inset.png", bbox_inches="tight", dpi=500)

sd = fits.getdata(sdistorted)
snorm = norm = ImageNormalize(sd, interval=ZScaleInterval())
plt.imshow(
    Cutout2D(sd, (231.4, 332.44), size=SIZE).data, origin='lower', cmap='gray', norm=snorm
)
plt.savefig("s_ccfbvc170119_inset.png", bbox_inches="tight", dpi=500)
plt.close()

plt.imshow(
    sd, origin='lower', cmap='gray', norm=snorm
)
plt.savefig("s_ccfbvc170119_inset.png", bbox_inches="tight", dpi=500)

# ############### 2. Trail ##############
trail_x_y = (156.76, 374.92)
image = fits.getdata(trail)
cutout = Cutout2D(image, trail_x_y, size=SIZE, mode='partial')
norm = ImageNormalize(cutout.data, interval=ZScaleInterval())

# print(cutout.to_original_position(trail_x_y))

# plt.imshow(cutout.data, origin='lower', cmap='gray', norm=norm)
# plt.show()

fig, ax = plt.subplots(figsize=[5, 4])
ax.imshow(image, origin="lower", cmap="gray", norm=norm)

# inset axes....
axins = ax.inset_axes([-0.6, 0.3, 0.47, 0.47])
axins.imshow(image, origin="lower", cmap="gray", norm=norm)
# sub region of the original image
x1, x2, y1, y2 = trail_x_y[0]-SIZE/2, trail_x_y[0]+SIZE/2, trail_x_y[1]-SIZE/2, trail_x_y[1]+SIZE/2
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels([])
axins.set_yticklabels([])

ax.indicate_inset_zoom(axins, edgecolor="springgreen")

plt.savefig("ccfbwi110033_inset.png", bbox_inches="tight", dpi=500)
plt.close()

plt.imshow(image, norm=ImageNormalize(image, interval=ZScaleInterval()), cmap='gray', origin='lower')
plt.savefig("ccfbwi110033_inset.png", bbox_inches="tight", dpi=500)

st = fits.getdata(strail)
snorm = ImageNormalize(st, interval=ZScaleInterval())
plt.imshow(
    Cutout2D(st, (231.4, 332.44), size=SIZE).data, origin='lower', cmap='gray', norm=snorm
)
plt.savefig("s_ccfbwi110033_inset.png", bbox_inches="tight", dpi=500)
plt.close()

plt.imshow(
    st, origin='lower', cmap='gray', norm=snorm
)
plt.savefig("s_ccfbwi110033_inset.png", bbox_inches="tight", dpi=500)


# ############### 3. Ideal ##############
ideal_x_y = (156.76, 374.92)
image = fits.getdata(ideal)
cutout = Cutout2D(image, ideal_x_y, size=SIZE, mode='partial')
norm = ImageNormalize(cutout.data, interval=ZScaleInterval())

fig, ax = plt.subplots(figsize=[5, 4])
ax.imshow(image, origin="lower", cmap="gray", norm=norm)

# inset axes....
axins = ax.inset_axes([-0.6, 0.3, 0.47, 0.47])
axins.imshow(image, origin="lower", cmap="gray", norm=norm)
# sub region of the original image
x1, x2, y1, y2 = ideal_x_y[0]-SIZE/2, ideal_x_y[0]+SIZE/2, ideal_x_y[1]-SIZE/2, ideal_x_y[1]+SIZE/2
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
axins.set_xticklabels([])
axins.set_yticklabels([])

ax.indicate_inset_zoom(axins, edgecolor="springgreen")

plt.savefig("ccfbtf170075_inset.png", bbox_inches="tight", dpi=500)
plt.close()

plt.imshow(image, norm=ImageNormalize(image, interval=ZScaleInterval()), cmap='gray', origin='lower')
plt.savefig("ccfbtf170075_inset.png", bbox_inches="tight", dpi=500)

si = fits.getdata(sideal)
snorm = ImageNormalize(si, interval=ZScaleInterval())
plt.imshow(
    Cutout2D(si, (231.4, 332.44), size=SIZE).data, origin='lower', cmap='gray', norm=snorm
)
plt.savefig("s_ccfbtf170075_inset.png", bbox_inches="tight", dpi=500)
plt.close()

plt.imshow(
    si, origin='lower', cmap='gray', norm=snorm
)
plt.savefig("s_ccfbtf170075_inset.png", bbox_inches="tight", dpi=500)
