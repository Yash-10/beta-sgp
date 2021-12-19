import numpy
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.visualization import ImageNormalize, ZScaleInterval

distorted = 'ccfbvc310078.fits'
trail = 'ccfbte210072.fits'
ideal = 'ccfbtf170075.fits'

SIZE = 300

################### 1. Distort ##################
distort_x_y = (477.122, 1264)
image = fits.getdata(distorted)
cutout = Cutout2D(image, distort_x_y, size=SIZE, mode='partial')
norm = ImageNormalize(cutout.data, interval=ZScaleInterval())

print(cutout.to_original_position(distort_x_y))

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

plt.savefig("ccfbvc310078_inset.png", bbox_inches="tight", dpi=500)


############### 2. Trail ##############
trail_x_y = (477.122, 1264)
image = fits.getdata(trail)
cutout = Cutout2D(image, trail_x_y, size=SIZE, mode='partial')
norm = ImageNormalize(cutout.data, interval=ZScaleInterval())

print(cutout.to_original_position(trail_x_y))

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

plt.savefig("ccfbte210072_inset.png", bbox_inches="tight", dpi=500)

############### 3. Ideal ##############
ideal_x_y = (477.122, 1264)
image = fits.getdata(ideal)
cutout = Cutout2D(image, ideal_x_y, size=SIZE, mode='partial')
norm = ImageNormalize(cutout.data, interval=ZScaleInterval())

print(cutout.to_original_position(ideal_x_y))

# plt.imshow(cutout.data, origin='lower', cmap='gray', norm=norm)
# plt.show()

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