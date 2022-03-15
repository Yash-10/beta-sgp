import glob
from astropy.io import fits
import matplotlib.pyplot as plt

from mpl_toolkits.axes_grid1 import make_axes_locatable

ORIGINAL = "SGP_original_images/*.fits"
RECONSTRUCTED = "SGP_reconstructed_images/*.fits"

for img in glob.glob(ORIGINAL):
    arr = fits.getdata(img)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.07)
    im = ax.imshow(arr, origin="lower")
    fig.colorbar(im, cax=cax, orientation='vertical')
    # ax.set_title("Original")
    plt.savefig(img+".png", bbox_inches="tight", dpi=500)
    # plt.show()
    #plt.close()

for img in glob.glob(RECONSTRUCTED):
    arr = fits.getdata(img)

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.07)
    im = ax.imshow(arr, origin="lower")
    fig.colorbar(im, cax=cax, orientation='vertical')
    # ax.set_title("Reconstructed")
    plt.savefig(img+".png", bbox_inches="tight", dpi=500)
    # plt.show()
    #plt.close()
plt.close(fig)
