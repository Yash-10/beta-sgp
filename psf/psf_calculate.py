import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from astropy.io import fits


class PSF:
    def __init__(self, txt_file):
        """Initialize a `PSF` object.

        Parameters
        ----------
        txt_file: str
            A file containing PSF attributes obtained from DIAPL.

        Notes
        -----
        For the structure of `txt_file`, see for example `psf/examples/psf_ccfbrd210048.bin.txt`

        """
        self.ldeg = 2
        self.sdeg = 1  # Unused
        with open(txt_file) as f:
            data = [float(l.rstrip("\n")) for l in f]
        self.hw = int(data[0])
        self.ndeg_spat = int(data[1])
        self.ndeg_local = int(data[2])
        self.ngauss = int(data[3])
        self.recenter = data[4]
        self.cos = data[5]
        self.sin = data[6]
        self.ax = data[7]
        self.ay = data[8]
        self.sigma_inc = data[9]
        self.sigma_mscale = data[10]
        self.fitrad = data[11]
        self.x_orig = data[12]
        self.y_orig = data[13]

        # The vector coefficients used to build the PSF model.
        self.vec_coeffs = data[14:]

        self.ntot = self.ngauss * (self.ndeg_local + 1) * (self.ndeg_local + 2) / 2;
        self.ntot *= (self.ndeg_spat + 1) * (self.ndeg_spat + 2) / 2;

    @property
    def coeffs(self):
        return self.vec_coeffs

    def calc_psf_pix(self, coeffs, x, y):
        """Calculates PSF pixel values.

        Parameters
        ----------
        coeffs: list
            List of PSF vector coefficients.
        x, y: int
            x and y are the local coordinates used to describe the PSF, in the range: [-psf.hw/2, +psf.hw/2].

        Notes
        -----
        References:
        [1] Pych, W. (2013). Difference Image Analysis Package (DIAPL2).
            Specifically, the `psf_core.c` script from the `phot` program was used.

        """
        # """Calculates PSF matrix locally, i.e. doesn't account for spatial PSF variation."""

        x1 = self.cos * x - self.sin * y
        y1 = self.sin * x + self.cos * y
        rr = self.ax * x1 * x1 + self.ay * y1 * y1

        psf_pix = 0.0
        icomp = 0

        for igauss in range(self.ngauss):
            f = np.exp(rr)
            a1 = 1.0
            for m in range(self.ldeg+1):
                a2 = 1.0
                for n in range(self.ldeg-m+1):
                    psf_pix += float(self.vec_coeffs[icomp])*f*a1*a2
                    icomp += 1
                    a2 *= y
                a1 *= x
            rr *= self.sigma_inc*self.sigma_inc

        return psf_pix

    def get_psf_mat(self):
        """Get the PSF matrix representation.

        Notes
        -----
        x and y ranges lie from [-15, 15) i.e. 31X31 because we generate 31X31 PSF matrices.

        """
        pix_locs = []
        psf_mat = np.zeros(961) # eg: 31*31 is the PSF matrix size to show.
        for i in range(-15, 15+1):
            for j in range(-15, 15+1):
                pix_locs.append((i, j))
                idx = j + self.hw + 31 * (i + self.hw)
                psf_mat[idx] = self.calc_psf_pix(self.vec_coeffs, j, i)
        # for i, pix_loc in enumerate(pix_locs):
        #     psf_mat[i] = self.calc_psf_pix(self.vec_coeffs, *pix_loc)
        self.psf_mat = psf_mat.reshape(31, 31)

        return self.psf_mat

    def show_psf_mat(self):
        """Shows the PSF as an image."""
        mat = self.get_psf_mat()
        plt.matshow(mat, origin='lower')
        plt.colorbar()
        plt.show()

    def check_symmetric(self, coeffs, rtol=1e-05, atol=1e-08):
        """Check if the matrix `coeffs` is symmetric or not. A helper function.

        Parameters
        ----------
        coeffs: 2D array.

        """
        return np.allclose(coeffs, coeffs.T, rtol=rtol, atol=atol)

    def normalize_psf_mat(self):
        """Normalizes the 2D PSF such that all pixels sum up to 1.
        Thic could be helpful if the brightness of the convolved image must not be changed.

        A helper function.

        """
        mat = self.get_psf_mat()
        mat = mat / np.sum(mat)
        return mat

    def init_psf(self, xpsf, ypsf):
        """Calculates the initial circular fit to the PSF.

        xpsf, ypsf: float
            The `xfit` and `yfit` point of a PSF object i.e. a single star.

        Notes
        -----
        Currently this cannot be used since we don't yet have candidate PSF object database.

        """
        ncomp = self.ngauss * (self.ldeg+1) * (self.ldeg + 2) / 2
        local_vec = [0.0] * ncomp
        # for icomp in range(ncomp):
        #     local_vec[icomp] = 0.0

        itot = 0
        a1 = 1.0
        for m in range(self.sdeg+1):
            a2 = 1.0
            for n in range(self.sdeg-m+1):
                for icomp in range(ncomp):
                    local_vec[icomp] += self.vec_coeffs[itot] * a1 * a2
                    itot += 1
                a2 *= ypsf - self.y_orig
            a1 *= xpsf - self.x_orig


if __name__ == "__main__":
    ### Draw PSF subplots for visualization ###
    ### PSF bin txt files must be present in the current working directory ###
    mats = []
    titles = []
    for i, file_ in enumerate(glob.glob("psf*.txt")):
        if file_.endswith(".txt"):
            print(file_)
            psf = PSF(file_)
            mat = psf.normalize_psf_mat()
            fits.writeto(file_.split(".")[0]+"_img.fits", mat, overwrite=True)
            mats.append(psf.get_psf_mat())
            titles.append(file_.split(".")[0])

    # # print(len(mats))
    # fig, ax = plt.subplots(3, 3, figsize=(5, 5))

    # fig.tight_layout()

    # ax[0, 0].imshow(mats[0], cmap="gray")
    # ax[0, 0].set_title(titles[0])
    # ax[0, 1].imshow(mats[1], cmap="gray")
    # ax[0, 1].set_title(titles[1])
    # ax[0, 2].imshow(mats[2], cmap="gray")
    # ax[0, 2].set_title(titles[2])
    # ax[1, 0].imshow(mats[3], cmap="gray")
    # ax[1, 0].set_title(titles[3])
    # ax[1, 1].imshow(mats[2], cmap="gray")
    # ax[1, 1].set_title(titles[4])
    # ax[1, 2].imshow(mats[3], cmap="gray")
    # ax[1, 2].set_title(titles[5])
    # ax[2, 0].imshow(mats[3], cmap="gray")
    # ax[2, 0].set_title(titles[6])
    # ax[2, 1].imshow(mats[2], cmap="gray")
    # ax[2, 1].set_title(titles[7])
    # ax[2, 2].imshow(mats[3], cmap="gray")
    # ax[2, 2].set_title(titles[8])
    # plt.show()

    # # Example: The image corresponding to this PSF had the worst FWHM estimate from fwhms.bash (~11)
    # plt.imshow(PSF("psfs/psf_ccfbvb230022.bin.txt").get_psf_mat(), cmap="gray")
    # plt.show()
