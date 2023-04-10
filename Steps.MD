1. Run: `./fwhms.bash` to get mask files.
2. Run `./psf_estimation.bash` to create subdivisions and PSF binary files containing PSF coefficients.
3. Run: `./get_psf_coeffs.bash` to get .txt files. It is human readable now.
4. Run: python psf_calculate.py to create PSF FITS files.
