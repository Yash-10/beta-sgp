1. Run: `./fwhms.bash` to get mask files.
2. Run `./psf_estimation.bash` to create subdivisions and PSF binary files containing PSF coefficients.
3. Run: `./get_psf_coeffs.bash` to get .txt files. It is human readable now.
4. Run: python psf_calculate.py to create PSF FITS files.








### Description of parameter files (for the star stamp experiment):

```
$ cat sfind.par 
C_MIN   = 0.4	// covariance threshold for detection
APRAD   = 1.2	// aperture radius in FWHMs
ANRAD1  = 20.0	// lower radius of background annulus
ANRAD2  = 25.0	// uppwer radius of background annulus
THRESH  = 2.0	// sigma threshold
DXY     = 1.0	// maximum distance of brightest pixel from centroid
BKG_ALG = mode	// supported modes are: mode, median, mean
VERBOSE = 1	// verbosity level
END


$ cat getpsf.par 
NBOX_X = 5
NBOX_Y = 5
NDEG_SPAT = 1
NDEG_LOCAL = 2
NGAUSS = 2
NPSF_MAX = 100
MIN_NBOX = 3
MIN_FLUX = 100.0
MAX_THRESH = 200.0
RAT_THRESH = 0.2
NSIG_DETECT = 3.0
NSIG_RAT = 3.0
PSFHW = 15
MAXHW = 1
ISOHW = 2
ISO_OFF = 1.0
ISO_SLO = 0.15
FITRAD = 3.0
NSIG_CLIP = 2.0
NITER_INIT = 3
NITER = 4
RECENTER = 1      
PSF_COS = 1.0
PSF_SIN = 0.0
PSF_AX = -0.8
PSF_AY = -0.8
SIGMA_INC = 0.548
SIGMA_MSCALE = 1.582
VERBOSE = 1
END


$ cat fwhmm.par 
#========================================================#
#                                                        #
#  fwhmm.par           version 3.5.0   2011.01.31        #
#                                                        #
#  Copyright (C) 2010-2011 by Wojtek Pych, CAMK PAN      #
#  pych@camk.edu.pl                                      #
#                                                        #
#========================================================#
#
SKY_MOD     = mode	# sky calc. algorithm (mode/quantile/median/mean)
QINTERVAL   = 0.3
APERTURE    = 8		# aperture size in pixels
IN_SKY_RAD  = 2		# sky annulus inner radius (times maxfwhm)
OUT_SKY_RAD = 5		# sky annulus outer radius (times maxfwhm)
MARGIN      = 5		# limit of the star distance from the edge
MIN_FWHM    = 2.6	# minimum FWHM of a good star
MAX_FWHM    = 10.0	# maximum FWHM of a good star
MIN_PEAK    = 5.0e2	# minimum peak of a good star
MAX_PEAK    = 2.0e4	# maximum peak of a good star
MIN_SKY     = 1.0	# minimum sky level
MAX_SKY     = 5000.0	# maximum sky level
MASKPAR     = bmask.par # bad pixels list's file name
VERBOSE     = 0		# verbose level 0,1,2
END

$ cat instrument.par
#========================================================#
#                                                        #
#  instrument.par         version 1.2.0   2009.01.12     #
#                                                        #
#  Copyright (C) 2009 by Wojtek Pych, CAMK PAN           #
#  pych@camk.edu.pl                                      #
#                                                        #
#========================================================#
#
GAIN      = 1.22                // camera gain [electrons/ADU]
MIN_LEVEL = 1.0         // minimum acceptable count
SAT_LEVEL = 65000.0     // maximum acceptable count
NX        = 410         // X size of the sub-frames
NY        = 410         // Y size of the sub-frames
MARG      = 20          // size of the overlap margin
FITS      = fits        // FITS file extension
END


```
