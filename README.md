# Scaled Gradient Projection with $\beta$-divergence

This repository contains the code implementation accompanying the paper: [**$\beta$-SGP: Scaled Gradient Projection with $\beta$-divergence for astronomical image restoration**](https://arxiv.org/abs/2207.10973). It is aimed at single-image deconvolution of astronomical images with a known Point Spread Function.

## Repository overview

<details>
<summary>Click here to see the folder structure</summary>
<pre>

.
├── images
│   ├── crowded_flux_subdiv.png
│   ├── crowded_subdiv_example.png
│   ├── ellipticity_ratio.png
│   ├── flux_frac_diff.png
│   ├── flux_line_plot_stamps.png
│   ├── flux_subdiv.png
│   ├── fwhm_ratio.png
│   └── subdiv_example.png
├── pre_processing
│   └── Automation.cl
├── psf
│   ├── get_psf_coeffs.bash
│   ├── psf_calculate.py
│   ├── psfccfbrd210048_1_1.bin.txt
│   ├── psfccfbrd210048_1_1_img.fits
│   ├── psf_estimation.bash
│   ├── psf_mat_show.ipynb
│   ├── psf_steps_and_params.MD
│   └── README.md
├── README.md
├── restoration
│   ├── application_sgp_star_stamps.py
│   ├── application_sgp_subdivisions.py
│   ├── flux_conserve_proj.py
│   ├── sgp.py
│   ├── simulated_test
│   │   ├── data
│   │   │   ├── NGC7027_255.mat
│   │   │   └── satellite_25500.mat
│   │   └── __init__.py
│   ├── simulation_test_sgp.py
│   ├── tests.py
│   └── utils.py
└── results
    ├── CROWDED_SUBDIV_BEST_BETA_INIT.npy
    ├── CROWDED_SUBDIV_EXEC_TIME_BETA.npy
    ├── CROWDED_SUBDIV_EXEC_TIME.npy
    ├── CROWDED_SUBDIV_NUM_ITERS_BETA.npy
    ├── CROWDED_SUBDIV_NUM_ITERS.npy
    ├── CROWDED_SUBDIV_ORIGCAT_2sigma.csv
    ├── CROWDED_SUBDIV_ORIGCAT.csv
    ├── CROWDED_SUBDIV_ORIG_FLUX_BETA.npy
    ├── CROWDED_SUBDIV_ORIG_FLUX.npy
    ├── CROWDED_SUBDIV_ORIGIMG_BETA.fits
    ├── CROWDED_SUBDIV_ORIGIMG.fits
    ├── CROWDED_SUBDIV_RESTORED_BETA.csv
    ├── CROWDED_SUBDIV_RESTORED_BETA_MATCHED.csv
    ├── CROWDED_SUBDIV_RESTORED.csv
    ├── CROWDED_SUBDIV_RESTORED_FLUX_BETA.npy
    ├── CROWDED_SUBDIV_RESTORED_FLUX.npy
    ├── CROWDED_SUBDIV_RESTOREDIMG_BETA.fits
    ├── CROWDED_SUBDIV_RESTOREDIMG.fits
    ├── CROWDED_SUBDIV_RESTORED_MATCHED.csv
    ├── ELLIPTICITY_RATIO_BETA.npy
    ├── ELLIPTICITY_RATIO.npy
    ├── EXEC_TIME_BETA.npy
    ├── EXEC_TIME.npy
    ├── FLUX_FRACTIONAL_DIFFERENCE_BETA.npy
    ├── FLUX_FRACTIONAL_DIFFERENCE.npy
    ├── FWHM_RATIO_BETA.npy
    ├── FWHM_RATIO.npy
    ├── NUM_ITERS_BETA.npy
    ├── NUM_ITERS.npy
    ├── ORIG_FLUX_BETA.npy
    ├── ORIG_FLUX.npy
    ├── RESTORED_FLUX_BETA.npy
    ├── RESTORED_FLUX.npy
    ├── SUBDIV_BEST_BETA_INIT.npy
    ├── SUBDIV_EXEC_TIME_BETA.npy
    ├── SUBDIV_EXEC_TIME.npy
    ├── SUBDIV_NUM_ITERS_BETA.npy
    ├── SUBDIV_NUM_ITERS.npy
    ├── SUBDIV_ORIGCAT.csv
    ├── SUBDIV_ORIG_FLUX_BETA.npy
    ├── SUBDIV_ORIG_FLUX.npy
    ├── SUBDIV_ORIGIMG_BETA.fits
    ├── SUBDIV_ORIGIMG.fits
    ├── SUBDIV_RESTORED_BETA.csv
    ├── SUBDIV_RESTORED_BETA_MATCHED.csv
    ├── SUBDIV_RESTORED.csv
    ├── SUBDIV_RESTORED_FLUX_BETA.npy
    ├── SUBDIV_RESTORED_FLUX.npy
    ├── SUBDIV_RESTOREDIMG_BETA.fits
    ├── SUBDIV_RESTOREDIMG.fits
    ├── SUBDIV_RESTORED_MATCHED.csv
    ├── WD_RADIAL_PROFILE_DISTANCE_BETA.npy
    └── WD_RADIAL_PROFILE_DISTANCE.npy

7 directories

</pre>
</details>

### `restoration`
- `sgp.py` contains implementation for SGP with both, $\beta$-divergence and KL divergence.
- `flux_conserve_proj.py` contains the flux conservation projection step code.
- `utils.py` contains some utility functions helpful in other scripts.

### `psf`

- `psf_calculate.py` calculates the PSF matrix from the parameters output by the `getpsf` code from the [DIAPL package](https://users.camk.edu.pl/pych/DIAPL/)<sup>1</sup>.

### `results`

- It contains the results in form of metrics embedded in .npy files.

### `pre_processing`

- `Automation.cl` is the IRAF automation script we generated to automate the process of removing bad bias and flat frames during the image reduction process.

## Example results and comparison

Example 1:
![Example 1](https://github.com/Yash-10/beta-sgp/blob/master/images/subdiv_example.png?raw=true)

Example 2:
![Example 2](https://github.com/Yash-10/beta-sgp/blob/master/images/crowded_subdiv_example.png?raw=true)

## Data availability

The M13 globular cluster I-filter images (244 images) are available [here](https://drive.google.com/file/d/13Vk2TpXgSB6IoLUIv-zdh-XI53wJp-0y/view?usp=sharing). These images have gone through the usual image reduction pipeline.

## Bugs or issues

If you find something not working as expected or anything weird, we would like to know and improve it! Please feel free to open an issue in the [issue tracker](https://github.com/Yash-10/fc_sgp-star-restoration/issues) or send an email to yashgondhalekar567@gmail.com

## Code motivation

The code presented here is a modified, Python implementation of the [Matlab SGP code of the SGP-dec software](https://www.unife.it/prin/software)<sup>3</sup>. However, it is not the official Python implementation of SGP-dec.

## Code status

Currently, SGP and $\beta$-SGP are written in two separate functions, despite both having many commonalities. This could make it slightly cumbersome to switch between both functions, especially given that $\beta$-SGP generalizes SGP. Future versions would focus on improving this aspect.

## References

[1] Pych, W. Difference Image Analysis Package (DIAPL).

[2] Efficient deconvolution methods for astronomical imaging: algorithms and IDL-GPU codes M.  Prato, R.  Cavicchioli, L.  Zanni, P.  Boccacci, M.  Bertero A&A 539 A133 (2012) DOI: 10.1051/0004-6361/201118681

[3] Bonettini S., Zanella R., Zanni L., 2009, InvPr, 25, 015002. doi:10.1088/0266-5611/25/1/015002

## License and copyright
The code here is licensed under the [MIT License](https://github.com/Yash-10/beta-sgp/blob/master/LICENSE).

Copyright (c) 2023 Yash Gondhalekar
