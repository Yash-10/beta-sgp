# Flux-Conserving Scaled Gradient projection (FC-SGP)

This repository contains the official code implementation accompanying the paper: *Image Improvement and Restoration in Optical Time Series. I. The Method*. It is aimed for single-image deconvolution of astronomical images with a known Point Spread Function.

## Repository overview

<details>
<summary>Click here to see the folder structure</summary>
<br>
.
├── Automation.cl
├── paper_plots
│   ├── ccfbtf170075_inset.png
│   ├── ccfbvc170119_inset.png
│   ├── ccfbwi110033_inset.png
│   ├── intro.py
│   ├── projection_time.png
│   ├── projection_time.py
│   ├── psf_mat_show.ipynb
│   ├── psfMatShow.png
│   ├── README.md
│   ├── s_ccfbtf170075_inset.png
│   ├── s_ccfbvc170119_inset.png
│   └── s_ccfbwi110033_inset.png
├── psf
│   ├── classify.py
│   ├── get_psf_coeffs.bash
│   ├── getpsf.par
│   ├── psf_calculate.py
│   ├── psf_ccfbrd210048.bin.txt
│   ├── psf_estimation.bash
│   ├── psf_models.tar.gz
│   └── README.md
├── README.md
├── restoration
│   ├── flux_conserve_proj.py
│   ├── original_sgp.py
│   ├── radprof_ellipticity.py
│   ├── rl.py
│   ├── sgp.py
│   └── sgp_validation.py
└── results
    ├── compare_sgp_and_fcsgp_kld.py
    ├── fc_sgp_output.ipynb
    ├── fc_sgp_params_and_metrics.csv
    ├── fcsgp_radprof_params_and_metrics.csv
    ├── plot_radprofiles.ipynb
    ├── plot_rl_sgp_fc-sgp_results.ipynb
    ├── radprof_params_and_metrics.csv
    ├── rl_params_and_metrics.csv
    ├── rl_sgp_fc-sgp_compare.ipynb
    ├── sgp_fcsgp_kld_compare.ipynb
    └── sgp_params_and_metrics.csv

4 directories, 39 files
</br>
</details>

### `restoration`
- It contains the implementation for the Richardson-Lucy (RL), Scaled Gradient Projection (SGP) and the Flux-Conserving Scaled Gradient projection (FC-SGP) algorithms in `rl.py`, `original_sgp.py`, and `sgp.py` resepctively.
- `flux_conserve_proj.py` contains the flux conservation projection step code.
- `sgp_validation.py` is for the optional validation step and `radprof_ellipticity.py` is used to calculate the radial profile and the Full-Width-Half-Maximum (FWHM).

### `psf`

- `psf_calculate.py` calculates the PSF matrix from the parameters output by the `getpsf` code from the [DIAPL package](https://users.camk.edu.pl/pych/DIAPL/)<sup>2</sup>.

### `results`

- It contains the output metric values for each algorithm: RL, SGP, and FC-SGP, and the corresponding code used.

### `paper_plots`

- Contains scripts used to produce the figures from the paper.

- `Automation.cl` is the IRAF automation script we generated to automate the process of removing bad bias and flat frames during the image reduction process.

## Code motivation

The code presented here is a modified, Python implementation of the [Matlab SGP code of the SGP-dec software](https://www.unife.it/prin/software)<sup>1</sup>.

## References

[1] Efficient deconvolution methods for astronomical imaging: algorithms and IDL-GPU codes M.  Prato, R.  Cavicchioli, L.  Zanni, P.  Boccacci, M.  Bertero A&A 539 A133 (2012) DOI: 10.1051/0004-6361/201118681
[2] Pych, W. Difference Image Analysis Package (DIAPL).