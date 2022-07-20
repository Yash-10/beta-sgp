# Flux-Conserving Scaled Gradient Projection (FC-SGP)

This repository contains the official code implementation accompanying the paper: *Image Improvement and Restoration in Optical Time Series. I. The Method*. It is aimed at single-image deconvolution of astronomical images with a known Point Spread Function.

## Repository overview

<details>
<summary>Click here to see the folder structure</summary>
<pre>
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
</pre>
</details>

### `restoration`
- It contains the implementation for the Richardson-Lucy (RL), Scaled Gradient Projection (SGP) and the Flux-Conserving Scaled Gradient projection (FC-SGP) algorithms in `rl.py`, `original_sgp.py`, and `sgp.py` respctively.
- `flux_conserve_proj.py` contains the flux conservation projection step code.
- `sgp_validation.py` is for the optional validation step, and `radprof_ellipticity.py` is used to calculate the radial profile and the Full-Width-Half-Maximum (FWHM).

### `psf`

- `psf_calculate.py` calculates the PSF matrix from the parameters output by the `getpsf` code from the [DIAPL package](https://users.camk.edu.pl/pych/DIAPL/)<sup>1</sup>.

### `results`

- It contains the output metric values for each algorithm: RL, SGP, and FC-SGP, and the corresponding code used.

### `paper_plots`

- Contains scripts used to produce the figures from the paper.

`Automation.cl` is the IRAF automation script we generated to automate the process of removing bad bias and flat frames during the image reduction process.

## Example result and comparison

![RL-SGP-FCSGP-comparison-image](https://github.com/Yash-10/fc_sgp-star-restoration/blob/master/readme_example.png?raw=true)

---
**NOTE**

While we say "SGP" in this example, there is a minute difference in our SGP implementation with the SGP algorithm originally proposed in [3], while our SGP implementation is exactly the same as implemented in [2]. Refer our paper for more clarification.

---

## Data availability

The M13 globular cluster I-filter images (244 in number) used in our study are available [here](https://www.dropbox.com/s/5d6lt81o97uiv5u/processed_M13_I_filter_images.tar.gz?dl=0). These images have gone through the usual image reduction pipeline.

## Bugs or issues

If you find something not working as expected or anything weird, we would like to know and improve it! Please feel free to open an issue in the [issue tracker](https://github.com/Yash-10/fc_sgp-star-restoration/issues) or send an email to yashgondhalekar567@gmail.com

## Code motivation

The code presented here is a modified, Python implementation of the [Matlab SGP code of the SGP-dec software](https://www.unife.it/prin/software)<sup>3</sup>.

## References

[1] Pych, W. Difference Image Analysis Package (DIAPL).

[2] Efficient deconvolution methods for astronomical imaging: algorithms and IDL-GPU codes M.  Prato, R.  Cavicchioli, L.  Zanni, P.  Boccacci, M.  Bertero A&A 539 A133 (2012) DOI: 10.1051/0004-6361/201118681

[3] Bonettini S., Zanella R., Zanni L., 2009, InvPr, 25, 015002. doi:10.1088/0266-5611/25/1/015002
