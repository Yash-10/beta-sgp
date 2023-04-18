# Scaled Gradient Projection with $\beta$-divergence

This repository contains the code implementation accompanying the paper: $\beta$-SGP: Scaled Gradient Projection with $\beta$-divergence for astronomical image restoration. It is aimed at single-image deconvolution of astronomical images with a known Point Spread Function.

arXiv preprint: https://arxiv.org/abs/2207.10973

## Repository overview

<details>
<summary>Click here to see the folder structure</summary>
<pre>
.
├── paper_plots
├── psf
├── restoration
│   └── test_data
└── results
    └── content
        └── sgp_experiments
            └── sgp_reconstruction_results
                ├── betadiv
                └── kldiv

10 directories

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

### `pre_processing1

`Automation.cl` is the IRAF automation script we generated to automate the process of removing bad bias and flat frames during the image reduction process.

## Example results and comparison

![Example result](https://github.com/Yash-10/beta-sgp/blob/master/readme_example.png?raw=true)

## Data availability

The M13 globular cluster I-filter images (244 images) are available [here](https://drive.google.com/file/d/13Vk2TpXgSB6IoLUIv-zdh-XI53wJp-0y/view?usp=sharing). These images have gone through the usual image reduction pipeline.

## Bugs or issues

If you find something not working as expected or anything weird, we would like to know and improve it! Please feel free to open an issue in the [issue tracker](https://github.com/Yash-10/fc_sgp-star-restoration/issues) or send an email to yashgondhalekar567@gmail.com

## Code motivation

The code presented here is a modified, Python implementation of the [Matlab SGP code of the SGP-dec software](https://www.unife.it/prin/software)<sup>3</sup>. However, it is not the official Python implementation of SGP-dec.

## References

[1] Pych, W. Difference Image Analysis Package (DIAPL).

[2] Efficient deconvolution methods for astronomical imaging: algorithms and IDL-GPU codes M.  Prato, R.  Cavicchioli, L.  Zanni, P.  Boccacci, M.  Bertero A&A 539 A133 (2012) DOI: 10.1051/0004-6361/201118681

[3] Bonettini S., Zanella R., Zanni L., 2009, InvPr, 25, 015002. doi:10.1088/0266-5611/25/1/015002
