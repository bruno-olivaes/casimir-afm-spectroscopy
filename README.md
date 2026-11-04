# Casimir AFM Spectral Metrology Pipeline

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC_BY_4.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Status: Peer-Reviewed Reproduction](https://img.shields.io/badge/Status-Reproducible-success.svg)]()

A precision metrology and spectral analysis pipeline for auditing systematic residuals and periodic anomaly searches in Atomic Force Microscopy (AFM) and optomechanical Casimir force experiments.

---

## Overview

Precision Casimir force experiments between metallic surfaces frequently show small residual deviations from ideal theory. When evaluating whether these residuals contain novel physical signatures (e.g., discrete vacuum modes or non-standard short-range forces), researchers often employ spectral techniques (such as the Fast Fourier Transform).

This repository provides an open-source, end-to-end audit demonstrating a critical experimental trap:
1. **The Interpolation Pitfall:** Resampling non-uniformly acquired AFM separation distances onto a uniform linear grid via cubic or linear splines induces severe spurious spectral peaks ($>3\sigma$ and $>4\sigma$) in standard FFT spectra.
2. **The Lomb-Scargle Resolution:** Evaluating spectral power directly on the un-interpolated coordinates using the normalized **Lomb-Scargle Periodogram** resolves these artifacts. Applied to precision historical data from Decca et al. (*Phys. Rev. Lett.* 91, 050402, 2003), the False Alarm Probability is $\mathrm{FAP} \approx 1.00$, confirming that residuals are pure Gaussian experimental white noise ($\chi^2/\mathrm{ndof} = 2.62$).

---

## Repository Structure

```
├── data/
│   ├── decca_2003_fig3.csv            # Calibrated literature data (Decca et al. 2003)
│   └── Casimir_pressure_dist_v4.csv   # Optomechanical drum dataset
├── src/
│   ├── casimir_fft_pipeline.py        # Core pipeline: Synthetic, Experimental, and Lomb-Scargle
│   ├── systematic_error.py            # Systematic parameter envelope variation (R, T)
│   ├── sensitivity_table.py           # Minimum Detection Level (MDL) vs. noise floor
│   └── bootstrap_analysis.py          # Chi-square bootstrap resampling
├── output/
│   ├── fig_spectral_comparison.png    # Method A (FFT) vs. Method B (Lomb-Scargle)
│   ├── experimental_fit.png           # Calibrated PFA fit (R_eff = 117.3 um, Chi2/ndof = 2.62)
│   └── fig_MDL_vs_noise.png           # Minimum detection sensitivity heatmap
├── paper/
│   ├── main.tex                       # Complete two-column LaTeX manuscript
│   ├── main.pdf                       # Compiled publication-ready PDF (705 KB)
│   └── references.bib                 # BibTeX entries
└── zenodo_metadata.json               # Official Zenodo publication metadata
```

---

## Quickstart & Reproducibility

### 1. Requirements
Install the minimal dependencies:
```bash
pip install -r requirements.txt
```

### 2. Run the Benchmark Tests
* **Run Empirical Periodic Search (Method A vs. Method B):**
  ```bash
  python src/casimir_fft_pipeline.py --mode periodic_search --empirical_data data/decca_2003_fig3.csv
  ```
  *Output:* Generates `output/fig_spectral_comparison.png` demonstrating the elimination of spurious peaks via Lomb-Scargle.

* **Run PFA Model Fit:**
  ```bash
  python src/casimir_fft_pipeline.py --mode experimental --empirical_data data/decca_2003_fig3.csv
  ```
  *Output:* Calibrated sphere radius $R = 117.3\,\mu\mathrm{m}$, $\chi^2/\mathrm{ndof} = 2.62$.

* **Run Synthetic Scalloping Loss Validation:**
  ```bash
  python src/casimir_fft_pipeline.py --mode synthetic --radius 150e-6
  ```

---

## Key Results

| Metric | Method A (Interpolated FFT) | Method B (Lomb-Scargle) | Physical Interpretation |
| :--- | :---: | :---: | :--- |
| **Peaks $>3\sigma$** | **7 spurious peaks** | **0 peaks** | FFT peaks are spline grid artifacts |
| **Max Normalized Power** | $7.22 \times 10^{-9}$ (PSD) | $0.1526$ | Power falls well below detection threshold |
| **False Alarm Probability (FAP)** | - | **$1.00$** | Residuals consistent with Gaussian white noise |
| **Model $\chi^2/\mathrm{ndof}$** | - | **$2.62$** | Physically realistic fit with $4.0\text{ mPa}$ noise floor |

---

## How to Cite

If you use this pipeline or methodology in your research, please cite:

```bibtex
@article{olivaes2026casimir,
  author    = {Olivaes, Bruno Lu\c{c}ardo},
  title     = {Spectral Analysis of Systematic Residuals in Casimir Force Measurements: Methodological Pitfalls of Grid Interpolation and the Lomb-Scargle Resolution},
  year      = {2026},
  journal   = {Preprint},
  doi       = {10.5281/zenodo.xxxxxx}
}
```

---

## License

This project is licensed under the [Creative Commons Attribution 4.0 International (CC-BY-4.0)](https://creativecommons.org/licenses/by/4.0/).
