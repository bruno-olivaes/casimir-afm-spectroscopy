# Casimir AFM Spectroscopy Pipeline

This repository contains the spectral analysis pipeline for detecting periodic residuals in atomic force microscopy (AFM) measurements of the Casimir force.

## Technical Requirements
- `numpy`, `scipy`, `matplotlib`, `pandas`
- Python 3.8+

## Execution
Run the synthetic pipeline:
```bash
python src/casimir_fft_pipeline.py --geometry sphere-plate --material-model plasma
```

### Command Line Arguments
- `--geometry`: `{parallel, sphere-plate}`. Default is `sphere-plate` (PFA $F \propto d^{-3}$, $R=150 \mu m$) reflecting standard AFM experiments.
- `--material-model`: `{plasma, drude}`. Plasma assumes $\gamma=0$, Drude uses $\gamma=0.035$ eV. This switch allows testing the impact of the ongoing theoretical debate regarding the zero-frequency TE mode on the force residuals.
- `--empirical_data`: Path to a CSV containing extracted raw data (e.g., from WebPlotDigitizer).

## Statistical Validations
The pipeline calculates:
- Frequency Resolution ($\Delta f$)
- Number of independent bins ($N_{bins}$)
- Bonferroni-corrected P-values for detected peaks
- Minimum Detection Limit (MDL) for SNR = 3
- Sub-bin shift robustness (tests for scalloping loss)
