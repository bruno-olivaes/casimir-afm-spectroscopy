# Casimir AFM Spectroscopy Pipeline

A methodology validation and analytical pipeline for systematic residuals in Casimir force experiments using Atomic Force Microscopy (AFM) and optomechanical drums.

## Modes of Operation

1. **Synthetic Mode**: Validates the FFT methodology by generating theoretical Casimir force data, injecting synthetic periodic anomalies, and computing the detection limits and scalloping loss.
   python src/casimir_fft_pipeline.py --mode synthetic --radius 150e-6
   *Data generated on uniform grid to avoid spectral artifacts from interpolation.*

2. **Discrimination Mode**: Employs classical statistics ($\chi^2$ likelihood ratio) to test theoretical models (e.g. Drude vs BCS vs Plasma) against high-precision data. Does not use FFT.
   python src/casimir_fft_pipeline.py --mode discrimination --empirical_data data/your_data.csv --radius 150e-6 --model-b plasma
   *WARNING: The Zenodo dataset shows P_BCS ≈ P_Drude numerically (difference < 10⁻⁵). Model discrimination requires experimental data or a synthetic dataset with divergent models. Use --model-b plasma to demonstrate macro-divergent discrimination.*

3. **Periodic Search Mode**: Placeholder for running FFT anomaly detection on true raw instrumental optomechanical sweep data.

## Technical Notes

* **Binning Error**: FFT frequency resolution is fundamentally limited by $\Delta f = 1 / (d_{max} - d_{min})$. Spectral peaks falling between bins will suffer amplitude degradation and frequency smearing.
* **Scalloping Loss**: Due to windowing (e.g., Hann window), a signal exactly halfway between two frequency bins can lose up to ~16% of its apparent amplitude (worst-case scalloping). Sub-bin shifts must be quantified.
* **False Positives (Interpolation)**: Taking empirical non-uniform or log-spaced AFM separation distances and forcibly interpolating them to a linear grid can introduce severe periodic artifacts in the high-frequency domain.
