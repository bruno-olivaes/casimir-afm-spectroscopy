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

## Limitations

1. **PFA Validity**: Proximity Force Approximation is only valid for  \ll R$. Verified for this configuration ({max} = 2 \mu m \ll R = 150 \mu m$).
2. **Roughness Expansion**: First-order perturbative expansion for roughness ($\eta_{rough} = 1 + 6(\delta_{rms}/d)^2$) is valid for  \gg \delta_{rms}$. Verified for this configuration ({min} = 100 nm \gg \delta_{rms} = 0.5 nm$).
3. **Plasma Model Proxy**: The plasma correction applies phenomenological finite conductivity scaling ($\eta_{plasma} \approx 1/(1+c/(\omega_p d))$) rather than exact numerical Lifshitz-Matsubara integration.

## Known Limitations

1. **Sensitivity Table False Positives:** The 4σ detection threshold yields elevated false positive rates (80–95%) for dense frequency grids (N > 1000 bins). Experimental application requires threshold calibration against instrument-specific noise characteristics.

2. **Scalloping Loss Measurement:** Sub-bin shift scalloping loss is quantified only when the shifted peak remains above detection threshold. Near-zero or negative values indicate sub-threshold detection and should be interpreted as "not measurable" rather than "zero loss."

3. **MDL Grid Dependence:** The Minimum Detection Level depends on the discrete amplitude grid (0.001, 0.002, 0.005, 0.01, 0.02, 0.05). Finer sampling is recommended for precise calibration curves.

4. **Bootstrap Proof-of-Concept:** The bootstrap demonstration uses the Zenodo theoretical dataset (BCS ≈ Drude) as a validation of the statistical method. Application to experimental data with divergent models is required for scientific conclusions.
