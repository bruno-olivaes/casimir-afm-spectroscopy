# Casimir Force & Nanomechanical Spectral Metrology Pipeline (v2.0)

[![Paper PDF](https://img.shields.io/badge/Paper-PDF_Manuscript-red.svg)](manuscript/main.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Version 2.0](https://img.shields.io/badge/Release-v2.0_Peer--Review_Audited-brightgreen.svg)]()

A precision metrology and spectral analysis pipeline for auditing systematic residuals, periodic anomaly searches, and electrostatic patch-potential backgrounds in micro-electro-mechanical torsional oscillator (MTO), optomechanical, and Atomic Force Microscopy (AFM/KPFM) Casimir experiments.

---

## What Changed in Version 2.0 (Post-Publication Audit)

1. **Lossless Vector Extraction of Decca et al. (2003):** Replaced the 53 raster-digitized points of v1 (which suffered from $\sim 7.1\text{ mPa}$ pixel quantization steps) with **direct PyMuPDF vector coordinate extraction** from the published PDF (`src/decca_vector_extract.py`), recovering all **$N = 497$ dynamic pressure residuals $\Delta P_c(z)$** (Fig. 4b, $\sigma_P = 0.313\text{ mPa}$) and **$N = 288$ static force residuals $\Delta F_c(z)$** (Fig. 3b, $\sigma_F = 0.229\text{ pN}$) with $<0.004\text{ mPa}$ axis-mapping accuracy.
2. **Exact Apparatus & Geometry Attribution:** Properly documents Decca et al. (*Phys. Rev. Lett.* 91, 050402, 2003) as an **Au-coated sphere ($R = 294.3 \pm 0.1\,\mu\mathrm{m}$) vs. Cu-coated MEMS torsional oscillator (MTO)** experiment, removing the v1 $R = 196\,\mu\mathrm{m}$ AFM misattribution.
3. **Resolution of the Zechmeister–Kürster / Scargle Underflow Bug:** Replaces `scipy.signal.lombscargle(..., normalize=True)` + unscaled Scargle exponential formula (which underflowed to $\mathrm{FAP} \equiv 1.00$ even on noiseless sine waves) with the heteroscedastic **Generalized Lomb-Scargle (GLS)** estimator (`astropy.timeseries.LombScargle`) and exact analytical **Baluev (2008)** extreme-value False Alarm Probabilities.
4. **Physical Inverse-Power Detrending vs. Polynomial Runge Ringing:** Demonstrates that subtracting a degree-3 polynomial across the steep short-range ($z < 400\text{ nm}$) drift in $\Delta P_c(z)$ induces spurious Runge oscillations ($\lambda = 583.5\text{ nm}$, $\chi^2/\mathrm{dof} = 3.74$, $\mathrm{FAP} = 1.49\times 10^{-45}$). Modeling the short-range background with the physical basis $\Delta P_{\mathrm{sys}}(z) = c_0 + c_4 (z_0/z)^4 + c_5 (z_0/z)^5$ reduces $\chi^2/\mathrm{dof}$ across all 497 points to **$1.14$** with $P_{\mathrm{ZK},\max} = 0.0332$ and **Baluev $\mathrm{FAP} = 0.163$** (and $\chi^2/\mathrm{dof} = 1.20$, **Baluev $\mathrm{FAP} = 0.651$** for $z \ge 500\text{ nm}$).
5. **2025 KPFM Patch-Potential & Superconducting Audit:** Integrates the 2D Hann-windowed electrostatic patch-potential spectrum $S_{VV}(\mathbf{k})$ from the raw Olympus Jupiter KPFM dataset (`W1P3_E4_measarea3 0003.h5`, $V_{\mathrm{rms}} = 42.42\text{ mV}$, $\delta_{\mathrm{rms}} = 2.22\text{ nm}$) of **de Jong et al. (*Nature Physics*, 2025)** through the exact Speake–Trenner Laplace transfer kernel, showing that surface patches generate a smooth monotonic background ($P_{\mathrm{patch}} \propto d^{-2.29}$) rather than periodic oscillations.

---

## Key Audited Results (v2.0)

| Dataset & Regime | $N$ | $\chi^2/\mathrm{dof}$ | Max GLS Power $P_{\mathrm{ZK},\max}$ | Baluev $\mathrm{FAP}$ | Metrological Conclusion |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Decca Fig. 4b $\Delta P_c$ (Raw, $z \ge 193\text{ nm}$)** | 497 | $41.29$ | $0.3010$ | $1.01\times 10^{-35}$ | Dominated by $<400\text{ nm}$ nonlinear MTO / $\varepsilon(\omega)$ drift |
| **Decca Fig. 4b $\Delta P_c$ (Cubic Poly, $z \ge 193\text{ nm}$)** | 497 | $3.74$ | $0.3626$ | $1.49\times 10^{-45}$ | **Trap:** Polynomial Runge ringing on $z^{-4}$ corner |
| **Decca Fig. 4b $\Delta P_c$ (Physical $\{1, z^{-4}, z^{-5}\}$)** | **497** | **$1.14$** | **$0.0332$** | **$0.163$** | **Null confirmed:** Consistent with white noise ($\sigma_P = 0.313\text{ mPa}$) |
| **Decca Fig. 4b $\Delta P_c$ (Linear Window $z \ge 500\text{ nm}$)** | **343** | **$1.20$** | **$0.0348$** | **$0.651$** | **Null confirmed** in asymptotic linear-response regime |
| **Decca Fig. 3b $\Delta F_c$ (Linear Window $z \ge 400\text{ nm}$)** | **222** | **$0.85$** | **$0.0588$** | **$0.332$** | **Null confirmed** in static force mode ($\sigma_F = 0.229\text{ pN}$) |

---

## Quickstart & Reproducibility

```bash
# 1. Extract lossless vector coordinates from Decca et al. (2003) PDF
python src/decca_vector_extract.py

# 2. Run the complete v2.0 Decca + Injection-Recovery MDL + de Jong (2025) KPFM audit
python src/casimir_fft_pipeline.py --mode full_v2

# 3. Run unit tests
pytest tests
```

---

## License

Creative Commons Attribution 4.0 International ([CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/)).
