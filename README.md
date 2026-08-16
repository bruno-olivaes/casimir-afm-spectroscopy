# Casimir AFM Spectroscopy Pipeline

This repository contains the spectral analysis pipeline for detecting periodic residuals in atomic force microscopy (AFM) measurements of the Casimir force.

## Repository Structure

* `src/`: Python source code for data analysis and simulation.
* `data/`: Raw and simulated AFM data.
* `output/`: Generated plots and analytical results (CSVs).
* `paper/`: LaTeX source files for the resulting manuscript.

## Installation and Execution

1. **Install Dependencies:**
   Install the required Python packages using pip:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Synthetic Pipeline:**
   Execute the core script to generate synthetic AFM data, apply the thermal and roughness corrections, add noise, inject a test signal, and perform the FFT analysis:
   ```bash
   python src/casimir_fft_pipeline.py
   ```

3. **Check Outputs:**
   The script will populate the `output/` directory with:
   - `fig_casimir_force.png`: The generated force-distance curves.
   - `fig_residuals.png`: The isolated residuals.
   - `fig_fft_spectrum.png`: The PSD spectrum showing the detected peaks.
   - `data_casimir_synthetic.csv`: The raw synthetic tabular data.
   - `data_fft_peaks.csv`: The detected spectral peaks and their significance.

## Scientific Context

This project aims to systematically quantify short-range anomalies in Casimir force measurements. The theoretical framework employs the high-temperature Drude model limit for thermal corrections and the Debye model for surface roughness, establishing a rigid baseline for spectral peak detection in the residuals.
