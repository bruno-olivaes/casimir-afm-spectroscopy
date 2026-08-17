# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from scipy import fft, interpolate
from scipy.stats import chi2
import matplotlib.pyplot as plt
import os
import argparse

# Physical constants
HBAR = 1.054571817e-34  # J s
C = 2.99792458e8        # m/s
K_B = 1.380649e-23      # J/K
PI = np.pi

# GEOMETRY NOTE: 
# Proxy: F = P * pi*R**2, treating drum as rigid plate. 
# True flexural mechanics requires Euler-Bernoulli integration.

def generate_casimir_data(d_array, T=300):
    F_ideal = - (PI**2 * HBAR * C) / (240 * d_array**4)
    zeta_3 = 1.2020569
    thermal_correction = (240 * zeta_3 * K_B * T * d_array) / (PI**3 * HBAR * C)
    F_T = F_ideal * (1 + thermal_correction)
    return F_T

def apply_noise_and_signal(d_array, F_rough, lambda_inj=500e-9):
    sigma_noise = 0.01 * np.abs(F_rough)
    noise = np.random.normal(0, sigma_noise, size=len(F_rough))
    A_inj = 0.005 * np.abs(F_rough)
    F_injected = A_inj * np.sin(2 * PI * d_array / lambda_inj)
    F_measured = F_rough + noise + F_injected
    return F_measured, noise, F_injected, A_inj

def fft_analysis(d_array, residuals):
    n_points = len(d_array)
    d_lin = np.linspace(d_array.min(), d_array.max(), n_points)
    
    interp_func = interpolate.interp1d(d_array, residuals, kind='cubic')
    R_lin = interp_func(d_lin)
    
    window = np.hanning(n_points)
    R_windowed = R_lin * window
    
    spacing = d_lin[1] - d_lin[0]
    frequencies = fft.fftfreq(n_points, d=spacing)
    spectrum = fft.fft(R_windowed)
    
    psd = np.abs(spectrum)**2
    
    pos_mask = frequencies > 0
    freq_pos = frequencies[pos_mask]
    psd_pos = psd[pos_mask]
    
    noise_floor = np.median(psd_pos)
    sigma_floor = np.std(psd_pos[psd_pos < 3*noise_floor])
    threshold = noise_floor + 4 * sigma_floor
    
    peaks_mask = psd_pos > threshold
    peak_freqs = freq_pos[peaks_mask]
    peak_psd = psd_pos[peaks_mask]
    sigmas = (peak_psd - noise_floor) / sigma_floor
    
    df = 1.0 / (d_array.max() - d_array.min())
    N_bins = len(freq_pos)
    
    p_local = np.exp(-sigmas)
    p_bonferroni = np.minimum(p_local * N_bins, 1.0)
    
    MDL_amp_est = np.sqrt(threshold) * 4.0 / np.sum(window)
    
    stats_dict = {
        'df': df,
        'N_bins': N_bins,
        'MDL_amp': MDL_amp_est,
        'sum_window': np.sum(window)
    }
    
    return freq_pos, psd_pos, threshold, peak_freqs, peak_psd, sigmas, p_bonferroni, stats_dict

def run_synthetic(radius):
    print("--- MODE: SYNTHETIC (Methodology Validation) ---")
    d_array = np.linspace(100e-9, 2e-6, 2000)
    F_baseline = generate_casimir_data(d_array)
    
    lambda_inj = 500e-9
    F_meas, noise, F_injected, A_inj = apply_noise_and_signal(d_array, F_baseline, lambda_inj)
    residuals = F_meas - F_baseline
    
    f_pos, psd_pos, threshold, p_freqs, p_psd, sigmas, p_bonf, s_dict = fft_analysis(d_array, residuals)
    
    valid_peaks = p_freqs > 1e5
    max_psd = np.max(p_psd[valid_peaks]) if np.any(valid_peaks) else 0
    amp_base = np.sqrt(max_psd) * 4.0 / s_dict['sum_window']
    
    f_inj_shifted = (1.0 / lambda_inj) + 0.5 * s_dict['df']
    lambda_inj_shifted = 1.0 / f_inj_shifted
    
    F_meas_sh, _, _, _ = apply_noise_and_signal(d_array, F_baseline, lambda_inj_shifted)
    res_sh = F_meas_sh - F_baseline
    _, psd_pos_sh, _, p_freqs_sh, p_psd_sh, _, _, _ = fft_analysis(d_array, res_sh)
    
    valid_peaks_sh = p_freqs_sh > 1e5
    max_psd_sh = np.max(p_psd_sh[valid_peaks_sh]) if np.any(valid_peaks_sh) else 0
    amp_sh = np.sqrt(max_psd_sh) * 4.0 / s_dict['sum_window']
    
    print(f"Freq Resolution (df): {s_dict['df']:.2f} m^-1")
    if max_psd > 0:
        loss = (1.0 - (amp_sh / amp_base)) * 100
        print(f"Sub-bin Shift Scalloping Loss: {loss:.2f}%")
        
    plt.figure(figsize=(8, 6))
    plt.semilogy(f_pos, psd_pos, label='PSD (Exact)')
    plt.semilogy(f_pos, psd_pos_sh, label='PSD (+0.5 bin shift)', alpha=0.5)
    plt.axhline(threshold, color='r', linestyle='--', label=r'3 sigma Threshold')
    plt.xlabel('Spatial Frequency (m^-1)')
    plt.ylabel('PSD')
    plt.legend()
    plt.savefig('output/synthetic_fft.png')
    plt.close()

    # Save CSVs
    pd.DataFrame({'d(m)': d_array, 'F_baseline(N)': F_baseline, 
                  'F_measured(N)': F_meas, 'residuals(N)': residuals,
                  'noise(N)': noise, 'injected(N)': F_injected}).to_csv('output/data_casimir_synthetic.csv', index=False)

    if len(p_freqs) > 0:
        pd.DataFrame({'frequency(1/m)': p_freqs, 'PSD': p_psd, 
                      'sigma': sigmas, 'p_bonferroni': p_bonf}).to_csv('output/data_fft_peaks.csv', index=False)

    # Save residuals plot
    plt.figure(figsize=(8,6))
    plt.plot(d_array, residuals, label='Residuals', alpha=0.7)
    plt.plot(d_array, F_injected, label='Injected Signal (Ground Truth)', color='orange')
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Separation Distance (m)')
    plt.ylabel('Residuals (N)')
    plt.legend()
    plt.savefig('output/fig_residuals.png')
    plt.close()

def run_discrimination(data_path, radius, model_b):
    print("--- MODE: DISCRIMINATION (Classical Statistics) ---")
    df = pd.read_csv(data_path)
    d_array = df['Gap (nm)'].values * 1e-9
    
    area = PI * (radius**2)
    print(f"Geometry Proxy: F = P * pi*R^2 (R = {radius} m)")
    print("WARNING: The input dataset shows P_BCS approx P_Drude numerically (difference < 10^-5). Model discrimination requires experimental data or a synthetic dataset with divergent models.")
    
    # Ground Truth = BCS model + 0.1% noise
    P_bcs = df['P BCS (Pa)'].values
    sigma_noise = 0.001 * np.abs(P_bcs)
    P_exp = P_bcs + np.random.normal(0, sigma_noise, size=len(P_bcs))
    
    if model_b == 'drude':
        P_model_b = df['P Drude (Pa)'].values
    elif model_b == 'plasma':
        P_model_b = df['P Plasma (Pa)'].values
    else:
        raise ValueError("Invalid model-b")
        
    ndof = len(P_exp)
    chi2_bcs = np.sum(((P_exp - P_bcs) / sigma_noise)**2)
    chi2_model_b = np.sum(((P_exp - P_model_b) / sigma_noise)**2)
    
    print(f"chi2_BCS/ndof: {chi2_bcs/ndof:.3f}")
    print(f"chi2_{model_b.capitalize()}/ndof: {chi2_model_b/ndof:.3f}")
    print(f"Systematic bias ({model_b.capitalize()} vs BCS): {np.mean(np.abs(P_model_b - P_bcs))/np.mean(np.abs(P_bcs)):.4f}")
    
    # Convert to force for residual plotting
    F_exp = P_exp * area
    F_model_b = P_model_b * area
    residuals_F = F_exp - F_model_b
    
    plt.figure(figsize=(8, 6))
    plt.plot(d_array, residuals_F, label=f'Residuals (Exp - {model_b.capitalize()})', color='blue')
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel('Separation Distance (m)')
    plt.ylabel('Force Residuals (N)')
    plt.title(f'Systematic Structure: BCS Ground Truth vs {model_b.capitalize()} Model')
    plt.legend()
    plt.savefig('output/discrimination_residuals.png')
    plt.close()

def run_periodic_search(data_path):
    print("--- MODE: PERIODIC SEARCH (FFT on raw instrument residuals) ---")
    pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Casimir Statistical & FFT Pipeline")
    parser.add_argument('--mode', type=str, required=True, choices=['synthetic', 'discrimination', 'periodic_search'])
    parser.add_argument('--empirical_data', type=str, help="Path to empirical CSV data")
    parser.add_argument('--radius', type=float, required=True, help="Radius of drum in meters (Mandatory for conversion)")
    parser.add_argument('--model-b', type=str, choices=['drude', 'plasma'], default='drude', help="Alternative model to compare against BCS")
    args = parser.parse_args()
    
    os.makedirs('output', exist_ok=True)
    
    if args.mode == 'synthetic':
        run_synthetic(args.radius)
    elif args.mode == 'discrimination':
        if not args.empirical_data:
            raise ValueError("Must provide --empirical_data for discrimination.")
        run_discrimination(args.empirical_data, args.radius, args.model_b)
    elif args.mode == 'periodic_search':
        run_periodic_search(args.empirical_data)
