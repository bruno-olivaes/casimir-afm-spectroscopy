import numpy as np
import pandas as pd
from scipy import fft, interpolate, stats
import matplotlib.pyplot as plt
import os
import argparse

# Physical constants
HBAR = 1.054571817e-34  # J s
C = 2.99792458e8        # m/s
K_B = 1.380649e-23      # J/K
PI = np.pi

# Default setup parameters
R_SPHERE = 150e-6       # 150 um sphere radius

def generate_casimir_data(d_array, geometry, material_model, T=300):
    '''
    Generates synthetic ideal Casimir force and applies thermal/roughness corrections.
    '''
    # 1. Ideal Casimir Force (Lifshitz T=0, perfect conductors)
    if geometry == 'parallel':
        # F ~ d^-4
        F_ideal = - (PI**2 * HBAR * C) / (240 * d_array**4)
    elif geometry == 'sphere-plate':
        # Proximity Force Approximation (PFA): F = 2*pi*R*E_parallel
        # E_parallel = - (pi^2 * hbar * c) / (720 * d^3)
        F_ideal = - (PI**3 * HBAR * C * R_SPHERE) / (360 * d_array**3)
    else:
        raise ValueError("Invalid geometry")

    # 2. Thermal correction & Material Model Switch
    # Note: The debate between Plasma vs Drude models in the Casimir effect is highly active.
    # The Drude model (gamma > 0) excludes the transverse electric (TE) zero-frequency mode,
    # leading to a reduction in the force at large distances compared to the Plasma model.
    # Here we use phenomenological thermal correction factors to simulate this effect.
    zeta_3 = 1.2020569
    
    if material_model == 'plasma':
        # Plasma: wp = 9.0 eV, gamma = 0
        # Standard high-temperature correction proxy for perfect/plasma conductors
        thermal_correction = (240 * zeta_3 * K_B * T * d_array) / (PI**3 * HBAR * C)
    elif material_model == 'drude':
        # Drude: wp = 9.0 eV, gamma = 0.035 eV
        # TE mode vanishes, thermal correction drops by approx a factor of 1/2 at large distances
        thermal_correction = 0.5 * (240 * zeta_3 * K_B * T * d_array) / (PI**3 * HBAR * C)
    else:
        raise ValueError("Invalid material model")

    F_T = F_ideal * (1 + thermal_correction)
    
    # 3. Roughness correction (Debye model)
    delta = 0.5e-9  # 0.5 nm RMS roughness
    F_rough = F_T * (1 + 6 * (delta / d_array)**2)
    
    return F_ideal, F_T, F_rough

def apply_noise_and_signal(d_array, F_rough, lambda_inj=500e-9):
    '''
    Adds Gaussian noise and an injected periodic signal.
    '''
    sigma_noise = 0.01 * np.abs(F_rough)
    noise = np.random.normal(0, sigma_noise)
    
    A_inj = 0.005 * np.abs(F_rough)
    F_injected = A_inj * np.sin(2 * PI * d_array / lambda_inj)
    F_measured = F_rough + noise + F_injected
    
    return F_measured, noise, F_injected, A_inj

def compute_residuals(F_measured, F_rough):
    return F_measured - F_rough

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
    threshold = noise_floor + 3 * sigma_floor
    
    peaks_mask = psd_pos > threshold
    peak_freqs = freq_pos[peaks_mask]
    peak_psd = psd_pos[peaks_mask]
    sigmas = (peak_psd - noise_floor) / sigma_floor
    
    # Statistical validations
    df = 1.0 / (d_array.max() - d_array.min())
    N_bins = len(freq_pos)
    
    # Bonferroni corrected P-value (assuming exponential distribution for PSD noise)
    # p_local = exp(- (peak - floor)/floor ) approx
    p_local = np.exp(-sigmas)
    p_bonferroni = np.minimum(p_local * N_bins, 1.0)
    
    # Minimum Detection Limit (MDL) for SNR = 3
    # Approximating from the PSD threshold back to amplitude in spatial domain
    # PSD amplitude relates to signal amplitude A by PSD ~ (A * N / 4)^2 for Hanning
    MDL_amp_est = np.sqrt(threshold) * 4 / np.sum(window)
    
    stats_dict = {
        'df': df,
        'N_bins': N_bins,
        'MDL_amp': MDL_amp_est
    }
    
    return freq_pos, psd_pos, threshold, peak_freqs, peak_psd, sigmas, p_bonferroni, stats_dict

def run_pipeline(args):
    print(f"Starting Casimir FFT Pipeline [Geometry: {args.geometry}, Model: {args.material_model}]")
    os.makedirs('output', exist_ok=True)
    
    if args.empirical_data:
        print(f"Processing empirical data from {args.empirical_data}...")
        # Placeholder for empirical data loading
        # df_emp = pd.read_csv(args.empirical_data)
        # d_array = df_emp['d'].values
        # F_measured = df_emp['F'].values
        # This will be fully implemented when user provides the CSV.
        print("Empirical data ingestion not fully mapped until CSV structure is provided. Defaulting to synthetic.")
    
    # Physical Range: 100 nm to 2 um
    # Justification: Above 2 um, the Casimir force (sub-aN to fN) is overwhelmed by 
    # cantilever thermal noise and seismic vibrations in standard AFM.
    d_min = 100e-9
    d_max = 2e-6
    N_points = 2000
    d_array = np.logspace(np.log10(d_min), np.log10(d_max), N_points)
    
    F_ideal, F_T, F_rough = generate_casimir_data(d_array, args.geometry, args.material_model, T=300)
    
    # Baseline injection
    lambda_inj = 500e-9
    F_measured, noise, F_injected, A_inj = apply_noise_and_signal(d_array, F_rough, lambda_inj)
    residuals = compute_residuals(F_measured, F_rough)
    
    freq_pos, psd_pos, threshold, peak_freqs, peak_psd, sigmas, p_bonf, s_dict = fft_analysis(d_array, residuals)
    
    # Sub-bin shift robustness test
    # Shift frequency by 0.5 bins
    f_inj = 1.0 / lambda_inj
    f_inj_shifted = f_inj + 0.5 * s_dict['df']
    lambda_inj_shifted = 1.0 / f_inj_shifted
    
    F_meas_shift, _, _, _ = apply_noise_and_signal(d_array, F_rough, lambda_inj_shifted)
    res_shift = compute_residuals(F_meas_shift, F_rough)
    _, psd_shift, _, p_freq_shift, p_psd_shift, sig_shift, p_bonf_shift, _ = fft_analysis(d_array, res_shift)
    
    print("\n--- Statistical Validation ---")
    print(f"Freq Resolution (df): {s_dict['df']:.2f} m^-1")
    print(f"Independent Bins (N_bins): {s_dict['N_bins']}")
    print(f"MDL (SNR=3): ~ {s_dict['MDL_amp']:.2e} N")
    
    print("\n--- Baseline Injection (lambda = 500 nm) ---")
    for f, p, sig, pb in zip(peak_freqs, peak_psd, sigmas, p_bonf):
        if f > 1e5: # Ignore DC leakage
            print(f"Peak @ {f:.2e} m^-1 | PSD: {p:.2e} | Sigma: {sig:.2f} | Bonf P-val: {pb:.2e}")
            
    print("\n--- Sub-bin Shift Injection (+0.5 df) ---")
    for f, p, sig, pb in zip(p_freq_shift, p_psd_shift, sig_shift, p_bonf_shift):
        if f > 1e5:
            print(f"Peak @ {f:.2e} m^-1 | PSD: {p:.2e} | Sigma: {sig:.2f} | Bonf P-val: {pb:.2e}")
            
    # Save outputs
    pd.DataFrame({
        'd(m)': d_array, 'F_ideal(N)': F_ideal, 'F_rough(N)': F_rough, 
        'F_measured(N)': F_measured, 'residuals(N)': residuals
    }).to_csv('output/data_casimir_synthetic.csv', index=False)
    
    # Plotting
    plt.figure(figsize=(8, 6))
    plt.loglog(d_array, np.abs(F_ideal), label=r'|F_{ideal}|')
    plt.loglog(d_array, np.abs(F_rough), label=r'|F_{rough}|')
    plt.loglog(d_array, np.abs(F_measured), label=r'|F_{measured}|', alpha=0.5)
    plt.xlabel(r'Separation Distance $d$ (m)')
    plt.ylabel(r'Casimir Force $|F|$ (N)')
    plt.title(f'Synthetic Force ({args.geometry}, {args.material_model})')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig('output/fig_casimir_force.png')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.semilogy(freq_pos, psd_pos, label='PSD (Baseline)')
    plt.semilogy(freq_pos, psd_shift, label='PSD (Shifted +0.5 bin)', alpha=0.5)
    plt.axhline(threshold, color='r', linestyle='--', label=r'3$\sigma$ Threshold')
    plt.xlabel(r'Spatial Frequency $(m^{-1})$')
    plt.ylabel(r'Power Spectral Density')
    plt.xlim(0, 5e6)
    plt.legend()
    plt.grid(True)
    plt.savefig('output/fig_fft_spectrum.png')
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Casimir FFT Pipeline")
    parser.add_argument('--geometry', type=str, choices=['parallel', 'sphere-plate'], default='sphere-plate')
    parser.add_argument('--material-model', type=str, choices=['plasma', 'drude'], default='plasma')
    parser.add_argument('--empirical_data', type=str, help="Path to empirical CSV data")
    args = parser.parse_args()
    
    run_pipeline(args)
