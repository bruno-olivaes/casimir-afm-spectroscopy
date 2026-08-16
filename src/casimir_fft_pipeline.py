import numpy as np
import pandas as pd
from scipy import fft, interpolate
import matplotlib.pyplot as plt
import os

# Physical constants
HBAR = 1.054571817e-34  # J s
C = 2.99792458e8        # m/s
K_B = 1.380649e-23      # J/K
PI = np.pi

def generate_casimir_data(d_array, T=300):
    '''
    Generates the ideal Casimir force and applies thermal and roughness corrections.
    '''
    # 1. Ideal Casimir Force (Lifshitz T=0, perfect conductors)
    F_ideal = - (PI**2 * HBAR * C) / (240 * d_array**4)
    
    # 2. Thermal correction (High-T limit for metals, approx. linear in T*d)
    # Reference: Klimchitskaya & Mostepanenko (2009), Bordag et al. (2009)
    zeta_3 = 1.2020569
    thermal_correction = (240 * zeta_3 * K_B * T * d_array) / (PI**3 * HBAR * C)
    F_T = F_ideal * (1 + thermal_correction)
    
    # 3. Roughness correction (Debye model)
    delta = 0.5e-9  # 0.5 nm RMS roughness
    F_rough = F_T * (1 + 6 * (delta / d_array)**2)
    
    return F_ideal, F_rough

def apply_noise_and_signal(d_array, F_rough):
    '''
    Adds Gaussian noise and an injected periodic signal.
    '''
    # 4. Instrumental noise (1% of signal)
    sigma_noise = 0.01 * np.abs(F_rough)
    noise = np.random.normal(0, sigma_noise)
    
    # 5. Injected periodic signal
    lambda_inj = 500e-9  # 500 nm
    A_inj = 0.005 * np.abs(F_rough)
    F_injected = A_inj * np.sin(2 * PI * d_array / lambda_inj)
    
    # Final measured data
    F_measured = F_rough + noise + F_injected
    
    return F_measured, noise, F_injected

def compute_residuals(F_measured, F_rough):
    '''
    Computes the residuals by subtracting the standard theoretical model.
    '''
    return F_measured - F_rough

def fft_analysis(d_array, residuals):
    '''
    Interpolates the log-spaced data to a linear grid and applies FFT.
    '''
    # Create a uniform linear grid for FFT
    n_points = len(d_array)
    d_lin = np.linspace(d_array.min(), d_array.max(), n_points)
    
    # Interpolate residuals
    interp_func = interpolate.interp1d(d_array, residuals, kind='cubic')
    R_lin = interp_func(d_lin)
    
    # Apply FFT with Hann window to reduce leakage
    window = np.hanning(n_points)
    R_windowed = R_lin * window
    
    # FFT spacing
    spacing = d_lin[1] - d_lin[0]
    frequencies = fft.fftfreq(n_points, d=spacing)
    spectrum = fft.fft(R_windowed)
    
    # Power Spectral Density
    psd = np.abs(spectrum)**2
    
    # Only keep positive frequencies
    pos_mask = frequencies > 0
    freq_pos = frequencies[pos_mask]
    psd_pos = psd[pos_mask]
    
    # Identify peaks (> 3 sigma of the background noise floor)
    noise_floor = np.median(psd_pos)
    sigma_floor = np.std(psd_pos[psd_pos < 3*noise_floor]) # approx noise std
    threshold = noise_floor + 3 * sigma_floor
    
    peaks_mask = psd_pos > threshold
    peak_freqs = freq_pos[peaks_mask]
    peak_psd = psd_pos[peaks_mask]
    sigmas = (peak_psd - noise_floor) / sigma_floor
    
    return freq_pos, psd_pos, threshold, peak_freqs, peak_psd, sigmas

def save_outputs(d_array, F_ideal, F_rough, F_measured, residuals, noise, freq_pos, psd_pos, threshold, peak_freqs, peak_psd, sigmas):
    os.makedirs('output', exist_ok=True)
    
    # CSVs
    df_data = pd.DataFrame({
        'd(m)': d_array,
        'F_ideal(N)': F_ideal,
        'F_rough(N)': F_rough,
        'F_measured(N)': F_measured,
        'residuals(N)': residuals,
        'noise(N)': noise
    })
    df_data.to_csv('output/data_casimir_synthetic.csv', index=False)
    
    df_peaks = pd.DataFrame({
        'peak_frequency(1/m)': peak_freqs,
        'peak_power': peak_psd,
        'significance_sigma': sigmas
    })
    # Filter out very low frequencies (DC leakage)
    df_peaks = df_peaks[df_peaks['peak_frequency(1/m)'] > 1e5]
    df_peaks.to_csv('output/data_fft_peaks.csv', index=False)
    
    # Plots
    plt.figure(figsize=(8, 6))
    plt.loglog(d_array, np.abs(F_ideal), label=r'|F_{ideal}|')
    plt.loglog(d_array, np.abs(F_rough), label=r'|F_{rough}|')
    plt.loglog(d_array, np.abs(F_measured), label=r'|F_{measured}|', alpha=0.5)
    plt.xlabel(r'Separation Distance $d$ (m)')
    plt.ylabel(r'Casimir Force $|F|$ (N)')
    plt.title('Synthetic Casimir Force')
    plt.legend()
    plt.grid(True, which="both", ls="--")
    plt.savefig('output/fig_casimir_force.png')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.plot(d_array, residuals, label='Residuals $R(d)$')
    plt.axhline(0, color='r', linestyle='--')
    plt.xlabel(r'Separation Distance $d$ (m)')
    plt.ylabel(r'Force Residuals (N)')
    plt.title('Force Residuals with Injected Signal')
    plt.legend()
    plt.grid(True)
    plt.savefig('output/fig_residuals.png')
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.semilogy(freq_pos, psd_pos, label='PSD')
    plt.axhline(threshold, color='r', linestyle='--', label=r'3$\sigma$ Threshold')
    for f in df_peaks['peak_frequency(1/m)']:
        plt.axvline(f, color='g', linestyle=':', alpha=0.5)
    plt.xlabel(r'Spatial Frequency $(m^{-1})$')
    plt.ylabel(r'Power Spectral Density')
    plt.title('Spectral Analysis of Residuals')
    plt.xlim(0, 5e6) # Focus on relevant range up to 5 um^-1
    plt.legend()
    plt.grid(True)
    plt.savefig('output/fig_fft_spectrum.png')
    plt.close()

if __name__ == '__main__':
    print('Starting Casimir FFT Pipeline...')
    
    d_min = 100e-9 # 100 nm
    d_max = 10e-6  # 10 um
    N_points = 2000
    d_array = np.logspace(np.log10(d_min), np.log10(d_max), N_points)
    
    F_ideal, F_rough = generate_casimir_data(d_array, T=300)
    F_measured, noise, F_injected = apply_noise_and_signal(d_array, F_rough)
    residuals = compute_residuals(F_measured, F_rough)
    freq_pos, psd_pos, threshold, peak_freqs, peak_psd, sigmas = fft_analysis(d_array, residuals)
    
    save_outputs(d_array, F_ideal, F_rough, F_measured, residuals, noise, freq_pos, psd_pos, threshold, peak_freqs, peak_psd, sigmas)
    
    print('Pipeline executed successfully. Outputs saved in output/ directory.')
