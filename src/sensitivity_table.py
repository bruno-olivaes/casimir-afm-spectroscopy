# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from casimir_fft_pipeline import generate_casimir_data, fft_analysis

def run_sensitivity_analysis():
    N_points_list = [500, 1000, 2000, 5000]
    noise_levels = [0.005, 0.01, 0.02, 0.05]
    d_ranges = [(100e-9, 1e-6), (100e-9, 2e-6)]
    lambda_inj = 500e-9
    amplitudes_fractions = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05]
    
    results = []
    
    for d_min, d_max in d_ranges:
        for N in N_points_list:
            d_array = np.linspace(d_min, d_max, N)
            F_baseline = generate_casimir_data(d_array)
            
            for noise_frac in noise_levels:
                sigma_noise = noise_frac * np.abs(F_baseline)
                noise = np.random.normal(0, sigma_noise, size=len(F_baseline))
                F_rough = F_baseline + noise
                
                mdl_found = None
                false_positive_rate = 0.0
                scalloping_loss_pct = 0.0
                
                for A_frac in amplitudes_fractions:
                    A_inj = A_frac * np.abs(F_baseline)
                    F_injected = A_inj * np.sin(2 * np.pi * d_array / lambda_inj)
                    F_meas = F_rough + F_injected
                    residuals = F_meas - F_baseline
                    
                    f_pos, psd_pos, threshold, p_freqs, p_psd, sigmas, p_bonf, s_dict = fft_analysis(d_array, residuals)
                    
                    f_inj_target = 1.0 / lambda_inj
                    tolerance = 3.0 * s_dict['df']
                    
                    detected_target = False
                    false_positives = 0
                    
                    for pf in p_freqs:
                        if abs(pf - f_inj_target) <= tolerance:
                            detected_target = True
                        elif pf > 1e5:
                            false_positives += 1
                            
                    if detected_target and mdl_found is None:
                        mdl_found = A_frac
                        false_positive_rate = false_positives / max(1, len(p_freqs))
                        
                        f_inj_shifted = (1.0 / lambda_inj) + 0.5 * s_dict['df']
                        F_injected_sh = A_inj * np.sin(2 * np.pi * d_array * f_inj_shifted)
                        F_meas_sh = F_rough + F_injected_sh
                        _, _, _, p_freqs_sh, p_psd_sh, _, _, _ = fft_analysis(d_array, F_meas_sh - F_baseline)
                        
                        max_psd = np.max(p_psd[p_freqs > 1e5]) if np.any(p_freqs > 1e5) else 0
                        max_psd_sh = np.max(p_psd_sh[p_freqs_sh > 1e5]) if np.any(p_freqs_sh > 1e5) else 0
                        
                        if max_psd > 0 and max_psd_sh > 0:
                            amp_base = np.sqrt(max_psd) * 4.0 / s_dict['sum_window']
                            amp_sh = np.sqrt(max_psd_sh) * 4.0 / s_dict['sum_window']
                            scalloping_loss_pct = (1.0 - (amp_sh / amp_base)) * 100
                        else:
                            scalloping_loss_pct = np.nan
                
                if mdl_found is None:
                    mdl_found = np.nan
                    
                results.append({
                    'N_points': N,
                    'd_min_m': d_min,
                    'd_max_m': d_max,
                    'noise_fraction': noise_frac,
                    'MDL_fraction': mdl_found,
                    'false_positive_rate': false_positive_rate,
                    'scalloping_loss_pct': scalloping_loss_pct
                })
                
    df_res = pd.DataFrame(results)
    os.makedirs('output', exist_ok=True)
    df_res.to_csv('output/sensitivity_table.csv', index=False)
    
    df_heat = df_res[(df_res['d_max_m'] == 2e-6)]
    pivot_heat = df_heat.pivot(index='N_points', columns='noise_fraction', values='MDL_fraction')
    
    plt.figure(figsize=(8, 6))
    plt.imshow(pivot_heat.values, cmap='viridis_r', origin='lower', aspect='auto')
    plt.colorbar(label='MDL (lower is better)')
    plt.xticks(ticks=np.arange(len(pivot_heat.columns)), labels=pivot_heat.columns)
    plt.yticks(ticks=np.arange(len(pivot_heat.index)), labels=pivot_heat.index)
    plt.xlabel('Noise Fraction')
    plt.ylabel('N Points')
    plt.title('MDL Sensitivity Heatmap')
    plt.savefig('output/fig_sensitivity_heatmap.png', dpi=300)
    plt.close()
    
    df_line = df_heat[df_heat['N_points'] == 2000]
    plt.figure(figsize=(8, 6))
    plt.plot(df_line['noise_fraction'], df_line['MDL_fraction'], marker='o', linestyle='-')
    plt.xlabel('Noise Fraction')
    plt.ylabel('MDL Fraction')
    plt.title('MDL vs Noise Level (N=2000)')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('output/fig_MDL_vs_noise.png', dpi=300)
    plt.close()
    print("Module A: Sensitivity Table generated successfully.")

if __name__ == '__main__':
    run_sensitivity_analysis()
