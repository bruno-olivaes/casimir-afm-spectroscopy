import os

# --- MÓDULO A: sensitivity_table.py ---
sensitivity_code = """# -*- coding: utf-8 -*-
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
                        
                        if max_psd > 0:
                            amp_base = np.sqrt(max_psd) * 4.0 / s_dict['sum_window']
                            amp_sh = np.sqrt(max_psd_sh) * 4.0 / s_dict['sum_window']
                            scalloping_loss_pct = (1.0 - (amp_sh / amp_base)) * 100
                
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
"""

# --- MÓDULO B: systematic_error.py ---
systematic_code = """# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from casimir_fft_pipeline import generate_casimir_data, C, PI, HBAR

def casimir_augmented_force(d_array, R_sphere, T, omega_p, delta_rough):
    \"\"\"
    Envelope computed by varying input parameters of the validated Casimir model. 
    Approximate asymptotic expansions are avoided to maintain consistency.
    \"\"\"
    P_ideal = generate_casimir_data(d_array, T)
    eta_rough = 1.0 + 6.0 * (delta_rough / d_array)**2
    # Non-expanded plasma skin depth scaling (lowest order Padé)
    eta_plasma = 1.0 / (1.0 + C / (omega_p * d_array))
    
    P_total = P_ideal * eta_rough * eta_plasma
    F_total = P_total * PI * (R_sphere**2)
    return F_total

def run_systematic_error():
    d_array = np.linspace(100e-9, 2e-6, 500)
    
    # Nominals
    R_nom = 150e-6
    T_nom = 300.0
    omega_p_nom = 9.0 * 1.60218e-19 / HBAR  # 9.0 eV to rad/s
    delta_nom = 0.5e-9
    
    F_nominal = casimir_augmented_force(d_array, R_nom, T_nom, omega_p_nom, delta_nom)
    
    param_uncertainties = {
        'R_sphere': 0.05 * R_nom,
        'T': 5.0,
        'omega_p': 0.10 * omega_p_nom,
        'delta_rough': 0.2e-9
    }
    
    F_min_1s = np.copy(F_nominal)
    F_max_1s = np.copy(F_nominal)
    F_min_2s = np.copy(F_nominal)
    F_max_2s = np.copy(F_nominal)
    
    for key, sigma_val in param_uncertainties.items():
        for mult in [-2, -1, 1, 2]:
            p_test = {
                'R_sphere': R_nom,
                'T': T_nom,
                'omega_p': omega_p_nom,
                'delta_rough': delta_nom
            }
            p_test[key] += mult * sigma_val
            
            F_test = casimir_augmented_force(d_array, p_test['R_sphere'], p_test['T'], p_test['omega_p'], p_test['delta_rough'])
            
            if abs(mult) == 1:
                F_min_1s = np.minimum(F_min_1s, F_test)
                F_max_1s = np.maximum(F_max_1s, F_test)
            elif abs(mult) == 2:
                F_min_2s = np.minimum(F_min_2s, F_test)
                F_max_2s = np.maximum(F_max_2s, F_test)

    # Convert to pN
    pN_conv = 1e12
    F_nominal_pN = np.abs(F_nominal) * pN_conv
    F_min_1s_pN = np.abs(F_max_1s) * pN_conv  # max absolute is min signed since force is negative
    F_max_1s_pN = np.abs(F_min_1s) * pN_conv
    F_min_2s_pN = np.abs(F_max_2s) * pN_conv
    F_max_2s_pN = np.abs(F_min_2s) * pN_conv

    df_err = pd.DataFrame({
        'd_nm': d_array * 1e9,
        'F_nominal_pN': F_nominal_pN,
        'F_minus_1sigma_pN': F_min_1s_pN,
        'F_plus_1sigma_pN': F_max_1s_pN,
        'F_minus_2sigma_pN': F_min_2s_pN,
        'F_plus_2sigma_pN': F_max_2s_pN
    })
    os.makedirs('output', exist_ok=True)
    df_err.to_csv('output/systematic_errors.csv', index=False)

    plt.figure(figsize=(8, 6))
    plt.loglog(d_array * 1e9, F_nominal_pN, label='Nominal', color='black', linewidth=2)
    plt.fill_between(d_array * 1e9, F_min_1s_pN, F_max_1s_pN, color='blue', alpha=0.4, label='1 Sigma Envelope')
    plt.fill_between(d_array * 1e9, F_min_2s_pN, F_max_2s_pN, color='lightblue', alpha=0.2, label='2 Sigma Envelope')
    plt.xlabel('Separation Distance d (nm)')
    plt.ylabel('|Force| (pN)')
    plt.title('Systematic Error Envelope')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.savefig('output/fig_systematic_envelope.png', dpi=300)
    plt.close()
    print("Module B: Systematic Error generated successfully.")

if __name__ == '__main__':
    run_systematic_error()
"""

# --- MÓDULO C: bootstrap_analysis.py ---
bootstrap_code = """# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_bootstrap_analysis(data_path):
    \"\"\"
    Bootstrap demonstration uses Zenodo theoretical dataset as proof-of-concept. 
    Application to experimental data (Decca 2003) is required for scientific conclusions.
    \"\"\"
    df = pd.read_csv(data_path)
    # Using BCS as ground truth empirical and Drude as theoretical model
    P_meas = df['P BCS (Pa)'].values
    sigma_noise = 0.001 * np.abs(P_meas)
    P_exp = P_meas + np.random.normal(0, sigma_noise, size=len(P_meas))
    P_model = df['P Drude (Pa)'].values
    
    chi2_0 = np.sum(((P_exp - P_model) / sigma_noise)**2)
    residuals = P_exp - P_model
    
    n_boot = 1000
    chi2_boot = np.zeros(n_boot)
    
    for i in range(n_boot):
        # Resample residuals with replacement
        res_boot = np.random.choice(residuals, size=len(residuals), replace=True)
        P_boot = P_model + res_boot
        chi2_boot[i] = np.sum(((P_boot - P_model) / sigma_noise)**2)
        
    chi2_mean = np.mean(chi2_boot)
    chi2_std = np.std(chi2_boot)
    p16 = np.percentile(chi2_boot, 16)
    p84 = np.percentile(chi2_boot, 84)
    p2_5 = np.percentile(chi2_boot, 2.5)
    p97_5 = np.percentile(chi2_boot, 97.5)
    p_value_boot = np.sum(chi2_boot > chi2_0) / n_boot
    
    os.makedirs('output', exist_ok=True)
    df_boot = pd.DataFrame({
        'chi2_nominal': [chi2_0],
        'chi2_mean': [chi2_mean],
        'chi2_std': [chi2_std],
        'p16': [p16],
        'p84': [p84],
        'p2.5': [p2_5],
        'p97.5': [p97_5],
        'p_value_bootstrap': [p_value_boot]
    })
    df_boot.to_csv('output/bootstrap_stats.csv', index=False)
    
    plt.figure(figsize=(8, 6))
    plt.hist(chi2_boot, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
    plt.axvline(chi2_0, color='red', linestyle='dashed', linewidth=2, label=f'Nominal $\chi^2$ = {chi2_0:.2f}')
    plt.axvline(p2_5, color='gray', linestyle='dotted', label='2.5%')
    plt.axvline(p97_5, color='gray', linestyle='dotted', label='97.5%')
    plt.xlabel('Bootstrap $\chi^2$')
    plt.ylabel('Frequency')
    plt.title('Non-parametric Bootstrap of $\chi^2$')
    plt.legend()
    plt.savefig('output/fig_bootstrap_chi2.png', dpi=300)
    plt.close()
    print("Module C: Bootstrap Analysis generated successfully.")

if __name__ == '__main__':
    data_file = 'data/Casimir_pressure_dist_v4.csv'
    if os.path.exists(data_file):
        run_bootstrap_analysis(data_file)
    else:
        print(f"File {data_file} not found. Cannot run bootstrap.")
"""

with open("src/sensitivity_table.py", "w", encoding="utf-8") as f:
    f.write(sensitivity_code)
with open("src/systematic_error.py", "w", encoding="utf-8") as f:
    f.write(systematic_code)
with open("src/bootstrap_analysis.py", "w", encoding="utf-8") as f:
    f.write(bootstrap_code)
