# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_bootstrap_analysis(data_path):
    """
    Bootstrap demonstration uses Zenodo theoretical dataset as proof-of-concept. 
    Application to experimental data (Decca 2003) is required for scientific conclusions.
    """
    df = pd.read_csv(data_path)
    # Using BCS as ground truth empirical and Drude as theoretical model
    P_meas = df['P BCS (Pa)'].values
    sigma_noise = 0.001 * np.abs(P_meas)
    P_exp = P_meas + np.random.normal(0, sigma_noise, size=len(P_meas))
    P_model = df['P Drude (Pa)'].values
    
    chi2_0 = np.sum(((P_exp - P_model) / sigma_noise)**2)
    std_residuals = (P_exp - P_model) / sigma_noise

    n_boot = 1000
    chi2_boot = np.zeros(n_boot)

    for i in range(n_boot):
        res_boot = np.random.choice(std_residuals, size=len(std_residuals), replace=True)
        chi2_boot[i] = np.sum(res_boot**2)
        
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
