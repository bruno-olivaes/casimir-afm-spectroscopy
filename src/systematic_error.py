# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from casimir_fft_pipeline import generate_casimir_data, PI

def run_systematic_error():
    """
    Envelope computed by varying R (+/-5%, +/-10%) and T (+/-5K) around nominal values, 
    using the validated generate_casimir_data() model. No asymptotic expansions are used.
    """
    d_array = np.linspace(100e-9, 2e-6, 500)
    
    # Nominals
    R_nom = 150e-6
    T_nom = 300.0
    
    P_nominal = generate_casimir_data(d_array, T_nom)
    F_nominal = P_nominal * PI * (R_nom**2)
    
    all_curves = [np.abs(F_nominal)]
    
    # Varying T
    for T_test in [295.0, 305.0]:
        P_test = generate_casimir_data(d_array, T_test)
        F_test = P_test * PI * (R_nom**2)
        all_curves.append(np.abs(F_test))
        
    # Varying R_sphere
    for R_mult in [0.90, 0.95, 1.05, 1.10]:
        R_test = R_nom * R_mult
        F_test = P_nominal * PI * (R_test**2)
        all_curves.append(np.abs(F_test))
        
    F_envelope_min = np.min(all_curves, axis=0)
    F_envelope_max = np.max(all_curves, axis=0)

    # Convert to pN
    pN_conv = 1e12
    F_nominal_pN = np.abs(F_nominal) * pN_conv
    F_envelope_min_pN = F_envelope_min * pN_conv
    F_envelope_max_pN = F_envelope_max * pN_conv

    df_err = pd.DataFrame({
        'd_nm': d_array * 1e9,
        'F_nominal_pN': F_nominal_pN,
        'F_envelope_min_pN': F_envelope_min_pN,
        'F_envelope_max_pN': F_envelope_max_pN
    })
    os.makedirs('output', exist_ok=True)
    df_err.to_csv('output/systematic_errors.csv', index=False)

    plt.figure(figsize=(8, 6))
    plt.loglog(d_array * 1e9, F_nominal_pN, label='Nominal', color='black', linewidth=2)
    plt.fill_between(d_array * 1e9, F_envelope_min_pN, F_envelope_max_pN, color='blue', alpha=0.3, label='Systematic Envelope')
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
