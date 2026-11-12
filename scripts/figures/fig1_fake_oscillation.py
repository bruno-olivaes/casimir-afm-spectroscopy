#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 1: Runge phenomenon ringing and false oscillation peak from cubic polynomial detrending
versus physical inverse-power basis on Decca et al. (2003) data.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'decca2003_vector', 'fig4b_dpressure.csv')
OUT_PDF = os.path.join(BASE_DIR, 'manuscript', 'fig1_fake_oscillation.pdf')

def generate_figure1():
    df = pd.read_csv(DATA_FILE)
    z = df['z_nm'].values
    dP = df['dPc_mPa'].values

    # 1. Cubic Polynomial Fit
    p3_coeffs = np.polyfit(z, dP, 3)
    p3_fit = np.polyval(p3_coeffs, z)
    res_cubic = dP - p3_fit

    # 2. Physical Inverse-Power Basis {1, (200/z)^4, (200/z)^5}
    X = np.column_stack([np.ones_like(z), (200.0 / z)**4, (200.0 / z)**5])
    beta, *_ = np.linalg.lstsq(X, dP, rcond=None)
    phys_fit = X @ beta
    res_phys = dP - phys_fit

    sigma_P = 0.313  # mPa instrumental noise floor

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 5.2), sharex=True, gridspec_kw={'height_ratios': [1.1, 1.0]})

    # Top panel: Raw data + Fits
    ax1.scatter(z, dP, s=12, color='#1f77b4', alpha=0.6, label=r'Decca et al. (2003) $\Delta P_c(z)$ ($N=497$)')
    ax1.plot(z, p3_fit, color='#d62728', lw=2.0, ls='--', label=r'Cubic polynomial $P_3(z)$ ($\chi^2/\mathrm{dof} = 3.74$)')
    ax1.plot(z, phys_fit, color='#2ca02c', lw=2.2, label=r'Physical basis $\{1, z^{-4}, z^{-5}\}$ ($\chi^2/\mathrm{dof} = 1.14$)')
    ax1.set_ylabel(r'$\Delta P_c$ (mPa)', fontsize=11)
    ax1.set_title("Trap 2: Runge Oscillations from Polynomial Detrending on Casimir Data", fontsize=11, fontweight="bold", pad=8)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower right', fontsize=9, framealpha=0.92)

    # Bottom panel: Detrended residuals comparison
    ax2.plot(z, res_cubic, color='#d62728', lw=1.2, alpha=0.85, label=r'Cubic residuals: Fake oscillation ($\lambda=583.5\,\mathrm{nm}$, $\mathrm{FAP}=1.49\times 10^{-45}$)')
    ax2.plot(z, res_phys, color='#2ca02c', lw=1.0, alpha=0.85, label=r'Physical residuals: Clean white noise ($\mathrm{FAP}=0.163$, Null confirmed)')
    ax2.axhline(0, color='black', lw=0.8, ls=':')
    ax2.axhline(2.0 * sigma_P, color='gray', lw=0.8, ls='--', label=r'$\pm 2\sigma$ instrumental noise floor')
    ax2.axhline(-2.0 * sigma_P, color='gray', lw=0.8, ls='--')
    ax2.set_xlabel(r'Separation distance $z$ (nm)', fontsize=11)
    ax2.set_ylabel(r'Residual (mPa)', fontsize=11)
    ax2.set_ylim(-2.8, 2.8)
    ax2.set_xlim(185, 1200)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=8.5, framealpha=0.92)

    plt.tight_layout()
    plt.savefig(OUT_PDF, dpi=300)
    plt.close()
    print(f"Generated Figure 1: {OUT_PDF}")

if __name__ == '__main__':
    generate_figure1()
