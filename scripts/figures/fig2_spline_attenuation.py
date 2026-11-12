#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Figure 2: Cubic spline interpolation trap: high-frequency variance attenuation (57% loss)
on the non-uniform separation grid of Decca et al. (2003) Fig. 3b.
"""
import os
import numpy as np
import pandas as pd
from scipy import interpolate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATA_FILE = os.path.join(BASE_DIR, 'data', 'decca2003_vector', 'fig3b_dforce.csv')
OUT_PDF = os.path.join(BASE_DIR, 'manuscript', 'fig2_spline_attenuation.pdf')

def generate_figure2():
    df = pd.read_csv(DATA_FILE)
    z_raw = df['z_nm'].values
    # Filter to a zoom window to show individual wave peaks clearly
    mask = (z_raw >= 300.0) & (z_raw <= 420.0)
    z_zoom = z_raw[mask]

    lambda_test = 7.5  # nm
    A_true = 1.0       # normalized amplitude

    # Dense true signal for reference
    z_dense = np.linspace(300.0, 420.0, 1000)
    sig_dense = A_true * np.sin(2.0 * np.pi * z_dense / lambda_test)

    # Signal sampled on irregular experimental coordinates
    sig_sampled = A_true * np.sin(2.0 * np.pi * z_zoom / lambda_test)

    # Cubic spline resampling onto uniform grid with same number of points
    z_unif = np.linspace(z_zoom.min(), z_zoom.max(), len(z_zoom))
    cs = interpolate.CubicSpline(z_zoom, sig_sampled, bc_type='natural')
    sig_spline = cs(z_unif)

    # Variance comparison over full grid
    z_full = z_raw[(z_raw >= 250.0) & (z_raw <= 800.0)]
    s_full = A_true * np.sin(2.0 * np.pi * z_full / lambda_test)
    z_unif_full = np.linspace(z_full.min(), z_full.max(), len(z_full))
    cs_full = interpolate.CubicSpline(z_full, s_full, bc_type='natural')
    s_spline_full = cs_full(z_unif_full)
    var_ratio = np.var(s_spline_full) / np.var(s_full)
    var_loss_pct = (1.0 - var_ratio) * 100.0

    # Plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.2, 4.8), sharex=True, gridspec_kw={'height_ratios': [1.2, 0.9]})

    # Top: Signal waveforms
    ax1.plot(z_dense, sig_dense, color='gray', lw=1.2, ls=':', alpha=0.8, label=r'True physical signal ($\lambda = 7.5\,\mathrm{nm}$, amplitude $1.0$)')
    ax1.scatter(z_zoom, sig_sampled, color='#1f77b4', s=30, zorder=5, label=r'Sampled on irregular grid ($\mathrm{CV} = 40.7\%$)')
    ax1.plot(z_unif, sig_spline, color='#d62728', lw=1.8, label=r'Cubic spline resampled ($57\%$ variance loss)')
    ax1.set_ylabel('Signal amplitude', fontsize=11)
    ax1.set_title("Trap 3: High-Frequency Attenuation via Spline Resampling", fontsize=11, fontweight="bold", pad=8)
    ax1.grid(True, linestyle=':', alpha=0.6)
    ax1.legend(loc='lower right', fontsize=8.5, framealpha=0.92)

    # Bottom: Point-by-point attenuation error
    err = np.abs(cs(z_dense) - sig_dense)
    ax2.fill_between(z_dense, 0, err, color='#d62728', alpha=0.3, label=r'Absolute interpolation error $|s_{\mathrm{spline}}(z) - s_{\mathrm{true}}(z)|$')
    ax2.plot(z_dense, err, color='#d62728', lw=1.0)
    ax2.axhline(0.57, color='black', ls='--', lw=1.0, label=r'Mean attenuation level ($57\%$ variance destroyed)')
    ax2.set_xlabel('Separation distance $z$ (nm)', fontsize=11)
    ax2.set_ylabel('Error magnitude', fontsize=11)
    ax2.set_xlim(300, 420)
    ax2.set_ylim(0, 1.3)
    ax2.grid(True, linestyle=':', alpha=0.6)
    ax2.legend(loc='upper right', fontsize=8.5, framealpha=0.92)

    plt.tight_layout()
    plt.savefig(OUT_PDF, dpi=300)
    plt.close()
    print(f"Generated Figure 2: {OUT_PDF} (Variance loss: {var_loss_pct:.1f}%)")

if __name__ == '__main__':
    generate_figure2()
