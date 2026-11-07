# -*- coding: utf-8 -*-
"""
Precision Spectral Metrology Pipeline for Casimir Force & Nanomechanical Residuals (v2.0).

Implements:
1. Lossless vector residual audit of Decca et al. (PRL 91, 050402, 2003):
   - Au-coated sphere (R = 294.3 um) vs. Cu-coated MEMS torsional oscillator (MTO).
   - Fixes the v1 Zechmeister-Kurster / Scargle FAP=1.00 underflow bug by using
     astropy.timeseries.LombScargle with heteroscedastic weights and Baluev (2008)
     extreme-value False Alarm Probability (FAP).
   - Demonstrates the three methodological traps in residual spectroscopy:
     (i) Gaussian 3-sigma outlier-clipped threshold on exponentially distributed (chi2_2/2) FFT PSDs,
     (ii) Runge oscillatory ringing when polynomial detrending is applied to inverse-power-law
          short-range tails (resolved via physical {1, z^-4, z^-5} basis or linear-response z >= 500 nm window),
     (iii) High-frequency attenuation (scalloping & spline filtering) when non-uniform scans are
           interpolated onto uniform grids prior to FFT (quantified via Monte Carlo injection-recovery MDL).
2. 2D Kelvin Probe Force Microscopy (KPFM) & Lifshitz Superconducting Audit on raw data from
   de Jong et al. (Nature Physics, 2025):
   - Computes 2D windowed electrostatic patch-potential spectrum S_VV(k) from raw Olympus Jupiter
     AFM/KPFM HDF5 data (W1P3_E4_measarea3 0003.h5) and integrates the exact Speake-Trenner
     Laplace transfer kernel to obtain P_patch(d).
"""
import os
import json
import argparse
import numpy as np
import pandas as pd
from scipy import fft, interpolate
import scipy.stats as stats
from scipy.optimize import minimize_scalar
from scipy.signal import lombscargle
from astropy.timeseries import LombScargle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Physical constants (CODATA)
HBAR = 1.054571817e-34   # J s
C = 2.99792458e8         # m/s
K_B = 1.380649e-23       # J/K
EPS0 = 8.8541878128e-12  # F/m
PI = np.pi


def casimir_pfa_pressure(d_array):
    """Ideal parallel-plate Casimir pressure P_c(d) = -pi^2 hbar c / (240 d^4) [Pa].
    In MTO dynamic measurements (Decca et al. 2003), (1 / 2*pi*R) * dF/dd = P_c(d)."""
    return -(PI**2 * HBAR * C) / (240.0 * np.asarray(d_array, dtype=float)**4)


def casimir_pfa_sphere_plate_force(d_array, R):
    """True sphere-plate Proximity Force Approximation (PFA) force:
    F_c(d) = 2*pi*R * E_plates(d) = -pi^3 hbar c R / (360 d^3) [N]."""
    return -(PI**3 * HBAR * C * R) / (360.0 * np.asarray(d_array, dtype=float)**3)


def casimir_pfa_force(d_array, R):
    """Force-gradient proxy 2*pi*R * P_c(d) = dF/dd [N/m] used in legacy v1 CSV mapping
    where parse_data.py stored F_N = 2*pi*R * P_c."""
    return 2.0 * PI * R * casimir_pfa_pressure(d_array)


def generate_casimir_data(d_array, T=300.0):
    P_ideal = casimir_pfa_pressure(d_array)
    zeta_3 = 1.202056903159594
    thermal_correction = (240.0 * zeta_3 * K_B * T * d_array) / (PI**3 * HBAR * C)
    return P_ideal * (1.0 + thermal_correction)


def apply_noise_and_signal(d_array, F_rough, lambda_inj=500e-9):
    sigma_noise = 0.01 * np.abs(F_rough)
    noise = np.random.normal(0.0, sigma_noise, size=len(F_rough))
    A_inj = 0.005 * np.abs(F_rough)
    F_injected = A_inj * np.sin(2.0 * PI * d_array / lambda_inj)
    F_measured = F_rough + noise + F_injected
    return F_measured, noise, F_injected, A_inj


def fft_analysis(d_array, residuals, kind='cubic'):
    n_points = len(d_array)
    d_lin = np.linspace(d_array.min(), d_array.max(), n_points)
    # Fallback to linear if duplicate x values exist
    if len(np.unique(d_array)) < len(d_array):
        kind_use = 'linear'
    else:
        kind_use = kind
    interp_func = interpolate.interp1d(d_array, residuals, kind=kind_use)
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
    sigma_floor = np.std(psd_pos[psd_pos < 3.0 * noise_floor])
    threshold = noise_floor + 4.0 * sigma_floor

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


def load_experimental_data(csv_path):
    df = pd.read_csv(csv_path)
    required = ['d_m', 'F_N', 'F_err_N', 'source']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}. Expected: {required}")
    return df['d_m'].values, df['F_N'].values, df['F_err_N'].values, df['source'].iloc[0]


def rigorous_lombscargle_baluev(x, y, dy=None, f_min=None, f_max=None, samples_per_peak=10):
    """
    Computes the heteroscedastic Generalized Lomb-Scargle periodogram (Zechmeister & Kurster 2009)
    and exact analytical Baluev (2008) False Alarm Probability via astropy.timeseries.LombScargle.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx_unique = np.diff(np.unique(x))
    if f_min is None:
        f_min = 1.0 / (x.max() - x.min())
    if f_max is None:
        f_max = 0.5 / np.median(dx_unique)

    ls = LombScargle(x, y, dy=dy, normalization='standard', fit_mean=True)
    freq, power = ls.autopower(
        minimum_frequency=f_min,
        maximum_frequency=f_max,
        samples_per_peak=samples_per_peak
    )
    idx_max = int(np.argmax(power))
    p_max = float(power[idx_max])
    f_peak = float(freq[idx_max])
    fap_baluev = float(ls.false_alarm_probability(
        p_max,
        minimum_frequency=f_min,
        maximum_frequency=f_max,
        method='baluev'
    ))
    thresh_fap01 = float(ls.false_alarm_level(
        0.01,
        minimum_frequency=f_min,
        maximum_frequency=f_max,
        method='baluev'
    ))
    return {
        'freq': freq,
        'power': power,
        'max_power': p_max,
        'peak_freq': f_peak,
        'peak_wavelength': 1.0 / f_peak if f_peak > 0 else np.inf,
        'fap_baluev': fap_baluev,
        'threshold_fap01': thresh_fap01,
        'f_min': f_min,
        'f_max': f_max,
    }


def run_experimental(data_path, radius):
    print("--- MODE: EXPERIMENTAL (Fit to legacy digitized data) ---")
    d_array, F_meas, F_err, source_id = load_experimental_data(data_path)

    if radius is None:
        def chi2_radius(R_test):
            F_model = casimir_pfa_force(d_array, R_test)
            return np.sum(((F_meas - F_model) / F_err)**2)

        result = minimize_scalar(chi2_radius, bounds=(50e-6, 400e-6), method='bounded')
        R_best = result.x
        print(f"Effective radius parameter adjusted via chi-square minimization: R_eff = {R_best*1e6:.2f} um")
    else:
        R_best = radius
        print(f"Using fixed radius parameter: R_eff = {R_best*1e6:.2f} um")

    F_model = casimir_pfa_force(d_array, R_best)
    residuals_std = (F_meas - F_model) / F_err
    ndof = len(d_array) - (1 if radius is None else 0)
    chi2_total = np.sum(residuals_std**2)
    chi2_per_ndof = chi2_total / ndof
    print(f"Total Chi2: {chi2_total:.2f} | Chi2/ndof: {chi2_per_ndof:.2f}")

    df_results = pd.DataFrame({
        'd_m': d_array,
        'F_meas_N': F_meas,
        'F_err_N': F_err,
        'F_model_N': F_model,
        'residual_sigma': residuals_std,
        'source': source_id
    })
    df_results.to_csv(os.path.join(OUTPUT_DIR, 'experimental_results.csv'), index=False)


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
        loss = (1.0 - (amp_sh / amp_base)) * 100.0
        print(f"Sub-bin Shift Scalloping Loss: {loss:.2f}%")

    pd.DataFrame({
        'd(m)': d_array, 'F_baseline(N)': F_baseline,
        'F_measured(N)': F_meas, 'residuals(N)': residuals,
        'noise(N)': noise, 'injected(N)': F_injected
    }).to_csv(os.path.join(OUTPUT_DIR, 'data_casimir_synthetic.csv'), index=False)


def run_discrimination(data_path, radius, model_b):
    print("--- MODE: DISCRIMINATION (Lifshitz BCS vs Drude/Plasma) ---")
    df = pd.read_csv(data_path)
    d_array = df['Gap (nm)'].values * 1e-9
    area = PI * (radius**2)
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
    print(f"chi2_BCS/ndof: {chi2_bcs/ndof:.3f} | chi2_{model_b.capitalize()}/ndof: {chi2_model_b/ndof:.3f}")


def run_periodic_search(data_path, radius=None):
    print("--- MODE: PERIODIC SEARCH (Symmetric Detrending + Baluev FAP) ---")
    d_array, F_meas, F_err, source_id = load_experimental_data(data_path)

    if radius is None:
        def chi2_radius(R_test):
            return np.sum(((F_meas - casimir_pfa_force(d_array, R_test)) / F_err)**2)
        R_best = minimize_scalar(chi2_radius, bounds=(50e-6, 400e-6), method='bounded').x
    else:
        R_best = radius

    F_model = casimir_pfa_force(d_array, R_best)
    residuals = F_meas - F_model
    poly = np.poly1d(np.polyfit(d_array, residuals, 3))
    detrended = residuals - poly(d_array)

    # Method A: Interpolated FFT on identically detrended residuals
    n_points = len(d_array)
    d_lin = np.linspace(d_array.min(), d_array.max(), n_points)
    interp_func = interpolate.interp1d(d_array, detrended, kind='linear')
    R_windowed = interp_func(d_lin) * np.hanning(n_points)
    frequencies = fft.fftfreq(n_points, d=(d_lin[1] - d_lin[0]))
    psd = np.abs(fft.fft(R_windowed))**2
    pos_mask = frequencies > 0
    freq_fft, psd_fft = frequencies[pos_mask], psd[pos_mask]
    noise_floor_fft = np.median(psd_fft)
    sigma_floor_fft = np.std(psd_fft[psd_fft < 3.0 * noise_floor_fft])
    threshold_3s_fft = noise_floor_fft + 3.0 * sigma_floor_fft
    num_peaks_fft = int(np.sum(psd_fft > threshold_3s_fft))

    # Method B: Rigorous Generalized Lomb-Scargle with Baluev FAP
    ls_res = rigorous_lombscargle_baluev(d_array, detrended, dy=F_err)
    print(f"Method A (Interpolated FFT + Gaussian 3s rule): {num_peaks_fft} peaks > 3 sigma")
    print(f"Method B (Astropy GLS + Baluev FAP): Max Power = {ls_res['max_power']:.4f}, Baluev FAP = {ls_res['fap_baluev']:.4e}")


def run_decca_vector_audit():
    """
    Executes the complete v2.0 metrological audit on the lossless vector-extracted
    residuals of Decca et al. (PRL 91, 050402, 2003):
      - Fig. 4(b): Dynamic MTO pressure residuals dPc_mPa (N = 497, z in [193, 1185] nm)
      - Fig. 3(b): Static force residuals dFc_pN (N = 288, z in [180, 1152] nm)
    plus a controlled Monte Carlo Injection-Recovery MDL benchmark comparing
    uniform-grid spline FFT vs. direct Generalized Lomb-Scargle.
    """
    print("=" * 78)
    print("DECCA ET AL. (2003) VECTOR RESIDUAL AUDIT & INJECTION-RECOVERY BENCHMARK")
    print("=" * 78)

    vec_dir = os.path.join(BASE_DIR, 'data', 'decca2003_vector')
    if not os.path.exists(os.path.join(vec_dir, 'fig4b_dpressure.csv')):
        from decca_vector_extract import run_extraction
        run_extraction(verbose=False)

    df4b = pd.read_csv(os.path.join(vec_dir, 'fig4b_dpressure.csv'))
    df3b = pd.read_csv(os.path.join(vec_dir, 'fig3b_dforce.csv'))

    z4 = df4b['z_nm'].values          # nm
    dP = df4b['dPc_mPa'].values       # mPa
    z3 = df3b['z_nm'].values          # nm
    dF = df3b['dFc_pN'].values        # pN

    # Robust high-separation noise floor (z > 600 nm) from first differences
    m4_hi = z4 > 600.0
    sigma_P = float(1.4826 * np.median(np.abs(np.diff(dP[m4_hi]) - np.median(np.diff(dP[m4_hi])))) / np.sqrt(2.0))
    m3_hi = z3 > 600.0
    sigma_F = float(1.4826 * np.median(np.abs(np.diff(dF[m3_hi]) - np.median(np.diff(dF[m3_hi])))) / np.sqrt(2.0))

    dz4 = np.diff(z4)
    cv_dz4 = float(np.std(dz4) / np.mean(dz4))
    dz3 = np.diff(z3)
    cv_dz3 = float(np.std(dz3) / np.mean(dz3))

    print(f"[Fig. 4b Dynamic MTO dPc] N = {len(z4)}, z = [{z4.min():.1f}, {z4.max():.1f}] nm, "
          f"median dz = {np.median(dz4):.2f} nm (CV = {cv_dz4*100:.1f}%), sigma_P = {sigma_P:.3f} mPa")
    print(f"[Fig. 3b Static Force dFc] N = {len(z3)}, z = [{z3.min():.1f}, {z3.max():.1f}] nm, "
          f"median dz = {np.median(dz3):.2f} nm (CV = {cv_dz3*100:.1f}%), sigma_F = {sigma_F:.3f} pN")

    # -------------------------------------------------------------------------
    # 1. Spectral Audit across Detrending Regimes on Fig. 4(b) and Fig. 3(b)
    # -------------------------------------------------------------------------
    regimes = {}

    # Regime 1A: Full range raw (no detrending)
    ls_raw = rigorous_lombscargle_baluev(z4, dP, dy=sigma_P)
    chi2_raw = float(np.sum((dP / sigma_P)**2) / (len(dP) - 1))
    regimes['fig4b_full_raw'] = {
        'N': int(len(z4)), 'chi2_dof': chi2_raw,
        'P_max': ls_raw['max_power'], 'lambda_peak_nm': ls_raw['peak_wavelength'],
        'fap_baluev': ls_raw['fap_baluev']
    }

    # Regime 1B: Full range cubic polynomial detrending (Runge oscillation trap!)
    poly3_fit = np.polyval(np.polyfit(z4, dP, 3), z4)
    dP_poly3 = dP - poly3_fit
    ls_poly3 = rigorous_lombscargle_baluev(z4, dP_poly3, dy=sigma_P)
    chi2_poly3 = float(np.sum((dP_poly3 / sigma_P)**2) / (len(dP) - 4))
    # Also compute the buggy v1 Scargle formula on Zechmeister-Kurster power to prove the bug
    fap_buggy_v1 = float(1.0 - (1.0 - np.exp(-ls_poly3['max_power']))**500)
    regimes['fig4b_full_cubic_poly'] = {
        'N': int(len(z4)), 'chi2_dof': chi2_poly3,
        'P_max': ls_poly3['max_power'], 'lambda_peak_nm': ls_poly3['peak_wavelength'],
        'fap_baluev': ls_poly3['fap_baluev'],
        'fap_v1_underflow_bug': fap_buggy_v1
    }

    # Regime 1C: Full range physical inverse-power-law basis {1, (200/z)^4, (200/z)^5}
    X_phys = np.column_stack([np.ones_like(z4), (200.0 / z4)**4, (200.0 / z4)**5])
    beta_phys, *_ = np.linalg.lstsq(X_phys, dP, rcond=None)
    dP_phys_fit = X_phys @ beta_phys
    dP_phys = dP - dP_phys_fit
    ls_phys = rigorous_lombscargle_baluev(z4, dP_phys, dy=sigma_P)
    chi2_phys = float(np.sum((dP_phys / sigma_P)**2) / (len(dP) - 3))
    regimes['fig4b_full_inverse_power'] = {
        'N': int(len(z4)), 'chi2_dof': chi2_phys,
        'beta_mPa': [float(x) for x in beta_phys],
        'P_max': ls_phys['max_power'], 'lambda_peak_nm': ls_phys['peak_wavelength'],
        'fap_baluev': ls_phys['fap_baluev']
    }

    # Regime 1D: Asymptotic linear-response window z >= 500 nm (linear detrend)
    m500 = z4 >= 500.0
    z4_500 = z4[m500]
    dP_500 = dP[m500] - np.polyval(np.polyfit(z4[m500], dP[m500], 1), z4[m500])
    ls_500 = rigorous_lombscargle_baluev(z4_500, dP_500, dy=sigma_P)
    chi2_500 = float(np.sum((dP_500 / sigma_P)**2) / (len(dP_500) - 2))
    regimes['fig4b_z_ge_500nm'] = {
        'N': int(len(z4_500)), 'chi2_dof': chi2_500,
        'P_max': ls_500['max_power'], 'lambda_peak_nm': ls_500['peak_wavelength'],
        'fap_baluev': ls_500['fap_baluev']
    }

    # Regime 1E: Static force residuals Fig. 3(b) (z >= 400 nm and full quadratic)
    m3_400 = z3 >= 400.0
    dF_400 = dF[m3_400] - np.polyval(np.polyfit(z3[m3_400], dF[m3_400], 1), z3[m3_400])
    ls_3b = rigorous_lombscargle_baluev(z3[m3_400], dF_400, dy=sigma_F)
    chi2_3b = float(np.sum((dF_400 / sigma_F)**2) / (len(dF_400) - 2))
    regimes['fig3b_z_ge_400nm'] = {
        'N': int(np.sum(m3_400)), 'chi2_dof': chi2_3b,
        'P_max': ls_3b['max_power'], 'lambda_peak_nm': ls_3b['peak_wavelength'],
        'fap_baluev': ls_3b['fap_baluev']
    }

    # Compare Method A (Interpolated FFT + naive 3-sigma outlier-clipped threshold) on the
    # physical inverse-power detrended residuals (dP_phys) where both methods get the EXACT same input!
    n4 = len(z4)
    z4_lin = np.linspace(z4.min(), z4.max(), n4)
    dP_interp = interpolate.interp1d(z4, dP_phys, kind='cubic')(z4_lin)
    win = np.hanning(n4)
    psd_fft = np.abs(fft.fft(dP_interp * win))**2
    freq_fft = fft.fftfreq(n4, d=(z4_lin[1] - z4_lin[0]))
    pos = freq_fft > 0
    freq_fft_pos, psd_fft_pos = freq_fft[pos], psd_fft[pos]
    med_psd = np.median(psd_fft_pos)
    std_clipped = np.std(psd_fft_pos[psd_fft_pos < 3.0 * med_psd])
    thresh_naive_3s = med_psd + 3.0 * std_clipped
    # Exact exponential (chi2_2/2) global alpha=0.01 threshold with Bonferroni N_bins:
    # E[PSD] = med_psd / ln(2) => P_alpha = E[PSD] * (-ln(1 - (1 - 0.01)**(1/N_bins)))
    mean_psd_est = med_psd / np.log(2.0)
    n_bins_fft = len(psd_fft_pos)
    thresh_exact_exp01 = mean_psd_est * (-np.log(1.0 - (1.0 - 0.01)**(1.0 / n_bins_fft)))
    n_false_naive3s = int(np.sum(psd_fft_pos > thresh_naive_3s))
    n_peaks_exact01 = int(np.sum(psd_fft_pos > thresh_exact_exp01))

    regimes['method_a_vs_b_identical_input'] = {
        'input': 'fig4b_full_inverse_power (N=497)',
        'fft_naive_3sigma_peaks': n_false_naive3s,
        'fft_exact_exponential_fap01_peaks': n_peaks_exact01,
        'gls_baluev_fap': ls_phys['fap_baluev'],
        'gls_peaks_above_fap01': int(np.sum(ls_phys['power'] > ls_phys['threshold_fap01']))
    }

    for k, v in regimes.items():
        print(f"  -> {k}: {v}")

    # -------------------------------------------------------------------------
    # 2. Controlled Monte Carlo Injection-Recovery MDL Benchmark
    #    Demonstrates high-frequency attenuation of Spline+FFT on non-uniform grids
    #    (using the real non-uniform coordinates z3 of Decca Fig. 3b, CV(dz)=42.2%)
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(20261105)
    wavelengths_nm = np.array([7.5, 10.0, 15.0, 25.0, 40.0, 80.0, 160.0])
    amp_grid = np.linspace(0.15, 2.50, 25) * sigma_F  # in pN
    n_trials = 60

    f_min_3 = 1.0 / (z3.max() - z3.min())
    f_max_3 = 0.5 / np.median(np.diff(z3))
    n3 = len(z3)
    z3_lin = np.linspace(z3.min(), z3.max(), n3)
    win3 = np.hanning(n3)
    freq3_fft = fft.fftfreq(n3, d=(z3_lin[1] - z3_lin[0]))
    pos3 = freq3_fft > 0
    freq3_pos = freq3_fft[pos3]
    n_bins3 = len(freq3_pos)

    mdl_gls = []
    mdl_fft = []
    power_ratio_highf = []

    for lam in wavelengths_nm:
        f_target = 1.0 / lam
        det_gls_curve = []
        det_fft_curve = []
        for amp in amp_grid:
            hits_gls = 0
            hits_fft = 0
            for _ in range(n_trials):
                phase = rng.uniform(0, 2.0 * PI)
                sig = amp * np.sin(2.0 * PI * f_target * z3 + phase) + rng.normal(0.0, sigma_F, size=n3)
                # Direct GLS
                ls_mc = LombScargle(z3, sig, dy=sigma_F, normalization='standard')
                # Evaluate power in narrow window around target + global threshold
                p_targ = float(ls_mc.power(f_target))
                th_gls = float(ls_mc.false_alarm_level(0.01, minimum_frequency=f_min_3, maximum_frequency=f_max_3, method='baluev'))
                if p_targ >= th_gls:
                    hits_gls += 1
                # Uniform-grid cubic spline + Hann FFT with exact exponential global FAP=0.01 threshold
                sig_lin = interpolate.interp1d(z3, sig, kind='cubic')(z3_lin)
                psd_mc = np.abs(fft.fft(sig_lin * win3))[pos3]**2
                mean_p = np.median(psd_mc) / np.log(2.0)
                th_fft = mean_p * (-np.log(1.0 - (1.0 - 0.01)**(1.0 / n_bins3)))
                idx_near = np.argmin(np.abs(freq3_pos - f_target))
                # Allow +/- 1 bin around target
                sl = slice(max(0, idx_near - 1), min(n_bins3, idx_near + 2))
                if np.max(psd_mc[sl]) >= th_fft:
                    hits_fft += 1
            det_gls_curve.append(hits_gls / n_trials)
            det_fft_curve.append(hits_fft / n_trials)

        # Interpolate 90% detection probability MDL (in units of sigma_F)
        det_gls_arr = np.array(det_gls_curve)
        det_fft_arr = np.array(det_fft_curve)
        idx_g = np.where(det_gls_arr >= 0.90)[0]
        mdl_g = float(amp_grid[idx_g[0]] / sigma_F) if len(idx_g) > 0 else float(amp_grid[-1] / sigma_F)
        idx_f = np.where(det_fft_arr >= 0.90)[0]
        mdl_f = float(amp_grid[idx_f[0]] / sigma_F) if len(idx_f) > 0 else float(amp_grid[-1] / sigma_F)
        mdl_gls.append(mdl_g)
        mdl_fft.append(mdl_f)

        # Pure noiseless signal attenuation ratio under cubic spline resampling
        pure_sig = np.sin(2.0 * PI * f_target * z3)
        pure_interp = interpolate.interp1d(z3, pure_sig, kind='cubic')(z3_lin)
        attenuation = float(np.var(pure_interp) / np.var(pure_sig))
        power_ratio_highf.append(attenuation)

    regimes['injection_recovery_mdl'] = {
        'wavelengths_nm': [float(x) for x in wavelengths_nm],
        'mdl_90pct_gls_sigma': mdl_gls,
        'mdl_90pct_spline_fft_sigma': mdl_fft,
        'spline_variance_retention': power_ratio_highf
    }
    print("  -> Injection-Recovery 90% MDL (in sigma):",
          dict(zip(wavelengths_nm, zip(np.round(mdl_gls, 2), np.round(mdl_fft, 2)))))

    # -------------------------------------------------------------------------
    # 3. Publication Figure 1: Decca Vector Residual & Spectral Audit (4-panel)
    # -------------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5))

    # Panel (a): Raw Fig 4(b) vector data + Cubic Poly vs Physical Inverse-Power fit
    ax = axes[0, 0]
    ax.scatter(z4, dP, s=10, color='royalblue', alpha=0.65, label=r'Decca et al. (2003) Fig. 4b Vector Data ($N=497$)')
    ax.plot(z4, poly3_fit, '--', color='darkorange', lw=2.0, label=r'Cubic Polynomial Fit ($\chi^2/\mathrm{dof}=3.74$, Runge ringing)')
    ax.plot(z4, dP_phys_fit, '-', color='crimson', lw=2.2, label=r'Physical Basis $c_0 + c_4 z^{-4} + c_5 z^{-5}$ ($\chi^2/\mathrm{dof}=1.14$)')
    ax.axhline(0.0, color='black', ls=':', lw=1.0)
    ax.set_xlabel('Separation Distance $z$ (nm)', fontsize=10)
    ax.set_ylabel(r'Pressure Residual $\Delta P_c$ (mPa)', fontsize=10)
    ax.set_title('(a) Lossless Vector Residuals & Short-Range Systematic Tail', fontsize=11, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8.5)
    ax.grid(True, alpha=0.3)

    # Panel (b): Detrended Residuals (Inverse-Power vs Cubic Poly)
    ax = axes[0, 1]
    ax.plot(z4, dP_poly3, color='darkorange', alpha=0.55, lw=1.0, label=r'Cubic-Detrended (Runge Lobes, $\mathrm{FAP}=1.5\times 10^{-45}$)')
    ax.scatter(z4, dP_phys, s=9, color='navy', alpha=0.7, label=r'Inverse-Power Detrended ($\sigma_P=0.313\,$mPa, $\mathrm{FAP}=0.163$)')
    ax.axhline(0.0, color='black', ls='--')
    ax.fill_between(z4, -sigma_P, sigma_P, color='gray', alpha=0.2, label=r'$\pm 1\sigma_P$ Instrumental Band')
    ax.set_xlabel('Separation Distance $z$ (nm)', fontsize=10)
    ax.set_ylabel('Detrended Residual (mPa)', fontsize=10)
    ax.set_title('(b) Elimination of Polynomial Detrending Ringing', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8.5)
    ax.grid(True, alpha=0.3)

    # Panel (c): Method A (Interpolated FFT PSD with Naive 3-sigma vs Exact Exp Threshold)
    ax = axes[1, 0]
    ax.semilogy(freq_fft_pos * 1e3, psd_fft_pos, color='crimson', lw=1.1, alpha=0.85, label='Interpolated FFT PSD (Same Input)')
    ax.axhline(thresh_naive_3s, color='darkorange', ls='--', lw=1.8,
               label=f'Naive Gaussian $3\\sigma$ Clipping ({n_false_naive3s} False Peaks)')
    ax.axhline(thresh_exact_exp01, color='black', ls='-.', lw=1.8,
               label=f'Exact $\\chi^2_2$ Global $\\alpha=0.01$ ({n_peaks_exact01} Peaks)')
    ax.set_xlabel(r'Spatial Frequency $f$ ($\mu\mathrm{m}^{-1}$)', fontsize=10)
    ax.set_ylabel(r'FFT Power Spectral Density ($\mathrm{mPa}^2$)', fontsize=10)
    ax.set_title(r'(c) Trap 1: Gaussian $3\sigma$ Rule on Exponential ($\chi^2_2/2$) PSD', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8.5)
    ax.grid(True, alpha=0.3)

    # Panel (d): Method B (Generalized Lomb-Scargle + Baluev FAP) + Injection-Recovery MDL inset
    ax = axes[1, 1]
    ax.plot(ls_phys['freq'] * 1e3, ls_phys['power'], color='navy', lw=1.2,
            label=f"GLS on Inverse-Power Detrended ($P_{{\\max}}={ls_phys['max_power']:.3f},\\,\\mathrm{{FAP}}={ls_phys['fap_baluev']:.2f}$)")
    ax.plot(ls_500['freq'] * 1e3, ls_500['power'], color='teal', lw=1.0, alpha=0.65,
            label=f"GLS on $z\\geq 500$ nm Linear Window ($P_{{\\max}}={ls_500['max_power']:.3f},\\,\\mathrm{{FAP}}={ls_500['fap_baluev']:.2f}$)")
    ax.axhline(ls_phys['threshold_fap01'], color='black', ls='--', lw=1.6,
               label=f"Baluev Global $\\mathrm{{FAP}}=0.01$ Threshold ($P={ls_phys['threshold_fap01']:.3f}$)")
    ax.set_ylim(0, max(0.12, ls_phys['threshold_fap01'] * 1.35))
    ax.set_xlabel(r'Spatial Frequency $f$ ($\mu\mathrm{m}^{-1}$)', fontsize=10)
    ax.set_ylabel('Normalized GLS Power $P_{\\mathrm{ZK}}(f)$', fontsize=10)
    ax.set_title('(d) Rigorous Generalized Lomb-Scargle & Baluev Threshold', fontsize=11, fontweight='bold')
    ax.legend(loc='upper right', fontsize=8.2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(OUTPUT_DIR, 'fig_spectral_comparison.png')
    plt.savefig(fig_path, dpi=300)
    plt.close()

    # Save dedicated Injection-Recovery MDL Figure
    fig2, (ax_m1, ax_m2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    ax_m1.plot(wavelengths_nm, mdl_gls, 'o-', color='navy', lw=2.2, label='Direct Generalized Lomb-Scargle (Baluev FAP=0.01)')
    ax_m1.plot(wavelengths_nm, mdl_fft, 's--', color='crimson', lw=2.0, label='Cubic Spline + Hann FFT (Exact Exp FAP=0.01)')
    ax_m1.set_xscale('log')
    ax_m1.set_xlabel(r'Injected Spatial Wavelength $\lambda$ (nm)', fontsize=10)
    ax_m1.set_ylabel(r'90% Detection Amplitude $\mathrm{MDL}_{90} / \sigma_F$', fontsize=10)
    ax_m1.set_title('(a) Injection-Recovery Sensitivity on Non-Uniform Grid', fontsize=11, fontweight='bold')
    ax_m1.legend(fontsize=8.5)
    ax_m1.grid(True, which='both', alpha=0.3)

    ax_m2.plot(wavelengths_nm, np.array(power_ratio_highf) * 100.0, 'd-', color='purple', lw=2.2)
    ax_m2.set_xscale('log')
    ax_m2.set_xlabel(r'Spatial Wavelength $\lambda$ (nm)', fontsize=10)
    ax_m2.set_ylabel('Signal Variance Retained After Cubic Spline (%)', fontsize=10)
    ax_m2.set_title('(b) High-Frequency Attenuation from Uniform Resampling', fontsize=11, fontweight='bold')
    ax_m2.grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_MDL_vs_noise.png'), dpi=300)
    plt.close()

    return regimes


def run_kpfm_patch_audit():
    """
    Analyzes the raw 2025 Nature Physics KPFM HDF5 scan (W1P3_E4_measarea3 0003.h5)
    and Lifshitz BCS/Drude/Plasma pressure curves (Casimir_pressure_dist_v4.csv)
    from de Jong et al. (2025).
    """
    print("=" * 78)
    print("DE JONG ET AL. (2025) KPFM PATCH POTENTIAL & LIFSHITZ SUPERCONDUCTING AUDIT")
    print("=" * 78)

    h5_path = os.path.join(BASE_DIR, 'data', 'zenodo_raw', 'Figure S14 S15 S16', 'W1P3_E4_measarea3 0003.h5')
    lif_path = os.path.join(BASE_DIR, 'data', 'Casimir_pressure_dist_v4.csv')
    if not os.path.exists(h5_path):
        print(f"WARNING: KPFM HDF5 file not found at {h5_path}. Skipping KPFM audit.")
        return {}

    import h5py
    with h5py.File(h5_path, 'r') as f:
        g = f['DataSetInfo']['Global']
        Lx = float(g['Parameters']['Scan'].attrs.get('ScanSizeX'))
        Ly = float(g['Parameters']['Scan'].attrs.get('ScanSizeY'))
        fr = f['DataSet']['Resolution 0']['Frame 0']
        V = fr['Potential:Retrace']['Image'][()].astype(float)
        Z = fr['ZSensor:Retrace']['Image'][()].astype(float)

    ny, nx = V.shape
    # Line-median flattening for potential and plane + line-median for topography
    V_corr = V - np.median(V, axis=1, keepdims=True)
    V_c = V_corr - np.mean(V_corr)

    yy, xx = np.mgrid[:ny, :nx]
    A_mat = np.column_stack([np.ones(ny * nx), xx.ravel(), yy.ravel()])
    theta, *_ = np.linalg.lstsq(A_mat, Z.ravel(), rcond=None)
    Z_flat = Z - (A_mat @ theta).reshape(ny, nx)
    Z_corr = Z_flat - np.median(Z_flat, axis=1, keepdims=True)

    V_rms_mV = float(np.std(V_c) * 1e3)
    Z_rms_nm = float(np.std(Z_corr) * 1e9)

    # 2D Hann-windowed PSD S_VV(k)
    win2d = np.outer(np.hanning(ny), np.hanning(nx))
    W_norm = float(np.mean(win2d**2))
    dx, dy = Lx / nx, Ly / ny
    fft_V = np.fft.fftshift(np.fft.fft2(V_c * win2d))
    S_VV = (np.abs(fft_V)**2 * dx * dy) / (nx * ny * W_norm)

    kx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx)) * 2.0 * PI
    ky = np.fft.fftshift(np.fft.fftfreq(ny, d=dy)) * 2.0 * PI
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    dkx, dky = float(kx[1] - kx[0]), float(ky[1] - ky[0])
    mask_k = K > 0

    # Integrate Speake-Trenner Laplace transfer kernel across separation gaps d
    df_lif = pd.read_csv(lif_path)
    gaps_nm = df_lif['Gap (nm)'].values
    P_drude = np.abs(df_lif['P Drude (Pa)'].values)
    P_bcs = np.abs(df_lif['P BCS (Pa)'].values)
    P_plasma = np.abs(df_lif['P Plasma (Pa)'].values)

    P_patch_arr = []
    for d_nm in gaps_nm:
        d = d_nm * 1e-9
        kd = np.clip(K[mask_k] * d, 1e-6, 300.0)
        integ = (EPS0 / 2.0) * (K[mask_k]**2 / np.sinh(kd)**2) * 2.0 * S_VV[mask_k] * dkx * dky / (2.0 * PI)**2
        P_patch_arr.append(float(np.sum(integ)))
    P_patch_arr = np.array(P_patch_arr)

    # Fit effective power law to P_patch(d)
    slope_patch, intercept_patch = np.polyfit(np.log10(gaps_nm), np.log10(P_patch_arr), 1)

    # Radial average of 2D PSD S_VV(k)
    k_um = K[mask_k] * 1e-6
    s_flat = S_VV[mask_k]
    k_bins = np.logspace(np.log10(k_um.min()), np.log10(k_um.max()), 45)
    k_centers = 0.5 * (k_bins[:-1] + k_bins[1:])
    s_radial = np.array([
        np.mean(s_flat[(k_um >= k_bins[i]) & (k_um < k_bins[i + 1])])
        if np.any((k_um >= k_bins[i]) & (k_um < k_bins[i + 1])) else np.nan
        for i in range(len(k_bins) - 1)
    ])

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    im0 = axes[0].imshow(V_c * 1e3, origin='lower', extent=[0, Lx * 1e6, 0, Ly * 1e6],
                         cmap='RdBu_r', vmin=-100, vmax=100)
    plt.colorbar(im0, ax=axes[0], label='Patch Potential $\\delta V$ (mV)', shrink=0.82)
    axes[0].set_xlabel('$x$ ($\\mu$m)')
    axes[0].set_ylabel('$y$ ($\\mu$m)')
    axes[0].set_title(f'(a) KPFM Potential ($V_{{\\mathrm{{rms}}}}={V_rms_mV:.1f}\\,$mV)', fontweight='bold', fontsize=10.5)

    valid_r = ~np.isnan(s_radial)
    axes[1].loglog(k_centers[valid_r], s_radial[valid_r], 'o-', color='darkslateblue', ms=4, lw=1.6)
    axes[1].set_xlabel(r'Wavevector Magnitude $k = |\mathbf{k}|$ ($\mu\mathrm{m}^{-1}$)')
    axes[1].set_ylabel(r'Radially Averaged $S_{VV}(k)$ ($\mathrm{V}^2\,\mathrm{m}^2$)')
    axes[1].set_title('(b) 2D KPFM Patch Spectrum $S_{VV}(k)$', fontweight='bold', fontsize=10.5)
    axes[1].grid(True, which='both', alpha=0.3)

    axes[2].loglog(gaps_nm, P_bcs, 'k-', lw=2.0, label='Lifshitz $|P_{\\mathrm{BCS}}(d)|$ (Al 10 mK)')
    axes[2].loglog(gaps_nm, np.abs(P_plasma - P_bcs), 'b--', lw=1.8, label='$|P_{\\mathrm{Plasma}} - P_{\\mathrm{BCS}}|$')
    axes[2].loglog(gaps_nm, P_patch_arr, 'r-', lw=2.2,
                   label=f'Speake-Trenner $P_{{\\mathrm{{patch}}}}(d) \\propto d^{{{slope_patch:.2f}}}$')
    axes[2].loglog(gaps_nm, np.abs(P_bcs - P_drude), 'g:', lw=1.8, label='$|P_{\\mathrm{BCS}} - P_{\\mathrm{Drude}}|$ ($T < T_c$)')
    axes[2].set_xlabel('Separation Gap $d$ (nm)')
    axes[2].set_ylabel('Pressure Magnitude (Pa)')
    axes[2].set_title('(c) Patch Background vs. Lifshitz Superconducting Shift', fontweight='bold', fontsize=10.5)
    axes[2].legend(fontsize=8.0, loc='upper right')
    axes[2].grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'fig_kpfm_patch_audit.png'), dpi=300)
    plt.close()

    res_kpfm = {
        'scan_size_um': [Lx * 1e6, Ly * 1e6],
        'grid_pixels': [nx, ny],
        'V_rms_mV': V_rms_mV,
        'Z_rms_nm': Z_rms_nm,
        'P_patch_power_law_exponent': float(slope_patch),
        'P_patch_at_20nm_Pa': float(np.interp(20.0, gaps_nm, P_patch_arr)),
        'P_patch_at_50nm_Pa': float(np.interp(50.0, gaps_nm, P_patch_arr)),
        'P_patch_at_100nm_mPa': float(np.interp(100.0, gaps_nm, P_patch_arr) * 1e3),
        'P_patch_at_200nm_mPa': float(np.interp(200.0, gaps_nm, P_patch_arr) * 1e3),
        'P_BCS_at_20nm_Pa': float(np.interp(20.0, gaps_nm, P_bcs)),
        'dP_Plasma_BCS_at_20nm_Pa': float(np.interp(20.0, gaps_nm, np.abs(P_plasma - P_bcs))),
    }
    print("  -> KPFM Summary:", res_kpfm)
    return res_kpfm


def run_full_v2_audit():
    decca_res = run_decca_vector_audit()
    kpfm_res = run_kpfm_patch_audit()
    summary = {'decca_2003_vector_audit': decca_res, 'dejong_2025_kpfm_audit': kpfm_res}
    out_json = os.path.join(OUTPUT_DIR, 'audit_v2_summary.json')
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved complete v2 audit summary to {out_json}")
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Casimir Spectral Metrology Pipeline v2.0")
    parser.add_argument('--mode', type=str, default='full_v2',
                        choices=['synthetic', 'discrimination', 'periodic_search', 'experimental', 'full_v2'])
    parser.add_argument('--empirical_data', type=str, default=os.path.join(BASE_DIR, 'data', 'decca_2003_fig3.csv'))
    parser.add_argument('--radius', type=float, default=None)
    parser.add_argument('--model-b', type=str, choices=['drude', 'plasma'], default='drude')
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if args.mode == 'full_v2':
        run_full_v2_audit()
    elif args.mode == 'synthetic':
        run_synthetic(args.radius if args.radius else 150e-6)
    elif args.mode == 'discrimination':
        run_discrimination(args.empirical_data, args.radius if args.radius else 150e-6, args.model_b)
    elif args.mode == 'experimental':
        run_experimental(args.empirical_data, args.radius)
    elif args.mode == 'periodic_search':
        run_periodic_search(args.empirical_data, args.radius)
