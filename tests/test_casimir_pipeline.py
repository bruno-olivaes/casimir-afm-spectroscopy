# -*- coding: utf-8 -*-
"""
Unit tests for Casimir Metrology & Generalized Lomb-Scargle Spectral Pipeline (v2.0).
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest
import numpy as np
import pandas as pd
from src.casimir_fft_pipeline import (
    generate_casimir_data,
    casimir_pfa_force,
    casimir_pfa_pressure,
    casimir_pfa_sphere_plate_force,
    rigorous_lombscargle_baluev,
    BASE_DIR,
)


def test_casimir_data_scaling():
    d1 = np.array([100e-9])
    d2 = np.array([200e-9])
    f1 = generate_casimir_data(d1, T=0)
    f2 = generate_casimir_data(d2, T=0)
    ratio = f1[0] / f2[0]
    assert abs(ratio - 16.0) < 0.05


def test_pfa_pressure_and_sphere_plate_force_scaling():
    d1 = np.array([100e-9])
    d2 = np.array([200e-9])
    R = 294.3e-6
    # Pressure (MTO force gradient / 2piR) scales as d^-4 (2^4 = 16)
    p1 = casimir_pfa_pressure(d1)
    p2 = casimir_pfa_pressure(d2)
    assert abs((p1[0] / p2[0]) - 16.0) < 1e-6

    # True sphere-plate force scales as d^-3 (2^3 = 8)
    fs1 = casimir_pfa_sphere_plate_force(d1, R)
    fs2 = casimir_pfa_sphere_plate_force(d2, R)
    assert abs((fs1[0] / fs2[0]) - 8.0) < 1e-6


def test_baluev_fap_detects_sine_and_rejects_white_noise():
    rng = np.random.default_rng(42)
    z = np.sort(rng.uniform(200.0, 1200.0, 400))
    sigma = 0.30
    # 1. Pure sine + noise -> Baluev FAP must be << 1e-10 (fixes v1 FAP=1.00 underflow bug)
    y_sine = 1.2 * np.sin(2.0 * np.pi * z / 80.0) + rng.normal(0.0, sigma, size=len(z))
    res_sine = rigorous_lombscargle_baluev(z, y_sine, dy=sigma)
    assert res_sine['fap_baluev'] < 1e-10
    assert abs(res_sine['peak_wavelength'] - 80.0) < 2.0

    # 2. Pure white noise -> Baluev FAP > 0.01
    y_noise = rng.normal(0.0, sigma, size=len(z))
    res_noise = rigorous_lombscargle_baluev(z, y_noise, dy=sigma)
    assert res_noise['fap_baluev'] > 0.01


def test_decca_vector_extracted_files_valid():
    vec_dir = os.path.join(BASE_DIR, 'data', 'decca2003_vector')
    p4b = os.path.join(vec_dir, 'fig4b_dpressure.csv')
    p3b = os.path.join(vec_dir, 'fig3b_dforce.csv')
    assert os.path.exists(p4b) and os.path.exists(p3b)
    df4 = pd.read_csv(p4b)
    df3 = pd.read_csv(p3b)
    assert len(df4) == 497
    assert len(df3) == 288
    # Strictly monotonic z_nm
    assert np.all(np.diff(df4['z_nm'].values) > 0)
    assert np.all(np.diff(df3['z_nm'].values) > 0)
