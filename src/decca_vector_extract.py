# -*- coding: utf-8 -*-
"""
Lossless vector extraction of the published data points of
Decca, Lopez, Fischbach & Krause, "Measurement of the Casimir force between
dissimilar metals", Phys. Rev. Lett. 91, 050402 (2003), arXiv:quant-ph/0306136v2.

Why this module exists
----------------------
Version 1 of this project used 53 points hand-digitised from a raster image of
Fig. 4(a) (|P_c| vs z). That digitisation had a vertical quantisation step of
~7.1 mPa (one pixel), i.e. about 20x larger than the real experimental scatter
visible in Fig. 4(b) (~0.31-0.35 mPa). Any "residual spectrum" computed from it
was dominated by digitisation error, not by the experiment.

The arXiv PDF stores Figs. 3 and 4 as *vector graphics*: every data marker is a
filled square path whose coordinates are stored with ~1e-4 pt resolution. We
therefore read the marker centres directly from the PDF content stream with
PyMuPDF and map them to physical units with a linear fit to the axis tick
marks (which are also vector paths).

Note on shared panel borders:
In Fig. 4, Panel (a) has its |P_c| = 0 mPa baseline at y_pt = 423.62, while
Panel (b) has its highest tick (+1 mPa) at y_pt = 429.02 and 0 mPa at 439.94.
Setting the Panel (b) upper cutoff to y_pt = 426.0 cleanly excludes the 35
asymptotic Panel (a) markers sitting on y_pt = 423.5. Similarly, in Fig. 3,
setting the Panel (b) upper cutoff to y_pt = 338.0 excludes the Panel (a)
baseline at y_pt = 335.12.

Output: data/decca2003_vector/*.csv
"""
import os
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PDF_PATH = os.path.join(BASE_DIR, 'data', 'decca_2003.pdf')
OUT_DIR = os.path.join(BASE_DIR, 'data', 'decca2003_vector')

X_TICKS_FIG4 = ([132.24, 195.24, 258.12, 321.12, 384.12, 447.06, 510.06],
                [0, 200, 400, 600, 800, 1000, 1200])
X_TICKS_FIG3 = ([125.58, 191.34, 256.98, 322.68, 388.38, 454.02, 519.78],
                [0, 200, 400, 600, 800, 1000, 1200])

PANELS = {
    'fig4a_pressure': dict(page=16, frame=(132.24, 276.68, 510.06, 426.00),
                           xt=X_TICKS_FIG4,
                           yt=([299.90, 330.80, 361.76, 392.72, 423.62], [800, 600, 400, 200, 0]),
                           xname='z_nm', yname='abs_Pc_mPa', marker=3.5),
    'fig4b_dpressure': dict(page=16, frame=(132.24, 426.00, 510.06, 549.38),
                            xt=X_TICKS_FIG4,
                            yt=([439.94, 461.78, 483.74, 505.58, 527.54, 549.38], [0, -2, -4, -6, -8, -10]),
                            xname='z_nm', yname='dPc_mPa', marker=3.5),
    'fig3a_force': dict(page=15, frame=(125.58, 181.82, 519.78, 338.00),
                        xt=X_TICKS_FIG3,
                        yt=([205.16, 231.26, 257.18, 283.16, 309.20, 335.12], [100, 80, 60, 40, 20, 0]),
                        xname='z_nm', yname='abs_Fc_pN', marker=4.0),
    'fig3b_dforce': dict(page=15, frame=(125.58, 338.00, 519.78, 466.40),
                         xt=X_TICKS_FIG3,
                         yt=([347.00, 376.82, 406.70, 436.52, 466.40], [0.5, 0.0, -0.5, -1.0, -1.5]),
                         xname='z_nm', yname='dFc_pN', marker=4.0),
}


def _linear_map(ticks):
    pix, val = np.asarray(ticks[0], float), np.asarray(ticks[1], float)
    a, b = np.polyfit(pix, val, 1)
    resid = val - (a * pix + b)
    return a, b, float(np.max(np.abs(resid)))


def extract_markers(page_index, marker_size, tol=0.25):
    import fitz  # PyMuPDF
    doc = fitz.open(PDF_PATH)
    page = doc[page_index]
    out = []
    for d in page.get_drawings():
        r = d['rect']
        if (d['type'] in ('fs', 'f') and abs(r.width - marker_size) < tol
                and abs(r.height - marker_size) < tol and len(d['items']) == 1):
            out.append(((r.x0 + r.x1) / 2.0, (r.y0 + r.y1) / 2.0, d.get('seqno', -1)))
    return np.array(out)


def deduplicate_by_x(df, xcol, ycol, tol_nm=0.02):
    """Merge markers that share an identical x coordinate within tol_nm."""
    df = df.sort_values(xcol).reset_index(drop=True)
    groups = (df[xcol].diff().fillna(1.0) > tol_nm).cumsum()
    agg = df.groupby(groups, as_index=False).agg({
        xcol: 'mean',
        ycol: 'mean',
        'pdf_x_pt': 'mean',
        'pdf_y_pt': 'mean'
    })
    return agg


def run_extraction(verbose=True):
    os.makedirs(OUT_DIR, exist_ok=True)
    cache = {}
    summary = []
    for name, P in PANELS.items():
        key = (P['page'], P['marker'])
        if key not in cache:
            cache[key] = extract_markers(*key)
        m = cache[key]
        x0, y0, x1, y1 = P['frame']
        inside = (m[:, 0] >= x0 - 0.01) & (m[:, 0] <= x1 + 0.01) & (m[:, 1] > y0) & (m[:, 1] <= y1 + 0.01)
        pts = m[inside]
        ax, bx, ex = _linear_map(P['xt'])
        ay, by, ey = _linear_map(P['yt'])
        xs = ax * pts[:, 0] + bx
        ys = ay * pts[:, 1] + by
        order = np.argsort(xs)
        df_raw = pd.DataFrame({P['xname']: xs[order], P['yname']: ys[order],
                               'pdf_x_pt': pts[order, 0], 'pdf_y_pt': pts[order, 1]})
        df = deduplicate_by_x(df_raw, P['xname'], P['yname'])
        df.to_csv(os.path.join(OUT_DIR, f'{name}.csv'), index=False)
        summary.append(dict(panel=name, n_raw=len(df_raw), n_unique_z=len(df),
                            z_min_nm=float(df[P['xname']].min()), z_max_nm=float(df[P['xname']].max()),
                            y_min=float(df[P['yname']].min()), y_max=float(df[P['yname']].max()),
                            x_units_per_pt=ax, y_units_per_pt=ay,
                            x_tick_lin_err=ex, y_tick_lin_err=ey))
    s = pd.DataFrame(summary)
    s.to_csv(os.path.join(OUT_DIR, 'extraction_summary.csv'), index=False)
    if verbose:
        with pd.option_context('display.width', 200, 'display.max_columns', 20):
            print(s)
    return s


if __name__ == '__main__':
    run_extraction()
