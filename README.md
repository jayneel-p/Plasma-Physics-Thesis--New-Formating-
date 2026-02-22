# NLCC — Nonlinear Cross Correlation (Ding et al. 1997)

Clone the repo, then run or import as a library.

## Run the scripts (from repo root)

```bash
# From project root (so nlcc is found)
python scripts/nlcc_run.py
# or
python scripts/time_series.py
```

You can pass an input file path instead of using the prompt:

```bash
python scripts/nlcc_run.py data_raw/VDP/test_VDP_ding.inp
```

Output goes to the `results/` directory at the project root.

### Probability distributions of NLCC metrics

To build probability distributions of G_xy, G_yx, S_xy across time windows (histogram + KDE):

```bash
python scripts/metrics_distribution.py
# or with an input file:
python scripts/metrics_distribution.py data_raw/VDP/test_VDP_ding.inp
```

This collects metrics from every window and embedding dimension, then saves `results/<run_name>_metrics_distribution.png`.

### L–H transition scan (long time series)

To check whether a long A/B time series shows an L–H-like transition:

```bash
python scripts/lh_transition_scan.py
# or: python scripts/lh_transition_scan.py path/to/test_AB.inp
```

The script runs NLCC in sliding windows over the **full** trace and plots **S_xy**, **G_xy**/ **G_yx**, and **RMS** vs time. In some devices (e.g. STOR-M), the L–H transition coincides with a **persistent sign change in S_xy** and/or a **sustained drop in fluctuation level (RMS)**. Use these as candidate indicators and compare with Hα/Dα or other diagnostics if available. Output: `results/<run_name>_lh_scan.png`.

## Use nlcc as a library

From your own code (with this repo on `PYTHONPATH` or installed), import the functions you need:

```python
from nlcc import (
    load_inp,
    load_signal,
    load_pair,
    conditional_dispersion_curves,
    nlcc_metrics_from_curves,
    eps_grid_from_inp_matlab,
)

# Load config and signals (paths can be anywhere)
cfg = load_inp("path/to/file.inp")
A = load_signal("path/to/A.dat")
B = load_signal("path/to/B.dat")
x, y = A["data"]["x"], B["data"]["x"]

# Build epsilon grid and run analysis on a window
eps_vals, _ = eps_grid_from_inp_matlab(cfg)
curves = conditional_dispersion_curves(x, y, dim=cfg["dim_min"], tau=cfg["tau"], eps_values=eps_vals, dM=cfg["dM"])
metrics = nlcc_metrics_from_curves(curves)
```

All paths in your script can be absolute or relative to your working directory; the package does not rely on hardcoded paths.
