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

Output goes to the `output/` directory at the project root.

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
