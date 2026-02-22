#!/usr/bin/env python3
"""
Build probability distributions from NLCC metrics (G_xy, G_yx, S_xy) across time windows.

Run from project root: python scripts/metrics_distribution.py [path/to/file.inp]

- Runs NLCC over all windows/dimensions and collects metrics.
- Plots histograms (empirical PDF) and optional KDE; saves to results/.
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from nlcc import (
    load_inp,
    load_signal,
    conditional_dispersion_curves,
    nlcc_metrics_from_curves,
    eps_grid_from_inp_matlab,
)


def collect_nlcc_metrics(inp_path):
    """
    Run NLCC over all time windows and dimensions; return a list of metric dicts.
    Each element has keys: I_time, D, G_xy, G_yx, S_xy (and optionally eps_xx, ...).
    """
    cfg = load_inp(inp_path)
    data_dir = Path(inp_path).parent
    A = load_signal(data_dir / cfg["input_A"])
    B = load_signal(data_dir / cfg["input_B"])
    x = A["data"]["x"]
    y = B["data"]["x"]
    dt = float(A["meta"]["Frequency"].split()[0])

    # Window indices (same logic as nlcc_run)
    inp_path_str = str(inp_path)
    if "VDP" in inp_path_str and ("test_VDP_ding" in inp_path_str or "test_VDP_vm" in inp_path_str):
        dt_ms = 1000.0 * dt
        I_begin = int(cfg["t_begin"] / dt_ms)
        I_end = int(cfg["t_end"] / dt_ms)
        I_step = int(cfg["t_step"] / dt_ms)
    else:
        I_begin = int(1000 * cfg["t_begin"]) - 1
        I_end = int(1000 * cfg["t_end"]) - 1
        I_step = int(1000 * cfg["t_step"])

    eps_vals, _ = eps_grid_from_inp_matlab(cfg)
    N = cfg["N"]
    tau = cfg["tau"]
    dM = cfg["dM"]

    records = []
    for I_time in range(I_begin, I_end + 1, I_step):
        xw = x[I_time : I_time + N]
        yw = y[I_time : I_time + N]
        if len(xw) < N or len(yw) < N:
            continue
        for D in range(cfg["dim_min"], cfg["dim_max"] + 1, cfg["dim_step"]):
            try:
                curves = conditional_dispersion_curves(
                    xw, yw, dim=D, tau=tau, eps_values=eps_vals,
                    normalization="range", dM=dM, max_pairs=None,
                )
                metrics = nlcc_metrics_from_curves(curves)
                records.append({
                    "I_time": I_time,
                    "D": D,
                    "G_xy": metrics["G_xy"],
                    "G_yx": metrics["G_yx"],
                    "S_xy": metrics["S_xy"],
                })
            except Exception:
                continue
    return records


def plot_distributions(records, output_path, use_kde=True):
    """
    Plot probability distributions for G_xy, G_yx, S_xy (histogram + optional KDE).
    """
    G_xy = np.array([r["G_xy"] for r in records if np.isfinite(r["G_xy"])])
    G_yx = np.array([r["G_yx"] for r in records if np.isfinite(r["G_yx"])])
    S_xy = np.array([r["S_xy"] for r in records if np.isfinite(r["S_xy"])])

    if len(G_xy) == 0 and len(G_yx) == 0 and len(S_xy) == 0:
        print("No finite metrics to plot.")
        return

    try:
        from scipy.stats import gaussian_kde
        has_kde = use_kde
    except ImportError:
        has_kde = False

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, data, label in [
        (axes[0], G_xy, r"$G_{xy}$"),
        (axes[1], G_yx, r"$G_{yx}$"),
        (axes[2], S_xy, r"$S_{xy}$"),
    ]:
        if len(data) == 0:
            ax.set_title(f"{label} (no data)")
            continue
        # Histogram (normalized as PDF)
        counts, bin_edges, _ = ax.hist(
            data, bins=min(30, max(10, len(data) // 5)),
            density=True, alpha=0.6, color="steelblue", edgecolor="white", label="histogram"
        )
        if has_kde and len(data) >= 2:
            kde = gaussian_kde(data)
            x_plot = np.linspace(data.min(), data.max(), 200)
            ax.plot(x_plot, kde(x_plot), "k-", lw=2, label="KDE")
        ax.set_xlabel(label)
        ax.set_ylabel("probability density")
        ax.set_title(f"{label}  (n={len(data)})")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    if len(sys.argv) > 1:
        inp_path = Path(sys.argv[1]).resolve()
    else:
        DATA = _PROJECT_ROOT / "data_raw"
        name = input("Preset (vdp_ding, vdp_vm, or path): ").strip()
        if name == "vdp_ding":
            inp_path = DATA / "VDP" / "test_VDP_ding.inp"
        elif name == "vdp_vm":
            inp_path = DATA / "VDP" / "test_VDP_vm.inp"
        else:
            inp_path = Path(name).resolve()
    if not inp_path.exists():
        print(f"Not found: {inp_path}")
        sys.exit(1)

    print("Collecting NLCC metrics over all windows/dimensions...")
    records = collect_nlcc_metrics(str(inp_path))
    print(f"Collected {len(records)} metric samples.")

    run_name = load_inp(str(inp_path))["run_name"]
    out_dir = _PROJECT_ROOT / "results"
    plot_distributions(records, out_dir / f"{run_name}_metrics_distribution.png", use_kde=True)


if __name__ == "__main__":
    main()
