#!/usr/bin/env python3
"""
Scan a long A/B time series with sliding NLCC windows to look for L–H transition signatures.

L–H transition (from literature, e.g. STOR-M, Ding/van Milligen):
- Directionality reversal: S_xy can flip sign (e.g. L-mode: outward propagation, H-mode: inward).
- Fluctuation level often drops in H-mode (edge transport barrier).
So we compute over time: S_xy, G_xy, G_yx (NLCC) and RMS (fluctuation level).
Look for: a persistent sign change in S_xy and/or a sustained drop in RMS as candidate L–H.

Usage (from project root):
  python scripts/lh_transition_scan.py [path/to/file.inp]
  python scripts/lh_transition_scan.py   # uses HT7 test_AB.inp by default

Output: results/<run_name>_lh_scan.png and printed summary.
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


def sliding_nlcc_scan(inp_path, step_frac=0.5, dim=None):
    """
    Run NLCC in sliding windows over the full A/B time series.

    step_frac: step = window_length * step_frac (0.5 = 50% overlap).
    dim: embedding dimension to use (None = use cfg["dim_min"]).

    Returns:
        t_ms: center time of each window (ms)
        S_xy, G_xy, G_yx: 1D arrays
        rms_a, rms_b: fluctuation level in each window
    """
    cfg = load_inp(inp_path)
    data_dir = Path(inp_path).parent
    A = load_signal(data_dir / cfg["input_A"])
    B = load_signal(data_dir / cfg["input_B"])
    x = A["data"]["x"]
    y = B["data"]["x"]
    t = A["data"]["t"]
    dt = float(A["meta"]["Frequency"].split()[0])

    N = cfg["N"]
    tau = cfg["tau"]
    dM = cfg["dM"]
    D = dim if dim is not None else cfg["dim_min"]
    eps_vals, _ = eps_grid_from_inp_matlab(cfg)

    step = max(1, int(N * step_frac))
    starts = list(range(0, len(x) - N + 1, step))
    n_windows = len(starts)
    if n_windows == 0:
        raise ValueError(f"Time series too short: {len(x)} samples, need at least N={N}")

    t_center_ms = []
    S_xy_list, G_xy_list, G_yx_list = [], [], []
    rms_a_list, rms_b_list = [], []

    for i, start in enumerate(starts):
        xw = x[start : start + N]
        yw = y[start : start + N]
        t_center_ms.append((t[start] + t[start + N - 1]) * 0.5 * 1e3)  # ms
        rms_a_list.append(np.sqrt(np.mean(xw**2)))
        rms_b_list.append(np.sqrt(np.mean(yw**2)))
        try:
            curves = conditional_dispersion_curves(
                xw, yw, dim=D, tau=tau, eps_values=eps_vals,
                normalization="range", dM=dM, max_pairs=None,
            )
            m = nlcc_metrics_from_curves(curves)
            S_xy_list.append(m["S_xy"])
            G_xy_list.append(m["G_xy"])
            G_yx_list.append(m["G_yx"])
        except Exception:
            S_xy_list.append(np.nan)
            G_xy_list.append(np.nan)
            G_yx_list.append(np.nan)

        if (i + 1) % 20 == 0 or i == 0:
            print(f"  Window {i+1}/{n_windows} (t_center = {t_center_ms[-1]:.2f} ms)")

    return (
        np.array(t_center_ms),
        np.array(S_xy_list),
        np.array(G_xy_list),
        np.array(G_yx_list),
        np.array(rms_a_list),
        np.array(rms_b_list),
        cfg,
    )


def plot_scan(t_ms, S_xy, G_xy, G_yx, rms_a, rms_b, run_name, output_path):
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(11, 8))

    # S_xy: directionality (sign change = candidate L–H in some devices)
    ax = axes[0]
    ax.plot(t_ms, S_xy, "k-", linewidth=0.8, alpha=0.9)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.7)
    ax.set_ylabel(r"$S_{xy}$")
    ax.set_title("Sliding-window NLCC: how to spot a possible L–H transition")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(t_ms[0], t_ms[-1])

    # G_xy, G_yx
    ax = axes[1]
    ax.plot(t_ms, G_xy, label=r"$G_{xy}$", linewidth=0.8, alpha=0.9)
    ax.plot(t_ms, G_yx, label=r"$G_{yx}$", linewidth=0.8, alpha=0.9)
    ax.set_ylabel("G")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    # Fluctuation level (sustained drop can indicate H-mode)
    ax = axes[2]
    rms = 0.5 * (rms_a + rms_b)
    ax.plot(t_ms, rms, "darkgreen", linewidth=0.8, alpha=0.9, label="mean RMS (A,B)")
    ax.set_ylabel("RMS")
    ax.set_xlabel("Time (ms)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.text(0.02, 0.02,
             "Interpretation: Look for (1) a persistent sign change in S_xy (e.g. + → −)\n"
             "and/or (2) a sustained drop in RMS. Compare with Hα/Dα or other diagnostics if available.",
             fontsize=8, verticalalignment="bottom", wrap=True,
             bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.4))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main():
    if len(sys.argv) > 1:
        inp_path = Path(sys.argv[1]).resolve()
    else:
        inp_path = _PROJECT_ROOT / "data_raw" / "HT7:EAST" / "test_AB.inp"
    if not inp_path.exists():
        print(f"Not found: {inp_path}")
        sys.exit(1)

    print("Sliding-window NLCC over full time series (this may take a few minutes)...")
    t_ms, S_xy, G_xy, G_yx, rms_a, rms_b, cfg = sliding_nlcc_scan(str(inp_path), step_frac=0.5)

    valid = np.isfinite(S_xy)
    if np.any(valid):
        s_min, s_max = np.nanmin(S_xy), np.nanmax(S_xy)
        sign_changes = np.where(np.diff(np.sign(S_xy)) != 0)[0]
        print("\n--- Summary ---")
        print(f"  S_xy range: [{s_min:.4f}, {s_max:.4f}]")
        print(f"  Number of S_xy sign changes: {len(sign_changes)}")
        if len(sign_changes) > 0:
            print("  Approx. times (ms) of sign change:", t_ms[sign_changes].tolist()[:10])
        print("\n  To judge L–H: look for a *persistent* S_xy reversal (e.g. stays negative after being positive)")
        print("  and/or a sustained drop in RMS. Cross-check with Hα/Dα or stored energy if you have them.")
    else:
        print("  No valid S_xy (check window length and data).")

    run_name = cfg["run_name"]
    out_dir = _PROJECT_ROOT / "results"
    plot_scan(t_ms, S_xy, G_xy, G_yx, rms_a, rms_b, run_name, out_dir / f"{run_name}_lh_scan.png")


if __name__ == "__main__":
    main()
