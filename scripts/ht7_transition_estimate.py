#!/usr/bin/env python3
"""
Estimate where the signal character changes in the HT-7 time series (first 50 ms).

The thesis describes: "quiescent phase → intermittent burst → broadband turbulence"
in the first 50 ms. This script has no L–H transition label in the repo; it only
estimates transitions between these three phases from the signal itself (e.g. sliding RMS).

Run from project root: python scripts/ht7_transition_estimate.py
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from nlcc import load_signal

# HT7 data path (project layout)
DATA = _PROJECT_ROOT / "data_raw" / "HT7:EAST"
T_FIRST_MS = 50.0   # thesis: "first 50 ms"
WINDOW_MS = 2.0     # sliding window for local activity (ms)
THRESHOLD_QUIET = 0.5   # fraction of global max RMS below which we call "quiescent"
THRESHOLD_BROADBAND = 0.85  # fraction above which we call "broadband" (high activity)


def main():
    path_a = DATA / "A.dat"
    path_b = DATA / "B.dat"
    if not path_a.exists() or not path_b.exists():
        print(f"HT7 data not found under {DATA}")
        sys.exit(1)

    A = load_signal(path_a)
    B = load_signal(path_b)
    t = A["data"]["t"]
    x = A["data"]["x"]
    y = B["data"]["x"]
    dt = float(A["meta"]["Frequency"].split()[0])

    # First 50 ms in samples
    n_50ms = int(round(T_FIRST_MS * 1e-3 / dt))
    if n_50ms > len(x):
        n_50ms = len(x)
    t_ms = (t[:n_50ms] - t[0]) * 1e3   # time in ms
    x_50 = x[:n_50ms]
    y_50 = y[:n_50ms]

    # Use mean of both probes as "activity" (or just A)
    sig = 0.5 * (x_50 + y_50)
    window_samples = max(1, int(round(WINDOW_MS * 1e-3 / dt)))
    # Sliding RMS
    rms = np.full_like(sig, np.nan, dtype=float)
    half = window_samples // 2
    for i in range(half, len(sig) - half):
        rms[i] = np.sqrt(np.mean(sig[i - half : i + half + 1] ** 2))
    rms[:half] = rms[half]
    rms[-half:] = rms[-half - 1]

    rms_max = np.nanmax(rms)
    rms_min = np.nanmin(rms)
    if rms_max <= rms_min:
        print("No variation in sliding RMS; cannot estimate phases.")
        return

    # Simple thresholds to mark phases (quiescent = low RMS, broadband = high RMS)
    quiet_mask = rms < rms_min + THRESHOLD_QUIET * (rms_max - rms_min)
    high_mask = rms >= rms_min + THRESHOLD_BROADBAND * (rms_max - rms_min)

    # First exit from quiescent (start of "intermittent burst")
    quiescent_end_idx = None
    for i in range(1, len(quiet_mask)):
        if quiet_mask[i - 1] and not quiet_mask[i]:
            quiescent_end_idx = i
            break
    # First entry into sustained high (start of "broadband")
    broadband_start_idx = None
    for i in range(1, len(high_mask)):
        if not high_mask[i - 1] and high_mask[i]:
            broadband_start_idx = i
            break

    # Report in ms
    def idx_to_ms(i):
        return float(t_ms[i]) if i is not None and i < len(t_ms) else None

    t_quiet_end_ms = idx_to_ms(quiescent_end_idx)
    t_broadband_start_ms = idx_to_ms(broadband_start_idx)

    print("HT-7 first 50 ms — estimated phase boundaries (from sliding RMS):")
    print("  (No L–H transition time is provided in the repo; this is the")
    print("   quiescent → intermittent burst → broadband evolution from the thesis.)")
    print()
    if t_quiet_end_ms is not None:
        print(f"  End of quiescent / start of intermittent burst: ~{t_quiet_end_ms:.2f} ms")
    else:
        print("  End of quiescent: not clearly detected")
    if t_broadband_start_ms is not None:
        print(f"  Start of sustained high activity (broadband):   ~{t_broadband_start_ms:.2f} ms")
    else:
        print("  Start of broadband: not clearly detected")
    print()
    print("  NLCC in the thesis uses a window at 100–105 ms (steady turbulence).")

    # Plot: time series + sliding RMS with phase bands
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 7))
    axes[0].plot(t_ms, x_50, linewidth=0.6, alpha=0.9, label="A (r1)")
    axes[0].set_ylabel("A")
    axes[0].legend(loc="upper right")
    axes[0].set_title("HT-7 first 50 ms — signal and estimated phases")

    axes[1].plot(t_ms, y_50, linewidth=0.6, alpha=0.9, label="B (r2)")
    axes[1].set_ylabel("B")
    axes[1].legend(loc="upper right")

    axes[2].plot(t_ms, rms, "k-", linewidth=1, label="sliding RMS")
    axes[2].set_ylabel("Sliding RMS")
    axes[2].set_xlabel("t (ms)")
    axes[2].legend(loc="upper right")
    if t_quiet_end_ms is not None:
        axes[2].axvline(t_quiet_end_ms, color="green", linestyle="--", alpha=0.8, label="quiescent end")
    if t_broadband_start_ms is not None:
        axes[2].axvline(t_broadband_start_ms, color="blue", linestyle="--", alpha=0.8, label="broadband start")

    out_dir = _PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "ht7_first50ms_phase_estimate.png"
    plt.tight_layout()
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    print(f"\nFigure saved: {out_file}")


if __name__ == "__main__":
    main()
