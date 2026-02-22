#!/usr/bin/env python3
"""NLCC runner with overview plots. Run from project root: python scripts/time_series.py"""
import sys
from pathlib import Path

# Allow importing nlcc when run from repo (e.g. clone from GitHub, run from any dir)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from nlcc import load_inp, load_signal, conditional_dispersion_curves, nlcc_metrics_from_curves, eps_grid_from_inp_matlab

def plot_window_timeseries(t, x, y, dt, cfg, output_dir, run_name, I_begin):
    """
    Plot A(t), B(t) over the first NLCC analysis window.

    Uses:
        start index = I_begin
        length      = N = cfg["N"]
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    N = cfg["N"]
    i0 = I_begin
    i1 = i0 + N
    if i1 > len(x):
        i1 = len(x)

    tw = t[i0:i1]
    xw = x[i0:i1]
    yw = y[i0:i1]

    # local time in ms relative to start of window
    t_local = (tw - tw[0]) * 1e3

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 5))

    ax[0].plot(t_local, xw, linewidth=0.8)
    ax[0].set_ylabel("A(t)")
    ax[0].set_title(f"{run_name}: time series in first NLCC window")

    ax[1].plot(t_local, yw, linewidth=0.8)
    ax[1].set_ylabel("B(t)")
    ax[1].set_xlabel("t [ms] (relative to window start)")

    fig.tight_layout()
    out_file = output_dir / f"{run_name}_timeseries_window.png"
    fig.savefig(out_file, dpi=200)
    print(f"✓ Window time-series figure saved to {out_file}")
    plt.close(fig)

def plot_overview_time_and_embedding(t, x, y, dt, cfg, output_dir, run_name):
    """
    Make quick overview plots for the current A/B pair:

    1) Short time series of A(t), B(t) over ~40 ms (or less if record is short).
    2) 2D delay embedding of A(t) with embedding dimension D = 2,
       using tau (in samples) from cfg["tau"].

    Saves:
        <output_dir>/<run_name>_timeseries.png
        <output_dir>/<run_name>_embed_D2.png
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # ---- Short time series window (~40 ms) ----
    T_plot = 40e-3  # 40 ms
    n_plot = int(T_plot / dt)
    if n_plot < 10:
        # If dt is large or record is short, just use up to 1000 points
        n_plot = min(len(x), 1000)
    else:
        n_plot = min(n_plot, len(x))

    t_ts = t[:n_plot] - t[0]
    x_ts = x[:n_plot]
    y_ts = y[:n_plot]

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    ax[0].plot(t_ts * 1e3, x_ts, linewidth=0.8)
    ax[0].set_ylabel("A(t)")
    ax[0].set_title(f"{run_name}: time series (first {t_ts[-1]*1e3:.1f} ms)")

    ax[1].plot(t_ts * 1e3, y_ts, linewidth=0.8)
    ax[1].set_ylabel("B(t)")
    ax[1].set_xlabel("t [ms]")

    fig.tight_layout()
    ts_file = output_dir / f"{run_name}_timeseries.png"
    fig.savefig(ts_file, dpi=200)
    print(f"✓ Time-series figure saved to {ts_file}")
    plt.close(fig)

    # ---- Simple phase-space analogue: Takens embedding of A with D=2 ----
    tau_samples = int(cfg["tau"])
    if tau_samples <= 0 or 2 * tau_samples >= len(x):
        print("! Skipping embedding plot: tau too large or invalid.")
        return

    start = 0
    end = len(x) - tau_samples
    X0 = x[start:end]
    X1 = x[start + tau_samples:start + tau_samples + (end - start)]

    # Optional decimation so plots don't become solid blobs
    n_points = len(X0)
    stride = max(1, n_points // 5000)
    X0 = X0[::stride]
    X1 = X1[::stride]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(X0, X1, linewidth=0.4, alpha=0.7)
    ax.set_xlabel("A(t)")
    ax.set_ylabel(f"A(t + {tau_samples}Δt)")
    ax.set_title(f"{run_name}: delay embedding (D=2, tau={tau_samples})")

    fig.tight_layout()
    emb_file = output_dir / f"{run_name}_embed_D2.png"
    fig.savefig(emb_file, dpi=200)
    print(f"✓ Embedding figure saved to {emb_file}")
    plt.close(fig)


def run_nlcc_full(inp_path, output_dir="output", plot=True):
    """
    Run full NLCC analysis matching MATLAB behavior as closely as possible.
    Also produces overview plots (short time series and D=2 embedding)
    for the input A/B signals.
    """
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Load parameters
    cfg = load_inp(inp_path)
    print(f"Configuration loaded from {inp_path}")
    print(f"  N={cfg['N']}, tau={cfg['tau']}, dM={cfg['dM']}")
    print(f"  Epsilon: log2(eps) from {cfg['log2_eps_min']} to {cfg['log2_eps_max']}")
    print(f"  Dimensions: {cfg['dim_min']} to {cfg['dim_max']} step {cfg['dim_step']}")
    print(f"  Time windows: {cfg['t_begin']}ms to {cfg['t_end']}ms step {cfg['t_step']}ms")

    # Load signals
    data_dir = Path(inp_path).parent
    A = load_signal(data_dir / cfg['input_A'])  # {"meta": meta, "data": {"t": t, "x": x}}
    B = load_signal(data_dir / cfg['input_B'])

    x = A["data"]["x"]
    y = B["data"]["x"]
    t = A["data"]["t"]

    print(f"  Signal A ({cfg['input_A']}): {len(x)} samples")
    print(f"  Signal B ({cfg['input_B']}): {len(y)} samples")

    # Extract dt from metadata (Frequency is treated as dt)
    freq_str = A["meta"]["Frequency"].split()[0]
    dt = float(freq_str)
    print(f"  dt = {dt:.6e} s = {dt*1e6:.3f} µs")

    # Overview plots: short time series + D=2 embedding
    plot_overview_time_and_embedding(t, x, y, dt, cfg, output_dir, cfg["run_name"])

    # Time-window indexing
    # VDP data: t_begin/t_end in ms, convert to sample index via dt
    data_dir_str = str(Path(inp_path).resolve().parent)
    if "VDP" in data_dir_str and "test_VDP" in str(inp_path):
        dt_ms = 1000.0 * dt
        I_begin = int(cfg["t_begin"] / dt_ms)
        I_end = int(cfg["t_end"] / dt_ms)
        I_step = int(cfg["t_step"] / dt_ms)
    else:
        # Tokamak case: t_begin, t_end, t_step are in ms
        # MATLAB: I_begin = 1000 * t_begin  (1-based)
        # Python: subtract 1 to convert to 0-based
        I_begin = int(1000 * cfg["t_begin"]) - 1
        I_end   = int(1000 * cfg["t_end"]) - 1
        I_step  = int(1000 * cfg["t_step"])

    print(f"  Time window sample indices: {I_begin} to {I_end} step {I_step}")

    # Build epsilon grid
    eps_vals, log2_eps = eps_grid_from_inp_matlab(cfg)
    print(f"  Epsilon grid: {len(eps_vals)} values")
    print(f"    log2(eps) from {log2_eps[0]:.3f} to {log2_eps[-1]:.3f}")
    print(f"    eps from {eps_vals[0]:.6e} to {eps_vals[-1]:.6e}")

    # Open .cor output file
    run_name = cfg["run_name"]
    out_file = output_dir / f"{run_name}.cor"

    with open(out_file, 'w') as fout:
        fout.write(f"NLCC Analysis Results\n")
        fout.write(f"Input A: {cfg['input_A']}\n")
        fout.write(f"Input B: {cfg['input_B']}\n")
        fout.write(f"N={cfg['N']}, tau={cfg['tau']}, dM={cfg['dM']}\n")
        fout.write("="*70 + "\n\n")

        # Loop over time windows
        for I_time in range(I_begin, I_end + 1, I_step):
            t_ms = I_time * dt * 1000.0

            print(f"\n  Processing time window at t={t_ms:.2f} ms (sample {I_time})")
            fout.write(f"\nTime window: t={t_ms:.2f} ms (sample {I_time})\n")
            fout.write("-"*70 + "\n")

            N = cfg["N"]
            xw = x[I_time: I_time + N]
            yw = y[I_time: I_time + N]

            if len(xw) < N or len(yw) < N:
                print(f"    Warning: Not enough data (got {len(xw)}/{len(yw)}, need {N})")
                fout.write("    ERROR: Insufficient data\n")
                continue

            print(f"    Extracted window: {len(xw)} samples")
            print(f"    A range: [{np.min(xw):.6e}, {np.max(xw):.6e}]")
            print(f"    B range: [{np.min(yw):.6e}, {np.max(yw):.6e}]")

            # Loop over embedding dimensions
            for D in range(cfg["dim_min"], cfg["dim_max"] + 1, cfg["dim_step"]):
                print(f"    Dimension D={D}... ", end='', flush=True)
                fout.write(f"\nDimension = {D}\n")
                fout.write(f"File X: {cfg['input_A']}            File Y:  {cfg['input_B']}\n")

                tau = cfg["tau"]
                dM = cfg["dM"]

                try:
                    # Compute dispersion curves
                    curves = conditional_dispersion_curves(
                        xw, yw,
                        dim=D,
                        tau=tau,
                        eps_values=eps_vals,
                        normalization="range",
                        dM=dM,
                        max_pairs=None,
                    )

                    # Save dispersion data to .dat
                    dat_file = output_dir / f"{run_name}{D}.dat"
                    with open(dat_file, 'w') as fdat:
                        fdat.write("log(E)     XX        XY          YY             YX\n")
                        for i in range(len(curves["log2_eps"])):
                            fdat.write(f"{curves['log2_eps'][i]:9.6f}  ")
                            fdat.write(f"{curves['sigma_xx'][i]:9.6f}  ")
                            fdat.write(f"{curves['sigma_xy'][i]:9.6f}  ")
                            fdat.write(f"{curves['sigma_yy'][i]:9.6f}  ")
                            fdat.write(f"{curves['sigma_yx'][i]:9.6f}\n")

                    # Compute metrics
                    metrics = nlcc_metrics_from_curves(curves)

                    # Write metrics (MATLAB-style) to .cor
                    fout.write("epsilon at which log2(epsilon)=0.9-------------\n")
                    fout.write("XX             XY             YY             YX\n")
                    fout.write(f"{metrics['eps_xx']:.6e}  {metrics['eps_xy']:.6e}  ")
                    fout.write(f"{metrics['eps_yy']:.6e}  {metrics['eps_yx']:.6e}\n")
                    fout.write(f"EpsXX= {metrics['eps_xx']:.6e}             EpsXY=  {metrics['eps_xy']:.6e}\n")
                    fout.write(f"EpsYY= {metrics['eps_yy']:.6e}             EpsYX=  {metrics['eps_yx']:.6e}\n")
                    fout.write(f"Gxy=EXY/sqrt(EXX EYY)=    {metrics['G_xy']:.6e}\n")
                    fout.write(f"Gyx=EYX/sqrt(EXX EYY)=    {metrics['G_yx']:.6e}\n")
                    fout.write(f"Sxy=(EYX-EXY)/(EXY+EYX)=  {metrics['S_xy']:.6e}\n")

                    print(f"G_xy={metrics['G_xy']:.4f}, G_yx={metrics['G_yx']:.4f}, S_xy={metrics['S_xy']:.4f}")

                except Exception as e:
                    print(f"ERROR: {e}")
                    fout.write(f"    ERROR: {e}\n")
                    import traceback
                    traceback.print_exc()
                    continue

    print(f"\n✓ Results written to {out_file}")

    if plot:
        plot_results(cfg, output_dir, run_name)

    print(f"  Time window sample indices: {I_begin} to {I_end} step {I_step}")

    # New: plot A,B over the first NLCC window (good for the thesis)
    plot_window_timeseries(t, x, y, dt, cfg, output_dir, run_name=cfg["run_name"], I_begin=I_begin)



def plot_results(cfg, output_dir, run_name):
    """
    Generate summary plots of sigma_xy and sigma_yx vs log2(eps)
    for each embedding dimension, similar to the MATLAB visualization.
    """
    output_dir = Path(output_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    dimensions = range(cfg["dim_min"], cfg["dim_max"] + 1, cfg["dim_step"])

    for D in dimensions:
        dat_file = output_dir / f"{run_name}{D}.dat"

        if not dat_file.exists():
            continue

        data = np.loadtxt(dat_file, skiprows=1)
        log2_eps = data[:, 0]
        sigma_xy = data[:, 2]
        sigma_yx = data[:, 4]

        ax1.plot(log2_eps, sigma_xy, '-o', markersize=3, label=f'M = {D}')
        ax2.plot(log2_eps, sigma_yx, '-o', markersize=3, label=f'M = {D}')

    ax1.set_xlabel('log2(eps)')
    ax1.set_ylabel('sigma XY(eps)')
    ax1.grid(True)
    ax1.legend()

    ax2.set_xlabel('log2(eps)')
    ax2.set_ylabel('sigma YX(eps)')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()

    fig_file = output_dir / f"{run_name}_dispersion.png"
    plt.savefig(fig_file, dpi=150)
    print(f"✓ Dispersion figure saved to {fig_file}")

    plt.show()


if __name__ == "__main__":
    # Paths relative to project root so it works after clone (run from repo root)
    DATA = _PROJECT_ROOT / "data_raw"

    # CLI overrides path; otherwise prompt for preset
    if len(sys.argv) > 1:
        inp_path = Path(sys.argv[1]).resolve()
    else:
        inp_path_prompt = input("Choose input file (HT7_AB, HT7_BA, VDP): ").strip()
        if inp_path_prompt == "HT7_AB":
            inp_path = DATA / "HT7:EAST" / "test_AB.inp"
        elif inp_path_prompt == "HT7_BA":
            inp_path = DATA / "HT7:EAST" / "test_BA.inp"
        elif inp_path_prompt == "VDP":
            inp_path = DATA / "VDP" / "test_VDP.inp"
        else:
            print("ERROR: choose HT7_AB, HT7_BA, or VDP")
            sys.exit(1)

    print("="*70)
    print("NLCC Runner - with overview plots")
    print("="*70)

    print("\nRunning full analysis (all windows, all dimensions)...\n")
    run_nlcc_full(str(inp_path), output_dir=str(_PROJECT_ROOT / "output"), plot=True)

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)
