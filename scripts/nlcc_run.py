"""
run_nlcc.py - Final corrected runner script

Critical fixes:
1. I_time is already in samples (multiply by 1000, not divide!)
2. MATLAB: I_begin = 1000.0 * tbegin (where tbegin is in ms)
3. Python 0-based indexing: x[I_time:I_time+N] maps to MATLAB's AN((I_time+1):(I_time+N))
"""

from nlcc.nlcc import range_normalize
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from nlcc.io import load_inp, load_signal
from nlcc.nlcc import (
    conditional_dispersion_curves,
    nlcc_metrics_from_curves,
    eps_grid_from_inp_matlab
)

def run_nlcc_full(inp_path, output_dir="output", plot=True):
    """
    Run full NLCC analysis matching MATLAB behavior exactly.
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
    A = load_signal(data_dir / cfg['input_A'])
    B = load_signal(data_dir / cfg['input_B'])

    x = A["data"]["x"]
    y = B["data"]["x"]

    print(f"  Signal A ({cfg['input_A']}): {len(x)} samples")
    print(f"  Signal B ({cfg['input_B']}): {len(y)} samples")

    # Extract dt from metadata
    freq_str = A["meta"]["Frequency"].split()[0]
    dt = float(freq_str)
    print(f"  dt = {dt:.6e} s = {dt*1e6:.3f} µs")


    I_begin = int(1000.0 * cfg["t_begin"])-1 #IMPORTANT: For weeks my code wouldn't match MATLAB and the error
                                             #           is due to how MATLAB and Python handle indexing (1-index vs 0-index)
    I_end   = int(1000.0 * cfg["t_end"])-1
    I_step  = int(1000.0 * cfg["t_step"])

    print(f"  Time window sample indices: {I_begin} to {I_end} step {I_step}")
    print(f"  (This assumes dt ≈ 1ms per sample)")

    # Verify this makes sense
    if dt > 0:
        implied_t_begin_ms = I_begin * dt * 1000
        print(f"  Verification: I_begin={I_begin} × dt={dt:.6e}s × 1000 = {implied_t_begin_ms:.1f}ms")
        if abs(implied_t_begin_ms - cfg["t_begin"]) > 1.0:
            print(f"  WARNING: Time mismatch! Expected {cfg['t_begin']}ms")

    # Build epsilon grid
    eps_vals, log2_eps = eps_grid_from_inp_matlab(cfg)
    print(f"  Epsilon grid: {len(eps_vals)} values")
    print(f"    log2(eps) from {log2_eps[0]:.3f} to {log2_eps[-1]:.3f}")
    print(f"    eps from {eps_vals[0]:.6e} to {eps_vals[-1]:.6e}")

    # Open output file
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

            # CRITICAL: Extract window matching MATLAB
            # MATLAB: AN11 = AN((I_time+1):(I_time+N))  [1-based, inclusive both ends]
            # Python: xw = x[I_time:I_time+N]  [0-based, exclusive end]
            # These are equivalent!

            N = cfg["N"]
            xw = x[I_time : I_time + N]
            yw = y[I_time : I_time + N]

            if len(xw) < N or len(yw) < N:
                print(f"    Warning: Not enough data (got {len(xw)}/{len(yw)}, need {N})")
                fout.write("    ERROR: Insufficient data\n")
                continue

            print(f"    Extracted window: {len(xw)} samples")
            print(f"    A range: [{np.min(xw):.6e}, {np.max(xw):.6e}]")
            print(f"    B range: [{np.min(yw):.6e}, {np.max(yw):.6e}]")

            # Loop over dimensions
            for D in range(cfg["dim_min"], cfg["dim_max"] + 1, cfg["dim_step"]):
                print(f"    Dimension D={D}... ", end='', flush=True)
                fout.write(f"\nDimension = {D}\n")
                fout.write(f"File X: {cfg['input_A']}            File Y:  {cfg['input_B']}\n")

                tau = cfg["tau"]
                dM = cfg["dM"]

                try:
                    # Compute curves
                    curves = conditional_dispersion_curves(
                        xw, yw,
                        dim=D,
                        tau=tau,
                        eps_values=eps_vals,
                        normalization="range",
                        dM=dM,
                        max_pairs=None,
                    )

                    # Save dispersion data
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

                    # Write to .cor file (matching MATLAB format)
                    fout.write(f"epsilon at which log2(epsilon)=0.9-------------\n")
                    fout.write(f"XX             XY             YY             YX\n")
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


def plot_results(cfg, output_dir, run_name):
    """
    Generate plots matching MATLAB visualization
    """
    output_dir = Path(output_dir)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    dimensions = range(cfg["dim_min"], cfg["dim_max"] + 1, cfg["dim_step"])

    for D in dimensions:
        dat_file = output_dir / f"{run_name}{D}.dat"

        if not dat_file.exists():
            continue

        # Load data
        data = np.loadtxt(dat_file, skiprows=1)
        log2_eps = data[:, 0]
        sigma_xy = data[:, 2]
        sigma_yx = data[:, 4]

        # Plot XY
        ax1.plot(log2_eps, sigma_xy, '-o', markersize=3, label=f'M = {D}')

        # Plot YX
        ax2.plot(log2_eps, sigma_yx, '-o', markersize=3, label=f'M = {D}')

    ax1.set_xlabel('logE (base 2)')
    ax1.set_ylabel('sigma XY(E)')
    ax1.grid(True)
    ax1.legend()

    ax2.set_xlabel('logE (base 2)')
    ax2.set_ylabel('sigma YX(E)')
    ax2.grid(True)
    ax2.legend()

    plt.tight_layout()

    fig_file = output_dir / f"{run_name}_figure.png"
    plt.savefig(fig_file, dpi=150)
    print(f"✓ Figure saved to {fig_file}")

    plt.show()


def run_quick_test(inp_path):
    """
    Quick test: single time window, single dimension, with diagnostics
    """
    cfg = load_inp(inp_path)

    data_dir = Path(inp_path).parent
    A = load_signal(data_dir / cfg['input_A'])
    B = load_signal(data_dir / cfg['input_B'])

    x = A["data"]["x"]
    y = B["data"]["x"]

    freq_str = A["meta"]["Frequency"].split()[0]
    dt = float(freq_str)

    # MATLAB indexing
    # Ding/NC_ASIPP behavior: t_begin is a raw sample index
    I_begin = int(cfg["t_begin"])
    N = int(cfg["N"])

    xw = x[I_begin: I_begin + N]
    yw = y[I_begin: I_begin + N]
    print(f"Window length: {len(xw)} samples")
    print(f"A: min={np.min(xw):.6e}, max={np.max(xw):.6e}")
    print(f"B: min={np.min(yw):.6e}, max={np.max(yw):.6e}")

    # Normalize
    xw_norm = range_normalize(xw)
    yw_norm = range_normalize(yw)

    print(f"\nNormalized A (first 5): {xw_norm[:5]}")
    print(f"Normalized B (first 5): {yw_norm[:5]}")

    eps_vals, log2_eps = eps_grid_from_inp_matlab(cfg)

    print(f"\nEpsilon grid: {len(eps_vals)} values")
    print(f"  First 3: {eps_vals[:3]}")
    print(f"  Last 3: {eps_vals[-3:]}")

    D = cfg["dim_max"]

    print(f"\nRunning NLCC with D={D}, tau={cfg['tau']}, dM={cfg['dM']}...")

    curves = conditional_dispersion_curves(
        xw, yw,
        dim=D,
        tau=cfg["tau"],
        eps_values=eps_vals,
        normalization="range",
        dM=cfg["dM"],
    )

    plt.figure(figsize=(10, 5))
    plt.plot(curves["log2_eps"], curves["sigma_xy"], '-o', label=f"σ_XY (D={D})", markersize=4)
    plt.plot(curves["log2_eps"], curves["sigma_yx"], '-o', label=f"σ_YX (D={D})", markersize=4)
    plt.xlabel("log₂ ε")
    plt.ylabel("σ")
    plt.grid(True)
    plt.legend()
    plt.title(f"NLCC curves - {cfg['input_A']} vs {cfg['input_B']}")
    plt.tight_layout()
    plt.show()

    metrics = nlcc_metrics_from_curves(curves)
    print("\nMetrics at σ=0.9:")
    for k, v in metrics.items():
        print(f"  {k:8s} = {v:.6e}")


if __name__ == "__main__":
    import sys

    # Default input file
    inp_path = ("../data_raw/HT7:EAST/test_AB"".inp")

    if len(sys.argv) > 1:
        inp_path = sys.argv[1]

    print("="*70)
    print("NLCC Runner - Python Implementation")
    print("="*70)

    # Choose mode
    mode = "full"  # Change to "full" for complete analysis

    if mode == "quick":
        print("\nRunning quick test (single window, single dimension)...\n")
        run_quick_test(inp_path)
    else:
        print("\nRunning full analysis (all windows, all dimensions)...\n")
        run_nlcc_full(inp_path, output_dir="output", plot=True)

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)

