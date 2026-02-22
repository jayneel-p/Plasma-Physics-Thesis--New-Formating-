"""
run_nlcc.py - NLCC runner. Run from project root: python scripts/nlcc_run.py
"""
import sys
from pathlib import Path

# So nlcc can be imported after cloning (run from any directory)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt

from nlcc import load_inp, load_signal, range_normalize, conditional_dispersion_curves, nlcc_metrics_from_curves, eps_grid_from_inp_matlab

def run_nlcc_full(inp_path, output_dir="results", plot=True):
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

    # Load signals from io file
    data_dir = Path(inp_path).parent
    A = load_signal(data_dir / cfg['input_A']) #{"meta": meta, "data": {"t": t, "x": x}}
    B = load_signal(data_dir / cfg['input_B'])

    x = A["data"]["x"]
    y = B["data"]["x"]
    t=A["data"]["t"]


    plt.plot(x,y)
    plt.show()
    plt.plot(t,x)
    plt.plot(t,y)
    plt.xlim(0,40)
    plt.ylim(-2,6)
    plt.show()


    print(f"  Signal A ({cfg['input_A']}): {len(x)} samples")
    print(f"  Signal B ({cfg['input_B']}): {len(y)} samples")

    # Extract dt from metadata
    freq_str = A["meta"]["Frequency"].split()[0]
    dt = float(freq_str)
    print(f"  dt = {dt:.6e} s = {dt*1e6:.3f} µs")

    # VDP data: t_begin/t_end in ms -> sample index via dt
    inp_path_str = str(inp_path)
    if "VDP" in inp_path_str and ("test_VDP_ding" in inp_path_str or "test_VDP_vm" in inp_path_str):
        dt_ms = 1000.0 * dt
        I_begin = int(cfg["t_begin"] / dt_ms)
        I_end = int(cfg["t_end"] / dt_ms)
        I_step = int(cfg["t_step"] / dt_ms)
    else:
        I_begin = int(1000 * cfg["t_begin"])-1 #IMPORTANT: For weeks my code wouldn't match MATLAB and the error is due to how MATLAB and Python handle indexing (1-index vs 0-index)
        I_end   = int(1000 * cfg["t_end"])-1
        I_step  = int(1000 * cfg["t_step"])

    print(f"  Time window sample indices: {I_begin} to {I_end} step {I_step}")

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

            #print statement (can ignore)
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





if __name__ == "__main__":
    DATA = _PROJECT_ROOT / "data_raw"

    if len(sys.argv) > 1:
        inp_path = Path(sys.argv[1]).resolve()
    else:
        inp_path_prompt = input("Choose input file (ht7_ab, ht7_ba, vdp_ding, vdp_vm): ").strip()
        if inp_path_prompt == "ht7_ab":
            inp_path = DATA / "HT7:EAST" / "test_AB.inp"
        elif inp_path_prompt == "ht7_ba":
            inp_path = DATA / "HT7:EAST" / "test_BA.inp"
        elif inp_path_prompt == "vdp_ding":
            inp_path = DATA / "VDP" / "test_VDP_ding.inp"
        elif inp_path_prompt == "vdp_vm":
            inp_path = DATA / "VDP" / "test_VDP_vm.inp"
        else:
            print("Choose ht7_ab, ht7_ba, vdp_ding, or vdp_vm")
            sys.exit(1)

    print("="*70)
    print("NLCC Runner")
    print("="*70)

    print("\nRunning full analysis (all windows, all dimensions)...\n")
    run_nlcc_full(str(inp_path), output_dir=str(_PROJECT_ROOT / "results"), plot=True)

    print("\n" + "="*70)
    print("Analysis complete!")
    print("="*70)

