#!/usr/bin/env python3
"""
sim_vdp_benchmarks.py

Generate synthetic time series from two coupled Van der Pol systems:

1. Ding-style Rayleigh VdP pair (Ding et al. 1997):
   x, y second-order oscillators with mutual (possibly asymmetric) coupling.

2. van Milligen 2014-style pair:
   Unidirectional coupling: oscillator 2 drives oscillator 1.

Outputs:
  - NLCC-compatible .dat files in ../data_raw/VDP/
  - Time-series and phase-space plots as PNGs in ../results/VDP/

Run this from the scripts/ directory.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# RHS definitions
# ---------------------------------------------------------------------------

def ding_coupled(
    t: float,
    z: np.ndarray,
    a1: float,
    a2: float,
    a: float,  # coupling x -> y
    b: float,  # coupling y -> x
) -> np.ndarray:
    """
    Ding-style coupled Rayleigh / Van der Pol system.

    State:
        z = [x, vx, y, vy]
    where
        x, y  : positions
        vx, vy: velocities (time derivatives of x, y)

    Equations (second-order form written as first-order system):

        x'  = vx
        vx' = (a1 - (x + b y)^2) * vx - (x + b y)

        y'  = vy
        vy' = (a2 - (y + a x)^2) * vy - (y + a x)

    a1, a2: Rayleigh parameters (usually ~O(1))
    a, b  : coupling strengths (x -> y and y -> x).
    """
    x, vx, y, vy = z

    dx = vx
    dvx = (a1 - (x + b * y) ** 2) * vx - (x + b * y)

    dy = vy
    dvy = (a2 - (y + a * x) ** 2) * vy - (y + a * x)

    return np.array([dx, dvx, dy, dvy], dtype=float)


def vm_coupled(
    t: float,
    z: np.ndarray,
    eps1: float,
    eps2: float,
) -> np.ndarray:
    """
    van Milligen 2014-style pair of VdP oscillators with unidirectional coupling.

    State:
        z = [x1, y1, x2, y2]
    where
        x1, x2 : positions
        y1, y2 : velocities

    Equations (from the TE test system):

        x1' = y1
        y1' = [eps1 - (x1 + x2)^2] * y1 - (x1 + x2)

        x2' = y2
        y2' = [eps2 - (x2)^2] * y2 - x2

    Here oscillator 2 drives oscillator 1 via the (x1 + x2) term, but
    oscillator 1 does not appear in oscillator 2's equation.
    """
    x1, y1, x2, y2 = z

    dx1 = y1
    dy1 = (eps1 - (x1 + x2) ** 2) * y1 - (x1 + x2)

    dx2 = y2
    dy2 = (eps2 - x2 ** 2) * y2 - x2

    return np.array([dx1, dy1, dx2, dy2], dtype=float)


# ---------------------------------------------------------------------------
# Generic RK4 and fixed-step integrator
# ---------------------------------------------------------------------------

def rk4_step(
    f,
    t: float,
    z: np.ndarray,
    dt: float,
    *f_args,
) -> np.ndarray:
    """
    Single explicit 4th-order Runge–Kutta step.

    Parameters
    ----------
    f : callable
        RHS function f(t, z, *f_args) -> dz/dt with same shape as z.
    t : float
        Current time.
    z : np.ndarray
        Current state vector.
    dt : float
        Time step.
    f_args : tuple
        Extra args passed to f.

    Returns
    -------
    z_next : np.ndarray
        Approximation to z(t + dt).
    """
    k1 = f(t,          z,             *f_args)
    k2 = f(t + 0.5*dt, z + 0.5*dt*k1, *f_args)
    k3 = f(t + 0.5*dt, z + 0.5*dt*k2, *f_args)
    k4 = f(t + dt,     z + dt*k3,     *f_args)

    return z + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


def simulate_system(
    rhs,
    z0: np.ndarray,
    dt: float,
    t_max: float,
    t_transient: float,
    *rhs_args,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fixed-step integration for a general ODE system.

    Parameters
    ----------
    rhs : callable
        RHS function rhs(t, z, *rhs_args) -> dz/dt.
    z0 : np.ndarray
        Initial state vector.
    dt : float
        Time step (> 0).
    t_max : float
        Final integration time (t runs from 0 to t_max).
    t_transient : float
        Length of transient to discard (0 <= t_transient < t_max).
    rhs_args : tuple
        Extra parameters passed to rhs.

    Returns
    -------
    t_rec : np.ndarray, shape (N_rec,)
        Recorded times, starting at 0 after transient removal.
    z_rec : np.ndarray, shape (N_rec, dim)
        Recorded states corresponding to t_rec.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if t_max <= 0.0:
        raise ValueError("t_max must be positive")
    if t_transient < 0.0:
        raise ValueError("t_transient must be non-negative")
    if t_transient >= t_max:
        raise ValueError("t_transient must be less than t_max")

    dim = z0.size
    n_steps = int(np.floor(t_max / dt)) + 1

    t_all = np.empty(n_steps, dtype=float)
    z_all = np.empty((n_steps, dim), dtype=float)

    t = 0.0
    z = np.array(z0, dtype=float)

    for n in range(n_steps):
        t_all[n] = t
        z_all[n] = z

        if n < n_steps - 1:
            z = rk4_step(rhs, t, z, dt, *rhs_args)
            t += dt

    keep = t_all >= t_transient
    if not np.any(keep):
        raise RuntimeError(
            "Transient cut removed all data; decrease t_transient or increase t_max"
        )

    t_rec = t_all[keep]
    t_rec = t_rec - t_rec[0]  # shift so first kept sample has t=0
    z_rec = z_all[keep]

    return t_rec, z_rec


# ---------------------------------------------------------------------------
# NLCC-friendly output
# ---------------------------------------------------------------------------

def write_dat(
    path: Path,
    t: np.ndarray,
    x: np.ndarray,
    dt: float,
) -> None:
    """
    Write a single time series to a .dat file in the NLCC-friendly format.

    Header format:
        Frequency=<dt in seconds>
        Trigger Time=0.000000e+00
        # t  value
        <t_0> <x_0>
        <t_1> <x_1>
        ...

    Note: in your NLCC I/O, "Frequency" is interpreted as the sampling
    interval dt (seconds), *not* Hz.
    """
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w") as f:
        f.write(f"Frequency={dt:.6e}\n")
        f.write("Trigger Time=0.000000e+00\n")
        f.write("# t  value\n")
        for t_val, val in zip(t, x):
            f.write(f"{t_val:.6e} {val:.6e}\n")


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------
def plot_vm(
    out_dir: Path,
    t: np.ndarray,
    x1: np.ndarray,
    y1: np.ndarray,
    x2: np.ndarray,
    y2: np.ndarray,
) -> None:
    """
    Make time-series and phase-space plots for the van Milligen system.

    - Time series: first T_plot seconds only.
    - Phase space: decimated trajectories.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Time series window ----
    T_plot = 50.0
    mask_ts = t <= t[0] + T_plot

    t_ts  = t[mask_ts]
    x1_ts = x1[mask_ts]
    x2_ts = x2[mask_ts]

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 5))
    ax[0].plot(t_ts, x1_ts, linewidth=0.8)
    ax[0].set_ylabel("x1(t)")
    ax[0].set_title(f"van Milligen VdP: time series (first {T_plot:g} s)")

    ax[1].plot(t_ts, x2_ts, linewidth=0.8)
    ax[1].set_ylabel("x2(t)")
    ax[1].set_xlabel("t")
    fig.tight_layout()
    fig.savefig(out_dir / "vm_timeseries.png", dpi=200)
    plt.close(fig)

    # ---- Phase space (decimated) ----
    stride = 10
    x1_ps = x1[::stride]
    y1_ps = y1[::stride]
    x2_ps = x2[::stride]
    y2_ps = y2[::stride]

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].plot(x1_ps, y1_ps, linewidth=0.6, alpha=0.8)
    axs[0].set_xlabel("x1")
    axs[0].set_ylabel("y1")
    axs[0].set_title("VM: x1–y1")

    axs[1].plot(x2_ps, y2_ps, linewidth=0.6, alpha=0.8)
    axs[1].set_xlabel("x2")
    axs[1].set_ylabel("y2")
    axs[1].set_title("VM: x2–y2")

    axs[2].plot(x2_ps, x1_ps, linewidth=0.6, alpha=0.8)
    axs[2].set_xlabel("x2")
    axs[2].set_ylabel("x1")
    axs[2].set_title("VM: x1–x2")

    fig.tight_layout()
    fig.savefig(out_dir / "vm_phase_space.png", dpi=200)
    plt.close(fig)


def plot_ding(
    out_dir: Path,
    t: np.ndarray,
    x: np.ndarray,
    vx: np.ndarray,
    y: np.ndarray,
    vy: np.ndarray,
) -> None:
    """
    Make time-series and phase-space plots for the Ding system.

    - Time series: show only a short window so the oscillations are readable.
    - Phase space: decimate points to avoid overplotting.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Time series: show only first T_plot seconds ----
    T_plot = 50.0   # change if you want a longer/shorter window
    mask_ts = t <= t[0] + T_plot

    t_ts = t[mask_ts]
    x_ts = x[mask_ts]
    y_ts = y[mask_ts]

    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(8, 5))

    ax[0].plot(t_ts, x_ts, linewidth=0.8)
    ax[0].set_ylabel("x(t)")
    ax[0].set_title(f"Ding VdP: time series (first {T_plot:g} s)")

    ax[1].plot(t_ts, y_ts, linewidth=0.8)
    ax[1].set_ylabel("y(t)")
    ax[1].set_xlabel("t")
    fig.tight_layout()
    fig.savefig(out_dir / "ding_timeseries.png", dpi=200)
    plt.close(fig)

    # ---- Phase space: decimate to reduce clutter ----
    stride = 10  # take every 10th point
    x_ps  = x[::stride]
    vx_ps = vx[::stride]
    y_ps  = y[::stride]
    vy_ps = vy[::stride]

    fig, axs = plt.subplots(1, 3, figsize=(12, 4))

    axs[0].plot(x_ps, vx_ps, linewidth=0.6, alpha=0.7)
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("vx")
    axs[0].set_title("Ding: x–vx")

    axs[1].plot(y_ps, vy_ps, linewidth=0.6, alpha=0.7)
    axs[1].set_xlabel("y")
    axs[1].set_ylabel("vy")
    axs[1].set_title("Ding: y–vy")

    axs[2].plot(x_ps, y_ps, linewidth=0.6, alpha=0.7)
    axs[2].set_xlabel("x")
    axs[2].set_ylabel("y")
    axs[2].set_title("Ding: x–y")

    fig.tight_layout()
    fig.savefig(out_dir / "ding_phase_space.png", dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Run both benchmark systems and write .dat + plots.

    - Ding system: writes x_vdp_ding.dat, y_vdp_ding.dat
    - van Milligen system: writes x_vdp_vm1.dat (x1), y_vdp_vm2.dat (x2)
    """
    # Directories relative to scripts/
    data_dir = Path("../data_raw") / "VDP"
    fig_dir = Path("../results") / "VDP"

    # ---------------------------
    # 1. Ding 1997-style system
    # ---------------------------
    # Parameters chosen to match the classic Ding example:
    #   a1 = a2 = 1, a = 2.5, b = -0.5
    a1_d = 1.0
    a2_d = 1.0
    a_d = 2.5
    b_d = -0.5

    dt_d = 0.01
    t_max_d = 2000.0
    t_trans_d = 100.0

    # Initial conditions (arbitrary, transient will wash out)
    z0_d = np.array([0.1, 0.0, 0.2, 0.0])

    t_d, z_d = simulate_system(
        ding_coupled,
        z0_d,
        dt_d,
        t_max_d,
        t_trans_d,
        a1_d,
        a2_d,
        a_d,
        b_d,
    )

    x_d = z_d[:, 0]
    vx_d = z_d[:, 1]
    y_d = z_d[:, 2]
    vy_d = z_d[:, 3]

    # NLCC-style outputs
    write_dat(data_dir / "x_vdp_ding.dat", t_d, x_d, dt_d)
    write_dat(data_dir / "y_vdp_ding.dat", t_d, y_d, dt_d)

    # Plots
    plot_ding(fig_dir / "Ding", t_d, x_d, vx_d, y_d, vy_d)

    print(f"Ding run: N = {x_d.size}, dt = {dt_d}, t in [{t_d[0]:.3f}, {t_d[-1]:.3f}]")

    # ---------------------------
    # 2. van Milligen 2014 system
    # ---------------------------
    # Parameters from the TE test:
    #   eps1 = 1.0, eps2 = 1.1, unidirectional coupling
    eps1 = 1.0
    eps2 = 1.1

    dt_vm = 0.01
    t_max_vm = 1000.0
    t_trans_vm = 100.0

    z0_vm = np.array([0.1, 0.0, 0.2, 0.0])

    t_vm, z_vm = simulate_system(
        vm_coupled,
        z0_vm,
        dt_vm,
        t_max_vm,
        t_trans_vm,
        eps1,
        eps2,
    )

    x1 = z_vm[:, 0]
    y1 = z_vm[:, 1]
    x2 = z_vm[:, 2]
    y2 = z_vm[:, 3]

    write_dat(data_dir / "x_vdp_vm1.dat", t_vm, x1, dt_vm)
    write_dat(data_dir / "y_vdp_vm2.dat", t_vm, x2, dt_vm)

    plot_vm(fig_dir / "vanMilligen", t_vm, x1, y1, x2, y2)

    print(f"van Milligen run: N = {x1.size}, dt = {dt_vm}, t in [{t_vm[0]:.3f}, {t_vm[-1]:.3f}]")


if __name__ == "__main__":
    main()
