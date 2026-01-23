from __future__ import annotations

"""
sim_vdp_to_dat.py

Generate synthetic time series from a pair of coupled Van der Pol oscillators,
in the spirit of Ding et al. (1997), and write them to data_raw/VDP/ as
x_vdp.dat and y_vdp.dat.

These files are meant to be fed into the NLCC pipeline exactly like A.dat/B.dat:
io.load_signal() should see the same header keys (Frequency, Trigger Time)
and two numeric columns (printed t, value).
"""

import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Dynamical system: coupled Van der Pol/Rayleigh oscillators
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
    Right-hand side for the coupled Rayleigh-type Van der Pol system.

    State:
        z = [x, vx, y, vy],
        where vx = dx/dt, vy = dy/dt.

    Equations:
        x'  = vx
        vx' = [a1 - (x + b*y)^2] * vx  - (x + b*y)

        y'  = vy
        vy' = [a2 - (y + a*x)^2] * vy  - (y + a*x)

    Parameters:
        a1, a2 : Rayleigh parameters of each oscillator (usually 1.0).
        a      : coupling strength from x → y.
        b      : coupling strength from y → x.

    Directionality:
        |a| > |b|  ⇒ x drives y
        |a| < |b|  ⇒ y drives x
        |a| = |b|  ⇒ symmetric coupling
    """
    x, vx, y, vy = z

    dx = vx
    dvx = (a1 - (x + b * y) ** 2) * vx - (x + b * y)

    dy = vy
    dvy = (a2 - (y + a * x) ** 2) * vy - (y + a * x)

    return np.array([dx, dvx, dy, dvy], dtype=float)


# ---------------------------------------------------------------------------
# Generic RK4 stepper
# ---------------------------------------------------------------------------

def rk4_step(
    f,
    t: float,
    z: np.ndarray,
    dt: float,
    *f_args,
) -> np.ndarray:
    """
    One explicit 4th-order Runge–Kutta step.

    Inputs
    ------
    f      : callable
             Right-hand side f(t, z, *f_args) → dz/dt (same shape as z).
    t      : float
             Current time.
    z      : np.ndarray
             Current state vector z(t).
    dt     : float
             Time step (assumed constant).
    f_args : tuple
             Extra parameters passed through to f.

    Returns
    -------
    z_next : np.ndarray
             Approximation to z(t + dt).
    """
    k1 = f(t,           z,               *f_args)
    k2 = f(t + 0.5*dt,  z + 0.5*dt*k1,   *f_args)
    k3 = f(t + 0.5*dt,  z + 0.5*dt*k2,   *f_args)
    k4 = f(t + dt,      z + dt*k3,       *f_args)

    return z + (dt / 6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4)


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------

def simulate_vdp_pair(
    a1: float,
    a2: float ,
    a: float ,
    b: float ,
    dt: float ,
    t_max: float ,
    t_transient: float ,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Integrate the coupled Van der Pol system using fixed-step RK4.

    Strategy
    --------
    - Integrate from t = 0 to t = t_max with constant step dt.
    - Store x(t), y(t) at *every* step.
    - After the integration, discard all samples with t < t_transient.
      The recorded time axis is then shifted so that the first kept
      sample has t_rec = 0.

    Parameters
    ----------
    mu          : float
                  Rayleigh parameter for both oscillators (a1 = a2 = mu).
    a, b        : float
                  Coupling strengths (x → y and y → x).
    dt          : float
                  Time step.
    t_max       : float
                  Final time of the integration.
    t_transient : float
                  Length of transient to discard (in the same units as t).

    Returns
    -------
    t_rec : np.ndarray, shape (N_rec,)
        Recorded times, starting at 0.
    x_rec : np.ndarray, shape (N_rec,)
        Recorded x(t) samples.
    y_rec : np.ndarray, shape (N_rec,)
        Recorded y(t) samples.
    """
    if dt <= 0.0:
        raise ValueError("dt must be positive")
    if t_max <= 0.0:
        raise ValueError("t_max must be positive")
    if t_transient < 0.0:
        raise ValueError("t_transient must be non-negative")
    if t_transient >= t_max:
        raise ValueError("t_transient must be less than t_max")

    # Number of stored time points (include both endpoints)
    n_steps = int(np.floor(t_max / dt)) + 1

    t_all = np.empty(n_steps, dtype=float)
    x_all = np.empty(n_steps, dtype=float)
    y_all = np.empty(n_steps, dtype=float)

    # Initial condition
    t = 0.0
    z = np.array([1.2, 0.0, 1.1, 0.0])

    for n in range(n_steps):
        # Store current state
        t_all[n] = t
        x_all[n] = z[0]
        y_all[n] = z[2]

        # Advance to next time, except after the last stored point
        if n < n_steps - 1:
            z = rk4_step(ding_coupled, t, z, dt, a1, a2, a, b)
            t += dt

    # Discard transient: keep t >= t_transient
    keep = t_all >= t_transient
    if not np.any(keep):
        raise RuntimeError("Transient cut removed all data; decrease t_transient or increase t_max")

    t_rec = t_all[keep] - t_all[keep][0]
    x_rec = x_all[keep]
    y_rec = y_all[keep]

    return t_rec, x_rec, y_rec


# ---------------------------------------------------------------------------
# Output helpers
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
# Main script
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Drive the simulation and write x_vdp.dat / y_vdp.dat into data_raw/VDP.

    Default choice |a| > |b| means oscillator x tends to drive y, so in NLCC
    you should see G_xy > G_yx if everything is behaving sensibly.
    """
    # Coupling parameters roughly in line with Ding-style examples
    a1 =1
    a2 = 1.1
    a = 1   # x → y (stronger)
    b = 0  # y → x (weaker)

    # Simulation parameters
    dt = 0.01

    # Enough time for transients to die
    t_max = 1000

    # Throw away the first 3000 units
    t_trans = 100
    t_rec, x_rec, y_rec = simulate_vdp_pair(
        a1=a1,
        a2=a2,
        a=a,
        b=b,
        dt=dt,
        t_max=t_max,
        t_transient=t_trans,
    )

    base_dir = Path("../data_raw") / "VDP"
    x_path = base_dir / "x_vdp.dat"
    y_path = base_dir / "y_vdp.dat"

    write_dat(x_path, t_rec, x_rec, dt)
    write_dat(y_path, t_rec, y_rec, dt)

    print(f"Wrote {x_path} and {y_path}")
    print(f"N = {x_rec.size}, dt = {dt}, t_rec in [{t_rec[0]:.3f}, {t_rec[-1]:.3f}]")


if __name__ == "__main__":
    main()
