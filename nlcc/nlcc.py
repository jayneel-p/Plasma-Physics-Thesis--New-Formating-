"""
nlcc.py - CORRECTED VERSION

Nonlinear "conditional cross-correlation" in the sense of
Ding et al. (1997) and the NC_ASIPP MATLAB/Fortran code.

Goal: produce σ(log2 ε) curves and NLCC metrics (G_xy, G_yx, S_xy)
that match the original NC_ASIPP outputs to machine precision.

Key implementation choices (all matched to Ding/NC_ASIPP):
  1. Normalise each TIME WINDOW by its range (divide by range, do NOT subtract min).
  2. Build delay-embedding vectors with M = tau*(D-1) and use the same
     pair loops as MATLAB (i < j).
  3. Compute distances exactly as in the double loop and divide by dM.
  4. Reproduce the epsilon grid used in NC_ASIPP.
  5. Interpolate ε at σ = 0.9 using the same piecewise logic as the MATLAB code.
"""

from __future__ import annotations

import numpy as np
from typing import Literal, Optional, Dict, Tuple


# ---------------------------------------------------------------------------
# Basic preprocessing
# ---------------------------------------------------------------------------

def range_normalize(x: np.ndarray) -> np.ndarray:
    """
    Range "normalization" used in Ding/NC_ASIPP.

    Args:
        x: 1D array-like, time series segment (window).

    Returns:
        1D array with entries x / (max(x) - min(x)).
        Note: we do NOT subtract xmin. This matches NC_ASIPP exactly.
    """
    x = np.asarray(x, dtype=float)
    xmin = np.nanmin(x)
    xmax = np.nanmax(x)
    denom = xmax - xmin
    if denom <= 0:
        # Flat signal -> zero out so distances vanish.
        return np.zeros_like(x, dtype=float)
    # CORRECT: divide by the range only (no shift).
    return x / denom


def zscore(x: np.ndarray) -> np.ndarray:
    """
    Standard z-score: (x - mean) / std.

    Mostly for testing; Ding/NC_ASIPP uses range scaling.
    Args:
        x: 1D array-like, time series segment.

    Returns:
        1D array, z-scored version of x (NaN-safe).
    """
    x = np.asarray(x, dtype=float)
    mu = np.nanmean(x)
    sd = np.nanstd(x)
    if sd <= 0:
        return np.zeros_like(x, dtype=float)
    return (x - mu) / sd


# ---------------------------------------------------------------------------
# Embedding
# ---------------------------------------------------------------------------

def embed(ts: np.ndarray, dim: int, tau: int) -> np.ndarray:
    """
    Takens delay embedding used in Ding's algorithm.

    Args:
        ts: 1D array, already normalized time series window.
        dim: embedding dimension D.
        tau: delay in samples.

    Returns:
        X: array of shape (N_embed, dim) where each row is
           [x_i, x_{i+tau}, ..., x_{i + (dim-1) tau}].

    Ding/NC_ASIPP equivalence:
        M = tau*(D-1)
        i = 1 : N - M - 1
        j = i+1 : N - M
        and then they sum over k = 0:tau:M.
    """
    ts = np.asarray(ts, dtype=float)
    if dim < 1:
        raise ValueError("dim must be >= 1")
    if tau < 1:
        raise ValueError("tau must be >= 1")

    # Total delay span M = tau * (D-1) (same as MATLAB).
    M = tau * (dim - 1)
    N = len(ts)
    # Number of valid starting points for embedding.
    N_embed = N - M

    if N_embed <= 1:
        raise ValueError("Time series too short for embedding with given dim and tau.")

    # Build the embedded matrix by shifting the window by tau each column.
    X = np.empty((N_embed, dim), dtype=float)
    for j in range(dim):
        X[:, j] = ts[j * tau : j * tau + N_embed]
    return X


# ---------------------------------------------------------------------------
# Pair selection and distance statistics
# ---------------------------------------------------------------------------

def _select_pairs(
    N: int,
    max_pairs: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Select index pairs (i, j) with i < j.

    Args:
        N: number of embedded points (rows of X/Y).
        max_pairs: if None, use all N*(N-1)/2 pairs.
                   if set, randomly subsample that many distinct pairs.
        rng: numpy Generator used for subsampling (for reproducibility).

    Returns:
        (i_idx, j_idx): integer arrays of same length with 0-based indices.

    Ding/NC_ASIPP equivalent:
        for i = 1 : N-M-1
            for j = i+1 : N-M
    """
    if N < 2:
        raise ValueError("Need at least 2 points to form pairs.")

    total_pairs = N * (N - 1) // 2

    # All pairs: explicit i<j loops (same structure as MATLAB).
    if max_pairs is None or max_pairs >= total_pairs:
        i_idx = []
        j_idx = []
        for i in range(N - 1):
            j_range = np.arange(i + 1, N, dtype=int)
            i_idx.append(np.full_like(j_range, i))
            j_idx.append(j_range)
        return np.concatenate(i_idx), np.concatenate(j_idx)

    # Subsample pairs (for large N we don't need all pairs).
    if rng is None:
        rng = np.random.default_rng(0)

    pair_ids = rng.choice(total_pairs, size=max_pairs, replace=False)
    i_idx = np.empty(max_pairs, dtype=int)
    j_idx = np.empty(max_pairs, dtype=int)

    count = 0
    k_global = 0  # flat index running over all (i,j) with i<j
    for i in range(N - 1):
        for j in range(i + 1, N):
            if k_global in pair_ids:
                i_idx[count] = i
                j_idx[count] = j
                count += 1
                if count == max_pairs:
                    return i_idx, j_idx
            k_global += 1

    return i_idx[:count], j_idx[:count]


def _mean_square_distances(
    X: np.ndarray,
    Y: np.ndarray,
    i: np.ndarray,
    j: np.ndarray,
    dM: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean squared distances DX and DY between embedded points.

    Args:
        X: array (N_embed, dim) from embed(x, dim, tau).
        Y: array (N_embed, dim) from embed(y, dim, tau).
        i, j: integer arrays of same length, index pairs (0-based).
        dM: divisor used in Ding/NC_ASIPP. If None, use dim.

    Returns:
        (DX2, DY2): arrays of shape (n_pairs,) with mean squared
        distances for X and Y. We only take sqrt later.

    Ding/NC_ASIPP equivalent (matlab lines 164–170):
        DX = 0.0; DY = 0.0
        for k = 0 : tau : M
            DX += (AN1(i+k) - AN1(j+k))^2
            DY += (BN1(i+k) - BN1(j+k))^2
        DX = DX / dM
        DXSQRT = sqrt(DX)
    """
    if X.shape[0] != Y.shape[0]:
        raise ValueError("X and Y must have the same number of rows.")
    if X.shape[0] <= 1:
        raise ValueError("Not enough embedded points.")

    # Differences along each embedding coordinate.
    diff_X = X[i] - X[j]  # (n_pairs, dim)
    diff_Y = Y[i] - Y[j]

    dim = X.shape[1]
    denom = dM if dM is not None else dim

    # Sum over embedding coordinates, divide by dM:
    DX2 = np.sum(diff_X**2, axis=1) / denom
    DY2 = np.sum(diff_Y**2, axis=1) / denom

    return DX2, DY2


# ---------------------------------------------------------------------------
# Conditional dispersion curves σ(ε)
# ---------------------------------------------------------------------------

def conditional_dispersion_curves(
    x: np.ndarray,
    y: np.ndarray,
    dim: int,
    tau: int,
    eps_values: np.ndarray,
    *,
    normalization: Literal["range", "zscore"] = "range",
    dM: Optional[int] = None,
    max_pairs: Optional[int] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute Ding-style conditional dispersion curves:

        σ_xx(ε), σ_xy(ε), σ_yy(ε), σ_yx(ε).

    Args:
        x, y: 1D arrays, time series windows (raw samples).
              This function handles the normalization internally.
        dim: embedding dimension D.
        tau: delay in samples.
        eps_values: 1D array of ε values where σ is evaluated.
        normalization: "range" (Ding/NC_ASIPP) or "zscore" (for tests).
        dM: divisor used in distance calculation (NC_ASIPP parameter dM).
        max_pairs: optional cap on number of point pairs (for speed).
        rng: numpy Generator used when subsampling pairs.

    Returns:
        dict with keys:
          "eps"        : ε values (1D array)
          "log2_eps"   : log2(ε)
          "sigma_xx"   : σ_xx(ε)
          "sigma_xy"   : σ_xy(ε)
          "sigma_yy"   : σ_yy(ε)
          "sigma_yx"   : σ_yx(ε)

    σ_xx(ε)   : dispersion of X within ε-neighbourhoods in X-space.
    σ_xy(ε)   : dispersion of Y within the same X-conditioned neighbourhoods.
    σ_yy,σ_yx : same story but conditioning on Y.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    N = min(len(x), len(y))
    if N <= 2:
        raise ValueError("Time series too short.")

    # Make sure both windows are same length.
    x = x[:N]
    y = y[:N]

    # --- Normalise the WINDOW (critical for matching Ding/NC_ASIPP). ---
    if normalization == "range":
        x_norm = range_normalize(x)
        y_norm = range_normalize(y)
    elif normalization == "zscore":
        x_norm = zscore(x)
        y_norm = zscore(y)
    else:
        raise ValueError(f"Unknown normalization '{normalization}'")

    # Delay embedding of each normalized window.
    X = embed(x_norm, dim=dim, tau=tau)
    Y = embed(y_norm, dim=dim, tau=tau)
    N_embed = X.shape[0]

    # Select (i, j) pairs of embedded points.
    i, j = _select_pairs(N_embed, max_pairs=max_pairs, rng=rng)
    if len(i) == 0:
        raise RuntimeError("No pairs selected; check N_embed and max_pairs.")

    # Mean squared distances between embedded points.
    DX2, DY2 = _mean_square_distances(X, Y, i, j, dM=dM)

    # Physical distances (same units as eps_values).
    DX = np.sqrt(DX2)
    DY = np.sqrt(DY2)

    eps_vals = np.asarray(eps_values, dtype=float)
    if eps_vals.ndim != 1:
        raise ValueError("eps_values must be a 1D array.")

    sigma_xx = np.zeros_like(eps_vals, dtype=float)
    sigma_xy = np.zeros_like(eps_vals, dtype=float)
    sigma_yy = np.zeros_like(eps_vals, dtype=float)
    sigma_yx = np.zeros_like(eps_vals, dtype=float)

    # For each ε, condition on balls in phase space.
    for k, eps in enumerate(eps_vals):
        # X-conditioned neighbourhood: DX < ε
        mask_X = DX < eps
        if np.any(mask_X):
            # How spread out X and Y are inside X-neighbourhoods.
            sigma_xx[k] = np.sqrt(np.mean(DX2[mask_X]))
            sigma_xy[k] = np.sqrt(np.mean(DY2[mask_X]))

        # Y-conditioned neighbourhood: DY < ε
        mask_Y = DY < eps
        if np.any(mask_Y):
            sigma_yy[k] = np.sqrt(np.mean(DY2[mask_Y]))
            sigma_yx[k] = np.sqrt(np.mean(DX2[mask_Y]))

    # Normalise each σ(ε) by its own max (as in NC_ASIPP).
    def _normalise_curve(s: np.ndarray) -> np.ndarray:
        if np.all(np.isnan(s)):
            return s.copy()
        finite = np.isfinite(s)
        if not finite.any():
            return s.copy()
        s_max = np.nanmax(s[finite])
        if s_max <= 0:
            return s.copy()
        out = s.copy()
        out[finite] = out[finite] / s_max
        return out

    sigma_xx_n = _normalise_curve(sigma_xx)
    sigma_xy_n = _normalise_curve(sigma_xy)
    sigma_yy_n = _normalise_curve(sigma_yy)
    sigma_yx_n = _normalise_curve(sigma_yx)

    return {
        "eps": eps_vals,
        "log2_eps": np.log2(eps_vals),
        "sigma_xx": sigma_xx_n,
        "sigma_xy": sigma_xy_n,
        "sigma_yy": sigma_yy_n,
        "sigma_yx": sigma_yx_n,
    }


# ---------------------------------------------------------------------------
# From σ(ε) curves to ε_XX, ε_XY, G, S
# ---------------------------------------------------------------------------

def eps_at_level(
    eps: np.ndarray,
    sigma: np.ndarray,
    level: float = 0.9,
) -> float:
    """
    Find ε where σ(ε) hits a given level (e.g. 0.9).

    We mimic the somewhat clunky piecewise logic of the MATLAB code:
    - if the curve never gets above 'level', extrapolate using last two points,
    - otherwise locate the first crossing and interpolate between neighbours.

    Args:
        eps: 1D array of ε values (monotone increasing).
        sigma: 1D array σ(ε) of the same shape.
        level: target level for interpolation (default 0.9).

    Returns:
        Scalar ε where σ ≈ level (float, possibly NaN if curve is degenerate).

    Ding/NC_ASIPP equivalent (lines 233–248):
        if(SUM1(i,EN) < 0.9)
            k1 = EN; k2 = EN-1
        else
            for L = 1 : Estep : EN
                if(SUM1(i,L) > 0.9)
                    k1 = L-1; k2 = L
        ...
        Eps4(i) = EPS(k1) + (EPS(k2)-EPS(k1))*(0.9-SUM1(i,k1))/(SUM1(i,k2)-SUM1(i,k1))

    Note: there was a bug in an earlier Python version where (e2 - s1) was used.
    That is fixed here: we correctly use (e2 - e1).
    """
    eps = np.asarray(eps, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if eps.shape != sigma.shape:
        raise ValueError("eps and sigma must have the same shape.")

    finite = np.isfinite(sigma)
    if not finite.any():
        return np.nan

    e = eps[finite]
    s = sigma[finite]

    # Case 1: last point is still below level -> extrapolate from last two points.
    if s[-1] < level:
        if len(s) < 2:
            # Single point: nothing to interpolate with.
            return e[-1]
        e1, e2 = e[-2], e[-1]
        s1, s2 = s[-2], s[-1]
    else:
        # Find first index where s > level.
        idx = np.argmax(s > level)
        if s[idx] <= level:
            # Never actually crosses; just return that ε.
            return e[idx]
        if idx == 0:
            # Crossing happens between first two points.
            e1, e2 = e[0], e[1]
            s1, s2 = s[0], s[1]
        else:
            e1, e2 = e[idx - 1], e[idx]
            s1, s2 = s[idx - 1], s[idx]

    if s2 == s1:
        # Flat segment -> no interpolation possible.
        return e1

    # Correct linear interpolation in ε:
    return float(e1 + (e2 - e1) * (level - s1) / (s2 - s1))


def nlcc_metrics_from_curves(
    curves: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """
    Compute NLCC metrics from σ(ε) curves.

    Args:
        curves: dict output from conditional_dispersion_curves.

    Returns:
        dict with:
          eps_xx, eps_xy, eps_yy, eps_yx  : ε at σ = 0.9
          G_xy = ε_XY / sqrt(ε_XX ε_YY)
          G_yx = ε_YX / sqrt(ε_XX ε_YY)
          S_xy = (ε_YX - ε_XY) / (ε_XY + ε_YX)

    This matches the NC_ASIPP post-processing (lines 250–258).
    """
    eps = curves["eps"]
    eps_xx = eps_at_level(eps, curves["sigma_xx"], level=0.9)
    eps_xy = eps_at_level(eps, curves["sigma_xy"], level=0.9)
    eps_yy = eps_at_level(eps, curves["sigma_yy"], level=0.9)
    eps_yx = eps_at_level(eps, curves["sigma_yx"], level=0.9)

    # Ding's G_xy and G_yx are normalized by sqrt(ε_XX ε_YY).
    G_xy = np.nan
    G_yx = np.nan
    if np.isfinite(eps_xx) and np.isfinite(eps_yy) and eps_xx > 0 and eps_yy > 0:
        denom = np.sqrt(eps_xx * eps_yy)
        if denom > 0:
            G_xy = eps_xy / denom
            G_yx = eps_yx / denom

    # Asymmetry S_xy.
    S_xy = np.nan
    denom_S = eps_xy + eps_yx
    if np.isfinite(denom_S) and denom_S != 0:
        S_xy = (eps_yx - eps_xy) / denom_S

    return {
        "eps_xx": float(eps_xx),
        "eps_xy": float(eps_xy),
        "eps_yy": float(eps_yy),
        "eps_yx": float(eps_yx),
        "G_xy": float(G_xy) if np.isfinite(G_xy) else np.nan,
        "G_yx": float(G_yx) if np.isfinite(G_yx) else np.nan,
        "S_xy": float(S_xy) if np.isfinite(S_xy) else np.nan,
    }


# ---------------------------------------------------------------------------
# Epsilon grid generation (matches MATLAB exactly)
# ---------------------------------------------------------------------------

def eps_grid_from_inp_matlab(cfg):
    """
    Build the epsilon grid exactly like NC_ASIPP.

    The MATLAB/Fortran code does:
        A = 2^Epsmin
        B = (Epsmax - Epsmin) * log(2) / EN
        EPS(I) = A * exp(B * I),  I = 1:Estep:EN

    This gives:
        log2(EPS) starts at log2_min + (log2_max - log2_min)/EN
        and ends exactly at log2_max, with constant log-spacing.

    Args:
        cfg: dict parsed from the .inp/.par file, with keys
             "log2_eps_min", "log2_eps_max", "eps_N", "eps_step".

    Returns:
        (eps_vals, log2_eps):
          eps_vals: 1D array of ε values.
          log2_eps: 1D array log2(ε_vals).
    """
    log2_min = cfg["log2_eps_min"]
    log2_max = cfg["log2_eps_max"]
    EN       = cfg["eps_N"]
    step     = int(cfg["eps_step"])

    A = 2.0 ** log2_min
    B = (log2_max - log2_min) * np.log(2.0) / EN

    # Note: indices start at 1 (matching MATLAB loop I=1:Estep:EN).
    idx = np.arange(1, EN + 1, step, dtype=float)
    eps_vals = A * np.exp(B * idx)
    log2_eps = np.log2(eps_vals)

    return eps_vals, log2_eps
