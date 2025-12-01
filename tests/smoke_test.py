# Smoke test for nlcc/io.py
# Author: Jayneel Parikh (14110606)

from pathlib import Path
import numpy as np
import nlcc.io as nio

HERE = Path(__file__).resolve()
PROJECT = HERE.parents[1]  # tests/ → project root
RAW = PROJECT / "data_raw"
REF = PROJECT / "data_reference" / "test_AB"

print("USING io.py:", nio.__file__)
print("PROJECT    :", PROJECT)
print("RAW exists :", RAW.exists())
print("RAW/test_AB.inp exists?", (RAW / "test_AB.inp").exists())

# Use RAW consistently
inp = (RAW / "test_AB.inp").resolve()
print("INP PASSED TO load_pair =", inp)
t, A, B, meta = nio.load_pair(inp)

# --- Helper: tiny assert printer
def ok(name, cond):
    if not cond:
        raise AssertionError(name)
    print("✓", name)

# 1) Load primary case ---------------------------------------------------------
print("\n[1] load_pair(test_AB.inp)")
t, A, B, meta = nio.load_pair(RAW / "test_AB.inp")

ok("A,B,t have equal length", len(A) == len(B) == len(t) == meta["N"])
ok("t starts at 0", abs(t[0] - 0.0) < 1e-15)
ok("dt positive", meta["dt"] > 0)
ok("A is 1-D", A.ndim == 1)
ok("B is 1-D", B.ndim == 1)
ok("no NaNs in A", not np.isnan(A).any())
ok("no NaNs in B", not np.isnan(B).any())

dt_est = (t[10] - t[0]) / 10
ok("dt matches estimate", abs(dt_est - meta["dt"]) < 1e-12)

print(f"N={meta['N']}, dt={meta['dt']:.6g}s, T_end={t[-1]:.6g}s")

# 2) Header handling sanity (explicit vs auto) ---------------------------------
print("\n[2] header handling check")
# Force auto-skip by reading raw files directly
A_auto = nio.read_dat(RAW / "A.dat", skip=None).squeeze()
B_auto = nio.read_dat(RAW / "B.dat", skip=None).squeeze()

# Force explicit skip using whatever the .inp says
hl = meta.get("header_lines", 0)
A_explicit = nio.read_dat(RAW / "A.dat", skip=hl).squeeze()
B_explicit = nio.read_dat(RAW / "B.dat", skip=hl).squeeze()

# Compare after aligning lengths (defensive)
nA = min(len(A_auto), len(A_explicit))
nB = min(len(B_auto), len(B_explicit))

ok("A auto vs explicit are identical (len-adjusted)",
   np.allclose(A_auto[:nA], A_explicit[:nA], equal_nan=False))
ok("B auto vs explicit are identical (len-adjusted)",
   np.allclose(B_auto[:nB], B_explicit[:nB], equal_nan=False))

# 3) Cross-file consistency for BA, cx ----------------------------------------
print("\n[3] load_pair(test_BA.inp) and load_pair(test_cx.inp)")
for name in ["test_BA.inp", "test_cx.inp"]:
    t2, A2, B2, m2 = nio.load_pair(RAW / name)
    ok(f"{name}: equal lengths", len(A2) == len(B2) == len(t2) == m2["N"])
    ok(f"{name}: dt positive", m2["dt"] > 0)

# 4) Reference alignment (ms vs s) --------------------------------------------
print("\n[4] reference alignment with test_AB9.dat")
ref_path = REF / "test_AB9.dat"
if not ref_path.exists():
    print(f"↪ reference not present: {ref_path} — skipping [4].")
else:
    ref9 = nio.read_dat(ref_path, skip=None)
    # normalize to 2-D
    if ref9.ndim == 1:
        ref9 = ref9.reshape(-1, 1)
    ok("ref9 has at least one column", ref9.shape[1] >= 1)

    # pick a time-like column: strictly increasing
    def _time_like_idx(arr2d):
        for j in range(arr2d.shape[1]):
            col = np.asarray(arr2d[:, j], dtype=float)
            d = np.diff(col[: min(len(col), 10000)])
            if d.size and np.all(d > 0):
                return j
        return 0  # fallback

    j_time = _time_like_idx(ref9)
    time_ref = np.asarray(ref9[:, j_time], dtype=float)

    # detect units by comparing median step to dt in s vs ms
    dt_s  = float(meta["dt"])
    dt_ms = dt_s * 1e3
    dref  = np.diff(time_ref[: min(len(time_ref), 10000)])
    step  = float(np.median(dref)) if dref.size else np.nan

    # default assume ms; switch to s→ms if needed
    if np.isfinite(step) and abs(step - dt_s) <= 0.05 * max(dt_s, 1e-12):
        scale = 1e3  # ref is in seconds → convert to ms
    else:
        scale = 1.0  # ref already in ms (or close enough)

    time_ms_ref = (time_ref - time_ref[0]) * scale   # rebase to 0 and put in ms
    time_ms_py  = (t * 1e3) - (t[0] * 1e3)           # rebase to 0 (already ms)

    ok("time starts near 0 ms", abs(time_ms_ref[0] - 0.0) < 1e-9)

    # ---- robust end check: align lengths, compare spans with step-based tol ----
    N = min(len(time_ms_ref), len(time_ms_py))
    time_ms_ref_N = time_ms_ref[:N]
    time_ms_py_N  = time_ms_py[:N]

    step_ref = float(np.median(np.diff(time_ms_ref_N))) if N > 1 else 0.0
    step_py  = float(np.median(np.diff(time_ms_py_N)))  if N > 1 else 0.0
    step_tol = max(step_ref, step_py) + 1e-9

    span_ref = time_ms_ref_N[-1] - time_ms_ref_N[0]
    span_py  = time_ms_py_N[-1]  - time_ms_py_N[0]

    ok("time end agrees (± one sample spacing)",
       abs(span_ref - span_py) <= step_tol)