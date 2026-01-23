# io.py
# Jayneel Parikh (14110606).
# Purpose: Load raw (or lightly converted) A/B files
# into structured dictionaries for use in the NLCC algorithm.
# A.dat/A.txt, B.dat/B.txt, and test_cx.inp -----> A (dict), B (dict), inp (dict)

import numpy as np
from pathlib import Path

def load_signal(filepath: object) -> object:
    """
    Load a single signal file containing:
      - a header with lines like 'Key=Value'
      - then two numeric columns (legacy: printed time, value)
    The printed time column in these files is low-precision and
    often useless (0.00, 0.00, ..., 0.10, ...), so we reconstruct
    the true time axis from the header:
        Frequency = dt  (in seconds)
        Trigger Time = t0 (in seconds, optional)
    Returns a dict with:
        {
          "meta": {...},
          "data": {
              "t": t-array (reconstructed),
              "x": signal values
          }
        }
    """

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    meta = {}
    data_started = False
    x_vals = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Header region: collect meta lines until we hit a numeric block
            if not data_started:
                parts = line.split()
                if len(parts) == 2:
                    # could be the start of the numeric block
                    try:
                        float(parts[0]); float(parts[1])
                        data_started = True
                        # Legacy files: first column is a low-precision time printout.
                        # We ignore it and only keep the second column as the signal.
                        x_vals.append(float(parts[1]))
                        continue
                    except ValueError:
                        pass

                # Still in header
                if "=" in line:
                    k, v = line.split("=", 1)
                    meta[k.strip()] = v.strip()
                continue

            # Numeric lines after header
            parts = line.split()
            if len(parts) == 2:
                try:
                    # Ignore the first (printed) time column, keep only x
                    float(parts[0])  # parsed only to sanity-check
                    x_vals.append(float(parts[1]))
                except ValueError:
                    # skip malformed lines silently
                    continue

    x = np.array(x_vals, dtype=float)
    N = len(x)

    # Reconstruct time using header info if available
    dt = None
    t0 = 0.0

    # Frequency line is like "1.001001001E-6 s"
    freq_str = meta.get("Frequency", None)
    if freq_str is not None:
        try:
            dt = float(freq_str.split()[0])
        except (ValueError, IndexError):
            dt = None

    # Trigger Time line is like "2E-5 s"
    trig_str = meta.get("Trigger Time", None)
    if trig_str is not None:
        try:
            t0 = float(trig_str.split()[0])
        except (ValueError, IndexError):
            t0 = 0.0

    if dt is not None and N > 0:
        t = t0 + dt * np.arange(N, dtype=float)
    else:
        # Fallback: uniform unit spacing if no header info
        t = np.arange(N, dtype=float)

    return {"meta": meta, "data": {"t": t, "x": x}}

def load_pair(nameA="A.txt", nameB="B.txt", folder="data_converted"):
    """
    Load a matched signal pair (A, B) from folder.
    Returns two dicts: (A, B).
    """

    folder = Path(folder)
    pathA = folder / nameA
    pathB = folder / nameB

    A = load_signal(pathA)
    B = load_signal(pathB)

    # Sanity checks
    if len(A["data"]["t"]) != len(B["data"]["t"]):
        print("Warning: A and B length mismatch.")
    elif not np.allclose(A["data"]["t"], B["data"]["t"], rtol=1e-5, atol=1e-8):
        print("Warning: A and B time vectors differ slightly.")

    return A, B

def load_inp(filepath):
    """
    Load a fixed-format .inp file (Ding/NC_ASIPP-style).
    Ignores comments after '!' and blank lines.
    Returns a structured dict with all parameters named.
    """

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    lines = []
    with open(filepath, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.split("!", 1)[0].strip()
            if line:
                lines.append(line)

    if len(lines) < 19:
        raise ValueError(f"Expected ≥19 lines, found {len(lines)}")

    cfg = {
        "n_of_files":   int(lines[0]),
        "input_A":      lines[1],
        "input_B":      lines[2],
        "run_name":     lines[3],
        "N":            int(lines[4]),
        "tau":          int(lines[5]),
        "log2_eps_min": float(lines[6]),
        "log2_eps_max": float(lines[7]),
        "eps_N":        int(lines[8]),
        "eps_step":     float(lines[9]),
        "dim_min":      int(lines[10]),
        "dim_max":      int(lines[11]),
        "dim_step":     int(lines[12]),
        "dM":           int(lines[13]),
        "t_begin":      int(lines[14]),
        "t_end":        int(lines[15]),
        "t_step":       int(lines[16]),
        "cal_A":        float(lines[17]),
        "cal_B":        float(lines[18]),
    }

    return cfg

#TEST SCRIPT BELOW, CAN IGNORE
if __name__ == "__main__":
    A, B = load_pair(folder="../data_converted")
    inp = load_inp("../data_raw/test_cx.inp")
    print(inp)
    print(f"A: {len(A['data']['x'])} samples, meta keys = {list(A['meta'])}")
    print(f"B: {len(B['data']['x'])} samples, meta keys = {list(B['meta'])}")
    print(len(A["data"]["t"]), len(B["data"]["t"]))
