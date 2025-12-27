#!/data/data/com.termux/files/usr/bin/python
"""
QDS Qutrit Competition Lab — Runner v0.1

- Uses qds_qutrit_core_v0.QuditSystem
- Runs one or more labelled strategies
- Computes bias vs uniform for Z-basis outcomes
- Exports summary CSV/JSON for the HTML lab to visualise
"""

import argparse
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np

import qds_qutrit_core_v0 as core


def parse_gate_sequence(seq_str):
    """
    Parse gate sequence like:
      "X0,Z0,H0"
    into:
      [("X", 0), ("Z", 0), ("H", 0)]
    """
    out = []
    if not seq_str:
        return out
    parts = [p.strip() for p in seq_str.split(",") if p.strip()]
    for p in parts:
        # e.g. "X0" or "H2"
        gate = p[0]
        site = int(p[1:])
        out.append((gate, site))
    return out



def get_gate(d: int, gate_name: str) -> np.ndarray:
    """
    Return a single-qudit gate matrix for the given dimension d
    and gate name.

    Supported (so far):
      - I : identity
      - X : cyclic shift
      - Z : phase (roots of unity on the diagonal)
      - H : Hadamard
            * d=2: usual qubit Hadamard
            * d=3: Fourier-like qutrit Hadamard
    """
    g = gate_name.upper()

    # Identity
    if g.startswith("I"):
        return np.eye(d, dtype=complex)

    # Z-like phase gate
    if g.startswith("Z"):
        phases = np.array(
            [np.exp(2j * np.pi * k / d) for k in range(d)],
            dtype=complex,
        )
        return np.diag(phases)

    # X-like cyclic shift
    if g.startswith("X"):
        X = np.zeros((d, d), dtype=complex)
        for i in range(d):
            X[(i + 1) % d, i] = 1.0
        return X

    # Qubit Hadamard (d = 2)
    if d == 2 and g == "H":
        return (1.0 / np.sqrt(2.0)) * np.array(
            [
                [1.0,  1.0],
                [1.0, -1.0],
            ],
            dtype=complex,
        )

    # Qutrit Hadamard / Fourier-like gate (d = 3)
    if d == 3 and g == "H":
        omega = np.exp(2j * np.pi / 3.0)
        H = (1.0 / np.sqrt(3.0)) * np.array(
            [
                [1.0,      1.0,       1.0],
                [1.0,      omega,     omega**2],
                [1.0,      omega**2,  omega],
            ],
            dtype=complex,
        )
        return H

    raise ValueError(f"Unknown gate '{gate_name}' for d={d}")

def run_single_strategy(
    label: str,
    d: int,
    n: int,
    gate_seq,
    shots: int,
    mode: str = "statevector",
    seed: int | None = None,
):
    """
    Run one strategy multiple times and collect Z-basis outcomes.
    """
    if seed is not None:
        np.random.seed(seed)

    counts = Counter()
    for _ in range(shots):
        sys = core.QuditSystem(d=d, n=n, mode=mode)
        for gate_name, site in gate_seq:
            U = get_gate(d, gate_name)
            sys.apply_gate(U, site)
        outcome, _ = sys.measure_z()
        counts[outcome] += 1

    # derive probabilities
    total = sum(counts.values()) or 1
    probs = {k: v / total for k, v in counts.items()}
    # complete with zeros up to full basis
    dim = d**n
    full_probs = {}
    for idx in range(dim):
        s = np.base_repr(idx, base=d).zfill(n)
        full_probs[s] = probs.get(s, 0.0)

    # compute bias vs uniform
    uniform_p = 1.0 / dim
    l1 = sum(abs(p - uniform_p) for p in full_probs.values())
    max_bias = max(abs(p - uniform_p) for p in full_probs.values())
    entropy = -sum(p * math.log(p, 2) for p in full_probs.values() if p > 0)

    return {
        "label": label,
        "d": d,
        "n": n,
        "shots": shots,
        "mode": mode,
        "gate_sequence": [(g, s) for g, s in gate_seq],
        "counts": dict(counts),
        "probs": full_probs,
        "metrics": {
            "l1_bias": l1,
            "max_bias": max_bias,
            "entropy_bits": entropy,
            "uniform_entropy_bits": math.log(dim, 2),
        },
    }


def run_from_args(args):
    # Two modes:
    #  1) Single strategy via CLI args
    #  2) Tournament via JSON config
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        return run_tournament(cfg, args)
    else:
        gate_seq = parse_gate_sequence(args.gate_seq)
        res = run_single_strategy(
            label=args.label,
            d=args.d,
            n=args.n,
            gate_seq=gate_seq,
            shots=args.shots,
            mode=args.mode,
            seed=args.seed,
        )
        return [res]


def run_tournament(cfg, args):
    """
    Config JSON shape:

    {
      "d": 3,
      "n": 1,
      "shots": 2000,
      "mode": "statevector",
      "strategies": [
        {"label": "X-only", "gate_seq": "X0"},
        {"label": "XZ", "gate_seq": "X0,Z0"}
      ]
    }
    """
    d = cfg.get("d", args.d)
    n = cfg.get("n", args.n)
    shots = cfg.get("shots", args.shots)
    mode = cfg.get("mode", args.mode)

    out = []
    for strat in cfg.get("strategies", []):
        label = strat["label"]
        gate_seq = parse_gate_sequence(strat.get("gate_seq", ""))
        res = run_single_strategy(
            label=label,
            d=d,
            n=n,
            gate_seq=gate_seq,
            shots=shots,
            mode=mode,
            seed=args.seed,
        )
        out.append(res)

    return out


def write_outputs(results, args):
    # stdout summary
    print("=== QDS Qutrit Runner v0.1 ===")
    for r in results:
        m = r["metrics"]
        print(
            f"[{r['label']}] d={r['d']} n={r['n']} shots={r['shots']} "
            f"mode={r['mode']} | "
            f"L1 bias={m['l1_bias']:.4f}  max bias={m['max_bias']:.4f} "
            f"entropy={m['entropy_bits']:.3f}/{m['uniform_entropy_bits']:.3f}"
        )

    # optional JSON
    if args.out_json:
        with open(args.out_json, "w", encoding="utf-8") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")
        print(f"Wrote JSONL: {args.out_json}")

    # optional CSV (one row per strategy)
    if args.out_csv:
        import csv

        fieldnames = [
            "label",
            "d",
            "n",
            "shots",
            "mode",
            "l1_bias",
            "max_bias",
            "entropy_bits",
            "uniform_entropy_bits",
        ]
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in results:
                m = r["metrics"]
                w.writerow(
                    {
                        "label": r["label"],
                        "d": r["d"],
                        "n": r["n"],
                        "shots": r["shots"],
                        "mode": r["mode"],
                        "l1_bias": m["l1_bias"],
                        "max_bias": m["max_bias"],
                        "entropy_bits": m["entropy_bits"],
                        "uniform_entropy_bits": m["uniform_entropy_bits"],
                    }
                )
        print(f"Wrote CSV: {args.out_csv}")


def main():
    p = argparse.ArgumentParser(
        description="QDS Qutrit Competition Runner v0.1"
    )
    p.add_argument("--d", type=int, default=3, help="dimension (2=qubit, 3=qutrit)")
    p.add_argument("--n", type=int, default=1, help="number of qudits")
    p.add_argument("--shots", type=int, default=2000, help="shots per strategy")
    p.add_argument("--mode", choices=["statevector", "density"], default="statevector")
    p.add_argument("--gate-seq", default="", help='e.g. "X0,Z0,H0"')
    p.add_argument("--label", default="strategy_v0", help="strategy label")
    p.add_argument("--seed", type=int, default=1234, help="RNG seed")
    p.add_argument(
        "--config",
        help="JSON config for tournament run (overrides gate-seq/label for multiple strategies)",
    )
    p.add_argument("--out-json", help="write JSONL results to this path")
    p.add_argument("--out-csv", help="write CSV summary to this path")

    args = p.parse_args()
    results = run_from_args(args)
    write_outputs(results, args)


if __name__ == "__main__":
    main()
