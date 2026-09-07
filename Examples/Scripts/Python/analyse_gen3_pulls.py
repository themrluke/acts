#!/usr/bin/env python3
"""Summarise the pull distributions written by the truth-seeded Kalman fit.

Reads ``tracksummary_kf.root`` and reports, for each track parameter, the width
of its pull distribution overall and in bins of eta.  A well-calibrated fit
gives unit-width pulls, so the objective is

    score = sum over parameters, sum over eta bins of (width - 1)^2

Run WITHOUT ``Python/Examples/python`` on PYTHONPATH -- that directory contains
a ``uproot.py`` shim which shadows the real uproot.
"""

import argparse
from pathlib import Path

import numpy as np
import uproot

PULLS = {
    "d0": "pull_eLOC0_fit",
    "z0": "pull_eLOC1_fit",
    "phi": "pull_ePHI_fit",
    "theta": "pull_eTHETA_fit",
    "qop": "pull_eQOP_fit",
}


def width(values, clip=5.0):
    """Robust Gaussian width: std of the sample after clipping outliers."""
    values = values[np.isfinite(values)]
    if values.size < 10:
        return np.nan, values.size
    kept = values[np.abs(values) < clip]
    return float(np.std(kept)), int(kept.size)


def summarise(path, eta_bins, eta_max):
    """Return (n_tracks, {param: overall width}, {param: [per-bin widths]})."""
    tree = uproot.open(path)["tracksummary"]
    branches = tree.arrays(list(PULLS.values()) + ["t_eta"], library="np")

    # One entry per event, each holding a variable-length array of tracks.
    eta = np.concatenate([np.asarray(a) for a in branches["t_eta"]])
    data = {
        name: np.concatenate([np.asarray(a) for a in branches[branch]])
        for name, branch in PULLS.items()
    }

    overall = {name: width(values)[0] for name, values in data.items()}

    edges = np.linspace(-eta_max, eta_max, eta_bins + 1)
    binned = {name: [] for name in PULLS}
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (eta >= lo) & (eta < hi)
        for name in PULLS:
            binned[name].append(width(data[name][mask])[0])

    return eta.size, overall, binned, edges


def report(label, path, eta_bins, eta_max):
    n, overall, binned, edges = summarise(path, eta_bins, eta_max)
    print(f"\n{label}  ({path})")
    print(f"  tracks: {n}")
    if n == 0:
        print("  no tracks -- nothing to fit")
        return None

    header = "  " + "eta bin".ljust(16) + "".join(f"{name:>9}" for name in PULLS)
    print(header)
    score = 0.0
    for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:])):
        row = f"  [{lo:+.1f}, {hi:+.1f})".ljust(18)
        for name in PULLS:
            w = binned[name][i]
            row += f"{w:>9.3f}" if np.isfinite(w) else f"{'--':>9}"
            if np.isfinite(w):
                score += (w - 1.0) ** 2
        print(row)
    row = "  overall".ljust(18)
    for name in PULLS:
        row += f"{overall[name]:>9.3f}"
    print(row)
    print(f"  score = sum (width - 1)^2 = {score:.4f}   (lower is better)")
    return score


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="*",
        default=["geant4=/home/themrluke/Datasets/acts/gen3_mapping/pulls/geant4/tracksummary_kf.root"],
        help="One or more LABEL=PATH entries to summarise",
    )
    parser.add_argument("--eta-bins", type=int, default=6)
    parser.add_argument("--eta-max", type=float, default=3.0)
    args = parser.parse_args()

    scores = {}
    for entry in args.inputs:
        label, _, path = entry.partition("=")
        if not path:
            label, path = Path(entry).parent.name, entry
        scores[label] = report(label, Path(path), args.eta_bins, args.eta_max)

    if len(scores) > 1:
        print("\nscores")
        for label, score in scores.items():
            if score is not None:
                print(f"  {label:<10} {score:.4f}")


if __name__ == "__main__":
    main()
