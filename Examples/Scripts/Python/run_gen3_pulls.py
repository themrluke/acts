#!/usr/bin/env python3
"""Pull-based validation driver for the Gen3 ODD material map.

Builds the Gen3 ODD *with our material map loaded* and runs truth-seeded Kalman
tracking, writing tracksummary_kf.root whose pull_e*_fit branches (d0, z0, phi,
theta, q/p) are the objective: a well-calibrated fit gives each pull a unit-width
Gaussian, so the score is sum over params of (width - 1)^2.  Summarise a run with
analyse_gen3_pulls.py.

The simulation-truth trap
-------------------------
With --fatras the particles are scattered through the *same* ACTS geometry the
KF then fits with, so the pulls come out ~1 BY CONSTRUCTION.  That is a plumbing
check of the chain, not a measurement of the map.

With --geant4 (the default) the hits come from Geant4 stepping through the full
DD4hep geometry, while the fit uses our map.  Truth and fit then have genuinely
different material, so the pulls say something about the map.  Geant4 runs in the
same process, so no simhit files or geoID translation are needed: the sensitive
surface mapper is built from the Gen3 tracking geometry, and the simhits carry
Gen3 geometry IDs directly.
"""

import argparse
from pathlib import Path

import acts
import acts.examples
from acts.examples.odd import getOpenDataDetector
from truth_tracking_kalman import runTruthTrackingKalman

u = acts.UnitConstants

SRCDIR = Path(__file__).resolve().parent.parent.parent.parent
DATADIR = Path("/home/themrluke/Datasets/acts/gen3_mapping")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fatras",
        dest="geant4",
        action="store_false",
        help="Simulate with Fatras instead of Geant4 (plumbing check only)",
    )
    parser.add_argument(
        "--kill-secondaries",
        action="store_true",
        help="Drop Geant4 secondaries. Measured to save no time on the ODD, so "
        "the default keeps the full physics.",
    )
    parser.add_argument("--events", type=int, default=100)
    parser.add_argument("--particles", type=int, default=10)
    parser.add_argument(
        "--eta", type=float, nargs=2, default=(-3.0, 3.0), metavar=("MIN", "MAX")
    )
    parser.add_argument("--map", type=Path, default=DATADIR / "gen3_mapping_map.json")
    parser.add_argument(
        "--digi",
        type=Path,
        default=SRCDIR / "Examples/Configs/odd-digi-smearing-config-gen3.json",
    )
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    outputDir = args.output or DATADIR / "pulls" / ("geant4" if args.geant4 else "fatras")
    outputDir.mkdir(parents=True, exist_ok=True)

    deco = acts.IMaterialDecorator.fromFile(args.map)
    detector = getOpenDataDetector(gen3=True, materialDecorator=deco)
    field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))

    s = acts.examples.Sequencer(events=args.events, numThreads=1)
    runTruthTrackingKalman(
        trackingGeometry=detector.trackingGeometry(),
        field=field,
        digiConfigFile=args.digi,
        outputDir=outputDir,
        numParticles=args.particles,
        etaRange=tuple(args.eta),
        geant4Detector=detector if args.geant4 else None,
        killSecondaries=args.kill_secondaries,
        s=s,
    ).run()
    print("wrote", outputDir / "tracksummary_kf.root")


if __name__ == "__main__":
    main()
