#!/usr/bin/env python3
"""Derive a Gen3-keyed digitisation config from the Gen1 one.

The ODD digitisation configs are keyed by volume geometry-ID.  Gen1 groups all
sensitive surfaces into 9 volumes; Gen3 gives every layer its own volume (48 of
them), so the Gen1 config matches almost nothing on Gen3 and tracks end up with
too few measurements to be reconstructed.

This script translates the config using the Gen1<->Gen3 surface mapping written
by ``generate_geoid_map.py``: every Gen1 volume in the input config is expanded
into the set of Gen3 volumes its sensitive surfaces map onto, each carrying the
same smearing values.

Usage
-----
    python3 Examples/Scripts/Python/make_gen3_digi_config.py \
        --geoid-map geoid_map_gen1_gen3.csv \
        --input Examples/Configs/odd-digi-smearing-config.json \
        --output Examples/Configs/odd-digi-smearing-config-gen3.json
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def load_volume_map(csv_path, prefix_a="gen1", prefix_b="gen3"):
    """Return {gen1_volume: sorted[gen3_volumes]} from the geoID map CSV."""
    volume_map = defaultdict(set)
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            volume_map[int(row[f"{prefix_a}_volume"])].add(
                int(row[f"{prefix_b}_volume"])
            )
    return {a: sorted(b) for a, b in sorted(volume_map.items())}


def translate(config, volume_map):
    """Expand each volume-keyed entry into its Gen3 volumes."""
    entries = []
    unmapped = []

    for entry in config["entries"]:
        if set(entry) - {"volume", "value"}:
            raise RuntimeError(
                f"Entry is keyed on more than a volume, cannot translate: "
                f"{sorted(set(entry) - {'value'})}"
            )
        volume = entry["volume"]
        targets = volume_map.get(volume)
        if not targets:
            unmapped.append(volume)
            continue
        for target in targets:
            entries.append({"volume": target, "value": entry["value"]})

    if unmapped:
        raise RuntimeError(
            f"No Gen3 volumes found for Gen1 volume(s) {unmapped}. "
            f"The geoID map covers {sorted(volume_map)}."
        )

    entries.sort(key=lambda e: e["volume"])
    return {**config, "entries": entries}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geoid-map", type=Path, required=True)
    parser.add_argument(
        "--input", type=Path, default=Path("Examples/Configs/odd-digi-smearing-config.json")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Examples/Configs/odd-digi-smearing-config-gen3.json"),
    )
    parser.add_argument("--prefix-a", default="gen1")
    parser.add_argument("--prefix-b", default="gen3")
    args = parser.parse_args()

    volume_map = load_volume_map(args.geoid_map, args.prefix_a, args.prefix_b)
    config = json.loads(args.input.read_text())
    translated = translate(config, volume_map)

    args.output.write_text(json.dumps(translated, indent=4) + "\n")

    for source, targets in volume_map.items():
        print(f"  volume {source:>3} -> {targets}")
    print(
        f"Written {len(translated['entries'])} entries "
        f"(from {len(config['entries'])}) to {args.output}"
    )


if __name__ == "__main__":
    main()
