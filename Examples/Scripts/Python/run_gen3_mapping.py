from pathlib import Path
import acts
from odd import getOpenDataDetector
from material_mapping import runMaterialMapping

# Gen3 detector, with the material targets we wired in OpenDataDetectorBuilder.cpp
detector = getOpenDataDetector(gen3=True)
tg = detector.trackingGeometry()

# The target surfaces are extracted from the tracking geometry and passed to the material mapping function
surfaces = tg.extractMaterialSurfaces()
print(f"Mapping onto {len(surfaces)} target surfaces from the Gen3 detector.")

# Dataset layout (organised 2026-08-19):
#   recordings/    Geant4 truth inputs
#   gen3_mapping/  this run's outputs (map + mapped/unmapped tracks)
data = Path("/home/themrluke/Datasets/acts")
recording = data / "recordings" / "odd_geant4_material_andi.root"
out_dir = data / "gen3_mapping"

# Now run it. outputFileBase must be a str (the code does base + "_map").
runMaterialMapping(
    surfaces,
    inputFile=recording,
    outputFileBase=str(out_dir / "gen3_mapping"),   # -> gen3_mapping/gen3_mapping_map.json etc.
).run()