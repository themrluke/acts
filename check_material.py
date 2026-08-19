from odd import getOpenDataDetector

detector = getOpenDataDetector(gen3=True)
tg = detector.trackingGeometry()
print("material surfaces:", len(tg.extractMaterialSurfaces()))
