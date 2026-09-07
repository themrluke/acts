#!/usr/bin/env python3

from pathlib import Path
from typing import Optional

import acts
import acts.examples

u = acts.UnitConstants


def runTruthTrackingKalman(
    trackingGeometry: acts.TrackingGeometry,
    field: acts.MagneticFieldProvider,
    digiConfigFile: Path,
    outputDir: Path,
    inputParticlePath: Optional[Path] = None,
    inputHitsPath: Optional[Path] = None,
    decorators=[],
    generatedParticleType: acts.PdgParticle = acts.PdgParticle.eMuon,
    reverseFilteringMomThreshold=0 * u.GeV,
    reverseFilteringCovarianceScaling=100.0,
    numParticles=1,
    etaRange=(-3.0, 3.0),
    geant4Detector=None,
    killSecondaries=False,
    linkForward: bool = False,
    useJosephFormulation: bool = False,
    s: acts.examples.Sequencer = None,
):
    from acts.examples.simulation import (
        addParticleGun,
        ParticleConfig,
        EtaConfig,
        PhiConfig,
        MomentumConfig,
        addFatras,
        addGeant4,
        addDigitization,
        ParticleSelectorConfig,
        addDigiParticleSelection,
    )

    from acts.examples.root import (
        RootParticleReader,
        RootSimHitReader,
        RootTrackStatesWriter,
        RootTrackSummaryWriter,
        RootTrackParameterPerformanceWriter,
    )

    from acts.examples.reconstruction import (
        addSeeding,
        SeedingAlgorithm,
        TrackSmearingSigmas,
        addKalmanTracks,
    )

    s = s or acts.examples.Sequencer(
        events=100, numThreads=-1, logLevel=acts.logging.INFO
    )

    for d in decorators:
        s.addContextDecorator(d)

    rnd = acts.examples.RandomNumbers(seed=42)
    outputDir = Path(outputDir)

    logger = acts.getDefaultLogger("Truth tracking example", acts.logging.INFO)

    if inputParticlePath is None:
        addParticleGun(
            s,
            ParticleConfig(
                num=numParticles, pdg=generatedParticleType, randomizeCharge=True
            ),
            EtaConfig(etaRange[0], etaRange[1], uniform=True),
            MomentumConfig(1.0 * u.GeV, 100.0 * u.GeV, transverse=True),
            PhiConfig(0.0, 360.0 * u.degree),
            vtxGen=acts.examples.GaussianVertexGenerator(
                mean=acts.Vector4(0, 0, 0, 0),
                stddev=acts.Vector4(0, 0, 0, 0),
            ),
            multiplicity=1,
            rnd=rnd,
        )
    else:
        logger.info("Reading particles from {}", inputParticlePath.resolve())
        assert inputParticlePath.exists()
        s.addReader(
            RootParticleReader(
                level=acts.logging.INFO,
                filePath=str(inputParticlePath.resolve()),
                outputParticles="particles_generated",
            )
        )
        s.addWhiteboardAlias("particles", "particles_generated")

    if inputHitsPath is None and geant4Detector is not None:
        # Geant4 simulates through its own (full) geometry, so the scattering
        # truth is independent of the material the fitter later uses.
        if s.config.numThreads != 1:
            raise ValueError("Geant 4 simulation does not support multi-threading")
        addGeant4(
            s,
            geant4Detector,
            trackingGeometry,
            field,
            rnd=rnd,
            killVolume=trackingGeometry.highestTrackingVolume,
            killAfterTime=25 * u.ns,
            killSecondaries=killSecondaries,
        )
    elif inputHitsPath is None:
        addFatras(
            s,
            trackingGeometry,
            field,
            rnd=rnd,
            enableInteractions=True,
        )
    else:
        logger.info("Reading hits from {}", inputHitsPath.resolve())
        assert inputHitsPath.exists()
        s.addReader(
            RootSimHitReader(
                level=acts.logging.INFO,
                filePath=str(inputHitsPath.resolve()),
                outputSimHits="simhits",
            )
        )
        s.addWhiteboardAlias("particles_simulated_selected", "particles_generated")

    addDigitization(
        s,
        trackingGeometry,
        field,
        digiConfigFile=digiConfigFile,
        rnd=rnd,
    )

    addDigiParticleSelection(
        s,
        ParticleSelectorConfig(
            pt=(0.9 * u.GeV, None),
            measurements=(7, None),
            removeNeutral=True,
            removeSecondaries=True,
        ),
    )

    addSeeding(
        s,
        trackingGeometry,
        field,
        rnd=rnd,
        inputParticles="particles_generated",
        seedingAlgorithm=SeedingAlgorithm.TruthSmeared,
        trackSmearingSigmas=TrackSmearingSigmas(
            # zero everything so the KF has a chance to find the measurements
            loc0=0,
            loc0PtA=0,
            loc0PtB=0,
            loc1=0,
            loc1PtA=0,
            loc1PtB=0,
            time=0,
            phi=0,
            theta=0,
            ptRel=0,
        ),
        particleHypothesis=acts.ParticleHypothesis.muon,
        initialSigmas=[
            1 * u.mm,
            1 * u.mm,
            1 * u.degree,
            1 * u.degree,
            0 / u.GeV,
            1 * u.ns,
        ],
        initialSigmaQoverPt=0.1 / u.GeV,
        initialSigmaPtRel=0.1,
        initialVarInflation=[1e0, 1e0, 1e0, 1e0, 1e0, 1e0],
    )

    addKalmanTracks(
        s,
        trackingGeometry,
        field,
        reverseFilteringMomThreshold,
        reverseFilteringCovarianceScaling,
        linkForward=linkForward,
        useJosephFormulation=useJosephFormulation,
    )

    s.addAlgorithm(
        acts.examples.TrackSelectorAlgorithm(
            level=acts.logging.INFO,
            inputTracks="tracks",
            outputTracks="selected-tracks",
            selectorConfig=acts.TrackSelector.Config(
                minMeasurements=7,
            ),
        )
    )
    s.addWhiteboardAlias("tracks", "selected-tracks")

    s.addWriter(
        RootTrackStatesWriter(
            level=acts.logging.INFO,
            inputTracks="tracks",
            inputParticles="particles_selected",
            inputTrackParticleMatching="track_particle_matching",
            inputSimHits="simhits",
            inputMeasurementSimHitsMap="measurement_simhits_map",
            filePath=str(outputDir / "trackstates_kf.root"),
        )
    )

    s.addWriter(
        RootTrackSummaryWriter(
            level=acts.logging.INFO,
            inputTracks="tracks",
            inputParticles="particles_selected",
            inputTrackParticleMatching="track_particle_matching",
            filePath=str(outputDir / "tracksummary_kf.root"),
        )
    )

    s.addWriter(
        RootTrackParameterPerformanceWriter(
            level=acts.logging.INFO,
            inputTracks="tracks",
            inputParticles="particles_selected",
            inputTrackParticleMatching="track_particle_matching",
            filePath=str(outputDir / "performance_kf.root"),
        )
    )

    return s


if "__main__" == __name__:
    srcdir = Path(__file__).resolve().parent.parent.parent.parent

    # ODD
    from acts.examples.odd import getOpenDataDetector

    detector = getOpenDataDetector()
    trackingGeometry = detector.trackingGeometry()
    digiConfigFile = srcdir / "Examples/Configs/odd-digi-smearing-config.json"

    ## GenericDetector
    # detector = acts.examples.GenericDetector()
    # trackingGeometry = detector.trackingGeometry()
    # digiConfigFile = (
    #     srcdir
    #     / "Examples/Configs/generic-digi-smearing-config.json"
    # )

    field = acts.ConstantBField(acts.Vector3(0, 0, 2 * u.T))

    runTruthTrackingKalman(
        trackingGeometry=trackingGeometry,
        field=field,
        digiConfigFile=digiConfigFile,
        outputDir=Path.cwd(),
    ).run()
