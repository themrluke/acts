// This file is part of the ACTS project.
//
// Copyright (C) 2016 CERN for the benefit of the ACTS project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

#include "ActsExamples/DD4hepDetector/OpenDataDetector.hpp"

#include "Acts/Geometry/TrackingGeometry.hpp"
#include "Acts/Material/IMaterialDecorator.hpp"
#include "Acts/Surfaces/Surface.hpp"
#include "ActsPlugins/DD4hep/DD4hepDetectorElement.hpp"
#include "ActsPlugins/DD4hep/OpenDataDetectorBuilder.hpp"
#include "ActsPlugins/Root/TGeoAxes.hpp"

#include <memory>

#include <DD4hep/Detector.h>

namespace ActsExamples {

OpenDataDetector::OpenDataDetector(const Config& cfg,
                                   const Acts::GeometryContext& gctx)
    : DD4hepDetectorBase{cfg}, m_cfg{cfg} {
  ACTS_INFO("OpenDataDetector construct");

  std::unique_ptr<Acts::TrackingGeometry> trackingGeometry;
  switch (m_cfg.constructionMethod) {
    case Config::ConstructionMethod::BarrelEndcap:
      trackingGeometry = ActsPlugins::DD4hep::buildOpenDataDetectorBarrelEndcap(
          dd4hepDetector(), gctx, logger());
      break;
    case Config::ConstructionMethod::DirectLayer:
      trackingGeometry = ActsPlugins::DD4hep::buildOpenDataDetectorDirectLayer(
          dd4hepDetector(), gctx, logger());
      break;
    case Config::ConstructionMethod::DirectLayerGrouped:
      trackingGeometry =
          ActsPlugins::DD4hep::buildOpenDataDetectorDirectLayerGrouped(
              dd4hepDetector(), gctx, logger());
      break;
    case Config::ConstructionMethod::TGeo:
      trackingGeometry =
          ActsPlugins::DD4hep::buildOpenDataDetectorBarrelEndcapViaTGeo(
              *dd4hepDetector().world().placement().ptr(), gctx, logger());
      break;
  }

  // Gen3 read-back: if a material map was supplied, decorate the geometry with
  // it before storing the (const) tracking geometry. The mutable traversal
  // visits every portal/face surface, so the per-layer designated faces are
  // matched by GeometryIdentifier; all other surfaces are a silent no-op.
  if (m_cfg.materialDecorator != nullptr) {
    ACTS_INFO("Applying material map to Gen3 ODD geometry");
    trackingGeometry->apply([&](Acts::Surface& surface) {
      m_cfg.materialDecorator->decorate(surface);
    });
  }

  m_trackingGeometry = std::move(trackingGeometry);
}

auto OpenDataDetector::config() const -> const Config& {
  return m_cfg;
}

std::shared_ptr<ActsPlugins::DD4hepDetectorElement>
OpenDataDetector::defaultDetectorElementFactory(
    const dd4hep::DetElement& element, ActsPlugins::TGeoAxes axes,
    double scale) {
  return std::make_shared<ActsPlugins::DD4hepDetectorElement>(element, axes,
                                                              scale);
}

}  // namespace ActsExamples
