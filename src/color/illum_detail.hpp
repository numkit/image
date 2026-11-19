// toolboxes/image/src/color/illum_detail.hpp
//
// Private (src-only) declaration shared between the engine-free compute in
// illum.cpp and its CallContext register half in illum_reg.cpp. NOT part of
// the public image API: illumgray_impl exposes the internal Minkowski-norm
// exponent (norm_exp) that the public illumgray() deliberately hardcodes to
// 1.0, so the 'Norm' name-value pair can only be honoured from the register
// wrapper.
//
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/value/value.hpp>

namespace numkit::image {

// Grey-world illuminant estimate with an explicit Minkowski norm exponent.
// illumgray() forwards here with norm_exp = 1.0.
Value illumgray_impl(const Value &A, keel::ArrayRef<double> P,
                     const Value &mask, double norm_exp,
                     keel::Allocator *mr);

} // namespace numkit::image
