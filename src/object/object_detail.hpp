// toolboxes/image/src/object/object_detail.hpp
//
// Private (src-only) declaration shared between the engine-free compute in
// object.cpp and its CallContext register half in object_reg.cpp. NOT part of
// the public image API — imgradient_mag is the magnitude-only gradient helper
// the imgradient register wrapper reuses.
//
// Phase 2b compute/register split — see project_layering_refactor memory.
#pragma once

#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/value/value.hpp>

namespace numkit::image {

// Gradient magnitude only (skips the per-pixel atan2 direction). Bit-identical
// to imgradient()'s first output.
Value imgradient_mag(const Value &I, keel::StringRef method,
                     keel::Allocator *mr);

} // namespace numkit::image
