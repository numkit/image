// toolboxes/image/src/texture/texture.cpp
//
// Texture Analysis — graycomatrix + graycoprops delegating to keel::image.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/image.h>
#include <keel/memory.h>
#include <numkit/image/texture/texture.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "texture_detail.hpp"

namespace numkit::image {

Value graycomatrix(const Value &I, keel::i32 numLevels, keel::i32 offR, keel::i32 offC,
                   keel::f64 gLow, keel::f64 gHigh, bool symmetric, keel::Allocator *mr)
{
    if (numLevels < 2)
        throw Error("graycomatrix: NumLevels must be >= 2",
                    0, 0, "graycomatrix", "", "numkit:graycomatrix:badLevels");
    const auto &d = I.dims();
    if (d.is3D())
        throw Error("graycomatrix: 2-D grayscale input only",
                    0, 0, "graycomatrix", "", "numkit:graycomatrix:unsupportedShape");
    if (gLow >= gHigh)
        throw Error("graycomatrix: GrayLimits[1] must exceed GrayLimits[0]",
                    0, 0, "graycomatrix", "", "numkit:graycomatrix:badLimits");

    const keel::usize R = d.rows();
    const keel::usize C = d.cols();
    const keel::usize nl = keel::as<keel::usize>(numLevels);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> img(R * C, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < R * C; ++i) img[i] = I.elemAsDouble(i);

    Value G = Value::matrix(nl, nl, ValueType::DOUBLE, mr);
    (void)keel::image::graycomatrix(img.as_ref(), R, C, numLevels, offR, offC, gLow, gHigh,
                                    symmetric,
                                    keel::MutArrayRef<keel::f64>(G.doubleDataMut(), nl * nl),
                                    scratch.allocator());
    return G;
}

Value graycoprops(const Value &G, keel::Allocator *mr)
{
    const auto &d = G.dims();
    if (d.rows() != d.cols() || d.rows() == 0)
        throw Error("graycoprops: GLCM must be square and non-empty",
                    0, 0, "graycoprops", "", "numkit:graycoprops:badShape");

    const keel::usize L = d.rows();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> gv(L * L, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < L * L; ++i) gv[i] = G.elemAsDouble(i);

    keel::image::GlcmProps props;
    const auto st = keel::image::graycoprops(gv.as_ref(), L, props);
    if (st != keel::image::ImageStatus::kOk)
        throw Error("graycoprops: GLCM has zero total mass",
                    0, 0, "graycoprops", "", "numkit:graycoprops:zeroSum");

    Value out = Value::structure();
    out.field("Contrast")    = Value::scalar(props.contrast, mr);
    out.field("Correlation") = Value::scalar(props.correlation, mr);
    out.field("Energy")      = Value::scalar(props.energy, mr);
    out.field("Homogeneity") = Value::scalar(props.homogeneity, mr);
    return out;
}

} // namespace numkit::image
