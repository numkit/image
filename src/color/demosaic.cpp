// toolboxes/image/src/color/demosaic.cpp
//
// Bayer mosaic -> RGB via Malvar-He-Cutler 5x5 linear interpolation,
// delegating the pure numerical kernel to keel::image.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/image.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <numkit/image/color/color.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

namespace numkit::image {

namespace {

inline void writeNativeInt(Value &out, keel::usize idx, keel::f64 v, ValueType T)
{
    if (v < 0.0) v = 0.0;
    switch (T) {
        case ValueType::UINT8:
            if (v > 255.0) v = 255.0;
            out.uint8DataMut()[idx] = keel::as<keel::u8>(keel::round(v));
            break;
        case ValueType::UINT16:
            if (v > 65535.0) v = 65535.0;
            out.uint16DataMut()[idx] = keel::as<keel::u16>(keel::round(v));
            break;
        case ValueType::UINT32:
            if (v > 4294967295.0) v = 4294967295.0;
            out.uint32DataMut()[idx] = keel::as<keel::u32>(keel::round(v));
            break;
        default:
            break;
    }
}

keel::image::BayerPattern parseAlignment(keel::StringRef s)
{
    if (s == "rggb") return keel::image::BayerPattern::kRggb;
    if (s == "bggr") return keel::image::BayerPattern::kBggr;
    if (s == "grbg") return keel::image::BayerPattern::kGrbg;
    if (s == "gbrg") return keel::image::BayerPattern::kGbrg;
    throw Error("demosaic: sensorAlignment must be 'rggb', 'bggr', 'grbg', or 'gbrg'",
                0, 0, "demosaic", "", "numkit:demosaic:alignment");
}

} // anonymous

Value demosaic(const Value &I, keel::StringRef sensorAlignment,
               keel::i32 bitsPerSample, keel::Allocator *mr)
{
    (void)bitsPerSample;
    const auto &dims = I.dims();
    if (dims.ndims() < 2)
        throw Error("demosaic: I must be a 2-D matrix",
                    0, 0, "demosaic", "", "numkit:demosaic:rank");
    const keel::usize M = dims.dim(0);
    const keel::usize N = dims.dim(1);
    if (M < 2 || N < 2 || (M % 2) != 0 || (N % 2) != 0)
        throw Error("demosaic: image dims must both be even and >= 2",
                    0, 0, "demosaic", "", "numkit:demosaic:invalidImageSize");
    if (I.numel() != M * N)
        throw Error("demosaic: I must be 2-D",
                    0, 0, "demosaic", "", "numkit:demosaic:shape");

    const ValueType T = I.type();
    if (T != ValueType::UINT8 && T != ValueType::UINT16 && T != ValueType::UINT32)
        throw Error("demosaic: image must be uint8, uint16, or uint32",
                    0, 0, "demosaic", "", "numkit:demosaic:type");

    const keel::image::BayerPattern pat = parseAlignment(sensorAlignment);
    const keel::f64 maxVal = (T == ValueType::UINT8)  ? 255.0
                           : (T == ValueType::UINT16) ? 65535.0
                                                      : 4294967295.0;

    const keel::usize npix = M * N;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> mosaic(npix, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < npix; ++i) mosaic[i] = I.elemAsDouble(i);

    keel::Array<keel::f64> r_ch(npix, 0.0, scratch.allocator());
    keel::Array<keel::f64> g_ch(npix, 0.0, scratch.allocator());
    keel::Array<keel::f64> b_ch(npix, 0.0, scratch.allocator());
    (void)keel::image::demosaic_mhc(mosaic.as_ref(), M, N, pat, maxVal,
                                    r_ch.as_mut_ref(), g_ch.as_mut_ref(), b_ch.as_mut_ref());

    Value RGB = Value::matrix3d(M, N, 3, T, mr);
    for (keel::usize i = 0; i < npix; ++i) {
        writeNativeInt(RGB, i, r_ch[i], T);
        writeNativeInt(RGB, i + npix, g_ch[i], T);
        writeNativeInt(RGB, i + 2u * npix, b_ch[i], T);
    }
    return RGB;
}

} // namespace numkit::image
