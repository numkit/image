// toolboxes/image/src/color/chromadapt.cpp
//
// Chromatic adaptation transform for white-balance correction,
// delegating per-pixel Bradford / Von Kries / Simple math to keel::image.

#include <keel/arena.h>
#include <keel/algo.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/image.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/string.h>
#include <numkit/image/color/color.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>


namespace numkit::image {

namespace {

keel::image::RgbColorSpace parseColorSpace(keel::StringRef s)
{
    if (s.equals_ignore_ascii_case("srgb"))           return keel::image::RgbColorSpace::kSrgb;
    if (s.equals_ignore_ascii_case("adobe-rgb-1998")) return keel::image::RgbColorSpace::kAdobeRgb1998;
    if (s.equals_ignore_ascii_case("prophoto-rgb"))   return keel::image::RgbColorSpace::kProPhotoRgb;
    if (s.equals_ignore_ascii_case("linear-rgb"))     return keel::image::RgbColorSpace::kLinearRgb;
    throw Error("chromadapt: ColorSpace must be 'srgb', 'adobe-rgb-1998', 'prophoto-rgb', or 'linear-rgb'",
                0, 0, "chromadapt", "", "numkit:chromadapt:colorSpace");
}

keel::image::ChromAdaptMethod parseMethod(keel::StringRef s)
{
    if (s.equals_ignore_ascii_case("bradford")) return keel::image::ChromAdaptMethod::kBradford;
    if (s.equals_ignore_ascii_case("vonkries")) return keel::image::ChromAdaptMethod::kVonKries;
    if (s.equals_ignore_ascii_case("simple"))   return keel::image::ChromAdaptMethod::kSimple;
    throw Error("chromadapt: Method must be 'bradford', 'vonkries', or 'simple'",
                0, 0, "chromadapt", "", "numkit:chromadapt:method");
}

Value packRgbValue(keel::ArrayRef<keel::f64> r, keel::ArrayRef<keel::f64> g,
                   keel::ArrayRef<keel::f64> b, keel::usize H, keel::usize W,
                   ValueType outClass, keel::Allocator *mr)
{
    const keel::usize npix = H * W;
    Value out = Value::matrix3d(H, W, 3, outClass, mr);
    const keel::ArrayRef<keel::f64> chs[3] = {r, g, b};
    for (keel::usize ch = 0; ch < 3u; ++ch) {
        const keel::usize off = ch * npix;
        switch (outClass) {
            case ValueType::UINT8: {
                keel::u8 *od = out.uint8DataMut() + off;
                for (keel::usize i = 0; i < npix; ++i) {
                    keel::f64 v = keel::round(chs[ch][i] * 255.0);
                    if (v < 0.0) v = 0.0;
                    if (v > 255.0) v = 255.0;
                    od[i] = keel::as<keel::u8>(v);
                }
                break;
            }
            case ValueType::UINT16: {
                keel::u16 *od = out.uint16DataMut() + off;
                for (keel::usize i = 0; i < npix; ++i) {
                    keel::f64 v = keel::round(chs[ch][i] * 65535.0);
                    if (v < 0.0) v = 0.0;
                    if (v > 65535.0) v = 65535.0;
                    od[i] = keel::as<keel::u16>(v);
                }
                break;
            }
            case ValueType::SINGLE: {
                keel::f32 *od = out.singleDataMut() + off;
                for (keel::usize i = 0; i < npix; ++i) od[i] = keel::as<keel::f32>(chs[ch][i]);
                break;
            }
            default: {
                keel::copy_n(out.doubleDataMut() + off, chs[ch].data(), npix);
                break;
            }
        }
    }
    return out;
}

} // namespace

Value chromadapt(const Value &A, const Value &illuminant_in,
                 keel::StringRef method, keel::StringRef color_space, keel::Allocator *mr)
{
    const auto &dA = A.dims();
    if (!dA.is3D() || dA.pages() != 3)
        throw Error("chromadapt: A must be H × W × 3 RGB",
                    0, 0, "chromadapt", "", "numkit:chromadapt:shape");
    const keel::usize H = dA.rows();
    const keel::usize W = dA.cols();
    if (H == 0 || W == 0)
        throw Error("chromadapt: A must be nonempty",
                    0, 0, "chromadapt", "", "numkit:chromadapt:empty");
    if (illuminant_in.numel() != 3)
        throw Error("chromadapt: illuminant must be a 3-element vector",
                    0, 0, "chromadapt", "", "numkit:chromadapt:illShape");

    const ValueType origClass = A.type();
    if (origClass != ValueType::UINT8 && origClass != ValueType::UINT16 &&
        origClass != ValueType::SINGLE && origClass != ValueType::DOUBLE)
        throw Error("chromadapt: A must be uint8 / uint16 / single / double",
                    0, 0, "chromadapt", "", "numkit:chromadapt:class");

    const auto meth = parseMethod(method);
    const auto cs = parseColorSpace(color_space);

    keel::f64 ill[3]{};
    const ValueType ill_t = illuminant_in.type();
    for (keel::usize k = 0; k < 3u; ++k) {
        keel::f64 v = illuminant_in.elemAsDouble(k);
        if (ill_t == ValueType::UINT8)  v /= 255.0;
        if (ill_t == ValueType::UINT16) v /= 65535.0;
        ill[k] = v;
    }
    if (ill[0] == 0.0 && ill[1] == 0.0 && ill[2] == 0.0)
        throw Error("chromadapt: illuminant cannot be [0 0 0]",
                    0, 0, "chromadapt", "", "numkit:chromadapt:illBlack");

    const keel::usize npix = H * W;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> in_r(npix, 0.0, scratch.allocator());
    keel::Array<keel::f64> in_g(npix, 0.0, scratch.allocator());
    keel::Array<keel::f64> in_b(npix, 0.0, scratch.allocator());
    const keel::f64 scale = (origClass == ValueType::UINT8)  ? (1.0 / 255.0)
                          : (origClass == ValueType::UINT16) ? (1.0 / 65535.0)
                                                             : 1.0;
    for (keel::usize i = 0; i < npix; ++i) {
        in_r[i] = A.elemAsDouble(i) * scale;
        in_g[i] = A.elemAsDouble(npix + i) * scale;
        in_b[i] = A.elemAsDouble(2u * npix + i) * scale;
    }

    keel::Array<keel::f64> out_r(npix, 0.0, scratch.allocator());
    keel::Array<keel::f64> out_g(npix, 0.0, scratch.allocator());
    keel::Array<keel::f64> out_b(npix, 0.0, scratch.allocator());
    (void)keel::image::chromadapt_planar(in_r.as_ref(), in_g.as_ref(), in_b.as_ref(),
                                         {ill[0], ill[1], ill[2]}, meth, cs,
                                         out_r.as_mut_ref(), out_g.as_mut_ref(), out_b.as_mut_ref());
    return packRgbValue(out_r.as_ref(), out_g.as_ref(), out_b.as_ref(), H, W, origClass, mr);
}

} // namespace numkit::image
