// toolboxes/image/src/color/illum.cpp
//
// Illumination estimation (illumwhite / illumgray / illumpca) and perceptual
// color difference (imcolordiff), delegating numerical kernels to keel::image.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/image.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>

#include <numkit/image/color/color.hpp>
#include <numkit/image/type_convert/type_convert.hpp>
#include "illum_detail.hpp"

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <cmath>

namespace numkit::image {

namespace {

void validate_image_and_mask(const Value &A, const Value &mask,
                             keel::usize &H, keel::usize &W,
                             keel::Array<keel::u8> &maskFlat,
                             const char *fn)
{
    if (!A.dims().is3D() || A.dims().pages() != 3)
        throw Error(keel::format("{}: input must be H×W×3", fn),
                    0, 0, fn, "", "numkit:image:shape");
    H = A.dims().rows();
    W = A.dims().cols();
    if (mask.isEmpty()) {
        maskFlat.clear();
        return;
    }
    const auto &md = mask.dims();
    if (md.rows() != H || md.cols() != W || md.pages() > 1)
        throw Error(keel::format("{}: Mask must be H×W matching the image", fn),
                    0, 0, fn, "", "numkit:image:maskShape");
    maskFlat.resize(H * W);
    for (keel::usize k = 0; k < H * W; ++k)
        maskFlat[k] = (mask.elemAsDouble(k) != 0.0) ? 1 : 0;
}

void unpack_rgb_planes(const Value &A, keel::usize npix,
                       keel::Array<keel::f64> &r,
                       keel::Array<keel::f64> &g,
                       keel::Array<keel::f64> &b)
{
    r.resize(npix);
    g.resize(npix);
    b.resize(npix);
    for (keel::usize k = 0; k < npix; ++k) {
        r[k] = A.elemAsDouble(0 * npix + k);
        g[k] = A.elemAsDouble(1 * npix + k);
        b[k] = A.elemAsDouble(2 * npix + k);
    }
}

keel::usize count_selected(keel::usize npix, keel::ArrayRef<keel::u8> mask)
{
    if (mask.empty()) return npix;
    keel::usize cnt = 0;
    for (keel::u8 v : mask) if (v != 0) ++cnt;
    return cnt;
}

Value make_row3(keel::image::Color3 c, keel::Allocator *mr)
{
    Value out = Value::matrix(1, 3, ValueType::DOUBLE, mr);
    out.doubleDataMut()[0] = c.c0;
    out.doubleDataMut()[1] = c.c1;
    out.doubleDataMut()[2] = c.c2;
    return out;
}

} // anonymous

Value illumwhite(const Value &A, keel::f64 P, const Value &mask,
                 keel::Allocator *mr)
{
    if (P < 0.0 || P >= 100.0)
        throw Error("illumwhite: percentile must satisfy 0 <= P < 100",
                    0, 0, "illumwhite", "", "numkit:illumwhite:percentile");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::usize H = 0, W = 0;
    keel::Array<keel::u8> maskFlat(scratch.allocator());
    validate_image_and_mask(A, mask, H, W, maskFlat, "illumwhite");
    if (H == 0 || W == 0)
        throw Error("illumwhite: image is empty",
                    0, 0, "illumwhite", "", "numkit:illumwhite:empty");
    const keel::usize npix = H * W;
    if (count_selected(npix, maskFlat.as_ref()) == 0)
        throw Error("illumwhite: no pixels selected by Mask",
                    0, 0, "illumwhite", "", "numkit:illumwhite:emptyMask");

    keel::Array<keel::f64> r(scratch.allocator()), g(scratch.allocator()), b(scratch.allocator());
    unpack_rgb_planes(A, npix, r, g, b);
    keel::image::Color3 res{};
    (void)keel::image::illumwhite_planar(r.as_ref(), g.as_ref(), b.as_ref(),
                                         maskFlat.as_ref(), P, res, scratch.allocator());
    return make_row3(res, mr);
}

Value illumgray_impl(const Value &A, keel::ArrayRef<keel::f64> P,
                     const Value &mask, keel::f64 norm_exp,
                     keel::Allocator *mr)
{
    keel::f64 p_lo = 1.0, p_hi = 1.0;
    if (P.size() == 1) {
        p_lo = P[0]; p_hi = P[0];
    } else if (P.size() == 2) {
        p_lo = P[0]; p_hi = P[1];
    } else if (!P.empty()) {
        throw Error("illumgray: percentile must be scalar or 2-vector",
                    0, 0, "illumgray", "", "numkit:illumgray:percentile");
    }
    if (p_lo < 0.0 || p_lo >= 100.0 || p_hi < 0.0 || p_hi >= 100.0
        || p_lo + p_hi > 100.0)
        throw Error("illumgray: percentiles must satisfy 0 <= P < 100 and "
                    "p_lo + p_hi <= 100",
                    0, 0, "illumgray", "", "numkit:illumgray:percentile");
    if (!(norm_exp > 0.0))
        throw Error("illumgray: Norm must be a positive scalar",
                    0, 0, "illumgray", "", "numkit:illumgray:norm");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::usize H = 0, W = 0;
    keel::Array<keel::u8> maskFlat(scratch.allocator());
    validate_image_and_mask(A, mask, H, W, maskFlat, "illumgray");
    if (H == 0 || W == 0)
        throw Error("illumgray: image is empty",
                    0, 0, "illumgray", "", "numkit:illumgray:empty");
    const keel::usize npix = H * W;
    const keel::usize nsel = count_selected(npix, maskFlat.as_ref());
    if (nsel == 0)
        throw Error("illumgray: no pixels selected by Mask",
                    0, 0, "illumgray", "", "numkit:illumgray:emptyMask");
    const keel::usize k_lo = keel::as<keel::usize>(std::floor(p_lo * 0.01 * keel::as<keel::f64>(nsel)));
    const keel::usize k_hi = keel::as<keel::usize>(std::floor(p_hi * 0.01 * keel::as<keel::f64>(nsel)));
    if (k_lo + k_hi >= nsel)
        throw Error("illumgray: percentiles trim all pixels",
                    0, 0, "illumgray", "", "numkit:illumgray:emptyTrim");

    keel::Array<keel::f64> r(scratch.allocator()), g(scratch.allocator()), b(scratch.allocator());
    unpack_rgb_planes(A, npix, r, g, b);
    keel::image::Color3 res{};
    (void)keel::image::illumgray_planar(r.as_ref(), g.as_ref(), b.as_ref(),
                                        maskFlat.as_ref(), p_lo, p_hi, norm_exp,
                                        res, scratch.allocator());
    return make_row3(res, mr);
}

Value illumgray(const Value &A, keel::ArrayRef<keel::f64> P,
                const Value &mask, keel::Allocator *mr)
{
    return illumgray_impl(A, P, mask, 1.0, mr);
}

Value illumpca(const Value &A, keel::f64 P, const Value &mask,
               keel::Allocator *mr)
{
    if (!(P > 0.0) || P > 50.0)
        throw Error("illumpca: percentage must satisfy 0 < P <= 50",
                    0, 0, "illumpca", "", "numkit:illumpca:percentage");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::usize H = 0, W = 0;
    keel::Array<keel::u8> maskFlat(scratch.allocator());
    validate_image_and_mask(A, mask, H, W, maskFlat, "illumpca");
    if (H == 0 || W == 0)
        throw Error("illumpca: image is empty",
                    0, 0, "illumpca", "", "numkit:illumpca:empty");
    const keel::usize npix = H * W;
    if (count_selected(npix, maskFlat.as_ref()) == 0)
        throw Error("illumpca: no pixels selected by Mask",
                    0, 0, "illumpca", "", "numkit:illumpca:emptyMask");

    keel::Array<keel::f64> r(scratch.allocator()), g(scratch.allocator()), b(scratch.allocator());
    unpack_rgb_planes(A, npix, r, g, b);
    keel::image::Color3 res{};
    const auto st = keel::image::illumpca_planar(r.as_ref(), g.as_ref(), b.as_ref(),
                                                 maskFlat.as_ref(), P, res, scratch.allocator());
    if (st != keel::image::ImageStatus::kOk)
        throw Error("illumpca: image contains Inf or NaN",
                    0, 0, "illumpca", "", "numkit:illumpca:nonfinite");
    return make_row3(res, mr);
}

Value imcolordiff(const Value &I1, const Value &I2,
                  keel::StringRef standard, bool is_input_lab,
                  keel::f64 kL, keel::f64 kC, keel::f64 kH, keel::f64 K1, keel::f64 K2,
                  keel::Allocator *mr)
{
    const bool is_2000 = standard.equals_ignore_ascii_case("ciede2000") ||
                         standard.equals_ignore_ascii_case("cie2000");
    if (!is_2000 && !standard.equals_ignore_ascii_case("cie94"))
        throw Error("imcolordiff: Standard must be 'CIE94' or 'CIEDE2000'",
                    0, 0, "imcolordiff", "", "numkit:imcolordiff:standard");
    if (!(kL > 0.0) || !(kC > 0.0) || !(kH > 0.0)
        || !(K1 > 0.0) || !(K2 > 0.0))
        throw Error("imcolordiff: kL, kC, kH, K1, K2 must all be positive",
                    0, 0, "imcolordiff", "", "numkit:imcolordiff:weights");

    const auto &d1 = I1.dims();
    const auto &d2 = I2.dims();
    keel::usize out_h = 0, out_w = 0;
    if (!d1.is3D() && !d2.is3D()) {
        if (d1.cols() != 3 || d2.cols() != 3 || d1.rows() != d2.rows())
            throw Error("imcolordiff: c-by-3 inputs must have matching rows",
                        0, 0, "imcolordiff", "", "numkit:imcolordiff:size");
        out_h = d1.rows();
        out_w = 1;
    } else if (d1.is3D() && d2.is3D()) {
        if (d1.pages() != 3 || d2.pages() != 3
            || d1.rows() != d2.rows() || d1.cols() != d2.cols())
            throw Error("imcolordiff: H-by-W-by-3 inputs must match in size",
                        0, 0, "imcolordiff", "", "numkit:imcolordiff:size");
        out_h = d1.rows();
        out_w = d1.cols();
    } else {
        throw Error("imcolordiff: I1 and I2 must both be c-by-3 or H-by-W-by-3",
                    0, 0, "imcolordiff", "", "numkit:imcolordiff:size");
    }

    const bool to_double = (I1.type() == ValueType::DOUBLE)
                        || (I2.type() == ValueType::DOUBLE);
    Value A = to_double ? im2double(I1, mr) : im2single(I1, mr);
    Value B = to_double ? im2double(I2, mr) : im2single(I2, mr);
    if (!is_input_lab) {
        A = rgb2lab(A, mr);
        B = rgb2lab(B, mr);
    }

    const keel::usize N = out_h * out_w;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> l1(scratch.allocator()), a1(scratch.allocator()), b1(scratch.allocator());
    keel::Array<keel::f64> l2(scratch.allocator()), a2(scratch.allocator()), b2(scratch.allocator());
    unpack_rgb_planes(A, N, l1, a1, b1);
    unpack_rgb_planes(B, N, l2, a2, b2);

    keel::Array<keel::f64> diff(N, 0.0, scratch.allocator());
    const auto std_enum = is_2000 ? keel::image::ColorDiffStandard::kCiede2000
                                  : keel::image::ColorDiffStandard::kCie94;
    (void)keel::image::colordiff_lab_planar(l1.as_ref(), a1.as_ref(), b1.as_ref(),
                                            l2.as_ref(), a2.as_ref(), b2.as_ref(),
                                            std_enum, kL, kC, kH, K1, K2, diff.as_mut_ref());

    Value out = Value::matrix(out_h, out_w,
                              to_double ? ValueType::DOUBLE : ValueType::SINGLE,
                              mr);
    if (to_double) {
        keel::f64 *od = out.doubleDataMut();
        for (keel::usize k = 0; k < N; ++k) od[k] = diff[k];
    } else {
        keel::f32 *od = out.singleDataMut();
        for (keel::usize k = 0; k < N; ++k) od[k] = keel::as<keel::f32>(diff[k]);
    }
    return out;
}

} // namespace numkit::image
