// toolboxes/image/src/color/color_extras.cpp
//
// Image Toolbox color-conversion extras (cycle 3 of the sweep):
//
//   rgb2lightness(RGB)             RGB → L (lightness, L* of CIE Lab).
//                                  Returns H×W single image.
//   [X, cmap] = rgb2ind(RGB, inmap) Quantize to fixed colormap by
//                                  nearest-neighbour search in RGB.
//                                  X is uint8/uint16 (1-based index;
//                                  uint8 if cmap rows ≤ 256, otherwise
//                                  uint16). cmap echoed unchanged.
//
// PMR HARD RULE: every fn takes std::pmr::memory_resource *mr.
//
// KNOWN GAPs (rgb2ind):
//   * `rgb2ind(RGB, Q)` minimum-variance quantization (median-cut style)
//     — deferred (~150 LOC algorithm work).
//   * `rgb2ind(RGB, tol)` uniform-grid quantization — deferred.
//   * `dithering` arg ('dither' | 'nodither') — deferred; always
//     equivalent to 'nodither' (no error diffusion).

#include <keel/core.h>
#include <keel/algo.h>
#include <keel/math.h>
#include <keel/format.h>
#include <keel/image.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/image/color/color.hpp>
#include <numkit/image/internal/type_gates.hpp>
#include <numkit/image/type_convert/type_convert.hpp>
#include <numkit/image/geom/geom.hpp>
#include <numkit/image/contrast/contrast.hpp>

#include <numkit/ops/matlab_mt19937.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <keel/static_array.h>

namespace numkit::image {

namespace {

void rgb2lightness_argcheck(const Value &RGB)
{
    const ValueType t0 = RGB.type();
    if (!(t0 == ValueType::DOUBLE || t0 == ValueType::SINGLE
          || t0 == ValueType::UINT8))
        internal::requireImageClass(RGB, "rgb2lightness", "RGB", 1,
                                    internal::ImgClasses::FloatOnly);
    if (!RGB.dims().is3D() || RGB.dims().pages() != 3)
        throw Error("rgb2lightness: input must be H×W×3",
                    0, 0, "rgb2lightness", "", "numkit:rgb2lightness:Shape");
}

void rgb2ind_inmap_argcheck(const Value &RGB, const Value &cmap)
{
    if (!RGB.dims().is3D() || RGB.dims().pages() != 3)
        throw Error("rgb2ind: input must be H×W×3",
                    0, 0, "rgb2ind", "", "numkit:rgb2ind:Shape");
    if (cmap.dims().is3D() || cmap.dims().cols() != 3)
        throw Error("rgb2ind: colormap must be K×3",
                    0, 0, "rgb2ind", "", "numkit:rgb2ind:CmapShape");
}

void rgbwide2ycbcr_argcheck(const Value &RGB, keel::i32 bits_per_sample)
{
    if (bits_per_sample != 10 && bits_per_sample != 12)
        throw Error("rgbwide2ycbcr: BPS must be 10 or 12",
                    0, 0, "rgbwide2ycbcr", "", "numkit:rgbwide2ycbcr:bps");
    if (RGB.type() != ValueType::UINT16)
        throw Error("rgbwide2ycbcr: RGB must be UINT16",
                    0, 0, "rgbwide2ycbcr", "", "numkit:rgbwide2ycbcr:class");

    const auto &d = RGB.dims();
    if (d.is3D()) {
        if (d.pages() != 3)
            throw Error("rgbwide2ycbcr: H×W×3 image expected",
                        0, 0, "rgbwide2ycbcr", "", "numkit:rgbwide2ycbcr:shape");
    } else {
        if (d.cols() != 3)
            throw Error("rgbwide2ycbcr: p×3 colour list expected",
                        0, 0, "rgbwide2ycbcr", "", "numkit:rgbwide2ycbcr:shape");
    }
}

void cmunique_xm_argcheck(const Value &MAP)
{
    if (MAP.dims().cols() != 3 || MAP.dims().is3D())
        throw Error("cmunique: MAP must be N×3",
                    0, 0, "cmunique", "", "numkit:cmunique:map");
}

void cmunique_rgb_argcheck(const Value &RGB)
{
    if (!RGB.dims().is3D() || RGB.dims().pages() != 3)
        throw Error("cmunique: RGB must be H×W×3",
                    0, 0, "cmunique", "", "numkit:cmunique:rgb");
}

void cmunique_i_argcheck(const Value &I)
{
    if (I.dims().ndims() > 2)
        throw Error("cmunique: I must be a 2-D intensity image",
                    0, 0, "cmunique", "", "numkit:cmunique:i");
}

void ycbcr2rgbwide_argcheck(const Value &YCBCR, keel::i32 bits_per_sample)
{
    if (bits_per_sample != 10 && bits_per_sample != 12)
        throw Error("ycbcr2rgbwide: BPS must be 10 or 12",
                    0, 0, "ycbcr2rgbwide", "", "numkit:ycbcr2rgbwide:bps");
    if (YCBCR.type() != ValueType::UINT16)
        throw Error("ycbcr2rgbwide: YCBCR must be UINT16",
                    0, 0, "ycbcr2rgbwide", "", "numkit:ycbcr2rgbwide:class");

    const auto &d = YCBCR.dims();
    if (d.is3D()) {
        if (d.pages() != 3)
            throw Error("ycbcr2rgbwide: H×W×3 image expected",
                        0, 0, "ycbcr2rgbwide", "", "numkit:ycbcr2rgbwide:shape");
    } else {
        if (d.cols() != 3)
            throw Error("ycbcr2rgbwide: p×3 colour list expected",
                        0, 0, "ycbcr2rgbwide", "", "numkit:ycbcr2rgbwide:shape");
    }
}

} // anonymous

// ── rgb2lightness ─────────────────────────────────────────────────────
// L = first channel of rgb2lab(RGB). MATLAB's rgb2lightness returns
// single. NK's rgb2lab returns double, so we cast on the way out.
Value rgb2lightness(const Value &RGB, keel::Allocator *mr)
{
    rgb2lightness_argcheck(RGB);
    // Probed R2025b: double / single / uint8 accepted (int16 and
    // complex rejected; uint8 returns a single result).
    {
        const ValueType t0 = RGB.type();
        if (!(t0 == ValueType::DOUBLE || t0 == ValueType::SINGLE
              || t0 == ValueType::UINT8))
            internal::requireImageClass(RGB, "rgb2lightness", "RGB", 1,
                                        internal::ImgClasses::FloatOnly);
    }
    if (!RGB.dims().is3D() || RGB.dims().pages() != 3)
        throw Error("rgb2lightness: input must be H×W×3",
                    0, 0, "rgb2lightness", "", "numkit:rgb2lightness:Shape");
    Value lab = rgb2lab(RGB, mr);
    const keel::usize H = lab.dims().rows();
    const keel::usize W = lab.dims().cols();
    Value out = Value::matrix(H, W, ValueType::SINGLE, mr);
    if (H == 0 || W == 0) return out;
    // Page 0 of column-major H×W×3 = first H*W elements.
    keel::f32 *dst = out.singleDataMut();
    if (lab.type() == ValueType::SINGLE) {
        keel::copy_n(dst, lab.singleData(), H * W);
    } else if (lab.type() == ValueType::DOUBLE) {
        const keel::f64 *src = lab.doubleData();
        for (keel::usize i = 0; i < H * W; ++i) dst[i] = keel::as<keel::f32>(src[i]);
    } else {
        // Fallback via elemAsDouble for any other lab type.
        for (keel::usize i = 0; i < H * W; ++i)
            dst[i] = keel::as<keel::f32>(lab.elemAsDouble(i));
    }
    // Probed R2025b: double input → double result (single / uint8 stay
    // single).
    if (RGB.type() == ValueType::DOUBLE)
        return internal::castResultToClass(out, ValueType::DOUBLE, mr);
    return out;
}

// ── rgb2ind ───────────────────────────────────────────────────────────
// Fixed-palette form only in v1: nearest-RGB quantization (no dithering).
//   X(i, j) = argmin_k ||RGB(i, j, :) - cmap(k, :)||^2   (0-based!)
// Output index class: uint8 if cmap rows ≤ 256, else uint16. MATLAB
// returns 0-based indices for integer output classes (1-based only when
// the output is double, which our v1 doesn't produce). Tie-breaking
// picks the lowest cmap index (matches MATLAB's nodither path).
keel::Pair<Value, Value>
rgb2ind_inmap(const Value &RGB, const Value &cmap, keel::Allocator *mr)
{
    rgb2ind_inmap_argcheck(RGB, cmap);
    const keel::usize H = RGB.dims().rows();
    const keel::usize W = RGB.dims().cols();
    const keel::usize K = cmap.dims().rows();
    const keel::usize HW = H * W;

    const ValueType idxT = (K <= 256) ? ValueType::UINT8 : ValueType::UINT16;
    Value X = Value::matrix(H, W, idxT, mr);
    if (HW == 0) return {X, cmap};

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> pix(HW * 3, scratch.allocator());
    const keel::f64 scale = (RGB.type() == ValueType::UINT8)  ? 1.0 / 255.0
                          : (RGB.type() == ValueType::UINT16) ? 1.0 / 65535.0
                          : 1.0;
    for (keel::usize p = 0; p < 3; ++p)
        for (keel::usize i = 0; i < HW; ++i)
            pix[p * HW + i] = RGB.elemAsDouble(p * HW + i) * scale;

    keel::Array<keel::f64> cm(K * 3, scratch.allocator());
    for (keel::usize c = 0; c < 3; ++c)
        for (keel::usize k = 0; k < K; ++k)
            cm[c * K + k] = cmap.elemAsDouble(k + c * K);

    keel::Array<keel::u32> out_idx(HW, scratch.allocator());
    (void)keel::image::rgb_to_ind_nearest_planar(
        keel::ArrayRef<keel::f64>(pix.data(), HW),
        keel::ArrayRef<keel::f64>(pix.data() + HW, HW),
        keel::ArrayRef<keel::f64>(pix.data() + 2 * HW, HW),
        keel::ArrayRef<keel::f64>(cm.data(), K),
        keel::ArrayRef<keel::f64>(cm.data() + K, K),
        keel::ArrayRef<keel::f64>(cm.data() + 2 * K, K),
        out_idx.as_mut_ref());

    if (idxT == ValueType::UINT8) {
        for (keel::usize i = 0; i < HW; ++i) X.uint8DataMut()[i] = keel::as<keel::u8>(out_idx[i]);
    } else {
        for (keel::usize i = 0; i < HW; ++i) X.uint16DataMut()[i] = keel::as<keel::u16>(out_idx[i]);
    }
    return {X, cmap};
}

Value rgbwide2ycbcr(const Value &RGB, keel::i32 bits_per_sample, keel::Allocator *mr)
{
    rgbwide2ycbcr_argcheck(RGB, bits_per_sample);
    const auto &d = RGB.dims();
    const bool is_image = d.is3D();
    const keel::usize H = d.rows();
    const keel::usize W = is_image ? d.cols() : 1;
    const keel::usize N = H * W;

    Value out = is_image ? Value::matrix3d(H, W, 3, ValueType::UINT16, mr)
                         : Value::matrix(H, 3, ValueType::UINT16, mr);
    const keel::u16 *src = RGB.uint16Data();
    keel::u16 *dst = out.uint16DataMut();
    const keel::usize n = is_image ? N : H;
    (void)keel::image::rgbwide_to_ycbcr_planar(
        keel::ArrayRef<keel::u16>(src, n),
        keel::ArrayRef<keel::u16>(src + n, n),
        keel::ArrayRef<keel::u16>(src + 2 * n, n),
        bits_per_sample,
        keel::MutArrayRef<keel::u16>(dst, n),
        keel::MutArrayRef<keel::u16>(dst + n, n),
        keel::MutArrayRef<keel::u16>(dst + 2 * n, n));
    return out;
}

// ── cmunique (remove duplicate colormap entries) ───────────────────
//
// MATLAB R2025b cmunique.m algorithm (transliterated verbatim):
//
//   tol = 1/1024
//   map  = round(map / tol) * tol                — quantise
//   [~, ndx] = sortrows(map, [3 2 1])             — sort by (B, G, R)
//   pos(ndx) = 1:length(ndx)                      — inverse perm
//   d  = all(abs(diff(map(ndx,:)))' < tol)'       — consecutive dup
//   loc = (1:length(ndx))' - [0; cumsum(d)]       — sorted→compressed
//   c(:) = loc(pos(c))                            — remap indices
//   ndx(d) = []                                   — drop dup rows
//   map = map(ndx, :)                             — compressed map
//   n  = histcounts(c, 1:nmap+1)
//   d  = (n == 0)                                  — unused rows
//   loc = (1:nmap) - cumsum(d)                    — compressed→final
//   c(:) = loc(c)
//   map(d, :) = []
//   if max(c) ≤ 256: c = uint8(c - 1)             — 0-based output
//
// For the (RGB) and (I) one-arg forms, build the per-pixel "big
// colormap" (1 row per pixel, im2double-converted) and `c =
// reshape(1:m*n, m, n)` first, then run the same compression pass.
//
// Reference: see MATLAB toolbox source
// `toolbox/matlab/graphics/graphics/graph3d/cmunique.m`.

namespace {

// Convert input pixel value (any class) to double in [0, 1] per
// MATLAB's im2double rules (uint8 / uint16 scaled by intmax; double
// and single passed through).
inline keel::f64 im2double_scalar(const Value &v, keel::usize i)
{
    switch (v.type()) {
        case ValueType::UINT8:  return v.uint8Data()[i] / 255.0;
        case ValueType::UINT16: return v.uint16Data()[i] / 65535.0;
        case ValueType::DOUBLE: return v.doubleData()[i];
        case ValueType::SINGLE: return keel::as<keel::f64>(v.singleData()[i]);
        default:
            throw Error("cmunique: unsupported image class",
                        0, 0, "cmunique", "", "numkit:cmunique:cls");
    }
}

// Core compression pass. `c` is a 1-based double index vector of
// length N referencing `map_in` (M × 3 colormap). Outputs `(Y,
// newmap)` per MATLAB rules — Y is uint8 if newmap has ≤ 256 rows.
keel::Pair<Value, Value>
cmunique_core(keel::Array<keel::f64> &c, keel::usize H, keel::usize W,
              const keel::Array<keel::f64> &map_in, keel::usize M,
              keel::Allocator *mr)
{
    const keel::usize N = c.size();
    keel::Array<keel::usize> idx_in(N, mr), idx_out(N, mr);
    for (keel::usize i = 0; i < N; ++i) idx_in[i] = keel::as<keel::usize>(c[i]) - 1;
    keel::Array<keel::f64> out_r(mr), out_g(mr), out_b(mr);
    (void)keel::image::colormap_unique(
        idx_in.as_ref(),
        keel::ArrayRef<keel::f64>(map_in.data(), M),
        keel::ArrayRef<keel::f64>(map_in.data() + M, M),
        keel::ArrayRef<keel::f64>(map_in.data() + 2 * M, M),
        idx_out.as_mut_ref(), out_r, out_g, out_b, mr);

    const keel::usize nmap2 = out_r.size();
    keel::usize max_c = 0;
    for (keel::usize i = 0; i < N; ++i) if (idx_out[i] + 1 > max_c) max_c = idx_out[i] + 1;
    const bool to_uint8 = (max_c <= 256);

    Value Y = to_uint8 ? Value::matrix(H, W, ValueType::UINT8, mr)
                       : Value::matrix(H, W, ValueType::DOUBLE, mr);
    if (to_uint8) {
        keel::u8 *yp = Y.uint8DataMut();
        for (keel::usize i = 0; i < N; ++i) yp[i] = keel::as<keel::u8>(idx_out[i]);
    } else {
        keel::f64 *yp = Y.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i) yp[i] = keel::as<keel::f64>(idx_out[i] + 1);
    }

    Value newmap = Value::matrix(nmap2, 3, ValueType::DOUBLE, mr);
    keel::f64 *np = newmap.doubleDataMut();
    if (nmap2 > 0) {
        keel::copy_n(np,             out_r.data(), nmap2);
        keel::copy_n(np + nmap2,     out_g.data(), nmap2);
        keel::copy_n(np + 2 * nmap2, out_b.data(), nmap2);
    }
    return {keel::move(Y), keel::move(newmap)};
}

} // anonymous

keel::Pair<Value, Value>
cmunique_xm(const Value &X, const Value &MAP, keel::Allocator *mr)
{
    cmunique_xm_argcheck(MAP);
    if (MAP.dims().cols() != 3 || MAP.dims().is3D())
        throw Error("cmunique: MAP must be N×3",
                    0, 0, "cmunique", "", "numkit:cmunique:map");
    const keel::usize M = MAP.dims().rows();
    const keel::usize H = X.dims().rows();
    const keel::usize W = X.dims().cols();
    const keel::usize N = X.numel();

    keel::Array<keel::f64> map_in(3 * M, mr);
    for (keel::usize c = 0; c < 3; ++c)
        for (keel::usize r = 0; r < M; ++r)
            map_in[c * M + r] = MAP.elemAsDouble(c * M + r);

    // Convert X to double + offset for integer classes (MATLAB:
    // c = double(a) + 1 for non-double X).
    keel::Array<keel::f64> c(N, mr);
    const bool isInt = (X.type() == ValueType::UINT8
                     || X.type() == ValueType::UINT16);
    for (keel::usize i = 0; i < N; ++i)
        c[i] = X.elemAsDouble(i) + (isInt ? 1.0 : 0.0);

    return cmunique_core(c, H, W, map_in, M, mr);
}

keel::Pair<Value, Value>
cmunique_rgb(const Value &RGB, keel::Allocator *mr)
{
    cmunique_rgb_argcheck(RGB);
    if (!RGB.dims().is3D() || RGB.dims().pages() != 3)
        throw Error("cmunique: RGB must be H×W×3",
                    0, 0, "cmunique", "", "numkit:cmunique:rgb");
    const keel::usize H = RGB.dims().rows();
    const keel::usize W = RGB.dims().cols();
    const keel::usize N = H * W;
    const keel::usize M = N;
    keel::Array<keel::f64> map_in(3 * M, mr);
    // map(:, c) = im2double(reshape(RGB(:,:,c), [], 1))
    for (keel::usize c = 0; c < 3; ++c)
        for (keel::usize k = 0; k < N; ++k)
            map_in[c * M + k] = im2double_scalar(RGB, c * N + k);
    // c = reshape(1:N, H, W) — column-major identity 1..N.
    keel::Array<keel::f64> idx(N, mr);
    for (keel::usize i = 0; i < N; ++i) idx[i] = keel::as<keel::f64>(i + 1);
    return cmunique_core(idx, H, W, map_in, M, mr);
}

keel::Pair<Value, Value>
cmunique_i(const Value &I, keel::Allocator *mr)
{
    cmunique_i_argcheck(I);
    if (I.dims().ndims() > 2)
        throw Error("cmunique: I must be a 2-D intensity image",
                    0, 0, "cmunique", "", "numkit:cmunique:i");
    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    const keel::usize N = H * W;
    const keel::usize M = N;
    keel::Array<keel::f64> map_in(3 * M, mr);
    for (keel::usize k = 0; k < N; ++k) {
        const keel::f64 v = im2double_scalar(I, k);
        map_in[0 * M + k] = v;
        map_in[1 * M + k] = v;
        map_in[2 * M + k] = v;
    }
    keel::Array<keel::f64> idx(N, mr);
    for (keel::usize i = 0; i < N; ++i) idx[i] = keel::as<keel::f64>(i + 1);
    return cmunique_core(idx, H, W, map_in, M, mr);
}

// ── ycbcr2rgbwide — inverse of rgbwide2ycbcr ───────────────────────
//
// Reference: ITU-R Rec. BT.2020-2 (10/2015) and BT.2100-2 (07/2018).
// Algorithm transliterated verbatim from MATLAB R2025b
// `images/colorspaces/+images/+color/+internal/ycbcr2rgbwideImpl.m`.
//
// Pipeline (per pixel):
//   1. Constants (bps = 10):
//        yzero        = 64                 (Y reference black)
//        ypeak        = 940                (Y reference white)
//        yrange       = ypeak − yzero      = 876
//        chromazero   = 2^(bps − 1)        = 512
//        chromarange  = 960 − 64           = 896 (Cb / Cr full nominal)
//        blackLevel   = 64                 (RGB reference black)
//        nominalPeak  = 940                (RGB reference white)
//        nominalRange = nominalPeak − blackLevel = 876
//      (12-bit equivalents are × 4 except chromarange = 3840 − 256.)
//   2. Y_n  = (Y  − yzero)/yrange;
//      Cb_n = (Cb − chromazero)/chromarange;
//      Cr_n = (Cr − chromazero)/chromarange;
//   3. R = 1.4746·Cr_n + Y_n;
//      B = 1.8814·Cb_n + Y_n;
//      G = (Y_n − 0.2627·R − 0.0593·B) / 0.6780.
//   4. RGB_out = uint16(rgb · nominalRange + blackLevel) with normal
//      uint16 saturation on out-of-range YCbCr inputs.
//
// MATLAB's uint16(...) is round-half-to-nearest-even with saturation
// at [0, 65535]; we mirror that with `+ 0.5` floor (round-half-up).
// Bit-equal on every probed test vector (round-trip with
// rgbwide2ycbcr).
Value ycbcr2rgbwide(const Value &YCBCR, keel::i32 bits_per_sample, keel::Allocator *mr)
{
    ycbcr2rgbwide_argcheck(YCBCR, bits_per_sample);
    const auto &d = YCBCR.dims();
    const bool is_image = d.is3D();
    const keel::usize H = d.rows();
    const keel::usize W = is_image ? d.cols() : 1;
    const keel::usize n = is_image ? (H * W) : H;

    Value out = is_image ? Value::matrix3d(H, W, 3, ValueType::UINT16, mr)
                         : Value::matrix(H, 3, ValueType::UINT16, mr);
    const keel::u16 *src = YCBCR.uint16Data();
    keel::u16 *dst = out.uint16DataMut();
    (void)keel::image::ycbcr_to_rgbwide_planar(
        keel::ArrayRef<keel::u16>(src, n),
        keel::ArrayRef<keel::u16>(src + n, n),
        keel::ArrayRef<keel::u16>(src + 2 * n, n),
        bits_per_sample,
        keel::MutArrayRef<keel::u16>(dst, n),
        keel::MutArrayRef<keel::u16>(dst + n, n),
        keel::MutArrayRef<keel::u16>(dst + 2 * n, n));
    return out;
}

namespace {

keel::image::WideColorSpace parse_wide_cs(keel::i32 bits_per_sample, keel::StringRef color_space,
                                          keel::StringRef linearization, const char *fn)
{
    const bool is_bt2020 = (color_space.equals_ignore_ascii_case("bt.2020")
                            || color_space.equals_ignore_ascii_case("bt2020"));
    const bool is_bt2100 = (color_space.equals_ignore_ascii_case("bt.2100")
                            || color_space.equals_ignore_ascii_case("bt2100"));
    if (!is_bt2020 && !is_bt2100)
        throw Error(keel::format("{}: ColorSpace must be 'BT.2020' or 'BT.2100'", fn),
                    0, 0, fn, "", keel::format("numkit:{}:cs", fn));
    if (bits_per_sample != 10 && bits_per_sample != 12)
        throw Error(keel::format("{}: bits_per_sample must be 10 or 12", fn),
                    0, 0, fn, "", "numkit:rgbwide2xyz:bps");
    if (is_bt2020) return keel::image::WideColorSpace::kBt2020;
    return linearization.equals_ignore_ascii_case("hlg")
               ? keel::image::WideColorSpace::kBt2100Hlg
               : keel::image::WideColorSpace::kBt2100Pq;
}

} // namespace

Value rgbwide2xyz(const Value &RGB, keel::i32 bits_per_sample,
                  keel::StringRef color_space, keel::StringRef linearization,
                  keel::Allocator *mr)
{
    const auto cs = parse_wide_cs(bits_per_sample, color_space, linearization, "rgbwide2xyz");
    const auto &d = RGB.dims();
    if (d.is3D() && d.pages() != 3)
        throw Error("rgbwide2xyz: RGB must be Nx3 or HxWx3",
                    0, 0, "rgbwide2xyz", "", "numkit:rgbwide2xyz:shape");
    const keel::usize H = d.rows(), W = d.cols();
    const bool is_image = d.is3D();
    const keel::usize Np = is_image ? (H * W) : H;

    keel::Array<keel::f64> in(3 * Np, mr);
    for (keel::usize i = 0; i < 3 * Np; ++i) in[i] = RGB.elemAsDouble(i);

    Value out = is_image ? Value::matrix3d(H, W, 3, ValueType::DOUBLE, mr)
                         : Value::matrix(H, 3, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    (void)keel::image::rgbwide_to_xyz_planar(
        keel::ArrayRef<keel::f64>(in.data(), Np),
        keel::ArrayRef<keel::f64>(in.data() + Np, Np),
        keel::ArrayRef<keel::f64>(in.data() + 2 * Np, Np),
        bits_per_sample, cs,
        keel::MutArrayRef<keel::f64>(od, Np),
        keel::MutArrayRef<keel::f64>(od + Np, Np),
        keel::MutArrayRef<keel::f64>(od + 2 * Np, Np));
    return out;
}

Value xyz2rgbwide(const Value &XYZ, keel::i32 bits_per_sample,
                  keel::StringRef color_space, keel::StringRef linearization,
                  keel::Allocator *mr)
{
    const auto cs = parse_wide_cs(bits_per_sample, color_space, linearization, "xyz2rgbwide");
    const auto &d = XYZ.dims();
    if (d.is3D() && d.pages() != 3)
        throw Error("xyz2rgbwide: XYZ must be Nx3 or HxWx3",
                    0, 0, "xyz2rgbwide", "", "numkit:xyz2rgbwide:shape");
    const keel::usize H = d.rows(), W = d.cols();
    const bool is_image = d.is3D();
    const keel::usize Np = is_image ? (H * W) : H;

    keel::Array<keel::f64> in(3 * Np, mr);
    for (keel::usize i = 0; i < 3 * Np; ++i) in[i] = XYZ.elemAsDouble(i);

    Value out = is_image ? Value::matrix3d(H, W, 3, ValueType::UINT16, mr)
                         : Value::matrix(H, 3, ValueType::UINT16, mr);
    keel::u16 *od = out.uint16DataMut();
    (void)keel::image::xyz_to_rgbwide_planar(
        keel::ArrayRef<keel::f64>(in.data(), Np),
        keel::ArrayRef<keel::f64>(in.data() + Np, Np),
        keel::ArrayRef<keel::f64>(in.data() + 2 * Np, Np),
        bits_per_sample, cs,
        keel::MutArrayRef<keel::u16>(od, Np),
        keel::MutArrayRef<keel::u16>(od + Np, Np),
        keel::MutArrayRef<keel::u16>(od + 2 * Np, Np));
    return out;
}

// ── imfuse (composite two images for visual comparison) ───────────
//
// MATLAB R2025b imfuse.m algorithm (programmatic, no imref2d form):
//
//   1. Zero-pad A and B to common size = element-wise max(sz_A, sz_B).
//   2. Per `scaling`:
//        "independent" → scale each to [0, 1] before im2uint8;
//        "joint"       → concatenate, scale together, split;
//        "none"        → just cast to uint8 (via im2uint8).
//   3. Per `method`:
//        "falsecolor"  → grayscale-convert, scale, assign per
//                        ColorChannels to RGB channels.
//        "blend"       → 0.5*A + 0.5*B in overlap (whole image
//                        without imref2d).
//        "diff"        → scale(|A-B|).
//        "checkerboard"→ 8x8 [1 0; 0 1] repmat → imresize-nearest
//                        to image size; A on 1, B on 0.
//        "montage"     → horizontal concat [A B].
namespace {

// Get H, W (ignoring channels) of a 2-D or 3-D image.
inline void hw_of(const Value &I, keel::usize &H, keel::usize &W) {
    H = I.dims().rows();
    W = I.dims().cols();
}

// Channels: 1 for HxW, 3 (or other) for HxWx3.
inline keel::usize channels_of(const Value &I) {
    return I.dims().is3D() ? I.dims().pages() : 1;
}

// Zero-pad I to (outH, outW) with the same channel count.
Value pad_zeros(const Value &I, keel::usize outH, keel::usize outW,
                keel::Allocator *mr)
{
    keel::usize H, W;
    hw_of(I, H, W);
    const keel::usize C = channels_of(I);
    if (H == outH && W == outW) return I;
    const ValueType T = I.type();
    Value out = (C == 1)
        ? Value::matrix(outH, outW, T, mr)
        : Value::matrix3d(outH, outW, C, T, mr);
    for (keel::usize ch = 0; ch < C; ++ch) {
        for (keel::usize c = 0; c < W; ++c) {
            for (keel::usize r = 0; r < H; ++r) {
                const keel::usize src_idx = (C == 1)
                    ? c * H + r
                    : ch * H * W + c * H + r;
                const keel::usize dst_idx = (C == 1)
                    ? c * outH + r
                    : ch * outH * outW + c * outH + r;
                const keel::f64 v = I.elemAsDouble(src_idx);
                switch (T) {
                    case ValueType::DOUBLE:  out.doubleDataMut()[dst_idx]  = v; break;
                    case ValueType::SINGLE:  out.singleDataMut()[dst_idx]  = keel::as<keel::f32>(v); break;
                    case ValueType::UINT8:   out.uint8DataMut()[dst_idx]   = keel::as<keel::u8>(v); break;
                    case ValueType::UINT16:  out.uint16DataMut()[dst_idx]  = keel::as<keel::u16>(v); break;
                    case ValueType::UINT32:  out.uint32DataMut()[dst_idx]  = keel::as<keel::u32>(v); break;
                    case ValueType::INT8:    out.int8DataMut()[dst_idx]    = keel::as<keel::i8>(v); break;
                    case ValueType::INT16:   out.int16DataMut()[dst_idx]   = keel::as<keel::i16>(v); break;
                    case ValueType::INT32:   out.int32DataMut()[dst_idx]   = keel::as<keel::i32>(v); break;
                    case ValueType::LOGICAL: out.logicalDataMut()[dst_idx] = v != 0.0 ? 1 : 0; break;
                    default: out.doubleDataMut()[dst_idx] = v; break;
                }
            }
        }
    }
    return out;
}

// MATLAB R2025b scaleGrayscaleImage: cast to single, scale [min,max] → [0,1].
Value scale_grayscale_image(const Value &I, keel::Allocator *mr) {
    if (I.isLogical()) return I;
    const keel::usize N = I.numel();
    Value out = (I.dims().is3D())
        ? Value::matrix3d(I.dims().rows(), I.dims().cols(),
                          I.dims().pages(), ValueType::SINGLE, mr)
        : Value::matrix(I.dims().rows(), I.dims().cols(),
                        ValueType::SINGLE, mr);
    keel::f32 *od = out.singleDataMut();
    if (N == 0) return out;
    keel::f32 mn = keel::Limits<keel::f32>::infinity();
    keel::f32 mx = -keel::Limits<keel::f32>::infinity();
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f32 v = keel::as<keel::f32>(I.elemAsDouble(i));
        od[i] = v;
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    if (mn == mx) return out;  // constant passthrough as-is
    const keel::f32 inv = 1.0f / (mx - mn);
    for (keel::usize i = 0; i < N; ++i) od[i] = (od[i] - mn) * inv;
    return out;
}

// Joint scale.
void scale_two_grayscale_images(Value &A, Value &B,
                                keel::Allocator *mr) {
    keel::f32 mn = keel::Limits<keel::f32>::infinity();
    keel::f32 mx = -keel::Limits<keel::f32>::infinity();
    for (keel::usize i = 0; i < A.numel(); ++i) {
        const keel::f32 v = keel::as<keel::f32>(A.elemAsDouble(i));
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    for (keel::usize i = 0; i < B.numel(); ++i) {
        const keel::f32 v = keel::as<keel::f32>(B.elemAsDouble(i));
        if (v < mn) mn = v;
        if (v > mx) mx = v;
    }
    auto rescale = [&](const Value &X) {
        Value Y = X.dims().is3D()
            ? Value::matrix3d(X.dims().rows(), X.dims().cols(),
                              X.dims().pages(), ValueType::SINGLE, mr)
            : Value::matrix(X.dims().rows(), X.dims().cols(),
                            ValueType::SINGLE, mr);
        keel::f32 *yd = Y.singleDataMut();
        for (keel::usize i = 0; i < X.numel(); ++i) {
            const keel::f32 v = keel::as<keel::f32>(X.elemAsDouble(i));
            yd[i] = (mn == mx) ? v : (v - mn) / (mx - mn);
        }
        return Y;
    };
    A = rescale(A);
    B = rescale(B);
}

void make_similar(Value &A, Value &B, keel::StringRef scaling,
                  keel::Allocator *mr)
{
    const bool aIsRGB = A.dims().is3D();
    const bool bIsRGB = B.dims().is3D();
    if (!aIsRGB && !bIsRGB) {
        if (scaling == "joint") {
            scale_two_grayscale_images(A, B, mr);
        } else if (scaling == "independent") {
            A = scale_grayscale_image(A, mr);
            B = scale_grayscale_image(B, mr);
        }
        A = im2uint8(A, mr);
        B = im2uint8(B, mr);
    } else if (aIsRGB && bIsRGB) {
        A = im2uint8(A, mr);
        B = im2uint8(B, mr);
    } else if (aIsRGB && !bIsRGB) {
        if (scaling != "none") B = scale_grayscale_image(B, mr);
        Value Bg = im2uint8(B, mr);
        const keel::usize H = Bg.dims().rows();
        const keel::usize W = Bg.dims().cols();
        Value Brep = Value::matrix3d(H, W, 3, ValueType::UINT8, mr);
        for (keel::usize ch = 0; ch < 3; ++ch)
            for (keel::usize i = 0; i < H * W; ++i)
                Brep.uint8DataMut()[ch * H * W + i] = Bg.uint8Data()[i];
        B = Brep;
        A = im2uint8(A, mr);
    } else {
        if (scaling != "none") A = scale_grayscale_image(A, mr);
        Value Ag = im2uint8(A, mr);
        const keel::usize H = Ag.dims().rows();
        const keel::usize W = Ag.dims().cols();
        Value Arep = Value::matrix3d(H, W, 3, ValueType::UINT8, mr);
        for (keel::usize ch = 0; ch < 3; ++ch)
            for (keel::usize i = 0; i < H * W; ++i)
                Arep.uint8DataMut()[ch * H * W + i] = Ag.uint8Data()[i];
        A = Arep;
        B = im2uint8(B, mr);
    }
}

void imfuse_argcheck(keel::StringRef method, keel::StringRef scaling,
                     const Value &channels)
{
    static const keel::StaticArray<const char *, 5> methods{
        "falsecolor", "blend", "diff", "checkerboard", "montage"};
    bool method_ok = false;
    for (auto m : methods) if (method == m) { method_ok = true; break; }
    if (!method_ok)
        throw Error(keel::format("imfuse: unknown METHOD '{}' "
                                 "(allowed: falsecolor / blend / diff / "
                                 "checkerboard / montage)", method),
                    0, 0, "imfuse", "", "numkit:imfuse:method");
    if (scaling != "independent" && scaling != "joint" && scaling != "none")
        throw Error(keel::format("imfuse: unknown SCALING '{}'", scaling),
                    0, 0, "imfuse", "", "numkit:imfuse:scaling");
    if (method == "falsecolor" && channels.numel() == 3) {
        for (keel::i32 k = 0; k < 3; ++k) {
            const keel::f64 v = channels.elemAsDouble(k);
            if (v != 0.0 && v != 1.0 && v != 2.0)
                throw Error("imfuse: ColorChannels values must be 0, 1, or 2",
                            0, 0, "imfuse", "", "numkit:imfuse:channels");
        }
    }
}

void tonemap_argcheck(const Value &HDR, keel::f64 lremap_lo, keel::f64 lremap_hi,
                      keel::f64 saturation, keel::i32 ntilesR, keel::i32 ntilesC)
{
    internal::requireImageClass(HDR, "tonemap", "HDR", 1,
                                internal::ImgClasses::FloatOnly);
    if (!(lremap_lo >= 0.0 && lremap_lo <= 1.0
       && lremap_hi >= 0.0 && lremap_hi <= 1.0
       && lremap_lo < lremap_hi))
        throw Error("tonemap: AdjustLightness must satisfy "
                    "0 <= lo < hi <= 1",
                    0, 0, "tonemap", "", "numkit:tonemap:adjust");
    if (!(saturation >= 0))
        throw Error("tonemap: AdjustSaturation must be non-negative",
                    0, 0, "tonemap", "", "numkit:tonemap:sat");
    if (ntilesR < 2 || ntilesC < 2)
        throw Error("tonemap: NumberOfTiles values must be >= 2",
                    0, 0, "tonemap", "", "numkit:tonemap:tiles");
    if (HDR.dims().is3D() && HDR.dims().pages() != 3)
        throw Error("tonemap: HDR must be H×W or H×W×3",
                    0, 0, "tonemap", "", "numkit:tonemap:shape");
}

} // anonymous (imfuse helpers)

Value imfuse(const Value &Ain, const Value &Bin,
             keel::StringRef method,
             keel::StringRef scaling,
             const Value &channels,
             keel::Allocator *mr)
{
    imfuse_argcheck(method, scaling, channels);
    static const keel::StaticArray<const char *, 5> methods{
        "falsecolor", "blend", "diff", "checkerboard", "montage"};
    bool method_ok = false;
    for (auto m : methods) if (method == m) { method_ok = true; break; }
    if (!method_ok)
        throw Error(keel::format("imfuse: unknown METHOD '{}' "
                                 "(allowed: falsecolor / blend / diff / "
                                 "checkerboard / montage)", method),
                    0, 0, "imfuse", "", "numkit:imfuse:method");
    if (scaling != "independent" && scaling != "joint" && scaling != "none")
        throw Error(keel::format("imfuse: unknown SCALING '{}'", scaling),
                    0, 0, "imfuse", "", "numkit:imfuse:scaling");

    keel::usize Ha, Wa, Hb, Wb;
    hw_of(Ain, Ha, Wa);
    hw_of(Bin, Hb, Wb);

    Value A, B;
    if (method == "montage") {
        const keel::usize H = keel::max(Ha, Hb);
        A = pad_zeros(Ain, H, Wa, mr);
        B = pad_zeros(Bin, H, Wb, mr);
    } else {
        const keel::usize H = keel::max(Ha, Hb);
        const keel::usize W = keel::max(Wa, Wb);
        A = pad_zeros(Ain, H, W, mr);
        B = pad_zeros(Bin, H, W, mr);
    }

    if (method == "falsecolor") {
        if (A.dims().is3D()) A = rgb2gray(A, mr);
        if (B.dims().is3D()) B = rgb2gray(B, mr);
        if (scaling == "joint")        scale_two_grayscale_images(A, B, mr);
        else if (scaling == "independent") {
            A = scale_grayscale_image(A, mr);
            B = scale_grayscale_image(B, mr);
        }
        A = im2uint8(A, mr);
        B = im2uint8(B, mr);
        keel::StaticArray<keel::i32, 3> ch{2, 1, 2};  // green-magenta default
        if (channels.numel() == 3) {
            for (keel::i32 k = 0; k < 3; ++k) {
                const keel::f64 v = channels.elemAsDouble(k);
                if (v != 0.0 && v != 1.0 && v != 2.0)
                    throw Error("imfuse: ColorChannels values must be 0, 1, or 2",
                                0, 0, "imfuse", "", "numkit:imfuse:channels");
                ch[k] = keel::as<keel::i32>(v);
            }
        }
        const keel::usize H = A.dims().rows();
        const keel::usize W = A.dims().cols();
        const keel::usize plane = H * W;
        Value R = Value::matrix3d(H, W, 3, ValueType::UINT8, mr);
        keel::u8 *rd = R.uint8DataMut();
        for (keel::i32 p = 0; p < 3; ++p) {
            const keel::u8 *src = (ch[p] == 1) ? A.uint8Data()
                                    : (ch[p] == 2) ? B.uint8Data()
                                    : nullptr;
            keel::u8 *dst = rd + p * plane;
            if (src) keel::mem_copy(dst, src, plane);
        }
        return R;
    }

    if (method == "blend") {
        make_similar(A, B, scaling, mr);
        const keel::usize H = A.dims().rows();
        const keel::usize W = A.dims().cols();
        const keel::usize C = channels_of(A);
        Value R = (C == 1)
            ? Value::matrix(H, W, ValueType::UINT8, mr)
            : Value::matrix3d(H, W, C, ValueType::UINT8, mr);
        const keel::usize N = H * W * C;
        const keel::u8 *ap = A.uint8Data();
        const keel::u8 *bp = B.uint8Data();
        keel::u8 *rp = R.uint8DataMut();
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f32 v = 0.5f * keel::as<keel::f32>(ap[i])
                          + 0.5f * keel::as<keel::f32>(bp[i]);
            keel::i32 rv = keel::as<keel::i32>(v + 0.5f);
            if (rv < 0) rv = 0; else if (rv > 255) rv = 255;
            rp[i] = keel::as<keel::u8>(rv);
        }
        return R;
    }

    if (method == "diff") {
        if (A.dims().is3D()) A = rgb2gray(A, mr);
        if (B.dims().is3D()) B = rgb2gray(B, mr);
        if (scaling == "joint") scale_two_grayscale_images(A, B, mr);
        else if (scaling == "independent") {
            A = scale_grayscale_image(A, mr);
            B = scale_grayscale_image(B, mr);
        }
        const keel::usize H = A.dims().rows();
        const keel::usize W = A.dims().cols();
        const keel::usize N = H * W;
        Value diff = Value::matrix(H, W, ValueType::SINGLE, mr);
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f32 va = keel::as<keel::f32>(A.elemAsDouble(i));
            const keel::f32 vb = keel::as<keel::f32>(B.elemAsDouble(i));
            diff.singleDataMut()[i] = keel::abs(va - vb);
        }
        diff = scale_grayscale_image(diff, mr);
        return im2uint8(diff, mr);
    }

    if (method == "checkerboard") {
        make_similar(A, B, scaling, mr);
        const keel::usize H = A.dims().rows();
        const keel::usize W = A.dims().cols();
        const keel::usize C = channels_of(A);
        // 16x16 base = [1 0; 0 1] tiled 8x8.
        Value base = Value::matrix(16, 16, ValueType::LOGICAL, mr);
        keel::u8 *bbp = base.logicalDataMut();
        for (keel::usize c = 0; c < 16; ++c)
            for (keel::usize r = 0; r < 16; ++r) {
                const keel::usize br = r / 8;
                const keel::usize bc = c / 8;
                bbp[c * 16 + r] = keel::as<keel::u8>((br + bc + 1) % 2);
            }
        Value mask = imresize(base, H, W, "nearest", mr);
        Value R = (C == 1)
            ? Value::matrix(H, W, ValueType::UINT8, mr)
            : Value::matrix3d(H, W, C, ValueType::UINT8, mr);
        const keel::u8 *m = mask.logicalData();
        const keel::u8 *ap = A.uint8Data();
        const keel::u8 *bp2 = B.uint8Data();
        keel::u8 *rp = R.uint8DataMut();
        const keel::usize plane = H * W;
        for (keel::usize ch = 0; ch < C; ++ch) {
            const keel::u8 *a_ch = ap + ch * plane;
            const keel::u8 *b_ch = bp2 + ch * plane;
            keel::u8 *r_ch = rp + ch * plane;
            for (keel::usize i = 0; i < plane; ++i)
                r_ch[i] = m[i] ? a_ch[i] : b_ch[i];
        }
        return R;
    }

    // method == "montage"
    make_similar(A, B, scaling, mr);
    const keel::usize H = A.dims().rows();
    const keel::usize WAa = A.dims().cols();
    const keel::usize WBb = B.dims().cols();
    const keel::usize C = channels_of(A);
    const keel::usize Wt = WAa + WBb;
    Value R = (C == 1)
        ? Value::matrix(H, Wt, ValueType::UINT8, mr)
        : Value::matrix3d(H, Wt, C, ValueType::UINT8, mr);
    const keel::u8 *ap = A.uint8Data();
    const keel::u8 *bp = B.uint8Data();
    keel::u8 *rp = R.uint8DataMut();
    for (keel::usize ch = 0; ch < C; ++ch) {
        for (keel::usize c = 0; c < WAa; ++c)
            for (keel::usize r = 0; r < H; ++r)
                rp[ch * H * Wt + c * H + r]
                    = ap[ch * H * WAa + c * H + r];
        for (keel::usize c = 0; c < WBb; ++c)
            for (keel::usize r = 0; r < H; ++r)
                rp[ch * H * Wt + (WAa + c) * H + r]
                    = bp[ch * H * WBb + c * H + r];
    }
    return R;
}

// ── tonemap (HDR → LDR for display) ─────────────────────────────
//
// MATLAB R2025b tonemap.m algorithm:
//   1. min_nz = min over non-zero entries of HDR.
//   2. Replace zeros with min_nz.
//   3. log2 → mat2gray (global min/max → [0, 1]).
//   4. Grayscale: adapthisteq(NumTiles) → imadjust(LRemap, [0 1]).
//      RGB: rgb2lab → L/100 → adapthisteq → imadjust → *100;
//           a,b channels × saturation; lab2rgb.
//   5. im2uint8.
//
// References:
//   G. Ward et al., "A Visibility Matching Tone Reproduction
//   Operator for High Dynamic Range Scenes", IEEE TVCG 3(4), 1997.
Value tonemap(const Value &HDR,
              keel::f64 lremap_lo, keel::f64 lremap_hi,
              keel::f64 saturation,
              keel::i32 ntilesR, keel::i32 ntilesC,
              keel::Allocator *mr)
{
    tonemap_argcheck(HDR, lremap_lo, lremap_hi, saturation, ntilesR, ntilesC);
    internal::requireImageClass(HDR, "tonemap", "HDR", 1,
                                internal::ImgClasses::FloatOnly);
    // Single HDR: compute on the double copy (ref gives identical
    // uint8 output for double and single — probed).
    if (HDR.type() == ValueType::SINGLE) {
        Value Hd = internal::coerceToDoubleImage(HDR, mr);
        return tonemap(Hd, lremap_lo, lremap_hi, saturation, ntilesR,
                       ntilesC, mr);
    }
    if (!(lremap_lo >= 0.0 && lremap_lo <= 1.0
       && lremap_hi >= 0.0 && lremap_hi <= 1.0
       && lremap_lo < lremap_hi))
        throw Error("tonemap: AdjustLightness must satisfy "
                    "0 <= lo < hi <= 1",
                    0, 0, "tonemap", "", "numkit:tonemap:adjust");
    if (!(saturation >= 0))
        throw Error("tonemap: AdjustSaturation must be non-negative",
                    0, 0, "tonemap", "", "numkit:tonemap:sat");
    if (ntilesR < 2 || ntilesC < 2)
        throw Error("tonemap: NumberOfTiles values must be >= 2",
                    0, 0, "tonemap", "", "numkit:tonemap:tiles");

    const keel::usize H = HDR.dims().rows();
    const keel::usize W = HDR.dims().cols();
    const bool isRGB = HDR.dims().is3D();
    if (isRGB && HDR.dims().pages() != 3)
        throw Error("tonemap: HDR must be H×W or H×W×3",
                    0, 0, "tonemap", "", "numkit:tonemap:shape");
    const keel::usize C = isRGB ? 3 : 1;
    const keel::usize plane = H * W;
    const keel::usize Ntot = plane * C;

    // Step 1+2: read into double; find min nonzero; replace zeros with min_nz.
    keel::Array<keel::f64> data(Ntot, 0.0, mr);
    keel::f64 min_nz = keel::Limits<keel::f64>::infinity();
    bool any_nz = false;
    for (keel::usize i = 0; i < Ntot; ++i) {
        const keel::f64 v = HDR.elemAsDouble(i);
        if (v < 0.0)
            throw Error("tonemap: HDR must be non-negative",
                        0, 0, "tonemap", "", "numkit:tonemap:negative");
        data[i] = v;
        if (v != 0.0) {
            any_nz = true;
            if (v < min_nz) min_nz = v;
        }
    }

    if (!any_nz) {
        // All zeros → output is all zeros uint8.
        Value out = isRGB
            ? Value::matrix3d(H, W, 3, ValueType::UINT8, mr)
            : Value::matrix(H, W, ValueType::UINT8, mr);
        keel::mem_zero(out.uint8DataMut(), Ntot);
        return out;
    }
    for (keel::usize i = 0; i < Ntot; ++i)
        if (data[i] == 0.0) data[i] = min_nz;

    // Step 3: log2, then mat2gray to [0, 1].
    keel::f64 gmin = keel::Limits<keel::f64>::infinity();
    keel::f64 gmax = -keel::Limits<keel::f64>::infinity();
    for (keel::usize i = 0; i < Ntot; ++i) {
        const keel::f64 v = keel::log2(data[i]);
        data[i] = v;
        if (v < gmin) gmin = v;
        if (v > gmax) gmax = v;
    }
    const keel::f64 span = gmax - gmin;
    if (span > 0.0) {
        const keel::f64 inv = 1.0 / span;
        for (keel::usize i = 0; i < Ntot; ++i) data[i] = (data[i] - gmin) * inv;
    } else {
        // Constant log image → all zeros after mat2gray.
        keel::fill(data.begin(), data.end(), 0.0);
    }

    // Wrap into a Value for downstream helpers.
    auto wrap_plane = [&](keel::usize ch) {
        Value V = Value::matrix(H, W, ValueType::DOUBLE, mr);
        for (keel::usize i = 0; i < plane; ++i)
            V.doubleDataMut()[i] = data[ch * plane + i];
        return V;
    };

    AdaptHistEqOptions opts;
    opts.numTilesR = ntilesR;
    opts.numTilesC = ntilesC;

    if (!isRGB) {
        // Grayscale: adapthisteq → imadjust.
        Value G = wrap_plane(0);
        G = adapthisteq(G, opts, mr);
        G = imadjust(G, lremap_lo, lremap_hi, 0.0, 1.0, 1.0, mr);
        return im2uint8(G, mr);
    }

    // RGB path.
    Value RGB = Value::matrix3d(H, W, 3, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < Ntot; ++i)
        RGB.doubleDataMut()[i] = data[i];
    Value Lab = rgb2lab(RGB, mr);  // H x W x 3 double

    // L / 100.
    for (keel::usize i = 0; i < plane; ++i)
        Lab.doubleDataMut()[i] /= 100.0;
    // Wrap L into a 2-D image; apply adapthisteq + imadjust; write back.
    Value Lplane = Value::matrix(H, W, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < plane; ++i)
        Lplane.doubleDataMut()[i] = Lab.doubleData()[i];
    Lplane = adapthisteq(Lplane, opts, mr);
    Lplane = imadjust(Lplane, lremap_lo, lremap_hi, 0.0, 1.0, 1.0, mr);
    for (keel::usize i = 0; i < plane; ++i)
        Lab.doubleDataMut()[i] = Lplane.elemAsDouble(i) * 100.0;
    // a, b * saturation.
    for (keel::usize i = 0; i < plane; ++i) {
        Lab.doubleDataMut()[plane + i]      *= saturation;
        Lab.doubleDataMut()[2 * plane + i] *= saturation;
    }
    Value rgb_out = lab2rgb(Lab, mr);  // H x W x 3 double in [0, 1] (clipped)
    return im2uint8(rgb_out, mr);
}

// ════════════════════════════════════════════════════════════════════
// labeloverlay — colour-overlay a label / mask matrix on a 2-D image
// ════════════════════════════════════════════════════════════════════
//
// Algorithm transliterated verbatim from MATLAB R2025b
//   toolbox/images/images/labeloverlay.m
//   toolbox/images/images/+images/+internal/LabelColormapHelper.m
//   toolbox/images/images/+images/+internal/labeloverlayalgo.m
//   toolbox/images/images/+images/+internal/greedyGraphColoring.m
//
// Pipeline:
//   1. im2single(A) → A in [0,1]; replicate grayscale to RGB.
//   2. maxLabel = max(L(:)), totalLabels = maxLabel + 1.
//   3. Resolve cmap:
//        - named string → feval(name, totalLabels)
//        - numeric Nx3 → as-is
//   4. ColorAssignment:
//        - auto: noshuffle if numeric, shuffle if string
//        - shuffle: cmap = cmap(randperm(N), :)
//                   randperm uses MATLAB MT19937 seed-0 (rng('default')).
//        - contrasting-neighbors: greedy BFS graph colouring on 8-conn.
//   5. alphaVal = 1 - transparency. Build alphamap of length N:
//        - if 0 ∈ included: alphamap(included+1) = alphaVal
//        - else: alphamap(included) = alphaVal; cmap = [cmap(1,:);cmap];
//                alphamap = [0, alphamap].
//   6. Per pixel: B(r,c,ch) = (1-α[L+1]) · A(r,c,ch) + α[L+1] · cmap[L+1,ch]
//   7. im2uint8(B) → uint8 H×W×3.
//
// Bit-identical with MATLAB on `noshuffle` and on `shuffle` for all
// probed seeds (default = 0). `contrasting-neighbors` is bit-identical
// where MATLAB's greedy BFS visits nodes in the same order (the
// canonical column-major BFS does on integer-typed L matrices).

// [Phase 2b] colormap/label2rgb cluster promoted from an anonymous namespace
// to numkit::image scope so color_extras_reg.cpp can reuse it (decls in
// color_extras_detail.hpp). Internal-only — not public API.

// ── jet(N) — MATLAB-canonical jet colormap ────────────────────────
// Reproduces MATLAB R2025b graphics/jet.m exactly for N >= 1.
//
//   n = ceil(m/4);
//   u = [(1:n)/n; ones(n-1,1); (n:-1:1)/n];
//   g = ceil(n/2) - (mod(m,4)==1) + (1:length(u))';
//   r = g + n;   b = g - n;
//   g(g>m)=[]; r(r>m)=[]; b(b<1)=[];
//   J(:,:) = 0;
//   J(r,1) = u(1:length(r));
//   J(g,2) = u(1:length(g));
//   J(b,3) = u(end-length(b)+1:end);
Value jet_colormap(keel::i32 m, keel::Allocator *mr)
{
    if (m < 1) m = 1;
    Value out = Value::matrix(keel::as<keel::usize>(m), 3,
                              ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    keel::fill(od, od + 3 * keel::as<keel::usize>(m), 0.0);
    const keel::i32 n = (m + 3) / 4;            // ceil(m/4)
    const keel::i32 ulen = n + (n - 1) + n;      // length(u)
    auto u_at = [&](keel::i32 i) -> keel::f64 {     // 1-based
        if (i <= n) return keel::f64(i) / keel::f64(n);
        if (i <= 2 * n - 1) return 1.0;
        return keel::f64(2 * n - 1 + n - (i - 1)) / keel::f64(n);  // (n:-1:1)/n
    };
    const keel::i32 gbase = (n + 1) / 2 - (m % 4 == 1 ? 1 : 0);   // 1-based base
    // For each k=1..ulen: g_k = gbase + k, r_k = g_k + n, b_k = g_k - n.
    keel::i32 rcount = 0, gcount = 0, bcount = 0;
    for (keel::i32 k = 1; k <= ulen; ++k) {
        const keel::i32 g = gbase + k;
        const keel::i32 r = g + n;
        const keel::i32 b = g - n;
        if (r >= 1 && r <= m) {
            // u(1..length(r)) — k-th valid r maps to u(rcount+1)
            ++rcount;
            od[0 * keel::as<keel::usize>(m) + (r - 1)] = u_at(rcount);
        }
        if (g >= 1 && g <= m) {
            ++gcount;
            od[1 * keel::as<keel::usize>(m) + (g - 1)] = u_at(gcount);
        }
        if (b >= 1 && b <= m) ++bcount;  // tally only; assigned below
    }
    // b uses u(end-length(b)+1 : end) — i.e. the last bcount entries.
    keel::i32 bsofar = 0;
    for (keel::i32 k = 1; k <= ulen; ++k) {
        const keel::i32 g = gbase + k;
        const keel::i32 b = g - n;
        if (b >= 1 && b <= m) {
            ++bsofar;
            od[2 * keel::as<keel::usize>(m) + (b - 1)] =
                u_at(ulen - bcount + bsofar);
        }
    }
    return out;
}

// ── hsv(N) — MATLAB-canonical HSV colormap ────────────────────────
// hsv2rgb([h s v]) with h = (0:N-1)'/N, s=v=1.
Value hsv_colormap(keel::i32 m, keel::Allocator *mr)
{
    if (m < 1) m = 1;
    Value out = Value::matrix(keel::as<keel::usize>(m), 3,
                              ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    for (keel::i32 i = 0; i < m; ++i) {
        const keel::f64 h = keel::f64(i) / keel::f64(m);     // [0,1)
        // HSV→RGB, with s=v=1.
        const keel::f64 hh = h * 6.0;
        const keel::i32 sect = keel::as<keel::i32>(keel::floor(hh));
        const keel::f64 f = hh - sect;
        keel::f64 r = 0, g = 0, b = 0;
        switch (sect % 6) {
            case 0: r = 1;     g = f;     b = 0;     break;
            case 1: r = 1 - f; g = 1;     b = 0;     break;
            case 2: r = 0;     g = 1;     b = f;     break;
            case 3: r = 0;     g = 1 - f; b = 1;     break;
            case 4: r = f;     g = 0;     b = 1;     break;
            case 5: r = 1;     g = 0;     b = 1 - f; break;
        }
        od[0 * keel::as<keel::usize>(m) + i] = r;
        od[1 * keel::as<keel::usize>(m) + i] = g;
        od[2 * keel::as<keel::usize>(m) + i] = b;
    }
    return out;
}

// ── parula(N) — MATLAB-canonical parula colormap ──────────────────
// 64-row reference table from MATLAB R2025b parula.m, linearly
// interpolated for any N. Reference table copy-pasted verbatim from
// MATLAB output: type(jet) replaced with type(parula).
static const keel::f64 kParulaRef[64 * 3] = {
    0.2422, 0.1504, 0.6603, 0.2504, 0.1650, 0.7076, 0.2578, 0.1818, 0.7511,
    0.2647, 0.1978, 0.7952, 0.2706, 0.2147, 0.8364, 0.2751, 0.2342, 0.8710,
    0.2783, 0.2559, 0.8991, 0.2803, 0.2782, 0.9221, 0.2813, 0.3006, 0.9414,
    0.2810, 0.3228, 0.9579, 0.2795, 0.3447, 0.9717, 0.2760, 0.3667, 0.9829,
    0.2699, 0.3892, 0.9906, 0.2602, 0.4123, 0.9952, 0.2440, 0.4358, 0.9988,
    0.2206, 0.4603, 0.9973, 0.1963, 0.4847, 0.9892, 0.1834, 0.5074, 0.9798,
    0.1786, 0.5289, 0.9682, 0.1764, 0.5499, 0.9520, 0.1687, 0.5703, 0.9359,
    0.1540, 0.5902, 0.9218, 0.1460, 0.6091, 0.9079, 0.1380, 0.6276, 0.8973,
    0.1248, 0.6459, 0.8883, 0.1113, 0.6635, 0.8763, 0.0952, 0.6798, 0.8598,
    0.0689, 0.6948, 0.8394, 0.0297, 0.7082, 0.8163, 0.0036, 0.7203, 0.7917,
    0.0067, 0.7312, 0.7660, 0.0433, 0.7411, 0.7394, 0.0964, 0.7500, 0.7120,
    0.1408, 0.7584, 0.6842, 0.1717, 0.7670, 0.6554, 0.1938, 0.7758, 0.6251,
    0.2161, 0.7843, 0.5923, 0.2470, 0.7918, 0.5567, 0.2906, 0.7973, 0.5188,
    0.3406, 0.8008, 0.4789, 0.3909, 0.8029, 0.4354, 0.4456, 0.8024, 0.3909,
    0.5044, 0.7993, 0.3480, 0.5616, 0.7942, 0.3045, 0.6174, 0.7876, 0.2612,
    0.6720, 0.7793, 0.2227, 0.7242, 0.7698, 0.1910, 0.7738, 0.7598, 0.1646,
    0.8203, 0.7498, 0.1535, 0.8634, 0.7406, 0.1596, 0.9035, 0.7330, 0.1774,
    0.9393, 0.7288, 0.2100, 0.9728, 0.7298, 0.2394, 0.9956, 0.7434, 0.2371,
    0.9970, 0.7659, 0.2199, 0.9952, 0.7893, 0.2028, 0.9892, 0.8136, 0.1885,
    0.9786, 0.8386, 0.1766, 0.9676, 0.8639, 0.1643, 0.9610, 0.8890, 0.1537,
    0.9597, 0.9135, 0.1423, 0.9628, 0.9373, 0.1265, 0.9691, 0.9606, 0.1064,
    0.9769, 0.9839, 0.0805
};

Value parula_colormap(keel::i32 m, keel::Allocator *mr)
{
    if (m < 1) m = 1;
    Value out = Value::matrix(keel::as<keel::usize>(m), 3,
                              ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    if (m == 1) {
        od[0] = kParulaRef[0];
        od[1] = kParulaRef[1];
        od[2] = kParulaRef[2];
        return out;
    }
    // Linear interp from 64-row reference. MATLAB uses interp1 with
    // 'linear' between sample positions (0:63)/63 → (0:m-1)/(m-1).
    for (keel::i32 i = 0; i < m; ++i) {
        const keel::f64 t = keel::f64(i) / keel::f64(m - 1);
        const keel::f64 src = t * 63.0;
        keel::i32 lo = keel::as<keel::i32>(keel::floor(src));
        keel::i32 hi = lo + 1;
        if (hi > 63) { hi = 63; lo = 62; }
        if (lo < 0)  { lo = 0;  hi = 1;  }
        const keel::f64 a = src - lo;
        for (keel::i32 ch = 0; ch < 3; ++ch) {
            const keel::f64 v = (1 - a) * kParulaRef[lo * 3 + ch]
                           + a       * kParulaRef[hi * 3 + ch];
            od[ch * keel::as<keel::usize>(m) + i] = v;
        }
    }
    return out;
}

// ── Other named colormaps — minimal set needed for labeloverlay ───
// gray, hot, cool, bone — all are simple closed-form linear ramps.
Value gray_colormap(keel::i32 m, keel::Allocator *mr)
{
    if (m < 1) m = 1;
    Value out = Value::matrix(keel::as<keel::usize>(m), 3,
                              ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    for (keel::i32 i = 0; i < m; ++i) {
        const keel::f64 g = (m == 1) ? 0.0 : keel::f64(i) / keel::f64(m - 1);
        od[0 * m + i] = g;
        od[1 * m + i] = g;
        od[2 * m + i] = g;
    }
    return out;
}

Value hot_colormap(keel::i32 m, keel::Allocator *mr)
{
    // From MATLAB R2025b graphics/hot.m:
    //   n = fix(3/8*m);  r = [(1:n)'/n; ones(m-n,1)];
    //   g = [zeros(n,1); (1:n)'/n; ones(m-2*n,1)];
    //   b = [zeros(2*n,1); (1:m-2*n)'/(m-2*n)];
    if (m < 1) m = 1;
    Value out = Value::matrix(keel::as<keel::usize>(m), 3,
                              ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    keel::fill(od, od + 3 * m, 0.0);
    const keel::i32 n = (3 * m) / 8;
    for (keel::i32 i = 0; i < m; ++i) {
        keel::f64 r, g, b;
        if (i < n)               r = keel::f64(i + 1) / keel::f64(n);
        else                     r = 1.0;
        if (i < n)               g = 0.0;
        else if (i < 2 * n)      g = keel::f64(i - n + 1) / keel::f64(n);
        else                     g = 1.0;
        if (i < 2 * n)           b = 0.0;
        else                     b = keel::f64(i - 2 * n + 1) / keel::f64(m - 2 * n);
        od[0 * m + i] = r;
        od[1 * m + i] = g;
        od[2 * m + i] = b;
    }
    return out;
}

Value cool_colormap(keel::i32 m, keel::Allocator *mr)
{
    // From graphics/cool.m: r = (0:m-1)'/(m-1); g = 1-r; b = ones(m,1).
    if (m < 1) m = 1;
    Value out = Value::matrix(keel::as<keel::usize>(m), 3,
                              ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    for (keel::i32 i = 0; i < m; ++i) {
        const keel::f64 r = (m == 1) ? 0.0 : keel::f64(i) / keel::f64(m - 1);
        od[0 * m + i] = r;
        od[1 * m + i] = 1.0 - r;
        od[2 * m + i] = 1.0;
    }
    return out;
}

Value bone_colormap(keel::i32 m, keel::Allocator *mr)
{
    // bone(m) = (7*gray(m) + hsv-blue ramp)/8 in MATLAB. Specifically:
    //   bone = (7*gray(m) + [zeros(...); (1:n)'/n; ones(...); ones(...)])/8
    // Use the form: bone(m) = (7*gray(m) + flipud(hot(m))(:,[3 2 1]))/8.
    if (m < 1) m = 1;
    Value g = gray_colormap(m, mr);
    Value h = hot_colormap(m, mr);
    Value out = Value::matrix(keel::as<keel::usize>(m), 3,
                              ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    const keel::f64 *gd = g.doubleData();
    const keel::f64 *hd = h.doubleData();
    for (keel::i32 i = 0; i < m; ++i) {
        // flipud(hot) with channel swap [3 2 1]:
        const keel::i32 j = m - 1 - i;
        const keel::f64 r2 = hd[2 * m + j];  // B of flipped → R
        const keel::f64 g2 = hd[1 * m + j];
        const keel::f64 b2 = hd[0 * m + j];  // R of flipped → B
        od[0 * m + i] = (7.0 * gd[0 * m + i] + r2) / 8.0;
        od[1 * m + i] = (7.0 * gd[1 * m + i] + g2) / 8.0;
        od[2 * m + i] = (7.0 * gd[2 * m + i] + b2) / 8.0;
    }
    return out;
}

// Dispatch a named MATLAB colormap. Throws for anything not in the
// supported set.
Value resolve_named_colormap(keel::StringRef name, keel::i32 N,
                             keel::Allocator *mr)
{
    if (name.equals_ignore_ascii_case("jet"))     return jet_colormap(N, mr);
    if (name.equals_ignore_ascii_case("hsv"))     return hsv_colormap(N, mr);
    if (name.equals_ignore_ascii_case("parula"))  return parula_colormap(N, mr);
    if (name.equals_ignore_ascii_case("gray") ||
        name.equals_ignore_ascii_case("grey"))
                                                  return gray_colormap(N, mr);
    if (name.equals_ignore_ascii_case("hot"))     return hot_colormap(N, mr);
    if (name.equals_ignore_ascii_case("cool"))    return cool_colormap(N, mr);
    if (name.equals_ignore_ascii_case("bone"))    return bone_colormap(N, mr);
    throw Error(keel::format("labeloverlay: unsupported colormap name '{}' "
                             "(supported: jet, hsv, parula, gray, hot, cool, bone)",
                             name),
                0, 0, "labeloverlay", "", "numkit:labeloverlay:cmapName");
}

// ── randperm(N) using MATLAB MT19937 with rng('default') ──────────
// Bit-identical with `rng('default'); randperm(N)` in MATLAB R2025b.
// Strategy: MATLAB's randperm(N) for N == k internally does
// `[~, p] = sort(rand(1, N))`. Using genRes53() draws, the indices
// after a stable sort give the permutation.
keel::Array<keel::i32> matlab_default_randperm(keel::i32 N)
{
    ::numkit::ops::MatlabMT19937 rng;   // seed 0 → state[0] = 5489
    keel::Array<keel::f64> u(N);
    for (keel::i32 i = 0; i < N; ++i) u[i] = rng.genRes53();
    keel::Array<keel::i32> p(N);
    for (keel::i32 i = 0; i < N; ++i) p[i] = i;          // 0-based for sort
    keel::stable_sort(p.begin(), p.end(),
                     [&](keel::i32 a, keel::i32 b) { return u[a] < u[b]; });
    for (keel::i32 &x : p) x += 1;                       // → 1-based result
    return p;
}

// Permute rows of an Nx3 cmap by `perm` (1-based).
Value permute_cmap_rows(const Value &cmap, keel::ArrayRef<keel::i32> perm,
                        keel::Allocator *mr)
{
    const keel::usize N = cmap.dims().rows();
    Value out = Value::matrix(N, 3, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    const keel::f64 *cd = cmap.doubleData();
    for (keel::usize i = 0; i < N; ++i) {
        const keel::usize src = keel::as<keel::usize>(perm[i] - 1);
        for (keel::i32 ch = 0; ch < 3; ++ch)
            od[ch * N + i] = cd[ch * N + src];
    }
    return out;
}

Value greedy_graph_coloring(const Value &L_in, const Value &cmap, keel::Allocator *mr)
{
    const keel::usize H = L_in.dims().rows();
    const keel::usize W = L_in.dims().cols();
    const keel::usize N = cmap.dims().rows();
    keel::Array<keel::i32> labs(H * W, mr);
    for (keel::usize i = 0; i < H * W; ++i)
        labs[i] = keel::as<keel::i32>(L_in.elemAsDouble(i));
    const keel::f64 *cd = cmap.doubleData();
    keel::Array<keel::f64> out_r(mr), out_g(mr), out_b(mr);
    (void)keel::image::greedy_graph_coloring_2d(
        labs.as_ref(), H, W,
        keel::ArrayRef<keel::f64>(cd, N),
        keel::ArrayRef<keel::f64>(cd + N, N),
        keel::ArrayRef<keel::f64>(cd + 2 * N, N),
        out_r, out_g, out_b, mr);

    Value out = Value::matrix(N, 3, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    if (N > 0) {
        keel::copy_n(od,         out_r.data(), N);
        keel::copy_n(od + N,     out_g.data(), N);
        keel::copy_n(od + 2 * N, out_b.data(), N);
    }
    return out;
}

// ── Lowercase helper ──────────────────────────────────────────────
keel::String lower(keel::StringRef s)
{
    keel::String lo;
    lo.reserve(s.size());
    for (char ch : s)
        lo.push_back(keel::to_ascii_lower(ch));
    return lo;
}

// Parse a MATLAB color-spec string into an RGB triplet in [0,1], matching
// label2rgb's parseZerocolor (single-letter ColorSpec or full color name).
// Returns false if the string is not a recognized color.
bool parseColorSpec(keel::StringRef name, keel::f64 rgb[3])
{
    const keel::String lo = lower(name);
    auto set = [&](keel::f64 r, keel::f64 g, keel::f64 b) {
        rgb[0] = r; rgb[1] = g; rgb[2] = b; return true;
    };
    if (lo == "y" || lo == "yellow")  return set(1, 1, 0);
    if (lo == "m" || lo == "magenta") return set(1, 0, 1);
    if (lo == "c" || lo == "cyan")    return set(0, 1, 1);
    if (lo == "r" || lo == "red")     return set(1, 0, 0);
    if (lo == "g" || lo == "green")   return set(0, 1, 0);
    if (lo == "b" || lo == "blue")    return set(0, 0, 1);
    if (lo == "w" || lo == "white")   return set(1, 1, 1);
    if (lo == "k" || lo == "black")   return set(0, 0, 0);
    return false;
}

// [Phase 2b] end of promoted colormap/label2rgb cluster

namespace {

void labeloverlay_argcheck(const Value &A_in, const Value &L_in,
                           keel::f64 transparency)
{
    const auto &dA = A_in.dims();
    const keel::usize H = dA.rows();
    const keel::usize W = dA.cols();
    if (H == 0 || W == 0)
        throw Error("labeloverlay: A must be nonempty",
                    0, 0, "labeloverlay", "", "numkit:labeloverlay:emptyA");
    const bool isRGB = dA.is3D() && dA.pages() == 3;
    const bool isGray = !dA.is3D() || (dA.is3D() && dA.pages() == 1);
    if (!isRGB && !isGray)
        throw Error("labeloverlay: A must be grayscale (H×W) or RGB (H×W×3)",
                    0, 0, "labeloverlay", "", "numkit:labeloverlay:shapeA");

    const auto &dL = L_in.dims();
    if (dL.is3D() && dL.pages() != 1)
        throw Error("labeloverlay: L must be 2-D",
                    0, 0, "labeloverlay", "", "numkit:labeloverlay:shapeL");
    if (dL.rows() != H || dL.cols() != W)
        throw Error("labeloverlay: size(L) must match size(A,1:2)",
                    0, 0, "labeloverlay", "", "numkit:labeloverlay:sizeMismatch");

    if (!keel::is_finite(transparency) || transparency < 0 || transparency > 1)
        throw Error("labeloverlay: Transparency must be in [0, 1]",
                    0, 0, "labeloverlay", "",
                    "numkit:labeloverlay:transparency");
}

} // anonymous

Value labeloverlay(const Value &A_in, const Value &L_in,
                   const Value &colormap,
                   keel::StringRef color_assignment,
                   const Value &included_labels,
                   keel::f64 transparency,
                   keel::Allocator *mr)
{
    labeloverlay_argcheck(A_in, L_in, transparency);
    // ── Validate A ────────────────────────────────────────────────
    const auto &dA = A_in.dims();
    const keel::usize H = dA.rows();
    const keel::usize W = dA.cols();
    if (H == 0 || W == 0)
        throw Error("labeloverlay: A must be nonempty",
                    0, 0, "labeloverlay", "", "numkit:labeloverlay:emptyA");
    const bool isRGB = dA.is3D() && dA.pages() == 3;
    const bool isGray = !dA.is3D() || (dA.is3D() && dA.pages() == 1);
    if (!isRGB && !isGray)
        throw Error("labeloverlay: A must be grayscale (H×W) or RGB (H×W×3)",
                    0, 0, "labeloverlay", "", "numkit:labeloverlay:shapeA");

    // ── Validate L ────────────────────────────────────────────────
    const auto &dL = L_in.dims();
    if (dL.is3D() && dL.pages() != 1)
        throw Error("labeloverlay: L must be 2-D",
                    0, 0, "labeloverlay", "", "numkit:labeloverlay:shapeL");
    if (dL.rows() != H || dL.cols() != W)
        throw Error("labeloverlay: size(L) must match size(A,1:2)",
                    0, 0, "labeloverlay", "", "numkit:labeloverlay:sizeMismatch");

    // Logical → 0/1 integer label matrix. Already a numeric label
    // matrix? Accept as-is, but validate non-negative integer.
    const keel::usize plane = H * W;
    keel::i32 maxLabel = 0;
    for (keel::usize i = 0; i < plane; ++i) {
        const keel::f64 v = L_in.elemAsDouble(i);
        if (v < 0 || keel::floor(v) != v || !keel::is_finite(v))
            throw Error("labeloverlay: L must be non-negative integer-valued",
                        0, 0, "labeloverlay", "", "numkit:labeloverlay:badL");
        const keel::i32 iv = keel::as<keel::i32>(v);
        if (iv > maxLabel) maxLabel = iv;
    }
    const keel::i32 totalLabels = maxLabel + 1;

    // ── Transparency ──────────────────────────────────────────────
    if (!keel::is_finite(transparency) || transparency < 0 || transparency > 1)
        throw Error("labeloverlay: Transparency must be in [0, 1]",
                    0, 0, "labeloverlay", "",
                    "numkit:labeloverlay:transparency");
    const keel::f64 alphaVal = 1.0 - transparency;

    // ── Resolve colormap ──────────────────────────────────────────
    // colormap arg can be: empty (use 'jet') / numeric Nx3 / string name.
    Value cmap;
    bool cmap_was_string = false;
    if (colormap.isEmpty()) {
        cmap = jet_colormap(totalLabels, mr);
        cmap_was_string = true;
    } else if (colormap.isChar() || colormap.isString()) {
        cmap = resolve_named_colormap(colormap.toString(), totalLabels, mr);
        cmap_was_string = true;
    } else {
        if (colormap.dims().cols() != 3 || colormap.dims().is3D())
            throw Error("labeloverlay: Colormap must be an Nx3 numeric array",
                        0, 0, "labeloverlay", "",
                        "numkit:labeloverlay:cmapShape");
        // Convert to DOUBLE (matches MATLAB's normalizeColormap → single
        // then double for indexing). Stays bit-identical because the
        // user-supplied array IS the source of truth.
        cmap = Value::matrix(colormap.dims().rows(), 3,
                             ValueType::DOUBLE, mr);
        for (keel::usize i = 0; i < cmap.numel(); ++i)
            cmap.doubleDataMut()[i] = colormap.elemAsDouble(i);
    }

    // ── Resolve ColorAssignment ───────────────────────────────────
    keel::String ca = lower(color_assignment.empty() ? keel::StringRef("auto")
                                                     : color_assignment);
    if (ca == "auto") {
        ca.assign(cmap_was_string ? "shuffle" : "noshuffle");
    } else if (ca != "shuffle" && ca != "noshuffle"
               && ca != "contrasting-neighbors") {
        throw Error("labeloverlay: ColorAssignment must be auto / shuffle / "
                    "noshuffle / contrasting-neighbors",
                    0, 0, "labeloverlay", "",
                    "numkit:labeloverlay:colorAssignment");
    }

    if (ca == "shuffle") {
        const keel::i32 N = keel::as<keel::i32>(cmap.dims().rows());
        keel::Array<keel::i32> p = matlab_default_randperm(N);
        cmap = permute_cmap_rows(cmap, p, mr);
    } else if (ca == "contrasting-neighbors") {
        cmap = greedy_graph_coloring(L_in, cmap, mr);
    }

    // ── Resolve IncludedLabels ────────────────────────────────────
    keel::Array<keel::i32> included;
    if (included_labels.isEmpty()) {
        for (keel::i32 k = 1; k <= maxLabel; ++k) included.push_back(k);
    } else {
        const keel::usize M = included_labels.numel();
        included.reserve(M);
        for (keel::usize i = 0; i < M; ++i) {
            const keel::f64 v = included_labels.elemAsDouble(i);
            if (v < 0 || keel::floor(v) != v || !keel::is_finite(v))
                throw Error("labeloverlay: IncludedLabels must be "
                            "non-negative integers",
                            0, 0, "labeloverlay", "",
                            "numkit:labeloverlay:includedBad");
            included.push_back(keel::as<keel::i32>(v));
        }
        for (keel::i32 v : included)
            if (v > maxLabel)
                throw Error("labeloverlay: IncludedLabels exceeds max label",
                            0, 0, "labeloverlay", "",
                            "numkit:labeloverlay:includedRange");
    }
    if (included.empty()) {
        // MATLAB behaviour: pass A through im2uint8.
        return im2uint8(A_in, mr);
    }
    if (keel::as<keel::i32>(cmap.dims().rows()) <
        keel::as<keel::i32>(included.size()))
        throw Error("labeloverlay: Colormap has fewer rows than the number "
                    "of labels",
                    0, 0, "labeloverlay", "",
                    "numkit:labeloverlay:cmapTooSmall");

    // ── Build alphamap ────────────────────────────────────────────
    bool zero_in_included = false;
    for (keel::i32 v : included) if (v == 0) { zero_in_included = true; break; }
    keel::Array<keel::f64> alphamap(cmap.dims().rows(), 0.0);
    if (zero_in_included) {
        for (keel::i32 k : included) {
            const keel::i32 idx = k + 1;
            if (idx - 1 < keel::as<keel::i32>(alphamap.size()))
                alphamap[idx - 1] = alphaVal;
        }
    } else {
        // alphamap(included) = alphaVal
        for (keel::i32 k : included) {
            if (k - 1 < keel::as<keel::i32>(alphamap.size()))
                alphamap[k - 1] = alphaVal;
        }
        // cmap = [cmap(1,:); cmap]; alphamap = [0, alphamap].
        const keel::usize N = cmap.dims().rows();
        Value c2 = Value::matrix(N + 1, 3, ValueType::DOUBLE, mr);
        const keel::f64 *cd = cmap.doubleData();
        keel::f64 *cd2 = c2.doubleDataMut();
        for (keel::i32 ch = 0; ch < 3; ++ch) {
            cd2[ch * (N + 1) + 0] = cd[ch * N + 0];
            for (keel::usize i = 0; i < N; ++i)
                cd2[ch * (N + 1) + (i + 1)] = cd[ch * N + i];
        }
        cmap = keel::move(c2);
        keel::Array<keel::f64> shifted(alphamap.size() + 1, 0.0);
        for (keel::usize i = 0; i < alphamap.size(); ++i)
            shifted[i + 1] = alphamap[i];
        alphamap = keel::move(shifted);
    }

    // ── Convert A to single in [0, 1] via im2single equivalent ────
    // We promote integer/logical inputs by class range; double/single
    // values are taken at face value (no rescale).
    auto a_pixel = [&](keel::usize r, keel::usize c, keel::i32 ch) -> keel::f64 {
        keel::usize off;
        if (isRGB) {
            off = keel::as<keel::usize>(ch) * H * W + c * H + r;
        } else {
            off = c * H + r;     // gray: same value across ch
        }
        const keel::f64 raw = A_in.elemAsDouble(off);
        switch (A_in.type()) {
            case ValueType::UINT8:   return raw / 255.0;
            case ValueType::UINT16:  return raw / 65535.0;
            case ValueType::INT16: {
                // im2single for int16: (x + 32768) / 65535.
                return (raw + 32768.0) / 65535.0;
            }
            case ValueType::LOGICAL: return raw == 0.0 ? 0.0 : 1.0;
            default:                  return raw;   // double / single
        }
    };

    // ── Pixel-wise blend → uint8 ──────────────────────────────────
    Value out = Value::matrix3d(H, W, 3, ValueType::UINT8, mr);
    keel::u8 *od = out.uint8DataMut();
    const keel::f64 *cd = cmap.doubleData();
    const keel::usize Nc = cmap.dims().rows();
    auto sat = [](keel::f64 v) -> keel::u8 {
        v = keel::round(v * 255.0);
        if (v < 0)   v = 0;
        if (v > 255) v = 255;
        return keel::as<keel::u8>(v);
    };
    for (keel::usize c = 0; c < W; ++c)
        for (keel::usize r = 0; r < H; ++r) {
            const keel::i32 lab = keel::as<keel::i32>(L_in.elemAsDouble(c * H + r));
            // MATLAB picks `alphamap(lab+1)` and `cmap(lab+1, :)` (1-based)
            // in both branches:
            //   • zero IN included: alphamap = zeros(N); alphamap(lab+1)=α.
            //     N == cmap rows (no prepend). C++ 0-based idx = lab.
            //   • zero NOT in included: alphamap = [0, alpha-padded];
            //     cmap = [cmap(1,:); cmap]. Both grow by 1; index still
            //     `lab + 1` in MATLAB ⇒ `lab` in C++.
            keel::i32 aidx = lab;
            if (aidx < 0) aidx = 0;
            if (aidx >= keel::as<keel::i32>(alphamap.size()))
                aidx = keel::as<keel::i32>(alphamap.size()) - 1;
            const keel::f64 a = alphamap[aidx];
            for (keel::i32 ch = 0; ch < 3; ++ch) {
                const keel::f64 Apx = a_pixel(r, c, ch);
                const keel::f64 Cpx = cd[ch * Nc + aidx];
                const keel::f64 blended = (1.0 - a) * Apx + a * Cpx;
                od[keel::as<keel::usize>(ch) * H * W + c * H + r] =
                    sat(blended);
            }
        }
    return out;
}
} // namespace numkit::image
