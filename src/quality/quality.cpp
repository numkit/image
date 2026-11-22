// toolboxes/image/src/quality/quality.cpp

#include <keel/array.h>
#include <keel/algo.h>
#include <keel/math.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <keel/image.h>
#include <numkit/image/quality/quality.hpp>
#include <numkit/image/internal/type_gates.hpp>

#include <numkit/image/filter/filter.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <cstdint>

namespace numkit::image {

namespace {

keel::f64 class_peak(ValueType t) {
    switch (t) {
        case ValueType::UINT8:  return 255.0;
        case ValueType::UINT16: return 65535.0;
        case ValueType::INT16:  return 32767.0;
        default:                return 1.0;
    }
}

// Element-wise difference squared, summed. Complex inputs use |zA−zB|²
// (probed R2025b: immse computes the absolute-square error, not the
// real-part difference).
keel::f64 sum_squared_error(const Value &A, const Value &B) {
    const keel::usize N = A.numel();
    keel::f64 s = 0.0;
    if (A.isComplex() || B.isComplex()) {
        // Either side may be real (complex pairs with double — probed).
        const bool ac = A.isComplex(), bc = B.isComplex();
        for (keel::usize i = 0; i < N; ++i) {
            const auto za = ac ? A.complexData()[i]
                               : keel::c64(A.elemAsDouble(i), 0.0);
            const auto zb = bc ? B.complexData()[i]
                               : keel::c64(B.elemAsDouble(i), 0.0);
            const auto d = za - zb;
            s += d.real() * d.real() + d.imag() * d.imag();
        }
        return s;
    }
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 d = A.elemAsDouble(i) - B.elemAsDouble(i);
        s += d * d;
    }
    return s;
}

void immse_argcheck(const Value &A, const Value &B) {
    internal::requireImageClassAllowComplex(A, "immse", "A", 1,
                                internal::ImgClasses::FloatInt);
    internal::requireImageClassAllowComplex(B, "immse", "B", 2,
                                internal::ImgClasses::FloatInt);
    const auto eff = [](const Value &v) {
        return v.type() == ValueType::COMPLEX ? ValueType::DOUBLE
                                              : v.type();
    };
    if (eff(A) != eff(B)) {
        throw Error(keel::format("immse: A and B must have the same class (got {} and {}).",
                                 mtypeName(A.type()), mtypeName(B.type())),
                    0, 0, "immse", "", "numkit:immse:differentClass");
    }
    if (A.numel() != B.numel())
        throw Error("immse: A and B must have the same number of elements",
                    0, 0, "immse", "", "numkit:immse:size");
}

void psnr_argcheck(const Value &A, const Value &B) {
    internal::requireImageClass(A, "psnr", "A", 1,
                                internal::ImgClasses::FloatInt);
    internal::requireImageClass(B, "psnr", "B", 2,
                                internal::ImgClasses::FloatInt);
    internal::requireSameClassPair(A, B, "psnr");
    if (A.numel() != B.numel())
        throw Error("psnr: A and B must have the same number of elements",
                    0, 0, "psnr", "", "numkit:psnr:size");
}

void ssim_argcheck(const Value &A, const Value &B) {
    internal::requireImageClass(A, "ssim", "A", 1,
                                internal::ImgClasses::FloatInt);
    internal::requireImageClass(B, "ssim", "B", 2,
                                internal::ImgClasses::FloatInt);
    internal::requireSameClassPair(A, B, "ssim");
    if (A.numel() != B.numel())
        throw Error("ssim: A and B must have the same number of elements",
                    0, 0, "ssim", "", "numkit:ssim:size");
}

void corr2_argcheck(const Value &A, const Value &B) {
    if (B.numel() != A.numel())
        throw Error("corr2: A and B must have the same number of elements",
                    0, 0, "corr2", "", "numkit:corr2:size");
}

} // anonymous

Value immse(const Value &A, const Value &B, keel::Allocator *mr) {
    immse_argcheck(A, B);
    const keel::f64 mse = (A.numel() > 0)
                       ? sum_squared_error(A, B) / keel::f64(A.numel())
                       : 0.0;
    return Value::scalar(mse, mr);
}

Value psnr(const Value &A, const Value &B, keel::f64 peak, keel::Allocator *mr) {
    psnr_argcheck(A, B);
    if (keel::is_nan(peak)) peak = class_peak(A.type());
    if (A.numel() == 0) return Value::scalar(keel::Limits<keel::f64>::infinity(), mr);
    const keel::f64 mse = sum_squared_error(A, B) / keel::f64(A.numel());
    if (mse == 0.0) return Value::scalar(keel::Limits<keel::f64>::infinity(), mr);
    const keel::f64 db = 10.0 * keel::log10(peak * peak / mse);
    return Value::scalar(db, mr);
}

Value ssim(const Value &A, const Value &B, keel::Allocator *mr) {
    ssim_argcheck(A, B);
    const keel::usize H = A.dims().rows();
    const keel::usize W = A.dims().cols();
    if (H == 0 || W == 0) return Value::scalar(1.0, mr);

    auto unit = [&](const Value &X) {
        keel::Array<keel::f64> out(X.numel(), mr);
        const keel::f64 scale = 1.0 / class_peak(X.type());
        for (keel::usize i = 0; i < X.numel(); ++i)
            out[i] = X.elemAsDouble(i) * scale;
        return out;
    };
    auto Au = unit(A), Bu = unit(B);
    return Value::scalar(keel::image::ssim(Au, Bu, H, W, 1.0, mr), mr);
}

Value mean2(const Value &A, keel::Allocator *mr)
{
    const keel::usize N = A.numel();
    if (N == 0) return Value::scalar(keel::kNaN, mr);
    long double s = 0.0L; // keel-lint: ignore
    for (keel::usize i = 0; i < N; ++i) s += A.elemAsDouble(i);
    return Value::scalar(keel::as<keel::f64>(s / keel::as<long double>(N)), // keel-lint: ignore
                         mr);
}

Value std2(const Value &A, keel::Allocator *mr)
{
    const keel::usize N = A.numel();
    if (N == 0) return Value::scalar(keel::kNaN, mr);
    if (N == 1) return Value::scalar(0.0, mr);
    long double s = 0.0L; // keel-lint: ignore
    for (keel::usize i = 0; i < N; ++i) s += A.elemAsDouble(i);
    const long double mu = s / keel::as<long double>(N); // keel-lint: ignore
    long double v = 0.0L; // keel-lint: ignore
    for (keel::usize i = 0; i < N; ++i) {
        const long double d = A.elemAsDouble(i) - mu; // keel-lint: ignore
        v += d * d;
    }
    // std2 wraps std(I(:)) → sample std (normalize by N-1).
    v /= keel::as<long double>(N - 1); // keel-lint: ignore
    return Value::scalar(keel::as<keel::f64>(keel::sqrt((keel::f64)v)), mr);
}

// ════════════════════════════════════════════════════════════════════
// multissim — Multi-scale SSIM (Wang/Simoncelli/Bovik 2003)
// ════════════════════════════════════════════════════════════════════
//
// Algorithm transliterated verbatim from MATLAB R2025b
//   toolbox/images/images/multissim.m + algmultissim.m + parserMultissim.m
//
// Pipeline:
//   1. Promote integer inputs to single (int16 has -intmin offset).
//   2. Build Gaussian kernel: filtSize = 2·ceil(3σ) + 1, σ user-set.
//   3. For each scale i = 1..numScales-1:
//      a. compute SSIM map without luminance:
//           ssimmap = (2σxy + C2) / (σx² + σy² + C2)
//         clamp to ≤ 1.
//      b. score[i] = mean(ssimmap)^scaleWeights[i] (clamp base to 0
//         for non-integer exponent).
//      c. lowpass (2×2 box) + downsample by 2.
//   4. At coarsest scale, compute SSIM with luminance:
//        ssimmap = (2μxμy+C1)(2σxy+C2) / ((μx²+μy²+C1)(σx²+σy²+C2))
//        clamp to ≤ 1.
//      score[numScales] = mean(ssimmap)^scaleWeights[numScales].
//   5. final score = prod(score[1..numScales]).
//
// Reference: Wang, Z., Simoncelli, E.P., Bovik, A.C., "Multiscale
//   structural similarity for image quality assessment", Asilomar
//   Conference on Signals, Systems & Computers, 2003.

namespace {

keel::f64 class_dynamic_range(ValueType t)
{
    switch (t) {
        case ValueType::UINT8:  return 255.0;
        case ValueType::UINT16: return 65535.0;
        case ValueType::INT16:  return 65535.0;   // diff(int16 range)
        default:                return 1.0;       // single, double
    }
}

// Default ScaleWeights: fspecial('gaussian', [1 N], 1) — sample
// gaussian at positions (0..N-1) - (N-1)/2 with σ=1, normalised to
// sum to 1. For N=1 returns [1.0]. For N=5 returns
// [0.0545, 0.2442, 0.4026, 0.2442, 0.0545].
keel::Array<keel::f64> default_scale_weights(keel::i32 N)
{
    if (N < 1) N = 1;
    keel::Array<keel::f64> w(N);
    if (N == 1) { w[0] = 1.0; return w; }
    const keel::f64 half = (N - 1) / 2.0;
    keel::f64 s = 0.0;
    for (keel::i32 i = 0; i < N; ++i) {
        const keel::f64 x = i - half;
        w[i] = keel::exp(-0.5 * x * x);
        s += w[i];
    }
    for (keel::f64 &v : w) v /= s;
    return w;
}

// Convert input image to DOUBLE for stable computation. Integer
// inputs are promoted: uint8/uint16 → x (NOT divided — multissim
// uses the raw integer pixel values plus the DynamicRange scaling
// for C constants). int16 → x - intmin = x + 32768.
Value to_double_image(const Value &A, keel::Allocator *mr)
{
    const keel::usize H = A.dims().rows();
    const keel::usize W = A.dims().cols();
    Value out = Value::matrix(H, W, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    const ValueType t = A.type();
    for (keel::usize i = 0; i < H * W; ++i) {
        keel::f64 v = A.elemAsDouble(i);
        if (t == ValueType::INT16) v += 32768.0;
        // uint8 / uint16 / single / double all keep raw value.
        od[i] = v;
    }
    return out;
}

// 2×2 box lowpass (ones/4) with replicate boundary, anchor at the
// kernel TOP-LEFT (matches MATLAB's imfilter anchor convention for
// even-sized kernels: anchor = floor((sz+1)/2) − 1 = 0 in 0-based).
//
// Out(r,c) = 0.25 · (X(r,c) + X(r+1,c) + X(r,c+1) + X(r+1,c+1))
// with row r+1 / col c+1 clamped via replicate at the bottom/right edges.
Value lowpass_2x2(const Value &X, keel::Allocator *mr)
{
    const keel::usize H = X.dims().rows();
    const keel::usize W = X.dims().cols();
    Value out = Value::matrix(H, W, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    const keel::f64 *xd = X.doubleData();
    for (keel::usize c = 0; c < W; ++c) {
        const keel::usize cn = (c + 1 < W) ? (c + 1) : (W - 1);  // replicate
        for (keel::usize r = 0; r < H; ++r) {
            const keel::usize rn = (r + 1 < H) ? (r + 1) : (H - 1);
            od[c * H + r] = 0.25 * (xd[c  * H + r ]
                                  + xd[c  * H + rn]
                                  + xd[cn * H + r ]
                                  + xd[cn * H + rn]);
        }
    }
    return out;
}

// Take every other row/column (1:2:end semantics, 0-based: rows/cols
// 0, 2, 4, ...). Returns a new H' × W' Value.
Value downsample_2(const Value &X, keel::Allocator *mr)
{
    const keel::usize H = X.dims().rows();
    const keel::usize W = X.dims().cols();
    const keel::usize Ho = (H + 1) / 2;
    const keel::usize Wo = (W + 1) / 2;
    Value out = Value::matrix(Ho, Wo, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    const keel::f64 *xd = X.doubleData();
    for (keel::usize c = 0; c < Wo; ++c)
        for (keel::usize r = 0; r < Ho; ++r)
            od[c * Ho + r] = xd[(c * 2) * H + (r * 2)];
    return out;
}

// Compute per-pixel SSIM map for one scale. If include_luminance is
// false, the map is (2σxy+C2)/(σx²+σy²+C2). If true, full SSIM
// (Wang-Bovik Eq. 13) with the luminance term.
Value ssim_map_double(const Value &A, const Value &B,
                      const Value &gK, keel::f64 C1, keel::f64 C2,
                      bool include_luminance,
                      keel::Allocator *mr)
{
    const keel::usize H = A.dims().rows();
    const keel::usize W = A.dims().cols();
    auto box = [&](const Value &X) {
        return imfilter(X, gK, PadMode::Replicate, 0.0, /*full=*/false,
                        /*flip_kernel=*/false, mr);
    };
    Value mu_a = box(A);
    Value mu_b = box(B);
    // Pre-products.
    Value AA = Value::matrix(H, W, ValueType::DOUBLE, mr);
    Value BB = Value::matrix(H, W, ValueType::DOUBLE, mr);
    Value AB = Value::matrix(H, W, ValueType::DOUBLE, mr);
    const keel::f64 *ad = A.doubleData();
    const keel::f64 *bd = B.doubleData();
    keel::f64 *aa = AA.doubleDataMut();
    keel::f64 *bb = BB.doubleDataMut();
    keel::f64 *ab = AB.doubleDataMut();
    for (keel::usize i = 0; i < H * W; ++i) {
        const keel::f64 a = ad[i], b = bd[i];
        aa[i] = a * a;
        bb[i] = b * b;
        ab[i] = a * b;
    }
    Value sa2 = box(AA);
    Value sb2 = box(BB);
    Value sab = box(AB);

    Value out = Value::matrix(H, W, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    const keel::f64 *mua_d = mu_a.doubleData();
    const keel::f64 *mub_d = mu_b.doubleData();
    const keel::f64 *sa2_d = sa2.doubleData();
    const keel::f64 *sb2_d = sb2.doubleData();
    const keel::f64 *sab_d = sab.doubleData();
    for (keel::usize i = 0; i < H * W; ++i) {
        const keel::f64 mua = mua_d[i];
        const keel::f64 mub = mub_d[i];
        const keel::f64 mu_xy = mua * mub;
        const keel::f64 mux2 = mua * mua;
        const keel::f64 muy2 = mub * mub;
        keel::f64 vx = sa2_d[i] - mux2;
        keel::f64 vy = sb2_d[i] - muy2;
        if (vx < 0.0) vx = 0.0;
        if (vy < 0.0) vy = 0.0;
        const keel::f64 cov = sab_d[i] - mu_xy;
        const keel::f64 num2 = 2.0 * cov + C2;
        const keel::f64 den2 = vx + vy + C2;
        if (include_luminance) {
            const keel::f64 num1 = 2.0 * mu_xy + C1;
            const keel::f64 den1 = mux2 + muy2 + C1;
            od[i] = (den1 != 0.0 && den2 != 0.0)
                  ? (num1 * num2) / (den1 * den2) : 1.0;
        } else {
            od[i] = (den2 != 0.0) ? num2 / den2 : 1.0;
        }
        // Clamp to <= 1 per MATLAB.
        if (od[i] > 1.0) od[i] = 1.0;
    }
    return out;
}

// Cast a DOUBLE Value to SINGLE (or pass through for DOUBLE).
Value cast_score_value(keel::f64 v, ValueType originalClass,
                       keel::Allocator *mr)
{
    if (originalClass == ValueType::DOUBLE)
        return Value::scalar(v, mr);
    Value out = Value::matrix(1, 1, ValueType::SINGLE, mr);
    out.singleDataMut()[0] = keel::as<keel::f32>(v);
    return out;
}

void multissim_argcheck(const Value &A, const Value &Iref, keel::i32 num_scales, keel::f64 sigma)
{
    internal::requireImageClass(A, "multissim", "A", 1,
                                internal::ImgClasses::FloatInt);
    internal::requireImageClass(Iref, "multissim", "ref", 2,
                                internal::ImgClasses::FloatInt);
    internal::requireSameClassPair(A, Iref, "multissim");
    if (A.dims().rows() != Iref.dims().rows()
        || A.dims().cols() != Iref.dims().cols())
        throw Error("multissim: I and Iref must have the same size",
                    0, 0, "multissim", "", "numkit:multissim:size");
    if (A.type() != Iref.type())
        throw Error("multissim: I and Iref must have the same class",
                    0, 0, "multissim", "", "numkit:multissim:class");
    if (num_scales < 1)
        throw Error("multissim: NumScales must be a positive integer",
                    0, 0, "multissim", "", "numkit:multissim:numScales");
    if (!keel::is_finite(sigma) || sigma <= 0.0)
        throw Error("multissim: Sigma must be a positive scalar",
                    0, 0, "multissim", "", "numkit:multissim:sigma");
}

} // namespace

Value multissim(const Value &A, const Value &Iref,
                keel::i32 num_scales,
                keel::ArrayRef<keel::f64> scale_weights_in,
                keel::f64 sigma, keel::f64 dynamic_range_in,
                keel::Array<Value> *quality_maps_out,
                keel::Allocator *mr)
{
    multissim_argcheck(A, Iref, num_scales, sigma);

    const ValueType origClass = A.type();
    const keel::usize H = A.dims().rows();
    const keel::usize W = A.dims().cols();

    // Verify downsampleability.
    {
        keel::usize h = H, w = W;
        for (keel::i32 i = 1; i < num_scales; ++i) {
            if (h <= 1 || w <= 1)
                throw Error(keel::format("multissim: image too small for NumScales={}", num_scales),
                            0, 0, "multissim", "",
                            "numkit:multissim:tooSmall");
            h = (h + 1) / 2;
            w = (w + 1) / 2;
        }
    }

    // ── Resolve dynamic range / C constants ─────────────────────
    const keel::f64 L = (dynamic_range_in > 0.0)
                   ? dynamic_range_in
                   : class_dynamic_range(origClass);
    const keel::f64 C1 = (0.01 * L) * (0.01 * L);
    const keel::f64 C2 = (0.03 * L) * (0.03 * L);

    // ── Resolve / normalize scale weights ───────────────────────
    keel::Array<keel::f64> sw = scale_weights_in.empty()
        ? default_scale_weights(num_scales)
        : keel::Array<keel::f64>::from(scale_weights_in);
    if (keel::as<keel::i32>(sw.size()) != num_scales)
        throw Error("multissim: length(ScaleWeights) must equal NumScales",
                    0, 0, "multissim", "",
                    "numkit:multissim:swLen");
    keel::f64 swsum = 0.0;
    for (keel::f64 v : sw) {
        if (v < 0.0)
            throw Error("multissim: ScaleWeights must be non-negative",
                        0, 0, "multissim", "",
                        "numkit:multissim:swNeg");
        swsum += v;
    }
    if (swsum <= 0.0)
        throw Error("multissim: ScaleWeights must have at least one "
                    "positive element",
                    0, 0, "multissim", "",
                    "numkit:multissim:swZero");
    for (keel::f64 &v : sw) v /= swsum;

    // ── Gaussian kernel ────────────────────────────────────────
    const keel::i32 filtRadius = keel::as<keel::i32>(keel::ceil(sigma * 3.0));
    const keel::i32 filtSize = 2 * filtRadius + 1;
    Value gK = fspecial("gaussian",
                        { keel::as<keel::f64>(filtSize),
                          keel::as<keel::f64>(filtSize), sigma }, mr);

    // ── Promote to DOUBLE for computation ───────────────────────
    Value Ad = to_double_image(A, mr);
    Value Bd = to_double_image(Iref, mr);

    // ── Multi-scale loop ────────────────────────────────────────
    if (quality_maps_out) quality_maps_out->clear();
    keel::f64 total_log_score = 0.0;
    bool has_log = true;            // accumulate in log domain for stability
    keel::f64 total_score = 1.0;       // fallback (in case log fails)

    auto apply_weight = [](keel::f64 base, keel::f64 exponent) -> keel::f64 {
        // base may be slightly negative due to float noise; clamp.
        if (exponent != keel::floor(exponent) && base < 0.0) base = 0.0;
        return keel::pow(base, exponent);
    };

    for (keel::i32 i = 0; i < num_scales; ++i) {
        const bool last = (i == num_scales - 1);
        Value ssimmap = ssim_map_double(Ad, Bd, gK, C1, C2,
                                        /*include_luminance=*/last, mr);
        // Mean of ssimmap.
        const keel::usize Nm = ssimmap.numel();
        long double s = 0.0L; // keel-lint: ignore
        const keel::f64 *sd = ssimmap.doubleData();
        for (keel::usize k = 0; k < Nm; ++k) s += sd[k];
        const keel::f64 mean_map = keel::as<keel::f64>(
            s / keel::as<long double>(Nm)); // keel-lint: ignore
        const keel::f64 per_scale = apply_weight(mean_map, sw[i]);
        total_score *= per_scale;
        if (mean_map > 0.0) {
            total_log_score += sw[i] * keel::log(mean_map);
        } else {
            has_log = false;
        }
        if (quality_maps_out) quality_maps_out->push_back(ssimmap);
        if (!last) {
            Ad = downsample_2(lowpass_2x2(Ad, mr), mr);
            Bd = downsample_2(lowpass_2x2(Bd, mr), mr);
        }
    }
    keel::f64 score = has_log ? keel::exp(total_log_score) : total_score;

    return cast_score_value(score, origClass, mr);
}

// ════════════════════════════════════════════════════════════════════
// multissim3 — 3-D multi-scale SSIM (Wang/Simoncelli/Bovik 2003)
// ════════════════════════════════════════════════════════════════════
//
// Volumetric extension of multissim. Same algorithm but:
//   - Gaussian smoothing is 3-D separable (3 successive 1-D convs
//     along rows / cols / slices) with the same filter size
//     2·ceil(3σ)+1.
//   - Lowpass is a 2×2×2 box (ones/8) with replicate boundary,
//     anchor at the top-front-left corner of the kernel (matches
//     MATLAB's imfilter even-size convention).
//   - Downsample factor 2 in ALL three dims (rows/cols/slices).
//
// Transliterated verbatim from MATLAB R2025b multissim3.m +
// algmultissim.m (numSpatialDims=3 branch) + parserMultissim.m.

namespace {

// Volume buffer mirror of the 2-D "image-in-DOUBLE" used by multissim.
struct Vol {
    keel::usize H = 0, W = 0, D = 0;
    keel::Array<keel::f64> v;  // column-major: idx = z*H*W + c*H + r
    Vol() = default;
    Vol(keel::usize h, keel::usize w, keel::usize d)
        : H(h), W(w), D(d), v(h * w * d, 0.0) {}
    keel::f64 &at(keel::usize r, keel::usize c, keel::usize z) {
        return v[z * H * W + c * H + r];
    }
    keel::f64 at(keel::usize r, keel::usize c, keel::usize z) const {
        return v[z * H * W + c * H + r];
    }
};

Vol to_double_vol(const Value &V)
{
    const keel::usize H = V.dims().rows();
    const keel::usize W = V.dims().cols();
    const keel::usize D = V.dims().is3D() ? V.dims().pages() : 1;
    Vol out(H, W, D);
    const ValueType t = V.type();
    const keel::usize N = H * W * D;
    for (keel::usize i = 0; i < N; ++i) {
        keel::f64 v = V.elemAsDouble(i);
        if (t == ValueType::INT16) v += 32768.0;
        out.v[i] = v;
    }
    return out;
}

Value vol_to_value(const Vol &V, keel::Allocator *mr)
{
    Value out = V.D == 1
        ? Value::matrix(V.H, V.W, ValueType::DOUBLE, mr)
        : Value::matrix3d(V.H, V.W, V.D, ValueType::DOUBLE, mr);
    keel::copy_n(out.doubleDataMut(), V.v.data(), V.v.size());
    return out;
}

// Build a 1-D Gaussian kernel with `filtSize` elements, mean = (filtSize-1)/2.
keel::Array<keel::f64> gauss1d(keel::f64 sigma, keel::i32 filtSize)
{
    keel::Array<keel::f64> k(filtSize);
    const keel::f64 half = (filtSize - 1) / 2.0;
    keel::f64 s = 0.0;
    for (keel::i32 i = 0; i < filtSize; ++i) {
        const keel::f64 x = i - half;
        k[i] = keel::exp(-0.5 * (x / sigma) * (x / sigma));
        s += k[i];
    }
    for (keel::f64 &v : k) v /= s;
    return k;
}

inline keel::usize reflect_clamp(keel::i64 i, keel::usize n)
{
    // Replicate boundary: clamp to [0, n-1].
    if (i < 0) return 0;
    if (i >= keel::as<keel::i64>(n)) return n - 1;
    return keel::as<keel::usize>(i);
}

// Separable 3-D Gaussian: convolve along rows, then cols, then slices.
// All passes use replicate boundary.
Vol gauss3d_filter(const Vol &In, keel::ArrayRef<keel::f64> k)
{
    const keel::usize H = In.H;
    const keel::usize W = In.W;
    const keel::usize D = In.D;
    const keel::i32 kn = keel::as<keel::i32>(k.size());
    const keel::i32 half = kn / 2;
    Vol tmp(H, W, D), out(H, W, D);
    // Rows pass: for each (c, z), convolve along r.
    for (keel::usize z = 0; z < D; ++z)
        for (keel::usize c = 0; c < W; ++c)
            for (keel::usize r = 0; r < H; ++r) {
                keel::f64 s = 0.0;
                for (keel::i32 j = -half; j <= half; ++j) {
                    const keel::usize rr = reflect_clamp(
                        keel::as<keel::i64>(r) + j, H);
                    s += k[keel::as<keel::usize>(j + half)] * In.at(rr, c, z);
                }
                tmp.at(r, c, z) = s;
            }
    // Cols pass.
    Vol tmp2(H, W, D);
    for (keel::usize z = 0; z < D; ++z)
        for (keel::usize c = 0; c < W; ++c)
            for (keel::usize r = 0; r < H; ++r) {
                keel::f64 s = 0.0;
                for (keel::i32 j = -half; j <= half; ++j) {
                    const keel::usize cc = reflect_clamp(
                        keel::as<keel::i64>(c) + j, W);
                    s += k[keel::as<keel::usize>(j + half)] * tmp.at(r, cc, z);
                }
                tmp2.at(r, c, z) = s;
            }
    // Slices pass.
    for (keel::usize z = 0; z < D; ++z)
        for (keel::usize c = 0; c < W; ++c)
            for (keel::usize r = 0; r < H; ++r) {
                keel::f64 s = 0.0;
                for (keel::i32 j = -half; j <= half; ++j) {
                    const keel::usize zz = reflect_clamp(
                        keel::as<keel::i64>(z) + j, D);
                    s += k[keel::as<keel::usize>(j + half)] * tmp2.at(r, c, zz);
                }
                out.at(r, c, z) = s;
            }
    return out;
}

// 2×2×2 box lowpass with replicate boundary, anchor at top-front-left.
//   Out(r,c,z) = (1/8) * sum_(i,j,k) In(r+i, c+j, z+k), i,j,k ∈ {0,1}
Vol lowpass_2x2x2(const Vol &In)
{
    const keel::usize H = In.H;
    const keel::usize W = In.W;
    const keel::usize D = In.D;
    Vol out(H, W, D);
    for (keel::usize z = 0; z < D; ++z) {
        const keel::usize zn = (z + 1 < D) ? (z + 1) : (D - 1);
        for (keel::usize c = 0; c < W; ++c) {
            const keel::usize cn = (c + 1 < W) ? (c + 1) : (W - 1);
            for (keel::usize r = 0; r < H; ++r) {
                const keel::usize rn = (r + 1 < H) ? (r + 1) : (H - 1);
                out.at(r, c, z) =
                    0.125 * (In.at(r, c, z) + In.at(rn, c, z)
                          + In.at(r, cn, z) + In.at(rn, cn, z)
                          + In.at(r, c, zn) + In.at(rn, c, zn)
                          + In.at(r, cn, zn) + In.at(rn, cn, zn));
            }
        }
    }
    return out;
}

Vol downsample_2x2x2(const Vol &In)
{
    const keel::usize H = (In.H + 1) / 2;
    const keel::usize W = (In.W + 1) / 2;
    const keel::usize D = (In.D + 1) / 2;
    Vol out(H, W, D);
    for (keel::usize z = 0; z < D; ++z)
        for (keel::usize c = 0; c < W; ++c)
            for (keel::usize r = 0; r < H; ++r)
                out.at(r, c, z) = In.at(r * 2, c * 2, z * 2);
    return out;
}

// 3-D SSIM map. Same formula as ssim_map_double but operates on Vol.
Vol ssim_map_vol(const Vol &A, const Vol &B,
                 keel::ArrayRef<keel::f64> k,
                 keel::f64 C1, keel::f64 C2, bool include_luminance)
{
    const keel::usize H = A.H, W = A.W, D = A.D;
    Vol AA(H, W, D), BB(H, W, D), AB(H, W, D);
    for (keel::usize i = 0; i < A.v.size(); ++i) {
        const keel::f64 a = A.v[i], b = B.v[i];
        AA.v[i] = a * a;
        BB.v[i] = b * b;
        AB.v[i] = a * b;
    }
    Vol mu_a = gauss3d_filter(A, k);
    Vol mu_b = gauss3d_filter(B, k);
    Vol sa2  = gauss3d_filter(AA, k);
    Vol sb2  = gauss3d_filter(BB, k);
    Vol sab  = gauss3d_filter(AB, k);
    Vol out(H, W, D);
    for (keel::usize i = 0; i < A.v.size(); ++i) {
        const keel::f64 mua = mu_a.v[i], mub = mu_b.v[i];
        const keel::f64 mu_xy = mua * mub;
        const keel::f64 mux2 = mua * mua;
        const keel::f64 muy2 = mub * mub;
        keel::f64 vx = sa2.v[i] - mux2;
        keel::f64 vy = sb2.v[i] - muy2;
        if (vx < 0.0) vx = 0.0;
        if (vy < 0.0) vy = 0.0;
        const keel::f64 cov = sab.v[i] - mu_xy;
        const keel::f64 num2 = 2.0 * cov + C2;
        const keel::f64 den2 = vx + vy + C2;
        keel::f64 m;
        if (include_luminance) {
            const keel::f64 num1 = 2.0 * mu_xy + C1;
            const keel::f64 den1 = mux2 + muy2 + C1;
            m = (den1 != 0.0 && den2 != 0.0) ? (num1 * num2) / (den1 * den2) : 1.0;
        } else {
            m = (den2 != 0.0) ? num2 / den2 : 1.0;
        }
        if (m > 1.0) m = 1.0;
        out.v[i] = m;
    }
    return out;
}

void multissim3_argcheck(const Value &V, const Value &Vref, keel::i32 num_scales, keel::f64 sigma)
{
    internal::requireImageClass(V, "multissim3", "V", 1,
                                internal::ImgClasses::FloatInt);
    internal::requireImageClass(Vref, "multissim3", "ref", 2,
                                internal::ImgClasses::FloatInt);
    internal::requireSameClassPair(V, Vref, "multissim3");
    if (V.dims().rows() != Vref.dims().rows()
        || V.dims().cols() != Vref.dims().cols()
        || V.dims().pages() != Vref.dims().pages())
        throw Error("multissim3: V and Vref must have the same size",
                    0, 0, "multissim3", "", "numkit:multissim3:size");
    if (V.type() != Vref.type())
        throw Error("multissim3: V and Vref must have the same class",
                    0, 0, "multissim3", "", "numkit:multissim3:class");
    if (num_scales < 1)
        throw Error("multissim3: NumScales must be a positive integer",
                    0, 0, "multissim3", "",
                    "numkit:multissim3:numScales");
    if (!keel::is_finite(sigma) || sigma <= 0.0)
        throw Error("multissim3: Sigma must be a positive scalar",
                    0, 0, "multissim3", "",
                    "numkit:multissim3:sigma");
}

} // namespace

Value multissim3(const Value &V, const Value &Vref,
                 keel::i32 num_scales,
                 keel::ArrayRef<keel::f64> scale_weights_in,
                 keel::f64 sigma, keel::f64 dynamic_range_in,
                 keel::Array<Value> *quality_maps_out,
                 keel::Allocator *mr)
{
    multissim3_argcheck(V, Vref, num_scales, sigma);

    const ValueType origClass = V.type();
    Vol Ad = to_double_vol(V);
    Vol Bd = to_double_vol(Vref);

    // Verify downsampleability.
    {
        keel::usize h = Ad.H, w = Ad.W, d = Ad.D;
        for (keel::i32 i = 1; i < num_scales; ++i) {
            if (h <= 1 || w <= 1 || d <= 1)
                throw Error(keel::format("multissim3: volume too small for NumScales={}", num_scales),
                            0, 0, "multissim3", "",
                            "numkit:multissim3:tooSmall");
            h = (h + 1) / 2;
            w = (w + 1) / 2;
            d = (d + 1) / 2;
        }
    }

    const keel::f64 L = (dynamic_range_in > 0.0)
                   ? dynamic_range_in
                   : class_dynamic_range(origClass);
    const keel::f64 C1 = (0.01 * L) * (0.01 * L);
    const keel::f64 C2 = (0.03 * L) * (0.03 * L);

    keel::Array<keel::f64> sw = scale_weights_in.empty()
        ? default_scale_weights(num_scales)
        : keel::Array<keel::f64>::from(scale_weights_in);
    if (keel::as<keel::i32>(sw.size()) != num_scales)
        throw Error("multissim3: length(ScaleWeights) must equal NumScales",
                    0, 0, "multissim3", "", "numkit:multissim3:swLen");
    keel::f64 swsum = 0.0;
    for (keel::f64 v : sw) {
        if (v < 0.0)
            throw Error("multissim3: ScaleWeights must be non-negative",
                        0, 0, "multissim3", "", "numkit:multissim3:swNeg");
        swsum += v;
    }
    if (swsum <= 0.0)
        throw Error("multissim3: ScaleWeights must have at least one "
                    "positive element",
                    0, 0, "multissim3", "", "numkit:multissim3:swZero");
    for (keel::f64 &v : sw) v /= swsum;

    const keel::i32 filtRadius = keel::as<keel::i32>(keel::ceil(sigma * 3.0));
    const keel::i32 filtSize = 2 * filtRadius + 1;
    keel::Array<keel::f64> gk = gauss1d(sigma, filtSize);

    if (quality_maps_out) quality_maps_out->clear();

    keel::f64 total_log_score = 0.0;
    bool has_log = true;
    keel::f64 total_score = 1.0;
    auto apply_weight = [](keel::f64 base, keel::f64 exp) -> keel::f64 {
        if (exp != keel::floor(exp) && base < 0.0) base = 0.0;
        return keel::pow(base, exp);
    };

    for (keel::i32 i = 0; i < num_scales; ++i) {
        const bool last = (i == num_scales - 1);
        Vol ssimmap = ssim_map_vol(Ad, Bd, gk, C1, C2,
                                   /*include_luminance=*/last);
        const keel::usize Nm = ssimmap.v.size();
        long double s = 0.0L; // keel-lint: ignore
        for (keel::f64 x : ssimmap.v) s += x;
        const keel::f64 mean_map = keel::as<keel::f64>(
            s / keel::as<long double>(Nm)); // keel-lint: ignore
        total_score *= apply_weight(mean_map, sw[i]);
        if (mean_map > 0.0) total_log_score += sw[i] * keel::log(mean_map);
        else                has_log = false;
        if (quality_maps_out) quality_maps_out->push_back(vol_to_value(ssimmap, mr));
        if (!last) {
            Vol Al = lowpass_2x2x2(Ad);
            Vol Bl = lowpass_2x2x2(Bd);
            Ad = downsample_2x2x2(Al);
            Bd = downsample_2x2x2(Bl);
        }
    }
    keel::f64 score = has_log ? keel::exp(total_log_score) : total_score;
    return cast_score_value(score, origClass, mr);
}

Value corr2(const Value &A, const Value &B, keel::Allocator *mr)
{
    corr2_argcheck(A, B);
    const keel::usize N = A.numel();
    if (N == 0) return Value::scalar(keel::kNaN, mr);
    long double sa = 0.0L, sb = 0.0L; // keel-lint: ignore
    for (keel::usize i = 0; i < N; ++i) {
        sa += A.elemAsDouble(i);
        sb += B.elemAsDouble(i);
    }
    const long double ma = sa / keel::as<long double>(N); // keel-lint: ignore
    const long double mb = sb / keel::as<long double>(N); // keel-lint: ignore
    long double cov = 0.0L, va = 0.0L, vb = 0.0L; // keel-lint: ignore
    for (keel::usize i = 0; i < N; ++i) {
        const long double da = A.elemAsDouble(i) - ma; // keel-lint: ignore
        const long double db = B.elemAsDouble(i) - mb; // keel-lint: ignore
        cov += da * db;
        va  += da * da;
        vb  += db * db;
    }
    const long double denom = keel::sqrt((keel::f64)(va * vb)); // keel-lint: ignore
    if (denom == 0.0L) return Value::scalar(keel::kNaN, mr);
    return Value::scalar(keel::as<keel::f64>(cov / denom), mr);
}

} // namespace numkit::image
