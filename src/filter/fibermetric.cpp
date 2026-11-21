// toolboxes/image/src/filter/fibermetric.cpp
//
// Multiscale Hessian-based vesselness filter (Frangi 1998 "Multiscale
// vessel enhancement filtering", MICCAI '98).
//
// Algorithm transliterated verbatim from MATLAB R2025b fibermetric.m
// + the internal C++ builtin images.internal.builtins.fibermetric.
//
// Per scale i ∈ thickness:
//   σ_i = thickness_i / 6.
//   Gaussian-smooth I with σ_i (FilterSize = 2·ceil(3σ_i) + 1).
//   Compute Hessian elements via central finite differences:
//     Ixx[r,c] = I[r,c+1] - 2·I[r,c] + I[r,c-1]
//     Iyy[r,c] = I[r+1,c] - 2·I[r,c] + I[r-1,c]
//     Ixy[r,c] = (I[r+1,c+1] - I[r+1,c-1] - I[r-1,c+1] + I[r-1,c-1])/4
//   Eigenvalues of [[Ixx Ixy];[Ixy Iyy]]:
//     trace = Ixx + Iyy, det = Ixx·Iyy − Ixy²
//     disc = sqrt((Ixx − Iyy)² + 4·Ixy²) / 2  (more stable)
//     λ⁻ = trace/2 − disc, λ⁺ = trace/2 + disc.
//     Sort so |λ₁| ≤ |λ₂|.
//   Vesselness:
//     if (bright && λ₂ > 0) || (dark && λ₂ < 0): V = 0.
//     else: Rβ = λ₁/λ₂; S = sqrt(λ₁² + λ₂²);
//           V = exp(-Rβ²/(2β²)) · (1 - exp(-S²/(2c²)));
//           β = 0.5 (Frangi default).
//
// 3-D variant: 3 eigenvalues |λ₁| ≤ |λ₂| ≤ |λ₃|.
//   Rα = |λ₂|/|λ₃|, Rβ = |λ₁|/sqrt(|λ₂·λ₃|), S² = sum(λᵢ²).
//   if (bright && λ₂>0||λ₃>0) || (dark && λ₂<0||λ₃<0): V = 0.
//   else: V = (1 - exp(-Rα²/(2α²))) · exp(-Rβ²/(2β²)) ·
//             (1 - exp(-S²/(2c²)));
//   α = β = 0.5.
//
// Per-pixel response is the MAX over all scales.
//
// PMR HARD RULE: every fn takes std::pmr::memory_resource *mr.

#define _USE_MATH_DEFINES
#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/image/filter/filter.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

#ifndef M_PI
#  define M_PI 3.14159265358979323846
#endif

namespace numkit::image {
namespace {

constexpr keel::f64 kFrangiBeta = 0.5;     // 2-D and 3-D blobness param
constexpr keel::f64 kFrangiAlpha = 0.5;    // 3-D plate-vs-line param

// Replicate-boundary clamp.
inline keel::usize clamp_idx(keel::i64 i, keel::usize n)
{
    if (i < 0) return 0;
    if (i >= keel::as<keel::i64>(n)) return n - 1;
    return keel::as<keel::usize>(i);
}

keel::f64 class_range(ValueType t)
{
    switch (t) {
        case ValueType::UINT8:  return 255.0;
        case ValueType::UINT16: return 65535.0;
        case ValueType::UINT32: return 4294967295.0;
        case ValueType::INT8:   return 255.0;
        case ValueType::INT16:  return 65535.0;
        case ValueType::INT32:  return 4294967295.0;
        default:                return 1.0;        // single, double
    }
}

// ── Convert input to working buffer (array of double) ─────────────
struct Buf {
    keel::usize H = 0, W = 0, D = 1;
    keel::Array<keel::f64> v;
    bool is3d() const { return D > 1; }
    keel::f64 &at(keel::usize r, keel::usize c, keel::usize z = 0) {
        return v[z * H * W + c * H + r];
    }
    keel::f64 at(keel::usize r, keel::usize c, keel::usize z = 0) const {
        return v[z * H * W + c * H + r];
    }
};

Buf to_double_buf(const Value &V)
{
    Buf b;
    b.H = V.dims().rows();
    b.W = V.dims().cols();
    b.D = V.dims().is3D() ? V.dims().pages() : 1;
    b.v.resize(b.H * b.W * b.D);
    const keel::usize N = b.v.size();
    for (keel::usize i = 0; i < N; ++i)
        b.v[i] = V.elemAsDouble(i);
    return b;
}

// ── 1-D Gaussian kernel, length = 2·radius+1 ─────────────────────
keel::Array<keel::f64> gauss1d(keel::f64 sigma, keel::i32 radius)
{
    const keel::i32 n = 2 * radius + 1;
    keel::Array<keel::f64> k(n);
    keel::f64 s = 0.0;
    for (keel::i32 i = 0; i < n; ++i) {
        const keel::f64 x = i - radius;
        k[keel::as<keel::usize>(i)] =
            std::exp(-0.5 * (x / sigma) * (x / sigma));
        s += k[keel::as<keel::usize>(i)];
    }
    for (keel::f64 &v : k) v /= s;
    return k;
}

// Separable 2-D Gaussian filter, replicate boundary.
Buf gauss2d(const Buf &In, keel::f64 sigma)
{
    const keel::i32 rad = keel::as<keel::i32>(std::ceil(3.0 * sigma));
    keel::Array<keel::f64> k = gauss1d(sigma, rad);
    const keel::usize H = In.H, W = In.W;
    Buf tmp; tmp.H = H; tmp.W = W; tmp.D = 1; tmp.v.resize(H * W, 0.0);
    Buf out; out.H = H; out.W = W; out.D = 1; out.v.resize(H * W, 0.0);
    // Rows pass.
    for (keel::usize c = 0; c < W; ++c)
        for (keel::usize r = 0; r < H; ++r) {
            keel::f64 s = 0.0;
            for (keel::i32 j = -rad; j <= rad; ++j) {
                const keel::usize rr = clamp_idx(
                    keel::as<keel::i64>(r) + j, H);
                s += k[keel::as<keel::usize>(j + rad)] * In.at(rr, c);
            }
            tmp.at(r, c) = s;
        }
    // Cols pass.
    for (keel::usize c = 0; c < W; ++c)
        for (keel::usize r = 0; r < H; ++r) {
            keel::f64 s = 0.0;
            for (keel::i32 j = -rad; j <= rad; ++j) {
                const keel::usize cc = clamp_idx(
                    keel::as<keel::i64>(c) + j, W);
                s += k[keel::as<keel::usize>(j + rad)] * tmp.at(r, cc);
            }
            out.at(r, c) = s;
        }
    return out;
}

// Separable 3-D Gaussian filter, replicate boundary.
Buf gauss3d(const Buf &In, keel::f64 sigma)
{
    const keel::i32 rad = keel::as<keel::i32>(std::ceil(3.0 * sigma));
    keel::Array<keel::f64> k = gauss1d(sigma, rad);
    const keel::usize H = In.H, W = In.W, D = In.D;
    Buf tmp;  tmp.H = H;  tmp.W = W;  tmp.D = D;  tmp.v.resize(H * W * D, 0.0);
    Buf tmp2; tmp2.H = H; tmp2.W = W; tmp2.D = D; tmp2.v.resize(H * W * D, 0.0);
    Buf out;  out.H = H;  out.W = W;  out.D = D;  out.v.resize(H * W * D, 0.0);
    // Rows.
    for (keel::usize z = 0; z < D; ++z)
        for (keel::usize c = 0; c < W; ++c)
            for (keel::usize r = 0; r < H; ++r) {
                keel::f64 s = 0.0;
                for (keel::i32 j = -rad; j <= rad; ++j) {
                    const keel::usize rr = clamp_idx(
                        keel::as<keel::i64>(r) + j, H);
                    s += k[keel::as<keel::usize>(j + rad)] * In.at(rr, c, z);
                }
                tmp.at(r, c, z) = s;
            }
    // Cols.
    for (keel::usize z = 0; z < D; ++z)
        for (keel::usize c = 0; c < W; ++c)
            for (keel::usize r = 0; r < H; ++r) {
                keel::f64 s = 0.0;
                for (keel::i32 j = -rad; j <= rad; ++j) {
                    const keel::usize cc = clamp_idx(
                        keel::as<keel::i64>(c) + j, W);
                    s += k[keel::as<keel::usize>(j + rad)] * tmp.at(r, cc, z);
                }
                tmp2.at(r, c, z) = s;
            }
    // Slices.
    for (keel::usize z = 0; z < D; ++z)
        for (keel::usize c = 0; c < W; ++c)
            for (keel::usize r = 0; r < H; ++r) {
                keel::f64 s = 0.0;
                for (keel::i32 j = -rad; j <= rad; ++j) {
                    const keel::usize zz = clamp_idx(
                        keel::as<keel::i64>(z) + j, D);
                    s += k[keel::as<keel::usize>(j + rad)] * tmp2.at(r, c, zz);
                }
                out.at(r, c, z) = s;
            }
    return out;
}

// ── 2-D Frangi vesselness at one scale ─────────────────────────────
// Hessian elements are γ-normalized by σ² (Frangi 1998, "Multiscale
// vessel enhancement filtering", Eq. before Eq. 7: γ = 2 for vessels).
Buf frangi2d(const Buf &Ig, keel::f64 c, bool bright, keel::f64 sigma)
{
    const keel::usize H = Ig.H, W = Ig.W;
    Buf out; out.H = H; out.W = W; out.D = 1; out.v.resize(H * W, 0.0);
    const keel::f64 beta_sq2 = 2.0 * kFrangiBeta * kFrangiBeta;
    const keel::f64 c_sq2    = 2.0 * c * c;
    const keel::f64 sigma2   = sigma * sigma;
    for (keel::usize cc = 0; cc < W; ++cc) {
        for (keel::usize r = 0; r < H; ++r) {
            const keel::usize rm = clamp_idx(keel::as<keel::i64>(r) - 1, H);
            const keel::usize rp = clamp_idx(keel::as<keel::i64>(r) + 1, H);
            const keel::usize cm = clamp_idx(keel::as<keel::i64>(cc) - 1, W);
            const keel::usize cp = clamp_idx(keel::as<keel::i64>(cc) + 1, W);
            const keel::f64 v   = Ig.at(r,  cc);
            const keel::f64 Ixx = sigma2 * (Ig.at(r,  cp) - 2.0 * v + Ig.at(r,  cm));
            const keel::f64 Iyy = sigma2 * (Ig.at(rp, cc) - 2.0 * v + Ig.at(rm, cc));
            const keel::f64 Ixy = sigma2 * 0.25 *
                                      ( Ig.at(rp, cp) - Ig.at(rp, cm)
                                      - Ig.at(rm, cp) + Ig.at(rm, cm));
            // Eigenvalues of 2x2 symmetric: trace/2 ± sqrt((diff/2)² + Ixy²).
            const keel::f64 mean = 0.5 * (Ixx + Iyy);
            const keel::f64 diff = 0.5 * (Ixx - Iyy);
            const keel::f64 disc = std::sqrt(diff * diff + Ixy * Ixy);
            keel::f64 l1 = mean - disc;
            keel::f64 l2 = mean + disc;
            // Order by |.|: |λ1| ≤ |λ2|.
            if (std::fabs(l1) > std::fabs(l2)) std::swap(l1, l2);
            // Polarity check.
            if (bright) { if (l2 > 0.0) { out.at(r, cc) = 0.0; continue; } }
            else        { if (l2 < 0.0) { out.at(r, cc) = 0.0; continue; } }
            if (l2 == 0.0) { out.at(r, cc) = 0.0; continue; }
            const keel::f64 Rb_sq = (l1 / l2) * (l1 / l2);
            const keel::f64 S_sq  = l1 * l1 + l2 * l2;
            const keel::f64 V = std::exp(-Rb_sq / beta_sq2)
                           * (1.0 - std::exp(-S_sq / c_sq2));
            out.at(r, cc) = V;
        }
    }
    return out;
}

// ── 3-D Frangi vesselness at one scale ─────────────────────────────
// 3x3 symmetric eigen decomposition via the closed-form trig solution
// (Smith, 1961; widely cited for 3x3 symmetric Hessians).
void eigvals_3x3(keel::f64 a11, keel::f64 a22, keel::f64 a33,
                 keel::f64 a12, keel::f64 a13, keel::f64 a23,
                 keel::f64 &l1, keel::f64 &l2, keel::f64 &l3)
{
    // Coefficients of characteristic polynomial λ³ − tr·λ² + ... − det.
    const keel::f64 p1 = a12 * a12 + a13 * a13 + a23 * a23;
    if (p1 == 0.0) {
        // Diagonal already.
        l1 = a11; l2 = a22; l3 = a33;
    } else {
        const keel::f64 q = (a11 + a22 + a33) / 3.0;
        const keel::f64 p2 = (a11 - q) * (a11 - q)
                        + (a22 - q) * (a22 - q)
                        + (a33 - q) * (a33 - q)
                        + 2.0 * p1;
        const keel::f64 p = std::sqrt(p2 / 6.0);
        if (p == 0.0) { l1 = l2 = l3 = q; return; }
        const keel::f64 b11 = (a11 - q) / p;
        const keel::f64 b22 = (a22 - q) / p;
        const keel::f64 b33 = (a33 - q) / p;
        const keel::f64 b12 = a12 / p;
        const keel::f64 b13 = a13 / p;
        const keel::f64 b23 = a23 / p;
        // det(B) / 2
        const keel::f64 r = 0.5 * (b11 * (b22 * b33 - b23 * b23)
                               - b12 * (b12 * b33 - b23 * b13)
                               + b13 * (b12 * b23 - b22 * b13));
        keel::f64 rcl = r;
        if (rcl < -1.0) rcl = -1.0;
        if (rcl >  1.0) rcl =  1.0;
        const keel::f64 phi = std::acos(rcl) / 3.0;
        l1 = q + 2.0 * p * std::cos(phi);
        l3 = q + 2.0 * p * std::cos(phi + 2.0 * M_PI / 3.0);
        l2 = 3.0 * q - l1 - l3;     // trace constraint
    }
    // Sort by absolute value: |l1| ≤ |l2| ≤ |l3|.
    if (std::fabs(l1) > std::fabs(l2)) std::swap(l1, l2);
    if (std::fabs(l2) > std::fabs(l3)) std::swap(l2, l3);
    if (std::fabs(l1) > std::fabs(l2)) std::swap(l1, l2);
}

Buf frangi3d(const Buf &Ig, keel::f64 c, bool bright, keel::f64 sigma)
{
    const keel::usize H = Ig.H, W = Ig.W, D = Ig.D;
    Buf out; out.H = H; out.W = W; out.D = D; out.v.resize(H * W * D, 0.0);
    const keel::f64 alpha_sq2 = 2.0 * kFrangiAlpha * kFrangiAlpha;
    const keel::f64 beta_sq2  = 2.0 * kFrangiBeta  * kFrangiBeta;
    const keel::f64 c_sq2     = 2.0 * c * c;
    const keel::f64 sigma2    = sigma * sigma;
    for (keel::usize z = 0; z < D; ++z) {
        const keel::usize zm = clamp_idx(keel::as<keel::i64>(z) - 1, D);
        const keel::usize zp = clamp_idx(keel::as<keel::i64>(z) + 1, D);
        for (keel::usize cc = 0; cc < W; ++cc) {
            const keel::usize cm = clamp_idx(keel::as<keel::i64>(cc) - 1, W);
            const keel::usize cp = clamp_idx(keel::as<keel::i64>(cc) + 1, W);
            for (keel::usize r = 0; r < H; ++r) {
                const keel::usize rm = clamp_idx(keel::as<keel::i64>(r) - 1, H);
                const keel::usize rp = clamp_idx(keel::as<keel::i64>(r) + 1, H);
                const keel::f64 v   = Ig.at(r, cc, z);
                const keel::f64 Ixx = sigma2 * (Ig.at(r,  cp, z) - 2.0 * v + Ig.at(r,  cm, z));
                const keel::f64 Iyy = sigma2 * (Ig.at(rp, cc, z) - 2.0 * v + Ig.at(rm, cc, z));
                const keel::f64 Izz = sigma2 * (Ig.at(r,  cc, zp) - 2.0 * v + Ig.at(r,  cc, zm));
                const keel::f64 Ixy = sigma2 * 0.25 *
                                          (Ig.at(rp, cp, z) - Ig.at(rp, cm, z)
                                         - Ig.at(rm, cp, z) + Ig.at(rm, cm, z));
                const keel::f64 Ixz = sigma2 * 0.25 *
                                          (Ig.at(r,  cp, zp) - Ig.at(r,  cm, zp)
                                         - Ig.at(r,  cp, zm) + Ig.at(r,  cm, zm));
                const keel::f64 Iyz = sigma2 * 0.25 *
                                          (Ig.at(rp, cc, zp) - Ig.at(rm, cc, zp)
                                         - Ig.at(rp, cc, zm) + Ig.at(rm, cc, zm));
                keel::f64 l1, l2, l3;
                eigvals_3x3(Ixx, Iyy, Izz, Ixy, Ixz, Iyz, l1, l2, l3);
                // Polarity: require both l2 and l3 to have the right sign.
                if (bright) {
                    if (l2 > 0.0 || l3 > 0.0) continue;
                } else {
                    if (l2 < 0.0 || l3 < 0.0) continue;
                }
                if (l2 == 0.0 || l3 == 0.0) continue;
                const keel::f64 Ra_sq  = (l2 / l3) * (l2 / l3);
                const keel::f64 prod23 = std::fabs(l2 * l3);
                if (prod23 == 0.0) continue;
                const keel::f64 Rb_sq = (l1 * l1) / prod23;
                const keel::f64 S_sq  = l1 * l1 + l2 * l2 + l3 * l3;
                const keel::f64 V = (1.0 - std::exp(-Ra_sq / alpha_sq2))
                               * std::exp(-Rb_sq / beta_sq2)
                               * (1.0 - std::exp(-S_sq / c_sq2));
                out.at(r, cc, z) = V;
            }
        }
    }
    return out;
}

// Cast result buffer to output Value (single for non-double inputs,
// double for double).
Value buf_to_value(const Buf &B, ValueType targetClass,
                   keel::Allocator *mr)
{
    const keel::usize N = B.v.size();
    Value out;
    if (B.is3d())
        out = Value::matrix3d(B.H, B.W, B.D, targetClass, mr);
    else
        out = Value::matrix(B.H, B.W, targetClass, mr);
    if (targetClass == ValueType::DOUBLE) {
        std::memcpy(out.doubleDataMut(), B.v.data(), N * sizeof(keel::f64));
    } else {
        // SINGLE
        keel::f32 *od = out.singleDataMut();
        for (keel::usize i = 0; i < N; ++i)
            od[i] = keel::as<keel::f32>(B.v[i]);
    }
    return out;
}

void fibermetric_argcheck(const Value &I, keel::ArrayRef<keel::f64> thickness_in,
                          keel::f64 structure_sensitivity)
{
    const auto &d = I.dims();
    if (d.rows() < 2 || d.cols() < 2)
        throw Error("fibermetric: I must be 2-D or 3-D with all dims >= 2",
                    0, 0, "fibermetric", "", "numkit:fibermetric:shape");
    const bool is3d = d.is3D() && d.pages() > 1;
    if (is3d && d.pages() < 2)
        throw Error("fibermetric: 3-D input must have at least 2 slices",
                    0, 0, "fibermetric", "", "numkit:fibermetric:slices");

    for (keel::f64 t : thickness_in)
        if (!std::isfinite(t) || t <= 0.0 || std::floor(t) != t)
            throw Error("fibermetric: thickness must be a vector of positive "
                        "integers",
                        0, 0, "fibermetric", "",
                        "numkit:fibermetric:thickness");

    keel::f64 c = structure_sensitivity;
    if (c <= 0.0) c = class_range(I.type()) / 100.0;
    if (!std::isfinite(c) || c <= 0.0)
        throw Error("fibermetric: StructureSensitivity must be > 0",
                    0, 0, "fibermetric", "",
                    "numkit:fibermetric:sens");
}

}  // namespace

Value fibermetric(const Value &I, keel::ArrayRef<keel::f64> thickness_in,
                  keel::f64 structure_sensitivity, bool bright_polarity,
                  keel::Allocator *mr)
{
    fibermetric_argcheck(I, thickness_in, structure_sensitivity);
    const auto &d = I.dims();
    if (d.rows() < 2 || d.cols() < 2)
        throw Error("fibermetric: I must be 2-D or 3-D with all dims >= 2",
                    0, 0, "fibermetric", "", "numkit:fibermetric:shape");
    const bool is3d = d.is3D() && d.pages() > 1;
    if (is3d && d.pages() < 2)
        throw Error("fibermetric: 3-D input must have at least 2 slices",
                    0, 0, "fibermetric", "", "numkit:fibermetric:slices");

    // Default thickness.
    static constexpr keel::f64 kDefaultThickness[6] = {4.0, 6.0, 8.0, 10.0, 12.0, 14.0};
    keel::ArrayRef<keel::f64> thickness = thickness_in.empty()
        ? keel::ArrayRef<keel::f64>(kDefaultThickness, 6)
        : thickness_in;
    for (keel::f64 t : thickness)
        if (!std::isfinite(t) || t <= 0.0 || std::floor(t) != t)
            throw Error("fibermetric: thickness must be a vector of positive "
                        "integers",
                        0, 0, "fibermetric", "",
                        "numkit:fibermetric:thickness");

    // Default c = range/100.
    keel::f64 c = structure_sensitivity;
    if (c <= 0.0) c = class_range(I.type()) / 100.0;
    if (!std::isfinite(c) || c <= 0.0)
        throw Error("fibermetric: StructureSensitivity must be > 0",
                    0, 0, "fibermetric", "",
                    "numkit:fibermetric:sens");

    // Output class.
    const ValueType outClass = (I.type() == ValueType::DOUBLE)
                             ? ValueType::DOUBLE : ValueType::SINGLE;

    Buf In = to_double_buf(I);
    Buf Out; Out.H = In.H; Out.W = In.W; Out.D = In.D;
    Out.v.resize(In.v.size(), 0.0);

    for (keel::f64 t : thickness) {
        const keel::f64 sigma = t / 6.0;
        Buf G = is3d ? gauss3d(In, sigma) : gauss2d(In, sigma);
        Buf V = is3d ? frangi3d(G, c, bright_polarity, sigma)
                     : frangi2d(G, c, bright_polarity, sigma);
        for (keel::usize i = 0; i < Out.v.size(); ++i)
            if (V.v[i] > Out.v[i]) Out.v[i] = V.v[i];
    }
    return buf_to_value(Out, outClass, mr);
}

} // namespace numkit::image
