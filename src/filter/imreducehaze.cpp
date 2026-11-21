// toolboxes/image/src/filter/imreducehaze.cpp
//
// Single-image dehazing via dark-channel prior. Two flavours:
//
//   • simpledcp (default): per-pixel dark channel D(x) = min_c I_c(x);
//                          atmospheric light via 5-level quadtree
//                          decomposition (Dubok et al. 2014 ICIP);
//                          5x5 guided-filter refinement (ε=0.01);
//                          ω = 0.9.
//   • approxdcp:           patched dark channel via min-erosion with
//                          square strel of size ceil(min(H,W)/400·15);
//                          atmospheric light from 0.1% brightest dark-
//                          channel pixels; guided-filter radius
//                          ceil(min(H,W)/50), ε=1e-4, subsample=4;
//                          ω = 0.95.
//
// Algorithm transliterated verbatim from MATLAB R2025b
//   toolbox/images/images/imreducehaze.m.
//
// References:
//   [1] He, K. "Single Image Haze Removal Using Dark Channel Prior."
//       Thesis, The Chinese University of Hong Kong, 2011.
//       (Earlier IEEE TPAMI 33(12) 2011 paper.)
//   [2] Dubok, P., Park, S., Hong, S., & Park, J. "Single Image
//       Dehazing with Image Entropy and Information Fidelity." ICIP
//       2014, pp 4037-4041.
//
// PMR HARD RULE: every fn takes std::pmr::memory_resource *mr.
//
// Composed entirely from existing nk primitives (imerode, imopen,
// imguidedfilter, rgb2gray, mat2gray, stretchlim, imadjust, im2*),
// so the haze-physics layer is the only new code here.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/math.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/image/filter/filter.hpp>
#include <numkit/image/morph/morph.hpp>
#include <numkit/image/contrast/contrast.hpp>
#include <numkit/image/type_convert/type_convert.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace numkit::image {
namespace {

// Lowercase helper (mirrors the one in color_extras).
inline keel::String lower_str(keel::StringRef s)
{
    keel::String lo;
    lo.reserve(s.size());
    for (char ch : s)
        lo.push_back(keel::to_ascii_lower(ch));
    return lo;
}

// ── im2single semantics (uint8 → x/255 single, uint16 → x/65535,
//   single passthrough, double clamp+cast).
Value to_single01(const Value &A, keel::Allocator *mr)
{
    const auto &d = A.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    const keel::usize P = d.is3D() ? d.pages() : 1;
    const ValueType t = A.type();
    if (t == ValueType::SINGLE) {
        // Clamp to [0,1] copy (algorithm assumes valid range).
        Value out = d.is3D()
            ? Value::matrix3d(H, W, P, ValueType::SINGLE, mr)
            : Value::matrix(H, W, ValueType::SINGLE, mr);
        const keel::f32 *src = A.singleData();
        keel::f32 *dst = out.singleDataMut();
        for (keel::usize i = 0; i < H * W * P; ++i) {
            keel::f32 v = src[i];
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            dst[i] = v;
        }
        return out;
    }
    Value out = d.is3D()
        ? Value::matrix3d(H, W, P, ValueType::SINGLE, mr)
        : Value::matrix(H, W, ValueType::SINGLE, mr);
    keel::f32 *dst = out.singleDataMut();
    for (keel::usize i = 0; i < H * W * P; ++i) {
        const keel::f64 raw = A.elemAsDouble(i);
        keel::f32 v;
        switch (t) {
            case ValueType::UINT8:  v = keel::as<keel::f32>(raw) / 255.0f;   break;
            case ValueType::UINT16: v = keel::as<keel::f32>(raw) / 65535.0f; break;
            case ValueType::INT8:   v = (keel::as<keel::f32>(raw) + 128.0f) / 255.0f;   break;
            case ValueType::INT16:  v = (keel::as<keel::f32>(raw) + 32768.0f) / 65535.0f; break;
            case ValueType::LOGICAL: v = raw == 0.0 ? 0.0f : 1.0f; break;
            default: {
                v = keel::as<keel::f32>(raw);
                if (v < 0.0f) v = 0.0f;
                if (v > 1.0f) v = 1.0f;
                break;
            }
        }
        dst[i] = v;
    }
    return out;
}

// ── min across pages (RGB → dark channel base, no spatial erosion) ──
// Output is H×W SINGLE.
Value min_across_pages(const Value &A, keel::Allocator *mr)
{
    const auto &d = A.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    const keel::usize P = d.is3D() ? d.pages() : 1;
    Value out = Value::matrix(H, W, ValueType::SINGLE, mr);
    keel::f32 *dst = out.singleDataMut();
    const keel::f32 *src = A.singleData();
    for (keel::usize i = 0; i < H * W; ++i) {
        keel::f32 m = src[i];
        for (keel::usize p = 1; p < P; ++p) {
            const keel::f32 v = src[p * H * W + i];
            if (v < m) m = v;
        }
        dst[i] = m;
    }
    return out;
}

// ── square strel of size n as a logical n×n matrix ──
Value make_square_strel(int n, keel::Allocator *mr)
{
    Value se = Value::matrix(keel::as<keel::usize>(n),
                              keel::as<keel::usize>(n),
                              ValueType::LOGICAL, mr);
    keel::u8 *d = se.logicalDataMut();
    for (int i = 0; i < n * n; ++i) d[i] = 1;
    return se;
}

// ── Dehaze: 1 - imopen(A./atm, strel) for approxdcp branch ────────
// A is SINGLE, atm is the brightest RGB triple (1×3) or scalar; we
// divide each plane and then min-across-pages.
Value normalize_by_atm_then_min(const Value &A, keel::ArrayRef<keel::f32> atm,
                                keel::Allocator *mr)
{
    const auto &d = A.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    const keel::usize P = d.is3D() ? d.pages() : 1;
    Value out = Value::matrix(H, W, ValueType::SINGLE, mr);
    keel::f32 *dst = out.singleDataMut();
    const keel::f32 *src = A.singleData();
    for (keel::usize i = 0; i < H * W; ++i) {
        keel::f32 m = std::numeric_limits<keel::f32>::infinity();
        for (keel::usize p = 0; p < P; ++p) {
            const keel::f32 a = atm[p > atm.size() - 1 ? atm.size() - 1 : p];
            const keel::f32 v = (a == 0.0f) ? std::numeric_limits<keel::f32>::infinity()
                                        : src[p * H * W + i] / a;
            if (v < m) m = v;
        }
        dst[i] = m;
    }
    return out;
}

// ── Recursive 5-level quadtree on the eroded dark channel; return
//    the (1-based row/col indices, inclusive both ends) of the final
//    sub-rectangle of A. Implements computeatmLightUsingQuadTree.
struct Rect { keel::usize r0, c0, r1, c1; };  // inclusive

Rect quadtree_decomp(const Value &dark, Rect Q, int numLevels)
{
    const keel::f32 *dd = dark.singleData();
    const keel::usize H = dark.dims().rows();
    auto mean_of = [&](const Rect &q) -> keel::f64 {
        keel::f64 s = 0.0;
        const keel::usize n = (q.r1 - q.r0 + 1) * (q.c1 - q.c0 + 1);
        for (keel::usize c = q.c0; c <= q.c1; ++c)
            for (keel::usize r = q.r0; r <= q.r1; ++r)
                s += dd[c * H + r];
        return s / keel::as<keel::f64>(n);
    };
    for (int lvl = 0; lvl < numLevels; ++lvl) {
        // MATLAB's quadrant indexing (1-based, all integer):
        //   row mid = (Q.r0 + Q.r1) / 2   (round-half-down via fix on
        //                                  even/odd; round in MATLAB
        //                                  source uses `round`)
        //   col mid = (Q.c0 + Q.c1) / 2
        // R2025b uses `round(...)` so we match that.
        const long rmid = keel::as<long>(std::round(
            (keel::as<keel::f64>(Q.r0) + keel::as<keel::f64>(Q.r1)) / 2.0));
        const long cmid = keel::as<long>(std::round(
            (keel::as<keel::f64>(Q.c0) + keel::as<keel::f64>(Q.c1)) / 2.0));
        const Rect q1{Q.r0,
                       Q.c0,
                       keel::as<keel::usize>(rmid),
                       keel::as<keel::usize>(cmid)};
        const Rect q2{Q.r0,
                       keel::as<keel::usize>(cmid + 1),
                       keel::as<keel::usize>(rmid),
                       Q.c1};
        const Rect q3{keel::as<keel::usize>(rmid + 1),
                       Q.c0,
                       Q.r1,
                       keel::as<keel::usize>(cmid)};
        const Rect q4{keel::as<keel::usize>(rmid + 1),
                       keel::as<keel::usize>(cmid + 1),
                       Q.r1,
                       Q.c1};
        const keel::f64 m[4] = {mean_of(q1), mean_of(q2), mean_of(q3), mean_of(q4)};
        int best = 0;
        for (int k = 1; k < 4; ++k)
            if (m[k] > m[best]) best = k;
        Q = best == 0 ? q1 : best == 1 ? q2 : best == 2 ? q3 : q4;
    }
    return Q;
}

// Compute atmospheric light using the simpledcp quadtree algorithm.
// A is SINGLE (H×W×P).
keel::Array<keel::f32> simpledcp_atm_light(const Value &A,
                                       keel::Allocator *mr)
{
    const auto &d = A.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    const keel::usize P = d.is3D() ? d.pages() : 1;
    Rect Q{0, 0, H - 1, W - 1};
    if (H >= 64 && W >= 64) {
        const int winSize = keel::as<int>(std::ceil(
            keel::as<keel::f64>(std::min(H, W)) / 400.0 * 15.0));
        Value se = make_square_strel(winSize, mr);
        Value darkChannel = min_across_pages(A, mr);
        Value dcEroded = imerode(darkChannel, se, mr);
        Q = quadtree_decomp(dcEroded, Q, 5);
    }
    // Pick pixel with minimum Euclidean distance to [1, 1, ..., 1]
    // in the final quadrant — i.e., the brightest pixel by L2 to ones.
    const keel::f32 *src = A.singleData();
    keel::f64 bestDist = std::numeric_limits<keel::f64>::infinity();
    keel::usize bestR = Q.r0, bestC = Q.c0;
    for (keel::usize c = Q.c0; c <= Q.c1; ++c)
        for (keel::usize r = Q.r0; r <= Q.r1; ++r) {
            keel::f64 s = 0.0;
            for (keel::usize p = 0; p < P; ++p) {
                const keel::f64 v = src[p * H * W + c * H + r];
                const keel::f64 d2 = keel::abs(1.0 - v);
                s += d2;     // sqrt(d2^2) == |d2|; matches MATLAB's
                            // sqrt(abs(brightIm-img).^2) then sum.
            }
            if (s < bestDist) {
                bestDist = s;
                bestR = r;
                bestC = c;
            }
        }
    keel::Array<keel::f32> atm(P);
    for (keel::usize p = 0; p < P; ++p)
        atm[p] = src[p * H * W + bestC * H + bestR];
    return atm;
}

// approxdcp atm light estimation: 0.1% brightest pixels of the
// (eroded) dark channel; among those, pick the one with the highest
// grayscale luminance (rgb2gray for RGB, A for grayscale).
keel::Array<keel::f32> approxdcp_atm_light(const Value &A,
                                       const Value &darkChannel,
                                       keel::Allocator *mr)
{
    const auto &d = A.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    const keel::usize P = d.is3D() ? d.pages() : 1;
    const keel::f32 *dc = darkChannel.singleData();
    // 256-bin imhist on the dark channel.
    constexpr int NBINS = 256;
    keel::Array<keel::usize> hist(NBINS, 0);
    for (keel::usize i = 0; i < H * W; ++i) {
        int b = keel::as<int>(std::floor(dc[i] * (NBINS - 1) + 0.5f));
        if (b < 0) b = 0;
        if (b >= NBINS) b = NBINS - 1;
        hist[b]++;
    }
    // Cumulative; find threshold for top 0.1% (1 - 0.001 = 0.999).
    const keel::f64 total = keel::as<keel::f64>(H * W);
    keel::f64 cum = 0;
    int binIdx = NBINS - 1;
    for (int b = 0; b < NBINS; ++b) {
        cum += hist[b];
        if (cum / total >= 0.999) { binIdx = b; break; }
    }
    const keel::f32 binWidth = 1.0f / keel::as<keel::f32>(NBINS - 1);
    const keel::f32 cutoff = keel::as<keel::f32>(binIdx) / keel::as<keel::f32>(NBINS - 1)
                       - binWidth * 0.5f;

    // Build mask of pixels >= cutoff.
    // Among masked pixels, pick the one with the highest grayscale value.
    Value gray;
    if (P == 3) {
        gray = rgb2gray(A, mr);
    } else {
        gray = A;
    }
    const keel::f32 *gv = gray.singleData();
    const keel::f32 *src = A.singleData();
    keel::f32 bestGray = -std::numeric_limits<keel::f32>::infinity();
    long bestIdx = -1;
    for (keel::usize i = 0; i < H * W; ++i) {
        if (dc[i] >= cutoff) {
            const keel::f32 gval = gv[i];
            if (gval > bestGray) {
                bestGray = gval;
                bestIdx = keel::as<long>(i);
            }
        }
    }
    keel::Array<keel::f32> atm(P);
    if (bestIdx < 0) {
        // No pixel above cutoff (shouldn't happen for sane input);
        // fall back to the pixel with maximum dark-channel value.
        keel::usize mi = 0;
        keel::f32 mv = dc[0];
        for (keel::usize i = 1; i < H * W; ++i)
            if (dc[i] > mv) { mv = dc[i]; mi = i; }
        for (keel::usize p = 0; p < P; ++p)
            atm[p] = src[p * H * W + mi];
    } else {
        for (keel::usize p = 0; p < P; ++p) {
            keel::f32 v = src[p * H * W + bestIdx];
            if (v == 0.0f)
                v = std::numeric_limits<keel::f32>::epsilon();  // eps('single')
            atm[p] = v;
        }
    }
    return atm;
}

// Per-pixel haze removal:
//   t_blend = 1 - omega * (1 - t)
//   radiance(x) = atm + (A(x) - atm) / max(t_blend, t0)
//   t_new = min(1, t_blend + amount)
//   B(x) = radiance(x) * t_new + atm * (1 - t_new)
//   B = clip(B, 0, 1)
Value recover_radiance(const Value &A, const Value &transmissionMap,
                       keel::ArrayRef<keel::f32> atm, keel::f64 amount,
                       keel::f64 omega, keel::Allocator *mr)
{
    const auto &d = A.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    const keel::usize P = d.is3D() ? d.pages() : 1;
    constexpr keel::f32 t0 = 0.1f;
    const keel::f32 *src = A.singleData();
    const keel::f32 *tm = transmissionMap.singleData();
    Value out;
    if (d.is3D())
        out = Value::matrix3d(H, W, P, ValueType::SINGLE, mr);
    else
        out = Value::matrix(H, W, ValueType::SINGLE, mr);
    keel::f32 *dst = out.singleDataMut();
    const keel::f32 famount = keel::as<keel::f32>(amount);
    const keel::f32 fomega = keel::as<keel::f32>(omega);
    for (keel::usize i = 0; i < H * W; ++i) {
        const keel::f32 t = tm[i];
        const keel::f32 tb = 1.0f - fomega * (1.0f - t);
        const keel::f32 tt = (tb > t0) ? tb : t0;
        const keel::f32 tnew = std::min(1.0f, tb + famount);
        for (keel::usize p = 0; p < P; ++p) {
            const keel::f32 a = atm[p > atm.size() - 1 ? atm.size() - 1 : p];
            const keel::f32 v = src[p * H * W + i];
            keel::f32 rad = a + (v - a) / tt;
            if (rad < 0.0f) rad = 0.0f;
            if (rad > 1.0f) rad = 1.0f;
            keel::f32 b = rad * tnew + a * (1.0f - tnew);
            if (b < 0.0f) b = 0.0f;
            if (b > 1.0f) b = 1.0f;
            dst[p * H * W + i] = b;
        }
    }
    return out;
}

// 1 - t (haze thickness, the second imreducehaze output).
Value one_minus(const Value &T, keel::Allocator *mr)
{
    const keel::usize H = T.dims().rows();
    const keel::usize W = T.dims().cols();
    Value out = Value::matrix(H, W, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    if (T.type() == ValueType::SINGLE) {
        const keel::f32 *td = T.singleData();
        for (keel::usize i = 0; i < H * W; ++i)
            od[i] = 1.0 - keel::as<keel::f64>(td[i]);
    } else {
        for (keel::usize i = 0; i < H * W; ++i)
            od[i] = 1.0 - T.elemAsDouble(i);
    }
    return out;
}

// Per-channel slice → H×W SINGLE Value (extract page p).
Value extract_plane(const Value &A, keel::usize p,
                    keel::Allocator *mr)
{
    const keel::usize H = A.dims().rows();
    const keel::usize W = A.dims().cols();
    Value out = Value::matrix(H, W, ValueType::SINGLE, mr);
    const keel::f32 *src = A.singleData() + p * H * W;
    std::memcpy(out.singleDataMut(), src, H * W * sizeof(keel::f32));
    return out;
}

// Stack P H×W single planes back into an H×W×P SINGLE.
Value stack_planes(keel::ArrayRef<Value> planes,
                   keel::Allocator *mr)
{
    const keel::usize P = planes.size();
    const keel::usize H = planes[0].dims().rows();
    const keel::usize W = planes[0].dims().cols();
    if (P == 1) {
        // Return a 2-D matrix copy to preserve grayscale shape.
        Value out = Value::matrix(H, W, ValueType::SINGLE, mr);
        std::memcpy(out.singleDataMut(), planes[0].singleData(),
                    H * W * sizeof(keel::f32));
        return out;
    }
    Value out = Value::matrix3d(H, W, P, ValueType::SINGLE, mr);
    for (keel::usize p = 0; p < P; ++p)
        std::memcpy(out.singleDataMut() + p * H * W,
                    planes[p].singleData(),
                    H * W * sizeof(keel::f32));
    return out;
}

// Global stretching post-processing:
//   gamma = 0.75 → A.^gamma
//   mat2gray
//   stretchlim [0.001, 0.999]
//   adjust clip: clip + 0.8 * (max(clip, mean(clip)) - clip)
//   imadjust (per-channel)
Value global_stretching(const Value &A, keel::Allocator *mr)
{
    const auto &d = A.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    const keel::usize P = d.is3D() ? d.pages() : 1;
    Value G = d.is3D() ? Value::matrix3d(H, W, P, ValueType::SINGLE, mr)
                        : Value::matrix(H, W, ValueType::SINGLE, mr);
    const keel::f32 *src = A.singleData();
    keel::f32 *dst = G.singleDataMut();
    for (keel::usize i = 0; i < H * W * P; ++i)
        dst[i] = std::pow(src[i], 0.75f);
    Value Nd = mat2gray(G, std::numeric_limits<keel::f64>::quiet_NaN(),
                        std::numeric_limits<keel::f64>::quiet_NaN(), mr);
    // mat2gray returns DOUBLE; convert to SINGLE for the rest of the
    // pipeline (cheap copy; keeps types consistent for stack_planes).
    Value N = d.is3D() ? Value::matrix3d(H, W, P, ValueType::SINGLE, mr)
                        : Value::matrix(H, W, ValueType::SINGLE, mr);
    {
        keel::f32 *nd = N.singleDataMut();
        const keel::f64 *dd_ = Nd.doubleData();
        for (keel::usize i = 0; i < H * W * P; ++i)
            nd[i] = keel::as<keel::f32>(dd_[i]);
    }
    constexpr keel::f64 alpha = 0.8;
    // MATLAB: clipLimit = stretchlim(N, [0.001 0.999]);          % 2×P
    //         clipLimit = clipLimit + alpha *
    //                     (max(clipLimit, mean(clipLimit, 2)) - clipLimit);
    // The mean is taken ACROSS CHANNELS (dim 2), so each row of
    // clipLimit (low / high) gets the same channel-mean value
    // broadcast across all P channels.  Then per-channel imadjust.
    keel::Array<keel::f64> lo_per(P), hi_per(P);
    for (keel::usize p = 0; p < P; ++p) {
        Value plane = (P == 1) ? N : extract_plane(N, p, mr);
        Value cl = stretchlim(plane, 0.001, 0.999, mr);
        lo_per[p] = cl.elemAsDouble(0);
        hi_per[p] = cl.elemAsDouble(1);
    }
    keel::f64 sum_lo = 0.0, sum_hi = 0.0;
    for (keel::usize p = 0; p < P; ++p) {
        sum_lo += lo_per[p];
        sum_hi += hi_per[p];
    }
    const keel::f64 mean_lo = sum_lo / keel::as<keel::f64>(P);
    const keel::f64 mean_hi = sum_hi / keel::as<keel::f64>(P);
    keel::Array<Value> outPlanes;
    outPlanes.reserve(P);
    for (keel::usize p = 0; p < P; ++p) {
        Value plane = (P == 1) ? N : extract_plane(N, p, mr);
        const keel::f64 lo_n = lo_per[p]
            + alpha * (std::max(lo_per[p], mean_lo) - lo_per[p]);
        const keel::f64 hi_n = hi_per[p]
            + alpha * (std::max(hi_per[p], mean_hi) - hi_per[p]);
        outPlanes.push_back(imadjust(plane, lo_n, hi_n, 0.0, 1.0, 1.0, mr));
    }
    return stack_planes(outPlanes, mr);
}

// Boost: B *= (1 + amount * boost * (1 - t))
//        where (1-t) is the transmissionMap input (1 - t_blend),
//        i.e. the *post-omega* transmission. The MATLAB code passes
//        1-T as the transmissionMap arg, where T is the haze
//        thickness (= 1 - t_blend). Therefore 1 - transmissionMap
//        passed in = T (thickness map). Hmm — re-read:
//           boostAmount = boostAmount * (1 - transmissionMap);
//           B = img .* (1 + (amount * boostAmount));
//        and is called as boosting(deHazed, amount, boostAmount,
//        1-T).  Inside, transmissionMap = 1-T, so 1 - transmissionMap
//        = T. So gain = boostAmount * T. Final B = dehazed*(1+amount*
//        boostAmount*T). Yes.
Value boost_postproc(const Value &dehazed, keel::f64 amount, keel::f64 boost,
                     const Value &T_thick, keel::Allocator *mr)
{
    const auto &d = dehazed.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    const keel::usize P = d.is3D() ? d.pages() : 1;
    Value out = d.is3D() ? Value::matrix3d(H, W, P, ValueType::SINGLE, mr)
                          : Value::matrix(H, W, ValueType::SINGLE, mr);
    const keel::f32 *src = dehazed.singleData();
    keel::f32 *dst = out.singleDataMut();
    const keel::f32 famount = keel::as<keel::f32>(amount);
    const keel::f32 fboost  = keel::as<keel::f32>(boost);
    // T_thick is DOUBLE H×W.
    for (keel::usize i = 0; i < H * W; ++i) {
        const keel::f32 Tv = keel::as<keel::f32>(T_thick.elemAsDouble(i));
        const keel::f32 g = 1.0f + famount * fboost * Tv;
        for (keel::usize p = 0; p < P; ++p) {
            keel::f32 v = src[p * H * W + i] * g;
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            dst[p * H * W + i] = v;
        }
    }
    return out;
}

// Cast final SINGLE output back to the input class.
Value cast_back(const Value &B, ValueType origClass,
                keel::Allocator *mr)
{
    if (origClass == ValueType::SINGLE) {
        // Already SINGLE; just clip [0,1].
        const keel::usize N = B.numel();
        Value out = B;
        // Make a copy under mr to ensure ownership.
        const auto &d = B.dims();
        if (d.is3D())
            out = Value::matrix3d(d.rows(), d.cols(), d.pages(),
                                  ValueType::SINGLE, mr);
        else
            out = Value::matrix(d.rows(), d.cols(), ValueType::SINGLE, mr);
        const keel::f32 *src = B.singleData();
        keel::f32 *dst = out.singleDataMut();
        for (keel::usize i = 0; i < N; ++i) {
            keel::f32 v = src[i];
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            dst[i] = v;
        }
        return out;
    }
    if (origClass == ValueType::DOUBLE) {
        const auto &d = B.dims();
        Value out = d.is3D() ? Value::matrix3d(d.rows(), d.cols(), d.pages(),
                                               ValueType::DOUBLE, mr)
                              : Value::matrix(d.rows(), d.cols(),
                                              ValueType::DOUBLE, mr);
        keel::f64 *dst = out.doubleDataMut();
        const keel::usize N = B.numel();
        if (B.type() == ValueType::SINGLE) {
            const keel::f32 *src = B.singleData();
            for (keel::usize i = 0; i < N; ++i) {
                keel::f64 v = keel::as<keel::f64>(src[i]);
                if (v < 0.0) v = 0.0;
                if (v > 1.0) v = 1.0;
                dst[i] = v;
            }
        } else {
            for (keel::usize i = 0; i < N; ++i) {
                keel::f64 v = B.elemAsDouble(i);
                if (v < 0.0) v = 0.0;
                if (v > 1.0) v = 1.0;
                dst[i] = v;
            }
        }
        return out;
    }
    if (origClass == ValueType::UINT8)  return im2uint8(B, mr);
    if (origClass == ValueType::UINT16) return im2uint16(B, mr);
    // Fallback (int8/int16): use im2uint8-ish — MATLAB only documents
    // single/double/uint8/uint16 as supported input classes.
    return im2uint8(B, mr);
}

void imreducehaze_argcheck(const Value &I, keel::f64 amount,
                           keel::StringRef method,
                           keel::StringRef contrast_enhancement,
                           keel::f64 boost_amount,
                           keel::String &m_out, keel::String &ce_out,
                           bool &isRGB_out)
{
    if (!std::isfinite(amount) || amount < 0.0 || amount > 1.0)
        throw Error("imreducehaze: amount must be in [0, 1]",
                    0, 0, "imreducehaze", "", "numkit:imreducehaze:amount");
    const auto &d = I.dims();
    if (d.rows() == 0 || d.cols() == 0)
        throw Error("imreducehaze: I must be non-empty",
                    0, 0, "imreducehaze", "", "numkit:imreducehaze:empty");
    const bool isRGB = d.is3D() && d.pages() == 3;
    const bool isGray = !d.is3D() || (d.is3D() && d.pages() == 1);
    if (!isRGB && !isGray)
        throw Error("imreducehaze: I must be H×W (grayscale) or H×W×3 (RGB)",
                    0, 0, "imreducehaze", "", "numkit:imreducehaze:shape");
    const ValueType origClass = I.type();
    if (origClass != ValueType::UINT8 && origClass != ValueType::UINT16
        && origClass != ValueType::SINGLE && origClass != ValueType::DOUBLE)
        throw Error("imreducehaze: I must be uint8 / uint16 / single / double",
                    0, 0, "imreducehaze", "", "numkit:imreducehaze:class");

    keel::String m = lower_str(method);
    if (m != "simpledcp" && m != "approxdcp")
        throw Error("imreducehaze: Method must be 'simpledcp' or 'approxdcp'",
                    0, 0, "imreducehaze", "", "numkit:imreducehaze:method");
    keel::String ce = lower_str(contrast_enhancement);
    if (ce != "global" && ce != "boost" && ce != "none")
        throw Error("imreducehaze: ContrastEnhancement must be 'global', "
                    "'boost', or 'none'",
                    0, 0, "imreducehaze", "",
                    "numkit:imreducehaze:contrastEnhancement");
    if (ce == "boost" && (boost_amount < 0.0 || boost_amount > 1.0
                          || !std::isfinite(boost_amount)))
        throw Error("imreducehaze: BoostAmount must be in [0, 1]",
                    0, 0, "imreducehaze", "",
                    "numkit:imreducehaze:boostAmount");
    m_out = std::move(m);
    ce_out = std::move(ce);
    isRGB_out = isRGB;
}

}  // namespace

Value imreducehaze(const Value &I, keel::f64 amount,
                   keel::StringRef method,
                   const Value &atmospheric_light,
                   keel::StringRef contrast_enhancement,
                   keel::f64 boost_amount,
                   Value &t_out, Value &L_out,
                   keel::Allocator *mr)
{
    keel::String m, ce;
    bool isRGB = false;
    imreducehaze_argcheck(I, amount, method, contrast_enhancement, boost_amount,
                          m, ce, isRGB);
    const ValueType origClass = I.type();

    // ── Short-circuit: amount == 0 → passthrough ──────────────────
    if (amount == 0.0) {
        t_out = Value::Empty;
        L_out = Value::Empty;
        return I;
    }

    // ── Promote to SINGLE [0,1] ──────────────────────────────────
    Value A = to_single01(I, mr);
    const keel::usize H = A.dims().rows();
    const keel::usize W = A.dims().cols();
    const keel::usize P = A.dims().is3D() ? A.dims().pages() : 1;

    // ── Resolve / estimate atmospheric light ─────────────────────
    keel::Array<keel::f32> atm;
    if (!atmospheric_light.isEmpty()) {
        const keel::usize N = atmospheric_light.numel();
        if (isRGB && N != 3)
            throw Error("imreducehaze: AtmosphericLight must be a 3-element "
                        "vector for RGB input",
                        0, 0, "imreducehaze", "",
                        "numkit:imreducehaze:atmRGB");
        if (!isRGB && N != 1)
            throw Error("imreducehaze: AtmosphericLight must be a scalar for "
                        "grayscale input",
                        0, 0, "imreducehaze", "",
                        "numkit:imreducehaze:atmGray");
        atm.resize(P);
        for (keel::usize p = 0; p < P; ++p)
            atm[p] = keel::as<keel::f32>(atmospheric_light.elemAsDouble(
                p >= N ? N - 1 : p));
    }

    // ── Dispatch on Method ───────────────────────────────────────
    keel::f64 omega;
    Value transmissionMap;
    int patchSize = keel::as<int>(std::ceil(
        keel::as<keel::f64>(std::min(H, W)) / 400.0 * 15.0));
    if (patchSize < 1) patchSize = 1;
    Value se = make_square_strel(patchSize, mr);

    if (m == "simpledcp") {
        if (atm.empty()) atm = simpledcp_atm_light(A, mr);
        // transmissionMap = 1 - min(A, [], 3)
        Value mp = min_across_pages(A, mr);
        const keel::usize N = mp.numel();
        Value tmap = Value::matrix(H, W, ValueType::SINGLE, mr);
        keel::f32 *td = tmap.singleDataMut();
        const keel::f32 *mpd = mp.singleData();
        for (keel::usize i = 0; i < N; ++i) td[i] = 1.0f - mpd[i];
        // Guided filter: 5×5 (radius 2), eps = 0.01.
        // Self-guided (tmap is the transmission map; used as own guide).
        transmissionMap = imguidedfilter(tmap, tmap, 5, 0.01, mr);
        // Clamp [0, 1].
        keel::f32 *tt = transmissionMap.singleDataMut();
        for (keel::usize i = 0; i < N; ++i) {
            if (tt[i] < 0.0f) tt[i] = 0.0f;
            if (tt[i] > 1.0f) tt[i] = 1.0f;
        }
        omega = 0.9;
    } else {  // approxdcp
        Value dcBase = min_across_pages(A, mr);
        Value darkChannel = imerode(dcBase, se, mr);
        if (atm.empty()) atm = approxdcp_atm_light(A, darkChannel, mr);
        // transmissionMap = 1 - imopen(A./atm, strel)
        Value normI = normalize_by_atm_then_min(A, atm, mr);
        Value opened = imopen(normI, se, mr);
        Value tmap = Value::matrix(H, W, ValueType::SINGLE, mr);
        keel::f32 *td = tmap.singleDataMut();
        const keel::f32 *od = opened.singleData();
        const keel::usize N = tmap.numel();
        for (keel::usize i = 0; i < N; ++i) td[i] = 1.0f - od[i];
        // Guided filter: nhood = 2*radius+1, radius=ceil(min(H,W)/50).
        // nk's imguidedfilter accepts grayscale guides only; for
        // RGB input we use rgb2gray(A) as guide (close approximation
        // to MATLAB's RGB-guide variant). MATLAB also uses a Fast-
        // Guided-Filter subsample factor of min(4, radius); nk's
        // imguidedfilter does not yet expose subsample, so we run
        // the full-resolution variant (slower but higher quality).
        int radius = keel::as<int>(std::ceil(
            keel::as<keel::f64>(std::min(H, W)) / 50.0));
        if (radius < 1) radius = 1;
        const int nhood = 2 * radius + 1;
        Value guide = (P == 3) ? rgb2gray(A, mr) : A;
        transmissionMap = imguidedfilter(tmap, guide, nhood, 1e-4, mr);
        // Clamp [0, 1].
        keel::f32 *tt = transmissionMap.singleDataMut();
        for (keel::usize i = 0; i < N; ++i) {
            if (tt[i] < 0.0f) tt[i] = 0.0f;
            if (tt[i] > 1.0f) tt[i] = 1.0f;
        }
        omega = 0.95;
    }

    // ── Haze thickness T = 1 - transmissionMap ──────────────────
    t_out = one_minus(transmissionMap, mr);

    // ── Recover scene radiance + blend by amount ────────────────
    Value deHazed = recover_radiance(A, transmissionMap, atm, amount,
                                     omega, mr);

    // ── L output ────────────────────────────────────────────────
    if (isRGB) {
        L_out = Value::matrix(1, 3, ValueType::DOUBLE, mr);
        keel::f64 *ld = L_out.doubleDataMut();
        ld[0] = atm[0]; ld[1] = atm[1]; ld[2] = atm[2];
    } else {
        L_out = Value::matrix(1, 1, ValueType::DOUBLE, mr);
        L_out.doubleDataMut()[0] = atm[0];
    }

    // ── Post-processing ─────────────────────────────────────────
    Value enhanced;
    if (ce == "global") {
        enhanced = global_stretching(deHazed, mr);
    } else if (ce == "boost") {
        const keel::f64 bA = (boost_amount > 0.0) ? boost_amount : 0.1;
        enhanced = boost_postproc(deHazed, amount, bA, t_out, mr);
    } else {
        enhanced = deHazed;
    }

    return cast_back(enhanced, origClass, mr);
}

} // namespace numkit::image
