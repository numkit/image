// toolboxes/image/src/segment/segment.cpp
//
// Lightweight segmentation utilities — no DL, no fancy numerics.
// Each entry is O(N) on the input. Composes with the cycle-22
// bwlabel / regionprops infrastructure.

#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/array.h>
#include <keel/math.h>
#include <numkit/image/segment/segment.hpp>
#include <numkit/image/internal/type_gates.hpp>
#include <numkit/image/filter/filter.hpp>
#include <numkit/image/geom/geom.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <keel/static_array.h>
#include <cstdint>
#include <utility>

namespace numkit::image {

namespace {

template <typename T, typename Cmp>
class ArrayPriorityQueue {
public:
    explicit ArrayPriorityQueue(Cmp cmp = Cmp{}, keel::Allocator *alloc = nullptr)
        : m_data(alloc), m_cmp(cmp) {}
    bool empty() const noexcept { return m_data.empty(); }
    const T &top() const noexcept { return m_data.front(); }
    template <typename... Args>
    void emplace(Args &&...args) {
        m_data.emplace_back(std::forward<Args>(args)...);
        std::push_heap(m_data.begin(), m_data.end(), m_cmp);
    }
    void push(const T &v) {
        m_data.push_back(v);
        std::push_heap(m_data.begin(), m_data.end(), m_cmp);
    }
    void pop() {
        std::pop_heap(m_data.begin(), m_data.end(), m_cmp);
        m_data.pop_back();
    }
private:
    keel::Array<T> m_data;
    Cmp m_cmp;
};

bool nonZero(keel::f64 v) { return v != 0.0; }

void requireSameShape(const Value &A, const Value &B, const char *fn) {
    if (A.numel() != B.numel() ||
        A.dims().rows() != B.dims().rows() ||
        A.dims().cols() != B.dims().cols())
        throw Error(keel::format("{}: inputs must have the same shape", fn),
                    0, 0, fn, "", "numkit:image:segment:shape");
}

} // anonymous

Value dice(const Value &A, const Value &B, keel::Allocator *mr)
{
    requireSameShape(A, B, "dice");
    const keel::usize N = A.numel();
    keel::i64 inter = 0, sumA = 0, sumB = 0;
    for (keel::usize i = 0; i < N; ++i) {
        const bool a = nonZero(A.elemAsDouble(i));
        const bool b = nonZero(B.elemAsDouble(i));
        if (a) ++sumA;
        if (b) ++sumB;
        if (a && b) ++inter;
    }
    const keel::i64 denom = sumA + sumB;
    const keel::f64 d = (denom == 0) ? 1.0
                                  : 2.0 * keel::f64(inter) / keel::f64(denom);
    return Value::scalar(d, mr);
}

Value jaccard(const Value &A, const Value &B, keel::Allocator *mr)
{
    requireSameShape(A, B, "jaccard");
    const keel::usize N = A.numel();
    keel::i64 inter = 0, uni = 0;
    for (keel::usize i = 0; i < N; ++i) {
        const bool a = nonZero(A.elemAsDouble(i));
        const bool b = nonZero(B.elemAsDouble(i));
        if (a && b) ++inter;
        if (a || b) ++uni;
    }
    const keel::f64 j = (uni == 0) ? 1.0
                                : keel::f64(inter) / keel::f64(uni);
    return Value::scalar(j, mr);
}

Value boundarymask(const Value &L, keel::i32 conn, keel::Allocator *mr)
{
    internal::requireImageClass(L, "boundarymask", "L", 1,
                                internal::ImgClasses::AnyNumericReal);
    if (conn != 4) conn = 8;
    const keel::i32 H = keel::as<keel::i32>(L.dims().rows());
    const keel::i32 W = keel::as<keel::i32>(L.dims().cols());
    Value out = Value::matrix(keel::as<keel::usize>(H),
                              keel::as<keel::usize>(W),
                              ValueType::LOGICAL, mr);
    if (H == 0 || W == 0) return out;

    // Read L as int (round); column-major access via elemAsDouble.
    auto labAt = [&](keel::i32 r, keel::i32 c) -> keel::i64 {
        return keel::as<keel::i64>(keel::round(
            L.elemAsDouble(keel::as<keel::usize>(c) * keel::as<keel::usize>(H) +
                            keel::as<keel::usize>(r))));
    };

    keel::u8 *od = out.logicalDataMut();
    static const keel::i32 dr8[8] = { -1,-1,-1, 0, 0, 1, 1, 1 };
    static const keel::i32 dc8[8] = { -1, 0, 1,-1, 1,-1, 0, 1 };
    static const keel::i32 dr4[4] = { -1, 1, 0, 0 };
    static const keel::i32 dc4[4] = {  0, 0,-1, 1 };
    const keel::i32 *dr = (conn == 4) ? dr4 : dr8;
    const keel::i32 *dc = (conn == 4) ? dc4 : dc8;
    const keel::i32 K   = (conn == 4) ? 4    : 8;

    for (keel::i32 c = 0; c < W; ++c)
        for (keel::i32 r = 0; r < H; ++r) {
            const keel::i64 me = labAt(r, c);
            // Background pixel (label 0) is never a boundary in the
            // MATLAB sense; for a pure binary mask we treat the
            // foreground perimeter.
            bool isBoundary = false;
            for (keel::i32 k = 0; k < K; ++k) {
                const keel::i32 nr = r + dr[k];
                const keel::i32 nc = c + dc[k];
                if (nr < 0 || nr >= H || nc < 0 || nc >= W) {
                    if (me != 0) { isBoundary = true; break; }
                    continue;
                }
                const keel::i64 ne = labAt(nr, nc);
                if (ne != me) { isBoundary = true; break; }
            }
            // Background-only neighbourhood: not a boundary.
            if (me == 0) isBoundary = false;
            od[keel::as<keel::usize>(c) * keel::as<keel::usize>(H) +
               keel::as<keel::usize>(r)] = isBoundary ? 1u : 0u;
        }
    return out;
}

Value label2idx(const Value &L, keel::Allocator *mr)
{
    const keel::usize N = L.numel();
    keel::i64 maxLab = 0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::i64 v = keel::as<keel::i64>(
            keel::round(L.elemAsDouble(i)));
        if (v > maxLab) maxLab = v;
    }
    Value cell = Value::cell(keel::as<keel::usize>(maxLab), 1, mr);
    if (maxLab == 0) return cell;

    // First pass: count per label.
    keel::Array<keel::usize> counts(keel::as<keel::usize>(maxLab) + 1, 0);
    for (keel::usize i = 0; i < N; ++i) {
        const keel::i64 v = keel::as<keel::i64>(
            keel::round(L.elemAsDouble(i)));
        if (v > 0) ++counts[keel::as<keel::usize>(v)];
    }
    // Allocate per-label column vectors (1-based indices).
    keel::Array<Value> bufs;
    bufs.reserve(maxLab);
    keel::Array<keel::f64 *> wptr(keel::as<keel::usize>(maxLab) + 1);
    keel::Array<keel::usize> cur(keel::as<keel::usize>(maxLab) + 1, 0);
    for (keel::i64 k = 1; k <= maxLab; ++k) {
        Value v = Value::matrix(counts[keel::as<keel::usize>(k)], 1,
                                ValueType::DOUBLE, mr);
        wptr[keel::as<keel::usize>(k)] = (counts[keel::as<keel::usize>(k)] > 0)
                                       ? v.doubleDataMut() : nullptr;
        bufs.push_back(std::move(v));
    }
    // Second pass: fill.
    for (keel::usize i = 0; i < N; ++i) {
        const keel::i64 v = keel::as<keel::i64>(
            keel::round(L.elemAsDouble(i)));
        if (v > 0) {
            wptr[keel::as<keel::usize>(v)][cur[keel::as<keel::usize>(v)]] =
                keel::as<keel::f64>(i + 1); // MATLAB 1-based
            ++cur[keel::as<keel::usize>(v)];
        }
    }
    for (keel::i64 k = 1; k <= maxLab; ++k)
        cell.cellAt(keel::as<keel::usize>(k - 1)) =
            std::move(bufs[keel::as<keel::usize>(k - 1)]);
    return cell;
}

Value grayconnected(const Value &I, keel::i32 row, keel::i32 col, keel::f64 tol, keel::Allocator *mr)
{
    internal::requireImageClass(I, "grayconnected", "I", 1,
                                internal::ImgClasses::FloatIntNoLog);
    const keel::i32 H = keel::as<keel::i32>(I.dims().rows());
    const keel::i32 W = keel::as<keel::i32>(I.dims().cols());
    Value out = Value::matrix(keel::as<keel::usize>(H),
                              keel::as<keel::usize>(W),
                              ValueType::LOGICAL, mr);
    if (H == 0 || W == 0) return out;

    // 1-based MATLAB → 0-based.
    const keel::i32 r0 = row - 1;
    const keel::i32 c0 = col - 1;
    if (r0 < 0 || r0 >= H || c0 < 0 || c0 >= W)
        throw Error("grayconnected: seed out of bounds",
                    0, 0, "grayconnected", "", "numkit:grayconnected:seed");

    // Auto-pick tolerance if caller passed a negative sentinel.
    if (tol < 0.0) {
        switch (I.type()) {
            case ValueType::UINT8:  tol = 32.0;            break;
            case ValueType::INT8:   tol = 32.0;            break;
            case ValueType::UINT16: tol = 32.0 * 256.0;    break;
            case ValueType::INT16:  tol = 32.0 * 256.0;    break;
            default:                tol = 0.125;            break; // [0, 1] floats
        }
    }

    auto idxAt = [&](keel::i32 r, keel::i32 c) {
        return keel::as<keel::usize>(c) * keel::as<keel::usize>(H) +
               keel::as<keel::usize>(r);
    };
    const keel::f64 seedVal = I.elemAsDouble(idxAt(r0, c0));
    keel::u8 *od = out.logicalDataMut();
    keel::Array<keel::u8> visited(keel::as<keel::usize>(H) *
                                       keel::as<keel::usize>(W), 0);

    // BFS with an explicit deque (vector + read head).
    keel::Array<std::pair<keel::i32, keel::i32>> q;
    q.reserve(keel::as<keel::usize>(H) * keel::as<keel::usize>(W) / 4 + 4);
    q.emplace_back(r0, c0);
    visited[keel::as<keel::usize>(r0) * keel::as<keel::usize>(W) +
            keel::as<keel::usize>(c0)] = 1;
    od[idxAt(r0, c0)] = 1;
    static const keel::i32 dr8[8] = { -1,-1,-1, 0, 0, 1, 1, 1 };
    static const keel::i32 dc8[8] = { -1, 0, 1,-1, 1,-1, 0, 1 };

    keel::usize head = 0;
    while (head < q.size()) {
        const auto [r, c] = q[head++];
        for (keel::i32 k = 0; k < 8; ++k) {
            const keel::i32 nr = r + dr8[k];
            const keel::i32 nc = c + dc8[k];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
            const keel::usize fi = keel::as<keel::usize>(nr) *
                                keel::as<keel::usize>(W) +
                              keel::as<keel::usize>(nc);
            if (visited[fi]) continue;
            const keel::f64 v = I.elemAsDouble(idxAt(nr, nc));
            if (keel::abs(v - seedVal) > tol) continue;
            visited[fi] = 1;
            od[idxAt(nr, nc)] = 1;
            q.emplace_back(nr, nc);
        }
    }
    return out;
}

// ── graydiffweight (FMM intensity-difference weights) ──────────────
//
// MATLAB R2025b graydiffweight.m → call sequence:
//   d = |I − refGrayVal|
//   if cutoff < Inf: isSuppressed = (d > cutoff)
//   d_scaled = imlinscale(d, [1e-3, 1])    — linear (min, max)→(1e-3, 1)
//   if cutoff: d_scaled(isSuppressed) = 1
//   W = 1 ./ (d_scaled .^ (1 / rolloffFactor))
// Output class is single if input was single, otherwise double.
// (The 4 input signatures collapse to a single scalar `refGrayVal`
// here; the adapter computes the mean over MASK / (C, R) / (C, R, P)
// before dispatching to this typed entry-point.)
Value graydiffweight(const Value &I, keel::f64 ref_gray_val,
                     keel::f64 rolloff_factor, keel::f64 cutoff,
                     keel::Allocator *mr)
{
    if (!(rolloff_factor > 0.0))
        throw Error("graydiffweight: RolloffFactor must be positive",
                    0, 0, "graydiffweight", "", "numkit:graydiffweight:rolloff");
    if (!(cutoff >= 0.0))
        throw Error("graydiffweight: GrayDifferenceCutoff must be "
                    "non-negative",
                    0, 0, "graydiffweight", "", "numkit:graydiffweight:cutoff");

    const ValueType outT = (I.type() == ValueType::SINGLE)
                          ? ValueType::SINGLE : ValueType::DOUBLE;
    const auto &d = I.dims();
    const keel::usize N = I.numel();
    Value W = d.is3D()
        ? Value::matrix3d(d.rows(), d.cols(), d.pages(), outT, mr)
        : Value::matrix(d.rows(), d.cols(), outT, mr);
    if (N == 0) return W;

    // Pass 1 — compute |I − ref| in DOUBLE temporary; track min/max.
    keel::Array<keel::f64> diff(N, mr);
    keel::f64 dmin = keel::Limits<keel::f64>::infinity();
    keel::f64 dmax = -keel::Limits<keel::f64>::infinity();
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 v = keel::abs(I.elemAsDouble(i) - ref_gray_val);
        diff[i] = v;
        if (v < dmin) dmin = v;
        if (v > dmax) dmax = v;
    }

    // Linear scale to [1e-3, 1].
    const keel::f64 lo = 1e-3, hi = 1.0;
    const keel::f64 range = dmax - dmin;
    const keel::f64 slope = (range > 0.0) ? (hi - lo) / range : 0.0;

    const keel::f64 inv_rolloff = 1.0 / rolloff_factor;
    const bool finite_cutoff = keel::is_finite(cutoff);

    auto store = [&](keel::usize i, keel::f64 v) {
        if (outT == ValueType::DOUBLE) W.doubleDataMut()[i] = v;
        else                            W.singleDataMut()[i] = keel::as<keel::f32>(v);
    };

    for (keel::usize i = 0; i < N; ++i) {
        keel::f64 s;
        if (range > 0.0) s = lo + slope * (diff[i] - dmin);
        else             s = lo;     // constant-image edge case
        if (finite_cutoff && diff[i] > cutoff) s = hi;
        const keel::f64 w = 1.0 / keel::pow(s, inv_rolloff);
        store(i, w);
    }
    return W;
}

// ── gradientweight (FMM gradient-based pixel weights) ──────────────
//
// MATLAB R2025b gradientweight.m / images.internal.imgradientdog
// algorithm:
//   r        = ceil(2*sigma)             (per axis)
//   x        = -r:r
//   hx(x)    = -x * exp(-x^2 / (2*sigma_x^2))
//   norm     = sum(hx(1..r))             (positive half — MATLAB 1-idx)
//   hx       = hx / norm
//   hy       = same with sigma_y, oriented vertical (column)
//   Gx       = imfilter(I, hx, 'replicate')
//   Gy       = imfilter(I, hy, 'replicate')
//   W = hypot(Gx, Gy)
//   W = imlinscale(W, [0 1])
//   W = W ^ (1 / rolloff_factor)
//   W = (1 - W) / (1 + W)
//   W(W < cutoff) = 1e-3
// Output class: single if input single, else double.
// 2-D only here (MATLAB calls imgradientdog3 for 3-D volumes).
Value gradientweight(const Value &I, keel::f64 sigma_x, keel::f64 sigma_y,
                     keel::f64 rolloff_factor, keel::f64 weight_cutoff,
                     keel::Allocator *mr)
{
    internal::requireImageClass(I, "gradientweight", "I", 1,
                                internal::ImgClasses::FloatIntNoLog);
    if (!(sigma_x > 0.0) || !keel::is_finite(sigma_x)
     || !(sigma_y > 0.0) || !keel::is_finite(sigma_y))
        throw Error("gradientweight: sigma must be positive and finite",
                    0, 0, "gradientweight", "", "numkit:gradientweight:sigma");
    if (!(rolloff_factor > 0.0) || !keel::is_finite(rolloff_factor))
        throw Error("gradientweight: RolloffFactor must be positive "
                    "and finite",
                    0, 0, "gradientweight", "", "numkit:gradientweight:rolloff");
    if (!(weight_cutoff >= 1e-3 && weight_cutoff <= 1.0))
        throw Error("gradientweight: WeightCutoff must be in [1e-3, 1]",
                    0, 0, "gradientweight", "", "numkit:gradientweight:cutoff");
    if (I.dims().is3D())
        throw Error("gradientweight: 3-D inputs not supported "
                    "(slice and call per page)",
                    0, 0, "gradientweight", "", "numkit:gradientweight:dim");

    const ValueType outT = (I.type() == ValueType::SINGLE)
                          ? ValueType::SINGLE : ValueType::DOUBLE;
    const keel::usize H = I.dims().rows();
    const keel::usize W_ = I.dims().cols();
    const keel::usize N = H * W_;

    // Empty: return same-shape DOUBLE empty.
    if (N == 0) return Value::matrix(H, W_, outT, mr);

    // Constant-image fast-path: |max - min| <= eps(maxabs)*1000 → W = 1.
    keel::f64 minI = keel::Limits<keel::f64>::infinity();
    keel::f64 maxI = -keel::Limits<keel::f64>::infinity();
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 v = I.elemAsDouble(i);
        if (v < minI) minI = v;
        if (v > maxI) maxI = v;
    }
    const keel::f64 absMax = std::max(keel::abs(minI), keel::abs(maxI));
    const keel::f64 e = keel::nextafter(absMax,
                          keel::Limits<keel::f64>::infinity()) - absMax;
    if (maxI - e * 1000.0 <= minI) {
        Value Wones = Value::matrix(H, W_, outT, mr);
        if (outT == ValueType::DOUBLE) {
            keel::f64 *p = Wones.doubleDataMut();
            for (keel::usize i = 0; i < N; ++i) p[i] = 1.0;
        } else {
            keel::f32 *p = Wones.singleDataMut();
            for (keel::usize i = 0; i < N; ++i) p[i] = 1.0f;
        }
        return Wones;
    }

    // Build DoG kernel along a single axis with given sigma. `norm_count`
    // is the number of left-half elements summed for normalisation
    // (MATLAB uses `filtRadius(1)` for BOTH hx and hy — so for hy on
    // anisotropic σ, this is the x-axis radius, not the y-axis radius.
    // We replicate the MATLAB R2025b behaviour exactly).
    auto build_dog = [&](keel::f64 sigma, keel::i32 norm_count, keel::usize &len) {
        const keel::i32 r = keel::as<keel::i32>(keel::ceil(2.0 * sigma));
        len = 2 * keel::as<keel::usize>(r) + 1;
        keel::Array<keel::f64> h(len, 0.0, mr);
        keel::f64 norm = 0.0;
        for (keel::i32 k = 0; k < keel::as<keel::i32>(len); ++k) {
            const keel::i32 x = k - r;
            h[k] = -keel::as<keel::f64>(x)
                 * keel::exp(-keel::as<keel::f64>(x * x)
                            / (2.0 * sigma * sigma));
            // MATLAB: norm = sum(h(1..norm_count)) -- 1-indexed
            // (k = 0..norm_count-1 in 0-indexed, the left half).
            if (k < norm_count) norm += h[k];
        }
        if (norm == 0.0) norm = 1.0;
        for (keel::usize k = 0; k < len; ++k) h[k] /= norm;
        return h;
    };

    const keel::i32 rx = keel::as<keel::i32>(keel::ceil(2.0 * sigma_x));
    keel::usize hxLen = 0, hyLen = 0;
    auto hxBuf = build_dog(sigma_x, rx, hxLen);
    auto hyBuf = build_dog(sigma_y, rx, hyLen);     // MATLAB-bug-compatible

    // Cast I to DOUBLE for filtering (matches MATLAB internal).
    Value Id;
    if (I.type() == ValueType::DOUBLE) {
        Id = I;
    } else {
        Id = Value::matrix(H, W_, ValueType::DOUBLE, mr);
        keel::f64 *p = Id.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i) p[i] = I.elemAsDouble(i);
    }

    // hx as 1×Nx row vector; hy as Ny×1 column vector.
    Value hx = Value::matrix(1, hxLen, ValueType::DOUBLE, mr);
    for (keel::usize k = 0; k < hxLen; ++k)
        hx.doubleDataMut()[k] = hxBuf[k];
    Value hy = Value::matrix(hyLen, 1, ValueType::DOUBLE, mr);
    for (keel::usize k = 0; k < hyLen; ++k)
        hy.doubleDataMut()[k] = hyBuf[k];

    // Gx, Gy via imfilter (replicate boundary, same size, correlation).
    Value Gx = imfilter(Id, hx, PadMode::Replicate, 0.0,
                        /*full=*/false, /*flip_kernel=*/false, mr);
    Value Gy = imfilter(Id, hy, PadMode::Replicate, 0.0,
                        /*full=*/false, /*flip_kernel=*/false, mr);

    // Gmag = hypot(Gx, Gy); track min/max for imlinscale.
    keel::Array<keel::f64> Gmag(N, 0.0, mr);
    keel::f64 gmin = keel::Limits<keel::f64>::infinity();
    keel::f64 gmax = -keel::Limits<keel::f64>::infinity();
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 gx = Gx.elemAsDouble(i);
        const keel::f64 gy = Gy.elemAsDouble(i);
        const keel::f64 g = keel::hypot(gx, gy);
        Gmag[i] = g;
        if (g < gmin) gmin = g;
        if (g > gmax) gmax = g;
    }
    // imlinscale to [0, 1].
    const keel::f64 gabsMin = std::min(keel::abs(gmin), keel::abs(gmax));
    const keel::f64 e2 = keel::nextafter(gabsMin,
                          keel::Limits<keel::f64>::infinity()) - gabsMin;
    const bool spread_ok = (gmax - gmin) > e2;
    const keel::f64 slope = spread_ok ? 1.0 / (gmax - gmin) : 0.0;

    const keel::f64 inv_rolloff = 1.0 / rolloff_factor;
    const keel::f64 floorOfW    = 1e-3;

    Value Wout = Value::matrix(H, W_, outT, mr);
    auto store = [&](keel::usize i, keel::f64 v) {
        if (outT == ValueType::DOUBLE) Wout.doubleDataMut()[i] = v;
        else                            Wout.singleDataMut()[i] = keel::as<keel::f32>(v);
    };
    for (keel::usize i = 0; i < N; ++i) {
        keel::f64 w = spread_ok ? slope * (Gmag[i] - gmin) : 0.0;
        // W = W^(1/rolloff); W = (1-W)/(1+W); cutoff
        w = keel::pow(w, inv_rolloff);
        w = (1.0 - w) / (1.0 + w);
        if (w < weight_cutoff) w = floorOfW;
        store(i, w);
    }
    return Wout;
}

// ── regionfill (Laplacian inpainting) ──────────────────────────────
//
// MATLAB R2025b regionfill.m algorithm:
//
//   1. maskPerimeter = (dilate(mask, cross) & !mask)
//        — the 1-pixel ring of known boundary values around each hole.
//   2. For each interior masked pixel (r, c):
//        4 J(r,c) − J_N − J_S − J_W − J_E = sum of perimeter neighbours
//      Pixels on the image border get a 3-neighbour stencil; corners
//      get 2.
//   3. Stack the equations into a sparse SPD linear system Ax = b
//      indexed by mask pixel and solve.
//
// MATLAB uses sparse-direct (UMFPACK via `\`); we use unpreconditioned
// conjugate gradient at tol 1e-12 against the implicit stencil matrix
// (no need to materialise A — apply Laplacian directly via the mask).
// CG converges in O(√n) iterations for the 2-D Laplacian, so a typical
// 1000-unknown mask takes ~30 iterations.
//
// References: Gonzalez & Woods §3.4 (Laplacian);
//             Press et al., *Numerical Recipes* §2.7.
Value regionfill(const Value &I, const Value &mask,
                 keel::Allocator *mr)
{
    internal::requireImageClass(I, "regionfill", "I", 1,
                                internal::ImgClasses::FloatInt);
    if (I.dims().is3D())
        throw Error("regionfill: I must be 2-D",
                    0, 0, "regionfill", "", "numkit:regionfill:mustBe2D");
    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    if (H < 3 || W < 3)
        throw Error("regionfill: I must be at least 3x3",
                    0, 0, "regionfill", "",
                    "numkit:regionfill:mustBeLargerThan2by2");
    if (mask.dims().rows() != H || mask.dims().cols() != W
     || mask.dims().is3D())
        throw Error("regionfill: MASK must be 2-D and the same size as I",
                    0, 0, "regionfill", "",
                    "numkit:regionfill:mustBeSameSizeAsI");
    const keel::usize N = H * W;

    // Convert mask to plain uint8 array (0/1) in column-major (matches
    // I's storage).
    keel::Array<keel::u8> M(N, 0, mr);
    if (mask.isLogical()) {
        const keel::u8 *src = mask.logicalData();
        for (keel::usize i = 0; i < N; ++i) M[i] = src[i] ? 1 : 0;
    } else {
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 v = mask.elemAsDouble(i);
            if (keel::is_nan(v))
                throw Error("regionfill: MASK cannot contain NaN",
                            0, 0, "regionfill", "",
                            "numkit:regionfill:maskNaN");
            M[i] = (v != 0.0) ? 1 : 0;
        }
    }

    // Cast I to DOUBLE for the solve; remember original class for output.
    const ValueType inT = I.type();
    keel::Array<keel::f64> Idata(N, 0.0, mr);
    for (keel::usize i = 0; i < N; ++i) Idata[i] = I.elemAsDouble(i);

    // Helper: column-major (r, c) → linear index.
    auto lin = [&](keel::usize r, keel::usize c) { return c * H + r; };

    // Compute mask perimeter: pixels (r,c) where M[r,c]==0 AND at least
    // one of the 4 N/S/W/E neighbours is masked.
    keel::Array<keel::u8> P(N, 0, mr);
    for (keel::usize c = 0; c < W; ++c) {
        for (keel::usize r = 0; r < H; ++r) {
            const keel::usize k = lin(r, c);
            if (M[k]) continue;
            bool touches = false;
            if (r > 0      && M[lin(r - 1, c)]) touches = true;
            if (r + 1 < H  && M[lin(r + 1, c)]) touches = true;
            if (c > 0      && M[lin(r, c - 1)]) touches = true;
            if (c + 1 < W  && M[lin(r, c + 1)]) touches = true;
            P[k] = touches ? 1 : 0;
        }
    }

    // Build mask-pixel-to-unknown index map.
    keel::Array<keel::i32> idxOf(N, -1, mr);
    keel::Array<keel::usize>  pixOf(mr);
    pixOf.reserve(N / 4);
    for (keel::usize i = 0; i < N; ++i) {
        if (M[i]) {
            idxOf[i] = keel::as<keel::i32>(pixOf.size());
            pixOf.push_back(i);
        }
    }
    const keel::usize nU = pixOf.size();

    // Output: start as copy of I (as DOUBLE so we can write the solve).
    keel::Array<keel::f64> Jout = Idata.clone(mr);

    if (nU == 0) {
        // No mask pixels → J = I unchanged.
        Value J = Value::matrix(H, W, ValueType::DOUBLE, mr);
        for (keel::usize i = 0; i < N; ++i) J.doubleDataMut()[i] = Jout[i];
        if (inT == ValueType::DOUBLE) return J;
        Value Jc = Value::matrix(H, W, inT, mr);
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 v = Jout[i];
            switch (inT) {
                case ValueType::SINGLE: Jc.singleDataMut()[i] = keel::as<keel::f32>(v); break;
                case ValueType::UINT8:  Jc.uint8DataMut()[i]  = keel::as<keel::u8>(std::clamp(v, 0.0, 255.0)); break;
                case ValueType::UINT16: Jc.uint16DataMut()[i] = keel::as<keel::u16>(std::clamp(v, 0.0, 65535.0)); break;
                case ValueType::INT8:   Jc.int8DataMut()[i]   = keel::as<keel::i8>(std::clamp(v, -128.0, 127.0)); break;
                case ValueType::INT16:  Jc.int16DataMut()[i]  = keel::as<keel::i16>(std::clamp(v, -32768.0, 32767.0)); break;
                case ValueType::INT32:  Jc.int32DataMut()[i]  = keel::as<keel::i32>(v); break;
                default: Jc.doubleDataMut()[i] = v; break;
            }
        }
        return Jc;
    }

    // Pre-compute per-unknown stencil data:
    //   numNbrs[i]    = # of on-grid neighbours (2/3/4)
    //   rhs[i]        = sum of perimeter-neighbour intensities
    //   nbrs[i]       = up to 4 unknown indices of N/S/W/E (or -1 if
    //                   off-grid or non-mask).
    keel::Array<keel::i32> numNbrs(nU, 0, mr);
    keel::Array<keel::f64>        rhs(nU, 0.0, mr);
    // Flat array of 4 neighbours per unknown.
    keel::Array<keel::i32>  nbrs(nU * 4, -1, mr);

    auto add_dir = [&](keel::usize i, keel::usize nr, keel::usize nc,
                       bool valid, keel::usize slot) {
        if (!valid) return;
        const keel::usize nk = lin(nr, nc);
        ++numNbrs[i];
        if (M[nk]) {
            nbrs[i * 4 + slot] = idxOf[nk];
        } else if (P[nk]) {
            rhs[i] += Idata[nk];
        }
        // else: in-grid, non-mask, non-perimeter — known but contributes 0.
    };

    for (keel::usize i = 0; i < nU; ++i) {
        const keel::usize k = pixOf[i];
        const keel::usize c = k / H;
        const keel::usize r = k % H;
        add_dir(i, r - 1, c, r > 0,      0); // N
        add_dir(i, r + 1, c, r + 1 < H,  1); // S
        add_dir(i, r, c - 1, c > 0,      2); // W
        add_dir(i, r, c + 1, c + 1 < W,  3); // E
    }

    // Apply A = stencil matrix (no materialisation):
    //   y[i] = numNbrs[i] * x[i] - sum_{j in nbrs[i]} x[j]
    auto matvec = [&](const keel::Array<keel::f64> &x,
                      keel::Array<keel::f64> &y) {
        for (keel::usize i = 0; i < nU; ++i) {
            keel::f64 s = keel::as<keel::f64>(numNbrs[i]) * x[i];
            const keel::i32 *nb = &nbrs[i * 4];
            if (nb[0] >= 0) s -= x[nb[0]];
            if (nb[1] >= 0) s -= x[nb[1]];
            if (nb[2] >= 0) s -= x[nb[2]];
            if (nb[3] >= 0) s -= x[nb[3]];
            y[i] = s;
        }
    };

    // Conjugate gradient solver, tol = 1e-12 (relative).
    keel::Array<keel::f64> x(nU, 0.0, mr);
    keel::Array<keel::f64> r(nU, 0.0, mr);
    keel::Array<keel::f64> p(nU, 0.0, mr);
    keel::Array<keel::f64> Ap(nU, 0.0, mr);

    for (keel::usize i = 0; i < nU; ++i) { r[i] = rhs[i]; p[i] = rhs[i]; }
    keel::f64 rdotr = 0.0;
    keel::f64 bnorm = 0.0;
    for (keel::usize i = 0; i < nU; ++i) {
        rdotr += r[i] * r[i];
        bnorm += rhs[i] * rhs[i];
    }
    const keel::f64 bn = keel::sqrt(bnorm);
    const keel::f64 tol = (bn > 0.0)
                         ? (1e-12 * bn) * (1e-12 * bn)
                         : 1e-24;

    const keel::usize maxiter = nU * 4 + 200;  // safety cap
    for (keel::usize it = 0; it < maxiter; ++it) {
        if (rdotr <= tol) break;
        matvec(p, Ap);
        keel::f64 pAp = 0.0;
        for (keel::usize i = 0; i < nU; ++i) pAp += p[i] * Ap[i];
        if (pAp <= 0.0) break;
        const keel::f64 alpha = rdotr / pAp;
        for (keel::usize i = 0; i < nU; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        keel::f64 rdotr_new = 0.0;
        for (keel::usize i = 0; i < nU; ++i) rdotr_new += r[i] * r[i];
        const keel::f64 beta = rdotr_new / rdotr;
        for (keel::usize i = 0; i < nU; ++i)
            p[i] = r[i] + beta * p[i];
        rdotr = rdotr_new;
    }

    // Write solution back to Jout.
    for (keel::usize i = 0; i < nU; ++i) Jout[pixOf[i]] = x[i];

    // Build typed output.
    Value J = Value::matrix(H, W, inT, mr);
    if (inT == ValueType::DOUBLE) {
        keel::f64 *p_ = J.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i) p_[i] = Jout[i];
    } else if (inT == ValueType::SINGLE) {
        keel::f32 *p_ = J.singleDataMut();
        for (keel::usize i = 0; i < N; ++i) p_[i] = keel::as<keel::f32>(Jout[i]);
    } else {
        // Integer classes: round + saturate.
        auto saturate = [&](keel::f64 v, keel::f64 lo, keel::f64 hi) {
            v = keel::round(v);
            if (v < lo) v = lo;
            if (v > hi) v = hi;
            return v;
        };
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 v = Jout[i];
            switch (inT) {
                case ValueType::UINT8:  J.uint8DataMut()[i]  = keel::as<keel::u8>(saturate(v, 0.0, 255.0)); break;
                case ValueType::UINT16: J.uint16DataMut()[i] = keel::as<keel::u16>(saturate(v, 0.0, 65535.0)); break;
                case ValueType::UINT32: J.uint32DataMut()[i] = keel::as<keel::u32>(saturate(v, 0.0, 4294967295.0)); break;
                case ValueType::INT8:   J.int8DataMut()[i]   = keel::as<keel::i8>(saturate(v, -128.0, 127.0)); break;
                case ValueType::INT16:  J.int16DataMut()[i]  = keel::as<keel::i16>(saturate(v, -32768.0, 32767.0)); break;
                case ValueType::INT32:  J.int32DataMut()[i]  = keel::as<keel::i32>(saturate(v, -2147483648.0, 2147483647.0)); break;
                default: J.doubleDataMut()[i] = v; break;
            }
        }
    }
    return J;
}

// ── poly2mask (polygon → binary mask scan conversion) ─────────────
//
// Empirically reverse-engineered from MATLAB R2025b's closed-source
// images.internal.builtins.poly2mask:
//
//   Pixel (r, c) ∈ BW iff its centre (c, r) is "inside" the polygon
//   under the following even-odd ray-casting variant:
//
//     For each non-horizontal edge (x1, y1) → (x2, y2):
//       if centre_y ∈ (min(y1,y2), max(y1,y2)]   ← half-open below,
//                                                   closed above
//          xi = x1 + (centre_y - y1) * (x2 - x1) / (y2 - y1)
//          if centre_x > xi                      ← strict greater
//             count ^= 1
//
//   The (ylo, yhi] interval ensures shared-vertex edges contribute
//   exactly once. The strict-greater x-test combined with the
//   half-open y rule reproduces MATLAB's bit-equal coverage of:
//     • integer-vertex rectangles aligned with the pixel grid,
//     • diagonal edges crossing pixel centres,
//     • degenerate (zero-area) sub-pixel polygons,
//     • polygons touching the image border.
//
// References: classic scanline polygon fill
//   Foley/van Dam et al., *Computer Graphics: P&P* §3.5;
//   X11 XFillPolygon half-open horizontal-edge convention.
Value poly2mask(const Value &X, const Value &Y,
                keel::usize M, keel::usize N,
                keel::Allocator *mr)
{
    const keel::usize nx = X.numel();
    const keel::usize ny = Y.numel();
    if (nx != ny)
        throw Error("poly2mask: X and Y must have the same length",
                    0, 0, "poly2mask", "",
                    "numkit:poly2mask:vectorSizeMismatch");

    Value BW = Value::matrix(M, N, ValueType::LOGICAL, mr);
    if (nx == 0 || M == 0 || N == 0) return BW;  // empty → all-false

    // Convert vertices to DOUBLE; auto-close if needed.
    keel::Array<keel::f64> vx(mr), vy(mr);
    vx.reserve(nx + 1);
    vy.reserve(nx + 1);
    for (keel::usize i = 0; i < nx; ++i) {
        vx.push_back(X.elemAsDouble(i));
        vy.push_back(Y.elemAsDouble(i));
    }
    if (vx.front() != vx.back() || vy.front() != vy.back()) {
        vx.push_back(vx.front());
        vy.push_back(vy.front());
    }
    const keel::usize nv = vx.size();

    // For each pixel, run the half-open even-odd ray-cast.
    keel::u8 *bw = BW.logicalDataMut();
    for (keel::usize c = 0; c < N; ++c) {
        const keel::f64 cx = keel::as<keel::f64>(c + 1);   // 1-based centre x
        for (keel::usize r = 0; r < M; ++r) {
            const keel::f64 cy = keel::as<keel::f64>(r + 1);
            keel::i32 crossings = 0;
            for (keel::usize i = 0; i + 1 < nv; ++i) {
                const keel::f64 y1 = vy[i], y2 = vy[i + 1];
                if (y1 == y2) continue;             // horizontal, skip
                const keel::f64 ylo = std::min(y1, y2);
                const keel::f64 yhi = std::max(y1, y2);
                if (cy <= ylo || cy > yhi) continue;
                const keel::f64 x1 = vx[i], x2 = vx[i + 1];
                const keel::f64 xi = x1 + (cy - y1) * (x2 - x1) / (y2 - y1);
                if (cx > xi) crossings ^= 1;
            }
            // Column-major linear index: c * M + r.
            bw[c * M + r] = keel::as<keel::u8>(crossings);
        }
    }
    return BW;
}

// ── roipoly (programmatic polygon ROI mask) ───────────────────────
//
// MATLAB R2025b roipoly.m algorithm (non-interactive forms):
//   1. Auto-close polygon: append (xi(1), yi(1)) if not already closed.
//   2. Map world → pixel:
//        roix = axes2pix(N, [xdata_lo xdata_hi], xi)
//        roiy = axes2pix(M, [ydata_lo ydata_hi], yi)
//   3. BW = poly2mask(roix, roiy, M, N).
//
// The interactive (figure-based) syntaxes are out of scope — they
// require a mouse and have no headless semantics.
Value roipoly(keel::f64 xdata_lo, keel::f64 xdata_hi,
              keel::f64 ydata_lo, keel::f64 ydata_hi,
              keel::usize M, keel::usize N,
              const Value &xi, const Value &yi,
              keel::Allocator *mr)
{
    if (xi.numel() != yi.numel())
        throw Error("roipoly: xi and yi must have the same length",
                    0, 0, "roipoly", "", "numkit:roipoly:xiyiMustBeSameLength");

    // Auto-close polygon.
    const keel::usize n0 = xi.numel();
    keel::Array<keel::f64> xc(mr), yc(mr);
    xc.reserve(n0 + 1);
    yc.reserve(n0 + 1);
    for (keel::usize i = 0; i < n0; ++i) {
        xc.push_back(xi.elemAsDouble(i));
        yc.push_back(yi.elemAsDouble(i));
    }
    if (n0 > 0 && (xc.front() != xc.back() || yc.front() != yc.back())) {
        xc.push_back(xc.front());
        yc.push_back(yc.front());
    }

    // Build extent vectors for axes2pix.
    Value xExt = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    xExt.doubleDataMut()[0] = xdata_lo;
    xExt.doubleDataMut()[1] = xdata_hi;
    Value yExt = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    yExt.doubleDataMut()[0] = ydata_lo;
    yExt.doubleDataMut()[1] = ydata_hi;

    // Wrap closed vertices as Value rows for axes2pix.
    const keel::usize nc = xc.size();
    Value xClosed = Value::matrix(nc, 1, ValueType::DOUBLE, mr);
    Value yClosed = Value::matrix(nc, 1, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < nc; ++i) {
        xClosed.doubleDataMut()[i] = xc[i];
        yClosed.doubleDataMut()[i] = yc[i];
    }
    Value roix = axes2pix(keel::as<keel::f64>(N), xExt, xClosed, mr);
    Value roiy = axes2pix(keel::as<keel::f64>(M), yExt, yClosed, mr);
    return poly2mask(roix, roiy, M, N, mr);
}

// ── graydist (gray-weighted geodesic distance transform) ────────
//
// MATLAB R2025b graydist algorithm:
//   T(seed) = 0
//   Edge cost p→q: χ(p,q) · (I(p) + I(q)) / 2
//     χ depends on method:
//       chessboard:        8-conn, χ = 1 for all neighbours
//       cityblock:         4-conn, χ = 1 (orth only)
//       quasi-euclidean:   8-conn, χ = 1 (orth), χ = √2 (diag)
//   T(p) = min over neighbours q: T(q) + cost(q→p)
//   Solved by Dijkstra with binary heap.
//
// Reference: Soille, *Morphological Image Analysis*, 2nd ed.,
//   Springer, §4.4 (chamfer geodesic distance transform).
Value graydist(const Value &I, const Value &seeds,
               keel::StringRef method,
               keel::Allocator *mr)
{
    internal::requireImageClass(I, "graydist", "A", 1,
                                internal::ImgClasses::AnyNumericReal);
    if (!seeds.isEmpty())
        internal::requireRealDoubleArg(seeds, "graydist", "IND", 2);
    if (I.dims().is3D())
        throw Error("graydist: I must be 2-D",
                    0, 0, "graydist", "", "numkit:graydist:dim");
    const bool is_cb = (method == "cityblock");
    const bool is_chess = (method == "chessboard");
    const bool is_qe = (method == "quasi-euclidean");
    if (!is_cb && !is_chess && !is_qe)
        throw Error("graydist: METHOD must be 'cityblock', "
                    "'chessboard', or 'quasi-euclidean'",
                    0, 0, "graydist", "", "numkit:graydist:method");

    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    const keel::usize N = H * W;

    // Output class: DOUBLE for DOUBLE input, SINGLE otherwise.
    const ValueType outT = (I.type() == ValueType::DOUBLE)
                            ? ValueType::DOUBLE : ValueType::SINGLE;
    Value T = Value::matrix(H, W, outT, mr);

    // Initialise with +Inf.
    if (outT == ValueType::DOUBLE) {
        keel::f64 *p = T.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i)
            p[i] = keel::Limits<keel::f64>::infinity();
    } else {
        keel::f32 *p = T.singleDataMut();
        for (keel::usize i = 0; i < N; ++i)
            p[i] = keel::Limits<keel::f32>::infinity();
    }
    if (N == 0 || seeds.numel() == 0) return T;

    // Working DOUBLE distance buffer (precision; cast back at end).
    keel::Array<keel::f64> dist(N, keel::Limits<keel::f64>::infinity(), mr);
    // Read image into a double array (uniform access).
    keel::Array<keel::f64> Idata(N, 0.0, mr);
    for (keel::usize i = 0; i < N; ++i) Idata[i] = I.elemAsDouble(i);

    // Min-heap as (dist, index) pairs.
    using PQNode = std::pair<keel::f64, keel::usize>;
    auto cmp = [](const PQNode &a, const PQNode &b) { return a.first > b.first; };
    ArrayPriorityQueue<PQNode, decltype(cmp)> pq(cmp, mr);

    // Seed pixels: T = 0, push.
    for (keel::usize k = 0; k < seeds.numel(); ++k) {
        const keel::f64 sv = seeds.elemAsDouble(k);
        if (!(sv >= 1) || sv != keel::floor(sv))
            throw Error("graydist: seed indices must be positive integers",
                        0, 0, "graydist", "", "numkit:graydist:seed");
        const keel::usize idx = keel::as<keel::usize>(sv) - 1;
        if (idx >= N)
            throw Error("graydist: seed index out of bounds",
                        0, 0, "graydist", "", "numkit:graydist:seedOOB");
        dist[idx] = 0.0;
        pq.emplace(0.0, idx);
    }

    // Neighbour offsets in (dr, dc) and chamfer weight.
    struct Step { keel::i32 dr, dc; keel::f64 chamfer; };
    keel::StaticArray<Step, 8> steps8{};
    keel::i32 nSteps = 0;
    if (is_cb) {
        steps8 = {Step{-1, 0, 1.0}, Step{1, 0, 1.0},
                  Step{0, -1, 1.0}, Step{0, 1, 1.0},
                  Step{}, Step{}, Step{}, Step{}};
        nSteps = 4;
    } else if (is_chess) {
        steps8 = {Step{-1, 0, 1.0}, Step{1, 0, 1.0},
                  Step{0, -1, 1.0}, Step{0, 1, 1.0},
                  Step{-1, -1, 1.0}, Step{-1, 1, 1.0},
                  Step{1, -1, 1.0},  Step{1, 1, 1.0}};
        nSteps = 8;
    } else {
        const keel::f64 s2 = keel::sqrt(2.0);
        steps8 = {Step{-1, 0, 1.0}, Step{1, 0, 1.0},
                  Step{0, -1, 1.0}, Step{0, 1, 1.0},
                  Step{-1, -1, s2}, Step{-1, 1, s2},
                  Step{1, -1, s2},  Step{1, 1, s2}};
        nSteps = 8;
    }

    // Dijkstra main loop.
    while (!pq.empty()) {
        auto [d, k] = pq.top();
        pq.pop();
        if (d > dist[k]) continue;  // stale heap entry
        const keel::usize r = k % H;
        const keel::usize c = k / H;
        const keel::f64 Iq = Idata[k];
        for (keel::i32 s = 0; s < nSteps; ++s) {
            const keel::i64 nr = keel::as<keel::i64>(r) + steps8[s].dr;
            const keel::i64 nc = keel::as<keel::i64>(c) + steps8[s].dc;
            if (nr < 0 || nc < 0
             || keel::as<keel::usize>(nr) >= H
             || keel::as<keel::usize>(nc) >= W) continue;
            const keel::usize nk = keel::as<keel::usize>(nc) * H
                                 + keel::as<keel::usize>(nr);
            const keel::f64 Ip = Idata[nk];
            const keel::f64 w = steps8[s].chamfer * (Iq + Ip) * 0.5;
            const keel::f64 nd = d + w;
            if (nd < dist[nk]) {
                dist[nk] = nd;
                pq.emplace(nd, nk);
            }
        }
    }

    // Write back to T with class cast.
    if (outT == ValueType::DOUBLE) {
        keel::f64 *p = T.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i) p[i] = dist[i];
    } else {
        keel::f32 *p = T.singleDataMut();
        for (keel::usize i = 0; i < N; ++i) p[i] = keel::as<keel::f32>(dist[i]);
    }
    return T;
}

// ── bwdistgeodesic (binary geodesic distance transform) ─────────
//
// Like graydist but the underlying image is a binary mask BW —
// only true pixels are traversable and the edge cost reduces to
// the raw chamfer χ(p, q). Output:
//   * 0     at seed pixels,
//   * finite at reachable traversable pixels,
//   * NaN   at false (non-traversable) pixels,
//   * Inf   at traversable but unreachable pixels.
// Output class is always SINGLE.
//
// Reference: P. Soille, *Morphological Image Analysis* §4.4.
Value bwdistgeodesic(const Value &BW, const Value &seeds,
                     keel::StringRef method,
                     keel::Allocator *mr)
{
    internal::requireImageClass(BW, "bwdistgeodesic", "BW", 1,
                                internal::ImgClasses::LogicalOnly);
    if (BW.dims().is3D())
        throw Error("bwdistgeodesic: BW must be 2-D",
                    0, 0, "bwdistgeodesic", "",
                    "numkit:bwdistgeodesic:dim");
    const bool is_cb = (method == "cityblock");
    const bool is_chess = (method == "chessboard");
    const bool is_qe = (method == "quasi-euclidean");
    if (!is_cb && !is_chess && !is_qe)
        throw Error("bwdistgeodesic: METHOD must be 'cityblock', "
                    "'chessboard', or 'quasi-euclidean'",
                    0, 0, "bwdistgeodesic", "",
                    "numkit:bwdistgeodesic:method");

    const keel::usize H = BW.dims().rows();
    const keel::usize W = BW.dims().cols();
    const keel::usize N = H * W;

    Value D = Value::matrix(H, W, ValueType::SINGLE, mr);
    if (N == 0) return D;

    // Read BW into uint8 array for fast access.
    keel::Array<keel::u8> mask(N, 0, mr);
    const bool isLog = BW.isLogical();
    for (keel::usize i = 0; i < N; ++i) {
        const bool on = isLog ? (BW.logicalData()[i] != 0)
                              : (BW.elemAsDouble(i) != 0.0);
        mask[i] = on ? 1 : 0;
    }

    // Initialise: NaN for false, +Inf for true.
    keel::f32 *dp = D.singleDataMut();
    const keel::f32 nan = keel::Limits<keel::f32>::quiet_nan();
    const keel::f32 inf = keel::Limits<keel::f32>::infinity();
    for (keel::usize i = 0; i < N; ++i)
        dp[i] = mask[i] ? inf : nan;

    if (seeds.numel() == 0) return D;

    // Working DOUBLE buffer for Dijkstra precision.
    keel::Array<keel::f64> dist(N, keel::Limits<keel::f64>::infinity(), mr);

    using PQNode = std::pair<keel::f64, keel::usize>;
    auto cmp = [](const PQNode &a, const PQNode &b) { return a.first > b.first; };
    ArrayPriorityQueue<PQNode, decltype(cmp)> pq(cmp, mr);

    // Seeds: validate, then set dist=0.
    for (keel::usize k = 0; k < seeds.numel(); ++k) {
        const keel::f64 sv = seeds.elemAsDouble(k);
        if (!(sv >= 1) || sv != keel::floor(sv))
            throw Error("bwdistgeodesic: seed indices must be positive "
                        "integers",
                        0, 0, "bwdistgeodesic", "",
                        "numkit:bwdistgeodesic:seed");
        const keel::usize idx = keel::as<keel::usize>(sv) - 1;
        if (idx >= N)
            throw Error("bwdistgeodesic: seed index out of bounds",
                        0, 0, "bwdistgeodesic", "",
                        "numkit:bwdistgeodesic:seedOOB");
        if (!mask[idx]) {
            // Probed R2025b: a seed off the mask is NOT an error — the
            // seed pixel comes back NaN and everything else stays Inf.
            dist[idx] = keel::Limits<keel::f64>::quiet_nan();
            continue;
        }
        dist[idx] = 0.0;
        pq.emplace(0.0, idx);
    }

    struct Step { keel::i32 dr, dc; keel::f64 chamfer; };
    keel::StaticArray<Step, 8> steps8{};
    keel::i32 nSteps = 0;
    if (is_cb) {
        steps8 = {Step{-1, 0, 1.0}, Step{1, 0, 1.0},
                  Step{0, -1, 1.0}, Step{0, 1, 1.0},
                  Step{}, Step{}, Step{}, Step{}};
        nSteps = 4;
    } else if (is_chess) {
        steps8 = {Step{-1, 0, 1.0}, Step{1, 0, 1.0},
                  Step{0, -1, 1.0}, Step{0, 1, 1.0},
                  Step{-1, -1, 1.0}, Step{-1, 1, 1.0},
                  Step{1, -1, 1.0},  Step{1, 1, 1.0}};
        nSteps = 8;
    } else {
        const keel::f64 s2 = keel::sqrt(2.0);
        steps8 = {Step{-1, 0, 1.0}, Step{1, 0, 1.0},
                  Step{0, -1, 1.0}, Step{0, 1, 1.0},
                  Step{-1, -1, s2}, Step{-1, 1, s2},
                  Step{1, -1, s2},  Step{1, 1, s2}};
        nSteps = 8;
    }

    while (!pq.empty()) {
        auto [d, k] = pq.top();
        pq.pop();
        if (d > dist[k]) continue;
        const keel::usize r = k % H;
        const keel::usize c = k / H;
        for (keel::i32 s = 0; s < nSteps; ++s) {
            const keel::i64 nr = keel::as<keel::i64>(r) + steps8[s].dr;
            const keel::i64 nc = keel::as<keel::i64>(c) + steps8[s].dc;
            if (nr < 0 || nc < 0
             || keel::as<keel::usize>(nr) >= H
             || keel::as<keel::usize>(nc) >= W) continue;
            const keel::usize nk = keel::as<keel::usize>(nc) * H
                                 + keel::as<keel::usize>(nr);
            if (!mask[nk]) continue;  // can't traverse barrier
            const keel::f64 nd = d + steps8[s].chamfer;
            if (nd < dist[nk]) {
                dist[nk] = nd;
                pq.emplace(nd, nk);
            }
        }
    }

    // Write reachable distances back; leave NaN at false pixels and
    // Inf at unreachable true pixels.
    for (keel::usize i = 0; i < N; ++i) {
        if (!mask[i]) continue;          // NaN preserved
        if (!keel::is_inf(dist[i]))
            dp[i] = keel::as<keel::f32>(dist[i]);
        // else: already Inf from initialisation
    }
    return D;
}

Value imoverlay(const Value &I, const Value &BW, const Value &color, keel::Allocator *mr)
{
    internal::requireImageClass(I, "imoverlay", "A", 1,
                                internal::ImgClasses::FloatIntLog);
    internal::requireImageClass(BW, "imoverlay", "BW", 2,
                                internal::ImgClasses::AnyNumericReal);
    // Probed R2025b: a complex colour triple is rejected.
    internal::requireImageClass(color, "imoverlay", "color", 3,
                                internal::ImgClasses::AnyNumericReal);
    if (color.numel() != 3)
        throw Error("imoverlay: color must be a 1×3 RGB triple",
                    0, 0, "imoverlay", "", "numkit:imoverlay:color");
    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    if (BW.dims().rows() != H || BW.dims().cols() != W)
        throw Error("imoverlay: BW must match the H × W of I",
                    0, 0, "imoverlay", "", "numkit:imoverlay:shape");

    // Detect input layout: H × W (grayscale) or H × W × 3 (RGB).
    bool isRGB;
    if (I.numel() == H * W) isRGB = false;
    else if (I.numel() == H * W * 3) isRGB = true;
    else
        throw Error("imoverlay: I must be H × W or H × W × 3",
                    0, 0, "imoverlay", "", "numkit:imoverlay:shape");

    // Read color. Auto-detect float (0..1) vs byte (0..255) by max
    // value: if all three channels ≤ 1.0 we assume 0..1, else 0..255.
    keel::f64 cIn[3] = { color.elemAsDouble(0),
                       color.elemAsDouble(1),
                       color.elemAsDouble(2) };
    const bool floatColour = (cIn[0] <= 1.0 && cIn[1] <= 1.0 &&
                               cIn[2] <= 1.0);
    keel::i32 cByte[3];
    for (keel::i32 k = 0; k < 3; ++k) {
        keel::f64 v = floatColour ? cIn[k] * 255.0 : cIn[k];
        if (v < 0.0) v = 0.0;
        if (v > 255.0) v = 255.0;
        cByte[k] = keel::as<keel::i32>(keel::round_i64(v));
    }

    // Helper: read I element as 0..255 byte.
    const ValueType srcT = I.type();
    auto pixelByte = [&](keel::usize y, keel::usize x, keel::i32 chan) {
        keel::usize idx;
        if (isRGB)
            idx = keel::as<keel::usize>(chan) * H * W + x * H + y;
        else
            idx = x * H + y;
        const keel::f64 v = I.elemAsDouble(idx);
        keel::f64 w = v;
        if (srcT != ValueType::UINT8 && srcT != ValueType::UINT16 &&
            srcT != ValueType::INT8  && srcT != ValueType::INT16) {
            // Floating point: assume [0, 1].
            w = v * 255.0;
        } else if (srcT == ValueType::UINT16) {
            w = v / 257.0;
        } else if (srcT == ValueType::INT16) {
            w = (v + 32768.0) / 257.0;
        }
        if (w < 0.0) w = 0.0;
        if (w > 255.0) w = 255.0;
        return keel::as<keel::u8>(keel::round_i64(w));
    };

    Value out = Value::matrix3d(H, W, 3, ValueType::UINT8, mr);
    keel::u8 *od = out.uint8DataMut();
    const keel::usize plane = H * W;
    for (keel::usize y = 0; y < H; ++y)
        for (keel::usize x = 0; x < W; ++x) {
            const bool flag = (BW.elemAsDouble(x * H + y) != 0.0);
            for (keel::i32 c = 0; c < 3; ++c) {
                const keel::usize outIdx = keel::as<keel::usize>(c) * plane +
                                       x * H + y;
                if (flag)
                    od[outIdx] = keel::as<keel::u8>(cByte[c]);
                else
                    od[outIdx] = pixelByte(y, x, isRGB ? c : 0);
            }
        }
    return out;
}

// ── reducepoly (Ramer-Douglas-Peucker polyline simplification) ────────
//
// MATLAB R2025b reducepoly.m: recursively splits the polyline at the
// vertex farthest from the chord between the current endpoints; a run is
// replaced by its endpoints once no interior vertex deviates by more than
// the (bbox-diagonal-normalised) tolerance.
//
// Reference: D. Douglas & T. Peucker, "Algorithms for the reduction of
// the number of points required to represent a digitized line or its
// caricature", Cartographica 10(2):112-122, 1973; Ramer 1972.
namespace {

// Recursively returns the kept vertex indices for the sub-polyline
// [lo, hi]. Mirrors douglasPeucker in reducepoly.m, including the
// first-farthest tie-break and the degenerate near-zero-chord branch.
template <typename T>
keel::Array<keel::i32> rdpKeep(const T *xs, const T *ys, keel::i32 lo, keel::i32 hi, T tolEff)
{
    const keel::i32 n = hi - lo + 1;
    if (n <= 2) {
        if (lo == hi) return {lo};
        return {lo, hi};
    }
    const T x1 = xs[lo], y1 = ys[lo], x2 = xs[hi], y2 = ys[hi];
    const T dx = x2 - x1, dy = y2 - y1;
    const T dNode = keel::sqrt(dx * dx + dy * dy);
    const keel::f64 EPS = 2.220446049250313e-16;  // MATLAB eps(1)

    T maxd = -1;
    keel::i32 far = -1;
    for (keel::i32 g = lo + 1; g <= hi - 1; ++g) {
        const T xk = xs[g], yk = ys[g];
        T d;
        if (keel::as<keel::f64>(dNode) > EPS) {
            // |det([1 x1 y1; 1 x2 y2; 1 xk yk])| / dNode (chord-perp dist).
            const T det = (x2 * yk - y2 * xk) - x1 * (yk - y2) + y1 * (xk - x2);
            d = keel::abs(det) / dNode;
        } else {
            const T ex = xk - x1, ey = yk - y1;
            d = keel::sqrt(ex * ex + ey * ey);
        }
        if (d > maxd) { maxd = d; far = g; }  // strict > keeps FIRST farthest
    }
    if (maxd > tolEff) {
        keel::Array<keel::i32> r1 = rdpKeep(xs, ys, lo, far, tolEff);
        keel::Array<keel::i32> r2 = rdpKeep(xs, ys, far, hi, tolEff);
        r1.append(r2.slice(1));  // drop duplicate `far`
        return r1;
    }
    return {lo, hi};
}

template <typename T>
keel::Array<keel::i32> rdpRun(const Value &P, keel::i32 n, keel::f64 tol)
{
    keel::Array<T> xs(n), ys(keel::as<keel::usize>(n));
    for (keel::i32 i = 0; i < n; ++i) {
        xs[keel::as<keel::usize>(i)] = static_cast<T>(P.elemAsDouble(keel::as<keel::usize>(i)));
        ys[keel::as<keel::usize>(i)] = static_cast<T>(P.elemAsDouble(keel::as<keel::usize>(i) + keel::as<keel::usize>(n)));
    }
    // scaling = norm(max(P) - min(P)) over each column (bbox diagonal).
    T minx = xs[0], maxx = xs[0], miny = ys[0], maxy = ys[0];
    for (keel::i32 i = 1; i < n; ++i) {
        minx = std::min(minx, xs[keel::usize(i)]); maxx = std::max(maxx, xs[keel::usize(i)]);
        miny = std::min(miny, ys[keel::usize(i)]); maxy = std::max(maxy, ys[keel::usize(i)]);
    }
    const T sdx = maxx - minx, sdy = maxy - miny;
    const T scaling = keel::sqrt(sdx * sdx + sdy * sdy);
    const keel::f64 EPS = 2.220446049250313e-16;
    const T tolEff = static_cast<T>(tol == 0.0 ? EPS : tol) * scaling;
    return rdpKeep<T>(xs.data(), ys.data(), 0, n - 1, tolEff);
}

} // namespace

Value reducepoly(const Value &P, keel::f64 tolerance, keel::Allocator *mr)
{
    const auto &d = P.dims();
    if (d.ndim() > 2 || (d.cols() != 2 && P.numel() != 0))
        throw Error("reducepoly: P must be an n-by-2 matrix",
                    0, 0, "reducepoly", "", "numkit:reducepoly:shape");
    if (tolerance < 0.0 || tolerance > 1.0)
        throw Error("reducepoly: tolerance must be in [0, 1]",
                    0, 0, "reducepoly", "", "numkit:reducepoly:tol");

    const keel::i32 n = keel::as<keel::i32>(d.rows());
    const ValueType cls = P.type();

    auto storeAt = [&](Value &out, keel::usize oidx, keel::f64 v) {
        switch (cls) {
            case ValueType::SINGLE: out.singleDataMut()[oidx] = keel::as<keel::f32>(v); break;
            case ValueType::UINT8:  out.uint8DataMut()[oidx]  = keel::as<keel::u8>(keel::round_i64(v)); break;
            case ValueType::UINT16: out.uint16DataMut()[oidx] = keel::as<keel::u16>(keel::round_i64(v)); break;
            case ValueType::UINT32: out.uint32DataMut()[oidx] = keel::as<keel::u32>(keel::round_i64(v)); break;
            case ValueType::INT8:   out.int8DataMut()[oidx]   = keel::as<keel::i8>(keel::round_i64(v)); break;
            case ValueType::INT16:  out.int16DataMut()[oidx]  = keel::as<keel::i16>(keel::round_i64(v)); break;
            case ValueType::INT32:  out.int32DataMut()[oidx]  = keel::as<keel::i32>(keel::round_i64(v)); break;
            case ValueType::INT64:  out.int64DataMut()[oidx]  = keel::as<keel::i64>(keel::round_i64(v)); break;
            default:                out.doubleDataMut()[oidx] = v; break;
        }
    };

    // n <= 1: return the input unchanged (per MATLAB).
    if (n <= 1) {
        Value out = Value::matrix(keel::as<keel::usize>(n), n == 0 ? 0 : 2, cls, mr);
        for (keel::i32 i = 0; i < n; ++i) {
            storeAt(out, keel::as<keel::usize>(i), P.elemAsDouble(keel::as<keel::usize>(i)));
            storeAt(out, keel::as<keel::usize>(i) + keel::as<keel::usize>(n),
                    P.elemAsDouble(keel::as<keel::usize>(i) + keel::as<keel::usize>(n)));
        }
        return out;
    }

    // Compute in double for DOUBLE input, single otherwise (matches MATLAB,
    // which casts non-float to single and keeps single/double as-is).
    const keel::Array<keel::i32> keep = (cls == ValueType::DOUBLE)
        ? rdpRun<keel::f64>(P, n, tolerance)
        : rdpRun<keel::f32>(P, n, tolerance);

    const keel::usize m = keep.size();
    Value out = Value::matrix(m, 2, cls, mr);
    for (keel::usize j = 0; j < m; ++j) {
        const keel::usize k = keel::as<keel::usize>(keep[j]);
        storeAt(out, j,           P.elemAsDouble(k));                       // x
        storeAt(out, j + m,       P.elemAsDouble(k + keel::as<keel::usize>(n)));  // y
    }
    return out;
}

namespace {

void watershed_argcheck(const Value &A, keel::i32 conn)
{
    internal::requireImageClass(A, "watershed", "A", 1,
                                internal::ImgClasses::AnyNumericReal);
    if (A.dims().is3D()) {
        throw Error("watershed: input 1 (A) must be a 2-D matrix.",
                    0, 0, "watershed", "", "numkit:watershed:expected2D");
    }
    const keel::usize n = A.numel();
    for (keel::usize i = 0; i < n; ++i) {
        if (keel::is_nan(A.elemAsDouble(i))) {
            throw Error("watershed: input 1 (A) must be non-NaN (got NaN).",
                        0, 0, "watershed", "", "numkit:watershed:expectedNonNaN");
        }
    }
    if (conn != 4 && conn != 8) {
        throw Error(keel::format("watershed: input 2 (conn) must be 4 or 8 (got {}).", conn),
                    0, 0, "watershed", "", "numkit:watershed:badConn");
    }
}

} // namespace

Value watershed(const Value &A, keel::i32 conn, keel::Allocator *mr)
{
    watershed_argcheck(A, conn);

    const keel::usize H = A.dims().rows();
    const keel::usize W = A.dims().cols();
    const keel::usize N = H * W;
    if (N == 0) {
        return Value::matrix(H, W, ValueType::UINT8, mr);
    }

    keel::Array<keel::f64> aval(N);
    for (keel::usize i = 0; i < N; ++i) {
        aval[i] = A.elemAsDouble(i);
    }

    // Column-major neighbor offsets:
    static constexpr keel::i32 dr8[8] = {-1,  0,  1, -1, 1, -1, 0, 1};
    static constexpr keel::i32 dc8[8] = {-1, -1, -1,  0, 0,  1, 1, 1};
    static constexpr keel::i32 dr4[4] = {-1,  1,  0,  0};
    static constexpr keel::i32 dc4[4] = { 0,  0, -1,  1};
    const keel::i32 *dr = (conn == 4) ? dr4 : dr8;
    const keel::i32 *dc = (conn == 4) ? dc4 : dc8;
    const keel::i32 K   = (conn == 4) ? 4   : 8;
    const keel::i32 Hi  = keel::as<keel::i32>(H);
    const keel::i32 Wi  = keel::as<keel::i32>(W);

    // Step 1: Identify regional minima plateaus under conn-connectivity.
    keel::Array<keel::u8> visited(N, 0);
    keel::Array<keel::u8> regmin(N, 0);
    keel::Array<keel::usize> comp;
    keel::Array<keel::usize> q;
    comp.reserve(64);
    q.reserve(64);

    for (keel::usize i = 0; i < N; ++i) {
        if (visited[i]) continue;
        visited[i] = 1;
        comp.clear();
        q.clear();
        comp.push_back(i);
        q.push_back(i);
        const keel::f64 v0 = aval[i];
        bool hasLower = false;
        keel::usize head = 0;
        while (head < q.size()) {
            const keel::usize curr = q[head++];
            const keel::i32 r = keel::as<keel::i32>(curr % H);
            const keel::i32 c = keel::as<keel::i32>(curr / H);
            for (keel::i32 k = 0; k < K; ++k) {
                const keel::i32 nr = r + dr[k];
                const keel::i32 nc = c + dc[k];
                if (nr < 0 || nr >= Hi || nc < 0 || nc >= Wi) continue;
                const keel::usize nb = keel::as<keel::usize>(nc) * H
                                     + keel::as<keel::usize>(nr);
                if (aval[nb] < v0) {
                    hasLower = true;
                } else if (aval[nb] == v0 && !visited[nb]) {
                    visited[nb] = 1;
                    comp.push_back(nb);
                    q.push_back(nb);
                }
            }
        }
        if (!hasLower) {
            for (keel::usize idx : comp) {
                regmin[idx] = 1;
            }
        }
    }

    // Step 2: Label connected components of regmin in column-major order.
    keel::Array<keel::u32> L(N, 0);
    keel::u32 numBasins = 0;
    for (keel::usize i = 0; i < N; ++i) {
        if (regmin[i] && L[i] == 0) {
            ++numBasins;
            L[i] = numBasins;
            q.clear();
            q.push_back(i);
            keel::usize head = 0;
            while (head < q.size()) {
                const keel::usize curr = q[head++];
                const keel::i32 r = keel::as<keel::i32>(curr % H);
                const keel::i32 c = keel::as<keel::i32>(curr / H);
                for (keel::i32 k = 0; k < K; ++k) {
                    const keel::i32 nr = r + dr[k];
                    const keel::i32 nc = c + dc[k];
                    if (nr < 0 || nr >= Hi || nc < 0 || nc >= Wi) continue;
                    const keel::usize nb = keel::as<keel::usize>(nc) * H
                                         + keel::as<keel::usize>(nr);
                    if (regmin[nb] && L[nb] == 0) {
                        L[nb] = numBasins;
                        q.push_back(nb);
                    }
                }
            }
        }
    }

    // Step 3: Seed Meyer's priority queue from neighbors of labeled minima.
    struct PQEntry {
        keel::f64 val;
        keel::usize ord;
        keel::usize idx;
        bool operator>(const PQEntry &other) const {
            if (val != other.val) return val > other.val;
            return ord > other.ord;
        }
    };
    auto pqCmp = [](const PQEntry &a, const PQEntry &b) { return a > b; };
    ArrayPriorityQueue<PQEntry, decltype(pqCmp)> pq(pqCmp, mr);
    keel::Array<keel::u8> inPQ(N, 0);
    for (keel::usize i = 0; i < N; ++i) {
        if (L[i] > 0) inPQ[i] = 1;
    }
    keel::usize ord = 0;
    for (keel::usize i = 0; i < N; ++i) {
        if (L[i] > 0) {
            const keel::i32 r = keel::as<keel::i32>(i % H);
            const keel::i32 c = keel::as<keel::i32>(i / H);
            for (keel::i32 k = 0; k < K; ++k) {
                const keel::i32 nr = r + dr[k];
                const keel::i32 nc = c + dc[k];
                if (nr < 0 || nr >= Hi || nc < 0 || nc >= Wi) continue;
                const keel::usize nb = keel::as<keel::usize>(nc) * H
                                     + keel::as<keel::usize>(nr);
                if (!inPQ[nb]) {
                    inPQ[nb] = 1;
                    pq.push(PQEntry{aval[nb], ++ord, nb});
                }
            }
        }
    }

    // Step 4: Flood from priority queue; pixels touching >= 2 distinct
    // basin labels remain 0 (watershed ridge lines).
    while (!pq.empty()) {
        const PQEntry top = pq.top();
        pq.pop();
        const keel::usize curr = top.idx;
        const keel::i32 r = keel::as<keel::i32>(curr % H);
        const keel::i32 c = keel::as<keel::i32>(curr / H);
        keel::u32 singleLabel = 0;
        bool multipleLabels = false;
        for (keel::i32 k = 0; k < K; ++k) {
            const keel::i32 nr = r + dr[k];
            const keel::i32 nc = c + dc[k];
            if (nr < 0 || nr >= Hi || nc < 0 || nc >= Wi) continue;
            const keel::u32 nl = L[keel::as<keel::usize>(nc) * H
                                     + keel::as<keel::usize>(nr)];
            if (nl > 0) {
                if (singleLabel == 0) {
                    singleLabel = nl;
                } else if (singleLabel != nl) {
                    multipleLabels = true;
                    break;
                }
            }
        }
        if (singleLabel > 0 && !multipleLabels) {
            L[curr] = singleLabel;
            for (keel::i32 k = 0; k < K; ++k) {
                const keel::i32 nr = r + dr[k];
                const keel::i32 nc = c + dc[k];
                if (nr < 0 || nr >= Hi || nc < 0 || nc >= Wi) continue;
                const keel::usize nb = keel::as<keel::usize>(nc) * H
                                     + keel::as<keel::usize>(nr);
                if (!inPQ[nb]) {
                    inPQ[nb] = 1;
                    pq.push(PQEntry{aval[nb], ++ord, nb});
                }
            }
        }
    }

    // Step 5: Materialize label matrix in smallest unsigned integer class.
    if (numBasins <= 255u) {
        Value out = Value::matrix(H, W, ValueType::UINT8, mr);
        keel::u8 *od = out.uint8DataMut();
        for (keel::usize i = 0; i < N; ++i) {
            od[i] = keel::as<keel::u8>(L[i]);
        }
        return out;
    } else if (numBasins <= 65535u) {
        Value out = Value::matrix(H, W, ValueType::UINT16, mr);
        keel::u16 *od = out.uint16DataMut();
        for (keel::usize i = 0; i < N; ++i) {
            od[i] = keel::as<keel::u16>(L[i]);
        }
        return out;
    } else {
        Value out = Value::matrix(H, W, ValueType::UINT32, mr);
        keel::u32 *od = out.uint32DataMut();
        for (keel::usize i = 0; i < N; ++i) {
            od[i] = L[i];
        }
        return out;
    }
}

} // namespace numkit::image
