// toolboxes/image/src/morph/morph.cpp

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/math.h>
#include <numkit/image/morph/morph.hpp>
#include <numkit/image/internal/type_gates.hpp>

namespace numkit::image::internal {
// Value-preserving class output for the morphology wrappers: multiply
// the intensity image (gathered to double) by a logical keep-mask and
// cast back to the input class (probed R2025b: imclearborder /
// imkeepborder / imfill preserve pixel VALUES, not just the mask).
inline Value maskImageByLogical(const Value &I, const Value &mask,
                                keel::Allocator *mr)
{
    Value Id = coerceToDoubleImage(I, mr);
    const keel::usize n = Id.numel();
    double *d = Id.doubleDataMut();
    const keel::u8 *m = mask.logicalData();
    for (keel::usize i = 0; i < n; ++i)
        if (!m[i]) d[i] = 0.0;
    return castResultToClass(Id, I.type(), mr);
}
} // namespace numkit::image::internal

#include <numkit/signal/convolution/convolution.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/fn_handle.hpp>

#include <keel/format.h>
#include <keel/hash_set.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

#include "bwmorph_luts.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numkit::image {

namespace {

Value pack_logical(keel::ArrayRef<keel::u8> mask, int rows, int cols, keel::Allocator *mr)
{
    Value out = Value::matrix(rows, cols, ValueType::LOGICAL, mr);
    if (mask.empty()) return out;
    keel::u8 *od = out.logicalDataMut();
    // Input is row-major mask; storage is column-major.
    for (int c = 0; c < cols; ++c)
        for (int r = 0; r < rows; ++r)
            od[(keel::usize)c * (keel::usize)rows + (keel::usize)r]
                = mask[(keel::usize)r * (keel::usize)cols + (keel::usize)c] ? 1 : 0;
    return out;
}

Value strel_square(int N, keel::Allocator *mr) {
    if (N < 1) N = 1;
    keel::Array<keel::u8> m((keel::usize)N * (keel::usize)N, 1);
    return pack_logical(m, N, N, mr);
}

Value strel_rect(int rows, int cols, keel::Allocator *mr) {
    if (rows < 1) rows = 1;
    if (cols < 1) cols = 1;
    keel::Array<keel::u8> m((keel::usize)rows * (keel::usize)cols, 1);
    return pack_logical(m, rows, cols, mr);
}

Value strel_diamond(int r, keel::Allocator *mr) {
    if (r < 1) r = 1;
    const int N = 2 * r + 1;
    keel::Array<keel::u8> m((keel::usize)N * (keel::usize)N, 0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            if (keel::abs(i - r) + keel::abs(j - r) <= r)
                m[(keel::usize)i * (keel::usize)N + (keel::usize)j] = 1;
    return pack_logical(m, N, N, mr);
}

// Build a true Euclidean disk neighbourhood: {(dy,dx) : dy²+dx² ≤ r²}.
Value strel_disk_euclidean(double r, keel::Allocator *mr) {
    const int R = (int)std::ceil(r);
    const int N = 2 * R + 1;
    keel::Array<keel::u8> m((keel::usize)N * (keel::usize)N, 0);
    const double r2 = r * r;
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            const double dy = i - R, dx = j - R;
            if (dy * dy + dx * dx <= r2)
                m[(keel::usize)i * (keel::usize)N + (keel::usize)j] = 1;
        }
    return pack_logical(m, N, N, mr);
}

struct GridPt {
    int r = 0, c = 0;
    bool operator==(const GridPt &o) const noexcept { return r == o.r && c == o.c; }
};
struct GridPtHash {
    keel::u64 operator()(const GridPt &p) const noexcept {
        const keel::u64 k = (static_cast<keel::u64>(static_cast<keel::u32>(p.r)) << 32)
                          | static_cast<keel::u64>(static_cast<keel::u32>(p.c));
        return keel::Hash<keel::u64>{}(k);
    }
};
using GridPtSet = keel::HashSet<GridPt, GridPtHash>;

// Minkowski sum (morphological dilation of point sets): P ⊕ {j·v : -rp≤j≤rp}.
void minksum_periodic(GridPtSet &P, int vr, int vc, int rp) {
    GridPtSet out;
    for (const auto &p : P)
        for (int j = -rp; j <= rp; ++j)
            out.insert(GridPt{p.r + j * vr, p.c + j * vc});
    P = keel::move(out);
}

// strel('disk', R, N): MATLAB R2025b. For R<3 or N==0 a true Euclidean
// disk; otherwise the radial decomposition via periodic lines (Adams 1993;
// Jones & Soille 1996) with N∈{4,6,8} basis directions (default 4), then
// compensating horizontal/vertical line strels. The final neighbourhood is
// the Minkowski sum (full dilation) of all decomposition strels.
Value strel_disk(double r_in, int n, keel::Allocator *mr) {
    if (r_in < 1.0) r_in = 1.0;
    const int r = (int)std::lround(r_in);
    if (r < 3 || (n != 4 && n != 6 && n != 8))
        return strel_disk_euclidean(r_in, mr);

    // Basis offset vectors (row, col) per N.
    static const std::array<std::pair<int,int>, 4> B4 =
        {{{1,0},{1,1},{0,1},{-1,1}}};
    static const std::array<std::pair<int,int>, 6> B6 =
        {{{1,0},{1,2},{2,1},{0,1},{-1,2},{-2,1}}};
    static const std::array<std::pair<int,int>, 8> B8 =
        {{{1,0},{2,1},{1,1},{1,2},{0,1},{-1,2},{-1,1},{-2,1}}};
    const std::pair<int,int> *V; int nb;
    if (n == 4)      { V = B4.data(); nb = 4; }
    else if (n == 6) { V = B6.data(); nb = 6; }
    else             { V = B8.data(); nb = 8; }

    // Radial extent of the periodic-line strels (Adams, p.328).
    const double theta = M_PI / (2.0 * n);
    const double k = 2.0 * (double)r /
                     (1.0 / std::tan(theta) + 1.0 / std::sin(theta));

    GridPtSet P;
    P.insert(GridPt{0, 0});
    for (int q = 0; q < nb; ++q) {
        const int vr = V[q].first, vc = V[q].second;
        const double norm = std::hypot((double)vr, (double)vc);
        const int rp = (int)std::floor(k / norm);
        minksum_periodic(P, vr, vc, rp);
    }

    // The decomposition is a little small; pad with H/V line strels.
    int maxRow = 0;
    for (const auto &p : P) maxRow = std::max(maxRow, keel::abs(p.r));
    const int radial_difference = r - maxRow;
    const int len = 2 * (radial_difference - 1) + 1;
    if (len >= 3) {
        const int half = (len - 1) / 2;
        minksum_periodic(P, 0, 1, half);   // line(len, 0)  — horizontal
        minksum_periodic(P, 1, 0, half);   // line(len, 90) — vertical
    }

    int R = 0, C = 0;
    for (const auto &p : P) { R = std::max(R, keel::abs(p.r));
                             C = std::max(C, keel::abs(p.c)); }
    const int H = 2 * R + 1, W = 2 * C + 1;
    keel::Array<keel::u8> m((keel::usize)H * (keel::usize)W, 0);
    for (const auto &p : P)
        m[(keel::usize)(p.r + R) * (keel::usize)W + (keel::usize)(p.c + C)] = 1;
    return pack_logical(m, H, W, mr);
}

Value strel_line(double len, double theta_deg, keel::Allocator *mr) {
    if (len < 1.0) len = 1.0;
    const double th = theta_deg * M_PI / 180.0;
    const double cx = (len - 1) * 0.5 * std::cos(th);
    const double cy = (len - 1) * 0.5 * std::sin(th);
    const int W = std::max(1, (int)std::ceil(2.0 * std::fabs(cx) + 1.0));
    const int H = std::max(1, (int)std::ceil(2.0 * std::fabs(cy) + 1.0));
    keel::Array<keel::u8> m((keel::usize)H * (keel::usize)W, 0);
    const double r0 = (H - 1) * 0.5;
    const double c0 = (W - 1) * 0.5;
    const int Nstep = std::max(1, (int)std::ceil(len));
    for (int k = 0; k < Nstep; ++k) {
        const double t = (Nstep > 1) ? (double(k) / double(Nstep - 1) - 0.5) * (len - 1) : 0.0;
        const int rr = (int)std::round(r0 + t * std::sin(th));
        const int cc = (int)std::round(c0 + t * std::cos(th));
        if (rr >= 0 && rr < H && cc >= 0 && cc < W)
            m[(keel::usize)rr * (keel::usize)W + (keel::usize)cc] = 1;
    }
    return pack_logical(m, H, W, mr);
}

void strel_argcheck(keel::StringRef shape, const Value &arbitrary_nhood)
{
    if (shape == "arbitrary" || shape.empty()) {
        if (arbitrary_nhood.numel() == 0)
            throw Error("strel('arbitrary', NHOOD): NHOOD missing",
                        0, 0, "strel", "", "numkit:strel:nargin");
    } else if (shape != "square" && shape != "rectangle" &&
               shape != "diamond" && shape != "disk" && shape != "line") {
        throw Error(keel::format("strel: unknown shape '{}'", shape), 0, 0, "strel", "",
                    "numkit:strel:badshape");
    }
}

} // anonymous

Value strel(keel::StringRef shape, keel::ArrayRef<double> params, const Value &arbitrary_nhood, keel::Allocator *mr)
{
    strel_argcheck(shape, arbitrary_nhood);
    // Compute the raw logical neighbourhood, then wrap it in a 1x1
    // struct with fields {Neighborhood, Dimensionality} matching
    // MATLAB's strel-object exposed properties. unpack_se() (used by
    // imerode / imdilate / etc.) accepts both the struct form and a
    // bare logical matrix, so legacy code continues to work.
    Value nhood;
    if (shape == "square") {
        const int N = params.empty() ? 3 : (int)params[0];
        nhood = strel_square(N, mr);
    } else if (shape == "rectangle") {
        const int r = params.size() >= 1 ? (int)params[0] : 3;
        const int c = params.size() >= 2 ? (int)params[1] : r;
        nhood = strel_rect(r, c, mr);
    } else if (shape == "diamond") {
        const int r = params.empty() ? 1 : (int)params[0];
        nhood = strel_diamond(r, mr);
    } else if (shape == "disk") {
        const double r = params.empty() ? 5.0 : params[0];
        const int    n = params.size() >= 2 ? (int)params[1] : 4;
        nhood = strel_disk(r, n, mr);
    } else if (shape == "line") {
        const double len = params.size() >= 1 ? params[0] : 3.0;
        const double th  = params.size() >= 2 ? params[1] : 0.0;
        nhood = strel_line(len, th, mr);
    } else {
        const auto &d = arbitrary_nhood.dims();
        const int H = (int)d.rows();
        const int W = (int)d.cols();
        keel::Array<keel::u8> m((keel::usize)H * (keel::usize)W, 0);
        for (int c = 0; c < W; ++c)
            for (int r = 0; r < H; ++r) {
                const double v = arbitrary_nhood.elemAsDouble((keel::usize)c * (keel::usize)H + (keel::usize)r);
                m[(keel::usize)r * (keel::usize)W + (keel::usize)c] = (v != 0.0) ? 1 : 0;
            }
        nhood = pack_logical(m, H, W, mr);
    }

    auto se = Value::structArray(1, 1, mr);
    auto &el = se.structArrayElem(0);
    el.put(keel::String::from("Neighborhood"), std::move(nhood));
    el.put(keel::String::from("Dimensionality"), Value::scalar(2.0, mr));
    return se;
}

// ════════════════════════════════════════════════════════════════════
// erode / dilate / open / close
// ════════════════════════════════════════════════════════════════════

namespace {

inline void store_classed_morph(Value &out, keel::usize i, double v, ValueType t) {
    switch (t) {
        case ValueType::DOUBLE:  out.doubleDataMut()[i]  = v; break;
        case ValueType::SINGLE:  out.singleDataMut()[i]  = (float)v; break;
        case ValueType::UINT8: {
            if (v < 0) v = 0; if (v > 255) v = 255;
            out.uint8DataMut()[i] = (keel::u8)std::lround(v); break;
        }
        case ValueType::UINT16: {
            if (v < 0) v = 0; if (v > 65535) v = 65535;
            out.uint16DataMut()[i] = (keel::u16)std::lround(v); break;
        }
        case ValueType::INT16: {
            if (v < -32768) v = -32768; if (v > 32767) v = 32767;
            out.int16DataMut()[i] = (keel::i16)std::lround(v); break;
        }
        case ValueType::LOGICAL:
            out.logicalDataMut()[i] = v != 0.0 ? 1 : 0; break;
        default:
            throw Error("morph: unsupported class", 0, 0, "morph", "",
                        "numkit:morph:badtype");
    }
}

// Convert SE input to a logical mask (offsets from centre).
struct SEInfo {
    int H, W;
    int half_r, half_c;
    keel::Array<keel::u8> mask;  // row-major
};

SEInfo unpack_se(const Value &SE) {
    // Accept both forms: a bare logical/numeric matrix OR a strel-style
    // 1x1 struct with field 'Neighborhood'.
    const Value *nhood = &SE;
    if (SE.isStruct() && SE.numel() >= 1) {
        const auto &fields = SE.structArrayElem(0);
        if (const Value *v = fields.get("Neighborhood")) nhood = v;
    }
    SEInfo s{};
    s.H = (int)nhood->dims().rows();
    s.W = (int)nhood->dims().cols();
    s.half_r = s.H / 2;
    s.half_c = s.W / 2;
    s.mask.assign((keel::usize)s.H * (keel::usize)s.W, 0);
    for (int c = 0; c < s.W; ++c)
        for (int r = 0; r < s.H; ++r) {
            const double v = nhood->elemAsDouble((keel::usize)c * (keel::usize)s.H + (keel::usize)r);
            s.mask[(keel::usize)r * (keel::usize)s.W + (keel::usize)c] = (v != 0.0) ? 1 : 0;
        }
    return s;
}

// Fast path for a flat SE on byte-backed input (LOGICAL / UINT8): dilation =
// max, erosion = min over the SE-marked, in-bounds input pixels (0 when no
// marked pixel is in bounds — exactly morph_op's convention). The interior
// (where the SE fully fits) is computed with a contiguous min/max inner loop
// the compiler auto-vectorises to pminub/pmaxub; the border falls back to a
// scalar reduction that mirrors morph_op byte-for-byte. ~20-40× over the
// generic elemAsDouble path on large images.
template <bool IsErode>
Value morph_flat_u8(const Value &I, const SEInfo &se, keel::Allocator *mr)
{
    const int H = (int)I.dims().rows();
    const int W = (int)I.dims().cols();
    Value out = Value::matrix(H, W, I.type(), mr);
    if (H == 0 || W == 0) return out;
    const keel::u8 *src = I.isLogical() ? I.logicalData() : I.uint8Data();
    keel::u8 *dst = out.isLogical() ? out.logicalDataMut() : out.uint8DataMut();

    // Marked offsets (dr, dc) relative to centre, plus their extent.
    struct Off { int dr, dc; };
    keel::Array<Off> off;
    off.reserve((keel::usize)se.H * (keel::usize)se.W);
    int drMin = 0, drMax = 0, dcMin = 0, dcMax = 0;
    for (int kj = 0; kj < se.W; ++kj)
        for (int ki = 0; ki < se.H; ++ki)
            if (se.mask[(keel::usize)ki * (keel::usize)se.W + (keel::usize)kj]) {
                const int dr = ki - se.half_r, dc = kj - se.half_c;
                off.push_back({dr, dc});
                drMin = std::min(drMin, dr); drMax = std::max(drMax, dr);
                dcMin = std::min(dcMin, dc); dcMax = std::max(dcMax, dc);
            }
    if (off.empty()) {                       // empty SE → all-0 (matches morph_op)
        std::fill(dst, dst + (keel::usize)H * (keel::usize)W, keel::u8{0});
        return out;
    }

    // Interior region where every marked offset lands in bounds.
    const int ir0 = std::max(0, -drMin), ir1 = std::min(H, H - drMax);
    const int ic0 = std::max(0, -dcMin), ic1 = std::min(W, W - dcMax);
    const keel::u8 INIT = IsErode ? keel::u8(255) : keel::u8(0);

    // Interior: contiguous min/max passes per marked offset (auto-vectorised).
    if (ir1 > ir0 && ic1 > ic0) {
        for (int oc = ic0; oc < ic1; ++oc) {
            keel::u8 *dcol = dst + (keel::usize)oc * (keel::usize)H;
            for (int r = ir0; r < ir1; ++r) dcol[r] = INIT;
            for (const Off &o : off) {
                const keel::u8 *scol = src + (keel::usize)(oc + o.dc) * (keel::usize)H + o.dr;
                if (IsErode)
                    for (int r = ir0; r < ir1; ++r) dcol[r] = std::min(dcol[r], scol[r]);
                else
                    for (int r = ir0; r < ir1; ++r) dcol[r] = std::max(dcol[r], scol[r]);
            }
        }
    }

    // Border: exact scalar reduction (0 when no marked pixel is in bounds).
    for (int oc = 0; oc < W; ++oc) {
        const bool colInterior = (oc >= ic0 && oc < ic1);
        for (int orow = 0; orow < H; ++orow) {
            if (colInterior && orow >= ir0 && orow < ir1) continue;  // done above
            int best = 0; bool covered = false;
            for (const Off &o : off) {
                const int rc = oc + o.dc, rr = orow + o.dr;
                if (rc < 0 || rc >= W || rr < 0 || rr >= H) continue;
                const int v = src[(keel::usize)rc * (keel::usize)H + (keel::usize)rr];
                if (!covered) { best = v; covered = true; }
                else if (IsErode) { if (v < best) best = v; }
                else { if (v > best) best = v; }
            }
            dst[(keel::usize)oc * (keel::usize)H + (keel::usize)orow] =
                static_cast<keel::u8>(covered ? best : 0);
        }
    }
    return out;
}

// 1-D windowed reduce (max for dilate, min for erode) along one contiguous
// column: out[r] = reduce over in[r+lo .. r+hi] ∩ [0,n). Block van Herk–
// Gil-Werman — exactly 3 reduce-ops per pixel, BRANCHLESS (vs the monotonic
// deque's two per-pixel while-loops). The OOB ends are padded with the
// identity (0 for max / 255 for min), which is equivalent to skipping them.
// Scratch P/g/h must each hold ≥ n + k - 1 bytes (k = hi - lo + 1).
template <bool IsErode>
inline void blockVanHerk(const keel::u8 *in, keel::u8 *out, int n,
                         int lo, int hi,
                         keel::u8 *P, keel::u8 *g, keel::u8 *h)
{
    auto MX = [](keel::u8 a, keel::u8 b) {
        return IsErode ? std::min(a, b) : std::max(a, b);
    };
    const int k = hi - lo + 1;
    const keel::u8 PAD = IsErode ? keel::u8(255) : keel::u8(0);
    const int off = -lo;          // ≥ 0 (lo ≤ 0)
    const int m = n + k - 1;      // padded length: out[r]=reduce(P[r..r+k-1])

    for (int i = 0; i < off; ++i)      P[i] = PAD;
    for (int i = 0; i < n; ++i)        P[off + i] = in[i];
    for (int i = off + n; i < m; ++i)  P[i] = PAD;

    // Forward running-reduce within each size-k block.
    for (int b = 0; b < m; b += k) {
        const int e = std::min(b + k, m);
        g[b] = P[b];
        for (int i = b + 1; i < e; ++i) g[i] = MX(g[i - 1], P[i]);
    }
    // Backward running-reduce within each block.
    for (int b = 0; b < m; b += k) {
        const int e = std::min(b + k, m);
        h[e - 1] = P[e - 1];
        for (int i = e - 2; i >= b; --i) h[i] = MX(h[i + 1], P[i]);
    }
    // out[r] = reduce(P[r .. r+k-1]) = MX(suffix at r, prefix at r+k-1).
    for (int r = 0; r < n; ++r) out[r] = MX(h[r], g[r + k - 1]);
}

// Returns true if `se` is usable by the van Herk path: centre marked, and
// every column's marked rows form a single contiguous run (true for disk,
// square, rectangle, diamond, octagon, line — anything convex per column).
inline bool vanHerkEligible(const SEInfo &se)
{
    if (se.half_r < 0 || se.half_c < 0) return false;
    if (!se.mask[(keel::usize)se.half_r * (keel::usize)se.W + (keel::usize)se.half_c]) return false;
    for (int kj = 0; kj < se.W; ++kj) {
        int lo = -1, hi = -1;
        for (int ki = 0; ki < se.H; ++ki)
            if (se.mask[(keel::usize)ki * (keel::usize)se.W + (keel::usize)kj]) { if (lo < 0) lo = ki; hi = ki; }
        for (int ki = lo; ki >= 0 && ki <= hi; ++ki)
            if (!se.mask[(keel::usize)ki * (keel::usize)se.W + (keel::usize)kj]) return false;  // gap
    }
    return true;
}

// van Herk morphology for a flat, marked-centre, per-column-contiguous SE on
// byte input: decompose the SE into its columns, take a vertical windowed
// reduce per column (O(1)/px via colWindowReduce), then a horizontal combine
// across SE columns (auto-vectorised). Cost ∝ SE width (diameter), not SE
// area — so it pulls ahead of morph_flat_u8 as the SE grows. Byte-identical
// to morph_flat_u8 (same marked-pixel set, same skip-OOB convention).
template <bool IsErode>
Value morph_vanherk_u8(const Value &I, const SEInfo &se, keel::Allocator *mr)
{
    const int H = (int)I.dims().rows();
    const int W = (int)I.dims().cols();
    Value out = Value::matrix(H, W, I.type(), mr);
    if (H == 0 || W == 0) return out;
    const keel::u8 *src = I.isLogical() ? I.logicalData() : I.uint8Data();
    keel::u8 *dst = out.isLogical() ? out.logicalDataMut() : out.uint8DataMut();
    const keel::u8 INIT = IsErode ? keel::u8(255) : keel::u8(0);
    std::fill(dst, dst + (keel::usize)H * (keel::usize)W, INIT);

    // Group SE columns by their vertical run (lo,hi). Symmetric SEs (disk,
    // square, diamond, …) have many columns sharing a run, so we compute the
    // vertical reduce V once per DISTINCT run and reuse it for every column
    // that shares it — fewer of the (scalar) vertical passes.
    struct Run { int lo, hi; };
    keel::Array<Run> runs;
    keel::Array<int> colRun;   // SE-column index (with marked pixels) → run id
    keel::Array<int> colDc;    // …and its horizontal shift
    int maxK = 1;
    for (int kj = 0; kj < se.W; ++kj) {
        int kiLo = -1, kiHi = -1;
        for (int ki = 0; ki < se.H; ++ki)
            if (se.mask[(keel::usize)ki * (keel::usize)se.W + (keel::usize)kj]) { if (kiLo < 0) kiLo = ki; kiHi = ki; }
        if (kiLo < 0) continue;                          // empty SE column
        const int lo = kiLo - se.half_r, hi = kiHi - se.half_r;
        int rid = -1;
        for (int t = 0; t < (int)runs.size(); ++t)
            if (runs[t].lo == lo && runs[t].hi == hi) { rid = t; break; }
        if (rid < 0) { rid = (int)runs.size(); runs.push_back({lo, hi}); }
        colRun.push_back(rid);
        colDc.push_back(kj - se.half_c);
        maxK = std::max(maxK, hi - lo + 1);
    }

    keel::ArenaTemp arena = keel::get_scratch(mr);
    keel::Array<keel::u8> V((keel::usize)H * (keel::usize)W, arena.allocator());
    keel::Array<keel::u8> P((keel::usize)(H + maxK), arena.allocator()),
                             G((keel::usize)(H + maxK), arena.allocator()),
                             Hh((keel::usize)(H + maxK), arena.allocator());  // tiny, reused across columns

    for (int rid = 0; rid < (int)runs.size(); ++rid) {
        const int lo = runs[rid].lo, hi = runs[rid].hi;
        // Vertical windowed reduce for this run, once, over every column.
        for (int c = 0; c < W; ++c)
            blockVanHerk<IsErode>(src + (keel::usize)c * (keel::usize)H,
                                  V.data() + (keel::usize)c * (keel::usize)H, H, lo, hi,
                                  P.data(), G.data(), Hh.data());
        // Combine into dst for every SE column that shares this run.
        for (int ci = 0; ci < (int)colRun.size(); ++ci) {
            if (colRun[ci] != rid) continue;
            const int dc = colDc[ci];
            const int oc0 = std::max(0, -dc), oc1 = std::min(W, W - dc);
            for (int oc = oc0; oc < oc1; ++oc) {
                keel::u8 *dcol = dst + (keel::usize)oc * (keel::usize)H;
                const keel::u8 *vcol = V.data() + (keel::usize)(oc + dc) * (keel::usize)H;
                if (IsErode)
                    for (int r = 0; r < H; ++r) dcol[r] = std::min(dcol[r], vcol[r]);
                else
                    for (int r = 0; r < H; ++r) dcol[r] = std::max(dcol[r], vcol[r]);
            }
        }
    }
    return out;
}

template <bool IsErode>
Value morph_op(const Value &I, const Value &SE, keel::Allocator *mr)
{
    auto se = unpack_se(SE);
    // Byte-backed flat-SE fast paths (LOGICAL / UINT8). van Herk (O(SE width))
    // for marked-centre per-column-contiguous SEs (disk/square/rect/diamond),
    // else the auto-vectorised O(SE area) reduction. Other types
    // (double/single/uint16/int16, incl. ±Inf grayscale) keep the generic
    // reduction below.
    if (I.type() == ValueType::LOGICAL || I.type() == ValueType::UINT8) {
        if (vanHerkEligible(se))
            return morph_vanherk_u8<IsErode>(I, se, mr);
        return morph_flat_u8<IsErode>(I, se, mr);
    }
    const int H = (int)I.dims().rows();
    const int W = (int)I.dims().cols();
    Value out = Value::matrix(H, W, I.type(), mr);
    if (H == 0 || W == 0) return out;

    for (int oc = 0; oc < W; ++oc) {
        for (int orow = 0; orow < H; ++orow) {
            // Use NaN as the "no SE pixel covered yet" sentinel so we can
            // tell apart an empty neighbourhood (→ output 0) from an
            // actual ±Inf reduction over real input values (preserve as
            // is — needed by callers like imimposemin which rely on
            // ±Inf propagation through reconstruction).
            double best = std::numeric_limits<double>::quiet_NaN();
            for (int kj = 0; kj < se.W; ++kj) {
                const int c_in = oc + kj - se.half_c;
                if (c_in < 0 || c_in >= W) continue;
                for (int ki = 0; ki < se.H; ++ki) {
                    if (!se.mask[(keel::usize)ki * (keel::usize)se.W + (keel::usize)kj]) continue;
                    const int r_in = orow + ki - se.half_r;
                    if (r_in < 0 || r_in >= H) continue;
                    const double v = I.elemAsDouble((keel::usize)c_in * (keel::usize)H + (keel::usize)r_in);
                    if (std::isnan(best)) {
                        best = v;
                    } else if (IsErode) {
                        if (v < best) best = v;
                    } else {
                        if (v > best) best = v;
                    }
                }
            }
            if (std::isnan(best)) {
                // No SE-marked pixel landed in bounds — fall back to 0
                // (the original convention for binary erosion with a
                // constant-0 boundary; grayscale callers don't rely on
                // this branch).
                best = 0.0;
            }
            store_classed_morph(out, (keel::usize)oc * (keel::usize)H + (keel::usize)orow, best, I.type());
        }
    }
    return out;
}

void imreconstruct_argcheck(const Value &marker, const Value &mask)
{
    internal::requireImageClass(marker, "imreconstruct", "marker", 1,
                                internal::ImgClasses::AnyNumericReal);
    internal::requireImageClass(mask, "imreconstruct", "mask", 2,
                                internal::ImgClasses::AnyNumericReal);
    if (marker.type() != mask.type())
        throw Error(keel::format("imreconstruct: marker and mask must have the same "
                                 "class (got {} and {}).",
                                 mtypeName(marker.type()), mtypeName(mask.type())),
                    0, 0, "imreconstruct", "",
                    "numkit:imreconstruct:notSameClass");
    if (mask.dims().rows() != marker.dims().rows()
        || mask.dims().cols() != marker.dims().cols())
        throw Error("imreconstruct: marker and mask must have the same shape",
                    0, 0, "imreconstruct", "", "numkit:imreconstruct:shape");
}

void imfill_holes_argcheck(const Value &BW)
{
    internal::requireImageClass(BW, "imfill", "BW", 1,
                                internal::ImgClasses::FloatIntLog);
}

void imregionalmax_argcheck(const Value &I)
{
    internal::requireImageClass(I, "imregionalmax", "I", 1,
                                internal::ImgClasses::AnyNumericReal);
}

void imregionalmin_argcheck(const Value &I)
{
    internal::requireImageClass(I, "imregionalmin", "I", 1,
                                internal::ImgClasses::AnyNumericReal);
}

void imhmax_argcheck(const Value &I, double h)
{
    internal::requireImageClass(I, "imhmax", "I", 1,
                                internal::ImgClasses::FloatInt);
    if (!(h >= 0.0))
        throw Error("imhmax: h must be ≥ 0",
                    0, 0, "imhmax", "", "numkit:imhmax:h");
}

void imhmin_argcheck(const Value &I, double h)
{
    internal::requireImageClass(I, "imhmin", "I", 1,
                                internal::ImgClasses::FloatInt);
    if (!(h >= 0.0))
        throw Error("imhmin: h must be ≥ 0",
                    0, 0, "imhmin", "", "numkit:imhmin:h");
}

void imimposemin_argcheck(const Value &I, const Value &BW)
{
    internal::requireImageClass(I, "imimposemin", "I", 1,
                                internal::ImgClasses::FloatInt);
    if (BW.dims().rows() != I.dims().rows() || BW.dims().cols() != I.dims().cols())
        throw Error("imimposemin: I and BW must have the same shape",
                    0, 0, "imimposemin", "", "numkit:imimposemin:shape");
}

void imclearborder_argcheck(const Value &BW)
{
    internal::requireImageClass(BW, "imclearborder", "BW", 1,
                                internal::ImgClasses::FloatIntLog);
}

} // anonymous

Value imerode(const Value &I, const Value &SE, keel::Allocator *mr) {
    return morph_op<true>(I, SE, mr);
}

Value imdilate(const Value &I, const Value &SE, keel::Allocator *mr) {
    return morph_op<false>(I, SE, mr);
}

Value imopen(const Value &I, const Value &SE, keel::Allocator *mr) {
    Value e = imerode(I, SE, mr);
    return imdilate(e, SE, mr);
}

Value imclose(const Value &I, const Value &SE, keel::Allocator *mr) {
    Value d = imdilate(I, SE, mr);
    return imerode(d, SE, mr);
}

// ════════════════════════════════════════════════════════════════════
// imreconstruct — morphological reconstruction by dilation
// ════════════════════════════════════════════════════════════════════
// J_{k+1} = min(imdilate(J_k, SE), mask), J_0 = marker.
// The fixed point exists because the iteration is monotone non-
// decreasing (each dilation only adds; the min-cap can never push a
// value above mask). Stopping criterion: J_{k+1} == J_k. Works for
// both binary and grayscale inputs because nk's `imdilate` is
// the grayscale max-in-neighborhood form (which reduces to binary
// dilation when the input is logical).

Value imreconstruct(const Value &marker, const Value &mask, int conn, keel::Allocator *mr)
{
    imreconstruct_argcheck(marker, mask);
    if (conn != 4) conn = 8;
    const keel::usize H = marker.dims().rows();
    const keel::usize W = marker.dims().cols();

    const keel::usize N = marker.numel();
    const ValueType srcT = marker.type();

    auto writeNative = [&](Value &dst, keel::usize i, double v) {
        switch (srcT) {
            case ValueType::DOUBLE: dst.doubleDataMut()[i] = v; break;
            case ValueType::SINGLE: dst.singleDataMut()[i] = float(v); break;
            case ValueType::UINT8: {
                if (v < 0.0) v = 0.0; if (v > 255.0) v = 255.0;
                dst.uint8DataMut()[i] = keel::u8(std::lround(v)); break;
            }
            case ValueType::UINT16: {
                if (v < 0.0) v = 0.0; if (v > 65535.0) v = 65535.0;
                dst.uint16DataMut()[i] = keel::u16(std::lround(v)); break;
            }
            case ValueType::LOGICAL:
                dst.logicalDataMut()[i] = (v != 0.0) ? 1u : 0u; break;
            default:
                dst.doubleDataMut()[i] = v; break;
        }
    };

    // Vincent (1993) fast hybrid reconstruction by dilation: a raster pass
    // and an anti-raster pass propagate most of the way, then a FIFO queue
    // finishes only where values can still rise. O(N) (plus bounded queue
    // traffic) instead of the naive O(N·(H+W)) iterate-dilate-until-stable.
    // Same fixed point: J = max-propagated marker, capped by the mask.
    keel::ArenaTemp arena = keel::get_scratch(mr);
    keel::Array<double> I(N, arena.allocator());   // mask
    keel::Array<double> J(N, arena.allocator());   // working reconstruction
    for (keel::usize i = 0; i < N; ++i) {
        I[i] = mask.elemAsDouble(i);
        J[i] = std::min(marker.elemAsDouble(i), I[i]);
    }

    struct Off { int dr, dc; };
    static constexpr Off kOff8[8] = {
        {-1,-1},{-1,0},{-1,1},{0,-1},{0,1},{1,-1},{1,0},{1,1} };
    static constexpr Off kOff4[4] = { {-1,0},{0,-1},{0,1},{1,0} };
    const Off *nbr = (conn == 8) ? kOff8 : kOff4;
    const int nnb  = (conn == 8) ? 8 : 4;

    const int Hi = static_cast<int>(H);
    const int Wi = static_cast<int>(W);
    auto idxOf = [Hi](int r, int c) -> keel::usize {
        return static_cast<keel::usize>(c) * static_cast<keel::usize>(Hi) + static_cast<keel::usize>(r);
    };

    // Raster scan (increasing linear index): pull from already-processed
    // (causal) neighbours — those with a smaller linear index.
    for (keel::usize i = 0; i < N; ++i) {
        const int r = static_cast<int>(i % H);
        const int c = static_cast<int>(i / H);
        double m = J[i];
        for (int k = 0; k < nnb; ++k) {
            const int nr = r + nbr[k].dr, nc = c + nbr[k].dc;
            if (nr < 0 || nr >= Hi || nc < 0 || nc >= Wi) continue;
            const keel::usize q = idxOf(nr, nc);
            if (q < i && J[q] > m) m = J[q];
        }
        J[i] = std::min(m, I[i]);
    }

    // Anti-raster scan (decreasing index): pull from anti-causal neighbours,
    // and seed the FIFO with pixels that still have a neighbour they could
    // raise.
    keel::Array<keel::usize> fifo(0, arena.allocator());
    fifo.reserve(N);
    for (keel::usize ii = N; ii-- > 0; ) {
        const keel::usize i = ii;
        const int r = static_cast<int>(i % H);
        const int c = static_cast<int>(i / H);
        double m = J[i];
        for (int k = 0; k < nnb; ++k) {
            const int nr = r + nbr[k].dr, nc = c + nbr[k].dc;
            if (nr < 0 || nr >= Hi || nc < 0 || nc >= Wi) continue;
            const keel::usize q = idxOf(nr, nc);
            if (q > i && J[q] > m) m = J[q];
        }
        J[i] = std::min(m, I[i]);
        for (int k = 0; k < nnb; ++k) {
            const int nr = r + nbr[k].dr, nc = c + nbr[k].dc;
            if (nr < 0 || nr >= Hi || nc < 0 || nc >= Wi) continue;
            const keel::usize q = idxOf(nr, nc);
            if (q > i && J[q] < J[i] && J[q] < I[q]) { fifo.push_back(i); break; }
        }
    }

    // Propagation: pop p, raise each neighbour q that is below J[p] and not
    // yet at its mask cap.
    keel::usize head = 0;
    while (head < fifo.size()) {
        const keel::usize p = fifo[head++];
        const int r = static_cast<int>(p % H);
        const int c = static_cast<int>(p / H);
        const double Jp = J[p];
        for (int k = 0; k < nnb; ++k) {
            const int nr = r + nbr[k].dr, nc = c + nbr[k].dc;
            if (nr < 0 || nr >= Hi || nc < 0 || nc >= Wi) continue;
            const keel::usize q = idxOf(nr, nc);
            if (J[q] < Jp && J[q] < I[q]) {
                J[q] = std::min(Jp, I[q]);
                fifo.push_back(q);
            }
        }
    }

    Value out = Value::matrix(H, W, srcT, mr);
    for (keel::usize i = 0; i < N; ++i) writeNative(out, i, J[i]);
    return out;
}

// ════════════════════════════════════════════════════════════════════
// imfill('holes')  — composes imreconstruct
// ════════════════════════════════════════════════════════════════════
// A "hole" is a 0-pixel NOT connectivity-reachable from the image
// border. We reconstruct the border-touching part of the complement
// (background reachable from the rim) through ~BW, then take the
// complement of that result — anything that wasn't reachable from
// the border becomes 1 in the output, which means foreground
// pixels stay 1 and holes are filled.

Value imfill_holes(const Value &BW, int conn, keel::Allocator *mr)
{
    imfill_holes_argcheck(BW);
    if (conn != 4) conn = 8;
    const keel::usize H = BW.dims().rows();
    const keel::usize W = BW.dims().cols();
    if (BW.type() != ValueType::LOGICAL) {
        const keel::usize N = H * W;
        if (N == 0) return Value::matrix(H, W, BW.type(), mr);
        // Morphological reconstruction by erosion (via complement dilation):
        //   T      = max(I)
        //   mask   = T - I
        //   marker = T - I on the image border (rim), -Inf in the interior
        //   J      = T - imreconstruct(marker, mask, conn)
        double maxI = -std::numeric_limits<double>::infinity();
        for (keel::usize i = 0; i < N; ++i) {
            const double v = BW.elemAsDouble(i);
            if (std::isfinite(v) && v > maxI) maxI = v;
        }
        if (!std::isfinite(maxI)) maxI = 0.0;
        const double NEG_INF = -std::numeric_limits<double>::infinity();
        Value marker = Value::matrix(H, W, ValueType::DOUBLE, mr);
        Value mask   = Value::matrix(H, W, ValueType::DOUBLE, mr);
        double *md = marker.doubleDataMut();
        double *kd = mask.doubleDataMut();
        for (keel::usize c = 0; c < W; ++c) {
            for (keel::usize r = 0; r < H; ++r) {
                const keel::usize idx = c * H + r;
                const double ic = maxI - BW.elemAsDouble(idx);
                kd[idx] = ic;
                const bool onRim = (r == 0 || c == 0 || r + 1 == H || c + 1 == W);
                md[idx] = onRim ? ic : NEG_INF;
            }
        }
        Value R = imreconstruct(marker, mask, conn, mr);
        Value J = Value::matrix(H, W, ValueType::DOUBLE, mr);
        double *jd = J.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i)
            jd[i] = maxI - R.elemAsDouble(i);
        return internal::castResultToClass(J, BW.type(), mr);
    }
    Value out = Value::matrix(H, W, ValueType::LOGICAL, mr);
    if (H == 0 || W == 0) return out;

    // marker = ~BW restricted to the image border (rim).
    // mask    = ~BW.
    Value marker = Value::matrix(H, W, ValueType::LOGICAL, mr);
    Value mask   = Value::matrix(H, W, ValueType::LOGICAL, mr);
    keel::u8 *md = marker.logicalDataMut();
    keel::u8 *kd = mask.logicalDataMut();
    for (keel::usize c = 0; c < W; ++c)
        for (keel::usize r = 0; r < H; ++r) {
            const keel::usize idx = c * H + r;
            const bool fg = (BW.elemAsDouble(idx) != 0.0);
            kd[idx] = fg ? 0u : 1u;            // ~BW
            const bool onRim = (r == 0 || c == 0 ||
                                  r + 1 == H || c + 1 == W);
            md[idx] = (onRim && !fg) ? 1u : 0u; // marker = rim ∩ ~BW
        }

    Value R = imreconstruct(marker, mask, conn, mr);

    // J = ~R.
    keel::u8 *od = out.logicalDataMut();
    for (keel::usize i = 0; i < H * W; ++i) {
        const bool r = (R.elemAsDouble(i) != 0.0);
        od[i] = r ? 0u : 1u;
    }
    return out;
}

// ════════════════════════════════════════════════════════════════════
// imregionalmax / imregionalmin
// ════════════════════════════════════════════════════════════════════
// Standard formula:  regmax(I) = (I − imreconstruct(I − 1, I)) > 0.
// The marker (I − 1) is below I except at regional maxima, where the
// reconstruction can't grow back up to I (because dilation only takes
// from neighbours that are themselves capped at I − 1). So pixels
// where the reconstruction equals I lie outside any regional max,
// and pixels where it stays below I are exactly the maxima.
// imregionalmin reuses imregionalmax on a value-inverted copy of I:
//   typeMax − I    for unsigned integer classes
//   − I             for signed / floating-point
// Inverting flips peaks↔troughs, so regional maxima of −I are the
// regional minima of I.

Value imregionalmax(const Value &I, int conn, keel::Allocator *mr)
{
    imregionalmax_argcheck(I);
    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    const keel::usize N = I.numel();
    Value out = Value::matrix(H, W, ValueType::LOGICAL, mr);
    if (N == 0) return out;

    // Build the marker = max(I − 1, lower_bound). Use DOUBLE through
    // the operation to avoid the integer-saturation surprise for
    // I = 0 / I = INT_MIN / etc.
    // Edge case: a +Inf pixel breaks the naïve "marker = I − 1" recipe
    // because (+Inf) − 1 = +Inf, and (mask = +Inf) > (recon = +Inf) is
    // false — so a +Inf would never be flagged. We fix this by clamping
    // marker at the largest finite element of I; the relative ordering
    // is preserved and Vincent's formula now correctly flags +Inf
    // pixels as regional maxima. The dual −Inf case isn't an issue for
    // imregionalmax (the formula correctly leaves −Inf as a non-max).
    Value marker = Value::matrix(H, W, ValueType::DOUBLE, mr);
    Value mask   = Value::matrix(H, W, ValueType::DOUBLE, mr);
    double *md = marker.doubleDataMut();
    double *kd = mask.doubleDataMut();
    double maxFin = -std::numeric_limits<double>::infinity();
    for (keel::usize i = 0; i < N; ++i) {
        const double v = I.elemAsDouble(i);
        if (std::isfinite(v) && v > maxFin) maxFin = v;
    }
    if (!std::isfinite(maxFin)) maxFin = 0.0;
    const double POS_INF = std::numeric_limits<double>::infinity();
    for (keel::usize i = 0; i < N; ++i) {
        const double v = I.elemAsDouble(i);
        kd[i] = v;
        md[i] = (v == POS_INF) ? maxFin : (v - 1.0);
    }
    Value R = imreconstruct(marker, mask, conn, mr);

    // Output = (I > R).
    keel::u8 *od = out.logicalDataMut();
    for (keel::usize i = 0; i < N; ++i) {
        const double iv = I.elemAsDouble(i);
        const double rv = R.elemAsDouble(i);
        od[i] = (iv > rv) ? 1u : 0u;
    }
    return out;
}

Value imregionalmin(const Value &I, int conn, keel::Allocator *mr)
{
    imregionalmin_argcheck(I);
    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    const keel::usize N = I.numel();
    Value out = Value::matrix(H, W, ValueType::LOGICAL, mr);
    if (N == 0) return out;

    // Invert into a DOUBLE copy: (high - I) flips the order so peaks
    // become troughs. high is chosen large enough that I_inv stays
    // non-negative for unsigned classes (purely cosmetic — only the
    // ordering matters).
    double high = 0.0;
    for (keel::usize i = 0; i < N; ++i) {
        const double v = I.elemAsDouble(i);
        if (v > high) high = v;
    }
    Value Iinv = Value::matrix(H, W, ValueType::DOUBLE, mr);
    double *id = Iinv.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i)
        id[i] = high - I.elemAsDouble(i);

    return imregionalmax(Iinv, conn, mr);
}

// ════════════════════════════════════════════════════════════════════
// imhmax / imhmin — h-extrema transforms
// ════════════════════════════════════════════════════════════════════
//   imhmax(I, h) = imreconstruct(I − h, I)
//   imhmin(I, h) = invert(imhmax(invert(I), h))
// h-maxima suppresses regional maxima shallower than h. Used as a
// precursor to imregionalmax to ignore small / noise-like peaks.

Value imhmax(const Value &I, double h, int conn, keel::Allocator *mr)
{
    imhmax_argcheck(I, h);
    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    const keel::usize N = I.numel();
    if (N == 0) return Value::matrix(H, W, I.type(), mr);

    Value marker = Value::matrix(H, W, ValueType::DOUBLE, mr);
    Value mask   = Value::matrix(H, W, ValueType::DOUBLE, mr);
    double *md = marker.doubleDataMut();
    double *kd = mask.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) {
        const double v = I.elemAsDouble(i);
        kd[i] = v;
        md[i] = v - h;
    }
    // Class-preserving result (probed R2025b): single / integer inputs
    // come back in the input class with saturating arithmetic.
    return internal::castResultToClass(
        imreconstruct(marker, mask, conn, mr), I.type(), mr);
}

Value imhmin(const Value &I, double h, int conn, keel::Allocator *mr)
{
    imhmin_argcheck(I, h);
    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    const keel::usize N = I.numel();
    if (N == 0) return Value::matrix(H, W, I.type(), mr);

    // Invert about a high value so peaks ↔ troughs, then reuse imhmax,
    // then invert back.
    double high = 0.0;
    for (keel::usize i = 0; i < N; ++i) {
        const double v = I.elemAsDouble(i);
        if (v > high) high = v;
    }
    Value Iinv = Value::matrix(H, W, ValueType::DOUBLE, mr);
    double *id = Iinv.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i)
        id[i] = high - I.elemAsDouble(i);

    Value Jinv = imhmax(Iinv, h, conn, mr);

    // Invert back: J = high - Jinv. Class-preserving result (probed
    // R2025b): single / integer inputs come back in the input class.
    Value J = Value::matrix(H, W, ValueType::DOUBLE, mr);
    double *jd = J.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i)
        jd[i] = high - Jinv.elemAsDouble(i);
    return internal::castResultToClass(J, I.type(), mr);
}

// ════════════════════════════════════════════════════════════════════
// imextendedmax / imextendedmin — extended-extrema transforms
// ════════════════════════════════════════════════════════════════════
//   imextendedmax(I, h) = imregionalmax(imhmax(I, h))
//   imextendedmin(I, h) = imregionalmin(imhmin(I, h))
// First flatten any peak shallower than h (imhmax), then locate the
// regional maxima of the result. Pixels in the output are exactly
// those that belong to a regional maximum at least h units above its
// surroundings — the "tall enough" peaks.

Value imextendedmax(const Value &I, double h, int conn, keel::Allocator *mr)
{
    return imregionalmax(imhmax(I, h, conn, mr), conn, mr);
}

Value imextendedmin(const Value &I, double h, int conn, keel::Allocator *mr)
{
    return imregionalmin(imhmin(I, h, conn, mr), conn, mr);
}

// ════════════════════════════════════════════════════════════════════
// imimposemin — minima imposition (Soille 1999, §6.3.1)
// ════════════════════════════════════════════════════════════════════
// Force the regional minima of `I` to be exactly the pixels marked
// in `BW`. Recipe matching MATLAB R2025b / Octave's imimposemin:
//   marker fm = −∞ at BW, +∞ elsewhere      (sentinel marker)
//   mask    m = min(I + h, fm)              (lifted image, drops back
//                                            to −∞ at marker pixels)
//   J         = R^E_m(fm)                   (erosion-reconstruction)
// where `h` is the per-class step:
//   integer classes  → h = 1
//   floating classes → h = (max(I) − min(I)) / 1000
// `h` lifts non-marker pixels just enough that any old regional
// minimum is no longer one (its old neighbours' value at I + h is
// strictly higher than I along the path to a marker). Marker pixels
// stay at −∞ and are the only regional minima of the result.
// Reconstruction by erosion is realised by complementing into the
// dilation domain. For floating-point we use ±Inf directly (matches
// MATLAB exactly); the existing imreconstruct propagates ±Inf as
// expected because every internal min/max is over finite differences.

Value imimposemin(const Value &I, const Value &BW, int conn, keel::Allocator *mr)
{
    imimposemin_argcheck(I, BW);
    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    const keel::usize N = I.numel();
    if (N == 0) return Value::matrix(H, W, I.type(), mr);

    // Per-class lift step h.
    double minI =  std::numeric_limits<double>::infinity();
    double maxI = -std::numeric_limits<double>::infinity();
    for (keel::usize i = 0; i < N; ++i) {
        const double v = I.elemAsDouble(i);
        if (v < minI) minI = v;
        if (v > maxI) maxI = v;
    }
    if (!std::isfinite(minI)) minI = 0.0;
    if (!std::isfinite(maxI)) maxI = 0.0;
    const ValueType inT = I.type();
    const bool is_float = (inT == ValueType::DOUBLE || inT == ValueType::SINGLE);
    const double h = is_float ? ((maxI - minI) / 1000.0) : 1.0;

    // marker fm: -Inf at BW, +Inf elsewhere.
    // mask    m: min(I + h, fm)  →  -Inf at BW, (I + h) elsewhere.
    Value mask   = Value::matrix(H, W, ValueType::DOUBLE, mr);
    Value marker = Value::matrix(H, W, ValueType::DOUBLE, mr);
    double *md = mask.doubleDataMut();
    double *gd = marker.doubleDataMut();
    const double NEG_INF = -std::numeric_limits<double>::infinity();
    const double POS_INF =  std::numeric_limits<double>::infinity();
    for (keel::usize i = 0; i < N; ++i) {
        const bool b = (BW.elemAsDouble(i) != 0.0);
        const double v = I.elemAsDouble(i);
        if (b) { md[i] = NEG_INF; gd[i] = NEG_INF; }
        else   { md[i] = v + h;   gd[i] = POS_INF; }
    }

    // Reconstruction by erosion via complement-trick over dilation:
    //   J = T − imreconstruct(T − fm, T − m, conn)
    // T must be a strict upper bound on every working value. With ±Inf
    // in fm/m the complement gives ∓Inf, which imreconstruct handles
    // as expected (any finite mask value caps from below).
    const double T = (std::isfinite(maxI) ? maxI : 0.0) + keel::abs(h) + 1.0;

    Value m_inv = Value::matrix(H, W, ValueType::DOUBLE, mr);
    Value g_inv = Value::matrix(H, W, ValueType::DOUBLE, mr);
    double *mi = m_inv.doubleDataMut();
    double *gi = g_inv.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) {
        mi[i] = T - md[i];   // +Inf at BW, T-(I+h) elsewhere
        gi[i] = T - gd[i];   // +Inf at BW, T-(+Inf)=−Inf? — clamp:
        // T − (+Inf) = −Inf; we want g_inv ≤ m_inv, both at marker
        // are equal (+Inf). At non-marker g_inv = −Inf < m_inv ≥ 0.
    }

    Value J_inv = imreconstruct(g_inv, m_inv, conn, mr);

    Value J = Value::matrix(H, W, ValueType::DOUBLE, mr);
    double *jd = J.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i)
        jd[i] = T - J_inv.elemAsDouble(i);
    // Class-preserving result (probed R2025b).
    return internal::castResultToClass(J, I.type(), mr);
}

// ════════════════════════════════════════════════════════════════════
// imclearborder — strip components touching the image rim
// ════════════════════════════════════════════════════════════════════
//   marker = BW restricted to the border pixels
//   R      = imreconstruct(marker, BW, conn)   — all FG reachable
//                                                from the rim
//   J      = BW & ~R                           — interior FG only
// For grayscale inputs the same recipe applies treating non-zero as
// foreground; we coerce to LOGICAL since MATLAB documents the result
// as a logical mask.

Value imclearborder(const Value &BW, int conn, keel::Allocator *mr)
{
    imclearborder_argcheck(BW);
    if (BW.type() != ValueType::LOGICAL)
        return internal::maskImageByLogical(
            BW,
            imclearborder(internal::coerceToLogicalImage(BW, mr), conn, mr),
            mr);
    if (conn != 4) conn = 8;
    const keel::usize H = BW.dims().rows();
    const keel::usize W = BW.dims().cols();
    Value out = Value::matrix(H, W, ValueType::LOGICAL, mr);
    if (H == 0 || W == 0) return out;

    Value marker = Value::matrix(H, W, ValueType::LOGICAL, mr);
    Value mask   = Value::matrix(H, W, ValueType::LOGICAL, mr);
    keel::u8 *md = marker.logicalDataMut();
    keel::u8 *kd = mask.logicalDataMut();
    for (keel::usize c = 0; c < W; ++c)
        for (keel::usize r = 0; r < H; ++r) {
            const keel::usize idx = c * H + r;
            const bool fg = (BW.elemAsDouble(idx) != 0.0);
            kd[idx] = fg ? 1u : 0u;
            const bool onRim = (r == 0 || c == 0 ||
                                  r + 1 == H || c + 1 == W);
            md[idx] = (onRim && fg) ? 1u : 0u;
        }

    Value R = imreconstruct(marker, mask, conn, mr);

    keel::u8 *od = out.logicalDataMut();
    for (keel::usize i = 0; i < H * W; ++i) {
        const bool b   = (BW.elemAsDouble(i) != 0.0);
        const bool rim = (R.elemAsDouble(i) != 0.0);
        od[i] = (b && !rim) ? 1u : 0u;
    }
    return out;
}

// ════════════════════════════════════════════════════════════════════
// imtophat / imbothat — top-hat residuals
// ════════════════════════════════════════════════════════════════════
//   imtophat(I, SE) = I − imopen(I, SE)
//   imbothat(I, SE) = imclose(I, SE) − I
// Both are subtractions in the image domain. We compute pixel-wise
// in double then store back in the input class with saturation
// (matching MATLAB's behaviour for integer types).

namespace {

Value tophat_subtract(const Value &lhs, const Value &rhs, keel::Allocator *mr)
{
    const keel::usize H = lhs.dims().rows();
    const keel::usize W = lhs.dims().cols();
    const keel::usize N = lhs.numel();
    Value out = Value::matrix(H, W, lhs.type(), mr);
    for (keel::usize i = 0; i < N; ++i) {
        const double v = lhs.elemAsDouble(i) - rhs.elemAsDouble(i);
        store_classed_morph(out, i, v, lhs.type());
    }
    return out;
}

void mmgradm_argcheck(const Value &I)
{
    internal::requireImageClass(I, "mmgradm", "I", 1,
                                internal::ImgClasses::FloatIntLog);
}

void bwpack_argcheck(const Value &BW)
{
    internal::requireImageClass(BW, "bwpack", "BW", 1,
                                internal::ImgClasses::AnyNumericReal);
}

void bwunpack_argcheck(const Value &BWP, keel::usize &M)
{
    internal::requireUint32Packed(BWP, "bwunpack");
    constexpr keel::usize CLASS_BITS = 32;
    const keel::usize pH = BWP.dims().rows();
    if (M == static_cast<keel::usize>(-1)) M = pH * CLASS_BITS;
    if (M > pH * CLASS_BITS)
        throw Error("bwunpack: M exceeds packed-row capacity",
                    0, 0, "bwunpack", "", "numkit:bwunpack:M");
}

void applylut_argcheck(const Value &BW, const Value &LUT, keel::usize &lutLen, keel::usize &nq, keel::usize &n)
{
    internal::requireImageClass(BW, "applylut", "BW", 1,
                                internal::ImgClasses::AnyNumericReal);
    lutLen = LUT.numel();
    if (lutLen == 0)
        throw Error("applylut: LUT must be non-empty",
                    0, 0, "applylut", "", "numkit:applylut:lutsize");
    nq = 0;
    while ((keel::usize{1} << nq) < lutLen) ++nq;
    if ((keel::usize{1} << nq) != lutLen)
        throw Error("applylut: LUT length must be 2^(n*n)",
                    0, 0, "applylut", "", "numkit:applylut:lutsize");
    n = static_cast<keel::usize>(std::round(std::sqrt((double)nq)));
    if (n * n != nq)
        throw Error("applylut: LUT length not 2^(n*n) for any integer n",
                    0, 0, "applylut", "", "numkit:applylut:lutshape");
}

void bwlookup_argcheck(const Value &lut)
{
    const keel::usize n = lut.numel();
    if (n != 16 && n != 512)
        throw Error("bwlookup: Expected LUT (argument 2) to have 16 or 512 elements.",
                    0, 0, "bwlookup", "", "numkit:bwlookup:lutsize");
}

void bwmorph3_argcheck(const Value &V, keel::StringRef op, int &opIdx)
{
    if      (op == "branchpoints") opIdx = 0;
    else if (op == "clean")        opIdx = 1;
    else if (op == "endpoints")    opIdx = 2;
    else if (op == "fill")         opIdx = 3;
    else if (op == "majority")     opIdx = 4;
    else if (op == "remove")       opIdx = 5;
    else throw Error(keel::format("bwmorph3: unknown operation '{}'", op),
                     0, 0, "bwmorph3", "", "numkit:bwmorph3:badOp");

    if (V.dims().ndim() > 3)
        throw Error("bwmorph3: input must be a 2-D or 3-D array",
                    0, 0, "bwmorph3", "", "numkit:bwmorph3:rank");
}

void bwhitmiss_argcheck(const Value &BW)
{
    internal::requireImageClass(BW, "bwhitmiss", "BW", 1,
                                internal::ImgClasses::AnyNumericReal);
}

void imkeepborder_argcheck(const Value &BW)
{
    internal::requireImageClass(BW, "imkeepborder", "BW", 1,
                                internal::ImgClasses::FloatIntLog);
}

} // anonymous

Value imtophat(const Value &I, const Value &SE, keel::Allocator *mr)
{
    Value opened = imopen(I, SE, mr);
    return tophat_subtract(I, opened, mr);
}

Value imbothat(const Value &I, const Value &SE, keel::Allocator *mr)
{
    Value closed = imclose(I, SE, mr);
    return tophat_subtract(closed, I, mr);
}


Value mmgradm(const Value &I, const Value &se_dil, const Value &se_ero, keel::Allocator *mr)
{
    // mmgradm is reference-absent (Octave image package API) — the
    // nk-defined contract takes the standard grayscale class set.
    mmgradm_argcheck(I);
    const bool dilEmpty = se_dil.numel() == 0;
    const bool eroEmpty = se_ero.numel() == 0;
    Value dilated = dilEmpty ? Value{} : imdilate(I, se_dil, mr);
    Value eroded  = eroEmpty ? Value{} : imerode(I, se_ero, mr);

    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    const keel::usize P = I.dims().is3D() ? I.dims().pages() : 1;
    const keel::usize N = H * W * P;

    if (I.type() == ValueType::LOGICAL) {
        Value out = I.dims().is3D()
            ? Value::matrix3d(H, W, P, ValueType::LOGICAL, mr)
            : Value::matrix(H, W, ValueType::LOGICAL, mr);
        keel::u8 *od = out.logicalDataMut();
        for (keel::usize i = 0; i < N; ++i) {
            const bool d = !dilEmpty && (dilated.elemAsDouble(i) != 0.0);
            const bool e = !eroEmpty && (eroded .elemAsDouble(i) != 0.0);
            // dilated & ~eroded for the standard form; half-gradients drop
            // one of the two terms.
            bool v;
            if (dilEmpty)        v = !e;
            else if (eroEmpty)   v =  d;
            else                 v =  d && !e;
            od[i] = v ? 1u : 0u;
        }
        return out;
    }

    // Numeric: dilated - eroded with class-saturated subtraction.
    Value out = I.dims().is3D()
        ? Value::matrix3d(H, W, P, I.type(), mr)
        : Value::matrix(H, W, I.type(), mr);

    auto satStore = [&](keel::usize i, double v) {
        switch (I.type()) {
            case ValueType::DOUBLE: out.doubleDataMut()[i] = v; break;
            case ValueType::SINGLE: out.singleDataMut()[i] =
                                    static_cast<float>(v); break;
            case ValueType::UINT8: {
                if (v < 0) v = 0; if (v > 255) v = 255;
                out.uint8DataMut()[i] = static_cast<keel::u8>(std::lround(v));
                break;
            }
            case ValueType::UINT16: {
                if (v < 0) v = 0; if (v > 65535) v = 65535;
                out.uint16DataMut()[i] = static_cast<keel::u16>(std::lround(v));
                break;
            }
            case ValueType::INT16: {
                if (v < -32768) v = -32768; if (v > 32767) v = 32767;
                out.int16DataMut()[i] = static_cast<keel::i16>(std::lround(v));
                break;
            }
            default: out.doubleDataMut()[i] = v; break;
        }
    };

    for (keel::usize i = 0; i < N; ++i) {
        double d = dilEmpty ? 0.0 : dilated.elemAsDouble(i);
        double e = eroEmpty ? 0.0 : eroded.elemAsDouble(i);
        double v;
        if (dilEmpty)       v = -e;     // internal: need careful sign for unsigned
        else if (eroEmpty)  v =  d;
        else                v =  d - e;
        satStore(i, v);
    }
    return out;
}

Value bwpack(const Value &BW, keel::Allocator *mr)
{
    bwpack_argcheck(BW);
    constexpr keel::usize CLASS_BITS = 32;
    const keel::usize H = BW.dims().rows();
    const keel::usize W = BW.dims().cols();
    const keel::usize outH = (H + CLASS_BITS - 1) / CLASS_BITS;
    Value out = Value::matrix(outH, W, ValueType::UINT32, mr);
    if (outH == 0 || W == 0) return out;
    keel::u32 *od = out.uint32DataMut();

    for (keel::usize c = 0; c < W; ++c) {
        for (keel::usize orow = 0; orow < outH; ++orow) {
            keel::u32 word = 0;
            for (keel::usize b = 0; b < CLASS_BITS; ++b) {
                const keel::usize r = orow * CLASS_BITS + b;
                if (r >= H) break;
                if (BW.elemAsDouble(c * H + r) != 0.0)
                    word |= (keel::u32{1} << b);
            }
            od[c * outH + orow] = word;
        }
    }
    return out;
}

Value bwunpack(const Value &BWP, keel::usize M, keel::Allocator *mr)
{
    bwunpack_argcheck(BWP, M);
    constexpr keel::usize CLASS_BITS = 32;
    const keel::usize pH = BWP.dims().rows();
    const keel::usize W  = BWP.dims().cols();

    Value out = Value::matrix(M, W, ValueType::LOGICAL, mr);
    if (M == 0 || W == 0) return out;
    keel::u8 *od = out.logicalDataMut();
    const keel::u32 *pd = BWP.uint32Data();

    for (keel::usize c = 0; c < W; ++c) {
        for (keel::usize r = 0; r < M; ++r) {
            const keel::usize orow = r / CLASS_BITS;
            const keel::usize b    = r % CLASS_BITS;
            const keel::u32 bit = (pd[c * pH + orow] >> b) & 1u;
            od[c * M + r] = static_cast<keel::u8>(bit);
        }
    }
    return out;
}

Value applylut(const Value &BW, const Value &LUT, keel::Allocator *mr)
{
    keel::usize lutLen = 0, nq = 0, n = 0;
    applylut_argcheck(BW, LUT, lutLen, nq, n);

    const keel::usize H = BW.dims().rows();
    const keel::usize W = BW.dims().cols();
    Value out = Value::matrix(H, W, LUT.type(), mr);
    if (H == 0 || W == 0) return out;

    // Weight kernel: reshape(2^[nq-1:-1:0], n, n) col-major.
    Value w = Value::matrix(n, n, ValueType::DOUBLE, mr);
    {
        double *wd = w.doubleDataMut();
        for (keel::usize i = 0; i < n * n; ++i)
            wd[i] = static_cast<double>(keel::usize{1} << (nq - 1 - i));
    }

    // BW promoted to double for filter2.
    Value bw_d = Value::matrix(H, W, ValueType::DOUBLE, mr);
    {
        double *bd = bw_d.doubleDataMut();
        for (keel::usize i = 0; i < H * W; ++i)
            bd[i] = (BW.elemAsDouble(i) != 0.0) ? 1.0 : 0.0;
    }

    // filter2(w, BW, 'same') = index per pixel.
    Value idx = signal::filter2(w, bw_d, "same", mr);
    const double *id = idx.doubleData();
    const keel::usize N = H * W;

    auto store_lut = [&](keel::usize i, double v) {
        switch (LUT.type()) {
            case ValueType::DOUBLE: out.doubleDataMut()[i] = v; break;
            case ValueType::SINGLE: out.singleDataMut()[i] =
                                    static_cast<float>(v); break;
            case ValueType::UINT8:  out.uint8DataMut()[i] =
                                    static_cast<keel::u8>(std::lround(v)); break;
            case ValueType::UINT16: out.uint16DataMut()[i] =
                                    static_cast<keel::u16>(std::lround(v)); break;
            case ValueType::INT16:  out.int16DataMut()[i] =
                                    static_cast<keel::i16>(std::lround(v)); break;
            case ValueType::LOGICAL: out.logicalDataMut()[i] =
                                    (v != 0.0) ? 1u : 0u; break;
            default: out.doubleDataMut()[i] = v; break;
        }
    };

    for (keel::usize i = 0; i < N; ++i) {
        keel::usize k = static_cast<keel::usize>(std::lround(id[i]));
        if (k >= lutLen) k = lutLen - 1;
        store_lut(i, LUT.elemAsDouble(k));
    }
    return out;
}

// bwlookup — modern neighbourhood-LUT filter. Identical index convention
// to applylut, restricted to the documented 16- (2×2) and 512- (3×3)
// element table sizes.
Value bwlookup(const Value &BW, const Value &lut, keel::Allocator *mr)
{
    bwlookup_argcheck(lut);
    return applylut(BW, lut, mr);
}


// bwmorph3 — morphological operations on a binary volume. Each voxel is
// evaluated from its 3×3×3 neighbourhood (zero-padded at the border).
// Clean-room port of MATLAB R2025b's bwmorph3Algorithm neighbourhood
// rules (count = number of set voxels in the 27-neighbourhood INCLUDING
// the centre; faces6 = the six 6-connected face neighbours):
//   branchpoints : centre set  AND  count > 3
//   clean        : centre set  AND  count != 1   (drop isolated voxels)
//   endpoints    : centre set  AND  count == 2   (centre + exactly one)
//   fill         : centre set  OR   faces6 == 6  (fill 6-conn holes)
//   majority     : count > 13                    (≥14 of 27)
//   remove       : centre set  AND  faces6 != 6  (strip interior voxels)
Value bwmorph3(const Value &V, keel::StringRef op, keel::Allocator *mr)
{
    int opIdx = 0;
    bwmorph3_argcheck(V, op, opIdx);
    enum Op { BRANCH = 0, CLEAN = 1, ENDP = 2, FILL = 3, MAJORITY = 4, REMOVE = 5 };
    const Op o = static_cast<Op>(opIdx);

    const auto &d = V.dims();
    const int H = static_cast<int>(d.rows());
    const int W = static_cast<int>(d.cols());
    const int P = d.is3D() ? static_cast<int>(d.pages()) : 1;

    Value out = (P > 1) ? Value::matrix3d(H, W, P, ValueType::LOGICAL, mr)
                        : Value::matrix(H, W, ValueType::LOGICAL, mr);
    if (H == 0 || W == 0 || P == 0) return out;
    keel::u8 *od = out.logicalDataMut();

    const keel::usize plane = static_cast<keel::usize>(H) * static_cast<keel::usize>(W);
    auto at = [&](int r, int c, int p) -> int {
        if (r < 0 || r >= H || c < 0 || c >= W || p < 0 || p >= P) return 0;
        return V.elemAsDouble(static_cast<keel::usize>(p) * plane
                              + static_cast<keel::usize>(c) * static_cast<keel::usize>(H)
                              + static_cast<keel::usize>(r)) != 0.0 ? 1 : 0;
    };

    for (int p = 0; p < P; ++p) {
        for (int c = 0; c < W; ++c) {
            for (int r = 0; r < H; ++r) {
                const int centre = at(r, c, p);
                // 27-neighbourhood count (includes centre) and 6 faces.
                int count = 0;
                for (int dp = -1; dp <= 1; ++dp)
                    for (int dc = -1; dc <= 1; ++dc)
                        for (int dr = -1; dr <= 1; ++dr)
                            count += at(r + dr, c + dc, p + dp);
                const int faces6 = at(r - 1, c, p) + at(r + 1, c, p)
                                 + at(r, c - 1, p) + at(r, c + 1, p)
                                 + at(r, c, p - 1) + at(r, c, p + 1);
                int v = 0;
                switch (o) {
                    case BRANCH:   v = (centre && count > 3); break;
                    case CLEAN:    v = (centre && count != 1); break;
                    case ENDP:     v = (centre && count == 2); break;
                    case FILL:     v = (centre || faces6 == 6); break;
                    case MAJORITY: v = (count > 13); break;
                    case REMOVE:   v = (centre && faces6 != 6); break;
                }
                const keel::usize idx = static_cast<keel::usize>(p) * plane
                                 + static_cast<keel::usize>(c) * static_cast<keel::usize>(H)
                                 + static_cast<keel::usize>(r);
                od[idx] = static_cast<keel::u8>(v ? 1 : 0);
            }
        }
    }
    return out;
}

Value bwhitmiss(const Value &BW, const Value &se1, const Value &se2, keel::Allocator *mr)
{
    bwhitmiss_argcheck(BW);
    const keel::usize H = BW.dims().rows();
    const keel::usize W = BW.dims().cols();
    Value out = Value::matrix(H, W, ValueType::LOGICAL, mr);
    if (H == 0 || W == 0) return out;

    // Foreground erosion: where se1 fits in BW.
    Value e1 = imerode(BW, se1, mr);

    // Build !BW as a logical image, then erode with se2.
    Value notBW = Value::matrix(H, W, ValueType::LOGICAL, mr);
    {
        keel::u8 *nd = notBW.logicalDataMut();
        for (keel::usize i = 0; i < H * W; ++i)
            nd[i] = (BW.elemAsDouble(i) != 0.0) ? 0u : 1u;
    }
    Value e2 = imerode(notBW, se2, mr);

    // J = e1 & e2.
    keel::u8 *od = out.logicalDataMut();
    for (keel::usize i = 0; i < H * W; ++i) {
        const bool a = (e1.elemAsDouble(i) != 0.0);
        const bool b = (e2.elemAsDouble(i) != 0.0);
        od[i] = (a && b) ? 1u : 0u;
    }
    return out;
}

// Exact dual of imclearborder: keep only the rim-reachable components.
//   marker = BW ∩ rim, J = imreconstruct(marker, BW, conn)
// imreconstruct already returns a LOGICAL when marker+mask are LOGICAL,
// so we just hand that through.
Value imkeepborder(const Value &BW, int conn, keel::Allocator *mr)
{
    imkeepborder_argcheck(BW);
    if (BW.type() != ValueType::LOGICAL)
        return internal::maskImageByLogical(
            BW,
            imkeepborder(internal::coerceToLogicalImage(BW, mr), conn, mr),
            mr);
    if (conn != 4) conn = 8;
    const keel::usize H = BW.dims().rows();
    const keel::usize W = BW.dims().cols();
    Value out = Value::matrix(H, W, ValueType::LOGICAL, mr);
    if (H == 0 || W == 0) return out;

    Value marker = Value::matrix(H, W, ValueType::LOGICAL, mr);
    Value mask   = Value::matrix(H, W, ValueType::LOGICAL, mr);
    keel::u8 *md = marker.logicalDataMut();
    keel::u8 *kd = mask.logicalDataMut();
    for (keel::usize c = 0; c < W; ++c)
        for (keel::usize r = 0; r < H; ++r) {
            const keel::usize idx = c * H + r;
            const bool fg = (BW.elemAsDouble(idx) != 0.0);
            kd[idx] = fg ? 1u : 0u;
            const bool onRim = (r == 0 || c == 0 ||
                                  r + 1 == H || c + 1 == W);
            md[idx] = (onRim && fg) ? 1u : 0u;
        }
    return imreconstruct(marker, mask, conn, mr);
}

// ════════════════════════════════════════════════════════════════════
// bwmorph — binary morphology operation dispatcher
// ════════════════════════════════════════════════════════════════════
// Clean-room reimplementation written from public references
// and the public references it cites:
//   * R. C. Gonzalez & R. E. Woods, Digital Image Processing, 4th ed.,
//     2018 — Ch. 9, binary morphology (dilate / erode / open / close /
//     hit-or-miss / boundary / thinning / thickening / skeletons);
//   * W. K. Pratt, Digital Image Processing, 4th ed., 2007 — §14,
//     morphological processing and 3x3 lookup-table pixel operations;
//   * L. Lam, S.-W. Lee, C. Y. Suen, "Thinning Methodologies — A
//     Comprehensive Survey", IEEE Trans. PAMI 14(9), 1992.
// The 512-entry 3x3-neighbourhood lookup tables are reference data
// (truth tables of the documented operations) — see bwmorph_luts.h.

namespace {

using Lut = std::array<keel::u8, 512>;

// ──────────────────────────────────────────────────────────────────
// The 3x3-neighbourhood lookup primitive.
// All buffers are column-major uint8 logical (0/1), size R*C; index
// (r,c) = r + c*R. For every pixel a 9-bit index is formed: bit k is
// the neighbour at row offset (k%3)-1, col offset (k/3)-1, with bit 4
// the centre pixel. Neighbours outside the image read as 0. The
// output pixel is lut[index].
// ──────────────────────────────────────────────────────────────────
void bwlookup3x3(const keel::u8 *src, keel::u8 *dst,
                 keel::usize R, keel::usize C, const Lut &lut)
{
    if (R == 0 || C == 0)
        return;

    const keel::isize Rs = static_cast<keel::isize>(R);
    const keel::isize Cs = static_cast<keel::isize>(C);

    for (keel::isize c = 0; c < Cs; ++c) {
        for (keel::isize r = 0; r < Rs; ++r) {
            unsigned idx = 0;
            // bit k = neighbour at (r + (k%3)-1, c + (k/3)-1)
            for (keel::isize dc = -1; dc <= 1; ++dc) {
                const keel::isize cc = c + dc;
                if (cc < 0 || cc >= Cs)
                    continue;  // whole column out of bounds → bits stay 0
                const keel::isize colBase = cc * Rs;
                for (keel::isize dr = -1; dr <= 1; ++dr) {
                    const keel::isize rr = r + dr;
                    if (rr < 0 || rr >= Rs)
                        continue;
                    if (src[colBase + rr]) {
                        const unsigned k =
                            static_cast<unsigned>((dr + 1) + (dc + 1) * 3);
                        idx |= (1u << k);
                    }
                }
            }
            dst[c * Rs + r] = lut[idx];
        }
    }
}

// ──────────────────────────────────────────────────────────────────
// Element-wise boolean helpers. Every buffer is treated as a logical
// 0/1 array of length n.
// ──────────────────────────────────────────────────────────────────
void boolAnd(const keel::u8 *a, const keel::u8 *b,
             keel::u8 *out, keel::usize n)
{
    for (keel::usize i = 0; i < n; ++i)
        out[i] = (a[i] && b[i]) ? 1 : 0;
}

void boolNot(const keel::u8 *a, keel::u8 *out, keel::usize n)
{
    for (keel::usize i = 0; i < n; ++i)
        out[i] = a[i] ? 0 : 1;
}

// out = a AND (NOT b)
void boolAndNot(const keel::u8 *a, const keel::u8 *b,
                keel::u8 *out, keel::usize n)
{
    for (keel::usize i = 0; i < n; ++i)
        out[i] = (a[i] && !b[i]) ? 1 : 0;
}

bool boolEqual(const keel::u8 *a, const keel::u8 *b, keel::usize n)
{
    for (keel::usize i = 0; i < n; ++i)
        if ((a[i] != 0) != (b[i] != 0))
            return false;
    return true;
}

bool anySet(const keel::u8 *a, keel::usize n)
{
    for (keel::usize i = 0; i < n; ++i)
        if (a[i])
            return true;
    return false;
}

// ──────────────────────────────────────────────────────────────────
// Composite-operation building blocks.
// ──────────────────────────────────────────────────────────────────

// open = erode then dilate.  bw is overwritten with the result.
void opOpen(keel::u8 *bw, keel::usize R, keel::usize C,
            keel::u8 *tmp)
{
    bwlookup3x3(bw, tmp, R, C, luterode);
    bwlookup3x3(tmp, bw, R, C, lutdilate);
}

// close = dilate then erode.  bw is overwritten with the result.
void opClose(keel::u8 *bw, keel::usize R, keel::usize C,
             keel::u8 *tmp)
{
    bwlookup3x3(bw, tmp, R, C, lutdilate);
    bwlookup3x3(tmp, bw, R, C, luterode);
}

// shrink — four checkerboard sub-iterations.
// For sub-iteration s = 0..3:
//   1. m    = lutshrink(bw)
//   2. cand = bw AND (NOT m)
//   3. overwrite bw with cand only on the active checkerboard
//      sub-field — pixels with (r%2,c%2) == (rOff,cOff), stepping by 2.
// Sub-field offsets in order s = 0,1,2,3: (0,0),(1,1),(1,0),(0,1).
void opShrink(keel::u8 *bw, keel::usize R, keel::usize C,
              keel::u8 *m, keel::u8 *cand)
{
    static constexpr int rOffTab[4] = {0, 1, 1, 0};
    static constexpr int cOffTab[4] = {0, 1, 0, 1};

    for (int s = 0; s < 4; ++s) {
        bwlookup3x3(bw, m, R, C, lutshrink);
        boolAndNot(bw, m, cand, R * C);

        const keel::usize rOff = static_cast<keel::usize>(rOffTab[s]);
        const keel::usize cOff = static_cast<keel::usize>(cOffTab[s]);
        for (keel::usize c = cOff; c < C; c += 2) {
            const keel::usize colBase = c * R;
            for (keel::usize r = rOff; r < R; r += 2)
                bw[colBase + r] = cand[colBase + r];
        }
    }
}

// spur.
//   1. bw = NOT bw
//   2. endPoints = lutspur(bw)
//   3. sub-field 0, offsets (0,0): bw = bw XOR endPoints on that field
//   4. sub-fields 1,2,3, offsets (1,0),(0,1),(1,1): re-apply lutspur to
//      the current bw → e; t = e AND endPoints; XOR t into bw on the
//      sub-field only.
//   5. bw = NOT bw
void opSpur(keel::u8 *bw, keel::usize R, keel::usize C,
            keel::u8 *endPoints, keel::u8 *e, keel::u8 *t)
{
    const keel::usize n = R * C;

    boolNot(bw, bw, n);
    bwlookup3x3(bw, endPoints, R, C, lutspur);

    static constexpr int rOffTab[4] = {0, 1, 0, 1};
    static constexpr int cOffTab[4] = {0, 0, 1, 1};

    for (int s = 0; s < 4; ++s) {
        const keel::u8 *fieldSrc;
        if (s == 0) {
            // XOR endPoints directly on sub-field 0.
            fieldSrc = endPoints;
        } else {
            // Re-evaluate spur end-points on the (now-updated) bw, then
            // mask against the original endPoints set.
            bwlookup3x3(bw, e, R, C, lutspur);
            boolAnd(e, endPoints, t, n);
            fieldSrc = t;
        }

        const keel::usize rOff = static_cast<keel::usize>(rOffTab[s]);
        const keel::usize cOff = static_cast<keel::usize>(cOffTab[s]);
        for (keel::usize c = cOff; c < C; c += 2) {
            const keel::usize colBase = c * R;
            for (keel::usize r = rOff; r < R; r += 2) {
                const keel::usize i = colBase + r;
                bw[i] = ((bw[i] != 0) != (fieldSrc[i] != 0)) ? 1 : 0;
            }
        }
    }

    boolNot(bw, bw, n);
}

// thicken.
void opThicken(keel::u8 *bw, keel::usize R, keel::usize C,
               const keel::ArenaTemp &arena)
{
    const keel::usize n = R * C;

    // ── Step 1: isolated-pixel boost ──────────────────────────────
    {
        keel::Array<keel::u8> iso(n, arena.allocator());
        bwlookup3x3(bw, iso.data(), R, C, lutiso);
        if (anySet(iso.data(), n)) {
            keel::Array<keel::u8> grow(n, arena.allocator());
            keel::Array<keel::u8> oneNbr(n, arena.allocator());
            bwlookup3x3(iso.data(), grow.data(), R, C, lutdilate);
            bwlookup3x3(bw, oneNbr.data(), R, C, lutsingle);
            for (keel::usize i = 0; i < n; ++i)
                if (oneNbr[i] && grow[i])
                    bw[i] = 1;
        }
    }

    // ── Step 2: padded thinning of the complement ─────────────────
    // Build a padded buffer (R+4)×(C+4), all 1, with NOT bw written
    // into the centre R×C region at offset (2,2).
    const keel::usize PR = R + 4;
    const keel::usize PC = C + 4;
    const keel::usize pn = PR * PC;

    keel::Array<keel::u8> c(pn, arena.allocator());
    c.assign(pn, 1);
    for (keel::usize cc = 0; cc < C; ++cc) {
        const keel::usize srcBase = cc * R;
        const keel::usize dstBase = (cc + 2) * PR + 2;
        for (keel::usize rr = 0; rr < R; ++rr)
            c[dstBase + rr] = bw[srcBase + rr] ? 0 : 1;
    }

    keel::Array<keel::u8> c1(pn, arena.allocator());
    keel::Array<keel::u8> c2(pn, arena.allocator());
    keel::Array<keel::u8> d(pn, arena.allocator());

    // c1 = thin1(c); c2 = thin2(c1); d = diag(c2)
    bwlookup3x3(c.data(), c1.data(), PR, PC, lutthin1);
    bwlookup3x3(c1.data(), c2.data(), PR, PC, lutthin2);
    bwlookup3x3(c2.data(), d.data(), PR, PC, lutdiag);

    // c = (c AND (NOT c2) AND d) OR c2
    for (keel::usize i = 0; i < pn; ++i) {
        const bool keep = c[i] && !c2[i] && d[i];
        c[i] = (keep || c2[i]) ? 1 : 0;
    }

    // Force the outer two-pixel border of c back to 1 (all four edges).
    for (keel::usize cc = 0; cc < PC; ++cc) {
        const keel::usize colBase = cc * PR;
        c[colBase + 0] = 1;
        c[colBase + 1] = 1;
        c[colBase + (PR - 2)] = 1;
        c[colBase + (PR - 1)] = 1;
    }
    for (keel::usize rr = 0; rr < PR; ++rr) {
        c[0 * PR + rr] = 1;
        c[1 * PR + rr] = 1;
        c[(PC - 2) * PR + rr] = 1;
        c[(PC - 1) * PR + rr] = 1;
    }

    // bw = NOT (centre R×C region of c)
    for (keel::usize cc = 0; cc < C; ++cc) {
        const keel::usize srcBase = (cc + 2) * PR + 2;
        const keel::usize dstBase = cc * R;
        for (keel::usize rr = 0; rr < R; ++rr)
            bw[dstBase + rr] = c[srcBase + rr] ? 0 : 1;
    }
}

// branchpoints.
void opBranchpoints(keel::u8 *bw, keel::usize R, keel::usize C,
                    const keel::ArenaTemp &arena)
{
    const keel::usize n = R * C;

    keel::Array<keel::u8> Cset(n, arena.allocator());
    keel::Array<keel::u8> Bset(n, arena.allocator());

    // Cset = lutbranchpoints(bw); Bset = lutbackcount4(bw) (small ints).
    bwlookup3x3(bw, Cset.data(), R, C, lutbranchpoints);
    bwlookup3x3(bw, Bset.data(), R, C, lutbackcount4);

    keel::Array<keel::u8> FC(n, arena.allocator());
    keel::Array<keel::u8> Vp(n, arena.allocator());
    keel::Array<keel::u8> Vq(n, arena.allocator());

    for (keel::usize i = 0; i < n; ++i) {
        const unsigned b = Bset[i];
        const bool E = (b == 1);
        const bool notE = !E;
        FC[i] = (notE && Cset[i]) ? 1 : 0;
        Vp[i] = (notE && b == 2) ? 1 : 0;
        Vq[i] = (notE && b > 2) ? 1 : 0;
    }

    keel::Array<keel::u8> Dset(n, arena.allocator());
    bwlookup3x3(Vq.data(), Dset.data(), R, C, lutdilate);

    // M = FC AND Vp AND Dset;  result = FC AND (NOT M)
    for (keel::usize i = 0; i < n; ++i) {
        const bool M = FC[i] && Vp[i] && Dset[i];
        bw[i] = (FC[i] && !M) ? 1 : 0;
    }
}

// ──────────────────────────────────────────────────────────────────
// Single-pass dispatch — runs ONE pass of `op` on `bw` in place.
// Returns false if `op` is not a recognised operation.
// ──────────────────────────────────────────────────────────────────
bool runOnePass(keel::StringRef op, keel::u8 *bw,
                keel::usize R, keel::usize C, keel::u8 *tmp,
                const keel::ArenaTemp &arena)
{
    const keel::usize n = R * C;

    // Single-table operations: one pass of the like-named table.
    const Lut *single = nullptr;
    if (op == "dilate")         single = &lutdilate;
    else if (op == "erode")     single = &luterode;
    else if (op == "bridge")    single = &lutbridge;
    else if (op == "clean")     single = &lutclean;
    else if (op == "diag")      single = &lutdiag;
    else if (op == "endpoints") single = &lutendpoints;
    else if (op == "fatten")    single = &lutfatten;
    else if (op == "fill")      single = &lutfill;
    else if (op == "hbreak")    single = &luthbreak;
    else if (op == "majority")  single = &lutmajority;
    else if (op == "perim4")    single = &lutper4;
    else if (op == "perim8")    single = &lutper8;
    else if (op == "remove")    single = &lutremove;

    if (single != nullptr) {
        bwlookup3x3(bw, tmp, R, C, *single);
        for (keel::usize i = 0; i < n; ++i)
            bw[i] = tmp[i];
        return true;
    }

    // Composite operations.
    if (op == "open") {
        opOpen(bw, R, C, tmp);
        return true;
    }
    if (op == "close") {
        opClose(bw, R, C, tmp);
        return true;
    }
    if (op == "bothat") {
        // bottom-hat: close(bw) AND NOT bw
        keel::Array<keel::u8> closed(keel::ArrayRef<keel::u8>(bw, n), arena.allocator());
        keel::Array<keel::u8> ctmp(n, arena.allocator());
        opClose(closed.data(), R, C, ctmp.data());
        boolAndNot(closed.data(), bw, bw, n);
        return true;
    }
    if (op == "tophat") {
        // top-hat: bw AND NOT open(bw)
        keel::Array<keel::u8> opened(keel::ArrayRef<keel::u8>(bw, n), arena.allocator());
        keel::Array<keel::u8> otmp(n, arena.allocator());
        opOpen(opened.data(), R, C, otmp.data());
        boolAndNot(bw, opened.data(), bw, n);
        return true;
    }
    if (op == "thin") {
        // one pass: thin1 then thin2.
        bwlookup3x3(bw, tmp, R, C, lutthin1);
        bwlookup3x3(tmp, bw, R, C, lutthin2);
        return true;
    }
    if (op == "skeleton" || op == "skel") {
        // one pass: the eight skeleton sub-tables in sequence, each on
        // the output of the previous.
        static const Lut *const skel[8] = {
            &lutskel1, &lutskel2, &lutskel3, &lutskel4,
            &lutskel5, &lutskel6, &lutskel7, &lutskel8};
        for (int k = 0; k < 8; ++k) {
            bwlookup3x3(bw, tmp, R, C, *skel[k]);
            for (keel::usize i = 0; i < n; ++i)
                bw[i] = tmp[i];
        }
        return true;
    }

    if (op == "shrink") {
        keel::Array<keel::u8> m(n, arena.allocator());
        keel::Array<keel::u8> cand(n, arena.allocator());
        opShrink(bw, R, C, m.data(), cand.data());
        return true;
    }

    if (op == "spur") {
        keel::Array<keel::u8> endPoints(n, arena.allocator());
        keel::Array<keel::u8> e(n, arena.allocator());
        keel::Array<keel::u8> t(n, arena.allocator());
        opSpur(bw, R, C, endPoints.data(), e.data(), t.data());
        return true;
    }

    if (op == "thicken") {
        opThicken(bw, R, C, arena);
        return true;
    }

    if (op == "branchpoints") {
        opBranchpoints(bw, R, C, arena);
        return true;
    }

    return false;  // unknown operation
}

void bwmorph_argcheck(const Value &BW, keel::StringRef op)
{
    internal::requireImageClass(BW, "bwmorph", "BW", 1,
                                internal::ImgClasses::AnyNumericReal);
    if (BW.dims().ndim() > 2 || BW.dims().is3D())
        throw numkit::Error("bwmorph: input must be a 2-D binary image",
                            0, 0, "bwmorph", "",
                            "numkit:bwmorph:unsupportedShape");
    static const char *const known[] = {
        "dilate", "erode", "bridge", "clean", "diag", "endpoints",
        "fatten", "fill", "hbreak", "majority", "perim4", "perim8",
        "remove", "open", "close", "bothat", "tophat", "thin",
        "skeleton", "skel", "shrink", "spur", "thicken",
        "branchpoints"};
    bool recognised = false;
    for (const char *k : known) {
        if (op == k) {
            recognised = true;
            break;
        }
    }
    if (!recognised)
        throw numkit::Error(keel::format("bwmorph: unknown operation '{}'", op),
                            0, 0, "bwmorph", "",
                            "numkit:bwmorph:badOp");
}

}  // anonymous namespace

// ──────────────────────────────────────────────────────────────────
// The iteration framework — public entry point.
// ──────────────────────────────────────────────────────────────────
Value bwmorph(const Value &BW, keel::StringRef op, int n,
              keel::Allocator *mr)
{
    bwmorph_argcheck(BW, op);

    const keel::usize R = BW.dims().rows();
    const keel::usize C = BW.dims().cols();
    const keel::usize N = R * C;

    keel::ArenaTemp arena = keel::get_scratch(mr);

    // Pack BW into a column-major logical (0/1) working buffer.
    keel::Array<keel::u8> bw(N, arena.allocator());
    for (keel::usize i = 0; i < N; ++i)
        bw[i] = (BW.elemAsDouble(i) != 0.0) ? 1 : 0;

    auto emitLogical = [&](const keel::u8 *buf) -> Value {
        Value out = Value::matrix(R, C, ValueType::LOGICAL, mr);
        keel::u8 *dst = out.logicalDataMut();
        for (keel::usize i = 0; i < N; ++i)
            dst[i] = buf[i] ? 1 : 0;
        return out;
    };

    // n == 0: no-op — return the packed input unchanged.
    if (n == 0)
        return emitLogical(bw.data());

    // Iterate: repeat one pass until stable or n times. n == -1 is the
    // "until stable" sentinel; a safety cap guards non-converging ops.
    constexpr int kSafetyCap = 10000;
    const bool untilStable = (n < 0);
    const int maxPasses = untilStable ? kSafetyCap : n;

    keel::Array<keel::u8> prev(N, arena.allocator());
    keel::Array<keel::u8> tmp(N, arena.allocator());

    for (int pass = 0; pass < maxPasses; ++pass) {
        for (keel::usize i = 0; i < N; ++i)
            prev[i] = bw[i];

        // A fresh per-pass arena keeps multi-buffer ops from
        // accumulating monotonic-arena allocations across passes.
        keel::ArenaTemp passArena = keel::get_scratch(mr);
        const bool ok =
            runOnePass(op, bw.data(), R, C, tmp.data(), passArena);
        if (!ok)
            throw numkit::Error(keel::format("bwmorph: unknown operation '{}'", op),
                                0, 0, "bwmorph", "",
                                "numkit:bwmorph:badOp");

        if (boolEqual(bw.data(), prev.data(), N))
            break;  // fixed point reached
    }

    return emitLogical(bw.data());
}

// ── bwtraceboundary (Moore-Neighbor boundary tracing) ─────────────
// Reverse-engineered from MATLAB R2025b (closed-source
// images.internal.builtins.bwtraceboundary):
//   8-conn dirs (CW order):  N NE E SE S SW W NW
//   4-conn dirs (CW order):  N E S W
//   Initial back_dir = opposite(fstep) (the notional "previous"
//   pixel direction).
//   Search start = (back_dir + 1) % nd for clockwise tracing
//                  (back_dir - 1 + nd) % nd for counterclockwise.
//   For each step:
//     scan nd neighbours in rotation order; first foreground hit =
//     next boundary pixel. Update back_dir to the OPPOSITE of the
//     move direction.
//   Stop when:
//     * len(B) >= m, or
//     * we just moved to P (loop closed, |B| >= 2), or
//     * no foreground neighbour found (isolated → append P, done).
// Reference: Moore-Neighbor tracing,
//   Pavlidis 1982 §7.5; classic 8-connected variant.
namespace {

// dr/dc tables for 8-conn (CW from N) and 4-conn (CW from N).
const int kDr8[8] = {-1, -1,  0,  1,  1,  1,  0, -1};
const int kDc8[8] = { 0,  1,  1,  1,  0, -1, -1, -1};
const int kDr4[4] = {-1,  0,  1,  0};
const int kDc4[4] = { 0,  1,  0, -1};

int name_to_dir8(keel::StringRef s) {
    if (s == "N")  return 0;
    if (s == "NE") return 1;
    if (s == "E")  return 2;
    if (s == "SE") return 3;
    if (s == "S")  return 4;
    if (s == "SW") return 5;
    if (s == "W")  return 6;
    if (s == "NW") return 7;
    return -1;
}
int name_to_dir4(keel::StringRef s) {
    if (s == "N") return 0;
    if (s == "E") return 1;
    if (s == "S") return 2;
    if (s == "W") return 3;
    return -1;
}

void bwtraceboundary_argcheck(const Value &BW, const Value &P,
                              keel::StringRef fstep, int conn,
                              keel::StringRef dir,
                              long &pr, long &pc, int &fidx)
{
    internal::requireImageClass(BW, "bwtraceboundary", "BW", 1,
                                internal::ImgClasses::AnyNumericReal);
    internal::requireRealDoubleArg(P, "bwtraceboundary", "P", 2);
    if (BW.dims().is3D())
        throw Error("bwtraceboundary: BW must be 2-D",
                    0, 0, "bwtraceboundary", "",
                    "numkit:bwtraceboundary:dim");
    if (conn != 4 && conn != 8)
        throw Error("bwtraceboundary: CONN must be 4 or 8",
                    0, 0, "bwtraceboundary", "",
                    "numkit:bwtraceboundary:conn");
    if (P.numel() != 2)
        throw Error("bwtraceboundary: P must be a 2-element vector",
                    0, 0, "bwtraceboundary", "",
                    "numkit:bwtraceboundary:p");
    if (dir != "clockwise" && dir != "counterclockwise")
        throw Error("bwtraceboundary: DIR must be 'clockwise' or "
                    "'counterclockwise'",
                    0, 0, "bwtraceboundary", "",
                    "numkit:bwtraceboundary:dir");

    const keel::usize M = BW.dims().rows();
    const keel::usize N = BW.dims().cols();
    pr = static_cast<long>(P.elemAsDouble(0)) - 1;
    pc = static_cast<long>(P.elemAsDouble(1)) - 1;
    if (pr < 0 || pc < 0
     || static_cast<keel::usize>(pr) >= M
     || static_cast<keel::usize>(pc) >= N)
        throw Error("bwtraceboundary: P out of bounds",
                    0, 0, "bwtraceboundary", "",
                    "numkit:bwtraceboundary:pbounds");

    const keel::usize k = static_cast<keel::usize>(pc) * M
                        + static_cast<keel::usize>(pr);
    const bool is_fg = BW.isLogical() ? (BW.logicalData()[k] != 0)
                                      : (BW.elemAsDouble(k) != 0.0);
    if (!is_fg)
        throw Error("bwtraceboundary: P must be a foreground pixel",
                    0, 0, "bwtraceboundary", "",
                    "numkit:bwtraceboundary:pfg");

    fidx = (conn == 8) ? name_to_dir8(fstep) : name_to_dir4(fstep);
    if (fidx < 0)
        throw Error("bwtraceboundary: invalid FSTEP for given CONN",
                    0, 0, "bwtraceboundary", "",
                    "numkit:bwtraceboundary:fstep");
}

void makelut_argcheck(int n)
{
    if (n != 2 && n != 3)
        throw Error("makelut: N must be 2 or 3.",
                    0, 0, "makelut", "", "numkit:makelut:badN");
}

} // anonymous

Value bwtraceboundary(const Value &BW, const Value &P,
                      keel::StringRef fstep, int conn,
                      keel::usize m, keel::StringRef dir,
                      keel::Allocator *mr)
{
    long pr = 0, pc = 0;
    int fidx = 0;
    bwtraceboundary_argcheck(BW, P, fstep, conn, dir, pr, pc, fidx);

    const keel::usize M = BW.dims().rows();
    const keel::usize N = BW.dims().cols();

    auto fg_at = [&](long r, long c) -> bool {
        if (r < 0 || c < 0
         || static_cast<keel::usize>(r) >= M
         || static_cast<keel::usize>(c) >= N) return false;
        const keel::usize k = static_cast<keel::usize>(c) * M
                            + static_cast<keel::usize>(r);
        if (BW.isLogical()) return BW.logicalData()[k] != 0;
        return BW.elemAsDouble(k) != 0.0;
    };

    const int nd = conn;
    const int *Dr = (conn == 8) ? kDr8 : kDr4;
    const int *Dc = (conn == 8) ? kDc8 : kDc4;

    const int rot = (dir == "clockwise") ? +1 : -1;

    // Initial back_dir = opposite(fstep). Search starts +rot from back_dir.
    int back_dir = (fidx + nd / 2) % nd;

    keel::Array<long> br(mr), bc(mr);
    br.reserve(64);
    bc.reserve(64);
    br.push_back(pr);
    bc.push_back(pc);

    long cur_r = pr, cur_c = pc;
    const keel::usize cap = (m == 0) ? std::numeric_limits<keel::usize>::max() : m;

    while (br.size() < cap) {
        bool found = false;
        for (int k = 0; k < nd; ++k) {
            int d = (back_dir + rot * (k + 1) + nd * 8) % nd;
            // ((back_dir + rot) + rot*k) mod nd — start at one step past back_dir.
            const long nr = cur_r + Dr[d];
            const long nc = cur_c + Dc[d];
            if (fg_at(nr, nc)) {
                br.push_back(nr);
                bc.push_back(nc);
                cur_r = nr;
                cur_c = nc;
                back_dir = (d + nd / 2) % nd;  // opposite of move direction
                found = true;
                break;
            }
        }
        if (!found) {
            // Isolated pixel: replicate P0 as the closing entry.
            br.push_back(pr);
            bc.push_back(pc);
            break;
        }
        // Loop closed: revisited start.
        if (cur_r == pr && cur_c == pc && br.size() >= 2) break;
    }

    // Pack output (Q × 2) of doubles, 1-based.
    const keel::usize Q = br.size();
    Value out = Value::matrix(Q, 2, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < Q; ++i) {
        out.doubleDataMut()[i]     = static_cast<double>(br[i] + 1);
        out.doubleDataMut()[Q + i] = static_cast<double>(bc[i] + 1);
    }
    return out;
}

// ── makelut — build a bwlookup/applylut table by evaluating `fun` on every
// n×n binary neighbourhood (`bwlookup(BW, makelut(fun, n))`). Core-free: the
// user function handle is invoked through the FnHandle callback (the engine
// adapter in bundle binds it to Engine::callFunctionHandle). The VM-pausable
// path for user-code handles is the separate MakelutCallbackBuiltin in bundle.
Value makelut(FnHandle fun, int n, keel::Allocator *mr)
{
    makelut_argcheck(n);
    const int nq = n * n;
    const keel::usize N = keel::usize{1} << nq;   // 16 or 512
    Value lut = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    double *ld = lut.doubleDataMut();

    // Reusable logical neighbourhood, mutated per table entry.
    Value nh = Value::matrix(static_cast<keel::usize>(n), static_cast<keel::usize>(n),
                             ValueType::LOGICAL, mr);
    keel::u8 *nd = nh.logicalDataMut();
    for (keel::usize k = 0; k < N; ++k) {
        for (int i = 0; i < nq; ++i)
            nd[i] = static_cast<keel::u8>((k >> (nq - 1 - i)) & keel::usize{1});
        std::array<Value, 1> o;
        fun(keel::ArrayRef<Value>(&nh, 1), keel::MutArrayRef<Value>(o.data(), 1), mr);
        const Value &r = o[0];
        if (r.numel() != 1)
            throw Error("makelut: fun must return a scalar",
                        0, 0, "makelut", "", "numkit:makelut:funScalar");
        ld[k] = r.toScalar();
    }
    return lut;
}

} // namespace numkit::image
