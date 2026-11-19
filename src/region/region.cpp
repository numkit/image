// toolboxes/image/src/region/region.cpp

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/math.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <numkit/image/region/region.hpp>
#include <numkit/image/internal/type_gates.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <tuple>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace numkit::image {

namespace {

// Two-pass union-find labelling. Returns labels image (row-major) and
// the maximum label value (= number of components).
std::pair<keel::Array<int>, int>
label_components(keel::ArrayRef<keel::u8> fg, int H, int W, int conn)
{
    keel::Array<int> L((keel::usize)H * (keel::usize)W, 0);
    keel::Array<int> parent;
    parent.reserve(64);
    parent.push_back(0);  // index 0 = "background"

    auto find = [&](int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    auto unite = [&](int x, int y) {
        x = find(x); y = find(y);
        if (x != y) parent[x] = y;
    };

    int next_label = 1;
    // MATLAB-compatible column-major scan order — labels are assigned
    // in the order each new component is first encountered in
    // column-major (col 0 row 0..H-1, col 1 row 0..H-1, ...) order.
    // Already-labelled neighbours at this scan point are: W (r, c-1),
    // N (r-1, c), and for 8-conn also NW (r-1, c-1) and SW (r+1, c-1).
    for (int c = 0; c < W; ++c) {
        for (int r = 0; r < H; ++r) {
            const keel::usize k = (keel::usize)r * (keel::usize)W + (keel::usize)c;
            if (!fg[k]) continue;
            int nb[4] = {0, 0, 0, 0};
            int nc = 0;
            if (c > 0)              { const int v = L[(keel::usize)r * (keel::usize)W + (keel::usize)(c - 1)];     if (v) nb[nc++] = v; }
            if (r > 0)              { const int v = L[(keel::usize)(r - 1) * (keel::usize)W + (keel::usize)c];     if (v) nb[nc++] = v; }
            if (conn == 8) {
                if (c > 0 && r > 0)            { const int v = L[(keel::usize)(r - 1) * (keel::usize)W + (keel::usize)(c - 1)]; if (v) nb[nc++] = v; }
                if (c > 0 && r + 1 < H)        { const int v = L[(keel::usize)(r + 1) * (keel::usize)W + (keel::usize)(c - 1)]; if (v) nb[nc++] = v; }
            }
            if (nc == 0) {
                parent.push_back(next_label);
                L[k] = next_label;
                ++next_label;
            } else {
                int m = nb[0];
                for (int i = 1; i < nc; ++i) if (nb[i] < m) m = nb[i];
                L[k] = m;
                for (int i = 0; i < nc; ++i) unite(nb[i], m);
            }
        }
    }

    // Second pass: collapse + relabel into 1..K in COLUMN-MAJOR
    // scan order (so labels match MATLAB's first-pixel-encountered
    // convention). This nested loop visits every pixel, so it fully
    // relabels L — do NOT add a follow-up pass over L: by this point L
    // already holds remapped labels (1..K), and feeding those back through
    // find()/remap (which are keyed on the ORIGINAL provisional labels)
    // corrupts the result (returned count K stops matching max(L)).
    keel::Array<int> remap(parent.size(), 0);
    int K = 0;
    for (int c = 0; c < W; ++c) {
        for (int r = 0; r < H; ++r) {
            const keel::usize i = (keel::usize)r * (keel::usize)W + (keel::usize)c;
            if (L[i] == 0) continue;
            const int root = find(L[i]);
            if (remap[(keel::usize)root] == 0) remap[(keel::usize)root] = ++K;
            L[i] = remap[(keel::usize)root];
        }
    }
    return {keel::move(L), K};
}

keel::Array<keel::u8> read_bw(const Value &BW) {
    const keel::usize H = BW.dims().rows();
    const keel::usize W = BW.dims().cols();
    keel::Array<keel::u8> out(H * W);
    // Convert column-major Value into row-major flat buffer.
    for (keel::usize r = 0; r < H; ++r)
        for (keel::usize c = 0; c < W; ++c)
            out[r * W + c] = (BW.elemAsDouble(c * H + r) != 0.0) ? 1 : 0;
    return out;
}

void bwconncomp_argcheck(const Value &BW) {
    internal::requireImageClass(BW, "bwconncomp", "BW", 1,
                                internal::ImgClasses::AnyNumericReal);
}

void bwperim_argcheck(const Value &BW) {
    internal::requireImageClassAllowComplex(BW, "bwperim", "BW", 1,
                                internal::ImgClasses::AnyNumericReal);
}

void bwareaopen_argcheck(const Value &BW) {
    internal::requireImageClass(BW, "bwareaopen", "BW", 1,
                                internal::ImgClasses::AnyNumericReal);
}

void bwboundaries_argcheck(const Value &BW) {
    internal::requireImageClass(BW, "bwboundaries", "BW", 1,
                                internal::ImgClasses::AnyNumericReal);
}

} // anonymous

std::tuple<Value, Value>
bwlabel(const Value &BW, int conn, keel::Allocator *mr)
{
    if (conn != 4) conn = 8;
    const int H = (int)BW.dims().rows();
    const int W = (int)BW.dims().cols();
    auto [L, K] = label_components(read_bw(BW), H, W, conn);
    Value out = Value::matrix(H, W, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    for (int c = 0; c < W; ++c)
        for (int r = 0; r < H; ++r)
            od[(keel::usize)c * (keel::usize)H + (keel::usize)r]
                = double(L[(keel::usize)r * (keel::usize)W + (keel::usize)c]);
    return std::make_tuple(std::move(out), Value::scalar(double(K), mr));
}

std::tuple<Value, Value>
bwlabeln(const Value &BW, int conn, keel::Allocator *mr)
{
    const int H = (int)BW.dims().dim(0);
    const int W = (int)BW.dims().dim(1);
    const int D = (int)(BW.numel() / std::max<keel::usize>(1, (keel::usize)H * (keel::usize)W));
    if (conn == 0) conn = (D > 1) ? 26 : 8;
    const int maxDist = (D > 1)
        ? ((conn == 6) ? 1 : (conn == 18 ? 2 : 3))
        : ((conn == 4) ? 1 : 2);
    const keel::usize N = BW.numel();
    keel::Array<int> L(N, 0);
    keel::Array<int> parent(1, 0);
    keel::Array<int> sz(1, 0);
    auto find = [&](int x) {
        while (parent[x] != x) x = parent[x];
        return x;
    };
    int next_label = 1;
    for (int d = 0; d < D; ++d) {
        for (int c = 0; c < W; ++c) {
            for (int r = 0; r < H; ++r) {
                const keel::usize idx = ((keel::usize)d * (keel::usize)W + (keel::usize)c) * (keel::usize)H + (keel::usize)r;
                if (BW.elemAsDouble(idx) == 0.0) continue;
                const int node = next_label++;
                parent.push_back(node);
                sz.push_back(1);
                L[idx] = node;
                for (int dd = -1; dd <= 0; ++dd) {
                    if (d + dd < 0) continue;
                    for (int dc = -1; dc <= (dd == 0 ? 0 : 1); ++dc) {
                        if (c + dc < 0 || c + dc >= W) continue;
                        for (int dr = -1; dr <= ((dd == 0 && dc == 0) ? -1 : 1); ++dr) {
                            if (r + dr < 0 || r + dr >= H) continue;
                            if (keel::abs(dr) + keel::abs(dc) + keel::abs(dd) > maxDist) continue;
                            const keel::usize nidx = ((keel::usize)(d + dd) * (keel::usize)W + (keel::usize)(c + dc)) * (keel::usize)H + (keel::usize)(r + dr);
                            if (L[nidx]) {
                                const int x = find(node);
                                const int y = find(L[nidx]);
                                if (x != y) {
                                    if (sz[(keel::usize)x] > sz[(keel::usize)y]) {
                                        parent[(keel::usize)y] = x;
                                        sz[(keel::usize)x] += sz[(keel::usize)y];
                                    } else {
                                        parent[(keel::usize)x] = y;
                                        sz[(keel::usize)y] += sz[(keel::usize)x];
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    keel::Array<int> remap(parent.size(), 0);
    int K = 0;
    for (keel::usize k = 1; k < parent.size(); ++k) {
        if (parent[k] == (int)k) remap[k] = ++K;
    }
    Value out = (BW.dims().pages() > 1)
        ? Value::matrix3d(BW.dims().rows(), BW.dims().cols(), BW.dims().pages(), ValueType::DOUBLE, mr)
        : Value::matrix(BW.dims().rows(), BW.dims().cols(), ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) {
        if (L[i] == 0) continue;
        od[i] = double(remap[(keel::usize)find(L[i])]);
    }
    return std::make_tuple(std::move(out), Value::scalar(double(K), mr));
}

Value bwconncomp(const Value &BW, int conn, keel::Allocator *mr)
{
    bwconncomp_argcheck(BW);
    if (conn != 4) conn = 8;
    const int H = (int)BW.dims().rows();
    const int W = (int)BW.dims().cols();
    auto [L, K] = label_components(read_bw(BW), H, W, conn);

    // MATLAB convention: bwconncomp returns a 1x1 struct with fields
    //   Connectivity (scalar)
    //   ImageSize    ([H, W])
    //   NumObjects   (scalar)
    //   PixelIdxList (1xK cell of column-vector 1-based linear indices,
    //                 column-major).
    auto cc = Value::structArray(1, 1, mr);
    auto &el = cc.structArrayElem(0);

    el.put(keel::String::from("Connectivity"), Value::scalar(double(conn), mr));

    Value sz = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    sz.doubleDataMut()[0] = double(H);
    sz.doubleDataMut()[1] = double(W);
    el.put(keel::String::from("ImageSize"), std::move(sz));

    el.put(keel::String::from("NumObjects"), Value::scalar(double(K), mr));

    // Build PixelIdxList: 1xK cell of column-vector indices.
    Value pixList = Value::cell(1, static_cast<keel::usize>(K), mr);
    if (K > 0) {
        keel::Array<keel::Array<int>> lists((keel::usize)K);
        for (int c = 0; c < W; ++c)
            for (int r = 0; r < H; ++r) {
                const int lab = L[(keel::usize)r * (keel::usize)W + (keel::usize)c];
                if (lab > 0)
                    lists[(keel::usize)lab - 1].push_back(c * H + r + 1);  // 1-based linear
            }
        for (keel::usize k = 0; k < (keel::usize)K; ++k) {
            const auto &v = lists[k];
            Value col = Value::matrix(v.size(), 1, ValueType::DOUBLE, mr);
            double *cd = col.doubleDataMut();
            for (keel::usize i = 0; i < v.size(); ++i) cd[i] = double(v[i]);
            pixList.cellAt(k) = std::move(col);
        }
    }
    el.put(keel::String::from("PixelIdxList"), std::move(pixList));

    return cc;
}

// ────────────────────────────────────────────────────────────────────
// labelmatrix / cc2bw — CC struct conversions
// ────────────────────────────────────────────────────────────────────
//
// Both functions accept a struct produced by bwconncomp and rasterise
// its PixelIdxList cells back into the image grid. Algorithms are
// straightforward direct transliterations of MATLAB R2025b
// labelmatrix.m / cc2bw.m.

namespace {

// Common helper: extract (H, W) image size and PixelIdxList from a CC
// struct. Validates field presence + types.
struct CCInfo {
    keel::usize H = 0, W = 0;
    keel::usize K = 0;             // NumObjects
    const Value *pixList = nullptr;  // 1×K cell of column-vector indices
};

CCInfo parse_cc(const Value &CC, const char *fn)
{
    CCInfo info;
    if (!CC.isStruct() || CC.numel() != 1)
        throw Error(keel::format("{}: CC must be a 1×1 struct from bwconncomp", fn),
                    0, 0, fn, "", keel::format("numkit:{}:notStruct", fn));
    const auto &el = CC.structArrayElem(0);
    auto need = [&](const char *name) -> const Value & {
        const Value *v = el.get(name);
        if (!v)
            throw Error(keel::format("{}: CC missing field '{}'", fn, name), 0, 0, fn, "",
                        keel::format("numkit:{}:noField", fn));
        return *v;
    };
    const Value &sz = need("ImageSize");
    if (sz.numel() < 2)
        throw Error(keel::format("{}: CC.ImageSize must have >= 2 dims", fn),
                    0, 0, fn, "",
                    keel::format("numkit:{}:sizeDim", fn));
    info.H = static_cast<keel::usize>(sz.elemAsDouble(0));
    info.W = static_cast<keel::usize>(sz.elemAsDouble(1));
    const Value &nob = need("NumObjects");
    info.K = static_cast<keel::usize>(nob.toScalar());
    const Value &pl = need("PixelIdxList");
    info.pixList = &pl;
    return info;
}

CCInfo labelmatrix_argcheck(const Value &CC)
{
    return parse_cc(CC, "labelmatrix");
}

CCInfo cc2bw_argcheck(const Value &CC, const Value &objects_to_keep, keel::Array<char> &keep)
{
    CCInfo info = parse_cc(CC, "cc2bw");
    const keel::usize K = info.K;
    keep.assign(K, 0);
    if (objects_to_keep.isEmpty()) {
        std::fill(keep.begin(), keep.end(), static_cast<char>(1));
    } else {
        const keel::usize N = objects_to_keep.numel();
        if (objects_to_keep.type() == ValueType::LOGICAL) {
            if (N != K)
                throw Error("cc2bw: ObjectsToKeep logical vector length "
                            "must equal NumObjects",
                            0, 0, "cc2bw", "",
                            "numkit:cc2bw:logicalLen");
            for (keel::usize i = 0; i < K; ++i)
                keep[i] = objects_to_keep.elemAsDouble(i) != 0.0 ? 1 : 0;
        } else {
            for (keel::usize i = 0; i < N; ++i) {
                const double v = objects_to_keep.elemAsDouble(i);
                if (!std::isfinite(v) || v <= 0.0 || std::floor(v) != v)
                    throw Error("cc2bw: ObjectsToKeep must be positive "
                                "integers or a logical vector",
                                0, 0, "cc2bw", "",
                                "numkit:cc2bw:badIdx");
                const keel::usize idx = static_cast<keel::usize>(v);
                if (idx > K)
                    throw Error("cc2bw: ObjectsToKeep index exceeds "
                                "NumObjects",
                                0, 0, "cc2bw", "",
                                "numkit:cc2bw:idxRange");
                keep[idx - 1] = 1;
            }
        }
    }
    return info;
}

}  // namespace

Value labelmatrix(const Value &CC, keel::Allocator *mr)
{
    const CCInfo info = labelmatrix_argcheck(CC);
    const keel::usize H = info.H, W = info.W, K = info.K;
    // Choose output class per MATLAB:
    //   K ≤ 255      → uint8
    //   K ≤ 65535    → uint16
    //   K ≤ 2³² - 1  → uint32
    //   else         → double
    ValueType ot;
    if      (K <= 255)        ot = ValueType::UINT8;
    else if (K <= 65535)      ot = ValueType::UINT16;
    else if (K <= 0xFFFFFFFFULL) ot = ValueType::UINT32;
    else                       ot = ValueType::DOUBLE;
    Value L = Value::matrix(H, W, ot, mr);
    // Zero-fill is already done by matrix constructor; defensive memset
    // is unnecessary.
    auto write_label = [&](keel::usize idx0, keel::usize lab) {
        // idx0: 0-based linear index in column-major H×W.
        if (idx0 >= H * W) return;
        switch (ot) {
            case ValueType::UINT8:  L.uint8DataMut()[idx0]
                = static_cast<keel::u8>(lab);  break;
            case ValueType::UINT16: L.uint16DataMut()[idx0]
                = static_cast<keel::u16>(lab); break;
            case ValueType::UINT32: L.uint32DataMut()[idx0]
                = static_cast<keel::u32>(lab); break;
            default:                L.doubleDataMut()[idx0]
                = static_cast<double>(lab);   break;
        }
    };
    for (keel::usize k = 0; k < K; ++k) {
        const Value &cell = info.pixList->cellAt(k);
        const keel::usize N = cell.numel();
        for (keel::usize i = 0; i < N; ++i) {
            const keel::usize idx1 = static_cast<keel::usize>(
                cell.elemAsDouble(i));   // 1-based
            if (idx1 == 0) continue;
            write_label(idx1 - 1, k + 1);
        }
    }
    return L;
}

Value cc2bw(const Value &CC, const Value &objects_to_keep,
            keel::Allocator *mr)
{
    keel::Array<char> keep;
    const CCInfo info = cc2bw_argcheck(CC, objects_to_keep, keep);
    const keel::usize H = info.H, W = info.W, K = info.K;

    Value BW = Value::matrix(H, W, ValueType::LOGICAL, mr);
    keel::u8 *bd = BW.logicalDataMut();
    std::fill(bd, bd + H * W, static_cast<keel::u8>(0));
    for (keel::usize k = 0; k < K; ++k) {
        if (!keep[k]) continue;
        const Value &cell = info.pixList->cellAt(k);
        const keel::usize N = cell.numel();
        for (keel::usize i = 0; i < N; ++i) {
            const keel::usize idx1 = static_cast<keel::usize>(
                cell.elemAsDouble(i));
            if (idx1 == 0 || idx1 > H * W) continue;
            bd[idx1 - 1] = 1;
        }
    }
    return BW;
}

Value bwarea(const Value &BW, keel::Allocator *mr) {
    const keel::usize N = BW.numel();
    double area = 0.0;
    for (keel::usize i = 0; i < N; ++i)
        if (BW.elemAsDouble(i) != 0.0) area += 1.0;
    return Value::scalar(area, mr);
}

Value bwperim(const Value &BW, int conn, keel::Allocator *mr) {
    // Probed R2025b: unlike the rest of the bw* family bwperim takes
    // complex BW and reads the real part.
    bwperim_argcheck(BW);
    if (conn != 4) conn = 8;
    const int H = (int)BW.dims().rows();
    const int W = (int)BW.dims().cols();
    auto fg = read_bw(BW);
    Value out = Value::matrix(H, W, ValueType::LOGICAL, mr);
    if (H == 0 || W == 0) return out;
    keel::u8 *od = out.logicalDataMut();
    for (int c = 0; c < W; ++c)
        for (int r = 0; r < H; ++r) {
            const keel::usize k = (keel::usize)r * (keel::usize)W + (keel::usize)c;
            if (!fg[k]) { od[(keel::usize)c * (keel::usize)H + (keel::usize)r] = 0; continue; }
            bool boundary = false;
            for (int dr = -1; dr <= 1 && !boundary; ++dr)
                for (int dc = -1; dc <= 1 && !boundary; ++dc) {
                    if (dr == 0 && dc == 0) continue;
                    if (conn == 4 && keel::abs(dr) + keel::abs(dc) > 1) continue;
                    const int rr = r + dr, cc = c + dc;
                    if (rr < 0 || rr >= H || cc < 0 || cc >= W) {
                        boundary = true; break;
                    }
                    if (!fg[(keel::usize)rr * (keel::usize)W + (keel::usize)cc]) boundary = true;
                }
            od[(keel::usize)c * (keel::usize)H + (keel::usize)r] = boundary ? 1 : 0;
        }
    return out;
}

Value bwareaopen(const Value &BW, int P, int conn, keel::Allocator *mr)
{
    bwareaopen_argcheck(BW);
    if (conn != 4) conn = 8;
    if (P < 1) P = 1;
    const int H = (int)BW.dims().rows();
    const int W = (int)BW.dims().cols();
    auto [L, K] = label_components(read_bw(BW), H, W, conn);
    keel::Array<int> count((keel::usize)K + 1, 0);
    for (int v : L) if (v > 0) ++count[(keel::usize)v];

    Value out = Value::matrix(H, W, ValueType::LOGICAL, mr);
    if (H == 0 || W == 0) return out;
    keel::u8 *od = out.logicalDataMut();
    for (int c = 0; c < W; ++c)
        for (int r = 0; r < H; ++r) {
            const int lab = L[(keel::usize)r * (keel::usize)W + (keel::usize)c];
            const bool keep = (lab > 0) && (count[(keel::usize)lab] >= P);
            od[(keel::usize)c * (keel::usize)H + (keel::usize)r] = keep ? 1 : 0;
        }
    return out;
}

// ════════════════════════════════════════════════════════════════════
// bwboundaries — Moore-neighbour outer-boundary trace
// ════════════════════════════════════════════════════════════════════
//
// For each connected component (labelled via the same union-find
// pass we already use for bwlabel/bwperim/bwareaopen), find the
// leftmost-topmost foreground pixel of that component and walk the
// outer boundary clockwise. We use Moore-neighbour tracing with
// Jacob's stopping criterion: stop when we revisit the start pixel
// from the same direction we left it.
//
// Boundary points are emitted as (row, col) pairs in MATLAB 1-based
// indexing. Inner holes are NOT traced (matches `'noholes'` mode);
// outer-only is what most scripts want and we'd need a second pass
// to gather hole boundaries from the complement.

Value bwboundaries(const Value &BW, int conn, Value *Lout, int *Nout, keel::Allocator *mr)
{
    bwboundaries_argcheck(BW);
    if (conn != 4) conn = 8;
    const int H = (int)BW.dims().rows();
    const int W = (int)BW.dims().cols();
    auto fg = read_bw(BW);
    auto [L, K] = label_components(fg, H, W, conn);

    // Optional 2nd/3rd outputs: label matrix (objects 1..K, no holes) and
    // object count. The internal L is row-major; the Value is column-major.
    if (Nout) *Nout = K;
    if (Lout) {
        Value lab = Value::matrix(static_cast<keel::usize>(H),
                                  static_cast<keel::usize>(W), ValueType::DOUBLE, mr);
        double *ld = lab.doubleDataMut();
        for (int r = 0; r < H; ++r)
            for (int c = 0; c < W; ++c)
                ld[(keel::usize)c * (keel::usize)H + (keel::usize)r] =
                    static_cast<double>(L[(keel::usize)r * (keel::usize)W + (keel::usize)c]);
        *Lout = std::move(lab);
    }

    Value cellCol = Value::cell(static_cast<keel::usize>(K), 1, mr);
    if (K == 0) return cellCol;

    // 8-direction CW deltas starting at "right" (dir=0):
    //   E, SE, S, SW, W, NW, N, NE.
    static const int dr[8] = { 0, 1, 1, 1, 0, -1, -1, -1 };
    static const int dc[8] = { 1, 1, 0,-1,-1, -1,  0,  1 };
    auto inside = [&](int r, int c) {
        return r >= 0 && r < H && c >= 0 && c < W;
    };
    const auto &Lbl = L;   // plain reference: capturing a structured binding
                           // directly in a lambda is a C++20 extension.
    auto isFg = [&](int r, int c, int targetLabel) {
        if (!inside(r, c)) return false;
        return Lbl[(keel::usize)r * (keel::usize)W + (keel::usize)c] == targetLabel;
    };

    // Start-pixel scan: first encounter of each label in row-major
    // order is its leftmost-topmost pixel. Cache that lookup.
    keel::Array<int> startRow(K + 1, -1);
    keel::Array<int> startCol(K + 1, -1);
    for (int r = 0; r < H && std::count(startRow.begin(),
                                         startRow.end(), -1) > 1; ++r)
        for (int c = 0; c < W; ++c) {
            const int lab = L[(keel::usize)r * (keel::usize)W + (keel::usize)c];
            if (lab > 0 && startRow[lab] < 0) {
                startRow[lab] = r;
                startCol[lab] = c;
            }
        }

    for (int lab = 1; lab <= K; ++lab) {
        const int r0 = startRow[lab];
        const int c0 = startCol[lab];
        keel::Array<int> rows, cols;
        rows.push_back(r0); cols.push_back(c0);

        // Single-pixel object: emit just the start point.
        bool hasNeighbour = false;
        for (int d = 0; d < 8; ++d)
            if (isFg(r0 + dr[d], c0 + dc[d], lab)) {
                hasNeighbour = true; break;
            }
        if (!hasNeighbour) {
            // Cell entry: 1×2 [row col] (1-based).
            Value m = Value::matrix(1, 2, ValueType::DOUBLE, mr);
            double *mb = m.doubleDataMut();
            mb[0] = double(r0 + 1);
            mb[1] = double(c0 + 1);
            cellCol.cellAt(static_cast<keel::usize>(lab - 1)) = m;
            continue;
        }

        // The previous-direction starts at the "from" we'd have used
        // if we entered the start from the west (4-conn) / NW (8-conn).
        int prevDir = (conn == 8) ? 7 : 6;   // NW or N
        int curR = r0, curC = c0;

        // Cap iteration at perimeter ≤ 8·numComponentPixels (very loose).
        const keel::usize cap = static_cast<keel::usize>(H) * static_cast<keel::usize>(W) * 4;
        keel::usize iter = 0;

        while (iter++ < cap) {
            // Look at the 8 neighbours starting one CW step past `prevDir`.
            int found = -1;
            for (int step = 1; step <= 8; ++step) {
                const int d = (prevDir + step) % 8;
                if (conn == 4 && (d & 1)) continue;   // skip diagonals
                const int nr = curR + dr[d];
                const int nc = curC + dc[d];
                if (isFg(nr, nc, lab)) { found = d; break; }
            }
            if (found < 0) break;   // isolated pixel reached
            const int nr = curR + dr[found];
            const int nc = curC + dc[found];
            // The direction we came FROM the new pixel is opposite of
            // the one we just took.
            prevDir = (found + 4) % 8;
            // Jacob's stop: back at start *and* would re-enter via the
            // same direction we left.
            if (nr == r0 && nc == c0) {
                // Emit closing entry.
                rows.push_back(r0);
                cols.push_back(c0);
                break;
            }
            rows.push_back(nr);
            cols.push_back(nc);
            curR = nr;
            curC = nc;
        }

        // Pack into a P×2 [row col] (1-based).
        const keel::usize P = rows.size();
        Value m = Value::matrix(P, 2, ValueType::DOUBLE, mr);
        double *mb = m.doubleDataMut();
        for (keel::usize i = 0; i < P; ++i) {
            mb[i]         = double(rows[i] + 1);
            mb[P + i]     = double(cols[i] + 1);
        }
        cellCol.cellAt(static_cast<keel::usize>(lab - 1)) = m;
    }
    return cellCol;
}

namespace {

// regionprops 'Perimeter' — MATLAB R2025b computePerimeterFromBoundary: trace
// the region's outer 8-connected boundary (Moore-neighbor) and apply the
// Vossepoel–Smeulders weighted estimator
//     perimeter = 0.980·Ne + 1.406·No − 0.091·Nc
// where Ne = axis steps, No = diagonal steps, Nc = orientation changes
// ("corners") around the closed boundary. The value is invariant to the trace
// start/direction, so we start at the region's top-left-most pixel (r0,c0) and
// go clockwise. Outer boundary only (holes not traced — MATLAB likewise warns
// Perimeter is meant for hole-free regions). Single-pixel regions return 0.
double regionPerimeter(keel::ArrayRef<int> L, int H, int W,
                       int lab, int r0, int c0)
{
    static const int Dr8[8] = {-1, -1,  0,  1,  1,  1,  0, -1};
    static const int Dc8[8] = { 0,  1,  1,  1,  0, -1, -1, -1};
    auto fg = [&](int r, int c) {
        return r >= 0 && r < H && c >= 0 && c < W &&
               L[(keel::usize)r * (keel::usize)W + (keel::usize)c] == lab;
    };
    keel::Array<int> br, bc;
    br.reserve(64);
    bc.reserve(64);
    br.push_back(r0);
    bc.push_back(c0);
    int back_dir = 6, cr = r0, cc = c0;   // back_dir = W (entered from background)
    const keel::usize cap = (keel::usize)H * (keel::usize)W * 4 + 8;
    while (br.size() < cap) {
        bool found = false;
        for (int k = 0; k < 8; ++k) {
            const int d = (back_dir + k + 1) % 8;          // clockwise from back_dir+1
            const int nr = cr + Dr8[d], nc = cc + Dc8[d];
            if (fg(nr, nc)) {
                br.push_back(nr);
                bc.push_back(nc);
                cr = nr;
                cc = nc;
                back_dir = (d + 4) % 8;                    // opposite of the move
                found = true;
                break;
            }
        }
        if (!found) break;                                  // isolated pixel
        if (cr == r0 && cc == c0 && br.size() >= 2) break;  // boundary closed
    }
    const keel::usize n = br.size();
    if (n < 3) return 0.0;                                  // ≤1 real step → 0
    const keel::usize steps = n - 1;
    auto stepType = [&](keel::usize i) {                         // 0 horiz, 1 vert, 2 diag
        const int dr = br[i + 1] - br[i], dc = bc[i + 1] - bc[i];
        return dr == 0 ? 0 : (dc == 0 ? 1 : 2);
    };
    long Ne = 0, No = 0, Nc = 0;
    for (keel::usize i = 0; i < steps; ++i) {
        const int t = stepType(i);
        if (t == 2) ++No; else ++Ne;
        if (t != stepType((i + 1) % steps)) ++Nc;           // circular corner count
    }
    return 0.980 * (double)Ne + 1.406 * (double)No - 0.091 * (double)Nc;
}

void regionprops_argcheck(const Value &BW_or_L, keel::ArrayRef<keel::String> propsIn)
{
    internal::requireImageClass(BW_or_L, "regionprops", "BW", 1,
                                internal::ImgClasses::AnyNumericReal);
    static const char *kKnown[] = {
        "all", "basic",
        "area", "centroid", "boundingbox", "perimeter",
        "majoraxislength", "minoraxislength", "eccentricity", "orientation",
        "equivdiameter", "extent", "pixelidxlist", "pixellist",
        "meanintensity", "maxintensity", "minintensity",
        "weightedcentroid", "pixelvalues",
    };
    for (const auto &q : propsIn) {
        keel::String lc = q.clone();
        for (char &ch : lc)
            if (ch >= 'A' && ch <= 'Z') ch = char(ch + 32);
        bool ok = false;
        for (const char *k : kKnown)
            if (lc == k) { ok = true; break; }
        if (!ok)
            throw Error(keel::format("regionprops: property '{}' is not supported in this revision", q),
                        0, 0, "regionprops", "",
                        "numkit:regionprops:badProperty");
    }
}

} // namespace

// ════════════════════════════════════════════════════════════════════
// regionprops — basic descriptors per labelled region
// ════════════════════════════════════════════════════════════════════

Value regionprops(const Value &BW_or_L, keel::ArrayRef<keel::String> propsIn, const Value &intensity, keel::Allocator *mr)
{
    regionprops_argcheck(BW_or_L, propsIn);
    const bool haveI = !intensity.isEmpty() && intensity.numel() > 0;
    // Detect input: integer-valued matrix → treat as label image,
    // else binary → run bwlabel internally.
    const int H = (int)BW_or_L.dims().rows();
    const int W = (int)BW_or_L.dims().cols();
    keel::Array<int> L((keel::usize)H * (keel::usize)W, 0);
    int K = 0;
    bool looksLabelled = false;
    {
        // Quick scan: are any values > 1 with integer fractional part?
        const keel::usize N = (keel::usize)H * (keel::usize)W;
        for (keel::usize i = 0; i < N; ++i) {
            const double v = BW_or_L.elemAsDouble(i);
            if (v > 1.5) { looksLabelled = true; break; }
        }
    }
    if (looksLabelled) {
        // Treat as label image directly; copy in row-major.
        for (int r = 0; r < H; ++r)
            for (int c = 0; c < W; ++c) {
                const int v = (int)std::round(
                    BW_or_L.elemAsDouble((keel::usize)c * (keel::usize)H + (keel::usize)r));
                L[(keel::usize)r * (keel::usize)W + (keel::usize)c] = v;
                if (v > K) K = v;
            }
    } else {
        auto [labels, kk] = label_components(read_bw(BW_or_L), H, W, 8);
        L = std::move(labels);
        K = kk;
    }

    // Property selection.
    auto contains = [&](const char *p) {
        for (const auto &q : propsIn) {
            if (q == p) return true;
            // Case-insensitive ASCII compare.
            if (q.size() != std::strlen(p)) continue;
            bool eq = true;
            for (keel::usize i = 0; i < q.size(); ++i) {
                char a = q[i], b = p[i];
                if (a >= 'A' && a <= 'Z') a = char(a + 32);
                if (b >= 'A' && b <= 'Z') b = char(b + 32);
                if (a != b) { eq = false; break; }
            }
            if (eq) return true;
        }
        return false;
    };

    // MATLAB: an empty property list (or 'basic') returns the BASIC set
    // {Area, Centroid, BoundingBox}; 'all' adds every shape measurement.
    const bool basic   = propsIn.empty() || contains("basic");
    const bool wantAll = contains("all") || contains("All");
    const bool wArea = basic || wantAll || contains("Area");
    const bool wCent = basic || wantAll || contains("Centroid");
    const bool wBbox = basic || wantAll || contains("BoundingBox");
    // Moment / area / bbox based scalar shape descriptors (closed-form,
    // bit-exact vs MATLAB R2025b). The four ellipse fields share one set
    // of second central moments. NOT part of the basic default set.
    const bool wMajor   = wantAll || contains("MajorAxisLength");
    const bool wMinor   = wantAll || contains("MinorAxisLength");
    const bool wEcc     = wantAll || contains("Eccentricity");
    const bool wOrient  = wantAll || contains("Orientation");
    const bool wEquivD  = wantAll || contains("EquivDiameter");
    const bool wExtent  = wantAll || contains("Extent");
    const bool wPerim   = wantAll || contains("Perimeter");
    const bool wEllipse = wMajor || wMinor || wEcc || wOrient;
    // Per-pixel list fields (column-major linear indices / [x y] list).
    const bool wPixIdx  = wantAll || contains("PixelIdxList");
    const bool wPixList = wantAll || contains("PixelList");
    // Intensity measurements (only when a grayscale image is supplied).
    const bool wMeanI  = haveI && (wantAll || contains("MeanIntensity"));
    const bool wMaxI   = haveI && (wantAll || contains("MaxIntensity"));
    const bool wMinI   = haveI && (wantAll || contains("MinIntensity"));
    const bool wWCent  = haveI && (wantAll || contains("WeightedCentroid"));
    const bool wPixVal = haveI && (wantAll || contains("PixelValues"));
    const bool needIntensity = wMeanI || wMaxI || wMinI || wWCent || wPixVal;
    const bool needPixels = wPixIdx || wPixList || needIntensity;

    Value sa = Value::structArray(static_cast<keel::usize>(K), 1, mr);
    if (K == 0) return sa;

    // Accumulators per label.
    const bool needMoments = wEllipse;
    keel::Array<long long> area(K + 1, 0);
    keel::Array<double> sumX(K + 1, 0.0), sumY(static_cast<keel::usize>(K + 1), 0.0);
    // First pixel (row-major scan order = top-left-most) per label, the
    // boundary-trace start for Perimeter.
    keel::Array<int> firstR(K + 1, -1), firstC(static_cast<keel::usize>(K + 1), -1);
    keel::Array<double> sumXX(K + 1, 0.0), sumYY(static_cast<keel::usize>(K + 1), 0.0), sumXY(static_cast<keel::usize>(K + 1), 0.0);
    keel::Array<int> minX(K + 1, INT_MAX), minY(static_cast<keel::usize>(K + 1), INT_MAX);
    keel::Array<int> maxX(K + 1, INT_MIN), maxY(static_cast<keel::usize>(K + 1), INT_MIN);
    // Column-major 1-based linear indices per label (for PixelIdxList /
    // PixelList). Collected in row-major visit order then sorted ascending
    // to match MATLAB's column-major ordering.
    keel::Array<keel::Array<long long>> pixIdx(needPixels ? static_cast<keel::usize>(K + 1) : 0);
    for (int r = 0; r < H; ++r)
        for (int c = 0; c < W; ++c) {
            const int lab = L[(keel::usize)r * (keel::usize)W + (keel::usize)c];
            if (lab <= 0 || lab > K) continue;
            ++area[(keel::usize)lab];
            if (firstR[(keel::usize)lab] < 0) { firstR[(keel::usize)lab] = r; firstC[(keel::usize)lab] = c; }
            sumX[(keel::usize)lab] += double(c);
            sumY[(keel::usize)lab] += double(r);
            if (needPixels)
                pixIdx[(keel::usize)lab].push_back(
                    static_cast<long long>(c) * H + r + 1);   // col-major 1-based
            if (needMoments) {
                sumXX[(keel::usize)lab] += double(c) * double(c);
                sumYY[(keel::usize)lab] += double(r) * double(r);
                sumXY[(keel::usize)lab] += double(c) * double(r);
            }
            if (c < minX[(keel::usize)lab]) minX[(keel::usize)lab] = c;
            if (r < minY[(keel::usize)lab]) minY[(keel::usize)lab] = r;
            if (c > maxX[(keel::usize)lab]) maxX[(keel::usize)lab] = c;
            if (r > maxY[(keel::usize)lab]) maxY[(keel::usize)lab] = r;
        }

    for (int lab = 1; lab <= K; ++lab) {
        auto &el = sa.structArrayElem(static_cast<keel::usize>(lab - 1));
        if (wArea) {
            el.put(keel::String::from("Area"), Value::scalar(double(area[(keel::usize)lab]), mr));
        }
        if (wCent) {
            // Centroid in MATLAB: 1-based (x, y) = (col+1, row+1) means.
            const double a = double(area[(keel::usize)lab]);
            Value cn = Value::matrix(1, 2, ValueType::DOUBLE, mr);
            cn.doubleDataMut()[0] = (a > 0.0) ? sumX[(keel::usize)lab] / a + 1.0 : 0.0;
            cn.doubleDataMut()[1] = (a > 0.0) ? sumY[(keel::usize)lab] / a + 1.0 : 0.0;
            el.put(keel::String::from("Centroid"), cn);
        }
        if (wBbox) {
            // BoundingBox: [xmin-0.5, ymin-0.5, width, height]
            // (MATLAB returns 0.5-aligned floats).
            Value bb = Value::matrix(1, 4, ValueType::DOUBLE, mr);
            bb.doubleDataMut()[0] = double(minX[(keel::usize)lab]) + 0.5;
            bb.doubleDataMut()[1] = double(minY[(keel::usize)lab]) + 0.5;
            bb.doubleDataMut()[2] = double(maxX[(keel::usize)lab] -
                                            minX[(keel::usize)lab] + 1);
            bb.doubleDataMut()[3] = double(maxY[(keel::usize)lab] -
                                            minY[(keel::usize)lab] + 1);
            el.put(keel::String::from("BoundingBox"), bb);
        }

        // ── Moment / area / bbox based scalar shape descriptors ──
        constexpr double kPi = 3.14159265358979323846;
        const double N = double(area[(keel::usize)lab]);
        if (wEquivD)
            el.put(keel::String::from("EquivDiameter"),
                   Value::scalar((N > 0.0) ? std::sqrt(4.0 * N / kPi) : 0.0, mr));
        if (wExtent) {
            const double bw = double(maxX[(keel::usize)lab] - minX[(keel::usize)lab] + 1);
            const double bh = double(maxY[(keel::usize)lab] - minY[(keel::usize)lab] + 1);
            const double bba = bw * bh;
            el.put(keel::String::from("Extent"),
                   Value::scalar((bba > 0.0) ? N / bba : 0.0, mr));
        }
        if (wPerim) {
            const double per = (area[(keel::usize)lab] > 0 && firstR[(keel::usize)lab] >= 0)
                ? regionPerimeter(L, H, W, lab,
                                  firstR[(keel::usize)lab], firstC[(keel::usize)lab])
                : 0.0;
            el.put(keel::String::from("Perimeter"), Value::scalar(per, mr));
        }
        if (wEllipse) {
            // Normalized second central moments with the +1/12 per-pixel
            // variance correction (uniform unit-pixel), matching MATLAB
            // regionprops. y is flipped for the orientation because image
            // rows increase downward.
            double major = 0.0, minor = 0.0, ecc = 0.0, orient = 0.0;
            if (N > 0.0) {
                const double xbar = sumX[(keel::usize)lab] / N;
                const double ybar = sumY[(keel::usize)lab] / N;
                const double uxx = sumXX[(keel::usize)lab] / N - xbar * xbar + 1.0 / 12.0;
                const double uyy = sumYY[(keel::usize)lab] / N - ybar * ybar + 1.0 / 12.0;
                const double uxy = sumXY[(keel::usize)lab] / N - xbar * ybar;
                const double common =
                    std::sqrt((uxx - uyy) * (uxx - uyy) + 4.0 * uxy * uxy);
                major = 2.0 * std::sqrt(2.0) * std::sqrt(uxx + uyy + common);
                minor = 2.0 * std::sqrt(2.0) * std::sqrt(uxx + uyy - common);
                if (major > 0.0) {
                    const double a = major / 2.0, b = minor / 2.0;
                    const double d = a * a - b * b;
                    ecc = 2.0 * std::sqrt(d > 0.0 ? d : 0.0) / major;
                }
                // Orientation in degrees, range [-90, 90].
                if (uxy == 0.0) {
                    orient = (uyy > uxx) ? 90.0 : 0.0;  // axis-aligned
                } else {
                    const double uxyM = -uxy;  // flip y (image rows go down)
                    double num, den;
                    if (uyy > uxx) {
                        num = uyy - uxx +
                              std::sqrt((uyy - uxx) * (uyy - uxx) + 4.0 * uxyM * uxyM);
                        den = 2.0 * uxyM;
                    } else {
                        num = 2.0 * uxyM;
                        den = uxx - uyy +
                              std::sqrt((uxx - uyy) * (uxx - uyy) + 4.0 * uxyM * uxyM);
                    }
                    orient = (180.0 / kPi) * std::atan(num / den);
                }
            }
            if (wMajor)  el.put(keel::String::from("MajorAxisLength"), Value::scalar(major, mr));
            if (wMinor)  el.put(keel::String::from("MinorAxisLength"), Value::scalar(minor, mr));
            if (wEcc)    el.put(keel::String::from("Eccentricity"),    Value::scalar(ecc, mr));
            if (wOrient) el.put(keel::String::from("Orientation"),     Value::scalar(orient, mr));
        }

        // ── Per-pixel list fields ──
        if (needPixels) {
            auto &idxList = pixIdx[(keel::usize)lab];
            std::sort(idxList.begin(), idxList.end());   // MATLAB col-major order
            const keel::usize P = idxList.size();
            if (wPixIdx) {
                Value pil = Value::matrix(P, 1, ValueType::DOUBLE, mr);
                double *pd = pil.doubleDataMut();
                for (keel::usize i = 0; i < P; ++i)
                    pd[i] = static_cast<double>(idxList[i]);
                el.put(keel::String::from("PixelIdxList"), std::move(pil));
            }
            if (wPixList) {
                // [x y] = [col row], 1-based, derived from the sorted index.
                Value plv = Value::matrix(P, 2, ValueType::DOUBLE, mr);
                double *pl = plv.doubleDataMut();
                for (keel::usize i = 0; i < P; ++i) {
                    const long long z = idxList[i] - 1;     // 0-based col-major
                    const long long row0 = z % H;
                    const long long col0 = z / H;
                    pl[i]     = static_cast<double>(col0 + 1);   // x = col
                    pl[P + i] = static_cast<double>(row0 + 1);   // y = row
                }
                el.put(keel::String::from("PixelList"), std::move(plv));
            }
            // ── Intensity measurements (grayscale image supplied) ──
            if (needIntensity) {
                double sumI = 0.0, sumIx = 0.0, sumIy = 0.0;
                double maxI = -std::numeric_limits<double>::infinity();
                double minI =  std::numeric_limits<double>::infinity();
                Value pvv;
                double *pv = nullptr;
                if (wPixVal) {
                    pvv = Value::matrix(P, 1, ValueType::DOUBLE, mr);
                    pv = pvv.doubleDataMut();
                }
                for (keel::usize i = 0; i < P; ++i) {
                    const long long z = idxList[i] - 1;     // col-major 0-based
                    const double iv = intensity.elemAsDouble(static_cast<keel::usize>(z));
                    sumI += iv;
                    if (iv > maxI) maxI = iv;
                    if (iv < minI) minI = iv;
                    if (wWCent) {
                        const long long row0 = z % H, col0 = z / H;
                        sumIx += iv * static_cast<double>(col0 + 1);
                        sumIy += iv * static_cast<double>(row0 + 1);
                    }
                    if (pv) pv[i] = iv;
                }
                if (wMeanI) el.put(keel::String::from("MeanIntensity"),
                                   Value::scalar(P ? sumI / double(P) : 0.0, mr));
                if (wMaxI)  el.put(keel::String::from("MaxIntensity"), Value::scalar(maxI, mr));
                if (wMinI)  el.put(keel::String::from("MinIntensity"), Value::scalar(minI, mr));
                if (wWCent) {
                    Value wc = Value::matrix(1, 2, ValueType::DOUBLE, mr);
                    wc.doubleDataMut()[0] = (sumI != 0.0) ? sumIx / sumI : 0.0;
                    wc.doubleDataMut()[1] = (sumI != 0.0) ? sumIy / sumI : 0.0;
                    el.put(keel::String::from("WeightedCentroid"), std::move(wc));
                }
                if (wPixVal) el.put(keel::String::from("PixelValues"), std::move(pvv));
            }
        }
    }
    return sa;
}

// ════════════════════════════════════════════════════════════════════
// bwdist — Euclidean distance transform
// ════════════════════════════════════════════════════════════════════
//
// Felzenszwalb-Huttenlocher (2004) 1-D parabolic-envelope distance
// transform applied row-wise then column-wise. Exact, O(H·W).
//
// The 1-D DT computes  D(x) = min_y { (x − y)² + f(y) }  in O(N).
// We use a sentinel f = +Inf for background pixels and 0 for
// foreground; running the 1-D DT along each row first squashes f
// horizontally, then the column pass combines that into the full
// 2-D Euclidean DT (proof: separability of squared distance in the
// max-of-min formulation).

namespace {

void dt1D(keel::ArrayRef<double> f, keel::Array<double> &d, keel::usize n)
{
    // v[k] is the index of the k-th parabola in the lower envelope;
    // z[k] is the boundary between parabolas k-1 and k.
    keel::Array<long long> v(n);
    keel::Array<double> z(n + 1);
    long long k = 0;
    v[0] = 0;
    z[0] = -std::numeric_limits<double>::infinity();
    z[1] =  std::numeric_limits<double>::infinity();
    for (long long q = 1; q < static_cast<long long>(n); ++q) {
        // Skip background pixels in f (they contribute Inf — no parabola).
        // Keep them anyway: their +Inf value pushes any candidate boundary
        // away to +Inf. That's fine.
        double s;
        for (;;) {
            const double fq = f[static_cast<keel::usize>(q)];
            const double fv = f[static_cast<keel::usize>(v[k])];
            const double dq = static_cast<double>(q);
            const double dv = static_cast<double>(v[k]);
            if (std::isinf(fq) && std::isinf(fv)) {
                // Both background — boundary is at -Inf, parabola q
                // dominates from k onward.
                s = -std::numeric_limits<double>::infinity();
            } else {
                s = ((fq + dq * dq) - (fv + dv * dv)) / (2.0 * (dq - dv));
            }
            if (s <= z[static_cast<keel::usize>(k)]) {
                --k;
                if (k < 0) break;
            } else break;
        }
        ++k;
        v[static_cast<keel::usize>(k)] = q;
        z[static_cast<keel::usize>(k)] = s;
        z[static_cast<keel::usize>(k) + 1] = std::numeric_limits<double>::infinity();
    }
    long long kk = 0;
    for (long long q = 0; q < static_cast<long long>(n); ++q) {
        while (z[static_cast<keel::usize>(kk) + 1] < static_cast<double>(q)) ++kk;
        const double dx = static_cast<double>(q - v[static_cast<keel::usize>(kk)]);
        d[static_cast<keel::usize>(q)] = dx * dx +
            f[static_cast<keel::usize>(v[static_cast<keel::usize>(kk)])];
    }
}

void bwdist_argcheck(const Value &BW)
{
    internal::requireImageClass(BW, "bwdist", "BW", 1,
                                internal::ImgClasses::AnyNumericReal);
}

void fchcode_argcheck(const Value &bound)
{
    internal::requireImageClass(bound, "fchcode", "bound", 1,
                                internal::ImgClasses::AnyNumericReal);
    if (bound.dims().cols() != 2)
        throw Error("fchcode: bound must be K-by-2",
                    0, 0, "fchcode", "", "numkit:fchcode:size");
}

void bwareafilt_argcheck(const Value &BW)
{
    internal::requireImageClass(BW, "bwareafilt", "BW", 1,
                                internal::ImgClasses::LogicalOnly);
}

void bweuler_argcheck(const Value &BW, int conn)
{
    if (conn != 4 && conn != 8)
        throw Error("bweuler: connectivity must be 4 or 8",
                    0, 0, "bweuler", "", "numkit:bweuler:conn");
    if (BW.dims().is3D())
        throw Error("bweuler: BW must have 2 dimensions",
                    0, 0, "bweuler", "", "numkit:bweuler:dims");
}

} // anonymous

Value bwdist(const Value &BW, keel::Allocator *mr)
{
    bwdist_argcheck(BW);
    const keel::usize H = BW.dims().rows();
    const keel::usize W = BW.dims().cols();
    // Probed R2025b: distances are single for every input class.
    Value out = Value::matrix(H, W, ValueType::SINGLE, mr);
    if (H == 0 || W == 0) return out;

    const double INF = std::numeric_limits<double>::infinity();
    // Pull BW into a flat row-major foreground/background buffer.
    keel::Array<double> f(H * W, 0.0);
    for (keel::usize r = 0; r < H; ++r)
        for (keel::usize c = 0; c < W; ++c) {
            // nk storage is column-major: idx = c*H + r.
            const double v = BW.elemAsDouble(c * H + r);
            f[r * W + c] = (v != 0.0) ? 0.0 : INF;
        }

    // Pass 1: distance transform along each row (horizontal DT).
    keel::Array<double> rowIn(W), rowOut(W);
    keel::Array<double> g(H * W, 0.0);
    for (keel::usize r = 0; r < H; ++r) {
        for (keel::usize c = 0; c < W; ++c) rowIn[c] = f[r * W + c];
        dt1D(rowIn, rowOut, W);
        for (keel::usize c = 0; c < W; ++c) g[r * W + c] = rowOut[c];
    }

    // Pass 2: distance transform along each column (vertical DT)
    // operating on the squared-distance result of pass 1.
    keel::Array<double> colIn(H), colOut(H);
    for (keel::usize c = 0; c < W; ++c) {
        for (keel::usize r = 0; r < H; ++r) colIn[r] = g[r * W + c];
        dt1D(colIn, colOut, H);
        for (keel::usize r = 0; r < H; ++r) {
            const double sq = colOut[r];
            // Take sqrt to get true Euclidean distance; +Inf survives.
            const double d = std::isinf(sq) ? INF : std::sqrt(sq);
            out.singleDataMut()[c * H + r] = static_cast<float>(d);
        }
    }
    return out;
}

Value roicolor(const Value &A, const Value &low_or_v, double high, bool is_range, keel::Allocator *mr)
{
    const auto &d = A.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    const keel::usize P = d.is3D() ? d.pages() : 1;
    Value out = d.is3D()
        ? Value::matrix3d(H, W, P, ValueType::LOGICAL, mr)
        : Value::matrix(H, W, ValueType::LOGICAL, mr);
    const keel::usize N = A.numel();
    if (N == 0) return out;
    keel::u8 *od = out.logicalDataMut();

    if (is_range) {
        const double lo = low_or_v.toScalar();
        const double hi = high;
        for (keel::usize i = 0; i < N; ++i) {
            const double v = A.elemAsDouble(i);
            od[i] = (v >= lo && v <= hi) ? 1u : 0u;
        }
    } else {
        const keel::usize M = low_or_v.numel();
        for (keel::usize i = 0; i < N; ++i) {
            const double v = A.elemAsDouble(i);
            keel::u8 b = 0;
            for (keel::usize k = 0; k < M; ++k)
                if (v == low_or_v.elemAsDouble(k)) { b = 1; break; }
            od[i] = b;
        }
    }
    return out;
}

Value fchcode(const Value &bound, keel::Allocator *mr)
{
    // fchcode is reference-absent (Octave image package API) — the
    // nk-defined contract takes real numeric coordinate matrices.
    fchcode_argcheck(bound);
    keel::usize K = bound.dims().rows();

    // Direction map matching Octave (rows = dr+2, cols = dc+2):
    //   dr=-1: dc=-1→3, dc=0→2, dc=+1→1
    //   dr= 0: dc=-1→4, dc=0→NaN, dc=+1→0
    //   dr=+1: dc=-1→5, dc=0→6, dc=+1→7
    auto dirCode = [](int dr, int dc) -> double {
        if (dr == -1 && dc == -1) return 3.0;
        if (dr == -1 && dc ==  0) return 2.0;
        if (dr == -1 && dc ==  1) return 1.0;
        if (dr ==  0 && dc == -1) return 4.0;
        if (dr ==  0 && dc ==  0) return std::nan("");
        if (dr ==  0 && dc ==  1) return 0.0;
        if (dr ==  1 && dc == -1) return 5.0;
        if (dr ==  1 && dc ==  0) return 6.0;
        if (dr ==  1 && dc ==  1) return 7.0;
        return std::nan("");
    };

    // Read bound rows into vectors r, c.
    keel::Array<double> rs(K), cs(K);
    for (keel::usize k = 0; k < K; ++k) {
        // Column-major: bound[k, 0] = data[0*K + k]; bound[k, 1] = data[1*K + k].
        rs[k] = bound.elemAsDouble(0 * K + k);
        cs[k] = bound.elemAsDouble(1 * K + k);
    }

    // Close the loop if not already closed.
    if (K >= 1 && (rs.front() != rs.back() || cs.front() != cs.back())) {
        rs.push_back(rs.front());
        cs.push_back(cs.front());
        K = rs.size();
    }
    const keel::usize nCodes = (K >= 1) ? K - 1 : 0;

    Value out = Value::structure(mr);

    // x0y0
    Value xy0 = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    if (K >= 1) {
        xy0.doubleDataMut()[0] = rs.empty() ? 0.0 : rs[0];
        xy0.doubleDataMut()[1] = cs.empty() ? 0.0 : cs[0];
    }
    out.field("x0y0") = xy0;

    // fcc — 1×nCodes.
    Value fcc = Value::matrix(1, nCodes, ValueType::DOUBLE, mr);
    double *fd = fcc.doubleDataMut();
    for (keel::usize k = 0; k < nCodes; ++k) {
        const int dr = static_cast<int>(rs[k + 1] - rs[k]);
        const int dc = static_cast<int>(cs[k + 1] - cs[k]);
        fd[k] = dirCode(dr, dc);
    }
    out.field("fcc") = fcc;

    // diff — mod 8 first-difference, cyclic.
    Value diffV = Value::matrix(1, nCodes, ValueType::DOUBLE, mr);
    double *dd = diffV.doubleDataMut();
    for (keel::usize k = 0; k < nCodes; ++k) {
        const double a = fd[k];
        const double b = fd[(k + 1) % nCodes];
        const double d = b - a;
        // mod-8 (positive remainder).
        double m = std::fmod(d, 8.0);
        if (m < 0.0) m += 8.0;
        dd[k] = m;
    }
    out.field("diff") = diffV;

    return out;
}

Value bwareafilt(const Value &BW, double lo, double hi, keel::usize n_keep, bool keep_largest, int conn, keel::Allocator *mr)
{
    bwareafilt_argcheck(BW);
    if (conn != 4) conn = 8;
    const int H = static_cast<int>(BW.dims().rows());
    const int W = static_cast<int>(BW.dims().cols());

    Value out = Value::matrix(static_cast<keel::usize>(H),
                              static_cast<keel::usize>(W),
                              ValueType::LOGICAL, mr);
    if (H == 0 || W == 0) return out;

    auto [L, K] = label_components(read_bw(BW), H, W, conn);
    if (K == 0) return out;

    // Compute area per label (1-based labels in L).
    keel::Array<keel::usize> area(static_cast<keel::usize>(K) + 1, 0);
    for (keel::usize i = 0; i < L.size(); ++i) {
        const int k = L[i];
        if (k > 0) ++area[static_cast<keel::usize>(k)];
    }

    // Decide kept labels.
    keel::Array<keel::u8> keep(static_cast<keel::usize>(K) + 1, 0);
    if (n_keep > 0) {
        // Top-N form. Sort labels [1..K] by area, ties by lower index.
        keel::Array<int> idx;
        idx.reserve(K);
        for (int k = 1; k <= K; ++k) idx.push_back(k);
        if (keep_largest) {
            std::sort(idx.begin(), idx.end(), [&](int a, int b) {
                if (area[(keel::usize)a] != area[(keel::usize)b])
                    return area[(keel::usize)a] > area[(keel::usize)b];
                return a < b;
            });
        } else {
            std::sort(idx.begin(), idx.end(), [&](int a, int b) {
                if (area[(keel::usize)a] != area[(keel::usize)b])
                    return area[(keel::usize)a] < area[(keel::usize)b];
                return a < b;
            });
        }
        const keel::usize take = std::min(n_keep, idx.size());
        for (keel::usize i = 0; i < take; ++i)
            keep[static_cast<keel::usize>(idx[i])] = 1;
    } else {
        // Range form.
        for (int k = 1; k <= K; ++k) {
            const double a = static_cast<double>(area[(keel::usize)k]);
            keep[(keel::usize)k] = (a >= lo && a <= hi) ? 1u : 0u;
        }
    }

    // Write output (col-major). L is row-major.
    keel::u8 *od = out.logicalDataMut();
    for (int c = 0; c < W; ++c)
        for (int r = 0; r < H; ++r) {
            const int k = L[(keel::usize)r * (keel::usize)W + (keel::usize)c];
            od[(keel::usize)c * (keel::usize)H + (keel::usize)r] =
                (k > 0 && keep[(keel::usize)k]) ? 1u : 0u;
        }
    return out;
}

std::tuple<Value, Value>
bwselect(const Value &BW, const Value &cols, const Value &rows, int conn, keel::Allocator *mr)
{
    if (conn != 4) conn = 8;
    const int H = static_cast<int>(BW.dims().rows());
    const int W = static_cast<int>(BW.dims().cols());

    Value out = Value::matrix(static_cast<keel::usize>(H),
                              static_cast<keel::usize>(W),
                              ValueType::LOGICAL, mr);
    Value idx_out = Value::matrix(0, 1, ValueType::DOUBLE, mr);
    if (H == 0 || W == 0)
        return std::make_tuple(std::move(out), std::move(idx_out));

    auto [L, K] = label_components(read_bw(BW), H, W, conn);
    if (K == 0) return std::make_tuple(std::move(out), std::move(idx_out));

    const keel::usize Nseed = std::min(cols.numel(), rows.numel());
    keel::Array<keel::u8> keep(static_cast<keel::usize>(K) + 1, 0);
    for (keel::usize s = 0; s < Nseed; ++s) {
        const int c = static_cast<int>(cols.elemAsDouble(s)) - 1;
        const int r = static_cast<int>(rows.elemAsDouble(s)) - 1;
        if (r < 0 || r >= H || c < 0 || c >= W) continue;
        const int k = L[(keel::usize)r * (keel::usize)W + (keel::usize)c];
        if (k > 0) keep[(keel::usize)k] = 1;
    }

    keel::u8 *od = out.logicalDataMut();
    keel::Array<double> idx_v;
    for (int c = 0; c < W; ++c)
        for (int r = 0; r < H; ++r) {
            const int k = L[(keel::usize)r * (keel::usize)W + (keel::usize)c];
            const bool sel = (k > 0 && keep[(keel::usize)k]);
            od[(keel::usize)c * (keel::usize)H + (keel::usize)r] = sel ? 1u : 0u;
            if (sel) idx_v.push_back(double((keel::usize)c * (keel::usize)H + (keel::usize)r + 1));
        }
    idx_out = Value::matrix(idx_v.size(), 1, ValueType::DOUBLE, mr);
    if (!idx_v.empty()) std::copy(idx_v.begin(), idx_v.end(),
                                  idx_out.doubleDataMut());
    return std::make_tuple(std::move(out), std::move(idx_out));
}

Value bweuler(const Value &BW, int conn, keel::Allocator *mr)
{
    bweuler_argcheck(BW, conn);
    // Pratt bit-quad LUT (Octave-image bweuler.m).
    static constexpr int lut8[16] = {0, 1, 1, 0, 1, 0, -2, -1,
                                     1, -2, 0, -1, 0, -1, -1, 0};
    static constexpr int lut4[16] = {0, 1, 1, 0, 1, 0,  2, -1,
                                     1,  2, 0, -1, 0, -1, -1, 0};
    const int *lut = (conn == 4) ? lut4 : lut8;

    const auto &d = BW.dims();

    const int M = static_cast<int>(d.rows());
    const int N = static_cast<int>(d.cols());

    auto get = [&](int r, int c) -> int {
        if (r < 0 || c < 0 || r >= M || c >= N) return 0;
        return BW.elemAsDouble(static_cast<keel::usize>(c) *
                               static_cast<keel::usize>(M) +
                               static_cast<keel::usize>(r)) != 0.0 ? 1 : 0;
    };

    int sum = 0;
    // Padded image is (M+1)×(N+1) with zero top row/left col, so
    // every 2×2 window slides over (r, c) ∈ [0..M] × [0..N], reading
    // BW[r-1, c-1], BW[r, c-1], BW[r-1, c], BW[r, c] (clamped to 0
    // outside [0, M-1] × [0, N-1]).
    for (int r = 0; r <= M; ++r) {
        for (int c = 0; c <= N; ++c) {
            const int p00 = get(r - 1, c - 1);
            const int p10 = get(r,     c - 1);
            const int p01 = get(r - 1, c);
            const int p11 = get(r,     c);
            const int idx = p00 + 2 * p10 + 4 * p01 + 8 * p11;
            sum += lut[idx];
        }
    }
    return Value::scalar(static_cast<double>(sum) / 4.0, mr);
}

} // namespace numkit::image
