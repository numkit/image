// toolboxes/image/src/filter/filter.cpp

#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/math.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <numkit/image/filter/filter.hpp>
#include <numkit/image/type_convert/type_convert.hpp>
#include <numkit/image/internal/type_gates.hpp>

#include <numkit/builtin/datafun.hpp>
#include <numkit/signal/convolution/convolution.hpp>
#include <numkit/signal/transforms/fft.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/ops/parallel_for.hpp>
#include "filter_detail.hpp"

#include <keel/random.h>
#include <algorithm>
#include <tuple>
#include <cmath>
#include <cstring>
#include <mutex>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numkit::image {

namespace {

// Index folder for non-constant pad modes. Returns the source index
// corresponding to the requested (possibly out-of-range) destination
// index. For Replicate, clamp to [0, n-1]; Symmetric mirror without
// duplicating edge; Circular wrap.
inline keel::i32 fold_index(keel::i32 i, keel::i32 n, PadMode m) {
    if (n <= 0) return 0;
    if (i >= 0 && i < n) return i;
    switch (m) {
        case PadMode::Replicate:
            return std::clamp(i, 0, n - 1);
        case PadMode::Symmetric: {
            // Reflect with edge included: ...a b c | c b a | a b c | c b a...
            const keel::i32 period = 2 * n;
            keel::i32 j = i % period;
            if (j < 0) j += period;
            if (j >= n) j = period - 1 - j;
            return j;
        }
        case PadMode::Circular: {
            keel::i32 j = i % n;
            if (j < 0) j += n;
            return j;
        }
        default:
            return std::clamp(i, 0, n - 1);
    }
}

void padarray_argcheck(const Value &x)
{
    switch (x.type()) {
        case ValueType::DOUBLE:
        case ValueType::SINGLE:
        case ValueType::UINT8:
        case ValueType::UINT16:
        case ValueType::INT16:
        case ValueType::LOGICAL:
            break;
        default:
            if (x.numel() > 0)
                throw Error("padarray: unsupported class", 0, 0, "padarray", "",
                            "numkit:padarray:badtype");
            break;
    }
}

void fspecial_argcheck(keel::StringRef type)
{
    if (type != "average" && type != "gaussian" && type != "laplacian" &&
        type != "log" && type != "sobel" && type != "prewitt" &&
        type != "disk" && type != "unsharp" && type != "motion")
        throw Error(keel::format("fspecial: unknown filter type '{}'", type),
                    0, 0, "fspecial", "", "numkit:fspecial:badtype");
}

} // anonymous

Value padarray(const Value &x, keel::ArrayRef<keel::i32> padsize, PadMode mode, keel::f64 pad_value, keel::StringRef direction, keel::Allocator *mr)
{
    padarray_argcheck(x);
    const auto &d = x.dims();
    keel::i32 H = keel::as<keel::i32>(d.rows());
    keel::i32 W = keel::as<keel::i32>(d.cols());
    keel::i32 P = d.is3D() ? keel::as<keel::i32>(d.pages()) : 1;

    keel::i32 padR = padsize.size() >= 1 ? padsize[0] : 0;
    keel::i32 padC = padsize.size() >= 2 ? padsize[1] : 0;
    keel::i32 padP = padsize.size() >= 3 ? padsize[2] : 0;

    bool pre = (direction == "pre"),
         post = (direction == "post");
    bool both = !pre && !post;

    keel::i32 prR = both ? padR : (pre ? padR : 0);
    keel::i32 poR = both ? padR : (post ? padR : 0);
    keel::i32 prC = both ? padC : (pre ? padC : 0);
    keel::i32 poC = both ? padC : (post ? padC : 0);
    keel::i32 prP = both ? padP : (pre ? padP : 0);
    keel::i32 poP = both ? padP : (post ? padP : 0);

    keel::i32 H2 = H + prR + poR;
    keel::i32 W2 = W + prC + poC;
    keel::i32 P2 = P + prP + poP;

    Value out;
    if (P2 == 1) out = Value::matrix(H2, W2, x.type(), mr);
    else         out = Value::matrix3d(H2, W2, P2, x.type(), mr);
    if (H2 == 0 || W2 == 0 || P2 == 0) return out;

    // Read source value at (r, c, p) folded according to mode; for
    // Constant mode, return pad_value when out of range.
    auto src = [&](keel::i32 r, keel::i32 c, keel::i32 p) -> keel::f64 {
        if (mode == PadMode::Constant) {
            if (r < 0 || r >= H || c < 0 || c >= W || p < 0 || p >= P) return pad_value;
            const keel::usize plane = keel::as<keel::usize>(H) * keel::as<keel::usize>(W);
            const keel::usize idx = (keel::usize)p * plane + (keel::usize)c * H + (keel::usize)r;
            return x.elemAsDouble(idx);
        }
        const keel::i32 rs = fold_index(r, H, mode);
        const keel::i32 cs = fold_index(c, W, mode);
        const keel::i32 ps = (P == 1) ? 0 : fold_index(p, P, mode);
        const keel::usize plane = keel::as<keel::usize>(H) * keel::as<keel::usize>(W);
        const keel::usize idx = (keel::usize)ps * plane + (keel::usize)cs * H + (keel::usize)rs;
        return x.elemAsDouble(idx);
    };

    // Storage helpers per output type.
    auto store = [&](keel::usize outIdx, keel::f64 v) {
        switch (x.type()) {
            case ValueType::DOUBLE: out.doubleDataMut()[outIdx] = v; break;
            case ValueType::SINGLE: out.singleDataMut()[outIdx] = (keel::f32)v; break;
            case ValueType::UINT8: {
                if (v < 0) v = 0; if (v > 255) v = 255;
                out.uint8DataMut()[outIdx] = (keel::u8)std::lround(v);
                break;
            }
            case ValueType::UINT16: {
                if (v < 0) v = 0; if (v > 65535) v = 65535;
                out.uint16DataMut()[outIdx] = (keel::u16)std::lround(v);
                break;
            }
            case ValueType::INT16: {
                if (v < -32768) v = -32768; if (v > 32767) v = 32767;
                out.int16DataMut()[outIdx] = (keel::i16)std::lround(v);
                break;
            }
            case ValueType::LOGICAL: {
                out.logicalDataMut()[outIdx] = v != 0.0 ? 1 : 0;
                break;
            }
            default:
                throw Error("padarray: unsupported class", 0, 0, "padarray", "",
                            "numkit:padarray:badtype");
        }
    };

    const keel::usize plane2 = (keel::usize)H2 * (keel::usize)W2;
    for (keel::i32 p = 0; p < P2; ++p) {
        for (keel::i32 c = 0; c < W2; ++c) {
            for (keel::i32 r = 0; r < H2; ++r) {
                const keel::f64 v = src(r - prR, c - prC, P == 1 ? 0 : p - prP);
                const keel::usize outIdx = (keel::usize)p * plane2 + (keel::usize)c * H2 + (keel::usize)r;
                store(outIdx, v);
            }
        }
    }
    return out;
}

// ════════════════════════════════════════════════════════════════════
// fspecial — kernel factory
// ════════════════════════════════════════════════════════════════════

namespace {

Value mat_double(keel::ArrayRef<keel::f64> v, keel::i32 rows, keel::i32 cols, keel::Allocator *mr) {
    Value out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (!v.empty()) std::memcpy(out.doubleDataMut(), v.data(), v.size() * sizeof(keel::f64));
    return out;
}

Value fspecial_average(keel::i32 rows, keel::i32 cols, keel::Allocator *mr) {
    const keel::f64 w = 1.0 / (keel::f64(rows) * keel::f64(cols));
    keel::Array<keel::f64> data(keel::usize(rows) * keel::usize(cols), w);
    return mat_double(data, rows, cols, mr);
}

Value fspecial_gaussian(keel::i32 rows, keel::i32 cols, keel::f64 sigma, keel::Allocator *mr) {
    if (sigma <= 0.0) throw Error("fspecial: sigma must be positive",
                                  0, 0, "fspecial", "", "numkit:fspecial:sigma");
    const keel::f64 cy = (rows - 1) / 2.0;
    const keel::f64 cx = (cols - 1) / 2.0;
    const keel::f64 inv2s2 = 1.0 / (2.0 * sigma * sigma);
    keel::Array<keel::f64> k(keel::usize(rows) * keel::usize(cols), 0.0);
    keel::f64 sum = 0.0;
    for (keel::i32 c = 0; c < cols; ++c)
        for (keel::i32 r = 0; r < rows; ++r) {
            const keel::f64 dy = r - cy, dx = c - cx;
            const keel::f64 v = std::exp(-(dx * dx + dy * dy) * inv2s2);
            k[keel::usize(c) * keel::usize(rows) + keel::usize(r)] = v;
            sum += v;
        }
    if (sum > 0.0) for (auto &v : k) v /= sum;
    return mat_double(k, rows, cols, mr);
}

Value fspecial_laplacian(keel::f64 alpha, keel::Allocator *mr) {
    // 3×3 Laplacian, controlled by alpha ∈ [0, 1] (default 0.2 in MATLAB).
    if (alpha < 0.0) alpha = 0.0; if (alpha > 1.0) alpha = 1.0;
    const keel::f64 a = alpha;
    const keel::f64 s = 1.0 / (a + 1.0);
    const keel::f64 k[9] = {
        s * a / 4.0,         s * (1.0 - a) / 4.0, s * a / 4.0,         // col 0
        s * (1.0 - a) / 4.0, s * (-1.0),          s * (1.0 - a) / 4.0, // col 1
        s * a / 4.0,         s * (1.0 - a) / 4.0, s * a / 4.0,         // col 2
    };
    return mat_double({k, 9}, 3, 3, mr);
}

Value fspecial_log(keel::i32 rows, keel::i32 cols, keel::f64 sigma, keel::Allocator *mr) {
    // Laplacian-of-Gaussian. Convention from MATLAB: zero-mean, normalized.
    const keel::f64 cy = (rows - 1) / 2.0;
    const keel::f64 cx = (cols - 1) / 2.0;
    const keel::f64 s2 = sigma * sigma;
    keel::Array<keel::f64> g(keel::usize(rows) * keel::usize(cols), 0.0);
    keel::Array<keel::f64> log_k(keel::usize(rows) * keel::usize(cols), 0.0);
    keel::f64 gsum = 0.0;
    for (keel::i32 c = 0; c < cols; ++c)
        for (keel::i32 r = 0; r < rows; ++r) {
            const keel::f64 dy = r - cy, dx = c - cx;
            const keel::f64 rsq = dx * dx + dy * dy;
            const keel::f64 e = std::exp(-rsq / (2.0 * s2));
            g[keel::usize(c) * keel::usize(rows) + keel::usize(r)] = e;
            gsum += e;
        }
    if (gsum > 0.0) for (auto &v : g) v /= gsum;
    for (keel::i32 c = 0; c < cols; ++c)
        for (keel::i32 r = 0; r < rows; ++r) {
            const keel::f64 dy = r - cy, dx = c - cx;
            const keel::f64 rsq = dx * dx + dy * dy;
            const keel::usize i = keel::usize(c) * keel::usize(rows) + keel::usize(r);
            log_k[i] = (rsq - 2.0 * s2) / (s2 * s2) * g[i];
        }
    // Mean subtraction so kernel sums to zero.
    keel::f64 mean = 0.0;
    for (auto v : log_k) mean += v;
    mean /= keel::f64(log_k.size());
    for (auto &v : log_k) v -= mean;
    return mat_double(log_k, rows, cols, mr);
}

Value fspecial_sobel(keel::Allocator *mr) {
    // MATLAB convention: [1 2 1; 0 0 0; -1 -2 -1]. Column-major storage:
    // col0 = [1, 0, -1], col1 = [2, 0, -2], col2 = [1, 0, -1].
    const keel::f64 k[9] = {1, 0, -1,   2, 0, -2,   1, 0, -1};
    return mat_double({k, 9}, 3, 3, mr);
}

Value fspecial_prewitt(keel::Allocator *mr) {
    // [1 1 1; 0 0 0; -1 -1 -1] column-major.
    const keel::f64 k[9] = {1, 0, -1,   1, 0, -1,   1, 0, -1};
    return mat_double({k, 9}, 3, 3, mr);
}

Value fspecial_disk(keel::f64 rad, keel::Allocator *mr) {
    if (rad <= 0.0) throw Error("fspecial: radius must be positive",
                                0, 0, "fspecial", "", "numkit:fspecial:radius");
    // MATLAB R2025b fspecial('disk',rad): each cell's weight is the EXACT
    // area of the unit cell that lies inside the disk of radius rad (the
    // sub-pixel area-coverage integral), not a linear-taper approximation.
    const keel::i32 crad = keel::i32(std::ceil(rad - 0.5));
    const keel::i32 side = 2 * crad + 1;
    const keel::f64 r2 = rad * rad;
    keel::Array<keel::f64> sg(keel::usize(side) * keel::usize(side), 0.0);
    auto clamp1 = [](keel::f64 v) { return v < -1.0 ? -1.0 : (v > 1.0 ? 1.0 : v); };
    auto sq = [](keel::f64 v) { return v * v; };
    for (keel::i32 ii = 0; ii < side; ++ii)
        for (keel::i32 jj = 0; jj < side; ++jj) {
            const keel::f64 y = keel::f64(-crad + ii);   // row coordinate
            const keel::f64 x = keel::f64(-crad + jj);   // col coordinate
            const keel::f64 ax = keel::abs(x), ay = keel::abs(y);
            const keel::f64 maxxy = std::max(ax, ay), minxy = std::min(ax, ay);
            const keel::f64 m1 = (r2 < sq(maxxy + 0.5) + sq(minxy - 0.5))
                ? (minxy - 0.5)
                : std::sqrt(std::max(0.0, r2 - sq(maxxy + 0.5)));
            const keel::f64 m2 = (r2 > sq(maxxy - 0.5) + sq(minxy + 0.5))
                ? (minxy + 0.5)
                : std::sqrt(std::max(0.0, r2 - sq(maxxy - 0.5)));
            const bool mask =
                ((r2 < sq(maxxy + 0.5) + sq(minxy + 0.5)) &&
                 (r2 > sq(maxxy - 0.5) + sq(minxy - 0.5))) ||
                ((minxy == 0.0) && (maxxy - 0.5 < rad) && (maxxy + 0.5 >= rad));
            keel::f64 s = 0.0;
            if (mask) {
                const keel::f64 a1 = std::asin(clamp1(m1 / rad));
                const keel::f64 a2 = std::asin(clamp1(m2 / rad));
                s = r2 * (0.5 * (a2 - a1) +
                          0.25 * (std::sin(2.0 * a2) - std::sin(2.0 * a1)))
                    - (maxxy - 0.5) * (m2 - m1) + (m1 - minxy + 0.5);
            }
            if (sq(maxxy + 0.5) + sq(minxy + 0.5) < r2) s += 1.0;
            sg[keel::usize(jj) * keel::usize(side) + keel::usize(ii)] = s;
        }
    auto at = [&](keel::i32 i, keel::i32 j) -> keel::f64 & {
        return sg[keel::usize(j) * keel::usize(side) + keel::usize(i)];   // (row i, col j)
    };
    at(crad, crad) = std::min(M_PI * r2, M_PI / 2.0);
    if (crad > 0 && rad > crad - 0.5 && r2 < sq(crad - 0.5) + 0.25) {
        const keel::f64 mm = std::sqrt(std::max(0.0, r2 - sq(crad - 0.5)));
        const keel::f64 mmn = mm / rad;
        const keel::f64 sg0 = 2.0 * (r2 * (0.5 * std::asin(clamp1(mmn)) +
                          0.25 * std::sin(2.0 * std::asin(clamp1(mmn))))
                          - mm * (crad - 0.5));
        at(2 * crad, crad) = sg0; at(crad, 2 * crad) = sg0;
        at(crad, 0) = sg0;        at(0, crad) = sg0;
        at(2 * crad - 1, crad) -= sg0; at(crad, 2 * crad - 1) -= sg0;
        at(crad, 1) -= sg0;            at(1, crad) -= sg0;
    }
    at(crad, crad) = std::min(at(crad, crad), 1.0);
    keel::f64 sum = 0.0;
    for (keel::f64 v : sg) sum += v;
    if (sum > 0.0) for (auto &v : sg) v /= sum;
    return mat_double(sg, side, side, mr);
}

Value fspecial_motion(keel::f64 len_in, keel::f64 theta, keel::Allocator *mr) {
    // MATLAB R2025b fspecial('motion',len,theta): a motion-blur PSF — a line
    // of `len` pixels at `theta` degrees (counter-clockwise from the
    // horizontal), anti-aliased by bilinear interpolation. Ported verbatim
    // from fspecial.m's 'motion' case (validated isequal across angles
    // 0/30/45/90/120/135 and lengths 5..15 before this port).
    const keel::f64 EPS = 2.2204460492503131e-16;   // eps
    const keel::f64 linewdt = 1.0;
    const keel::f64 len  = std::max(1.0, len_in);
    const keel::f64 half = (len - 1.0) / 2.0;
    keel::f64 tmod = std::fmod(theta, 180.0);
    if (tmod < 0.0) tmod += 180.0;
    const keel::f64 phi    = tmod / 180.0 * M_PI;
    const keel::f64 cosphi = std::cos(phi), sinphi = std::sin(phi);
    const keel::f64 xsign  = (cosphi > 0.0) ? 1.0 : (cosphi < 0.0 ? -1.0 : 0.0);

    const keel::i64 sx = (keel::i64)std::trunc(half * cosphi + linewdt * xsign - len * EPS);
    const keel::i64 sy = (keel::i64)std::trunc(half * sinphi + linewdt - len * EPS);
    const keel::i32 ncol = (keel::i32)std::labs(sx) + 1;
    const keel::i32 nrow = (keel::i32)sy + 1;

    // Half-matrix of (signed) perpendicular distances to the rotated line.
    keel::Array<keel::f64> D((keel::usize)nrow * (keel::usize)ncol, 0.0);
    for (keel::i32 r = 0; r < nrow; ++r)
        for (keel::i32 c = 0; c < ncol; ++c) {
            const keel::f64 x = (keel::f64)c * xsign;
            const keel::f64 y = (keel::f64)r;
            keel::f64 d = y * cosphi - x * sinphi;
            const keel::f64 rad = std::hypot(x, y);
            if (rad >= half && keel::abs(d) <= linewdt) {
                // Pixel past the line's end: distance to the end-point.
                const keel::f64 x2 = (cosphi != 0.0)
                    ? half - keel::abs((x + d * sinphi) / cosphi) : 0.0;
                d = std::sqrt(d * d + x2 * x2);
            }
            keel::f64 w = linewdt + EPS - keel::abs(d);
            if (w < 0.0) w = 0.0;
            D[(keel::usize)r * (keel::usize)ncol + (keel::usize)c] = w;
        }

    // Unfold the half-matrix to the full PSF via 180-degree symmetry.
    const keel::i32 H = 2 * nrow - 1, W = 2 * ncol - 1;
    keel::Array<keel::f64> Hm((keel::usize)H * (keel::usize)W, 0.0);
    auto Dat = [&](keel::i32 r, keel::i32 c) { return D[(keel::usize)r * (keel::usize)ncol + (keel::usize)c]; };
    for (keel::i32 r = 0; r < nrow; ++r)                       // rot90(D,2) top-left
        for (keel::i32 c = 0; c < ncol; ++c)
            Hm[(keel::usize)r * (keel::usize)W + (keel::usize)c] = Dat(nrow - 1 - r, ncol - 1 - c);
    for (keel::i32 r = 0; r < nrow; ++r)                       // D bottom-right
        for (keel::i32 c = 0; c < ncol; ++c)
            Hm[(keel::usize)(nrow - 1 + r) * (keel::usize)W + (keel::usize)(ncol - 1 + c)] = Dat(r, c);

    keel::f64 s = 0.0;
    for (keel::f64 v : Hm) s += v;
    s += EPS * len * len;
    for (auto &v : Hm) v /= s;

    // Column-major output; flipud when cosphi > 0 (MATLAB convention).
    keel::Array<keel::f64> out((keel::usize)H * (keel::usize)W);
    for (keel::i32 c = 0; c < W; ++c)
        for (keel::i32 r = 0; r < H; ++r) {
            const keel::i32 rr = (cosphi > 0.0) ? (H - 1 - r) : r;
            out[(keel::usize)c * (keel::usize)H + (keel::usize)r] = Hm[(keel::usize)rr * (keel::usize)W + (keel::usize)c];
        }
    return mat_double(out, H, W, mr);
}

Value fspecial_unsharp(keel::f64 alpha, keel::Allocator *mr) {
    // MATLAB R2025b fspecial('unsharp',alpha): the 3x3 unsharp-contrast
    // enhancement filter h = [-a a-1 -a; a-1 a+5 a-1; -a a-1 -a]/(a+1),
    // with alpha in [0,1] (default 0.2). Sums to 1 (a sharpening Laplacian:
    // corners -a, edge-mids a-1, centre a+5).
    if (!(alpha >= 0.0 && alpha <= 1.0))
        throw Error("fspecial: unsharp alpha must be in [0, 1]",
                    0, 0, "fspecial", "", "numkit:fspecial:alpha");
    const keel::f64 a = alpha;
    const keel::f64 d = a + 1.0;
    keel::f64 k[9] = {
        -a,    a - 1, -a,       // col 0 (symmetric, so col == row order)
        a - 1, a + 5, a - 1,    // col 1
        -a,    a - 1, -a        // col 2
    };
    for (keel::f64 &v : k) v /= d;
    return mat_double({k, 9}, 3, 3, mr);
}

} // anonymous

Value fspecial(keel::StringRef type, keel::ArrayRef<keel::f64> params, keel::Allocator *mr)
{
    fspecial_argcheck(type);
    if (type == "average") {
        keel::i32 rows = params.size() >= 1 ? keel::i32(params[0]) : 3;
        keel::i32 cols = params.size() >= 2 ? keel::i32(params[1]) : rows;
        return fspecial_average(rows, cols, mr);
    }
    if (type == "gaussian") {
        keel::i32 rows = params.size() >= 1 ? keel::i32(params[0]) : 3;
        keel::i32 cols = params.size() >= 2 ? keel::i32(params[1]) : rows;
        keel::f64 sigma = params.size() >= 3 ? params[2] : 0.5;
        return fspecial_gaussian(rows, cols, sigma, mr);
    }
    if (type == "laplacian") {
        keel::f64 alpha = params.size() >= 1 ? params[0] : 0.2;
        return fspecial_laplacian(alpha, mr);
    }
    if (type == "log") {
        keel::i32 rows = params.size() >= 1 ? keel::i32(params[0]) : 5;
        keel::i32 cols = params.size() >= 2 ? keel::i32(params[1]) : rows;
        keel::f64 sigma = params.size() >= 3 ? params[2] : 0.5;
        return fspecial_log(rows, cols, sigma, mr);
    }
    if (type == "sobel")   return fspecial_sobel(mr);
    if (type == "prewitt") return fspecial_prewitt(mr);
    if (type == "disk") {
        keel::f64 radius = params.size() >= 1 ? params[0] : 5.0;
        return fspecial_disk(radius, mr);
    }
    if (type == "unsharp") {
        keel::f64 alpha = params.size() >= 1 ? params[0] : 0.2;
        return fspecial_unsharp(alpha, mr);
    }
    if (type == "motion") {
        keel::f64 len   = params.size() >= 1 ? params[0] : 9.0;
        keel::f64 theta = params.size() >= 2 ? params[1] : 0.0;
        return fspecial_motion(len, theta, mr);
    }
    throw Error(keel::format("fspecial: unknown filter type '{}'", type),
                0, 0, "fspecial", "", "numkit:fspecial:badtype");
}

// ════════════════════════════════════════════════════════════════════
// imfilter / imgaussfilt / imboxfilt / medfilt2
// ════════════════════════════════════════════════════════════════════

namespace {

inline keel::f64 sample_padded(const Value &I, keel::i32 r, keel::i32 c, keel::i32 H, keel::i32 W,
                            PadMode mode, keel::f64 pv) {
    if (mode == PadMode::Constant) {
        if (r < 0 || r >= H || c < 0 || c >= W) return pv;
    } else {
        r = fold_index(r, H, mode);
        c = fold_index(c, W, mode);
    }
    return I.elemAsDouble((keel::usize)c * (keel::usize)H + (keel::usize)r);
}

inline void store_classed(Value &out, keel::usize i, keel::f64 v, ValueType t) {
    switch (t) {
        case ValueType::DOUBLE: out.doubleDataMut()[i] = v; break;
        case ValueType::SINGLE: out.singleDataMut()[i] = (keel::f32)v; break;
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
        default:
            throw Error("imfilter: unsupported class", 0, 0, "imfilter", "",
                        "numkit:imfilter:badtype");
    }
}

void imfilter_argcheck(const Value &I, const Value &h)
{
    internal::requireImageClassAllowComplex(I, "imfilter", "I", 1,
                                internal::ImgClasses::FloatIntLog);
    internal::requireImageClassAllowComplex(h, "imfilter", "h", 2,
                                internal::ImgClasses::DoubleOnly);
}

void imboxfilt_argcheck(const Value &I)
{
    internal::requireImageClass(I, "imboxfilt", "I", 1,
                                internal::ImgClasses::FloatInt);
}

void integralBoxFilter_argcheck(const Value &I, keel::i32 fH, keel::i32 fW)
{
    if (fH <= 0 || fW <= 0)
        throw Error("integralBoxFilter: filterSize must be positive",
                    0, 0, "integralBoxFilter", "", "numkit:integralBoxFilter:badSize");
    if ((fH & 1) == 0 || (fW & 1) == 0)
        throw Error("integralBoxFilter: filterSize must be odd",
                    0, 0, "integralBoxFilter", "", "numkit:integralBoxFilter:notOdd");

    const auto &d = I.dims();
    if (d.ndim() > 3)
        throw Error("integralBoxFilter: input must be 2-D or 3-D integral image",
                    0, 0, "integralBoxFilter", "", "numkit:integralBoxFilter:rank");

    const keel::usize H1 = d.rows();
    const keel::usize W1 = d.cols();
    if (H1 < 1 || W1 < 1)
        throw Error("integralBoxFilter: integral image must be at least (1+1) × (1+1)",
                    0, 0, "integralBoxFilter", "", "numkit:integralBoxFilter:tiny");
    const keel::usize H = H1 - 1;
    const keel::usize W = W1 - 1;
    if (keel::as<keel::usize>(fH) > H || keel::as<keel::usize>(fW) > W)
        throw Error("integralBoxFilter: filter size too large for integral image",
                    0, 0, "integralBoxFilter", "", "numkit:integralBoxFilter:tooLarge");
}

void integralBoxFilter3_argcheck(const Value &I, keel::i32 fH, keel::i32 fW, keel::i32 fP)
{
    if (fH <= 0 || fW <= 0 || fP <= 0)
        throw Error("integralBoxFilter3: filterSize must be positive",
                    0, 0, "integralBoxFilter3", "", "numkit:integralBoxFilter3:badSize");
    if ((fH & 1) == 0 || (fW & 1) == 0 || (fP & 1) == 0)
        throw Error("integralBoxFilter3: Expected filterSize to be odd.",
                    0, 0, "integralBoxFilter3", "", "numkit:integralBoxFilter3:notOdd");

    const auto &d = I.dims();
    if (d.ndim() > 3)
        throw Error("integralBoxFilter3: input must be a 3-D integral volume",
                    0, 0, "integralBoxFilter3", "", "numkit:integralBoxFilter3:rank");
    const keel::usize H1 = d.rows();
    const keel::usize W1 = d.cols();
    const keel::usize D1 = d.is3D() ? d.pages() : 1;
    if (H1 < 2 || W1 < 2 || D1 < 2)
        throw Error("integralBoxFilter3: integral volume must be at least (1+1)^3",
                    0, 0, "integralBoxFilter3", "", "numkit:integralBoxFilter3:tiny");
    const keel::usize H = H1 - 1, W = W1 - 1, D = D1 - 1;
    if (keel::as<keel::usize>(fH) > H || keel::as<keel::usize>(fW) > W
        || keel::as<keel::usize>(fP) > D)
        throw Error("integralBoxFilter3: Filter size is too large for integral image.",
                    0, 0, "integralBoxFilter3", "", "numkit:integralBoxFilter3:tooLarge");
}

void modefilt_argcheck(const Value &A, keel::i32 fH, keel::i32 fW, keel::i32 fD,
                       keel::StringRef padopt, keel::i32 &padMode_out)
{
    const bool is3D = (fD > 0);
    if (fH <= 0 || fW <= 0 || (is3D && fD <= 0))
        throw Error("modefilt: filter size must be positive",
                    0, 0, "modefilt", "", "numkit:modefilt:badSize");
    if ((fH & 1) == 0 || (fW & 1) == 0 || (is3D && (fD & 1) == 0))
        throw Error("modefilt: filter size must be odd",
                    0, 0, "modefilt", "", "numkit:modefilt:notOdd");

    if (padopt.empty() || padopt == "symmetric") padMode_out = 0;
    else if (padopt == "replicate")              padMode_out = 1;
    else if (padopt == "zeros" || padopt == "zero") padMode_out = 2;
    else throw Error("modefilt: padopt must be 'symmetric', 'replicate', or 'zeros'",
                     0, 0, "modefilt", "", "numkit:modefilt:badPad");

    const auto &d = A.dims();
    if (d.ndim() > 3 || (!is3D && d.ndim() > 2))
        throw Error("modefilt: input rank does not match filter rank",
                    0, 0, "modefilt", "", "numkit:modefilt:rank");
}

void medfilt2_argcheck(const Value &I)
{
    internal::requireImageClass(I, "medfilt2", "I", 1,
                                internal::ImgClasses::FloatIntLog);
}

void imsharpen_argcheck(const Value &I, keel::f64 threshold)
{
    internal::requireImageClass(I, "imsharpen", "I", 1,
                                internal::ImgClasses::FloatInt);
    if (!(threshold >= 0.0 && threshold <= 1.0))
        throw Error("imsharpen: threshold must be in [0, 1]",
                    0, 0, "imsharpen", "", "numkit:imsharpen:threshold");
}

} // anonymous

Value imfilter(const Value &I, const Value &h, PadMode boundary, keel::f64 pad_value, bool full, bool flip_kernel, keel::Allocator *mr)
{
    imfilter_argcheck(I, h);
    if (I.isComplex() || h.isComplex()) {
        auto extractPart = [&](const Value &V, bool imagPart) -> Value {
            const auto &d = V.dims();
            Value out = Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
            keel::f64 *od = out.doubleDataMut();
            const keel::usize N = V.numel();
            if (V.isComplex()) {
                const auto *cd = V.complexData();
                for (keel::usize i = 0; i < N; ++i)
                    od[i] = imagPart ? cd[i].imag() : cd[i].real();
            } else if (imagPart) {
                for (keel::usize i = 0; i < N; ++i) od[i] = 0.0;
            } else {
                for (keel::usize i = 0; i < N; ++i) od[i] = V.elemAsDouble(i);
            }
            return out;
        };
        Value Ir = extractPart(I, false);
        Value Ii = extractPart(I, true);
        Value hr = extractPart(h, false);
        Value hi = extractPart(h, true);
        Value rr = imfilter(Ir, hr, boundary, pad_value, full, flip_kernel, mr);
        const keel::usize outH = rr.dims().rows();
        const keel::usize outW = rr.dims().cols();
        Value ri = h.isComplex() ? imfilter(Ir, hi, boundary, pad_value, full, flip_kernel, mr)
                                 : Value::matrix(outH, outW, ValueType::DOUBLE, mr);
        Value ir = I.isComplex() ? imfilter(Ii, hr, boundary, 0.0, full, flip_kernel, mr)
                                 : Value::matrix(outH, outW, ValueType::DOUBLE, mr);
        Value ii = (I.isComplex() && h.isComplex())
            ? imfilter(Ii, hi, boundary, 0.0, full, flip_kernel, mr)
            : Value::matrix(outH, outW, ValueType::DOUBLE, mr);
        Value out = Value::matrix(outH, outW, ValueType::COMPLEX, mr);
        keel::c64 *od = out.complexDataMut();
        const keel::f64 *rrd = rr.doubleData();
        const keel::f64 *rid = ri.doubleData();
        const keel::f64 *ird = ir.doubleData();
        const keel::f64 *iid = ii.doubleData();
        for (keel::usize i = 0; i < outH * outW; ++i)
            od[i] = keel::c64(rrd[i] - iid[i], rid[i] + ird[i]);
        return out;
    }
    // Logical input → logical output: the reference filters the double
    // values and re-binarises (probed R2025b: imfilter(logical, h) is
    // logical). Compute on a double copy, cast back.
    if (I.type() == ValueType::LOGICAL)
        return internal::castResultToClass(
            imfilter(internal::coerceToDoubleImage(I, mr), h, boundary,
                     pad_value, full, flip_kernel, mr),
            ValueType::LOGICAL, mr);
    const keel::i32 H = (keel::i32)I.dims().rows();
    const keel::i32 W = (keel::i32)I.dims().cols();
    const keel::i32 kH = (keel::i32)h.dims().rows();
    const keel::i32 kW = (keel::i32)h.dims().cols();
    // 'same'-mode anchor: MATLAB centres the kernel at 1-based index
    // floor((K+1)/2), i.e. 0-based offset floor((K-1)/2). For ODD K this
    // equals K/2 (unchanged); for EVEN K it is K/2 - 1, so the window is
    // not shifted by a pixel relative to MATLAB.
    const keel::i32 half_r = (kH - 1) / 2;
    const keel::i32 half_c = (kW - 1) / 2;

    keel::i32 outH, outW;
    if (full) { outH = H + kH - 1; outW = W + kW - 1; }
    else      { outH = H; outW = W; }

    Value out = Value::matrix(outH, outW, I.type(), mr);
    if (outH == 0 || outW == 0) return out;

    // Read kernel into a contiguous double buffer.
    keel::Array<keel::f64> kbuf((keel::usize)kH * (keel::usize)kW);
    for (keel::usize i = 0; i < kbuf.size(); ++i) kbuf[i] = h.elemAsDouble(i);

    // For 'same' correlation: output(r,c) = Σ_{ki, kj} kernel[ki, kj]
    //                                       · I[r + ki - half_r, c + kj - half_c].
    // For 'conv': flip the kernel by 180° before applying.
    auto getK = [&](keel::i32 ki, keel::i32 kj) -> keel::f64 {
        const keel::i32 kki = flip_kernel ? (kH - 1 - ki) : ki;
        const keel::i32 kkj = flip_kernel ? (kW - 1 - kj) : kj;
        return kbuf[(keel::usize)kkj * (keel::usize)kH + (keel::usize)kki];
    };

    if (full) {
        // 'full' mode: index translation inside the larger output grid.
        for (keel::i32 oc = 0; oc < outW; ++oc) {
            for (keel::i32 orow = 0; orow < outH; ++orow) {
                keel::f64 s = 0.0;
                for (keel::i32 kj = 0; kj < kW; ++kj) {
                    const keel::i32 c_in = oc - kj;
                    if (c_in < 0 || c_in >= W) continue;
                    for (keel::i32 ki = 0; ki < kH; ++ki) {
                        const keel::i32 r_in = orow - ki;
                        if (r_in < 0 || r_in >= H) continue;
                        s += getK(ki, kj) * I.elemAsDouble((keel::usize)c_in * (keel::usize)H + (keel::usize)r_in);
                    }
                }
                store_classed(out, (keel::usize)oc * (keel::usize)outH + (keel::usize)orow, s, I.type());
            }
        }
    } else if (I.type() == ValueType::DOUBLE) {
        // SIMD-friendly fast path for DOUBLE input: the interior (where the
        // whole kernel fits) is computed as one FMA pass per kernel tap over
        // contiguous output columns — the inner loop auto-vectorises. Taps
        // are accumulated in the SAME kj-major / ki-minor order as the scalar
        // loop, so the result is bit-for-bit identical. The border keeps the
        // scalar sample_padded reduction (exact boundary semantics).
        const keel::f64 *src = I.doubleData();
        keel::f64 *dst = out.doubleDataMut();
        const keel::i32 ir0 = std::max(0, half_r), ir1 = std::min(H, H - kH + 1 + half_r);
        const keel::i32 ic0 = std::max(0, half_c), ic1 = std::min(W, W - kW + 1 + half_c);

        if (ir1 > ir0 && ic1 > ic0) {
            // Each output column writes a disjoint dst column and only reads
            // src + the read-only kernel, so the column loop is data-parallel.
            // parallel_for is a no-op (plain fn(0,n)) unless NUMKIT_WITH_THREADS
            // is set; thread only when the interior work dwarfs the pool
            // hand-off (~1M tap-multiplies), else stay on the caller's thread.
            const keel::usize ncols = (keel::usize)(ic1 - ic0);
            const keel::usize work  = ncols * (keel::usize)(ir1 - ir0) *
                                      (keel::usize)kH * (keel::usize)kW;
            const keel::usize threshold =
                (work >= (keel::usize{1} << 20)) ? keel::usize{1} : ncols + 1;
            numkit::detail::parallel_for(ncols, threshold,
                [&](keel::usize cs, keel::usize ce) {
                    for (keel::usize cc = cs; cc < ce; ++cc) {
                        const keel::i32 oc = ic0 + (keel::i32)cc;
                        keel::f64 *dcol = dst + (keel::usize)oc * (keel::usize)H;
                        for (keel::i32 r = ir0; r < ir1; ++r) dcol[r] = 0.0;
                        for (keel::i32 kj = 0; kj < kW; ++kj) {
                            const keel::i32 c_in = oc + kj - half_c;
                            for (keel::i32 ki = 0; ki < kH; ++ki) {
                                const keel::f64 w = getK(ki, kj);
                                const keel::f64 *scol = src + (keel::usize)c_in * (keel::usize)H + (ki - half_r);
                                for (keel::i32 r = ir0; r < ir1; ++r) dcol[r] += w * scol[r];
                            }
                        }
                    }
                });
        }
        // Border pixels (kernel overhangs the edge) — scalar, exact.
        for (keel::i32 oc = 0; oc < W; ++oc) {
            const bool colInterior = (oc >= ic0 && oc < ic1);
            for (keel::i32 orow = 0; orow < H; ++orow) {
                if (colInterior && orow >= ir0 && orow < ir1) continue;
                keel::f64 s = 0.0;
                for (keel::i32 kj = 0; kj < kW; ++kj) {
                    const keel::i32 c_in = oc + kj - half_c;
                    for (keel::i32 ki = 0; ki < kH; ++ki) {
                        const keel::i32 r_in = orow + ki - half_r;
                        s += getK(ki, kj) * sample_padded(I, r_in, c_in, H, W,
                                                          boundary, pad_value);
                    }
                }
                dst[(keel::usize)oc * (keel::usize)H + (keel::usize)orow] = s;
            }
        }
    } else {
        for (keel::i32 oc = 0; oc < W; ++oc) {
            for (keel::i32 orow = 0; orow < H; ++orow) {
                keel::f64 s = 0.0;
                for (keel::i32 kj = 0; kj < kW; ++kj) {
                    const keel::i32 c_in = oc + kj - half_c;
                    for (keel::i32 ki = 0; ki < kH; ++ki) {
                        const keel::i32 r_in = orow + ki - half_r;
                        s += getK(ki, kj) * sample_padded(I, r_in, c_in, H, W,
                                                          boundary, pad_value);
                    }
                }
                store_classed(out, (keel::usize)oc * (keel::usize)H + (keel::usize)orow, s, I.type());
            }
        }
    }
    return out;
}

// Apply a separable 2-D filter — the 1-D kernel `g` along rows, then along
// columns — in DOUBLE, then convert back to I's class. For a 2-D kernel that
// equals g⊗g (Gaussian, box) this is exactly the full 2-D convolution but
// costs O(2k) instead of O(k²), keeps the intermediate un-quantised between
// passes (matches MATLAB), and hits imfilter's fast double path. 2-D only.
static Value applySeparableSym(const Value &I, keel::ArrayRef<keel::f64> g,
                               keel::Allocator *mr)
{
    const keel::i32 k = (keel::i32)g.size();
    Value gRow = mat_double(g, 1, k, mr);
    Value gCol = mat_double(g, k, 1, mr);
    const keel::i32 H = (keel::i32)I.dims().rows(), W = (keel::i32)I.dims().cols();

    Value Id;
    if (I.type() == ValueType::DOUBLE) {
        Id = I;
    } else {
        Id = Value::matrix(H, W, ValueType::DOUBLE, mr);
        keel::f64 *dd = Id.doubleDataMut();
        for (keel::usize i = 0; i < (keel::usize)H * (keel::usize)W; ++i) dd[i] = I.elemAsDouble(i);
    }
    Value tmp = imfilter(Id,  gRow, PadMode::Replicate, 0.0, false, false, mr);
    Value res = imfilter(tmp, gCol, PadMode::Replicate, 0.0, false, false, mr);
    if (I.type() == ValueType::DOUBLE) return res;

    Value out = Value::matrix(H, W, I.type(), mr);
    const keel::f64 *rd = res.doubleData();
    for (keel::usize i = 0; i < (keel::usize)H * (keel::usize)W; ++i) store_classed(out, i, rd[i], I.type());
    return out;
}

Value imgaussfilt(const Value &I, keel::f64 sigma, keel::i32 filter_size, keel::Allocator *mr)
{
    if (filter_size <= 0) {
        // Default: 2·ceil(2σ) + 1.
        filter_size = 2 * (keel::i32)std::ceil(2.0 * sigma) + 1;
        if (filter_size < 3) filter_size = 3;
    }
    if (filter_size % 2 == 0) filter_size += 1;

    // Non-2-D inputs keep the original (2-D-kernel) path unchanged.
    if (I.dims().is3D() || sigma <= 0.0)
        return imfilter(I, fspecial_gaussian(filter_size, filter_size, sigma, mr),
                        PadMode::Replicate, 0.0, false, false, mr);

    // SEPARABLE: a 2-D Gaussian is exactly the outer product of two 1-D
    // Gaussians (g_row ⊗ g_col == fspecial_gaussian, since sum2d == sum1d²),
    // so two 1-D passes cost O(2k) instead of O(k²). We filter in DOUBLE so
    // the intermediate isn't re-quantised between passes (matches MATLAB) and
    // so both passes hit imfilter's fast double path, then convert back to the
    // input class.
    const keel::f64 cx = (filter_size - 1) / 2.0;
    const keel::f64 inv2s2 = 1.0 / (2.0 * sigma * sigma);
    keel::Array<keel::f64> g((keel::usize)filter_size);
    keel::f64 sum = 0.0;
    for (keel::i32 i = 0; i < filter_size; ++i) {
        const keel::f64 d = i - cx;
        g[i] = std::exp(-d * d * inv2s2);
        sum += g[i];
    }
    if (sum > 0.0) for (auto &v : g) v /= sum;
    return applySeparableSym(I, g, mr);
}

Value imboxfilt(const Value &I, keel::i32 filter_size, keel::Allocator *mr)
{
    imboxfilt_argcheck(I);
    if (filter_size <= 0) filter_size = 3;
    if (filter_size % 2 == 0) filter_size += 1;
    // Non-2-D keeps the original 2-D-kernel path.
    if (I.dims().is3D())
        return imfilter(I, fspecial_average(filter_size, filter_size, mr),
                        PadMode::Replicate, 0.0, false, false, mr);

    // INTEGRAL IMAGE: build a summed-area table of the replicate-padded image,
    // then each window mean is 4 lookups — O(1) per pixel regardless of k
    // (flat in filter size, like MATLAB), and uint8 needs no per-pass double
    // filtering. Result equals the replicate box mean (same as the separable
    // / 2-D-kernel path) up to FP summation order.
    const keel::i32 k = filter_size, hk = (k - 1) / 2;
    const keel::i32 H = (keel::i32)I.dims().rows(), W = (keel::i32)I.dims().cols();
    Value out = Value::matrix(H, W, I.type(), mr);
    if (H == 0 || W == 0) return out;

    const keel::i32 Hp = H + 2 * hk, Wp = W + 2 * hk;   // replicate-padded extent
    const keel::i32 Hs = Hp + 1, Ws = Wp + 1;           // integral-image extent
    keel::ArenaTemp arena = keel::get_scratch(mr);

    // uint8 fast path: INTEGER summed-area table (uint32 — typed read, no
    // elemAsDouble; half the bytes of the double SAT; integer mean). Used when
    // the full-image sum can't overflow uint32 (H·W·255 < 2³², i.e. up to
    // ~16.8 Mpx). Bit-identical to the double SAT: (2·sum + k²)/(2·k²) is
    // round-half-up == lround(sum/k²) for the non-negative box sum.
    if (I.type() == ValueType::UINT8 &&
        (keel::u64)H * (keel::u64)W * 255ULL < (1ULL << 32)) {
        const keel::u8 *src = I.uint8Data();
        keel::u8 *dst = out.uint8DataMut();
        keel::Array<keel::u32> Si((keel::usize)Hs * (keel::usize)Ws, arena.allocator());
        auto Q = [&](keel::i32 r, keel::i32 c) -> keel::u32 & { return Si[(keel::usize)c * (keel::usize)Hs + (keel::usize)r]; };
        for (keel::i32 c = 0; c < Ws; ++c) Q(0, c) = 0;
        for (keel::i32 r = 0; r < Hs; ++r) Q(r, 0) = 0;
        for (keel::i32 c = 1; c < Ws; ++c) {
            const keel::i32 sc = std::min(std::max((c - 1) - hk, 0), W - 1);
            const keel::usize srcCol = (keel::usize)sc * (keel::usize)H;
            for (keel::i32 r = 1; r < Hs; ++r) {
                const keel::i32 sr = std::min(std::max((r - 1) - hk, 0), H - 1);
                Q(r, c) = (keel::u32)src[srcCol + (keel::usize)sr]
                          + Q(r - 1, c) + Q(r, c - 1) - Q(r - 1, c - 1);
            }
        }
        const keel::u32 k2 = (keel::u32)k * (keel::u32)k;
        for (keel::i32 c = 0; c < W; ++c)
            for (keel::i32 r = 0; r < H; ++r) {
                const keel::u32 sum =
                    (Q(r + k, c + k) + Q(r, c)) - (Q(r, c + k) + Q(r + k, c));
                dst[(keel::usize)c * (keel::usize)H + (keel::usize)r] =
                    (keel::u8)((2u * sum + k2) / (2u * k2));
            }
        return out;
    }

    keel::Array<keel::f64> S((keel::usize)Hs * (keel::usize)Ws, arena.allocator());

    // Summed-area table S (column-major): S[c*Hs + r] = Σ P[0..r-1, 0..c-1],
    // where P is I replicate-padded by hk. First row/col are zero; we read
    // the padded source value on the fly (clamped indices) — no explicit pad
    // buffer. Recurrence: S[r][c] = P[r-1][c-1] + S[r-1][c] + S[r][c-1] - S[r-1][c-1].
    auto Sat = [&](keel::i32 r, keel::i32 c) -> keel::f64 & { return S[(keel::usize)c * (keel::usize)Hs + (keel::usize)r]; };
    for (keel::i32 c = 0; c < Ws; ++c) Sat(0, c) = 0.0;
    for (keel::i32 r = 0; r < Hs; ++r) Sat(r, 0) = 0.0;
    for (keel::i32 c = 1; c < Ws; ++c) {
        const keel::i32 sc = std::min(std::max((c - 1) - hk, 0), W - 1);   // padded→source col
        const keel::usize srcCol = (keel::usize)sc * (keel::usize)H;
        for (keel::i32 r = 1; r < Hs; ++r) {
            const keel::i32 sr = std::min(std::max((r - 1) - hk, 0), H - 1); // padded→source row
            const keel::f64 p = I.elemAsDouble(srcCol + (keel::usize)sr);
            Sat(r, c) = p + Sat(r - 1, c) + Sat(r, c - 1) - Sat(r - 1, c - 1);
        }
    }

    const keel::f64 invk2 = 1.0 / ((keel::f64)k * (keel::f64)k);
    for (keel::i32 c = 0; c < W; ++c)
        for (keel::i32 r = 0; r < H; ++r) {
            // Output (r,c) → padded window rows [r, r+k), cols [c, c+k).
            const keel::f64 sum = Sat(r + k, c + k) - Sat(r, c + k) - Sat(r + k, c) + Sat(r, c);
            store_classed(out, (keel::usize)c * (keel::usize)H + (keel::usize)r, sum * invk2, I.type());
        }
    return out;
}

// integralBoxFilter — 2-D box filter on a precomputed integral image.
//
// Given integral image I of shape (H+1) x (W+1) [optionally x C], for each
// output pixel (oi, oj) in the underlying-image coordinates the underlying
// box [oi..oi+fH-1, oj..oj+fW-1] has sum
//     I[oi+fH, oj+fW] - I[oi, oj+fW] - I[oi+fH, oj] + I[oi, oj]
// (4 lookups → O(1) per pixel regardless of filter size).
// Output is (H - fH + 1) x (W - fW + 1) — only the no-boundary core.
// For 3-D input the filter is applied per-channel.
Value integralBoxFilter(const Value &I, keel::i32 fH, keel::i32 fW, keel::f64 normFactor,
                         keel::Allocator *mr)
{
    integralBoxFilter_argcheck(I, fH, fW);
    const auto &d = I.dims();
    const keel::usize H1 = d.rows();
    const keel::usize W1 = d.cols();
    const keel::usize H = H1 - 1;
    const keel::usize W = W1 - 1;

    const keel::usize C = (d.ndim() == 3) ? d.pages() : 1;
    const keel::usize outH = H - keel::as<keel::usize>(fH) + 1;
    const keel::usize outW = W - keel::as<keel::usize>(fW) + 1;

    Value out = (C == 1)
        ? Value::matrix(outH, outW, ValueType::DOUBLE, mr)
        : Value::matrix3d(outH, outW, C, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();

    const keel::usize inPlane = H1 * W1;
    // normFactor is a MULTIPLIER (MATLAB semantics): out = boxSum * normFactor.
    // Default (1/(fH·fW)) → mean; pass 1 for raw sum.

    for (keel::usize c = 0; c < C; ++c) {
        for (keel::usize oj = 0; oj < outW; ++oj) {
            const keel::usize c0 = oj;
            const keel::usize c1 = oj + keel::as<keel::usize>(fW);
            for (keel::usize oi = 0; oi < outH; ++oi) {
                const keel::usize r0 = oi;
                const keel::usize r1 = oi + keel::as<keel::usize>(fH);
                // Column-major indexing for I: idx = c·inPlane + col·H1 + row.
                const keel::usize base = c * inPlane;
                const keel::f64 s =
                      I.elemAsDouble(base + c1 * H1 + r1)
                    - I.elemAsDouble(base + c1 * H1 + r0)
                    - I.elemAsDouble(base + c0 * H1 + r1)
                    + I.elemAsDouble(base + c0 * H1 + r0);
                const keel::usize dstIdx = (C == 1)
                    ? (oj * outH + oi)
                    : (c * outH * outW + oj * outH + oi);
                od[dstIdx] = s * normFactor;
            }
        }
    }
    return out;
}

// integralBoxFilter3 — 3-D box filter on a precomputed integral volume.
//
// Given integral volume I of shape (H+1) × (W+1) × (D+1) (output of
// integralImage3), for each output voxel (oi, oj, ok) the underlying box
// [oi..oi+fH-1, oj..oj+fW-1, ok..ok+fP-1] has sum given by the 8-corner
// inclusion-exclusion query on I (constant time regardless of box size).
// Output is (H - fH + 1) × (W - fW + 1) × (D - fP + 1) — the no-boundary
// core, matching MATLAB. `normFactor` is a MULTIPLIER (MATLAB semantics):
// each box sum is multiplied by it.
Value integralBoxFilter3(const Value &I, keel::i32 fH, keel::i32 fW, keel::i32 fP,
                         keel::f64 normFactor, keel::Allocator *mr)
{
    integralBoxFilter3_argcheck(I, fH, fW, fP);
    const auto &d = I.dims();
    const keel::usize H1 = d.rows();
    const keel::usize W1 = d.cols();
    const keel::usize D1 = d.is3D() ? d.pages() : 1;
    const keel::usize H = H1 - 1, W = W1 - 1, D = D1 - 1;

    const keel::usize outH = H - keel::as<keel::usize>(fH) + 1;
    const keel::usize outW = W - keel::as<keel::usize>(fW) + 1;
    const keel::usize outD = D - keel::as<keel::usize>(fP) + 1;

    Value out = (outD > 1)
        ? Value::matrix3d(outH, outW, outD, ValueType::DOUBLE, mr)
        : Value::matrix(outH, outW, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();

    const keel::usize plane = H1 * W1;  // page stride in I
    // Column-major, page-major index into the integral volume I.
    auto Iat = [&](keel::usize r, keel::usize c, keel::usize p) -> keel::f64 {
        return I.elemAsDouble(p * plane + c * H1 + r);
    };

    for (keel::usize ok = 0; ok < outD; ++ok) {
        const keel::usize p0 = ok, p1 = ok + keel::as<keel::usize>(fP);
        for (keel::usize oj = 0; oj < outW; ++oj) {
            const keel::usize c0 = oj, c1 = oj + keel::as<keel::usize>(fW);
            for (keel::usize oi = 0; oi < outH; ++oi) {
                const keel::usize r0 = oi, r1 = oi + keel::as<keel::usize>(fH);
                const keel::f64 s =
                      Iat(r1, c1, p1)
                    - Iat(r0, c1, p1) - Iat(r1, c0, p1) - Iat(r1, c1, p0)
                    + Iat(r0, c0, p1) + Iat(r0, c1, p0) + Iat(r1, c0, p0)
                    - Iat(r0, c0, p0);
                const keel::usize dstIdx = ok * outH * outW + oj * outH + oi;
                od[dstIdx] = s * normFactor;
            }
        }
    }
    return out;
}

// modefilt — 2-D mode filter. For each output pixel we histogram the
// neighbourhood and pick the most-common value (ties → smallest value).
//
// For UINT8 / LOGICAL we use a 256-bucket array (cache-friendly, no
// hashing). For all other numeric types we fall back to an ordered
// std::map<double, int> per pixel — slower but generic enough.
//
// Padding modes (MATLAB-compatible):
//   "symmetric" (default) — mirror reflection without duplicating edge
//                            (so abcde → cba|abcde|edc)
//   "replicate"           — repeat edge pixel (aaaa|abcde|eeee)
//   "zeros"               — zero pad
// Internal 2-D and 3-D worker: when fD > 0 the input is treated as a
// 3-D volume and the filter window is fH × fW × fD; otherwise pure 2-D.
static Value modefilt_impl(const Value &A, keel::i32 fH, keel::i32 fW, keel::i32 fD,
                            keel::StringRef padopt,
                            keel::Allocator *mr)
{
    keel::i32 padMode = 0;  // 0=symmetric, 1=replicate, 2=zeros
    modefilt_argcheck(A, fH, fW, fD, padopt, padMode);
    const bool is3D = (fD > 0);
    const auto &d = A.dims();
    const keel::i32 H = keel::as<keel::i32>(d.rows());
    const keel::i32 W = keel::as<keel::i32>(d.cols());
    const keel::i32 D = is3D ? keel::as<keel::i32>(d.is3D() ? d.pages() : 1) : 1;
    const keel::i32 hh = fH / 2;
    const keel::i32 hw = fW / 2;
    const keel::i32 hd = is3D ? (fD / 2) : 0;
    Value out = is3D ? Value::matrix3d(H, W, D, A.type(), mr)
                     : Value::matrix(H, W, A.type(), mr);

    // Index access with padding: returns the input index for sample
    // location (rr, cc, pp), or -1 if the pixel should be treated as zero.
    auto reflect = [&](keel::i32 v, keel::i32 N) -> keel::i32 {
        if (padMode == 1) {  // replicate
            if (v < 0) return 0;
            if (v >= N) return N - 1;
            return v;
        }
        if (padMode == 2) {  // zeros — caller checks for -1
            if (v < 0 || v >= N) return -1;
            return v;
        }
        // symmetric (mirror without dup).
        while (v < 0 || v >= N) {
            if (v < 0)      v = -v - 1;
            if (v >= N)     v = 2 * N - v - 1;
        }
        return v;
    };
    auto padIdx = [&](keel::i32 rr, keel::i32 cc, keel::i32 pp) -> keel::i64 {
        keel::i32 r = reflect(rr, H);
        keel::i32 c = reflect(cc, W);
        keel::i32 p = is3D ? reflect(pp, D) : 0;
        if (r < 0 || c < 0 || (is3D && p < 0)) return -1;
        const keel::i64 plane = keel::as<keel::i64>(H) * W;
        return keel::as<keel::i64>(p) * plane
             + keel::as<keel::i64>(c) * H + r;
    };

    auto getDouble = [&](keel::i64 idx) -> keel::f64 {
        return idx < 0 ? 0.0 : A.elemAsDouble(keel::as<keel::usize>(idx));
    };

    // Fast UINT8 / LOGICAL path with 256-bucket histogram.
    const ValueType vt = A.type();
    const bool fastByte = (vt == ValueType::UINT8 || vt == ValueType::LOGICAL);

    const keel::i64 plane = keel::as<keel::i64>(H) * W;
    const keel::i32 pMax = is3D ? D : 1;
    keel::Array<keel::f64> vals;
    if (!fastByte)
        vals.reserve(keel::as<keel::usize>(fH) * keel::as<keel::usize>(fW) *
                     keel::as<keel::usize>(is3D ? fD : 1));
    for (keel::i32 p = 0; p < pMax; ++p) {
        for (keel::i32 c = 0; c < W; ++c) {
            for (keel::i32 r = 0; r < H; ++r) {
                if (fastByte) {
                keel::i32 hist[256] = {0};
                for (keel::i32 dp = -hd; dp <= hd; ++dp)
                for (keel::i32 dc = -hw; dc <= hw; ++dc)
                    for (keel::i32 dr = -hh; dr <= hh; ++dr) {
                        const keel::i64 idx = padIdx(r + dr, c + dc, p + dp);
                        const keel::i32 v = keel::as<keel::i32>(getDouble(idx));
                        if (v >= 0 && v < 256) ++hist[v];
                    }
                // Ties → smallest value wins (matches MATLAB's documented
                // `mode` behaviour). MATLAB's modefilt internal MEX uses an
                // undocumented order-dependent rule for ties that doesn't
                // match base mode in all edge cases; we follow the spec.
                keel::i32 bestVal = 0, bestCnt = -1;
                for (keel::i32 v = 0; v < 256; ++v) {
                    if (hist[v] > bestCnt) {
                        bestCnt = hist[v];
                        bestVal = v;
                    }
                }
                const keel::i64 dstIdx = keel::as<keel::i64>(p) * plane
                                          + keel::as<keel::i64>(c) * H + r;
                if (vt == ValueType::UINT8)
                    out.uint8DataMut()[dstIdx] =
                        keel::as<keel::u8>(bestVal);
                else
                    out.logicalDataMut()[dstIdx] =
                        keel::as<keel::u8>(bestVal != 0);
                continue;
            }
            // Generic path: sort window values and pick highest-frequency run.
            vals.clear();
            for (keel::i32 dp = -hd; dp <= hd; ++dp)
            for (keel::i32 dc = -hw; dc <= hw; ++dc)
                for (keel::i32 dr = -hh; dr <= hh; ++dr) {
                    const keel::i64 idx = padIdx(r + dr, c + dc, p + dp);
                    if (idx < 0 && padMode != 2) continue;
                    vals.push_back(getDouble(idx));
                }
            std::sort(vals.begin(), vals.end());
            // Smallest-wins-on-tie (see fastByte path comment).
            keel::i32 bestCnt = -1;
            keel::f64 bestVal = 0.0;
            for (keel::usize vi = 0; vi < vals.size(); ) {
                keel::usize vj = vi + 1;
                while (vj < vals.size() && vals[vj] == vals[vi]) ++vj;
                const keel::i32 cnt = keel::as<keel::i32>(vj - vi);
                if (cnt > bestCnt) {
                    bestCnt = cnt;
                    bestVal = vals[vi];
                }
                vi = vj;
            }
            // Store into output of matching dtype.
            const keel::usize dst = keel::as<keel::usize>(p) * keel::as<keel::usize>(plane)
                              + keel::as<keel::usize>(c) * H + keel::as<keel::usize>(r);
            switch (vt) {
                case ValueType::DOUBLE: out.doubleDataMut()[dst] = bestVal; break;
                case ValueType::SINGLE: out.singleDataMut()[dst] = keel::as<keel::f32>(bestVal); break;
                case ValueType::UINT16: out.uint16DataMut()[dst] = keel::as<keel::u16>(bestVal); break;
                case ValueType::UINT32: out.uint32DataMut()[dst] = keel::as<keel::u32>(bestVal); break;
                case ValueType::INT8:   out.int8DataMut()[dst]   = keel::as<keel::i8>(bestVal); break;
                case ValueType::INT16:  out.int16DataMut()[dst]  = keel::as<keel::i16>(bestVal); break;
                case ValueType::INT32:  out.int32DataMut()[dst]  = keel::as<keel::i32>(bestVal); break;
                default:                out.doubleDataMut()[dst] = bestVal; break;
            }
            }
        }
    }
    return out;
}

// Public 2-D entry point — pure 2-D semantics (fD = 0 means no 3rd dim).
Value modefilt(const Value &A, keel::i32 fH, keel::i32 fW,
               keel::StringRef padopt, keel::Allocator *mr)
{
    return modefilt_impl(A, fH, fW, /*fD=*/0, padopt, mr);
}

// Public 3-D entry point — fD is the depth-axis window length.
Value modefilt3D(const Value &A, keel::i32 fH, keel::i32 fW, keel::i32 fD,
                 keel::StringRef padopt, keel::Allocator *mr)
{
    return modefilt_impl(A, fH, fW, fD, padopt, mr);
}

// Median of 9 via a branchless 19-comparator sorting network (Devillard's
// opt_med9) — faster than nth_element for the ubiquitous 3×3 window because
// it has no partitioning/branch overhead. Median = element at sorted index 4.
inline keel::f64 med9(keel::f64 p[9])
{
    #define NUMKIT_PIX_SORT(a, b) do { if (p[a] > p[b]) { const keel::f64 t = p[a]; p[a] = p[b]; p[b] = t; } } while (0)
    NUMKIT_PIX_SORT(1, 2); NUMKIT_PIX_SORT(4, 5); NUMKIT_PIX_SORT(7, 8);
    NUMKIT_PIX_SORT(0, 1); NUMKIT_PIX_SORT(3, 4); NUMKIT_PIX_SORT(6, 7);
    NUMKIT_PIX_SORT(1, 2); NUMKIT_PIX_SORT(4, 5); NUMKIT_PIX_SORT(7, 8);
    NUMKIT_PIX_SORT(0, 3); NUMKIT_PIX_SORT(5, 8); NUMKIT_PIX_SORT(4, 7);
    NUMKIT_PIX_SORT(3, 6); NUMKIT_PIX_SORT(1, 4); NUMKIT_PIX_SORT(2, 5);
    NUMKIT_PIX_SORT(4, 7); NUMKIT_PIX_SORT(4, 2); NUMKIT_PIX_SORT(6, 4);
    NUMKIT_PIX_SORT(4, 2);
    #undef NUMKIT_PIX_SORT
    return p[4];
}

// Huang's sliding-histogram median for UINT8 with an ODD×ODD window: maintain
// a 256-bin histogram of the window and the median bin `med` with `below` =
// count of pixels < med; sliding the window one row keeps a running median
// in O(window-width) per pixel (vs O(k² log k²) for a per-pixel nth_element).
// Zero-padded border (MATLAB medfilt2 default); n = rows·cols is odd so the
// median is the single th-th (0-based, th = n/2) order statistic — identical
// to nth_element's window[n/2]. Scans down columns (contiguous, column-major).
static Value medfilt2_huang_u8(const Value &I, keel::i32 rows, keel::i32 cols,
                               keel::Allocator *mr)
{
    const keel::i32 H = (keel::i32)I.dims().rows(), W = (keel::i32)I.dims().cols();
    const keel::i32 hr = (rows - 1) / 2, hc = (cols - 1) / 2;
    Value out = Value::matrix(H, W, ValueType::UINT8, mr);
    if (H == 0 || W == 0) return out;
    const keel::u8 *src = I.uint8Data();
    keel::u8 *dst = out.uint8DataMut();

    const keel::i32 Hp = H + rows - 1, Wp = W + cols - 1;        // zero-padded extent
    keel::ArenaTemp arena = keel::get_scratch(mr);
    keel::Array<keel::u8> Pp((keel::usize)Hp * (keel::usize)Wp, arena.allocator());
    std::fill(Pp.begin(), Pp.end(), keel::u8{0});
    for (keel::i32 c = 0; c < W; ++c) {
        const keel::usize scol = (keel::usize)c * (keel::usize)H;
        keel::u8 *pcol = Pp.data() + (keel::usize)(c + hc) * (keel::usize)Hp + (keel::usize)hr;
        for (keel::i32 r = 0; r < H; ++r) pcol[r] = src[scol + (keel::usize)r];
    }

    const keel::i32 th = (rows * cols) / 2;     // 0-based median index (n odd)
    keel::i32 hist[256];
    for (keel::i32 oc = 0; oc < W; ++oc) {
        std::fill(hist, hist + 256, 0);
        for (keel::i32 wc = 0; wc < cols; ++wc) {
            const keel::u8 *col = Pp.data() + (keel::usize)(oc + wc) * (keel::usize)Hp;
            for (keel::i32 wr = 0; wr < rows; ++wr) ++hist[col[wr]];
        }
        keel::i32 med = 0, below = 0;
        while (below + hist[med] <= th) { below += hist[med]; ++med; }
        dst[(keel::usize)oc * (keel::usize)H + 0] = (keel::u8)med;

        for (keel::i32 orow = 1; orow < H; ++orow) {
            const keel::i32 rOut = orow - 1;            // window row leaving (top)
            const keel::i32 rIn  = orow + rows - 1;     // window row entering (bottom)
            for (keel::i32 wc = 0; wc < cols; ++wc) {
                const keel::u8 *col = Pp.data() + (keel::usize)(oc + wc) * (keel::usize)Hp;
                const keel::u8 vo = col[rOut]; --hist[vo]; if (vo < med) --below;
                const keel::u8 vi = col[rIn];  ++hist[vi]; if (vi < med) ++below;
            }
            while (below > th)                 { --med; below -= hist[med]; }
            while (below + hist[med] <= th)    { below += hist[med]; ++med; }
            dst[(keel::usize)oc * (keel::usize)H + (keel::usize)orow] = (keel::u8)med;
        }
    }
    return out;
}

Value medfilt2(const Value &I, keel::i32 rows, keel::i32 cols, keel::Allocator *mr)
{
    medfilt2_argcheck(I);
    // Logical input → logical output (probed R2025b): median on the
    // double copy, re-binarise.
    if (I.type() == ValueType::LOGICAL)
        return internal::castResultToClass(
            medfilt2(internal::coerceToDoubleImage(I, mr), rows, cols, mr),
            ValueType::LOGICAL, mr);
    // Fast path: uint8 + odd×odd window (n odd) → Huang sliding histogram.
    if (I.type() == ValueType::UINT8 && rows > 0 && cols > 0 && (rows & 1) && (cols & 1))
        return medfilt2_huang_u8(I, rows, cols, mr);

    const keel::i32 H = (keel::i32)I.dims().rows();
    const keel::i32 W = (keel::i32)I.dims().cols();
    const keel::i32 half_r = rows / 2;
    const keel::i32 half_c = cols / 2;
    Value out = Value::matrix(H, W, I.type(), mr);
    if (H == 0 || W == 0) return out;

    // Fast path: 3×3 double → branchless median-of-9 sorting network (the
    // uint8 3×3 case already took the Huang path above). Zero-pad border.
    if (rows == 3 && cols == 3 && I.type() == ValueType::DOUBLE) {
        const keel::f64 *src = I.doubleData();
        keel::f64 *dst = out.doubleDataMut();
        for (keel::i32 oc = 0; oc < W; ++oc)
            for (keel::i32 orow = 0; orow < H; ++orow) {
                keel::f64 p[9];
                keel::i32 m = 0;
                for (keel::i32 kj = 0; kj < 3; ++kj)
                    for (keel::i32 ki = 0; ki < 3; ++ki) {
                        const keel::i32 r_in = orow + ki - 1, c_in = oc + kj - 1;
                        p[m++] = (r_in < 0 || r_in >= H || c_in < 0 || c_in >= W)
                                     ? 0.0 : src[(keel::usize)c_in * (keel::usize)H + (keel::usize)r_in];
                    }
                dst[(keel::usize)oc * (keel::usize)H + (keel::usize)orow] = med9(p);
            }
        return out;
    }

    keel::Array<keel::f64> window;
    window.reserve((keel::usize)rows * (keel::usize)cols);

    // Typed pointers for the common image classes — avoids the per-element
    // elemAsDouble() type-switch in the hot window-gather loop.
    const keel::f64  *sd = (I.type() == ValueType::DOUBLE) ? I.doubleData() : nullptr;
    const keel::u8 *su = (I.type() == ValueType::UINT8)  ? I.uint8Data()  : nullptr;
    auto getv = [&](keel::usize idx) -> keel::f64 {
        if (sd) return sd[idx];
        if (su) return keel::as<keel::f64>(su[idx]);
        return I.elemAsDouble(idx);
    };

    for (keel::i32 oc = 0; oc < W; ++oc) {
        for (keel::i32 orow = 0; orow < H; ++orow) {
            window.clear();
            for (keel::i32 kj = 0; kj < cols; ++kj) {
                for (keel::i32 ki = 0; ki < rows; ++ki) {
                    const keel::i32 r_in = orow + ki - half_r;
                    const keel::i32 c_in = oc + kj - half_c;
                    // MATLAB medfilt2 default: zero-pad outside.
                    if (r_in < 0 || r_in >= H || c_in < 0 || c_in >= W) {
                        window.push_back(0.0);
                    } else {
                        window.push_back(getv((keel::usize)c_in * (keel::usize)H + (keel::usize)r_in));
                    }
                }
            }
            std::nth_element(window.begin(),
                             window.begin() + window.size() / 2,
                             window.end());
            keel::f64 med = window[window.size() / 2];
            if ((window.size() & 1) == 0) {
                // Even count: average of two middle values. nth_element put
                // the upper-middle in place; find max of the lower half.
                keel::f64 left_max = -std::numeric_limits<keel::f64>::infinity();
                for (auto it = window.begin();
                     it != window.begin() + window.size() / 2; ++it)
                    if (*it > left_max) left_max = *it;
                med = 0.5 * (left_max + med);
            }
            store_classed(out, (keel::usize)oc * (keel::usize)H + (keel::usize)orow, med, I.type());
        }
    }
    return out;
}

// ════════════════════════════════════════════════════════════════════
// imsharpen — unsharp masking
// ════════════════════════════════════════════════════════════════════
//
//   high  = I − imgaussfilt(I, sigma=radius)
//   mask  = (|high| ≥ threshold · max|high|)            // 1 if no thresh
//   B     = saturate(I + amount · mask · high)
//
// MATLAB R2025b defaults: radius=1, amount=0.8, threshold=0. Output
// has the same class as the input (saturating cast for ints).

Value imsharpen(const Value &I, keel::f64 radius, keel::f64 amount, keel::f64 threshold, keel::Allocator *mr)
{
    imsharpen_argcheck(I, threshold);
    if (!(radius > 0.0)) radius = 1.0;

    // Gaussian filter size: 2*ceil(2*sigma)+1 — MATLAB default.
    keel::i32 fs = 2 * (keel::i32)std::ceil(2.0 * radius) + 1;
    if (fs < 3) fs = 3;
    if (fs % 2 == 0) fs += 1;

    Value blurred = imgaussfilt(I, radius, fs, mr);

    const keel::usize Hh = I.dims().rows();
    const keel::usize Ww = I.dims().cols();
    const keel::usize N = I.numel();
    const bool useThreshold = (threshold > 0.0);

    Value out = Value::matrix(Hh, Ww, I.type(), mr);

    // Typed DOUBLE fast path: raw pointers, no per-element elemAsDouble /
    // store_classed dispatch. When threshold == 0 (the MATLAB default) the
    // high-pass needs no max-abs reduction and no temp buffer, so it fuses
    // into a single vectorisable pass  out = I + amount·(I − blurred).
    // Bit-for-bit identical to the generic path (same ops, same order;
    // store_classed(DOUBLE) is a plain write).
    if (I.type() == ValueType::DOUBLE && !I.isComplex() &&
        blurred.type() == ValueType::DOUBLE && !blurred.isComplex()) {
        const keel::f64 *Id = I.doubleData();
        const keel::f64 *Bd = blurred.doubleData();
        keel::f64 *od = out.doubleDataMut();
        if (!useThreshold) {
            for (keel::usize i = 0; i < N; ++i)
                od[i] = Id[i] + amount * (Id[i] - Bd[i]);
        } else {
            keel::Array<keel::f64> Hbuf(N);
            keel::f64 max_abs = 0.0;
            for (keel::usize i = 0; i < N; ++i) {
                const keel::f64 v = Id[i] - Bd[i];
                Hbuf[i] = v;
                const keel::f64 a = keel::abs(v);
                if (a > max_abs) max_abs = a;
            }
            const keel::f64 thr_abs = threshold * max_abs;
            for (keel::usize i = 0; i < N; ++i) {
                keel::f64 h = Hbuf[i];
                if (keel::abs(h) < thr_abs) h = 0.0;
                od[i] = Id[i] + amount * h;
            }
        }
        return out;
    }

    // Generic path (single / uintN / int16, or complex) — unchanged.
    keel::Array<keel::f64> Hbuf(N);
    keel::f64 max_abs = 0.0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 v = I.elemAsDouble(i) - blurred.elemAsDouble(i);
        Hbuf[i] = v;
        const keel::f64 a = keel::abs(v);
        if (a > max_abs) max_abs = a;
    }
    const keel::f64 thr_abs = threshold * max_abs;
    for (keel::usize i = 0; i < N; ++i) {
        keel::f64 h = Hbuf[i];
        if (useThreshold && keel::abs(h) < thr_abs) h = 0.0;
        const keel::f64 v = I.elemAsDouble(i) + amount * h;
        store_classed(out, i, v, I.type());
    }
    return out;
}

// ════════════════════════════════════════════════════════════════════
// im2col — rearrange neighborhoods into matrix columns
// ════════════════════════════════════════════════════════════════════
//
// MATLAB convention: input A is H×W column-major; block size is m×n
// (m=rows, n=cols).
//
// "sliding": output has m·n rows × (H − m + 1)·(W − n + 1) cols.
//   Column index iterates column-major over top-left block positions
//   (r0=0..H−m, c0=0..W−n with r0 fastest); within each column the
//   block is laid out column-major (br=0..m−1, bc=0..n−1, br fastest).
// "distinct": A is partitioned into ceil(H/m) × ceil(W/n) tiles,
//   zero-padded at the right/bottom when needed. Output is m·n ×
//   ceil(H/m)·ceil(W/n). Same column-major ordering rules.

namespace {

template <typename T>
void im2col_sliding_ptr(const T *ad, T *bd, keel::usize H, keel::usize W,
                        keel::i32 m, keel::i32 n)
{
    const keel::usize outH = (keel::usize)(m * n);
    const keel::usize Hp = H - (keel::usize)m + 1;
    const keel::usize Wp = W - (keel::usize)n + 1;
    keel::usize col = 0;
    for (keel::usize c0 = 0; c0 < Wp; ++c0) {
        for (keel::usize r0 = 0; r0 < Hp; ++r0) {
            // Within-block: column-major (bc outer, br inner).
            keel::usize off = col * outH;
            for (keel::i32 bc = 0; bc < n; ++bc) {
                const keel::usize Acol = (c0 + (keel::usize)bc) * H;
                for (keel::i32 br = 0; br < m; ++br) {
                    bd[off++] = ad[Acol + r0 + (keel::usize)br];
                }
            }
            ++col;
        }
    }
}

template <typename T>
void im2col_distinct_ptr(const T *ad, T *bd, keel::usize H, keel::usize W,
                         keel::i32 m, keel::i32 n)
{
    const keel::usize outH = (keel::usize)(m * n);
    const keel::usize Hb = (H + (keel::usize)m - 1) / (keel::usize)m;
    const keel::usize Wb = (W + (keel::usize)n - 1) / (keel::usize)n;
    // Output starts zero-initialised by Value::matrix; we just write
    // the in-bounds elements.
    keel::usize col = 0;
    for (keel::usize bj = 0; bj < Wb; ++bj) {
        for (keel::usize bi = 0; bi < Hb; ++bi) {
            keel::usize off = col * outH;
            for (keel::i32 bc = 0; bc < n; ++bc) {
                const keel::usize cc = bj * (keel::usize)n + (keel::usize)bc;
                if (cc >= W) { off += (keel::usize)m; continue; }
                const keel::usize Acol = cc * H;
                for (keel::i32 br = 0; br < m; ++br) {
                    const keel::usize rr = bi * (keel::usize)m + (keel::usize)br;
                    if (rr < H) bd[off] = ad[Acol + rr];
                    ++off;
                }
            }
            ++col;
        }
    }
}

void im2col_argcheck(const Value &A, keel::i32 m, keel::i32 n, keel::StringRef block_type)
{
    internal::requireImageClassAllowComplex(A, "im2col", "A", 1,
                                internal::ImgClasses::FloatIntLog);
    if (m <= 0 || n <= 0)
        throw Error("im2col: block size must be positive",
                    0, 0, "im2col", "", "numkit:im2col:size");
    if (block_type != "sliding" && !block_type.empty() && block_type != "distinct")
        throw Error("im2col: block_type must be 'sliding' or 'distinct'",
                    0, 0, "im2col", "", "numkit:im2col:type");
}

void imbilatfilt_argcheck(const Value &I, keel::f64 degreeOfSmoothing)
{
    internal::requireImageClass(I, "imbilatfilt", "I", 1,
                                internal::ImgClasses::FloatInt);
    if (!(degreeOfSmoothing > 0.0))
        throw Error("imbilatfilt: degreeOfSmoothing must be > 0",
                    0, 0, "imbilatfilt", "", "numkit:imbilatfilt:dos");
}

} // anonymous

Value im2col(const Value &A, keel::i32 m, keel::i32 n, keel::StringRef block_type, keel::Allocator *mr)
{
    im2col_argcheck(A, m, n, block_type);
    const keel::usize H = A.dims().rows();
    const keel::usize W = A.dims().cols();
    const ValueType T = A.type();

    keel::usize outH = (keel::usize)(m * n);
    keel::usize outW;
    bool sliding = (block_type == "sliding" || block_type.empty());
    if (sliding) {
        // Block larger than the image in a sliding gather → an EMPTY
        // double (outH×0) result, matching the reference (probed:
        // im2col('ab', [3 3]) is zeros(9,0), not an error — the empty
        // result is double even for class-preserving inputs).
        outW = ((keel::usize)m > H || (keel::usize)n > W)
                   ? 0
                   : (H - (keel::usize)m + 1) * (W - (keel::usize)n + 1);
    } else if (block_type == "distinct") {
        const keel::usize Hb = (H + (keel::usize)m - 1) / (keel::usize)m;
        const keel::usize Wb = (W + (keel::usize)n - 1) / (keel::usize)n;
        outW = Hb * Wb;
    } else {
        throw Error("im2col: block_type must be 'sliding' or 'distinct'",
                    0, 0, "im2col", "", "numkit:im2col:type");
    }

    if (outH == 0 || outW == 0)
        return Value::matrix(outH, 0, ValueType::DOUBLE, mr);
    Value B = Value::matrix(outH, outW, T, mr);

    auto run = [&](const auto *ad, auto *bd) {
        if (sliding) im2col_sliding_ptr(ad, bd, H, W, m, n);
        else         im2col_distinct_ptr(ad, bd, H, W, m, n);
    };

    if (T == ValueType::COMPLEX) {
        // Complex gather — values preserved (probed R2025b).
        auto *ad = A.complexData();
        auto *bd = B.complexDataMut();
        const keel::usize outHc = outH;
        if (sliding) {
            const keel::usize Hp = (H >= (keel::usize)m) ? H - (keel::usize)m + 1 : 0;
            const keel::usize Wp = (W >= (keel::usize)n) ? W - (keel::usize)n + 1 : 0;
            keel::usize col = 0;
            for (keel::usize c0 = 0; c0 < Wp; ++c0)
                for (keel::usize r0 = 0; r0 < Hp; ++r0) {
                    keel::usize off = col * outHc;
                    for (keel::i32 bc = 0; bc < n; ++bc) {
                        const keel::usize Acol = (c0 + (keel::usize)bc) * H;
                        for (keel::i32 br = 0; br < m; ++br)
                            bd[off++] = ad[Acol + r0 + (keel::usize)br];
                    }
                    ++col;
                }
        } else {
            const keel::usize Hb = (H + (keel::usize)m - 1) / (keel::usize)m;
            const keel::usize Wb = (W + (keel::usize)n - 1) / (keel::usize)n;
            keel::usize col = 0;
            for (keel::usize bj = 0; bj < Wb; ++bj)
                for (keel::usize bi = 0; bi < Hb; ++bi) {
                    keel::usize off = col * outHc;
                    for (keel::i32 bc = 0; bc < n; ++bc) {
                        const keel::usize cc = bj * (keel::usize)n + (keel::usize)bc;
                        if (cc >= W) { off += (keel::usize)m; continue; }
                        const keel::usize Acol = cc * H;
                        for (keel::i32 br = 0; br < m; ++br) {
                            const keel::usize rr = bi * (keel::usize)m + (keel::usize)br;
                            bd[off++] = (rr < H) ? ad[Acol + rr]
                                                 : keel::c64{};
                        }
                    }
                    ++col;
                }
        }
        return B;
    }

    switch (T) {
        case ValueType::DOUBLE:  run(A.doubleData(),  B.doubleDataMut());  break;
        case ValueType::SINGLE:  run(A.singleData(),  B.singleDataMut());  break;
        case ValueType::UINT8:   run(A.uint8Data(),   B.uint8DataMut());   break;
        case ValueType::UINT16:  run(A.uint16Data(),  B.uint16DataMut());  break;
        case ValueType::INT16:   run(A.int16Data(),   B.int16DataMut());   break;
        case ValueType::LOGICAL: run(A.logicalData(), B.logicalDataMut()); break;
        default:
            throw Error("im2col: unsupported class",
                        0, 0, "im2col", "", "numkit:im2col:badtype");
    }
    return B;
}

// ════════════════════════════════════════════════════════════════════
// imbilatfilt — bilateral (edge-preserving) filter
// ════════════════════════════════════════════════════════════════════
//
// Per-pixel weighted average where the weight is the product of a
// spatial Gaussian and a range Gaussian over intensity difference:
//
//   J(x) = (1/W) · sum_y[ exp(−d²/(2σ_s²)) · exp(−Δ²/(2·dos)) · I(y) ]
//   W    = sum_y[ exp(−d²/(2σ_s²)) · exp(−Δ²/(2·dos)) ]
//
// where d is the spatial offset, Δ = I(x) − I(y), σ_s = spatialSigma,
// dos = degreeOfSmoothing (interpreted as the range Gaussian's
// VARIANCE, matching MATLAB R2025b). Boundary mode = replicate.
//
// Window: 2·ceil(2·σ_s) + 1 in each axis. Spatial-weight table is
// precomputed once per call. Range weight is per (centre, neighbour)
// pair — we eat the per-pixel exp; same cost as MATLAB's reference.

Value imbilatfilt(const Value &I, keel::f64 degreeOfSmoothing, keel::f64 spatialSigma, keel::Allocator *mr)
{
    imbilatfilt_argcheck(I, degreeOfSmoothing);
    if (!(spatialSigma > 0.0)) spatialSigma = 1.0;

    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    const keel::usize N = I.numel();
    Value out = Value::matrix(H, W, I.type(), mr);
    if (N == 0) return out;

    const keel::i32 half = std::max(1, (keel::i32)std::ceil(2.0 * spatialSigma));
    const keel::i32 win = 2 * half + 1;
    // The reference requires the window to be SMALLER than the image
    // (probed: imbilatfilt on a 5×5 with the default σs=1 → window 5
    // dies with "Expected NeighborhoodSize to be a scalar with
    // value < 5").
    if ((keel::usize)win >= std::min(H, W))
        throw Error(keel::format("imbilatfilt: NeighborhoodSize (window {}) "
                                 "must be smaller than the image (min dimension {}).",
                                 win, std::min(H, W)),
                    0, 0, "imbilatfilt", "",
                    "numkit:imbilatfilt:nhoodTooBig");
    const keel::f64 inv2ss = 0.5 / (spatialSigma * spatialSigma);
    const keel::f64 inv2dos = 0.5 / degreeOfSmoothing;

    // Precompute spatial weights — flat (win*win) row-major (di, dj).
    keel::Array<keel::f64> sw((keel::usize)win * (keel::usize)win);
    for (keel::i32 di = -half; di <= half; ++di)
        for (keel::i32 dj = -half; dj <= half; ++dj)
            sw[(keel::usize)(di + half) * (keel::usize)win + (keel::usize)(dj + half)]
                = std::exp(-(keel::f64)(di*di + dj*dj) * inv2ss);

    auto clamp = [](keel::i32 v, keel::i32 lo, keel::i32 hi) {
        if (v < lo) return lo;
        if (v > hi) return hi;
        return v;
    };

    for (keel::usize oc = 0; oc < W; ++oc) {
        for (keel::usize orow = 0; orow < H; ++orow) {
            const keel::f64 Ic = I.elemAsDouble(oc * H + orow);
            keel::f64 sum_w = 0.0, sum_v = 0.0;
            for (keel::i32 dj = -half; dj <= half; ++dj) {
                const keel::i32 c = clamp((keel::i32)oc + dj, 0, (keel::i32)W - 1);
                for (keel::i32 di = -half; di <= half; ++di) {
                    const keel::i32 r = clamp((keel::i32)orow + di, 0, (keel::i32)H - 1);
                    const keel::f64 v = I.elemAsDouble((keel::usize)c * H + (keel::usize)r);
                    const keel::f64 dI = v - Ic;
                    const keel::f64 rw = std::exp(-(dI * dI) * inv2dos);
                    const keel::f64 w = sw[(keel::usize)(di + half) * (keel::usize)win
                                          + (keel::usize)(dj + half)] * rw;
                    sum_w += w;
                    sum_v += w * v;
                }
            }
            const keel::f64 j = (sum_w > 0.0) ? sum_v / sum_w : Ic;
            store_classed(out, oc * H + orow, j, I.type());
        }
    }
    return out;
}

// ════════════════════════════════════════════════════════════════════
// col2im — inverse of im2col
// ════════════════════════════════════════════════════════════════════
//
// Two modes mirroring im2col:
//
// "sliding"  — B is 1 × (mm−m+1)·(nn−n+1). Each column of B is a
//              scalar (already-reduced) filter response for one
//              window. Output is just a reshape into (mm−m+1) ×
//              (nn−n+1), column-major. (im2col + correlation per
//              column = col2im(sliding) of the row of dot-products.)
//
// "distinct" — B is m·n × Hb·Wb where Hb=⌈mm/m⌉, Wb=⌈nn/n⌉. Each
//              column was an m×n tile (column-major). Output is mm
//              × nn — we drop the trailing rows/cols of the last
//              row-tile / col-tile, which is where im2col padded
//              with zeros.

namespace {

template <typename T>
void col2im_distinct_ptr(const T *bd, T *ad, keel::usize mm, keel::usize nn,
                         keel::i32 m, keel::i32 n)
{
    const keel::usize Hb = (mm + (keel::usize)m - 1) / (keel::usize)m;
    const keel::usize outH = (keel::usize)(m * n);
    keel::usize col = 0;
    for (keel::usize bj = 0; bj < (nn + (keel::usize)n - 1) / (keel::usize)n; ++bj) {
        for (keel::usize bi = 0; bi < Hb; ++bi) {
            keel::usize off = col * outH;
            for (keel::i32 bc = 0; bc < n; ++bc) {
                const keel::usize cc = bj * (keel::usize)n + (keel::usize)bc;
                if (cc >= nn) { off += (keel::usize)m; continue; }
                const keel::usize Acol = cc * mm;
                for (keel::i32 br = 0; br < m; ++br) {
                    const keel::usize rr = bi * (keel::usize)m + (keel::usize)br;
                    if (rr < mm) ad[Acol + rr] = bd[off];
                    ++off;
                }
            }
            ++col;
        }
    }
}

void col2im_argcheck(keel::i32 m, keel::i32 n, keel::i32 mm, keel::i32 nn, keel::StringRef block_type)
{
    if (m <= 0 || n <= 0 || mm <= 0 || nn <= 0)
        throw Error("col2im: dimensions must be positive",
                    0, 0, "col2im", "", "numkit:col2im:size");
    if (block_type != "sliding" && !block_type.empty() && block_type != "distinct")
        throw Error("col2im: block_type must be 'sliding' or 'distinct'",
                    0, 0, "col2im", "", "numkit:col2im:type");
}

void imnoise_argcheck(const Value &I, keel::StringRef mode)
{
    internal::requireImageClassAllowComplex(I, "imnoise", "I", 1,
                                internal::ImgClasses::FloatInt);
    if (mode != "gaussian" && mode != "salt & pepper" && mode != "salt&pepper" &&
        mode != "speckle" && mode != "poisson" && mode != "localvar")
        throw Error(keel::format("imnoise: unknown mode '{}'", mode),
                    0, 0, "imnoise", "", "numkit:imnoise:mode");
}

} // anonymous

Value col2im(const Value &B, keel::i32 m, keel::i32 n, keel::i32 mm, keel::i32 nn, keel::StringRef block_type, keel::Allocator *mr)
{
    col2im_argcheck(m, n, mm, nn, block_type);
    const keel::usize MM = (keel::usize)mm;
    const keel::usize NN = (keel::usize)nn;
    const ValueType T = B.type();

    bool sliding = (block_type == "sliding" || block_type.empty());
    if (sliding) {
        if ((keel::usize)m > MM || (keel::usize)n > NN)
            throw Error("col2im(sliding): block larger than output image",
                        0, 0, "col2im", "", "numkit:col2im:size");
        const keel::usize Hp = MM - (keel::usize)m + 1;
        const keel::usize Wp = NN - (keel::usize)n + 1;
        if (B.numel() != Hp * Wp)
            throw Error("col2im(sliding): B size mismatch — expected 1×(mm−m+1)·(nn−n+1)",
                        0, 0, "col2im", "", "numkit:col2im:nelems");
        Value A = Value::matrix(Hp, Wp, T, mr);
        // Both A and B are column-major linearised; values flow 1:1.
        const keel::usize N = Hp * Wp;
        switch (T) {
            case ValueType::DOUBLE:
                std::memcpy(A.doubleDataMut(), B.doubleData(),
                            N * sizeof(keel::f64)); break;
            case ValueType::SINGLE:
                std::memcpy(A.singleDataMut(), B.singleData(),
                            N * sizeof(keel::f32)); break;
            case ValueType::UINT8:
                std::memcpy(A.uint8DataMut(), B.uint8Data(), N); break;
            case ValueType::UINT16:
                std::memcpy(A.uint16DataMut(), B.uint16Data(),
                            N * sizeof(keel::u16)); break;
            case ValueType::INT16:
                std::memcpy(A.int16DataMut(), B.int16Data(),
                            N * sizeof(keel::i16)); break;
            case ValueType::LOGICAL:
                std::memcpy(A.logicalDataMut(), B.logicalData(), N); break;
            default:
                throw Error("col2im: unsupported class",
                            0, 0, "col2im", "", "numkit:col2im:badtype");
        }
        return A;
    }

    if (block_type != "distinct")
        throw Error("col2im: block_type must be 'sliding' or 'distinct'",
                    0, 0, "col2im", "", "numkit:col2im:type");

    const keel::usize Hb = (MM + (keel::usize)m - 1) / (keel::usize)m;
    const keel::usize Wb = (NN + (keel::usize)n - 1) / (keel::usize)n;
    const keel::usize expRows = (keel::usize)(m * n);
    const keel::usize expCols = Hb * Wb;
    if (B.dims().rows() != expRows || B.dims().cols() != expCols)
        throw Error("col2im(distinct): B shape mismatch — expected m·n × ⌈mm/m⌉·⌈nn/n⌉",
                    0, 0, "col2im", "", "numkit:col2im:shape");

    Value A = Value::matrix(MM, NN, T, mr);
    if (MM == 0 || NN == 0) return A;

    switch (T) {
        case ValueType::DOUBLE:  col2im_distinct_ptr(B.doubleData(),  A.doubleDataMut(),  MM, NN, m, n); break;
        case ValueType::SINGLE:  col2im_distinct_ptr(B.singleData(),  A.singleDataMut(),  MM, NN, m, n); break;
        case ValueType::UINT8:   col2im_distinct_ptr(B.uint8Data(),   A.uint8DataMut(),   MM, NN, m, n); break;
        case ValueType::UINT16:  col2im_distinct_ptr(B.uint16Data(),  A.uint16DataMut(),  MM, NN, m, n); break;
        case ValueType::INT16:   col2im_distinct_ptr(B.int16Data(),   A.int16DataMut(),   MM, NN, m, n); break;
        case ValueType::LOGICAL: col2im_distinct_ptr(B.logicalData(), A.logicalDataMut(), MM, NN, m, n); break;
        default:
            throw Error("col2im: unsupported class",
                        0, 0, "col2im", "", "numkit:col2im:badtype");
    }
    return A;
}

// ════════════════════════════════════════════════════════════════════
// imnoise — additive / multiplicative / shot noise
// ════════════════════════════════════════════════════════════════════
//
// All modes operate on the unit-range [0, 1] interpretation of I:
//   integer classes are scaled by their max (255 for uint8, 65535 for
//   uint16, 32768±32767 for int16); double/single are taken as-is.
// Output is saturated back to the input class.
//
// Modes (matching MATLAB R2025b imnoise semantics):
//   gaussian       J = I + m + sqrt(var) * N(0, 1)
//   localvar       J = I + sqrt(V[x]) * N(0, 1)         V is variance map
//   salt & pepper  fraction d / 2 of pixels each set to 0 and 1
//   speckle        J = I + I * sqrt(var) * N(0, 1)      multiplicative
//   poisson        J = Poisson(I * scale) / scale       scale per class

Value imnoise(::numkit::ops::RngContext &rng, const Value &I, keel::StringRef mode, const Value &p1, const Value &p2, keel::Allocator *mr)
{
    imnoise_argcheck(I, mode);
    const ValueType T = I.type();
    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    const keel::usize N = I.numel();

    if (I.isComplex()) {
        const auto *cd = I.complexData();
        if (mode == "salt & pepper" || mode == "salt&pepper") {
            const keel::f64 d = (p1.numel() > 0) ? p1.toScalar() : 0.05;
            Value out = Value::matrix(H, W, ValueType::COMPLEX, mr);
            auto *od = out.complexDataMut();
            for (keel::usize i = 0; i < N; ++i) {
                keel::c64 z = cd[i];
                const keel::f64 u = keel::random::uniform_f64(rng);
                if      (u < d * 0.5) z = keel::c64(0.0, 0.0);
                else if (u < d)       z = keel::c64(1.0, 0.0);
                od[i] = z;
            }
            return out;
        }
        Value Ir = Value::matrix(H, W, ValueType::DOUBLE, mr);
        keel::f64 *ird = Ir.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i) ird[i] = cd[i].real();
        Value Jr = imnoise(rng, Ir, mode, p1, p2, mr);
        Value out = Value::matrix(H, W, ValueType::COMPLEX, mr);
        auto *od = out.complexDataMut();
        const keel::f64 *jrd = Jr.doubleData();
        for (keel::usize i = 0; i < N; ++i)
            od[i] = keel::c64(jrd[i], cd[i].imag());
        return out;
    }

    auto toUnit = [&](keel::f64 v) -> keel::f64 {
        switch (T) {
            case ValueType::UINT8:   return v / 255.0;
            case ValueType::UINT16:  return v / 65535.0;
            case ValueType::INT16:   return (v + 32768.0) / 65535.0;
            case ValueType::LOGICAL: return v != 0.0 ? 1.0 : 0.0;
            default:                 return v;
        }
    };
    auto fromUnit = [&](Value &out, keel::usize i, keel::f64 v) {
        if (v < 0.0) v = 0.0;
        if (v > 1.0) v = 1.0;
        switch (T) {
            case ValueType::UINT8:
                out.uint8DataMut()[i] =
                    (keel::u8)std::lround(v * 255.0); break;
            case ValueType::UINT16:
                out.uint16DataMut()[i] =
                    (keel::u16)std::lround(v * 65535.0); break;
            case ValueType::INT16: {
                keel::f64 w = std::lround(v * 65535.0) - 32768.0;
                if (w < -32768.0) w = -32768.0;
                if (w >  32767.0) w =  32767.0;
                out.int16DataMut()[i] = (keel::i16)w; break;
            }
            case ValueType::LOGICAL:
                out.logicalDataMut()[i] = v >= 0.5 ? 1u : 0u; break;
            case ValueType::SINGLE:
                out.singleDataMut()[i] = (keel::f32)v; break;
            default:
                out.doubleDataMut()[i] = v; break;
        }
    };

    Value out = Value::matrix(H, W, T, mr);


    if (mode == "gaussian") {
        const keel::f64 m = (p1.numel() > 0) ? p1.toScalar() : 0.0;
        const keel::f64 v = (p2.numel() > 0) ? p2.toScalar() : 0.01;
        const keel::f64 sigma = std::sqrt(std::max(v, 0.0));
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 x = toUnit(I.elemAsDouble(i));
            fromUnit(out, i, x + m + keel::random::normal_f64(rng, 0.0, sigma));
        }
    }
    else if (mode == "salt & pepper" || mode == "salt&pepper") {
        const keel::f64 d = (p1.numel() > 0) ? p1.toScalar() : 0.05;
        for (keel::usize i = 0; i < N; ++i) {
            keel::f64 x = toUnit(I.elemAsDouble(i));
            const keel::f64 u = keel::random::uniform_f64(rng);
            if      (u < d * 0.5) x = 0.0;       // pepper
            else if (u < d)       x = 1.0;       // salt
            fromUnit(out, i, x);
        }
    }
    else if (mode == "speckle") {
        const keel::f64 v = (p1.numel() > 0) ? p1.toScalar() : 0.04;
        const keel::f64 sigma = std::sqrt(std::max(v, 0.0));
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 x = toUnit(I.elemAsDouble(i));
            fromUnit(out, i, x + x * keel::random::normal_f64(rng, 0.0, sigma));
        }
    }
    else if (mode == "poisson") {
        // Scale per class: integer classes use their max as the count
        // unit; double/single use a large fixed scale so noise is small.
        keel::f64 scale;
        switch (T) {
            case ValueType::UINT8:   scale = 255.0;   break;
            case ValueType::UINT16:  scale = 65535.0; break;
            default:                 scale = 1e12;    break;
        }
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 x = toUnit(I.elemAsDouble(i));
            const keel::f64 mean = std::max(x * scale, 0.0);
            const keel::u64 k = keel::random::poisson_u64(rng, mean);
            fromUnit(out, i, keel::as<keel::f64>(k) / scale);
        }
    }
    else if (mode == "localvar") {
        if (p1.numel() != N)
            throw Error("imnoise('localvar', V): V must match I in size",
                        0, 0, "imnoise", "", "numkit:imnoise:localvar");
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 x = toUnit(I.elemAsDouble(i));
            const keel::f64 v = std::max(p1.elemAsDouble(i), 0.0);
            fromUnit(out, i, x + keel::random::normal_f64(rng, 0.0, std::sqrt(v)));
        }
    }
    else {
        throw Error(keel::format("imnoise: unknown mode '{}'", mode),
                    0, 0, "imnoise", "", "numkit:imnoise:mode");
    }
    return out;
}

namespace {

// Build a list of (dr, dc) offsets for the non-zero entries of a
// 2-D `domain` mask, plus the kernel half-extents.
struct DomainOffsets {
    keel::Array<std::pair<keel::i32, keel::i32>> offs;
    keel::i32 rh = 0, ch = 0;
};

DomainOffsets domain_offsets(const Value &domain) {
    DomainOffsets r;
    const keel::usize kH = domain.dims().rows();
    const keel::usize kW = domain.dims().cols();
    r.rh = keel::as<keel::i32>(kH) / 2;
    r.ch = keel::as<keel::i32>(kW) / 2;
    r.offs.reserve(kH * kW);
    for (keel::usize c = 0; c < kW; ++c)
        for (keel::usize row = 0; row < kH; ++row)
            if (domain.elemAsDouble(c * kH + row) != 0.0)
                r.offs.push_back({keel::as<keel::i32>(row) - r.rh,
                                  keel::as<keel::i32>(c)   - r.ch});
    return r;
}

inline keel::f64 sample_sym(const Value &I, keel::i32 r, keel::i32 c, keel::i32 H, keel::i32 W) {
    r = fold_index(r, H, PadMode::Symmetric);
    c = fold_index(c, W, PadMode::Symmetric);
    return I.elemAsDouble(keel::as<keel::usize>(c) *
                          keel::as<keel::usize>(H) +
                          keel::as<keel::usize>(r));
}

void stdfilt_argcheck(const Value &I)
{
    internal::requireImageClass(I, "stdfilt", "I", 1,
                                internal::ImgClasses::FloatIntLog);
}

void ordfilt2_argcheck(const Value &A, const Value &domain)
{
    internal::requireImageClass(A, "ordfilt2", "A", 1,
                                internal::ImgClasses::FloatIntLog);
    internal::requireImageClass(domain, "ordfilt2", "domain", 3,
                                internal::ImgClasses::AnyNumericReal);
}

void imsmooth_argcheck(const Value &I, keel::StringRef name, keel::f64 sigma)
{
    internal::requireImageClass(I, "imsmooth", "I", 1,
                                internal::ImgClasses::FloatIntLog);
    if (!name.equals_ignore_ascii_case("gaussian"))
        throw Error("imsmooth: only 'Gaussian' mode is implemented",
                    0, 0, "imsmooth", "", "numkit:imsmooth:mode");
    if (!(sigma > 0.0))
        throw Error("imsmooth: sigma must be positive",
                    0, 0, "imsmooth", "", "numkit:imsmooth:sigma");
}

void imboxfilt3_argcheck(const Value &V)
{
    internal::requireImageClass(V, "imboxfilt3", "A", 1,
                                internal::ImgClasses::FloatInt);
}

void imgaussfilt3_argcheck(const Value &V)
{
    internal::requireImageClass(V, "imgaussfilt3", "A", 1,
                                internal::ImgClasses::FloatInt);
}

void medfilt3_argcheck(const Value &V, keel::i32 M, keel::i32 N, keel::i32 P)
{
    internal::requireImageClass(V, "medfilt3", "A", 1,
                                internal::ImgClasses::FloatIntLog);
    const keel::i32 m = (M <= 0) ? 3 : M;
    const keel::i32 n = (N <= 0) ? 3 : N;
    const keel::i32 p = (P <= 0) ? 3 : P;
    if (m % 2 == 0 || n % 2 == 0 || p % 2 == 0)
        throw Error("medfilt3: filter sizes must be odd",
                    0, 0, "medfilt3", "", "numkit:medfilt3:size");
}

void convmtx2_argcheck(const Value &h, keel::i32 m, keel::i32 n)
{
    if (h.dims().is3D())
        throw Error("convmtx2: kernel must be 2-D",
                    0, 0, "convmtx2", "", "numkit:convmtx2:dims");
    if (m <= 0 || n <= 0)
        throw Error("convmtx2: m, n must be positive integers",
                    0, 0, "convmtx2", "", "numkit:convmtx2:size");
}

void entropyfilt_argcheck(const Value &I)
{
    internal::requireImageClass(I, "entropyfilt", "I", 1,
                                internal::ImgClasses::FloatIntNoI16);
    internal::requireNonemptyImage(I, "entropyfilt", "I", 1);
}

void rangefilt_argcheck(const Value &I)
{
    internal::requireImageClass(I, "rangefilt", "I", 1,
                                internal::ImgClasses::FloatIntLog);
}

void freqz2_argcheck(const Value &h)
{
    if (h.dims().is3D())
        throw Error("freqz2: kernel must be 2-D",
                    0, 0, "freqz2", "", "numkit:freqz2:dims");
}

void roifilt2_combine_argcheck(const Value &I, const Value &filtRes, const Value &BW)
{
    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    if (BW.dims().rows() != H || BW.dims().cols() != W)
        throw Error("roifilt2: I and BW must be the same size",
                    0, 0, "roifilt2", "", "numkit:roifilt2:imageMaskSizeMismatch");
    if (filtRes.dims().rows() != H || filtRes.dims().cols() != W)
        throw Error("roifilt2: filtered result size mismatch",
                    0, 0, "roifilt2", "", "numkit:roifilt2:imageSizeMismatch");
}

void roifilt2_argcheck(const Value &h, const Value &I)
{
    internal::requireImageClassAllowComplex(I, "roifilt2", "I", 2,
                                internal::ImgClasses::FloatInt);
    internal::requireImageClassAllowComplex(h, "roifilt2", "h", 1,
                                internal::ImgClasses::DoubleOnly);
    if (I.dims().ndim() > 2)
        throw Error("roifilt2: I must be a 2-D image",
                    0, 0, "roifilt2", "", "numkit:roifilt2:imageMustBe2D");
}

void imguidedfilter_argcheck(const Value &A, keel::i32 nhood)
{
    internal::requireImageClass(A, "imguidedfilter", "A", 1,
                                internal::ImgClasses::FloatIntLog);
    if (A.dims().is3D())
        throw Error("imguidedfilter: A must be 2-D",
                    0, 0, "imguidedfilter", "",
                    "numkit:imguidedfilter:dim");
    if (nhood < 1 || (nhood % 2) == 0)
        throw Error("imguidedfilter: NeighborhoodSize must be positive odd "
                    "integer",
                    0, 0, "imguidedfilter", "",
                    "numkit:imguidedfilter:nhood");
}

void imdiffusefilt_argcheck(const Value &I, keel::StringRef connectivity,
                            keel::StringRef conduction)
{
    internal::requireImageClass(I, "imdiffusefilt", "I", 1,
                                internal::ImgClasses::FloatInt);
    if (I.dims().is3D())
        throw Error("imdiffusefilt: 3-D inputs not yet supported",
                    0, 0, "imdiffusefilt", "",
                    "numkit:imdiffusefilt:dim");
    if (connectivity != "maximal" && connectivity != "minimal")
        throw Error("imdiffusefilt: Connectivity must be 'maximal' or "
                    "'minimal'",
                    0, 0, "imdiffusefilt", "",
                    "numkit:imdiffusefilt:conn");
    if (conduction != "exponential" && conduction != "quadratic")
        throw Error("imdiffusefilt: ConductionMethod must be "
                    "'exponential' or 'quadratic'",
                    0, 0, "imdiffusefilt", "",
                    "numkit:imdiffusefilt:cond");
}

void imgaborfilt_argcheck(const Value &A_in, keel::f64 wavelength, keel::f64 sfb, keel::f64 aspect)
{
    if (A_in.dims().is3D())
        throw Error("imgaborfilt: A must be 2-D",
                    0, 0, "imgaborfilt", "", "numkit:imgaborfilt:dim");
    if (!(wavelength >= 2.0))
        throw Error("imgaborfilt: wavelength must be >= 2",
                    0, 0, "imgaborfilt", "", "numkit:imgaborfilt:wavelength");
    if (!(sfb > 0))
        throw Error("imgaborfilt: SpatialFrequencyBandwidth must be > 0",
                    0, 0, "imgaborfilt", "", "numkit:imgaborfilt:sfb");
    if (!(aspect > 0))
        throw Error("imgaborfilt: SpatialAspectRatio must be > 0",
                    0, 0, "imgaborfilt", "", "numkit:imgaborfilt:aspect");
}

} // anonymous

Value stdfilt(const Value &I, const Value &domain, keel::Allocator *mr)
{
    stdfilt_argcheck(I);
    const keel::i32 H = keel::as<keel::i32>(I.dims().rows());
    const keel::i32 W = keel::as<keel::i32>(I.dims().cols());
    Value out = Value::matrix((keel::usize)H, (keel::usize)W, ValueType::DOUBLE, mr);
    if (H == 0 || W == 0) return out;

    Value dom_local;
    const Value *dom = &domain;
    if (domain.numel() == 0) {
        dom_local = Value::matrix(3, 3, ValueType::LOGICAL, mr);
        for (keel::usize i = 0; i < 9; ++i) dom_local.logicalDataMut()[i] = 1;
        dom = &dom_local;
    }

    // INTEGRAL-IMAGE fast path for a RECTANGULAR neighbourhood (the default
    // ones(3) and any full ones(m,n)): two summed-area tables (Σx and Σx²)
    // over the symmetric-padded image give per-window sum & sum-of-squares in
    // O(1)/px, then var = (Σx² − (Σx)²/n)/(n−1). This is the one-pass form
    // MATLAB's stdfilt also uses, so values match (the per-offset loop below
    // — two-pass — is kept for non-rectangular neighbourhoods).
    {
        const keel::i32 kH = (keel::i32)dom->dims().rows(), kW = (keel::i32)dom->dims().cols();
        bool rect = (kH > 0 && kW > 0);
        for (keel::usize i = 0; rect && i < dom->numel(); ++i)
            if (dom->elemAsDouble(i) == 0.0) rect = false;
        if (rect) {
            const keel::i32 rh = kH / 2, ch = kW / 2;     // match domain_offsets centring
            const keel::f64 n = (keel::f64)kH * (keel::f64)kW;
            const keel::i32 Hp = H + kH - 1, Wp = W + kW - 1;
            const keel::i32 Hs = Hp + 1, Ws = Wp + 1;
            keel::ArenaTemp arena = keel::get_scratch(mr);
            keel::Array<keel::f64> Sx((keel::usize)Hs * (keel::usize)Ws, arena.allocator());
            keel::Array<keel::f64> Sxx((keel::usize)Hs * (keel::usize)Ws, arena.allocator());
            auto X  = [&](keel::i32 r, keel::i32 c) -> keel::f64 & { return Sx[(keel::usize)c * (keel::usize)Hs + (keel::usize)r]; };
            auto XX = [&](keel::i32 r, keel::i32 c) -> keel::f64 & { return Sxx[(keel::usize)c * (keel::usize)Hs + (keel::usize)r]; };
            // Typed source read (no per-element elemAsDouble dispatch) for the
            // common image classes; integer values are exact in a double SAT.
            const keel::u8  *su8 = (I.type() == ValueType::UINT8)  ? I.uint8Data()  : nullptr;
            const keel::f64        *sdd = (I.type() == ValueType::DOUBLE) ? I.doubleData() : nullptr;
            for (keel::i32 c = 0; c < Ws; ++c) { X(0, c) = 0.0; XX(0, c) = 0.0; }
            for (keel::i32 r = 0; r < Hs; ++r) { X(r, 0) = 0.0; XX(r, 0) = 0.0; }
            for (keel::i32 c = 1; c < Ws; ++c) {
                const keel::i32 sc = fold_index((c - 1) - ch, W, PadMode::Symmetric);
                const keel::usize scol = (keel::usize)sc * (keel::usize)H;
                for (keel::i32 r = 1; r < Hs; ++r) {
                    const keel::i32 sr = fold_index((r - 1) - rh, H, PadMode::Symmetric);
                    const keel::usize idx = scol + (keel::usize)sr;
                    const keel::f64 p = su8 ? (keel::f64)su8[idx] : sdd ? sdd[idx] : I.elemAsDouble(idx);
                    X(r, c)  = p       + X(r - 1, c)  + X(r, c - 1)  - X(r - 1, c - 1);
                    XX(r, c) = p * p   + XX(r - 1, c) + XX(r, c - 1) - XX(r - 1, c - 1);
                }
            }
            keel::f64 *od2 = out.doubleDataMut();
            for (keel::i32 c = 0; c < W; ++c)
                for (keel::i32 r = 0; r < H; ++r) {
                    const keel::f64 s  = X(r + kH, c + kW)  - X(r, c + kW)  - X(r + kH, c)  + X(r, c);
                    const keel::f64 s2 = XX(r + kH, c + kW) - XX(r, c + kW) - XX(r + kH, c) + XX(r, c);
                    keel::f64 var = (n > 1.0) ? (s2 - s * s / n) / (n - 1.0) : 0.0;
                    if (var < 0.0) var = 0.0;   // clamp tiny negatives from FP cancellation
                    od2[(keel::usize)c * (keel::usize)H + (keel::usize)r] = std::sqrt(var);
                }
            return out;
        }
    }

    const DomainOffsets dom_offs = domain_offsets(*dom);
    const keel::usize M = dom_offs.offs.size();
    if (M == 0) return out;

    keel::f64 *od = out.doubleDataMut();
    for (keel::i32 c = 0; c < W; ++c) {
        for (keel::i32 r = 0; r < H; ++r) {
            keel::f64 sum = 0.0;
            for (keel::usize k = 0; k < M; ++k)
                sum += sample_sym(I, r + dom_offs.offs[k].first,
                                     c + dom_offs.offs[k].second, H, W);
            const keel::f64 mu = sum / keel::as<keel::f64>(M);
            keel::f64 v = 0.0;
            for (keel::usize k = 0; k < M; ++k) {
                const keel::f64 d = sample_sym(I,
                                            r + dom_offs.offs[k].first,
                                            c + dom_offs.offs[k].second,
                                            H, W) - mu;
                v += d * d;
            }
            od[(keel::usize)c * (keel::usize)H + (keel::usize)r] =
                (M > 1) ? std::sqrt(v / keel::as<keel::f64>(M - 1)) : 0.0;
        }
    }
    return out;
}

Value ordfilt2(const Value &A, keel::i32 nth, const Value &domain, const Value &S, PadMode boundary, keel::f64 pad_value, keel::Allocator *mr)
{
    ordfilt2_argcheck(A, domain);
    // Logical input → logical output (probed R2025b): order-statistic on
    // the double copy, re-binarise.
    if (A.type() == ValueType::LOGICAL)
        return internal::castResultToClass(
            ordfilt2(internal::coerceToDoubleImage(A, mr), nth, domain, S,
                     boundary, pad_value, mr),
            ValueType::LOGICAL, mr);
    const keel::i32 H = keel::as<keel::i32>(A.dims().rows());
    const keel::i32 W = keel::as<keel::i32>(A.dims().cols());
    Value out = Value::matrix((keel::usize)H, (keel::usize)W, A.type(), mr);
    if (H == 0 || W == 0) return out;

    const keel::usize kH = domain.dims().rows();
    const keel::usize kW = domain.dims().cols();
    const keel::i32 rh = keel::as<keel::i32>(kH) / 2;
    const keel::i32 ch = keel::as<keel::i32>(kW) / 2;
    const bool has_S = (S.numel() == domain.numel());

    struct Off { keel::i32 dr; keel::i32 dc; keel::f64 s; };
    keel::Array<Off> offs;
    offs.reserve(kH * kW);
    for (keel::usize c = 0; c < kW; ++c)
        for (keel::usize r = 0; r < kH; ++r)
            if (domain.elemAsDouble(c * kH + r) != 0.0) {
                const keel::f64 s = has_S ? S.elemAsDouble(c * kH + r) : 0.0;
                offs.push_back({keel::as<keel::i32>(r) - rh,
                                keel::as<keel::i32>(c) - ch, s});
            }
    const keel::i32 M = keel::as<keel::i32>(offs.size());
    if (M == 0) return out;
    if (nth < 1 || nth > M)
        throw Error("ordfilt2: nth-order index out of range",
                    0, 0, "ordfilt2", "", "numkit:ordfilt2:nth");

    auto sample = [&](keel::i32 r, keel::i32 c) -> keel::f64 {
        if (boundary == PadMode::Constant) {
            if (r < 0 || c < 0 || r >= H || c >= W) return pad_value;
        } else {
            r = fold_index(r, H, boundary);
            c = fold_index(c, W, boundary);
        }
        return A.elemAsDouble((keel::usize)c * (keel::usize)H + (keel::usize)r);
    };

    keel::Array<keel::f64> nb(M);
    for (keel::i32 c = 0; c < W; ++c) {
        for (keel::i32 r = 0; r < H; ++r) {
            for (keel::i32 k = 0; k < M; ++k) {
                nb[(keel::usize)k] = sample(r + offs[(keel::usize)k].dr,
                                       c + offs[(keel::usize)k].dc)
                                + offs[(keel::usize)k].s;
            }
            std::nth_element(nb.begin(),
                             nb.begin() + (nth - 1),
                             nb.end());
            store_classed(out, (keel::usize)c * (keel::usize)H + (keel::usize)r,
                          nb[(keel::usize)(nth - 1)], A.type());
        }
    }
    return out;
}

std::tuple<Value, Value>
wiener2(const Value &I, keel::usize nh, keel::usize nw, keel::f64 noise, keel::Allocator *mr)
{
    const ValueType cls = I.type();
    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    if (H == 0 || W == 0) {
        Value out = Value::matrix(H, W, cls, mr);
        return {std::move(out), Value::scalar(0.0, mr)};
    }

    // Promote to double in [0, 1] for processing.
    Value Id = (cls == ValueType::DOUBLE)
        ? I
        : (cls == ValueType::SINGLE
           ? im2double(I, mr)
           : im2double(I, mr));
    if (Id.type() != ValueType::DOUBLE)
        Id = im2double(Id, mr);

    if (nh == 0) nh = 3;
    if (nw == 0) nw = 3;

    // Build h×w box kernel.
    Value k = Value::matrix(nh, nw, ValueType::DOUBLE, mr);
    {
        const keel::f64 a = 1.0 / keel::as<keel::f64>(nh * nw);
        keel::f64 *kd = k.doubleDataMut();
        for (keel::usize i = 0; i < nh * nw; ++i) kd[i] = a;
    }

    // Squared image.
    Value Id_sq = Value::matrix(H, W, ValueType::DOUBLE, mr);
    {
        const keel::f64 *id = Id.doubleData();
        keel::f64 *od = Id_sq.doubleDataMut();
        for (keel::usize i = 0; i < H * W; ++i) od[i] = id[i] * id[i];
    }

    // Local mean and mean-of-squares via zero-pad conv2 'same'.
    Value mean_im = signal::conv2(Id, k, "same", mr);
    Value mean_sq = signal::conv2(Id_sq, k, "same", mr);

    // variance_im = mean_sq - mean_im^2.
    Value var_im = Value::matrix(H, W, ValueType::DOUBLE, mr);
    {
        keel::f64 *vd = var_im.doubleDataMut();
        const keel::f64 *m  = mean_im.doubleData();
        const keel::f64 *ms = mean_sq.doubleData();
        for (keel::usize i = 0; i < H * W; ++i)
            vd[i] = ms[i] - m[i] * m[i];
    }

    // Estimate noise as mean of variance if not supplied.
    if (std::isnan(noise)) {
        long double s = 0.0L; // keel-lint: ignore
        for (keel::usize i = 0; i < H * W; ++i)
            s += var_im.doubleData()[i];
        noise = keel::as<keel::f64>(s / keel::as<long double>(H * W)); // keel-lint: ignore
    }

    // var_orig = max(0, var_im - noise).
    // out = mean + var_orig / (var_orig + noise) * (Id - mean).
    Value out_d = Value::matrix(H, W, ValueType::DOUBLE, mr);
    {
        const keel::f64 *id = Id.doubleData();
        const keel::f64 *m  = mean_im.doubleData();
        keel::f64 *vd       = var_im.doubleDataMut();
        keel::f64 *od       = out_d.doubleDataMut();
        for (keel::usize i = 0; i < H * W; ++i) {
            keel::f64 vo = vd[i] - noise;
            if (vo < 0.0) vo = 0.0;
            const keel::f64 denom = vo + noise;
            const keel::f64 w = (denom > 0.0) ? vo / denom : 0.0;
            od[i] = m[i] + w * (id[i] - m[i]);
        }
    }

    // Cast back to input class.
    Value out;
    if (cls == ValueType::DOUBLE) out = std::move(out_d);
    else if (cls == ValueType::SINGLE) {
        out = Value::matrix(H, W, ValueType::SINGLE, mr);
        const keel::f64 *o = out_d.doubleData();
        keel::f32 *of = out.singleDataMut();
        for (keel::usize i = 0; i < H * W; ++i) of[i] = keel::as<keel::f32>(o[i]);
    } else if (cls == ValueType::UINT8) out = im2uint8(out_d, mr);
    else if (cls == ValueType::UINT16) out = im2uint16(out_d, mr);
    else if (cls == ValueType::INT16)  out = im2int16(out_d, mr);
    else                                out = std::move(out_d);

    return {std::move(out), Value::scalar(noise, mr)};
}

Value imsmooth(const Value &I, keel::StringRef name, keel::f64 sigma, keel::Allocator *mr)
{
    imsmooth_argcheck(I, name, sigma);

    const ValueType cls = I.type();
    const keel::i32 H = keel::as<keel::i32>(I.dims().rows());
    const keel::i32 W = keel::as<keel::i32>(I.dims().cols());
    const keel::i32 P = I.dims().is3D() ? keel::as<keel::i32>(I.dims().pages()) : 1;
    const keel::i32 h = keel::as<keel::i32>(std::ceil(3.0 * sigma));
    const keel::i32 K = 2 * h + 1;

    keel::Array<keel::f64> f((keel::usize)K);
    keel::f64 s = 0.0;
    for (keel::i32 i = -h; i <= h; ++i) {
        f[(keel::usize)(i + h)] = std::exp(-(keel::f64)i * i / (2.0 * sigma * sigma));
        s += f[(keel::usize)(i + h)];
    }
    for (keel::i32 i = 0; i < K; ++i) f[(keel::usize)i] /= s;

    Value out = (P > 1) ? Value::matrix3d(H, W, P, cls, mr)
                        : Value::matrix(H, W, cls, mr);
    if (H == 0 || W == 0) return out;

    for (keel::i32 p = 0; p < P; ++p) {
        keel::Array<keel::f64> tmp((keel::usize)H * (keel::usize)W, 0.0);
        for (keel::i32 r = 0; r < H; ++r) {
            for (keel::i32 c = 0; c < W; ++c) {
                keel::f64 acc = 0.0;
                for (keel::i32 k = -h; k <= h; ++k) {
                    const keel::i32 cs = fold_index(c + k, W, PadMode::Symmetric);
                    const keel::usize srcIdx =
                        (keel::usize)p * (keel::usize)H * (keel::usize)W +
                        (keel::usize)cs * (keel::usize)H + (keel::usize)r;
                    acc += f[(keel::usize)(k + h)] * I.elemAsDouble(srcIdx);
                }
                tmp[(keel::usize)c * (keel::usize)H + (keel::usize)r] = acc;
            }
        }
        for (keel::i32 c = 0; c < W; ++c) {
            for (keel::i32 r = 0; r < H; ++r) {
                keel::f64 acc = 0.0;
                for (keel::i32 k = -h; k <= h; ++k) {
                    const keel::i32 rs = fold_index(r + k, H, PadMode::Symmetric);
                    acc += f[(keel::usize)(k + h)] *
                           tmp[(keel::usize)c * (keel::usize)H + (keel::usize)rs];
                }
                store_classed(out,
                              (keel::usize)p * (keel::usize)H * (keel::usize)W +
                                  (keel::usize)c * (keel::usize)H + (keel::usize)r,
                              acc, cls);
            }
        }
    }
    return out;
}

// ════════════════════════════════════════════════════════════════════
// imboxfilt3 — 3-D box (mean) filter, separable + replicate boundary
// ════════════════════════════════════════════════════════════════════
Value imboxfilt3(const Value &V, keel::i32 fH, keel::i32 fW, keel::i32 fP, keel::Allocator *mr)
{
    imboxfilt3_argcheck(V);
    if (fH <= 0) fH = 3;
    if (fW <= 0) fW = 3;
    if (fP <= 0) fP = 3;
    if (fH % 2 == 0) fH += 1;
    if (fW % 2 == 0) fW += 1;
    if (fP % 2 == 0) fP += 1;

    const ValueType cls = V.type();
    const keel::i32 H = keel::as<keel::i32>(V.dims().rows());
    const keel::i32 W = keel::as<keel::i32>(V.dims().cols());
    const keel::i32 P = V.dims().is3D() ? keel::as<keel::i32>(V.dims().pages()) : 1;
    const keel::i32 hh = fH / 2;
    const keel::i32 hw = fW / 2;
    const keel::i32 hp = fP / 2;

    Value out = (P > 1) ? Value::matrix3d(H, W, P, cls, mr)
                        : Value::matrix(H, W, cls, mr);
    if (H == 0 || W == 0 || P == 0) return out;

    const keel::usize total = (keel::usize)H * (keel::usize)W * (keel::usize)P;
    keel::Array<keel::f64> a(total), b(total);
    // Load V into a (double).
    for (keel::usize i = 0; i < total; ++i) a[i] = V.elemAsDouble(i);

    // Pass 1: average along rows (H axis).
    for (keel::i32 p = 0; p < P; ++p) {
        for (keel::i32 c = 0; c < W; ++c) {
            for (keel::i32 r = 0; r < H; ++r) {
                keel::f64 acc = 0.0;
                for (keel::i32 k = -hh; k <= hh; ++k) {
                    const keel::i32 rs = fold_index(r + k, H, PadMode::Replicate);
                    acc += a[(keel::usize)p * (keel::usize)H * (keel::usize)W +
                              (keel::usize)c * (keel::usize)H + (keel::usize)rs];
                }
                b[(keel::usize)p * (keel::usize)H * (keel::usize)W +
                  (keel::usize)c * (keel::usize)H + (keel::usize)r] = acc / (keel::f64)fH;
            }
        }
    }
    // Pass 2: average along cols (W axis).
    for (keel::i32 p = 0; p < P; ++p) {
        for (keel::i32 r = 0; r < H; ++r) {
            for (keel::i32 c = 0; c < W; ++c) {
                keel::f64 acc = 0.0;
                for (keel::i32 k = -hw; k <= hw; ++k) {
                    const keel::i32 cs = fold_index(c + k, W, PadMode::Replicate);
                    acc += b[(keel::usize)p * (keel::usize)H * (keel::usize)W +
                              (keel::usize)cs * (keel::usize)H + (keel::usize)r];
                }
                a[(keel::usize)p * (keel::usize)H * (keel::usize)W +
                  (keel::usize)c * (keel::usize)H + (keel::usize)r] = acc / (keel::f64)fW;
            }
        }
    }
    // Pass 3: average along pages (P axis).
    for (keel::i32 p = 0; p < P; ++p) {
        for (keel::i32 c = 0; c < W; ++c) {
            for (keel::i32 r = 0; r < H; ++r) {
                keel::f64 acc = 0.0;
                for (keel::i32 k = -hp; k <= hp; ++k) {
                    const keel::i32 ps = fold_index(p + k, P, PadMode::Replicate);
                    acc += a[(keel::usize)ps * (keel::usize)H * (keel::usize)W +
                              (keel::usize)c * (keel::usize)H + (keel::usize)r];
                }
                store_classed(out,
                              (keel::usize)p * (keel::usize)H * (keel::usize)W +
                                  (keel::usize)c * (keel::usize)H + (keel::usize)r,
                              acc / (keel::f64)fP, cls);
            }
        }
    }
    return out;
}

// 1-D Gaussian kernel of length 2*ceil(2σ)+1, normalised.
static keel::Array<keel::f64> gauss1d_kernel(keel::f64 sigma) {
    if (!(sigma > 0.0)) sigma = 0.5;
    keel::i32 half = keel::as<keel::i32>(std::ceil(2.0 * sigma));
    if (half < 1) half = 1;
    const keel::i32 K = 2 * half + 1;
    keel::Array<keel::f64> k(K);
    keel::f64 sum = 0.0;
    const keel::f64 inv2s2 = 0.5 / (sigma * sigma);
    for (keel::i32 i = -half; i <= half; ++i) {
        const keel::f64 w = std::exp(-keel::as<keel::f64>(i * i) * inv2s2);
        k[keel::as<keel::usize>(i + half)] = w;
        sum += w;
    }
    const keel::f64 inv = 1.0 / sum;
    for (keel::f64 &x : k) x *= inv;
    return k;
}

Value imgaussfilt3(const Value &V, keel::f64 sigH, keel::f64 sigW, keel::f64 sigP, keel::Allocator *mr)
{
    imgaussfilt3_argcheck(V);
    const ValueType cls = V.type();
    const keel::i32 H = keel::as<keel::i32>(V.dims().rows());
    const keel::i32 W = keel::as<keel::i32>(V.dims().cols());
    const keel::i32 P = V.dims().is3D() ? keel::as<keel::i32>(V.dims().pages()) : 1;
    Value out = (P > 1) ? Value::matrix3d(H, W, P, cls, mr)
                        : Value::matrix(H, W, cls, mr);
    if (H == 0 || W == 0 || P == 0) return out;

    const auto kH = gauss1d_kernel(sigH);
    const auto kW = gauss1d_kernel(sigW);
    const auto kP = gauss1d_kernel(sigP);
    const keel::i32 hH = keel::as<keel::i32>(kH.size()) / 2;
    const keel::i32 hW = keel::as<keel::i32>(kW.size()) / 2;
    const keel::i32 hP = keel::as<keel::i32>(kP.size()) / 2;

    const keel::usize total = keel::as<keel::usize>(H)
                       * keel::as<keel::usize>(W)
                       * keel::as<keel::usize>(P);
    keel::Array<keel::f64> a(total), b(total);
    for (keel::usize i = 0; i < total; ++i) a[i] = V.elemAsDouble(i);

    // Pass 1: along H axis.
    for (keel::i32 p = 0; p < P; ++p) {
        for (keel::i32 c = 0; c < W; ++c) {
            for (keel::i32 r = 0; r < H; ++r) {
                keel::f64 acc = 0.0;
                for (keel::i32 k = -hH; k <= hH; ++k) {
                    const keel::i32 rs = fold_index(r + k, H, PadMode::Replicate);
                    acc += kH[keel::as<keel::usize>(k + hH)] *
                           a[(keel::usize)p * H * W + (keel::usize)c * H + (keel::usize)rs];
                }
                b[(keel::usize)p * H * W + (keel::usize)c * H + (keel::usize)r] = acc;
            }
        }
    }
    // Pass 2: along W axis.
    for (keel::i32 p = 0; p < P; ++p) {
        for (keel::i32 r = 0; r < H; ++r) {
            for (keel::i32 c = 0; c < W; ++c) {
                keel::f64 acc = 0.0;
                for (keel::i32 k = -hW; k <= hW; ++k) {
                    const keel::i32 cs = fold_index(c + k, W, PadMode::Replicate);
                    acc += kW[keel::as<keel::usize>(k + hW)] *
                           b[(keel::usize)p * H * W + (keel::usize)cs * H + (keel::usize)r];
                }
                a[(keel::usize)p * H * W + (keel::usize)c * H + (keel::usize)r] = acc;
            }
        }
    }
    // Pass 3: along P axis.
    for (keel::i32 p = 0; p < P; ++p) {
        for (keel::i32 c = 0; c < W; ++c) {
            for (keel::i32 r = 0; r < H; ++r) {
                keel::f64 acc = 0.0;
                for (keel::i32 k = -hP; k <= hP; ++k) {
                    const keel::i32 ps = fold_index(p + k, P, PadMode::Replicate);
                    acc += kP[keel::as<keel::usize>(k + hP)] *
                           a[(keel::usize)ps * H * W + (keel::usize)c * H + (keel::usize)r];
                }
                store_classed(out,
                              (keel::usize)p * H * W + (keel::usize)c * H + (keel::usize)r,
                              acc, cls);
            }
        }
    }
    return out;
}

Value medfilt3(const Value &V, keel::i32 M, keel::i32 N, keel::i32 P, keel::Allocator *mr)
{
    medfilt3_argcheck(V, M, N, P);
    if (V.type() == ValueType::LOGICAL)
        return internal::castResultToClass(
            medfilt3(internal::coerceToDoubleImage(V, mr), M, N, P, mr),
            ValueType::LOGICAL, mr);
    if (M <= 0) M = 3;
    if (N <= 0) N = 3;
    if (P <= 0) P = 3;

    const ValueType cls = V.type();
    const keel::i32 H = keel::as<keel::i32>(V.dims().rows());
    const keel::i32 W = keel::as<keel::i32>(V.dims().cols());
    const keel::i32 D = V.dims().is3D() ? keel::as<keel::i32>(V.dims().pages()) : 1;

    Value out = (D > 1) ? Value::matrix3d(H, W, D, cls, mr)
                        : Value::matrix(H, W, cls, mr);
    if (H == 0 || W == 0 || D == 0) return out;

    const keel::i32 hM = M / 2;
    const keel::i32 hN = N / 2;
    const keel::i32 hP = P / 2;
    const keel::usize plane = keel::as<keel::usize>(H) * keel::as<keel::usize>(W);
    const keel::usize nbhd = keel::as<keel::usize>(M) * keel::as<keel::usize>(N)
                      * keel::as<keel::usize>(P);

    keel::Array<keel::f64> window;
    window.reserve(nbhd);
    for (keel::i32 p = 0; p < D; ++p) {
        for (keel::i32 c = 0; c < W; ++c) {
            for (keel::i32 r = 0; r < H; ++r) {
                window.clear();
                for (keel::i32 kp = -hP; kp <= hP; ++kp) {
                    const keel::i32 ps = fold_index(p + kp, D, PadMode::Symmetric);
                    for (keel::i32 kc = -hN; kc <= hN; ++kc) {
                        const keel::i32 cs = fold_index(c + kc, W, PadMode::Symmetric);
                        for (keel::i32 kr = -hM; kr <= hM; ++kr) {
                            const keel::i32 rs = fold_index(r + kr, H, PadMode::Symmetric);
                            const keel::usize idx = keel::as<keel::usize>(ps) * plane
                                             + keel::as<keel::usize>(cs) * H
                                             + keel::as<keel::usize>(rs);
                            window.push_back(V.elemAsDouble(idx));
                        }
                    }
                }
                std::nth_element(window.begin(),
                                 window.begin() + window.size() / 2,
                                 window.end());
                keel::f64 med = window[window.size() / 2];
                if ((window.size() & 1) == 0) {
                    keel::f64 left_max = -std::numeric_limits<keel::f64>::infinity();
                    for (auto it = window.begin();
                         it != window.begin() + window.size() / 2; ++it)
                        if (*it > left_max) left_max = *it;
                    med = 0.5 * (left_max + med);
                }
                store_classed(out,
                              keel::as<keel::usize>(p) * plane
                                + keel::as<keel::usize>(c) * H
                                + keel::as<keel::usize>(r),
                              med, cls);
            }
        }
    }
    return out;
}

Value convmtx2(const Value &h, keel::i32 m, keel::i32 n, keel::Allocator *mr)
{
    convmtx2_argcheck(h, m, n);
    const auto &dh = h.dims();
    const keel::i32 M = keel::as<keel::i32>(dh.rows());
    const keel::i32 N = keel::as<keel::i32>(dh.cols());
    const keel::usize out_rows = keel::as<keel::usize>(m + M - 1);
    const keel::usize out_cols = keel::as<keel::usize>(n + N - 1);
    const keel::usize P = out_rows * out_cols;
    const keel::usize Q = keel::as<keel::usize>(m) * keel::as<keel::usize>(n);

    Value T = Value::matrix(P, Q, ValueType::DOUBLE, mr);
    if (P == 0 || Q == 0) return T;
    keel::f64 *Td = T.doubleDataMut();
    // T is col-major; column k corresponds to input position (r_in, c_in)
    // (1-based). For each kernel cell (i, j), set
    // T(out_r + (out_c-1)*out_rows, k) = h(i, j),
    // where out_r = r_in + i - 1, out_c = c_in + j - 1.
    for (keel::i32 c_in = 0; c_in < n; ++c_in) {
        for (keel::i32 r_in = 0; r_in < m; ++r_in) {
            const keel::usize k = keel::as<keel::usize>(c_in) * keel::as<keel::usize>(m)
                           + keel::as<keel::usize>(r_in);
            for (keel::i32 j = 0; j < N; ++j) {
                for (keel::i32 i = 0; i < M; ++i) {
                    const keel::f64 hv = h.elemAsDouble(
                        keel::as<keel::usize>(j) * keel::as<keel::usize>(M)
                        + keel::as<keel::usize>(i));
                    if (hv == 0.0) continue;
                    const keel::usize out_r = keel::as<keel::usize>(r_in + i);
                    const keel::usize out_c = keel::as<keel::usize>(c_in + j);
                    const keel::usize p = out_c * out_rows + out_r;
                    Td[k * P + p] = hv;
                }
            }
        }
    }
    return T;
}

Value entropyfilt(const Value &I, const Value &domain, keel::Allocator *mr)
{
    entropyfilt_argcheck(I);
    const bool isLogical = (I.type() == ValueType::LOGICAL);
    const keel::i32 nbins = isLogical ? 2 : 256;
    // For non-logical, non-uint8 inputs, cast through im2uint8 to match
    // Octave-image's "do this for everything except logical/uint8" rule.
    Value Iu = (isLogical || I.type() == ValueType::UINT8)
        ? I : im2uint8(I, mr);

    const keel::i32 H = keel::as<keel::i32>(Iu.dims().rows());
    const keel::i32 W = keel::as<keel::i32>(Iu.dims().cols());
    Value out = Value::matrix((keel::usize)H, (keel::usize)W, ValueType::DOUBLE, mr);
    if (H == 0 || W == 0) return out;

    Value dom_local;
    const Value *dom = &domain;
    if (domain.numel() == 0) {
        // Default: ones(9).
        dom_local = Value::matrix(9, 9, ValueType::LOGICAL, mr);
        for (keel::usize i = 0; i < 81; ++i) dom_local.logicalDataMut()[i] = 1;
        dom = &dom_local;
    }
    const DomainOffsets dom_offs = domain_offsets(*dom);
    const keel::usize M = dom_offs.offs.size();
    if (M == 0) return out;

    keel::f64 *od = out.doubleDataMut();
    keel::Array<keel::i32> hist(nbins, 0);
    for (keel::i32 c = 0; c < W; ++c) {
        for (keel::i32 r = 0; r < H; ++r) {
            std::fill(hist.begin(), hist.end(), 0);
            for (keel::usize k = 0; k < M; ++k) {
                const keel::f64 v = sample_sym(Iu,
                                            r + dom_offs.offs[k].first,
                                            c + dom_offs.offs[k].second,
                                            H, W);
                keel::i32 b = keel::as<keel::i32>(v);
                if (b < 0)        b = 0;
                if (b >= nbins)   b = nbins - 1;
                ++hist[(keel::usize)b];
            }
            const keel::f64 total = keel::as<keel::f64>(M);
            keel::f64 H_bits = 0.0;
            for (keel::i32 b = 0; b < nbins; ++b) {
                const keel::i32 n = hist[(keel::usize)b];
                if (n <= 0) continue;
                const keel::f64 p = keel::as<keel::f64>(n) / total;
                H_bits -= p * std::log2(p);
            }
            od[(keel::usize)c * (keel::usize)H + (keel::usize)r] = H_bits;
        }
    }
    return out;
}

Value rangefilt(const Value &I, const Value &domain, keel::Allocator *mr)
{
    rangefilt_argcheck(I);
    // Logical input → logical output (probed R2025b).
    if (I.type() == ValueType::LOGICAL)
        return internal::castResultToClass(
            rangefilt(internal::coerceToDoubleImage(I, mr), domain, mr),
            ValueType::LOGICAL, mr);
    const keel::i32 H = keel::as<keel::i32>(I.dims().rows());
    const keel::i32 W = keel::as<keel::i32>(I.dims().cols());
    // int16 input → uint16 range (probed R2025b: the reference computes
    // the range in the unsigned domain for signed inputs).
    const ValueType outT = (I.type() == ValueType::INT16)
                               ? ValueType::UINT16 : I.type();
    Value out = Value::matrix((keel::usize)H, (keel::usize)W, outT, mr);
    if (H == 0 || W == 0) return out;

    Value dom_local;
    const Value *dom = &domain;
    if (domain.numel() == 0) {
        dom_local = Value::matrix(3, 3, ValueType::LOGICAL, mr);
        for (keel::usize i = 0; i < 9; ++i) dom_local.logicalDataMut()[i] = 1;
        dom = &dom_local;
    }
    const DomainOffsets dom_offs = domain_offsets(*dom);
    const keel::usize M = dom_offs.offs.size();
    if (M == 0) return out;

    for (keel::i32 c = 0; c < W; ++c) {
        for (keel::i32 r = 0; r < H; ++r) {
            keel::f64 mn =  std::numeric_limits<keel::f64>::infinity();
            keel::f64 mx = -std::numeric_limits<keel::f64>::infinity();
            for (keel::usize k = 0; k < M; ++k) {
                const keel::f64 v = sample_sym(I,
                                            r + dom_offs.offs[k].first,
                                            c + dom_offs.offs[k].second,
                                            H, W);
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            store_classed(out, (keel::usize)c * (keel::usize)H + (keel::usize)r,
                          mx - mn, outT);
        }
    }
    return out;
}

std::tuple<Value, Value, Value>
freqz2(const Value &h, keel::usize M, keel::usize N, keel::Allocator *mr)
{
    freqz2_argcheck(h);
    const keel::usize P = h.dims().rows();
    const keel::usize Q = h.dims().cols();
    if (M == 0) M = 64;
    if (N == 0) N = 64;

    // freqspace-style row/col frequency grids: f[k] = -1 + 2(k-1)/N for
    // k = 1..N. (Even N: starts at -1 and stops at 1 - 2/N.)
    Value f1V = Value::matrix(M, 1, ValueType::DOUBLE, mr);
    Value f2V = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::f64 *f1 = f1V.doubleDataMut();
    keel::f64 *f2 = f2V.doubleDataMut();
    for (keel::usize i = 0; i < M; ++i) f1[i] = -1.0 + 2.0 * keel::as<keel::f64>(i) / keel::as<keel::f64>(M);
    for (keel::usize j = 0; j < N; ++j) f2[j] = -1.0 + 2.0 * keel::as<keel::f64>(j) / keel::as<keel::f64>(N);

    Value HV = Value::matrix(M, N, ValueType::COMPLEX, mr);
    if (P == 0 || Q == 0) return {std::move(HV), std::move(f1V), std::move(f2V)};
    keel::c64 *Hd = HV.complexDataMut();

    // H[i, j] = Σ_p Σ_q h[p,q] · exp(+iπ·(f1[i]·(p - cp) + f2[j]·(q - cq)))
    // where (cp, cq) = (⌊(P-1)/2⌋, ⌊(Q-1)/2⌋) are the kernel centre
    // offsets. Matches MATLAB: real for symmetric h, complex otherwise.
    const keel::f64 cp = keel::as<keel::f64>((P > 0 ? (P - 1) / 2 : 0));
    const keel::f64 cq = keel::as<keel::f64>((Q > 0 ? (Q - 1) / 2 : 0));
    keel::Array<keel::c64> phase_p_i(M * P), phase_q_j(N * Q);
    for (keel::usize i = 0; i < M; ++i)
        for (keel::usize p = 0; p < P; ++p) {
            const keel::f64 a = M_PI * f1[i] * (keel::as<keel::f64>(p) - cp);
            phase_p_i[p + i * P] = {std::cos(a), std::sin(a)};
        }
    for (keel::usize j = 0; j < N; ++j)
        for (keel::usize q = 0; q < Q; ++q) {
            const keel::f64 a = M_PI * f2[j] * (keel::as<keel::f64>(q) - cq);
            phase_q_j[q + j * Q] = {std::cos(a), std::sin(a)};
        }
    for (keel::usize j = 0; j < N; ++j) {
        for (keel::usize i = 0; i < M; ++i) {
            keel::c64 sum{0.0, 0.0};
            for (keel::usize q = 0; q < Q; ++q) {
                const keel::c64 pq = phase_q_j[q + j * Q];
                for (keel::usize p = 0; p < P; ++p) {
                    const keel::f64 hpq = h.elemAsDouble(p + q * P);
                    sum += hpq * (phase_p_i[p + i * P] * pq);
                }
            }
            Hd[i + j * M] = sum;
        }
    }
    return {std::move(HV), std::move(f1V), std::move(f2V)};
}

// ── roifilt2 (filter a region of interest) ────────────────────────
//
// MATLAB R2025b roifilt2.m:
//   J = roifilt2(h, I, BW)   — filter I with the 2-D linear filter h,
//                              keep only the masked (BW) pixels.
//   J = roifilt2(I, BW, fun) — apply fun to I, keep only masked pixels.
//
// In both forms the output equals I outside the mask and the filtered
// value inside it. For the filter form MATLAB crops to the ROI bounding
// box (padded by ceil(size(h)/2)) purely as an optimisation; since that
// pad always covers the filter radius, the masked pixels' values equal
// those of imfilter over the whole image, so we filter the full image
// directly. imfilter is invoked with MATLAB's defaults (correlation,
// zero boundary, same size). The function form's output class follows
// the class of fun's result; the filter form's follows class(I).
// out = cast(I, class(filtRes)); out(BW) = filtRes(BW).
Value roifilt2_combine(const Value &I, const Value &filtRes, const Value &BW,
                       keel::Allocator *mr)
{
    roifilt2_combine_argcheck(I, filtRes, BW);
    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    const keel::usize N = H * W;
    if (filtRes.isComplex() || I.isComplex()) {
        Value out = Value::matrix(H, W, ValueType::COMPLEX, mr);
        auto *od = out.complexDataMut();
        auto getC = [](const Value &v, keel::usize idx) -> keel::c64 {
            return v.isComplex() ? v.complexData()[idx]
                                 : keel::c64(v.elemAsDouble(idx), 0.0);
        };
        for (keel::usize i = 0; i < N; ++i) {
            const bool masked = BW.elemAsDouble(i) != 0.0;
            od[i] = masked ? getC(filtRes, i) : getC(I, i);
        }
        return out;
    }
    const ValueType cls = filtRes.type();
    Value out = Value::matrix(H, W, cls, mr);
    auto store = [&](keel::usize i, keel::f64 v) {
        switch (cls) {
            case ValueType::SINGLE: out.singleDataMut()[i] = keel::as<keel::f32>(v); break;
            case ValueType::UINT8:  out.uint8DataMut()[i]  = keel::as<keel::u8>(std::lround(v)); break;
            case ValueType::UINT16: out.uint16DataMut()[i] = keel::as<keel::u16>(std::lround(v)); break;
            case ValueType::UINT32: out.uint32DataMut()[i] = keel::as<keel::u32>(std::lround(v)); break;
            case ValueType::INT8:   out.int8DataMut()[i]   = keel::as<keel::i8>(std::lround(v)); break;
            case ValueType::INT16:  out.int16DataMut()[i]  = keel::as<keel::i16>(std::lround(v)); break;
            case ValueType::INT32:  out.int32DataMut()[i]  = keel::as<keel::i32>(std::lround(v)); break;
            case ValueType::INT64:  out.int64DataMut()[i]  = keel::as<keel::i64>(std::llround(v)); break;
            case ValueType::LOGICAL:out.logicalDataMut()[i]= (v != 0.0) ? 1u : 0u; break;
            default:                out.doubleDataMut()[i] = v; break;
        }
    };
    for (keel::usize i = 0; i < N; ++i) {
        const bool masked = BW.elemAsDouble(i) != 0.0;
        store(i, (masked ? filtRes : I).elemAsDouble(i));
    }
    return out;
}

// Filter form: J = roifilt2(h, I, BW).
Value roifilt2(const Value &h, const Value &I, const Value &BW,
               keel::Allocator *mr)
{
    roifilt2_argcheck(h, I);
    // imfilter defaults: correlation, zero boundary, same size.
    Value filtFull = imfilter(I, h, PadMode::Constant, 0.0, false, false, mr);
    return roifilt2_combine(I, filtFull, BW, mr);
}

// ── imguidedfilter (Guided Image Filter, He/Sun/Tang 2013) ────────
//
// Algorithm 1 from K. He, J. Sun, X. Tang,
//   "Guided Image Filtering", IEEE TPAMI 35(6), 1397-1409, 2013.
//
//   meanI  = box(G)
//   meanP  = box(A)
//   corrI  = box(G·G)
//   corrIP = box(G·A)
//   varI  = corrI − meanI²
//   covIP = corrIP − meanI·meanP
//   a = covIP / (varI + ε)
//   b = meanP − a·meanI
//   meana = box(a)
//   meanb = box(b)
//   B = meana·G + meanb
//
// Grayscale guide only here. Default ε = 0.01 · range².
Value imguidedfilter(const Value &A, const Value &G_in, keel::i32 nhood,
                     keel::f64 eps_in,
                     keel::Allocator *mr)
{
    imguidedfilter_argcheck(A, nhood);
    const keel::usize H = A.dims().rows();
    const keel::usize W = A.dims().cols();
    if (H == 0 || W == 0) return A;

    Value G = G_in;
    if (G.numel() == 0) G = A;  // self-guidance
    if (G.dims().rows() != H || G.dims().cols() != W || G.dims().is3D())
        throw Error("imguidedfilter: G must be the same H × W as A "
                    "(RGB guidance not yet supported)",
                    0, 0, "imguidedfilter", "",
                    "numkit:imguidedfilter:gshape");

    // Resolve default ε per A's class.
    keel::f64 eps = eps_in;
    if (eps < 0) {
        keel::f64 range = 1.0;
        switch (A.type()) {
            case ValueType::UINT8:  range = 255.0;       break;
            case ValueType::UINT16: range = 65535.0;     break;
            case ValueType::UINT32: range = 4294967295.0;break;
            case ValueType::INT8:   range = 255.0;       break;  // [-128,127] diff=255
            case ValueType::INT16:  range = 65535.0;     break;
            case ValueType::INT32:  range = 4294967295.0;break;
            default:                range = 1.0;         break;  // double/single/logical
        }
        eps = 0.01 * range * range;
    }

    // Cast A, G → DOUBLE.
    const keel::usize N = H * W;
    Value Ad = Value::matrix(H, W, ValueType::DOUBLE, mr);
    Value Gd = Value::matrix(H, W, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < N; ++i) Ad.doubleDataMut()[i] = A.elemAsDouble(i);
    for (keel::usize i = 0; i < N; ++i) Gd.doubleDataMut()[i] = G.elemAsDouble(i);

    // I.*I, I.*P (element-wise products).
    Value II = Value::matrix(H, W, ValueType::DOUBLE, mr);
    Value IP = Value::matrix(H, W, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 gi = Gd.doubleData()[i];
        const keel::f64 ai = Ad.doubleData()[i];
        II.doubleDataMut()[i] = gi * gi;
        IP.doubleDataMut()[i] = gi * ai;
    }

    const Value meanI  = imboxfilt(Gd, nhood, mr);
    const Value meanP  = imboxfilt(Ad, nhood, mr);
    const Value corrI  = imboxfilt(II, nhood, mr);
    const Value corrIP = imboxfilt(IP, nhood, mr);

    // a, b.
    Value a_ = Value::matrix(H, W, ValueType::DOUBLE, mr);
    Value b_ = Value::matrix(H, W, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 mI = meanI.doubleData()[i];
        const keel::f64 mP = meanP.doubleData()[i];
        const keel::f64 varI  = corrI.doubleData()[i]  - mI * mI;
        const keel::f64 covIP = corrIP.doubleData()[i] - mI * mP;
        const keel::f64 a = covIP / (varI + eps);
        a_.doubleDataMut()[i] = a;
        b_.doubleDataMut()[i] = mP - a * mI;
    }

    const Value meana = imboxfilt(a_, nhood, mr);
    const Value meanb = imboxfilt(b_, nhood, mr);

    // B = meana·G + meanb.
    Value Bd = Value::matrix(H, W, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < N; ++i)
        Bd.doubleDataMut()[i] = meana.doubleData()[i] * Gd.doubleData()[i]
                              + meanb.doubleData()[i];

    // Cast back to class(A).
    const ValueType outT = A.type();
    if (outT == ValueType::DOUBLE) return Bd;
    Value Bout = Value::matrix(H, W, outT, mr);
    auto saturate = [&](keel::f64 v, keel::f64 lo, keel::f64 hi) {
        v = std::round(v);
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        return v;
    };
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 v = Bd.doubleData()[i];
        switch (outT) {
            case ValueType::SINGLE:  Bout.singleDataMut()[i] = keel::as<keel::f32>(v); break;
            case ValueType::UINT8:   Bout.uint8DataMut()[i]  = keel::as<keel::u8>(saturate(v, 0.0, 255.0)); break;
            case ValueType::UINT16:  Bout.uint16DataMut()[i] = keel::as<keel::u16>(saturate(v, 0.0, 65535.0)); break;
            case ValueType::INT8:    Bout.int8DataMut()[i]   = keel::as<keel::i8>(saturate(v, -128.0, 127.0)); break;
            case ValueType::INT16:   Bout.int16DataMut()[i]  = keel::as<keel::i16>(saturate(v, -32768.0, 32767.0)); break;
            case ValueType::INT32:   Bout.int32DataMut()[i]  = keel::as<keel::i32>(saturate(v, -2147483648.0, 2147483647.0)); break;
            case ValueType::UINT32:  Bout.uint32DataMut()[i] = keel::as<keel::u32>(saturate(v, 0.0, 4294967295.0)); break;
            case ValueType::LOGICAL: Bout.logicalDataMut()[i] = v >= 0.5 ? 1 : 0; break;
            default: Bout.doubleDataMut()[i] = v; break;
        }
    }
    return Bout;
}

// ── imdiffusefilt (Perona-Malik anisotropic diffusion) ───────────
//
// MATLAB R2025b imdiffusefilt.m + anisotropicDiffusion2D.m algorithm.
// References: [1] Perona & Malik 1990; [2] Gerig et al. 1992.
Value imdiffusefilt(const Value &I,
                    const Value &thresh,
                    keel::usize N_in,
                    keel::StringRef connectivity,
                    keel::StringRef conduction,
                    keel::Allocator *mr)
{
    imdiffusefilt_argcheck(I, connectivity, conduction);

    const keel::usize M = I.dims().rows();
    const keel::usize W = I.dims().cols();
    if (M == 0 || W == 0) return I;

    // Default GradientThreshold based on input class.
    const ValueType inT = I.type();
    auto class_range_diff = [&]() {
        switch (inT) {
            case ValueType::UINT8:  return 255.0;
            case ValueType::UINT16: return 65535.0;
            case ValueType::UINT32: return 4294967295.0;
            case ValueType::INT8:   return 255.0;
            case ValueType::INT16:  return 65535.0;
            case ValueType::INT32:  return 4294967295.0;
            default:                return 1.0;
        }
    };

    // Resolve gradientThreshold vector (length nThresh).
    keel::Array<keel::f64> threshVec(mr);
    if (thresh.numel() == 0) {
        threshVec.push_back(0.1 * class_range_diff());
    } else {
        threshVec.reserve(thresh.numel());
        for (keel::usize i = 0; i < thresh.numel(); ++i) {
            const keel::f64 v = thresh.elemAsDouble(i);
            if (!(v > 0) || !std::isfinite(v))
                throw Error("imdiffusefilt: GradientThreshold must be a "
                            "positive finite scalar or vector",
                            0, 0, "imdiffusefilt", "",
                            "numkit:imdiffusefilt:thresh");
            threshVec.push_back(v);
        }
    }

    // Default N.
    keel::usize N = N_in;
    if (N == 0) {
        N = (threshVec.size() == 1) ? 5 : threshVec.size();
    }
    if (threshVec.size() == 1) {
        // Replicate scalar threshold to length N.
        threshVec.resize(N, threshVec[0]);
    } else if (threshVec.size() != N) {
        throw Error("imdiffusefilt: numel(GradientThreshold) must equal "
                    "NumberOfIterations",
                    0, 0, "imdiffusefilt", "",
                    "numkit:imdiffusefilt:threshLen");
    }

    // Work in double precision (matches MATLAB for double; for
    // non-double inputs MATLAB uses single — we use double which
    // gives identical results for our integer-bit-comparison cases).
    const keel::usize Npix = M * W;
    keel::Array<keel::f64> img(Npix, 0.0, mr);
    for (keel::usize i = 0; i < Npix; ++i) img[i] = I.elemAsDouble(i);

    // Padded image with replicate border: (M+2) x (W+2).
    const keel::usize Mp = M + 2;
    const keel::usize Wp = W + 2;
    keel::Array<keel::f64> pad(Mp * Wp, 0.0, mr);

    auto refresh_padded = [&](const keel::Array<keel::f64> &src) {
        // Interior: pad(r+1, c+1) = src(r, c) for r=0..M-1, c=0..W-1.
        for (keel::usize c = 0; c < W; ++c)
            for (keel::usize r = 0; r < M; ++r)
                pad[(c + 1) * Mp + (r + 1)] = src[c * M + r];
        // Top/bottom rows replicate.
        for (keel::usize c = 0; c < W; ++c) {
            pad[(c + 1) * Mp + 0]      = src[c * M + 0];
            pad[(c + 1) * Mp + Mp - 1] = src[c * M + M - 1];
        }
        // Left/right cols replicate (including corners).
        for (keel::usize r = 0; r < Mp; ++r) {
            pad[0 * Mp + r]            = pad[1 * Mp + r];
            pad[(Wp - 1) * Mp + r]     = pad[(W) * Mp + r];
        }
    };

    auto conductance = [&](keel::f64 diff, keel::f64 K) {
        const keel::f64 r = diff / K;
        if (conduction == "exponential") return std::exp(-(r * r));
        return 1.0 / (1.0 + r * r);
    };

    const bool isMax = (connectivity == "maximal");
    const keel::f64 diffusionRate = isMax ? (1.0 / 8.0) : (1.0 / 4.0);
    const keel::f64 inv_dd2 = 0.5;  // 1 / dd² = 1/2 for diagonal

    for (keel::usize it = 0; it < N; ++it) {
        const keel::f64 K = threshVec[it];
        refresh_padded(img);

        // Accessor: padded(r, c) → pad[c * Mp + r].
        auto P = [&](keel::usize r, keel::usize c) {
            return pad[c * Mp + r];
        };

        keel::Array<keel::f64> upd(Npix, 0.0, mr);

        for (keel::usize c = 0; c < W; ++c) {
            for (keel::usize r = 0; r < M; ++r) {
                // 0-indexed in image; padded coords = (r+1, c+1).
                const keel::f64 ci = img[c * M + r];
                // diffNorth at this pixel = above - center =
                //   pad(r, c+1) - pad(r+1, c+1)
                const keel::f64 dN = P(r,     c + 1) - ci;
                // diffSouth = below - center
                const keel::f64 dS = P(r + 2, c + 1) - ci;
                // diffEast  = right - center
                const keel::f64 dE = P(r + 1, c + 2) - ci;
                // diffWest  = left - center
                const keel::f64 dW = P(r + 1, c)     - ci;
                const keel::f64 fN = conductance(dN, K) * dN;
                const keel::f64 fS = conductance(dS, K) * dS;
                const keel::f64 fE = conductance(dE, K) * dE;
                const keel::f64 fW = conductance(dW, K) * dW;
                keel::f64 delta = fN + fS + fE + fW;
                if (isMax) {
                    const keel::f64 dNW = P(r,     c)     - ci;
                    const keel::f64 dNE = P(r,     c + 2) - ci;
                    const keel::f64 dSW = P(r + 2, c)     - ci;
                    const keel::f64 dSE = P(r + 2, c + 2) - ci;
                    const keel::f64 fNW = conductance(dNW, K) * dNW;
                    const keel::f64 fNE = conductance(dNE, K) * dNE;
                    const keel::f64 fSW = conductance(dSW, K) * dSW;
                    const keel::f64 fSE = conductance(dSE, K) * dSE;
                    delta += inv_dd2 * (fNW + fNE + fSW + fSE);
                }
                upd[c * M + r] = ci + diffusionRate * delta;
            }
        }
        keel::swap(img, upd);
    }

    // Cast back to input class.
    Value out;
    if (inT == ValueType::DOUBLE) {
        out = Value::matrix(M, W, ValueType::DOUBLE, mr);
        for (keel::usize i = 0; i < Npix; ++i) out.doubleDataMut()[i] = img[i];
        return out;
    }
    out = Value::matrix(M, W, inT, mr);
    auto saturate = [&](keel::f64 v, keel::f64 lo, keel::f64 hi) {
        v = std::round(v);
        if (v < lo) v = lo;
        if (v > hi) v = hi;
        return v;
    };
    for (keel::usize i = 0; i < Npix; ++i) {
        const keel::f64 v = img[i];
        switch (inT) {
            case ValueType::SINGLE:  out.singleDataMut()[i] = keel::as<keel::f32>(v); break;
            case ValueType::UINT8:   out.uint8DataMut()[i]  = keel::as<keel::u8>(saturate(v, 0.0, 255.0)); break;
            case ValueType::UINT16:  out.uint16DataMut()[i] = keel::as<keel::u16>(saturate(v, 0.0, 65535.0)); break;
            case ValueType::UINT32:  out.uint32DataMut()[i] = keel::as<keel::u32>(saturate(v, 0.0, 4294967295.0)); break;
            case ValueType::INT8:    out.int8DataMut()[i]   = keel::as<keel::i8>(saturate(v, -128.0, 127.0)); break;
            case ValueType::INT16:   out.int16DataMut()[i]  = keel::as<keel::i16>(saturate(v, -32768.0, 32767.0)); break;
            case ValueType::INT32:   out.int32DataMut()[i]  = keel::as<keel::i32>(saturate(v, -2147483648.0, 2147483647.0)); break;
            default: out.doubleDataMut()[i] = v; break;
        }
    }
    return out;
}

// ── imgaborfilt (single-filter Gabor magnitude + phase) ──────────
//
// MATLAB R2025b algorithm (gaborFilterFFT.m + gabor.m):
//   SigmaX = wavelength/π · √(log2/2) · (2^B + 1)/(2^B - 1)
//   SigmaY = SigmaX / aspect
//   r      = max(⌈7·SigmaX⌉, ⌈7·SigmaY⌉)
//   Pad A by [r r] replicate, FFT-2D padded image.
//   Build H in frequency domain:
//     u_n = normalised frequency vector length N
//     v_n = normalised frequency vector length M
//     U', V' = rotated (u, v) by orientation
//     σ_u = 1/(2π·SigmaX), σ_v = 1/(2π·SigmaY), f = 1/λ
//     A_const = 2π·SigmaX·SigmaY
//     H = A_const · exp(-½ ((U'-f)²/σ_u² + V'²/σ_v²))
//   out_padded = ifft2(A_fft · ifftshift(H))
//   out = crop r border → (M, N)
//   mag = |out|, phase = angle(out)
//
// References:
//   Jain & Farrokhnia 1991; Kruizinga & Petkov 1999.
void imgaborfilt(const Value &A_in, keel::f64 wavelength, keel::f64 orientation,
                 keel::f64 sfb, keel::f64 aspect,
                 Value &mag_out, Value &phase_out,
                 keel::Allocator *mr)
{
    imgaborfilt_argcheck(A_in, wavelength, sfb, aspect);

    const keel::usize M = A_in.dims().rows();
    const keel::usize N = A_in.dims().cols();
    const ValueType outT = (A_in.type() == ValueType::SINGLE)
                            ? ValueType::SINGLE : ValueType::DOUBLE;

    // Sigma parameters.
    const keel::f64 sigmaX = wavelength / M_PI * std::sqrt(std::log(2.0) / 2.0)
                        * (std::pow(2.0, sfb) + 1.0)
                        / (std::pow(2.0, sfb) - 1.0);
    const keel::f64 sigmaY = sigmaX / aspect;
    const keel::i64 Rx = keel::as<keel::i64>(std::ceil(7.0 * sigmaX));
    const keel::i64 Ry = keel::as<keel::i64>(std::ceil(7.0 * sigmaY));
    const keel::i64 r = std::max(Rx, Ry);

    // Pad A to (M + 2r) x (N + 2r) with replicate.
    const keel::usize Mp = M + 2 * r;
    const keel::usize Np = N + 2 * r;
    // Use existing padarray (filter.cpp) if available; fall back to manual.
    Value Ad = Value::matrix(M, N, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < M * N; ++i) Ad.doubleDataMut()[i] = A_in.elemAsDouble(i);
    Value Apad = padarray(Ad, {keel::as<keel::i32>(r), keel::as<keel::i32>(r)},
                          PadMode::Replicate, 0.0, "both", mr);

    // FFT-2D of padded image.
    Value Afft = signal::fft2(Apad, -1, -1, mr);
    // Normalise Afft to COMPLEX storage.
    const keel::usize Np_pix = Mp * Np;
    keel::Array<keel::c64> Afc(Np_pix, keel::c64{0, 0}, mr);
    if (Afft.isComplex()) {
        const keel::c64 *src = Afft.complexData();
        for (keel::usize i = 0; i < Np_pix; ++i) Afc[i] = src[i];
    } else {
        for (keel::usize i = 0; i < Np_pix; ++i)
            Afc[i] = keel::c64{Afft.elemAsDouble(i), 0.0};
    }

    // Normalised frequency vectors u (length Np), v (length Mp).
    auto freq_vec = [&](keel::usize Nlen, keel::Array<keel::f64> &out) {
        out.resize(Nlen);
        if (Nlen == 1) { out[0] = 0.0; return; }
        if (Nlen % 2 == 1) {
            const keel::f64 lo = -0.5 + 1.0 / (2.0 * Nlen);
            const keel::f64 hi =  0.5 - 1.0 / (2.0 * Nlen);
            const keel::f64 step = (hi - lo) / (Nlen - 1);
            for (keel::usize i = 0; i < Nlen; ++i) out[i] = lo + step * i;
        } else {
            const keel::f64 lo = -0.5;
            const keel::f64 hi = 0.5 - 1.0 / Nlen;
            const keel::f64 step = (hi - lo) / (Nlen - 1);
            for (keel::usize i = 0; i < Nlen; ++i) out[i] = lo + step * i;
        }
    };
    keel::Array<keel::f64> u(mr), v(mr);
    freq_vec(Np, u);
    freq_vec(Mp, v);

    const keel::f64 theta_rad = orientation * M_PI / 180.0;
    const keel::f64 cosT = std::cos(theta_rad);
    const keel::f64 sinT = std::sin(theta_rad);
    const keel::f64 sigmau = 1.0 / (2.0 * M_PI * sigmaX);
    const keel::f64 sigmav = 1.0 / (2.0 * M_PI * sigmaY);
    const keel::f64 freq   = 1.0 / wavelength;
    const keel::f64 A_const = 2.0 * M_PI * sigmaX * sigmaY;

    // Build H in COMPLEX domain (real values; imag = 0). Stored col-major:
    // H(r, c) → c*Mp + r.
    keel::Array<keel::f64> H(Np_pix, 0.0, mr);
    for (keel::usize c = 0; c < Np; ++c) {
        const keel::f64 uc = u[c];
        for (keel::usize r2 = 0; r2 < Mp; ++r2) {
            const keel::f64 vc = v[r2];
            const keel::f64 Upr = uc * cosT - vc * sinT;
            const keel::f64 Vpr = uc * sinT + vc * cosT;
            const keel::f64 e = -0.5 * ((Upr - freq) * (Upr - freq) / (sigmau * sigmau)
                                   + Vpr * Vpr / (sigmav * sigmav));
            H[c * Mp + r2] = A_const * std::exp(e);
        }
    }

    // ifftshift(H): for each dim, shift by floor(N/2) (no rounding).
    // ifftshift is the inverse of fftshift; for even N it's identical.
    // Easy impl: cyclic shift each dim by -floor(N/2) (which equals
    // shift right by ceil(N/2) when going from "0 at center" to "0 at origin").
    auto ifftshift_2d = [&](const keel::Array<keel::f64> &in,
                            keel::Array<keel::f64> &out) {
        out.resize(Np_pix);
        const keel::usize shR_eff = Mp / 2;          // floor(Mp/2) for fftshift
        const keel::usize shC_eff = Np / 2;
        // ifftshift = inverse of fftshift. For ifftshift, the shift amount equals (N - floor(N/2)) = ceil(N/2):
        const keel::usize aR = Mp - shR_eff;
        const keel::usize aC = Np - shC_eff;
        for (keel::usize c = 0; c < Np; ++c) {
            const keel::usize cs = (c + aC) % Np;
            for (keel::usize r2 = 0; r2 < Mp; ++r2) {
                const keel::usize rs = (r2 + aR) % Mp;
                out[c * Mp + r2] = in[cs * Mp + rs];
            }
        }
    };
    keel::Array<keel::f64> Hshift(mr);
    ifftshift_2d(H, Hshift);

    // Multiply: prod = Afc .* Hshift (real-valued H).
    keel::Array<keel::c64> prod(Np_pix, keel::c64{0, 0}, mr);
    for (keel::usize i = 0; i < Np_pix; ++i) {
        prod[i] = keel::c64{Afc[i].real() * Hshift[i],
                          Afc[i].imag() * Hshift[i]};
    }

    // ifft2.
    Value Prod = Value::matrix(Mp, Np, ValueType::COMPLEX, mr);
    keel::c64 *pp = Prod.complexDataMut();
    for (keel::usize i = 0; i < Np_pix; ++i) pp[i] = prod[i];
    Value Out = signal::ifft2(Prod, -1, -1, mr);

    // Output is complex; if accidentally collapsed to DOUBLE (real-symmetric),
    // pad imag = 0.
    auto out_at = [&](keel::usize i) -> keel::c64 {
        if (Out.isComplex()) return Out.complexData()[i];
        return keel::c64{Out.elemAsDouble(i), 0.0};
    };

    // Unpad: take rows [r, r+M-1] cols [r, r+N-1].
    mag_out = Value::matrix(M, N, outT, mr);
    phase_out = Value::matrix(M, N, outT, mr);
    for (keel::usize c = 0; c < N; ++c) {
        for (keel::usize rr = 0; rr < M; ++rr) {
            const keel::usize src = (c + r) * Mp + (rr + r);
            const keel::c64 z = out_at(src);
            const keel::f64 m = std::hypot(z.real(), z.imag());
            const keel::f64 p = std::atan2(z.imag(), z.real());
            const keel::usize dst = c * M + rr;
            if (outT == ValueType::DOUBLE) {
                mag_out.doubleDataMut()[dst]   = m;
                phase_out.doubleDataMut()[dst] = p;
            } else {
                mag_out.singleDataMut()[dst]   = keel::as<keel::f32>(m);
                phase_out.singleDataMut()[dst] = keel::as<keel::f32>(p);
            }
        }
    }
}

// ── imnlmfilt (Non-Local Means denoising) ────────────────────────
//
// MATLAB R2025b imnlmfilt: exhaustive search-window NLM with
// box-blur patch-distance (Buades-Coll-Morel 2005).
//
//   for each pixel p:
//     for each q in S×S search window:
//       d² = (1/|N|) Σ_{x∈N} (I_pad(p+x) - I_pad(q+x))²
//       w(p, q) = exp(-d² / h²)
//     w(p, p) is set to max over non-self weights ("Buades trick")
//     J(p) = Σ_q w·I(q) / Σ_q w
//
// Default h via Immerkaer-1996 noise estimate:
//   h = |L*I| · √(π/2) / (6·(W-2)·(H-2))
//   L = [1 -2 1; -2 4 -2; 1 -2 1] (Laplacian kernel)
//
// References:
//   Buades-Coll-Morel "A Non-Local Algorithm for Image Denoising"
//   CVPR 2005;
//   Immerkaer "Fast Noise Variance Estimation" CVIU 1996.
//
// 2-D grayscale only.
namespace {

keel::f64 immerkaer_dos(const Value &I, keel::Allocator * /*mr*/) {
    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    if (H < 3 || W < 3) return 1.0;

    // MATLAB always casts to single in estimateDegreeOfSmoothing.m, so
    // we do the same to bit-match (matters at the ~1e-8 level).
    const keel::f32 K[3][3] = {{1, -2, 1}, {-2, 4, -2}, {1, -2, 1}};
    const keel::usize Mp = H + 2;
    const keel::usize Wp = W + 2;
    auto Ival = [&](keel::usize r, keel::usize c) -> keel::f32 {
        return keel::as<keel::f32>(I.elemAsDouble(c * H + r));
    };
    keel::f32 sum_abs = 0.0f;
    for (keel::usize oc = 0; oc < Wp; ++oc) {
        for (keel::usize orr = 0; orr < Mp; ++orr) {
            keel::f32 s = 0.0f;
            for (keel::i32 i = 0; i < 3; ++i) {
                for (keel::i32 j = 0; j < 3; ++j) {
                    const keel::i64 ir = keel::as<keel::i64>(orr) - i;
                    const keel::i64 ic = keel::as<keel::i64>(oc) - j;
                    if (ir < 0 || ic < 0
                     || keel::as<keel::usize>(ir) >= H
                     || keel::as<keel::usize>(ic) >= W) continue;
                    s += K[i][j] * Ival(ir, ic);
                }
            }
            sum_abs += std::fabs(s);
        }
    }
    const keel::f32 dos = sum_abs * std::sqrt(keel::as<keel::f32>(M_PI) * 0.5f)
                    / (6.0f * (W - 2) * (H - 2));
    return (dos == 0.0f)
            ? keel::as<keel::f64>(std::numeric_limits<keel::f32>::epsilon())
            : keel::as<keel::f64>(dos);
}

void imnlmfilt_argcheck(const Value &I, keel::i32 swsize, keel::i32 cwsize)
{
    if (I.dims().is3D())
        throw Error("imnlmfilt: RGB / colour inputs not yet supported",
                    0, 0, "imnlmfilt", "", "numkit:imnlmfilt:dim");
    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    if (H < 21 || W < 21)
        throw Error("imnlmfilt: image must be at least 21x21",
                    0, 0, "imnlmfilt", "", "numkit:imnlmfilt:size");
    if (swsize < 1 || (swsize % 2) == 0)
        throw Error("imnlmfilt: SearchWindowSize must be a positive odd "
                    "integer",
                    0, 0, "imnlmfilt", "", "numkit:imnlmfilt:sws");
    if (cwsize < 1 || (cwsize % 2) == 0)
        throw Error("imnlmfilt: ComparisonWindowSize must be a positive "
                    "odd integer",
                    0, 0, "imnlmfilt", "", "numkit:imnlmfilt:cws");
    if (cwsize > swsize)
        throw Error("imnlmfilt: ComparisonWindowSize must be <= "
                    "SearchWindowSize",
                    0, 0, "imnlmfilt", "", "numkit:imnlmfilt:cwsTooLarge");
    if (keel::as<keel::usize>(swsize) > std::min(H, W))
        throw Error("imnlmfilt: SearchWindowSize must be <= min(H, W)",
                    0, 0, "imnlmfilt", "", "numkit:imnlmfilt:swsTooLarge");
}

} // anonymous

void imnlmfilt(const Value &I, keel::f64 dos,
               keel::i32 swsize, keel::i32 cwsize,
               Value &J_out, keel::f64 &est_out,
               keel::Allocator *mr)
{
    imnlmfilt_argcheck(I, swsize, cwsize);
    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();

    const ValueType outT = I.type();
    const keel::usize N = H * W;

    // Cast I to DOUBLE.
    keel::Array<keel::f64> Iv(N, 0.0, mr);
    for (keel::usize i = 0; i < N; ++i) Iv[i] = I.elemAsDouble(i);

    // Resolve h.
    keel::f64 h = dos;
    if (h < 0) {
        Value Ivv = Value::matrix(H, W, ValueType::DOUBLE, mr);
        for (keel::usize i = 0; i < N; ++i) Ivv.doubleDataMut()[i] = Iv[i];
        h = immerkaer_dos(Ivv, mr);
    }
    est_out = h;

    const keel::i32 sh = (swsize - 1) / 2;
    const keel::i32 ch = (cwsize - 1) / 2;
    const keel::i32 pad = sh + ch;
    const keel::usize Mp = H + 2 * pad;
    const keel::usize Wp = W + 2 * pad;

    // Build replicate-padded image.
    keel::Array<keel::f64> P(Mp * Wp, 0.0, mr);
    for (keel::usize c = 0; c < Wp; ++c) {
        for (keel::usize r = 0; r < Mp; ++r) {
            const keel::i64 ir = keel::as<keel::i64>(r) - pad;
            const keel::i64 ic = keel::as<keel::i64>(c) - pad;
            const keel::i64 irc = std::clamp(ir, 0L, keel::as<keel::i64>(H - 1));
            const keel::i64 icc = std::clamp(ic, 0L, keel::as<keel::i64>(W - 1));
            P[c * Mp + r] = Iv[keel::as<keel::usize>(icc) * H
                              + keel::as<keel::usize>(irc)];
        }
    }

    const keel::f64 inv_h2 = 1.0 / (h * h);
    const keel::f64 inv_cwsq = 1.0 / keel::as<keel::f64>(cwsize * cwsize);

    // Build output.
    keel::Array<keel::f64> Out(N, 0.0, mr);

    for (keel::usize c = 0; c < W; ++c) {
        for (keel::usize r = 0; r < H; ++r) {
            const keel::usize pr = r + pad;
            const keel::usize pc = c + pad;
            keel::f64 sum_w = 0.0;
            keel::f64 sum_wI = 0.0;
            keel::f64 max_w = 0.0;

            // Iterate search window q = p + (dr, dc).
            for (keel::i32 dc = -sh; dc <= sh; ++dc) {
                for (keel::i32 dr = -sh; dr <= sh; ++dr) {
                    if (dr == 0 && dc == 0) continue;  // skip self
                    const keel::usize qr = pr + dr;
                    const keel::usize qc = pc + dc;
                    // Compute squared patch distance.
                    keel::f64 d2 = 0.0;
                    for (keel::i32 ac = -ch; ac <= ch; ++ac) {
                        for (keel::i32 ar = -ch; ar <= ch; ++ar) {
                            const keel::f64 a = P[(pc + ac) * Mp + (pr + ar)];
                            const keel::f64 b = P[(qc + ac) * Mp + (qr + ar)];
                            const keel::f64 d = a - b;
                            d2 += d * d;
                        }
                    }
                    d2 *= inv_cwsq;
                    const keel::f64 w = std::exp(-d2 * inv_h2);
                    sum_w  += w;
                    sum_wI += w * P[qc * Mp + qr];
                    if (w > max_w) max_w = w;
                }
            }
            // Centre pixel uses max non-self weight.
            sum_w  += max_w;
            sum_wI += max_w * Iv[c * H + r];

            Out[c * H + r] = sum_wI / sum_w;
        }
    }

    // Cast back to input class.
    J_out = Value::matrix(H, W, outT, mr);
    if (outT == ValueType::DOUBLE) {
        keel::f64 *p = J_out.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i) p[i] = Out[i];
    } else if (outT == ValueType::SINGLE) {
        keel::f32 *p = J_out.singleDataMut();
        for (keel::usize i = 0; i < N; ++i) p[i] = keel::as<keel::f32>(Out[i]);
    } else {
        auto saturate = [&](keel::f64 v, keel::f64 lo, keel::f64 hi) {
            v = std::round(v);
            if (v < lo) v = lo;
            if (v > hi) v = hi;
            return v;
        };
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 v = Out[i];
            switch (outT) {
                case ValueType::UINT8:  J_out.uint8DataMut()[i]  = keel::as<keel::u8>(saturate(v, 0.0, 255.0)); break;
                case ValueType::UINT16: J_out.uint16DataMut()[i] = keel::as<keel::u16>(saturate(v, 0.0, 65535.0)); break;
                case ValueType::INT8:   J_out.int8DataMut()[i]   = keel::as<keel::i8>(saturate(v, -128.0, 127.0)); break;
                case ValueType::INT16:  J_out.int16DataMut()[i]  = keel::as<keel::i16>(saturate(v, -32768.0, 32767.0)); break;
                case ValueType::INT32:  J_out.int32DataMut()[i]  = keel::as<keel::i32>(saturate(v, -2147483648.0, 2147483647.0)); break;
                case ValueType::UINT32: J_out.uint32DataMut()[i] = keel::as<keel::u32>(saturate(v, 0.0, 4294967295.0)); break;
                default: J_out.doubleDataMut()[i] = v; break;
            }
        }
    }
}

} // namespace numkit::image
