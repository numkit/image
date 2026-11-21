// toolboxes/image/src/object/object.cpp
//
// Gradient + edge detection. For first cut: Sobel / Prewitt / Roberts.
// Canny / log / zerocross share the simpler-Sobel path with hysteresis
// thresholding; full Canny non-max suppression deferred.

#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/image/object/object.hpp>
#include <numkit/image/internal/type_gates.hpp>

#include <numkit/image/contrast/contrast.hpp>
#include <numkit/image/filter/filter.hpp>
#include <numkit/image/morph/morph.hpp>
#include <numkit/image/type_convert/type_convert.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <tuple>
#include <keel/static_array.h>
#include <cmath>
#include <cstdint>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace numkit::image {

namespace {

// Build a small 3×3 (or 2×2) kernel as a DOUBLE Value (column-major).
Value make_kernel(keel::ArrayRef<keel::f64> flat_rowmajor, int rows, int cols, keel::Allocator *mr)
{
    Value k = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    keel::f64 *kd = k.doubleDataMut();
    for (int c = 0; c < cols; ++c)
        for (int r = 0; r < rows; ++r)
            kd[(keel::usize)c * (keel::usize)rows + (keel::usize)r]
                = flat_rowmajor[(keel::usize)r * (keel::usize)cols + (keel::usize)c];
    return k;
}

// Convolve `I` with `k` using imfilter (Replicate boundary, same size,
// correlation = no flip — these are gradient kernels meant to be applied
// directly).
Value apply_kernel(const Value &I, const Value &k, keel::Allocator *mr)
{
    return imfilter(I, k, PadMode::Replicate, 0.0, /*full=*/false, /*flip_kernel=*/false, mr);
}

} // anonymous

std::tuple<Value, Value>
imgradientxy(const Value &I, keel::StringRef method, keel::Allocator *mr)
{
    // Choose kernels (Gx for horizontal, Gy for vertical).
    keel::Array<keel::f64> kx, ky;
    int cols = 3;
    // Kernel sign convention matches MATLAB imgradientxy: Gx is positive
    // where intensity increases left→right (negative on the left column,
    // positive on the right), Gy positive where intensity increases
    // top→bottom. (MATLAB uses hx = [-1 0 1; -2 0 2; -1 0 1] for sobel.)
    if (method == "prewitt") {
        kx = {-1, 0, 1,
              -1, 0, 1,
              -1, 0, 1 };
        ky = {-1,-1,-1,
               0, 0, 0,
               1, 1, 1 };
    } else if (method == "central") {
        // 1×3 / 3×1 central difference.
        kx = {-0.5, 0, 0.5}; cols = 3;
        ky = {-0.5, 0, 0.5};
    } else if (method == "intermediate") {
        kx = {-1, 1}; cols = 2;
        ky = {-1, 1};
    } else {  // sobel default
        kx = {-1, 0, 1,
              -2, 0, 2,
              -1, 0, 1 };
        ky = {-1,-2,-1,
               0, 0, 0,
               1, 2, 1 };
    }

    Value Kx, Ky;
    if (method == "central" || method == "intermediate") {
        Kx = make_kernel(kx, 1, cols, mr);
        Ky = make_kernel(ky, cols, 1, mr);  // transpose for vertical
    } else {
        Kx = make_kernel(kx, 3, 3, mr);
        Ky = make_kernel(ky, 3, 3, mr);
    }

    Value Gx = apply_kernel(I, Kx, mr);
    Value Gy = apply_kernel(I, Ky, mr);
    return std::make_tuple(std::move(Gx), std::move(Gy));
}

// ── 3-D imgradientxyz / imgradient3 ────────────────────────────────
//
// MATLAB R2025b imgradientxyz.m uses non-standard 3-D Sobel kernels
// with [1, 3, 3, 1]-style weights (NOT the naive [1, 2, 1] 2-D
// extension); the actual hx/hy/hz tensors are reproduced verbatim
// from the MATLAB source below. Prewitt uses [1, 1, 1]; central is
// `gradient(V)`; intermediate is forward `diff` with the trailing
// slice zeroed.
namespace {

// Convolve V (3-D DOUBLE) with a 3×3×3 kernel K (27 doubles, row-
// then-col-then-page layout matching `K[r, c, p]` indexing) using
// 'replicate' boundary handling. Returns DOUBLE same shape.
//
// Convention: V is stored column-major H × W × D, linear index
//   k(r, c, p) = r + H * c + H * W * p
// Sobel-style direct correlation (no kernel flip — kernel embeds
// the sign convention).
Value conv3d_replicate(const Value &V, const keel::f64 K[27],
                       keel::Allocator *mr)
{
    const auto &d = V.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    const keel::usize D = d.is3D() ? d.pages() : 1;
    Value out = (D > 1)
        ? Value::matrix3d(H, W, D, ValueType::DOUBLE, mr)
        : Value::matrix(H, W, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    const keel::f64 *vd = V.doubleData();

    auto clamp = [](long long x, keel::usize n) -> keel::usize {
        if (x < 0) return 0;
        if (x >= keel::as<long long>(n)) return n - 1;
        return keel::as<keel::usize>(x);
    };

    for (keel::usize p = 0; p < D; ++p) {
        for (keel::usize c = 0; c < W; ++c) {
            for (keel::usize r = 0; r < H; ++r) {
                keel::f64 sum = 0.0;
                for (int dp = -1; dp <= 1; ++dp) {
                    const keel::usize pp = clamp(
                        keel::as<long long>(p) + dp, D);
                    for (int dc = -1; dc <= 1; ++dc) {
                        const keel::usize cc = clamp(
                            keel::as<long long>(c) + dc, W);
                        for (int dr = -1; dr <= 1; ++dr) {
                            const keel::usize rr = clamp(
                                keel::as<long long>(r) + dr, H);
                            const int ki = (dr + 1)
                                         + 3 * (dc + 1)
                                         + 9 * (dp + 1);
                            sum += K[ki] * vd[rr + H * cc + H * W * pp];
                        }
                    }
                }
                od[r + H * c + H * W * p] = sum;
            }
        }
    }
    return out;
}

// Cast V to DOUBLE (or SINGLE if outT is SINGLE).
Value promote_to_double(const Value &V, keel::Allocator *mr)
{
    const keel::usize N = V.numel();
    const auto &d = V.dims();
    Value out = d.is3D()
        ? Value::matrix3d(d.rows(), d.cols(), d.pages(), ValueType::DOUBLE, mr)
        : Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) od[i] = V.elemAsDouble(i);
    return out;
}

Value zeros_like(const Value &V, ValueType t,
                 keel::Allocator *mr)
{
    const auto &d = V.dims();
    return d.is3D()
        ? Value::matrix3d(d.rows(), d.cols(), d.pages(), t, mr)
        : Value::matrix(d.rows(), d.cols(), t, mr);
}

} // anonymous

std::tuple<Value, Value, Value>
imgradientxyz(const Value &V, keel::StringRef method, keel::Allocator *mr)
{
    if (!V.dims().is3D())
        throw Error("imgradientxyz: V must be 3-D",
                    0, 0, "imgradientxyz", "", "numkit:imgradientxyz:rank");
    const ValueType inT = V.type();
    const ValueType outT = (inT == ValueType::SINGLE)
                         ? ValueType::SINGLE : ValueType::DOUBLE;
    Value Vd = promote_to_double(V, mr);

    Value Gx, Gy, Gz;
    if (method == "sobel") {
        // MATLAB R2025b imgradientxyz.m kernels.
        // hx (X = horizontal / cols, derivative along c):
        //   page 1: [-1 0 1; -3 0 3; -1 0 1]
        //   page 2: [-3 0 3; -6 0 6; -3 0 3]
        //   page 3: [-1 0 1; -3 0 3; -1 0 1]
        // hy (Y = vertical / rows, derivative along r):
        //   page 1: [-1 -3 -1; 0 0 0; 1 3 1]
        //   page 2: [-3 -6 -3; 0 0 0; 3 6 3]
        //   page 3: [-1 -3 -1; 0 0 0; 1 3 1]
        // hz (Z = depth / pages, derivative along p):
        //   page 1: [-1 -3 -1; -3 -6 -3; -1 -3 -1]
        //   page 2: [ 0  0  0;  0  0  0;  0  0  0]
        //   page 3: [ 1  3  1;  3  6  3;  1  3  1]
        // K index = (dr+1) + 3*(dc+1) + 9*(dp+1).
        auto build = [](const keel::f64 rows3[3][3][3], keel::f64 K[27]) {
            for (int p = 0; p < 3; ++p)
                for (int c = 0; c < 3; ++c)
                    for (int r = 0; r < 3; ++r)
                        K[r + 3 * c + 9 * p] = rows3[p][r][c];
        };
        const keel::f64 hx3[3][3][3] = {
            {{-1, 0, 1}, {-3, 0, 3}, {-1, 0, 1}},
            {{-3, 0, 3}, {-6, 0, 6}, {-3, 0, 3}},
            {{-1, 0, 1}, {-3, 0, 3}, {-1, 0, 1}},
        };
        const keel::f64 hy3[3][3][3] = {
            {{-1, -3, -1}, {0, 0, 0}, {1, 3, 1}},
            {{-3, -6, -3}, {0, 0, 0}, {3, 6, 3}},
            {{-1, -3, -1}, {0, 0, 0}, {1, 3, 1}},
        };
        const keel::f64 hz3[3][3][3] = {
            {{-1, -3, -1}, {-3, -6, -3}, {-1, -3, -1}},
            {{ 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}},
            {{ 1,  3,  1}, { 3,  6,  3}, { 1,  3,  1}},
        };
        keel::f64 Kx[27], Ky[27], Kz[27];
        build(hx3, Kx); build(hy3, Ky); build(hz3, Kz);
        Gx = conv3d_replicate(Vd, Kx, mr);
        Gy = conv3d_replicate(Vd, Ky, mr);
        Gz = conv3d_replicate(Vd, Kz, mr);
    } else if (method == "prewitt") {
        auto build = [](const keel::f64 rows3[3][3][3], keel::f64 K[27]) {
            for (int p = 0; p < 3; ++p)
                for (int c = 0; c < 3; ++c)
                    for (int r = 0; r < 3; ++r)
                        K[r + 3 * c + 9 * p] = rows3[p][r][c];
        };
        const keel::f64 hx3[3][3][3] = {
            {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}},
            {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}},
            {{-1, 0, 1}, {-1, 0, 1}, {-1, 0, 1}},
        };
        const keel::f64 hy3[3][3][3] = {
            {{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}},
            {{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}},
            {{-1, -1, -1}, {0, 0, 0}, {1, 1, 1}},
        };
        const keel::f64 hz3[3][3][3] = {
            {{-1, -1, -1}, {-1, -1, -1}, {-1, -1, -1}},
            {{ 0,  0,  0}, { 0,  0,  0}, { 0,  0,  0}},
            {{ 1,  1,  1}, { 1,  1,  1}, { 1,  1,  1}},
        };
        keel::f64 Kx[27], Ky[27], Kz[27];
        build(hx3, Kx); build(hy3, Ky); build(hz3, Kz);
        Gx = conv3d_replicate(Vd, Kx, mr);
        Gy = conv3d_replicate(Vd, Ky, mr);
        Gz = conv3d_replicate(Vd, Kz, mr);
    } else if (method == "central" || method == "intermediate") {
        // gradient(V) for 'central'; forward diff with zero pad for
        // 'intermediate'. Both are straight per-axis differences.
        const auto &d = Vd.dims();
        const keel::usize H = d.rows();
        const keel::usize W = d.cols();
        const keel::usize D = d.pages();
        Gx = zeros_like(Vd, ValueType::DOUBLE, mr);
        Gy = zeros_like(Vd, ValueType::DOUBLE, mr);
        Gz = zeros_like(Vd, ValueType::DOUBLE, mr);
        keel::f64 *gxd = Gx.doubleDataMut();
        keel::f64 *gyd = Gy.doubleDataMut();
        keel::f64 *gzd = Gz.doubleDataMut();
        const keel::f64 *vd = Vd.doubleData();
        auto idx = [&](keel::usize r, keel::usize c, keel::usize p) {
            return r + H * c + H * W * p;
        };
        if (method == "central") {
            // Gx[r,c,p] = (V[r,c+1,p] - V[r,c-1,p]) / 2  (central),
            // forward at c=0 and backward at c=W-1.
            for (keel::usize p = 0; p < D; ++p)
                for (keel::usize r = 0; r < H; ++r)
                    for (keel::usize c = 0; c < W; ++c) {
                        if (W == 1) { gxd[idx(r, c, p)] = 0.0; continue; }
                        if (c == 0)
                            gxd[idx(r, c, p)] = vd[idx(r, 1, p)] - vd[idx(r, 0, p)];
                        else if (c == W - 1)
                            gxd[idx(r, c, p)] = vd[idx(r, W - 1, p)] - vd[idx(r, W - 2, p)];
                        else
                            gxd[idx(r, c, p)] = 0.5 * (vd[idx(r, c + 1, p)] - vd[idx(r, c - 1, p)]);
                    }
            for (keel::usize p = 0; p < D; ++p)
                for (keel::usize c = 0; c < W; ++c)
                    for (keel::usize r = 0; r < H; ++r) {
                        if (H == 1) { gyd[idx(r, c, p)] = 0.0; continue; }
                        if (r == 0)
                            gyd[idx(r, c, p)] = vd[idx(1, c, p)] - vd[idx(0, c, p)];
                        else if (r == H - 1)
                            gyd[idx(r, c, p)] = vd[idx(H - 1, c, p)] - vd[idx(H - 2, c, p)];
                        else
                            gyd[idx(r, c, p)] = 0.5 * (vd[idx(r + 1, c, p)] - vd[idx(r - 1, c, p)]);
                    }
            for (keel::usize c = 0; c < W; ++c)
                for (keel::usize r = 0; r < H; ++r)
                    for (keel::usize p = 0; p < D; ++p) {
                        if (D == 1) { gzd[idx(r, c, p)] = 0.0; continue; }
                        if (p == 0)
                            gzd[idx(r, c, p)] = vd[idx(r, c, 1)] - vd[idx(r, c, 0)];
                        else if (p == D - 1)
                            gzd[idx(r, c, p)] = vd[idx(r, c, D - 1)] - vd[idx(r, c, D - 2)];
                        else
                            gzd[idx(r, c, p)] = 0.5 * (vd[idx(r, c, p + 1)] - vd[idx(r, c, p - 1)]);
                    }
        } else {
            // intermediate: forward diff with last slice = 0.
            for (keel::usize p = 0; p < D; ++p)
                for (keel::usize r = 0; r < H; ++r)
                    for (keel::usize c = 0; c + 1 < W; ++c)
                        gxd[idx(r, c, p)] = vd[idx(r, c + 1, p)] - vd[idx(r, c, p)];
            for (keel::usize p = 0; p < D; ++p)
                for (keel::usize c = 0; c < W; ++c)
                    for (keel::usize r = 0; r + 1 < H; ++r)
                        gyd[idx(r, c, p)] = vd[idx(r + 1, c, p)] - vd[idx(r, c, p)];
            for (keel::usize c = 0; c < W; ++c)
                for (keel::usize r = 0; r < H; ++r)
                    for (keel::usize p = 0; p + 1 < D; ++p)
                        gzd[idx(r, c, p)] = vd[idx(r, c, p + 1)] - vd[idx(r, c, p)];
        }
    } else {
        throw Error("imgradientxyz: method must be 'sobel', 'prewitt', "
                    "'central', or 'intermediate'",
                    0, 0, "imgradientxyz", "", "numkit:imgradientxyz:method");
    }

    if (outT == ValueType::SINGLE) {
        // Cast to single.
        auto cast_single = [&](Value &G) {
            const keel::usize N = G.numel();
            Value gs = zeros_like(G, ValueType::SINGLE, mr);
            for (keel::usize i = 0; i < N; ++i)
                gs.singleDataMut()[i] = keel::as<keel::f32>(G.doubleData()[i]);
            G = std::move(gs);
        };
        cast_single(Gx); cast_single(Gy); cast_single(Gz);
    }
    return std::make_tuple(std::move(Gx), std::move(Gy), std::move(Gz));
}

std::tuple<Value, Value, Value>
imgradient3_from_grads(const Value &Gx, const Value &Gy, const Value &Gz,
                       keel::Allocator *mr)
{
    if (Gx.dims().rows() != Gy.dims().rows()
        || Gx.dims().cols() != Gy.dims().cols()
        || Gx.dims().pages() != Gy.dims().pages()
        || Gx.dims().rows() != Gz.dims().rows()
        || Gx.dims().cols() != Gz.dims().cols()
        || Gx.dims().pages() != Gz.dims().pages())
        throw Error("imgradient3: Gx, Gy, Gz must have the same size",
                    0, 0, "imgradient3", "", "numkit:imgradient3:size");
    const keel::usize N = Gx.numel();
    const ValueType outT = (Gx.type() == ValueType::SINGLE
                         || Gy.type() == ValueType::SINGLE
                         || Gz.type() == ValueType::SINGLE)
                         ? ValueType::SINGLE : ValueType::DOUBLE;
    auto alloc = [&]() {
        return zeros_like(Gx, outT, mr);
    };
    Value Gmag = alloc(), Gaz = alloc(), Gelev = alloc();
    constexpr keel::f64 D2R = 180.0 / 3.14159265358979323846;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 gx = Gx.elemAsDouble(i);
        const keel::f64 gy = Gy.elemAsDouble(i);
        const keel::f64 gz = Gz.elemAsDouble(i);
        const keel::f64 mag = std::hypot(std::hypot(gx, gy), gz);
        const keel::f64 az  = std::atan2(-gy, gx) * D2R;
        const keel::f64 el  = std::atan2(gz, std::hypot(gx, gy)) * D2R;
        if (outT == ValueType::DOUBLE) {
            Gmag.doubleDataMut()[i]  = mag;
            Gaz.doubleDataMut()[i]   = az;
            Gelev.doubleDataMut()[i] = el;
        } else {
            Gmag.singleDataMut()[i]  = keel::as<keel::f32>(mag);
            Gaz.singleDataMut()[i]   = keel::as<keel::f32>(az);
            Gelev.singleDataMut()[i] = keel::as<keel::f32>(el);
        }
    }
    return std::make_tuple(std::move(Gmag), std::move(Gaz), std::move(Gelev));
}

std::tuple<Value, Value, Value>
imgradient3(const Value &V, keel::StringRef method, keel::Allocator *mr)
{
    auto [Gx, Gy, Gz] = imgradientxyz(V, method, mr);
    return imgradient3_from_grads(Gx, Gy, Gz, mr);
}

std::tuple<Value, Value>
imgradient(const Value &I, keel::StringRef method, keel::Allocator *mr)
{
    auto [Gx, Gy] = imgradientxy(I, method, mr);
    const keel::usize N = Gx.numel();
    Value Gmag = Value::matrix(Gx.dims().rows(), Gx.dims().cols(),
                                ValueType::DOUBLE, mr);
    Value Gdir = Value::matrix(Gx.dims().rows(), Gx.dims().cols(),
                                ValueType::DOUBLE, mr);
    keel::f64 *gm = Gmag.doubleDataMut();
    keel::f64 *gd = Gdir.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 gxv = Gx.elemAsDouble(i);
        const keel::f64 gyv = Gy.elemAsDouble(i);
        gm[i] = std::sqrt(gxv * gxv + gyv * gyv);
        gd[i] = std::atan2(-gyv, gxv) * 180.0 / M_PI;
    }
    return std::make_tuple(std::move(Gmag), std::move(Gdir));
}

// Magnitude-only variant — skips the per-pixel atan2 direction, which costs
// nothing when the caller wants only Gmag (the common `m = imgradient(I)`).
// Gmag is bit-identical to imgradient()'s first output.
Value imgradient_mag(const Value &I, keel::StringRef method, keel::Allocator *mr)
{
    auto [Gx, Gy] = imgradientxy(I, method, mr);
    const keel::usize N = Gx.numel();
    Value Gmag = Value::matrix(Gx.dims().rows(), Gx.dims().cols(), ValueType::DOUBLE, mr);
    keel::f64 *gm = Gmag.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 gxv = Gx.elemAsDouble(i), gyv = Gy.elemAsDouble(i);
        gm[i] = std::sqrt(gxv * gxv + gyv * gyv);
    }
    return Gmag;
}

namespace {

// Threshold (binarise) gradient magnitude into LOGICAL.
Value threshold_to_logical(const Value &G, keel::f64 thresh, keel::Allocator *mr)
{
    const keel::usize H = G.dims().rows();
    const keel::usize W = G.dims().cols();
    Value out = Value::matrix(H, W, ValueType::LOGICAL, mr);
    keel::u8 *od = out.logicalDataMut();
    const keel::usize N = H * W;
    for (keel::usize i = 0; i < N; ++i)
        od[i] = (G.elemAsDouble(i) > thresh) ? 1 : 0;
    return out;
}

keel::f64 auto_threshold(const Value &G, keel::f64 frac) {
    const keel::usize N = G.numel();
    keel::f64 mx = 0.0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 v = std::fabs(G.elemAsDouble(i));
        if (v > mx) mx = v;
    }
    return frac * mx;
}

// Proper Canny edge detector: Gaussian smoothing → Sobel gradient →
// non-maximum suppression (thins ridges to 1 px) → double-threshold
// hysteresis. `userHigh` is the MATLAB-style HIGH threshold as a fraction
// of the peak gradient magnitude (low = 0.4·high); NaN → an auto fraction.
Value canny_edge(const Value &Iin, keel::f64 userHigh, keel::Allocator *mr)
{
    const int H = keel::as<int>(Iin.dims().rows());
    const int W = keel::as<int>(Iin.dims().cols());
    const keel::usize N = keel::as<keel::usize>(H) * keel::as<keel::usize>(W);

    Value out = Value::matrix(H, W, ValueType::LOGICAL, mr);
    keel::u8 *od = out.logicalDataMut();
    std::fill(od, od + N, keel::u8{0});
    if (N == 0) return out;

    // 1–2. Derivative-of-Gaussian gradient (MATLAB Canny approach, σ = √2):
    // smooth along one axis with a Gaussian g and differentiate along the
    // other with its derivative dg. Separable, so two 1-D passes per axis.
    // This avoids the extra smoothing that a Gaussian-blur-then-Sobel chain
    // would add (which weakens weak edges below threshold).
    const keel::f64 sigma = std::sqrt(2.0);
    const int hw = std::max(1, keel::as<int>(std::ceil(3.0 * sigma)));
    const int L = 2 * hw + 1;
    keel::Array<keel::f64> g(L), dg(keel::as<keel::usize>(L));
    keel::f64 gsum = 0.0;
    for (int i = 0; i < L; ++i) {
        const keel::f64 x = i - hw;
        g[i] = std::exp(-(x * x) / (2.0 * sigma * sigma));
        gsum += g[i];
    }
    for (int i = 0; i < L; ++i) g[i] /= gsum;
    for (int i = 0; i < L; ++i) {
        const keel::f64 x = i - hw;
        dg[i] = -x * std::exp(-(x * x) / (2.0 * sigma * sigma));  // scale irrelevant (relative threshold)
    }
    // Filter in DOUBLE. imfilter on an integer image clamps each pass to the
    // integer range — which would zero every negative derivative response
    // (i.e. every falling edge), losing ~half the edges. MATLAB likewise
    // converts to floating point before the Canny gradient.
    Value Id = Value::matrix(H, W, ValueType::DOUBLE, mr);
    {
        keel::f64 *dd = Id.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i) dd[i] = Iin.elemAsDouble(i);
    }

    Value gRow  = make_kernel(g,  1, L, mr);
    Value gCol  = make_kernel(g,  L, 1, mr);
    Value dgRow = make_kernel(dg, 1, L, mr);
    Value dgCol = make_kernel(dg, L, 1, mr);
    Value GxV = apply_kernel(apply_kernel(Id, gCol, mr), dgRow, mr);
    Value GyV = apply_kernel(apply_kernel(Id, gRow, mr), dgCol, mr);

    keel::ArenaTemp arena = keel::get_scratch(mr);
    keel::Array<keel::f64> Gx(N, arena.allocator()), Gy(N, arena.allocator()), mag(N, arena.allocator());
    keel::f64 maxMag = 0.0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 a = GxV.elemAsDouble(i), b = GyV.elemAsDouble(i);
        Gx[i] = a; Gy[i] = b;
        const keel::f64 m = std::sqrt(a * a + b * b);
        mag[i] = m;
        if (m > maxMag) maxMag = m;
    }
    if (maxMag <= 0.0) return out;

    auto at = [&](int r, int c) -> keel::f64 {
        return mag[keel::as<keel::usize>(c) * keel::as<keel::usize>(H) + keel::as<keel::usize>(r)];
    };

    // 3. Non-maximum suppression with sub-pixel interpolation along the
    // gradient direction (as MATLAB does). Interpolating the magnitude
    // between the two straddling neighbours yields better-connected 1-px
    // ridges than 4-way quantisation, which matters for closed-contour use
    // (imfill). Border pixels can't be interpolated → left as non-edges.
    keel::Array<keel::f64> nms(N, arena.allocator());
    std::fill(nms.begin(), nms.end(), 0.0);
    for (int c = 1; c < W - 1; ++c)
        for (int r = 1; r < H - 1; ++r) {
            const keel::usize i = keel::as<keel::usize>(c) * keel::as<keel::usize>(H) + keel::as<keel::usize>(r);
            const keel::f64 m = mag[i];
            if (m <= 0.0) continue;
            const keel::f64 gx = Gx[i], gy = Gy[i];
            const keel::f64 agx = std::fabs(gx), agy = std::fabs(gy);
            keel::f64 mf, mb, w;
            if (agx >= agy) {                        // gradient more horizontal
                w = (agx > 0.0) ? agy / agx : 0.0;
                if (gx * gy >= 0.0) {                // ↘ / ↖
                    mf = (1.0 - w) * at(r, c + 1) + w * at(r + 1, c + 1);
                    mb = (1.0 - w) * at(r, c - 1) + w * at(r - 1, c - 1);
                } else {                             // ↗ / ↙
                    mf = (1.0 - w) * at(r, c + 1) + w * at(r - 1, c + 1);
                    mb = (1.0 - w) * at(r, c - 1) + w * at(r + 1, c - 1);
                }
            } else {                                 // gradient more vertical
                w = (agy > 0.0) ? agx / agy : 0.0;
                if (gx * gy >= 0.0) {
                    mf = (1.0 - w) * at(r + 1, c) + w * at(r + 1, c + 1);
                    mb = (1.0 - w) * at(r - 1, c) + w * at(r - 1, c - 1);
                } else {
                    mf = (1.0 - w) * at(r + 1, c) + w * at(r + 1, c - 1);
                    mb = (1.0 - w) * at(r - 1, c) + w * at(r - 1, c + 1);
                }
            }
            if (m >= mf && m >= mb) nms[i] = m;
        }

    // 4. Thresholds, relative to the peak gradient magnitude.
    const keel::f64 highFrac = (std::isnan(userHigh) || userHigh <= 0.0) ? 0.2 : userHigh;
    const keel::f64 highT = highFrac * maxMag;
    const keel::f64 lowT  = 0.4 * highT;

    // 5. Hysteresis: strong pixels (≥ highT) seed a flood that keeps any
    // weak pixel (≥ lowT) reachable through 8-connectivity.
    keel::Array<keel::usize> stack(0, arena.allocator());
    stack.reserve(N / 8 + 1);
    for (keel::usize i = 0; i < N; ++i)
        if (nms[i] >= highT) { od[i] = 1; stack.push_back(i); }
    static constexpr int dr8[8] = { -1, -1, -1, 0, 0, 1, 1, 1 };
    static constexpr int dc8[8] = { -1, 0, 1, -1, 1, -1, 0, 1 };
    while (!stack.empty()) {
        const keel::usize p = stack.back(); stack.pop_back();
        const int r = keel::as<int>(p % keel::as<keel::usize>(H));
        const int c = keel::as<int>(p / keel::as<keel::usize>(H));
        for (int k = 0; k < 8; ++k) {
            const int nr = r + dr8[k], nc = c + dc8[k];
            if (nr < 0 || nr >= H || nc < 0 || nc >= W) continue;
            const keel::usize q = keel::as<keel::usize>(nc) * keel::as<keel::usize>(H) + keel::as<keel::usize>(nr);
            if (!od[q] && nms[q] >= lowT) { od[q] = 1; stack.push_back(q); }
        }
    }
    return out;
}

} // anonymous

Value edge(const Value &I, keel::StringRef methodRaw, keel::f64 thresh_lo, keel::f64 /*thresh_hi*/, keel::Allocator *mr)
{
    // Method names are case-insensitive (MATLAB-compatible).
    keel::String method = keel::String::from(methodRaw);
    for (char &ch : method) ch = keel::to_ascii_lower(ch);

    // Sobel / Prewitt / Roberts threshold the gradient magnitude; Canny does
    // full non-max suppression + hysteresis; log / zerocross mark zero
    // crossings.
    if (method == "roberts") {
        // 2×2 Roberts kernels.
        Value Kx = make_kernel({ 1, 0, 0, -1 }, 2, 2, mr);
        Value Ky = make_kernel({ 0, 1, -1, 0 }, 2, 2, mr);
        Value Gx = apply_kernel(I, Kx, mr);
        Value Gy = apply_kernel(I, Ky, mr);
        const keel::usize N = Gx.numel();
        Value G = Value::matrix(Gx.dims().rows(), Gx.dims().cols(),
                                 ValueType::DOUBLE, mr);
        keel::f64 *gd = G.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 a = Gx.elemAsDouble(i), b = Gy.elemAsDouble(i);
            gd[i] = std::sqrt(a * a + b * b);
        }
        if (std::isnan(thresh_lo)) thresh_lo = auto_threshold(G, 0.5);
        return threshold_to_logical(G, thresh_lo, mr);
    }
    if (method == "log" || method == "zerocross") {
        // Apply LoG kernel, then mark zero-crossings as edges.
        const int hsz = 5;
        Value Klog = fspecial("log", { (keel::f64)hsz, (keel::f64)hsz, 0.5 }, mr);
        Value Y = apply_kernel(I, Klog, mr);
        const int H = (int)Y.dims().rows();
        const int W = (int)Y.dims().cols();
        Value out = Value::matrix(H, W, ValueType::LOGICAL, mr);
        keel::u8 *od = out.logicalDataMut();
        const keel::f64 th = std::isnan(thresh_lo) ? auto_threshold(Y, 0.05) : thresh_lo;
        auto y = [&](int r, int c) { return Y.elemAsDouble((keel::usize)c * (keel::usize)H + (keel::usize)r); };
        for (int c = 0; c < W; ++c)
            for (int r = 0; r < H; ++r) {
                bool zc = false;
                for (int dr = -1; dr <= 1 && !zc; ++dr)
                    for (int dc = -1; dc <= 1 && !zc; ++dc) {
                        if (dr == 0 && dc == 0) continue;
                        const int rr = r + dr, cc = c + dc;
                        if (rr < 0 || rr >= H || cc < 0 || cc >= W) continue;
                        const keel::f64 a = y(r, c), b = y(rr, cc);
                        if (a * b < 0.0 && std::fabs(a - b) > th) zc = true;
                    }
                od[(keel::usize)c * (keel::usize)H + (keel::usize)r] = zc ? 1 : 0;
            }
        return out;
    }
    if (method == "canny") {
        // Full Canny: Gaussian smooth → Sobel gradient → non-max
        // suppression → double-threshold hysteresis. `thresh_lo` carries
        // the user's scalar threshold, which MATLAB treats as the HIGH
        // threshold (a fraction of the peak gradient magnitude); NaN → auto.
        return canny_edge(I, thresh_lo, mr);
    }
    // Default sobel / prewitt path: grad-magnitude threshold.
    const keel::StringRef mth = (method == "prewitt") ? "prewitt" : "sobel";
    auto [Gmag, _] = imgradient(I, mth, mr);
    if (std::isnan(thresh_lo)) thresh_lo = auto_threshold(Gmag, 0.4);
    return threshold_to_logical(Gmag, thresh_lo, mr);
}

// ── cornermetric (Harris / Shi-Tomasi) ────────────────────────────
//
// MATLAB R2025b cornermetric.m algorithm (transliterated):
//
//   Convert I → double via im2double.
//   Dx = imfilter(I, [-1 0 1],  'replicate', 'conv')
//   Dy = imfilter(I, [-1 0 1]', 'replicate', 'conv')
//   Trim 1px border on Dx, Dy (i.e. take rows 2..end-1, cols 2..end-1).
//   A = Dx², B = Dy², C = Dx·Dy.
//   W = filter_coef * filter_coef'      (outer-product 2-D kernel)
//   A,B,C ← imfilter(., W, 'replicate', 'full', 'conv').
//   removed = (len(filter_coef) - 1) / 2 - 1; crop back to image size.
//   Harris:           C* = A·B - C² - k·(A+B)²
//   MinimumEigenvalue: C* = ((A+B) - sqrt((A-B)² + 4·C²)) / 2.
//
// References:
//   Harris & Stephens, "A Combined Corner and Edge Detector", 1988.
//   Shi & Tomasi,     "Good Features to Track",                1994.
Value cornermetric(const Value &I, keel::StringRef method,
                   keel::f64 sensitivity_factor,
                   const Value &filter_coef,
                   keel::Allocator *mr)
{
    // Probed R2025b: every numeric class (result double); complex dies
    // with expectedReal.
    internal::requireImageClass(I, "cornermetric", "I", 1,
                                internal::ImgClasses::FloatIntLog);
    if (I.dims().is3D())
        throw Error("cornermetric: I must be 2-D",
                    0, 0, "cornermetric", "", "numkit:cornermetric:dim");
    if (method != "Harris" && method != "MinimumEigenvalue")
        throw Error("cornermetric: METHOD must be 'Harris' or "
                    "'MinimumEigenvalue'",
                    0, 0, "cornermetric", "", "numkit:cornermetric:method");
    if (method == "Harris"
        && (!(sensitivity_factor > 0.0) || !(sensitivity_factor < 0.25)))
        throw Error("cornermetric: SensitivityFactor must be in (0, 0.25)",
                    0, 0, "cornermetric", "", "numkit:cornermetric:k");

    // Resolve filter coef vector.
    keel::Array<keel::f64> fcoef(mr);
    if (filter_coef.numel() == 0) {
        // Default fspecial('gaussian', [5 1], 1.5).
        // Hardcoded exact values: g = exp(-x²/(2σ²)), normalised.
        const keel::f64 raw[5] = {
            0.12007838770739, 0.23388075658030,
            0.29208171142462,
            0.23388075658030, 0.12007838770739};
        fcoef.assign(keel::ArrayRef<keel::f64>(raw, 5));
    } else {
        if (filter_coef.numel() < 3 || (filter_coef.numel() % 2) == 0)
            throw Error("cornermetric: FilterCoefficients length must "
                        "be odd and ≥ 3",
                        0, 0, "cornermetric", "",
                        "numkit:cornermetric:filter");
        fcoef.reserve(filter_coef.numel());
        for (keel::usize i = 0; i < filter_coef.numel(); ++i)
            fcoef.push_back(filter_coef.elemAsDouble(i));
    }
    const keel::usize Lc = fcoef.size();
    const keel::usize halfLc = Lc / 2;       // (Lc-1)/2
    if (halfLc < 1)
        throw Error("cornermetric: filter coef too short",
                    0, 0, "cornermetric", "", "numkit:cornermetric:filter");
    const keel::usize removed = halfLc - 1;  // matches MATLAB

    // Promote I → DOUBLE via im2double.
    Value Id = im2double(I, mr);
    const keel::usize H = Id.dims().rows();
    const keel::usize W = Id.dims().cols();
    if (H < 3 || W < 3)
        throw Error("cornermetric: image too small (need ≥ 3x3)",
                    0, 0, "cornermetric", "", "numkit:cornermetric:tooSmall");

    // Build [-1 0 1] row + column kernels.
    Value hx = Value::matrix(1, 3, ValueType::DOUBLE, mr);
    hx.doubleDataMut()[0] = -1; hx.doubleDataMut()[1] = 0; hx.doubleDataMut()[2] = 1;
    Value hy = Value::matrix(3, 1, ValueType::DOUBLE, mr);
    hy.doubleDataMut()[0] = -1; hy.doubleDataMut()[1] = 0; hy.doubleDataMut()[2] = 1;

    Value Dx = imfilter(Id, hx, PadMode::Replicate, 0.0,
                        /*full=*/false, /*flip_kernel=*/true, mr);
    Value Dy = imfilter(Id, hy, PadMode::Replicate, 0.0,
                        /*full=*/false, /*flip_kernel=*/true, mr);

    // Trim 1px border: take rows 1..H-2, cols 1..W-2 (0-indexed).
    const keel::usize Ht = H - 2;
    const keel::usize Wt = W - 2;
    if (Ht == 0 || Wt == 0)
        throw Error("cornermetric: image too small after gradient trim",
                    0, 0, "cornermetric", "", "numkit:cornermetric:tooSmall");

    // Build A = Dx², B = Dy², C = Dx·Dy at trimmed shape.
    Value A = Value::matrix(Ht, Wt, ValueType::DOUBLE, mr);
    Value B = Value::matrix(Ht, Wt, ValueType::DOUBLE, mr);
    Value Cmat = Value::matrix(Ht, Wt, ValueType::DOUBLE, mr);
    for (keel::usize c = 0; c < Wt; ++c) {
        for (keel::usize r = 0; r < Ht; ++r) {
            const keel::usize src = (c + 1) * H + (r + 1);
            const keel::f64 dx = Dx.doubleData()[src];
            const keel::f64 dy = Dy.doubleData()[src];
            const keel::usize dst = c * Ht + r;
            A.doubleDataMut()[dst] = dx * dx;
            B.doubleDataMut()[dst] = dy * dy;
            Cmat.doubleDataMut()[dst] = dx * dy;
        }
    }

    // W = filter_coef * filter_coef' (Lc x Lc outer-product kernel).
    Value Wk = Value::matrix(Lc, Lc, ValueType::DOUBLE, mr);
    for (keel::usize c = 0; c < Lc; ++c)
        for (keel::usize r = 0; r < Lc; ++r)
            Wk.doubleDataMut()[c * Lc + r] = fcoef[r] * fcoef[c];

    // Filter A, B, C with 'full' shape under Replicate boundary —
    // MATLAB's `imfilter(.,W,'replicate','full','conv')` pads the
    // input by (Lc-1)/2 replicate on each side and then runs the
    // convolution at the larger size. nk's `imfilter(full=true)`
    // uses zero-padding regardless of PadMode, so we pre-pad by
    // (Lc-1)/2 with `padarray` and run a 'same' convolution.
    auto full_replicate = [&](Value X) {
        X = padarray(X, {keel::as<int>(halfLc), keel::as<int>(halfLc)}, PadMode::Replicate, 0.0, "both", mr);
        return imfilter(X, Wk, PadMode::Replicate, 0.0,
                        /*full=*/false, /*flip_kernel=*/true, mr);
    };
    A = full_replicate(A);
    B = full_replicate(B);
    Cmat = full_replicate(Cmat);

    // Crop back to (H, W) image size: rows [removed+1, end-removed].
    const keel::usize Ha = A.dims().rows();
    const keel::usize Wa = A.dims().cols();
    if (Ha < H || Wa < W)
        throw Error("cornermetric: internal shape error after filter",
                    0, 0, "cornermetric", "", "numkit:cornermetric:shape");
    auto crop_value = [&](const Value &X) {
        Value Y = Value::matrix(H, W, ValueType::DOUBLE, mr);
        for (keel::usize c = 0; c < W; ++c) {
            for (keel::usize r = 0; r < H; ++r) {
                const keel::usize sr = r + removed;
                const keel::usize sc = c + removed;
                Y.doubleDataMut()[c * H + r]
                    = X.doubleData()[sc * Ha + sr];
            }
        }
        return Y;
    };
    A = crop_value(A);
    B = crop_value(B);
    Cmat = crop_value(Cmat);

    // Compute cornerness.
    Value out = Value::matrix(H, W, ValueType::DOUBLE, mr);
    const keel::f64 *ap = A.doubleData();
    const keel::f64 *bp = B.doubleData();
    const keel::f64 *cp = Cmat.doubleData();
    keel::f64 *op = out.doubleDataMut();
    const keel::usize N = H * W;
    if (method == "Harris") {
        const keel::f64 k = sensitivity_factor;
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 sum = ap[i] + bp[i];
            op[i] = ap[i] * bp[i] - cp[i] * cp[i] - k * sum * sum;
        }
    } else { // MinimumEigenvalue
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 sum = ap[i] + bp[i];
            const keel::f64 diff = ap[i] - bp[i];
            op[i] = (sum - std::sqrt(diff * diff + 4.0 * cp[i] * cp[i])) * 0.5;
        }
    }
    return out;
}

// ── corner (peak detection on the cornermetric) ────────────────────
//
// MATLAB R2025b corner.m: compute the cornermetric, keep its local
// maxima above QualityLevel·max(metric), take each connected peak's
// centroid, sort by strength (descending), and return up to N integer
// [x y] = [col row] coordinates. Built on the shipped cornermetric; the
// border is excluded naturally (the metric there is ≤ 0 < threshold).
Value corner(const Value &I, keel::StringRef method, int maxN,
             keel::f64 quality_level, keel::f64 sensitivity,
             const Value &filter_coef, keel::Allocator *mr)
{
    internal::requireImageClass(I, "corner", "I", 1,
                                internal::ImgClasses::FloatIntLog);
    Value Cm = cornermetric(I, method, sensitivity, filter_coef, mr);
    const keel::usize M = Cm.dims().rows();
    const keel::usize N = Cm.dims().cols();
    const keel::usize MN = M * N;

    auto emptyOut = [&]() { return Value::matrix(0, 2, ValueType::DOUBLE, mr); };
    if (MN == 0) return emptyOut();
    const keel::f64 *m = Cm.doubleData();

    keel::f64 mx = m[0];
    for (keel::usize i = 1; i < MN; ++i) if (m[i] > mx) mx = m[i];
    if (!(mx > 0.0)) return emptyOut();
    const keel::f64 thr = quality_level * mx;

    keel::ArenaTemp arena = keel::get_scratch(mr);
    auto at = [&](keel::usize r, keel::usize c) -> keel::f64 { return m[c * M + r]; };

    // Local-maximum mask: strictly above threshold AND ≥ every in-bounds
    // 8-neighbor.
    keel::Array<keel::u8> isMax(MN, arena.allocator());
    for (keel::usize i = 0; i < MN; ++i) isMax[i] = 0;
    for (keel::usize c = 0; c < N; ++c) {
        for (keel::usize r = 0; r < M; ++r) {
            const keel::f64 v = at(r, c);
            if (!(v > thr)) continue;
            bool peak = true;
            for (int dc = -1; dc <= 1 && peak; ++dc)
                for (int dr = -1; dr <= 1 && peak; ++dr) {
                    if (dr == 0 && dc == 0) continue;
                    const long long rr = (long long)r + dr, cc = (long long)c + dc;
                    if (rr < 0 || cc < 0 || rr >= (long long)M || cc >= (long long)N)
                        continue;
                    if (at((keel::usize)rr, (keel::usize)cc) > v) peak = false;
                }
            if (peak) isMax[c * M + r] = 1;
        }
    }

    // Connected components (8-conn), discovered in column-major order so
    // equal-strength peaks tie-break by ascending linear index (matching
    // MATLAB's find() order). Each component → centroid + peak strength.
    // `key` quantises the strength to ~1e-9·max so that ULP-level noise
    // (nk's cornermetric matches MATLAB only to ~1e-8, and is not
    // bit-exactly symmetric at symmetric corners) collapses to a single
    // value — MATLAB's metric IS bit-exact there, so its equal-strength
    // corners keep find() order under the strength sort; quantising lets
    // ours do the same while still ordering genuinely distinct strengths.
    const keel::f64 quantum = mx * 1e-9;
    struct Peak { keel::f64 row, col, val; long long key; };
    keel::Array<Peak> peaks(0, arena.allocator());
    keel::Array<keel::u8> seen(MN, arena.allocator());
    for (keel::usize i = 0; i < MN; ++i) seen[i] = 0;
    keel::Array<keel::usize> stack(0, arena.allocator());
    for (keel::usize idx = 0; idx < MN; ++idx) {
        if (!isMax[idx] || seen[idx]) continue;
        keel::f64 sumR = 0, sumC = 0, best = m[idx];
        keel::usize cnt = 0;
        stack.clear();
        stack.push_back(idx);
        seen[idx] = 1;
        while (!stack.empty()) {
            const keel::usize p = stack.back(); stack.pop_back();
            const keel::usize pr = p % M, pc = p / M;
            sumR += (keel::f64)pr; sumC += (keel::f64)pc; ++cnt;
            if (m[p] > best) best = m[p];
            for (int dc = -1; dc <= 1; ++dc)
                for (int dr = -1; dr <= 1; ++dr) {
                    if (dr == 0 && dc == 0) continue;
                    const long long rr = (long long)pr + dr, cc = (long long)pc + dc;
                    if (rr < 0 || cc < 0 || rr >= (long long)M || cc >= (long long)N)
                        continue;
                    const keel::usize q = (keel::usize)cc * M + (keel::usize)rr;
                    if (isMax[q] && !seen[q]) { seen[q] = 1; stack.push_back(q); }
                }
        }
        peaks.push_back({ sumR / (keel::f64)cnt, sumC / (keel::f64)cnt, best,
                          std::llround(best / quantum) });
    }

    // Stable sort by quantised strength descending (ties keep column-major
    // order — MATLAB's find() order over equal-strength corners).
    std::stable_sort(peaks.begin(), peaks.end(),
                     [](const Peak &a, const Peak &b) { return a.key > b.key; });

    keel::usize K = peaks.size();
    if (maxN >= 0 && (keel::usize)maxN < K) K = (keel::usize)maxN;

    // Output K×2: column 0 = x (col), column 1 = y (row), 1-indexed,
    // centroid rounded half-away-from-zero (MATLAB round).
    Value out = Value::matrix(K, 2, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < K; ++i) {
        od[i]     = (keel::f64)(std::llround(peaks[i].col) + 1);   // x
        od[K + i] = (keel::f64)(std::llround(peaks[i].row) + 1);   // y
    }
    return out;
}

// ── hough (Standard Hough Transform) ───────────────────────────────
//
// MATLAB R2025b hough.m algorithm:
//   M, N = size(BW); D = sqrt((M-1)² + (N-1)²)
//   q = ceil(D / rhoRes); nrho = 2*q + 1
//   rho = linspace(-q*rhoRes, q*rhoRes, nrho)
//   theta default = -90 : 1 : 89  (180 bins, [-90, 90))
//   H = zeros(nrho, ntheta)
//   for each true pixel (r, c)  // 1-indexed
//     x = c - 1, y = r - 1       // 0-indexed image coords
//     for each theta_k:
//       rho_val = x*cos(theta_k) + y*sin(theta_k)
//       bin = round(rho_val / rhoRes) + q + 1  // 1-indexed
//       H[bin, k] += 1
//
// Reference: Gonzalez, Woods & Eddins, "Digital Image Processing
// Using MATLAB", 2nd ed., Gatesmark, 2009.
void hough(const Value &BW, keel::f64 rho_res,
           const Value &theta_deg,
           Value &H_out, Value &T_out, Value &R_out,
           keel::Allocator *mr)
{
    if (BW.dims().is3D())
        throw Error("hough: BW must be 2-D",
                    0, 0, "hough", "", "numkit:hough:dim");
    if (!(rho_res > 0.0) || !std::isfinite(rho_res))
        throw Error("hough: RhoResolution must be a positive scalar",
                    0, 0, "hough", "", "numkit:hough:rho");

    const keel::usize M = BW.dims().rows();
    const keel::usize N = BW.dims().cols();

    // Build theta grid (degrees → radians).
    keel::Array<keel::f64> theta(mr);
    if (theta_deg.numel() == 0) {
        theta.reserve(180);
        for (int k = -90; k <= 89; ++k)
            theta.push_back(keel::as<keel::f64>(k));
    } else {
        theta.reserve(theta_deg.numel());
        for (keel::usize i = 0; i < theta_deg.numel(); ++i) {
            const keel::f64 v = theta_deg.elemAsDouble(i);
            if (v < -90.0 || v >= 90.0)
                throw Error("hough: Theta values must lie in [-90, 90)",
                            0, 0, "hough", "", "numkit:hough:theta");
            theta.push_back(v);
        }
    }
    const keel::usize ntheta = theta.size();

    // ρ grid.
    const keel::f64 D = std::sqrt(
        keel::as<keel::f64>(M - 1) * (M - 1)
      + keel::as<keel::f64>(N - 1) * (N - 1));
    const keel::usize q = (D == 0.0)
        ? 0
        : keel::as<keel::usize>(std::ceil(D / rho_res));
    const keel::usize nrho = 2 * q + 1;
    Value R = Value::matrix(1, nrho, ValueType::DOUBLE, mr);
    if (nrho == 1) {
        R.doubleDataMut()[0] = 0.0;
    } else {
        const keel::f64 lo = -keel::as<keel::f64>(q) * rho_res;
        const keel::f64 step = rho_res;
        for (keel::usize i = 0; i < nrho; ++i)
            R.doubleDataMut()[i] = lo + step * keel::as<keel::f64>(i);
    }

    // T echo.
    Value T = Value::matrix(1, ntheta, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < ntheta; ++i)
        T.doubleDataMut()[i] = theta[i];

    // Pre-compute cos/sin of theta (in radians).
    constexpr keel::f64 DEG2RAD = M_PI / 180.0;
    keel::Array<keel::f64> cosT(ntheta, 0.0, mr);
    keel::Array<keel::f64> sinT(ntheta, 0.0, mr);
    for (keel::usize k = 0; k < ntheta; ++k) {
        cosT[k] = std::cos(theta[k] * DEG2RAD);
        sinT[k] = std::sin(theta[k] * DEG2RAD);
    }

    // Accumulator (col-major: H(r, c) → c*nrho + r).
    Value H = Value::matrix(nrho, ntheta, ValueType::DOUBLE, mr);
    keel::f64 *hp = H.doubleDataMut();

    // Iterate true pixels of BW (logical or numeric "non-zero").
    const bool islog = BW.isLogical();
    for (keel::usize c = 0; c < N; ++c) {
        for (keel::usize r = 0; r < M; ++r) {
            const keel::usize k = c * M + r;
            const bool on = islog
                ? (BW.logicalData()[k] != 0)
                : (BW.elemAsDouble(k) != 0.0);
            if (!on) continue;
            const keel::f64 x = keel::as<keel::f64>(c);  // 0-indexed
            const keel::f64 y = keel::as<keel::f64>(r);
            for (keel::usize tk = 0; tk < ntheta; ++tk) {
                const keel::f64 rho_val = x * cosT[tk] + y * sinT[tk];
                // bin = round(rho_val / rho_res) + q + 1 (1-indexed)
                long bin = keel::as<long>(std::lround(rho_val / rho_res))
                         + keel::as<long>(q);  // 0-indexed
                if (bin < 0 || keel::as<keel::usize>(bin) >= nrho) continue;
                hp[tk * nrho + bin] += 1.0;
            }
        }
    }

    H_out = std::move(H);
    T_out = std::move(T);
    R_out = std::move(R);
}

// ── houghpeaks (peak extraction from Hough accumulator) ───────────
Value houghpeaks(const Value &H, keel::usize numpeaks,
                 keel::f64 threshold,
                 keel::usize nhoodRho, keel::usize nhoodTheta,
                 const Value &theta_deg,
                 keel::Allocator *mr)
{
    // Probed R2025b: the accumulator must be real double (single and
    // uint8 accumulators are rejected).
    internal::requireImageClass(H, "houghpeaks", "H", 1,
                                internal::ImgClasses::DoubleOnly);
    if (H.dims().is3D())
        throw Error("houghpeaks: H must be 2-D",
                    0, 0, "houghpeaks", "", "numkit:houghpeaks:dim");
    const keel::usize nrho = H.dims().rows();
    const keel::usize ntheta = H.dims().cols();
    const keel::usize N = nrho * ntheta;

    // Default neighbourhood: ceil(size(H)/50), bumped up to next odd, min 1.
    auto next_odd = [](keel::usize v) -> keel::usize {
        if (v < 1) v = 1;
        if ((v % 2) == 0) v += 1;
        return v;
    };
    if (nhoodRho == 0)   nhoodRho   = next_odd((nrho + 49) / 50);
    if (nhoodTheta == 0) nhoodTheta = next_odd((ntheta + 49) / 50);
    if ((nhoodRho   % 2) == 0) ++nhoodRho;
    if ((nhoodTheta % 2) == 0) ++nhoodTheta;

    // Default threshold: 0.5 * max(H(:)).
    keel::f64 mxH = 0.0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 v = H.elemAsDouble(i);
        if (v > mxH) mxH = v;
    }
    if (threshold < 0.0) threshold = 0.5 * mxH;

    // Detect antisymmetric theta range.
    bool isThetaAntisym = false;
    if (theta_deg.numel() >= 2) {
        const keel::usize nT = theta_deg.numel();
        keel::f64 minT = theta_deg.elemAsDouble(0);
        keel::f64 maxT = theta_deg.elemAsDouble(0);
        for (keel::usize i = 1; i < nT; ++i) {
            const keel::f64 v = theta_deg.elemAsDouble(i);
            if (v < minT) minT = v;
            if (v > maxT) maxT = v;
        }
        const keel::f64 thetaRes
            = std::fabs(maxT - minT) / keel::as<keel::f64>(nT - 1);
        isThetaAntisym
            = std::fabs(minT + thetaRes * nhoodTheta) <= maxT;
    } else if (theta_deg.numel() == 0) {
        // Default -90:1:89: minT=-90, maxT=89, thetaRes=1.
        isThetaAntisym = std::fabs(-90.0 + 1.0 * nhoodTheta) <= 89.0;
    }

    // Working copy of H (DOUBLE).
    keel::Array<keel::f64> Hwork(N, 0.0, mr);
    for (keel::usize i = 0; i < N; ++i) Hwork[i] = H.elemAsDouble(i);

    const keel::usize halfR = nhoodRho / 2;
    const keel::usize halfT = nhoodTheta / 2;

    keel::Array<keel::usize> peak_r(mr), peak_c(mr);
    peak_r.reserve(numpeaks);
    peak_c.reserve(numpeaks);

    while (peak_r.size() < numpeaks) {
        // Find global max.
        keel::usize maxIdx = 0;
        keel::f64 maxV = Hwork[0];
        for (keel::usize i = 1; i < N; ++i) {
            if (Hwork[i] > maxV) { maxV = Hwork[i]; maxIdx = i; }
        }
        if (maxV < threshold) break;
        const keel::usize p = maxIdx % nrho;  // 0-indexed rho
        const keel::usize q = maxIdx / nrho;  // 0-indexed theta
        peak_r.push_back(p);
        peak_c.push_back(q);

        // Suppress nhood.
        const long p1 = keel::as<long>(p) - keel::as<long>(halfR);
        const long p2 = keel::as<long>(p) + keel::as<long>(halfR);
        const long q1 = keel::as<long>(q) - keel::as<long>(halfT);
        const long q2 = keel::as<long>(q) + keel::as<long>(halfT);
        for (long qq = q1; qq <= q2; ++qq) {
            for (long pp = p1; pp <= p2; ++pp) {
                // Out-of-bounds in rho: drop.
                if (pp < 0 || pp >= keel::as<long>(nrho)) continue;
                long qWrap = qq;
                long pWrap = pp;
                if (qWrap < 0 || qWrap >= keel::as<long>(ntheta)) {
                    if (isThetaAntisym) {
                        if (qWrap < 0) {
                            qWrap += keel::as<long>(ntheta);
                            pWrap = keel::as<long>(nrho) - 1 - pp;
                        } else {
                            qWrap -= keel::as<long>(ntheta);
                            pWrap = keel::as<long>(nrho) - 1 - pp;
                        }
                        if (pWrap < 0 || pWrap >= keel::as<long>(nrho))
                            continue;
                    } else {
                        continue;
                    }
                }
                Hwork[keel::as<keel::usize>(qWrap) * nrho
                    + keel::as<keel::usize>(pWrap)] = 0.0;
            }
        }
    }

    // Pack output (P × 2): each row [rho_idx, theta_idx], 1-indexed.
    const keel::usize nP = peak_r.size();
    Value P = Value::matrix(nP, 2, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < nP; ++i) {
        P.doubleDataMut()[i]      = keel::as<keel::f64>(peak_r[i] + 1);
        P.doubleDataMut()[nP + i] = keel::as<keel::f64>(peak_c[i] + 1);
    }
    return P;
}

// ── houghlines (line segment extraction from Hough peaks) ────────
//
// MATLAB R2025b houghlines.m algorithm:
//   nonzeropix = [x y]; with x=col-1, y=row-1 from find(BW).
//   For each peak (rbin, cbin):
//     theta_c = theta(cbin) * pi/180
//     rho_xy  = x*cos(theta_c) + y*sin(theta_c)
//     slope = (nrho - 1) / (rho(end) - rho(1))
//     rho_bin_index = round(slope * (rho_xy - rho(1)) + 1)
//     idx = where rho_bin_index == rbin
//     r = y(idx) + 1, c = x(idx) + 1  (back to 1-based pixel coords)
//     resort pixels along the dominant axis (row vs col range)
//     xy = [c r]
//     diff_sq = sum((xy(i+1,:) - xy(i,:))^2, 2)
//     fillgap_idx = where diff_sq > fillgap^2
//     split xy at gaps; for each segment [p1, p2]:
//       if ||p2 - p1||^2 >= minlength^2:
//         add line struct {point1=p1, point2=p2, theta, rho}
//
// Reference: Gonzalez/Woods/Eddins, *Digital Image Processing
// Using MATLAB*, Prentice Hall 2003.
Value houghlines(const Value &BW, const Value &theta_deg,
                 const Value &rho, const Value &peaks,
                 keel::f64 fillgap, keel::f64 minlength,
                 keel::Allocator *mr)
{
    internal::requireImageClass(BW, "houghlines", "BW", 1,
                                internal::ImgClasses::AnyNumericReal);
    if (BW.dims().is3D())
        throw Error("houghlines: BW must be 2-D",
                    0, 0, "houghlines", "", "numkit:houghlines:dim");
    if (!(fillgap > 0.0) || !std::isfinite(fillgap))
        throw Error("houghlines: FillGap must be positive",
                    0, 0, "houghlines", "", "numkit:houghlines:fillgap");
    if (!(minlength > 0.0) || !std::isfinite(minlength))
        throw Error("houghlines: MinLength must be positive",
                    0, 0, "houghlines", "", "numkit:houghlines:minlength");

    const keel::usize M = BW.dims().rows();
    const keel::usize N = BW.dims().cols();
    const keel::usize nrho = rho.numel();
    const keel::usize ntheta = theta_deg.numel();
    if (nrho < 2 || ntheta == 0)
        throw Error("houghlines: rho/theta vectors too small",
                    0, 0, "houghlines", "", "numkit:houghlines:vec");
    if (peaks.dims().cols() != 2 || peaks.dims().is3D())
        throw Error("houghlines: PEAKS must be P × 2",
                    0, 0, "houghlines", "", "numkit:houghlines:peaks");
    const keel::usize nPeaks = peaks.dims().rows();

    // Gather nonzero pixel (x, y) = (col-1, row-1).
    const bool islog = BW.isLogical();
    keel::Array<keel::f64> px(mr), py(mr);
    px.reserve(M * N / 16);
    py.reserve(M * N / 16);
    // Walk column-major to match MATLAB's find() order.
    for (keel::usize c = 0; c < N; ++c) {
        for (keel::usize r = 0; r < M; ++r) {
            const keel::usize k = c * M + r;
            const bool on = islog
                ? (BW.logicalData()[k] != 0)
                : (BW.elemAsDouble(k) != 0.0);
            if (!on) continue;
            px.push_back(keel::as<keel::f64>(c));
            py.push_back(keel::as<keel::f64>(r));
        }
    }
    const keel::usize nP = px.size();

    constexpr keel::f64 DEG2RAD = M_PI / 180.0;
    const keel::f64 rho_first = rho.elemAsDouble(0);
    const keel::f64 rho_last  = rho.elemAsDouble(nrho - 1);
    const keel::f64 slope     = keel::as<keel::f64>(nrho - 1)
                           / (rho_last - rho_first);
    const keel::f64 fillgap_sq   = fillgap   * fillgap;
    const keel::f64 minlength_sq = minlength * minlength;

    // Collect line segments.
    struct Seg {
        keel::f64 p1x, p1y, p2x, p2y;
        keel::f64 theta, rho;
    };
    keel::Array<Seg> segs(mr);

    keel::Array<keel::usize> idx(mr);
    keel::Array<keel::usize> rseg(mr), cseg(mr);
    idx.reserve(nP / 4);
    rseg.reserve(nP / 4);
    cseg.reserve(nP / 4);

    for (keel::usize k = 0; k < nPeaks; ++k) {
        const long rbin = keel::as<long>(peaks.elemAsDouble(k));
        const long cbin = keel::as<long>(peaks.elemAsDouble(nPeaks + k));
        if (cbin < 1 || cbin > keel::as<long>(ntheta)) continue;
        const keel::f64 theta_c = theta_deg.elemAsDouble(cbin - 1) * DEG2RAD;
        const keel::f64 cosT = std::cos(theta_c);
        const keel::f64 sinT = std::sin(theta_c);

        // Pick pixels whose rho-bin matches.
        idx.clear();
        for (keel::usize i = 0; i < nP; ++i) {
            const keel::f64 rho_xy = px[i] * cosT + py[i] * sinT;
            const long bin = keel::as<long>(std::lround(
                slope * (rho_xy - rho_first) + 1.0));
            if (bin == rbin) idx.push_back(i);
        }
        if (idx.empty()) continue;

        // Convert back to 1-based (r, c).
        rseg.clear(); cseg.clear();
        for (keel::usize i : idx) {
            rseg.push_back(keel::as<keel::usize>(py[i]) + 1);
            cseg.push_back(keel::as<keel::usize>(px[i]) + 1);
        }

        // Determine sort key: r_range vs c_range.
        keel::usize rmin = rseg[0], rmax = rseg[0];
        keel::usize cmin = cseg[0], cmax = cseg[0];
        for (keel::usize i = 1; i < rseg.size(); ++i) {
            if (rseg[i] < rmin) rmin = rseg[i];
            if (rseg[i] > rmax) rmax = rseg[i];
            if (cseg[i] < cmin) cmin = cseg[i];
            if (cseg[i] > cmax) cmax = cseg[i];
        }
        const keel::usize r_range = rmax - rmin;
        const keel::usize c_range = cmax - cmin;
        // Sort rows by (key1, key2): if r_range > c_range, key1=r, key2=c;
        // else key1=c, key2=r.
        keel::Array<keel::usize> order(mr);
        order.reserve(rseg.size());
        for (keel::usize i = 0; i < rseg.size(); ++i) order.push_back(i);
        if (r_range > c_range) {
            std::sort(order.begin(), order.end(),
                      [&](keel::usize a, keel::usize b) {
                          if (rseg[a] != rseg[b]) return rseg[a] < rseg[b];
                          return cseg[a] < cseg[b];
                      });
        } else {
            std::sort(order.begin(), order.end(),
                      [&](keel::usize a, keel::usize b) {
                          if (cseg[a] != cseg[b]) return cseg[a] < cseg[b];
                          return rseg[a] < rseg[b];
                      });
        }

        // Build sorted (xy = [c r]).
        keel::Array<long> xs(mr), ys(mr);
        xs.reserve(rseg.size());
        ys.reserve(rseg.size());
        for (keel::usize i : order) {
            xs.push_back(keel::as<long>(cseg[i]));
            ys.push_back(keel::as<long>(rseg[i]));
        }

        // diff_sq = (xs[i+1]-xs[i])^2 + (ys[i+1]-ys[i])^2.
        // Find gaps where diff_sq > fillgap^2.
        keel::Array<keel::usize> gap_at(mr);
        for (keel::usize i = 0; i + 1 < xs.size(); ++i) {
            const long dx = xs[i + 1] - xs[i];
            const long dy = ys[i + 1] - ys[i];
            const keel::f64 d2 = keel::as<keel::f64>(dx) * dx
                            + keel::as<keel::f64>(dy) * dy;
            if (d2 > fillgap_sq) gap_at.push_back(i + 1);  // 1-based start
        }
        // Segments split by gap_at indices: idx = [0; gap_at; size].
        // MATLAB: for p=1:length(idx)-1: p1=xy(idx(p)+1,:); p2=xy(idx(p+1),:);
        keel::Array<keel::usize> splits(mr);
        splits.push_back(0);
        for (auto g : gap_at) splits.push_back(g);
        splits.push_back(xs.size());
        for (keel::usize p = 0; p + 1 < splits.size(); ++p) {
            const keel::usize lo = splits[p];      // exclusive lower (gap start)
            const keel::usize hi = splits[p + 1];  // inclusive upper
            if (hi <= lo) continue;
            // p1 = xy(lo+1) in MATLAB 1-idx; here 0-indexed → xs[lo].
            // p2 = xy(hi) in MATLAB → xs[hi-1].
            const long p1x = xs[lo];
            const long p1y = ys[lo];
            const long p2x = xs[hi - 1];
            const long p2y = ys[hi - 1];
            const keel::f64 linelen_sq
                = keel::as<keel::f64>(p2x - p1x) * (p2x - p1x)
                + keel::as<keel::f64>(p2y - p1y) * (p2y - p1y);
            if (linelen_sq >= minlength_sq) {
                Seg s;
                s.p1x = keel::as<keel::f64>(p1x);
                s.p1y = keel::as<keel::f64>(p1y);
                s.p2x = keel::as<keel::f64>(p2x);
                s.p2y = keel::as<keel::f64>(p2y);
                s.theta = theta_deg.elemAsDouble(cbin - 1);
                s.rho   = rho.elemAsDouble(rbin - 1);
                segs.push_back(s);
            }
        }
    }

    // Build 1 × N struct array.
    const keel::usize nL = segs.size();
    if (nL == 0) {
        // Empty struct with the four fields (MATLAB: 1 × 0 struct).
        Value out = Value::structArray(1, 0, mr);
        return out;
    }
    Value out = Value::structArray(1, nL, mr);
    for (keel::usize i = 0; i < nL; ++i) {
        Value p1 = Value::matrix(1, 2, ValueType::DOUBLE, mr);
        p1.doubleDataMut()[0] = segs[i].p1x;
        p1.doubleDataMut()[1] = segs[i].p1y;
        Value p2 = Value::matrix(1, 2, ValueType::DOUBLE, mr);
        p2.doubleDataMut()[0] = segs[i].p2x;
        p2.doubleDataMut()[1] = segs[i].p2y;
        out.setField(i, "point1", p1);
        out.setField(i, "point2", p2);
        out.setField(i, "theta", Value::scalar(segs[i].theta, mr));
        out.setField(i, "rho",   Value::scalar(segs[i].rho,   mr));
    }
    return out;
}

namespace {

keel::String toLowerAscii(keel::StringRef s)
{
    keel::String out = keel::String::from(s);
    for (char &ch : out) {
        ch = keel::to_ascii_lower(ch);
    }
    return out;
}

void imfindcircles_argcheck(const Value &A, keel::f64 rmin, keel::f64 rmax,
                            keel::StringRef object_polarity,
                            keel::StringRef method,
                            keel::f64 sensitivity,
                            keel::f64 edge_threshold)
{
    internal::requireImageClass(A, "imfindcircles", "A", 1,
                                internal::ImgClasses::FloatIntLog);
    internal::requireNonemptyImage(A, "imfindcircles", "A", 1);
    const Dims &d = A.dims();
    if (d.ndim() > 3 || (d.is3D() && d.pages() != 3)
        || d.rows() <= 1 || d.cols() <= 1) {
        throw Error("imfindcircles: input 1 (A) must be a 2-D matrix "
                    "(both dimensions > 1) or an M-by-N-by-3 RGB image.",
                    0, 0, "imfindcircles", "",
                    "numkit:imfindcircles:invalidInputImage");
    }
    if (!std::isfinite(rmin) || !std::isfinite(rmax)
        || !(rmin > 0.0) || !(rmax >= rmin)) {
        throw Error("imfindcircles: input 2 (radiusRange) must have "
                    "positive finite bounds with rmin <= rmax.",
                    0, 0, "imfindcircles", "",
                    "numkit:imfindcircles:invalidRadiusRange");
    }
    if (rmin != rmax && (rmin != std::floor(rmin) || rmax != std::floor(rmax))) {
        throw Error("imfindcircles: 2-element radiusRange must contain "
                    "positive integers.",
                    0, 0, "imfindcircles", "",
                    "numkit:imfindcircles:expectedInteger");
    }
    const keel::String pol = toLowerAscii(object_polarity);
    if (pol != "bright" && pol != "dark") {
        throw Error(keel::format("imfindcircles: ObjectPolarity must be 'bright' or 'dark' (got '{}').",
                                 object_polarity),
                    0, 0, "imfindcircles", "",
                    "numkit:imfindcircles:unrecognizedObjectPolarity");
    }
    const keel::String mth = toLowerAscii(method);
    if (mth != "phasecode" && mth != "twostage") {
        throw Error(keel::format("imfindcircles: Method must be 'PhaseCode' or 'TwoStage' (got '{}').",
                                 method),
                    0, 0, "imfindcircles", "",
                    "numkit:imfindcircles:unrecognizedMethod");
    }
    if (!std::isfinite(sensitivity) || sensitivity < 0.0 || sensitivity > 1.0) {
        throw Error("imfindcircles: Sensitivity must be a finite scalar in [0, 1].",
                    0, 0, "imfindcircles", "",
                    "numkit:imfindcircles:outOfRangeSensitivity");
    }
    if (edge_threshold >= 0.0 || std::isnan(edge_threshold) || std::isinf(edge_threshold)) {
        if (!std::isfinite(edge_threshold) || edge_threshold < 0.0 || edge_threshold > 1.0) {
            throw Error("imfindcircles: EdgeThreshold must be a finite scalar in [0, 1].",
                        0, 0, "imfindcircles", "",
                        "numkit:imfindcircles:outOfRangeEdgeThreshold");
        }
    }
}

} // namespace

ImfindcirclesResult
imfindcircles(const Value &A, keel::f64 rmin, keel::f64 rmax,
              keel::StringRef object_polarity,
              keel::StringRef method,
              keel::f64 sensitivity,
              keel::f64 edge_threshold,
              keel::Allocator *mr)
{
    imfindcircles_argcheck(A, rmin, rmax, object_polarity, method,
                           sensitivity, edge_threshold);

    auto emptyResult = [&]() -> ImfindcirclesResult {
        return {
            Value::matrix(0, 0, ValueType::DOUBLE, mr),
            Value::matrix(0, 0, ValueType::DOUBLE, mr),
            Value::matrix(0, 0, ValueType::DOUBLE, mr)
        };
    };

    const keel::usize totalElem = A.numel();
    if (totalElem == 0) return emptyResult();

    // Flat image check.
    const keel::f64 firstVal = A.elemAsDouble(0);
    bool isFlat = true;
    for (keel::usize i = 1; i < totalElem; ++i) {
        if (A.elemAsDouble(i) != firstVal) {
            isFlat = false;
            break;
        }
    }
    if (isFlat) return emptyResult();

    // Convert to 2-D grayscale floating-point image matching chaccum's getGrayImage.
    Value G;
    if (A.dims().is3D()) {
        G = rgb2gray(A, mr);
        if (isIntegerType(G.type())) {
            G = im2single(G, mr);
        }
    } else {
        if (A.type() == ValueType::LOGICAL) {
            Value As = im2single(A, mr);
            Value gaussFilt = fspecial("gaussian", {5.0, 5.0, 1.5}, mr);
            G = imfilter(As, gaussFilt, PadMode::Replicate, 0.0,
                         /*full=*/false, /*flip_kernel=*/false, mr);
        } else if (isIntegerType(A.type())) {
            G = im2single(A, mr);
        } else {
            G = A;
        }
    }

    const keel::usize M = G.dims().rows();
    const keel::usize N = G.dims().cols();
    const keel::usize MN = M * N;
    const bool useSingle = (G.type() == ValueType::SINGLE);

    // Sobel gradient: hy = -fspecial('sobel'), hx = hy', convolved with replicate padding.
    Value hx = make_kernel({-1, 0, 1, -2, 0, 2, -1, 0, 1}, 3, 3, mr);
    Value hy = make_kernel({-1, -2, -1, 0, 0, 0, 1, 2, 1}, 3, 3, mr);
    Value GxVal = imfilter(G, hx, PadMode::Replicate, 0.0,
                           /*full=*/false, /*flip_kernel=*/true, mr);
    Value GyVal = imfilter(G, hy, PadMode::Replicate, 0.0,
                           /*full=*/false, /*flip_kernel=*/true, mr);

    keel::Array<keel::f64> Gx(MN), Gy(MN), gradMag(MN);
    keel::f64 Gmax = 0.0;
    for (keel::usize i = 0; i < MN; ++i) {
        const keel::f64 gx = GxVal.elemAsDouble(i);
        const keel::f64 gy = GyVal.elemAsDouble(i);
        Gx[i] = gx;
        Gy[i] = gy;
        const keel::f64 gm = useSingle
            ? keel::as<keel::f64>(std::hypot(keel::as<keel::f32>(gx), keel::as<keel::f32>(gy)))
            : std::hypot(gx, gy);
        gradMag[i] = gm;
        if (gm > Gmax) Gmax = gm;
    }
    if (!(Gmax > 0.0)) return emptyResult();

    keel::f64 edgeThresh = edge_threshold;
    if (edgeThresh < 0.0) {
        Value Gnorm = Value::matrix(M, N, useSingle ? ValueType::SINGLE : ValueType::DOUBLE, mr);
        if (useSingle) {
            keel::f32 *p = Gnorm.singleDataMut();
            const keel::f32 gmaxF = keel::as<keel::f32>(Gmax);
            for (keel::usize i = 0; i < MN; ++i) {
                p[i] = keel::as<keel::f32>(gradMag[i]) / gmaxF;
            }
        } else {
            keel::f64 *p = Gnorm.doubleDataMut();
            for (keel::usize i = 0; i < MN; ++i) {
                p[i] = gradMag[i] / Gmax;
            }
        }
        auto [lvl, _] = graythresh(Gnorm, mr);
        edgeThresh = lvl.toScalar();
    }

    const keel::f64 t = useSingle
        ? keel::as<keel::f64>(keel::as<keel::f32>(Gmax) * keel::as<keel::f32>(edgeThresh))
        : (Gmax * edgeThresh);

    struct EdgePix {
        keel::f64 ex;
        keel::f64 ey;
        keel::f64 ux;
        keel::f64 uy;
    };
    keel::Array<EdgePix> edges;
    edges.reserve(MN / 8);
    for (keel::usize c = 0; c < N; ++c) {
        for (keel::usize r = 0; r < M; ++r) {
            const keel::usize idx = c * M + r;
            if (gradMag[idx] > t) {
                const keel::f64 ux = useSingle
                    ? keel::as<keel::f64>(keel::as<keel::f32>(Gx[idx]) / keel::as<keel::f32>(gradMag[idx]))
                    : (Gx[idx] / gradMag[idx]);
                const keel::f64 uy = useSingle
                    ? keel::as<keel::f64>(keel::as<keel::f32>(Gy[idx]) / keel::as<keel::f32>(gradMag[idx]))
                    : (Gy[idx] / gradMag[idx]);
                edges.push_back({
                    keel::as<keel::f64>(c + 1),
                    keel::as<keel::f64>(r + 1),
                    ux,
                    uy
                });
            }
        }
    }
    if (edges.empty()) return emptyResult();

    const keel::String polLower = toLowerAscii(object_polarity);
    const keel::String methodLower = toLowerAscii(method);

    keel::Array<keel::f64> radVec;
    if (rmin < rmax) {
        for (keel::f64 r = rmin; r <= rmax + 1e-9; r += 0.5) {
            radVec.push_back(r);
        }
    } else {
        radVec.push_back(rmin);
    }
    const keel::usize nR = radVec.size();
    const keel::f64 polSign = (polLower == "bright") ? 1.0 : -1.0;

    keel::Array<keel::f64> RR(nR), w0_re(nR), w0_im(nR, 0.0);
    const keel::f64 lnR1 = std::log(radVec.front());
    const keel::f64 lnRend = std::log(radVec.back());
    for (keel::usize k = 0; k < nR; ++k) {
        RR[k] = polSign * radVec[k];
        const keel::f64 invCirc = 1.0 / (2.0 * M_PI * radVec[k]);
        if (methodLower == "phasecode" && nR > 1) {
            const keel::f64 phi = ((std::log(radVec[k]) - lnR1) / (lnRend - lnR1) * 2.0 * M_PI) - M_PI;
            w0_re[k] = std::cos(phi) * invCirc;
            w0_im[k] = std::sin(phi) * invCirc;
        } else {
            w0_re[k] = invCirc;
            w0_im[k] = 0.0;
        }
    }

    keel::Array<keel::f64> accumRe(MN, 0.0), accumIm(MN, 0.0);
    const long Ml = keel::as<long>(M);
    const long Nl = keel::as<long>(N);
    for (const auto &ep : edges) {
        for (keel::usize k = 0; k < nR; ++k) {
            const long xc = std::lround(ep.ex + (-RR[k]) * ep.ux);
            const long yc = std::lround(ep.ey + (-RR[k]) * ep.uy);
            if (xc >= 1 && xc <= Nl && yc >= 1 && yc <= Ml) {
                const keel::usize aidx = keel::as<keel::usize>(xc - 1) * M
                                       + keel::as<keel::usize>(yc - 1);
                accumRe[aidx] += w0_re[k];
                accumIm[aidx] += w0_im[k];
            }
        }
    }

    keel::Array<keel::f64> accumAbs(MN, 0.0);
    bool anyNonZero = false;
    for (keel::usize i = 0; i < MN; ++i) {
        accumAbs[i] = std::hypot(accumRe[i], accumIm[i]);
        if (accumAbs[i] != 0.0) anyNonZero = true;
    }
    if (!anyNonZero) return emptyResult();

    bool accumFlat = true;
    for (keel::usize i = 1; i < MN; ++i) {
        if (accumAbs[i] != accumAbs[0]) {
            accumFlat = false;
            break;
        }
    }
    if (accumFlat) return emptyResult();

    // 5x5 zero-padded median filter on accumAbs if min(M, N) > 5.
    Value HdVal = Value::matrix(M, N, ValueType::DOUBLE, mr);
    keel::f64 *hdPtr = HdVal.doubleDataMut();
    if (std::min(M, N) > 5) {
        keel::StaticArray<keel::f64, 25> win{};
        for (keel::usize c = 0; c < N; ++c) {
            for (keel::usize r = 0; r < M; ++r) {
                keel::usize k = 0;
                for (int dc = -2; dc <= 2; ++dc) {
                    const long nc = keel::as<long>(c) + dc;
                    for (int dr = -2; dr <= 2; ++dr) {
                        const long nr = keel::as<long>(r) + dr;
                        if (nr < 0 || nr >= Ml || nc < 0 || nc >= Nl) {
                            win[k++] = 0.0;
                        } else {
                            win[k++] = accumAbs[keel::as<keel::usize>(nc) * M
                                              + keel::as<keel::usize>(nr)];
                        }
                    }
                }
                std::nth_element(win.begin(), win.begin() + 12, win.end());
                hdPtr[c * M + r] = win[12];
            }
        }
    } else {
        for (keel::usize i = 0; i < MN; ++i) {
            hdPtr[i] = accumAbs[i];
        }
    }

    const keel::f64 accumThresh = 1.0 - sensitivity;
    const keel::f64 epsT = (accumThresh > 0.0)
        ? (std::nextafter(accumThresh, std::numeric_limits<keel::f64>::infinity()) - accumThresh)
        : 0.0;
    const keel::f64 suppThreshold = std::max(accumThresh - epsT, 0.0);
    HdVal = imhmax(HdVal, suppThreshold, 8, mr);
    const keel::f64 *hd = HdVal.doubleData();

    // Find 8-connected regional maxima plateaus of Hd in column-major order.
    static constexpr int dr8[8] = {-1, 0, 1, -1, 1, -1, 0, 1};
    static constexpr int dc8[8] = {-1, -1, -1, 0, 0, 1, 1, 1};
    keel::Array<keel::u8> visited(MN, 0);
    keel::Array<keel::usize> comp, q;
    comp.reserve(32);
    q.reserve(32);

    struct Candidate {
        keel::f64 xc;
        keel::f64 yc;
        keel::f64 metric;
    };
    keel::Array<Candidate> cands;

    for (keel::usize i = 0; i < MN; ++i) {
        if (visited[i]) continue;
        visited[i] = 1;
        comp.clear();
        q.clear();
        comp.push_back(i);
        q.push_back(i);
        const keel::f64 v0 = hd[i];
        bool hasHigher = false;
        keel::usize head = 0;
        while (head < q.size()) {
            const keel::usize curr = q[head++];
            const long r = keel::as<long>(curr % M);
            const long c = keel::as<long>(curr / M);
            for (int k = 0; k < 8; ++k) {
                const long nr = r + dr8[k];
                const long nc = c + dc8[k];
                if (nr < 0 || nr >= Ml || nc < 0 || nc >= Nl) continue;
                const keel::usize nb = keel::as<keel::usize>(nc) * M
                                     + keel::as<keel::usize>(nr);
                if (hd[nb] > v0) {
                    hasHigher = true;
                } else if (hd[nb] == v0 && !visited[nb]) {
                    visited[nb] = 1;
                    comp.push_back(nb);
                    q.push_back(nb);
                }
            }
        }
        if (!hasHigher) {
            keel::f64 wsum = 0.0, xsum = 0.0, ysum = 0.0;
            for (keel::usize idx : comp) {
                const keel::f64 w = accumAbs[idx];
                const keel::f64 r1 = keel::as<keel::f64>(idx % M) + 1.0;
                const keel::f64 c1 = keel::as<keel::f64>(idx / M) + 1.0;
                wsum += w;
                xsum += c1 * w;
                ysum += r1 * w;
            }
            if (wsum > 0.0) {
                const keel::f64 xc = xsum / wsum;
                const keel::f64 yc = ysum / wsum;
                const long r_idx = std::clamp<long>(std::lround(yc) - 1, 0, Ml - 1);
                const long c_idx = std::clamp<long>(std::lround(xc) - 1, 0, Nl - 1);
                const keel::f64 met = hd[keel::as<keel::usize>(c_idx) * M
                                    + keel::as<keel::usize>(r_idx)];
                if (met >= accumThresh) {
                    cands.push_back({xc, yc, met});
                }
            }
        }
    }

    if (cands.empty()) return emptyResult();

    std::stable_sort(cands.begin(), cands.end(),
                     [](const Candidate &a, const Candidate &b) {
                         return a.metric > b.metric;
                     });

    const keel::usize K = cands.size();
    Value centersOut = Value::matrix(K, 2, ValueType::DOUBLE, mr);
    Value radiiOut   = Value::matrix(K, 1, ValueType::DOUBLE, mr);
    Value metricOut  = Value::matrix(K, 1, ValueType::DOUBLE, mr);
    keel::f64 *cp = centersOut.doubleDataMut();
    keel::f64 *rp = radiiOut.doubleDataMut();
    keel::f64 *mp = metricOut.doubleDataMut();

    for (keel::usize k = 0; k < K; ++k) {
        cp[k]     = cands[k].xc;
        cp[K + k] = cands[k].yc;
        mp[k]     = cands[k].metric;

        if (rmin == rmax) {
            rp[k] = rmin;
        } else if (methodLower == "phasecode") {
            const long r_idx = std::clamp<long>(std::lround(cands[k].yc) - 1, 0, Ml - 1);
            const long c_idx = std::clamp<long>(std::lround(cands[k].xc) - 1, 0, Nl - 1);
            const keel::usize aidx = keel::as<keel::usize>(c_idx) * M
                                   + keel::as<keel::usize>(r_idx);
            const keel::f64 cenPhase = std::atan2(accumIm[aidx], accumRe[aidx]);
            rp[k] = std::exp(((cenPhase + M_PI) / (2.0 * M_PI) * (lnRend - lnR1)) + lnR1);
        } else {
            // Two-stage radial histogram around (xc, yc).
            const long left   = std::max<long>(keel::as<long>(std::floor(cands[k].xc - rmax)), 1);
            const long right  = std::min<long>(keel::as<long>(std::ceil(cands[k].xc + rmax)), Nl);
            const long top    = std::max<long>(keel::as<long>(std::floor(cands[k].yc - rmax)), 1);
            const long bottom = std::min<long>(keel::as<long>(std::ceil(cands[k].yc + rmax)), Ml);
            const long r1_int = keel::as<long>(std::lround(rmin));
            const long r2_int = keel::as<long>(std::lround(rmax));
            const keel::usize nBins = keel::as<keel::usize>(r2_int - r1_int + 1);
            keel::Array<keel::f64> hbin(nBins, 0.0);
            for (long cc = left; cc <= right; ++cc) {
                for (long rr = top; rr <= bottom; ++rr) {
                    const long rbin = std::lround(std::hypot(
                        keel::as<keel::f64>(cc) - cands[k].xc,
                        keel::as<keel::f64>(rr) - cands[k].yc));
                    if (rbin >= r1_int && rbin <= r2_int) {
                        hbin[keel::as<keel::usize>(rbin - r1_int)] +=
                            gradMag[keel::as<keel::usize>(cc - 1) * M
                                  + keel::as<keel::usize>(rr - 1)];
                    }
                }
            }
            keel::f64 bestH = -1.0;
            long bestR = r1_int;
            for (keel::usize b = 0; b < nBins; ++b) {
                const keel::f64 rcur = keel::as<keel::f64>(r1_int + keel::as<long>(b));
                const keel::f64 hnorm = hbin[b] / (2.0 * M_PI * rcur);
                if (hnorm > bestH) {
                    bestH = hnorm;
                    bestR = r1_int + keel::as<long>(b);
                }
            }
            rp[k] = keel::as<keel::f64>(bestR);
        }
    }

    return {std::move(centersOut), std::move(radiiOut), std::move(metricOut)};
}

} // namespace numkit::image
