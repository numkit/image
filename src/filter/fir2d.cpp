// toolboxes/image/src/filter/fir2d.cpp
//
// 2-D FIR filter design family — fsamp2 / ftrans2 / fwind1 / fwind2.
//
// Algorithms transliterated verbatim from MATLAB R2025b
//   fsamp2.m, ftrans2.m, fwind1.m, fwind2.m.
//
// All four functions return DOUBLE 2-D filters intended for use with
// filter2 (equivalent to applying the filter with the rotated-by-180°
// convention).
//
// Reference: Lim, J. S. (1990). Two-Dimensional Signal and Image
// Processing, Prentice-Hall, §3.4-3.5 (window method and
// transformation method), §4.2 (frequency sampling).
//
// PMR HARD RULE: every fn takes std::pmr::memory_resource *mr.

#include <numkit/image/filter/filter.hpp>
#include <numkit/image/internal/type_gates.hpp>

#include <numkit/signal/transforms/fft.hpp>
#include <numkit/signal/transforms/transform_helpers.hpp>
#include <numkit/signal/convolution/convolution.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <vector>

namespace numkit::image {
namespace {

// rot90(A, 2) — 180° rotation == reverse all elements.
Value rot90_2(const Value &A, std::pmr::memory_resource *mr)
{
    const std::size_t H = A.dims().rows();
    const std::size_t W = A.dims().cols();
    if (A.isComplex()) {
        Value out = Value::matrix(H, W, ValueType::COMPLEX, mr);
        auto *od = out.complexDataMut();
        const auto *ad = A.complexData();
        for (std::size_t c = 0; c < W; ++c)
            for (std::size_t r = 0; r < H; ++r)
                od[c * H + r] = ad[(W - 1 - c) * H + (H - 1 - r)];
        return out;
    }
    Value out = Value::matrix(H, W, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    for (std::size_t c = 0; c < W; ++c)
        for (std::size_t r = 0; r < H; ++r)
            od[c * H + r] = A.elemAsDouble((W - 1 - c) * H + (H - 1 - r));
    return out;
}

// MATLAB-style ifftshift: shift each axis by ceil(N/2) (negative
// direction). ifftshift([1 2 3 4 5]) = [3 4 5 1 2] — the center
// value 3 (at index 2) moves to index 0.
//
// To compute output[c] we read from (c + ceil(N/2)) mod N.
// For odd N this equals (c + floor(N/2)) mod N.
Value ifftshift2d(const Value &A, std::pmr::memory_resource *mr)
{
    const std::size_t H = A.dims().rows();
    const std::size_t W = A.dims().cols();
    const std::size_t hh = (H + 1) / 2;  // ceil
    const std::size_t ww = (W + 1) / 2;
    if (A.isComplex()) {
        Value out = Value::matrix(H, W, ValueType::COMPLEX, mr);
        auto *od = out.complexDataMut();
        const auto *ad = A.complexData();
        for (std::size_t c = 0; c < W; ++c) {
            const std::size_t cs = (c + ww) % W;
            for (std::size_t r = 0; r < H; ++r) {
                const std::size_t rs = (r + hh) % H;
                od[c * H + r] = ad[cs * H + rs];
            }
        }
        return out;
    }
    Value out = Value::matrix(H, W, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    for (std::size_t c = 0; c < W; ++c) {
        const std::size_t cs = (c + ww) % W;
        for (std::size_t r = 0; r < H; ++r) {
            const std::size_t rs = (r + hh) % H;
            od[c * H + r] = A.elemAsDouble(cs * H + rs);
        }
    }
    return out;
}

// MATLAB-style fftshift: shift each axis by floor(N/2). The value
// originally at index 0 moves to index floor(N/2).
// fftshift([1 2 3 4 5]) = [4 5 1 2 3] — value 1 (at index 0) moves
// to index 2 (= floor(5/2)).
//
// Reading the output forward: output[c] = input[(c + ceil(N/2)) mod N]
// is the inverse mapping (where to *read from*). Equivalent.
//
// For ODD dims, fftshift == ifftshift. For EVEN dims they differ in
// shift direction by 1.
Value fftshift2d(const Value &A, std::pmr::memory_resource *mr)
{
    const std::size_t H = A.dims().rows();
    const std::size_t W = A.dims().cols();
    const std::size_t hh = (H + 1) / 2;  // ceil
    const std::size_t ww = (W + 1) / 2;
    if (A.isComplex()) {
        Value out = Value::matrix(H, W, ValueType::COMPLEX, mr);
        auto *od = out.complexDataMut();
        const auto *ad = A.complexData();
        for (std::size_t c = 0; c < W; ++c) {
            const std::size_t cs = (c + W - ww) % W;
            for (std::size_t r = 0; r < H; ++r) {
                const std::size_t rs = (r + H - hh) % H;
                od[c * H + r] = ad[cs * H + rs];
            }
        }
        return out;
    }
    Value out = Value::matrix(H, W, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    for (std::size_t c = 0; c < W; ++c) {
        const std::size_t cs = (c + W - ww) % W;
        for (std::size_t r = 0; r < H; ++r) {
            const std::size_t rs = (r + H - hh) % H;
            od[c * H + r] = A.elemAsDouble(cs * H + rs);
        }
    }
    return out;
}

// Helper to coerce a Value to DOUBLE matrix (no class-rescaling; just
// type-cast).
Value as_double(const Value &V, std::pmr::memory_resource *mr)
{
    if (V.type() == ValueType::DOUBLE) return V;
    const auto &d = V.dims();
    Value out = Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    const std::size_t N = V.numel();
    for (std::size_t i = 0; i < N; ++i) od[i] = V.elemAsDouble(i);
    return out;
}

// Bilinear interpolation of a 2-D real or complex matrix onto m_out x n_out.
Value bilinear_interp_hd(const Value &Hd, std::size_t m_out, std::size_t n_out,
                         std::pmr::memory_resource *mr)
{
    const std::size_t Hh = Hd.dims().rows();
    const std::size_t Wh = Hd.dims().cols();
    const double dr_src = 2.0 / static_cast<double>(Hh);
    const double dc_src = 2.0 / static_cast<double>(Wh);
    const double dr_dst = 2.0 / static_cast<double>(m_out);
    const double dc_dst = 2.0 / static_cast<double>(n_out);
    if (Hd.isComplex()) {
        Value out = Value::matrix(m_out, n_out, ValueType::COMPLEX, mr);
        auto *od = out.complexDataMut();
        const auto *cd = Hd.complexData();
        for (std::size_t c = 0; c < n_out; ++c) {
            const double cx = (static_cast<double>(c) - (n_out - 1) / 2.0) * dc_dst;
            const double sf_c = (cx + 1.0) / dc_src - 0.5;
            const long c0 = std::max<long>(0, static_cast<long>(std::floor(sf_c)));
            const long c1 = std::min<long>(static_cast<long>(Wh) - 1, c0 + 1);
            const double ac = sf_c - c0;
            for (std::size_t r = 0; r < m_out; ++r) {
                const double cy = (static_cast<double>(r) - (m_out - 1) / 2.0) * dr_dst;
                const double sf_r = (cy + 1.0) / dr_src - 0.5;
                const long r0 = std::max<long>(0, static_cast<long>(std::floor(sf_r)));
                const long r1 = std::min<long>(static_cast<long>(Hh) - 1, r0 + 1);
                const double ar = sf_r - r0;
                const auto v00 = cd[static_cast<std::size_t>(c0) * Hh + r0];
                const auto v01 = cd[static_cast<std::size_t>(c0) * Hh + r1];
                const auto v10 = cd[static_cast<std::size_t>(c1) * Hh + r0];
                const auto v11 = cd[static_cast<std::size_t>(c1) * Hh + r1];
                const auto v0 = v00 + ar * (v01 - v00);
                const auto v1 = v10 + ar * (v11 - v10);
                od[c * m_out + r] = v0 + ac * (v1 - v0);
            }
        }
        return out;
    }
    Value Hd_d = as_double(Hd, mr);
    Value out = Value::matrix(m_out, n_out, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    for (std::size_t c = 0; c < n_out; ++c) {
        const double cx = (static_cast<double>(c) - (n_out - 1) / 2.0) * dc_dst;
        const double sf_c = (cx + 1.0) / dc_src - 0.5;
        const long c0 = std::max<long>(0, static_cast<long>(std::floor(sf_c)));
        const long c1 = std::min<long>(static_cast<long>(Wh) - 1, c0 + 1);
        const double ac = sf_c - c0;
        for (std::size_t r = 0; r < m_out; ++r) {
            const double cy = (static_cast<double>(r) - (m_out - 1) / 2.0) * dr_dst;
            const double sf_r = (cy + 1.0) / dr_src - 0.5;
            const long r0 = std::max<long>(0, static_cast<long>(std::floor(sf_r)));
            const long r1 = std::min<long>(static_cast<long>(Hh) - 1, r0 + 1);
            const double ar = sf_r - r0;
            const double v00 = Hd_d.elemAsDouble(static_cast<std::size_t>(c0) * Hh + r0);
            const double v01 = Hd_d.elemAsDouble(static_cast<std::size_t>(c0) * Hh + r1);
            const double v10 = Hd_d.elemAsDouble(static_cast<std::size_t>(c1) * Hh + r0);
            const double v11 = Hd_d.elemAsDouble(static_cast<std::size_t>(c1) * Hh + r1);
            const double v0 = v00 + ar * (v01 - v00);
            const double v1 = v10 + ar * (v11 - v10);
            od[c * m_out + r] = v0 + ac * (v1 - v0);
        }
    }
    return out;
}

}  // namespace

// ── fsamp2 ────────────────────────────────────────────────────────
Value fsamp2(const Value &HdIn, const Value &f1, const Value &f2,
             const std::vector<std::size_t> & /*siz*/,
             std::pmr::memory_resource *mr)
{
    Value Hd0 = (HdIn.type() == ValueType::CHAR)
        ? internal::coerceToDoubleImage(HdIn, mr) : Value(HdIn);
    internal::requireImageClassAllowComplex(Hd0, "fsamp2", "Hd", 1,
                                internal::ImgClasses::FloatIntLog);
    const Value &Hd = Hd0;
    if (!f1.isEmpty() || !f2.isEmpty())
        throw Error("fsamp2: non-uniform-spacing form "
                    "fsamp2(f1, f2, Hd, [m n]) is not yet supported — "
                    "use the uniform fsamp2(Hd) form.",
                    0, 0, "fsamp2", "", "numkit:fsamp2:nonuniform");
    if (Hd.numel() == 0)
        throw Error("fsamp2: Hd must be nonempty",
                    0, 0, "fsamp2", "", "numkit:fsamp2:empty");
    if (Hd.dims().is3D() || Hd.dims().rows() < 1 || Hd.dims().cols() < 1)
        throw Error("fsamp2: Hd must be a nonempty 2-D matrix",
                    0, 0, "fsamp2", "", "numkit:fsamp2:shape");

    if (Hd.isComplex()) {
        const std::size_t H = Hd.dims().rows();
        const std::size_t W = Hd.dims().cols();
        Value Hdr = Value::matrix(H, W, ValueType::DOUBLE, mr);
        Value Hdi = Value::matrix(H, W, ValueType::DOUBLE, mr);
        double *rd = Hdr.doubleDataMut();
        double *id = Hdi.doubleDataMut();
        const auto *cd = Hd.complexData();
        for (std::size_t k = 0; k < H * W; ++k) {
            rd[k] = cd[k].real();
            id[k] = cd[k].imag();
        }
        Value hr = fsamp2(Hdr, Value::Empty, Value::Empty, {}, mr);
        Value hi = fsamp2(Hdi, Value::Empty, Value::Empty, {}, mr);
        Value out = Value::matrix(H, W, ValueType::COMPLEX, mr);
        auto *od = out.complexDataMut();
        const double *hrd = hr.doubleData();
        const double *hid = hi.doubleData();
        for (std::size_t k = 0; k < H * W; ++k)
            od[k] = std::complex<double>(hrd[k], hid[k]);
        return out;
    }

    Value Hd_d = as_double(Hd, mr);
    // 1. Inverse fftshift — use numkit::signal's tested implementation
    //    via `rot90(fftshift(rot90(A, 2)), 2)`, MATLAB's idiom for
    //    ifftshift.
    Value shifted = rot90_2(signal::fftshift(rot90_2(Hd_d, mr), mr), mr);
    // 2. ifft2
    Value h_complex = signal::ifft2(shifted, -1, -1, mr);
    // 3. fftshift on the real part of the output.
    Value h_shifted;
    if (h_complex.type() == ValueType::DOUBLE) {
        h_shifted = signal::fftshift(h_complex, mr);
    } else if (h_complex.type() == ValueType::COMPLEX) {
        const std::size_t H = h_complex.dims().rows();
        const std::size_t W = h_complex.dims().cols();
        Value real_part = Value::matrix(H, W, ValueType::DOUBLE, mr);
        double *rd = real_part.doubleDataMut();
        const std::complex<double> *cd = h_complex.complexData();
        for (std::size_t i = 0; i < H * W; ++i) rd[i] = cd[i].real();
        h_shifted = signal::fftshift(real_part, mr);
    } else {
        h_shifted = signal::fftshift(as_double(h_complex, mr), mr);
    }
    return internal::castResultToClass(
        rot90_2(h_shifted, mr),
        Hd0.type() == ValueType::SINGLE ? ValueType::SINGLE
                                        : ValueType::DOUBLE,
        mr);
}

// ── ftrans2 ──────────────────────────────────────────────────────
Value ftrans2(const Value &b_in, const Value &t_in,
              std::pmr::memory_resource *mr)
{
    Value b = as_double(b_in, mr);
    const std::size_t L = b.numel();
    if (L == 0)
        throw Error("ftrans2: b must be nonempty",
                    0, 0, "ftrans2", "", "numkit:ftrans2:bEmpty");
    if (L % 2 == 0)
        throw Error("ftrans2: b must be odd-length",
                    0, 0, "ftrans2", "", "numkit:ftrans2:bLen");
    // Check zero.
    bool all_zero = true;
    for (std::size_t i = 0; i < L; ++i)
        if (b.elemAsDouble(i) != 0.0) { all_zero = false; break; }
    if (all_zero)
        throw Error("ftrans2: b must have at least one nonzero element",
                    0, 0, "ftrans2", "", "numkit:ftrans2:bZero");
    // Check symmetry: b == rot90(b, 2) within sqrt(eps).
    const double sqrt_eps = std::sqrt(std::numeric_limits<double>::epsilon());
    for (std::size_t i = 0; i < L; ++i) {
        const double a = b.elemAsDouble(i);
        const double c = b.elemAsDouble(L - 1 - i);
        if (std::fabs(a - c) > sqrt_eps)
            throw Error("ftrans2: b must be symmetric",
                        0, 0, "ftrans2", "", "numkit:ftrans2:bSym");
    }
    // Resolve transform matrix t.
    Value t;
    if (t_in.isEmpty()) {
        // Default McClellan.
        t = Value::matrix(3, 3, ValueType::DOUBLE, mr);
        double *td = t.doubleDataMut();
        // Column-major fill of [1 2 1; 2 -4 2; 1 2 1]/8.
        // MATLAB layout: row-major rows above translate to col-major:
        // (r=0,c=0)=1, (1,0)=2, (2,0)=1, (0,1)=2, (1,1)=-4, (2,1)=2, etc.
        td[0] = 1.0/8.0; td[1] = 2.0/8.0; td[2] = 1.0/8.0;
        td[3] = 2.0/8.0; td[4] = -4.0/8.0; td[5] = 2.0/8.0;
        td[6] = 1.0/8.0; td[7] = 2.0/8.0; td[8] = 1.0/8.0;
    } else {
        t = as_double(t_in, mr);
        bool tzero = true;
        for (std::size_t i = 0; i < t.numel(); ++i)
            if (t.elemAsDouble(i) != 0.0) { tzero = false; break; }
        if (tzero)
            throw Error("ftrans2: t must have at least one nonzero element",
                        0, 0, "ftrans2", "", "numkit:ftrans2:tZero");
    }

    // Convert b to Chebyshev coefficients a:
    //   b is length L = 2n+1, symmetric around index n.
    //   shifted = ifftshift(b) so b becomes [b(n), b(n+1), ..., b(2n),
    //                                         b(0), ..., b(n-1)].
    //   a = [shifted(1), 2*shifted(2:n+1)] (1-based).
    const long n = (static_cast<long>(L) - 1) / 2;
    std::vector<double> b_vec(L);
    for (std::size_t i = 0; i < L; ++i) b_vec[i] = b.elemAsDouble(i);
    // ifftshift on 1-D: roll by floor(L/2).
    const long shift = static_cast<long>(L) / 2;
    std::vector<double> bsh(L);
    for (std::size_t i = 0; i < L; ++i)
        bsh[i] = b_vec[(static_cast<long>(i) + shift) % static_cast<long>(L)];
    std::vector<double> a(static_cast<std::size_t>(n + 1));
    a[0] = bsh[0];
    for (long k = 1; k <= n; ++k) a[static_cast<std::size_t>(k)] = 2.0 * bsh[static_cast<std::size_t>(k)];

    // Chebyshev recursion.
    const std::size_t Mt = t.dims().rows();
    const std::size_t Nt = t.dims().cols();
    const std::size_t insetR = (Mt - 1) / 2;
    const std::size_t insetC = (Nt - 1) / 2;
    // P0 = 1 (scalar 1x1).
    Value P0 = Value::matrix(1, 1, ValueType::DOUBLE, mr);
    P0.doubleDataMut()[0] = 1.0;
    Value P1 = t;
    // h = a(2) * P1 (a is 0-indexed in C++; a(2) MATLAB = a[1] C++).
    auto scale_mat = [&](const Value &A, double s) {
        const std::size_t H = A.dims().rows();
        const std::size_t W = A.dims().cols();
        Value out = Value::matrix(H, W, ValueType::DOUBLE, mr);
        double *od = out.doubleDataMut();
        for (std::size_t i = 0; i < H * W; ++i)
            od[i] = A.elemAsDouble(i) * s;
        return out;
    };
    Value h = scale_mat(P1, a[1]);
    // Add a(1) * P0 at position (insetR, insetC) (0-based).
    auto add_at = [&](Value &dst, const Value &src,
                      std::size_t r0, std::size_t c0) {
        const std::size_t Hs = src.dims().rows();
        const std::size_t Ws = src.dims().cols();
        const std::size_t Hd = dst.dims().rows();
        double *dd = dst.doubleDataMut();
        for (std::size_t c = 0; c < Ws; ++c)
            for (std::size_t r = 0; r < Hs; ++r)
                dd[(c0 + c) * Hd + (r0 + r)] += src.elemAsDouble(c * Hs + r);
    };
    add_at(h, scale_mat(P0, a[0]), insetR, insetC);
    std::size_t rowsT = insetR + 1, colsT = insetC + 1; // 1-based MATLAB
    (void)rowsT; (void)colsT;
    // For i = 3..n+1 (MATLAB) → C++ i = 2..n (after a[k-1] indexing).
    std::size_t cur_rows_off = insetR;
    std::size_t cur_cols_off = insetC;
    for (long i = 2; i <= n; ++i) {
        // P2 = 2 * conv2(t, P1)  (default 'full' shape).
        Value conv = signal::conv2(t, P1, "full", mr);
        Value P2 = scale_mat(conv, 2.0);
        cur_rows_off += insetR;
        cur_cols_off += insetC;
        // P2(rows, cols) -= P0  (at offset insetR*(i-1), insetC*(i-1) ... etc)
        const std::size_t Hp = P2.dims().rows();
        double *p2d = P2.doubleDataMut();
        const std::size_t Hp0 = P0.dims().rows();
        const std::size_t Wp0 = P0.dims().cols();
        for (std::size_t c = 0; c < Wp0; ++c)
            for (std::size_t r = 0; r < Hp0; ++r)
                p2d[(cur_cols_off + c) * Hp + (cur_rows_off + r)]
                    -= P0.elemAsDouble(c * Hp0 + r);
        // Wrap insertion: h_new = a[i] * P2; then add h (old) at offset (insetR, insetC).
        Value h_new = scale_mat(P2, a[static_cast<std::size_t>(i)]);
        add_at(h_new, h, insetR, insetC);
        h = std::move(h_new);
        P0 = P1;
        P1 = P2;
    }
    // rot90(h, 2) for filter2.
    return rot90_2(h, mr);
}

// ── fwind1 ───────────────────────────────────────────────────────
// Linear interpolation utility used by Huang's method.
template <typename T>
T linear_interp_t(const std::vector<double> &xs,
                  const std::vector<T> &ys, double q)
{
    if (q <= xs.front()) return ys.front();
    if (q >= xs.back())  return ys.back();
    const double dx = xs[1] - xs[0];
    const double t = (q - xs[0]) / dx;
    const std::size_t i = static_cast<std::size_t>(std::floor(t));
    const double a = t - static_cast<double>(i);
    return (1.0 - a) * ys[i] + a * ys[i + 1];
}

Value fwind1(const Value &Hd, const Value &win1, const Value &win2,
             std::pmr::memory_resource *mr)
{
    internal::requireImageClassAllowComplex(Hd, "fwind1", "Hd", 1,
                                internal::ImgClasses::FloatOnly);
    internal::requireImageClassAllowComplex(win1, "fwind1", "win", 2,
                                internal::ImgClasses::FloatOnly);
    if (!win2.isEmpty()) {
        internal::requireImageClassAllowComplex(win2, "fwind1", "win2", 3,
                                    internal::ImgClasses::FloatOnly);
    }
    if (Hd.numel() == 0)
        throw Error("fwind1: Hd must be nonempty",
                    0, 0, "fwind1", "", "numkit:fwind1:empty");
    if (win1.numel() < 2)
        throw Error("fwind1: window length must be >= 2",
                    0, 0, "fwind1", "", "numkit:fwind1:winShort");

    const std::size_t n = win1.numel();
    const bool winComplex = win1.isComplex() || (!win2.isEmpty() && win2.isComplex());
    const double sqrt_eps = std::sqrt(std::numeric_limits<double>::epsilon());

    auto getC = [](const Value &v, std::size_t idx) -> std::complex<double> {
        return v.isComplex() ? v.complexData()[idx]
                             : std::complex<double>(v.elemAsDouble(idx), 0.0);
    };

    Value W;
    std::size_t m_out, n_out;
    if (win2.isEmpty()) {
        m_out = n_out = n;
        std::vector<double> t(n);
        for (std::size_t i = 0; i < n; ++i)
            t[i] = (static_cast<double>(i) - (n - 1) / 2.0) * (2.0 / (n - 1));
        if (winComplex) {
            std::vector<std::complex<double>> w1(n);
            for (std::size_t i = 0; i < n; ++i) w1[i] = getC(win1, i);
            for (std::size_t i = 0; i < n; ++i)
                if (std::abs(w1[i] - w1[n - 1 - i]) > sqrt_eps)
                    throw Error("fwind1: 1-D window must be symmetric",
                                0, 0, "fwind1", "", "numkit:fwind1:winSym");
            W = Value::matrix(n, n, ValueType::COMPLEX, mr);
            auto *wd = W.complexDataMut();
            for (std::size_t c = 0; c < n; ++c)
                for (std::size_t r = 0; r < n; ++r) {
                    const double rr = std::sqrt(t[c] * t[c] + t[r] * t[r]);
                    wd[c * n + r] = (rr < t.front() || rr > t.back())
                        ? std::complex<double>(0.0, 0.0)
                        : linear_interp_t(t, w1, rr);
                }
        } else {
            std::vector<double> w1(n);
            for (std::size_t i = 0; i < n; ++i) w1[i] = win1.elemAsDouble(i);
            for (std::size_t i = 0; i < n; ++i)
                if (std::fabs(w1[i] - w1[n - 1 - i]) > sqrt_eps)
                    throw Error("fwind1: 1-D window must be symmetric",
                                0, 0, "fwind1", "", "numkit:fwind1:winSym");
            W = Value::matrix(n, n, ValueType::DOUBLE, mr);
            double *wd = W.doubleDataMut();
            for (std::size_t c = 0; c < n; ++c)
                for (std::size_t r = 0; r < n; ++r) {
                    const double rr = std::sqrt(t[c] * t[c] + t[r] * t[r]);
                    wd[c * n + r] = (rr < t.front() || rr > t.back())
                        ? 0.0
                        : linear_interp_t(t, w1, rr);
                }
        }
    } else {
        const std::size_t m = win2.numel();
        m_out = m;  n_out = n;
        if (winComplex) {
            W = Value::matrix(m, n, ValueType::COMPLEX, mr);
            auto *wd = W.complexDataMut();
            for (std::size_t c = 0; c < n; ++c)
                for (std::size_t r = 0; r < m; ++r)
                    wd[c * m + r] = getC(win2, r) * getC(win1, c);
        } else {
            W = Value::matrix(m, n, ValueType::DOUBLE, mr);
            double *wd = W.doubleDataMut();
            for (std::size_t c = 0; c < n; ++c)
                for (std::size_t r = 0; r < m; ++r)
                    wd[c * m + r] = win2.elemAsDouble(r) * win1.elemAsDouble(c);
        }
    }

    const std::size_t Hh = Hd.dims().rows();
    const std::size_t Wh = Hd.dims().cols();
    Value Hd_used = (Hh == m_out && Wh == n_out)
        ? Hd
        : bilinear_interp_hd(Hd, m_out, n_out, mr);

    Value h = fsamp2(Hd_used, Value::Empty, Value::Empty, {}, mr);
    const std::size_t M = h.dims().rows();
    const std::size_t N = h.dims().cols();
    if (h.isComplex() || W.isComplex()) {
        Value out = Value::matrix(M, N, ValueType::COMPLEX, mr);
        auto *od = out.complexDataMut();
        for (std::size_t i = 0; i < M * N; ++i)
            od[i] = getC(h, i) * getC(W, i);
        return out;
    }
    Value out = Value::matrix(M, N, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    for (std::size_t i = 0; i < M * N; ++i)
        od[i] = h.elemAsDouble(i) * W.elemAsDouble(i);
    return out;
}

// ── fwind2 ───────────────────────────────────────────────────────
Value fwind2(const Value &Hd, const Value &W,
             std::pmr::memory_resource *mr)
{
    internal::requireImageClassAllowComplex(Hd, "fwind2", "Hd", 1,
                                internal::ImgClasses::FloatOnly);
    internal::requireImageClassAllowComplex(W, "fwind2", "win", 2,
                                internal::ImgClasses::FloatOnly);
    if (Hd.numel() == 0)
        throw Error("fwind2: Hd must be nonempty",
                    0, 0, "fwind2", "", "numkit:fwind2:empty");
    if (W.numel() == 0)
        throw Error("fwind2: W must be nonempty",
                    0, 0, "fwind2", "", "numkit:fwind2:wEmpty");

    auto getC = [](const Value &v, std::size_t idx) -> std::complex<double> {
        return v.isComplex() ? v.complexData()[idx]
                             : std::complex<double>(v.elemAsDouble(idx), 0.0);
    };

    Value Wouter;
    const Value *Wp = &W;
    if (W.dims().rows() == 1 || W.dims().cols() == 1) {
        const std::size_t n = W.numel();
        if (W.isComplex()) {
            Wouter = Value::matrix(n, n, ValueType::COMPLEX, mr);
            auto *od2 = Wouter.complexDataMut();
            for (std::size_t c = 0; c < n; ++c)
                for (std::size_t r = 0; r < n; ++r)
                    od2[c * n + r] = getC(W, r) * getC(W, c);
        } else {
            Value Wd = as_double(W, mr);
            Wouter = Value::matrix(n, n, ValueType::DOUBLE, mr);
            double *od2 = Wouter.doubleDataMut();
            for (std::size_t c = 0; c < n; ++c)
                for (std::size_t r = 0; r < n; ++r)
                    od2[c * n + r] = Wd.elemAsDouble(r) * Wd.elemAsDouble(c);
        }
        Wp = &Wouter;
    }
    const Value &Wref = *Wp;
    const std::size_t Mh = Hd.dims().rows();
    const std::size_t Nh = Hd.dims().cols();
    const std::size_t Mw = Wref.dims().rows();
    const std::size_t Nw = Wref.dims().cols();
    Value Hd_used = (Mh == Mw && Nh == Nw)
        ? Hd
        : bilinear_interp_hd(Hd, Mw, Nw, mr);

    Value h = fsamp2(Hd_used, Value::Empty, Value::Empty, {}, mr);
    if (h.isComplex() || Wref.isComplex()) {
        Value out = Value::matrix(Mw, Nw, ValueType::COMPLEX, mr);
        auto *od = out.complexDataMut();
        for (std::size_t i = 0; i < Mw * Nw; ++i)
            od[i] = getC(h, i) * getC(Wref, i);
        return out;
    }
    Value out = Value::matrix(Mw, Nw, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    for (std::size_t i = 0; i < Mw * Nw; ++i)
        od[i] = h.elemAsDouble(i) * Wref.elemAsDouble(i);
    if (Hd.type() == ValueType::SINGLE || W.type() == ValueType::SINGLE)
        return internal::castResultToClass(out, ValueType::SINGLE, mr);
    return out;
}

} // namespace numkit::image
