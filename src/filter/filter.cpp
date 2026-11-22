// toolboxes/image/src/filter/filter.cpp

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/image.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/random.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <numkit/builtin/datafun.hpp>
#include <numkit/image/filter/filter.hpp>
#include <numkit/image/internal/type_gates.hpp>
#include <numkit/image/type_convert/type_convert.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "filter_detail.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <tuple>

namespace numkit::image {

namespace {

inline keel::image::PadMode to_keel_pad(PadMode m) noexcept
{
    switch (m) {
        case PadMode::Constant:  return keel::image::PadMode::kConstant;
        case PadMode::Replicate: return keel::image::PadMode::kReplicate;
        case PadMode::Symmetric: return keel::image::PadMode::kSymmetric;
        case PadMode::Circular:  return keel::image::PadMode::kCircular;
    }
    return keel::image::PadMode::kConstant;
}

Value mat_double(keel::ArrayRef<keel::f64> v, keel::usize rows, keel::usize cols,
                 keel::Allocator *mr)
{
    Value out = Value::matrix(rows, cols, ValueType::DOUBLE, mr);
    if (!v.empty())
        std::memcpy(out.doubleDataMut(), v.data(), v.size() * sizeof(keel::f64));
    return out;
}

inline void store_classed(Value &out, keel::usize i, keel::f64 v, ValueType t)
{
    switch (t) {
        case ValueType::DOUBLE: out.doubleDataMut()[i] = v; break;
        case ValueType::SINGLE: out.singleDataMut()[i] = keel::as<keel::f32>(v); break;
        case ValueType::UINT8: {
            if (v < 0.0) v = 0.0;
            if (v > 255.0) v = 255.0;
            out.uint8DataMut()[i] = keel::as<keel::u8>(std::lround(v));
            break;
        }
        case ValueType::UINT16: {
            if (v < 0.0) v = 0.0;
            if (v > 65535.0) v = 65535.0;
            out.uint16DataMut()[i] = keel::as<keel::u16>(std::lround(v));
            break;
        }
        case ValueType::INT16: {
            if (v < -32768.0) v = -32768.0;
            if (v > 32767.0) v = 32767.0;
            out.int16DataMut()[i] = keel::as<keel::i16>(std::lround(v));
            break;
        }
        case ValueType::LOGICAL:
            out.logicalDataMut()[i] = (v != 0.0) ? 1u : 0u;
            break;
        default:
            throw Error("imfilter: unsupported class", 0, 0, "imfilter", "",
                        "numkit:imfilter:badtype");
    }
}

inline void store_classed_wide(Value &out, keel::usize i, keel::f64 v, ValueType t)
{
    auto sat = [](keel::f64 x, keel::f64 lo, keel::f64 hi) -> keel::f64 {
        x = std::round(x);
        if (x < lo) x = lo;
        if (x > hi) x = hi;
        return x;
    };
    switch (t) {
        case ValueType::DOUBLE:  out.doubleDataMut()[i]  = v; break;
        case ValueType::SINGLE:  out.singleDataMut()[i]  = keel::as<keel::f32>(v); break;
        case ValueType::UINT8:   out.uint8DataMut()[i]   = keel::as<keel::u8>(sat(v, 0.0, 255.0)); break;
        case ValueType::UINT16:  out.uint16DataMut()[i]  = keel::as<keel::u16>(sat(v, 0.0, 65535.0)); break;
        case ValueType::UINT32:  out.uint32DataMut()[i]  = keel::as<keel::u32>(sat(v, 0.0, 4294967295.0)); break;
        case ValueType::INT8:    out.int8DataMut()[i]    = keel::as<keel::i8>(sat(v, -128.0, 127.0)); break;
        case ValueType::INT16:   out.int16DataMut()[i]   = keel::as<keel::i16>(sat(v, -32768.0, 32767.0)); break;
        case ValueType::INT32:   out.int32DataMut()[i]   = keel::as<keel::i32>(sat(v, -2147483648.0, 2147483647.0)); break;
        case ValueType::LOGICAL: out.logicalDataMut()[i] = (v >= 0.5) ? 1u : 0u; break;
        default:                 out.doubleDataMut()[i]  = v; break;
    }
}

keel::Array<keel::u8> unpack_domain(const Value &domain, keel::usize default_sz,
                                    keel::usize &dom_h, keel::usize &dom_w,
                                    keel::Allocator *alloc)
{
    if (domain.numel() == 0) {
        dom_h = default_sz;
        dom_w = default_sz;
        return keel::Array<keel::u8>(dom_h * dom_w, keel::u8{1}, alloc);
    }
    dom_h = domain.dims().rows();
    dom_w = domain.dims().cols();
    const keel::usize n = dom_h * dom_w;
    keel::Array<keel::u8> out(n, keel::u8{0}, alloc);
    for (keel::usize i = 0; i < n; ++i)
        out[i] = (domain.elemAsDouble(i) != 0.0) ? 1u : 0u;
    return out;
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

keel::image::PadMode modefilt_argcheck(const Value &A, keel::i32 fH, keel::i32 fW, keel::i32 fD,
                                       keel::StringRef padopt)
{
    const bool is3D = (fD > 0);
    if (fH <= 0 || fW <= 0 || (is3D && fD <= 0))
        throw Error("modefilt: filter size must be positive",
                    0, 0, "modefilt", "", "numkit:modefilt:badSize");
    if ((fH & 1) == 0 || (fW & 1) == 0 || (is3D && (fD & 1) == 0))
        throw Error("modefilt: filter size must be odd",
                    0, 0, "modefilt", "", "numkit:modefilt:notOdd");

    keel::image::PadMode pm = keel::image::PadMode::kSymmetric;
    if (padopt.empty() || padopt == "symmetric") pm = keel::image::PadMode::kSymmetric;
    else if (padopt == "replicate")              pm = keel::image::PadMode::kReplicate;
    else if (padopt == "zeros" || padopt == "zero") pm = keel::image::PadMode::kConstant;
    else throw Error("modefilt: padopt must be 'symmetric', 'replicate', or 'zeros'",
                     0, 0, "modefilt", "", "numkit:modefilt:badPad");

    const auto &d = A.dims();
    if (d.ndim() > 3 || (!is3D && d.ndim() > 2))
        throw Error("modefilt: input rank does not match filter rank",
                    0, 0, "modefilt", "", "numkit:modefilt:rank");
    return pm;
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

template <typename T>
void im2col_sliding_ptr(const T *ad, T *bd, keel::usize H, keel::usize W,
                        keel::i32 m, keel::i32 n)
{
    const keel::usize outH = keel::as<keel::usize>(m * n);
    const keel::usize Hp = H - keel::as<keel::usize>(m) + 1;
    const keel::usize Wp = W - keel::as<keel::usize>(n) + 1;
    keel::usize col = 0;
    for (keel::usize c0 = 0; c0 < Wp; ++c0) {
        for (keel::usize r0 = 0; r0 < Hp; ++r0) {
            keel::usize off = col * outH;
            for (keel::i32 bc = 0; bc < n; ++bc) {
                const keel::usize Acol = (c0 + keel::as<keel::usize>(bc)) * H;
                for (keel::i32 br = 0; br < m; ++br)
                    bd[off++] = ad[Acol + r0 + keel::as<keel::usize>(br)];
            }
            ++col;
        }
    }
}

template <typename T>
void im2col_distinct_ptr(const T *ad, T *bd, keel::usize H, keel::usize W,
                         keel::i32 m, keel::i32 n)
{
    const keel::usize outH = keel::as<keel::usize>(m * n);
    const keel::usize Hb = (H + keel::as<keel::usize>(m) - 1) / keel::as<keel::usize>(m);
    const keel::usize Wb = (W + keel::as<keel::usize>(n) - 1) / keel::as<keel::usize>(n);
    keel::usize col = 0;
    for (keel::usize bj = 0; bj < Wb; ++bj) {
        for (keel::usize bi = 0; bi < Hb; ++bi) {
            keel::usize off = col * outH;
            for (keel::i32 bc = 0; bc < n; ++bc) {
                const keel::usize cc = bj * keel::as<keel::usize>(n) + keel::as<keel::usize>(bc);
                if (cc >= W) { off += keel::as<keel::usize>(m); continue; }
                const keel::usize Acol = cc * H;
                for (keel::i32 br = 0; br < m; ++br) {
                    const keel::usize rr = bi * keel::as<keel::usize>(m) + keel::as<keel::usize>(br);
                    if (rr < H) bd[off] = ad[Acol + rr];
                    ++off;
                }
            }
            ++col;
        }
    }
}

template <typename T>
void col2im_distinct_ptr(const T *bd, T *ad, keel::usize mm, keel::usize nn,
                         keel::i32 m, keel::i32 n)
{
    const keel::usize Hb = (mm + keel::as<keel::usize>(m) - 1) / keel::as<keel::usize>(m);
    const keel::usize outH = keel::as<keel::usize>(m * n);
    keel::usize col = 0;
    for (keel::usize bj = 0; bj < (nn + keel::as<keel::usize>(n) - 1) / keel::as<keel::usize>(n); ++bj) {
        for (keel::usize bi = 0; bi < Hb; ++bi) {
            keel::usize off = col * outH;
            for (keel::i32 bc = 0; bc < n; ++bc) {
                const keel::usize cc = bj * keel::as<keel::usize>(n) + keel::as<keel::usize>(bc);
                if (cc >= nn) { off += keel::as<keel::usize>(m); continue; }
                const keel::usize Acol = cc * mm;
                for (keel::i32 br = 0; br < m; ++br) {
                    const keel::usize rr = bi * keel::as<keel::usize>(m) + keel::as<keel::usize>(br);
                    if (rr < mm) ad[Acol + rr] = bd[off];
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

Value padarray(const Value &x, keel::ArrayRef<keel::i32> padsize, PadMode mode,
               keel::f64 pad_value, keel::StringRef direction, keel::Allocator *mr)
{
    padarray_argcheck(x);
    const auto &d = x.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    const keel::usize P = d.is3D() ? d.pages() : 1;

    const keel::i32 padR = padsize.size() >= 1 ? padsize[0] : 0;
    const keel::i32 padC = padsize.size() >= 2 ? padsize[1] : 0;
    const keel::i32 padP = padsize.size() >= 3 ? padsize[2] : 0;

    const bool pre  = (direction == "pre");
    const bool post = (direction == "post");
    const bool both = !pre && !post;

    const keel::i32 prR = both ? padR : (pre ? padR : 0);
    const keel::i32 poR = both ? padR : (post ? padR : 0);
    const keel::i32 prC = both ? padC : (pre ? padC : 0);
    const keel::i32 poC = both ? padC : (post ? padC : 0);
    const keel::i32 prP = both ? padP : (pre ? padP : 0);
    const keel::i32 poP = both ? padP : (post ? padP : 0);

    const keel::usize H2 = H + keel::as<keel::usize>(prR + poR);
    const keel::usize W2 = W + keel::as<keel::usize>(prC + poC);
    const keel::usize P2 = P + keel::as<keel::usize>(prP + poP);

    Value out = (P2 == 1) ? Value::matrix(H2, W2, x.type(), mr)
                          : Value::matrix3d(H2, W2, P2, x.type(), mr);
    if (H2 == 0 || W2 == 0 || P2 == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const keel::usize n_in = H * W * P;
    const keel::usize n_out = H2 * W2 * P2;
    keel::Array<keel::f64> src_f(n_in, scratch.allocator());
    for (keel::usize i = 0; i < n_in; ++i)
        src_f[i] = x.elemAsDouble(i);

    if (x.type() == ValueType::DOUBLE) {
        (void)keel::image::padarray_f64(src_f.as_ref(), H, W, P, prR, poR, prC, poC, prP, poP,
                                        to_keel_pad(mode), pad_value,
                                        keel::MutArrayRef<keel::f64>(out.doubleDataMut(), n_out));
        return out;
    }

    keel::Array<keel::f64> dst_f(n_out, scratch.allocator());
    (void)keel::image::padarray_f64(src_f.as_ref(), H, W, P, prR, poR, prC, poC, prP, poP,
                                    to_keel_pad(mode), pad_value, dst_f.as_mut_ref());
    for (keel::usize i = 0; i < n_out; ++i)
        store_classed(out, i, dst_f[i], x.type());
    return out;
}

Value fspecial(keel::StringRef type, keel::ArrayRef<keel::f64> params, keel::Allocator *mr)
{
    fspecial_argcheck(type);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::image::Kernel2D k{};

    if (type == "average") {
        const keel::i32 rows = params.size() >= 1 ? keel::as<keel::i32>(params[0]) : 3;
        const keel::i32 cols = params.size() >= 2 ? keel::as<keel::i32>(params[1]) : rows;
        (void)keel::image::fspecial_average(rows, cols, k, scratch.allocator());
    } else if (type == "gaussian") {
        const keel::i32 rows = params.size() >= 1 ? keel::as<keel::i32>(params[0]) : 3;
        const keel::i32 cols = params.size() >= 2 ? keel::as<keel::i32>(params[1]) : rows;
        const keel::f64 sigma = params.size() >= 3 ? params[2] : 0.5;
        if (sigma <= 0.0)
            throw Error("fspecial: sigma must be positive",
                        0, 0, "fspecial", "", "numkit:fspecial:sigma");
        (void)keel::image::fspecial_gaussian(rows, cols, sigma, k, scratch.allocator());
    } else if (type == "laplacian") {
        const keel::f64 alpha = params.size() >= 1 ? params[0] : 0.2;
        (void)keel::image::fspecial_laplacian(alpha, k, scratch.allocator());
    } else if (type == "log") {
        const keel::i32 rows = params.size() >= 1 ? keel::as<keel::i32>(params[0]) : 5;
        const keel::i32 cols = params.size() >= 2 ? keel::as<keel::i32>(params[1]) : rows;
        const keel::f64 sigma = params.size() >= 3 ? params[2] : 0.5;
        (void)keel::image::fspecial_log(rows, cols, sigma, k, scratch.allocator());
    } else if (type == "sobel") {
        (void)keel::image::fspecial_sobel(k, scratch.allocator());
    } else if (type == "prewitt") {
        (void)keel::image::fspecial_prewitt(k, scratch.allocator());
    } else if (type == "disk") {
        const keel::f64 radius = params.size() >= 1 ? params[0] : 5.0;
        if (radius <= 0.0)
            throw Error("fspecial: radius must be positive",
                        0, 0, "fspecial", "", "numkit:fspecial:radius");
        (void)keel::image::fspecial_disk(radius, k, scratch.allocator());
    } else if (type == "unsharp") {
        const keel::f64 alpha = params.size() >= 1 ? params[0] : 0.2;
        if (!(alpha >= 0.0 && alpha <= 1.0))
            throw Error("fspecial: unsharp alpha must be in [0, 1]",
                        0, 0, "fspecial", "", "numkit:fspecial:alpha");
        (void)keel::image::fspecial_unsharp(alpha, k, scratch.allocator());
    } else if (type == "motion") {
        const keel::f64 len   = params.size() >= 1 ? params[0] : 9.0;
        const keel::f64 theta = params.size() >= 2 ? params[1] : 0.0;
        (void)keel::image::fspecial_motion(len, theta, k, scratch.allocator());
    }
    return mat_double(k.data_col_major.as_ref(), k.rows, k.cols, mr);
}

Value imfilter(const Value &I, const Value &h, PadMode boundary, keel::f64 pad_value, bool full,
               bool flip_kernel, keel::Allocator *mr)
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
    if (I.type() == ValueType::LOGICAL)
        return internal::castResultToClass(
            imfilter(internal::coerceToDoubleImage(I, mr), h, boundary,
                     pad_value, full, flip_kernel, mr),
            ValueType::LOGICAL, mr);

    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    const keel::usize kH = h.dims().rows();
    const keel::usize kW = h.dims().cols();
    const keel::usize outH = full ? (H + kH - 1) : H;
    const keel::usize outW = full ? (W + kW - 1) : W;

    Value out = Value::matrix(outH, outW, I.type(), mr);
    if (outH == 0 || outW == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> kbuf(kH * kW, scratch.allocator());
    for (keel::usize i = 0; i < kH * kW; ++i)
        kbuf[i] = h.elemAsDouble(i);

    if (I.type() == ValueType::DOUBLE) {
        (void)keel::image::imfilter_2d(
            keel::ArrayRef<keel::f64>(I.doubleData(), H * W), H, W,
            kbuf.as_ref(), kH, kW, to_keel_pad(boundary), pad_value, full, flip_kernel,
            keel::MutArrayRef<keel::f64>(out.doubleDataMut(), outH * outW));
        return out;
    }

    keel::Array<keel::f64> src_f(H * W, scratch.allocator());
    for (keel::usize i = 0; i < H * W; ++i)
        src_f[i] = I.elemAsDouble(i);
    keel::Array<keel::f64> dst_f(outH * outW, scratch.allocator());
    (void)keel::image::imfilter_2d(src_f.as_ref(), H, W, kbuf.as_ref(), kH, kW,
                                   to_keel_pad(boundary), pad_value, full, flip_kernel,
                                   dst_f.as_mut_ref());
    for (keel::usize i = 0; i < outH * outW; ++i)
        store_classed(out, i, dst_f[i], I.type());
    return out;
}

Value imgaussfilt(const Value &I, keel::f64 sigma, keel::i32 filter_size, keel::Allocator *mr)
{
    if (filter_size <= 0) {
        filter_size = 2 * keel::as<keel::i32>(std::ceil(2.0 * sigma)) + 1;
        if (filter_size < 3) filter_size = 3;
    }
    if (filter_size % 2 == 0) filter_size += 1;

    if (I.dims().is3D() || sigma <= 0.0) {
        const keel::f64 p[3] = {keel::as<keel::f64>(filter_size),
                                keel::as<keel::f64>(filter_size), sigma};
        return imfilter(I, fspecial("gaussian", {p, 3}, mr),
                        PadMode::Replicate, 0.0, false, false, mr);
    }

    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    const keel::usize N = H * W;
    Value out = Value::matrix(H, W, I.type(), mr);
    if (N == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    if (I.type() == ValueType::DOUBLE) {
        (void)keel::image::gaussian_filter_2d(
            keel::ArrayRef<keel::f64>(I.doubleData(), N), H, W, sigma, filter_size, false,
            keel::MutArrayRef<keel::f64>(out.doubleDataMut(), N), scratch.allocator());
        return out;
    }

    keel::Array<keel::f64> src_f(N, scratch.allocator());
    keel::Array<keel::f64> dst_f(N, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i)
        src_f[i] = I.elemAsDouble(i);
    (void)keel::image::gaussian_filter_2d(src_f.as_ref(), H, W, sigma, filter_size, false,
                                          dst_f.as_mut_ref(), scratch.allocator());
    for (keel::usize i = 0; i < N; ++i)
        store_classed(out, i, dst_f[i], I.type());
    return out;
}

Value imboxfilt(const Value &I, keel::i32 filter_size, keel::Allocator *mr)
{
    imboxfilt_argcheck(I);
    if (filter_size <= 0) filter_size = 3;
    if (filter_size % 2 == 0) filter_size += 1;
    if (I.dims().is3D()) {
        const keel::f64 p[2] = {keel::as<keel::f64>(filter_size),
                                keel::as<keel::f64>(filter_size)};
        return imfilter(I, fspecial("average", {p, 2}, mr),
                        PadMode::Replicate, 0.0, false, false, mr);
    }

    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    const keel::usize N = H * W;
    Value out = Value::matrix(H, W, I.type(), mr);
    if (N == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    if (I.type() == ValueType::UINT8) {
        (void)keel::image::box_filter_2d_u8_replicate(
            keel::ArrayRef<keel::u8>(I.uint8Data(), N), H, W, filter_size,
            keel::MutArrayRef<keel::u8>(out.uint8DataMut(), N), scratch.allocator());
        return out;
    }
    if (I.type() == ValueType::DOUBLE) {
        (void)keel::image::box_filter_2d_replicate(
            keel::ArrayRef<keel::f64>(I.doubleData(), N), H, W, filter_size,
            keel::MutArrayRef<keel::f64>(out.doubleDataMut(), N), scratch.allocator());
        return out;
    }

    keel::Array<keel::f64> src_f(N, scratch.allocator());
    keel::Array<keel::f64> dst_f(N, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i)
        src_f[i] = I.elemAsDouble(i);
    (void)keel::image::box_filter_2d_replicate(src_f.as_ref(), H, W, filter_size,
                                               dst_f.as_mut_ref(), scratch.allocator());
    for (keel::usize i = 0; i < N; ++i)
        store_classed(out, i, dst_f[i], I.type());
    return out;
}

Value integralBoxFilter(const Value &I, keel::i32 fH, keel::i32 fW, keel::f64 normFactor,
                        keel::Allocator *mr)
{
    integralBoxFilter_argcheck(I, fH, fW);
    const auto &d = I.dims();
    const keel::usize H1 = d.rows();
    const keel::usize W1 = d.cols();
    const keel::usize C = (d.ndim() == 3) ? d.pages() : 1;
    const keel::usize outH = (H1 - 1) - keel::as<keel::usize>(fH) + 1;
    const keel::usize outW = (W1 - 1) - keel::as<keel::usize>(fW) + 1;

    Value out = (C == 1) ? Value::matrix(outH, outW, ValueType::DOUBLE, mr)
                         : Value::matrix3d(outH, outW, C, ValueType::DOUBLE, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const keel::usize inN = H1 * W1 * C;
    keel::Array<keel::f64> sat(inN, scratch.allocator());
    for (keel::usize i = 0; i < inN; ++i)
        sat[i] = I.elemAsDouble(i);

    (void)keel::image::integral_box_filter_2d(
        sat.as_ref(), H1, W1, C, fH, fW, normFactor,
        keel::MutArrayRef<keel::f64>(out.doubleDataMut(), outH * outW * C));
    return out;
}

Value integralBoxFilter3(const Value &I, keel::i32 fH, keel::i32 fW, keel::i32 fP,
                         keel::f64 normFactor, keel::Allocator *mr)
{
    integralBoxFilter3_argcheck(I, fH, fW, fP);
    const auto &d = I.dims();
    const keel::usize H1 = d.rows();
    const keel::usize W1 = d.cols();
    const keel::usize D1 = d.is3D() ? d.pages() : 1;
    const keel::usize outH = (H1 - 1) - keel::as<keel::usize>(fH) + 1;
    const keel::usize outW = (W1 - 1) - keel::as<keel::usize>(fW) + 1;
    const keel::usize outD = (D1 - 1) - keel::as<keel::usize>(fP) + 1;

    Value out = (outD > 1) ? Value::matrix3d(outH, outW, outD, ValueType::DOUBLE, mr)
                           : Value::matrix(outH, outW, ValueType::DOUBLE, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const keel::usize inN = H1 * W1 * D1;
    keel::Array<keel::f64> sat(inN, scratch.allocator());
    for (keel::usize i = 0; i < inN; ++i)
        sat[i] = I.elemAsDouble(i);

    (void)keel::image::integral_box_filter_3d(
        sat.as_ref(), H1, W1, D1, fH, fW, fP, normFactor,
        keel::MutArrayRef<keel::f64>(out.doubleDataMut(), outH * outW * outD));
    return out;
}

static Value modefilt_impl(const Value &A, keel::i32 fH, keel::i32 fW, keel::i32 fD,
                           keel::StringRef padopt, keel::Allocator *mr)
{
    const keel::image::PadMode pm = modefilt_argcheck(A, fH, fW, fD, padopt);
    const bool is3D = (fD > 0);
    const auto &d = A.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    const keel::usize D = is3D ? (d.is3D() ? d.pages() : 1) : 1;
    const keel::usize N = H * W * D;
    const ValueType vt = A.type();

    Value out = is3D ? Value::matrix3d(H, W, D, vt, mr)
                     : Value::matrix(H, W, vt, mr);
    if (N == 0) return out;

    if (vt == ValueType::UINT8 || vt == ValueType::LOGICAL) {
        const keel::u8 *src = (vt == ValueType::UINT8) ? A.uint8Data() : A.logicalData();
        keel::u8 *dst = (vt == ValueType::UINT8) ? out.uint8DataMut() : out.logicalDataMut();
        (void)keel::image::modefilt_u8(keel::ArrayRef<keel::u8>(src, N), H, W, D, fH, fW, fD, pm,
                                       keel::MutArrayRef<keel::u8>(dst, N));
        if (vt == ValueType::LOGICAL) {
            for (keel::usize i = 0; i < N; ++i) dst[i] = dst[i] ? 1u : 0u;
        }
        return out;
    }

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> src_f(N, scratch.allocator());
    keel::Array<keel::f64> dst_f(N, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i)
        src_f[i] = A.elemAsDouble(i);
    (void)keel::image::modefilt_f64(src_f.as_ref(), H, W, D, fH, fW, fD, pm, dst_f.as_mut_ref(),
                                    scratch.allocator());

    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 v = dst_f[i];
        switch (vt) {
            case ValueType::DOUBLE: out.doubleDataMut()[i] = v; break;
            case ValueType::SINGLE: out.singleDataMut()[i] = keel::as<keel::f32>(v); break;
            case ValueType::UINT16: out.uint16DataMut()[i] = keel::as<keel::u16>(v); break;
            case ValueType::UINT32: out.uint32DataMut()[i] = keel::as<keel::u32>(v); break;
            case ValueType::INT8:   out.int8DataMut()[i]   = keel::as<keel::i8>(v); break;
            case ValueType::INT16:  out.int16DataMut()[i]  = keel::as<keel::i16>(v); break;
            case ValueType::INT32:  out.int32DataMut()[i]  = keel::as<keel::i32>(v); break;
            default:                out.doubleDataMut()[i] = v; break;
        }
    }
    return out;
}

Value modefilt(const Value &A, keel::i32 fH, keel::i32 fW,
               keel::StringRef padopt, keel::Allocator *mr)
{
    return modefilt_impl(A, fH, fW, 0, padopt, mr);
}

Value modefilt3D(const Value &A, keel::i32 fH, keel::i32 fW, keel::i32 fD,
                 keel::StringRef padopt, keel::Allocator *mr)
{
    return modefilt_impl(A, fH, fW, fD, padopt, mr);
}

Value medfilt2(const Value &I, keel::i32 rows, keel::i32 cols, keel::Allocator *mr)
{
    medfilt2_argcheck(I);
    if (I.type() == ValueType::LOGICAL)
        return internal::castResultToClass(
            medfilt2(internal::coerceToDoubleImage(I, mr), rows, cols, mr),
            ValueType::LOGICAL, mr);

    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    const keel::usize N = H * W;
    Value out = Value::matrix(H, W, I.type(), mr);
    if (N == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    if (I.type() == ValueType::UINT8) {
        (void)keel::image::medfilt2_u8(keel::ArrayRef<keel::u8>(I.uint8Data(), N), H, W, rows,
                                       cols, keel::MutArrayRef<keel::u8>(out.uint8DataMut(), N),
                                       scratch.allocator());
        return out;
    }
    if (I.type() == ValueType::DOUBLE) {
        (void)keel::image::medfilt2_f64(keel::ArrayRef<keel::f64>(I.doubleData(), N), H, W, rows,
                                        cols, keel::MutArrayRef<keel::f64>(out.doubleDataMut(), N),
                                        scratch.allocator());
        return out;
    }

    keel::Array<keel::f64> src_f(N, scratch.allocator());
    keel::Array<keel::f64> dst_f(N, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i)
        src_f[i] = I.elemAsDouble(i);
    (void)keel::image::medfilt2_f64(src_f.as_ref(), H, W, rows, cols, dst_f.as_mut_ref(),
                                    scratch.allocator());
    for (keel::usize i = 0; i < N; ++i)
        store_classed(out, i, dst_f[i], I.type());
    return out;
}

Value imsharpen(const Value &I, keel::f64 radius, keel::f64 amount, keel::f64 threshold,
                keel::Allocator *mr)
{
    imsharpen_argcheck(I, threshold);
    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    const keel::usize N = H * W;
    Value out = Value::matrix(H, W, I.type(), mr);
    if (N == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    if (I.type() == ValueType::DOUBLE && !I.isComplex()) {
        (void)keel::image::imsharpen_2d(keel::ArrayRef<keel::f64>(I.doubleData(), N), H, W, radius,
                                        amount, threshold,
                                        keel::MutArrayRef<keel::f64>(out.doubleDataMut(), N),
                                        scratch.allocator());
        return out;
    }

    keel::Array<keel::f64> src_f(N, scratch.allocator());
    keel::Array<keel::f64> dst_f(N, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i)
        src_f[i] = I.elemAsDouble(i);
    (void)keel::image::imsharpen_2d(src_f.as_ref(), H, W, radius, amount, threshold,
                                    dst_f.as_mut_ref(), scratch.allocator());
    for (keel::usize i = 0; i < N; ++i)
        store_classed(out, i, dst_f[i], I.type());
    return out;
}

Value im2col(const Value &A, keel::i32 m, keel::i32 n, keel::StringRef block_type,
             keel::Allocator *mr)
{
    im2col_argcheck(A, m, n, block_type);
    const keel::usize H = A.dims().rows();
    const keel::usize W = A.dims().cols();
    const ValueType T = A.type();

    const keel::usize outH = keel::as<keel::usize>(m * n);
    keel::usize outW = 0;
    const bool sliding = (block_type == "sliding" || block_type.empty());
    if (sliding) {
        outW = (keel::as<keel::usize>(m) > H || keel::as<keel::usize>(n) > W)
                   ? 0
                   : (H - keel::as<keel::usize>(m) + 1) * (W - keel::as<keel::usize>(n) + 1);
    } else if (block_type == "distinct") {
        const keel::usize Hb = (H + keel::as<keel::usize>(m) - 1) / keel::as<keel::usize>(m);
        const keel::usize Wb = (W + keel::as<keel::usize>(n) - 1) / keel::as<keel::usize>(n);
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
        run(A.complexData(), B.complexDataMut());
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

Value imbilatfilt(const Value &I, keel::f64 degreeOfSmoothing, keel::f64 spatialSigma,
                  keel::Allocator *mr)
{
    imbilatfilt_argcheck(I, degreeOfSmoothing);
    if (!(spatialSigma > 0.0)) spatialSigma = 1.0;

    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    const keel::usize N = I.numel();
    Value out = Value::matrix(H, W, I.type(), mr);
    if (N == 0) return out;

    const keel::i32 half = std::max(1, keel::as<keel::i32>(std::ceil(2.0 * spatialSigma)));
    const keel::i32 win = 2 * half + 1;
    if (keel::as<keel::usize>(win) >= std::min(H, W))
        throw Error(keel::format("imbilatfilt: NeighborhoodSize (window {}) "
                                 "must be smaller than the image (min dimension {}).",
                                 win, std::min(H, W)),
                    0, 0, "imbilatfilt", "",
                    "numkit:imbilatfilt:nhoodTooBig");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    if (I.type() == ValueType::DOUBLE) {
        (void)keel::image::imbilatfilt_2d(
            keel::ArrayRef<keel::f64>(I.doubleData(), N), H, W, degreeOfSmoothing, spatialSigma,
            keel::MutArrayRef<keel::f64>(out.doubleDataMut(), N), scratch.allocator());
        return out;
    }

    keel::Array<keel::f64> src_f(N, scratch.allocator());
    keel::Array<keel::f64> dst_f(N, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i)
        src_f[i] = I.elemAsDouble(i);
    (void)keel::image::imbilatfilt_2d(src_f.as_ref(), H, W, degreeOfSmoothing, spatialSigma,
                                      dst_f.as_mut_ref(), scratch.allocator());
    for (keel::usize i = 0; i < N; ++i)
        store_classed(out, i, dst_f[i], I.type());
    return out;
}

Value col2im(const Value &B, keel::i32 m, keel::i32 n, keel::i32 mm, keel::i32 nn,
             keel::StringRef block_type, keel::Allocator *mr)
{
    col2im_argcheck(m, n, mm, nn, block_type);
    const keel::usize MM = keel::as<keel::usize>(mm);
    const keel::usize NN = keel::as<keel::usize>(nn);
    const ValueType T = B.type();

    const bool sliding = (block_type == "sliding" || block_type.empty());
    if (sliding) {
        if (keel::as<keel::usize>(m) > MM || keel::as<keel::usize>(n) > NN)
            throw Error("col2im(sliding): block larger than output image",
                        0, 0, "col2im", "", "numkit:col2im:size");
        const keel::usize Hp = MM - keel::as<keel::usize>(m) + 1;
        const keel::usize Wp = NN - keel::as<keel::usize>(n) + 1;
        if (B.numel() != Hp * Wp)
            throw Error("col2im(sliding): B size mismatch — expected 1×(mm−m+1)·(nn−n+1)",
                        0, 0, "col2im", "", "numkit:col2im:nelems");
        Value A = Value::matrix(Hp, Wp, T, mr);
        const keel::usize N = Hp * Wp;
        switch (T) {
            case ValueType::DOUBLE:  std::memcpy(A.doubleDataMut(), B.doubleData(), N * sizeof(keel::f64)); break;
            case ValueType::SINGLE:  std::memcpy(A.singleDataMut(), B.singleData(), N * sizeof(keel::f32)); break;
            case ValueType::UINT8:   std::memcpy(A.uint8DataMut(), B.uint8Data(), N); break;
            case ValueType::UINT16:  std::memcpy(A.uint16DataMut(), B.uint16Data(), N * sizeof(keel::u16)); break;
            case ValueType::INT16:   std::memcpy(A.int16DataMut(), B.int16Data(), N * sizeof(keel::i16)); break;
            case ValueType::LOGICAL: std::memcpy(A.logicalDataMut(), B.logicalData(), N); break;
            default:
                throw Error("col2im: unsupported class", 0, 0, "col2im", "", "numkit:col2im:badtype");
        }
        return A;
    }

    if (block_type != "distinct")
        throw Error("col2im: block_type must be 'sliding' or 'distinct'",
                    0, 0, "col2im", "", "numkit:col2im:type");

    const keel::usize Hb = (MM + keel::as<keel::usize>(m) - 1) / keel::as<keel::usize>(m);
    const keel::usize Wb = (NN + keel::as<keel::usize>(n) - 1) / keel::as<keel::usize>(n);
    const keel::usize expRows = keel::as<keel::usize>(m * n);
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
            throw Error("col2im: unsupported class", 0, 0, "col2im", "", "numkit:col2im:badtype");
    }
    return A;
}

Value imnoise(::numkit::ops::RngContext &rng, const Value &I, keel::StringRef mode,
              const Value &p1, const Value &p2, keel::Allocator *mr)
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
                out.uint8DataMut()[i] = keel::as<keel::u8>(std::lround(v * 255.0)); break;
            case ValueType::UINT16:
                out.uint16DataMut()[i] = keel::as<keel::u16>(std::lround(v * 65535.0)); break;
            case ValueType::INT16: {
                keel::f64 w = std::lround(v * 65535.0) - 32768.0;
                if (w < -32768.0) w = -32768.0;
                if (w >  32767.0) w =  32767.0;
                out.int16DataMut()[i] = keel::as<keel::i16>(w); break;
            }
            case ValueType::LOGICAL:
                out.logicalDataMut()[i] = v >= 0.5 ? 1u : 0u; break;
            case ValueType::SINGLE:
                out.singleDataMut()[i] = keel::as<keel::f32>(v); break;
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
    } else if (mode == "salt & pepper" || mode == "salt&pepper") {
        const keel::f64 d = (p1.numel() > 0) ? p1.toScalar() : 0.05;
        for (keel::usize i = 0; i < N; ++i) {
            keel::f64 x = toUnit(I.elemAsDouble(i));
            const keel::f64 u = keel::random::uniform_f64(rng);
            if      (u < d * 0.5) x = 0.0;
            else if (u < d)       x = 1.0;
            fromUnit(out, i, x);
        }
    } else if (mode == "speckle") {
        const keel::f64 v = (p1.numel() > 0) ? p1.toScalar() : 0.04;
        const keel::f64 sigma = std::sqrt(std::max(v, 0.0));
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 x = toUnit(I.elemAsDouble(i));
            fromUnit(out, i, x + x * keel::random::normal_f64(rng, 0.0, sigma));
        }
    } else if (mode == "poisson") {
        keel::f64 scale = 1e12;
        if (T == ValueType::UINT8)       scale = 255.0;
        else if (T == ValueType::UINT16) scale = 65535.0;
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 x = toUnit(I.elemAsDouble(i));
            const keel::f64 mean = std::max(x * scale, 0.0);
            const keel::u64 k = keel::random::poisson_u64(rng, mean);
            fromUnit(out, i, keel::as<keel::f64>(k) / scale);
        }
    } else if (mode == "localvar") {
        if (p1.numel() != N)
            throw Error("imnoise('localvar', V): V must match I in size",
                        0, 0, "imnoise", "", "numkit:imnoise:localvar");
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 x = toUnit(I.elemAsDouble(i));
            const keel::f64 v = std::max(p1.elemAsDouble(i), 0.0);
            fromUnit(out, i, x + keel::random::normal_f64(rng, 0.0, std::sqrt(v)));
        }
    } else {
        throw Error(keel::format("imnoise: unknown mode '{}'", mode),
                    0, 0, "imnoise", "", "numkit:imnoise:mode");
    }
    return out;
}

Value stdfilt(const Value &I, const Value &domain, keel::Allocator *mr)
{
    stdfilt_argcheck(I);
    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    const keel::usize N = H * W;
    Value out = Value::matrix(H, W, ValueType::DOUBLE, mr);
    if (N == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::usize dom_h = 0, dom_w = 0;
    keel::Array<keel::u8> dom = unpack_domain(domain, 3, dom_h, dom_w, scratch.allocator());

    keel::Array<keel::f64> src_f(N, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i)
        src_f[i] = I.elemAsDouble(i);

    (void)keel::image::stdfilt_2d(src_f.as_ref(), H, W, dom.as_ref(), dom_h, dom_w,
                                  keel::MutArrayRef<keel::f64>(out.doubleDataMut(), N),
                                  scratch.allocator());
    return out;
}

Value ordfilt2(const Value &A, keel::i32 nth, const Value &domain, const Value &S,
               PadMode boundary, keel::f64 pad_value, keel::Allocator *mr)
{
    ordfilt2_argcheck(A, domain);
    if (A.type() == ValueType::LOGICAL)
        return internal::castResultToClass(
            ordfilt2(internal::coerceToDoubleImage(A, mr), nth, domain, S,
                     boundary, pad_value, mr),
            ValueType::LOGICAL, mr);

    const keel::usize H = A.dims().rows();
    const keel::usize W = A.dims().cols();
    const keel::usize N = H * W;
    Value out = Value::matrix(H, W, A.type(), mr);
    if (N == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const keel::usize kH = domain.dims().rows();
    const keel::usize kW = domain.dims().cols();
    const keel::usize domN = kH * kW;
    keel::Array<keel::u8> dom(domN, scratch.allocator());
    keel::i32 M = 0;
    for (keel::usize i = 0; i < domN; ++i) {
        const bool set = (domain.elemAsDouble(i) != 0.0);
        dom[i] = set ? 1u : 0u;
        if (set) ++M;
    }
    if (M == 0) return out;
    if (nth < 1 || nth > M)
        throw Error("ordfilt2: nth-order index out of range",
                    0, 0, "ordfilt2", "", "numkit:ordfilt2:nth");

    keel::Array<keel::f64> s_buf(scratch.allocator());
    if (S.numel() == domN) {
        s_buf.resize(domN);
        for (keel::usize i = 0; i < domN; ++i)
            s_buf[i] = S.elemAsDouble(i);
    }

    keel::Array<keel::f64> src_f(N, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i)
        src_f[i] = A.elemAsDouble(i);

    if (A.type() == ValueType::DOUBLE) {
        (void)keel::image::ordfilt2_2d(
            src_f.as_ref(), H, W, nth, dom.as_ref(), kH, kW, s_buf.as_ref(),
            to_keel_pad(boundary), pad_value,
            keel::MutArrayRef<keel::f64>(out.doubleDataMut(), N), scratch.allocator());
        return out;
    }

    keel::Array<keel::f64> dst_f(N, scratch.allocator());
    (void)keel::image::ordfilt2_2d(src_f.as_ref(), H, W, nth, dom.as_ref(), kH, kW,
                                   s_buf.as_ref(), to_keel_pad(boundary), pad_value,
                                   dst_f.as_mut_ref(), scratch.allocator());
    for (keel::usize i = 0; i < N; ++i)
        store_classed(out, i, dst_f[i], A.type());
    return out;
}

std::tuple<Value, Value>
wiener2(const Value &I, keel::usize nh, keel::usize nw, keel::f64 noise, keel::Allocator *mr)
{
    const ValueType cls = I.type();
    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    const keel::usize N = H * W;
    if (N == 0) {
        Value out = Value::matrix(H, W, cls, mr);
        return {std::move(out), Value::scalar(0.0, mr)};
    }

    Value Id = (cls == ValueType::DOUBLE) ? I : im2double(I, mr);
    if (Id.type() != ValueType::DOUBLE)
        Id = im2double(Id, mr);

    Value out_d = Value::matrix(H, W, ValueType::DOUBLE, mr);
    keel::f64 noise_out = 0.0;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    (void)keel::image::wiener2_2d(
        keel::ArrayRef<keel::f64>(Id.doubleData(), N), H, W, nh, nw, noise,
        keel::MutArrayRef<keel::f64>(out_d.doubleDataMut(), N), noise_out, scratch.allocator());

    Value out;
    if (cls == ValueType::DOUBLE) {
        out = std::move(out_d);
    } else if (cls == ValueType::SINGLE) {
        out = Value::matrix(H, W, ValueType::SINGLE, mr);
        const keel::f64 *o = out_d.doubleData();
        keel::f32 *of = out.singleDataMut();
        for (keel::usize i = 0; i < N; ++i) of[i] = keel::as<keel::f32>(o[i]);
    } else if (cls == ValueType::UINT8)  out = im2uint8(out_d, mr);
    else if (cls == ValueType::UINT16)   out = im2uint16(out_d, mr);
    else if (cls == ValueType::INT16)    out = im2int16(out_d, mr);
    else                                 out = std::move(out_d);

    return {std::move(out), Value::scalar(noise_out, mr)};
}

Value imsmooth(const Value &I, keel::StringRef name, keel::f64 sigma, keel::Allocator *mr)
{
    imsmooth_argcheck(I, name, sigma);
    const ValueType cls = I.type();
    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    const keel::usize P = I.dims().is3D() ? I.dims().pages() : 1;
    const keel::usize plane = H * W;

    Value out = (P > 1) ? Value::matrix3d(H, W, P, cls, mr)
                        : Value::matrix(H, W, cls, mr);
    if (plane == 0) return out;

    const keel::i32 ksize = 2 * keel::as<keel::i32>(std::ceil(3.0 * sigma)) + 1;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> src_f(plane, scratch.allocator());
    keel::Array<keel::f64> dst_f(plane, scratch.allocator());

    for (keel::usize p = 0; p < P; ++p) {
        const keel::usize base = p * plane;
        for (keel::usize i = 0; i < plane; ++i)
            src_f[i] = I.elemAsDouble(base + i);
        (void)keel::image::gaussian_filter_2d(src_f.as_ref(), H, W, sigma, ksize, true,
                                              dst_f.as_mut_ref(), scratch.allocator());
        for (keel::usize i = 0; i < plane; ++i)
            store_classed(out, base + i, dst_f[i], cls);
    }
    return out;
}

Value imboxfilt3(const Value &V, keel::i32 fH, keel::i32 fW, keel::i32 fP, keel::Allocator *mr)
{
    imboxfilt3_argcheck(V);
    const ValueType cls = V.type();
    const keel::usize H = V.dims().rows();
    const keel::usize W = V.dims().cols();
    const keel::usize P = V.dims().is3D() ? V.dims().pages() : 1;
    const keel::usize total = H * W * P;
    Value out = (P > 1) ? Value::matrix3d(H, W, P, cls, mr)
                        : Value::matrix(H, W, cls, mr);
    if (total == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> src_f(total, scratch.allocator());
    for (keel::usize i = 0; i < total; ++i)
        src_f[i] = V.elemAsDouble(i);

    if (cls == ValueType::DOUBLE) {
        (void)keel::image::imboxfilt3_f64(
            src_f.as_ref(), H, W, P, fH, fW, fP,
            keel::MutArrayRef<keel::f64>(out.doubleDataMut(), total), scratch.allocator());
        return out;
    }

    keel::Array<keel::f64> dst_f(total, scratch.allocator());
    (void)keel::image::imboxfilt3_f64(src_f.as_ref(), H, W, P, fH, fW, fP, dst_f.as_mut_ref(),
                                      scratch.allocator());
    for (keel::usize i = 0; i < total; ++i)
        store_classed(out, i, dst_f[i], cls);
    return out;
}

Value imgaussfilt3(const Value &V, keel::f64 sigH, keel::f64 sigW, keel::f64 sigP,
                   keel::Allocator *mr)
{
    imgaussfilt3_argcheck(V);
    const ValueType cls = V.type();
    const keel::usize H = V.dims().rows();
    const keel::usize W = V.dims().cols();
    const keel::usize P = V.dims().is3D() ? V.dims().pages() : 1;
    const keel::usize total = H * W * P;
    Value out = (P > 1) ? Value::matrix3d(H, W, P, cls, mr)
                        : Value::matrix(H, W, cls, mr);
    if (total == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> src_f(total, scratch.allocator());
    for (keel::usize i = 0; i < total; ++i)
        src_f[i] = V.elemAsDouble(i);

    if (cls == ValueType::DOUBLE) {
        (void)keel::image::imgaussfilt3_f64(
            src_f.as_ref(), H, W, P, sigH, sigW, sigP,
            keel::MutArrayRef<keel::f64>(out.doubleDataMut(), total), scratch.allocator());
        return out;
    }

    keel::Array<keel::f64> dst_f(total, scratch.allocator());
    (void)keel::image::imgaussfilt3_f64(src_f.as_ref(), H, W, P, sigH, sigW, sigP,
                                        dst_f.as_mut_ref(), scratch.allocator());
    for (keel::usize i = 0; i < total; ++i)
        store_classed(out, i, dst_f[i], cls);
    return out;
}

Value medfilt3(const Value &V, keel::i32 M, keel::i32 N, keel::i32 P, keel::Allocator *mr)
{
    medfilt3_argcheck(V, M, N, P);
    if (V.type() == ValueType::LOGICAL)
        return internal::castResultToClass(
            medfilt3(internal::coerceToDoubleImage(V, mr), M, N, P, mr),
            ValueType::LOGICAL, mr);

    const ValueType cls = V.type();
    const keel::usize H = V.dims().rows();
    const keel::usize W = V.dims().cols();
    const keel::usize D = V.dims().is3D() ? V.dims().pages() : 1;
    const keel::usize total = H * W * D;
    Value out = (D > 1) ? Value::matrix3d(H, W, D, cls, mr)
                        : Value::matrix(H, W, cls, mr);
    if (total == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> src_f(total, scratch.allocator());
    for (keel::usize i = 0; i < total; ++i)
        src_f[i] = V.elemAsDouble(i);

    if (cls == ValueType::DOUBLE) {
        (void)keel::image::medfilt3_f64(
            src_f.as_ref(), H, W, D, M, N, P,
            keel::MutArrayRef<keel::f64>(out.doubleDataMut(), total), scratch.allocator());
        return out;
    }

    keel::Array<keel::f64> dst_f(total, scratch.allocator());
    (void)keel::image::medfilt3_f64(src_f.as_ref(), H, W, D, M, N, P, dst_f.as_mut_ref(),
                                    scratch.allocator());
    for (keel::usize i = 0; i < total; ++i)
        store_classed(out, i, dst_f[i], cls);
    return out;
}

Value convmtx2(const Value &h, keel::i32 m, keel::i32 n, keel::Allocator *mr)
{
    convmtx2_argcheck(h, m, n);
    const keel::usize M = h.dims().rows();
    const keel::usize N = h.dims().cols();
    const keel::usize mu = keel::as<keel::usize>(m);
    const keel::usize nu = keel::as<keel::usize>(n);
    const keel::usize P = (mu + M - 1) * (nu + N - 1);
    const keel::usize Q = mu * nu;

    Value T = Value::matrix(P, Q, ValueType::DOUBLE, mr);
    if (P == 0 || Q == 0) return T;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> h_f(M * N, scratch.allocator());
    for (keel::usize i = 0; i < M * N; ++i)
        h_f[i] = h.elemAsDouble(i);
    (void)keel::image::convmtx2_f64(h_f.as_ref(), M, N, mu, nu,
                                    keel::MutArrayRef<keel::f64>(T.doubleDataMut(), P * Q));
    return T;
}

Value entropyfilt(const Value &I, const Value &domain, keel::Allocator *mr)
{
    entropyfilt_argcheck(I);
    const bool isLogical = (I.type() == ValueType::LOGICAL);
    const keel::i32 nbins = isLogical ? 2 : 256;
    Value Iu = (isLogical || I.type() == ValueType::UINT8) ? I : im2uint8(I, mr);

    const keel::usize H = Iu.dims().rows();
    const keel::usize W = Iu.dims().cols();
    const keel::usize N = H * W;
    Value out = Value::matrix(H, W, ValueType::DOUBLE, mr);
    if (N == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::usize dom_h = 0, dom_w = 0;
    keel::Array<keel::u8> dom = unpack_domain(domain, 9, dom_h, dom_w, scratch.allocator());

    const keel::u8 *src = isLogical ? Iu.logicalData() : Iu.uint8Data();
    (void)keel::image::entropyfilt_u8(
        keel::ArrayRef<keel::u8>(src, N), H, W, nbins, dom.as_ref(), dom_h, dom_w,
        keel::MutArrayRef<keel::f64>(out.doubleDataMut(), N), scratch.allocator());
    return out;
}

Value rangefilt(const Value &I, const Value &domain, keel::Allocator *mr)
{
    rangefilt_argcheck(I);
    if (I.type() == ValueType::LOGICAL)
        return internal::castResultToClass(
            rangefilt(internal::coerceToDoubleImage(I, mr), domain, mr),
            ValueType::LOGICAL, mr);

    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    const keel::usize N = H * W;
    const ValueType outT = (I.type() == ValueType::INT16) ? ValueType::UINT16 : I.type();
    Value out = Value::matrix(H, W, outT, mr);
    if (N == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::usize dom_h = 0, dom_w = 0;
    keel::Array<keel::u8> dom = unpack_domain(domain, 3, dom_h, dom_w, scratch.allocator());

    keel::Array<keel::f64> src_f(N, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i)
        src_f[i] = I.elemAsDouble(i);

    if (outT == ValueType::DOUBLE) {
        (void)keel::image::rangefilt_2d(
            src_f.as_ref(), H, W, dom.as_ref(), dom_h, dom_w,
            keel::MutArrayRef<keel::f64>(out.doubleDataMut(), N), scratch.allocator());
        return out;
    }

    keel::Array<keel::f64> dst_f(N, scratch.allocator());
    (void)keel::image::rangefilt_2d(src_f.as_ref(), H, W, dom.as_ref(), dom_h, dom_w,
                                    dst_f.as_mut_ref(), scratch.allocator());
    for (keel::usize i = 0; i < N; ++i)
        store_classed(out, i, dst_f[i], outT);
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

    Value f1V = Value::matrix(M, 1, ValueType::DOUBLE, mr);
    Value f2V = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    Value HV  = Value::matrix(M, N, ValueType::COMPLEX, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> h_f(P * Q, scratch.allocator());
    for (keel::usize i = 0; i < P * Q; ++i)
        h_f[i] = h.elemAsDouble(i);

    (void)keel::image::freqz2_2d(
        h_f.as_ref(), P, Q, M, N,
        keel::MutArrayRef<keel::c64>(HV.complexDataMut(), M * N),
        keel::MutArrayRef<keel::f64>(f1V.doubleDataMut(), M),
        keel::MutArrayRef<keel::f64>(f2V.doubleDataMut(), N),
        scratch.allocator());
    return {std::move(HV), std::move(f1V), std::move(f2V)};
}

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
    for (keel::usize i = 0; i < N; ++i) {
        const bool masked = BW.elemAsDouble(i) != 0.0;
        const keel::f64 v = (masked ? filtRes : I).elemAsDouble(i);
        switch (cls) {
            case ValueType::SINGLE:  out.singleDataMut()[i]  = keel::as<keel::f32>(v); break;
            case ValueType::UINT8:   out.uint8DataMut()[i]   = keel::as<keel::u8>(std::lround(v)); break;
            case ValueType::UINT16:  out.uint16DataMut()[i]  = keel::as<keel::u16>(std::lround(v)); break;
            case ValueType::UINT32:  out.uint32DataMut()[i]  = keel::as<keel::u32>(std::lround(v)); break;
            case ValueType::INT8:    out.int8DataMut()[i]    = keel::as<keel::i8>(std::lround(v)); break;
            case ValueType::INT16:   out.int16DataMut()[i]   = keel::as<keel::i16>(std::lround(v)); break;
            case ValueType::INT32:   out.int32DataMut()[i]   = keel::as<keel::i32>(std::lround(v)); break;
            case ValueType::INT64:   out.int64DataMut()[i]   = keel::as<keel::i64>(std::lround(v)); break;
            case ValueType::LOGICAL: out.logicalDataMut()[i] = (v != 0.0) ? 1u : 0u; break;
            default:                 out.doubleDataMut()[i]  = v; break;
        }
    }
    return out;
}

Value roifilt2(const Value &h, const Value &I, const Value &BW, keel::Allocator *mr)
{
    roifilt2_argcheck(h, I);
    Value filtFull = imfilter(I, h, PadMode::Constant, 0.0, false, false, mr);
    return roifilt2_combine(I, filtFull, BW, mr);
}

Value imguidedfilter(const Value &A, const Value &G_in, keel::i32 nhood,
                     keel::f64 eps_in, keel::Allocator *mr)
{
    imguidedfilter_argcheck(A, nhood);
    const keel::usize H = A.dims().rows();
    const keel::usize W = A.dims().cols();
    if (H == 0 || W == 0) return A;

    const Value &G = (G_in.numel() == 0) ? A : G_in;
    if (G.dims().rows() != H || G.dims().cols() != W || G.dims().is3D())
        throw Error("imguidedfilter: G must be the same H × W as A "
                    "(RGB guidance not yet supported)",
                    0, 0, "imguidedfilter", "",
                    "numkit:imguidedfilter:gshape");

    keel::f64 eps = eps_in;
    if (eps < 0.0) {
        keel::f64 range = 1.0;
        switch (A.type()) {
            case ValueType::UINT8:  range = 255.0;        break;
            case ValueType::UINT16: range = 65535.0;      break;
            case ValueType::UINT32: range = 4294967295.0; break;
            case ValueType::INT8:   range = 255.0;        break;
            case ValueType::INT16:  range = 65535.0;      break;
            case ValueType::INT32:  range = 4294967295.0; break;
            default:                range = 1.0;          break;
        }
        eps = 0.01 * range * range;
    }

    const keel::usize N = H * W;
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> ad(N, scratch.allocator());
    keel::Array<keel::f64> gd(N, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i) {
        ad[i] = A.elemAsDouble(i);
        gd[i] = G.elemAsDouble(i);
    }

    const ValueType outT = A.type();
    if (outT == ValueType::DOUBLE) {
        Value out = Value::matrix(H, W, ValueType::DOUBLE, mr);
        (void)keel::image::guided_filter_gray(
            ad.as_ref(), gd.as_ref(), H, W, nhood, eps,
            keel::MutArrayRef<keel::f64>(out.doubleDataMut(), N), scratch.allocator());
        return out;
    }

    keel::Array<keel::f64> bd(N, scratch.allocator());
    (void)keel::image::guided_filter_gray(ad.as_ref(), gd.as_ref(), H, W, nhood, eps,
                                          bd.as_mut_ref(), scratch.allocator());
    Value out = Value::matrix(H, W, outT, mr);
    for (keel::usize i = 0; i < N; ++i)
        store_classed_wide(out, i, bd[i], outT);
    return out;
}

Value imdiffusefilt(const Value &I, const Value &thresh, keel::usize N_in,
                    keel::StringRef connectivity, keel::StringRef conduction,
                    keel::Allocator *mr)
{
    imdiffusefilt_argcheck(I, connectivity, conduction);
    const keel::usize M = I.dims().rows();
    const keel::usize W = I.dims().cols();
    if (M == 0 || W == 0) return I;

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

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> threshVec(scratch.allocator());
    if (thresh.numel() == 0) {
        threshVec.push_back(0.1 * class_range_diff());
    } else {
        threshVec.reserve(thresh.numel());
        for (keel::usize i = 0; i < thresh.numel(); ++i) {
            const keel::f64 v = thresh.elemAsDouble(i);
            if (!(v > 0.0) || !std::isfinite(v))
                throw Error("imdiffusefilt: GradientThreshold must be a "
                            "positive finite scalar or vector",
                            0, 0, "imdiffusefilt", "",
                            "numkit:imdiffusefilt:thresh");
            threshVec.push_back(v);
        }
    }

    keel::usize N = N_in;
    if (N == 0)
        N = (threshVec.size() == 1) ? 5 : threshVec.size();
    if (threshVec.size() == 1) {
        threshVec.resize(N, threshVec[0]);
    } else if (threshVec.size() != N) {
        throw Error("imdiffusefilt: numel(GradientThreshold) must equal "
                    "NumberOfIterations",
                    0, 0, "imdiffusefilt", "",
                    "numkit:imdiffusefilt:threshLen");
    }

    const keel::usize Npix = M * W;
    keel::Array<keel::f64> src_f(Npix, scratch.allocator());
    for (keel::usize i = 0; i < Npix; ++i)
        src_f[i] = I.elemAsDouble(i);

    const bool isMax = (connectivity == "maximal");
    const bool isExp = (conduction == "exponential");

    if (inT == ValueType::DOUBLE) {
        Value out = Value::matrix(M, W, ValueType::DOUBLE, mr);
        (void)keel::image::imdiffusefilt_2d(
            src_f.as_ref(), M, W, threshVec.as_ref(), isMax, isExp,
            keel::MutArrayRef<keel::f64>(out.doubleDataMut(), Npix), scratch.allocator());
        return out;
    }

    keel::Array<keel::f64> dst_f(Npix, scratch.allocator());
    (void)keel::image::imdiffusefilt_2d(src_f.as_ref(), M, W, threshVec.as_ref(), isMax, isExp,
                                        dst_f.as_mut_ref(), scratch.allocator());
    Value out = Value::matrix(M, W, inT, mr);
    for (keel::usize i = 0; i < Npix; ++i)
        store_classed_wide(out, i, dst_f[i], inT);
    return out;
}

void imgaborfilt(const Value &A_in, keel::f64 wavelength, keel::f64 orientation,
                 keel::f64 sfb, keel::f64 aspect,
                 Value &mag_out, Value &phase_out, keel::Allocator *mr)
{
    imgaborfilt_argcheck(A_in, wavelength, sfb, aspect);
    const keel::usize M = A_in.dims().rows();
    const keel::usize N = A_in.dims().cols();
    const keel::usize total = M * N;
    const ValueType outT = (A_in.type() == ValueType::SINGLE)
                               ? ValueType::SINGLE : ValueType::DOUBLE;

    mag_out = Value::matrix(M, N, outT, mr);
    phase_out = Value::matrix(M, N, outT, mr);
    if (total == 0) return;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> src_f(total, scratch.allocator());
    for (keel::usize i = 0; i < total; ++i)
        src_f[i] = A_in.elemAsDouble(i);

    if (outT == ValueType::DOUBLE) {
        (void)keel::image::imgaborfilt_2d(
            src_f.as_ref(), M, N, wavelength, orientation, sfb, aspect,
            keel::MutArrayRef<keel::f64>(mag_out.doubleDataMut(), total),
            keel::MutArrayRef<keel::f64>(phase_out.doubleDataMut(), total),
            scratch.allocator());
        return;
    }

    keel::Array<keel::f64> mag_f(total, scratch.allocator());
    keel::Array<keel::f64> phase_f(total, scratch.allocator());
    (void)keel::image::imgaborfilt_2d(src_f.as_ref(), M, N, wavelength, orientation, sfb, aspect,
                                      mag_f.as_mut_ref(), phase_f.as_mut_ref(),
                                      scratch.allocator());
    keel::f32 *md = mag_out.singleDataMut();
    keel::f32 *pd = phase_out.singleDataMut();
    for (keel::usize i = 0; i < total; ++i) {
        md[i] = keel::as<keel::f32>(mag_f[i]);
        pd[i] = keel::as<keel::f32>(phase_f[i]);
    }
}

void imnlmfilt(const Value &I, keel::f64 dos, keel::i32 swsize, keel::i32 cwsize,
               Value &J_out, keel::f64 &est_out, keel::Allocator *mr)
{
    imnlmfilt_argcheck(I, swsize, cwsize);
    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    const keel::usize N = H * W;
    const ValueType outT = I.type();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> src_f(N, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i)
        src_f[i] = I.elemAsDouble(i);

    J_out = Value::matrix(H, W, outT, mr);
    if (outT == ValueType::DOUBLE) {
        (void)keel::image::imnlmfilt_2d(
            src_f.as_ref(), H, W, dos, swsize, cwsize,
            keel::MutArrayRef<keel::f64>(J_out.doubleDataMut(), N), est_out,
            scratch.allocator());
        return;
    }

    keel::Array<keel::f64> dst_f(N, scratch.allocator());
    (void)keel::image::imnlmfilt_2d(src_f.as_ref(), H, W, dos, swsize, cwsize, dst_f.as_mut_ref(),
                                    est_out, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i)
        store_classed_wide(J_out, i, dst_f[i], outT);
}

} // namespace numkit::image
