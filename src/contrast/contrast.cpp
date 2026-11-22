// toolboxes/image/src/contrast/contrast.cpp
//
// Contrast enhancement, adaptive thresholding, multi-level Otsu, histogram
// matching, flat-field correction, and quantization — delegating numerical
// kernels to keel::image.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/image.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>

#include <numkit/image/contrast/contrast.hpp>
#include <numkit/image/internal/type_gates.hpp>
#include <numkit/image/type_convert/type_convert.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/value_type.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <tuple>

namespace numkit::image {

namespace {

inline keel::f64 element_to_unit(const Value &x, keel::usize i)
{
    const keel::f64 v = x.elemAsDouble(i);
    switch (x.type()) {
        case ValueType::DOUBLE:
        case ValueType::SINGLE:  return v;
        case ValueType::UINT8:   return v / 255.0;
        case ValueType::UINT16:  return v / 65535.0;
        case ValueType::INT16:   return (v + 32768.0) / 65535.0;
        case ValueType::LOGICAL: return v != 0.0 ? 1.0 : 0.0;
        default:                 return v;
    }
}

inline keel::i32 default_nbins(const Value &I)
{
    switch (I.type()) {
        case ValueType::UINT8:  return 256;
        case ValueType::UINT16: return 65536;
        default:                return 64;
    }
}

inline void store_classed(Value &out, keel::usize i, keel::f64 v, ValueType t)
{
    switch (t) {
        case ValueType::DOUBLE: out.doubleDataMut()[i] = v; break;
        case ValueType::SINGLE: out.singleDataMut()[i] = keel::as<keel::f32>(v); break;
        case ValueType::UINT8: {
            keel::f64 w = std::round(v * 255.0);
            if (w < 0) w = 0; if (w > 255) w = 255;
            out.uint8DataMut()[i] = keel::as<keel::u8>(w); break;
        }
        case ValueType::UINT16: {
            keel::f64 w = std::round(v * 65535.0);
            if (w < 0) w = 0; if (w > 65535) w = 65535;
            out.uint16DataMut()[i] = keel::as<keel::u16>(w); break;
        }
        case ValueType::INT16: {
            keel::f64 w = std::round(v * 65535.0) - 32768.0;
            if (w < -32768) w = -32768; if (w > 32767) w = 32767;
            out.int16DataMut()[i] = keel::as<keel::i16>(w); break;
        }
        default:
            throw Error("contrast: unsupported class", 0, 0, "contrast", "",
                        "numkit:contrast:badtype");
    }
}

Value allocSameShape(const Value &I, ValueType t, keel::Allocator *mr)
{
    const auto &d = I.dims();
    if (I.isScalar()) return Value::matrix(1, 1, t, mr);
    if (d.is3D())     return Value::matrix3d(d.rows(), d.cols(), d.pages(), t, mr);
    return Value::matrix(d.rows(), d.cols(), t, mr);
}

void unpackUnit(const Value &I, keel::Array<keel::f64> &buf)
{
    const keel::usize n = I.numel();
    buf.resize(n);
    for (keel::usize i = 0; i < n; ++i) buf[i] = element_to_unit(I, i);
}

void classRange(ValueType t, keel::f64 &lo, keel::f64 &hi)
{
    switch (t) {
        case ValueType::DOUBLE:
        case ValueType::SINGLE:
        case ValueType::LOGICAL: lo = 0.0;                       hi = 1.0;                       break;
        case ValueType::CHAR:    lo = 0.0;                       hi = 65535.0;                   break;
        case ValueType::INT8:    lo = -128.0;                    hi = 127.0;                     break;
        case ValueType::INT16:   lo = -32768.0;                  hi = 32767.0;                   break;
        case ValueType::INT32:   lo = -2147483648.0;             hi = 2147483647.0;              break;
        case ValueType::INT64:   lo = -9223372036854775808.0;    hi = 9223372036854775807.0;     break;
        case ValueType::UINT8:   lo = 0.0;                       hi = 255.0;                     break;
        case ValueType::UINT16:  lo = 0.0;                       hi = 65535.0;                   break;
        case ValueType::UINT32:  lo = 0.0;                       hi = 4294967295.0;              break;
        case ValueType::UINT64:  lo = 0.0;                       hi = 18446744073709551615.0;    break;
        default:                 lo = 0.0;                       hi = 1.0;                       break;
    }
}

void adapthisteq_argcheck(const Value &I, const AdaptHistEqOptions &opts)
{
    if (opts.numTilesR < 2 || opts.numTilesC < 2)
        throw Error("adapthisteq: NumTiles must be >= 2 in each dimension",
                    0, 0, "adapthisteq", "", "numkit:adapthisteq:badTiles");
    if (!(opts.clipLimit >= 0.0 && opts.clipLimit <= 1.0))
        throw Error("adapthisteq: ClipLimit must be in [0, 1]",
                    0, 0, "adapthisteq", "", "numkit:adapthisteq:badClip");
    if (opts.nBins < 2)
        throw Error("adapthisteq: NBins must be >= 2",
                    0, 0, "adapthisteq", "", "numkit:adapthisteq:badBins");
    if (opts.distribution != "uniform" && opts.distribution != "rayleigh" &&
        opts.distribution != "exponential")
        throw Error("adapthisteq: Distribution must be 'uniform', 'rayleigh' or 'exponential'",
                    0, 0, "adapthisteq", "", "numkit:adapthisteq:badDistribution");
    if (opts.range != "full" && opts.range != "original")
        throw Error("adapthisteq: Range must be 'full' or 'original'",
                    0, 0, "adapthisteq", "", "numkit:adapthisteq:badRange");

    const Dims &d = I.dims();
    if (d.ndim() > 2 || (d.ndim() == 3 && d.pages() > 1))
        throw Error("adapthisteq: input must be a 2-D greyscale image",
                    0, 0, "adapthisteq", "", "numkit:adapthisteq:unsupportedShape");
    if (I.isComplex() || I.isCell() || I.isStruct() ||
        I.isString() || I.isFuncHandle())
        throw Error("adapthisteq: input must be a real numeric image",
                    0, 0, "adapthisteq", "", "numkit:adapthisteq:unsupportedShape");
    if (d.rows() == 0 || d.cols() == 0)
        throw Error("adapthisteq: input image is empty",
                    0, 0, "adapthisteq", "", "numkit:adapthisteq:unsupportedShape");
}

Value graysliceQuantize(const Value &I, keel::ArrayRef<keel::f64> thresh,
                        keel::usize n_levels, keel::Allocator *mr)
{
    const keel::usize N = I.numel();
    const ValueType outT = (n_levels < 256) ? ValueType::UINT8 : ValueType::DOUBLE;
    Value out = allocSameShape(I, outT, mr);
    if (N == 0) return out;
    const bool baseOne = (outT == ValueType::DOUBLE);
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 iv = I.elemAsDouble(i);
        keel::usize cnt = 0;
        for (keel::usize k = 0; k < thresh.size(); ++k) {
            if (thresh[k] <= iv) ++cnt;
            else                 break;
        }
        if (outT == ValueType::UINT8) {
            out.uint8DataMut()[i] = keel::as<keel::u8>(cnt > 255 ? 255 : cnt);
        } else {
            out.doubleDataMut()[i] = keel::as<keel::f64>(cnt) + (baseOne ? 1.0 : 0.0);
        }
    }
    return out;
}

} // anonymous

std::tuple<Value, Value>
imhist(const Value &I, keel::i32 n, keel::Allocator *mr)
{
    if (n <= 0) n = default_nbins(I);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::i64> counts(keel::as<keel::usize>(n), 0, scratch.allocator());
    const keel::usize N = I.numel();
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 u = element_to_unit(I, i);
        if (std::isnan(u)) continue;
        keel::i32 bin = keel::as<keel::i32>(std::round(u * (n - 1)));
        if (bin < 0) bin = 0;
        if (bin >= n) bin = n - 1;
        ++counts[keel::as<keel::usize>(bin)];
    }
    Value c = Value::matrix(keel::as<keel::usize>(n), 1, ValueType::DOUBLE, mr);
    Value x = Value::matrix(keel::as<keel::usize>(n), 1, ValueType::DOUBLE, mr);
    keel::f64 *cd = c.doubleDataMut();
    keel::f64 *xd = x.doubleDataMut();
    keel::f64 lo = 0.0, hi = 1.0;
    switch (I.type()) {
        case ValueType::UINT8:  hi = 255.0; break;
        case ValueType::UINT16: hi = 65535.0; break;
        case ValueType::INT16:  lo = -32768.0; hi = 32767.0; break;
        default: break;
    }
    const keel::f64 step = (n > 1) ? (hi - lo) / keel::as<keel::f64>(n - 1) : 0.0;
    for (keel::i32 i = 0; i < n; ++i) {
        cd[i] = keel::as<keel::f64>(counts[keel::as<keel::usize>(i)]);
        xd[i] = lo + i * step;
    }
    return std::make_tuple(std::move(c), std::move(x));
}

Value stretchlim(const Value &I, keel::f64 low_tol, keel::f64 high_tol, keel::Allocator *mr)
{
    internal::requireImageClass(I, "stretchlim", "I", 1, internal::ImgClasses::FloatInt);
    internal::requireNonemptyImage(I, "stretchlim", "I", 1);

    const auto &d = I.dims();
    const bool is_rgb = d.is3D() && d.pages() == 3;
    const keel::usize plane = d.is3D() ? d.rows() * d.cols() : I.numel();
    const keel::usize channels = is_rgb ? 3 : 1;
    const keel::i32 nbins = (I.type() == ValueType::UINT8) ? 256 : 65536;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> unit(plane * channels, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < plane * channels; ++i) unit[i] = element_to_unit(I, i);

    Value out = Value::matrix(2, channels, ValueType::DOUBLE, mr);
    (void)keel::image::stretchlim_unit(
        unit.as_ref(), plane, channels, nbins, low_tol, high_tol,
        keel::MutArrayRef<keel::f64>(out.doubleDataMut(), 2 * channels),
        scratch.allocator());
    return out;
}

Value imadjust(const Value &I, keel::f64 low_in, keel::f64 high_in,
               keel::f64 low_out, keel::f64 high_out, keel::f64 gamma,
               keel::Allocator *mr)
{
    internal::requireImageClass(I, "imadjust", "I", 1, internal::ImgClasses::FloatInt);
    if (std::isnan(low_in) || std::isnan(high_in)) {
        Value lim = stretchlim(I, 0.01, 0.99, mr);
        if (std::isnan(low_in))  low_in  = lim.elemAsDouble(0);
        if (std::isnan(high_in)) high_in = lim.elemAsDouble(1);
    }

    const keel::usize N = I.numel();
    Value out = allocSameShape(I, I.type(), mr);
    if (N == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> in_u(scratch.allocator());
    unpackUnit(I, in_u);
    keel::Array<keel::f64> out_u(N, 0.0, scratch.allocator());
    (void)keel::image::imadjust_unit(in_u.as_ref(), low_in, high_in, low_out, high_out, gamma,
                                     out_u.as_mut_ref());
    for (keel::usize i = 0; i < N; ++i) store_classed(out, i, out_u[i], I.type());
    return out;
}

Value histeq(const Value &I, keel::i32 n, keel::Allocator *mr)
{
    if (n <= 0) n = 64;
    const keel::usize N = I.numel();
    if (N == 0) return allocSameShape(I, I.type(), mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> in_u(scratch.allocator());
    unpackUnit(I, in_u);
    keel::Array<keel::f64> out_u(N, 0.0, scratch.allocator());
    (void)keel::image::histeq_unit(in_u.as_ref(), n, out_u.as_mut_ref(), scratch.allocator());

    Value out = allocSameShape(I, I.type(), mr);
    for (keel::usize i = 0; i < N; ++i) store_classed(out, i, out_u[i], I.type());
    return out;
}

Value adapthisteq(const Value &I, const AdaptHistEqOptions &opts, keel::Allocator *mr)
{
    adapthisteq_argcheck(I, opts);
    const auto &d = I.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    const ValueType outType = I.type();

    const auto dist = (opts.distribution == "rayleigh")    ? keel::image::ClaheDistribution::kRayleigh
                    : (opts.distribution == "exponential") ? keel::image::ClaheDistribution::kExponential
                                                           : keel::image::ClaheDistribution::kUniform;
    const bool rangeOriginal = (opts.range == "original");

    keel::f64 classLo = 0.0, classHi = 1.0;
    classRange(outType, classLo, classHi);

    keel::ArenaTemp arena = keel::get_scratch(mr);
    keel::Array<keel::f64> src(H * W, 0.0, arena.allocator());
    for (keel::usize i = 0; i < H * W; ++i) src[i] = I.elemAsDouble(i);
    keel::Array<keel::f64> outImg(H * W, 0.0, arena.allocator());

    (void)keel::image::clahe_2d(src.as_ref(), H, W, opts.numTilesR, opts.numTilesC,
                                opts.clipLimit, opts.nBins, dist, opts.alpha,
                                classLo, classHi, rangeOriginal, outImg.as_mut_ref(),
                                arena.allocator());

    Value result = Value::matrix(H, W, outType, mr);
    const bool intOut = isIntegerType(outType) || outType == ValueType::CHAR
                     || outType == ValueType::LOGICAL;
    for (keel::usize i = 0; i < H * W; ++i) {
        keel::f64 stored = outImg[i];
        if (intOut) {
            stored = (stored >= 0.0) ? std::floor(stored + 0.5) : std::ceil(stored - 0.5);
            if (stored < classLo) stored = classLo;
            if (stored > classHi) stored = classHi;
        }
        result.elemSet(i, Value::scalar(stored, arena.allocator()));
    }
    return result;
}

std::tuple<Value, Value>
otsuthresh(const Value &counts_v, keel::Allocator *mr)
{
    const keel::usize L = counts_v.numel();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> c(L, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < L; ++i) c[i] = counts_v.elemAsDouble(i);
    const auto r = keel::image::otsu_threshold(c.as_ref(), scratch.allocator());
    const keel::f64 thresh = (L > 1) ? (r.bin_level / keel::as<keel::f64>(L - 1)) : 0.0;
    return std::make_tuple(Value::scalar(thresh, mr), Value::scalar(r.effectiveness, mr));
}

std::tuple<Value, Value>
graythresh(const Value &I, keel::Allocator *mr)
{
    internal::requireImageClass(I, "graythresh", "I", 1, internal::ImgClasses::FloatInt);
    auto [counts, _] = imhist(I, 256, mr);
    return otsuthresh(counts, mr);
}

std::tuple<Value, Value>
multithresh(const Value &I, keel::i32 N, keel::Allocator *mr)
{
    if (N > 5)
        throw Error("multithresh: N > 5 not supported (exhaustive search would be too slow)",
                    0, 0, "multithresh", "", "numkit:multithresh:tooMany");
    if (N < 1) N = 1;

    const ValueType ty = I.type();
    const bool isInt = (ty == ValueType::UINT8 || ty == ValueType::UINT16
                     || ty == ValueType::INT16 || ty == ValueType::INT8
                     || ty == ValueType::LOGICAL);
    const keel::usize M = I.numel();

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> raw(M, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < M; ++i) raw[i] = I.elemAsDouble(i);

    Value t_out = Value::matrix(1, keel::as<keel::usize>(N), ValueType::DOUBLE, mr);
    keel::f64 em = 0.0;
    (void)keel::image::multithresh_otsu(
        raw.as_ref(), N, isInt,
        keel::MutArrayRef<keel::f64>(t_out.doubleDataMut(), keel::as<keel::usize>(N)),
        em, scratch.allocator());
    return std::make_tuple(std::move(t_out), Value::scalar(em, mr));
}

Value imbinarize(const Value &I, keel::f64 thresh, keel::Allocator *mr)
{
    internal::requireImageClass(I, "imbinarize", "I", 1, internal::ImgClasses::FloatInt);
    const keel::usize N = I.numel();
    Value out = allocSameShape(I, ValueType::LOGICAL, mr);
    if (N == 0) return out;
    keel::u8 *od = out.logicalDataMut();
    for (keel::usize i = 0; i < N; ++i) od[i] = (element_to_unit(I, i) > thresh) ? 1 : 0;
    return out;
}

Value imbinarize(const Value &I, const Value &T, keel::Allocator *mr)
{
    internal::requireImageClass(I, "imbinarize", "I", 1, internal::ImgClasses::FloatInt);
    if (T.numel() != I.numel())
        throw Error("imbinarize: per-pixel T must have the same number of elements as I",
                    0, 0, "imbinarize", "", "numkit:imbinarize:Tshape");
    const keel::usize N = I.numel();
    Value out = allocSameShape(I, ValueType::LOGICAL, mr);
    if (N == 0) return out;
    keel::u8 *od = out.logicalDataMut();
    for (keel::usize i = 0; i < N; ++i)
        od[i] = (element_to_unit(I, i) > T.elemAsDouble(i)) ? 1 : 0;
    return out;
}

Value imquantize(const Value &I, const Value &levels, keel::Allocator *mr)
{
    internal::requireImageClass(I, "imquantize", "I", 1, internal::ImgClasses::FloatInt);
    internal::requireImageClass(levels, "imquantize", "levels", 2, internal::ImgClasses::AnyNumericReal);
    const keel::usize Lcount = levels.numel();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> lv(Lcount, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < Lcount; ++i) lv[i] = levels.elemAsDouble(i);
    std::sort(lv.begin(), lv.end());

    const keel::usize N = I.numel();
    Value out = allocSameShape(I, ValueType::DOUBLE, mr);
    if (N == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 u = element_to_unit(I, i);
        keel::usize cls = 1;
        for (keel::usize k = 0; k < Lcount; ++k) {
            if (u <= lv[k]) break;
            ++cls;
        }
        od[i] = keel::as<keel::f64>(cls);
    }
    return out;
}

Value imhistmatch(const Value &I, const Value &ref, keel::i32 nbins,
                  Value *hgramOut, keel::Allocator *mr)
{
    if (nbins <= 0) nbins = 64;
    if (nbins < 2) nbins = 2;

    const keel::usize Ni = I.numel();
    const keel::usize Nr = ref.numel();
    if (Ni == 0 || Nr == 0) {
        if (hgramOut) *hgramOut = Value::matrix(1, keel::as<keel::usize>(nbins), ValueType::DOUBLE, mr);
        return Value::matrix(I.dims().rows(), I.dims().cols(), I.type(), mr);
    }

    keel::i32 npts = 256;
    switch (I.type()) {
        case ValueType::UINT16:
        case ValueType::INT16:   npts = 65536; break;
        case ValueType::LOGICAL: npts = 2;     break;
        default:                 npts = 256;   break;
    }

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> in_u(scratch.allocator()), ref_u(scratch.allocator());
    unpackUnit(I, in_u);
    unpackUnit(ref, ref_u);
    keel::Array<keel::f64> out_u(Ni, 0.0, scratch.allocator());

    Value hg;
    keel::MutArrayRef<keel::f64> hg_ref;
    if (hgramOut) {
        hg = Value::matrix(1, keel::as<keel::usize>(nbins), ValueType::DOUBLE, mr);
        hg_ref = keel::MutArrayRef<keel::f64>(hg.doubleDataMut(), keel::as<keel::usize>(nbins));
    }

    (void)keel::image::imhistmatch_unit(in_u.as_ref(), ref_u.as_ref(), npts, nbins,
                                        out_u.as_mut_ref(), hg_ref, scratch.allocator());

    const auto &d = I.dims();
    Value out = d.is3D() ? Value::matrix3d(d.rows(), d.cols(), d.pages(), I.type(), mr)
                         : Value::matrix(d.rows(), d.cols(), I.type(), mr);
    for (keel::usize i = 0; i < Ni; ++i) store_classed(out, i, out_u[i], I.type());
    if (hgramOut) *hgramOut = std::move(hg);
    return out;
}

Value adaptthresh(const Value &I, keel::f64 sensitivity, keel::i32 neighborhood,
                  keel::StringRef statistic, keel::Allocator *mr)
{
    internal::requireImageClass(I, "adaptthresh", "I", 1, internal::ImgClasses::FloatInt);
    if (!(sensitivity >= 0.0 && sensitivity <= 1.0))
        throw Error("adaptthresh: sensitivity must be in [0, 1]",
                    0, 0, "adaptthresh", "", "numkit:adaptthresh:sens");

    const keel::StringRef s = statistic.empty() ? keel::StringRef("mean") : statistic;
    bool is_gauss = false;
    if (s == "mean" || s == "Mean" || s == "MEAN")             is_gauss = false;
    else if (s == "gaussian" || s == "Gaussian" || s == "GAUSSIAN") is_gauss = true;
    else
        throw Error("adaptthresh: statistic must be 'mean' or 'gaussian'",
                    0, 0, "adaptthresh", "", "numkit:adaptthresh:stat");

    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> in_u(H * W, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < H * W; ++i) in_u[i] = element_to_unit(I, i);

    Value T = Value::matrix(H, W, ValueType::DOUBLE, mr);
    (void)keel::image::adaptthresh_2d(
        in_u.as_ref(), H, W, sensitivity, neighborhood, is_gauss,
        keel::MutArrayRef<keel::f64>(T.doubleDataMut(), H * W),
        scratch.allocator());
    return T;
}

Value imflatfield(const Value &I, keel::f64 sigma, const Value &mask, keel::Allocator *mr)
{
    const ValueType classin = I.type();
    const auto &d = I.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    const keel::usize pages = d.is3D() ? d.pages() : 1;
    const keel::usize plane = H * W;
    const keel::usize N = I.numel();

    if (mask.numel() > 0 && (mask.dims().rows() != H || mask.dims().cols() != W))
        throw Error("imflatfield: mask must match spatial dims of I",
                    0, 0, "imflatfield", "", "numkit:imflatfield:masksize");

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> in_u(N, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i) in_u[i] = element_to_unit(I, i);

    keel::Array<keel::u8> mask_u8(scratch.allocator());
    if (mask.numel() > 0) {
        mask_u8.resize(plane);
        for (keel::usize i = 0; i < plane; ++i) mask_u8[i] = (mask.elemAsDouble(i) != 0.0) ? 1 : 0;
    }

    keel::Array<keel::f64> out_u(N, 0.0, scratch.allocator());
    (void)keel::image::flatfield_2d(in_u.as_ref(), H, W, pages, sigma, mask_u8.as_ref(),
                                    out_u.as_mut_ref(), scratch.allocator());

    Value out = d.is3D() ? Value::matrix3d(H, W, pages, classin, mr)
                         : Value::matrix(H, W, classin, mr);
    for (keel::usize i = 0; i < N; ++i) store_classed(out, i, out_u[i], classin);
    return out;
}

Value wcodemat(const Value &X, keel::i32 nb, keel::StringRef opt, keel::i32 absol, keel::Allocator *mr)
{
    if (nb < 1)
        throw Error("wcodemat: NB must be a positive integer",
                    0, 0, "wcodemat", "", "numkit:wcodemat:nb");

    keel::String lo;
    lo.reserve(opt.size());
    for (char c : opt) lo.push_back(keel::to_ascii_lower(c));
    if (lo.empty()) lo.assign("mat");
    keel::image::WcodematMode mode = keel::image::WcodematMode::kMat;
    if (lo == "mat")      mode = keel::image::WcodematMode::kMat;
    else if (lo == "row") mode = keel::image::WcodematMode::kRow;
    else if (lo == "col") mode = keel::image::WcodematMode::kCol;
    else
        throw Error("wcodemat: opt must be 'mat', 'row', or 'col'",
                    0, 0, "wcodemat", "", "numkit:wcodemat:opt");

    const auto &d = X.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    const keel::usize P = d.is3D() ? d.pages() : 1;
    Value out = d.is3D() ? Value::matrix3d(H, W, P, ValueType::DOUBLE, mr)
                         : Value::matrix(H, W, ValueType::DOUBLE, mr);
    const keel::usize N = X.numel();
    if (N == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> xv(N, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i) xv[i] = X.elemAsDouble(i);
    (void)keel::image::wcodemat(xv.as_ref(), H, W, P, nb, mode, absol != 0,
                                keel::MutArrayRef<keel::f64>(out.doubleDataMut(), N));
    return out;
}

Value entropy(const Value &I, keel::i32 nbins, keel::Allocator *mr)
{
    const bool isLogical = (I.type() == ValueType::LOGICAL);
    if (nbins <= 0) nbins = isLogical ? 2 : 256;

    Value Iu = isLogical ? I : im2uint8(I, mr);
    auto [counts, _bins] = imhist(Iu, nbins, mr);
    const keel::f64 h = keel::image::shannon_entropy(
        keel::ArrayRef<keel::f64>(counts.doubleData(), keel::as<keel::usize>(nbins)));
    return Value::scalar(h, mr);
}

Value grayslice(const Value &I, keel::i32 N, keel::Allocator *mr)
{
    if (N < 1)
        throw Error("grayslice: N must be a positive integer",
                    0, 0, "grayslice", "", "numkit:grayslice:n");
    const ValueType ct = I.type();
    const bool isInt16 = (ct == ValueType::INT16);
    const bool isFloat = (ct == ValueType::DOUBLE || ct == ValueType::SINGLE);
    const keel::f64 n_scalar = keel::as<keel::f64>(N);
    const keel::usize k_max = keel::as<keel::usize>(N - 1);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> thresh(scratch.allocator());
    thresh.reserve(k_max);
    const keel::f64 scale =
          (ct == ValueType::UINT8)  ? 255.0
        : (ct == ValueType::UINT16) ? 65535.0
        : isInt16                   ? 65535.0
        : 1.0;
    for (keel::usize k = 1; k <= k_max; ++k) {
        const keel::f64 v_unit = keel::as<keel::f64>(k) / n_scalar;
        keel::f64 v_class = isFloat ? v_unit : std::round(v_unit * scale);
        if (!isFloat && isInt16) v_class -= 32768.0;
        thresh.push_back(v_class);
    }
    return graysliceQuantize(I, thresh.as_ref(), k_max + 1, mr);
}

Value grayslice(const Value &I, keel::ArrayRef<keel::f64> levels, keel::Allocator *mr)
{
    const ValueType ct = I.type();
    const bool isFloat = (ct == ValueType::DOUBLE || ct == ValueType::SINGLE);
    const keel::usize M = levels.size();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> thresh(levels, scratch.allocator());
    std::sort(thresh.begin(), thresh.end());
    const keel::usize N = I.numel();
    if (isFloat && N > 0) {
        keel::f64 imin =  std::numeric_limits<keel::f64>::infinity();
        keel::f64 imax = -std::numeric_limits<keel::f64>::infinity();
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 v = I.elemAsDouble(i);
            if (v < imin) imin = v;
            if (v > imax) imax = v;
        }
        for (keel::usize i = 0; i < M; ++i) {
            if (thresh[i] < imin) thresh[i] = imin;
            if (thresh[i] > imax) thresh[i] = imax;
        }
    }
    return graysliceQuantize(I, thresh.as_ref(), M + 1, mr);
}

} // namespace numkit::image
