// toolboxes/image/src/transform/transform.cpp
//
// Thin Value/argcheck wrappers delegating 2D DCT/IDCT, integral images,
// phantom/checkerboard generators, PSF/OTF conversions, normxcorr2, and
// Wiener/Tikhonov deconvolution to keel::image.

#include <keel/core.h>
#include <keel/algo.h>
#include <keel/format.h>
#include <keel/image.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <keel/static_array.h>
#include <keel/math.h>
#include <keel/tuple.h>

#include <numkit/image/transform/transform.hpp>
#include <numkit/image/internal/type_gates.hpp>
#include <numkit/image/filter/filter.hpp>

#include <numkit/signal/transforms/fft.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

namespace numkit::image {

namespace {

keel::Array<keel::f64> extract_double_buf(const Value &V, keel::Allocator *mr)
{
    const keel::usize N = V.numel();
    keel::Array<keel::f64> buf(N, mr);
    if (V.type() == ValueType::DOUBLE) {
        keel::copy_n(buf.data(), V.doubleData(), N);
    } else {
        for (keel::usize i = 0; i < N; ++i) buf[i] = V.elemAsDouble(i);
    }
    return buf;
}

Value complex_or_real_value(keel::ArrayRef<keel::c64> data, keel::usize H, keel::usize W,
                            keel::Allocator *mr)
{
    const keel::usize N = data.size();
    keel::f64 max_re = 0.0, max_im = 0.0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 ar = keel::abs(data[i].re);
        const keel::f64 ai = keel::abs(data[i].im);
        if (ar > max_re) max_re = ar;
        if (ai > max_im) max_im = ai;
    }
    if (max_im <= 1e-12 * keel::max(1.0, max_re)) {
        Value out = Value::matrix(H, W, ValueType::DOUBLE, mr);
        keel::f64 *od = out.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i) od[i] = data[i].re;
        return out;
    }
    Value out = Value::matrix(H, W, ValueType::COMPLEX, mr);
    keel::copy_n(out.complexDataMut(), data.data(), N);
    return out;
}

void dctmtx_argcheck(keel::f64 Nd)
{
    if (!(Nd > 0.0) || keel::floor(Nd) != Nd)
        throw Error("dctmtx: N must be a positive integer",
                    0, 0, "dctmtx", "", "numkit:dctmtx:arg");
}

Value make_std_ellipses(keel::image::PhantomModel model, keel::Allocator *mr)
{
    Value E = Value::matrix(10, 6, ValueType::DOUBLE, mr);
    (void)keel::image::phantom_ellipses(model, keel::MutArrayRef<keel::f64>(E.doubleDataMut(), 60));
    return E;
}

Value phantom_argcheck(const Value &model_or_E, keel::Allocator *mr)
{
    if (model_or_E.numel() == 0) {
        return make_std_ellipses(keel::image::PhantomModel::kModifiedSheppLogan, mr);
    }
    if (model_or_E.isChar() || model_or_E.isString()) {
        const keel::String m = model_or_E.toString();
        keel::String lo;
        lo.reserve(m.size());
        for (char c : m) lo.push_back(keel::to_ascii_lower(c));
        if (lo == "shepp-logan")
            return make_std_ellipses(keel::image::PhantomModel::kSheppLogan, mr);
        if (lo == "modified shepp-logan")
            return make_std_ellipses(keel::image::PhantomModel::kModifiedSheppLogan, mr);
        throw Error("phantom: unknown MODEL", 0, 0, "phantom", "", "numkit:phantom:model");
    }
    if (model_or_E.dims().cols() != 6)
        throw Error("phantom: E must be N-by-6", 0, 0, "phantom", "", "numkit:phantom:E");
    return model_or_E;
}

void bestblk_argcheck(const Value &IMS, keel::f64 k, Value &IMSC, keel::Allocator *mr)
{
    internal::requireRealKernelImage(IMS, "bestblk", "IMS", 1);
    if (IMS.type() == ValueType::CHAR)
        IMSC = internal::coerceToDoubleImage(IMS, mr);
    else
        internal::requireImageClass(IMS, "bestblk", "IMS", 1, internal::ImgClasses::FloatInt);
    const Value &IMSu = (IMS.type() == ValueType::CHAR) ? IMSC : IMS;
    if (IMSu.numel() < 2)
        throw Error("bestblk: IMS must have at least 2 elements",
                    0, 0, "bestblk", "", "numkit:bestblk:size");
    if (!(k >= 1.0))
        throw Error("bestblk: K must be a positive scalar",
                    0, 0, "bestblk", "", "numkit:bestblk:k");
}

void fftconv2_argcheck(const Value &A, const Value &B)
{
    internal::requireImageClass(A, "fftconv2", "A", 1, internal::ImgClasses::FloatIntLog);
    internal::requireImageClass(B, "fftconv2", "B", 2, internal::ImgClasses::FloatIntLog);
}

void psf2otf_argcheck(const Value &PSF, keel::ArrayRef<keel::usize> outsize,
                      keel::usize &inH, keel::usize &inW, keel::usize &outH, keel::usize &outW)
{
    const auto &d = PSF.dims();
    inH = d.rows();
    inW = d.cols();
    const bool is1D = (inH == 1 || inW == 1);
    outH = inH;
    outW = inW;
    if (outsize.size() >= 2) {
        outH = outsize[0];
        outW = outsize[1];
    } else if (outsize.size() == 1) {
        const keel::usize L = outsize[0];
        if (is1D) {
            if (inH == 1) { outH = 1; outW = L; }
            else          { outH = L; outW = 1; }
        } else { outH = L; outW = L; }
    }
    if (outH < inH || outW < inW)
        throw Error("psf2otf: OUTSIZE must be larger than PSF size",
                    0, 0, "psf2otf", "", "numkit:psf2otf:outsize");
}

void otf2psf_argcheck(const Value &OTF, keel::ArrayRef<keel::usize> outsize,
                      keel::usize &inH, keel::usize &inW, keel::usize &outH, keel::usize &outW)
{
    const auto &d = OTF.dims();
    inH = d.rows();
    inW = d.cols();
    const bool is1D = (inH == 1 || inW == 1);
    outH = inH;
    outW = inW;
    if (outsize.size() == 1) {
        const keel::usize L = outsize[0];
        if (is1D) {
            if (inH == 1) { outH = 1; outW = L; }
            else          { outH = L; outW = 1; }
        } else { outH = L; outW = L; }
    } else if (outsize.size() >= 2) {
        outH = outsize[0];
        outW = outsize[1];
    }
    if (outH > inH || outW > inW)
        throw Error("otf2psf: OUTSIZE must not exceed the size of the OTF "
                    "array in any dimension",
                    0, 0, "otf2psf", "", "numkit:otf2psf:outsize");
}

void normxcorr2_argcheck(const Value &templ, const Value &img)
{
    internal::requireImageClass(templ, "normxcorr2", "TEMPLATE", 1,
                                internal::ImgClasses::FloatIntNoLog);
    internal::requireImageClass(img, "normxcorr2", "INPUT", 2,
                                internal::ImgClasses::FloatIntNoLog);
}

Value to_double_unit(const Value &I, keel::Allocator *mr)
{
    const auto &d = I.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    Value out = d.is3D()
        ? Value::matrix3d(H, W, d.pages(), ValueType::DOUBLE, mr)
        : Value::matrix(H, W, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    const keel::usize N = I.numel();
    switch (I.type()) {
        case ValueType::DOUBLE:
            keel::copy_n(od, I.doubleData(), N); break;
        case ValueType::SINGLE:
            for (keel::usize i = 0; i < N; ++i) od[i] = I.singleData()[i]; break;
        case ValueType::UINT8:
            for (keel::usize i = 0; i < N; ++i) od[i] = I.uint8Data()[i] / 255.0; break;
        case ValueType::UINT16:
            for (keel::usize i = 0; i < N; ++i) od[i] = I.uint16Data()[i] / 65535.0; break;
        case ValueType::INT16:
            for (keel::usize i = 0; i < N; ++i)
                od[i] = (I.int16Data()[i] + 32768.0) / 65535.0;
            break;
        default:
            throw Error("deconvwnr: unsupported image class",
                        0, 0, "deconvwnr", "", "numkit:deconvwnr:cls");
    }
    return out;
}

Value real_back_to_class(const Value &Jd, ValueType outT, keel::Allocator *mr)
{
    const auto &d = Jd.dims();
    Value out = d.is3D()
        ? Value::matrix3d(d.rows(), d.cols(), d.pages(), outT, mr)
        : Value::matrix(d.rows(), d.cols(), outT, mr);
    const keel::usize N = Jd.numel();
    const keel::f64 *src = Jd.doubleData();
    auto sat_u8  = [](keel::f64 v) {
        if (v <= 0.0) return keel::u8{0};
        if (v >= 1.0) return keel::u8{255};
        return keel::as<keel::u8>(keel::round_i64(v * 255.0));
    };
    auto sat_u16 = [](keel::f64 v) {
        if (v <= 0.0) return keel::u16{0};
        if (v >= 1.0) return keel::u16{65535};
        return keel::as<keel::u16>(keel::round_i64(v * 65535.0));
    };
    auto sat_i16 = [](keel::f64 v) {
        if (v <= 0.0) return keel::i16{-32768};
        if (v >= 1.0) return keel::i16{32767};
        return keel::as<keel::i16>(keel::round_i64(v * 65535.0 - 32768.0));
    };
    switch (outT) {
        case ValueType::DOUBLE:
            keel::copy_n(out.doubleDataMut(), src, N); break;
        case ValueType::SINGLE:
            for (keel::usize i = 0; i < N; ++i)
                out.singleDataMut()[i] = keel::as<keel::f32>(src[i]);
            break;
        case ValueType::UINT8:
            for (keel::usize i = 0; i < N; ++i)
                out.uint8DataMut()[i] = sat_u8(src[i]);
            break;
        case ValueType::UINT16:
            for (keel::usize i = 0; i < N; ++i)
                out.uint16DataMut()[i] = sat_u16(src[i]);
            break;
        case ValueType::INT16:
            for (keel::usize i = 0; i < N; ++i)
                out.int16DataMut()[i] = sat_i16(src[i]);
            break;
        default:
            throw Error("deconvwnr: unsupported output class",
                        0, 0, "deconvwnr", "", "numkit:deconvwnr:cls");
    }
    return out;
}

void deconvwnr_nsr_argcheck(const Value &I, const Value &PSF)
{
    internal::requireImageClass(I, "deconvwnr", "I", 1, internal::ImgClasses::FloatInt);
    internal::requireImageClass(PSF, "deconvwnr", "PSF", 2, internal::ImgClasses::DoubleOnly);
    if (PSF.dims().is3D())
        throw Error("deconvwnr: PSF must be 2-D", 0, 0, "deconvwnr", "", "numkit:deconvwnr:psf");
}

void deconvwnr_acf_argcheck(const Value &PSF)
{
    if (PSF.dims().is3D())
        throw Error("deconvwnr: PSF must be 2-D", 0, 0, "deconvwnr", "", "numkit:deconvwnr:psf");
}

void edgetaper_argcheck(const Value &I, const Value &PSF)
{
    const auto &dI   = I.dims();
    const auto &dPSF = PSF.dims();
    if (dI.is3D())
        throw Error("edgetaper: I must be 2-D (slice 3-D inputs and call per page)",
                    0, 0, "edgetaper", "", "numkit:edgetaper:rank");
    if (dPSF.is3D())
        throw Error("edgetaper: PSF must be 2-D",
                    0, 0, "edgetaper", "", "numkit:edgetaper:psfRank");
    const keel::usize H  = dI.rows(), W  = dI.cols();
    const keel::usize PH = dPSF.rows(), PW = dPSF.cols();
    if (H * W < 2)
        throw Error("edgetaper: I must have at least 2 elements",
                    0, 0, "edgetaper", "", "numkit:edgetaper:tooSmall");
    if (PH * PW < 2)
        throw Error("edgetaper: PSF must have at least 2 elements",
                    0, 0, "edgetaper", "", "numkit:edgetaper:psfTooSmall");
    if (2 * PH > H || 2 * PW > W)
        throw Error("edgetaper: PSF size must be smaller than half of "
                    "the image size in any nonsingleton dimension",
                    0, 0, "edgetaper", "", "numkit:edgetaper:psfSize");
}

void deconvreg_argcheck(const Value &PSF, keel::f64 np, keel::f64 lo, keel::f64 hi)
{
    if (PSF.dims().is3D())
        throw Error("deconvreg: PSF must be 2-D (slice 3-D inputs and call per page)",
                    0, 0, "deconvreg", "", "numkit:deconvreg:psf");
    if (PSF.numel() < 2)
        throw Error("deconvreg: PSF must have at least 2 elements",
                    0, 0, "deconvreg", "", "numkit:deconvreg:psfTooSmall");
    if (!keel::is_finite(np) || np < 0.0)
        throw Error("deconvreg: NP must be a finite non-negative scalar",
                    0, 0, "deconvreg", "", "numkit:deconvreg:np");
    if (!keel::is_finite(lo) || !keel::is_finite(hi) || lo < 0.0 || hi < lo)
        throw Error("deconvreg: LRANGE must be finite, non-negative, and lo <= hi",
                    0, 0, "deconvreg", "", "numkit:deconvreg:lrange");
}

} // anonymous

Value dct2(const Value &A, keel::Allocator *mr)
{
    const keel::usize H = A.dims().rows();
    const keel::usize W = A.dims().cols();
    Value out = Value::matrix(H, W, ValueType::DOUBLE, mr);
    if (H == 0 || W == 0) return out;
    auto in = extract_double_buf(A, mr);
    (void)keel::image::dct_2d(in.as_ref(), H, W,
                              keel::MutArrayRef<keel::f64>(out.doubleDataMut(), H * W), mr);
    return out;
}

Value idct2(const Value &A, keel::Allocator *mr)
{
    const keel::usize H = A.dims().rows();
    const keel::usize W = A.dims().cols();
    Value out = Value::matrix(H, W, ValueType::DOUBLE, mr);
    if (H == 0 || W == 0) return out;
    auto in = extract_double_buf(A, mr);
    (void)keel::image::idct_2d(in.as_ref(), H, W,
                               keel::MutArrayRef<keel::f64>(out.doubleDataMut(), H * W), mr);
    return out;
}

Value dctmtx(keel::f64 Nd, keel::Allocator *mr)
{
    dctmtx_argcheck(Nd);
    const keel::usize N = keel::as<keel::usize>(Nd);
    Value D = Value::matrix(N, N, ValueType::DOUBLE, mr);
    if (N == 0) return D;
    (void)keel::image::dct_matrix(N, keel::MutArrayRef<keel::f64>(D.doubleDataMut(), N * N));
    return D;
}

Value integralImage(const Value &I, keel::Allocator *mr)
{
    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    Value out = Value::matrix(H + 1, W + 1, ValueType::DOUBLE, mr);
    if (H == 0 || W == 0) return out;
    auto in = extract_double_buf(I, mr);
    (void)keel::image::integral_image_2d(
        in.as_ref(), H, W, keel::MutArrayRef<keel::f64>(out.doubleDataMut(), (H + 1) * (W + 1)));
    return out;
}

Value integralImage3(const Value &V, keel::Allocator *mr)
{
    const auto &d = V.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    const keel::usize P = d.is3D() ? d.pages() : 1;
    Value out = Value::matrix3d(H + 1, W + 1, P + 1, ValueType::DOUBLE, mr);
    if (H == 0 || W == 0 || P == 0) return out;
    auto in = extract_double_buf(V, mr);
    (void)keel::image::integral_image_3d(
        in.as_ref(), H, W, P,
        keel::MutArrayRef<keel::f64>(out.doubleDataMut(), (H + 1) * (W + 1) * (P + 1)));
    return out;
}

keel::Tuple<Value, Value>
phantom(const Value &model_or_E, keel::usize n, keel::Allocator *mr)
{
    Value E = phantom_argcheck(model_or_E, mr);
    if (n == 0) n = 256;
    Value head = Value::matrix(n, n, ValueType::DOUBLE, mr);
    const keel::usize nE = E.dims().rows();
    auto ed = extract_double_buf(E, mr);
    (void)keel::image::phantom_from_ellipses(
        ed.as_ref(), nE, n, keel::MutArrayRef<keel::f64>(head.doubleDataMut(), n * n), mr);
    return {keel::move(head), keel::move(E)};
}

Value bestblk(const Value &IMS, keel::f64 k, keel::Allocator *mr)
{
    Value IMSC;
    bestblk_argcheck(IMS, k, IMSC, mr);
    const Value &IMSu = (IMS.type() == ValueType::CHAR) ? IMSC : IMS;

    const keel::usize nd = IMSu.numel();
    const keel::i64 K = keel::as<keel::i64>(keel::floor(k));
    Value out = Value::matrix(1, nd, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();

    for (keel::usize d = 0; d < nd; ++d) {
        const keel::i64 dim = keel::as<keel::i64>(keel::floor(IMSu.elemAsDouble(d)));
        if (dim < 1)
            throw Error("bestblk: IMS entries must be positive",
                        0, 0, "bestblk", "", "numkit:bestblk:vals");
        if (dim <= K) { od[d] = keel::as<keel::f64>(dim); continue; }
        const keel::f64 lo_d = keel::min(keel::as<keel::f64>(dim) / 10.0,
                                         keel::as<keel::f64>(K) / 2.0);
        keel::i64 lo = keel::max<keel::i64>(1, keel::as<keel::i64>(keel::ceil(lo_d)));
        keel::i64 bestP   = K;
        keel::i64 bestPad = ((K - dim % K) % K + K) % K;
        for (keel::i64 p = K - 1; p >= lo; --p) {
            const keel::i64 pad = ((p - dim % p) % p + p) % p;
            if (pad < bestPad) {
                bestPad = pad;
                bestP   = p;
            }
        }
        od[d] = keel::as<keel::f64>(bestP);
    }
    return internal::castResultToClass(out, IMSu.type(), mr);
}

Value fftconv2(const Value &A, const Value &B, keel::StringRef shape, keel::Allocator *mr)
{
    fftconv2_argcheck(A, B);
    const keel::usize ra = A.dims().rows(), ca = A.dims().cols();
    const keel::usize rb = B.dims().rows(), cb = B.dims().cols();
    if (ra == 0 || ca == 0 || rb == 0 || cb == 0)
        return Value::matrix(0, 0, ValueType::COMPLEX, mr);

    const keel::usize Hf = ra + rb - 1;
    const keel::usize Wf = ca + cb - 1;
    auto next_pow2 = [](keel::usize n) {
        keel::usize p = 1;
        while (p < n) p <<= 1;
        return p ? p : 1;
    };
    const keel::usize Hp = next_pow2(Hf);
    const keel::usize Wp = next_pow2(Wf);

    auto pad_post = [&](const Value &X, keel::usize Hpad, keel::usize Wpad) {
        return padarray(X, {(keel::i32)(Hpad - X.dims().rows()), (keel::i32)(Wpad - X.dims().cols())},
                        PadMode::Constant, 0.0, "post", mr);
    };
    Value Ap = pad_post(A, Hp, Wp);
    Value Bp = pad_post(B, Hp, Wp);

    Value FA = signal::fft2(Ap, -1, -1, mr);
    Value FB = signal::fft2(Bp, -1, -1, mr);

    Value FY = Value::matrix(Hp, Wp, ValueType::COMPLEX, mr);
    {
        const keel::c64 *a_ = FA.complexData();
        const keel::c64 *b_ = FB.complexData();
        keel::c64 *y_       = FY.complexDataMut();
        const keel::usize N = Hp * Wp;
        for (keel::usize i = 0; i < N; ++i) y_[i] = a_[i] * b_[i];
    }

    Value Yfull = signal::ifft2(FY, -1, -1, mr);
    Value Y = Value::matrix(Hf, Wf, Yfull.type(), mr);
    if (Yfull.type() == ValueType::COMPLEX) {
        const keel::c64 *src = Yfull.complexData();
        keel::c64 *dst       = Y.complexDataMut();
        for (keel::usize c = 0; c < Wf; ++c)
            for (keel::usize r = 0; r < Hf; ++r)
                dst[c * Hf + r] = src[c * Hp + r];
    } else {
        const keel::f64 *src = Yfull.doubleData();
        keel::f64 *dst       = Y.doubleDataMut();
        for (keel::usize c = 0; c < Wf; ++c)
            for (keel::usize r = 0; r < Hf; ++r)
                dst[c * Hf + r] = src[c * Hp + r];
    }

    keel::String sh;
    sh.reserve(shape.size());
    for (char c : shape) sh.push_back(keel::to_ascii_lower(c));
    if (sh.empty()) sh.assign("full");
    if (sh == "full") return Y;

    keel::usize r0 = 0, c0 = 0, outH = 0, outW = 0;
    if (sh == "same") {
        r0 = rb / 2; c0 = cb / 2; outH = ra; outW = ca;
    } else if (sh == "valid") {
        if (ra < rb || ca < cb) return Value::matrix(0, 0, Y.type(), mr);
        r0 = rb - 1; c0 = cb - 1; outH = ra - rb + 1; outW = ca - cb + 1;
    } else {
        throw Error("fftconv2: shape must be 'full', 'same', or 'valid'",
                    0, 0, "fftconv2", "", "numkit:fftconv2:shape");
    }
    if (outH == 0 || outW == 0) return Value::matrix(0, 0, Y.type(), mr);

    Value Ycrop = Value::matrix(outH, outW, Y.type(), mr);
    if (Y.type() == ValueType::COMPLEX) {
        const keel::c64 *src = Y.complexData();
        keel::c64 *dst       = Ycrop.complexDataMut();
        for (keel::usize c = 0; c < outW; ++c)
            for (keel::usize r = 0; r < outH; ++r)
                dst[c * outH + r] = src[(c + c0) * Hf + (r + r0)];
    } else {
        const keel::f64 *src = Y.doubleData();
        keel::f64 *dst       = Ycrop.doubleDataMut();
        for (keel::usize c = 0; c < outW; ++c)
            for (keel::usize r = 0; r < outH; ++r)
                dst[c * outH + r] = src[(c + c0) * Hf + (r + r0)];
    }
    return Ycrop;
}

Value psf2otf(const Value &PSF, keel::ArrayRef<keel::usize> outsize, keel::Allocator *mr)
{
    keel::usize inH = 0, inW = 0, outH = 0, outW = 0;
    psf2otf_argcheck(PSF, outsize, inH, inW, outH, outW);
    auto in = extract_double_buf(PSF, mr);
    keel::Array<keel::c64> otf(outH * outW, mr);
    (void)keel::image::psf_to_otf(in.as_ref(), inH, inW, outH, outW, otf.as_mut_ref(), mr);
    return complex_or_real_value(otf.as_ref(), outH, outW, mr);
}

Value otf2psf(const Value &OTF, keel::ArrayRef<keel::usize> outsize, keel::Allocator *mr)
{
    keel::usize inH = 0, inW = 0, outH = 0, outW = 0;
    otf2psf_argcheck(OTF, outsize, inH, inW, outH, outW);
    const keel::usize inN = inH * inW;
    keel::Array<keel::c64> in(inN, mr);
    if (OTF.isComplex()) {
        keel::copy_n(in.data(), OTF.complexData(), inN);
    } else {
        for (keel::usize i = 0; i < inN; ++i) in[i] = keel::c64{OTF.elemAsDouble(i), 0.0};
    }
    keel::Array<keel::c64> psf(outH * outW, mr);
    (void)keel::image::otf_to_psf(in.as_ref(), inH, inW, outH, outW, psf.as_mut_ref(), mr);
    return complex_or_real_value(psf.as_ref(), outH, outW, mr);
}

Value normxcorr2(const Value &templ, const Value &img, keel::Allocator *mr)
{
    normxcorr2_argcheck(templ, img);
    const keel::usize mH = templ.dims().rows(), mW = templ.dims().cols();
    const keel::usize bH = img.dims().rows(),   bW = img.dims().cols();
    const keel::usize outH = bH + mH - 1,       outW = bW + mW - 1;
    Value c_out = Value::matrix(outH, outW, ValueType::DOUBLE, mr);
    if (mH == 0 || mW == 0 || bH == 0 || bW == 0) return c_out;
    auto tbuf = extract_double_buf(templ, mr);
    auto ibuf = extract_double_buf(img, mr);
    (void)keel::image::normxcorr2(
        tbuf.as_ref(), mH, mW, ibuf.as_ref(), bH, bW,
        keel::MutArrayRef<keel::f64>(c_out.doubleDataMut(), outH * outW), mr);
    return c_out;
}

Value checkerboard(keel::usize side, keel::usize M, keel::usize N, keel::Allocator *mr)
{
    const keel::usize H = 2 * M * side;
    const keel::usize W = 2 * N * side;
    Value out = Value::matrix(H, W, ValueType::DOUBLE, mr);
    if (H == 0 || W == 0 || side == 0) return out;
    (void)keel::image::checkerboard(side, M, N,
                                    keel::MutArrayRef<keel::f64>(out.doubleDataMut(), H * W), mr);
    return out;
}

Value deconvwnr(const Value &I, const Value &PSF, keel::f64 nsr, keel::Allocator *mr)
{
    deconvwnr_nsr_argcheck(I, PSF);
    const ValueType inT = I.type();
    Value Id = to_double_unit(I, mr);
    Value PSFd = to_double_unit(PSF, mr);

    const auto &dI = Id.dims();
    const keel::usize H = dI.rows(), W = dI.cols();
    const bool is3 = dI.is3D();
    const keel::usize P = is3 ? dI.pages() : 1;
    const keel::usize plane = H * W;

    Value J = is3 ? Value::matrix3d(H, W, P, ValueType::DOUBLE, mr)
                  : Value::matrix(H, W, ValueType::DOUBLE, mr);
    const keel::f64 su1[1] = {nsr};
    const keel::f64 sx1[1] = {1.0};
    const keel::ArrayRef<keel::f64> psf_ref(PSFd.doubleData(), PSFd.numel());
    for (keel::usize pg = 0; pg < P; ++pg) {
        (void)keel::image::deconvwnr_2d(
            keel::ArrayRef<keel::f64>(Id.doubleData() + pg * plane, plane), H, W,
            psf_ref, PSFd.dims().rows(), PSFd.dims().cols(),
            keel::ArrayRef<keel::f64>(su1, 1), keel::ArrayRef<keel::f64>(sx1, 1),
            keel::MutArrayRef<keel::f64>(J.doubleDataMut() + pg * plane, plane), mr);
    }
    return (inT == ValueType::DOUBLE) ? J : real_back_to_class(J, inT, mr);
}

Value deconvwnr(const Value &I, const Value &PSF,
                const Value &ncorr, const Value &icorr, keel::Allocator *mr)
{
    deconvwnr_acf_argcheck(PSF);
    const ValueType inT = I.type();
    Value Id = to_double_unit(I, mr);
    Value PSFd = to_double_unit(PSF, mr);

    const auto &dI = Id.dims();
    const keel::usize H = dI.rows(), W = dI.cols();
    const bool is3 = dI.is3D();
    const keel::usize P = is3 ? dI.pages() : 1;
    const keel::usize plane = H * W;

    auto extract_acf = [&](const Value &acf, const char *name) -> keel::Array<keel::f64> {
        if (acf.numel() == 1) {
            keel::Array<keel::f64> v(mr);
            v.push_back(acf.toScalar());
            return v;
        }
        if (acf.dims().rows() == H && acf.dims().cols() == W && !acf.dims().is3D()) {
            return extract_double_buf(acf, mr);
        }
        throw Error(keel::format("deconvwnr: {} must be a scalar or same size as I "
                                 "(1-D extrapolation form not supported)", name),
                    0, 0, "deconvwnr", "", "numkit:deconvwnr:acf");
    };

    auto su = extract_acf(ncorr, "NCORR");
    auto sx = extract_acf(icorr, "ICORR");

    Value J = is3 ? Value::matrix3d(H, W, P, ValueType::DOUBLE, mr)
                  : Value::matrix(H, W, ValueType::DOUBLE, mr);
    const keel::ArrayRef<keel::f64> psf_ref(PSFd.doubleData(), PSFd.numel());
    for (keel::usize pg = 0; pg < P; ++pg) {
        (void)keel::image::deconvwnr_2d(
            keel::ArrayRef<keel::f64>(Id.doubleData() + pg * plane, plane), H, W,
            psf_ref, PSFd.dims().rows(), PSFd.dims().cols(),
            su.as_ref(), sx.as_ref(),
            keel::MutArrayRef<keel::f64>(J.doubleDataMut() + pg * plane, plane), mr);
    }
    return (inT == ValueType::DOUBLE) ? J : real_back_to_class(J, inT, mr);
}

Value edgetaper(const Value &I, const Value &PSF, keel::Allocator *mr)
{
    edgetaper_argcheck(I, PSF);
    const ValueType inT = I.type();
    Value Id = to_double_unit(I, mr);
    Value PSFd = to_double_unit(PSF, mr);
    const keel::usize H = Id.dims().rows(), W = Id.dims().cols();
    Value Jd = Value::matrix(H, W, ValueType::DOUBLE, mr);
    (void)keel::image::edgetaper_2d(
        keel::ArrayRef<keel::f64>(Id.doubleData(), H * W), H, W,
        keel::ArrayRef<keel::f64>(PSFd.doubleData(), PSFd.numel()),
        PSFd.dims().rows(), PSFd.dims().cols(),
        keel::MutArrayRef<keel::f64>(Jd.doubleDataMut(), H * W), mr);
    if (inT == ValueType::DOUBLE) return Jd;
    return real_back_to_class(Jd, inT, mr);
}

DeconvregResult deconvreg(const Value &I, const Value &PSF,
                          keel::f64 np, keel::f64 lo, keel::f64 hi,
                          const Value &regop, keel::Allocator *mr)
{
    deconvreg_argcheck(PSF, np, lo, hi);
    const ValueType inT = I.type();
    Value Id = to_double_unit(I, mr);
    Value PSFd = to_double_unit(PSF, mr);

    const keel::usize H = Id.dims().rows(), W = Id.dims().cols();
    if (H < 3 || W < 3)
        throw Error("deconvreg: image too small (min 3x3)",
                    0, 0, "deconvreg", "", "numkit:deconvreg:tooSmall");

    bool allzero = true;
    for (keel::usize i = 0; i < PSFd.numel(); ++i)
        if (PSFd.doubleData()[i] != 0.0) { allzero = false; break; }
    if (allzero)
        throw Error("deconvreg: PSF cannot be all zeros",
                    0, 0, "deconvreg", "", "numkit:deconvreg:psfAllZero");

    Value regop_d;
    keel::ArrayRef<keel::f64> regop_ref{};
    keel::usize rh = 0, rw = 0;
    if (!regop.isEmpty()) {
        if (regop.dims().is3D())
            throw Error("deconvreg: REGOP must be 2-D",
                        0, 0, "deconvreg", "", "numkit:deconvreg:regop");
        regop_d = to_double_unit(regop, mr);
        rh = regop_d.dims().rows();
        rw = regop_d.dims().cols();
        regop_ref = keel::ArrayRef<keel::f64>(regop_d.doubleData(), regop_d.numel());
    }

    Value Jd = Value::matrix(H, W, ValueType::DOUBLE, mr);
    keel::f64 lagra = 0.0;
    (void)keel::image::deconvreg_2d(
        keel::ArrayRef<keel::f64>(Id.doubleData(), H * W), H, W,
        keel::ArrayRef<keel::f64>(PSFd.doubleData(), PSFd.numel()),
        PSFd.dims().rows(), PSFd.dims().cols(),
        np, lo, hi, regop_ref, rh, rw,
        keel::MutArrayRef<keel::f64>(Jd.doubleDataMut(), H * W), lagra, mr);

    Value Jout = (inT == ValueType::DOUBLE) ? Jd : real_back_to_class(Jd, inT, mr);
    return DeconvregResult{keel::move(Jout), lagra};
}

} // namespace numkit::image
