// toolboxes/image/src/filter/fir2d.cpp
//
// 2-D FIR filter design family (fsamp2 / ftrans2 / fwind1 / fwind2),
// delegating numerical kernels to keel::image.

#include <keel/arena.h>
#include <keel/algo.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/image.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/ref.h>

#include <numkit/image/filter/filter.hpp>
#include <numkit/image/internal/type_gates.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>


namespace numkit::image {
namespace {

void copyToF64(const Value &v, keel::Array<keel::f64> &out)
{
    const keel::usize n = v.numel();
    out.resize(n);
    for (keel::usize i = 0; i < n; ++i) out[i] = v.elemAsDouble(i);
}

void copyToC64(const Value &v, keel::Array<keel::c64> &out)
{
    const keel::usize n = v.numel();
    out.resize(n);
    if (v.isComplex()) {
        const auto *cd = v.complexData();
        for (keel::usize i = 0; i < n; ++i) out[i] = cd[i];
    } else {
        for (keel::usize i = 0; i < n; ++i) out[i] = keel::c64(v.elemAsDouble(i), 0.0);
    }
}

void fsamp2_argcheck(const Value &Hd, const Value &f1, const Value &f2)
{
    internal::requireImageClassAllowComplex(Hd, "fsamp2", "Hd", 1,
                                            internal::ImgClasses::FloatIntLog);
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
}

void ftrans2_argcheck(const Value &b)
{
    const keel::usize L = b.numel();
    if (L == 0)
        throw Error("ftrans2: b must be nonempty",
                    0, 0, "ftrans2", "", "numkit:ftrans2:bEmpty");
    if (L % 2 == 0)
        throw Error("ftrans2: b must be odd-length",
                    0, 0, "ftrans2", "", "numkit:ftrans2:bLen");
}

void fwind1_argcheck(const Value &Hd, const Value &win1, const Value &win2)
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
}

void fwind2_argcheck(const Value &Hd, const Value &W)
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
}

} // namespace

Value fsamp2(const Value &HdIn, const Value &f1, const Value &f2,
             keel::ArrayRef<keel::usize> /*siz*/,
             keel::Allocator *mr)
{
    Value Hd0 = (HdIn.type() == ValueType::CHAR)
        ? internal::coerceToDoubleImage(HdIn, mr) : Value(HdIn);
    fsamp2_argcheck(Hd0, f1, f2);

    const keel::usize H = Hd0.dims().rows();
    const keel::usize W = Hd0.dims().cols();
    keel::ArenaTemp scratch = keel::get_scratch(mr);

    if (Hd0.isComplex()) {
        keel::Array<keel::c64> hd(scratch.allocator());
        copyToC64(Hd0, hd);
        Value out = Value::matrix(H, W, ValueType::COMPLEX, mr);
        (void)keel::image::fsamp2_complex(hd.as_ref(), H, W,
                                          keel::MutArrayRef<keel::c64>(out.complexDataMut(), H * W),
                                          scratch.allocator());
        return out;
    }

    keel::Array<keel::f64> hd(scratch.allocator());
    copyToF64(Hd0, hd);
    Value out = Value::matrix(H, W, ValueType::DOUBLE, mr);
    (void)keel::image::fsamp2_real(hd.as_ref(), H, W,
                                   keel::MutArrayRef<keel::f64>(out.doubleDataMut(), H * W),
                                   scratch.allocator());
    return internal::castResultToClass(
        out,
        Hd0.type() == ValueType::SINGLE ? ValueType::SINGLE : ValueType::DOUBLE,
        mr);
}

Value ftrans2(const Value &b_in, const Value &t_in, keel::Allocator *mr)
{
    ftrans2_argcheck(b_in);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> b(scratch.allocator());
    copyToF64(b_in, b);
    const keel::usize L = b.size();

    bool all_zero = true;
    for (keel::usize i = 0; i < L; ++i)
        if (b[i] != 0.0) { all_zero = false; break; }
    if (all_zero)
        throw Error("ftrans2: b must have at least one nonzero element",
                    0, 0, "ftrans2", "", "numkit:ftrans2:bZero");

    const keel::f64 sqrt_eps = keel::sqrt(keel::Limits<keel::f64>::kEpsilon);
    for (keel::usize i = 0; i < L; ++i) {
        if (keel::abs(b[i] - b[L - 1 - i]) > sqrt_eps)
            throw Error("ftrans2: b must be symmetric",
                        0, 0, "ftrans2", "", "numkit:ftrans2:bSym");
    }

    keel::Array<keel::f64> t(scratch.allocator());
    keel::usize mt = 0, nt = 0;
    if (!t_in.isEmpty()) {
        copyToF64(t_in, t);
        mt = t_in.dims().rows();
        nt = t_in.dims().cols();
        bool tzero = true;
        for (keel::usize i = 0; i < t.size(); ++i)
            if (t[i] != 0.0) { tzero = false; break; }
        if (tzero)
            throw Error("ftrans2: t must have at least one nonzero element",
                        0, 0, "ftrans2", "", "numkit:ftrans2:tZero");
    }

    keel::Array<keel::f64> h_buf(scratch.allocator());
    keel::usize oh = 0, ow = 0;
    (void)keel::image::ftrans2(b.as_ref(), t.as_ref(), mt, nt,
                               h_buf, oh, ow, scratch.allocator());
    Value out = Value::matrix(oh, ow, ValueType::DOUBLE, mr);
    keel::copy_n(out.doubleDataMut(), h_buf.data(), oh * ow);
    return out;
}

Value fwind1(const Value &Hd, const Value &win1, const Value &win2,
             keel::Allocator *mr)
{
    fwind1_argcheck(Hd, win1, win2);

    const keel::usize hh = Hd.dims().rows();
    const keel::usize wh = Hd.dims().cols();
    const keel::usize n = win1.numel();
    const keel::usize m_out = win2.isEmpty() ? n : win2.numel();
    const keel::usize n_out = n;
    const bool anyComplex = Hd.isComplex() || win1.isComplex()
                         || (!win2.isEmpty() && win2.isComplex());
    keel::ArenaTemp scratch = keel::get_scratch(mr);

    if (anyComplex) {
        keel::Array<keel::c64> hd(scratch.allocator()), w1(scratch.allocator()), w2(scratch.allocator());
        copyToC64(Hd, hd);
        copyToC64(win1, w1);
        if (!win2.isEmpty()) copyToC64(win2, w2);
        Value out = Value::matrix(m_out, n_out, ValueType::COMPLEX, mr);
        const auto st = keel::image::fwind1_complex(
            hd.as_ref(), hh, wh, w1.as_ref(), w2.as_ref(),
            keel::MutArrayRef<keel::c64>(out.complexDataMut(), m_out * n_out),
            scratch.allocator());
        if (st != keel::image::ImageStatus::kOk)
            throw Error("fwind1: 1-D window must be symmetric",
                        0, 0, "fwind1", "", "numkit:fwind1:winSym");
        return out;
    }

    keel::Array<keel::f64> hd(scratch.allocator()), w1(scratch.allocator()), w2(scratch.allocator());
    copyToF64(Hd, hd);
    copyToF64(win1, w1);
    if (!win2.isEmpty()) copyToF64(win2, w2);
    Value out = Value::matrix(m_out, n_out, ValueType::DOUBLE, mr);
    const auto st = keel::image::fwind1_real(
        hd.as_ref(), hh, wh, w1.as_ref(), w2.as_ref(),
        keel::MutArrayRef<keel::f64>(out.doubleDataMut(), m_out * n_out),
        scratch.allocator());
    if (st != keel::image::ImageStatus::kOk)
        throw Error("fwind1: 1-D window must be symmetric",
                    0, 0, "fwind1", "", "numkit:fwind1:winSym");
    return out;
}

Value fwind2(const Value &Hd, const Value &W, keel::Allocator *mr)
{
    fwind2_argcheck(Hd, W);

    const keel::usize hh = Hd.dims().rows();
    const keel::usize wh = Hd.dims().cols();
    const keel::usize hw = W.dims().rows();
    const keel::usize ww = W.dims().cols();
    const bool is1d = (hw == 1 || ww == 1);
    const keel::usize mw = is1d ? W.numel() : hw;
    const keel::usize nw = is1d ? W.numel() : ww;
    const bool anyComplex = Hd.isComplex() || W.isComplex();
    keel::ArenaTemp scratch = keel::get_scratch(mr);

    if (anyComplex) {
        keel::Array<keel::c64> hd(scratch.allocator()), w(scratch.allocator());
        copyToC64(Hd, hd);
        copyToC64(W, w);
        Value out = Value::matrix(mw, nw, ValueType::COMPLEX, mr);
        (void)keel::image::fwind2_complex(
            hd.as_ref(), hh, wh, w.as_ref(), hw, ww,
            keel::MutArrayRef<keel::c64>(out.complexDataMut(), mw * nw),
            scratch.allocator());
        return out;
    }

    keel::Array<keel::f64> hd(scratch.allocator()), w(scratch.allocator());
    copyToF64(Hd, hd);
    copyToF64(W, w);
    Value out = Value::matrix(mw, nw, ValueType::DOUBLE, mr);
    (void)keel::image::fwind2_real(
        hd.as_ref(), hh, wh, w.as_ref(), hw, ww,
        keel::MutArrayRef<keel::f64>(out.doubleDataMut(), mw * nw),
        scratch.allocator());
    if (Hd.type() == ValueType::SINGLE || W.type() == ValueType::SINGLE)
        return internal::castResultToClass(out, ValueType::SINGLE, mr);
    return out;
}

} // namespace numkit::image
