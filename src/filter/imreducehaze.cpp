// toolboxes/image/src/filter/imreducehaze.cpp
//
// Single-image dehazing via dark-channel prior, delegating numerical kernels
// to keel::image::dehaze.

#include <keel/arena.h>
#include <keel/algo.h>
#include <keel/math.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/image.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>

#include <numkit/image/filter/filter.hpp>
#include <numkit/image/type_convert/type_convert.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>


namespace numkit::image {
namespace {

inline keel::String lower_str(keel::StringRef s)
{
    keel::String lo;
    lo.reserve(s.size());
    for (char ch : s) lo.push_back(keel::to_ascii_lower(ch));
    return lo;
}

void to_single01_buf(const Value &A, keel::Array<keel::f32> &out)
{
    const keel::usize total = A.numel();
    out.resize(total);
    const ValueType t = A.type();
    if (t == ValueType::SINGLE) {
        const keel::f32 *src = A.singleData();
        for (keel::usize i = 0; i < total; ++i) {
            keel::f32 v = src[i];
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            out[i] = v;
        }
        return;
    }
    for (keel::usize i = 0; i < total; ++i) {
        const keel::f64 raw = A.elemAsDouble(i);
        keel::f32 v = 0.0f;
        switch (t) {
            case ValueType::UINT8:   v = keel::as<keel::f32>(raw) / 255.0f; break;
            case ValueType::UINT16:  v = keel::as<keel::f32>(raw) / 65535.0f; break;
            case ValueType::INT8:    v = (keel::as<keel::f32>(raw) + 128.0f) / 255.0f; break;
            case ValueType::INT16:   v = (keel::as<keel::f32>(raw) + 32768.0f) / 65535.0f; break;
            case ValueType::LOGICAL: v = (raw == 0.0) ? 0.0f : 1.0f; break;
            default:
                v = keel::as<keel::f32>(raw);
                if (v < 0.0f) v = 0.0f;
                if (v > 1.0f) v = 1.0f;
                break;
        }
        out[i] = v;
    }
}

Value cast_back_f32(keel::ArrayRef<keel::f32> buf, keel::usize H, keel::usize W, keel::usize P,
                    bool is3D, ValueType origClass, keel::Allocator *mr)
{
    const keel::usize N = buf.size();
    if (origClass == ValueType::SINGLE) {
        Value out = is3D ? Value::matrix3d(H, W, P, ValueType::SINGLE, mr)
                         : Value::matrix(H, W, ValueType::SINGLE, mr);
        keel::f32 *dst = out.singleDataMut();
        for (keel::usize i = 0; i < N; ++i) {
            keel::f32 v = buf[i];
            if (v < 0.0f) v = 0.0f;
            if (v > 1.0f) v = 1.0f;
            dst[i] = v;
        }
        return out;
    }
    if (origClass == ValueType::DOUBLE) {
        Value out = is3D ? Value::matrix3d(H, W, P, ValueType::DOUBLE, mr)
                         : Value::matrix(H, W, ValueType::DOUBLE, mr);
        keel::f64 *dst = out.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i) {
            keel::f64 v = keel::as<keel::f64>(buf[i]);
            if (v < 0.0) v = 0.0;
            if (v > 1.0) v = 1.0;
            dst[i] = v;
        }
        return out;
    }
    Value tmp = is3D ? Value::matrix3d(H, W, P, ValueType::SINGLE, mr)
                     : Value::matrix(H, W, ValueType::SINGLE, mr);
    keel::copy_n(tmp.singleDataMut(), buf.data(), N);
    if (origClass == ValueType::UINT16) return im2uint16(tmp, mr);
    return im2uint8(tmp, mr);
}

void imreducehaze_argcheck(const Value &I, keel::f64 amount,
                           keel::StringRef method,
                           keel::StringRef contrast_enhancement,
                           keel::f64 boost_amount,
                           keel::String &m_out, keel::String &ce_out,
                           bool &isRGB_out)
{
    if (!keel::is_finite(amount) || amount < 0.0 || amount > 1.0)
        throw Error("imreducehaze: amount must be in [0, 1]",
                    0, 0, "imreducehaze", "", "numkit:imreducehaze:amount");
    const auto &d = I.dims();
    if (d.rows() == 0 || d.cols() == 0)
        throw Error("imreducehaze: I must be non-empty",
                    0, 0, "imreducehaze", "", "numkit:imreducehaze:empty");
    const bool isRGB = d.is3D() && d.pages() == 3;
    const bool isGray = !d.is3D() || (d.is3D() && d.pages() == 1);
    if (!isRGB && !isGray)
        throw Error("imreducehaze: I must be H×W (grayscale) or H×W×3 (RGB)",
                    0, 0, "imreducehaze", "", "numkit:imreducehaze:shape");
    const ValueType origClass = I.type();
    if (origClass != ValueType::UINT8 && origClass != ValueType::UINT16
        && origClass != ValueType::SINGLE && origClass != ValueType::DOUBLE)
        throw Error("imreducehaze: I must be uint8 / uint16 / single / double",
                    0, 0, "imreducehaze", "", "numkit:imreducehaze:class");

    keel::String m = lower_str(method);
    if (m != "simpledcp" && m != "approxdcp")
        throw Error("imreducehaze: Method must be 'simpledcp' or 'approxdcp'",
                    0, 0, "imreducehaze", "", "numkit:imreducehaze:method");
    keel::String ce = lower_str(contrast_enhancement);
    if (ce != "global" && ce != "boost" && ce != "none")
        throw Error("imreducehaze: ContrastEnhancement must be 'global', "
                    "'boost', or 'none'",
                    0, 0, "imreducehaze", "",
                    "numkit:imreducehaze:contrastEnhancement");
    if (ce == "boost" && (boost_amount < 0.0 || boost_amount > 1.0
                          || !keel::is_finite(boost_amount)))
        throw Error("imreducehaze: BoostAmount must be in [0, 1]",
                    0, 0, "imreducehaze", "",
                    "numkit:imreducehaze:boostAmount");
    m_out = keel::move(m);
    ce_out = keel::move(ce);
    isRGB_out = isRGB;
}

} // namespace

Value imreducehaze(const Value &I, keel::f64 amount,
                   keel::StringRef method,
                   const Value &atmospheric_light,
                   keel::StringRef contrast_enhancement,
                   keel::f64 boost_amount,
                   Value &t_out, Value &L_out,
                   keel::Allocator *mr)
{
    keel::String m, ce;
    bool isRGB = false;
    imreducehaze_argcheck(I, amount, method, contrast_enhancement, boost_amount,
                          m, ce, isRGB);
    const ValueType origClass = I.type();

    if (amount == 0.0) {
        t_out = Value::Empty;
        L_out = Value::Empty;
        return I;
    }

    const auto &d = I.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    const keel::usize P = d.is3D() ? d.pages() : 1;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f32> atm_override(scratch.allocator());
    if (!atmospheric_light.isEmpty()) {
        const keel::usize N = atmospheric_light.numel();
        if (isRGB && N != 3)
            throw Error("imreducehaze: AtmosphericLight must be a 3-element "
                        "vector for RGB input",
                        0, 0, "imreducehaze", "",
                        "numkit:imreducehaze:atmRGB");
        if (!isRGB && N != 1)
            throw Error("imreducehaze: AtmosphericLight must be a scalar for "
                        "grayscale input",
                        0, 0, "imreducehaze", "",
                        "numkit:imreducehaze:atmGray");
        atm_override.resize(P);
        for (keel::usize p = 0; p < P; ++p)
            atm_override[p] = keel::as<keel::f32>(
                atmospheric_light.elemAsDouble(p >= N ? N - 1 : p));
    }

    keel::Array<keel::f32> in_buf(scratch.allocator());
    to_single01_buf(I, in_buf);
    keel::Array<keel::f32> out_buf(H * W * P, 0.0f, scratch.allocator());

    t_out = Value::matrix(H, W, ValueType::DOUBLE, mr);
    L_out = Value::matrix(1, P, ValueType::DOUBLE, mr);

    const auto dm = (m == "simpledcp") ? keel::image::DehazeMethod::kSimpleDcp
                                       : keel::image::DehazeMethod::kApproxDcp;
    const auto dc = (ce == "global") ? keel::image::DehazeContrast::kGlobal
                  : (ce == "boost")  ? keel::image::DehazeContrast::kBoost
                                     : keel::image::DehazeContrast::kNone;

    (void)keel::image::dehaze(
        in_buf.as_ref(), H, W, P, amount, dm, atm_override.as_ref(), dc, boost_amount,
        out_buf.as_mut_ref(),
        keel::MutArrayRef<keel::f64>(t_out.doubleDataMut(), H * W),
        keel::MutArrayRef<keel::f64>(L_out.doubleDataMut(), P),
        scratch.allocator());

    return cast_back_f32(out_buf.as_ref(), H, W, P, d.is3D(), origClass, mr);
}

} // namespace numkit::image
