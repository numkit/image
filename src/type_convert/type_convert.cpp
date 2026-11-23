// toolboxes/image/src/type_convert/type_convert.cpp

#include <keel/core.h>
#include <keel/math.h>
#include <keel/format.h>
#include <keel/image.h>
#include <keel/memory.h>
#include <keel/array.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/image/type_convert/type_convert.hpp>
#include <numkit/image/internal/type_gates.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <keel/tuple.h>

namespace numkit::image {

namespace {

template <typename T>
inline T satCast(keel::f64 v) {
    if (keel::is_nan(v)) return T{};
    constexpr keel::f64 lo = keel::as<keel::f64>(keel::Limits<T>::kLowest);
    constexpr keel::f64 hi = keel::as<keel::f64>(keel::Limits<T>::kMax);
    if (v <= lo) return keel::Limits<T>::kLowest;
    if (v >= hi) return keel::Limits<T>::kMax;
    return static_cast<T>(keel::round_i64(v));
}

inline Value alloc_like(const Value &x, ValueType t, keel::Allocator *mr) {
    const auto &d = x.dims();
    if (x.isScalar()) return Value::scalar(0.0, mr);
    if (d.is3D()) return Value::matrix3d(d.rows(), d.cols(), d.pages(), t, mr);
    return Value::matrix(d.rows(), d.cols(), t, mr);
}

// Convert one element of source type to a value in [0, 1] (double).
inline keel::f64 element_to_unit(const Value &x, keel::usize i) {
    const keel::f64 v = x.elemAsDouble(i);
    switch (x.type()) {
        case ValueType::DOUBLE:
        case ValueType::SINGLE:  return v;                       // already [0,1]
        case ValueType::UINT8:   return v / 255.0;
        case ValueType::UINT16:  return v / 65535.0;
        case ValueType::INT16:   return (v + 32768.0) / 65535.0;
        case ValueType::LOGICAL: return v != 0.0 ? 1.0 : 0.0;
        default:                 return v;
    }
}

// Generic float-output helper.
template <typename Float>
Value to_float_impl(const Value &x, ValueType target, keel::Allocator *mr) {
    const keel::usize n = x.numel();
    Value out;
    if (x.isScalar()) {
        out = Value::matrix(1, 1, target, mr);
    } else {
        out = alloc_like(x, target, mr);
    }
    if (n == 0) return out;

    Float *od = nullptr;
    if constexpr (keel::IsSame<Float, keel::f64>) od = out.doubleDataMut();
    else                                          od = out.singleDataMut();

    // Typed raw-pointer paths (no per-element element_to_unit/elemAsDouble
    // dispatch) — contiguous inner loops auto-vectorise. Divisors match
    // element_to_unit exactly (÷255 / ÷65535, not ×reciprocal) to stay
    // bit-for-bit identical with the generic path.
    if (!x.isComplex()) {
        switch (x.type()) {
            case ValueType::DOUBLE: {
                const keel::f64 *s = x.doubleData();
                for (keel::usize i = 0; i < n; ++i) od[i] = static_cast<Float>(s[i]);
                return out;
            }
            case ValueType::SINGLE: {
                const keel::f32 *s = x.singleData();
                for (keel::usize i = 0; i < n; ++i) od[i] = static_cast<Float>(s[i]);
                return out;
            }
            case ValueType::UINT8: {
                const keel::u8 *s = x.uint8Data();
                for (keel::usize i = 0; i < n; ++i) od[i] = static_cast<Float>(s[i] / 255.0);
                return out;
            }
            case ValueType::UINT16: {
                const keel::u16 *s = x.uint16Data();
                for (keel::usize i = 0; i < n; ++i) od[i] = static_cast<Float>(s[i] / 65535.0);
                return out;
            }
            case ValueType::INT16: {
                const keel::i16 *s = x.int16Data();
                for (keel::usize i = 0; i < n; ++i)
                    od[i] = static_cast<Float>((s[i] + 32768.0) / 65535.0);
                return out;
            }
            case ValueType::LOGICAL: {
                const keel::u8 *s = x.logicalData();
                for (keel::usize i = 0; i < n; ++i) od[i] = s[i] ? Float(1) : Float(0);
                return out;
            }
            default: break;
        }
    }
    for (keel::usize i = 0; i < n; ++i)
        od[i] = static_cast<Float>(element_to_unit(x, i));
    return out;
}

// Saturating int-output. Float source: round(unit · scale) + bias_after_round
// (MATLAB rounds the un-shifted product first, then subtracts the bias). Using
// the shifted intermediate would corrupt half-integer cases (e.g. 0.5*65535
// → 32767.5; rounded gives 32768, then -32768 → 0; the alternative
// rounds (32767.5 - 32768) = -0.5 → -1, which is wrong).
template <typename Int>
Value to_int_impl(const Value &x, ValueType target, keel::f64 scale, keel::f64 bias_after, keel::Allocator *mr) {
    const keel::usize n = x.numel();
    Value out;
    if (x.isScalar()) out = Value::matrix(1, 1, target, mr);
    else              out = alloc_like(x, target, mr);
    if (n == 0) return out;

    Int *od = nullptr;
    if      constexpr (keel::IsSame<Int, keel::u8>)  od = out.uint8DataMut();
    else if constexpr (keel::IsSame<Int, keel::u16>) od = out.uint16DataMut();
    else if constexpr (keel::IsSame<Int, keel::i16>)  od = out.int16DataMut();
    else                                              od = nullptr;

    // Float → int: scale unit value by `scale` then round, plus optional bias.
    // Int → int: use proportional rescale (handled per-source below for accuracy).
    //
    // The float/logical path runs a branchless `convert` lambda over a typed raw
    // pointer — no per-element element_to_unit/elemAsDouble dispatch and no
    // std::lround (errno + rounding-mode overhead, does not vectorise). For
    // u·scale ≥ 0, round-half-away-from-zero == trunc(u·scale + 0.5); the clamp
    // forces u ∈ [0,1] first, so this is bit-exact with the old lround + satCast
    // (incl. NaN → 0). The clean contiguous loop auto-vectorises under AVX2.
    const keel::f64 sc = scale, bi = bias_after;
    auto convert = [sc, bi](keel::f64 v) -> Int {
        constexpr keel::f64 clamp_lo =
            keel::as<keel::f64>(keel::Limits<Int>::kLowest);
        constexpr keel::f64 clamp_hi =
            keel::as<keel::f64>(keel::Limits<Int>::kMax);
        const bool nan = !(v == v);
        keel::f64 u = nan ? 0.0 : v;
        if (u < 0.0) u = 0.0;
        else if (u > 1.0) u = 1.0;
        keel::f64 t = keel::trunc(u * sc + 0.5) + bi;
        if (t < clamp_lo) t = clamp_lo;
        else if (t > clamp_hi) t = clamp_hi;
        return nan ? Int{} : static_cast<Int>(t);
    };
    switch (x.type()) {
        case ValueType::DOUBLE:
        case ValueType::SINGLE:
        case ValueType::LOGICAL: {
            if (x.type() == ValueType::DOUBLE && !x.isComplex()) {
                const keel::f64 *s = x.doubleData();
                for (keel::usize i = 0; i < n; ++i) od[i] = convert(s[i]);
            } else if (x.type() == ValueType::SINGLE && !x.isComplex()) {
                const keel::f32 *s = x.singleData();
                for (keel::usize i = 0; i < n; ++i)
                    od[i] = convert(keel::as<keel::f64>(s[i]));
            } else if (x.type() == ValueType::LOGICAL) {
                const keel::u8 *s = x.logicalData();
                for (keel::usize i = 0; i < n; ++i) od[i] = convert(s[i] ? 1.0 : 0.0);
            } else {  // complex DOUBLE/SINGLE — keep real-part dispatch
                for (keel::usize i = 0; i < n; ++i)
                    od[i] = convert(element_to_unit(x, i));
            }
            break;
        }
        case ValueType::UINT8: {
            const keel::u8 *src = x.uint8Data();
            if constexpr (keel::IsSame<Int, keel::u8>) {
                for (keel::usize i = 0; i < n; ++i) od[i] = src[i];
            } else if constexpr (keel::IsSame<Int, keel::u16>) {
                // 0xAB → 0xABAB (bit-replication).
                for (keel::usize i = 0; i < n; ++i) od[i] = (keel::u16)((src[i] << 8) | src[i]);
            } else { // int16
                for (keel::usize i = 0; i < n; ++i) {
                    const keel::u16 u = (keel::u16)((src[i] << 8) | src[i]);
                    od[i] = satCast<keel::i16>((keel::f64)u - 32768.0);
                }
            }
            break;
        }
        case ValueType::UINT16: {
            const keel::u16 *src = x.uint16Data();
            if constexpr (keel::IsSame<Int, keel::u16>) {
                for (keel::usize i = 0; i < n; ++i) od[i] = src[i];
            } else if constexpr (keel::IsSame<Int, keel::u8>) {
                // Round high byte: round(src/257).
                for (keel::usize i = 0; i < n; ++i)
                    od[i] = satCast<keel::u8>((keel::f64)src[i] / 257.0);
            } else { // int16
                for (keel::usize i = 0; i < n; ++i)
                    od[i] = satCast<keel::i16>((keel::f64)src[i] - 32768.0);
            }
            break;
        }
        case ValueType::INT16: {
            const keel::i16 *src = x.int16Data();
            if constexpr (keel::IsSame<Int, keel::i16>) {
                for (keel::usize i = 0; i < n; ++i) od[i] = src[i];
            } else if constexpr (keel::IsSame<Int, keel::u16>) {
                for (keel::usize i = 0; i < n; ++i)
                    od[i] = satCast<keel::u16>((keel::f64)src[i] + 32768.0);
            } else { // uint8
                for (keel::usize i = 0; i < n; ++i) {
                    const keel::f64 u16 = (keel::f64)src[i] + 32768.0;
                    od[i] = satCast<keel::u8>(u16 / 257.0);
                }
            }
            break;
        }
        default:
            throw Error("im2int: unsupported source class", 0, 0, "im2int", "",
                        "numkit:im2int:badtype");
    }
    return out;
}

void rgb2gray_argcheck(const Value &x)
{
    const auto &d = x.dims();
    if (!d.is3D() || d.pages() != 3)
        throw Error("rgb2gray: input must be H×W×3", 0, 0, "rgb2gray", "",
                    "numkit:rgb2gray:size");
}

} // anonymous

Value im2double(const Value &x, keel::Allocator *mr) {
    if (x.type() == ValueType::DOUBLE) return x;
    return to_float_impl<keel::f64>(x, ValueType::DOUBLE, mr);
}

Value im2single(const Value &x, keel::Allocator *mr) {
    if (x.type() == ValueType::SINGLE) return x;
    return to_float_impl<keel::f32>(x, ValueType::SINGLE, mr);
}

Value im2uint8(const Value &x, keel::Allocator *mr) {
    if (x.type() == ValueType::UINT8) return x;
    return to_int_impl<keel::u8>(x, ValueType::UINT8, 255.0, 0.0, mr);
}

Value im2uint16(const Value &x, keel::Allocator *mr) {
    if (x.type() == ValueType::UINT16) return x;
    return to_int_impl<keel::u16>(x, ValueType::UINT16, 65535.0, 0.0, mr);
}

Value im2int16(const Value &x, keel::Allocator *mr) {
    if (x.type() == ValueType::INT16) return x;
    // [0, 1] float → [-32768, 32767]: round(x*65535) - 32768.
    return to_int_impl<keel::i16>(x, ValueType::INT16, 65535.0, -32768.0, mr);
}

Value mat2gray(const Value &x, keel::f64 lo, keel::f64 hi, keel::Allocator *mr)
{
    const keel::usize n = x.numel();
    if (keel::is_nan(lo) || keel::is_nan(hi)) {
        // Auto-detect range, ignoring NaNs.
        lo =  keel::Limits<keel::f64>::infinity();
        hi = -keel::Limits<keel::f64>::infinity();
        for (keel::usize i = 0; i < n; ++i) {
            const keel::f64 v = x.elemAsDouble(i);
            if (keel::is_nan(v)) continue;
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        if (!keel::is_finite(lo) || !keel::is_finite(hi)) {
            lo = 0.0; hi = 0.0;
        }
    }
    Value out = alloc_like(x, ValueType::DOUBLE, mr);
    if (x.isScalar()) out = Value::matrix(1, 1, ValueType::DOUBLE, mr);
    if (n == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    if (lo == hi) {
        // MATLAB sets all entries ≤ lo to 0 and others to 1; for uniform
        // input that's all-0 unless any element exceeds hi (no element does).
        for (keel::usize i = 0; i < n; ++i) od[i] = 0.0;
        return out;
    }
    const keel::f64 inv = 1.0 / (hi - lo);
    for (keel::usize i = 0; i < n; ++i) {
        keel::f64 v = (x.elemAsDouble(i) - lo) * inv;
        if (v < 0.0) v = 0.0;
        if (v > 1.0) v = 1.0;
        od[i] = v;
    }
    return out;
}

Value rgb2gray(const Value &x, keel::Allocator *mr)
{
    rgb2gray_argcheck(x);
    const auto &d = x.dims();
    const keel::usize H = d.rows(), W = d.cols();
    const keel::usize plane = H * W;
    Value out = Value::matrix(H, W, x.type(), mr);

    auto pix = [&](keel::usize i, keel::usize p){ return x.elemAsDouble(p * plane + i); };
    auto store = [&](keel::usize i, keel::f64 v) {
        switch (x.type()) {
            case ValueType::DOUBLE: out.doubleDataMut()[i] = v; break;
            case ValueType::SINGLE: out.singleDataMut()[i] = (keel::f32)v; break;
            case ValueType::UINT8:  out.uint8DataMut()[i]  = satCast<keel::u8>(v); break;
            case ValueType::UINT16: out.uint16DataMut()[i] = satCast<keel::u16>(v); break;
            case ValueType::INT16:  out.int16DataMut()[i]  = satCast<keel::i16>(v); break;
            default:
                throw Error("rgb2gray: unsupported class", 0, 0, "rgb2gray", "",
                            "numkit:rgb2gray:badtype");
        }
    };

    if (x.type() == ValueType::UINT8) {
        const keel::u8 *R = x.uint8Data();
        (void)keel::image::rgb_to_gray_u8_planar(
            {R, plane}, {R + plane, plane}, {R + 2 * plane, plane},
            {out.uint8DataMut(), plane});
        return out;
    }
    if (x.type() == ValueType::DOUBLE) {
        const keel::f64 *R = x.doubleData();
        (void)keel::image::rgb_to_gray_planar(
            {R, plane}, {R + plane, plane}, {R + 2 * plane, plane},
            {out.doubleDataMut(), plane});
        return out;
    }

    // Generic (single / uint16 / int16).
    for (keel::usize i = 0; i < plane; ++i) {
        const keel::f64 y = keel::image::rgb_to_gray(pix(i, 0), pix(i, 1), pix(i, 2));
        store(i, y);
    }
    return out;
}

Value im2gray(const Value &x, keel::Allocator *mr) {
    const auto &d = x.dims();
    if (d.is3D() && d.pages() == 3) return rgb2gray(x, mr);
    return x;  // already grayscale (or unsupported shape — pass through)
}

namespace {

bool spatial_pages_eq(const Value &x, keel::usize want) {
    const auto &d = x.dims();
    if (d.is3D()) return d.pages() == want;
    return want == 1;
}

bool is_int_image_class(ValueType t) {
    return t == ValueType::UINT8 || t == ValueType::UINT16 ||
           t == ValueType::INT16;
}

bool is_float_unit_image(const Value &x) {
    if (x.type() != ValueType::DOUBLE && x.type() != ValueType::SINGLE)
        return false;
    const keel::usize N = x.numel();
    bool any_real = false;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 v = x.elemAsDouble(i);
        if (keel::is_nan(v)) continue;
        if (v < 0.0 || v > 1.0) return false;
        any_real = true;
    }
    return any_real;
}

bool is_pos_int_float(const Value &x) {
    if (x.type() != ValueType::DOUBLE && x.type() != ValueType::SINGLE)
        return false;
    const keel::usize N = x.numel();
    if (N == 0) return false;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 v = x.elemAsDouble(i);
        if (!keel::is_finite(v)) return false;
        if (v < 1.0 || v != keel::floor(v)) return false;
    }
    return true;
}

bool all_zero_or_one(const Value &x) {
    const keel::usize N = x.numel();
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 v = x.elemAsDouble(i);
        if (keel::is_nan(v)) return false;
        if (v != 0.0 && v != 1.0) return false;
    }
    return true;
}

inline Value bool_scalar(bool b, keel::Allocator *mr) {
    Value out = Value::matrix(1, 1, ValueType::LOGICAL, mr);
    out.logicalDataMut()[0] = b ? 1u : 0u;
    return out;
}

void iptnum2ordinal_argcheck(keel::f64 n)
{
    if (!keel::is_finite(n) || n <= 0.0 || n != keel::floor(n))
        throw Error("iptnum2ordinal: num must be a real positive integer",
                    0, 0, "iptnum2ordinal", "", "numkit:iptnum2ordinal:n");
}

void imcast_argcheck(keel::StringRef type, keel::String &lo)
{
    lo.clear();
    lo.reserve(type.size());
    for (char c : type) lo.push_back(keel::to_ascii_lower(c));
    if (lo != "double" && lo != "single" && lo != "uint8" &&
        lo != "uint16" && lo != "int16" && lo != "logical")
        throw Error("imcast: unsupported TYPE", 0, 0, "imcast", "",
                    "numkit:imcast:type");
}

} // anonymous

Value iptnum2ordinal(keel::f64 n, keel::Allocator *mr)
{
    iptnum2ordinal_argcheck(n);

    static const char *kWords[] = {
        nullptr,    "first",      "second",   "third",
        "fourth",   "fifth",      "sixth",    "seventh",
        "eighth",   "ninth",      "tenth",    "eleventh",
        "twelfth",  "thirteenth", "fourteenth","fifteenth",
        "sixteenth","seventeenth","eighteenth","nineteenth",
        "twentieth"
    };

    const keel::i64 ll = keel::as<keel::i64>(n);
    keel::String s;
    if (ll >= 1 && ll <= 20) {
        s.assign(kWords[ll]);
    } else {
        s = keel::format("{}", ll);
        const char last = s.back();
        const char *suffix = "th";
        if      (last == '1') suffix = "st";
        else if (last == '2') suffix = "nd";
        else if (last == '3') suffix = "rd";
        s.append(suffix);
    }
    return Value::fromString(s, mr);
}

Value imcast(const Value &I, keel::StringRef type, keel::Allocator *mr)
{
    keel::String lo;
    imcast_argcheck(type, lo);

    auto cls_of = [](ValueType t) -> const char * {
        switch (t) {
            case ValueType::DOUBLE:  return "double";
            case ValueType::SINGLE:  return "single";
            case ValueType::UINT8:   return "uint8";
            case ValueType::UINT16:  return "uint16";
            case ValueType::INT16:   return "int16";
            case ValueType::LOGICAL: return "logical";
            default:                 return "?";
        }
    };

    if (lo == cls_of(I.type())) return I;

    if (lo == "double")  return im2double(I, mr);
    if (lo == "single")  return im2single(I, mr);
    if (lo == "uint8")   return im2uint8(I, mr);
    if (lo == "uint16")  return im2uint16(I, mr);
    if (lo == "int16")   return im2int16(I, mr);
    if (lo == "logical") {
        const auto &d = I.dims();
        Value out = d.is3D()
            ? Value::matrix3d(d.rows(), d.cols(), d.pages(),
                              ValueType::LOGICAL, mr)
            : Value::matrix(d.rows(), d.cols(), ValueType::LOGICAL, mr);
        const keel::usize N = I.numel();
        keel::u8 *od = out.logicalDataMut();
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 v = I.elemAsDouble(i);
            od[i] = (v != 0.0 && !keel::is_nan(v)) ? 1u : 0u;
        }
        return out;
    }
    throw Error("imcast: unsupported TYPE", 0, 0, "imcast", "",
                "numkit:imcast:type");
}

namespace {

Value gray_colormap(keel::i32 n, keel::Allocator *mr)
{
    Value map = Value::matrix(keel::as<keel::usize>(n), 3,
                              ValueType::DOUBLE, mr);
    keel::f64 *md = map.doubleDataMut();
    if (n <= 0) return map;
    if (n == 1) {
        md[0] = md[1] = md[2] = 0.0;
        return map;
    }
    for (keel::i32 i = 0; i < n; ++i) {
        const keel::f64 v = keel::as<keel::f64>(i) / keel::as<keel::f64>(n - 1);
        md[0 * n + i] = v;
        md[1 * n + i] = v;
        md[2 * n + i] = v;
    }
    return map;
}

void gray2ind_argcheck(keel::i32 n)
{
    if (n < 1 || n > 65536)
        throw Error("gray2ind: N must be in [1, 65536]",
                    0, 0, "gray2ind", "", "numkit:gray2ind:n");
}

void ind2gray_argcheck(const Value &idx, const Value &map)
{
    const ValueType xt = idx.type();
    if (!(xt == ValueType::DOUBLE || xt == ValueType::SINGLE
          || xt == ValueType::UINT8 || xt == ValueType::UINT16
          || xt == ValueType::LOGICAL))
        internal::requireImageClass(idx, "ind2gray", "X", 1,
                                    internal::ImgClasses::DoubleOnly);
    internal::requireColormap(map, "ind2gray", 2);
    if (map.numel() == 0)
        throw Error("ind2gray: requires (X, MAP) with non-empty MAP",
                    0, 0, "ind2gray", "", "numkit:ind2gray:nargin");
    if (map.dims().cols() != 3 || map.dims().is3D())
        throw Error("ind2gray: MAP must be N-by-3",
                    0, 0, "ind2gray", "", "numkit:ind2gray:map");
    if (map.dims().rows() < 1)
        throw Error("ind2gray: MAP must have at least one row",
                    0, 0, "ind2gray", "", "numkit:ind2gray:emptyMap");
}

void ind2rgb_argcheck(const Value &idx, const Value &map)
{
    const ValueType xt = idx.type();
    if (!(xt == ValueType::DOUBLE || xt == ValueType::SINGLE
          || xt == ValueType::UINT8 || xt == ValueType::UINT16
          || xt == ValueType::INT16 || xt == ValueType::CHAR))
        internal::requireImageClass(idx, "ind2rgb", "X", 1,
                                    internal::ImgClasses::DoubleOnly);
    internal::requireRealKernelImage(map, "ind2rgb", "map", 2);
    if (map.type() != ValueType::DOUBLE)
        internal::requireImageClass(map, "ind2rgb", "map", 2,
                                    internal::ImgClasses::AnyNumericReal);
}

void getrangefromclass_argcheck(const Value &I)
{
    switch (I.type()) {
        case ValueType::DOUBLE:
        case ValueType::SINGLE:
        case ValueType::LOGICAL:
        case ValueType::UINT8:
        case ValueType::UINT16:
        case ValueType::INT16:
            break;
        default:
            throw Error("getrangefromclass: unrecognized image class",
                        0, 0, "getrangefromclass", "",
                        "numkit:getrangefromclass:cls");
    }
}

void intlut_argcheck(const Value &A, const Value &LUT, keel::usize &expectedLen)
{
    const ValueType atype = A.type();
    const ValueType ltype = LUT.type();
    switch (atype) {
        case ValueType::UINT8:  expectedLen = 256;   break;
        case ValueType::UINT16:
        case ValueType::INT16:  expectedLen = 65536; break;
        default:
            throw Error("intlut: A must be uint8, uint16, or int16",
                        0, 0, "intlut", "", "numkit:intlut:atype");
    }
    if (ltype != ValueType::UINT8 && ltype != ValueType::UINT16 &&
        ltype != ValueType::INT16)
        throw Error("intlut: LUT must be uint8, uint16, or int16",
                    0, 0, "intlut", "", "numkit:intlut:luttype");
    if (LUT.numel() != expectedLen)
        throw Error("intlut: LUT length does not match input class range",
                    0, 0, "intlut", "", "numkit:intlut:lutsize");
}

} // anonymous

keel::Tuple<Value, Value>
gray2ind(const Value &I, keel::i32 n, keel::Allocator *mr)
{
    gray2ind_argcheck(n);

    const ValueType cls = I.type();
    const auto &d = I.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    const keel::usize N = I.numel();
    const ValueType outT = (n <= 256) ? ValueType::UINT8
                                       : ValueType::UINT16;
    Value out = d.is3D()
        ? Value::matrix3d(H, W, d.pages(), outT, mr)
        : Value::matrix(H, W, outT, mr);
    Value map = gray_colormap(n, mr);
    if (N == 0) return {keel::move(out), keel::move(map)};

    keel::f64 low = 0.0, scale = 1.0;
    const bool isFloat = (cls == ValueType::DOUBLE || cls == ValueType::SINGLE);
    switch (cls) {
        case ValueType::UINT8:  scale = 255.0;                       break;
        case ValueType::UINT16: scale = 65535.0;                     break;
        case ValueType::INT16:  low = -32768.0; scale = 65535.0;     break;
        case ValueType::DOUBLE:
        case ValueType::SINGLE:
        case ValueType::LOGICAL: scale = 1.0;                        break;
        default:
            throw Error("gray2ind: unsupported class",
                        0, 0, "gray2ind", "", "numkit:gray2ind:cls");
    }
    if (isFloat) {
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 v = I.elemAsDouble(i);
            if (v < 0.0 || v > 1.0)
                throw Error("gray2ind: float values must be in [0, 1]",
                            0, 0, "gray2ind", "", "numkit:gray2ind:range");
        }
    }
    const keel::f64 k = keel::as<keel::f64>(n - 1) / scale;
    // MATLAB / Octave's integer cast rounds half-away-from-zero, not
    // truncates toward zero.
    if (outT == ValueType::UINT8) {
        keel::u8 *od = out.uint8DataMut();
        for (keel::usize i = 0; i < N; ++i) {
            keel::f64 v = keel::round_i64((I.elemAsDouble(i) - low) * k);
            if (v < 0)   v = 0;
            if (v > 255) v = 255;
            od[i] = keel::as<keel::u8>(v);
        }
    } else {
        keel::u16 *od = out.uint16DataMut();
        for (keel::usize i = 0; i < N; ++i) {
            keel::f64 v = keel::round_i64((I.elemAsDouble(i) - low) * k);
            if (v < 0)     v = 0;
            if (v > 65535) v = 65535;
            od[i] = keel::as<keel::u16>(v);
        }
    }
    return {keel::move(out), keel::move(map)};
}

Value ind2gray(const Value &idx, const Value &map, keel::Allocator *mr)
{
    ind2gray_argcheck(idx, map);
    // Logical indices coerce to double 0/1 (ref returns a double
    // grayscale image — probed).
    Value idxUsed = (idx.type() == ValueType::LOGICAL)
        ? internal::coerceToDoubleImage(idx, mr) : Value(idx);
    const keel::i32 M = keel::as<keel::i32>(map.dims().rows());

    // Build the grey colormap (length M, DOUBLE in [0, 1]).
    constexpr keel::f64 Cr = 0.298936021293775;
    constexpr keel::f64 Cg = 0.587043074451121;
    constexpr keel::f64 Cb = 0.114020904255103;
    keel::Array<keel::f64> graycm(M, mr);
    for (keel::i32 k = 0; k < M; ++k) {
        // map is column-major M×3: map[r, c] = data[c*M + r].
        const keel::f64 r = map.elemAsDouble(0 * M + k);
        const keel::f64 g = map.elemAsDouble(1 * M + k);
        const keel::f64 b = map.elemAsDouble(2 * M + k);
        graycm[k] = Cr * r + Cg * g + Cb * b;
    }

    const auto &d = idxUsed.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    const keel::usize N = idxUsed.numel();
    const ValueType outT = idxUsed.type();   // class-preserving

    Value out = d.is3D()
        ? Value::matrix3d(H, W, d.pages(), outT, mr)
        : Value::matrix(H, W, outT, mr);
    if (N == 0) return out;

    // Index → grey conversion. MATLAB index conventions:
    //   * float X is 1-based, clamped to [1, M].
    //   * integer X is 0-based, LUT padded to vs = 256 / 65536 with the
    //     last grey value beyond `M - 1`.
    auto float_lookup = [&](keel::f64 v) -> keel::f64 {
        keel::i64 k = keel::as<keel::i64>(v);
        if (k < 1)  k = 1;
        if (k > M)  k = M;
        return graycm[keel::as<keel::usize>(k - 1)];
    };
    auto int_lookup_u8 = [&](keel::u32 v) -> keel::u8 {
        keel::usize k = v;
        if (k >= keel::as<keel::usize>(M))
            k = keel::as<keel::usize>(M - 1);
        keel::f64 scaled = graycm[k] * 255.0;
        if (scaled < 0.0)   scaled = 0.0;
        if (scaled > 255.0) scaled = 255.0;
        return keel::as<keel::u8>(keel::round_i64(scaled));
    };
    auto int_lookup_u16 = [&](keel::u32 v) -> keel::u16 {
        keel::usize k = v;
        if (k >= keel::as<keel::usize>(M))
            k = keel::as<keel::usize>(M - 1);
        keel::f64 scaled = graycm[k] * 65535.0;
        if (scaled < 0.0)     scaled = 0.0;
        if (scaled > 65535.0) scaled = 65535.0;
        return keel::as<keel::u16>(keel::round_i64(scaled));
    };

    switch (outT) {
        case ValueType::DOUBLE: {
            keel::f64 *od = out.doubleDataMut();
            for (keel::usize i = 0; i < N; ++i)
                od[i] = float_lookup(idxUsed.elemAsDouble(i));
            break;
        }
        case ValueType::SINGLE: {
            keel::f32 *od = out.singleDataMut();
            for (keel::usize i = 0; i < N; ++i)
                od[i] = keel::as<keel::f32>(float_lookup(idxUsed.elemAsDouble(i)));
            break;
        }
        case ValueType::UINT8: {
            keel::u8 *od = out.uint8DataMut();
            const keel::u8 *src = idx.uint8Data();
            for (keel::usize i = 0; i < N; ++i)
                od[i] = int_lookup_u8(src[i]);
            break;
        }
        case ValueType::UINT16: {
            keel::u16 *od = out.uint16DataMut();
            const keel::u16 *src = idx.uint16Data();
            for (keel::usize i = 0; i < N; ++i)
                od[i] = int_lookup_u16(src[i]);
            break;
        }
        case ValueType::LOGICAL: {
            // MATLAB historical compat: logical → uint8-like 0/1 lookup
            // (the MATLAB source uses intlut with vs = 256 here too).
            keel::u8 *od = out.uint8DataMut();
            const keel::u8 *src = idx.logicalData();
            for (keel::usize i = 0; i < N; ++i)
                od[i] = int_lookup_u8(src[i] ? 1u : 0u);
            break;
        }
        default:
            throw Error("ind2gray: X must be double, single, uint8, uint16, "
                        "or logical",
                        0, 0, "ind2gray", "", "numkit:ind2gray:cls");
    }
    return out;
}

Value ind2rgb(const Value &idx, const Value &map, keel::Allocator *mr)
{
    ind2rgb_argcheck(idx, map);
    Value rgbIdxC = (idx.type() == ValueType::CHAR)
        ? internal::coerceToDoubleImage(idx, mr) : Value(idx);
    Value rgbMapC = (map.type() != ValueType::DOUBLE)
        ? internal::coerceToDoubleImage(map, mr) : Value(map);
    if (rgbMapC.dims().cols() != 3)
        throw Error("ind2rgb: map must be N-by-3",
                    0, 0, "ind2rgb", "", "numkit:ind2rgb:map");
    const auto &d = rgbIdxC.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    const keel::usize N = rgbIdxC.numel();
    const keel::i32 M = keel::as<keel::i32>(rgbMapC.dims().rows());

    Value out = Value::matrix3d(H, W, 3, ValueType::DOUBLE, mr);
    if (N == 0) return out;
    keel::f64 *od = out.doubleDataMut();
    const keel::usize plane = H * W;

    const bool isFloatIdx = (idx.type() == ValueType::DOUBLE ||
                             idx.type() == ValueType::SINGLE);
    for (keel::usize i = 0; i < N; ++i) {
        keel::i64 k;
        const keel::f64 v = rgbIdxC.elemAsDouble(i);
        if (keel::is_nan(v) || !keel::is_finite(v)) {
            // NaN / Inf clip to last colormap row.
            k = M - 1;
        } else {
            k = keel::as<keel::i64>(v);
            if (isFloatIdx) k -= 1;
        }
        if (k < 0)  k = 0;
        if (k >= M) k = M - 1;
        // map is M×3, col-major: map[r, c] = data[c*M + r].
        od[0 * plane + i] = rgbMapC.elemAsDouble(0 * M + k);
        od[1 * plane + i] = rgbMapC.elemAsDouble(1 * M + k);
        od[2 * plane + i] = rgbMapC.elemAsDouble(2 * M + k);
    }
    return out;
}

Value getrangefromclass(const Value &I, keel::Allocator *mr)
{
    getrangefromclass_argcheck(I);
    Value r = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    keel::f64 *rd = r.doubleDataMut();
    switch (I.type()) {
        case ValueType::DOUBLE:
        case ValueType::SINGLE:
        case ValueType::LOGICAL: rd[0] = 0.0;       rd[1] = 1.0;     break;
        case ValueType::UINT8:   rd[0] = 0.0;       rd[1] = 255.0;   break;
        case ValueType::UINT16:  rd[0] = 0.0;       rd[1] = 65535.0; break;
        case ValueType::INT16:   rd[0] = -32768.0;  rd[1] = 32767.0; break;
        default:
            throw Error("getrangefromclass: unrecognized image class",
                        0, 0, "getrangefromclass", "",
                        "numkit:getrangefromclass:cls");
    }
    return r;
}

Value isbw(const Value &BW, keel::StringRef mode, keel::Allocator *mr)
{
    if (!spatial_pages_eq(BW, 1)) return bool_scalar(false, mr);
    if (BW.numel() == 0)          return bool_scalar(false, mr);
    if (mode == "logical")
        return bool_scalar(BW.type() == ValueType::LOGICAL, mr);
    if (mode != "non-logical")
        throw Error("isbw: MODE must be 'logical' or 'non-logical'",
                    0, 0, "isbw", "", "numkit:isbw:mode");
    if (BW.type() == ValueType::LOGICAL) return bool_scalar(true, mr);
    if (is_int_image_class(BW.type()) ||
        BW.type() == ValueType::DOUBLE || BW.type() == ValueType::SINGLE)
        return bool_scalar(all_zero_or_one(BW), mr);
    return bool_scalar(false, mr);
}

Value isgray(const Value &I, keel::Allocator *mr)
{
    if (!spatial_pages_eq(I, 1)) return bool_scalar(false, mr);
    if (I.numel() == 0)          return bool_scalar(false, mr);
    if (is_int_image_class(I.type())) return bool_scalar(true, mr);
    return bool_scalar(is_float_unit_image(I), mr);
}

Value isind(const Value &I, keel::Allocator *mr)
{
    if (!spatial_pages_eq(I, 1)) return bool_scalar(false, mr);
    if (I.numel() == 0)          return bool_scalar(false, mr);
    if (I.type() == ValueType::UINT8 || I.type() == ValueType::UINT16)
        return bool_scalar(true, mr);
    return bool_scalar(is_pos_int_float(I), mr);
}

Value isrgb(const Value &I, keel::Allocator *mr)
{
    if (!spatial_pages_eq(I, 3)) return bool_scalar(false, mr);
    if (I.numel() == 0)          return bool_scalar(false, mr);
    if (is_int_image_class(I.type())) return bool_scalar(true, mr);
    return bool_scalar(is_float_unit_image(I), mr);
}

Value iscolormap(const Value &cmap, keel::Allocator *mr)
{
    const auto &d = cmap.dims();
    if (d.is3D())                return bool_scalar(false, mr);
    if (d.cols() != 3)           return bool_scalar(false, mr);
    if (cmap.numel() == 0)       return bool_scalar(false, mr);
    const ValueType t = cmap.type();
    if (t != ValueType::DOUBLE && t != ValueType::SINGLE)
        return bool_scalar(false, mr);
    if (cmap.isComplex())        return bool_scalar(false, mr);
    return bool_scalar(true, mr);
}

Value intlut(const Value &A, const Value &LUT, keel::Allocator *mr)
{
    keel::usize expectedLen = 0;
    intlut_argcheck(A, LUT, expectedLen);
    const ValueType atype = A.type();
    const ValueType ltype = LUT.type();

    Value out = alloc_like(A, ltype, mr);
    const keel::usize N = A.numel();
    if (N == 0) return out;

    const keel::i32 shift = (atype == ValueType::INT16) ? 32768 : 0;

    auto idx_at = [&](keel::usize i) -> keel::usize {
        return keel::as<keel::usize>(keel::as<keel::i64>(A.elemAsDouble(i))
                                   + shift);
    };

    switch (ltype) {
        case ValueType::UINT8: {
            const keel::u8 *lut = LUT.uint8Data();
            keel::u8 *od = out.uint8DataMut();
            for (keel::usize i = 0; i < N; ++i) od[i] = lut[idx_at(i)];
            break;
        }
        case ValueType::UINT16: {
            const keel::u16 *lut = LUT.uint16Data();
            keel::u16 *od = out.uint16DataMut();
            for (keel::usize i = 0; i < N; ++i) od[i] = lut[idx_at(i)];
            break;
        }
        case ValueType::INT16: {
            const keel::i16 *lut = LUT.int16Data();
            keel::i16 *od = out.int16DataMut();
            for (keel::usize i = 0; i < N; ++i) od[i] = lut[idx_at(i)];
            break;
        }
        default: break;
    }
    return out;
}

} // namespace numkit::image
