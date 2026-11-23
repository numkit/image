// toolboxes/image/src/arithmetic/arithmetic.cpp
//
// Image arithmetic. Saturating element-wise operations for integer
// classes; pass-through for floating types. Output class = X's class.

#include <keel/array.h>
#include <keel/math.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/complex.h>
#include <numkit/image/arithmetic/arithmetic.hpp>
#include <numkit/image/internal/type_gates.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>

namespace numkit::image {

namespace {

// Integer saturation helper. Cast `v` (double) to T with clipping to
// T's representable range, rounding to nearest.
template <typename T>
inline T satCast(keel::f64 v) {
    if (keel::is_nan(v)) return T{};
    constexpr keel::f64 lo = keel::as<keel::f64>(keel::Limits<T>::kLowest);
    constexpr keel::f64 hi = keel::as<keel::f64>(keel::Limits<T>::kMax);
    if (v <= lo) return keel::Limits<T>::kLowest;
    if (v >= hi) return keel::Limits<T>::kMax;
    return static_cast<T>(keel::round_i64(v));
}
template <> inline keel::f64 satCast<keel::f64>(keel::f64 v) { return v; }
template <> inline keel::f32  satCast<keel::f32>(keel::f64 v)  { return keel::as<keel::f32>(v); }

// Allocate an output Value matching X's shape and class.
inline Value makeOutLike(const Value &x, keel::Allocator *mr) {
    const auto &d = x.dims();
    if (x.isScalar()) return Value::scalar(0.0, mr);  // overwritten by caller
    if (d.is3D()) return Value::matrix3d(d.rows(), d.cols(), d.pages(), x.type(), mr);
    return Value::matrix(d.rows(), d.cols(), x.type(), mr);
}

// Dispatch a binary op on element pairs (xi, yi) producing xi.class.
template <typename Op>
Value binop(const Value &x, const Value &y, Op op, keel::Allocator *mr) {
    const keel::usize n = std::max(x.numel(), y.numel());
    if (n == 0) return makeOutLike(x, mr);
    Value out;
    if (x.isScalar() && !y.isScalar()) {
        // Output shape matches y, but class matches x.
        const auto &d = y.dims();
        if (d.is3D()) out = Value::matrix3d(d.rows(), d.cols(), d.pages(), x.type(), mr);
        else          out = Value::matrix(d.rows(), d.cols(), x.type(), mr);
    } else {
        out = makeOutLike(x, mr);
    }

    auto getX = [&](keel::usize i){ return x.elemAsDouble(x.isScalar() ? 0 : i); };
    auto getY = [&](keel::usize i){ return y.elemAsDouble(y.isScalar() ? 0 : i); };

    switch (x.type()) {
        case ValueType::DOUBLE: {
            keel::f64 *od = out.doubleDataMut();
            for (keel::usize i = 0; i < n; ++i) od[i] = satCast<keel::f64>(op(getX(i), getY(i)));
            break;
        }
        case ValueType::SINGLE: {
            keel::f32 *od = out.singleDataMut();
            for (keel::usize i = 0; i < n; ++i) od[i] = satCast<keel::f32>(op(getX(i), getY(i)));
            break;
        }
        case ValueType::UINT8: {
            keel::u8 *od = out.uint8DataMut();
            for (keel::usize i = 0; i < n; ++i) od[i] = satCast<keel::u8>(op(getX(i), getY(i)));
            break;
        }
        case ValueType::UINT16: {
            keel::u16 *od = out.uint16DataMut();
            for (keel::usize i = 0; i < n; ++i) od[i] = satCast<keel::u16>(op(getX(i), getY(i)));
            break;
        }
        case ValueType::INT16: {
            keel::i16 *od = out.int16DataMut();
            for (keel::usize i = 0; i < n; ++i) od[i] = satCast<keel::i16>(op(getX(i), getY(i)));
            break;
        }
        case ValueType::LOGICAL: {
            keel::u8 *od = out.logicalDataMut();
            for (keel::usize i = 0; i < n; ++i) od[i] = satCast<keel::u8>(op(getX(i), getY(i))) ? 1 : 0;
            break;
        }
        default:
            throw Error("imarith: unsupported image class", 0, 0, "imarith", "",
                        "numkit:imarith:badtype");
    }
    return out;
}

// Unary variant.
template <typename Op>
Value unop(const Value &x, Op op, keel::Allocator *mr) {
    const keel::usize n = x.numel();
    Value out = makeOutLike(x, mr);
    if (n == 0) return out;
    auto getX = [&](keel::usize i){ return x.elemAsDouble(i); };
    switch (x.type()) {
        case ValueType::DOUBLE: {
            keel::f64 *od = out.doubleDataMut();
            for (keel::usize i = 0; i < n; ++i) od[i] = satCast<keel::f64>(op(getX(i)));
            break;
        }
        case ValueType::SINGLE: {
            keel::f32 *od = out.singleDataMut();
            for (keel::usize i = 0; i < n; ++i) od[i] = satCast<keel::f32>(op(getX(i)));
            break;
        }
        case ValueType::UINT8: {
            keel::u8 *od = out.uint8DataMut();
            for (keel::usize i = 0; i < n; ++i) od[i] = satCast<keel::u8>(op(getX(i)));
            break;
        }
        case ValueType::UINT16: {
            keel::u16 *od = out.uint16DataMut();
            for (keel::usize i = 0; i < n; ++i) od[i] = satCast<keel::u16>(op(getX(i)));
            break;
        }
        case ValueType::INT16: {
            keel::i16 *od = out.int16DataMut();
            for (keel::usize i = 0; i < n; ++i) od[i] = satCast<keel::i16>(op(getX(i)));
            break;
        }
        case ValueType::LOGICAL: {
            keel::u8 *od = out.logicalDataMut();
            for (keel::usize i = 0; i < n; ++i) od[i] = satCast<keel::u8>(op(getX(i))) ? 1 : 0;
            break;
        }
        default:
            throw Error("imarith: unsupported image class", 0, 0, "imarith", "",
                        "numkit:imarith:badtype");
    }
    return out;
}

// Saturated max/min for X's class.
inline keel::f64 classMax(ValueType t) {
    switch (t) {
        case ValueType::UINT8:  return 255.0;
        case ValueType::UINT16: return 65535.0;
        case ValueType::INT16:  return 32767.0;
        case ValueType::LOGICAL: return 1.0;
        case ValueType::SINGLE:
        case ValueType::DOUBLE: return 1.0;  // floating-point images are conventionally [0, 1]
        default: return 1.0;
    }
}

void imarith_argcheck(const Value &x)
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
            throw Error("imarith: unsupported image class", 0, 0, "imarith", "",
                        "numkit:imarith:badtype");
    }
}

void imlincomb_argcheck(keel::ArrayRef<keel::f64> coefs,
                        keel::ArrayRef<Value> images,
                        ValueType output_class)
{
    if (coefs.size() != images.size() && coefs.size() != images.size() + 1)
        throw Error("imlincomb: number of coefficients must equal number of images "
                    "(optional one extra for the additive constant)",
                    0, 0, "imlincomb", "", "numkit:imlincomb:nargin");
    if (images.empty()) return;
    const keel::usize n = images[0].numel();
    for (keel::usize k = 0; k < images.size(); ++k) {
        if (images[k].numel() != n)
            throw Error("imlincomb: all images must have the same size",
                        0, 0, "imlincomb", "", "numkit:imlincomb:size");
    }
    switch (output_class) {
        case ValueType::DOUBLE:
        case ValueType::SINGLE:
        case ValueType::UINT8:
        case ValueType::UINT16:
        case ValueType::INT16:
            break;
        default:
            throw Error("imlincomb: unsupported output class",
                        0, 0, "imlincomb", "", "numkit:imlincomb:badtype");
    }
}

void imapplymatrix_argcheck(const Value &M, const Value &x, ValueType output_class)
{
    internal::requireImageClassAllowComplex(M, "imapplymatrix", "M", 1,
                                internal::ImgClasses::AnyNumericReal);
    internal::requireImageClassAllowComplex(x, "imapplymatrix", "X", 2,
                                internal::ImgClasses::FloatInt);
    const auto &dx = x.dims();
    const keel::usize pages = dx.is3D() ? dx.pages() : 1;
    const keel::usize Mcols = M.dims().cols();
    if (Mcols != pages)
        throw Error("imapplymatrix: matrix columns must match the third dimension of X",
                    0, 0, "imapplymatrix", "", "numkit:imapplymatrix:size");
    if (M.isComplex() || x.isComplex()) {
        if (output_class != ValueType::DOUBLE && output_class != ValueType::SINGLE &&
            output_class != ValueType::UINT8 && output_class != ValueType::UINT16 &&
            output_class != ValueType::INT16 && output_class != ValueType::COMPLEX)
            throw Error("imapplymatrix: unsupported output class",
                        0, 0, "imapplymatrix", "", "numkit:imapplymatrix:badtype");
    }
}

} // anonymous

Value imadd(const Value &x, const Value &y, keel::Allocator *mr) {
    imarith_argcheck(x);
    return binop(x, y, [](keel::f64 a, keel::f64 b){ return a + b; }, mr);
}

Value imsubtract(const Value &x, const Value &y, keel::Allocator *mr) {
    imarith_argcheck(x);
    return binop(x, y, [](keel::f64 a, keel::f64 b){ return a - b; }, mr);
}

Value immultiply(const Value &x, const Value &y, keel::Allocator *mr) {
    imarith_argcheck(x);
    return binop(x, y, [](keel::f64 a, keel::f64 b){ return a * b; }, mr);
}

Value imdivide(const Value &x, const Value &y, keel::Allocator *mr) {
    imarith_argcheck(x);
    return binop(x, y, [](keel::f64 a, keel::f64 b){
        if (b == 0.0) {
            if (a > 0.0) return keel::Limits<keel::f64>::infinity();
            if (a < 0.0) return -keel::Limits<keel::f64>::infinity();
            return keel::Limits<keel::f64>::quiet_nan();
        }
        return a / b;
    }, mr);
}

Value imabsdiff(const Value &x, const Value &y, keel::Allocator *mr) {
    imarith_argcheck(x);
    return binop(x, y, [](keel::f64 a, keel::f64 b){ return keel::abs(a - b); }, mr);
}

Value imcomplement(const Value &x, keel::Allocator *mr) {
    imarith_argcheck(x);
    const keel::f64 cmax = classMax(x.type());
    return unop(x, [=](keel::f64 v){ return cmax - v; }, mr);
}

Value imlincomb(keel::ArrayRef<keel::f64> coefs, keel::ArrayRef<Value> images, ValueType output_class, keel::Allocator *mr)
{
    imlincomb_argcheck(coefs, images, output_class);
    if (coefs.size() != images.size() && coefs.size() != images.size() + 1)
        throw Error("imlincomb: number of coefficients must equal number of images "
                    "(optional one extra for the additive constant)",
                    0, 0, "imlincomb", "", "numkit:imlincomb:nargin");
    if (images.empty()) return Value::scalar(0.0, mr);

    const Value &x0 = images[0];
    const auto &d = x0.dims();
    const keel::usize n = x0.numel();
    Value out;
    if (d.is3D()) out = Value::matrix3d(d.rows(), d.cols(), d.pages(), output_class, mr);
    else          out = Value::matrix(d.rows(), d.cols(), output_class, mr);

    // Accumulate in double, then cast at the end.
    keel::Array<keel::f64> acc(n, 0.0);
    const keel::f64 bias = (coefs.size() == images.size() + 1) ? coefs.back() : 0.0;
    for (keel::usize k = 0; k < images.size(); ++k) {
        const Value &xk = images[k];
        const keel::f64 ck = coefs[k];
        if (xk.numel() != n)
            throw Error("imlincomb: all images must have the same size",
                        0, 0, "imlincomb", "", "numkit:imlincomb:size");
        for (keel::usize i = 0; i < n; ++i) acc[i] += ck * xk.elemAsDouble(i);
    }
    for (keel::usize i = 0; i < n; ++i) acc[i] += bias;

    // Cast acc into output_class with saturation.
    switch (output_class) {
        case ValueType::DOUBLE:
            std::copy(acc.begin(), acc.end(), out.doubleDataMut()); break;
        case ValueType::SINGLE: {
            keel::f32 *od = out.singleDataMut();
            for (keel::usize i = 0; i < n; ++i) od[i] = satCast<keel::f32>(acc[i]);
            break;
        }
        case ValueType::UINT8: {
            keel::u8 *od = out.uint8DataMut();
            for (keel::usize i = 0; i < n; ++i) od[i] = satCast<keel::u8>(acc[i]);
            break;
        }
        case ValueType::UINT16: {
            keel::u16 *od = out.uint16DataMut();
            for (keel::usize i = 0; i < n; ++i) od[i] = satCast<keel::u16>(acc[i]);
            break;
        }
        case ValueType::INT16: {
            keel::i16 *od = out.int16DataMut();
            for (keel::usize i = 0; i < n; ++i) od[i] = satCast<keel::i16>(acc[i]);
            break;
        }
        default:
            throw Error("imlincomb: unsupported output class",
                        0, 0, "imlincomb", "", "numkit:imlincomb:badtype");
    }
    return out;
}

Value imapplymatrix(const Value &M, const Value &x, ValueType output_class, keel::Allocator *mr)
{
    imapplymatrix_argcheck(M, x, output_class);
    // Probed R2025b: M coerces every numeric class (logical X dies at
    // the output-class check); complex M / complex X both compute
    // complex results in the reference.
    internal::requireImageClassAllowComplex(M, "imapplymatrix", "M", 1,
                                internal::ImgClasses::AnyNumericReal);
    internal::requireImageClassAllowComplex(x, "imapplymatrix", "X", 2,
                                internal::ImgClasses::FloatInt);
    const auto &dx = x.dims();
    const keel::usize pages = dx.is3D() ? dx.pages() : 1;
    const keel::usize Mrows = M.dims().rows();
    const keel::usize Mcols = M.dims().cols();
    if (Mcols != pages)
        throw Error("imapplymatrix: matrix columns must match the third dimension of X",
                    0, 0, "imapplymatrix", "", "numkit:imapplymatrix:size");

    const keel::usize H = dx.rows();
    const keel::usize W = dx.cols();
    const keel::usize plane = H * W;

    if (M.isComplex() || x.isComplex()) {
        if (output_class != ValueType::DOUBLE && output_class != ValueType::SINGLE &&
            output_class != ValueType::UINT8 && output_class != ValueType::UINT16 &&
            output_class != ValueType::INT16 && output_class != ValueType::COMPLEX)
            throw Error("imapplymatrix: unsupported output class",
                        0, 0, "imapplymatrix", "", "numkit:imapplymatrix:badtype");
        Value out = (Mrows == 1)
            ? Value::matrix(H, W, ValueType::COMPLEX, mr)
            : Value::matrix3d(H, W, Mrows, ValueType::COMPLEX, mr);
        auto Mc = [&](keel::usize r, keel::usize c) -> keel::c64 {
            const keel::usize idx = c * Mrows + r;
            return M.isComplex() ? M.complexData()[idx]
                                 : keel::c64(M.elemAsDouble(idx), 0.0);
        };
        auto Xc = [&](keel::usize i, keel::usize p) -> keel::c64 {
            const keel::usize idx = p * plane + i;
            return x.isComplex() ? x.complexData()[idx]
                                 : keel::c64(x.elemAsDouble(idx), 0.0);
        };
        keel::c64 *od = out.complexDataMut();
        for (keel::usize r = 0; r < Mrows; ++r) {
            for (keel::usize i = 0; i < plane; ++i) {
                keel::c64 acc(0.0, 0.0);
                for (keel::usize c = 0; c < pages; ++c) acc += Mc(r, c) * Xc(i, c);
                od[r * plane + i] = acc;
            }
        }
        return out;
    }

    Value out;
    if (Mrows == 1) out = Value::matrix(H, W, output_class, mr);
    else            out = Value::matrix3d(H, W, Mrows, output_class, mr);

    // Accumulate per output pixel in double.
    auto Md = [&](keel::usize r, keel::usize c){ return M.elemAsDouble(c * Mrows + r); };
    auto Xd = [&](keel::usize i, keel::usize p){ return x.elemAsDouble(p * plane + i); };

    auto store = [&](keel::usize outIdx, keel::f64 v) {
        switch (output_class) {
            case ValueType::DOUBLE:  out.doubleDataMut()[outIdx]  = satCast<keel::f64>(v); break;
            case ValueType::SINGLE:  out.singleDataMut()[outIdx]  = satCast<keel::f32>(v); break;
            case ValueType::UINT8:   out.uint8DataMut()[outIdx]   = satCast<keel::u8>(v); break;
            case ValueType::UINT16:  out.uint16DataMut()[outIdx]  = satCast<keel::u16>(v); break;
            case ValueType::INT16:   out.int16DataMut()[outIdx]   = satCast<keel::i16>(v); break;
            default:
                throw Error("imapplymatrix: unsupported output class",
                            0, 0, "imapplymatrix", "", "numkit:imapplymatrix:badtype");
        }
    };

    for (keel::usize r = 0; r < Mrows; ++r) {
        for (keel::usize i = 0; i < plane; ++i) {
            keel::f64 acc = 0.0;
            for (keel::usize c = 0; c < pages; ++c) acc += Md(r, c) * Xd(i, c);
            store(r * plane + i, acc);
        }
    }
    return out;
}

} // namespace numkit::image
