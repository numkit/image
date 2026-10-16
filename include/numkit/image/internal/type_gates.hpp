/// @file type_gates.hpp
/// @ingroup group_image
// toolboxes/image/include/nk/image/internal/type_gates.hpp
//
// Shared <fn>_argcheck cores for toolboxes/image (argcheck separation,
// dev-docs/todo/argcheck_separation.md — boundary rule: raw arguments in,
// all message digestion here; call sites are one pass-through line).
// PRIVATE to image TUs — not part of the public library API.
//
// Rejection classes reference-probed against R2025b on 2026-09-18
// (build/probe12..13_image_ref.m, ~940 mirrored class probes). The
// reference's image toolbox takes a DIFFERENT class set per entry —
// imadjust's family rejects logical, medfilt2's accepts it, entropyfilt
// rejects int16 while stdfilt takes it, bwareafilt demands logical while
// bwareaopen takes every numeric class — so the gates are parameterized
// by a per-entry mask instead of one blanket rule. Complex arrays are
// rejected wherever the reference's validator says "expectedReal" (a
// complex-stored array with zero imaginary part is real to the
// reference; nk collapses +0i to real storage, so isComplex() is
// exactly the nonzero-imaginary test).

#pragma once

#include <string>
#include <string_view>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/value_type.hpp>   // mtypeName (got-clauses)

namespace numkit::image::internal {

/// @addtogroup group_image
/// @{

/// Render "2x3x4" for got-clauses.
inline std::string shapeOf(const Value &v)
{
    const Dims &d = v.dims();
    std::string s = std::to_string(d.dim(0));
    for (int k = 1; k < d.ndim(); ++k) {
        s += "x";
        s += std::to_string(d.dim(k));
    }
    return s;
}

/// Per-entry accepted-class masks, each named after its probe family.
namespace ImgClasses {
inline constexpr unsigned DoubleOnly     = 1u << 0;   ///< houghpeaks H, imfilter kernel h
inline constexpr unsigned FloatOnly      = 1u << 1;   ///< tonemap HDR (double, single)
inline constexpr unsigned FloatInt       = 1u << 2;   ///< imadjust/imbinarize/... (logical rejected)
inline constexpr unsigned FloatIntLog    = 1u << 3;   ///< medfilt2/ordfilt2/... (logical accepted)
inline constexpr unsigned AnyNumericReal = 1u << 4;   ///< bw* family: every numeric class incl. logical
inline constexpr unsigned LogicalOnly    = 1u << 5;   ///< bwareafilt/bwpropfilt/bwdistgeodesic
inline constexpr unsigned FloatIntNoI16  = 1u << 6;   ///< entropyfilt (int16 rejected)
inline constexpr unsigned FloatIntNoLog  = 1u << 7;   ///< gradientweight/grayconnected (logical rejected)
}   // namespace ImgClasses

/// True when v's class is inside mask.
inline bool classAllowed(const Value &v, unsigned mask)
{
    const ValueType t = v.type();
    if (t == ValueType::DOUBLE)
        // every family takes double EXCEPT the logical-only entries
        // (bwareafilt / bwpropfilt / bwdistgeodesic — probed R2025b).
        return mask != ImgClasses::LogicalOnly;
    if (t == ValueType::SINGLE)
        return (mask & (ImgClasses::FloatOnly | ImgClasses::FloatInt
                        | ImgClasses::FloatIntLog | ImgClasses::FloatIntNoI16
                        | ImgClasses::FloatIntNoLog
                        | ImgClasses::AnyNumericReal)) != 0;
    if (t == ValueType::LOGICAL)
        return (mask & (ImgClasses::FloatIntLog | ImgClasses::FloatIntNoI16
                        | ImgClasses::AnyNumericReal
                        | ImgClasses::LogicalOnly)) != 0;
    if (isIntegerType(t)) {
        if (t == ValueType::INT16)
            return (mask & (ImgClasses::FloatInt | ImgClasses::FloatIntLog
                            | ImgClasses::FloatIntNoLog
                            | ImgClasses::AnyNumericReal)) != 0;
        return (mask & (ImgClasses::FloatInt | ImgClasses::FloatIntLog
                        | ImgClasses::FloatIntNoI16 | ImgClasses::FloatIntNoLog
                        | ImgClasses::AnyNumericReal)) != 0;
    }
    return false;
}

/// Human listing of a mask's classes (message text).
inline std::string describeClasses(unsigned mask)
{
    if (mask == ImgClasses::DoubleOnly)
        return "double";
    if (mask == ImgClasses::FloatOnly)
        return "double or single";
    if (mask == ImgClasses::LogicalOnly)
        return "logical";
    if (mask == ImgClasses::FloatIntNoI16)
        return "double, single, logical, uint8, uint16 (int16 rejected)";
    if (mask == ImgClasses::FloatIntNoLog)
        return "double, single, uint8, uint16, int16";
    if (mask == ImgClasses::FloatInt)
        return "double, single, uint8, uint16, int16";
    if (mask == ImgClasses::FloatIntLog)
        return "double, single, logical, uint8, uint16, int16";
    return "numeric";
}

/// @brief The workhorse gate: v must be a real matrix of one of the
/// entry's accepted classes. Wrong class → :inputType; complex →
/// :expectedReal (the reference's validator wording, probed per entry);
/// cell/string/struct/char always land in :inputType — char CODE-value
/// coercion is NOT an image contract anywhere in the probed set.
inline void requireImageClass(const Value &v, std::string_view fn,
                              std::string_view argName, int argNum,
                              unsigned mask)
{
    const ValueType t = v.type();
    if (t == ValueType::COMPLEX) {
        throw Error(std::string(fn) + ": input " + std::to_string(argNum)
                        + " (" + std::string(argName) + ") must be real — "
                        + "complex images are rejected here (got a "
                        + shapeOf(v) + " complex array).",
                    0, 0, std::string(fn), "",
                    "nk:" + std::string(fn) + ":expectedReal");
    }
    if (!classAllowed(v, mask)) {
        throw Error(std::string(fn) + ": input " + std::to_string(argNum)
                        + " (" + std::string(argName) + ") must be one of: "
                        + describeClasses(mask) + " (got a " + shapeOf(v)
                        + " " + mtypeName(t) + " array).",
                    0, 0, std::string(fn), "",
                    "nk:" + std::string(fn) + ":inputType");
    }
}

/// @brief Complex-capable gate for the four kernel families whose
/// reference COMPUTES complex values (imfilter/roifilt2 kernels,
/// im2col, imnoise, immse): class check as above but complex input
/// passes the gate — the caller either supports it or raises its own
/// :complexUnsupported stub.
inline void requireImageClassAllowComplex(const Value &v, std::string_view fn,
                                          std::string_view argName, int argNum,
                                          unsigned mask)
{
    if (v.type() == ValueType::COMPLEX)
        return;
    requireImageClass(v, fn, argName, argNum, mask);
}

/// @brief Complex-kernel stub (imfilter/roifilt2/imnoise): the
/// reference computes a complex result; this kernel is double-only, so
/// refuse instead of dropping imaginary parts
/// (bugs/opened/image/complex-kernels.md).
inline void requireRealKernelImage(const Value &v, std::string_view fn,
                                   std::string_view argName, int argNum)
{
    if (v.type() != ValueType::COMPLEX)
        return;
    throw Error(std::string(fn) + ": input " + std::to_string(argNum)
                    + " (" + std::string(argName) + ") is complex — complex "
                    + "kernels are not supported yet; the reference "
                    + "computes a complex result here, so refusing beats "
                    + "returning a wrong real one (got a " + shapeOf(v)
                    + " complex array).",
                0, 0, std::string(fn), "",
                "nk:" + std::string(fn) + ":complexUnsupported");
}

/// @brief Nonempty gate (imadjust's auto stretchlim, entropyfilt):
/// the reference rejects empty images with expectedNonempty.
inline void requireNonemptyImage(const Value &v, std::string_view fn,
                                 std::string_view argName, int argNum)
{
    if (v.numel() > 0)
        return;
    throw Error(std::string(fn) + ": input " + std::to_string(argNum)
                    + " (" + std::string(argName) + ") must be nonempty.",
                0, 0, std::string(fn), "",
                "nk:" + std::string(fn) + ":expectedNonempty");
}

/// @brief Real-valued check for DOUBLE-only auxiliary args (seeds,
/// P points, colors): complex → math:mustBeReal in the reference.
inline void requireRealDoubleArg(const Value &v, std::string_view fn,
                                 std::string_view argName, int argNum)
{
    if (v.type() == ValueType::COMPLEX)
        throw Error(std::string(fn) + ": input " + std::to_string(argNum)
                        + " (" + std::string(argName) + ") must be real.",
                    0, 0, std::string(fn), "",
                    "nk:" + std::string(fn) + ":mustBeReal");
    if (v.type() != ValueType::DOUBLE)
        throw Error(std::string(fn) + ": input " + std::to_string(argNum)
                        + " (" + std::string(argName) + ") must be double "
                        + "(got a " + mtypeName(v.type()) + " array).",
                    0, 0, std::string(fn), "",
                    "nk:" + std::string(fn) + ":inputType");
}

/// @brief Colormap gate (label2rgb/ind2rgb/ind2gray map, imfuse):
/// double M×3, real. single/uint8 maps are rejected by the reference
/// (invalidMapType / badMapMatrix — probed R2025b).
inline void requireColormap(const Value &v, std::string_view fn,
                            int argNum = 2)
{
    if (v.type() == ValueType::DOUBLE && !v.isComplex()
        && !v.dims().is3D() && v.dims().cols() == 3)
        return;
    throw Error(std::string(fn) + ": input " + std::to_string(argNum)
                    + " (map) must be a real double M-by-3 colormap "
                    + "(got a " + shapeOf(v) + " " + mtypeName(v.type())
                    + " array).",
                0, 0, std::string(fn), "",
                "nk:" + std::string(fn) + ":invalidColormap");
}

/// @brief Same-class pair gate for the quality family (immse/psnr/
/// ssim/multissim): the reference requires class(A) == class(B)
/// (images:validate:differentClassMatrices).
inline void requireSameClassPair(const Value &A, const Value &B,
                                 std::string_view fn)
{
    if (A.type() == B.type())
        return;
    throw Error(std::string(fn) + ": A and B must have the same class "
                    + "(got " + mtypeName(A.type()) + " and "
                    + mtypeName(B.type()) + ").",
                0, 0, std::string(fn), "",
                "nk:" + std::string(fn) + ":differentClass");
}

/// @brief Packed-binary gate (bwunpack): BWP must be uint32
/// (the reference rejects even logical with invalidType — probed).
inline void requireUint32Packed(const Value &v, std::string_view fn)
{
    if (v.type() == ValueType::UINT32)
        return;
    throw Error(std::string(fn) + ": input 1 (BWP) must be uint32 "
                    + "(got a " + mtypeName(v.type()) + " array).",
                0, 0, std::string(fn), "",
                "nk:" + std::string(fn) + ":inputType");
}

/// @brief Coerce a gated image (single / logical / integer / char when
/// the entry's contract allows code values, e.g. fsamp2) into a double
/// matrix of the same shape. Complex and non-coercible classes pass
/// through UNCHANGED so the caller's gate voices the rejection.
inline Value coerceToDoubleImage(const Value &v,
                                 std::pmr::memory_resource *mr = nullptr)
{
    const ValueType t = v.type();
    if (t == ValueType::DOUBLE || t == ValueType::COMPLEX)
        return v;
    if (t == ValueType::SINGLE || t == ValueType::LOGICAL
        || t == ValueType::CHAR || isIntegerType(t)) {
        const Dims &d = v.dims();
        Value out = d.is3D()
            ? Value::matrix3d(d.rows(), d.cols(), d.pages(),
                              ValueType::DOUBLE, mr)
            : Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
        const std::size_t n = v.numel();
        double *od = out.doubleDataMut();
        for (std::size_t i = 0; i < n; ++i)
            od[i] = v.elemAsDouble(i);
        return out;
    }
    return v;
}

/// @brief Binarise any real image to logical (nonzero → true) — the
/// bw* kernels' input view; used by the class-preserving wrappers.
inline Value coerceToLogicalImage(const Value &v,
                                  std::pmr::memory_resource *mr = nullptr)
{
    if (v.type() == ValueType::LOGICAL)
        return v;
    const Dims &d = v.dims();
    Value out = d.is3D()
        ? Value::matrix3d(d.rows(), d.cols(), d.pages(),
                          ValueType::LOGICAL, mr)
        : Value::matrix(d.rows(), d.cols(), ValueType::LOGICAL, mr);
    const std::size_t n = v.numel();
    uint8_t *od = out.logicalDataMut();
    for (std::size_t i = 0; i < n; ++i)
        od[i] = v.elemAsDouble(i) != 0.0 ? 1 : 0;
    return out;
}

/// @brief Cast a double result back to the input's class (the
/// class-preserving families: imhmax/imhmin/imimposemin, imclearborder,
/// imfill, imkeepborder, imfilter-on-logical, bwdist-to-single).
/// Saturating for integer classes (matches the reference's integer
/// image arithmetic); logical maps nonzero → true.
inline Value castResultToClass(const Value &result, ValueType target,
                               std::pmr::memory_resource *mr = nullptr)
{
    const ValueType rt = result.type();
    if (rt == target)
        return result;
    const std::size_t n = result.numel();
    Value out;
    const Dims &d = result.dims();
    if (d.is3D())
        out = Value::matrix3d(d.rows(), d.cols(), d.pages(), target, mr);
    else
        out = Value::matrix(d.rows(), d.cols(), target, mr);
    switch (target) {
        case ValueType::DOUBLE: {
            // The compute buffers are single in a few kernels; the
            // double-target copy must materialise (default would pass
            // the single result through unchanged).
            double *od = out.doubleDataMut();
            for (std::size_t i = 0; i < n; ++i)
                od[i] = result.elemAsDouble(i);
            break;
        }
        case ValueType::SINGLE: {
            float *od = out.singleDataMut();
            for (std::size_t i = 0; i < n; ++i)
                od[i] = static_cast<float>(result.elemAsDouble(i));
            break;
        }
        case ValueType::LOGICAL: {
            uint8_t *od = out.logicalDataMut();
            for (std::size_t i = 0; i < n; ++i)
                od[i] = result.elemAsDouble(i) != 0.0 ? 1 : 0;
            break;
        }
        case ValueType::UINT8: {
            uint8_t *od = out.uint8DataMut();
            for (std::size_t i = 0; i < n; ++i) {
                const double x = result.elemAsDouble(i);
                od[i] = static_cast<uint8_t>(x < 0.0 ? 0.0 : (x > 255.0 ? 255.0 : x));
            }
            break;
        }
        case ValueType::UINT16: {
            uint16_t *od = out.uint16DataMut();
            for (std::size_t i = 0; i < n; ++i) {
                const double x = result.elemAsDouble(i);
                od[i] = static_cast<uint16_t>(x < 0.0 ? 0.0 : (x > 65535.0 ? 65535.0 : x));
            }
            break;
        }
        case ValueType::INT16: {
            int16_t *od = out.int16DataMut();
            for (std::size_t i = 0; i < n; ++i) {
                const double x = result.elemAsDouble(i);
                od[i] = static_cast<int16_t>(x < -32768.0 ? -32768.0 : (x > 32767.0 ? 32767.0 : x));
            }
            break;
        }
        default:
            return result;   // double and everything else: pass through
    }
    return out;
}

/// @} group_image

} // namespace numkit::image::internal
