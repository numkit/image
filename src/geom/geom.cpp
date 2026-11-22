// toolboxes/image/src/geom/geom.cpp
//
// Geometric transforms (imresize, imresize3, imcrop, imcrop3, imrotate,
// imtranslate, axes2pix, impyramid), delegating numerical kernels to keel::image.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/complex.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/image.h>
#include <keel/memory.h>
#include <keel/ref.h>

#include <numkit/image/geom/geom.hpp>
#include <numkit/image/internal/type_gates.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>
#include "geom_detail.hpp"

#include <algorithm>
#include <cmath>

namespace numkit::image {

namespace {

struct Shape { keel::usize H, W, C; };

Shape shapeOf(const Value &A)
{
    const auto &d = A.dims();
    Shape s{};
    s.H = d.rows();
    s.W = d.cols();
    if (A.numel() == s.H * s.W)          s.C = 1;
    else if (A.numel() == s.H * s.W * 3) s.C = 3;
    else if (A.numel() == s.H * s.W * 4) s.C = 4;
    else
        throw Error("image geom: input must be H×W or H×W×{1,3,4}",
                    0, 0, "geom", "", "numkit:image:geom:shape");
    return s;
}

inline keel::usize idx3D(const Shape &s, keel::usize y, keel::usize x, keel::usize c)
{
    return c * s.H * s.W + x * s.H + y;
}

Value makeOut(keel::usize H, keel::usize W, keel::usize C, ValueType t, keel::Allocator *mr)
{
    if (C == 1) return Value::matrix(H, W, t, mr);
    return Value::matrix3d(H, W, C, t, mr);
}

void copyInToF64(const Value &A, keel::Array<keel::f64> &buf)
{
    const keel::usize n = A.numel();
    buf.resize(n);
    for (keel::usize i = 0; i < n; ++i) buf[i] = A.elemAsDouble(i);
}

void writeF64ToOut(keel::ArrayRef<keel::f64> src, Value &out, ValueType t)
{
    const keel::usize n = src.size();
    for (keel::usize i = 0; i < n; ++i) writeNative(out, i, src[i], t);
}

bool methodIsNearest(keel::StringRef m)
{
    return m == "nearest" || m == "Nearest" || m == "NEAREST" ||
           m == "near"    || m == "n";
}

keel::image::ResizeKernel parseResize2Kernel(keel::StringRef m)
{
    if (methodIsNearest(m)) return keel::image::ResizeKernel::kNearest;
    if (m == "bilinear" || m == "Bilinear" || m == "linear" || m == "Linear" ||
        m == "triangle" || m == "Triangle")
        return keel::image::ResizeKernel::kTriangle;
    return keel::image::ResizeKernel::kCubic;
}

keel::image::ResizeKernel parseResize3Kernel(keel::StringRef m)
{
    if (m == "nearest" || m == "Nearest")   return keel::image::ResizeKernel::kNearest;
    if (m == "box" || m == "Box")           return keel::image::ResizeKernel::kBox;
    if (m == "triangle" || m == "Triangle" ||
        m == "linear"   || m == "Linear"   ||
        m == "bilinear" || m == "Bilinear") return keel::image::ResizeKernel::kTriangle;
    if (m == "cubic" || m == "Cubic" ||
        m == "bicubic" || m == "Bicubic")   return keel::image::ResizeKernel::kCubic;
    if (m == "lanczos2" || m == "Lanczos2") return keel::image::ResizeKernel::kLanczos2;
    if (m == "lanczos3" || m == "Lanczos3") return keel::image::ResizeKernel::kLanczos3;
    throw Error(keel::format("imresize3: unknown method '{}'", m),
                0, 0, "imresize3", "", "numkit:imresize3:method");
}

void imresize_scale_argcheck(keel::f64 scale)
{
    if (!(scale > 0.0))
        throw Error("imresize: scale must be > 0",
                    0, 0, "imresize", "", "numkit:imresize:scale");
}

void imcrop3_argcheck(const Value &V, const Value &cuboid)
{
    if (cuboid.numel() != 6)
        throw Error("imcrop3: CUBOID must be a 6-element vector "
                    "[XMIN YMIN ZMIN WIDTH HEIGHT DEPTH]",
                    0, 0, "imcrop3", "", "numkit:imcrop3:cuboid");
    if (V.dims().ndims() < 3)
        throw Error("imcrop3: V must be at least 3-D",
                    0, 0, "imcrop3", "", "numkit:imcrop3:rank");
}

void axes2pix_argcheck(const Value &ext)
{
    internal::requireImageClassAllowComplex(ext, "axes2pix", "EXTENT", 2,
                                            internal::ImgClasses::FloatOnly);
    if (ext.numel() < 1)
        throw Error("axes2pix: EXTENT must be a non-empty vector",
                    0, 0, "axes2pix", "", "numkit:axes2pix:extent");
}

void impyramid_argcheck(keel::StringRef type, bool &reduce_out)
{
    if      (type == "reduce" || type == "Reduce") reduce_out = true;
    else if (type == "expand" || type == "Expand") reduce_out = false;
    else throw Error("impyramid: type must be 'reduce' or 'expand'",
                     0, 0, "impyramid", "", "numkit:impyramid:type");
}

void imresize3_argcheck(const Value &V)
{
    const auto &dims = V.dims();
    if (dims.ndims() < 2)
        throw Error("imresize3: V must be a 3-D volume",
                    0, 0, "imresize3", "", "numkit:imresize3:rank");
    const keel::usize inR = dims.dim(0);
    const keel::usize inC = (dims.ndims() >= 2) ? dims.dim(1) : 1;
    const keel::usize inD = (dims.ndims() >= 3) ? dims.dim(2) : 1;
    if (V.numel() != inR * inC * inD)
        throw Error("imresize3: V must be a 3-D numeric volume",
                    0, 0, "imresize3", "", "numkit:imresize3:shape");
}

void imresize3_scale_argcheck(keel::f64 scale)
{
    if (!(scale > 0.0))
        throw Error("imresize3: scale must be > 0",
                    0, 0, "imresize3", "", "numkit:imresize3:scale");
}

} // anonymous

Value imresize(const Value &A, keel::usize outH, keel::usize outW, keel::StringRef method, keel::Allocator *mr)
{
    const Shape s = shapeOf(A);
    const ValueType t = A.type();
    if (outH == 0 || outW == 0) return makeOut(outH, outW, s.C, t, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> in_buf(scratch.allocator());
    copyInToF64(A, in_buf);
    keel::Array<keel::f64> out_buf(outH * outW * s.C, 0.0, scratch.allocator());
    const auto k = parseResize2Kernel(method);
    (void)keel::image::resize_2d(in_buf.as_ref(), s.H, s.W, s.C, outH, outW,
                                 keel::as<keel::f64>(outH) / keel::as<keel::f64>(s.H),
                                 keel::as<keel::f64>(outW) / keel::as<keel::f64>(s.W),
                                 k, true, out_buf.as_mut_ref(), scratch.allocator());
    Value B = makeOut(outH, outW, s.C, t, mr);
    writeF64ToOut(out_buf.as_ref(), B, t);
    return B;
}

Value imresize(const Value &A, keel::f64 scale, keel::StringRef method, keel::Allocator *mr)
{
    const Shape s = shapeOf(A);
    imresize_scale_argcheck(scale);
    const keel::usize outH = keel::as<keel::usize>(std::round(scale * keel::as<keel::f64>(s.H)));
    const keel::usize outW = keel::as<keel::usize>(std::round(scale * keel::as<keel::f64>(s.W)));
    const ValueType t = A.type();
    if (outH == 0 || outW == 0) return makeOut(outH, outW, s.C, t, mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> in_buf(scratch.allocator());
    copyInToF64(A, in_buf);
    keel::Array<keel::f64> out_buf(outH * outW * s.C, 0.0, scratch.allocator());
    const auto k = parseResize2Kernel(method);
    (void)keel::image::resize_2d(in_buf.as_ref(), s.H, s.W, s.C, outH, outW,
                                 scale, scale, k, true, out_buf.as_mut_ref(),
                                 scratch.allocator());
    Value B = makeOut(outH, outW, s.C, t, mr);
    writeF64ToOut(out_buf.as_ref(), B, t);
    return B;
}

Value imcrop(const Value &A, keel::f64 xmin, keel::f64 ymin, keel::f64 width, keel::f64 height, keel::Allocator *mr)
{
    const Shape s = shapeOf(A);
    const ValueType t = A.type();
    keel::i32 x0 = std::max(keel::as<keel::i32>(std::round(xmin)) - 1, 0);
    keel::i32 y0 = std::max(keel::as<keel::i32>(std::round(ymin)) - 1, 0);
    keel::i32 w  = std::max(keel::as<keel::i32>(std::round(width))  + 1, 1);
    keel::i32 h  = std::max(keel::as<keel::i32>(std::round(height)) + 1, 1);
    if (x0 + w > keel::as<keel::i32>(s.W)) w = keel::as<keel::i32>(s.W) - x0;
    if (y0 + h > keel::as<keel::i32>(s.H)) h = keel::as<keel::i32>(s.H) - y0;
    if (w <= 0 || h <= 0)
        return makeOut(0, 0, s.C, t, mr);

    Value B = makeOut(keel::as<keel::usize>(h), keel::as<keel::usize>(w), s.C, t, mr);
    Shape sd{keel::as<keel::usize>(h), keel::as<keel::usize>(w), s.C};
    for (keel::usize c = 0; c < s.C; ++c)
        for (keel::i32 y = 0; y < h; ++y)
            for (keel::i32 x = 0; x < w; ++x) {
                const keel::f64 v = A.elemAsDouble(
                    idx3D(s, keel::as<keel::usize>(y0 + y), keel::as<keel::usize>(x0 + x), c));
                writeNative(B, idx3D(sd, keel::as<keel::usize>(y), keel::as<keel::usize>(x), c), v, t);
            }
    return B;
}

Value imcrop3(const Value &V, const Value &cuboid, keel::Allocator *mr)
{
    imcrop3_argcheck(V, cuboid);

    const auto &dV = V.dims();
    const keel::i32 nd = dV.ndims();
    const keel::usize H = dV.dim(0);
    const keel::usize W = dV.dim(1);
    const keel::usize D = dV.dim(2);
    const keel::usize T = (nd >= 4) ? dV.dim(3) : 1;

    auto rd = [&](keel::usize k) {
        return keel::as<keel::i64>(std::lround(cuboid.elemAsDouble(k)));
    };
    const keel::i64 xmin = rd(0);
    const keel::i64 ymin = rd(1);
    const keel::i64 zmin = rd(2);
    const keel::i64 xmax = keel::as<keel::i64>(std::lround(
        cuboid.elemAsDouble(0) + cuboid.elemAsDouble(3)));
    const keel::i64 ymax = keel::as<keel::i64>(std::lround(
        cuboid.elemAsDouble(1) + cuboid.elemAsDouble(4)));
    const keel::i64 zmax = keel::as<keel::i64>(std::lround(
        cuboid.elemAsDouble(2) + cuboid.elemAsDouble(5)));

    auto in_range = [](keel::i64 lo, keel::i64 hi, keel::usize N) {
        return lo >= 1 && lo <= keel::as<keel::i64>(N)
            && hi >= 1 && hi <= keel::as<keel::i64>(N) && lo <= hi;
    };
    if (!in_range(xmin, xmax, W) || !in_range(ymin, ymax, H) || !in_range(zmin, zmax, D))
        throw Error("imcrop3: cuboid is out of bounds of the input "
                    "volume — lower bound must be >= 1 and upper "
                    "bound must not exceed the image size",
                    0, 0, "imcrop3", "", "numkit:imcrop3:cropCuboidOutofBounds");

    const keel::usize outH = keel::as<keel::usize>(ymax - ymin + 1);
    const keel::usize outW = keel::as<keel::usize>(xmax - xmin + 1);
    const keel::usize outD = keel::as<keel::usize>(zmax - zmin + 1);

    const ValueType t = V.type();
    Value B;
    if (T == 1) {
        B = (outD == 1) ? Value::matrix(outH, outW, t, mr)
                        : Value::matrix3d(outH, outW, outD, t, mr);
    } else {
        const keel::usize dims4[4] = {outH, outW, outD, T};
        B = Value::matrixND(dims4, 4, t, mr);
    }

    auto src_lin = [&](keel::usize r, keel::usize c, keel::usize p, keel::usize tt) {
        return r + H * c + H * W * p + H * W * D * tt;
    };
    auto dst_lin = [&](keel::usize r, keel::usize c, keel::usize p, keel::usize tt) {
        return r + outH * c + outH * outW * p + outH * outW * outD * tt;
    };

    auto write_one = [&](keel::usize di, keel::usize si) {
        switch (t) {
            case ValueType::DOUBLE:  B.doubleDataMut()[di]  = V.doubleData()[si];  break;
            case ValueType::SINGLE:  B.singleDataMut()[di]  = V.singleData()[si];  break;
            case ValueType::UINT8:   B.uint8DataMut()[di]   = V.uint8Data()[si];   break;
            case ValueType::UINT16:  B.uint16DataMut()[di]  = V.uint16Data()[si];  break;
            case ValueType::UINT32:  B.uint32DataMut()[di]  = V.uint32Data()[si];  break;
            case ValueType::UINT64:  B.uint64DataMut()[di]  = V.uint64Data()[si];  break;
            case ValueType::INT8:    B.int8DataMut()[di]    = V.int8Data()[si];    break;
            case ValueType::INT16:   B.int16DataMut()[di]   = V.int16Data()[si];   break;
            case ValueType::INT32:   B.int32DataMut()[di]   = V.int32Data()[si];   break;
            case ValueType::INT64:   B.int64DataMut()[di]   = V.int64Data()[si];   break;
            case ValueType::LOGICAL: B.logicalDataMut()[di] = V.logicalData()[si]; break;
            default:
                throw Error("imcrop3: unsupported class for V",
                            0, 0, "imcrop3", "", "numkit:imcrop3:cls");
        }
    };

    const keel::usize y0 = keel::as<keel::usize>(ymin - 1);
    const keel::usize x0 = keel::as<keel::usize>(xmin - 1);
    const keel::usize z0 = keel::as<keel::usize>(zmin - 1);

    for (keel::usize tt = 0; tt < T; ++tt)
        for (keel::usize p = 0; p < outD; ++p)
            for (keel::usize c = 0; c < outW; ++c)
                for (keel::usize r = 0; r < outH; ++r)
                    write_one(dst_lin(r, c, p, tt), src_lin(y0 + r, x0 + c, z0 + p, tt));
    return B;
}

namespace {

keel::image::RotateInterp parseRotateInterp(keel::StringRef m)
{
    if (methodIsNearest(m)) return keel::image::RotateInterp::kNearest;
    if (m == "bicubic" || m == "Bicubic" || m == "BICUBIC" ||
        m == "cubic"   || m == "Cubic"   || m == "CUBIC")
        return keel::image::RotateInterp::kBicubic;
    return keel::image::RotateInterp::kBilinear;
}

} // anonymous

Value imrotate(const Value &A, keel::f64 angle, keel::StringRef method, keel::StringRef bbox, keel::Allocator *mr)
{
    const Shape s = shapeOf(A);
    const ValueType t = A.type();
    const auto interp = parseRotateInterp(method);
    const bool isCrop = (bbox == "crop" || bbox == "Crop");

    keel::usize outH = 0, outW = 0;
    keel::image::rotate_2d_dims(s.H, s.W, angle, isCrop, outH, outW);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> in_buf(scratch.allocator());
    copyInToF64(A, in_buf);
    keel::Array<keel::f64> out_buf(outH * outW * s.C, 0.0, scratch.allocator());
    (void)keel::image::rotate_2d(in_buf.as_ref(), s.H, s.W, s.C, angle, interp, isCrop,
                                 out_buf.as_mut_ref(), outH, outW);
    Value B = makeOut(outH, outW, s.C, t, mr);
    writeF64ToOut(out_buf.as_ref(), B, t);
    return B;
}

Value imtranslate(const Value &A, keel::f64 dx, keel::f64 dy, keel::Allocator *mr)
{
    const Shape s = shapeOf(A);
    const ValueType t = A.type();
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> in_buf(scratch.allocator());
    copyInToF64(A, in_buf);
    keel::Array<keel::f64> out_buf(s.H * s.W * s.C, 0.0, scratch.allocator());
    (void)keel::image::translate_2d(in_buf.as_ref(), s.H, s.W, s.C, dx, dy, out_buf.as_mut_ref());
    Value B = makeOut(s.H, s.W, s.C, t, mr);
    writeF64ToOut(out_buf.as_ref(), B, t);
    return B;
}

Value axes2pix(keel::f64 n, const Value &extent, const Value &axesCoord, keel::Allocator *mr)
{
    Value extentCoerced;
    const Value *extP = &extent;
    if (extent.type() == ValueType::CHAR) {
        extentCoerced = internal::coerceToDoubleImage(extent, mr);
        extP = &extentCoerced;
    }
    const Value &ext = *extP;
    axes2pix_argcheck(ext);

    const auto &d = axesCoord.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    const keel::usize N = axesCoord.numel();

    if (ext.isComplex() || axesCoord.isComplex()) {
        auto getC = [](const Value &v, keel::usize idx) -> keel::c64 {
            return v.isComplex() ? v.complexData()[idx] : keel::c64(v.elemAsDouble(idx), 0.0);
        };
        const keel::c64 e0 = getC(ext, 0);
        const keel::c64 e1 = getC(ext, ext.numel() - 1);
        const bool degenerate = (n == 1.0) || (e0 == e1);
        const keel::c64 pixelWidth = degenerate ? keel::c64(1.0, 0.0) : (e1 - e0) / (n - 1.0);
        Value out = d.is3D()
            ? Value::matrix3d(H, W, d.pages(), ValueType::COMPLEX, mr)
            : Value::matrix(H, W, ValueType::COMPLEX, mr);
        keel::c64 *od = out.complexDataMut();
        for (keel::usize i = 0; i < N; ++i) {
            const keel::c64 a = getC(axesCoord, i);
            od[i] = degenerate ? (a - e0 + 1.0) : ((a - e0) / pixelWidth + 1.0);
        }
        return out;
    }

    const keel::f64 e0 = ext.elemAsDouble(0);
    const keel::f64 e1 = ext.elemAsDouble(ext.numel() - 1);
    const bool degenerate = (n == 1.0) || (e0 == e1);
    const keel::f64 pixelWidth = degenerate ? 1.0 : (e1 - e0) / (n - 1.0);

    Value out = d.is3D()
        ? Value::matrix3d(H, W, d.pages(), ValueType::DOUBLE, mr)
        : Value::matrix(H, W, ValueType::DOUBLE, mr);
    if (axesCoord.numel() == 0)
        return internal::castResultToClass(out, ext.type(), mr);

    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 a = axesCoord.elemAsDouble(i);
        od[i] = degenerate ? (a - e0 + 1.0) : ((a - e0) / pixelWidth + 1.0);
    }
    return internal::castResultToClass(out, ext.type(), mr);
}

Value impyramid(const Value &A, keel::StringRef type, keel::Allocator *mr)
{
    const Shape s = shapeOf(A);
    bool reduce = false;
    impyramid_argcheck(type, reduce);
    const ValueType t = A.type();

    const keel::usize Hout = reduce ? (s.H + 1) / 2 : (s.H ? 2 * s.H - 1 : 0);
    const keel::usize Wout = reduce ? (s.W + 1) / 2 : (s.W ? 2 * s.W - 1 : 0);
    Value B = makeOut(Hout, Wout, s.C, t, mr);
    if (s.H == 0 || s.W == 0 || Hout == 0 || Wout == 0) return B;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> in_buf(scratch.allocator());
    copyInToF64(A, in_buf);
    keel::Array<keel::f64> out_buf(Hout * Wout * s.C, 0.0, scratch.allocator());
    if (reduce) {
        (void)keel::image::pyramid_reduce_2d(in_buf.as_ref(), s.H, s.W, s.C,
                                             out_buf.as_mut_ref(), scratch.allocator());
    } else {
        (void)keel::image::pyramid_expand_2d(in_buf.as_ref(), s.H, s.W, s.C,
                                             out_buf.as_mut_ref(), scratch.allocator());
    }
    writeF64ToOut(out_buf.as_ref(), B, t);
    return B;
}

Value imresize3_core(const Value &V,
                     keel::usize outR, keel::usize outC, keel::usize outD,
                     keel::f64 scaleY, keel::f64 scaleX, keel::f64 scaleZ,
                     keel::StringRef method, bool antialias,
                     keel::Allocator *mr)
{
    imresize3_argcheck(V);
    const auto &dims = V.dims();
    const keel::usize inR = dims.dim(0);
    const keel::usize inC = (dims.ndims() >= 2) ? dims.dim(1) : 1;
    const keel::usize inD = (dims.ndims() >= 3) ? dims.dim(2) : 1;

    const ValueType T = V.type();
    if (outR == 0 || outC == 0 || outD == 0)
        return Value::matrix3d(outR, outC, outD, T, mr);

    const auto k = parseResize3Kernel(method);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> in_buf(scratch.allocator());
    copyInToF64(V, in_buf);
    keel::Array<keel::f64> out_buf(outR * outC * outD, 0.0, scratch.allocator());
    (void)keel::image::resize_3d(in_buf.as_ref(), inR, inC, inD, outR, outC, outD,
                                 scaleY, scaleX, scaleZ, k, antialias,
                                 out_buf.as_mut_ref(), scratch.allocator());
    Value B = Value::matrix3d(outR, outC, outD, T, mr);
    writeF64ToOut(out_buf.as_ref(), B, T);
    return B;
}

Value imresize3(const Value &V,
                keel::usize outR, keel::usize outC, keel::usize outD,
                keel::StringRef method, bool antialias,
                keel::Allocator *mr)
{
    const auto &dims = V.dims();
    const keel::usize inR = dims.dim(0);
    const keel::usize inC = (dims.ndims() >= 2) ? dims.dim(1) : 1;
    const keel::usize inD = (dims.ndims() >= 3) ? dims.dim(2) : 1;
    return imresize3_core(V, outR, outC, outD,
        keel::as<keel::f64>(outR) / keel::as<keel::f64>(inR ? inR : 1),
        keel::as<keel::f64>(outC) / keel::as<keel::f64>(inC ? inC : 1),
        keel::as<keel::f64>(outD) / keel::as<keel::f64>(inD ? inD : 1),
        method, antialias, mr);
}

Value imresize3(const Value &V, keel::f64 scale,
                keel::StringRef method, bool antialias,
                keel::Allocator *mr)
{
    imresize3_scale_argcheck(scale);
    const auto &dims = V.dims();
    const keel::usize inR = dims.dim(0);
    const keel::usize inC = (dims.ndims() >= 2) ? dims.dim(1) : 1;
    const keel::usize inD = (dims.ndims() >= 3) ? dims.dim(2) : 1;
    return imresize3_core(V,
        keel::as<keel::usize>(std::round(scale * keel::as<keel::f64>(inR))),
        keel::as<keel::usize>(std::round(scale * keel::as<keel::f64>(inC))),
        keel::as<keel::usize>(std::round(scale * keel::as<keel::f64>(inD))),
        scale, scale, scale, method, antialias, mr);
}

} // namespace numkit::image
