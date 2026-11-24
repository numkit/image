// toolboxes/image/src/segment/segment.cpp
//
// Thin Value/argcheck wrappers delegating dice, jaccard, boundarymask,
// grayconnected, graydiffweight, gradientweight, regionfill, poly2mask,
// graydist, bwdistgeodesic, reducepoly, and watershed to keel::image.

#include <keel/core.h>
#include <keel/algo.h>
#include <keel/format.h>
#include <keel/image.h>
#include <keel/memory.h>
#include <keel/array.h>
#include <keel/math.h>
#include <keel/static_array.h>

#include <numkit/image/segment/segment.hpp>
#include <numkit/image/internal/type_gates.hpp>
#include <numkit/image/geom/geom.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

namespace numkit::image {

namespace {

void requireSameShape(const Value &A, const Value &B, const char *fn)
{
    if (A.numel() != B.numel() || A.dims().rows() != B.dims().rows()
        || A.dims().cols() != B.dims().cols())
        throw Error(keel::format("{}: inputs must have the same shape", fn),
                    0, 0, fn, "", "numkit:image:segment:shape");
}

keel::Array<keel::u8> extract_bw(const Value &V, keel::Allocator *mr)
{
    const keel::usize N = V.numel();
    keel::Array<keel::u8> out(N, mr);
    if (V.isLogical()) {
        const keel::u8 *src = V.logicalData();
        for (keel::usize i = 0; i < N; ++i) out[i] = src[i] ? 1u : 0u;
    } else {
        for (keel::usize i = 0; i < N; ++i) out[i] = (V.elemAsDouble(i) != 0.0) ? 1u : 0u;
    }
    return out;
}

keel::Array<keel::f64> extract_double(const Value &V, keel::Allocator *mr)
{
    const keel::usize N = V.numel();
    keel::Array<keel::f64> out(N, mr);
    if (V.type() == ValueType::DOUBLE) {
        keel::copy_n(out.data(), V.doubleData(), N);
    } else {
        for (keel::usize i = 0; i < N; ++i) out[i] = V.elemAsDouble(i);
    }
    return out;
}

keel::image::GeodesicMetric parse_geodesic_metric(keel::StringRef method, const char *fn)
{
    if (method == "cityblock")       return keel::image::GeodesicMetric::kCityblock;
    if (method == "chessboard")      return keel::image::GeodesicMetric::kChessboard;
    if (method == "quasi-euclidean") return keel::image::GeodesicMetric::kQuasiEuclidean;
    throw Error(keel::format("{}: METHOD must be 'cityblock', 'chessboard', or 'quasi-euclidean'",
                             fn),
                0, 0, fn, "", keel::format("numkit:{}:method", fn));
}

void watershed_argcheck(const Value &A, keel::i32 conn)
{
    internal::requireImageClass(A, "watershed", "A", 1, internal::ImgClasses::AnyNumericReal);
    if (A.dims().is3D())
        throw Error("watershed: input 1 (A) must be a 2-D matrix.",
                    0, 0, "watershed", "", "numkit:watershed:expected2D");
    for (keel::usize i = 0; i < A.numel(); ++i) {
        if (keel::is_nan(A.elemAsDouble(i)))
            throw Error("watershed: input 1 (A) must be non-NaN (got NaN).",
                        0, 0, "watershed", "", "numkit:watershed:expectedNonNaN");
    }
    if (conn != 4 && conn != 8)
        throw Error(keel::format("watershed: input 2 (conn) must be 4 or 8 (got {}).", conn),
                    0, 0, "watershed", "", "numkit:watershed:badConn");
}

} // anonymous

Value dice(const Value &A, const Value &B, keel::Allocator *mr)
{
    requireSameShape(A, B, "dice");
    auto ba = extract_bw(A, mr);
    auto bb = extract_bw(B, mr);
    return Value::scalar(keel::image::dice_coeff(ba.as_ref(), bb.as_ref()), mr);
}

Value jaccard(const Value &A, const Value &B, keel::Allocator *mr)
{
    requireSameShape(A, B, "jaccard");
    auto ba = extract_bw(A, mr);
    auto bb = extract_bw(B, mr);
    return Value::scalar(keel::image::jaccard_index(ba.as_ref(), bb.as_ref()), mr);
}

Value boundarymask(const Value &L, keel::i32 conn, keel::Allocator *mr)
{
    internal::requireImageClass(L, "boundarymask", "L", 1, internal::ImgClasses::AnyNumericReal);
    const keel::usize H = L.dims().rows(), W = L.dims().cols();
    Value out = Value::matrix(H, W, ValueType::LOGICAL, mr);
    if (H == 0 || W == 0) return out;
    keel::Array<keel::i64> labs(H * W, mr);
    for (keel::usize i = 0; i < H * W; ++i)
        labs[i] = keel::as<keel::i64>(keel::round(L.elemAsDouble(i)));
    (void)keel::image::boundarymask_2d(labs.as_ref(), H, W, conn,
                                       keel::MutArrayRef<keel::u8>(out.logicalDataMut(), H * W));
    return out;
}

Value label2idx(const Value &L, keel::Allocator *mr)
{
    const keel::usize N = L.numel();
    keel::i64 maxLab = 0;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::i64 v = keel::as<keel::i64>(keel::round(L.elemAsDouble(i)));
        if (v > maxLab) maxLab = v;
    }
    Value cell = Value::cell(keel::as<keel::usize>(maxLab), 1, mr);
    if (maxLab == 0) return cell;

    keel::Array<keel::usize> counts(keel::as<keel::usize>(maxLab) + 1, 0, mr);
    for (keel::usize i = 0; i < N; ++i) {
        const keel::i64 v = keel::as<keel::i64>(keel::round(L.elemAsDouble(i)));
        if (v > 0) ++counts[keel::as<keel::usize>(v)];
    }
    keel::Array<Value> bufs;
    bufs.reserve(keel::as<keel::usize>(maxLab));
    keel::Array<keel::f64 *> wptr(keel::as<keel::usize>(maxLab) + 1, nullptr, mr);
    keel::Array<keel::usize> cur(keel::as<keel::usize>(maxLab) + 1, 0, mr);
    for (keel::i64 k = 1; k <= maxLab; ++k) {
        const keel::usize cnt = counts[keel::as<keel::usize>(k)];
        Value v = Value::matrix(cnt, 1, ValueType::DOUBLE, mr);
        wptr[keel::as<keel::usize>(k)] = (cnt > 0) ? v.doubleDataMut() : nullptr;
        bufs.push_back(keel::move(v));
    }
    for (keel::usize i = 0; i < N; ++i) {
        const keel::i64 v = keel::as<keel::i64>(keel::round(L.elemAsDouble(i)));
        if (v > 0) {
            const keel::usize u = keel::as<keel::usize>(v);
            wptr[u][cur[u]++] = keel::as<keel::f64>(i + 1);
        }
    }
    for (keel::i64 k = 1; k <= maxLab; ++k)
        cell.cellAt(keel::as<keel::usize>(k - 1)) = keel::move(bufs[keel::as<keel::usize>(k - 1)]);
    return cell;
}

Value grayconnected(const Value &I, keel::i32 row, keel::i32 col, keel::f64 tol,
                    keel::Allocator *mr)
{
    internal::requireImageClass(I, "grayconnected", "I", 1, internal::ImgClasses::FloatIntNoLog);
    const keel::usize H = I.dims().rows(), W = I.dims().cols();
    Value out = Value::matrix(H, W, ValueType::LOGICAL, mr);
    if (H == 0 || W == 0) return out;

    const keel::i32 r0 = row - 1, c0 = col - 1;
    if (r0 < 0 || r0 >= keel::as<keel::i32>(H) || c0 < 0 || c0 >= keel::as<keel::i32>(W))
        throw Error("grayconnected: seed out of bounds",
                    0, 0, "grayconnected", "", "numkit:grayconnected:seed");
    if (tol < 0.0) {
        switch (I.type()) {
            case ValueType::UINT8:
            case ValueType::INT8:   tol = 32.0; break;
            case ValueType::UINT16:
            case ValueType::INT16:  tol = 32.0 * 256.0; break;
            default:                tol = 0.125; break;
        }
    }
    auto in = extract_double(I, mr);
    (void)keel::image::grayconnected_2d(
        in.as_ref(), H, W, r0, c0, tol,
        keel::MutArrayRef<keel::u8>(out.logicalDataMut(), H * W), mr);
    return out;
}

Value graydiffweight(const Value &I, keel::f64 ref_gray_val,
                     keel::f64 rolloff_factor, keel::f64 cutoff, keel::Allocator *mr)
{
    if (!(rolloff_factor > 0.0))
        throw Error("graydiffweight: RolloffFactor must be positive",
                    0, 0, "graydiffweight", "", "numkit:graydiffweight:rolloff");
    if (!(cutoff >= 0.0))
        throw Error("graydiffweight: GrayDifferenceCutoff must be non-negative",
                    0, 0, "graydiffweight", "", "numkit:graydiffweight:cutoff");

    const ValueType outT = (I.type() == ValueType::SINGLE) ? ValueType::SINGLE : ValueType::DOUBLE;
    const auto &d = I.dims();
    const keel::usize N = I.numel();
    Value W = d.is3D() ? Value::matrix3d(d.rows(), d.cols(), d.pages(), outT, mr)
                       : Value::matrix(d.rows(), d.cols(), outT, mr);
    if (N == 0) return W;

    auto in = extract_double(I, mr);
    keel::Array<keel::f64> wbuf(N, 0.0, mr);
    (void)keel::image::graydiffweight_f64(in.as_ref(), ref_gray_val, rolloff_factor, cutoff,
                                          wbuf.as_mut_ref());
    if (outT == ValueType::DOUBLE) {
        keel::copy_n(W.doubleDataMut(), wbuf.data(), N);
    } else {
        keel::f32 *p = W.singleDataMut();
        for (keel::usize i = 0; i < N; ++i) p[i] = keel::as<keel::f32>(wbuf[i]);
    }
    return W;
}

Value gradientweight(const Value &I, keel::f64 sigma_x, keel::f64 sigma_y,
                     keel::f64 rolloff_factor, keel::f64 weight_cutoff, keel::Allocator *mr)
{
    internal::requireImageClass(I, "gradientweight", "I", 1, internal::ImgClasses::FloatIntNoLog);
    if (!(sigma_x > 0.0) || !keel::is_finite(sigma_x)
        || !(sigma_y > 0.0) || !keel::is_finite(sigma_y))
        throw Error("gradientweight: sigma must be positive and finite",
                    0, 0, "gradientweight", "", "numkit:gradientweight:sigma");
    if (!(rolloff_factor > 0.0) || !keel::is_finite(rolloff_factor))
        throw Error("gradientweight: RolloffFactor must be positive and finite",
                    0, 0, "gradientweight", "", "numkit:gradientweight:rolloff");
    if (!(weight_cutoff >= 1e-3 && weight_cutoff <= 1.0))
        throw Error("gradientweight: WeightCutoff must be in [1e-3, 1]",
                    0, 0, "gradientweight", "", "numkit:gradientweight:cutoff");
    if (I.dims().is3D())
        throw Error("gradientweight: 3-D inputs not supported (slice and call per page)",
                    0, 0, "gradientweight", "", "numkit:gradientweight:dim");

    const ValueType outT = (I.type() == ValueType::SINGLE) ? ValueType::SINGLE : ValueType::DOUBLE;
    const keel::usize H = I.dims().rows(), W_ = I.dims().cols(), N = H * W_;
    Value Wout = Value::matrix(H, W_, outT, mr);
    if (N == 0) return Wout;

    auto in = extract_double(I, mr);
    keel::Array<keel::f64> wbuf(N, 0.0, mr);
    (void)keel::image::gradientweight_2d(in.as_ref(), H, W_, sigma_x, sigma_y,
                                         rolloff_factor, weight_cutoff, wbuf.as_mut_ref(), mr);
    if (outT == ValueType::DOUBLE) {
        keel::copy_n(Wout.doubleDataMut(), wbuf.data(), N);
    } else {
        keel::f32 *p = Wout.singleDataMut();
        for (keel::usize i = 0; i < N; ++i) p[i] = keel::as<keel::f32>(wbuf[i]);
    }
    return Wout;
}

Value regionfill(const Value &I, const Value &mask, keel::Allocator *mr)
{
    internal::requireImageClass(I, "regionfill", "I", 1, internal::ImgClasses::FloatInt);
    if (I.dims().is3D())
        throw Error("regionfill: I must be 2-D", 0, 0, "regionfill", "", "numkit:regionfill:mustBe2D");
    const keel::usize H = I.dims().rows(), W = I.dims().cols();
    if (H < 3 || W < 3)
        throw Error("regionfill: I must be at least 3x3",
                    0, 0, "regionfill", "", "numkit:regionfill:mustBeLargerThan2by2");
    if (mask.dims().rows() != H || mask.dims().cols() != W || mask.dims().is3D())
        throw Error("regionfill: MASK must be 2-D and the same size as I",
                    0, 0, "regionfill", "", "numkit:regionfill:mustBeSameSizeAsI");

    const keel::usize N = H * W;
    keel::Array<keel::u8> M(N, 0, mr);
    if (mask.isLogical()) {
        const keel::u8 *src = mask.logicalData();
        for (keel::usize i = 0; i < N; ++i) M[i] = src[i] ? 1u : 0u;
    } else {
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 v = mask.elemAsDouble(i);
            if (keel::is_nan(v))
                throw Error("regionfill: MASK cannot contain NaN",
                            0, 0, "regionfill", "", "numkit:regionfill:maskNaN");
            M[i] = (v != 0.0) ? 1u : 0u;
        }
    }

    auto idata = extract_double(I, mr);
    keel::Array<keel::f64> jout(N, 0.0, mr);
    (void)keel::image::regionfill_2d(idata.as_ref(), M.as_ref(), H, W, jout.as_mut_ref(), mr);

    const ValueType inT = I.type();
    Value J = Value::matrix(H, W, inT, mr);
    if (inT == ValueType::DOUBLE) {
        keel::copy_n(J.doubleDataMut(), jout.data(), N);
    } else if (inT == ValueType::SINGLE) {
        for (keel::usize i = 0; i < N; ++i) J.singleDataMut()[i] = keel::as<keel::f32>(jout[i]);
    } else {
        auto saturate = [](keel::f64 v, keel::f64 lo, keel::f64 hi) {
            v = keel::round(v);
            return keel::clamp(v, lo, hi);
        };
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 v = jout[i];
            switch (inT) {
                case ValueType::UINT8:  J.uint8DataMut()[i]  = keel::as<keel::u8>(saturate(v, 0.0, 255.0)); break;
                case ValueType::UINT16: J.uint16DataMut()[i] = keel::as<keel::u16>(saturate(v, 0.0, 65535.0)); break;
                case ValueType::UINT32: J.uint32DataMut()[i] = keel::as<keel::u32>(saturate(v, 0.0, 4294967295.0)); break;
                case ValueType::INT8:   J.int8DataMut()[i]   = keel::as<keel::i8>(saturate(v, -128.0, 127.0)); break;
                case ValueType::INT16:  J.int16DataMut()[i]  = keel::as<keel::i16>(saturate(v, -32768.0, 32767.0)); break;
                case ValueType::INT32:  J.int32DataMut()[i]  = keel::as<keel::i32>(saturate(v, -2147483648.0, 2147483647.0)); break;
                default:                J.doubleDataMut()[i] = v; break;
            }
        }
    }
    return J;
}

Value poly2mask(const Value &X, const Value &Y, keel::usize M, keel::usize N, keel::Allocator *mr)
{
    if (X.numel() != Y.numel())
        throw Error("poly2mask: X and Y must have the same length",
                    0, 0, "poly2mask", "", "numkit:poly2mask:vectorSizeMismatch");
    Value BW = Value::matrix(M, N, ValueType::LOGICAL, mr);
    if (X.numel() == 0 || M == 0 || N == 0) return BW;
    auto vx = extract_double(X, mr);
    auto vy = extract_double(Y, mr);
    (void)keel::image::poly2mask_2d(vx.as_ref(), vy.as_ref(), M, N,
                                    keel::MutArrayRef<keel::u8>(BW.logicalDataMut(), M * N), mr);
    return BW;
}

Value roipoly(keel::f64 xdata_lo, keel::f64 xdata_hi,
              keel::f64 ydata_lo, keel::f64 ydata_hi,
              keel::usize M, keel::usize N,
              const Value &xi, const Value &yi, keel::Allocator *mr)
{
    if (xi.numel() != yi.numel())
        throw Error("roipoly: xi and yi must have the same length",
                    0, 0, "roipoly", "", "numkit:roipoly:xiyiMustBeSameLength");
    const keel::usize n0 = xi.numel();
    keel::Array<keel::f64> xc(mr), yc(mr);
    xc.reserve(n0 + 1); yc.reserve(n0 + 1);
    for (keel::usize i = 0; i < n0; ++i) {
        xc.push_back(xi.elemAsDouble(i));
        yc.push_back(yi.elemAsDouble(i));
    }
    if (n0 > 0 && (xc.front() != xc.back() || yc.front() != yc.back())) {
        xc.push_back(xc.front()); yc.push_back(yc.front());
    }
    Value xExt = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    xExt.doubleDataMut()[0] = xdata_lo; xExt.doubleDataMut()[1] = xdata_hi;
    Value yExt = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    yExt.doubleDataMut()[0] = ydata_lo; yExt.doubleDataMut()[1] = ydata_hi;
    const keel::usize nc = xc.size();
    Value xClosed = Value::matrix(nc, 1, ValueType::DOUBLE, mr);
    Value yClosed = Value::matrix(nc, 1, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < nc; ++i) {
        xClosed.doubleDataMut()[i] = xc[i];
        yClosed.doubleDataMut()[i] = yc[i];
    }
    Value roix = axes2pix(keel::as<keel::f64>(N), xExt, xClosed, mr);
    Value roiy = axes2pix(keel::as<keel::f64>(M), yExt, yClosed, mr);
    return poly2mask(roix, roiy, M, N, mr);
}

Value graydist(const Value &I, const Value &seeds, keel::StringRef method, keel::Allocator *mr)
{
    internal::requireImageClass(I, "graydist", "A", 1, internal::ImgClasses::AnyNumericReal);
    if (!seeds.isEmpty())
        internal::requireRealDoubleArg(seeds, "graydist", "IND", 2);
    if (I.dims().is3D())
        throw Error("graydist: I must be 2-D", 0, 0, "graydist", "", "numkit:graydist:dim");
    const auto metric = parse_geodesic_metric(method, "graydist");

    const keel::usize H = I.dims().rows(), W = I.dims().cols(), N = H * W;
    const ValueType outT = (I.type() == ValueType::DOUBLE) ? ValueType::DOUBLE : ValueType::SINGLE;
    Value T = Value::matrix(H, W, outT, mr);

    keel::Array<keel::usize> sidx(mr);
    sidx.reserve(seeds.numel());
    for (keel::usize k = 0; k < seeds.numel(); ++k) {
        const keel::f64 sv = seeds.elemAsDouble(k);
        if (!(sv >= 1) || sv != keel::floor(sv))
            throw Error("graydist: seed indices must be positive integers",
                        0, 0, "graydist", "", "numkit:graydist:seed");
        const keel::usize idx = keel::as<keel::usize>(sv) - 1;
        if (idx >= N)
            throw Error("graydist: seed index out of bounds",
                        0, 0, "graydist", "", "numkit:graydist:seedOOB");
        sidx.push_back(idx);
    }

    auto idata = extract_double(I, mr);
    keel::Array<keel::f64> dist(N, 0.0, mr);
    (void)keel::image::graydist_2d(idata.as_ref(), H, W, sidx.as_ref(), metric, dist.as_mut_ref(),
                                   mr);
    if (outT == ValueType::DOUBLE) {
        keel::copy_n(T.doubleDataMut(), dist.data(), N);
    } else {
        keel::f32 *p = T.singleDataMut();
        for (keel::usize i = 0; i < N; ++i) p[i] = keel::as<keel::f32>(dist[i]);
    }
    return T;
}

Value bwdistgeodesic(const Value &BW, const Value &seeds, keel::StringRef method,
                     keel::Allocator *mr)
{
    internal::requireImageClass(BW, "bwdistgeodesic", "BW", 1, internal::ImgClasses::LogicalOnly);
    if (BW.dims().is3D())
        throw Error("bwdistgeodesic: BW must be 2-D",
                    0, 0, "bwdistgeodesic", "", "numkit:bwdistgeodesic:dim");
    const auto metric = parse_geodesic_metric(method, "bwdistgeodesic");

    const keel::usize H = BW.dims().rows(), W = BW.dims().cols(), N = H * W;
    Value D = Value::matrix(H, W, ValueType::SINGLE, mr);
    if (N == 0) return D;

    keel::Array<keel::usize> sidx(mr);
    sidx.reserve(seeds.numel());
    for (keel::usize k = 0; k < seeds.numel(); ++k) {
        const keel::f64 sv = seeds.elemAsDouble(k);
        if (!(sv >= 1) || sv != keel::floor(sv))
            throw Error("bwdistgeodesic: seed indices must be positive integers",
                        0, 0, "bwdistgeodesic", "", "numkit:bwdistgeodesic:seed");
        const keel::usize idx = keel::as<keel::usize>(sv) - 1;
        if (idx >= N)
            throw Error("bwdistgeodesic: seed index out of bounds",
                        0, 0, "bwdistgeodesic", "", "numkit:bwdistgeodesic:seedOOB");
        sidx.push_back(idx);
    }

    auto mask = extract_bw(BW, mr);
    keel::Array<keel::f64> dist(N, 0.0, mr);
    (void)keel::image::bwdistgeodesic_2d(mask.as_ref(), H, W, sidx.as_ref(), metric,
                                         dist.as_mut_ref(), mr);
    keel::f32 *dp = D.singleDataMut();
    for (keel::usize i = 0; i < N; ++i) dp[i] = keel::as<keel::f32>(dist[i]);
    return D;
}

Value imoverlay(const Value &I, const Value &BW, const Value &color, keel::Allocator *mr)
{
    internal::requireImageClass(I, "imoverlay", "A", 1, internal::ImgClasses::FloatIntLog);
    internal::requireImageClass(BW, "imoverlay", "BW", 2, internal::ImgClasses::AnyNumericReal);
    internal::requireImageClass(color, "imoverlay", "color", 3,
                                internal::ImgClasses::AnyNumericReal);
    if (color.numel() != 3)
        throw Error("imoverlay: color must be a 1×3 RGB triple",
                    0, 0, "imoverlay", "", "numkit:imoverlay:color");
    const keel::usize H = I.dims().rows(), W = I.dims().cols();
    if (BW.dims().rows() != H || BW.dims().cols() != W)
        throw Error("imoverlay: BW must match the H × W of I",
                    0, 0, "imoverlay", "", "numkit:imoverlay:shape");

    bool isRGB;
    if (I.numel() == H * W) isRGB = false;
    else if (I.numel() == H * W * 3) isRGB = true;
    else
        throw Error("imoverlay: I must be H × W or H × W × 3",
                    0, 0, "imoverlay", "", "numkit:imoverlay:shape");

    keel::f64 cIn[3] = {color.elemAsDouble(0), color.elemAsDouble(1), color.elemAsDouble(2)};
    const bool floatColour = (cIn[0] <= 1.0 && cIn[1] <= 1.0 && cIn[2] <= 1.0);
    keel::u8 cByte[3];
    for (keel::i32 k = 0; k < 3; ++k) {
        const keel::f64 v = keel::clamp(floatColour ? cIn[k] * 255.0 : cIn[k], 0.0, 255.0);
        cByte[k] = keel::as<keel::u8>(keel::round_i64(v));
    }

    const ValueType srcT = I.type();
    auto pixelByte = [&](keel::usize y, keel::usize x, keel::i32 chan) {
        const keel::usize idx = isRGB ? (keel::as<keel::usize>(chan) * H * W + x * H + y)
                                      : (x * H + y);
        const keel::f64 v = I.elemAsDouble(idx);
        keel::f64 w = v;
        if (srcT != ValueType::UINT8 && srcT != ValueType::UINT16 && srcT != ValueType::INT8
            && srcT != ValueType::INT16) {
            w = v * 255.0;
        } else if (srcT == ValueType::UINT16) {
            w = v / 257.0;
        } else if (srcT == ValueType::INT16) {
            w = (v + 32768.0) / 257.0;
        }
        return keel::as<keel::u8>(keel::round_i64(keel::clamp(w, 0.0, 255.0)));
    };

    Value out = Value::matrix3d(H, W, 3, ValueType::UINT8, mr);
    keel::u8 *od = out.uint8DataMut();
    const keel::usize plane = H * W;
    for (keel::usize y = 0; y < H; ++y)
        for (keel::usize x = 0; x < W; ++x) {
            const bool flag = (BW.elemAsDouble(x * H + y) != 0.0);
            for (keel::i32 c = 0; c < 3; ++c) {
                const keel::usize outIdx = keel::as<keel::usize>(c) * plane + x * H + y;
                od[outIdx] = flag ? cByte[c] : pixelByte(y, x, isRGB ? c : 0);
            }
        }
    return out;
}

Value reducepoly(const Value &P, keel::f64 tolerance, keel::Allocator *mr)
{
    const auto &d = P.dims();
    if (d.ndim() > 2 || (d.cols() != 2 && P.numel() != 0))
        throw Error("reducepoly: P must be an n-by-2 matrix",
                    0, 0, "reducepoly", "", "numkit:reducepoly:shape");
    if (tolerance < 0.0 || tolerance > 1.0)
        throw Error("reducepoly: tolerance must be in [0, 1]",
                    0, 0, "reducepoly", "", "numkit:reducepoly:tol");

    const keel::usize n = d.rows();
    const ValueType cls = P.type();
    auto storeAt = [&](Value &out, keel::usize oidx, keel::f64 v) {
        switch (cls) {
            case ValueType::SINGLE: out.singleDataMut()[oidx] = keel::as<keel::f32>(v); break;
            case ValueType::UINT8:  out.uint8DataMut()[oidx]  = keel::as<keel::u8>(keel::round_i64(v)); break;
            case ValueType::UINT16: out.uint16DataMut()[oidx] = keel::as<keel::u16>(keel::round_i64(v)); break;
            case ValueType::UINT32: out.uint32DataMut()[oidx] = keel::as<keel::u32>(keel::round_i64(v)); break;
            case ValueType::INT8:   out.int8DataMut()[oidx]   = keel::as<keel::i8>(keel::round_i64(v)); break;
            case ValueType::INT16:  out.int16DataMut()[oidx]  = keel::as<keel::i16>(keel::round_i64(v)); break;
            case ValueType::INT32:  out.int32DataMut()[oidx]  = keel::as<keel::i32>(keel::round_i64(v)); break;
            case ValueType::INT64:  out.int64DataMut()[oidx]  = keel::as<keel::i64>(keel::round_i64(v)); break;
            default:                out.doubleDataMut()[oidx] = v; break;
        }
    };

    if (n <= 1) {
        Value out = Value::matrix(n, n == 0 ? 0 : 2, cls, mr);
        for (keel::usize i = 0; i < n; ++i) {
            storeAt(out, i,     P.elemAsDouble(i));
            storeAt(out, i + n, P.elemAsDouble(i + n));
        }
        return out;
    }

    keel::Array<keel::i32> keep(mr);
    if (cls == ValueType::DOUBLE) {
        keel::Array<keel::f64> xs(n, mr), ys(n, mr);
        for (keel::usize i = 0; i < n; ++i) {
            xs[i] = P.elemAsDouble(i);
            ys[i] = P.elemAsDouble(i + n);
        }
        (void)keel::image::reducepoly_f64(xs.as_ref(), ys.as_ref(), tolerance, keep);
    } else {
        keel::Array<keel::f32> xs(n, mr), ys(n, mr);
        for (keel::usize i = 0; i < n; ++i) {
            xs[i] = keel::as<keel::f32>(P.elemAsDouble(i));
            ys[i] = keel::as<keel::f32>(P.elemAsDouble(i + n));
        }
        (void)keel::image::reducepoly_f32(xs.as_ref(), ys.as_ref(), tolerance, keep);
    }

    const keel::usize m = keep.size();
    Value out = Value::matrix(m, 2, cls, mr);
    for (keel::usize j = 0; j < m; ++j) {
        const keel::usize k = keel::as<keel::usize>(keep[j]);
        storeAt(out, j,     P.elemAsDouble(k));
        storeAt(out, j + m, P.elemAsDouble(k + n));
    }
    return out;
}

Value watershed(const Value &A, keel::i32 conn, keel::Allocator *mr)
{
    watershed_argcheck(A, conn);
    const keel::usize H = A.dims().rows(), W = A.dims().cols(), N = H * W;
    if (N == 0) return Value::matrix(H, W, ValueType::UINT8, mr);

    auto aval = extract_double(A, mr);
    keel::Array<keel::u32> L(N, 0, mr);
    keel::u32 numBasins = 0;
    (void)keel::image::watershed_2d(aval.as_ref(), H, W, conn, L.as_mut_ref(), numBasins, mr);

    if (numBasins <= 255u) {
        Value out = Value::matrix(H, W, ValueType::UINT8, mr);
        for (keel::usize i = 0; i < N; ++i) out.uint8DataMut()[i] = keel::as<keel::u8>(L[i]);
        return out;
    }
    if (numBasins <= 65535u) {
        Value out = Value::matrix(H, W, ValueType::UINT16, mr);
        for (keel::usize i = 0; i < N; ++i) out.uint16DataMut()[i] = keel::as<keel::u16>(L[i]);
        return out;
    }
    Value out = Value::matrix(H, W, ValueType::UINT32, mr);
    keel::copy_n(out.uint32DataMut(), L.data(), N);
    return out;
}

} // namespace numkit::image
