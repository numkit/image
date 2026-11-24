// toolboxes/image/src/object/object.cpp
//
// Thin Value/argcheck wrappers delegating 2D/3D gradients, edge detection
// (Sobel/Prewitt/Roberts/LoG/ZeroCross/Canny), cornermetric/corner, Hough
// transform/peaks/lines, and imfindcircles to keel::image.

#include <keel/core.h>
#include <keel/algo.h>
#include <keel/format.h>
#include <keel/image.h>
#include <keel/memory.h>
#include <keel/array.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <keel/tuple.h>
#include <keel/math.h>

#include <numkit/image/object/object.hpp>
#include <numkit/image/internal/type_gates.hpp>
#include <numkit/image/color/color.hpp>
#include <numkit/image/filter/filter.hpp>
#include <numkit/image/type_convert/type_convert.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

namespace numkit::image {

namespace {

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

keel::Array<keel::u8> extract_bw(const Value &BW, keel::Allocator *mr)
{
    const keel::usize N = BW.numel();
    keel::Array<keel::u8> out(N, mr);
    if (BW.isLogical()) {
        const keel::u8 *src = BW.logicalData();
        for (keel::usize i = 0; i < N; ++i) out[i] = src[i] ? 1u : 0u;
    } else {
        for (keel::usize i = 0; i < N; ++i) out[i] = (BW.elemAsDouble(i) != 0.0) ? 1u : 0u;
    }
    return out;
}

keel::image::GradientMethod parse_grad_method_2d(keel::StringRef method)
{
    if (method == "prewitt")      return keel::image::GradientMethod::kPrewitt;
    if (method == "central")      return keel::image::GradientMethod::kCentral;
    if (method == "intermediate") return keel::image::GradientMethod::kIntermediate;
    return keel::image::GradientMethod::kSobel;
}

Value zeros_like(const Value &V, ValueType t, keel::Allocator *mr)
{
    const auto &d = V.dims();
    return d.is3D() ? Value::matrix3d(d.rows(), d.cols(), d.pages(), t, mr)
                    : Value::matrix(d.rows(), d.cols(), t, mr);
}

keel::String toLowerAscii(keel::StringRef s)
{
    keel::String out = keel::String::from(s);
    out.to_ascii_lower_in_place();
    return out;
}

void imfindcircles_argcheck(const Value &A, keel::f64 rmin, keel::f64 rmax,
                            keel::StringRef object_polarity,
                            keel::StringRef method,
                            keel::f64 sensitivity,
                            keel::f64 edge_threshold)
{
    internal::requireImageClass(A, "imfindcircles", "A", 1, internal::ImgClasses::FloatIntLog);
    internal::requireNonemptyImage(A, "imfindcircles", "A", 1);
    const Dims &d = A.dims();
    if (d.ndim() > 3 || (d.is3D() && d.pages() != 3) || d.rows() <= 1 || d.cols() <= 1) {
        throw Error("imfindcircles: input 1 (A) must be a 2-D matrix "
                    "(both dimensions > 1) or an M-by-N-by-3 RGB image.",
                    0, 0, "imfindcircles", "", "numkit:imfindcircles:invalidInputImage");
    }
    if (!keel::is_finite(rmin) || !keel::is_finite(rmax) || !(rmin > 0.0) || !(rmax >= rmin)) {
        throw Error("imfindcircles: input 2 (radiusRange) must have "
                    "positive finite bounds with rmin <= rmax.",
                    0, 0, "imfindcircles", "", "numkit:imfindcircles:invalidRadiusRange");
    }
    if (rmin != rmax && (rmin != keel::floor(rmin) || rmax != keel::floor(rmax))) {
        throw Error("imfindcircles: 2-element radiusRange must contain positive integers.",
                    0, 0, "imfindcircles", "", "numkit:imfindcircles:expectedInteger");
    }
    const keel::String pol = toLowerAscii(object_polarity);
    if (pol != "bright" && pol != "dark") {
        throw Error(keel::format("imfindcircles: ObjectPolarity must be 'bright' or 'dark' (got '{}').",
                                 object_polarity),
                    0, 0, "imfindcircles", "", "numkit:imfindcircles:unrecognizedObjectPolarity");
    }
    const keel::String mth = toLowerAscii(method);
    if (mth != "phasecode" && mth != "twostage") {
        throw Error(keel::format("imfindcircles: Method must be 'PhaseCode' or 'TwoStage' (got '{}').",
                                 method),
                    0, 0, "imfindcircles", "", "numkit:imfindcircles:unrecognizedMethod");
    }
    if (!keel::is_finite(sensitivity) || sensitivity < 0.0 || sensitivity > 1.0) {
        throw Error("imfindcircles: Sensitivity must be a finite scalar in [0, 1].",
                    0, 0, "imfindcircles", "", "numkit:imfindcircles:outOfRangeSensitivity");
    }
    if (edge_threshold >= 0.0 || keel::is_nan(edge_threshold) || keel::is_inf(edge_threshold)) {
        if (!keel::is_finite(edge_threshold) || edge_threshold < 0.0 || edge_threshold > 1.0) {
            throw Error("imfindcircles: EdgeThreshold must be a finite scalar in [0, 1].",
                        0, 0, "imfindcircles", "", "numkit:imfindcircles:outOfRangeEdgeThreshold");
        }
    }
}

} // anonymous

keel::Tuple<Value, Value>
imgradientxy(const Value &I, keel::StringRef method, keel::Allocator *mr)
{
    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    Value Gx = Value::matrix(H, W, ValueType::DOUBLE, mr);
    Value Gy = Value::matrix(H, W, ValueType::DOUBLE, mr);
    if (H == 0 || W == 0) return keel::make_tuple(keel::move(Gx), keel::move(Gy));
    auto in = extract_double(I, mr);
    (void)keel::image::imgradientxy_2d(
        in.as_ref(), H, W, parse_grad_method_2d(method),
        keel::MutArrayRef<keel::f64>(Gx.doubleDataMut(), H * W),
        keel::MutArrayRef<keel::f64>(Gy.doubleDataMut(), H * W));
    return keel::make_tuple(keel::move(Gx), keel::move(Gy));
}

keel::Tuple<Value, Value, Value>
imgradientxyz(const Value &V, keel::StringRef method, keel::Allocator *mr)
{
    if (!V.dims().is3D())
        throw Error("imgradientxyz: V must be 3-D",
                    0, 0, "imgradientxyz", "", "numkit:imgradientxyz:rank");
    keel::image::GradientMethod gm;
    if      (method == "sobel")        gm = keel::image::GradientMethod::kSobel;
    else if (method == "prewitt")      gm = keel::image::GradientMethod::kPrewitt;
    else if (method == "central")      gm = keel::image::GradientMethod::kCentral;
    else if (method == "intermediate") gm = keel::image::GradientMethod::kIntermediate;
    else
        throw Error("imgradientxyz: method must be 'sobel', 'prewitt', 'central', or 'intermediate'",
                    0, 0, "imgradientxyz", "", "numkit:imgradientxyz:method");

    const ValueType outT = (V.type() == ValueType::SINGLE) ? ValueType::SINGLE : ValueType::DOUBLE;
    const keel::usize H = V.dims().rows(), W = V.dims().cols(), D = V.dims().pages();
    const keel::usize N = H * W * D;
    auto in = extract_double(V, mr);
    Value Gx = Value::matrix3d(H, W, D, ValueType::DOUBLE, mr);
    Value Gy = Value::matrix3d(H, W, D, ValueType::DOUBLE, mr);
    Value Gz = Value::matrix3d(H, W, D, ValueType::DOUBLE, mr);
    (void)keel::image::imgradientxyz_3d(
        in.as_ref(), H, W, D, gm,
        keel::MutArrayRef<keel::f64>(Gx.doubleDataMut(), N),
        keel::MutArrayRef<keel::f64>(Gy.doubleDataMut(), N),
        keel::MutArrayRef<keel::f64>(Gz.doubleDataMut(), N));

    if (outT == ValueType::SINGLE) {
        auto cast_single = [&](Value &G) {
            Value gs = Value::matrix3d(H, W, D, ValueType::SINGLE, mr);
            for (keel::usize i = 0; i < N; ++i)
                gs.singleDataMut()[i] = keel::as<keel::f32>(G.doubleData()[i]);
            G = keel::move(gs);
        };
        cast_single(Gx); cast_single(Gy); cast_single(Gz);
    }
    return keel::make_tuple(keel::move(Gx), keel::move(Gy), keel::move(Gz));
}

keel::Tuple<Value, Value, Value>
imgradient3_from_grads(const Value &Gx, const Value &Gy, const Value &Gz, keel::Allocator *mr)
{
    if (Gx.dims().rows() != Gy.dims().rows() || Gx.dims().cols() != Gy.dims().cols()
        || Gx.dims().pages() != Gy.dims().pages() || Gx.dims().rows() != Gz.dims().rows()
        || Gx.dims().cols() != Gz.dims().cols() || Gx.dims().pages() != Gz.dims().pages())
        throw Error("imgradient3: Gx, Gy, Gz must have the same size",
                    0, 0, "imgradient3", "", "numkit:imgradient3:size");
    const keel::usize N = Gx.numel();
    const ValueType outT = (Gx.type() == ValueType::SINGLE || Gy.type() == ValueType::SINGLE
                            || Gz.type() == ValueType::SINGLE)
                               ? ValueType::SINGLE : ValueType::DOUBLE;
    Value Gmag = zeros_like(Gx, outT, mr);
    Value Gaz  = zeros_like(Gx, outT, mr);
    Value Gelev = zeros_like(Gx, outT, mr);
    constexpr keel::f64 D2R = 180.0 / keel::kPi;
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 gx = Gx.elemAsDouble(i), gy = Gy.elemAsDouble(i), gz = Gz.elemAsDouble(i);
        const keel::f64 mag = keel::hypot(keel::hypot(gx, gy), gz);
        const keel::f64 az  = keel::atan2(-gy, gx) * D2R;
        const keel::f64 el  = keel::atan2(gz, keel::hypot(gx, gy)) * D2R;
        if (outT == ValueType::DOUBLE) {
            Gmag.doubleDataMut()[i] = mag; Gaz.doubleDataMut()[i] = az; Gelev.doubleDataMut()[i] = el;
        } else {
            Gmag.singleDataMut()[i]  = keel::as<keel::f32>(mag);
            Gaz.singleDataMut()[i]   = keel::as<keel::f32>(az);
            Gelev.singleDataMut()[i] = keel::as<keel::f32>(el);
        }
    }
    return keel::make_tuple(keel::move(Gmag), keel::move(Gaz), keel::move(Gelev));
}

keel::Tuple<Value, Value, Value>
imgradient3(const Value &V, keel::StringRef method, keel::Allocator *mr)
{
    auto [Gx, Gy, Gz] = imgradientxyz(V, method, mr);
    return imgradient3_from_grads(Gx, Gy, Gz, mr);
}

keel::Tuple<Value, Value>
imgradient(const Value &I, keel::StringRef method, keel::Allocator *mr)
{
    auto [Gx, Gy] = imgradientxy(I, method, mr);
    const keel::usize N = Gx.numel();
    Value Gmag = Value::matrix(Gx.dims().rows(), Gx.dims().cols(), ValueType::DOUBLE, mr);
    Value Gdir = Value::matrix(Gx.dims().rows(), Gx.dims().cols(), ValueType::DOUBLE, mr);
    const keel::f64 *gxp = Gx.doubleData(), *gyp = Gy.doubleData();
    keel::f64 *gm = Gmag.doubleDataMut(), *gd = Gdir.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) {
        gm[i] = keel::sqrt(gxp[i] * gxp[i] + gyp[i] * gyp[i]);
        gd[i] = keel::atan2(-gyp[i], gxp[i]) * 180.0 / keel::kPi;
    }
    return keel::make_tuple(keel::move(Gmag), keel::move(Gdir));
}

Value imgradient_mag(const Value &I, keel::StringRef method, keel::Allocator *mr)
{
    auto [Gx, Gy] = imgradientxy(I, method, mr);
    const keel::usize N = Gx.numel();
    Value Gmag = Value::matrix(Gx.dims().rows(), Gx.dims().cols(), ValueType::DOUBLE, mr);
    const keel::f64 *gxp = Gx.doubleData(), *gyp = Gy.doubleData();
    keel::f64 *gm = Gmag.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) gm[i] = keel::sqrt(gxp[i] * gxp[i] + gyp[i] * gyp[i]);
    return Gmag;
}

Value edge(const Value &I, keel::StringRef methodRaw, keel::f64 thresh_lo, keel::f64 /*thresh_hi*/,
           keel::Allocator *mr)
{
    keel::String method = toLowerAscii(methodRaw);
    keel::image::EdgeMethod em = keel::image::EdgeMethod::kSobel;
    if      (method == "prewitt")   em = keel::image::EdgeMethod::kPrewitt;
    else if (method == "roberts")   em = keel::image::EdgeMethod::kRoberts;
    else if (method == "log")       em = keel::image::EdgeMethod::kLog;
    else if (method == "zerocross") em = keel::image::EdgeMethod::kZeroCross;
    else if (method == "canny")     em = keel::image::EdgeMethod::kCanny;

    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    Value out = Value::matrix(H, W, ValueType::LOGICAL, mr);
    if (H == 0 || W == 0) return out;
    auto in = extract_double(I, mr);
    (void)keel::image::edge_2d(in.as_ref(), H, W, em, thresh_lo,
                               keel::MutArrayRef<keel::u8>(out.logicalDataMut(), H * W), mr);
    return out;
}

Value cornermetric(const Value &I, keel::StringRef method,
                   keel::f64 sensitivity_factor, const Value &filter_coef,
                   keel::Allocator *mr)
{
    internal::requireImageClass(I, "cornermetric", "I", 1, internal::ImgClasses::FloatIntLog);
    if (I.dims().is3D())
        throw Error("cornermetric: I must be 2-D",
                    0, 0, "cornermetric", "", "numkit:cornermetric:dim");
    if (method != "Harris" && method != "MinimumEigenvalue")
        throw Error("cornermetric: METHOD must be 'Harris' or 'MinimumEigenvalue'",
                    0, 0, "cornermetric", "", "numkit:cornermetric:method");
    if (method == "Harris" && (!(sensitivity_factor > 0.0) || !(sensitivity_factor < 0.25)))
        throw Error("cornermetric: SensitivityFactor must be in (0, 0.25)",
                    0, 0, "cornermetric", "", "numkit:cornermetric:k");

    keel::Array<keel::f64> fcoef(mr);
    if (filter_coef.numel() > 0) {
        if (filter_coef.numel() < 3 || (filter_coef.numel() % 2) == 0)
            throw Error("cornermetric: FilterCoefficients length must be odd and ≥ 3",
                        0, 0, "cornermetric", "", "numkit:cornermetric:filter");
        fcoef.reserve(filter_coef.numel());
        for (keel::usize i = 0; i < filter_coef.numel(); ++i)
            fcoef.push_back(filter_coef.elemAsDouble(i));
    }

    Value Id = im2double(I, mr);
    const keel::usize H = Id.dims().rows(), W = Id.dims().cols();
    if (H < 3 || W < 3)
        throw Error("cornermetric: image too small (need ≥ 3x3)",
                    0, 0, "cornermetric", "", "numkit:cornermetric:tooSmall");

    Value out = Value::matrix(H, W, ValueType::DOUBLE, mr);
    const auto cm = (method == "Harris") ? keel::image::CornerMethod::kHarris
                                         : keel::image::CornerMethod::kMinimumEigenvalue;
    (void)keel::image::cornermetric_2d(
        keel::ArrayRef<keel::f64>(Id.doubleData(), H * W), H, W, cm, sensitivity_factor,
        fcoef.as_ref(), keel::MutArrayRef<keel::f64>(out.doubleDataMut(), H * W), mr);
    return out;
}

Value corner(const Value &I, keel::StringRef method, keel::i32 maxN,
             keel::f64 quality_level, keel::f64 sensitivity,
             const Value &filter_coef, keel::Allocator *mr)
{
    internal::requireImageClass(I, "corner", "I", 1, internal::ImgClasses::FloatIntLog);
    Value Cm = cornermetric(I, method, sensitivity, filter_coef, mr);
    const keel::usize M = Cm.dims().rows(), N = Cm.dims().cols();
    keel::Array<keel::image::BoundaryPoint> peaks(mr);
    (void)keel::image::corner_peaks_2d(
        keel::ArrayRef<keel::f64>(Cm.doubleData(), M * N), M, N, maxN, quality_level, peaks, mr);

    const keel::usize K = peaks.size();
    Value out = Value::matrix(K, 2, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < K; ++i) {
        od[i]     = keel::as<keel::f64>(peaks[i].col + 1);
        od[K + i] = keel::as<keel::f64>(peaks[i].row + 1);
    }
    return out;
}

void hough(const Value &BW, keel::f64 rho_res, const Value &theta_deg,
           Value &H_out, Value &T_out, Value &R_out, keel::Allocator *mr)
{
    if (BW.dims().is3D())
        throw Error("hough: BW must be 2-D", 0, 0, "hough", "", "numkit:hough:dim");
    if (!(rho_res > 0.0) || !keel::is_finite(rho_res))
        throw Error("hough: RhoResolution must be a positive scalar",
                    0, 0, "hough", "", "numkit:hough:rho");

    keel::Array<keel::f64> theta(mr);
    if (theta_deg.numel() == 0) {
        theta.reserve(180);
        for (keel::i32 k = -90; k <= 89; ++k) theta.push_back(keel::as<keel::f64>(k));
    } else {
        theta.reserve(theta_deg.numel());
        for (keel::usize i = 0; i < theta_deg.numel(); ++i) {
            const keel::f64 v = theta_deg.elemAsDouble(i);
            if (v < -90.0 || v >= 90.0)
                throw Error("hough: Theta values must lie in [-90, 90)",
                            0, 0, "hough", "", "numkit:hough:theta");
            theta.push_back(v);
        }
    }

    auto bw = extract_bw(BW, mr);
    keel::Array<keel::f64> h_arr(mr), rho_arr(mr);
    (void)keel::image::hough_2d(bw.as_ref(), BW.dims().rows(), BW.dims().cols(), rho_res,
                                theta.as_ref(), h_arr, rho_arr, mr);

    const keel::usize nrho = rho_arr.size(), ntheta = theta.size();
    H_out = Value::matrix(nrho, ntheta, ValueType::DOUBLE, mr);
    if (!h_arr.empty()) keel::copy_n(H_out.doubleDataMut(), h_arr.data(), h_arr.size());
    T_out = Value::matrix(1, ntheta, ValueType::DOUBLE, mr);
    if (!theta.empty()) keel::copy_n(T_out.doubleDataMut(), theta.data(), ntheta);
    R_out = Value::matrix(1, nrho, ValueType::DOUBLE, mr);
    if (!rho_arr.empty()) keel::copy_n(R_out.doubleDataMut(), rho_arr.data(), nrho);
}

Value houghpeaks(const Value &H, keel::usize numpeaks, keel::f64 threshold,
                 keel::usize nhoodRho, keel::usize nhoodTheta,
                 const Value &theta_deg, keel::Allocator *mr)
{
    internal::requireImageClass(H, "houghpeaks", "H", 1, internal::ImgClasses::DoubleOnly);
    if (H.dims().is3D())
        throw Error("houghpeaks: H must be 2-D", 0, 0, "houghpeaks", "", "numkit:houghpeaks:dim");

    keel::Array<keel::f64> theta(mr);
    if (theta_deg.numel() > 0) {
        theta.reserve(theta_deg.numel());
        for (keel::usize i = 0; i < theta_deg.numel(); ++i)
            theta.push_back(theta_deg.elemAsDouble(i));
    }

    keel::Array<keel::image::BoundaryPoint> peaks(mr);
    (void)keel::image::houghpeaks_2d(
        keel::ArrayRef<keel::f64>(H.doubleData(), H.numel()),
        H.dims().rows(), H.dims().cols(), numpeaks, threshold, nhoodRho, nhoodTheta,
        theta.as_ref(), peaks, mr);

    const keel::usize nP = peaks.size();
    Value P = Value::matrix(nP, 2, ValueType::DOUBLE, mr);
    for (keel::usize i = 0; i < nP; ++i) {
        P.doubleDataMut()[i]      = keel::as<keel::f64>(peaks[i].row + 1);
        P.doubleDataMut()[nP + i] = keel::as<keel::f64>(peaks[i].col + 1);
    }
    return P;
}

Value houghlines(const Value &BW, const Value &theta_deg, const Value &rho, const Value &peaks,
                 keel::f64 fillgap, keel::f64 minlength, keel::Allocator *mr)
{
    internal::requireImageClass(BW, "houghlines", "BW", 1, internal::ImgClasses::AnyNumericReal);
    if (BW.dims().is3D())
        throw Error("houghlines: BW must be 2-D", 0, 0, "houghlines", "", "numkit:houghlines:dim");
    if (!(fillgap > 0.0) || !keel::is_finite(fillgap))
        throw Error("houghlines: FillGap must be positive",
                    0, 0, "houghlines", "", "numkit:houghlines:fillgap");
    if (!(minlength > 0.0) || !keel::is_finite(minlength))
        throw Error("houghlines: MinLength must be positive",
                    0, 0, "houghlines", "", "numkit:houghlines:minlength");
    if (rho.numel() < 2 || theta_deg.numel() == 0)
        throw Error("houghlines: rho/theta vectors too small",
                    0, 0, "houghlines", "", "numkit:houghlines:vec");
    if (peaks.dims().cols() != 2 || peaks.dims().is3D())
        throw Error("houghlines: PEAKS must be P × 2",
                    0, 0, "houghlines", "", "numkit:houghlines:peaks");

    auto bw = extract_bw(BW, mr);
    auto th = extract_double(theta_deg, mr);
    auto rh = extract_double(rho, mr);
    const keel::usize nPeaks = peaks.dims().rows();
    keel::Array<keel::image::BoundaryPoint> p0(nPeaks, mr);
    for (keel::usize k = 0; k < nPeaks; ++k) {
        p0[k].row = keel::as<keel::i64>(peaks.elemAsDouble(k)) - 1;
        p0[k].col = keel::as<keel::i64>(peaks.elemAsDouble(nPeaks + k)) - 1;
    }

    keel::Array<keel::image::HoughLine> segs(mr);
    (void)keel::image::houghlines_2d(bw.as_ref(), BW.dims().rows(), BW.dims().cols(),
                                     th.as_ref(), rh.as_ref(), p0.as_ref(),
                                     fillgap, minlength, segs, mr);

    const keel::usize nL = segs.size();
    if (nL == 0) return Value::structArray(1, 0, mr);
    Value out = Value::structArray(1, nL, mr);
    for (keel::usize i = 0; i < nL; ++i) {
        Value p1 = Value::matrix(1, 2, ValueType::DOUBLE, mr);
        p1.doubleDataMut()[0] = segs[i].p1x + 1.0;
        p1.doubleDataMut()[1] = segs[i].p1y + 1.0;
        Value p2 = Value::matrix(1, 2, ValueType::DOUBLE, mr);
        p2.doubleDataMut()[0] = segs[i].p2x + 1.0;
        p2.doubleDataMut()[1] = segs[i].p2y + 1.0;
        out.setField(i, "point1", p1);
        out.setField(i, "point2", p2);
        out.setField(i, "theta", Value::scalar(segs[i].theta, mr));
        out.setField(i, "rho",   Value::scalar(segs[i].rho,   mr));
    }
    return out;
}

ImfindcirclesResult
imfindcircles(const Value &A, keel::f64 rmin, keel::f64 rmax,
              keel::StringRef object_polarity, keel::StringRef method,
              keel::f64 sensitivity, keel::f64 edge_threshold, keel::Allocator *mr)
{
    imfindcircles_argcheck(A, rmin, rmax, object_polarity, method, sensitivity, edge_threshold);
    auto emptyResult = [&]() -> ImfindcirclesResult {
        return {Value::matrix(0, 0, ValueType::DOUBLE, mr),
                Value::matrix(0, 0, ValueType::DOUBLE, mr),
                Value::matrix(0, 0, ValueType::DOUBLE, mr)};
    };
    if (A.numel() == 0) return emptyResult();

    // Match reference grayscale conversion before calling keel::image::imfindcircles_2d.
    Value G;
    if (A.dims().is3D()) {
        G = rgb2gray(A, mr);
        if (isIntegerType(G.type())) G = im2single(G, mr);
    } else if (A.type() == ValueType::LOGICAL) {
        Value As = im2single(A, mr);
        Value gaussFilt = fspecial("gaussian", {5.0, 5.0, 1.5}, mr);
        G = imfilter(As, gaussFilt, PadMode::Replicate, 0.0, false, false, mr);
    } else if (isIntegerType(A.type())) {
        G = im2single(A, mr);
    } else {
        G = A;
    }

    const bool useSingle = (G.type() == ValueType::SINGLE);
    auto gbuf = extract_double(G, mr);
    const auto pol = (toLowerAscii(object_polarity) == "bright")
                         ? keel::image::CirclePolarity::kBright
                         : keel::image::CirclePolarity::kDark;
    const auto mth = (toLowerAscii(method) == "phasecode")
                         ? keel::image::CircleMethod::kPhaseCode
                         : keel::image::CircleMethod::kTwoStage;

    keel::Array<keel::image::CircleDetection> circles(mr);
    (void)keel::image::imfindcircles_2d(
        gbuf.as_ref(), G.dims().rows(), G.dims().cols(), rmin, rmax, pol, mth,
        sensitivity, edge_threshold, useSingle, circles, mr);

    const keel::usize K = circles.size();
    if (K == 0) return emptyResult();
    Value centersOut = Value::matrix(K, 2, ValueType::DOUBLE, mr);
    Value radiiOut   = Value::matrix(K, 1, ValueType::DOUBLE, mr);
    Value metricOut  = Value::matrix(K, 1, ValueType::DOUBLE, mr);
    keel::f64 *cp = centersOut.doubleDataMut();
    keel::f64 *rp = radiiOut.doubleDataMut();
    keel::f64 *mp = metricOut.doubleDataMut();
    for (keel::usize k = 0; k < K; ++k) {
        cp[k]     = circles[k].center_x + 1.0;
        cp[K + k] = circles[k].center_y + 1.0;
        rp[k]     = circles[k].radius;
        mp[k]     = circles[k].metric;
    }
    return {keel::move(centersOut), keel::move(radiiOut), keel::move(metricOut)};
}

} // namespace numkit::image
