// toolboxes/image/src/filter/locallapfilt.cpp
//
// Fast Local Laplacian Filtering delegating the pyramid & remapping
// numerical kernel to keel::image.

#include <keel/arena.h>
#include <keel/algo.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/image.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <numkit/image/filter/filter.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>


namespace numkit::image {

namespace {

keel::Array<keel::f32> readSinglePlane(const Value &I, keel::usize page, keel::Allocator *alloc)
{
    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    const keel::usize N = H * W;
    const keel::usize off = page * N;
    const ValueType t = I.type();
    keel::Array<keel::f32> out(N, 0.0f, alloc);
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 raw = I.elemAsDouble(off + i);
        switch (t) {
            case ValueType::UINT8:   out[i] = keel::as<keel::f32>(raw) / 255.0f; break;
            case ValueType::UINT16:  out[i] = keel::as<keel::f32>(raw) / 65535.0f; break;
            case ValueType::INT8:    out[i] = (keel::as<keel::f32>(raw) + 128.0f) / 255.0f; break;
            case ValueType::INT16:   out[i] = (keel::as<keel::f32>(raw) + 32768.0f) / 65535.0f; break;
            case ValueType::LOGICAL: out[i] = (raw == 0.0) ? 0.0f : 1.0f; break;
            default:                 out[i] = keel::as<keel::f32>(raw); break;
        }
    }
    return out;
}

void writeSinglePlaneTo(Value &out, keel::usize page, keel::ArrayRef<keel::f32> plane, ValueType target)
{
    const keel::usize N = plane.size();
    const keel::usize off = page * N;
    switch (target) {
        case ValueType::UINT8: {
            keel::u8 *od = out.uint8DataMut() + off;
            for (keel::usize i = 0; i < N; ++i) {
                keel::f32 v = keel::as<keel::f32>(keel::round(keel::as<keel::f64>(plane[i]) * 255.0));
                if (v < 0.0f) v = 0.0f;
                if (v > 255.0f) v = 255.0f;
                od[i] = keel::as<keel::u8>(v);
            }
            break;
        }
        case ValueType::UINT16: {
            keel::u16 *od = out.uint16DataMut() + off;
            for (keel::usize i = 0; i < N; ++i) {
                keel::f32 v = keel::as<keel::f32>(keel::round(keel::as<keel::f64>(plane[i]) * 65535.0));
                if (v < 0.0f) v = 0.0f;
                if (v > 65535.0f) v = 65535.0f;
                od[i] = keel::as<keel::u16>(v);
            }
            break;
        }
        case ValueType::INT8: {
            keel::i8 *od = out.int8DataMut() + off;
            for (keel::usize i = 0; i < N; ++i) {
                keel::f32 v = keel::as<keel::f32>(keel::round(keel::as<keel::f64>(plane[i]) * 255.0 - 128.0));
                if (v < -128.0f) v = -128.0f;
                if (v > 127.0f) v = 127.0f;
                od[i] = keel::as<keel::i8>(v);
            }
            break;
        }
        case ValueType::INT16: {
            keel::i16 *od = out.int16DataMut() + off;
            for (keel::usize i = 0; i < N; ++i) {
                keel::f32 v = keel::as<keel::f32>(keel::round(keel::as<keel::f64>(plane[i]) * 65535.0 - 32768.0));
                if (v < -32768.0f) v = -32768.0f;
                if (v > 32767.0f) v = 32767.0f;
                od[i] = keel::as<keel::i16>(v);
            }
            break;
        }
        case ValueType::SINGLE: {
            keel::copy_n(out.singleDataMut() + off, plane.data(), N);
            break;
        }
        default: {
            keel::f64 *od = out.doubleDataMut() + off;
            for (keel::usize i = 0; i < N; ++i) od[i] = keel::as<keel::f64>(plane[i]);
            break;
        }
    }
}

} // namespace

Value locallapfilt(const Value &I, keel::f64 sigma, keel::f64 alpha, keel::f64 beta,
                   keel::i32 num_intensity_levels, bool process_luminance,
                   keel::Allocator *mr)
{
    if (!keel::is_finite(sigma) || sigma < 0.0)
        throw Error("locallapfilt: sigma must be a non-negative finite scalar",
                    0, 0, "locallapfilt", "", "numkit:locallapfilt:sigma");
    if (!keel::is_finite(alpha) || alpha <= 0.0)
        throw Error("locallapfilt: alpha must be a positive finite scalar",
                    0, 0, "locallapfilt", "", "numkit:locallapfilt:alpha");
    if (!keel::is_finite(beta) || beta < 0.0)
        throw Error("locallapfilt: beta must be a non-negative finite scalar",
                    0, 0, "locallapfilt", "", "numkit:locallapfilt:beta");

    const auto &dI = I.dims();
    if (dI.rows() == 0 || dI.cols() == 0)
        throw Error("locallapfilt: I must be non-empty",
                    0, 0, "locallapfilt", "", "numkit:locallapfilt:empty");
    const bool isRGB = dI.is3D() && dI.pages() == 3;
    const bool isGray = !dI.is3D() || (dI.is3D() && dI.pages() == 1);
    if (!isRGB && !isGray)
        throw Error("locallapfilt: I must be H×W (grayscale) or H×W×3 (RGB)",
                    0, 0, "locallapfilt", "", "numkit:locallapfilt:shape");

    if ((alpha == 1.0 && beta == 1.0) || (sigma == 0.0 && beta == 1.0)) {
        return I;
    }

    const ValueType origClass = I.type();
    const ValueType outClass = (origClass == ValueType::UINT8 || origClass == ValueType::UINT16 ||
                                origClass == ValueType::INT8 || origClass == ValueType::INT16 ||
                                origClass == ValueType::SINGLE)
                                   ? origClass
                                   : ValueType::DOUBLE;
    const keel::usize H = dI.rows();
    const keel::usize W = dI.cols();
    const keel::usize N = H * W;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const keel::f32 fsigma = keel::as<keel::f32>(sigma);
    const keel::f32 falpha = keel::as<keel::f32>(alpha);
    const keel::f32 fbeta  = keel::as<keel::f32>(beta);

    if (isGray) {
        const auto in_p = readSinglePlane(I, 0, scratch.allocator());
        keel::Array<keel::f32> out_p(N, 0.0f, scratch.allocator());
        (void)keel::image::local_laplacian_gray(in_p.as_ref(), H, W, fsigma, falpha, fbeta,
                                                num_intensity_levels, out_p.as_mut_ref(),
                                                scratch.allocator());
        Value out = Value::matrix(H, W, outClass, mr);
        writeSinglePlaneTo(out, 0, out_p.as_ref(), outClass);
        return out;
    }

    const auto in_r = readSinglePlane(I, 0, scratch.allocator());
    const auto in_g = readSinglePlane(I, 1, scratch.allocator());
    const auto in_b = readSinglePlane(I, 2, scratch.allocator());
    keel::Array<keel::f32> out_r(N, 0.0f, scratch.allocator());
    keel::Array<keel::f32> out_g(N, 0.0f, scratch.allocator());
    keel::Array<keel::f32> out_b(N, 0.0f, scratch.allocator());
    (void)keel::image::local_laplacian_rgb(in_r.as_ref(), in_g.as_ref(), in_b.as_ref(), H, W,
                                           fsigma, falpha, fbeta, num_intensity_levels,
                                           process_luminance, out_r.as_mut_ref(),
                                           out_g.as_mut_ref(), out_b.as_mut_ref(),
                                           scratch.allocator());
    Value out = Value::matrix3d(H, W, 3, outClass, mr);
    writeSinglePlaneTo(out, 0, out_r.as_ref(), outClass);
    writeSinglePlaneTo(out, 1, out_g.as_ref(), outClass);
    writeSinglePlaneTo(out, 2, out_b.as_ref(), outClass);
    return out;
}

} // namespace numkit::image
