// toolboxes/image/src/filter/fibermetric.cpp
//
// Multiscale Hessian-based vesselness filter (Frangi 1998),
// delegating 2D/3D Gaussian + Hessian eigenanalysis to keel::image.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/image.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <numkit/image/filter/filter.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

namespace numkit::image {

namespace {

keel::f64 class_range(ValueType t)
{
    switch (t) {
        case ValueType::UINT8:  return 255.0;
        case ValueType::UINT16: return 65535.0;
        case ValueType::UINT32: return 4294967295.0;
        case ValueType::INT8:   return 255.0;
        case ValueType::INT16:  return 65535.0;
        case ValueType::INT32:  return 4294967295.0;
        default:                return 1.0;
    }
}

} // namespace

Value fibermetric(const Value &I, keel::ArrayRef<keel::f64> thickness_in,
                  keel::f64 structure_sensitivity, bool bright_polarity,
                  keel::Allocator *mr)
{
    const auto &d = I.dims();
    if (d.rows() < 2 || d.cols() < 2)
        throw Error("fibermetric: I must be 2-D or 3-D with all dims >= 2",
                    0, 0, "fibermetric", "", "numkit:fibermetric:shape");
    const bool is3d = d.is3D() && d.pages() > 1;
    if (is3d && d.pages() < 2)
        throw Error("fibermetric: 3-D input must have at least 2 slices",
                    0, 0, "fibermetric", "", "numkit:fibermetric:slices");

    static constexpr keel::f64 kDefaultThickness[6] = {4.0, 6.0, 8.0, 10.0, 12.0, 14.0};
    const keel::ArrayRef<keel::f64> thickness = thickness_in.empty()
        ? keel::ArrayRef<keel::f64>(kDefaultThickness, 6)
        : thickness_in;
    for (keel::f64 t : thickness) {
        if (!keel::is_finite(t) || t <= 0.0 || keel::floor(t) != t)
            throw Error("fibermetric: thickness must be a vector of positive integers",
                        0, 0, "fibermetric", "", "numkit:fibermetric:thickness");
    }

    keel::f64 c = structure_sensitivity;
    if (c <= 0.0) c = class_range(I.type()) / 100.0;
    if (!keel::is_finite(c) || c <= 0.0)
        throw Error("fibermetric: StructureSensitivity must be > 0",
                    0, 0, "fibermetric", "", "numkit:fibermetric:sens");

    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    const keel::usize D = is3d ? d.pages() : 1u;
    const keel::usize N = H * W * D;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> in_buf(N, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i) in_buf[i] = I.elemAsDouble(i);

    const ValueType outClass = (I.type() == ValueType::DOUBLE) ? ValueType::DOUBLE : ValueType::SINGLE;
    if (outClass == ValueType::DOUBLE) {
        Value out = is3d ? Value::matrix3d(H, W, D, ValueType::DOUBLE, mr)
                         : Value::matrix(H, W, ValueType::DOUBLE, mr);
        keel::MutArrayRef<keel::f64> od(out.doubleDataMut(), N);
        if (is3d) {
            (void)keel::image::fibermetric_3d(in_buf.as_ref(), H, W, D, thickness, c,
                                              bright_polarity, od, scratch.allocator());
        } else {
            (void)keel::image::fibermetric_2d(in_buf.as_ref(), H, W, thickness, c,
                                              bright_polarity, od, scratch.allocator());
        }
        return out;
    }

    keel::Array<keel::f64> out_buf(N, 0.0, scratch.allocator());
    if (is3d) {
        (void)keel::image::fibermetric_3d(in_buf.as_ref(), H, W, D, thickness, c,
                                          bright_polarity, out_buf.as_mut_ref(), scratch.allocator());
    } else {
        (void)keel::image::fibermetric_2d(in_buf.as_ref(), H, W, thickness, c,
                                          bright_polarity, out_buf.as_mut_ref(), scratch.allocator());
    }
    Value out = is3d ? Value::matrix3d(H, W, D, ValueType::SINGLE, mr)
                     : Value::matrix(H, W, ValueType::SINGLE, mr);
    keel::f32 *od = out.singleDataMut();
    for (keel::usize i = 0; i < N; ++i) od[i] = keel::as<keel::f32>(out_buf[i]);
    return out;
}

} // namespace numkit::image
