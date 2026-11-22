// toolboxes/image/src/filter/filter_design.cpp
//
// 3-D filter kernel design (fspecial3) and 2-D windowed FIR design (fwind2),
// delegating numerical kernels to keel::image.

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/image.h>
#include <keel/memory.h>
#include <keel/string.h>
#include <numkit/image/filter/filter.hpp>

#include <numkit/value/error.hpp>
#include <numkit/value/value.hpp>

#include <cstring>

namespace numkit::image {

namespace {

struct Size3 { keel::usize H, W, P; };
struct Sig3 { keel::f64 sx, sy, sz; };

Size3 parseHsize(const Value &hsize)
{
    if (hsize.isEmpty() || hsize.isUnset()) return {5, 5, 5};
    if (hsize.isScalar()) {
        const keel::usize n = keel::as<keel::usize>(hsize.toScalar());
        return {n, n, n};
    }
    if (hsize.numel() == 3) {
        return {keel::as<keel::usize>(hsize.elemAsDouble(0)),
                keel::as<keel::usize>(hsize.elemAsDouble(1)),
                keel::as<keel::usize>(hsize.elemAsDouble(2))};
    }
    throw Error("fspecial3: hsize must be scalar or 3-element vector",
                0, 0, "fspecial3", "", "numkit:fspecial3:BadHsize");
}

Sig3 parseSigma3(const Value &sig, keel::f64 defv = 1.0)
{
    if (sig.isEmpty() || sig.isUnset()) return {defv, defv, defv};
    if (sig.isScalar()) {
        const keel::f64 s = sig.toScalar();
        return {s, s, s};
    }
    if (sig.numel() == 3) {
        return {sig.elemAsDouble(0), sig.elemAsDouble(1), sig.elemAsDouble(2)};
    }
    throw Error("fspecial3: sigma must be scalar or 3-element vector",
                0, 0, "fspecial3", "", "numkit:fspecial3:BadSigma");
}

Sig3 parseSemiaxes(const Value &v, keel::f64 defv = 5.0)
{
    if (v.isEmpty() || v.isUnset()) return {defv, defv, defv};
    if (v.isScalar()) {
        const keel::f64 a = v.toScalar();
        return {a, a, a};
    }
    if (v.numel() == 3) {
        return {v.elemAsDouble(0), v.elemAsDouble(1), v.elemAsDouble(2)};
    }
    throw Error("fspecial3: semiaxes must be a scalar or 3-element vector",
                0, 0, "fspecial3", "", "numkit:fspecial3:BadSemiaxes");
}

} // namespace

Value fspecial3(keel::StringRef type, const Value &a1, const Value &a2, keel::Allocator *mr)
{
    if (type == "average") {
        const Size3 s = parseHsize(a1);
        Value out = Value::matrix3d(s.H, s.W, s.P, ValueType::DOUBLE, mr);
        (void)keel::image::fspecial3_average(s.H, s.W, s.P,
                                             keel::MutArrayRef<keel::f64>(out.doubleDataMut(), s.H * s.W * s.P));
        return out;
    }
    if (type == "gaussian") {
        const Size3 s = parseHsize(a1);
        const Sig3 sg = parseSigma3(a2, 1.0);
        Value out = Value::matrix3d(s.H, s.W, s.P, ValueType::DOUBLE, mr);
        (void)keel::image::fspecial3_gaussian(s.H, s.W, s.P, sg.sx, sg.sy, sg.sz,
                                              keel::MutArrayRef<keel::f64>(out.doubleDataMut(), s.H * s.W * s.P));
        return out;
    }
    if (type == "ellipsoid") {
        const Sig3 ax = parseSemiaxes(a1, 5.0);
        if (ax.sx <= 0.0 || ax.sy <= 0.0 || ax.sz <= 0.0)
            throw Error("fspecial3: ellipsoid semiaxes must be positive",
                        0, 0, "fspecial3", "", "numkit:fspecial3:BadSemiaxes");
        keel::ArenaTemp scratch = keel::get_scratch(mr);
        keel::Array<keel::f64> buf(scratch.allocator());
        keel::usize h = 0, w = 0, p = 0;
        (void)keel::image::fspecial3_ellipsoid(ax.sx, ax.sy, ax.sz, buf, h, w, p, scratch.allocator());
        Value out = Value::matrix3d(h, w, p, ValueType::DOUBLE, mr);
        std::memcpy(out.doubleDataMut(), buf.data(), buf.size() * sizeof(keel::f64));
        return out;
    }
    if (type == "laplacian") {
        const keel::f64 g1 = (!a1.isEmpty() && !a1.isUnset()) ? a1.toScalar() : 0.0;
        const keel::f64 g2 = (!a2.isEmpty() && !a2.isUnset()) ? a2.toScalar() : 0.0;
        if (g1 < 0.0)
            throw Error("fspecial3: Expected input number 2, GAMMA1, to be nonnegative.",
                        0, 0, "fspecial3", "", "numkit:fspecial3:BadGamma");
        if (g2 < 0.0)
            throw Error("fspecial3: Expected input number 3, GAMMA2, to be nonnegative.",
                        0, 0, "fspecial3", "", "numkit:fspecial3:BadGamma");
        if (g1 + g2 > 1.0)
            throw Error("fspecial3: GAMMA1 + GAMMA2 should be less than or equal to 1 and greater than 0.",
                        0, 0, "fspecial3", "", "numkit:fspecial3:BadGamma");
        Value out = Value::matrix3d(3, 3, 3, ValueType::DOUBLE, mr);
        (void)keel::image::fspecial3_laplacian(g1, g2, keel::MutArrayRef<keel::f64>(out.doubleDataMut(), 27u));
        return out;
    }
    if (type == "log") {
        const Size3 s = parseHsize(a1);
        const Sig3 sg = parseSigma3(a2, 1.0);
        keel::ArenaTemp scratch = keel::get_scratch(mr);
        Value out = Value::matrix3d(s.H, s.W, s.P, ValueType::DOUBLE, mr);
        (void)keel::image::fspecial3_log(s.H, s.W, s.P, sg.sx, sg.sy, sg.sz,
                                         keel::MutArrayRef<keel::f64>(out.doubleDataMut(), s.H * s.W * s.P),
                                         scratch.allocator());
        return out;
    }
    if (type == "sobel" || type == "prewitt") {
        keel::String dir = keel::String::from("X");
        if (!a1.isEmpty() && !a1.isUnset() && (a1.isChar() || a1.isString()))
            dir = a1.toString();
        keel::image::Axis3D axis = keel::image::Axis3D::kX;
        if (dir.equals_ignore_ascii_case("x")) axis = keel::image::Axis3D::kX;
        else if (dir.equals_ignore_ascii_case("y")) axis = keel::image::Axis3D::kY;
        else if (dir.equals_ignore_ascii_case("z")) axis = keel::image::Axis3D::kZ;
        else throw Error("fspecial3: direction must be 'X', 'Y', or 'Z'",
                         0, 0, "fspecial3", "", "numkit:fspecial3:BadDir");
        Value out = Value::matrix3d(3, 3, 3, ValueType::DOUBLE, mr);
        (void)keel::image::fspecial3_gradient(type == "sobel", axis,
                                              keel::MutArrayRef<keel::f64>(out.doubleDataMut(), 27u));
        return out;
    }
    throw Error(keel::format("fspecial3: unsupported type '{}'", type),
                0, 0, "fspecial3", "", "numkit:fspecial3:BadType");
}

Value fwind2(keel::Allocator *mr, const Value &Hd, const Value &w)
{
    if (Hd.dims().is3D() || w.dims().is3D())
        throw Error("fwind2: inputs must be 2-D", 0, 0, "fwind2", "", "numkit:fwind2:Not2D");
    const keel::usize H = Hd.dims().rows();
    const keel::usize W = Hd.dims().cols();
    if (w.dims().rows() != H || w.dims().cols() != W)
        throw Error("fwind2: window must match Hd shape",
                    0, 0, "fwind2", "", "numkit:fwind2:ShapeMismatch");

    Value h = Value::matrix(H, W, ValueType::DOUBLE, mr);
    if (H == 0 || W == 0) return h;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const keel::usize N = H * W;
    keel::Array<keel::f64> hdv(N, 0.0, scratch.allocator());
    keel::Array<keel::f64> wv(N, 0.0, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i) {
        hdv[i] = Hd.elemAsDouble(i);
        wv[i] = w.elemAsDouble(i);
    }
    (void)keel::image::fwind2(hdv.as_ref(), wv.as_ref(), H, W,
                              keel::MutArrayRef<keel::f64>(h.doubleDataMut(), N));
    return h;
}

} // namespace numkit::image
