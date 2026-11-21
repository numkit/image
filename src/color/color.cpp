// toolboxes/image/src/color/color.cpp
//
// Colour-space conversions. Portable scalar implementation; SIMD
// optimisation deferred to a later phase.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/math.h>
#include <keel/image.h>
#include <keel/string.h>
#include <numkit/image/color/color.hpp>
#include <numkit/image/internal/type_gates.hpp>
#include <numkit/image/type_convert/type_convert.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <keel/static_array.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#include <cmath>
#include <cstdint>
#include <cstring>
#include <utility>

namespace numkit::image {

namespace {

// Detect input layout: returns (H, W, P) where P is 3, and a triple of
// strides (s_pix, s_chan) so that x[pix_idx, chan] = data[pix_idx*s_pix
// + chan*s_chan]. Supports H×W×3 (s_pix=1, s_chan=H*W) and N×3 colormap
// (s_pix=1, s_chan=N).
struct Layout {
    keel::usize H, W, npix, s_pix, s_chan;
    bool is_3d;
};

Layout detect_layout(const Value &x, const char *fn) {
    const auto &d = x.dims();
    Layout lay{};
    if (d.is3D() && d.pages() == 3) {
        lay.H = d.rows(); lay.W = d.cols();
        lay.npix = lay.H * lay.W;
        lay.s_pix = 1;
        lay.s_chan = lay.npix;
        lay.is_3d = true;
    } else if (!d.is3D() && d.cols() == 3) {
        lay.H = d.rows(); lay.W = 1;
        lay.npix = d.rows();
        lay.s_pix = 1;
        lay.s_chan = lay.npix;
        lay.is_3d = false;
    } else {
        throw Error(keel::format("{}: input must be H×W×3 or N×3", fn),
                    0, 0, fn, "", keel::format("numkit:{}:size", fn));
    }
    return lay;
}

// Read a [0,1] float from an integer image element with class scaling.
inline double element_to_unit(const Value &x, keel::usize i) {
    const double v = x.elemAsDouble(i);
    switch (x.type()) {
        case ValueType::DOUBLE:
        case ValueType::SINGLE:  return v;
        case ValueType::UINT8:   return v / 255.0;
        case ValueType::UINT16:  return v / 65535.0;
        case ValueType::INT16:   return (v + 32768.0) / 65535.0;
        case ValueType::LOGICAL: return v != 0.0 ? 1.0 : 0.0;
        default:                 return v;
    }
}

Value alloc_out_double(const Layout &lay, keel::Allocator *mr) {
    if (lay.is_3d) return Value::matrix3d(lay.H, lay.W, 3, ValueType::DOUBLE, mr);
    return Value::matrix(lay.H, 3, ValueType::DOUBLE, mr);
}

// Generic per-pixel transform.
template <typename Op>
Value pixel_transform(const Value &x, const char *fn, Op op, keel::Allocator *mr) {
    auto lay = detect_layout(x, fn);
    Value out = alloc_out_double(lay, mr);
    if (lay.npix == 0) return out;
    double *od = out.doubleDataMut();

    for (keel::usize p = 0; p < lay.npix; ++p) {
        const double a = element_to_unit(x, p + 0 * lay.s_chan);
        const double b = element_to_unit(x, p + 1 * lay.s_chan);
        const double c = element_to_unit(x, p + 2 * lay.s_chan);
        keel::StaticArray<double, 3> r = op(a, b, c);
        od[p + 0 * lay.s_chan] = r[0];
        od[p + 1 * lay.s_chan] = r[1];
        od[p + 2 * lay.s_chan] = r[2];
    }
    return out;
}

// Same as pixel_transform but without the unit-scale conversion: input
// is read as raw doubles (used for HSV / YCbCr / Lab → RGB where the
// input is already in the appropriate domain).
template <typename Op>
Value pixel_transform_raw(const Value &x, const char *fn, Op op, keel::Allocator *mr) {
    auto lay = detect_layout(x, fn);
    Value out = alloc_out_double(lay, mr);
    if (lay.npix == 0) return out;
    double *od = out.doubleDataMut();

    for (keel::usize p = 0; p < lay.npix; ++p) {
        const double a = x.elemAsDouble(p + 0 * lay.s_chan);
        const double b = x.elemAsDouble(p + 1 * lay.s_chan);
        const double c = x.elemAsDouble(p + 2 * lay.s_chan);
        keel::StaticArray<double, 3> r = op(a, b, c);
        od[p + 0 * lay.s_chan] = r[0];
        od[p + 1 * lay.s_chan] = r[1];
        od[p + 2 * lay.s_chan] = r[2];
    }
    return out;
}

// Convert a [0,1] DOUBLE result into the requested output class, matching
// MATLAB's class-preserving colour conversions: DOUBLE passes through,
// SINGLE is cast, and integer classes are scaled to their full studio
// range, rounded, and saturated.
Value finalize_class(const Value &res01, ValueType cls, keel::Allocator *mr) {
    if (cls == ValueType::DOUBLE) return res01;
    const auto &d = res01.dims();
    Value out = d.is3D() ? Value::matrix3d(d.rows(), d.cols(), d.pages(), cls, mr)
                         : Value::matrix(d.rows(), d.cols(), cls, mr);
    const keel::usize N = res01.numel();
    const double *rd = res01.doubleData();
    for (keel::usize i = 0; i < N; ++i) {
        const double v = rd[i];
        const double c = std::clamp(v, 0.0, 1.0);
        switch (cls) {
            case ValueType::SINGLE: out.singleDataMut()[i] = static_cast<float>(v); break;
            case ValueType::UINT8:  out.uint8DataMut()[i]  = static_cast<keel::u8>(std::lround(c * 255.0)); break;
            case ValueType::UINT16: out.uint16DataMut()[i] = static_cast<keel::u16>(std::lround(c * 65535.0)); break;
            case ValueType::INT16:  out.int16DataMut()[i]  = static_cast<keel::i16>(std::lround(c * 65535.0) - 32768); break;
            default:                 out.doubleDataMut()[i] = v; break;
        }
    }
    return out;
}

// Build a [0,1] DOUBLE copy of an image, scaling integer classes by their
// max (used to normalise integer YCbCr input before the inverse).
Value to_unit_double(const Value &x, keel::Allocator *mr) {
    const auto &d = x.dims();
    Value out = d.is3D() ? Value::matrix3d(d.rows(), d.cols(), d.pages(), ValueType::DOUBLE, mr)
                         : Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
    const keel::usize N = x.numel();
    double *od = out.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) od[i] = element_to_unit(x, i);
    return out;
}

void color_layout_argcheck(const Value &x, const char *fn) {
    (void)detect_layout(x, fn);
}

} // anonymous

// ════════════════════════════════════════════════════════════════════
// RGB ↔ HSV  (MATLAB convention: all channels in [0, 1])
// ════════════════════════════════════════════════════════════════════

Value rgb2hsv(const Value &x, keel::Allocator *mr) {
    color_layout_argcheck(x, "rgb2hsv");
    return pixel_transform(x, "rgb2hsv", [](double r, double g, double b) {
        const auto c = keel::image::rgb_to_hsv(r, g, b);
        return keel::StaticArray<double, 3>{c.c0, c.c1, c.c2};
    }, mr);
}

Value hsv2rgb(const Value &x, keel::Allocator *mr) {
    color_layout_argcheck(x, "hsv2rgb");
    return pixel_transform_raw(x, "hsv2rgb", [](double h, double s, double v) {
        const auto c = keel::image::hsv_to_rgb(h, s, v);
        return keel::StaticArray<double, 3>{c.c0, c.c1, c.c2};
    }, mr);
}

Value rgb2ntsc(const Value &x, keel::Allocator *mr) {
    color_layout_argcheck(x, "rgb2ntsc");
    return pixel_transform(x, "rgb2ntsc", [](double r, double g, double b) {
        const auto c = keel::image::rgb_to_ntsc(r, g, b);
        return keel::StaticArray<double, 3>{c.c0, c.c1, c.c2};
    }, mr);
}

Value ntsc2rgb(const Value &x, keel::Allocator *mr) {
    color_layout_argcheck(x, "ntsc2rgb");
    return pixel_transform_raw(x, "ntsc2rgb", [](double y, double i, double q) {
        const auto c = keel::image::ntsc_to_rgb(y, i, q);
        return keel::StaticArray<double, 3>{c.c0, c.c1, c.c2};
    }, mr);
}

Value rgb2ycbcr(const Value &x, keel::Allocator *mr) {
    color_layout_argcheck(x, "rgb2ycbcr");
    Value res = pixel_transform(x, "rgb2ycbcr", [](double r, double g, double b) {
        const auto c = keel::image::rgb_to_ycbcr(r, g, b);
        return keel::StaticArray<double, 3>{c.c0, c.c1, c.c2};
    }, mr);
    return finalize_class(res, x.type(), mr);
}

Value ycbcr2rgb(const Value &x, keel::Allocator *mr) {
    color_layout_argcheck(x, "ycbcr2rgb");
    const ValueType cls = x.type();
    const bool isInt = (cls == ValueType::UINT8 || cls == ValueType::UINT16
                        || cls == ValueType::INT16);
    Value src = isInt ? to_unit_double(x, mr) : x;
    Value res = pixel_transform_raw(src, "ycbcr2rgb", [](double y, double cb, double cr) {
        const auto c = keel::image::ycbcr_to_rgb(y, cb, cr);
        return keel::StaticArray<double, 3>{c.c0, c.c1, c.c2};
    }, mr);
    return finalize_class(res, cls, mr);
}

Value rgb2xyz(const Value &x, keel::Allocator *mr) {
    color_layout_argcheck(x, "rgb2xyz");
    return pixel_transform(x, "rgb2xyz", [](double r, double g, double b) {
        const auto c = keel::image::rgb_to_xyz(r, g, b);
        return keel::StaticArray<double, 3>{c.c0, c.c1, c.c2};
    }, mr);
}

Value xyz2rgb(const Value &x, keel::Allocator *mr) {
    color_layout_argcheck(x, "xyz2rgb");
    return pixel_transform_raw(x, "xyz2rgb", [](double X, double Y, double Z) {
        const auto c = keel::image::xyz_to_rgb(X, Y, Z);
        return keel::StaticArray<double, 3>{c.c0, c.c1, c.c2};
    }, mr);
}

Value xyz2lab(const Value &x, keel::Allocator *mr) {
    color_layout_argcheck(x, "xyz2lab");
    return pixel_transform_raw(x, "xyz2lab", [](double X, double Y, double Z) {
        const auto c = keel::image::xyz_to_lab(X, Y, Z);
        return keel::StaticArray<double, 3>{c.c0, c.c1, c.c2};
    }, mr);
}

Value lab2xyz(const Value &x, keel::Allocator *mr) {
    color_layout_argcheck(x, "lab2xyz");
    return pixel_transform_raw(x, "lab2xyz", [](double L, double a, double b) {
        const auto c = keel::image::lab_to_xyz(L, a, b);
        return keel::StaticArray<double, 3>{c.c0, c.c1, c.c2};
    }, mr);
}

Value rgb2lab(const Value &x, keel::Allocator *mr) {
    color_layout_argcheck(x, "rgb2xyz");
    return pixel_transform(x, "rgb2xyz", [](double r, double g, double b) {
        const auto c = keel::image::rgb_to_lab(r, g, b);
        return keel::StaticArray<double, 3>{c.c0, c.c1, c.c2};
    }, mr);
}

Value lab2rgb(const Value &x, keel::Allocator *mr) {
    color_layout_argcheck(x, "lab2xyz");
    return pixel_transform_raw(x, "lab2xyz", [](double L, double a, double b) {
        const auto c = keel::image::lab_to_rgb(L, a, b);
        return keel::StaticArray<double, 3>{c.c0, c.c1, c.c2};
    }, mr);
}

// ════════════════════════════════════════════════════════════════════
// imsplit — split H×W×P → P planes of H×W
// ════════════════════════════════════════════════════════════════════
//
// NK packs an H×W×P volume column-major across all three dims:
// element (r, c, p) sits at linear offset p·H·W + c·H + r. Splitting
// is therefore a contiguous H·W copy per channel — no transpose, no
// stride trickery.

namespace {

inline void copy_plane(const Value &src, keel::usize plane,
                       keel::usize H, keel::usize W, Value &dst)
{
    const keel::usize N = H * W;
    const keel::usize off = plane * N;
    const ValueType T = src.type();
    switch (T) {
        case ValueType::DOUBLE:
            std::memcpy(dst.doubleDataMut(),
                        src.doubleData() + off, N * sizeof(double));
            break;
        case ValueType::SINGLE:
            std::memcpy(dst.singleDataMut(),
                        src.singleData() + off, N * sizeof(float));
            break;
        case ValueType::UINT8:
            std::memcpy(dst.uint8DataMut(),
                        src.uint8Data() + off, N);
            break;
        case ValueType::UINT16:
            std::memcpy(dst.uint16DataMut(),
                        src.uint16Data() + off, N * sizeof(keel::u16));
            break;
        case ValueType::INT16:
            std::memcpy(dst.int16DataMut(),
                        src.int16Data() + off, N * sizeof(keel::i16));
            break;
        case ValueType::LOGICAL:
            std::memcpy(dst.logicalDataMut(),
                        src.logicalData() + off, N);
            break;
        default:
            // Generic fallback — re-read elements.
            for (keel::usize i = 0; i < N; ++i) {
                const double v = src.elemAsDouble(off + i);
                dst.doubleDataMut()[i] = v;
            }
    }
}

} // anonymous

void imsplit(const Value &I, keel::Array<Value> &planes, keel::Allocator *mr)
{
    const auto &d = I.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    const keel::usize P = d.is3D() ? d.pages() : 1;
    planes.clear();
    planes.reserve(P);
    for (keel::usize p = 0; p < P; ++p) {
        Value plane = Value::matrix(H, W, I.type(), mr);
        copy_plane(I, p, H, W, plane);
        planes.push_back(std::move(plane));
    }
}

namespace {

// Normalise an RGB-vector argument to an N×3 row-of-triplets layout.
// Accepts 3-element vectors of any orientation (1×3, 3×1, or 3-D
// linear) and N×3 matrices. Returns the data flattened into a
// keel::Array<double> of length 3*N (row-major: per-colour triple
// stored consecutively).
keel::Array<double> rgb_rows(const Value &v, const char *name) {
    const auto &d = v.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    if (v.numel() == 3) {
        keel::Array<double> out(3);
        for (keel::usize i = 0; i < 3; ++i) out[i] = v.elemAsDouble(i);
        return out;
    }
    if (W != 3)
        throw Error(keel::format("{} must be a 3-element or N-by-3 array", name),
                    0, 0, "colorangle", "", "numkit:colorangle:shape");
    keel::Array<double> out(3 * H);
    for (keel::usize r = 0; r < H; ++r)
        for (keel::usize c = 0; c < 3; ++c)
            // col-major source: idx = c*H + r → row-major dest: r*3 + c.
            out[r * 3 + c] = v.elemAsDouble(c * H + r);
    return out;
}

} // anonymous

namespace {

// Detect LAB layout: Mx3 colormap or MxNx3 image. Returns (npix, plane)
// where plane is the per-channel stride (npix == plane). Throws on
// other shapes.
struct LabLayout {
    keel::usize H, W, npix;
    bool is_3d;
};

LabLayout detect_lab_layout(const Value &v, const char *fn) {
    const auto &d = v.dims();
    LabLayout L{};
    if (d.is3D() && d.pages() == 3) {
        L.H = d.rows(); L.W = d.cols();
        L.npix = L.H * L.W;
        L.is_3d = true;
    } else if (!d.is3D() && d.cols() == 3) {
        L.H = d.rows(); L.W = 1;
        L.npix = d.rows();
        L.is_3d = false;
    } else {
        throw Error(keel::format("{}: LAB must be Mx3 or MxNx3", fn),
                    0, 0, fn, "", keel::format("numkit:{}:size", fn));
    }
    return L;
}

template <typename T>
Value lab_alloc_int(const LabLayout &L, ValueType cls, keel::Allocator *mr)
{
    return L.is_3d ? Value::matrix3d(L.H, L.W, 3, cls, mr)
                   : Value::matrix(L.H, 3, cls, mr);
}

// Per-pixel channel reader / writer in the same layout as detect_lab_layout.
inline keel::usize lab_idx(const LabLayout &L, keel::usize pix, keel::usize ch) {
    return pix + ch * L.npix;
}

void lab_layout_argcheck(const Value &lab, const char *fn)
{
    (void)detect_lab_layout(lab, fn);
    switch (lab.type()) {
        case ValueType::DOUBLE:
        case ValueType::SINGLE:
        case ValueType::UINT8:
        case ValueType::UINT16:
            break;
        default:
            throw Error(keel::format("{}: unsupported LAB class", fn),
                        0, 0, fn, "", keel::format("numkit:{}:cls", fn));
    }
}

} // anonymous

Value lab2double(const Value &lab, keel::Allocator *mr)
{
    if (lab.type() == ValueType::DOUBLE) return lab;
    lab_layout_argcheck(lab, "lab2double");
    auto L = detect_lab_layout(lab, "lab2double");
    Value out = L.is_3d
        ? Value::matrix3d(L.H, L.W, 3, ValueType::DOUBLE, mr)
        : Value::matrix(L.H, 3, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();

    for (keel::usize p = 0; p < L.npix; ++p) {
        const double l_ = lab.elemAsDouble(lab_idx(L, p, 0));
        const double a_ = lab.elemAsDouble(lab_idx(L, p, 1));
        const double b_ = lab.elemAsDouble(lab_idx(L, p, 2));
        double L_, a, b;
        switch (lab.type()) {
            case ValueType::UINT8:
                L_ = l_ * (100.0 / 255.0);
                a  = a_ - 128.0;
                b  = b_ - 128.0;
                break;
            case ValueType::UINT16:
                L_ = l_ * (100.0 / 65280.0);
                a  = a_ * (255.0 / 65280.0) - 128.0;
                b  = b_ * (255.0 / 65280.0) - 128.0;
                break;
            case ValueType::SINGLE:
                L_ = l_; a = a_; b = b_;
                break;
            default:
                throw Error("lab2double: unsupported LAB class",
                            0, 0, "lab2double", "", "numkit:lab2double:cls");
        }
        od[lab_idx(L, p, 0)] = L_;
        od[lab_idx(L, p, 1)] = a;
        od[lab_idx(L, p, 2)] = b;
    }
    return out;
}

Value lab2single(const Value &lab, keel::Allocator *mr)
{
    if (lab.type() == ValueType::SINGLE) return lab;
    lab_layout_argcheck(lab, "lab2double");
    Value asDouble = lab2double(lab, mr);
    auto L = detect_lab_layout(asDouble, "lab2single");
    Value out = L.is_3d
        ? Value::matrix3d(L.H, L.W, 3, ValueType::SINGLE, mr)
        : Value::matrix(L.H, 3, ValueType::SINGLE, mr);
    float *of = out.singleDataMut();
    const double *src = asDouble.doubleData();
    for (keel::usize i = 0; i < 3 * L.npix; ++i)
        of[i] = static_cast<float>(src[i]);
    return out;
}

Value lab2uint8(const Value &lab, keel::Allocator *mr)
{
    if (lab.type() == ValueType::UINT8) return lab;
    lab_layout_argcheck(lab, "lab2uint8");
    auto L = detect_lab_layout(lab, "lab2uint8");
    Value out = L.is_3d
        ? Value::matrix3d(L.H, L.W, 3, ValueType::UINT8, mr)
        : Value::matrix(L.H, 3, ValueType::UINT8, mr);
    keel::u8 *od = out.uint8DataMut();

    for (keel::usize p = 0; p < L.npix; ++p) {
        for (keel::usize ch = 0; ch < 3; ++ch) {
            const double v = lab.elemAsDouble(lab_idx(L, p, ch));
            double w;
            switch (lab.type()) {
                case ValueType::SINGLE:
                case ValueType::DOUBLE: {
                    if (std::isnan(v)) { w = 255.0; break; }
                    // MATLAB-compat: (v * 255) / 100 (exact for integer L)
                    // vs v * (255/100) where 255/100 = 2.5499… in double
                    // — half-cases (e.g. L=50 → 127.5) round to 128 only
                    // with the multiply-first form.
                    w = (ch == 0) ? (v * 255.0) / 100.0
                                  : v + 128.0;
                    break;
                }
                case ValueType::UINT16:
                    w = v / 256.0;
                    break;
                default:
                    throw Error("lab2uint8: unsupported LAB class",
                                0, 0, "lab2uint8", "", "numkit:lab2uint8:cls");
            }
            if (w < 0)   w = 0;
            if (w > 255) w = 255;
            od[lab_idx(L, p, ch)] = static_cast<keel::u8>(std::lround(w));
        }
    }
    return out;
}

Value lab2uint16(const Value &lab, keel::Allocator *mr)
{
    if (lab.type() == ValueType::UINT16) return lab;
    lab_layout_argcheck(lab, "lab2uint16");
    auto L = detect_lab_layout(lab, "lab2uint16");
    Value out = L.is_3d
        ? Value::matrix3d(L.H, L.W, 3, ValueType::UINT16, mr)
        : Value::matrix(L.H, 3, ValueType::UINT16, mr);
    keel::u16 *od = out.uint16DataMut();

    for (keel::usize p = 0; p < L.npix; ++p) {
        for (keel::usize ch = 0; ch < 3; ++ch) {
            const double v = lab.elemAsDouble(lab_idx(L, p, ch));
            double w;
            switch (lab.type()) {
                case ValueType::SINGLE:
                case ValueType::DOUBLE: {
                    if (std::isnan(v)) { w = 65535.0; break; }
                    // MATLAB-compat: multiply first to keep integer L
                    // exact (v * 65280 / 100). Same fix as lab2uint8.
                    // For a/b: 65280/255 = 256 exact, so order doesn't
                    // matter — kept as-is.
                    w = (ch == 0) ? (v * 65280.0) / 100.0
                                  : (v + 128.0) * (65280.0 / 255.0);
                    break;
                }
                case ValueType::UINT8:
                    w = v * 256.0;
                    break;
                default:
                    throw Error("lab2uint16: unsupported LAB class",
                                0, 0, "lab2uint16", "", "numkit:lab2uint16:cls");
            }
            if (w < 0)     w = 0;
            if (w > 65535) w = 65535;
            od[lab_idx(L, p, ch)] = static_cast<keel::u16>(std::lround(w));
        }
    }
    return out;
}

namespace {

void wavelength_to_rgb(double lambda, double gamma,
                       double &r, double &g, double &b)
{
    r = g = b = 0.0;
    if (lambda < 380.0 || lambda > 780.0) return;   // out-of-band → black

    if (lambda < 440.0) {
        r = -(lambda - 440.0) / 60.0;
        b = 1.0;
    } else if (lambda < 490.0) {
        g = (lambda - 440.0) / 50.0;
        b = 1.0;
    } else if (lambda < 510.0) {
        g = 1.0;
        b = -(lambda - 510.0) / 20.0;
    } else if (lambda < 580.0) {
        r = (lambda - 510.0) / 70.0;
        g = 1.0;
    } else if (lambda < 645.0) {
        r = 1.0;
        g = -(lambda - 645.0) / 65.0;
    } else {
        r = 1.0;
    }

    double factor = 0.0;
    if (lambda >= 380.0 && lambda < 420.0)
        factor = 0.3 + 0.7 * (lambda - 380.0) / 40.0;
    else if (lambda <= 700.0)
        factor = 1.0;
    else if (lambda <= 780.0)
        factor = 0.3 + 0.7 * (780.0 - lambda) / 80.0;

    r = std::pow(r * factor, gamma);
    g = std::pow(g * factor, gamma);
    b = std::pow(b * factor, gamma);
}

void colorgradient_argcheck(const Value &C, const Value &w, int n)
{
    internal::requireImageClass(C, "colorgradient", "C", 1,
                                internal::ImgClasses::AnyNumericReal);
    if (C.dims().cols() != 3)
        throw Error("colorgradient: C must be K-by-3",
                    0, 0, "colorgradient", "", "numkit:colorgradient:C");
    const int K = static_cast<int>(C.dims().rows());
    if (K < 2)
        throw Error("colorgradient: C must have at least 2 rows",
                    0, 0, "colorgradient", "", "numkit:colorgradient:rows");
    if (n < 2)
        throw Error("colorgradient: n must be >= 2",
                    0, 0, "colorgradient", "", "numkit:colorgradient:n");
    if (w.numel() != 0 && static_cast<int>(w.numel()) != K - 1)
        throw Error("colorgradient: must have one weight per interval",
                    0, 0, "colorgradient", "", "numkit:colorgradient:w");
}

void wavelength2rgb_argcheck(double gamma, keel::StringRef out_class)
{
    if (!(gamma >= 0.0 && gamma <= 1.0))
        throw Error("wavelength2rgb: gamma must be in [0, 1]",
                    0, 0, "wavelength2rgb", "", "numkit:wavelength2rgb:gamma");
    if (!out_class.empty() &&
        !out_class.equals_ignore_ascii_case("double") &&
        !out_class.equals_ignore_ascii_case("single") &&
        !out_class.equals_ignore_ascii_case("uint8") &&
        !out_class.equals_ignore_ascii_case("uint16") &&
        !out_class.equals_ignore_ascii_case("int16"))
        throw Error("wavelength2rgb: unsupported class",
                    0, 0, "wavelength2rgb", "", "numkit:wavelength2rgb:cls");
}

void colorangle_argcheck(const Value &rgb1, const Value &rgb2)
{
    const auto a = rgb_rows(rgb1, "RGB1");
    const auto b = rgb_rows(rgb2, "RGB2");
    const keel::usize Na = a.size() / 3;
    const keel::usize Nb = b.size() / 3;
    if (Na != Nb && Na != 1 && Nb != 1)
        throw Error("colorangle: RGB1 and RGB2 must broadcast (N or 1)",
                    0, 0, "colorangle", "", "numkit:colorangle:size");
}

void xyz2double_argcheck(const Value &xyz)
{
    const auto &d = xyz.dims();
    const bool ok_2d = !d.is3D() && d.cols() == 3;
    const bool ok_3d = d.is3D() && d.pages() == 3;
    if (!ok_2d && !ok_3d)
        throw Error("xyz2double: input must be M-by-3 or H-by-W-by-3",
                    0, 0, "xyz2double", "", "numkit:xyz2double:size");
    const ValueType t = xyz.type();
    if (t != ValueType::DOUBLE && t != ValueType::UINT16)
        throw Error("xyz2double: input must be uint16 or double",
                    0, 0, "xyz2double", "", "numkit:xyz2double:type");
}

void contrast_argcheck(const Value &x, int m)
{
    if (m < 2)
        throw Error("contrast: M must be >= 2",
                    0, 0, "contrast", "", "numkit:contrast:m");
    const keel::usize Nx = x.numel();
    if (Nx == 0)
        throw Error("contrast: input image must be non-empty",
                    0, 0, "contrast", "", "numkit:contrast:empty");
}

void brighten_argcheck(const Value &map, double beta)
{
    if (!(beta > -1.0 && beta < 1.0))
        throw Error("brighten: BETA must be a scalar in the range (-1, 1)",
                    0, 0, "brighten", "", "numkit:brighten:beta");
    const auto &d = map.dims();
    if (d.is3D() || d.cols() != 3)
        throw Error("brighten: input must be an N-by-3 colormap",
                    0, 0, "brighten", "", "numkit:brighten:shape");
}

void xyz2uint16_argcheck(const Value &xyz)
{
    const auto &d = xyz.dims();
    const bool ok_2d = !d.is3D() && d.cols() == 3;
    const bool ok_3d = d.is3D() && d.pages() == 3;
    if (!ok_2d && !ok_3d)
        throw Error("xyz2uint16: input must be M-by-3 or H-by-W-by-3",
                    0, 0, "xyz2uint16", "", "numkit:xyz2uint16:size");
    const ValueType t = xyz.type();
    if (t != ValueType::DOUBLE && t != ValueType::UINT16)
        throw Error("xyz2uint16: input must be uint16 or double",
                    0, 0, "xyz2uint16", "", "numkit:xyz2uint16:type");
}

void deltaE_argcheck(const Value &I1, const Value &I2)
{
    const auto &d1 = I1.dims();
    const auto &d2 = I2.dims();
    if (!d1.is3D() && !d2.is3D()) {
        if (d1.cols() != 3 || d2.cols() != 3 ||
            d1.rows() != d2.rows())
            throw Error("deltaE: M-by-3 inputs must agree in row count",
                        0, 0, "deltaE", "", "numkit:deltaE:size");
    } else if (d1.is3D() && d2.is3D()) {
        if (d1.pages() != 3 || d2.pages() != 3 ||
            d1.rows() != d2.rows() || d1.cols() != d2.cols())
            throw Error("deltaE: H-by-W-by-3 inputs must agree in size",
                        0, 0, "deltaE", "", "numkit:deltaE:size");
    } else {
        throw Error("deltaE: I1 and I2 must both be M-by-3 or H-by-W-by-3",
                    0, 0, "deltaE", "", "numkit:deltaE:size");
    }
}

void whitepoint_argcheck(keel::StringRef illuminant)
{
    if (!illuminant.empty() &&
        !illuminant.equals_ignore_ascii_case("a") &&
        !illuminant.equals_ignore_ascii_case("c") &&
        !illuminant.equals_ignore_ascii_case("d50") &&
        !illuminant.equals_ignore_ascii_case("d55") &&
        !illuminant.equals_ignore_ascii_case("d65") &&
        !illuminant.equals_ignore_ascii_case("e") &&
        !illuminant.equals_ignore_ascii_case("icc"))
        throw Error(keel::format("whitepoint: unsupported illuminant '{}' "
                                 "(use a, c, d50, d55, d65, e, or icc)",
                                 illuminant),
                    0, 0, "whitepoint", "", "numkit:whitepoint:illum");
}

void cmap2gray_argcheck(const Value &cmap)
{
    const auto &d = cmap.dims();
    if (d.is3D() || d.cols() != 3 || d.rows() < 1)
        throw Error("cmap2gray: CMAP must be an N-by-3 colormap",
                    0, 0, "cmap2gray", "", "numkit:cmap2gray:shape");
}

void label2rgb_argcheck(const Value &L, const Value &cmap,
                        const Value &background)
{
    internal::requireImageClass(L, "label2rgb", "L", 1,
                                internal::ImgClasses::AnyNumericReal);
    if (!cmap.isEmpty() && !cmap.isChar() && !cmap.isString())
        internal::requireColormap(cmap, "label2rgb", 2);
    const auto &dL = L.dims();
    if (dL.is3D() && dL.pages() != 1)
        throw Error("label2rgb: L must be a 2-D labelled image",
                    0, 0, "label2rgb", "", "numkit:label2rgb:dims");

    if (cmap.dims().cols() != 3)
        throw Error("label2rgb: CMAP must be N-by-3",
                    0, 0, "label2rgb", "", "numkit:label2rgb:cmap");

    if (background.numel() != 3 && background.numel() != 0)
        throw Error("label2rgb: background must be a 3-element RGB triplet",
                    0, 0, "label2rgb", "", "numkit:label2rgb:bg");
}

} // anonymous

Value colorgradient(const Value &C, const Value &w, int n, keel::Allocator *mr)
{
    colorgradient_argcheck(C, w, n);
    // colorgradient is reference-absent (Octave image package API) —
    // the nk-defined contract takes real numeric colours.
    internal::requireImageClass(C, "colorgradient", "C", 1,
                                internal::ImgClasses::AnyNumericReal);
    if (C.dims().cols() != 3)
        throw Error("colorgradient: C must be K-by-3",
                    0, 0, "colorgradient", "", "numkit:colorgradient:C");
    const int K = static_cast<int>(C.dims().rows());
    if (K < 2)
        throw Error("colorgradient: C must have at least 2 rows",
                    0, 0, "colorgradient", "", "numkit:colorgradient:rows");
    if (n < 2)
        throw Error("colorgradient: n must be >= 2",
                    0, 0, "colorgradient", "", "numkit:colorgradient:n");

    keel::Array<double> wv;
    wv.reserve(K - 1);
    if (w.numel() == 0) {
        wv.resize(K - 1, 1.0);
    } else {
        if (static_cast<int>(w.numel()) != K - 1)
            throw Error("colorgradient: must have one weight per interval",
                        0, 0, "colorgradient", "", "numkit:colorgradient:w");
        for (keel::usize i = 0; i < w.numel(); ++i)
            wv.push_back(w.elemAsDouble(i));
    }

    double total = 0.0;
    for (double x : wv) total += x;
    if (total == 0.0) total = 1.0;
    keel::Array<int> wpos(K, 0);
    double acc = 0.0;
    wpos[0] = 1;
    for (int i = 0; i < K - 1; ++i) {
        acc += wv[(keel::usize)i];
        wpos[(keel::usize)(i + 1)] = static_cast<int>(
            1 + std::lround((n - 1) * acc / total));
    }

    Value map = Value::matrix(static_cast<keel::usize>(n), 3,
                              ValueType::DOUBLE, mr);
    double *md = map.doubleDataMut();

    auto Cval = [&](int r, int c) {
        return C.elemAsDouble(static_cast<keel::usize>(c) * (keel::usize)K + (keel::usize)r);
    };

    for (int i = 0; i < K - 1; ++i) {
        const int p0 = wpos[(keel::usize)i];
        const int p1 = wpos[(keel::usize)(i + 1)];
        if (p0 == p1) continue;
        const int len = p1 - p0 + 1;
        for (int ch = 0; ch < 3; ++ch) {
            const double a = Cval(i,     ch);
            const double b = Cval(i + 1, ch);
            for (int k = 0; k < len; ++k) {
                const double t = (len == 1) ? 0.0 :
                                 static_cast<double>(k) / (len - 1);
                md[(keel::usize)ch * (keel::usize)n + (keel::usize)(p0 - 1 + k)] =
                    a + t * (b - a);
            }
        }
    }
    return map;
}

Value wavelength2rgb(const Value &wavelength, keel::StringRef out_class, double gamma, keel::Allocator *mr)
{
    wavelength2rgb_argcheck(gamma, out_class);
    if (!(gamma >= 0.0 && gamma <= 1.0))
        throw Error("wavelength2rgb: gamma must be in [0, 1]",
                    0, 0, "wavelength2rgb", "", "numkit:wavelength2rgb:gamma");

    const auto &d = wavelength.dims();
    const keel::usize N = wavelength.numel();
    if (N == 0)
        return Value::matrix(0, 0, ValueType::DOUBLE, mr);

    // Decide output shape:
    //   scalar λ            → 1×3 row.
    //   row vector 1×N       → 1×N×3.
    //   column vector N×1   → N×1×3.
    //   matrix M×N (>=2x2)  → M×N×3.
    Value out;
    keel::usize H, W, plane;
    if (N == 1) {
        out = Value::matrix(1, 3, ValueType::DOUBLE, mr);
        H = 1; W = 3; plane = 3;
    } else {
        H = d.rows(); W = d.cols();
        out = Value::matrix3d(H, W, 3, ValueType::DOUBLE, mr);
        plane = H * W;
    }
    double *od = out.doubleDataMut();

    if (N == 1) {
        double r, g, b;
        wavelength_to_rgb(wavelength.toScalar(), gamma, r, g, b);
        od[0] = r; od[1] = g; od[2] = b;
    } else {
        for (keel::usize i = 0; i < N; ++i) {
            double r, g, b;
            wavelength_to_rgb(wavelength.elemAsDouble(i), gamma, r, g, b);
            od[0 * plane + i] = r;
            od[1 * plane + i] = g;
            od[2 * plane + i] = b;
        }
    }

    // Cast to requested class.
    if (out_class.empty() || out_class.equals_ignore_ascii_case("double")) return out;
    if (out_class.equals_ignore_ascii_case("single")) return im2single(out, mr);
    if (out_class.equals_ignore_ascii_case("uint8"))  return im2uint8(out, mr);
    if (out_class.equals_ignore_ascii_case("uint16")) return im2uint16(out, mr);
    if (out_class.equals_ignore_ascii_case("int16"))  return im2int16(out, mr);
    throw Error("wavelength2rgb: unsupported class",
                0, 0, "wavelength2rgb", "", "numkit:wavelength2rgb:cls");
}

Value colorangle(const Value &rgb1, const Value &rgb2, keel::Allocator *mr)
{
    colorangle_argcheck(rgb1, rgb2);
    const auto a = rgb_rows(rgb1, "RGB1");
    const auto b = rgb_rows(rgb2, "RGB2");
    const keel::usize Na = a.size() / 3;
    const keel::usize Nb = b.size() / 3;
    if (Na != Nb && Na != 1 && Nb != 1)
        throw Error("colorangle: RGB1 and RGB2 must broadcast (N or 1)",
                    0, 0, "colorangle", "", "numkit:colorangle:size");
    const keel::usize N = std::max(Na, Nb);

    Value out = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    if (N == 0) return out;
    double *od = out.doubleDataMut();

    for (keel::usize i = 0; i < N; ++i) {
        const double *ai = &a[(Na == 1 ? 0 : i) * 3];
        const double *bi = &b[(Nb == 1 ? 0 : i) * 3];
        const double dot = ai[0]*bi[0] + ai[1]*bi[1] + ai[2]*bi[2];
        const double na  = std::sqrt(ai[0]*ai[0] + ai[1]*ai[1] + ai[2]*ai[2]);
        const double nb  = std::sqrt(bi[0]*bi[0] + bi[1]*bi[1] + bi[2]*bi[2]);
        double angle;
        if (na == 0.0 && nb == 0.0)      angle = 0.0;
        else if (na == 0.0 || nb == 0.0) angle = std::nan("");
        else {
            double c = dot / (na * nb);
            if (c >  1.0) c =  1.0;
            if (c < -1.0) c = -1.0;
            angle = std::acos(c) * 180.0 / M_PI;
        }
        od[i] = angle;
    }
    return out;
}

Value gray_cmap(int n, keel::Allocator *mr)
{
    if (n <= 0) return Value::matrix(0, 3, ValueType::DOUBLE, mr);
    Value out = Value::matrix(static_cast<keel::usize>(n), 3, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    if (n == 1) {
        od[0] = od[1] = od[2] = 0.0;
        return out;
    }
    const double inv = 1.0 / static_cast<double>(n - 1);
    for (int i = 0; i < n; ++i) {
        const double v = static_cast<double>(i) * inv;
        // col-major: column 0 (R), column 1 (G), column 2 (B).
        od[0 * n + i] = v;
        od[1 * n + i] = v;
        od[2 * n + i] = v;
    }
    return out;
}

Value hot_cmap(int n, keel::Allocator *mr)
{
    if (n <= 0) return Value::matrix(0, 3, ValueType::DOUBLE, mr);
    Value out = Value::matrix(static_cast<keel::usize>(n), 3, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    auto R = [&](int r) -> double & { return od[0 * n + r]; };
    auto G = [&](int r) -> double & { return od[1 * n + r]; };
    auto B = [&](int r) -> double & { return od[2 * n + r]; };
    if (n == 1) {
        R(0) = G(0) = B(0) = 1.0;
        return out;
    }
    if (n == 2) {
        R(0) = G(0) = 1.0; B(0) = 0.5;
        R(1) = G(1) = B(1) = 1.0;
        return out;
    }
    // n > 2
    const int idx = static_cast<int>(std::floor(3.0 / 8.0 * n));
    const int nel = idx;
    // R: r(1:idx) = i/nel; r(idx+1:end) = 1.
    for (int i = 0; i < n; ++i) R(i) = 1.0;
    for (int i = 0; i < nel; ++i) R(i) = static_cast<double>(i + 1) / nel;
    // G: g(1:idx) = 0; g(idx+1:2*idx) = (1..idx)/idx; g(2*idx+1:end) = 1.
    for (int i = 0; i < n; ++i) G(i) = 0.0;
    for (int i = 0; i < idx; ++i) G(idx + i) = static_cast<double>(i + 1) / idx;
    for (int i = 2 * idx; i < n; ++i) G(i) = 1.0;
    // B: b(idx2:end) = (1..nel2)/nel2 where idx2 = 2*idx, nel2 = n - 2*idx.
    for (int i = 0; i < n; ++i) B(i) = 0.0;
    const int idx2 = 2 * idx;
    const int nel2 = n - idx2;
    for (int i = 0; i < nel2; ++i) B(idx2 + i) = static_cast<double>(i + 1) / nel2;
    return out;
}

Value cool_cmap(int n, keel::Allocator *mr)
{
    if (n <= 0) return Value::matrix(0, 3, ValueType::DOUBLE, mr);
    Value out = Value::matrix(static_cast<keel::usize>(n), 3, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    if (n == 1) {
        od[0]     = 0.0;  // R
        od[1]     = 1.0;  // G
        od[2]     = 1.0;  // B
        return out;
    }
    const double inv = 1.0 / static_cast<double>(n - 1);
    for (int i = 0; i < n; ++i) {
        const double r = static_cast<double>(i) * inv;
        od[0 * n + i] = r;
        od[1 * n + i] = 1.0 - r;
        od[2 * n + i] = 1.0;
    }
    return out;
}

Value spring_cmap(int n, keel::Allocator *mr)
{
    if (n <= 0) return Value::matrix(0, 3, ValueType::DOUBLE, mr);
    Value out = Value::matrix(static_cast<keel::usize>(n), 3, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    if (n == 1) {
        od[0] = 1.0; od[1] = 0.0; od[2] = 1.0;
        return out;
    }
    const double inv = 1.0 / static_cast<double>(n - 1);
    for (int i = 0; i < n; ++i) {
        const double g = static_cast<double>(i) * inv;
        od[0 * n + i] = 1.0;
        od[1 * n + i] = g;
        od[2 * n + i] = 1.0 - g;
    }
    return out;
}

Value summer_cmap(int n, keel::Allocator *mr)
{
    if (n <= 0) return Value::matrix(0, 3, ValueType::DOUBLE, mr);
    Value out = Value::matrix(static_cast<keel::usize>(n), 3, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    if (n == 1) {
        od[0] = 0.0; od[1] = 0.5; od[2] = 0.4;
        return out;
    }
    const double inv = 1.0 / static_cast<double>(n - 1);
    for (int i = 0; i < n; ++i) {
        const double r = static_cast<double>(i) * inv;
        od[0 * n + i] = r;
        od[1 * n + i] = 0.5 + 0.5 * r;
        od[2 * n + i] = 0.4;
    }
    return out;
}

Value autumn_cmap(int n, keel::Allocator *mr)
{
    if (n <= 0) return Value::matrix(0, 3, ValueType::DOUBLE, mr);
    Value out = Value::matrix(static_cast<keel::usize>(n), 3, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    if (n == 1) {
        od[0] = 1.0; od[1] = 0.0; od[2] = 0.0;
        return out;
    }
    const double inv = 1.0 / static_cast<double>(n - 1);
    for (int i = 0; i < n; ++i) {
        od[0 * n + i] = 1.0;
        od[1 * n + i] = static_cast<double>(i) * inv;
        od[2 * n + i] = 0.0;
    }
    return out;
}

Value winter_cmap(int n, keel::Allocator *mr)
{
    if (n <= 0) return Value::matrix(0, 3, ValueType::DOUBLE, mr);
    Value out = Value::matrix(static_cast<keel::usize>(n), 3, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    if (n == 1) {
        od[0] = 0.0; od[1] = 0.0; od[2] = 1.0;
        return out;
    }
    const double inv = 1.0 / static_cast<double>(n - 1);
    for (int i = 0; i < n; ++i) {
        const double g = static_cast<double>(i) * inv;
        od[0 * n + i] = 0.0;
        od[1 * n + i] = g;
        od[2 * n + i] = 1.0 - 0.5 * g;
    }
    return out;
}

Value copper_cmap(int n, keel::Allocator *mr)
{
    if (n <= 0) return Value::matrix(0, 3, ValueType::DOUBLE, mr);
    Value out = Value::matrix(static_cast<keel::usize>(n), 3, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    if (n == 1) {
        od[0] = od[1] = od[2] = 0.0;
        return out;
    }
    const double inv = 1.0 / static_cast<double>(n - 1);
    for (int i = 0; i < n; ++i) {
        const double x = static_cast<double>(i) * inv;
        double r = 1.25 * x;
        if (r > 1.0) r = 1.0;
        od[0 * n + i] = r;
        od[1 * n + i] = 0.7812 * x;
        od[2 * n + i] = 0.4975 * x;
    }
    return out;
}

Value pink_cmap(int n, keel::Allocator *mr)
{
    if (n <= 0) return Value::matrix(0, 3, ValueType::DOUBLE, mr);
    Value out = Value::matrix(static_cast<keel::usize>(n), 3, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    if (n == 1) {
        const double v = std::sqrt(1.0 / 3.0);
        od[0] = v; od[1] = v; od[2] = v;
        return out;
    }
    if (n == 2) {
        const double s = std::sqrt(1.0 / 3.0);
        const double s2 = std::sqrt(1.0 / 6.0);
        od[0 * n + 0] = s;  od[1 * n + 0] = s;  od[2 * n + 0] = s2;
        od[0 * n + 1] = 1;  od[1 * n + 1] = 1;  od[2 * n + 1] = 1;
        return out;
    }
    // n > 2.
    const int IDX = static_cast<int>(std::floor(3.0 / 8.0 * n));
    const double base = 1.0 / (3.0 * IDX);
    const double inv  = 1.0 / static_cast<double>(n - 1);
    keel::Array<double> r(n), g(n), b(n);

    // R channel.
    const double r_end = (2.0 / 3.0) * (IDX - 1) * inv + 1.0 / 3.0;
    if (IDX == 1) {
        r[0] = base;
    } else {
        for (int i = 0; i < IDX; ++i)
            r[i] = base + (r_end - base) * i / (IDX - 1);
    }
    for (int i = IDX; i < n; ++i)
        r[i] = (2.0 / 3.0) * i * inv + 1.0 / 3.0;

    // G channel: 3-piece. Initial fill, then linspace overwrites overlap.
    for (int i = 0; i < IDX; ++i)
        g[i] = (2.0 / 3.0) * i * inv;
    {
        const double a = (2.0 / 3.0) * (IDX - 1) * inv;
        const double bb = (2.0 / 3.0) * (2 * IDX - 1) * inv + 1.0 / 3.0;
        const int len = IDX + 1;
        for (int j = 0; j < len; ++j) {
            const int pos = IDX - 1 + j;
            if (pos < 0 || pos >= n) continue;
            g[pos] = a + (bb - a) * j / (len - 1);
        }
    }
    for (int i = 2 * IDX; i < n; ++i)
        g[i] = (2.0 / 3.0) * i * inv + 1.0 / 3.0;

    // B channel: linear up to 2*IDX-1, then linspace to 1.
    const int upto = std::min(2 * IDX, n);
    for (int i = 0; i < upto; ++i)
        b[i] = (2.0 / 3.0) * i * inv;
    {
        const int b_start = 2 * IDX - 1;
        if (b_start >= 0 && b_start < n) {
            const int len = n - 2 * IDX + 1;
            const double a = (2.0 / 3.0) * b_start * inv;
            for (int j = 0; j < len; ++j) {
                const int pos = b_start + j;
                if (pos >= n) break;
                b[pos] = a + (1.0 - a) * j / (len - 1);
            }
        }
    }

    for (int i = 0; i < n; ++i) {
        od[0 * n + i] = std::sqrt(r[i]);
        od[1 * n + i] = std::sqrt(g[i]);
        od[2 * n + i] = std::sqrt(b[i]);
    }
    return out;
}

Value hsv_cmap(int n, keel::Allocator *mr)
{
    if (n <= 0) return Value::matrix(0, 3, ValueType::DOUBLE, mr);
    if (n == 1) {
        Value out = Value::matrix(1, 3, ValueType::DOUBLE, mr);
        double *od = out.doubleDataMut();
        od[0] = 1.0; od[1] = 0.0; od[2] = 0.0;
        return out;
    }
    // Build N×3 [hue, 1, 1] then dispatch through hsv2rgb.
    Value hsv_in = Value::matrix(static_cast<keel::usize>(n), 3, ValueType::DOUBLE, mr);
    double *id = hsv_in.doubleDataMut();
    const double inv = 1.0 / static_cast<double>(n);
    for (int i = 0; i < n; ++i) {
        id[0 * n + i] = static_cast<double>(i) * inv;  // hue
        id[1 * n + i] = 1.0;                            // saturation
        id[2 * n + i] = 1.0;                            // value
    }
    return hsv2rgb(hsv_in, mr);
}

Value flag_cmap(int n, keel::Allocator *mr)
{
    if (n <= 0) return Value::matrix(0, 3, ValueType::DOUBLE, mr);
    Value out = Value::matrix(static_cast<keel::usize>(n), 3, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    static constexpr double kFlag[4][3] = {
        {1.0, 0.0, 0.0}, {1.0, 1.0, 1.0}, {0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}
    };
    for (int i = 0; i < n; ++i) {
        const int k = i % 4;
        od[0 * n + i] = kFlag[k][0];
        od[1 * n + i] = kFlag[k][1];
        od[2 * n + i] = kFlag[k][2];
    }
    return out;
}

Value prism_cmap(int n, keel::Allocator *mr)
{
    if (n <= 0) return Value::matrix(0, 3, ValueType::DOUBLE, mr);
    Value out = Value::matrix(static_cast<keel::usize>(n), 3, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    static constexpr double kPrism[6][3] = {
        {1.0, 0.0,        0.0},
        {1.0, 0.5,        0.0},
        {1.0, 1.0,        0.0},
        {0.0, 1.0,        0.0},
        {0.0, 0.0,        1.0},
        {2.0 / 3.0, 0.0,  1.0}
    };
    for (int i = 0; i < n; ++i) {
        const int k = i % 6;
        od[0 * n + i] = kPrism[k][0];
        od[1 * n + i] = kPrism[k][1];
        od[2 * n + i] = kPrism[k][2];
    }
    return out;
}

Value lines_cmap(int n, keel::Allocator *mr)
{
    if (n <= 0) return Value::matrix(0, 3, ValueType::DOUBLE, mr);
    Value out = Value::matrix(static_cast<keel::usize>(n), 3, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    if (n == 1) {
        // MATLAB special-case: lines(1) → [0 0 1] (last fallback blue).
        od[0] = 0.0; od[1] = 0.0; od[2] = 1.0;
        return out;
    }
    // MATLAB R2025b factory axes colororder (7 rows).
    static constexpr double kLines[7][3] = {
        {0.066, 0.443, 0.745},
        {0.866, 0.329, 0.000},
        {0.929, 0.694, 0.125},
        {0.521, 0.086, 0.819},
        {0.231, 0.666, 0.196},
        {0.184, 0.745, 0.937},
        {0.819, 0.015, 0.545}
    };
    for (int i = 0; i < n; ++i) {
        const int k = i % 7;
        od[0 * n + i] = kLines[k][0];
        od[1 * n + i] = kLines[k][1];
        od[2 * n + i] = kLines[k][2];
    }
    return out;
}

Value bone_cmap(int n, keel::Allocator *mr)
{
    if (n <= 0) return Value::matrix(0, 3, ValueType::DOUBLE, mr);
    Value out = Value::matrix(static_cast<keel::usize>(n), 3, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    if (n == 1) {
        od[0] = 0.125; od[1] = 0.125; od[2] = 0.125;
        return out;
    }
    if (n == 2) {
        od[0 * n + 0] = 1.0 / 16.0;
        od[1 * n + 0] = 0.125;
        od[2 * n + 0] = 0.125;
        od[0 * n + 1] = 1.0;
        od[1 * n + 1] = 1.0;
        od[2 * n + 1] = 1.0;
        return out;
    }
    // n > 2.
    const double inv = 1.0 / static_cast<double>(n - 1);
    auto x = [&](int i) { return static_cast<double>(i) * inv; };
    keel::Array<double> r(n), g(n), b(n);

    // R channel.
    const int IDX_R = static_cast<int>(std::floor(0.75 * n));
    const int nel_R = n - IDX_R + 1;
    const int rem8  = n % 8;
    double base_R = 0.0;
    if (rem8 == 2 || rem8 == 4)        base_R = 1.0 / (16.0 + 2.0 * (n - rem8));
    else if (rem8 == 5 || rem8 == 7)   base_R = 1.0 / (24.0 + 2.0 * (n - rem8));
    for (int i = 0; i < IDX_R; ++i) r[i] = (7.0 / 8.0) * x(i);
    {
        const double a  = (7.0 / 8.0) * x(IDX_R - 1) + base_R;
        const double bb = 1.0;
        for (int j = 0; j < nel_R; ++j) {
            const int pos = IDX_R - 1 + j;
            if (pos < 0 || pos >= n) continue;
            r[pos] = a + (bb - a) * j / (nel_R - 1);
        }
    }

    // G channel.
    const int IDX_G = static_cast<int>(std::floor(0.375 * n));
    for (int i = 0; i < IDX_G; ++i) g[i] = (7.0 / 8.0) * x(i);
    {
        const int len = IDX_G + 1;  // 1-based length idx..2*idx
        const double a  = (7.0 / 8.0) * x(IDX_G - 1);
        const double bb = (7.0 / 8.0) * x(2 * IDX_G - 1) + 0.125;
        for (int j = 0; j < len; ++j) {
            const int pos = IDX_G - 1 + j;
            if (pos < 0 || pos >= n) continue;
            g[pos] = a + (bb - a) * j / (len - 1);
        }
    }
    for (int i = 2 * IDX_G; i < n; ++i)
        g[i] = (7.0 / 8.0) * x(i) + 0.125;

    // B channel.
    {
        const double base_B = 1.0 / (8.0 * IDX_G);
        const double a  = base_B;
        const double bb = (7.0 / 8.0) * x(IDX_G - 1) + 0.125;
        const int nel_B = IDX_G;
        for (int j = 0; j < nel_B; ++j) {
            const int pos = j;
            if (pos < 0 || pos >= n) continue;
            b[pos] = a + (bb - a) * j / (nel_B - 1);
        }
    }
    for (int i = IDX_G - 1; i < n; ++i)
        b[i] = (7.0 / 8.0) * x(i) + 0.125;

    for (int i = 0; i < n; ++i) {
        od[0 * n + i] = r[i];
        od[1 * n + i] = g[i];
        od[2 * n + i] = b[i];
    }
    return out;
}

Value rgb2lin(const Value &A, keel::Allocator *mr)
{
    // MATLAB R2025b convention: int classes float to single; else keep
    // input class. Range outside [0, 1] is allowed — the function uses
    // sign(x) * f(|x|) so negatives mirror through the gamma curve.
    Value in;
    if (A.type() == ValueType::DOUBLE) in = A;
    else if (A.type() == ValueType::SINGLE) in = A;
    else in = im2single(A, mr);

    auto sRGBtoLin = [](double x) -> double {
        const double s = (x < 0.0) ? -1.0 : 1.0;
        const double ax = keel::abs(x);
        if (ax < 0.04045) return s * (ax / 12.92);
        return s * std::pow((ax + 0.055) / 1.055, 2.4);
    };

    const auto &d = in.dims();
    Value out = d.is3D()
        ? Value::matrix3d(d.rows(), d.cols(), d.pages(), in.type(), mr)
        : Value::matrix(d.rows(), d.cols(), in.type(), mr);
    const keel::usize N = in.numel();

    if (in.type() == ValueType::DOUBLE) {
        const double *pin = in.doubleData();
        double *pout = out.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i) pout[i] = sRGBtoLin(pin[i]);
    } else {
        const float *pin = in.singleData();
        float *pout = out.singleDataMut();
        for (keel::usize i = 0; i < N; ++i)
            pout[i] = static_cast<float>(sRGBtoLin(static_cast<double>(pin[i])));
    }
    return out;
}

Value xyz2double(const Value &xyz, keel::Allocator *mr)
{
    xyz2double_argcheck(xyz);
    const auto &d = xyz.dims();
    const bool ok_2d = !d.is3D() && d.cols() == 3;
    const bool ok_3d = d.is3D() && d.pages() == 3;
    if (!ok_2d && !ok_3d)
        throw Error("xyz2double: input must be M-by-3 or H-by-W-by-3",
                    0, 0, "xyz2double", "", "numkit:xyz2double:size");
    const ValueType t = xyz.type();
    if (t != ValueType::DOUBLE && t != ValueType::UINT16)
        throw Error("xyz2double: input must be uint16 or double",
                    0, 0, "xyz2double", "", "numkit:xyz2double:type");

    Value out = ok_3d
        ? Value::matrix3d(d.rows(), d.cols(), d.pages(), ValueType::DOUBLE, mr)
        : Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
    const keel::usize N = xyz.numel();
    double *od = out.doubleDataMut();
    if (t == ValueType::DOUBLE) {
        const double *id = xyz.doubleData();
        for (keel::usize i = 0; i < N; ++i) od[i] = id[i];
    } else {
        // ICC: uint16 32768 ↔ 1.0 → divide by 32768.
        const keel::u16 *id = xyz.uint16Data();
        constexpr double inv32768 = 1.0 / 32768.0;
        for (keel::usize i = 0; i < N; ++i)
            od[i] = static_cast<double>(id[i]) * inv32768;
    }
    return out;
}

Value contrast(const Value &x, int m, keel::Allocator *mr)
{
    contrast_argcheck(x, m);
    if (m < 2)
        throw Error("contrast: M must be >= 2",
                    0, 0, "contrast", "", "numkit:contrast:m");
    const keel::usize Nx = x.numel();
    if (Nx == 0)
        throw Error("contrast: input image must be non-empty",
                    0, 0, "contrast", "", "numkit:contrast:empty");

    double xmin = x.elemAsDouble(0), xmax = xmin;
    for (keel::usize i = 1; i < Nx; ++i) {
        const double v = x.elemAsDouble(i);
        if (v < xmin) xmin = v;
        if (v > xmax) xmax = v;
    }
    const double range = xmax - xmin;
    // MATLAB rounds (m-1)*(x-xmin)/range. With xmin == xmax, div by zero —
    // MATLAB yields NaN map; we throw to keep output well-defined.
    if (!(range > 0.0))
        throw Error("contrast: image must have non-zero intensity range",
                    0, 0, "contrast", "", "numkit:contrast:flat");

    keel::Array<double> sorted;
    sorted.reserve(Nx + static_cast<keel::usize>(m + 1));
    const double inv_range = 1.0 / range;
    for (keel::usize i = 0; i < Nx; ++i) {
        const double v = x.elemAsDouble(i);
        sorted.push_back(std::round((m - 1) * (v - xmin) * inv_range));
    }
    for (int k = 0; k <= m; ++k)
        sorted.push_back(static_cast<double>(k));
    std::sort(sorted.begin(), sorted.end());

    // f = 1-based positions where sorted[i+1] != sorted[i].
    keel::Array<double> f;
    f.reserve(m);
    for (keel::usize i = 0; i + 1 < sorted.size(); ++i)
        if (sorted[i + 1] != sorted[i])
            f.push_back(static_cast<double>(i + 1));
    if (f.empty())
        return Value::matrix(0, 3, ValueType::DOUBLE, mr);
    const double fmax = f.back();  // sorted, last is max
    const keel::usize N = f.size();
    Value out = Value::matrix(N, 3, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) {
        const double v = f[i] / fmax;
        od[0 * N + i] = v;
        od[1 * N + i] = v;
        od[2 * N + i] = v;
    }
    return out;
}

Value brighten(const Value &map, double beta, keel::Allocator *mr)
{
    brighten_argcheck(map, beta);
    if (!(beta > -1.0 && beta < 1.0))
        throw Error("brighten: BETA must be a scalar in the range (-1, 1)",
                    0, 0, "brighten", "", "numkit:brighten:beta");
    const double gamma = (beta > 0.0) ? (1.0 - beta) : (1.0 / (1.0 + beta));
    const auto &d = map.dims();
    if (d.is3D() || d.cols() != 3)
        throw Error("brighten: input must be an N-by-3 colormap",
                    0, 0, "brighten", "", "numkit:brighten:shape");
    const keel::usize N = map.numel();
    Value out = Value::matrix(d.rows(), 3, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) {
        const double v = map.elemAsDouble(i);
        od[i] = std::pow(v, gamma);
    }
    return out;
}

Value xyz2uint16(const Value &xyz, keel::Allocator *mr)
{
    xyz2uint16_argcheck(xyz);
    const auto &d = xyz.dims();
    const bool ok_2d = !d.is3D() && d.cols() == 3;
    const bool ok_3d = d.is3D() && d.pages() == 3;
    if (!ok_2d && !ok_3d)
        throw Error("xyz2uint16: input must be M-by-3 or H-by-W-by-3",
                    0, 0, "xyz2uint16", "", "numkit:xyz2uint16:size");
    const ValueType t = xyz.type();
    if (t != ValueType::DOUBLE && t != ValueType::UINT16)
        throw Error("xyz2uint16: input must be uint16 or double",
                    0, 0, "xyz2uint16", "", "numkit:xyz2uint16:type");

    Value out = ok_3d
        ? Value::matrix3d(d.rows(), d.cols(), d.pages(), ValueType::UINT16, mr)
        : Value::matrix(d.rows(), d.cols(), ValueType::UINT16, mr);
    const keel::usize N = xyz.numel();
    keel::u16 *od = out.uint16DataMut();
    if (t == ValueType::UINT16) {
        const keel::u16 *id = xyz.uint16Data();
        for (keel::usize i = 0; i < N; ++i) od[i] = id[i];
    } else {
        const double *id = xyz.doubleData();
        for (keel::usize i = 0; i < N; ++i) {
            const double v = id[i] * 32768.0;
            long r = (v <= 0.0) ? 0L : std::lround(v);
            if (r < 0L) r = 0L;
            if (r > 65535L) r = 65535L;
            od[i] = static_cast<keel::u16>(r);
        }
    }
    return out;
}

Value deltaE(const Value &I1, const Value &I2, bool isInputLab, keel::Allocator *mr)
{
    deltaE_argcheck(I1, I2);
    const auto &d1 = I1.dims();
    const auto &d2 = I2.dims();

    // Determine output shape based on input dims.
    keel::usize out_h, out_w;
    if (!d1.is3D() && !d2.is3D()) {
        if (d1.cols() != 3 || d2.cols() != 3 ||
            d1.rows() != d2.rows())
            throw Error("deltaE: M-by-3 inputs must agree in row count",
                        0, 0, "deltaE", "", "numkit:deltaE:size");
        out_h = d1.rows();
        out_w = 1;
    } else if (d1.is3D() && d2.is3D()) {
        if (d1.pages() != 3 || d2.pages() != 3 ||
            d1.rows() != d2.rows() || d1.cols() != d2.cols())
            throw Error("deltaE: H-by-W-by-3 inputs must agree in size",
                        0, 0, "deltaE", "", "numkit:deltaE:size");
        out_h = d1.rows();
        out_w = d1.cols();
    } else {
        throw Error("deltaE: I1 and I2 must both be M-by-3 or H-by-W-by-3",
                    0, 0, "deltaE", "", "numkit:deltaE:size");
    }

    // Class promotion: double if either is double, else single.
    const bool to_double = (I1.type() == ValueType::DOUBLE) ||
                           (I2.type() == ValueType::DOUBLE);
    Value A = to_double ? im2double(I1, mr) : im2single(I1, mr);
    Value B = to_double ? im2double(I2, mr) : im2single(I2, mr);
    if (!isInputLab) {
        A = rgb2lab(A, mr);
        B = rgb2lab(B, mr);
    }

    Value out = Value::matrix(out_h, out_w,
                              to_double ? ValueType::DOUBLE : ValueType::SINGLE,
                              mr);
    const keel::usize plane = out_h * out_w;
    for (keel::usize i = 0; i < plane; ++i) {
        const double l1 = A.elemAsDouble(0 * plane + i);
        const double a1 = A.elemAsDouble(1 * plane + i);
        const double b1 = A.elemAsDouble(2 * plane + i);
        const double l2 = B.elemAsDouble(0 * plane + i);
        const double a2 = B.elemAsDouble(1 * plane + i);
        const double b2 = B.elemAsDouble(2 * plane + i);
        const double dl = l1 - l2, da = a1 - a2, db = b1 - b2;
        const double v = std::sqrt(dl*dl + da*da + db*db);
        if (to_double) out.doubleDataMut()[i] = v;
        else           out.singleDataMut()[i] = static_cast<float>(v);
    }
    return out;
}

Value whitepoint(keel::StringRef illuminant, keel::Allocator *mr)
{
    whitepoint_argcheck(illuminant);

    double X = 0.0, Y = 0.0, Z = 0.0;
    if (illuminant.equals_ignore_ascii_case("a"))        { X = 1.0985; Y = 1.0; Z = 0.3558; }
    else if (illuminant.equals_ignore_ascii_case("c"))   { X = 0.9807; Y = 1.0; Z = 1.1823; }
    else if (illuminant.equals_ignore_ascii_case("d50")) { X = 0.96419865576090109; Y = 1.0;
                                                           Z = 0.82511648321920425; }
    else if (illuminant.equals_ignore_ascii_case("d55")) { X = 0.9568; Y = 1.0; Z = 0.9214; }
    else if (illuminant.equals_ignore_ascii_case("d65")) { X = 0.95047; Y = 1.0; Z = 1.08883; }
    else if (illuminant.equals_ignore_ascii_case("e"))   { X = 1.0; Y = 1.0; Z = 1.0; }
    else if (illuminant.empty() || illuminant.equals_ignore_ascii_case("icc"))
                                                         { X = 0.96420288085938; Y = 1.0;
                                                           Z = 0.82489013671875; }
    else
        throw Error(keel::format("whitepoint: unsupported illuminant '{}' "
                                 "(use a, c, d50, d55, d65, e, or icc)",
                                 illuminant),
                    0, 0, "whitepoint", "", "numkit:whitepoint:illum");

    Value out = Value::matrix(1, 3, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    od[0] = X; od[1] = Y; od[2] = Z;
    return out;
}

Value lin2rgb(const Value &A, keel::Allocator *mr)
{
    Value in;
    if (A.type() == ValueType::DOUBLE) in = A;
    else if (A.type() == ValueType::SINGLE) in = A;
    else in = im2single(A, mr);

    auto linTosRGB = [](double x) -> double {
        const double s = (x < 0.0) ? -1.0 : 1.0;
        const double ax = keel::abs(x);
        if (ax <= 0.0031308) return s * (12.92 * ax);
        return s * (1.055 * std::pow(ax, 1.0 / 2.4) - 0.055);
    };

    const auto &d = in.dims();
    Value out = d.is3D()
        ? Value::matrix3d(d.rows(), d.cols(), d.pages(), in.type(), mr)
        : Value::matrix(d.rows(), d.cols(), in.type(), mr);
    const keel::usize N = in.numel();

    if (in.type() == ValueType::DOUBLE) {
        const double *pin = in.doubleData();
        double *pout = out.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i) pout[i] = linTosRGB(pin[i]);
    } else {
        const float *pin = in.singleData();
        float *pout = out.singleDataMut();
        for (keel::usize i = 0; i < N; ++i)
            pout[i] = static_cast<float>(linTosRGB(static_cast<double>(pin[i])));
    }
    return out;
}

Value white_cmap(int n, keel::Allocator *mr)
{
    if (n <= 0) return Value::matrix(0, 3, ValueType::DOUBLE, mr);
    Value out = Value::matrix(static_cast<keel::usize>(n), 3, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    const keel::usize total = static_cast<keel::usize>(n) * 3;
    for (keel::usize i = 0; i < total; ++i) od[i] = 1.0;
    return out;
}

Value cmap2gray(const Value &cmap, keel::Allocator *mr)
{
    cmap2gray_argcheck(cmap);
    const auto &d = cmap.dims();
    if (d.is3D() || d.cols() != 3 || d.rows() < 1)
        throw Error("cmap2gray: CMAP must be an N-by-3 colormap",
                    0, 0, "cmap2gray", "", "numkit:cmap2gray:shape");
    const keel::usize N = d.rows();
    Value out = Value::matrix(N, 3, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();
    // MATLAB R2020b+ derives Y from inv(YIQ→RGB) — first row of inv
    // matrix [1 0.956 0.621; 1 -0.272 -0.647; 1 -1.106 1.703]. Full
    // double-precision (matches MATLAB's runtime `inv` to 1 ULP):
    // [0.29893602129377539, 0.58704307445112136, 0.11402090425510331].
    // Output is N×3 with the same grayscale value replicated across
    // R/G/B, clipped to [0, 1].
    constexpr double Cr = 0.29893602129377539;
    constexpr double Cg = 0.58704307445112136;
    constexpr double Cb = 0.11402090425510331;
    for (keel::usize i = 0; i < N; ++i) {
        const double r = cmap.elemAsDouble(0 * N + i);
        const double g = cmap.elemAsDouble(1 * N + i);
        const double b = cmap.elemAsDouble(2 * N + i);
        double y = Cr * r + Cg * g + Cb * b;
        if (y < 0.0) y = 0.0;
        if (y > 1.0) y = 1.0;
        od[0 * N + i] = y;
        od[1 * N + i] = y;
        od[2 * N + i] = y;
    }
    return out;
}

Value label2rgb(const Value &L, const Value &cmap, const Value &background, keel::Allocator *mr)
{
    label2rgb_argcheck(L, cmap, background);
    internal::requireImageClass(L, "label2rgb", "L", 1,
                                internal::ImgClasses::AnyNumericReal);
    if (!cmap.isEmpty() && !cmap.isChar() && !cmap.isString())
        internal::requireColormap(cmap, "label2rgb", 2);
    const auto &dL = L.dims();
    const keel::usize H = dL.rows();
    const keel::usize W = dL.cols();
    if (dL.is3D() && dL.pages() != 1)
        throw Error("label2rgb: L must be a 2-D labelled image",
                    0, 0, "label2rgb", "", "numkit:label2rgb:dims");

    if (cmap.dims().cols() != 3)
        throw Error("label2rgb: CMAP must be N-by-3",
                    0, 0, "label2rgb", "", "numkit:label2rgb:cmap");
    const keel::usize N = cmap.dims().rows();

    double bg[3] = {1.0, 1.0, 1.0};
    if (background.numel() == 3) {
        bg[0] = background.elemAsDouble(0);
        bg[1] = background.elemAsDouble(1);
        bg[2] = background.elemAsDouble(2);
    } else if (background.numel() != 0)
        throw Error("label2rgb: background must be a 3-element RGB triplet",
                    0, 0, "label2rgb", "", "numkit:label2rgb:bg");

    Value out = Value::matrix3d(H, W, 3, ValueType::UINT8, mr);
    if (H == 0 || W == 0) return out;
    keel::u8 *od = out.uint8DataMut();
    const keel::usize plane = H * W;
    const double *cd = cmap.doubleData();

    auto sat = [](double v) -> keel::u8 {
        v = std::round(v * 255.0);
        if (v < 0)   v = 0;
        if (v > 255) v = 255;
        return static_cast<keel::u8>(v);
    };

    for (keel::usize i = 0; i < plane; ++i) {
        const double lv = L.elemAsDouble(i);
        const keel::i64 lab = static_cast<keel::i64>(lv);
        if (lv < 0 || lab != static_cast<keel::i64>(lv))
            throw Error("label2rgb: L must be non-negative integer-valued",
                        0, 0, "label2rgb", "", "numkit:label2rgb:value");
        if (lab == 0) {
            od[0 * plane + i] = sat(bg[0]);
            od[1 * plane + i] = sat(bg[1]);
            od[2 * plane + i] = sat(bg[2]);
        } else {
            const keel::usize row = static_cast<keel::usize>(lab) - 1;
            if (row >= N)
                throw Error("label2rgb: CMAP has fewer rows than max label",
                            0, 0, "label2rgb", "", "numkit:label2rgb:short");
            // cmap is N×3 column-major: idx[row, ch] = ch * N + row.
            od[0 * plane + i] = sat(cd[0 * N + row]);
            od[1 * plane + i] = sat(cd[1 * N + row]);
            od[2 * plane + i] = sat(cd[2 * N + row]);
        }
    }
    return out;
}

} // namespace numkit::image
