// toolboxes/image/src/geom/geom_reg.cpp
//
// Register half of the image geom builtins: the CallContext wrappers
// delegating to the engine-free compute in geom.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif
#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <cmath>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include <numkit/image/geom/geom.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>
#include "geom/geom_detail.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <tuple>
#include <utility>

namespace numkit::image {

namespace detail {

static keel::String argString(const Value &v) {
    if (!v.isChar() && !v.isString())
        throw Error("image geom: expected a string argument",
                    0, 0, "geom", "", "numkit:image:geom:type");
    return v.toString();
}

void imresize_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                  CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "imresize");
    // no max: name-value tail in the reference
    auto *mr = ctx.engine->allocator();
    keel::String method = keel::String::from("bicubic");   // MATLAB default
    if (args.size() >= 3 && !args[2].isEmpty()) method = argString(args[2]);
    if (args[1].numel() == 1) {
        outs[0] = imresize(args[0], args[1].toScalar(), method, mr);
    } else if (args[1].numel() == 2) {
        const keel::usize outH = keel::usize(args[1].elemAsDouble(0));
        const keel::usize outW = keel::usize(args[1].elemAsDouble(1));
        outs[0] = imresize(args[0], outH, outW, method, mr);
    } else {
        throw Error("imresize: 2nd arg must be a scalar scale "
                    "or a 2-element [outH outW]",
                    0, 0, "imresize", "", "numkit:imresize:size");
    }
}

void imcrop_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 5, "imcrop");
    if (args[1].numel() < 4)
        throw Error("imcrop: rect must have 4 elements",
                    0, 0, "imcrop", "", "numkit:imcrop:rect");
    outs[0] = imcrop(args[0], args[1].elemAsDouble(0), args[1].elemAsDouble(1), args[1].elemAsDouble(2), args[1].elemAsDouble(3), ctx.engine->allocator());
}

void imcrop3_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                 CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "imcrop3");
    outs[0] = imcrop3(args[0], args[1], ctx.engine->allocator());
}

void imrotate_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                  CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 4, "imrotate");
    // MATLAB R2025b default method is 'nearest' (not bilinear), and the
    // 3rd argument may be EITHER a method or a bbox keyword:
    //   imrotate(A, angle, method [, bbox])   or   imrotate(A, angle, bbox)
    keel::String method = keel::String::from("nearest");
    keel::String bbox   = keel::String::from("loose");
    auto isBboxKw = [](keel::StringRef s) {
        return s == "loose" || s == "Loose" || s == "crop" || s == "Crop";
    };
    if (args.size() >= 3 && !args[2].isEmpty()) {
        keel::String s3 = argString(args[2]);
        if (isBboxKw(s3)) {
            bbox = keel::move(s3);           // imrotate(A, angle, bbox)
        } else {
            method = keel::move(s3);         // imrotate(A, angle, method [, bbox])
            if (args.size() >= 4 && !args[3].isEmpty()) bbox = argString(args[3]);
        }
    }
    outs[0] = imrotate(args[0], args[1].toScalar(), method, bbox, ctx.engine->allocator());
}

void imtranslate_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                     CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "imtranslate");
    // no max: name-value tail in the reference
    if (args[1].numel() < 2)
        throw Error("imtranslate: vector must have 2 elements",
                    0, 0, "imtranslate", "", "numkit:imtranslate:vec");
    outs[0] = imtranslate(args[0], args[1].elemAsDouble(0), args[1].elemAsDouble(1), ctx.engine->allocator());
}

void axes2pix_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 3, "axes2pix");
    const keel::f64 n = args[0].toScalar();
    outs[0] = axes2pix(n, args[1], args[2], ctx.engine->allocator());
}

void impyramid_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                   CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "impyramid");
    const keel::String type = argString(args[1]);
    outs[0] = impyramid(args[0], type, ctx.engine->allocator());
}

// ── imrotate3 (3-D volume rotation, MATLAB R2025b imrotate3.m) ───────
//
// Rotate a 3-D numeric volume by `angle` degrees CCW (right-hand
// rule) around the axis vector W = [Wx Wy Wz] passing through the
// centre of the volume. Output size:
//
//   bbox = "loose"  (default)
//     outDim[j] = ceil( sum_k inDim[k] * |R[j][k]| )
//   bbox = "crop"
//     outDim = inDim
//
// Inverse warp:
//   (xs, ys, zs) = inCentre + R * ((xo, yo, zo) - outCentre)
//
// MATLAB's spatial X / Y / Z = col / row / page. Standard Rodrigues
// rotation matrix is used directly: a right-hand-rule CCW rotation
// around W of the centred output delta maps to the source delta.
//
// References:
//   * Rodrigues 1840 — rotation matrix from axis-angle.
//   * Keys 1981 — Catmull-Rom cubic (a = −0.5).
namespace {

struct Rot3 { keel::f64 r[9]; };          // row-major 3 × 3

Rot3 rodrigues(keel::f64 Wx, keel::f64 Wy, keel::f64 Wz, keel::f64 theta)
{
    const keel::f64 n = std::sqrt(Wx*Wx + Wy*Wy + Wz*Wz);
    if (n < 1e-15)
        throw Error("imrotate3: rotation axis must be non-zero",
                    0, 0, "imrotate3", "", "numkit:imrotate3:axis");
    const keel::f64 ux = Wx / n, uy = Wy / n, uz = Wz / n;
    const keel::f64 c = std::cos(theta), s = std::sin(theta), t = 1.0 - c;
    Rot3 R;
    R.r[0] = t*ux*ux + c;    R.r[1] = t*ux*uy - s*uz; R.r[2] = t*ux*uz + s*uy;
    R.r[3] = t*uy*ux + s*uz; R.r[4] = t*uy*uy + c;    R.r[5] = t*uy*uz - s*ux;
    R.r[6] = t*uz*ux - s*uy; R.r[7] = t*uz*uy + s*ux; R.r[8] = t*uz*uz + c;
    return R;
}

inline keel::f64 cubic_keys(keel::f64 x)
{
    const keel::f64 a = std::fabs(x);
    if (a < 1.0) return ((1.5 * a - 2.5) * a * a + 1.0);
    if (a < 2.0) return (((-0.5 * a + 2.5) * a - 4.0) * a + 2.0);
    return 0.0;
}

inline keel::f64 getV3(const Value &V, keel::usize H, keel::usize W, keel::usize D,
                    keel::i32 xi, keel::i32 yi, keel::i32 zi, keel::f64 fill)
{
    // 1-indexed spatial (xi, yi, zi). Out-of-bounds → fill.
    if (xi < 1 || xi > keel::i32(W) || yi < 1 || yi > keel::i32(H) ||
        zi < 1 || zi > keel::i32(D))
        return fill;
    return V.elemAsDouble(keel::usize(yi - 1)
                        + keel::usize(xi - 1) * H
                        + keel::usize(zi - 1) * H * W);
}

keel::f64 samp_nearest3(const Value &V, keel::usize H, keel::usize W, keel::usize D,
                     keel::f64 xs, keel::f64 ys, keel::f64 zs, keel::f64 fill)
{
    const keel::i32 xi = keel::as<keel::i32>(std::floor(xs + 0.5));
    const keel::i32 yi = keel::as<keel::i32>(std::floor(ys + 0.5));
    const keel::i32 zi = keel::as<keel::i32>(std::floor(zs + 0.5));
    return getV3(V, H, W, D, xi, yi, zi, fill);
}

keel::f64 samp_trilinear(const Value &V, keel::usize H, keel::usize W, keel::usize D,
                      keel::f64 xs, keel::f64 ys, keel::f64 zs, keel::f64 fill)
{
    const keel::i32 x0 = keel::as<keel::i32>(std::floor(xs));
    const keel::i32 y0 = keel::as<keel::i32>(std::floor(ys));
    const keel::i32 z0 = keel::as<keel::i32>(std::floor(zs));
    const keel::f64 fx = xs - keel::f64(x0);
    const keel::f64 fy = ys - keel::f64(y0);
    const keel::f64 fz = zs - keel::f64(z0);
    const keel::f64 c000 = getV3(V, H, W, D, x0,     y0,     z0,     fill);
    const keel::f64 c100 = getV3(V, H, W, D, x0 + 1, y0,     z0,     fill);
    const keel::f64 c010 = getV3(V, H, W, D, x0,     y0 + 1, z0,     fill);
    const keel::f64 c110 = getV3(V, H, W, D, x0 + 1, y0 + 1, z0,     fill);
    const keel::f64 c001 = getV3(V, H, W, D, x0,     y0,     z0 + 1, fill);
    const keel::f64 c101 = getV3(V, H, W, D, x0 + 1, y0,     z0 + 1, fill);
    const keel::f64 c011 = getV3(V, H, W, D, x0,     y0 + 1, z0 + 1, fill);
    const keel::f64 c111 = getV3(V, H, W, D, x0 + 1, y0 + 1, z0 + 1, fill);
    const keel::f64 c00 = c000 * (1 - fx) + c100 * fx;
    const keel::f64 c10 = c010 * (1 - fx) + c110 * fx;
    const keel::f64 c01 = c001 * (1 - fx) + c101 * fx;
    const keel::f64 c11 = c011 * (1 - fx) + c111 * fx;
    const keel::f64 c0  = c00  * (1 - fy) + c10  * fy;
    const keel::f64 c1  = c01  * (1 - fy) + c11  * fy;
    return c0 * (1 - fz) + c1 * fz;
}

keel::f64 samp_tricubic(const Value &V, keel::usize H, keel::usize W, keel::usize D,
                     keel::f64 xs, keel::f64 ys, keel::f64 zs, keel::f64 fill)
{
    const keel::i32 x0 = keel::as<keel::i32>(std::floor(xs));
    const keel::i32 y0 = keel::as<keel::i32>(std::floor(ys));
    const keel::i32 z0 = keel::as<keel::i32>(std::floor(zs));
    const keel::f64 fx = xs - keel::f64(x0);
    const keel::f64 fy = ys - keel::f64(y0);
    const keel::f64 fz = zs - keel::f64(z0);
    keel::f64 wx[4], wy[4], wz[4];
    for (keel::i32 k = -1; k <= 2; ++k) {
        wx[k + 1] = cubic_keys(keel::f64(k) - fx);
        wy[k + 1] = cubic_keys(keel::f64(k) - fy);
        wz[k + 1] = cubic_keys(keel::f64(k) - fz);
    }
    keel::f64 sum = 0.0;
    for (keel::i32 kz = -1; kz <= 2; ++kz)
        for (keel::i32 kx = -1; kx <= 2; ++kx) {
            const keel::f64 wxz = wx[kx + 1] * wz[kz + 1];
            for (keel::i32 ky = -1; ky <= 2; ++ky)
                sum += wy[ky + 1] * wxz *
                       getV3(V, H, W, D, x0 + kx, y0 + ky, z0 + kz, fill);
        }
    return sum;
}

} // anonymous (imrotate3 helpers)

Value imrotate3(const Value &V, keel::f64 angle_deg,
                keel::f64 Wx, keel::f64 Wy, keel::f64 Wz,
                keel::StringRef method, keel::StringRef bbox,
                keel::f64 fill,
                keel::Allocator *mr)
{
    const auto &dims = V.dims();
    if (dims.ndims() < 2)
        throw Error("imrotate3: V must be a 3-D volume",
                    0, 0, "imrotate3", "", "numkit:imrotate3:rank");
    const keel::usize H = dims.dim(0);
    const keel::usize W = dims.dim(1);
    const keel::usize D = (dims.ndims() >= 3) ? dims.dim(2) : 1;
    if (V.numel() != H * W * D)
        throw Error("imrotate3: V must be a 3-D numeric volume",
                    0, 0, "imrotate3", "", "numkit:imrotate3:shape");

    const ValueType T = V.type();
    const keel::f64 theta = angle_deg * M_PI / 180.0;
    const Rot3 R = rodrigues(Wx, Wy, Wz, theta);

    auto snap = [](keel::f64 v) {
        const keel::f64 r = std::round(v);
        return (std::fabs(v - r) < 1e-9) ? r : v;
    };

    keel::usize outH, outW, outD;
    if (bbox == "crop" || bbox == "Crop") {
        outH = H; outW = W; outD = D;
    } else if (bbox == "loose" || bbox == "Loose" || bbox.empty()) {
        const keel::f64 Wd = keel::f64(W), Hd = keel::f64(H), Dd = keel::f64(D);
        // outDim[axis] = sum_k inDim[k] * |R[axis][k]|
        const keel::f64 xe = Wd*std::fabs(R.r[0]) + Hd*std::fabs(R.r[1]) + Dd*std::fabs(R.r[2]);
        const keel::f64 ye = Wd*std::fabs(R.r[3]) + Hd*std::fabs(R.r[4]) + Dd*std::fabs(R.r[5]);
        const keel::f64 ze = Wd*std::fabs(R.r[6]) + Hd*std::fabs(R.r[7]) + Dd*std::fabs(R.r[8]);
        outW = std::max(keel::usize(1), keel::usize(std::ceil(snap(xe))));
        outH = std::max(keel::usize(1), keel::usize(std::ceil(snap(ye))));
        outD = std::max(keel::usize(1), keel::usize(std::ceil(snap(ze))));
    } else {
        throw Error("imrotate3: bbox must be 'loose' or 'crop'",
                    0, 0, "imrotate3", "", "numkit:imrotate3:bbox");
    }

    Value B = (outD == 1) ? Value::matrix(outH, outW, T, mr)
                          : Value::matrix3d(outH, outW, outD, T, mr);

    const keel::f64 cxOut = (keel::f64(outW) + 1.0) * 0.5;
    const keel::f64 cyOut = (keel::f64(outH) + 1.0) * 0.5;
    const keel::f64 czOut = (keel::f64(outD) + 1.0) * 0.5;
    const keel::f64 cxIn  = (keel::f64(W) + 1.0) * 0.5;
    const keel::f64 cyIn  = (keel::f64(H) + 1.0) * 0.5;
    const keel::f64 czIn  = (keel::f64(D) + 1.0) * 0.5;

    keel::i32 code = 1;  // 0 = nearest, 1 = linear, 2 = cubic
    if      (method == "nearest" || method == "Nearest") code = 0;
    else if (method == "linear"  || method == "Linear" ||
             method == "bilinear")                       code = 1;
    else if (method == "cubic"   || method == "Cubic")   code = 2;
    else
        throw Error("imrotate3: method must be 'nearest', 'linear', or 'cubic'",
                    0, 0, "imrotate3", "", "numkit:imrotate3:method");

    for (keel::usize p = 1; p <= outD; ++p) {
        const keel::f64 dzo = keel::f64(p) - czOut;
        for (keel::usize c = 1; c <= outW; ++c) {
            const keel::f64 dxo = keel::f64(c) - cxOut;
            for (keel::usize r = 1; r <= outH; ++r) {
                const keel::f64 dyo = keel::f64(r) - cyOut;
                const keel::f64 dxi = R.r[0]*dxo + R.r[1]*dyo + R.r[2]*dzo;
                const keel::f64 dyi = R.r[3]*dxo + R.r[4]*dyo + R.r[5]*dzo;
                const keel::f64 dzi = R.r[6]*dxo + R.r[7]*dyo + R.r[8]*dzo;
                const keel::f64 xs = cxIn + dxi;
                const keel::f64 ys = cyIn + dyi;
                const keel::f64 zs = czIn + dzi;
                keel::f64 v;
                switch (code) {
                    case 0: v = samp_nearest3 (V, H, W, D, xs, ys, zs, fill); break;
                    case 1: v = samp_trilinear(V, H, W, D, xs, ys, zs, fill); break;
                    default:v = samp_tricubic (V, H, W, D, xs, ys, zs, fill); break;
                }
                writeNative(B, (r - 1) + (c - 1) * outH + (p - 1) * outH * outW, v, T);
            }
        }
    }
    return B;
}

// imresize3 — MATLAB:
//   B = imresize3(V, scale)
//   B = imresize3(V, [r c d])
//   B = imresize3(___, method)
//   B = imresize3(___, Name=Value)         Method / Antialiasing / Scale / OutputSize
void imresize3_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                   CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "imresize3");
    // no max: name-value tail in the reference
    auto *mr = ctx.engine->allocator();
    const Value &V = args[0];

    // Defaults
    keel::String method = keel::String::from("cubic");
    bool antialiasSet  = false;
    bool antialias     = true;        // initial — recomputed once scale known
    bool haveScaleNum  = false;       // scalar Scale OR per-axis Scale
    keel::f64 scaleScalar = 1.0;
    keel::Array<keel::f64> scaleVec3;    // length 3 if 3-vec scale provided
    bool haveSize3     = false;
    keel::usize outRows = 0, outCols = 0, outDeps = 0;

    // Positional: scale/size, then optional method (string).
    keel::usize i = 1;
    if (args.size() >= 2 && !args[1].isEmpty()
        && !args[1].isChar() && !args[1].isString()) {
        const Value &s = args[1];
        if (s.numel() == 1) {
            haveScaleNum = true; scaleScalar = s.toScalar();
        } else if (s.numel() == 3) {
            haveSize3 = true;
            outRows = keel::as<keel::usize>(s.elemAsDouble(0));
            outCols = keel::as<keel::usize>(s.elemAsDouble(1));
            outDeps = keel::as<keel::usize>(s.elemAsDouble(2));
        } else {
            throw Error("imresize3: 2nd arg must be scalar scale or "
                        "[numrows numcols numplanes]",
                        0, 0, "imresize3", "", "numkit:imresize3:size");
        }
        i = 2;
    }
    if (i < args.size() && (args[i].isChar() || args[i].isString())) {
        keel::String s = argString(args[i]);
        // First trailing string is the method positional. Subsequent
        // strings are NV-pair names handled in the loop below.
        // Heuristic: only treat as positional method if it's NOT a
        // known NV-pair name.
        const bool isNV = (s == "Antialiasing" || s == "antialiasing" ||
                           s == "Method"       || s == "method" ||
                           s == "OutputSize"   || s == "outputsize" ||
                           s == "Scale"        || s == "scale");
        if (!isNV) { method = keel::move(s); ++i; }
    }

    // NV pairs.
    while (i + 1 < args.size()) {
        const keel::String name = argString(args[i]);
        const Value &val = args[i + 1];
        if (name == "Antialiasing" || name == "antialiasing") {
            antialiasSet = true;
            antialias = (val.toScalar() != 0.0);
        } else if (name == "Method" || name == "method") {
            method = argString(val);
        } else if (name == "OutputSize" || name == "outputsize") {
            if (val.numel() != 3)
                throw Error("imresize3: OutputSize must be a 3-element vector",
                            0, 0, "imresize3", "", "numkit:imresize3:size");
            haveSize3 = true;
            outRows = keel::as<keel::usize>(val.elemAsDouble(0));
            outCols = keel::as<keel::usize>(val.elemAsDouble(1));
            outDeps = keel::as<keel::usize>(val.elemAsDouble(2));
        } else if (name == "Scale" || name == "scale") {
            if (val.numel() == 1) {
                haveScaleNum = true; scaleScalar = val.toScalar();
            } else if (val.numel() == 3) {
                const keel::f64 sv[3] = {val.elemAsDouble(0),
                                      val.elemAsDouble(1),
                                      val.elemAsDouble(2)};
                scaleVec3.assign(keel::ArrayRef<keel::f64>(sv, 3));
            } else {
                throw Error("imresize3: Scale must be a positive number "
                            "or a 3-element vector",
                            0, 0, "imresize3", "", "numkit:imresize3:scaleNV");
            }
        } else {
            throw Error(keel::format("imresize3: unknown name-value parameter '{}'", name),
                        0, 0, "imresize3", "", "numkit:imresize3:nv");
        }
        i += 2;
    }

    // Resolve output dims.
    const auto &d = V.dims();
    const keel::usize inR = d.dim(0);
    const keel::usize inC = (d.ndims() >= 2) ? d.dim(1) : 1;
    const keel::usize inD = (d.ndims() >= 3) ? d.dim(2) : 1;
    if (!haveSize3) {
        keel::f64 sR, sC, sD;
        if (!scaleVec3.empty()) {
            sR = scaleVec3[0]; sC = scaleVec3[1]; sD = scaleVec3[2];
        } else if (haveScaleNum) {
            sR = sC = sD = scaleScalar;
        } else {
            throw Error("imresize3: must specify scale, size, or "
                        "Scale/OutputSize NV",
                        0, 0, "imresize3", "", "numkit:imresize3:noSize");
        }
        if (!(sR > 0 && sC > 0 && sD > 0))
            throw Error("imresize3: scale factors must be > 0",
                        0, 0, "imresize3", "", "numkit:imresize3:scale");
        outRows = keel::as<keel::usize>(std::round(sR * keel::f64(inR)));
        outCols = keel::as<keel::usize>(std::round(sC * keel::f64(inC)));
        outDeps = keel::as<keel::usize>(std::round(sD * keel::f64(inD)));
    }

    // Per-axis scale that drives the u formula. MATLAB uses the
    // user-supplied scale when given (scalar or 3-vec), and out/in
    // when only a size vector is given — these produce DIFFERENT
    // values even when round(scale * inLen) == outLen.
    keel::f64 scY, scX, scZ;
    if (haveSize3 && scaleVec3.empty() && !haveScaleNum) {
        scY = keel::f64(outRows) / keel::f64(inR ? inR : 1);
        scX = keel::f64(outCols) / keel::f64(inC ? inC : 1);
        scZ = keel::f64(outDeps) / keel::f64(inD ? inD : 1);
    } else if (!scaleVec3.empty()) {
        scY = scaleVec3[0]; scX = scaleVec3[1]; scZ = scaleVec3[2];
    } else {  // scalar scale (positional or 'Scale' NV)
        scY = scX = scZ = scaleScalar;
    }

    // Resolve antialiasing default = true if any output dim < input dim
    if (!antialiasSet) {
        antialias = (outRows < inR) || (outCols < inC) || (outDeps < inD);
    }

    outs[0] = imresize3_core(V, outRows, outCols, outDeps,
                             scY, scX, scZ, method, antialias, mr);
}

// imrotate3 — MATLAB:
//   B = imrotate3(V, angle, W)
//   B = imrotate3(V, angle, W, method)
//   B = imrotate3(V, angle, W, method, bbox)
//   B = imrotate3(___, "FillValues", v)
void imrotate3_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                   keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 7, "imrotate3");
    auto *mr = ctx.engine->allocator();
    const keel::f64 angle = args[1].toScalar();
    if (args[2].numel() != 3)
        throw Error("imrotate3: axis W must be a 3-element vector",
                    0, 0, "imrotate3", "", "numkit:imrotate3:axis");
    const keel::f64 Wx = args[2].elemAsDouble(0);
    const keel::f64 Wy = args[2].elemAsDouble(1);
    const keel::f64 Wz = args[2].elemAsDouble(2);

    keel::String method = keel::String::from("linear");
    keel::String bbox   = keel::String::from("loose");
    keel::f64 fill        = 0.0;
    keel::usize i = 3;
    if (i < args.size() && (args[i].isChar() || args[i].isString())) {
        keel::String s = argString(args[i]);
        if (s != "FillValues" && s != "fillvalues") { method = keel::move(s); ++i; }
    }
    if (i < args.size() && (args[i].isChar() || args[i].isString())) {
        keel::String s = argString(args[i]);
        if (s != "FillValues" && s != "fillvalues") { bbox = keel::move(s); ++i; }
    }
    while (i + 1 < args.size()) {
        const keel::String name = argString(args[i]);
        const Value &val = args[i + 1];
        if (name == "FillValues" || name == "fillvalues") {
            fill = val.toScalar();
        } else {
            throw Error(keel::format("imrotate3: unknown name-value parameter '{}'", name),
                        0, 0, "imrotate3", "", "numkit:imrotate3:nv");
        }
        i += 2;
    }
    outs[0] = imrotate3(args[0], angle, Wx, Wy, Wz, method, bbox, fill, mr);
}

} // namespace detail

} // namespace numkit::image
