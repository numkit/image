// toolboxes/image/src/morph/morph.cpp

#include <keel/arena.h>
#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/image.h>
#include <keel/math.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/static_array.h>
#include <numkit/image/internal/type_gates.hpp>
#include <numkit/image/morph/morph.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/fn_handle.hpp>
#include <numkit/value/value.hpp>

#include <utility>

namespace numkit::image::internal {
// Value-preserving class output for the morphology wrappers: multiply
// the intensity image (gathered to double) by a logical keep-mask and
// cast back to the input class.
inline Value maskImageByLogical(const Value &I, const Value &mask,
                                keel::Allocator *mr)
{
    Value Id = coerceToDoubleImage(I, mr);
    const keel::usize n = Id.numel();
    keel::f64 *d = Id.doubleDataMut();
    const keel::u8 *m = mask.logicalData();
    for (keel::usize i = 0; i < n; ++i)
        if (!m[i]) d[i] = 0.0;
    return castResultToClass(Id, I.type(), mr);
}
} // namespace numkit::image::internal

namespace numkit::image {

namespace {

Value pack_logical_col(keel::ArrayRef<keel::u8> mask_col, keel::usize rows, keel::usize cols,
                       keel::Allocator *mr)
{
    Value out = Value::matrix(rows, cols, ValueType::LOGICAL, mr);
    const keel::usize n = rows * cols;
    if (n == 0) return out;
    keel::u8 *od = out.logicalDataMut();
    for (keel::usize i = 0; i < n; ++i)
        od[i] = mask_col[i] ? 1u : 0u;
    return out;
}

void strel_argcheck(keel::StringRef shape, const Value &arbitrary_nhood)
{
    if (shape == "arbitrary" || shape.empty()) {
        if (arbitrary_nhood.numel() == 0)
            throw Error("strel('arbitrary', NHOOD): NHOOD missing",
                        0, 0, "strel", "", "numkit:strel:nargin");
    } else if (shape != "square" && shape != "rectangle" &&
               shape != "diamond" && shape != "disk" && shape != "line") {
        throw Error(keel::format("strel: unknown shape '{}'", shape), 0, 0, "strel", "",
                    "numkit:strel:badshape");
    }
}

inline void store_classed_morph(Value &out, keel::usize i, keel::f64 v, ValueType t)
{
    switch (t) {
        case ValueType::DOUBLE: out.doubleDataMut()[i] = v; break;
        case ValueType::SINGLE: out.singleDataMut()[i] = keel::as<keel::f32>(v); break;
        case ValueType::UINT8: {
            if (v < 0.0) v = 0.0;
            if (v > 255.0) v = 255.0;
            out.uint8DataMut()[i] = keel::as<keel::u8>(keel::round_i64(v));
            break;
        }
        case ValueType::UINT16: {
            if (v < 0.0) v = 0.0;
            if (v > 65535.0) v = 65535.0;
            out.uint16DataMut()[i] = keel::as<keel::u16>(keel::round_i64(v));
            break;
        }
        case ValueType::INT16: {
            if (v < -32768.0) v = -32768.0;
            if (v > 32767.0) v = 32767.0;
            out.int16DataMut()[i] = keel::as<keel::i16>(keel::round_i64(v));
            break;
        }
        case ValueType::LOGICAL:
            out.logicalDataMut()[i] = v != 0.0 ? 1u : 0u;
            break;
        default:
            throw Error("morph: unsupported class", 0, 0, "morph", "",
                        "numkit:morph:badtype");
    }
}

struct SEColMajor {
    keel::usize rows = 0;
    keel::usize cols = 0;
    keel::Array<keel::u8> mask;
};

SEColMajor unpack_se(const Value &SE, keel::Allocator *alloc)
{
    const Value *nhood = &SE;
    if (SE.isStruct() && SE.numel() >= 1) {
        const auto &fields = SE.structArrayElem(0);
        if (const Value *v = fields.get("Neighborhood")) nhood = v;
    }
    SEColMajor s{};
    s.rows = nhood->dims().rows();
    s.cols = nhood->dims().cols();
    const keel::usize n = s.rows * s.cols;
    s.mask = keel::Array<keel::u8>(n, keel::u8{0}, alloc);
    if (nhood->isLogical()) {
        const keel::u8 *src = nhood->logicalData();
        for (keel::usize i = 0; i < n; ++i)
            s.mask[i] = src[i] ? 1u : 0u;
    } else {
        for (keel::usize i = 0; i < n; ++i)
            s.mask[i] = (nhood->elemAsDouble(i) != 0.0) ? 1u : 0u;
    }
    return s;
}

template <bool IsErode>
Value morph_op(const Value &I, const Value &SE, keel::Allocator *mr)
{
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    SEColMajor se = unpack_se(SE, scratch.allocator());
    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    Value out = Value::matrix(H, W, I.type(), mr);
    if (H == 0 || W == 0) return out;

    if (I.type() == ValueType::LOGICAL || I.type() == ValueType::UINT8) {
        const keel::u8 *src = I.isLogical() ? I.logicalData() : I.uint8Data();
        keel::u8 *dst = out.isLogical() ? out.logicalDataMut() : out.uint8DataMut();
        const keel::usize n = H * W;
        if constexpr (IsErode) {
            (void)keel::image::morph_erode_u8(
                keel::ArrayRef<keel::u8>(src, n), H, W, se.mask.as_ref(), se.rows, se.cols,
                keel::MutArrayRef<keel::u8>(dst, n), scratch.allocator());
        } else {
            (void)keel::image::morph_dilate_u8(
                keel::ArrayRef<keel::u8>(src, n), H, W, se.mask.as_ref(), se.rows, se.cols,
                keel::MutArrayRef<keel::u8>(dst, n), scratch.allocator());
        }
        return out;
    }

    const keel::usize n = H * W;
    keel::Array<keel::f64> src_f(n, scratch.allocator());
    keel::Array<keel::f64> dst_f(n, scratch.allocator());
    for (keel::usize i = 0; i < n; ++i)
        src_f[i] = I.elemAsDouble(i);
    if constexpr (IsErode) {
        (void)keel::image::morph_erode_f64(src_f.as_ref(), H, W, se.mask.as_ref(), se.rows, se.cols,
                                           dst_f.as_mut_ref());
    } else {
        (void)keel::image::morph_dilate_f64(src_f.as_ref(), H, W, se.mask.as_ref(), se.rows, se.cols,
                                            dst_f.as_mut_ref());
    }
    for (keel::usize i = 0; i < n; ++i)
        store_classed_morph(out, i, dst_f[i], I.type());
    return out;
}

void imreconstruct_argcheck(const Value &marker, const Value &mask)
{
    internal::requireImageClass(marker, "imreconstruct", "marker", 1,
                                internal::ImgClasses::AnyNumericReal);
    internal::requireImageClass(mask, "imreconstruct", "mask", 2,
                                internal::ImgClasses::AnyNumericReal);
    if (marker.type() != mask.type())
        throw Error(keel::format("imreconstruct: marker and mask must have the same "
                                 "class (got {} and {}).",
                                 mtypeName(marker.type()), mtypeName(mask.type())),
                    0, 0, "imreconstruct", "",
                    "numkit:imreconstruct:notSameClass");
    if (mask.dims().rows() != marker.dims().rows()
        || mask.dims().cols() != marker.dims().cols())
        throw Error("imreconstruct: marker and mask must have the same shape",
                    0, 0, "imreconstruct", "", "numkit:imreconstruct:shape");
}

void imfill_holes_argcheck(const Value &BW)
{
    internal::requireImageClass(BW, "imfill", "BW", 1,
                                internal::ImgClasses::FloatIntLog);
}

void imregionalmax_argcheck(const Value &I)
{
    internal::requireImageClass(I, "imregionalmax", "I", 1,
                                internal::ImgClasses::AnyNumericReal);
}

void imregionalmin_argcheck(const Value &I)
{
    internal::requireImageClass(I, "imregionalmin", "I", 1,
                                internal::ImgClasses::AnyNumericReal);
}

void imhmax_argcheck(const Value &I, keel::f64 h)
{
    internal::requireImageClass(I, "imhmax", "I", 1,
                                internal::ImgClasses::FloatInt);
    if (!(h >= 0.0))
        throw Error("imhmax: h must be ≥ 0",
                    0, 0, "imhmax", "", "numkit:imhmax:h");
}

void imhmin_argcheck(const Value &I, keel::f64 h)
{
    internal::requireImageClass(I, "imhmin", "I", 1,
                                internal::ImgClasses::FloatInt);
    if (!(h >= 0.0))
        throw Error("imhmin: h must be ≥ 0",
                    0, 0, "imhmin", "", "numkit:imhmin:h");
}

void imimposemin_argcheck(const Value &I, const Value &BW)
{
    internal::requireImageClass(I, "imimposemin", "I", 1,
                                internal::ImgClasses::FloatInt);
    if (BW.dims().rows() != I.dims().rows() || BW.dims().cols() != I.dims().cols())
        throw Error("imimposemin: I and BW must have the same shape",
                    0, 0, "imimposemin", "", "numkit:imimposemin:shape");
}

void imclearborder_argcheck(const Value &BW)
{
    internal::requireImageClass(BW, "imclearborder", "BW", 1,
                                internal::ImgClasses::FloatIntLog);
}

Value tophat_subtract(const Value &lhs, const Value &rhs, keel::Allocator *mr)
{
    const keel::usize H = lhs.dims().rows();
    const keel::usize W = lhs.dims().cols();
    const keel::usize N = lhs.numel();
    Value out = Value::matrix(H, W, lhs.type(), mr);
    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 v = lhs.elemAsDouble(i) - rhs.elemAsDouble(i);
        store_classed_morph(out, i, v, lhs.type());
    }
    return out;
}

void mmgradm_argcheck(const Value &I)
{
    internal::requireImageClass(I, "mmgradm", "I", 1,
                                internal::ImgClasses::FloatIntLog);
}

void bwpack_argcheck(const Value &BW)
{
    internal::requireImageClass(BW, "bwpack", "BW", 1,
                                internal::ImgClasses::AnyNumericReal);
}

void bwunpack_argcheck(const Value &BWP, keel::usize &M)
{
    internal::requireUint32Packed(BWP, "bwunpack");
    constexpr keel::usize CLASS_BITS = 32;
    const keel::usize pH = BWP.dims().rows();
    if (M == keel::as<keel::usize>(-1)) M = pH * CLASS_BITS;
    if (M > pH * CLASS_BITS)
        throw Error("bwunpack: M exceeds packed-row capacity",
                    0, 0, "bwunpack", "", "numkit:bwunpack:M");
}

void applylut_argcheck(const Value &BW, const Value &LUT, keel::usize &lutLen, keel::usize &nq,
                       keel::usize &n)
{
    internal::requireImageClass(BW, "applylut", "BW", 1,
                                internal::ImgClasses::AnyNumericReal);
    lutLen = LUT.numel();
    if (lutLen == 0)
        throw Error("applylut: LUT must be non-empty",
                    0, 0, "applylut", "", "numkit:applylut:lutsize");
    nq = 0;
    while ((keel::usize{1} << nq) < lutLen) ++nq;
    if ((keel::usize{1} << nq) != lutLen)
        throw Error("applylut: LUT length must be 2^(n*n)",
                    0, 0, "applylut", "", "numkit:applylut:lutsize");
    n = keel::as<keel::usize>(keel::round(keel::sqrt(keel::as<keel::f64>(nq))));
    if (n * n != nq)
        throw Error("applylut: LUT length not 2^(n*n) for any integer n",
                    0, 0, "applylut", "", "numkit:applylut:lutshape");
}

void bwlookup_argcheck(const Value &lut)
{
    const keel::usize n = lut.numel();
    if (n != 16 && n != 512)
        throw Error("bwlookup: Expected LUT (argument 2) to have 16 or 512 elements.",
                    0, 0, "bwlookup", "", "numkit:bwlookup:lutsize");
}

keel::image::Bwmorph3Op bwmorph3_argcheck(const Value &V, keel::StringRef op)
{
    keel::image::Bwmorph3Op parsed = keel::image::Bwmorph3Op::kBranchpoints;
    if      (op == "branchpoints") parsed = keel::image::Bwmorph3Op::kBranchpoints;
    else if (op == "clean")        parsed = keel::image::Bwmorph3Op::kClean;
    else if (op == "endpoints")    parsed = keel::image::Bwmorph3Op::kEndpoints;
    else if (op == "fill")         parsed = keel::image::Bwmorph3Op::kFill;
    else if (op == "majority")     parsed = keel::image::Bwmorph3Op::kMajority;
    else if (op == "remove")       parsed = keel::image::Bwmorph3Op::kRemove;
    else throw Error(keel::format("bwmorph3: unknown operation '{}'", op),
                     0, 0, "bwmorph3", "", "numkit:bwmorph3:badOp");

    if (V.dims().ndim() > 3)
        throw Error("bwmorph3: input must be a 2-D or 3-D array",
                    0, 0, "bwmorph3", "", "numkit:bwmorph3:rank");
    return parsed;
}

void bwhitmiss_argcheck(const Value &BW)
{
    internal::requireImageClass(BW, "bwhitmiss", "BW", 1,
                                internal::ImgClasses::AnyNumericReal);
}

void imkeepborder_argcheck(const Value &BW)
{
    internal::requireImageClass(BW, "imkeepborder", "BW", 1,
                                internal::ImgClasses::FloatIntLog);
}

keel::image::BwmorphOp bwmorph_argcheck(const Value &BW, keel::StringRef op)
{
    internal::requireImageClass(BW, "bwmorph", "BW", 1,
                                internal::ImgClasses::AnyNumericReal);
    if (BW.dims().ndim() > 2 || BW.dims().is3D())
        throw numkit::Error("bwmorph: input must be a 2-D binary image",
                            0, 0, "bwmorph", "",
                            "numkit:bwmorph:unsupportedShape");

    if (op == "dilate")         return keel::image::BwmorphOp::kDilate;
    if (op == "erode")          return keel::image::BwmorphOp::kErode;
    if (op == "bridge")         return keel::image::BwmorphOp::kBridge;
    if (op == "clean")          return keel::image::BwmorphOp::kClean;
    if (op == "diag")           return keel::image::BwmorphOp::kDiag;
    if (op == "endpoints")      return keel::image::BwmorphOp::kEndpoints;
    if (op == "fatten")         return keel::image::BwmorphOp::kFatten;
    if (op == "fill")           return keel::image::BwmorphOp::kFill;
    if (op == "hbreak")         return keel::image::BwmorphOp::kHbreak;
    if (op == "majority")       return keel::image::BwmorphOp::kMajority;
    if (op == "perim4")         return keel::image::BwmorphOp::kPerim4;
    if (op == "perim8")         return keel::image::BwmorphOp::kPerim8;
    if (op == "remove")         return keel::image::BwmorphOp::kRemove;
    if (op == "open")           return keel::image::BwmorphOp::kOpen;
    if (op == "close")          return keel::image::BwmorphOp::kClose;
    if (op == "bothat")         return keel::image::BwmorphOp::kBothat;
    if (op == "tophat")         return keel::image::BwmorphOp::kTophat;
    if (op == "thin")           return keel::image::BwmorphOp::kThin;
    if (op == "skeleton" || op == "skel") return keel::image::BwmorphOp::kSkel;
    if (op == "shrink")         return keel::image::BwmorphOp::kShrink;
    if (op == "spur")           return keel::image::BwmorphOp::kSpur;
    if (op == "thicken")        return keel::image::BwmorphOp::kThicken;
    if (op == "branchpoints")   return keel::image::BwmorphOp::kBranchpoints;

    throw numkit::Error(keel::format("bwmorph: unknown operation '{}'", op),
                        0, 0, "bwmorph", "",
                        "numkit:bwmorph:badOp");
}

keel::i32 name_to_dir8(keel::StringRef s)
{
    if (s == "N")  return 0;
    if (s == "NE") return 1;
    if (s == "E")  return 2;
    if (s == "SE") return 3;
    if (s == "S")  return 4;
    if (s == "SW") return 5;
    if (s == "W")  return 6;
    if (s == "NW") return 7;
    return -1;
}

keel::i32 name_to_dir4(keel::StringRef s)
{
    if (s == "N") return 0;
    if (s == "E") return 1;
    if (s == "S") return 2;
    if (s == "W") return 3;
    return -1;
}

void bwtraceboundary_argcheck(const Value &BW, const Value &P,
                              keel::StringRef fstep, keel::i32 conn,
                              keel::StringRef dir,
                              keel::i64 &pr, keel::i64 &pc, keel::i32 &fidx)
{
    internal::requireImageClass(BW, "bwtraceboundary", "BW", 1,
                                internal::ImgClasses::AnyNumericReal);
    internal::requireRealDoubleArg(P, "bwtraceboundary", "P", 2);
    if (BW.dims().is3D())
        throw Error("bwtraceboundary: BW must be 2-D",
                    0, 0, "bwtraceboundary", "",
                    "numkit:bwtraceboundary:dim");
    if (conn != 4 && conn != 8)
        throw Error("bwtraceboundary: CONN must be 4 or 8",
                    0, 0, "bwtraceboundary", "",
                    "numkit:bwtraceboundary:conn");
    if (P.numel() != 2)
        throw Error("bwtraceboundary: P must be a 2-element vector",
                    0, 0, "bwtraceboundary", "",
                    "numkit:bwtraceboundary:p");
    if (dir != "clockwise" && dir != "counterclockwise")
        throw Error("bwtraceboundary: DIR must be 'clockwise' or "
                    "'counterclockwise'",
                    0, 0, "bwtraceboundary", "",
                    "numkit:bwtraceboundary:dir");

    const keel::usize M = BW.dims().rows();
    const keel::usize N = BW.dims().cols();
    pr = keel::as<keel::i64>(P.elemAsDouble(0)) - 1;
    pc = keel::as<keel::i64>(P.elemAsDouble(1)) - 1;
    if (pr < 0 || pc < 0
     || keel::as<keel::usize>(pr) >= M
     || keel::as<keel::usize>(pc) >= N)
        throw Error("bwtraceboundary: P out of bounds",
                    0, 0, "bwtraceboundary", "",
                    "numkit:bwtraceboundary:pbounds");

    const keel::usize k = keel::as<keel::usize>(pc) * M
                        + keel::as<keel::usize>(pr);
    const bool is_fg = BW.isLogical() ? (BW.logicalData()[k] != 0)
                                      : (BW.elemAsDouble(k) != 0.0);
    if (!is_fg)
        throw Error("bwtraceboundary: P must be a foreground pixel",
                    0, 0, "bwtraceboundary", "",
                    "numkit:bwtraceboundary:pfg");

    fidx = (conn == 8) ? name_to_dir8(fstep) : name_to_dir4(fstep);
    if (fidx < 0)
        throw Error("bwtraceboundary: invalid FSTEP for given CONN",
                    0, 0, "bwtraceboundary", "",
                    "numkit:bwtraceboundary:fstep");
}

void makelut_argcheck(keel::i32 n)
{
    if (n != 2 && n != 3)
        throw Error("makelut: N must be 2 or 3.",
                    0, 0, "makelut", "", "numkit:makelut:badN");
}

} // anonymous

Value strel(keel::StringRef shape, keel::ArrayRef<keel::f64> params, const Value &arbitrary_nhood,
            keel::Allocator *mr)
{
    strel_argcheck(shape, arbitrary_nhood);
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::image::StrelMask sm{};
    Value nhood;

    if (shape == "square") {
        const keel::i32 N = params.empty() ? 3 : keel::as<keel::i32>(params[0]);
        (void)keel::image::strel_square(N, sm, scratch.allocator());
        nhood = pack_logical_col(sm.mask_col_major.as_ref(), sm.rows, sm.cols, mr);
    } else if (shape == "rectangle") {
        const keel::i32 r = params.size() >= 1 ? keel::as<keel::i32>(params[0]) : 3;
        const keel::i32 c = params.size() >= 2 ? keel::as<keel::i32>(params[1]) : r;
        (void)keel::image::strel_rect(r, c, sm, scratch.allocator());
        nhood = pack_logical_col(sm.mask_col_major.as_ref(), sm.rows, sm.cols, mr);
    } else if (shape == "diamond") {
        const keel::i32 r = params.empty() ? 1 : keel::as<keel::i32>(params[0]);
        (void)keel::image::strel_diamond(r, sm, scratch.allocator());
        nhood = pack_logical_col(sm.mask_col_major.as_ref(), sm.rows, sm.cols, mr);
    } else if (shape == "disk") {
        const keel::f64 r = params.empty() ? 5.0 : params[0];
        const keel::i32 n = params.size() >= 2 ? keel::as<keel::i32>(params[1]) : 4;
        (void)keel::image::strel_disk(r, n, sm, scratch.allocator());
        nhood = pack_logical_col(sm.mask_col_major.as_ref(), sm.rows, sm.cols, mr);
    } else if (shape == "line") {
        const keel::f64 len = params.size() >= 1 ? params[0] : 3.0;
        const keel::f64 th  = params.size() >= 2 ? params[1] : 0.0;
        (void)keel::image::strel_line(len, th, sm, scratch.allocator());
        nhood = pack_logical_col(sm.mask_col_major.as_ref(), sm.rows, sm.cols, mr);
    } else {
        const auto &d = arbitrary_nhood.dims();
        const keel::usize H = d.rows();
        const keel::usize W = d.cols();
        nhood = Value::matrix(H, W, ValueType::LOGICAL, mr);
        keel::u8 *od = nhood.logicalDataMut();
        for (keel::usize i = 0; i < H * W; ++i)
            od[i] = (arbitrary_nhood.elemAsDouble(i) != 0.0) ? 1u : 0u;
    }

    auto se = Value::structArray(1, 1, mr);
    auto &el = se.structArrayElem(0);
    el.put(keel::String::from("Neighborhood"), std::move(nhood));
    el.put(keel::String::from("Dimensionality"), Value::scalar(2.0, mr));
    return se;
}

Value imerode(const Value &I, const Value &SE, keel::Allocator *mr)
{
    return morph_op<true>(I, SE, mr);
}

Value imdilate(const Value &I, const Value &SE, keel::Allocator *mr)
{
    return morph_op<false>(I, SE, mr);
}

Value imopen(const Value &I, const Value &SE, keel::Allocator *mr)
{
    Value e = imerode(I, SE, mr);
    return imdilate(e, SE, mr);
}

Value imclose(const Value &I, const Value &SE, keel::Allocator *mr)
{
    Value d = imdilate(I, SE, mr);
    return imerode(d, SE, mr);
}

Value imreconstruct(const Value &marker, const Value &mask, keel::i32 conn, keel::Allocator *mr)
{
    imreconstruct_argcheck(marker, mask);
    if (conn != 4) conn = 8;
    const keel::usize H = marker.dims().rows();
    const keel::usize W = marker.dims().cols();
    const keel::usize N = marker.numel();
    const ValueType srcT = marker.type();

    Value out = Value::matrix(H, W, srcT, mr);
    if (N == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> marker_f(N, scratch.allocator());
    keel::Array<keel::f64> mask_f(N, scratch.allocator());
    keel::Array<keel::f64> recon_f(N, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i) {
        marker_f[i] = marker.elemAsDouble(i);
        mask_f[i] = mask.elemAsDouble(i);
    }

    (void)keel::image::imreconstruct_f64(marker_f.as_ref(), mask_f.as_ref(), H, W, conn,
                                         recon_f.as_mut_ref(), scratch.allocator());

    for (keel::usize i = 0; i < N; ++i) {
        keel::f64 v = recon_f[i];
        switch (srcT) {
            case ValueType::DOUBLE: out.doubleDataMut()[i] = v; break;
            case ValueType::SINGLE: out.singleDataMut()[i] = keel::as<keel::f32>(v); break;
            case ValueType::UINT8: {
                if (v < 0.0) v = 0.0;
                if (v > 255.0) v = 255.0;
                out.uint8DataMut()[i] = keel::as<keel::u8>(keel::round_i64(v));
                break;
            }
            case ValueType::UINT16: {
                if (v < 0.0) v = 0.0;
                if (v > 65535.0) v = 65535.0;
                out.uint16DataMut()[i] = keel::as<keel::u16>(keel::round_i64(v));
                break;
            }
            case ValueType::LOGICAL:
                out.logicalDataMut()[i] = (v != 0.0) ? 1u : 0u;
                break;
            default:
                out.doubleDataMut()[i] = v;
                break;
        }
    }
    return out;
}

Value imfill_holes(const Value &BW, keel::i32 conn, keel::Allocator *mr)
{
    imfill_holes_argcheck(BW);
    if (conn != 4) conn = 8;
    const keel::usize H = BW.dims().rows();
    const keel::usize W = BW.dims().cols();
    const keel::usize N = H * W;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    if (BW.type() != ValueType::LOGICAL) {
        if (N == 0) return Value::matrix(H, W, BW.type(), mr);
        keel::Array<keel::f64> img_f(N, scratch.allocator());
        for (keel::usize i = 0; i < N; ++i)
            img_f[i] = BW.elemAsDouble(i);
        Value J = Value::matrix(H, W, ValueType::DOUBLE, mr);
        (void)keel::image::imfill_holes_gray(img_f.as_ref(), H, W, conn,
                                             keel::MutArrayRef<keel::f64>(J.doubleDataMut(), N),
                                             scratch.allocator());
        return internal::castResultToClass(J, BW.type(), mr);
    }

    Value out = Value::matrix(H, W, ValueType::LOGICAL, mr);
    if (N == 0) return out;
    (void)keel::image::imfill_holes_binary(
        keel::ArrayRef<keel::u8>(BW.logicalData(), N), H, W, conn,
        keel::MutArrayRef<keel::u8>(out.logicalDataMut(), N), scratch.allocator());
    return out;
}

Value imregionalmax(const Value &I, keel::i32 conn, keel::Allocator *mr)
{
    imregionalmax_argcheck(I);
    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    const keel::usize N = I.numel();
    Value out = Value::matrix(H, W, ValueType::LOGICAL, mr);
    if (N == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> img_f(N, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i)
        img_f[i] = I.elemAsDouble(i);
    (void)keel::image::imregionalmax_2d(
        img_f.as_ref(), H, W, conn, keel::MutArrayRef<keel::u8>(out.logicalDataMut(), N),
        scratch.allocator());
    return out;
}

Value imregionalmin(const Value &I, keel::i32 conn, keel::Allocator *mr)
{
    imregionalmin_argcheck(I);
    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    const keel::usize N = I.numel();
    Value out = Value::matrix(H, W, ValueType::LOGICAL, mr);
    if (N == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> img_f(N, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i)
        img_f[i] = I.elemAsDouble(i);
    (void)keel::image::imregionalmin_2d(
        img_f.as_ref(), H, W, conn, keel::MutArrayRef<keel::u8>(out.logicalDataMut(), N),
        scratch.allocator());
    return out;
}

Value imhmax(const Value &I, keel::f64 h, keel::i32 conn, keel::Allocator *mr)
{
    imhmax_argcheck(I, h);
    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    const keel::usize N = I.numel();
    if (N == 0) return Value::matrix(H, W, I.type(), mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> img_f(N, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i)
        img_f[i] = I.elemAsDouble(i);
    Value J = Value::matrix(H, W, ValueType::DOUBLE, mr);
    (void)keel::image::imhmax_2d(img_f.as_ref(), H, W, h, conn,
                                 keel::MutArrayRef<keel::f64>(J.doubleDataMut(), N),
                                 scratch.allocator());
    return internal::castResultToClass(J, I.type(), mr);
}

Value imhmin(const Value &I, keel::f64 h, keel::i32 conn, keel::Allocator *mr)
{
    imhmin_argcheck(I, h);
    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    const keel::usize N = I.numel();
    if (N == 0) return Value::matrix(H, W, I.type(), mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> img_f(N, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i)
        img_f[i] = I.elemAsDouble(i);
    Value J = Value::matrix(H, W, ValueType::DOUBLE, mr);
    (void)keel::image::imhmin_2d(img_f.as_ref(), H, W, h, conn,
                                 keel::MutArrayRef<keel::f64>(J.doubleDataMut(), N),
                                 scratch.allocator());
    return internal::castResultToClass(J, I.type(), mr);
}

Value imextendedmax(const Value &I, keel::f64 h, keel::i32 conn, keel::Allocator *mr)
{
    return imregionalmax(imhmax(I, h, conn, mr), conn, mr);
}

Value imextendedmin(const Value &I, keel::f64 h, keel::i32 conn, keel::Allocator *mr)
{
    return imregionalmin(imhmin(I, h, conn, mr), conn, mr);
}

Value imimposemin(const Value &I, const Value &BW, keel::i32 conn, keel::Allocator *mr)
{
    imimposemin_argcheck(I, BW);
    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    const keel::usize N = I.numel();
    if (N == 0) return Value::matrix(H, W, I.type(), mr);

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::f64> img_f(N, scratch.allocator());
    keel::Array<keel::u8> bw_u8(N, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i) {
        img_f[i] = I.elemAsDouble(i);
        bw_u8[i] = (BW.elemAsDouble(i) != 0.0) ? 1u : 0u;
    }
    const bool is_float = (I.type() == ValueType::DOUBLE || I.type() == ValueType::SINGLE);
    Value J = Value::matrix(H, W, ValueType::DOUBLE, mr);
    (void)keel::image::imimposemin_2d(img_f.as_ref(), bw_u8.as_ref(), H, W, is_float, conn,
                                      keel::MutArrayRef<keel::f64>(J.doubleDataMut(), N),
                                      scratch.allocator());
    return internal::castResultToClass(J, I.type(), mr);
}

Value imclearborder(const Value &BW, keel::i32 conn, keel::Allocator *mr)
{
    imclearborder_argcheck(BW);
    if (BW.type() != ValueType::LOGICAL)
        return internal::maskImageByLogical(
            BW,
            imclearborder(internal::coerceToLogicalImage(BW, mr), conn, mr),
            mr);
    if (conn != 4) conn = 8;
    const keel::usize H = BW.dims().rows();
    const keel::usize W = BW.dims().cols();
    const keel::usize N = H * W;
    Value out = Value::matrix(H, W, ValueType::LOGICAL, mr);
    if (N == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    (void)keel::image::imclearborder_binary(
        keel::ArrayRef<keel::u8>(BW.logicalData(), N), H, W, conn,
        keel::MutArrayRef<keel::u8>(out.logicalDataMut(), N), scratch.allocator());
    return out;
}

Value imtophat(const Value &I, const Value &SE, keel::Allocator *mr)
{
    Value opened = imopen(I, SE, mr);
    return tophat_subtract(I, opened, mr);
}

Value imbothat(const Value &I, const Value &SE, keel::Allocator *mr)
{
    Value closed = imclose(I, SE, mr);
    return tophat_subtract(closed, I, mr);
}

Value mmgradm(const Value &I, const Value &se_dil, const Value &se_ero, keel::Allocator *mr)
{
    mmgradm_argcheck(I);
    const bool dilEmpty = se_dil.numel() == 0;
    const bool eroEmpty = se_ero.numel() == 0;
    Value dilated = dilEmpty ? Value{} : imdilate(I, se_dil, mr);
    Value eroded  = eroEmpty ? Value{} : imerode(I, se_ero, mr);

    const keel::usize H = I.dims().rows();
    const keel::usize W = I.dims().cols();
    const keel::usize P = I.dims().is3D() ? I.dims().pages() : 1;
    const keel::usize N = H * W * P;

    if (I.type() == ValueType::LOGICAL) {
        Value out = I.dims().is3D()
            ? Value::matrix3d(H, W, P, ValueType::LOGICAL, mr)
            : Value::matrix(H, W, ValueType::LOGICAL, mr);
        keel::u8 *od = out.logicalDataMut();
        for (keel::usize i = 0; i < N; ++i) {
            const bool d = !dilEmpty && (dilated.elemAsDouble(i) != 0.0);
            const bool e = !eroEmpty && (eroded.elemAsDouble(i) != 0.0);
            bool v = false;
            if (dilEmpty)      v = !e;
            else if (eroEmpty) v = d;
            else               v = d && !e;
            od[i] = v ? 1u : 0u;
        }
        return out;
    }

    Value out = I.dims().is3D()
        ? Value::matrix3d(H, W, P, I.type(), mr)
        : Value::matrix(H, W, I.type(), mr);

    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 d = dilEmpty ? 0.0 : dilated.elemAsDouble(i);
        const keel::f64 e = eroEmpty ? 0.0 : eroded.elemAsDouble(i);
        const keel::f64 v = dilEmpty ? -e : (eroEmpty ? d : (d - e));
        store_classed_morph(out, i, v, I.type());
    }
    return out;
}

Value bwpack(const Value &BW, keel::Allocator *mr)
{
    bwpack_argcheck(BW);
    constexpr keel::usize CLASS_BITS = 32;
    const keel::usize H = BW.dims().rows();
    const keel::usize W = BW.dims().cols();
    const keel::usize outH = (H + CLASS_BITS - 1) / CLASS_BITS;
    Value out = Value::matrix(outH, W, ValueType::UINT32, mr);
    if (outH == 0 || W == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    const keel::usize N = H * W;
    keel::Array<keel::u8> bw_u8(N, scratch.allocator());
    if (BW.isLogical()) {
        const keel::u8 *src = BW.logicalData();
        for (keel::usize i = 0; i < N; ++i)
            bw_u8[i] = src[i] ? 1u : 0u;
    } else {
        for (keel::usize i = 0; i < N; ++i)
            bw_u8[i] = (BW.elemAsDouble(i) != 0.0) ? 1u : 0u;
    }
    (void)keel::image::bwpack_2d(
        bw_u8.as_ref(), H, W, keel::MutArrayRef<keel::u32>(out.uint32DataMut(), outH * W));
    return out;
}

Value bwunpack(const Value &BWP, keel::usize M, keel::Allocator *mr)
{
    bwunpack_argcheck(BWP, M);
    const keel::usize pH = BWP.dims().rows();
    const keel::usize W  = BWP.dims().cols();
    Value out = Value::matrix(M, W, ValueType::LOGICAL, mr);
    if (M == 0 || W == 0) return out;

    (void)keel::image::bwunpack_2d(
        keel::ArrayRef<keel::u32>(BWP.uint32Data(), pH * W), pH, W, M,
        keel::MutArrayRef<keel::u8>(out.logicalDataMut(), M * W));
    return out;
}

Value applylut(const Value &BW, const Value &LUT, keel::Allocator *mr)
{
    keel::usize lutLen = 0, nq = 0, n = 0;
    applylut_argcheck(BW, LUT, lutLen, nq, n);

    const keel::usize H = BW.dims().rows();
    const keel::usize W = BW.dims().cols();
    const keel::usize N = H * W;
    Value out = Value::matrix(H, W, LUT.type(), mr);
    if (N == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::u8> bw_u8(N, scratch.allocator());
    for (keel::usize i = 0; i < N; ++i)
        bw_u8[i] = (BW.elemAsDouble(i) != 0.0) ? 1u : 0u;

    keel::Array<keel::f64> lut_f(lutLen, scratch.allocator());
    for (keel::usize i = 0; i < lutLen; ++i)
        lut_f[i] = LUT.elemAsDouble(i);

    keel::Array<keel::f64> out_f(N, scratch.allocator());
    (void)keel::image::bwlookup_nxn(bw_u8.as_ref(), H, W, lut_f.as_ref(), n, out_f.as_mut_ref());

    for (keel::usize i = 0; i < N; ++i) {
        const keel::f64 v = out_f[i];
        switch (LUT.type()) {
            case ValueType::DOUBLE: out.doubleDataMut()[i] = v; break;
            case ValueType::SINGLE: out.singleDataMut()[i] = keel::as<keel::f32>(v); break;
            case ValueType::UINT8:  out.uint8DataMut()[i] = keel::as<keel::u8>(keel::round_i64(v)); break;
            case ValueType::UINT16: out.uint16DataMut()[i] = keel::as<keel::u16>(keel::round_i64(v)); break;
            case ValueType::INT16:  out.int16DataMut()[i] = keel::as<keel::i16>(keel::round_i64(v)); break;
            case ValueType::LOGICAL: out.logicalDataMut()[i] = (v != 0.0) ? 1u : 0u; break;
            default: out.doubleDataMut()[i] = v; break;
        }
    }
    return out;
}

Value bwlookup(const Value &BW, const Value &lut, keel::Allocator *mr)
{
    bwlookup_argcheck(lut);
    return applylut(BW, lut, mr);
}

Value bwmorph3(const Value &V, keel::StringRef op, keel::Allocator *mr)
{
    const keel::image::Bwmorph3Op k_op = bwmorph3_argcheck(V, op);
    const auto &d = V.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    const keel::usize P = d.is3D() ? d.pages() : 1;
    const keel::usize N = H * W * P;

    Value out = (P > 1) ? Value::matrix3d(H, W, P, ValueType::LOGICAL, mr)
                        : Value::matrix(H, W, ValueType::LOGICAL, mr);
    if (N == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::u8> vol_u8(N, scratch.allocator());
    if (V.isLogical()) {
        const keel::u8 *src = V.logicalData();
        for (keel::usize i = 0; i < N; ++i)
            vol_u8[i] = src[i] ? 1u : 0u;
    } else {
        for (keel::usize i = 0; i < N; ++i)
            vol_u8[i] = (V.elemAsDouble(i) != 0.0) ? 1u : 0u;
    }
    (void)keel::image::bwmorph3_3d(vol_u8.as_ref(), H, W, P, k_op,
                                   keel::MutArrayRef<keel::u8>(out.logicalDataMut(), N));
    return out;
}

Value bwhitmiss(const Value &BW, const Value &se1, const Value &se2, keel::Allocator *mr)
{
    bwhitmiss_argcheck(BW);
    const keel::usize H = BW.dims().rows();
    const keel::usize W = BW.dims().cols();
    Value out = Value::matrix(H, W, ValueType::LOGICAL, mr);
    if (H == 0 || W == 0) return out;

    Value e1 = imerode(BW, se1, mr);
    Value notBW = Value::matrix(H, W, ValueType::LOGICAL, mr);
    {
        keel::u8 *nd = notBW.logicalDataMut();
        for (keel::usize i = 0; i < H * W; ++i)
            nd[i] = (BW.elemAsDouble(i) != 0.0) ? 0u : 1u;
    }
    Value e2 = imerode(notBW, se2, mr);

    keel::u8 *od = out.logicalDataMut();
    for (keel::usize i = 0; i < H * W; ++i) {
        const bool a = (e1.elemAsDouble(i) != 0.0);
        const bool b = (e2.elemAsDouble(i) != 0.0);
        od[i] = (a && b) ? 1u : 0u;
    }
    return out;
}

Value imkeepborder(const Value &BW, keel::i32 conn, keel::Allocator *mr)
{
    imkeepborder_argcheck(BW);
    if (BW.type() != ValueType::LOGICAL)
        return internal::maskImageByLogical(
            BW,
            imkeepborder(internal::coerceToLogicalImage(BW, mr), conn, mr),
            mr);
    if (conn != 4) conn = 8;
    const keel::usize H = BW.dims().rows();
    const keel::usize W = BW.dims().cols();
    const keel::usize N = H * W;
    Value out = Value::matrix(H, W, ValueType::LOGICAL, mr);
    if (N == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    (void)keel::image::imkeepborder_binary(
        keel::ArrayRef<keel::u8>(BW.logicalData(), N), H, W, conn,
        keel::MutArrayRef<keel::u8>(out.logicalDataMut(), N), scratch.allocator());
    return out;
}

Value bwmorph(const Value &BW, keel::StringRef op, keel::i32 n, keel::Allocator *mr)
{
    const keel::image::BwmorphOp k_op = bwmorph_argcheck(BW, op);
    const keel::usize R = BW.dims().rows();
    const keel::usize C = BW.dims().cols();
    const keel::usize N = R * C;
    Value out = Value::matrix(R, C, ValueType::LOGICAL, mr);
    if (N == 0) return out;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::u8> bw_u8(N, scratch.allocator());
    if (BW.isLogical()) {
        const keel::u8 *src = BW.logicalData();
        for (keel::usize i = 0; i < N; ++i)
            bw_u8[i] = src[i] ? 1u : 0u;
    } else {
        for (keel::usize i = 0; i < N; ++i)
            bw_u8[i] = (BW.elemAsDouble(i) != 0.0) ? 1u : 0u;
    }

    (void)keel::image::bwmorph_2d(
        bw_u8.as_ref(), R, C, k_op, n, keel::MutArrayRef<keel::u8>(out.logicalDataMut(), N),
        scratch.allocator());
    return out;
}

Value bwtraceboundary(const Value &BW, const Value &P,
                      keel::StringRef fstep, keel::i32 conn,
                      keel::usize m, keel::StringRef dir,
                      keel::Allocator *mr)
{
    keel::i64 pr = 0, pc = 0;
    keel::i32 fidx = 0;
    bwtraceboundary_argcheck(BW, P, fstep, conn, dir, pr, pc, fidx);

    const keel::usize M = BW.dims().rows();
    const keel::usize N = BW.dims().cols();
    const keel::usize total = M * N;

    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::u8> bw_u8(total, scratch.allocator());
    if (BW.isLogical()) {
        const keel::u8 *src = BW.logicalData();
        for (keel::usize i = 0; i < total; ++i)
            bw_u8[i] = src[i] ? 1u : 0u;
    } else {
        for (keel::usize i = 0; i < total; ++i)
            bw_u8[i] = (BW.elemAsDouble(i) != 0.0) ? 1u : 0u;
    }

    keel::Array<keel::image::BoundaryPoint> pts(scratch.allocator());
    (void)keel::image::bwtraceboundary_2d(bw_u8.as_ref(), M, N, pr, pc, fidx, conn, m,
                                          dir == "clockwise", pts);

    const keel::usize Q = pts.size();
    Value out = Value::matrix(Q, 2, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < Q; ++i) {
        od[i]     = keel::as<keel::f64>(pts[i].row + 1);
        od[Q + i] = keel::as<keel::f64>(pts[i].col + 1);
    }
    return out;
}

Value makelut(FnHandle fun, keel::i32 n, keel::Allocator *mr)
{
    makelut_argcheck(n);
    const keel::i32 nq = n * n;
    const keel::usize N = keel::usize{1} << nq;
    Value lut = Value::matrix(N, 1, ValueType::DOUBLE, mr);
    keel::f64 *ld = lut.doubleDataMut();

    Value nh = Value::matrix(keel::as<keel::usize>(n), keel::as<keel::usize>(n),
                             ValueType::LOGICAL, mr);
    keel::u8 *nd = nh.logicalDataMut();
    for (keel::usize k = 0; k < N; ++k) {
        for (keel::i32 i = 0; i < nq; ++i)
            nd[i] = keel::as<keel::u8>((k >> (nq - 1 - i)) & keel::usize{1});
        keel::StaticArray<Value, 1> o;
        fun(keel::ArrayRef<Value>(&nh, 1), keel::MutArrayRef<Value>(o.data(), 1), mr);
        const Value &r = o[0];
        if (r.numel() != 1)
            throw Error("makelut: fun must return a scalar",
                        0, 0, "makelut", "", "numkit:makelut:funScalar");
        ld[k] = r.toScalar();
    }
    return lut;
}

} // namespace numkit::image
