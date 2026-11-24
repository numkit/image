// toolboxes/image/src/region/region.cpp
//
// Thin Value/argcheck wrappers delegating connected components, binary
// perimeter/area/Euler/distance transform, Moore-Neighbor boundaries,
// regionprops, and Freeman chain codes to keel::image.

#include <keel/array.h>
#include <keel/algo.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/image.h>
#include <keel/memory.h>
#include <keel/math.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/tuple.h>

#include <numkit/image/region/region.hpp>
#include <numkit/image/internal/type_gates.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

namespace numkit::image {

namespace {

keel::Array<keel::u8> read_bw_col_major(const Value &BW, keel::Allocator *mr)
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

void bwconncomp_argcheck(const Value &BW)
{
    internal::requireImageClass(BW, "bwconncomp", "BW", 1, internal::ImgClasses::AnyNumericReal);
}

void bwperim_argcheck(const Value &BW)
{
    internal::requireImageClassAllowComplex(BW, "bwperim", "BW", 1,
                                            internal::ImgClasses::AnyNumericReal);
}

void bwareaopen_argcheck(const Value &BW)
{
    internal::requireImageClass(BW, "bwareaopen", "BW", 1, internal::ImgClasses::AnyNumericReal);
}

void bwboundaries_argcheck(const Value &BW)
{
    internal::requireImageClass(BW, "bwboundaries", "BW", 1, internal::ImgClasses::AnyNumericReal);
}

struct CCInfo {
    keel::usize H = 0, W = 0;
    keel::usize K = 0;
    const Value *pixList = nullptr;
};

CCInfo parse_cc(const Value &CC, const char *fn)
{
    CCInfo info;
    if (!CC.isStruct() || CC.numel() != 1)
        throw Error(keel::format("{}: CC must be a 1×1 struct from bwconncomp", fn),
                    0, 0, fn, "", keel::format("numkit:{}:notStruct", fn));
    const auto &el = CC.structArrayElem(0);
    auto need = [&](const char *name) -> const Value & {
        const Value *v = el.get(name);
        if (!v)
            throw Error(keel::format("{}: CC missing field '{}'", fn, name), 0, 0, fn, "",
                        keel::format("numkit:{}:noField", fn));
        return *v;
    };
    const Value &sz = need("ImageSize");
    if (sz.numel() < 2)
        throw Error(keel::format("{}: CC.ImageSize must have >= 2 dims", fn),
                    0, 0, fn, "", keel::format("numkit:{}:sizeDim", fn));
    info.H = keel::as<keel::usize>(sz.elemAsDouble(0));
    info.W = keel::as<keel::usize>(sz.elemAsDouble(1));
    info.K = keel::as<keel::usize>(need("NumObjects").toScalar());
    info.pixList = &need("PixelIdxList");
    return info;
}

CCInfo labelmatrix_argcheck(const Value &CC) { return parse_cc(CC, "labelmatrix"); }

CCInfo cc2bw_argcheck(const Value &CC, const Value &objects_to_keep, keel::Array<char> &keep)
{
    CCInfo info = parse_cc(CC, "cc2bw");
    const keel::usize K = info.K;
    keep.assign(K, 0);
    if (objects_to_keep.isEmpty()) {
        keel::fill(keep.begin(), keep.end(), keel::as<char>(1));
    } else {
        const keel::usize N = objects_to_keep.numel();
        if (objects_to_keep.type() == ValueType::LOGICAL) {
            if (N != K)
                throw Error("cc2bw: ObjectsToKeep logical vector length must equal NumObjects",
                            0, 0, "cc2bw", "", "numkit:cc2bw:logicalLen");
            for (keel::usize i = 0; i < K; ++i)
                keep[i] = objects_to_keep.elemAsDouble(i) != 0.0 ? 1 : 0;
        } else {
            for (keel::usize i = 0; i < N; ++i) {
                const keel::f64 v = objects_to_keep.elemAsDouble(i);
                if (!keel::is_finite(v) || v <= 0.0 || keel::floor(v) != v)
                    throw Error("cc2bw: ObjectsToKeep must be positive integers or a logical vector",
                                0, 0, "cc2bw", "", "numkit:cc2bw:badIdx");
                const keel::usize idx = keel::as<keel::usize>(v);
                if (idx > K)
                    throw Error("cc2bw: ObjectsToKeep index exceeds NumObjects",
                                0, 0, "cc2bw", "", "numkit:cc2bw:idxRange");
                keep[idx - 1] = 1;
            }
        }
    }
    return info;
}

void regionprops_argcheck(const Value &BW_or_L, keel::ArrayRef<keel::String> propsIn)
{
    internal::requireImageClass(BW_or_L, "regionprops", "BW", 1,
                                internal::ImgClasses::AnyNumericReal);
    static const char *kKnown[] = {
        "all", "basic",
        "area", "centroid", "boundingbox", "perimeter",
        "majoraxislength", "minoraxislength", "eccentricity", "orientation",
        "equivdiameter", "extent", "pixelidxlist", "pixellist",
        "meanintensity", "maxintensity", "minintensity",
        "weightedcentroid", "pixelvalues",
    };
    for (const auto &q : propsIn) {
        keel::String lc = q.clone();
        for (char &ch : lc)
            if (ch >= 'A' && ch <= 'Z') ch = char(ch + 32);
        bool ok = false;
        for (const char *k : kKnown)
            if (lc == k) { ok = true; break; }
        if (!ok)
            throw Error(keel::format("regionprops: property '{}' is not supported in this revision", q),
                        0, 0, "regionprops", "", "numkit:regionprops:badProperty");
    }
}

void bwdist_argcheck(const Value &BW)
{
    internal::requireImageClass(BW, "bwdist", "BW", 1, internal::ImgClasses::AnyNumericReal);
}

void fchcode_argcheck(const Value &bound)
{
    internal::requireImageClass(bound, "fchcode", "bound", 1,
                                internal::ImgClasses::AnyNumericReal);
    if (bound.dims().cols() != 2)
        throw Error("fchcode: bound must be K-by-2", 0, 0, "fchcode", "", "numkit:fchcode:size");
}

void bwareafilt_argcheck(const Value &BW)
{
    internal::requireImageClass(BW, "bwareafilt", "BW", 1, internal::ImgClasses::LogicalOnly);
}

void bweuler_argcheck(const Value &BW, keel::i32 conn)
{
    if (conn != 4 && conn != 8)
        throw Error("bweuler: connectivity must be 4 or 8",
                    0, 0, "bweuler", "", "numkit:bweuler:conn");
    if (BW.dims().is3D())
        throw Error("bweuler: BW must have 2 dimensions",
                    0, 0, "bweuler", "", "numkit:bweuler:dims");
}

} // anonymous

keel::Tuple<Value, Value>
bwlabel(const Value &BW, keel::i32 conn, keel::Allocator *mr)
{
    if (conn != 4) conn = 8;
    const keel::usize H = BW.dims().rows();
    const keel::usize W = BW.dims().cols();
    const keel::usize N = H * W;
    auto fg = read_bw_col_major(BW, mr);
    keel::Array<keel::i32> L(N, 0, mr);
    keel::i32 K = 0;
    (void)keel::image::bwlabel_2d(fg.as_ref(), H, W, conn, L.as_mut_ref(), K, mr);

    Value out = Value::matrix(H, W, ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) od[i] = keel::as<keel::f64>(L[i]);
    return keel::make_tuple(keel::move(out), Value::scalar(keel::as<keel::f64>(K), mr));
}

keel::Tuple<Value, Value>
bwlabeln(const Value &BW, keel::i32 conn, keel::Allocator *mr)
{
    const keel::usize H = BW.dims().dim(0);
    const keel::usize W = BW.dims().dim(1);
    const keel::usize D = BW.numel() / keel::max<keel::usize>(1, H * W);
    const keel::usize N = BW.numel();
    auto fg = read_bw_col_major(BW, mr);
    keel::Array<keel::i32> L(N, 0, mr);
    keel::i32 K = 0;
    (void)keel::image::bwlabel_3d(fg.as_ref(), H, W, D, conn, L.as_mut_ref(), K, mr);

    Value out = (BW.dims().pages() > 1)
        ? Value::matrix3d(BW.dims().rows(), BW.dims().cols(), BW.dims().pages(),
                          ValueType::DOUBLE, mr)
        : Value::matrix(BW.dims().rows(), BW.dims().cols(), ValueType::DOUBLE, mr);
    keel::f64 *od = out.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) od[i] = keel::as<keel::f64>(L[i]);
    return keel::make_tuple(keel::move(out), Value::scalar(keel::as<keel::f64>(K), mr));
}

Value bwconncomp(const Value &BW, keel::i32 conn, keel::Allocator *mr)
{
    bwconncomp_argcheck(BW);
    if (conn != 4) conn = 8;
    const keel::usize H = BW.dims().rows();
    const keel::usize W = BW.dims().cols();
    const keel::usize N = H * W;
    auto fg = read_bw_col_major(BW, mr);
    keel::Array<keel::i32> L(N, 0, mr);
    keel::i32 K = 0;
    (void)keel::image::bwlabel_2d(fg.as_ref(), H, W, conn, L.as_mut_ref(), K, mr);

    auto cc = Value::structArray(1, 1, mr);
    auto &el = cc.structArrayElem(0);
    el.put(keel::String::from("Connectivity"), Value::scalar(keel::as<keel::f64>(conn), mr));

    Value sz = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    sz.doubleDataMut()[0] = keel::as<keel::f64>(H);
    sz.doubleDataMut()[1] = keel::as<keel::f64>(W);
    el.put(keel::String::from("ImageSize"), keel::move(sz));
    el.put(keel::String::from("NumObjects"), Value::scalar(keel::as<keel::f64>(K), mr));

    Value pixList = Value::cell(1, keel::as<keel::usize>(K), mr);
    if (K > 0) {
        keel::Array<keel::Array<keel::i32>> lists(keel::as<keel::usize>(K));
        for (keel::usize i = 0; i < N; ++i) {
            const keel::i32 lab = L[i];
            if (lab > 0) lists[keel::as<keel::usize>(lab - 1)].push_back(keel::as<keel::i32>(i + 1));
        }
        for (keel::usize k = 0; k < keel::as<keel::usize>(K); ++k) {
            const auto &v = lists[k];
            Value col = Value::matrix(v.size(), 1, ValueType::DOUBLE, mr);
            keel::f64 *cd = col.doubleDataMut();
            for (keel::usize i = 0; i < v.size(); ++i) cd[i] = keel::as<keel::f64>(v[i]);
            pixList.cellAt(k) = keel::move(col);
        }
    }
    el.put(keel::String::from("PixelIdxList"), keel::move(pixList));
    return cc;
}

Value labelmatrix(const Value &CC, keel::Allocator *mr)
{
    const CCInfo info = labelmatrix_argcheck(CC);
    const keel::usize H = info.H, W = info.W, K = info.K;
    ValueType ot;
    if      (K <= 255)           ot = ValueType::UINT8;
    else if (K <= 65535)         ot = ValueType::UINT16;
    else if (K <= 0xFFFFFFFFULL) ot = ValueType::UINT32;
    else                         ot = ValueType::DOUBLE;
    Value L = Value::matrix(H, W, ot, mr);
    auto write_label = [&](keel::usize idx0, keel::usize lab) {
        if (idx0 >= H * W) return;
        switch (ot) {
            case ValueType::UINT8:  L.uint8DataMut()[idx0]  = keel::as<keel::u8>(lab);  break;
            case ValueType::UINT16: L.uint16DataMut()[idx0] = keel::as<keel::u16>(lab); break;
            case ValueType::UINT32: L.uint32DataMut()[idx0] = keel::as<keel::u32>(lab); break;
            default:                L.doubleDataMut()[idx0] = keel::as<keel::f64>(lab); break;
        }
    };
    for (keel::usize k = 0; k < K; ++k) {
        const Value &cell = info.pixList->cellAt(k);
        for (keel::usize i = 0; i < cell.numel(); ++i) {
            const keel::usize idx1 = keel::as<keel::usize>(cell.elemAsDouble(i));
            if (idx1 > 0) write_label(idx1 - 1, k + 1);
        }
    }
    return L;
}

Value cc2bw(const Value &CC, const Value &objects_to_keep, keel::Allocator *mr)
{
    keel::Array<char> keep;
    const CCInfo info = cc2bw_argcheck(CC, objects_to_keep, keep);
    const keel::usize H = info.H, W = info.W, K = info.K;
    Value BW = Value::matrix(H, W, ValueType::LOGICAL, mr);
    keel::u8 *bd = BW.logicalDataMut();
    keel::fill(bd, bd + H * W, keel::u8{0});
    for (keel::usize k = 0; k < K; ++k) {
        if (!keep[k]) continue;
        const Value &cell = info.pixList->cellAt(k);
        for (keel::usize i = 0; i < cell.numel(); ++i) {
            const keel::usize idx1 = keel::as<keel::usize>(cell.elemAsDouble(i));
            if (idx1 > 0 && idx1 <= H * W) bd[idx1 - 1] = 1;
        }
    }
    return BW;
}

Value bwarea(const Value &BW, keel::Allocator *mr)
{
    const keel::usize N = BW.numel();
    keel::f64 area = 0.0;
    for (keel::usize i = 0; i < N; ++i)
        if (BW.elemAsDouble(i) != 0.0) area += 1.0;
    return Value::scalar(area, mr);
}

Value bwperim(const Value &BW, keel::i32 conn, keel::Allocator *mr)
{
    bwperim_argcheck(BW);
    const keel::usize H = BW.dims().rows();
    const keel::usize W = BW.dims().cols();
    Value out = Value::matrix(H, W, ValueType::LOGICAL, mr);
    if (H == 0 || W == 0) return out;
    auto fg = read_bw_col_major(BW, mr);
    (void)keel::image::bwperim_2d(fg.as_ref(), H, W, conn,
                                  keel::MutArrayRef<keel::u8>(out.logicalDataMut(), H * W));
    return out;
}

Value bwareaopen(const Value &BW, keel::i32 P, keel::i32 conn, keel::Allocator *mr)
{
    bwareaopen_argcheck(BW);
    const keel::usize H = BW.dims().rows();
    const keel::usize W = BW.dims().cols();
    Value out = Value::matrix(H, W, ValueType::LOGICAL, mr);
    if (H == 0 || W == 0) return out;
    auto fg = read_bw_col_major(BW, mr);
    (void)keel::image::bwareaopen_2d(fg.as_ref(), H, W, P, conn,
                                     keel::MutArrayRef<keel::u8>(out.logicalDataMut(), H * W), mr);
    return out;
}

Value bwboundaries(const Value &BW, keel::i32 conn, Value *Lout, keel::i32 *Nout,
                   keel::Allocator *mr)
{
    bwboundaries_argcheck(BW);
    const keel::usize H = BW.dims().rows();
    const keel::usize W = BW.dims().cols();
    const keel::usize N = H * W;
    auto fg = read_bw_col_major(BW, mr);
    keel::Array<keel::i32> L(N, 0, mr);
    keel::i32 K = 0;
    keel::Array<keel::image::BoundaryPoint> pts(mr);
    keel::Array<keel::usize> offs(mr);
    (void)keel::image::bwboundaries_2d(fg.as_ref(), H, W, conn, L.as_mut_ref(), K, pts, offs, mr);

    if (Nout) *Nout = K;
    if (Lout) {
        Value lab = Value::matrix(H, W, ValueType::DOUBLE, mr);
        keel::f64 *ld = lab.doubleDataMut();
        for (keel::usize i = 0; i < N; ++i) ld[i] = keel::as<keel::f64>(L[i]);
        *Lout = keel::move(lab);
    }

    Value cellCol = Value::cell(keel::as<keel::usize>(K), 1, mr);
    for (keel::usize k = 0; k < keel::as<keel::usize>(K); ++k) {
        const keel::usize beg = offs[k];
        const keel::usize P = offs[k + 1] - beg;
        Value m = Value::matrix(P, 2, ValueType::DOUBLE, mr);
        keel::f64 *mb = m.doubleDataMut();
        for (keel::usize i = 0; i < P; ++i) {
            mb[i]     = keel::as<keel::f64>(pts[beg + i].row + 1);
            mb[P + i] = keel::as<keel::f64>(pts[beg + i].col + 1);
        }
        cellCol.cellAt(k) = keel::move(m);
    }
    return cellCol;
}

Value regionprops(const Value &BW_or_L, keel::ArrayRef<keel::String> propsIn,
                  const Value &intensity, keel::Allocator *mr)
{
    regionprops_argcheck(BW_or_L, propsIn);
    const bool haveI = !intensity.isEmpty() && intensity.numel() > 0;
    const keel::usize H = BW_or_L.dims().rows();
    const keel::usize W = BW_or_L.dims().cols();
    const keel::usize N = H * W;

    keel::Array<keel::i32> L(N, 0, mr);
    keel::i32 K = 0;
    bool looksLabelled = false;
    for (keel::usize i = 0; i < N; ++i) {
        if (BW_or_L.elemAsDouble(i) > 1.5) { looksLabelled = true; break; }
    }
    if (looksLabelled) {
        for (keel::usize i = 0; i < N; ++i) {
            const keel::i32 v = keel::as<keel::i32>(keel::round(BW_or_L.elemAsDouble(i)));
            L[i] = v;
            if (v > K) K = v;
        }
    } else {
        auto fg = read_bw_col_major(BW_or_L, mr);
        (void)keel::image::bwlabel_2d(fg.as_ref(), H, W, 8, L.as_mut_ref(), K, mr);
    }

    auto contains = [&](const char *p) {
        for (const auto &q : propsIn) {
            if (q.equals_ignore_ascii_case(p)) return true;
        }
        return false;
    };

    const bool basic   = propsIn.empty() || contains("basic");
    const bool wantAll = contains("all");
    const bool wArea   = basic || wantAll || contains("Area");
    const bool wCent   = basic || wantAll || contains("Centroid");
    const bool wBbox   = basic || wantAll || contains("BoundingBox");
    const bool wMajor  = wantAll || contains("MajorAxisLength");
    const bool wMinor  = wantAll || contains("MinorAxisLength");
    const bool wEcc    = wantAll || contains("Eccentricity");
    const bool wOrient = wantAll || contains("Orientation");
    const bool wEquivD = wantAll || contains("EquivDiameter");
    const bool wExtent = wantAll || contains("Extent");
    const bool wPerim  = wantAll || contains("Perimeter");
    const bool wEllipse = wMajor || wMinor || wEcc || wOrient;
    const bool wPixIdx  = wantAll || contains("PixelIdxList");
    const bool wPixList = wantAll || contains("PixelList");
    const bool wMeanI  = haveI && (wantAll || contains("MeanIntensity"));
    const bool wMaxI   = haveI && (wantAll || contains("MaxIntensity"));
    const bool wMinI   = haveI && (wantAll || contains("MinIntensity"));
    const bool wWCent  = haveI && (wantAll || contains("WeightedCentroid"));
    const bool wPixVal = haveI && (wantAll || contains("PixelValues"));
    const bool needIntensity = wMeanI || wMaxI || wMinI || wWCent || wPixVal;
    const bool needPixels = wPixIdx || wPixList || needIntensity;

    Value sa = Value::structArray(keel::as<keel::usize>(K), 1, mr);
    if (K == 0) return sa;

    keel::u32 flags = keel::image::kRegionPropBasic;
    if (wEllipse) flags |= keel::image::kRegionPropEllipse;
    if (wPerim)   flags |= keel::image::kRegionPropPerimeter;

    keel::Array<keel::image::RegionProps2D> props(mr);
    (void)keel::image::region_props_2d(L.as_ref(), H, W, K, flags, props, mr);

    // Collect column-major 1-based pixel indices per label (already sorted ascending).
    keel::Array<keel::Array<keel::i64>> pixIdx(needPixels ? keel::as<keel::usize>(K) : 0);
    if (needPixels) {
        for (keel::usize i = 0; i < N; ++i) {
            const keel::i32 lab = L[i];
            if (lab > 0 && lab <= K)
                pixIdx[keel::as<keel::usize>(lab - 1)].push_back(keel::as<keel::i64>(i + 1));
        }
    }

    for (keel::usize idx = 0; idx < keel::as<keel::usize>(K); ++idx) {
        auto &el = sa.structArrayElem(idx);
        const auto &p = props[idx];
        if (wArea)
            el.put(keel::String::from("Area"), Value::scalar(keel::as<keel::f64>(p.area), mr));
        if (wCent) {
            Value cn = Value::matrix(1, 2, ValueType::DOUBLE, mr);
            cn.doubleDataMut()[0] = (p.area > 0) ? (p.centroid_x + 1.0) : 0.0;
            cn.doubleDataMut()[1] = (p.area > 0) ? (p.centroid_y + 1.0) : 0.0;
            el.put(keel::String::from("Centroid"), keel::move(cn));
        }
        if (wBbox) {
            Value bb = Value::matrix(1, 4, ValueType::DOUBLE, mr);
            bb.doubleDataMut()[0] = keel::as<keel::f64>(p.min_col) + 0.5;
            bb.doubleDataMut()[1] = keel::as<keel::f64>(p.min_row) + 0.5;
            bb.doubleDataMut()[2] = keel::as<keel::f64>(p.max_col - p.min_col + 1);
            bb.doubleDataMut()[3] = keel::as<keel::f64>(p.max_row - p.min_row + 1);
            el.put(keel::String::from("BoundingBox"), keel::move(bb));
        }
        if (wEquivD) el.put(keel::String::from("EquivDiameter"), Value::scalar(p.equiv_diameter, mr));
        if (wExtent) el.put(keel::String::from("Extent"), Value::scalar(p.extent, mr));
        if (wPerim)  el.put(keel::String::from("Perimeter"), Value::scalar(p.perimeter, mr));
        if (wMajor)  el.put(keel::String::from("MajorAxisLength"), Value::scalar(p.major_axis_length, mr));
        if (wMinor)  el.put(keel::String::from("MinorAxisLength"), Value::scalar(p.minor_axis_length, mr));
        if (wEcc)    el.put(keel::String::from("Eccentricity"), Value::scalar(p.eccentricity, mr));
        if (wOrient) el.put(keel::String::from("Orientation"), Value::scalar(p.orientation_deg, mr));

        if (needPixels) {
            const auto &idxList = pixIdx[idx];
            const keel::usize P = idxList.size();
            if (wPixIdx) {
                Value pil = Value::matrix(P, 1, ValueType::DOUBLE, mr);
                keel::f64 *pd = pil.doubleDataMut();
                for (keel::usize i = 0; i < P; ++i) pd[i] = keel::as<keel::f64>(idxList[i]);
                el.put(keel::String::from("PixelIdxList"), keel::move(pil));
            }
            if (wPixList) {
                Value plv = Value::matrix(P, 2, ValueType::DOUBLE, mr);
                keel::f64 *pl = plv.doubleDataMut();
                for (keel::usize i = 0; i < P; ++i) {
                    const keel::i64 z = idxList[i] - 1;
                    pl[i]     = keel::as<keel::f64>(z / keel::as<keel::i64>(H) + 1);
                    pl[P + i] = keel::as<keel::f64>(z % keel::as<keel::i64>(H) + 1);
                }
                el.put(keel::String::from("PixelList"), keel::move(plv));
            }
            if (needIntensity) {
                keel::f64 sumI = 0.0, sumIx = 0.0, sumIy = 0.0;
                keel::f64 maxI = -keel::Limits<keel::f64>::infinity();
                keel::f64 minI =  keel::Limits<keel::f64>::infinity();
                Value pvv;
                keel::f64 *pv = nullptr;
                if (wPixVal) {
                    pvv = Value::matrix(P, 1, ValueType::DOUBLE, mr);
                    pv = pvv.doubleDataMut();
                }
                for (keel::usize i = 0; i < P; ++i) {
                    const keel::i64 z = idxList[i] - 1;
                    const keel::f64 iv = intensity.elemAsDouble(keel::as<keel::usize>(z));
                    sumI += iv;
                    if (iv > maxI) maxI = iv;
                    if (iv < minI) minI = iv;
                    if (wWCent) {
                        const keel::i64 row0 = z % keel::as<keel::i64>(H);
                        const keel::i64 col0 = z / keel::as<keel::i64>(H);
                        sumIx += iv * keel::as<keel::f64>(col0 + 1);
                        sumIy += iv * keel::as<keel::f64>(row0 + 1);
                    }
                    if (pv) pv[i] = iv;
                }
                if (wMeanI) el.put(keel::String::from("MeanIntensity"),
                                   Value::scalar(P ? sumI / keel::as<keel::f64>(P) : 0.0, mr));
                if (wMaxI)  el.put(keel::String::from("MaxIntensity"), Value::scalar(maxI, mr));
                if (wMinI)  el.put(keel::String::from("MinIntensity"), Value::scalar(minI, mr));
                if (wWCent) {
                    Value wc = Value::matrix(1, 2, ValueType::DOUBLE, mr);
                    wc.doubleDataMut()[0] = (sumI != 0.0) ? sumIx / sumI : 0.0;
                    wc.doubleDataMut()[1] = (sumI != 0.0) ? sumIy / sumI : 0.0;
                    el.put(keel::String::from("WeightedCentroid"), keel::move(wc));
                }
                if (wPixVal) el.put(keel::String::from("PixelValues"), keel::move(pvv));
            }
        }
    }
    return sa;
}

Value bwdist(const Value &BW, keel::Allocator *mr)
{
    bwdist_argcheck(BW);
    const keel::usize H = BW.dims().rows();
    const keel::usize W = BW.dims().cols();
    Value out = Value::matrix(H, W, ValueType::SINGLE, mr);
    if (H == 0 || W == 0) return out;
    auto fg = read_bw_col_major(BW, mr);
    keel::Array<keel::f64> dist(H * W, 0.0, mr);
    (void)keel::image::bwdist_euclidean_2d(fg.as_ref(), H, W, dist.as_mut_ref(), mr);
    keel::f32 *od = out.singleDataMut();
    for (keel::usize i = 0; i < H * W; ++i) od[i] = keel::as<keel::f32>(dist[i]);
    return out;
}

Value roicolor(const Value &A, const Value &low_or_v, keel::f64 high, bool is_range,
               keel::Allocator *mr)
{
    const auto &d = A.dims();
    const keel::usize H = d.rows(), W = d.cols(), P = d.is3D() ? d.pages() : 1;
    Value out = d.is3D() ? Value::matrix3d(H, W, P, ValueType::LOGICAL, mr)
                         : Value::matrix(H, W, ValueType::LOGICAL, mr);
    const keel::usize N = A.numel();
    if (N == 0) return out;
    keel::u8 *od = out.logicalDataMut();
    if (is_range) {
        const keel::f64 lo = low_or_v.toScalar();
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 v = A.elemAsDouble(i);
            od[i] = (v >= lo && v <= high) ? 1u : 0u;
        }
    } else {
        const keel::usize M = low_or_v.numel();
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 v = A.elemAsDouble(i);
            keel::u8 b = 0;
            for (keel::usize k = 0; k < M; ++k)
                if (v == low_or_v.elemAsDouble(k)) { b = 1; break; }
            od[i] = b;
        }
    }
    return out;
}

Value fchcode(const Value &bound, keel::Allocator *mr)
{
    fchcode_argcheck(bound);
    const keel::usize K = bound.dims().rows();
    keel::Array<keel::f64> rs(K, mr), cs(K, mr);
    for (keel::usize k = 0; k < K; ++k) {
        rs[k] = bound.elemAsDouble(k);
        cs[k] = bound.elemAsDouble(K + k);
    }
    keel::Array<keel::f64> fcc_arr(mr), diff_arr(mr);
    (void)keel::image::freeman_chain_code(rs.as_ref(), cs.as_ref(), fcc_arr, diff_arr);

    Value out = Value::structure(mr);
    Value xy0 = Value::matrix(1, 2, ValueType::DOUBLE, mr);
    if (K >= 1) {
        xy0.doubleDataMut()[0] = rs[0];
        xy0.doubleDataMut()[1] = cs[0];
    }
    out.field("x0y0") = keel::move(xy0);

    const keel::usize nCodes = fcc_arr.size();
    Value fcc = Value::matrix(1, nCodes, ValueType::DOUBLE, mr);
    Value diffV = Value::matrix(1, nCodes, ValueType::DOUBLE, mr);
    if (nCodes > 0) {
        keel::copy_n(fcc.doubleDataMut(), fcc_arr.data(), nCodes);
        keel::copy_n(diffV.doubleDataMut(), diff_arr.data(), nCodes);
    }
    out.field("fcc") = keel::move(fcc);
    out.field("diff") = keel::move(diffV);
    return out;
}

Value bwareafilt(const Value &BW, keel::f64 lo, keel::f64 hi, keel::usize n_keep,
                 bool keep_largest, keel::i32 conn, keel::Allocator *mr)
{
    bwareafilt_argcheck(BW);
    const keel::usize H = BW.dims().rows();
    const keel::usize W = BW.dims().cols();
    Value out = Value::matrix(H, W, ValueType::LOGICAL, mr);
    if (H == 0 || W == 0) return out;
    auto fg = read_bw_col_major(BW, mr);
    (void)keel::image::bwareafilt_2d(fg.as_ref(), H, W, lo, hi, n_keep, keep_largest, conn,
                                     keel::MutArrayRef<keel::u8>(out.logicalDataMut(), H * W), mr);
    return out;
}

keel::Tuple<Value, Value>
bwselect(const Value &BW, const Value &cols, const Value &rows, keel::i32 conn,
         keel::Allocator *mr)
{
    const keel::usize H = BW.dims().rows();
    const keel::usize W = BW.dims().cols();
    Value out = Value::matrix(H, W, ValueType::LOGICAL, mr);
    Value idx_out = Value::matrix(0, 1, ValueType::DOUBLE, mr);
    if (H == 0 || W == 0)
        return keel::make_tuple(keel::move(out), keel::move(idx_out));

    const keel::usize nSeed = keel::min(cols.numel(), rows.numel());
    keel::Array<keel::i32> srows(nSeed, mr), scols(nSeed, mr);
    for (keel::usize s = 0; s < nSeed; ++s) {
        scols[s] = keel::as<keel::i32>(cols.elemAsDouble(s)) - 1;
        srows[s] = keel::as<keel::i32>(rows.elemAsDouble(s)) - 1;
    }
    auto fg = read_bw_col_major(BW, mr);
    keel::u8 *od = out.logicalDataMut();
    (void)keel::image::bwselect_2d(fg.as_ref(), H, W, srows.as_ref(), scols.as_ref(), conn,
                                   keel::MutArrayRef<keel::u8>(od, H * W), mr);

    keel::Array<keel::f64> idx_v(mr);
    for (keel::usize i = 0; i < H * W; ++i) {
        if (od[i]) idx_v.push_back(keel::as<keel::f64>(i + 1));
    }
    idx_out = Value::matrix(idx_v.size(), 1, ValueType::DOUBLE, mr);
    if (!idx_v.empty()) keel::copy_n(idx_out.doubleDataMut(), idx_v.data(), idx_v.size());
    return keel::make_tuple(keel::move(out), keel::move(idx_out));
}

Value bweuler(const Value &BW, keel::i32 conn, keel::Allocator *mr)
{
    bweuler_argcheck(BW, conn);
    const keel::usize H = BW.dims().rows();
    const keel::usize W = BW.dims().cols();
    auto fg = read_bw_col_major(BW, mr);
    keel::f64 euler = 0.0;
    (void)keel::image::bweuler_2d(fg.as_ref(), H, W, conn, euler);
    return Value::scalar(euler, mr);
}

} // namespace numkit::image
