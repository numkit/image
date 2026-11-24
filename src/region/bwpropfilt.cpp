// toolboxes/image/src/region/bwpropfilt.cpp
//
// Thin Value/argcheck wrapper delegating all 17 bwpropfilt component
// attributes to keel::image::region_component_attrs.

#include <keel/core.h>
#include <keel/algo.h>
#include <keel/array.h>
#include <keel/format.h>
#include <keel/image.h>
#include <keel/memory.h>
#include <keel/string.h>
#include <keel/utf8.h>

#include <numkit/image/region/region.hpp>
#include <numkit/image/internal/type_gates.hpp>
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

namespace numkit::image {
namespace {

inline keel::String lower_str(keel::StringRef s)
{
    keel::String lo;
    lo.reserve(s.size());
    for (char ch : s) lo.push_back(keel::to_ascii_lower(ch));
    return lo;
}

struct CCInfoBPF {
    keel::i32 H = 0, W = 0;
    keel::i32 conn = 8;
    keel::Array<keel::i32> pixel_rows;
    keel::Array<keel::i32> pixel_cols;
    keel::Array<keel::usize> comp_offsets;
};

CCInfoBPF parse_cc(const Value &CC, const char *fn, keel::Allocator *mr)
{
    CCInfoBPF info;
    info.pixel_rows = keel::Array<keel::i32>(mr);
    info.pixel_cols = keel::Array<keel::i32>(mr);
    info.comp_offsets = keel::Array<keel::usize>(mr);
    if (!CC.isStruct() || CC.numel() != 1)
        throw Error(keel::format("{}: CC must be a 1×1 struct", fn),
                    0, 0, fn, "", keel::format("numkit:{}:notStruct", fn));
    const auto &el = CC.structArrayElem(0);
    auto need = [&](const char *name) -> const Value & {
        const Value *v = el.get(name);
        if (!v)
            throw Error(keel::format("{}: CC missing '{}'", fn, name),
                        0, 0, fn, "", keel::format("numkit:{}:noField", fn));
        return *v;
    };
    const Value &sz = need("ImageSize");
    if (sz.numel() < 2)
        throw Error(keel::format("{}: CC.ImageSize too short", fn),
                    0, 0, fn, "", keel::format("numkit:{}:sizeDim", fn));
    info.H = keel::as<keel::i32>(sz.elemAsDouble(0));
    info.W = keel::as<keel::i32>(sz.elemAsDouble(1));
    if (const Value *vConn = el.get("Connectivity"))
        info.conn = keel::as<keel::i32>(vConn->toScalar());
    const keel::usize K = keel::as<keel::usize>(need("NumObjects").toScalar());
    const Value &pl = need("PixelIdxList");
    info.comp_offsets.reserve(K + 1);
    info.comp_offsets.push_back(0);
    for (keel::usize k = 0; k < K; ++k) {
        const Value &cell = pl.cellAt(k);
        const keel::usize N = cell.numel();
        for (keel::usize i = 0; i < N; ++i) {
            const keel::usize idx1 = keel::as<keel::usize>(cell.elemAsDouble(i));
            if (idx1 == 0) {
                info.pixel_rows.push_back(0);
                info.pixel_cols.push_back(0);
                continue;
            }
            const keel::i32 idx0 = keel::as<keel::i32>(idx1 - 1);
            const keel::i32 col = idx0 / info.H;
            const keel::i32 row = idx0 - col * info.H;
            info.pixel_rows.push_back(row);
            info.pixel_cols.push_back(col);
        }
        info.comp_offsets.push_back(info.pixel_rows.size());
    }
    return info;
}

keel::image::RegionAttr parse_region_attr(keel::StringRef attrib_lower, const Value &marker)
{
    using A = keel::image::RegionAttr;
    if (attrib_lower == "area")            return A::kArea;
    if (attrib_lower == "equivdiameter")   return A::kEquivDiameter;
    if (attrib_lower == "extent")          return A::kExtent;
    if (attrib_lower == "majoraxislength") return A::kMajorAxisLength;
    if (attrib_lower == "minoraxislength") return A::kMinorAxisLength;
    if (attrib_lower == "eccentricity")    return A::kEccentricity;
    if (attrib_lower == "orientation")     return A::kOrientation;
    if (attrib_lower == "perimeter")       return A::kPerimeter;
    if (attrib_lower == "perimeterold")    return A::kPerimeterOld;
    if (attrib_lower == "circularity")     return A::kCircularity;
    if (attrib_lower == "convexarea")      return A::kConvexArea;
    if (attrib_lower == "solidity")        return A::kSolidity;
    if (attrib_lower == "eulernumber")     return A::kEulerNumber;
    if (attrib_lower == "filledarea")      return A::kFilledArea;
    if (attrib_lower == "minintensity") {
        if (marker.isEmpty())
            throw Error("bwpropfilt: MinIntensity needs a marker image",
                        0, 0, "bwpropfilt", "", "numkit:bwpropfilt:needMarker");
        return A::kMinIntensity;
    }
    if (attrib_lower == "maxintensity") {
        if (marker.isEmpty())
            throw Error("bwpropfilt: MaxIntensity needs a marker image",
                        0, 0, "bwpropfilt", "", "numkit:bwpropfilt:needMarker");
        return A::kMaxIntensity;
    }
    if (attrib_lower == "meanintensity") {
        if (marker.isEmpty())
            throw Error("bwpropfilt: MeanIntensity needs a marker image",
                        0, 0, "bwpropfilt", "", "numkit:bwpropfilt:needMarker");
        return A::kMeanIntensity;
    }
    throw Error(keel::format("bwpropfilt: unknown attribute '{}' "
                             "(supported: Area, Circularity, ConvexArea, "
                             "Eccentricity, EquivDiameter, EulerNumber, Extent, "
                             "FilledArea, MajorAxisLength, MaxIntensity, "
                             "MeanIntensity, MinIntensity, MinorAxisLength, "
                             "Orientation, Perimeter, PerimeterOld, Solidity)",
                             attrib_lower),
                0, 0, "bwpropfilt", "", "numkit:bwpropfilt:badAttr");
}

void bwpropfilt_argcheck(const Value &BW, const Value &CC)
{
    if (CC.isEmpty() && !BW.isEmpty())
        internal::requireImageClass(BW, "bwpropfilt", "IN", 1,
                                    internal::ImgClasses::LogicalOnly);
}

} // namespace

Value bwpropfilt(const Value &BW, const Value &CC, const Value &marker,
                 keel::StringRef attribute,
                 keel::f64 p_min, keel::f64 p_max,
                 keel::usize keep_n, bool keep_largest,
                 keel::i32 conn, keel::Allocator *mr)
{
    bwpropfilt_argcheck(BW, CC);
    const bool from_cc = !CC.isEmpty();
    const keel::String attrib_lo = lower_str(attribute);

    Value cc_owned;
    const Value *cc_use = nullptr;
    if (from_cc) {
        cc_use = &CC;
    } else {
        if (BW.isEmpty() || BW.numel() == 0) {
            return Value::matrix(BW.dims().rows(), BW.dims().cols(), ValueType::LOGICAL, mr);
        }
        if (conn != 4) conn = 8;
        cc_owned = bwconncomp(BW, conn, mr);
        cc_use = &cc_owned;
    }
    CCInfoBPF info = parse_cc(*cc_use, "bwpropfilt", mr);
    const keel::usize K = info.comp_offsets.size() - 1;
    if (K == 0) {
        if (from_cc) return CC;
        return Value::matrix(keel::as<keel::usize>(info.H),
                             keel::as<keel::usize>(info.W), ValueType::LOGICAL, mr);
    }

    const keel::image::RegionAttr attr = parse_region_attr(attrib_lo, marker);
    keel::Array<keel::f64> marker_buf(mr);
    if (!marker.isEmpty()) {
        marker_buf.resize(marker.numel());
        for (keel::usize i = 0; i < marker.numel(); ++i)
            marker_buf[i] = marker.elemAsDouble(i);
    }
    keel::Array<keel::f64> vals(K, 0.0, mr);
    (void)keel::image::region_component_attrs(
        info.pixel_rows.as_ref(), info.pixel_cols.as_ref(), info.comp_offsets.as_ref(),
        keel::as<keel::usize>(info.H), keel::as<keel::usize>(info.W), info.conn, attr,
        marker_buf.as_ref(), vals.as_mut_ref(), mr);

    keel::Array<keel::u8> keep(K, 0, mr);
    if (keep_n > 0) {
        keel::Array<keel::usize> idx(K, mr);
        for (keel::usize i = 0; i < K; ++i) idx[i] = i;
        if (keep_largest) {
            keel::stable_sort(idx.begin(), idx.end(),
                              [&](keel::usize a, keel::usize b) { return vals[a] > vals[b]; });
        } else {
            keel::stable_sort(idx.begin(), idx.end(),
                              [&](keel::usize a, keel::usize b) { return vals[a] < vals[b]; });
        }
        const keel::usize p = keel::min(keep_n, idx.size());
        const keel::f64 cutoff = vals[idx[p - 1]];
        for (keel::usize i = 0; i < K; ++i) {
            if (keep_largest ? (vals[i] >= cutoff) : (vals[i] <= cutoff)) keep[i] = 1;
        }
    } else {
        for (keel::usize i = 0; i < K; ++i)
            keep[i] = (vals[i] >= p_min && vals[i] <= p_max) ? 1u : 0u;
    }

    if (from_cc) {
        Value out = Value::structArray(1, 1, mr);
        auto &el = out.structArrayElem(0);
        const auto &src = cc_use->structArrayElem(0);
        if (const Value *v = src.get("Connectivity"))
            el.put(keel::String::from("Connectivity"), *v);
        if (const Value *v = src.get("ImageSize"))
            el.put(keel::String::from("ImageSize"), *v);
        keel::usize Kkept = 0;
        for (keel::usize k = 0; k < K; ++k) if (keep[k]) ++Kkept;
        el.put(keel::String::from("NumObjects"), Value::scalar(keel::as<keel::f64>(Kkept), mr));
        Value cell = Value::cell(1, Kkept, mr);
        const Value &orig = *src.get("PixelIdxList");
        keel::usize w = 0;
        for (keel::usize k = 0; k < K; ++k)
            if (keep[k]) cell.cellAt(w++) = orig.cellAt(k);
        el.put(keel::String::from("PixelIdxList"), keel::move(cell));
        return out;
    }

    Value outBW = Value::matrix(keel::as<keel::usize>(info.H),
                                keel::as<keel::usize>(info.W), ValueType::LOGICAL, mr);
    keel::u8 *od = outBW.logicalDataMut();
    keel::fill(od, od + keel::as<keel::usize>(info.H) * keel::as<keel::usize>(info.W), keel::u8{0});
    for (keel::usize k = 0; k < K; ++k) {
        if (!keep[k]) continue;
        const keel::usize beg = info.comp_offsets[k];
        const keel::usize end = info.comp_offsets[k + 1];
        for (keel::usize i = beg; i < end; ++i)
            od[keel::as<keel::usize>(info.pixel_cols[i] * info.H + info.pixel_rows[i])] = 1;
    }
    return outBW;
}

} // namespace numkit::image
