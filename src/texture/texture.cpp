// toolboxes/image/src/texture/texture.cpp
//
// Texture Analysis — graycomatrix + graycoprops.

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <numkit/image/texture/texture.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include "texture_detail.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace numkit::image {

namespace {

// Map an arbitrary numeric Value element to its double representation.
inline keel::f64 elem_as_double(const Value &v, keel::usize i)
{
    return v.elemAsDouble(i);
}

void graycomatrix_argcheck(const Value &I, keel::i32 numLevels, keel::f64 gLow, keel::f64 gHigh)
{
    if (numLevels < 2)
        throw Error("graycomatrix: NumLevels must be >= 2",
                    0, 0, "graycomatrix", "", "numkit:graycomatrix:badLevels");

    const auto &d = I.dims();
    if (d.is3D())
        throw Error("graycomatrix: 2-D grayscale input only",
                    0, 0, "graycomatrix", "", "numkit:graycomatrix:unsupportedShape");

    if (gLow >= gHigh)
        throw Error("graycomatrix: GrayLimits[1] must exceed GrayLimits[0]",
                    0, 0, "graycomatrix", "", "numkit:graycomatrix:badLimits");
}

void graycoprops_argcheck(const Value &G)
{
    const auto &d = G.dims();
    if (d.rows() != d.cols() || d.rows() == 0)
        throw Error("graycoprops: GLCM must be square and non-empty",
                    0, 0, "graycoprops", "", "numkit:graycoprops:badShape");
}

} // anonymous namespace

Value graycomatrix(const Value &I, keel::i32 numLevels, keel::i32 offR, keel::i32 offC, keel::f64 gLow, keel::f64 gHigh, bool symmetric, keel::Allocator *mr)
{
    graycomatrix_argcheck(I, numLevels, gLow, gHigh);

    const auto &d = I.dims();
    const keel::usize R = d.rows();
    const keel::usize C = d.cols();

    // Pre-quantise the image into level indices in [0, numLevels-1].
    // MATLAB clamps out-of-range to the closest end-bin.
    keel::ArenaTemp scratch = keel::get_scratch(mr);
    keel::Array<keel::i32> q(R * C, scratch.allocator());
    const keel::f64 span = gHigh - gLow;
    const keel::f64 scale = keel::as<keel::f64>(numLevels) / span;
    for (keel::usize i = 0; i < R * C; ++i) {
        const keel::f64 v = elem_as_double(I, i);
        if (std::isnan(v)) {
            q[i] = -1;    // NaN — skip when forming pairs
            continue;
        }
        keel::i32 level = keel::as<keel::i32>(std::floor((v - gLow) * scale));
        if (level < 0)            level = 0;
        if (level >= numLevels)   level = numLevels - 1;
        q[i] = level;
    }

    // Allocate output (numLevels × numLevels, column-major).
    Value G = Value::matrix(keel::as<keel::usize>(numLevels),
                             keel::as<keel::usize>(numLevels),
                             ValueType::DOUBLE, mr);
    keel::f64 *Gd = G.doubleDataMut();
    std::memset(Gd, 0,
                sizeof(keel::f64) * keel::as<keel::usize>(numLevels) *
                                  keel::as<keel::usize>(numLevels));

    // Walk every pair (p, p+offset) that lies inside the image.
    for (keel::usize c = 0; c < C; ++c) {
        const keel::i64 c2 = keel::as<keel::i64>(c) + offC;
        if (c2 < 0 || c2 >= keel::as<keel::i64>(C)) continue;
        for (keel::usize r = 0; r < R; ++r) {
            const keel::i64 r2 = keel::as<keel::i64>(r) + offR;
            if (r2 < 0 || r2 >= keel::as<keel::i64>(R)) continue;
            const keel::i32 i_lvl = q[r + c * R];                // first pixel
            const keel::i32 j_lvl = q[r2 + c2 * R];              // offset pixel
            if (i_lvl < 0 || j_lvl < 0) continue;          // NaN guard
            Gd[i_lvl + j_lvl * numLevels] += 1.0;
            if (symmetric)
                Gd[j_lvl + i_lvl * numLevels] += 1.0;
        }
    }
    return G;
}

Value graycoprops(const Value &G, keel::Allocator *mr)
{
    graycoprops_argcheck(G);
    const auto &d = G.dims();
    const keel::usize L = d.rows();

    // Normalise to a joint probability.
    keel::f64 total = 0.0;
    for (keel::usize i = 0; i < L * L; ++i) total += G.elemAsDouble(i);
    if (total <= 0.0)
        throw Error("graycoprops: GLCM has zero total mass",
                    0, 0, "graycoprops", "", "numkit:graycoprops:zeroSum");

    // p(i, j) — already accessible via G.elemAsDouble(i + j*L) / total.
    // Compute marginal means / variances (1-based "intensity values"
    // matching MATLAB's convention: rows / cols treated as 1..L).
    keel::f64 mu_i = 0.0, mu_j = 0.0;
    for (keel::usize j = 0; j < L; ++j)
        for (keel::usize i = 0; i < L; ++i) {
            const keel::f64 p = G.elemAsDouble(i + j * L) / total;
            mu_i += keel::as<keel::f64>(i + 1) * p;
            mu_j += keel::as<keel::f64>(j + 1) * p;
        }
    keel::f64 var_i = 0.0, var_j = 0.0;
    for (keel::usize j = 0; j < L; ++j)
        for (keel::usize i = 0; i < L; ++i) {
            const keel::f64 p = G.elemAsDouble(i + j * L) / total;
            const keel::f64 di = (i + 1) - mu_i;
            const keel::f64 dj = (j + 1) - mu_j;
            var_i += di * di * p;
            var_j += dj * dj * p;
        }
    const keel::f64 sigma_i = std::sqrt(var_i);
    const keel::f64 sigma_j = std::sqrt(var_j);

    keel::f64 contrast = 0.0, energy = 0.0, homogeneity = 0.0, correlation = 0.0;
    for (keel::usize j = 0; j < L; ++j)
        for (keel::usize i = 0; i < L; ++i) {
            const keel::f64 p = G.elemAsDouble(i + j * L) / total;
            const keel::f64 di = keel::as<keel::f64>(i) - keel::as<keel::f64>(j);
            contrast    += di * di * p;
            energy      += p * p;
            homogeneity += p / (1.0 + std::fabs(di));
            if (sigma_i > 0.0 && sigma_j > 0.0)
                correlation += (((i + 1) - mu_i) * ((j + 1) - mu_j) * p) /
                               (sigma_i * sigma_j);
        }
    // MATLAB convention: when sigma_i or sigma_j == 0, correlation is NaN
    // (constant images). We replicate.
    if (!(sigma_i > 0.0 && sigma_j > 0.0))
        correlation = std::nan("");

    auto field_scalar = [&](keel::f64 v) { return Value::scalar(v, mr); };

    Value out = Value::structure();
    out.field("Contrast")    = field_scalar(contrast);
    out.field("Correlation") = field_scalar(correlation);
    out.field("Energy")      = field_scalar(energy);
    out.field("Homogeneity") = field_scalar(homogeneity);
    return out;
}

} // namespace numkit::image
