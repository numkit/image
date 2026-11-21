// toolboxes/image/src/morph/morph_reg.cpp
//
// Register half of the image morphology builtins: the CallContext wrappers
// (two detail blocks, mirroring the interleaved compute TU) plus the VM-pausable
// makelut callback (MakelutCallbackBuiltin + registerMakelutCallbackBuiltin),
// all delegating to the engine-free compute in morph.cpp. library.cpp
// forward-declares + registers these by name. core/callback_builtin.hpp +
// core/vm.hpp (the continuation machinery) live here on the register side.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/image/morph/morph.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace numkit::image {

// [Phase 2b] relocated from morph.cpp compute: the reg-only strel default
// (diamond_cross) and the synchronous Engine-driven makelut. The VM-pausable
// path is MakelutCallbackBuiltin below.
namespace {
Value diamond_cross(keel::Allocator *mr) {
    Value se = Value::matrix(3, 3, ValueType::LOGICAL, mr);
    keel::u8 *d = se.logicalDataMut();
    // Column-major fill: cols [c0(r0..2), c1(r0..2), c2(r0..2)].
    // Pattern: [0 1 0; 1 1 1; 0 1 0] — center cross.
    static constexpr keel::u8 pat[9] = {0,1,0, 1,1,1, 0,1,0};
    for (keel::usize i = 0; i < 9; ++i) d[i] = pat[i];
    return se;
}
} // anonymous

// makelut's compute (Value makelut(FnHandle, int, mr)) lives in the image
// toolbox (morph.cpp) — core-free. makelut_reg below is the synchronous
// fallback adapter (builtin handles); the VM-pausable user-code path is
// MakelutCallbackBuiltin further down.

namespace detail {

void strel_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 0, 4, "strel");
    if (args.empty()) {
        outs[0] = strel("square", {3.0}, Value(), ctx.engine->allocator());
        return;
    }
    keel::String shape = keel::String::from("square");
    if (args[0].isChar() || args[0].isString()) shape = args[0].toString();
    keel::Array<keel::f64> params;
    Value arbitrary;
    for (keel::usize i = 1; i < args.size(); ++i) {
        if (!(args[i].isChar() || args[i].isString())) {
            if (shape == "arbitrary") { arbitrary = args[i]; continue; }
            for (keel::usize j = 0; j < args[i].numel(); ++j)
                params.push_back(args[i].elemAsDouble(j));
        }
    }
    outs[0] = strel(shape, params, arbitrary, ctx.engine->allocator());
}

#define NUMKIT_MORPH_REG(name, mxArgs)                                               \
    void name##_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,                 \
                    keel::MutArrayRef<Value> outs, CallContext &ctx)                         \
    {                                                                             \
    argcheck::checkNargin(args.size(), 2, mxArgs, #name);                   \
        outs[0] = name(args[0], args[1], ctx.engine->allocator());                \
    }

NUMKIT_MORPH_REG(imerode, 5)
NUMKIT_MORPH_REG(imdilate, 5)
NUMKIT_MORPH_REG(imopen, 2)
NUMKIT_MORPH_REG(imclose, 2)

#undef NUMKIT_MORPH_REG

void imreconstruct_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                       keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "imreconstruct");
    const int conn = (args.size() >= 3 && !args[2].isEmpty())
                     ? argcheck::expectedOneOf(
            argcheck::expectedReal(args[2], 3, "conn", "imreconstruct"),
            {4, 8}, 3, "conn", "imreconstruct") : 8;
    outs[0] = imreconstruct(args[0], args[1], conn, ctx.engine->allocator());
}

void imfill_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "imfill");
    auto *mr = ctx.engine->allocator();
    // Currently we support `imfill(BW, 'holes' [, conn])` only.
    if (args.size() < 2 ||
        !(args[1].isChar() || args[1].isString()))
        throw Error("imfill: only the 'holes' mode is implemented",
                    0, 0, "imfill", "", "numkit:imfill:mode");
    const keel::String mode = args[1].toString();
    if (mode != "holes" && mode != "Holes" && mode != "HOLES")
        throw Error("imfill: only 'holes' mode is implemented "
                    "(seed-list mode not yet supported)",
                    0, 0, "imfill", "", "numkit:imfill:mode");
    // Default connectivity is 4 — MATLAB's imfill uses conndef(2,'minimal').
    // With 8-connectivity the background flood leaks through 1-px diagonal
    // gaps in the foreground, so enclosed regions wrongly read as
    // border-reachable and don't fill.
    const int conn = (args.size() >= 3 && !args[2].isEmpty())
                     ? argcheck::expectedOneOf(
            argcheck::expectedReal(args[2], 3, "conn", "imfill"),
            {4, 8}, 3, "conn", "imfill") : 4;
    outs[0] = imfill_holes(args[0], conn, mr);
}

void imregionalmax_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                       keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "imregionalmax");
    const int conn = (args.size() >= 2 && !args[1].isEmpty())
                     ? argcheck::expectedOneOf(
            argcheck::expectedReal(args[1], 2, "conn", "imregionalmax"),
            {4, 8}, 2, "conn", "imregionalmax") : 8;
    outs[0] = imregionalmax(args[0], conn, ctx.engine->allocator());
}

void imregionalmin_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                       keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "imregionalmin");
    const int conn = (args.size() >= 2 && !args[1].isEmpty())
                     ? argcheck::expectedOneOf(
            argcheck::expectedReal(args[1], 2, "conn", "imregionalmin"),
            {4, 8}, 2, "conn", "imregionalmin") : 8;
    outs[0] = imregionalmin(args[0], conn, ctx.engine->allocator());
}

void imhmax_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "imhmax");
    const keel::f64 h = args[1].toScalar();
    const int conn = (args.size() >= 3 && !args[2].isEmpty())
                     ? argcheck::expectedOneOf(
            argcheck::expectedReal(args[2], 3, "conn", "imhmax"),
            {4, 8}, 3, "conn", "imhmax") : 8;
    outs[0] = imhmax(args[0], h, conn, ctx.engine->allocator());
}

void imhmin_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "imhmin");
    // no max: name-value tail in the reference
    const keel::f64 h = args[1].toScalar();
    const int conn = (args.size() >= 3 && !args[2].isEmpty())
                     ? argcheck::expectedOneOf(
            argcheck::expectedReal(args[2], 3, "conn", "imhmin"),
            {4, 8}, 3, "conn", "imhmin") : 8;
    outs[0] = imhmin(args[0], h, conn, ctx.engine->allocator());
}

void imextendedmax_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                       keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "imextendedmax");
    const keel::f64 h = args[1].toScalar();
    const int conn = (args.size() >= 3 && !args[2].isEmpty())
                     ? argcheck::expectedOneOf(
            argcheck::expectedReal(args[2], 3, "conn", "imextendedmax"),
            {4, 8}, 3, "conn", "imextendedmax") : 8;
    outs[0] = imextendedmax(args[0], h, conn, ctx.engine->allocator());
}

void imextendedmin_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                       keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "imextendedmin");
    const keel::f64 h = args[1].toScalar();
    const int conn = (args.size() >= 3 && !args[2].isEmpty())
                     ? argcheck::expectedOneOf(
            argcheck::expectedReal(args[2], 3, "conn", "imextendedmin"),
            {4, 8}, 3, "conn", "imextendedmin") : 8;
    outs[0] = imextendedmin(args[0], h, conn, ctx.engine->allocator());
}

void imimposemin_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                     keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "imimposemin");
    const int conn = (args.size() >= 3 && !args[2].isEmpty())
                     ? argcheck::expectedOneOf(
            argcheck::expectedReal(args[2], 3, "conn", "imimposemin"),
            {4, 8}, 3, "conn", "imimposemin") : 8;
    outs[0] = imimposemin(args[0], args[1], conn, ctx.engine->allocator());
}

void imclearborder_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                       keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "imclearborder");
    const int conn = (args.size() >= 2 && !args[1].isEmpty())
                     ? argcheck::expectedOneOf(
            argcheck::expectedReal(args[1], 2, "conn", "imclearborder"),
            {4, 8}, 2, "conn", "imclearborder") : 8;
    outs[0] = imclearborder(args[0], conn, ctx.engine->allocator());
}

void imkeepborder_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                      keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "imkeepborder");
    // no max: name-value tail in the reference
    const int conn = (args.size() >= 2 && !args[1].isEmpty())
                     ? argcheck::expectedOneOf(
            argcheck::expectedReal(args[1], 2, "conn", "imkeepborder"),
            {4, 8}, 2, "conn", "imkeepborder") : 8;
    outs[0] = imkeepborder(args[0], conn, ctx.engine->allocator());
}

void imtophat_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "imtophat");
    outs[0] = imtophat(args[0], args[1], ctx.engine->allocator());
}

void imbothat_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "imbothat");
    outs[0] = imbothat(args[0], args[1], ctx.engine->allocator());
}

void mmgradm_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "mmgradm");
    // no max: absent from the R2025b reference build
    auto *mr = ctx.engine->allocator();
    // Defaults: arg omitted → elementary cross. Arg explicitly empty
    // means half-gradient (the C++ function reads .numel() == 0).
    Value sed = (args.size() >= 2) ? args[1] : diamond_cross(mr);
    Value see = (args.size() >= 3) ? args[2] : diamond_cross(mr);
    outs[0] = mmgradm(args[0], sed, see, mr);
}

void bwpack_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "bwpack");
    outs[0] = bwpack(args[0], ctx.engine->allocator());
}

void bwunpack_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "bwunpack");
    keel::usize M = keel::as<keel::usize>(-1);
    if (args.size() >= 2 && !args[1].isEmpty())
        M = keel::as<keel::usize>(argcheck::expectedPositiveInteger(
            argcheck::expectedReal(args[1], 2, "M", "bwunpack"), 2, "M", "bwunpack"));
    outs[0] = bwunpack(args[0], M, ctx.engine->allocator());
}

void applylut_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "applylut");
    outs[0] = applylut(args[0], args[1], ctx.engine->allocator());
}

void bwlookup_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "bwlookup");
    outs[0] = bwlookup(args[0], args[1], ctx.engine->allocator());
}

void makelut_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "makelut");
    // no max: variadic passthrough to the callback
    const int n = argcheck::expectedOneOf(
            argcheck::expectedReal(args[1], 2, "n", "makelut"),
            {2, 3}, 2, "n", "makelut");
    const Value &handle = args[0];
    // Bind the user/builtin function handle into a core-free FnHandle for the
    // toolbox compute (the VM-pausable user-code path is MakelutCallbackBuiltin).
    auto cb = [&ctx, &handle](keel::ArrayRef<Value> a, keel::MutArrayRef<Value> o,
                              keel::Allocator * /*mr*/) {
        auto r = ctx.engine->callFunctionHandleMulti(handle, a, o.size());
        for (keel::usize i = 0; i < o.size() && i < r.size(); ++i)
            o[i] = std::move(r[i]);
    };
    outs[0] = makelut(cb, n, ctx.engine->allocator());
}

void bwmorph3_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "bwmorph3");
    // no max: strel-object + option tail
    if (!(args[1].isChar() || args[1].isString()))
        throw Error("bwmorph3: requires (V, operation)",
                    0, 0, "bwmorph3", "", "numkit:bwmorph3:nargin");
    keel::String op = args[1].toString();
    outs[0] = bwmorph3(args[0], op, ctx.engine->allocator());
}

void bwhitmiss_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                   keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "bwhitmiss");
    auto *mr = ctx.engine->allocator();
    Value se1, se2;
    if (args.size() == 2) {
        // Single interval matrix with values in {-1, 0, 1}.
        const Value &iv = args[1];
        const keel::usize H = iv.dims().rows();
        const keel::usize W = iv.dims().cols();
        const keel::usize N = iv.numel();
        se1 = Value::matrix(H, W, ValueType::LOGICAL, mr);
        se2 = Value::matrix(H, W, ValueType::LOGICAL, mr);
        keel::u8 *s1 = se1.logicalDataMut();
        keel::u8 *s2 = se2.logicalDataMut();
        for (keel::usize i = 0; i < N; ++i) {
            const keel::f64 v = iv.elemAsDouble(i);
            s1[i] = (v ==  1.0) ? 1u : 0u;
            s2[i] = (v == -1.0) ? 1u : 0u;
        }
    } else {
        se1 = args[1];
        se2 = args[2];
    }
    outs[0] = bwhitmiss(args[0], se1, se2, mr);
}

} // namespace detail

namespace detail {

void bwmorph_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "bwmorph");
    if (!args[1].isChar())
        throw Error("bwmorph: op must be a string",
                    0, 0, "bwmorph", "", "numkit:bwmorph:badOp");
    keel::String op = args[1].toString();
    // Normalise to lowercase.
    for (auto &ch : op) ch = keel::to_ascii_lower(ch);
    // MATLAB accepts "skel" as a prefix-match for "skeleton".
    if (op == "skel") op.assign("skeleton");

    int n = 1;
    if (args.size() >= 3 && !args[2].isEmpty()) {
        const keel::f64 v = args[2].toScalar();
        if (std::isinf(v)) n = -1;
        else               n = keel::as<int>(v);
    }
    outs[0] = bwmorph(args[0], op, n, ctx.engine->allocator());
}

void bwtraceboundary_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                         keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 6, "bwtraceboundary");
    auto *mr = ctx.engine->allocator();
    if (!args[2].isChar() && !args[2].isString())
        throw Error("bwtraceboundary: FSTEP must be a string",
                    0, 0, "bwtraceboundary", "",
                    "numkit:bwtraceboundary:fstep");
    keel::String fstep = args[2].toString();
    // Upper-case fstep.
    for (auto &ch : fstep)
        ch = keel::to_ascii_upper(ch);

    int conn = 8;
    keel::usize m_max = std::numeric_limits<keel::usize>::max();
    keel::String dir = keel::String::from("clockwise");
    if (args.size() >= 4 && !args[3].isEmpty())
        conn = argcheck::expectedOneOf(
        argcheck::expectedReal(args[3], 4, "conn", "bwtraceboundary"),
        {4, 8}, 4, "conn", "bwtraceboundary");
    if (args.size() >= 5 && !args[4].isEmpty()) {
        const keel::f64 v = args[4].toScalar();
        if (!std::isinf(v)) {
            if (!(v > 0))
                throw Error("bwtraceboundary: N must be positive or Inf",
                            0, 0, "bwtraceboundary", "",
                            "numkit:bwtraceboundary:n");
            m_max = keel::as<keel::usize>(v);
        }
    }
    if (args.size() >= 6 && !args[5].isEmpty()) {
        if (!args[5].isChar() && !args[5].isString())
            throw Error("bwtraceboundary: DIR must be a string",
                        0, 0, "bwtraceboundary", "",
                        "numkit:bwtraceboundary:dirArg");
        dir = args[5].toString();
        for (char &ch : dir)
            ch = keel::to_ascii_lower(ch);
    }
    outs[0] = bwtraceboundary(args[0], args[1], fstep, conn,
                              m_max, dir, mr);
}

} // namespace detail

} // namespace numkit::image
