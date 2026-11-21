// toolboxes/image/src/color/illum_reg.cpp
//
// Register half of the image illuminant builtins: the CallContext wrappers
// delegating to the engine-free compute in illum.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <keel/array.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <numkit/image/color/color.hpp>
#include "color/illum_detail.hpp"

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <cstddef>

namespace numkit::image {

namespace detail {

// Name-value parser shared by illumwhite/illumgray; supports the
// two MATLAB options 'Mask' and 'Norm' (the latter is illumgray-only,
// caller passes `allow_norm = false` for illumwhite).
static void parse_nv_pairs(keel::ArrayRef<Value> args, keel::usize start_idx,
                           bool allow_norm, Value &mask, keel::f64 &norm_exp,
                           const char *fn)
{
    while (start_idx + 1 < args.size()) {
        if (!args[start_idx].isChar() && !args[start_idx].isString())
            throw Error(keel::format("{}: expected option name string", fn),
                        0, 0, fn, "", "numkit:image:badNvArg");
        keel::String name = args[start_idx].toString();
        if (name.as_ref().equals_ignore_ascii_case("mask")) {
            mask = args[start_idx + 1];
        } else if (allow_norm && name.as_ref().equals_ignore_ascii_case("norm")) {
            norm_exp = args[start_idx + 1].toScalar();
        } else {
            throw Error(keel::format("{}: unknown option '{}'", fn, name),
                        0, 0, fn, "", "numkit:image:unknownNvArg");
        }
        start_idx += 2;
    }
    if (start_idx < args.size())
        throw Error(keel::format("{}: trailing unpaired name-value arg", fn),
                    0, 0, fn, "", "numkit:image:unpairedNv");
}

void illumwhite_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                    keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "illumwhite");
    auto *mr = ctx.engine->allocator();
    keel::f64 P = 1.0;       // MATLAB default percentile = 1%.
    keel::usize i = 1;
    if (args.size() >= 2 && !args[1].isChar() && !args[1].isString()) {
        P = args[1].toScalar();
        i = 2;
    }
    Value mask = Value::Empty;
    keel::f64 dummy_norm = 1.0;
    parse_nv_pairs(args, i, /*allow_norm=*/false, mask, dummy_norm,
                   "illumwhite");
    outs[0] = illumwhite(args[0], P, mask, mr);
}

void imcolordiff_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                     keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "imcolordiff");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->allocator();
    keel::String standard = keel::String::from("CIE94");
    bool is_input_lab = false;
    keel::f64 kL = 1.0, kC = 1.0, kH = 1.0, K1 = 0.045, K2 = 0.015;
    keel::usize i = 2;
    while (i + 1 < args.size()) {
        if (!args[i].isChar() && !args[i].isString())
            throw Error("imcolordiff: expected option name string",
                        0, 0, "imcolordiff", "", "numkit:imcolordiff:badNvArg");
        keel::String name = args[i].toString();
        keel::StringRef nref = name.as_ref();
        if      (nref.equals_ignore_ascii_case("standard"))   standard     = args[i + 1].toString();
        else if (nref.equals_ignore_ascii_case("isinputlab")) is_input_lab = (args[i + 1].toScalar() != 0.0);
        else if (nref.equals_ignore_ascii_case("kl"))         kL = args[i + 1].toScalar();
        else if (nref.equals_ignore_ascii_case("kc"))         kC = args[i + 1].toScalar();
        else if (nref.equals_ignore_ascii_case("kh"))         kH = args[i + 1].toScalar();
        else if (nref.equals_ignore_ascii_case("k1"))         K1 = args[i + 1].toScalar();
        else if (nref.equals_ignore_ascii_case("k2"))         K2 = args[i + 1].toScalar();
        else
            throw Error(keel::format("imcolordiff: unknown option '{}'", name),
                        0, 0, "imcolordiff", "", "numkit:imcolordiff:unknownNv");
        i += 2;
    }
    if (i < args.size())
        throw Error("imcolordiff: trailing unpaired name-value arg",
                    0, 0, "imcolordiff", "", "numkit:imcolordiff:unpaired");
    outs[0] = imcolordiff(args[0], args[1], standard, is_input_lab,
                          kL, kC, kH, K1, K2, mr);
}

void illumpca_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "illumpca");
    auto *mr = ctx.engine->allocator();
    keel::f64 P = 3.5;       // MATLAB default percentage.
    keel::usize i = 1;
    if (args.size() >= 2 && !args[1].isChar() && !args[1].isString()) {
        P = args[1].toScalar();
        i = 2;
    }
    Value mask = Value::Empty;
    keel::f64 dummy_norm = 1.0;
    parse_nv_pairs(args, i, /*allow_norm=*/false, mask, dummy_norm,
                   "illumpca");
    outs[0] = illumpca(args[0], P, mask, mr);
}

void illumgray_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                   keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 6, "illumgray");
    auto *mr = ctx.engine->allocator();
    keel::Array<keel::f64> P;
    keel::usize i = 1;
    if (args.size() >= 2 && !args[1].isChar() && !args[1].isString()) {
        const keel::usize n = args[1].numel();
        if (n == 1) P.push_back(args[1].toScalar());
        else if (n == 2) {
            P.push_back(args[1].elemAsDouble(0));
            P.push_back(args[1].elemAsDouble(1));
        } else if (n != 0) {
            throw Error("illumgray: percentile must be scalar or 2-vector",
                        0, 0, "illumgray", "", "numkit:illumgray:percentile");
        }
        i = 2;
    }
    Value mask = Value::Empty;
    keel::f64 norm_exp = 1.0;
    parse_nv_pairs(args, i, /*allow_norm=*/true, mask, norm_exp,
                   "illumgray");
    outs[0] = illumgray_impl(args[0], P, mask, norm_exp, mr);
}

} // namespace detail
} // namespace numkit::image
