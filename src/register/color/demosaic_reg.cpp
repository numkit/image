// toolboxes/image/src/color/demosaic_reg.cpp
//
// Register half of the image demosaic builtin: the CallContext wrapper
// delegating to the engine-free compute in demosaic.cpp. library.cpp
// forward-declares + registers it by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/format.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <numkit/image/color/color.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>


namespace numkit::image {

namespace detail {

static keel::String asString(const Value &v) {
    if (!v.isChar() && !v.isString())
        throw Error("demosaic: expected a string argument",
                    0, 0, "demosaic", "", "numkit:demosaic:type");
    return v.toString();
}

void demosaic_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "demosaic");  // no max: name-value tail in the reference
    auto *mr = ctx.engine->allocator();
    const keel::String align = asString(args[1]);
    keel::i32 bps = 0;
    for (keel::usize i = 2; i + 1 < args.size(); i += 2) {
        const keel::String name = asString(args[i]);
        if (name.equals_ignore_ascii_case("bitspersample"))
            bps = argcheck::expectedOneOf(
                argcheck::expectedReal(args[i + 1], keel::i32(i) + 2, "BitsPerSample", "demosaic"),
                {8, 10, 12, 16}, keel::i32(i) + 2, "BitsPerSample", "demosaic");
        else
            throw Error(keel::format("demosaic: unknown name-value parameter '{}'", name),
                        0, 0, "demosaic", "", "numkit:demosaic:nv");
    }
    outs[0] = demosaic(args[0], align, bps, mr);
}

} // namespace detail

} // namespace numkit::image
