// toolboxes/image/src/color/chromadapt_reg.cpp
//
// Register half of the image chromadapt builtin: the CallContext wrapper
// (NV-pair parsing) delegating to the engine-free compute in chromadapt.cpp.
// library.cpp forward-declares + registers it by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/format.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <numkit/image/color/color.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <cstddef>

namespace numkit::image {

namespace detail {

void chromadapt_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                    keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 6, "chromadapt");
    auto *mr = ctx.engine->allocator();
    auto is_string = [](const Value &v) { return v.isChar() || v.isString(); };
    keel::String method = keel::String::from("bradford");
    keel::String colorSpace = keel::String::from("srgb");
    keel::usize i = 2;
    while (i + 1 < args.size()) {
        if (!is_string(args[i]))
            throw Error("chromadapt: expected NV-pair name string",
                        0, 0, "chromadapt", "", "numkit:chromadapt:badNv");
        keel::String name = args[i].toString();
        if (name.as_ref().equals_ignore_ascii_case("method"))
            method = args[i + 1].toString();
        else if (name.as_ref().equals_ignore_ascii_case("colorspace"))
            colorSpace = args[i + 1].toString();
        else throw Error(keel::format("chromadapt: unknown option '{}'", name),
                         0, 0, "chromadapt", "",
                         "numkit:chromadapt:unknownNv");
        i += 2;
    }
    outs[0] = chromadapt(args[0], args[1], method, colorSpace, mr);
}

}  // namespace detail
}  // namespace numkit::image
