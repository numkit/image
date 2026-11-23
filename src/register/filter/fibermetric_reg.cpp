// toolboxes/image/src/filter/fibermetric_reg.cpp
//
// Register half of the image fibermetric builtins: the CallContext wrappers
// delegating to the engine-free compute in fibermetric.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <keel/array.h>
#include <keel/math.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <numkit/image/filter/filter.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>


namespace numkit::image {

namespace detail {

void fibermetric_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                     keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 6, "fibermetric");
    auto *mr = ctx.engine->allocator();
    auto is_string = [](const Value &v) { return v.isChar() || v.isString(); };

    const Value &I = args[0];
    keel::Array<keel::f64> thickness;
    keel::f64 c = -1.0;
    bool bright = true;

    keel::usize i = 1;
    if (i < args.size() && !is_string(args[i])) {
        const Value &v = args[i];
        const keel::usize N = v.numel();
        thickness.resize(N);
        for (keel::usize k = 0; k < N; ++k)
            thickness[k] = v.elemAsDouble(k);
        ++i;
    }
    while (i + 1 < args.size()) {
        if (!is_string(args[i]))
            throw Error("fibermetric: expected NV-pair name string",
                        0, 0, "fibermetric", "",
                        "numkit:fibermetric:badNv");
        keel::String name = args[i].toString();
        if (name.as_ref().equals_ignore_ascii_case("structuresensitivity")) {
            c = args[i + 1].toScalar();
            if (!keel::is_finite(c) || c <= 0.0)
                throw Error("fibermetric: StructureSensitivity must be "
                            "a positive finite scalar",
                            0, 0, "fibermetric", "",
                            "numkit:fibermetric:sens");
        } else if (name.as_ref().equals_ignore_ascii_case("objectpolarity")) {
            keel::String s = args[i + 1].toString();
            if (s.as_ref().equals_ignore_ascii_case("bright"))    bright = true;
            else if (s.as_ref().equals_ignore_ascii_case("dark")) bright = false;
            else throw Error("fibermetric: ObjectPolarity must be "
                             "'bright' or 'dark'",
                             0, 0, "fibermetric", "",
                             "numkit:fibermetric:polarity");
        } else {
            throw Error(keel::format("fibermetric: unknown option '{}'", name),
                        0, 0, "fibermetric", "",
                        "numkit:fibermetric:unknownNv");
        }
        i += 2;
    }
    outs[0] = fibermetric(I, thickness, c, bright, mr);
}

}  // namespace detail
}  // namespace numkit::image
