// toolboxes/image/src/color/raw_planar_reg.cpp
//
// Register half of the image raw/planar builtins (raw2planar / planar2raw):
// CallContext wrappers delegating to the engine-free compute in raw_planar.cpp.
// library.cpp forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/image/color/color.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <cstddef>

namespace numkit::image {

namespace detail {

void raw2planar_reg(Span<const Value> args, size_t /*nargout*/,
                    Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "raw2planar");
    // no max: name-value tail in the reference
    outs[0] = raw2planar(args[0], ctx.engine->resource());
}

void planar2raw_reg(Span<const Value> args, size_t /*nargout*/,
                    Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "planar2raw");
    // no max: name-value tail in the reference
    outs[0] = planar2raw(args[0], ctx.engine->resource());
}

} // namespace detail

} // namespace numkit::image
