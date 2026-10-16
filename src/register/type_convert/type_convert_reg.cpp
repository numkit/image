// toolboxes/image/src/type_convert/type_convert_reg.cpp
//
// Register half of the image type_convert builtins: the CallContext wrappers
// delegating to the engine-free compute in type_convert.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/image/type_convert/type_convert.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace numkit::image {

namespace detail {

#define NUMKIT_IM_UNARY_REG(name, mxArgs)                                           \
    void name##_reg(Span<const Value> args, size_t /*nargout*/,                \
                    Span<Value> outs, CallContext &ctx)                        \
    {                                                                            \
    argcheck::checkNargin(args.size(), 1, mxArgs, #name);                  \
        outs[0] = name(args[0], ctx.engine->resource());                        \
    }

// No count wall in the reference: option tail.
#define NUMKIT_IM_UNARY_NV_REG(name)                                                 \
    void name##_reg(Span<const Value> args, size_t /*nargout*/,                \
                    Span<Value> outs, CallContext &ctx)                        \
    {                                                                            \
    argcheck::checkNargin(args.size(), 1, #name);                           \
        outs[0] = name(args[0], ctx.engine->resource());                        \
    }

NUMKIT_IM_UNARY_REG(im2double, 2)
NUMKIT_IM_UNARY_REG(im2single, 2)
NUMKIT_IM_UNARY_REG(im2uint8, 2)
NUMKIT_IM_UNARY_REG(im2uint16, 2)
NUMKIT_IM_UNARY_REG(im2int16, 1)
NUMKIT_IM_UNARY_REG(im2gray, 1)
NUMKIT_IM_UNARY_NV_REG(rgb2gray)

#undef NUMKIT_IM_UNARY_REG

void mat2gray_reg(Span<const Value> args, size_t /*nargout*/,
                  Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "mat2gray");
    double lo = std::numeric_limits<double>::quiet_NaN();
    double hi = std::numeric_limits<double>::quiet_NaN();
    if (args.size() >= 2 && !args[1].isEmpty()) {
        if (args[1].numel() != 2)
            throw Error("mat2gray: range must be a 2-element vector",
                        0, 0, "mat2gray", "", "nk:mat2gray:size");
        lo = args[1].elemAsDouble(0);
        hi = args[1].elemAsDouble(1);
    }
    outs[0] = mat2gray(args[0], lo, hi, ctx.engine->resource());
}

void iptnum2ordinal_reg(Span<const Value> args, size_t /*nargout*/,
                        Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "iptnum2ordinal");
    outs[0] = iptnum2ordinal(args[0].toScalar(), ctx.engine->resource());
}

void imcast_reg(Span<const Value> args, size_t /*nargout*/,
                Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "imcast");
    // no max: absent from the R2025b reference build
    if (!args[1].isChar() && !args[1].isString())
        throw Error("imcast: TYPE must be a string",
                    0, 0, "imcast", "", "nk:imcast:type");
    outs[0] = imcast(args[0], args[1].toString(), ctx.engine->resource());
}

void gray2ind_reg(Span<const Value> args, size_t nargout,
                  Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "gray2ind");
    int n = (args[0].type() == ValueType::LOGICAL) ? 2 : 64;
    if (args.size() >= 2 && !args[1].isEmpty())
        {
        const double nv = argcheck::expectedReal(args[1], 2, "n", "gray2ind");
        argcheck::expectedInteger(nv, 2, "n", "gray2ind");
        argcheck::expectedBetween(nv, 1, 65536, 2, "n", "gray2ind");
        n = static_cast<int>(nv);
    }
    auto [ind, map] = gray2ind(args[0], n, ctx.engine->resource());
    outs[0] = std::move(ind);
    if (nargout > 1) outs[1] = std::move(map);
}

void ind2gray_reg(Span<const Value> args, size_t /*nargout*/,
                  Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "ind2gray");
    Value mp;
    if (args.size() >= 2 && !args[1].isEmpty()) mp = args[1];
    outs[0] = ind2gray(args[0], mp, ctx.engine->resource());
}

void ind2rgb_reg(Span<const Value> args, size_t /*nargout*/,
                 Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "ind2rgb");
    outs[0] = ind2rgb(args[0], args[1], ctx.engine->resource());
}

void getrangefromclass_reg(Span<const Value> args, size_t /*nargout*/,
                           Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "getrangefromclass");
    outs[0] = getrangefromclass(args[0], ctx.engine->resource());
}

void isbw_reg(Span<const Value> args, size_t /*nargout*/,
              Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "isbw");
    // no max: absent from the R2025b reference build
    std::string mode = "logical";
    if (args.size() >= 2 && !args[1].isEmpty()) {
        if (!args[1].isChar() && !args[1].isString())
            throw Error("isbw: MODE must be a string",
                        0, 0, "isbw", "", "nk:isbw:mode");
        mode = args[1].toString();
    }
    outs[0] = isbw(args[0], mode, ctx.engine->resource());
}

void isgray_reg(Span<const Value> args, size_t /*nargout*/,
                Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "isgray");
    // no max: absent from the R2025b reference build
    outs[0] = isgray(args[0], ctx.engine->resource());
}

void isind_reg(Span<const Value> args, size_t /*nargout*/,
               Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "isind");
    // no max: absent from the R2025b reference build
    outs[0] = isind(args[0], ctx.engine->resource());
}

void isrgb_reg(Span<const Value> args, size_t /*nargout*/,
               Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "isrgb");
    // no max: absent from the R2025b reference build
    outs[0] = isrgb(args[0], ctx.engine->resource());
}

void iscolormap_reg(Span<const Value> args, size_t /*nargout*/,
                    Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "iscolormap");
    // no max: absent from the R2025b reference build
    outs[0] = iscolormap(args[0], ctx.engine->resource());
}

void intlut_reg(Span<const Value> args, size_t /*nargout*/,
                Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "intlut");
    outs[0] = intlut(args[0], args[1], ctx.engine->resource());
}

} // namespace detail
} // namespace numkit::image
