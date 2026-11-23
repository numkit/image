// toolboxes/image/src/type_convert/type_convert_reg.cpp
//
// Register half of the image type_convert builtins: the CallContext wrappers
// delegating to the engine-free compute in type_convert.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <numkit/image/type_convert/type_convert.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cstdlib>
#include <tuple>
#include <utility>

namespace numkit::image {

namespace detail {

#define NUMKIT_IM_UNARY_REG(name, mxArgs)                                           \
    void name##_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,                \
                    keel::MutArrayRef<Value> outs, CallContext &ctx)                        \
    {                                                                            \
    argcheck::checkNargin(args.size(), 1, mxArgs, #name);                  \
        outs[0] = name(args[0], ctx.engine->allocator());                        \
    }

// No count wall in the reference: option tail.
#define NUMKIT_IM_UNARY_NV_REG(name)                                                 \
    void name##_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,                \
                    keel::MutArrayRef<Value> outs, CallContext &ctx)                        \
    {                                                                            \
    argcheck::checkNargin(args.size(), 1, #name);                           \
        outs[0] = name(args[0], ctx.engine->allocator());                        \
    }

NUMKIT_IM_UNARY_REG(im2double, 2)
NUMKIT_IM_UNARY_REG(im2single, 2)
NUMKIT_IM_UNARY_REG(im2uint8, 2)
NUMKIT_IM_UNARY_REG(im2uint16, 2)
NUMKIT_IM_UNARY_REG(im2int16, 1)
NUMKIT_IM_UNARY_REG(im2gray, 1)
NUMKIT_IM_UNARY_NV_REG(rgb2gray)

#undef NUMKIT_IM_UNARY_REG

void mat2gray_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "mat2gray");
    keel::f64 lo = keel::Limits<keel::f64>::quiet_nan();
    keel::f64 hi = keel::Limits<keel::f64>::quiet_nan();
    if (args.size() >= 2 && !args[1].isEmpty()) {
        if (args[1].numel() != 2)
            throw Error("mat2gray: range must be a 2-element vector",
                        0, 0, "mat2gray", "", "numkit:mat2gray:size");
        lo = args[1].elemAsDouble(0);
        hi = args[1].elemAsDouble(1);
    }
    outs[0] = mat2gray(args[0], lo, hi, ctx.engine->allocator());
}

void iptnum2ordinal_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                        keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "iptnum2ordinal");
    outs[0] = iptnum2ordinal(args[0].toScalar(), ctx.engine->allocator());
}

void imcast_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "imcast");
    // no max: absent from the R2025b reference build
    if (!args[1].isChar() && !args[1].isString())
        throw Error("imcast: TYPE must be a string",
                    0, 0, "imcast", "", "numkit:imcast:type");
    keel::String targetType = args[1].toString();
    outs[0] = imcast(args[0], targetType, ctx.engine->allocator());
}

void gray2ind_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "gray2ind");
    keel::i32 n = (args[0].type() == ValueType::LOGICAL) ? 2 : 64;
    if (args.size() >= 2 && !args[1].isEmpty())
        {
        const keel::f64 nv = argcheck::expectedReal(args[1], 2, "n", "gray2ind");
        argcheck::expectedInteger(nv, 2, "n", "gray2ind");
        argcheck::expectedBetween(nv, 1, 65536, 2, "n", "gray2ind");
        n = keel::as<keel::i32>(nv);
    }
    auto [ind, map] = gray2ind(args[0], n, ctx.engine->allocator());
    outs[0] = std::move(ind);
    if (nargout > 1) outs[1] = std::move(map);
}

void ind2gray_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "ind2gray");
    Value mp;
    if (args.size() >= 2 && !args[1].isEmpty()) mp = args[1];
    outs[0] = ind2gray(args[0], mp, ctx.engine->allocator());
}

void ind2rgb_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "ind2rgb");
    outs[0] = ind2rgb(args[0], args[1], ctx.engine->allocator());
}

void getrangefromclass_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                           keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "getrangefromclass");
    outs[0] = getrangefromclass(args[0], ctx.engine->allocator());
}

void isbw_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
              keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "isbw");
    // no max: absent from the R2025b reference build
    keel::String mode = keel::String::from("logical");
    if (args.size() >= 2 && !args[1].isEmpty()) {
        if (!args[1].isChar() && !args[1].isString())
            throw Error("isbw: MODE must be a string",
                        0, 0, "isbw", "", "numkit:isbw:mode");
        mode = args[1].toString();
    }
    outs[0] = isbw(args[0], mode, ctx.engine->allocator());
}

void isgray_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "isgray");
    // no max: absent from the R2025b reference build
    outs[0] = isgray(args[0], ctx.engine->allocator());
}

void isind_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "isind");
    // no max: absent from the R2025b reference build
    outs[0] = isind(args[0], ctx.engine->allocator());
}

void isrgb_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "isrgb");
    // no max: absent from the R2025b reference build
    outs[0] = isrgb(args[0], ctx.engine->allocator());
}

void iscolormap_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                    keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "iscolormap");
    // no max: absent from the R2025b reference build
    outs[0] = iscolormap(args[0], ctx.engine->allocator());
}

void intlut_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "intlut");
    outs[0] = intlut(args[0], args[1], ctx.engine->allocator());
}

} // namespace detail
} // namespace numkit::image
