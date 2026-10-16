// toolboxes/image/src/texture/texture_reg.cpp
//
// Register half of the image texture builtins: the CallContext wrappers
// delegating to the engine-free compute in texture.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/image/texture/texture.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include "texture/texture_detail.hpp"

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

static bool eqIgnoreCase(const std::string &a, const char *b)
{
    if (a.size() != std::strlen(b)) return false;
    for (std::size_t i = 0; i < a.size(); ++i)
        if (std::tolower(a[i]) != std::tolower(b[i])) return false;
    return true;
}

void graycomatrix_reg(Span<const Value> args, std::size_t /*nargout*/,
                      Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "graycomatrix");
    // no max: name-value tail ('Offset' pairs)
    const Value &I = args[0];

    int numLevels = (I.type() == ValueType::LOGICAL) ? 2
                  : (I.type() == ValueType::UINT8)   ? 8
                                                     : 8;
    int offR = 0, offC = 1;
    double gLow = 0.0, gHigh = 0.0;
    bool   limitsSet = false;
    bool   symmetric = false;

    for (std::size_t i = 1; i + 1 < args.size(); i += 2) {
        if (!args[i].isChar())
            throw Error("graycomatrix: NV-pair name must be a string",
                         0, 0, "graycomatrix", "",
                         "numkit:graycomatrix:badNVName");
        const std::string key = args[i].toString();
        const Value &v        = args[i + 1];
        if (eqIgnoreCase(key, "NumLevels"))
            numLevels = argcheck::expectedPositiveInteger(
                argcheck::expectedReal(v, int(i) + 2, "NumLevels", "graycomatrix"),
                int(i) + 2, "NumLevels", "graycomatrix");
        else if (eqIgnoreCase(key, "Offset")) {
            if (v.numel() < 2)
                throw Error("graycomatrix: Offset must be 2-element",
                             0, 0, "graycomatrix", "",
                             "numkit:graycomatrix:badOffset");
            offR = static_cast<int>(v.elemAsDouble(0));
            offC = static_cast<int>(v.elemAsDouble(1));
        }
        else if (eqIgnoreCase(key, "GrayLimits")) {
            if (v.isEmpty()) {
                // 'GrayLimits', [] -> auto [min(I(:)) max(I(:))] over the
                // actual data (MATLAB-documented), for any class.
                data_gray_limits(I, gLow, gHigh);
                limitsSet = true;
            }
            else if (v.numel() < 2)
                throw Error("graycomatrix: GrayLimits must be 2-element",
                             0, 0, "graycomatrix", "",
                             "numkit:graycomatrix:badLimits");
            else {
                gLow  = v.elemAsDouble(0);
                gHigh = v.elemAsDouble(1);
                limitsSet = true;
            }
        }
        else if (eqIgnoreCase(key, "Symmetric")) {
            symmetric = (v.toScalar() != 0.0);
        }
        else
            throw Error("graycomatrix: unknown NV-pair key '" + key + "'",
                         0, 0, "graycomatrix", "",
                         "numkit:graycomatrix:badNVKey");
    }
    if (!limitsSet) default_gray_limits(I, gLow, gHigh);

    outs[0] = graycomatrix(I, numLevels, offR, offC, gLow, gHigh, symmetric, ctx.engine->resource());
}

void graycoprops_reg(Span<const Value> args, std::size_t /*nargout*/,
                     Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "graycoprops");
    outs[0] = graycoprops(args[0], ctx.engine->resource());
}

} // namespace detail

} // namespace numkit::image
