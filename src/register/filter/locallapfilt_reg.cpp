// toolboxes/image/src/filter/locallapfilt_reg.cpp
//
// Register half of the image locallapfilt builtins: the CallContext wrappers
// delegating to the engine-free compute in locallapfilt.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/format.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <numkit/image/filter/filter.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace numkit::image {

namespace detail {

void locallapfilt_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                      keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 10, "locallapfilt");
    auto *mr = ctx.engine->allocator();

    const Value &I = args[0];
    const double sigma = args[1].toScalar();
    const double alpha = args[2].toScalar();
    auto is_string = [](const Value &v) { return v.isChar() || v.isString(); };

    double beta = 1.0;
    int nlevels = -1;          // -1 = auto
    bool processLuminance = true;

    keel::usize i = 3;
    // Optional positional beta (if 4th arg is numeric scalar and not a string).
    if (i < args.size() && !is_string(args[i])) {
        if (args[i].numel() != 1)
            throw Error("locallapfilt: beta must be a scalar",
                        0, 0, "locallapfilt", "",
                        "numkit:locallapfilt:betaShape");
        beta = args[i].toScalar();
        ++i;
    }

    while (i + 1 < args.size()) {
        if (!is_string(args[i]))
            throw Error("locallapfilt: expected NV-pair name string",
                        0, 0, "locallapfilt", "",
                        "numkit:locallapfilt:badNv");
        keel::String name = args[i].toString();
        if (name.as_ref().equals_ignore_ascii_case("numintensitylevels")) {
            const Value &v = args[i + 1];
            if (is_string(v)) {
                keel::String s = v.toString();
                if (!s.as_ref().equals_ignore_ascii_case("auto"))
                    throw Error("locallapfilt: NumIntensityLevels string "
                                "must be 'auto'",
                                0, 0, "locallapfilt", "",
                                "numkit:locallapfilt:nIntStr");
                nlevels = -1;
            } else {
                nlevels = argcheck::expectedPositiveInteger(
            argcheck::expectedReal(v, int(i) + 2, "NumIntensityLevels", "locallapfilt"), 0, "NumIntensityLevels", "locallapfilt");
                if (nlevels < 1)
                    throw Error("locallapfilt: NumIntensityLevels must be >= 1",
                                0, 0, "locallapfilt", "",
                                "numkit:locallapfilt:nIntRange");
            }
        } else if (name.as_ref().equals_ignore_ascii_case("colormode")) {
            keel::String s = args[i + 1].toString();
            if (s.as_ref().equals_ignore_ascii_case("luminance"))     processLuminance = true;
            else if (s.as_ref().equals_ignore_ascii_case("separate")) processLuminance = false;
            else throw Error("locallapfilt: ColorMode must be 'luminance' or "
                             "'separate'",
                             0, 0, "locallapfilt", "",
                             "numkit:locallapfilt:colorMode");
        } else {
            throw Error(keel::format("locallapfilt: unknown option '{}'", name),
                        0, 0, "locallapfilt", "",
                        "numkit:locallapfilt:unknownNv");
        }
        i += 2;
    }

    outs[0] = locallapfilt(I, sigma, alpha, beta, nlevels,
                           processLuminance, mr);
}

} // namespace detail

} // namespace numkit::image
