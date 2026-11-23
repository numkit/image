// toolboxes/image/src/filter/imreducehaze_reg.cpp
//
// Register half of the image imreducehaze builtins: the CallContext wrappers
// delegating to the engine-free compute in imreducehaze.cpp. library.cpp
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
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>
#include <utility>

namespace numkit::image {

namespace detail {

void imreducehaze_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                      keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 10, "imreducehaze");
    auto *mr = ctx.engine->allocator();
    auto is_string = [](const Value &v) { return v.isChar() || v.isString(); };

    const Value &I = args[0];
    keel::f64 amount = 1.0;
    keel::String method = keel::String::from("simpledcp");
    Value atmLight;
    keel::String ce = keel::String::from("global");
    keel::f64 boostAmount = 0.0;  // sentinel: default-on-demand
    bool boostExplicitlySet = false;

    keel::usize i = 1;
    if (i < args.size() && !is_string(args[i])) {
        amount = args[i].toScalar();
        ++i;
    }
    while (i + 1 < args.size()) {
        if (!is_string(args[i]))
            throw Error("imreducehaze: expected NV-pair name string",
                        0, 0, "imreducehaze", "",
                        "numkit:imreducehaze:badNv");
        keel::String name = args[i].toString();
        keel::StringRef nref = name.as_ref();
        if (nref.equals_ignore_ascii_case("method")) {
            method = args[i + 1].toString();
        } else if (nref.equals_ignore_ascii_case("atmosphericlight")) {
            atmLight = args[i + 1];
        } else if (nref.equals_ignore_ascii_case("contrastenhancement")) {
            ce = args[i + 1].toString();
        } else if (nref.equals_ignore_ascii_case("boostamount")) {
            boostAmount = args[i + 1].toScalar();
            boostExplicitlySet = true;
        } else {
            throw Error(keel::format("imreducehaze: unknown option '{}'", name),
                        0, 0, "imreducehaze", "",
                        "numkit:imreducehaze:unknownNv");
        }
        i += 2;
    }
    if (boostExplicitlySet) {
        // Match MATLAB: BoostAmount only allowed with boost contrast.
        if (!ce.as_ref().equals_ignore_ascii_case("boost"))
            throw Error("imreducehaze: BoostAmount may only be specified "
                        "when ContrastEnhancement is 'boost'",
                        0, 0, "imreducehaze", "",
                        "numkit:imreducehaze:boostNotAllowed");
    }

    Value T, L;
    outs[0] = imreducehaze(I, amount, method, atmLight, ce,
                           boostAmount > 0.0 ? boostAmount : 0.1,
                           T, L, mr);
    if (nargout >= 2 && outs.size() >= 2) outs[1] = std::move(T);
    if (nargout >= 3 && outs.size() >= 3) outs[2] = std::move(L);
}

}  // namespace detail
}  // namespace numkit::image
