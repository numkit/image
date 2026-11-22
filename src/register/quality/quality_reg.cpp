// toolboxes/image/src/quality/quality_reg.cpp
//
// Register half of the image quality builtins: the CallContext wrappers
// delegating to the engine-free compute in quality.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <keel/array.h>
#include <keel/math.h>
#include <keel/core.h>
#include <keel/format.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/image/quality/quality.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <utility>

namespace numkit::image {

namespace detail {

void immse_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "immse");
    outs[0] = immse(args[0], args[1], ctx.engine->allocator());
}

void psnr_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
              keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "psnr");
    const keel::f64 peak = (args.size() >= 3 && !args[2].isEmpty())
                        ? args[2].toScalar() : keel::kNaN;
    outs[0] = psnr(args[0], args[1], peak, ctx.engine->allocator());
}

void ssim_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
              keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "ssim");
    // no max: option-string tail
    outs[0] = ssim(args[0], args[1], ctx.engine->allocator());
}

void mean2_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "mean2");
    outs[0] = mean2(args[0], ctx.engine->allocator());
}

void std2_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
              keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "std2");
    outs[0] = std2(args[0], ctx.engine->allocator());
}

void corr2_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
               keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "corr2");
    outs[0] = corr2(args[0], args[1], ctx.engine->allocator());
}

void multissim3_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                    keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "multissim3");
    // no max: option-string tail
    auto *mr = ctx.engine->allocator();
    auto is_string = [](const Value &v) { return v.isChar() || v.isString(); };

    keel::i32 num_scales = 5;
    keel::Array<keel::f64> scale_weights;
    keel::f64 sigma = 1.5;
    keel::f64 dynamic_range = -1.0;

    keel::usize i = 2;
    while (i + 1 < args.size()) {
        if (!is_string(args[i]))
            throw Error("multissim3: expected NV-pair name string",
                        0, 0, "multissim3", "", "numkit:multissim3:badNv");
        keel::String name = args[i].toString();
        keel::String nlo;
        for (char ch : name)
            nlo.push_back(keel::to_ascii_lower(ch));
        if (nlo == "numscales") {
            num_scales = argcheck::expectedPositiveInteger(
                argcheck::expectedReal(args[i + 1], keel::i32(i) + 2, "NumScales", "multissim3"),
                keel::i32(i) + 2, "NumScales", "multissim3");
        } else if (nlo == "scaleweights") {
            const Value &v = args[i + 1];
            const keel::usize N = v.numel();
            scale_weights.resize(N);
            for (keel::usize k = 0; k < N; ++k)
                scale_weights[k] = v.elemAsDouble(k);
        } else if (nlo == "sigma") {
            sigma = args[i + 1].toScalar();
        } else if (nlo == "dynamicrange") {
            dynamic_range = args[i + 1].toScalar();
        } else {
            throw Error(keel::format("multissim3: unknown option '{}'", name),
                        0, 0, "multissim3", "",
                        "numkit:multissim3:unknownNv");
        }
        i += 2;
    }
    keel::Array<Value> qmaps;
    outs[0] = multissim3(args[0], args[1], num_scales, scale_weights,
                         sigma, dynamic_range,
                         nargout >= 2 ? &qmaps : nullptr, mr);
    if (nargout >= 2 && outs.size() >= 2) {
        Value cell = Value::cell(1, qmaps.size(), mr);
        for (keel::usize k = 0; k < qmaps.size(); ++k)
            cell.cellAt(k) = std::move(qmaps[k]);
        outs[1] = std::move(cell);
    }
}

void multissim_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                   keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "multissim");
    // no max: option-string tail
    auto *mr = ctx.engine->allocator();
    auto is_string = [](const Value &v) { return v.isChar() || v.isString(); };

    keel::i32 num_scales = 5;
    keel::Array<keel::f64> scale_weights;
    keel::f64 sigma = 1.5;
    keel::f64 dynamic_range = -1.0;

    keel::usize i = 2;
    while (i + 1 < args.size()) {
        if (!is_string(args[i]))
            throw Error("multissim: expected NV-pair name string",
                        0, 0, "multissim", "", "numkit:multissim:badNv");
        keel::String name = args[i].toString();
        keel::String nlo;
        for (char ch : name)
            nlo.push_back(keel::to_ascii_lower(ch));
        if (nlo == "numscales") {
            num_scales = argcheck::expectedPositiveInteger(
                argcheck::expectedReal(args[i + 1], keel::i32(i) + 2, "NumScales", "multissim"),
                keel::i32(i) + 2, "NumScales", "multissim");
        } else if (nlo == "scaleweights") {
            const Value &v = args[i + 1];
            const keel::usize N = v.numel();
            scale_weights.resize(N);
            for (keel::usize k = 0; k < N; ++k)
                scale_weights[k] = v.elemAsDouble(k);
        } else if (nlo == "sigma") {
            sigma = args[i + 1].toScalar();
        } else if (nlo == "dynamicrange") {
            dynamic_range = args[i + 1].toScalar();
        } else {
            throw Error(keel::format("multissim: unknown option '{}'", name),
                        0, 0, "multissim", "",
                        "numkit:multissim:unknownNv");
        }
        i += 2;
    }
    keel::Array<Value> qmaps;
    outs[0] = multissim(args[0], args[1], num_scales, scale_weights,
                        sigma, dynamic_range,
                        nargout >= 2 ? &qmaps : nullptr, mr);
    if (nargout >= 2 && outs.size() >= 2) {
        // Return cell array of per-scale maps.
        Value cell = Value::cell(1, qmaps.size(), mr);
        for (keel::usize k = 0; k < qmaps.size(); ++k)
            cell.cellAt(k) = std::move(qmaps[k]);
        outs[1] = std::move(cell);
    }
}

} // namespace detail
} // namespace numkit::image
