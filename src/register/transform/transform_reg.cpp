// toolboxes/image/src/transform/transform_reg.cpp
//
// Register half of the image transform builtins: the CallContext wrappers
// delegating to the engine-free compute in transform.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <numkit/image/transform/transform.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/scratch.hpp>
#include <numkit/value/span.hpp>

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

void integralImage_reg(Span<const Value> args, size_t /*nargout*/,
                       Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "integralImage");
    outs[0] = integralImage(args[0], ctx.engine->resource());
}

void integralImage3_reg(Span<const Value> args, size_t /*nargout*/,
                        Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "integralImage3");
    // no max: name-value tail in the reference
    outs[0] = integralImage3(args[0], ctx.engine->resource());
}

void dct2_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
              CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "dct2");
    outs[0] = dct2(args[0], ctx.engine->resource());
}

void idct2_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
               CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "idct2");
    outs[0] = idct2(args[0], ctx.engine->resource());
}

void dctmtx_reg(Span<const Value> args, size_t /*nargout*/, Span<Value> outs,
                CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 1, "dctmtx");
    outs[0] = dctmtx(args[0].toScalar(), ctx.engine->resource());
}

void phantom_reg(Span<const Value> args, size_t nargout,
                 Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 0, "phantom");
    auto *mr = ctx.engine->resource();
    Value model_or_E;
    size_t n = 0;
    // Argument forms:
    //   phantom()                            -> defaults
    //   phantom(model_str | E)               -> single arg
    //   phantom(N)                           -> single numeric scalar
    //   phantom(model_str | E, N)            -> two args
    if (args.size() == 1) {
        const Value &a = args[0];
        if (!a.isEmpty() && (a.isChar() || a.isString() || a.numel() != 1))
            model_or_E = a;
        else if (!a.isEmpty())
            n = argcheck::arrayDim(a, "phantom");
    } else if (args.size() >= 2) {
        if (!args[0].isEmpty()) model_or_E = args[0];
        if (!args[1].isEmpty()) n = argcheck::arrayDim(args[1], "phantom");
    }
    auto [head, E] = phantom(model_or_E, n, mr);
    outs[0] = std::move(head);
    if (nargout > 1) outs[1] = std::move(E);
}

void normxcorr2_reg(Span<const Value> args, size_t /*nargout*/,
                    Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "normxcorr2");
    outs[0] = normxcorr2(args[0], args[1], ctx.engine->resource());
}

void psf2otf_reg(Span<const Value> args, size_t /*nargout*/,
                 Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "psf2otf");
    auto *mr = ctx.engine->resource();
    ScratchArena scratch(mr);
    ScratchVec<size_t> outsizeBuf(&scratch);
    if (args.size() >= 2 && !args[1].isEmpty()) {
        const size_t n = args[1].numel();
        outsizeBuf.reserve(n);
        for (size_t i = 0; i < n; ++i)
            outsizeBuf.push_back(static_cast<size_t>(args[1].elemAsDouble(i)));
    }
    outs[0] = psf2otf(args[0], Span<const size_t>(outsizeBuf.data(), outsizeBuf.size()), mr);
}

void bestblk_reg(Span<const Value> args, size_t nargout,
                 Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "bestblk");
    auto *mr = ctx.engine->resource();
    const double k = (args.size() >= 2 && !args[1].isEmpty())
                       ? args[1].toScalar() : 100.0;
    Value v = bestblk(args[0], k, mr);
    if (nargout <= 1) { outs[0] = std::move(v); return; }
    // Multi-output form: split row vector into scalars.
    const size_t nd = v.numel();
    const size_t M = std::min<size_t>(nargout, nd);
    for (size_t i = 0; i < M; ++i) {
        outs[i] = Value::scalar(v.elemAsDouble(i), mr);
    }
}

void fftconv2_reg(Span<const Value> args, size_t /*nargout*/,
                  Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "fftconv2");
    // no max: absent from the R2025b reference build
    std::string shape = "full";
    if (args.size() >= 3 && !args[2].isEmpty()) {
        if (!args[2].isChar() && !args[2].isString())
            throw Error("fftconv2: shape must be a string",
                        0, 0, "fftconv2", "", "numkit:fftconv2:shape");
        shape = args[2].toString();
    }
    outs[0] = fftconv2(args[0], args[1], shape, ctx.engine->resource());
}

void edgetaper_reg(Span<const Value> args, std::size_t /*nargout*/,
                   Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 2, "edgetaper");
    outs[0] = edgetaper(args[0], args[1], ctx.engine->resource());
}

void deconvreg_reg(Span<const Value> args, std::size_t nargout,
                   Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 5, "deconvreg");
    auto *mr = ctx.engine->resource();
    double np = 0.0;
    double lo = 1e-9, hi = 1e9;
    Value regop = Value::Empty;
    if (args.size() >= 3 && !args[2].isEmpty()) {
        if (args[2].numel() != 1)
            throw Error("deconvreg: NP must be a scalar",
                        0, 0, "deconvreg", "", "numkit:deconvreg:np");
        np = args[2].toScalar();
    }
    if (args.size() >= 4 && !args[3].isEmpty()) {
        const Value &lr = args[3];
        if (lr.numel() == 1) {
            lo = hi = lr.toScalar();
        } else if (lr.numel() == 2) {
            lo = lr.elemAsDouble(0);
            hi = lr.elemAsDouble(1);
            if (hi < lo)
                throw Error("deconvreg: LRANGE must satisfy lo <= hi",
                            0, 0, "deconvreg", "", "numkit:deconvreg:lrange");
        } else {
            throw Error("deconvreg: LRANGE must be a scalar or 2-element vector",
                        0, 0, "deconvreg", "", "numkit:deconvreg:lrange");
        }
    }
    if (args.size() >= 5 && !args[4].isEmpty())
        regop = args[4];
    auto r = deconvreg(args[0], args[1], np, lo, hi, regop, mr);
    outs[0] = std::move(r.J);
    if (nargout > 1) outs[1] = Value::scalar(r.lagra, mr);
}

void deconvwnr_reg(Span<const Value> args, std::size_t /*nargout*/,
                   Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 4, "deconvwnr");
    auto *mr = ctx.engine->resource();
    if (args.size() == 2) {
        outs[0] = deconvwnr(args[0], args[1], 0.0, mr);
    } else if (args.size() == 3) {
        // 3-arg form: scalar NSR (most common case).
        if (args[2].numel() != 1)
            throw Error("deconvwnr: 3-arg NSR must be a scalar; use the "
                        "4-arg (NCORR, ICORR) form for array spectra",
                        0, 0, "deconvwnr", "", "numkit:deconvwnr:nsr");
        outs[0] = deconvwnr(args[0], args[1], args[2].toScalar(), mr);
    } else {
        outs[0] = deconvwnr(args[0], args[1], args[2], args[3], mr);
    }
}

void otf2psf_reg(Span<const Value> args, size_t /*nargout*/,
                 Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "otf2psf");
    auto *mr = ctx.engine->resource();
    ScratchArena scratch(mr);
    ScratchVec<size_t> outsizeBuf(&scratch);
    if (args.size() >= 2 && !args[1].isEmpty()) {
        const size_t n = args[1].numel();
        outsizeBuf.reserve(n);
        for (size_t i = 0; i < n; ++i)
            outsizeBuf.push_back(static_cast<size_t>(args[1].elemAsDouble(i)));
    }
    outs[0] = otf2psf(args[0], Span<const size_t>(outsizeBuf.data(), outsizeBuf.size()), mr);
}

void checkerboard_reg(Span<const Value> args, size_t /*nargout*/,
                      Span<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 0, 3, "checkerboard");
    size_t side = 10, M = 4, N = 4;
    if (args.size() >= 1 && !args[0].isEmpty()) {
        const double s = args[0].toScalar();
        if (s < 0.0 || s != std::floor(s))
            throw Error("checkerboard: SIDE must be a non-negative integer",
                        0, 0, "checkerboard", "", "numkit:checkerboard:side");
        side = static_cast<size_t>(s);
    }
    if (args.size() >= 2 && !args[1].isEmpty()) {
        const Value &v = args[1];
        if (v.numel() == 1)
            M = N = static_cast<size_t>(argcheck::expectedPositiveInteger(
                argcheck::expectedReal(v, 2, "M", "checkerboard"), 2, "M", "checkerboard"));
        else if (v.numel() >= 2) {
            M = static_cast<size_t>(v.elemAsDouble(0));
            N = static_cast<size_t>(v.elemAsDouble(1));
        }
    }
    if (args.size() >= 3 && !args[2].isEmpty())
        N = static_cast<size_t>(argcheck::expectedPositiveInteger(
            argcheck::expectedReal(args[2], 3, "N", "checkerboard"), 3, "N", "checkerboard"));
    outs[0] = checkerboard(side, M, N, ctx.engine->resource());
}

} // namespace detail

} // namespace numkit::image
