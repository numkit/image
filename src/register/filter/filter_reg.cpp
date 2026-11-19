// toolboxes/image/src/filter/filter_reg.cpp
//
// Register half of the image filter builtins: the CallContext wrappers (two
// detail blocks, mirroring the interleaved compute TU) plus the VM-pausable
// nlfilter callback (NlfilterCallbackBuiltin + registerNlfilterCallbackBuiltin),
// all delegating to the engine-free compute in filter.cpp. library.cpp
// forward-declares + registers these by name. core/callback_builtin.hpp +
// core/vm.hpp (the continuation machinery) live here on the register side.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/array.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/image/filter/filter.hpp>
#include <numkit/image/type_convert/type_convert.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/argcheck.hpp>
#include <numkit/value/error.hpp>
#include "filter/filter_detail.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <utility>

namespace numkit::image {

// [Phase 2b] relocated from filter.cpp compute: the synchronous Engine-driven
// neighbourhood filters (nlfilter/colfilt). The VM-pausable nlfilter path is
// NlfilterCallbackBuiltin below.

// ── nlfilter (general sliding-neighbourhood) ──────────────────────
//
// MATLAB R2025b nlfilter.m:
//   B = nlfilter(A, [m n], fun)
//   B = nlfilter(A, 'indexed', [m n], fun)
//
// For every pixel (i, j) ∈ A extract the m × n window
//   x = aa(i + 0..m-1, j + 0..n-1)
// from the padded image `aa` and call `fun(x)`. The output element
// is `b(i, j) = fun(x)`. Output class equals the class of the FIRST
// invocation of `fun` (matches MATLAB's `mkconstarray(class(...))`).
// Default `padval = 0`; `'indexed'` form uses `padval = 1` for
// `single` / `double` `A`, otherwise `padval = 0`.
//
// The dispatch goes through Engine::callFunctionHandle, matching the
// pattern adopted in toolboxes/ode/ode45 (function_ref couldn't carry
// func-handle semantics through the round-trip).
Value nlfilter(const Value &A,
               keel::usize m, keel::usize n, FnHandle fun,
               bool indexed,
               keel::Allocator *mr)
{
    if (m < 1 || n < 1)
        throw Error("nlfilter: neighbourhood size must be positive",
                    0, 0, "nlfilter", "", "numkit:nlfilter:nhood");

    const auto &dA = A.dims();
    if (dA.is3D())
        throw Error("nlfilter: A must be a 2-D image",
                    0, 0, "nlfilter", "", "numkit:nlfilter:rank");
    const keel::usize H = dA.rows();
    const keel::usize W = dA.cols();
    const ValueType inT = A.type();

    // Invoke the user callback on one input (core-free via FnHandle;
    // the bundle adapter binds it to Engine::callFunctionHandle).
    auto callFun = [&](const Value &arg) -> Value {
        std::array<Value, 1> o;
        fun(keel::ArrayRef<Value>(&arg, 1), keel::MutArrayRef<Value>(o.data(), 1), mr);
        return std::move(o[0]);
    };

    // Padding: 'indexed' uses 1.0 for single/double, else 0.
    double padval = 0.0;
    if (indexed) {
        padval = (inT == ValueType::DOUBLE || inT == ValueType::SINGLE)
               ? 1.0 : 0.0;
    }

    // Pad above by floor((m-1)/2) rows, below by ceil((m-1)/2);
    // left by floor((n-1)/2), right by ceil((n-1)/2). (MATLAB:
    // mkconstarray(class(a), padval, size(a)+nhood-1) — then drops
    // the original A into the offset block.)
    const keel::usize pad_top  = (m - 1) / 2;
    const keel::usize pad_left = (n - 1) / 2;
    const keel::usize Hpad     = H + m - 1;
    const keel::usize Wpad     = W + n - 1;

    // Build padded array in DOUBLE (we always read out via
    // elemAsDouble; this saves a class-specific dispatch).
    keel::Array<double> aa(Hpad * Wpad, padval, mr);
    for (keel::usize j = 0; j < W; ++j)
        for (keel::usize i = 0; i < H; ++i)
            aa[(j + pad_left) * Hpad + (i + pad_top)] = A.elemAsDouble(j * H + i);

    // Allocate scratch window (DOUBLE — the class for `fun` is what
    // the kernel receives; MATLAB nlfilter forwards the same class
    // as `A`, but we choose DOUBLE here for simplicity. Tests use
    // class-agnostic kernels like `@(x) mean(x(:))`).
    Value window = Value::matrix(m, n, ValueType::DOUBLE, mr);
    double *wd = window.doubleDataMut();

    // First-call invocation determines output class.
    auto fill_window = [&](keel::usize i, keel::usize j) {
        for (keel::usize c = 0; c < n; ++c)
            for (keel::usize r = 0; r < m; ++r)
                wd[c * m + r] = aa[(j + c) * Hpad + (i + r)];
    };

    fill_window(0, 0);
    Value first = callFun(window);
    if (first.numel() != 1)
        throw Error("nlfilter: fun must return a scalar",
                    0, 0, "nlfilter", "", "numkit:nlfilter:funScalar");
    const ValueType outT = first.type();
    Value B = Value::matrix(H, W, outT, mr);

    auto store = [&](keel::usize r, keel::usize c, const Value &v) {
        const keel::usize idx = c * H + r;
        const double d = v.toScalar();
        switch (outT) {
            case ValueType::DOUBLE:  B.doubleDataMut()[idx]  = d; break;
            case ValueType::SINGLE:  B.singleDataMut()[idx]  = static_cast<float>(d); break;
            case ValueType::UINT8:   B.uint8DataMut()[idx]   = static_cast<keel::u8>(d); break;
            case ValueType::UINT16:  B.uint16DataMut()[idx]  = static_cast<keel::u16>(d); break;
            case ValueType::UINT32:  B.uint32DataMut()[idx]  = static_cast<keel::u32>(d); break;
            case ValueType::UINT64:  B.uint64DataMut()[idx]  = static_cast<keel::u64>(d); break;
            case ValueType::INT8:    B.int8DataMut()[idx]    = static_cast<keel::i8>(d); break;
            case ValueType::INT16:   B.int16DataMut()[idx]   = static_cast<keel::i16>(d); break;
            case ValueType::INT32:   B.int32DataMut()[idx]   = static_cast<keel::i32>(d); break;
            case ValueType::INT64:   B.int64DataMut()[idx]   = static_cast<keel::i64>(d); break;
            case ValueType::LOGICAL: B.logicalDataMut()[idx] = d != 0.0 ? 1 : 0; break;
            default:
                throw Error("nlfilter: unsupported fun output class",
                            0, 0, "nlfilter", "", "numkit:nlfilter:outCls");
        }
    };

    store(0, 0, first);

    for (keel::usize i = 0; i < H; ++i) {
        for (keel::usize j = 0; j < W; ++j) {
            if (i == 0 && j == 0) continue;  // already filled
            fill_window(i, j);
            Value r = callFun(window);
            if (r.numel() != 1)
                throw Error("nlfilter: fun must return a scalar at all (i,j)",
                            0, 0, "nlfilter", "", "numkit:nlfilter:funScalar");
            store(i, j, r);
        }
    }
    return B;
}


// ── colfilt (column-wise neighbourhood) ───────────────────────────
//
// MATLAB R2025b colfilt.m:
//   B = colfilt(A, [m n], block_type, fun)         (whole-matrix)
//   B = colfilt(A, [m n], [mblock nblock], block_type, fun)
//   B = colfilt(A, 'indexed', …)
//
// block_type ∈ {'sliding', 'distinct'} (case-insensitive, abbrev'd
// by leading char). Sliding mode:
//   1. Pad A by (m-1, n-1) with 0 (or 1 for 'indexed' double/single).
//   2. X is the matrix whose columns are the m*n elements of every
//      m × n window centred on (i, j); shape m*n × (H*W).
//   3. Call fun(X) — must return a row vector 1 × (H*W).
//   4. Reshape into H × W.
// Distinct mode:
//   1. Pad A to next multiple of [m, n].
//   2. X has one column per distinct m × n block.
//   3. fun(X) must return a same-size matrix; the columns are then
//      unpacked back into blocks (col2im 'distinct').
//   4. Crop the assembled image back to size(A).
//
// The optional [mblock nblock] arg is purely a memory optimisation
// (MATLAB explicitly notes: "does not change the result"). The
// engine adapter accepts and ignores it.
//
// Output class equals the class of fun()'s return value.
Value colfilt(const Value &A,
              keel::usize m, keel::usize n,
              keel::StringRef block_type, FnHandle fun,
              bool indexed,
              keel::Allocator *mr)
{
    if (m < 1 || n < 1)
        throw Error("colfilt: block size must be positive",
                    0, 0, "colfilt", "", "numkit:colfilt:nhood");

    const auto &dA = A.dims();
    if (dA.is3D())
        throw Error("colfilt: A must be a 2-D image",
                    0, 0, "colfilt", "", "numkit:colfilt:rank");
    const keel::usize H = dA.rows();
    const keel::usize W = dA.cols();
    const ValueType inT = A.type();

    // Invoke the user callback on one input (core-free via FnHandle;
    // the bundle adapter binds it to Engine::callFunctionHandle).
    auto callFun = [&](const Value &arg) -> Value {
        std::array<Value, 1> o;
        fun(keel::ArrayRef<Value>(&arg, 1), keel::MutArrayRef<Value>(o.data(), 1), mr);
        return std::move(o[0]);
    };

    if (block_type.empty())
        throw Error("colfilt: block_type must be 'sliding' or 'distinct'",
                    0, 0, "colfilt", "", "numkit:colfilt:blockType");
    const char first = keel::to_ascii_lower(block_type[0]);
    if (first != 's' && first != 'd')
        throw Error("colfilt: block_type must be 'sliding' or 'distinct'",
                    0, 0, "colfilt", "", "numkit:colfilt:blockType");

    // Common padval rule.
    double padval = 0.0;
    if (indexed && (inT == ValueType::DOUBLE || inT == ValueType::SINGLE))
        padval = 1.0;

    if (first == 's') {
        // ── Sliding ─────────────────────────────────────────────
        const keel::usize pad_top  = (m - 1) / 2;
        const keel::usize pad_left = (n - 1) / 2;
        const keel::usize Hpad = H + m - 1;
        const keel::usize Wpad = W + n - 1;
        const keel::usize Ncol = H * W;

        // Build X = m*n × Ncol in DOUBLE.
        Value X = Value::matrix(m * n, Ncol, ValueType::DOUBLE, mr);
        double *xd = X.doubleDataMut();

        // Pad row-by-col into a temporary scratch.
        keel::Array<double> aa(Hpad * Wpad, padval, mr);
        for (keel::usize j = 0; j < W; ++j)
            for (keel::usize i = 0; i < H; ++i)
                aa[(j + pad_left) * Hpad + (i + pad_top)] =
                    A.elemAsDouble(j * H + i);

        // For each centre (i, j), gather m*n window values into
        // column k = j*H + i.
        for (keel::usize j = 0; j < W; ++j) {
            for (keel::usize i = 0; i < H; ++i) {
                const keel::usize k = j * H + i;
                keel::usize r = 0;
                // im2col 'sliding' iteration order: column-major
                // within each window, i.e. (col-major) flatten.
                for (keel::usize wc = 0; wc < n; ++wc)
                    for (keel::usize wr = 0; wr < m; ++wr)
                        xd[k * (m * n) + (r++)]
                            = aa[(j + wc) * Hpad + (i + wr)];
            }
        }

        Value result = callFun(X);
        if (result.numel() != Ncol)
            throw Error("colfilt: sliding fun must return a 1 × N row "
                        "vector (one value per column)",
                        0, 0, "colfilt", "", "numkit:colfilt:funShape");

        // Reshape result (regardless of [1, Ncol] or [Ncol, 1]) into H × W.
        Value B = Value::matrix(H, W, result.type(), mr);
        const auto outT = result.type();
        for (keel::usize k = 0; k < Ncol; ++k) {
            const double v = result.elemAsDouble(k);
            const keel::usize idx = k;
            switch (outT) {
                case ValueType::DOUBLE: B.doubleDataMut()[idx]  = v; break;
                case ValueType::SINGLE: B.singleDataMut()[idx]  = static_cast<float>(v); break;
                case ValueType::UINT8:  B.uint8DataMut()[idx]   = static_cast<keel::u8>(v); break;
                case ValueType::UINT16: B.uint16DataMut()[idx]  = static_cast<keel::u16>(v); break;
                case ValueType::INT16:  B.int16DataMut()[idx]   = static_cast<keel::i16>(v); break;
                case ValueType::INT32:  B.int32DataMut()[idx]   = static_cast<keel::i32>(v); break;
                case ValueType::LOGICAL: B.logicalDataMut()[idx] = v != 0 ? 1 : 0; break;
                default:
                    throw Error("colfilt: unsupported fun output class",
                                0, 0, "colfilt", "", "numkit:colfilt:outCls");
            }
        }
        return B;
    }

    // ── Distinct ────────────────────────────────────────────────────
    const keel::usize mpad = (H % m) ? (m - H % m) : 0;
    const keel::usize npad = (W % n) ? (n - W % n) : 0;
    const keel::usize Hpad = H + mpad;
    const keel::usize Wpad = W + npad;
    const keel::usize mblocks = Hpad / m;
    const keel::usize nblocks = Wpad / n;
    const keel::usize Ncol    = mblocks * nblocks;

    // Pad to multiple of (m, n).
    keel::Array<double> aa(Hpad * Wpad, padval, mr);
    for (keel::usize j = 0; j < W; ++j)
        for (keel::usize i = 0; i < H; ++i)
            aa[j * Hpad + i] = A.elemAsDouble(j * H + i);

    // Build X = m*n × Ncol; column index k = bj * mblocks + bi.
    Value X = Value::matrix(m * n, Ncol, ValueType::DOUBLE, mr);
    double *xd = X.doubleDataMut();
    for (keel::usize bj = 0; bj < nblocks; ++bj) {
        for (keel::usize bi = 0; bi < mblocks; ++bi) {
            const keel::usize k = bj * mblocks + bi;
            keel::usize r = 0;
            for (keel::usize wc = 0; wc < n; ++wc)
                for (keel::usize wr = 0; wr < m; ++wr)
                    xd[k * (m * n) + (r++)]
                        = aa[(bj * n + wc) * Hpad + (bi * m + wr)];
        }
    }

    Value result = callFun(X);
    if (result.numel() != m * n * Ncol)
        throw Error("colfilt: distinct fun must return a matrix of the "
                    "same shape as its input (m*n × N)",
                    0, 0, "colfilt", "", "numkit:colfilt:funShape");

    // Reassemble padded image, then crop.
    const auto outT = result.type();
    keel::Array<double> bb(Hpad * Wpad, 0.0, mr);
    for (keel::usize bj = 0; bj < nblocks; ++bj) {
        for (keel::usize bi = 0; bi < mblocks; ++bi) {
            const keel::usize k = bj * mblocks + bi;
            keel::usize r = 0;
            for (keel::usize wc = 0; wc < n; ++wc)
                for (keel::usize wr = 0; wr < m; ++wr) {
                    const double v = result.elemAsDouble(k * (m * n) + (r++));
                    bb[(bj * n + wc) * Hpad + (bi * m + wr)] = v;
                }
        }
    }

    Value B = Value::matrix(H, W, outT, mr);
    for (keel::usize j = 0; j < W; ++j) {
        for (keel::usize i = 0; i < H; ++i) {
            const double v = bb[j * Hpad + i];
            const keel::usize idx = j * H + i;
            switch (outT) {
                case ValueType::DOUBLE: B.doubleDataMut()[idx]  = v; break;
                case ValueType::SINGLE: B.singleDataMut()[idx]  = static_cast<float>(v); break;
                case ValueType::UINT8:  B.uint8DataMut()[idx]   = static_cast<keel::u8>(v); break;
                case ValueType::UINT16: B.uint16DataMut()[idx]  = static_cast<keel::u16>(v); break;
                case ValueType::INT16:  B.int16DataMut()[idx]   = static_cast<keel::i16>(v); break;
                case ValueType::INT32:  B.int32DataMut()[idx]   = static_cast<keel::i32>(v); break;
                case ValueType::LOGICAL: B.logicalDataMut()[idx] = v != 0 ? 1 : 0; break;
                default:
                    throw Error("colfilt: unsupported fun output class",
                                0, 0, "colfilt", "", "numkit:colfilt:outCls");
            }
        }
    }
    return B;
}

namespace detail {

namespace {
keel::String to_lower_ascii(keel::StringRef s) {
    keel::String out;
    out.reserve(s.size());
    for (char ch : s)
        out.push_back(keel::to_ascii_lower(ch));
    return out;
}

PadMode parse_pad_mode(const Value &v, double &pad_value, bool &is_value) {
    is_value = false;
    pad_value = 0.0;
    if (v.isChar() || v.isString()) {
        auto s = v.toString();
        if (s == "replicate") return PadMode::Replicate;
        if (s == "symmetric") return PadMode::Symmetric;
        if (s == "circular")  return PadMode::Circular;
        // Unknown mode → treat as scalar 0.
    }
    if (v.numel() == 1) {
        is_value = true;
        pad_value = v.toScalar();
    }
    return PadMode::Constant;
}
} // anonymous

void padarray_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 4, "padarray");
    keel::Array<int> padsize;
    {
        const Value &p = args[1];
        const keel::usize n = p.numel();
        padsize.resize(n);
        for (keel::usize i = 0; i < n; ++i) padsize[i] = int(p.elemAsDouble(i));
    }

    PadMode mode = PadMode::Constant;
    double pad_value = 0.0;
    keel::String direction = keel::String::from("both");

    // Parse optional trailing args: pad_value-or-mode (one) and/or direction.
    for (keel::usize i = 2; i < args.size(); ++i) {
        const Value &a = args[i];
        if (a.isChar() || a.isString()) {
            keel::String s = a.toString();
            if (s == "pre" || s == "post" || s == "both") {
                direction = std::move(s);
            } else {
                bool dummy;
                mode = parse_pad_mode(a, pad_value, dummy);
            }
        } else {
            // Treat as scalar pad value.
            pad_value = a.toScalar();
            mode = PadMode::Constant;
        }
    }

    outs[0] = padarray(args[0], padsize, mode, pad_value, direction, ctx.engine->allocator());
}

void fspecial_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "fspecial");
    keel::String type = keel::String::from("gaussian");
    if (!args.empty() && (args[0].isChar() || args[0].isString()))
        type = args[0].toString();

    keel::Array<double> params;
    if (type == "motion") {
        // 'motion' takes two INDEPENDENT scalars [len, theta]; do NOT
        // size-double the 2nd arg the way the size-based filters do.
        if (args.size() >= 2 && args[1].numel() >= 1)
            params.push_back(args[1].elemAsDouble(0));
        if (args.size() >= 3 && args[2].numel() >= 1)
            params.push_back(args[2].elemAsDouble(0));
    } else {
        // Second positional arg can be a scalar (square size) or a 2-vector
        // [rows cols]; a scalar size is doubled so a following scalar (sigma /
        // alpha) lands in slot 3.
        if (args.size() >= 2) {
            const Value &v = args[1];
            if (v.numel() == 1) {
                params.push_back(v.toScalar());
                params.push_back(v.toScalar());
            } else if (v.numel() == 2) {
                params.push_back(v.elemAsDouble(0));
                params.push_back(v.elemAsDouble(1));
            } else if (v.numel() > 0) {
                for (keel::usize i = 0; i < v.numel(); ++i)
                    params.push_back(v.elemAsDouble(i));
            }
        }
        // Third positional arg = sigma / alpha / radius (scalar).
        if (args.size() >= 3 && args[2].numel() == 1)
            params.push_back(args[2].toScalar());
    }

    outs[0] = fspecial(type, params, ctx.engine->allocator());
}

void imfilter_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 5, "imfilter");
    PadMode boundary = PadMode::Constant;
    double pad_value = 0.0;
    bool full = false;
    bool flip_kernel = false;  // 'corr' default
    for (keel::usize i = 2; i < args.size(); ++i) {
        const Value &a = args[i];
        if (a.isChar() || a.isString()) {
            keel::String s = to_lower_ascii(a.toString());
            if      (s.as_ref().starts_with("rep"))  boundary = PadMode::Replicate;
            else if (s.as_ref().starts_with("sym"))  boundary = PadMode::Symmetric;
            else if (s.as_ref().starts_with("circ")) boundary = PadMode::Circular;
            else if (s.as_ref().starts_with("full")) full = true;
            else if (s.as_ref().starts_with("same")) full = false;
            else if (s.as_ref().starts_with("conv")) flip_kernel = true;
            else if (s.as_ref().starts_with("corr")) flip_kernel = false;
        } else {
            pad_value = a.toScalar();
            boundary = PadMode::Constant;
        }
    }
    outs[0] = imfilter(args[0], args[1], boundary, pad_value, full, flip_kernel, ctx.engine->allocator());
}

void imgaussfilt_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                     keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "imgaussfilt");
    // no max: name-value tail in the reference
    double sigma = (args.size() >= 2 && !args[1].isEmpty()) ? args[1].toScalar() : 0.5;
    int fs = 0;  // auto
    // Look for 'FilterSize' name-value pair.
    for (keel::usize i = 2; i + 1 < args.size(); ++i) {
        if ((args[i].isChar() || args[i].isString())
            && args[i].toString() == "FilterSize") {
            fs = (int)args[i + 1].toScalar();
            break;
        }
    }
    outs[0] = imgaussfilt(args[0], sigma, fs, ctx.engine->allocator());
}

void imboxfilt_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                   keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "imboxfilt");
    // no max: name-value tail in the reference
    int fs = (args.size() >= 2 && !args[1].isEmpty()) ? (int)args[1].toScalar() : 3;
    outs[0] = imboxfilt(args[0], fs, ctx.engine->allocator());
}

void modefilt_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                   keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "modefilt");
    // Defaults adapt to input rank — MATLAB picks [3 3] for 2-D, [3 3 3] for 3-D.
    const bool input3D = (args[0].dims().ndim() == 3);
    int fH = 3, fW = 3, fD = input3D ? 3 : 0;
    keel::String padopt = keel::String::from("symmetric");
    if (args.size() >= 2 && !args[1].isEmpty()) {
        if (args[1].isChar() || args[1].isString()) {
            padopt = args[1].toString();
        } else {
            if (args[1].numel() == 1) {
                fH = fW = argcheck::expectedPositiveInteger(
            argcheck::expectedReal(args[1], 2, "window", "modefilt"), 2, "window", "modefilt");
                if (input3D) fD = fH;
            } else if (args[1].numel() == 2) {
                fH = static_cast<int>(args[1].elemAsDouble(0));
                fW = static_cast<int>(args[1].elemAsDouble(1));
                if (input3D) fD = 1;  // no 3rd dim filter specified
            } else if (args[1].numel() >= 3) {
                fH = static_cast<int>(args[1].elemAsDouble(0));
                fW = static_cast<int>(args[1].elemAsDouble(1));
                fD = static_cast<int>(args[1].elemAsDouble(2));
            }
        }
    }
    if (args.size() >= 3 && !args[2].isEmpty()) {
        if (args[2].isChar() || args[2].isString())
            padopt = args[2].toString();
        else
            throw Error("modefilt: padopt must be a string",
                        0, 0, "modefilt", "", "numkit:modefilt:badPad");
    }
    if (input3D)
        outs[0] = modefilt3D(args[0], fH, fW, fD, padopt, ctx.engine->allocator());
    else
        outs[0] = modefilt(args[0], fH, fW, padopt, ctx.engine->allocator());
}

void integralBoxFilter_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                            keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "integralBoxFilter");
    // no max: name-value tail in the reference// Defaults match MATLAB: 3-by-3 box.
    int fH = 3, fW = 3;
    keel::usize nvStart = 1;
    if (args.size() >= 2 && !args[1].isEmpty()
        && !args[1].isChar() && !args[1].isString()) {
        const Value &fsArg = args[1];
        if (fsArg.numel() == 1) {
            fH = fW = argcheck::expectedPositiveInteger(
            argcheck::expectedReal(fsArg, 2, "window", "integralBoxFilter"), 2, "window", "integralBoxFilter");
        } else if (fsArg.numel() == 2) {
            fH = static_cast<int>(fsArg.elemAsDouble(0));
            fW = static_cast<int>(fsArg.elemAsDouble(1));
        } else {
            throw Error("integralBoxFilter: filterSize must be a scalar or 2-element vector",
                        0, 0, "integralBoxFilter", "", "numkit:integralBoxFilter:badSize");
        }
        nvStart = 2;
    }

    // MATLAB default NormalizationFactor = 1/(fH·fW) (mean); it is a
    // multiplier applied to the box sum, NOT a divisor.
    double normFactor = 1.0 / (static_cast<double>(fH) * static_cast<double>(fW));
    // NV-pair: NormalizationFactor.
    for (keel::usize i = nvStart; i + 1 < args.size(); i += 2) {
        if (!args[i].isChar() && !args[i].isString())
            throw Error("integralBoxFilter: name-value name must be a string",
                        0, 0, "integralBoxFilter", "", "numkit:integralBoxFilter:badNVName");
        const keel::String key = args[i].toString();
        if (key.as_ref().equals_ignore_ascii_case("normalizationfactor")) {
            normFactor = args[i + 1].toScalar();
        } else {
            throw Error(keel::format("integralBoxFilter: unknown name-value key '{}'", key),
                        0, 0, "integralBoxFilter", "", "numkit:integralBoxFilter:badNVKey");
        }
    }
    outs[0] = integralBoxFilter(args[0], fH, fW, normFactor, ctx.engine->allocator());
}

void integralBoxFilter3_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                            keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "integralBoxFilter3");
    // no max: name-value tail in the reference// Default: 3×3×3 box.
    int fH = 3, fW = 3, fP = 3;
    keel::usize nvStart = 1;
    if (args.size() >= 2 && !args[1].isEmpty()
        && !args[1].isChar() && !args[1].isString()) {
        const Value &fsArg = args[1];
        if (fsArg.numel() == 1) {
            fH = fW = fP = argcheck::expectedPositiveInteger(
            argcheck::expectedReal(fsArg, 2, "window", "integralBoxFilter3"), 2, "window", "integralBoxFilter3");
        } else if (fsArg.numel() == 3) {
            fH = static_cast<int>(fsArg.elemAsDouble(0));
            fW = static_cast<int>(fsArg.elemAsDouble(1));
            fP = static_cast<int>(fsArg.elemAsDouble(2));
        } else {
            throw Error("integralBoxFilter3: filterSize must be a scalar or 3-element vector",
                        0, 0, "integralBoxFilter3", "", "numkit:integralBoxFilter3:badSize");
        }
        nvStart = 2;
    }

    // MATLAB default NormalizationFactor = 1/prod(filterSize) (mean).
    double normFactor = 1.0 / (static_cast<double>(fH) * static_cast<double>(fW)
                               * static_cast<double>(fP));
    for (keel::usize i = nvStart; i + 1 < args.size(); i += 2) {
        if (!args[i].isChar() && !args[i].isString())
            throw Error("integralBoxFilter3: name-value name must be a string",
                        0, 0, "integralBoxFilter3", "", "numkit:integralBoxFilter3:badNVName");
        const keel::String key = args[i].toString();
        if (key.as_ref().equals_ignore_ascii_case("normalizationfactor")) {
            normFactor = args[i + 1].toScalar();
        } else {
            throw Error("integralBoxFilter3: unknown name-value key",
                        0, 0, "integralBoxFilter3", "", "numkit:integralBoxFilter3:badNVKey");
        }
    }
    outs[0] = integralBoxFilter3(args[0], fH, fW, fP, normFactor, ctx.engine->allocator());
}

void medfilt3_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "medfilt3");
    // no max: name-value tail in the reference
    int M = 3, N = 3, P = 3;
    if (args.size() >= 2 && !args[1].isEmpty()) {
        const Value &v = args[1];
        if (v.numel() == 1) {
            M = N = P = argcheck::expectedPositiveInteger(
                argcheck::expectedReal(v, 2, "window", "medfilt3"), 2, "window", "medfilt3");
        } else if (v.numel() >= 3) {
            M = static_cast<int>(v.elemAsDouble(0));
            N = static_cast<int>(v.elemAsDouble(1));
            P = static_cast<int>(v.elemAsDouble(2));
        }
    }
    outs[0] = medfilt3(args[0], M, N, P, ctx.engine->allocator());
}

void imgaussfilt3_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                      keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "imgaussfilt3");
    // no max: name-value tail in the reference
    double sigH = 0.5, sigW = 0.5, sigP = 0.5;
    if (args.size() >= 2 && !args[1].isEmpty()) {
        const Value &v = args[1];
        if (v.numel() == 1) {
            sigH = sigW = sigP = v.toScalar();
        } else if (v.numel() >= 3) {
            sigH = v.elemAsDouble(0);
            sigW = v.elemAsDouble(1);
            sigP = v.elemAsDouble(2);
        }
    }
    outs[0] = imgaussfilt3(args[0], sigH, sigW, sigP, ctx.engine->allocator());
}

void convmtx2_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 3, "convmtx2");
    int m = 0, n = 0;
    if (args.size() >= 3) {
        m = argcheck::expectedPositiveInteger(
            argcheck::expectedReal(args[1], 2, "m", "convmtx2"), 2, "m", "convmtx2");
        n = argcheck::expectedPositiveInteger(
            argcheck::expectedReal(args[2], 3, "n", "convmtx2"), 3, "n", "convmtx2");
    } else {
        const Value &v = args[1];
        if (v.numel() != 2)
            throw Error("convmtx2: 2nd arg must be a 2-element vector or pair (m, n)",
                        0, 0, "convmtx2", "", "numkit:convmtx2:size");
        m = static_cast<int>(v.elemAsDouble(0));
        n = static_cast<int>(v.elemAsDouble(1));
    }
    outs[0] = convmtx2(args[0], m, n, ctx.engine->allocator());
}

void imboxfilt3_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                    keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "imboxfilt3");
    // no max: name-value tail in the reference
    int fH = 3, fW = 3, fP = 3;
    if (args.size() >= 2 && !args[1].isEmpty()) {
        const Value &v = args[1];
        if (v.numel() == 1) {
            fH = fW = fP = (int)v.toScalar();
        } else if (v.numel() >= 3) {
            fH = (int)v.elemAsDouble(0);
            fW = (int)v.elemAsDouble(1);
            fP = (int)v.elemAsDouble(2);
        }
    }
    outs[0] = imboxfilt3(args[0], fH, fW, fP, ctx.engine->allocator());
}

void freqz2_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "freqz2");
    keel::usize M = 64, N = 64;
    if (args.size() >= 2 && !args[1].isEmpty()) {
        const Value &v = args[1];
        if (v.numel() == 1) {
            M = N = static_cast<keel::usize>(argcheck::expectedPositiveInteger(
            argcheck::expectedReal(v, 2, "n", "freqz2"), 2, "n", "freqz2"));
        } else if (v.numel() >= 2) {
            M = static_cast<keel::usize>(v.elemAsDouble(0));
            N = static_cast<keel::usize>(v.elemAsDouble(1));
        }
    }
    if (args.size() >= 3 && !args[2].isEmpty())
        N = static_cast<keel::usize>(argcheck::expectedPositiveInteger(
            argcheck::expectedReal(args[2], 3, "n", "freqz2"), 3, "n", "freqz2"));
    auto [H, f1, f2] = freqz2(args[0], M, N, ctx.engine->allocator());
    outs[0] = std::move(H);
    if (nargout > 1) outs[1] = std::move(f1);
    if (nargout > 2) outs[2] = std::move(f2);
}

void medfilt2_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 3, "medfilt2");
    int rows = 3, cols = 3;
    if (args.size() >= 2 && !args[1].isEmpty()) {
        const Value &v = args[1];
        if (v.numel() == 1) {
            rows = cols = (int)v.toScalar();
        } else if (v.numel() >= 2) {
            rows = (int)v.elemAsDouble(0);
            cols = (int)v.elemAsDouble(1);
        }
    }
    outs[0] = medfilt2(args[0], rows, cols, ctx.engine->allocator());
}

void im2col_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, 4, "im2col");
    int m = 0, n = 0;
    const Value &sz = args[1];
    if (sz.numel() == 1) {
        m = n = (int)sz.toScalar();
    } else if (sz.numel() >= 2) {
        m = (int)sz.elemAsDouble(0);
        n = (int)sz.elemAsDouble(1);
    } else {
        throw Error("im2col: block size must be scalar or 2-vector",
                    0, 0, "im2col", "", "numkit:im2col:size");
    }
    keel::String mode = keel::String::from("sliding");
    if (args.size() >= 3 && (args[2].isChar() || args[2].isString()))
        mode = args[2].toString();
    outs[0] = im2col(args[0], m, n, mode, ctx.engine->allocator());
}

void imbilatfilt_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                     keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "imbilatfilt");
    // no max: name-value tail in the reference// Default DegreeOfSmoothing depends on input class range — MATLAB
    // uses 0.01 · diff(getrangefromclass(I)). For our purposes:
    //   double / single / logical → 0.01    (range = 1)
    //   uint8                     → 6.50    (= 0.01·255²·1e-3 ish; matches MATLAB)
    //   uint16                    → tiny but absolute; we use 0.01·65535²·1e-3
    // Actually MATLAB stores it as variance of the range Gaussian
    // expressed in the same units as I. For unit-range double the
    // canonical value is 0.01; for integer classes we scale by the
    // range so the *relative* sensitivity matches.
    const Value &I = args[0];
    double dos_default;
    switch (I.type()) {
        case ValueType::UINT8:
            dos_default = 0.01 * 255.0 * 255.0; break;
        case ValueType::UINT16:
            dos_default = 0.01 * 65535.0 * 65535.0; break;
        case ValueType::INT16:
            dos_default = 0.01 * 65535.0 * 65535.0; break;
        default:
            dos_default = 0.01; break;
    }
    double dos    = (args.size() >= 2 && !args[1].isEmpty())
                    ? args[1].toScalar() : dos_default;
    double sigma  = (args.size() >= 3 && !args[2].isEmpty())
                    ? args[2].toScalar() : 1.0;
    outs[0] = imbilatfilt(I, dos, sigma, ctx.engine->allocator());
}

void col2im_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "col2im");
    // no max: variadic trailing block sizes by design
    int m, n, mm, nn;
    {
        const Value &b = args[1];
        if (b.numel() < 1)
            throw Error("col2im: empty block size",
                        0, 0, "col2im", "", "numkit:col2im:size");
        m = (int)b.elemAsDouble(0);
        n = (b.numel() >= 2) ? (int)b.elemAsDouble(1) : m;
    }
    {
        const Value &s = args[2];
        if (s.numel() < 1)
            throw Error("col2im: empty image size",
                        0, 0, "col2im", "", "numkit:col2im:size");
        mm = (int)s.elemAsDouble(0);
        nn = (s.numel() >= 2) ? (int)s.elemAsDouble(1) : mm;
    }
    keel::String mode = keel::String::from("sliding");
    if (args.size() >= 4 && (args[3].isChar() || args[3].isString()))
        mode = args[3].toString();
    outs[0] = col2im(args[0], m, n, mm, nn, mode, ctx.engine->allocator());
}

void imnoise_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "imnoise");
    keel::String mode = keel::String::from("gaussian");
    if (args.size() >= 2) {
        if (!(args[1].isChar() || args[1].isString()))
            throw Error("imnoise: mode must be a string",
                        0, 0, "imnoise", "", "numkit:imnoise:mode");
        mode = args[1].toString();
    }
    Value p1, p2;
    if (args.size() >= 3) p1 = args[2];
    if (args.size() >= 4) p2 = args[3];
    outs[0] = imnoise(ctx.engine->rng(), args[0], mode, p1, p2, ctx.engine->allocator());
}

void imsharpen_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                   keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 7, "imsharpen");
    double radius = 1.0, amount = 0.8, threshold = 0.0;
    // Accept either positional (I, radius, amount, threshold) or
    // name-value pairs ('Radius', r, 'Amount', a, 'Threshold', t).
    keel::usize i = 1;
    bool sawNV = false;
    while (i < args.size()) {
        if (args[i].isChar() || args[i].isString()) {
            sawNV = true;
            const keel::String nm = args[i].toString();
            if (i + 1 >= args.size())
                throw Error(keel::format("imsharpen: missing value for '{}'", nm),
                            0, 0, "imsharpen", "", "numkit:imsharpen:nv");
            const double v = args[i + 1].toScalar();
            if (nm == "Radius" || nm == "radius") radius = v;
            else if (nm == "Amount" || nm == "amount") amount = v;
            else if (nm == "Threshold" || nm == "threshold") threshold = v;
            else throw Error(keel::format("imsharpen: unknown option '{}'", nm),
                             0, 0, "imsharpen", "", "numkit:imsharpen:opt");
            i += 2;
        } else {
            if (sawNV)
                throw Error("imsharpen: positional after name-value",
                            0, 0, "imsharpen", "", "numkit:imsharpen:syntax");
            const double v = args[i].toScalar();
            if      (i == 1) radius    = v;
            else if (i == 2) amount    = v;
            else if (i == 3) threshold = v;
            ++i;
        }
    }
    outs[0] = imsharpen(args[0], radius, amount, threshold, ctx.engine->allocator());
}

void stdfilt_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "stdfilt");
    Value dom;
    if (args.size() >= 2 && !args[1].isEmpty()) dom = args[1];
    outs[0] = stdfilt(args[0], dom, ctx.engine->allocator());
}

void rangefilt_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                   keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "rangefilt");
    Value dom;
    if (args.size() >= 2 && !args[1].isEmpty()) dom = args[1];
    outs[0] = rangefilt(args[0], dom, ctx.engine->allocator());
}

void imsmooth_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "imsmooth");
    // no max: absent from the R2025b reference build
    auto *mr = ctx.engine->allocator();
    keel::String name = keel::String::from("Gaussian");
    double sigma = 0.5;
    // imsmooth(I, scalar) — Octave shorthand: scalar is sigma, name=Gaussian.
    if (args.size() >= 2 && !args[1].isEmpty()) {
        if (args[1].isChar() || args[1].isString()) {
            name = args[1].toString();
            if (args.size() >= 3 && !args[2].isEmpty())
                sigma = args[2].toScalar();
        } else if (args[1].numel() == 1) {
            sigma = args[1].toScalar();
        }
    }
    outs[0] = imsmooth(args[0], name, sigma, mr);
}

void entropyfilt_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                     keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 2, "entropyfilt");
    Value dom;
    if (args.size() >= 2 && !args[1].isEmpty()) dom = args[1];
    outs[0] = entropyfilt(args[0], dom, ctx.engine->allocator());
}

void ordfilt2_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, 5, "ordfilt2");
    auto *mr = ctx.engine->allocator();
    const int nth = argcheck::expectedPositiveInteger(
            argcheck::expectedReal(args[1], 2, "order", "ordfilt2"), 2, "order", "ordfilt2");
    Value domain = args[2];
    if (domain.numel() == 1) {
        // Scalar domain → square true(N).
        const int n = argcheck::expectedPositiveInteger(
            argcheck::expectedReal(domain, 3, "domain", "ordfilt2"), 3, "domain", "ordfilt2");
        domain = Value::matrix((keel::usize)n, (keel::usize)n, ValueType::LOGICAL, mr);
        keel::u8 *dd = domain.logicalDataMut();
        for (keel::usize i = 0; i < (keel::usize)n * (keel::usize)n; ++i) dd[i] = 1;
    }
    Value S;
    PadMode pad = PadMode::Constant;
    double pad_value = 0.0;
    for (keel::usize i = 3; i < args.size(); ++i) {
        const Value &a = args[i];
        if (a.isEmpty()) continue;
        if (a.isChar() || a.isString()) {
            const keel::String s = a.toString();
            if      (s == "replicate") pad = PadMode::Replicate;
            else if (s == "symmetric") pad = PadMode::Symmetric;
            else if (s == "circular")  pad = PadMode::Circular;
            else throw Error("ordfilt2: unknown padding mode",
                              0, 0, "ordfilt2", "", "numkit:ordfilt2:pad");
        } else if (a.numel() == 1) {
            pad = PadMode::Constant;
            pad_value = a.toScalar();
        } else if (a.dims().rows() == domain.dims().rows() &&
                   a.dims().cols() == domain.dims().cols()) {
            S = a;
        } else {
            throw Error("ordfilt2: unrecognized argument shape",
                        0, 0, "ordfilt2", "", "numkit:ordfilt2:arg");
        }
    }
    outs[0] = ordfilt2(args[0], nth, domain, S, pad, pad_value, mr);
}

void wiener2_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 4, "wiener2");
    keel::usize nh = 3, nw = 3;
    double noise = std::nan("");
    if (args.size() >= 2 && !args[1].isEmpty()) {
        if (args[1].numel() == 1) {
            // Single scalar = noise (Octave convention when no nhood).
            noise = args[1].toScalar();
        } else if (args[1].numel() >= 2) {
            nh = static_cast<keel::usize>(args[1].elemAsDouble(0));
            nw = static_cast<keel::usize>(args[1].elemAsDouble(1));
        }
    }
    if (args.size() >= 3 && !args[2].isEmpty())
        noise = args[2].toScalar();
    auto [denoised, n] =
        wiener2(args[0], nh, nw, noise, ctx.engine->allocator());
    outs[0] = std::move(denoised);
    if (nargout > 1) outs[1] = std::move(n);
}

} // namespace detail

namespace detail {

void roifilt2_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "roifilt2");
    // no max: name-value tail in the reference
    auto *mr = ctx.engine->allocator();
    // Function-handle form: J = roifilt2(I, BW, fun).
    if (args[2].isFuncHandle() || args[2].isChar() || args[2].isString()) {
        const Value &I = args[0];
        const Value &BW = args[1];
        if (I.dims().ndim() > 2)
            throw Error("roifilt2: I must be a 2-D image",
                        0, 0, "roifilt2", "", "numkit:roifilt2:imageMustBe2D");
        Value filtRes = ctx.engine->callFunctionHandle(
            args[2], keel::ArrayRef<Value>(&I, 1));
        outs[0] = roifilt2_combine(I, filtRes, BW, mr);
        return;
    }
    // Filter form: J = roifilt2(h, I, BW).
    outs[0] = roifilt2(args[0], args[1], args[2], mr);
}

void nlfilter_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                  keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "nlfilter");
    // no max: variadic passthrough to the callback
    auto *mr = ctx.engine->allocator();

    bool indexed = false;
    keel::usize k = 1;
    if (args[1].isChar() || args[1].isString()) {
        keel::String s = args[1].toString();
        if (!s.as_ref().equals_ignore_ascii_case("indexed"))
            throw Error(keel::format("nlfilter: unknown literal '{}' (expected 'indexed')", s),
                        0, 0, "nlfilter", "", "numkit:nlfilter:badLiteral");
        indexed = true;
        k = 2;
    }
    if (k + 1 >= args.size())
        throw Error("nlfilter: requires (A, [m n], fun) "
                    "or (A, 'indexed', [m n], fun)",
                    0, 0, "nlfilter", "", "numkit:nlfilter:nargin");
    const Value &nh = args[k];
    const Value &fn = args[k + 1];
    if (nh.numel() != 2)
        throw Error("nlfilter: neighbourhood must be a 2-element vector",
                    0, 0, "nlfilter", "", "numkit:nlfilter:nhood");
    const keel::usize m = static_cast<keel::usize>(nh.elemAsDouble(0));
    const keel::usize n = static_cast<keel::usize>(nh.elemAsDouble(1));
    if (!fn.isFuncHandle())
        throw Error("nlfilter: 3rd argument must be a function handle",
                    0, 0, "nlfilter", "", "numkit:nlfilter:fun");
    auto cb = [&ctx, &fn](keel::ArrayRef<Value> a, keel::MutArrayRef<Value> o,
                          keel::Allocator * /*mr*/) {
        auto r = ctx.engine->callFunctionHandleMulti(fn, a, o.size());
        for (keel::usize i = 0; i < o.size() && i < r.size(); ++i)
            o[i] = std::move(r[i]);
    };
    outs[0] = nlfilter(args[0], m, n, cb, indexed, mr);
}

void colfilt_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                 keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 4, "colfilt");
    // no max: variadic passthrough to the callback
    auto *mr = ctx.engine->allocator();

    bool indexed = false;
    keel::usize k = 1;
    if (args[1].isChar() || args[1].isString()) {
        keel::String s = args[1].toString();
        if (!s.as_ref().equals_ignore_ascii_case("indexed"))
            throw Error(keel::format("colfilt: unknown literal '{}' (expected 'indexed')", s),
                        0, 0, "colfilt", "", "numkit:colfilt:badLiteral");
        indexed = true;
        k = 2;
    }
    if (k + 2 >= args.size())
        throw Error("colfilt: requires (A, [m n], block_type, fun)",
                    0, 0, "colfilt", "", "numkit:colfilt:nargin");
    const Value &nh = args[k];
    if (nh.numel() != 2)
        throw Error("colfilt: neighbourhood must be a 2-element vector",
                    0, 0, "colfilt", "", "numkit:colfilt:nhood");
    const keel::usize m = static_cast<keel::usize>(nh.elemAsDouble(0));
    const keel::usize n = static_cast<keel::usize>(nh.elemAsDouble(1));

    // Detect optional [mblock nblock] — present iff arg[k+1] is a
    // 2-element numeric vector AND arg[k+2] is a string AND arg[k+3]
    // exists (the function handle).
    keel::usize bi = k + 1;
    if (!args[bi].isChar() && !args[bi].isString()
        && args[bi].numel() == 2
        && (bi + 2) < args.size()
        && (args[bi + 1].isChar() || args[bi + 1].isString())) {
        // mblock / nblock is purely a memory optimisation per MATLAB
        // docs — ignore it and proceed.
        bi = bi + 1;
    }
    if (bi + 1 >= args.size())
        throw Error("colfilt: missing block_type and/or fun argument",
                    0, 0, "colfilt", "", "numkit:colfilt:nargin");
    if (!args[bi].isChar() && !args[bi].isString())
        throw Error("colfilt: block_type must be 'sliding' or 'distinct'",
                    0, 0, "colfilt", "", "numkit:colfilt:blockType");
    const keel::String kind = args[bi].toString();
    const Value &fn = args[bi + 1];
    if (!fn.isFuncHandle())
        throw Error("colfilt: fun must be a function handle",
                    0, 0, "colfilt", "", "numkit:colfilt:fun");
    auto cb = [&ctx, &fn](keel::ArrayRef<Value> a, keel::MutArrayRef<Value> o,
                          keel::Allocator * /*mr*/) {
        auto r = ctx.engine->callFunctionHandleMulti(fn, a, o.size());
        for (keel::usize i = 0; i < o.size() && i < r.size(); ++i)
            o[i] = std::move(r[i]);
    };
    outs[0] = colfilt(args[0], m, n, kind, cb, indexed, mr);
}

void imnlmfilt_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                   keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 7, "imnlmfilt");
    auto *mr = ctx.engine->allocator();
    auto is_string = [](const Value &v) { return v.isChar() || v.isString(); };

    double dos = -1.0;
    int swsize = 21;
    int cwsize = 5;
    keel::usize i = 1;
    while (i + 1 < args.size()) {
        if (!is_string(args[i]))
            throw Error("imnlmfilt: expected NV-pair name",
                        0, 0, "imnlmfilt", "", "numkit:imnlmfilt:badNv");
        keel::String name = args[i].toString();
        keel::StringRef nref = name.as_ref();
        if (nref.equals_ignore_ascii_case("degreeofsmoothing")) {
            dos = args[i + 1].toScalar();
            if (!(dos > 0))
                throw Error("imnlmfilt: DegreeOfSmoothing must be positive",
                            0, 0, "imnlmfilt", "", "numkit:imnlmfilt:dos");
        } else if (nref.equals_ignore_ascii_case("searchwindowsize")) {
            swsize = argcheck::expectedPositiveInteger(
                argcheck::expectedReal(args[i + 1], int(i) + 2, "SearchWindowSize", "imnlmfilt"),
                int(i) + 2, "SearchWindowSize", "imnlmfilt");
        } else if (nref.equals_ignore_ascii_case("comparisonwindowsize")) {
            cwsize = argcheck::expectedPositiveInteger(
                argcheck::expectedReal(args[i + 1], int(i) + 2, "ComparisonWindowSize", "imnlmfilt"),
                int(i) + 2, "ComparisonWindowSize", "imnlmfilt");
        } else {
            throw Error(keel::format("imnlmfilt: unknown option '{}'", name),
                        0, 0, "imnlmfilt", "",
                        "numkit:imnlmfilt:unknownNv");
        }
        i += 2;
    }
    Value J;
    double est;
    imnlmfilt(args[0], dos, swsize, cwsize, J, est, mr);
    outs[0] = std::move(J);
    if (nargout >= 2) outs[1] = Value::scalar(est, mr);
}

void imgaborfilt_reg(keel::ArrayRef<Value> args, keel::usize nargout,
                     keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 3, "imgaborfilt");
    // no max: name-value tail in the reference
    auto *mr = ctx.engine->allocator();
    auto is_string = [](const Value &v) { return v.isChar() || v.isString(); };

    const Value &A = args[0];
    const double wavelength  = args[1].toScalar();
    const double orientation = args[2].toScalar();
    double sfb = 1.0;
    double aspect = 0.5;

    keel::usize i = 3;
    while (i + 1 < args.size()) {
        if (!is_string(args[i]))
            throw Error("imgaborfilt: expected NV-pair name",
                        0, 0, "imgaborfilt", "", "numkit:imgaborfilt:badNv");
        keel::String name = args[i].toString();
        keel::StringRef nref = name.as_ref();
        if (nref.equals_ignore_ascii_case("spatialfrequencybandwidth"))
            sfb = args[i + 1].toScalar();
        else if (nref.equals_ignore_ascii_case("spatialaspectratio"))
            aspect = args[i + 1].toScalar();
        else
            throw Error(keel::format("imgaborfilt: unknown option '{}'", name),
                        0, 0, "imgaborfilt", "",
                        "numkit:imgaborfilt:unknownNv");
        i += 2;
    }
    Value mag, phase;
    imgaborfilt(A, wavelength, orientation, sfb, aspect, mag, phase, mr);
    outs[0] = std::move(mag);
    if (nargout >= 2) outs[1] = std::move(phase);
}

void imdiffusefilt_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                       keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "imdiffusefilt");
    // no max: inputParser name-value tail
    auto *mr = ctx.engine->allocator();
    auto is_string = [](const Value &v) { return v.isChar() || v.isString(); };

    Value thresh;
    keel::usize N = 0;
    keel::String connectivity = keel::String::from("maximal");
    keel::String conduction = keel::String::from("exponential");

    keel::usize i = 1;
    while (i + 1 < args.size()) {
        if (!is_string(args[i]))
            throw Error("imdiffusefilt: expected NV-pair name",
                        0, 0, "imdiffusefilt", "",
                        "numkit:imdiffusefilt:badNv");
        keel::String name = args[i].toString();
        keel::StringRef nref = name.as_ref();
        if (nref.equals_ignore_ascii_case("gradientthreshold")) {
            thresh = args[i + 1];
        } else if (nref.equals_ignore_ascii_case("numberofiterations")) {
            const double v = args[i + 1].toScalar();
            if (!(v > 0) || v != std::floor(v))
                throw Error("imdiffusefilt: NumberOfIterations must be a "
                            "positive integer",
                            0, 0, "imdiffusefilt", "",
                            "numkit:imdiffusefilt:n");
            N = static_cast<keel::usize>(v);
        } else if (nref.equals_ignore_ascii_case("connectivity")) {
            connectivity = to_lower_ascii(args[i + 1].toString());
        } else if (nref.equals_ignore_ascii_case("conductionmethod")) {
            conduction = to_lower_ascii(args[i + 1].toString());
        } else {
            throw Error(keel::format("imdiffusefilt: unknown option '{}'", name),
                        0, 0, "imdiffusefilt", "",
                        "numkit:imdiffusefilt:unknownNv");
        }
        i += 2;
    }
    outs[0] = imdiffusefilt(args[0], thresh, N, connectivity, conduction, mr);
}

void imguidedfilter_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/,
                        keel::MutArrayRef<Value> outs, CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, 6, "imguidedfilter");
    auto *mr = ctx.engine->allocator();
    auto is_string = [](const Value &v) { return v.isChar() || v.isString(); };

    Value A = args[0];
    Value G;  // empty → self-guide
    int nhood = 5;
    double eps = -1.0;  // sentinel → class-based default

    keel::usize i = 1;
    if (i < args.size() && !is_string(args[i])) {
        G = args[i];
        ++i;
    }
    while (i + 1 < args.size()) {
        if (!is_string(args[i]))
            throw Error("imguidedfilter: expected NV-pair name",
                        0, 0, "imguidedfilter", "",
                        "numkit:imguidedfilter:badNv");
        keel::String name = args[i].toString();
        keel::StringRef nref = name.as_ref();
        if (nref.equals_ignore_ascii_case("neighborhoodsize")) {
            const Value &v = args[i + 1];
            if (v.numel() == 1) {
                nhood = argcheck::expectedPositiveInteger(
                    argcheck::expectedReal(v, int(i) + 2, "Neighborhood", "imguidedfilter"),
                    int(i) + 2, "Neighborhood", "imguidedfilter");
            } else if (v.numel() == 2) {
                const int n0 = static_cast<int>(v.elemAsDouble(0));
                const int n1 = static_cast<int>(v.elemAsDouble(1));
                if (n0 != n1)
                    throw Error("imguidedfilter: non-square NeighborhoodSize "
                                "not yet supported",
                                0, 0, "imguidedfilter", "",
                                "numkit:imguidedfilter:nhoodNonsq");
                nhood = n0;
            } else {
                throw Error("imguidedfilter: NeighborhoodSize must be a "
                            "scalar or 2-element vector",
                            0, 0, "imguidedfilter", "",
                            "numkit:imguidedfilter:nhoodSize");
            }
        } else if (nref.equals_ignore_ascii_case("degreeofsmoothing")) {
            eps = args[i + 1].toScalar();
            if (!(eps > 0) || !std::isfinite(eps))
                throw Error("imguidedfilter: DegreeOfSmoothing must be a "
                            "positive finite scalar",
                            0, 0, "imguidedfilter", "",
                            "numkit:imguidedfilter:dos");
        } else {
            throw Error(keel::format("imguidedfilter: unknown option '{}'", name),
                        0, 0, "imguidedfilter", "",
                        "numkit:imguidedfilter:unknownNv");
        }
        i += 2;
    }
    outs[0] = imguidedfilter(A, G, nhood, eps, mr);
}

} // namespace detail

} // namespace numkit::image
