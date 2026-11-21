// toolboxes/image/src/contrast/contrast.cpp

#include <keel/core.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/arena.h>
#include <keel/array.h>
#include <keel/math.h>
#include <keel/image.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/image/contrast/contrast.hpp>
#include <numkit/image/filter/filter.hpp>
#include <numkit/image/type_convert/type_convert.hpp>
#include <numkit/image/internal/type_gates.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/value_type.hpp>

#include <algorithm>
#include <tuple>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace numkit::image {

namespace {

// Convert any image element to a unit-range double in [0, 1].
inline double element_to_unit(const Value &x, keel::usize i) {
    const double v = x.elemAsDouble(i);
    switch (x.type()) {
        case ValueType::DOUBLE:
        case ValueType::SINGLE:  return v;
        case ValueType::UINT8:   return v / 255.0;
        case ValueType::UINT16:  return v / 65535.0;
        case ValueType::INT16:   return (v + 32768.0) / 65535.0;
        case ValueType::LOGICAL: return v != 0.0 ? 1.0 : 0.0;
        default:                 return v;
    }
}

inline int default_nbins(const Value &I) {
    switch (I.type()) {
        case ValueType::UINT8:  return 256;
        case ValueType::UINT16: return 65536;
        default:                return 64;
    }
}

inline void store_classed(Value &out, keel::usize i, double v, ValueType t) {
    switch (t) {
        case ValueType::DOUBLE: out.doubleDataMut()[i] = v; break;
        case ValueType::SINGLE: out.singleDataMut()[i] = (float)v; break;
        case ValueType::UINT8: {
            double w = std::round(v * 255.0);
            if (w < 0) w = 0; if (w > 255) w = 255;
            out.uint8DataMut()[i] = (keel::u8)w; break;
        }
        case ValueType::UINT16: {
            double w = std::round(v * 65535.0);
            if (w < 0) w = 0; if (w > 65535) w = 65535;
            out.uint16DataMut()[i] = (keel::u16)w; break;
        }
        case ValueType::INT16: {
            double w = std::round(v * 65535.0) - 32768.0;
            if (w < -32768) w = -32768; if (w > 32767) w = 32767;
            out.int16DataMut()[i] = (keel::i16)w; break;
        }
        default:
            throw Error("contrast: unsupported class", 0, 0, "contrast", "",
                        "numkit:contrast:badtype");
    }
}

void stretchlim_argcheck(const Value &I)
{
    internal::requireImageClass(I, "stretchlim", "I", 1,
                                internal::ImgClasses::FloatInt);
    internal::requireNonemptyImage(I, "stretchlim", "I", 1);
}

void imadjust_argcheck(const Value &I)
{
    internal::requireImageClass(I, "imadjust", "I", 1,
                                internal::ImgClasses::FloatInt);
}

void adapthisteq_argcheck(const Value &I, const AdaptHistEqOptions &opts)
{
    if (opts.numTilesR < 2 || opts.numTilesC < 2)
        throw Error("adapthisteq: NumTiles must be >= 2 in each dimension",
                    0, 0, "adapthisteq", "", "numkit:adapthisteq:badTiles");
    if (!(opts.clipLimit >= 0.0 && opts.clipLimit <= 1.0))
        throw Error("adapthisteq: ClipLimit must be in [0, 1]",
                    0, 0, "adapthisteq", "", "numkit:adapthisteq:badClip");
    if (opts.nBins < 2)
        throw Error("adapthisteq: NBins must be >= 2",
                    0, 0, "adapthisteq", "", "numkit:adapthisteq:badBins");
    if (opts.distribution != "uniform" && opts.distribution != "rayleigh" &&
        opts.distribution != "exponential")
        throw Error("adapthisteq: Distribution must be 'uniform', "
                    "'rayleigh' or 'exponential'",
                    0, 0, "adapthisteq", "", "numkit:adapthisteq:badDistribution");
    if (opts.range != "full" && opts.range != "original")
        throw Error("adapthisteq: Range must be 'full' or 'original'",
                    0, 0, "adapthisteq", "", "numkit:adapthisteq:badRange");

    const Dims &d = I.dims();
    if (d.ndim() > 2 || (d.ndim() == 3 && d.pages() > 1))
        throw Error("adapthisteq: input must be a 2-D greyscale image",
                    0, 0, "adapthisteq", "", "numkit:adapthisteq:unsupportedShape");
    if (I.isComplex() || I.isCell() || I.isStruct() ||
        I.isString() || I.isFuncHandle())
        throw Error("adapthisteq: input must be a real numeric image",
                    0, 0, "adapthisteq", "", "numkit:adapthisteq:unsupportedShape");
    if (d.rows() == 0 || d.cols() == 0)
        throw Error("adapthisteq: input image is empty",
                    0, 0, "adapthisteq", "", "numkit:adapthisteq:unsupportedShape");
}

void graythresh_argcheck(const Value &I)
{
    internal::requireImageClass(I, "graythresh", "I", 1,
                                internal::ImgClasses::FloatInt);
}

void multithresh_argcheck(int N)
{
    if (N > 5)
        throw Error("multithresh: N > 5 not supported (exhaustive search would be too slow)",
                    0, 0, "multithresh", "", "numkit:multithresh:tooMany");
}

void imbinarize_argcheck(const Value &I)
{
    internal::requireImageClass(I, "imbinarize", "I", 1,
                                internal::ImgClasses::FloatInt);
}

void imbinarize_argcheck(const Value &I, const Value &T)
{
    internal::requireImageClass(I, "imbinarize", "I", 1,
                                internal::ImgClasses::FloatInt);
    if (T.numel() != I.numel())
        throw Error("imbinarize: per-pixel T must have the same number "
                    "of elements as I",
                    0, 0, "imbinarize", "", "numkit:imbinarize:Tshape");
}

void imquantize_argcheck(const Value &I, const Value &levels)
{
    internal::requireImageClass(I, "imquantize", "I", 1,
                                internal::ImgClasses::FloatInt);
    internal::requireImageClass(levels, "imquantize", "levels", 2,
                                internal::ImgClasses::AnyNumericReal);
}

void adaptthresh_argcheck(const Value &I, double sensitivity,
                          keel::StringRef statistic)
{
    internal::requireImageClass(I, "adaptthresh", "I", 1,
                                internal::ImgClasses::FloatInt);
    if (!(sensitivity >= 0.0 && sensitivity <= 1.0))
        throw Error("adaptthresh: sensitivity must be in [0, 1]",
                    0, 0, "adaptthresh", "", "numkit:adaptthresh:sens");
    const keel::StringRef s = statistic.empty() ? keel::StringRef("mean")
                                                : statistic;
    if (s != "mean" && s != "Mean" && s != "MEAN" &&
        s != "gaussian" && s != "Gaussian" && s != "GAUSSIAN")
        throw Error("adaptthresh: statistic must be 'mean' or 'gaussian'",
                    0, 0, "adaptthresh", "", "numkit:adaptthresh:stat");
}

void imflatfield_argcheck(const Value &I, const Value &mask)
{
    const auto &d = I.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    if (mask.numel() > 0 && (mask.dims().rows() != H || mask.dims().cols() != W))
        throw Error("imflatfield: mask must match spatial dims of I",
                    0, 0, "imflatfield", "", "numkit:imflatfield:masksize");
}

void wcodemat_argcheck(int nb, keel::StringRef opt)
{
    if (nb < 1)
        throw Error("wcodemat: NB must be a positive integer",
                    0, 0, "wcodemat", "", "numkit:wcodemat:nb");
    keel::String lo;
    lo.reserve(opt.size());
    for (char c : opt) lo.push_back(keel::to_ascii_lower(c));
    if (lo.empty()) lo.assign("mat");
    if (lo != "mat" && lo != "row" && lo != "col")
        throw Error("wcodemat: opt must be 'mat', 'row', or 'col'",
                    0, 0, "wcodemat", "", "numkit:wcodemat:opt");
}

void grayslice_argcheck(int N)
{
    if (N < 1)
        throw Error("grayslice: N must be a positive integer",
                    0, 0, "grayslice", "", "numkit:grayslice:n");
}

} // anonymous

std::tuple<Value, Value>
imhist(const Value &I, int n, keel::Allocator *mr)
{
    if (n <= 0) n = default_nbins(I);
    keel::Array<keel::i64> counts(n, 0);
    const keel::usize N = I.numel();
    for (keel::usize i = 0; i < N; ++i) {
        const double u = element_to_unit(I, i);
        // Map u (∈ [0, 1]) to bin index 0..n-1 with edges spaced 1/(n-1)
        // so bin centres are 0, 1/(n-1), ..., 1 (matches MATLAB).
        if (std::isnan(u)) continue;
        int bin = (int)std::round(u * (n - 1));
        if (bin < 0) bin = 0;
        if (bin >= n) bin = n - 1;
        ++counts[bin];
    }
    Value c = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    Value x = Value::matrix(n, 1, ValueType::DOUBLE, mr);
    double *cd = c.doubleDataMut();
    double *xd = x.doubleDataMut();
    // Bin locations x span the input CLASS's display range, not [0,1]:
    // double/single/logical -> [0,1]; uint8 -> [0,255]; uint16 ->
    // [0,65535]; int16 -> [-32768,32767]. (Counts above are computed on
    // the normalized value, so only the x labels depend on class.)
    double lo = 0.0, hi = 1.0;
    switch (I.type()) {
        case ValueType::UINT8:  hi = 255.0; break;
        case ValueType::UINT16: hi = 65535.0; break;
        case ValueType::INT16:  lo = -32768.0; hi = 32767.0; break;
        default: break;  // DOUBLE / SINGLE / LOGICAL / other -> [0,1]
    }
    const double step = (n > 1) ? (hi - lo) / double(n - 1) : 0.0;
    for (int i = 0; i < n; ++i) {
        cd[i] = (double)counts[i];
        xd[i] = lo + i * step;
    }
    return std::make_tuple(std::move(c), std::move(x));
}

Value stretchlim(const Value &I, double low_tol, double high_tol, keel::Allocator *mr)
{
    stretchlim_argcheck(I);
    internal::requireImageClass(I, "stretchlim", "I", 1,
                                internal::ImgClasses::FloatInt);
    internal::requireNonemptyImage(I, "stretchlim", "I", 1);
    if (low_tol < 0.0)  low_tol  = 0.01;
    if (high_tol > 1.0 || high_tol <= low_tol) high_tol = 0.99;

    // Separate per-channel for H×W×3 RGB; otherwise single column.
    const auto &d = I.dims();
    const bool is_rgb = d.is3D() && d.pages() == 3;
    const keel::usize plane = d.is3D() ? d.rows() * d.cols() : I.numel();
    const int channels = is_rgb ? 3 : 1;

    Value out = Value::matrix(2, channels, ValueType::DOUBLE, mr);
    double *od = out.doubleDataMut();

    // MATLAB's stretchlim uses a histogram with strict-greater cumulative
    // thresholds, NOT sample-percentile. The bin count is class-dependent:
    // 256 bins for uint8, 65536 for uint16 / int16 / single / double. For
    // uint8 the bin index is the value directly; for the wider classes the
    // input is normalized to [0,1] and rebinned. (Previously a fixed 256
    // bins gave coarsely-quantized limits on double images — e.g.
    // stretchlim([0.1 0.2 0.9 0.95]) returned [26/255 242/255] instead of
    // [6554/65535 62258/65535].)
    const int NBINS = (I.type() == ValueType::UINT8) ? 256 : 65536;
    keel::Array<keel::usize> hist(NBINS, 0);
    keel::Array<keel::usize> cumhist(NBINS, 0);
    const double N = static_cast<double>(plane);

    for (int ch = 0; ch < channels; ++ch) {
        std::fill(hist.begin(), hist.end(), 0);
        for (keel::usize i = 0; i < plane; ++i) {
            const double u = element_to_unit(I, ch * plane + i);
            int bin = static_cast<int>(std::lround(u * (NBINS - 1)));
            if (bin < 0) bin = 0;
            if (bin >= NBINS) bin = NBINS - 1;
            ++hist[bin];
        }
        cumhist[0] = hist[0];
        for (int k = 1; k < NBINS; ++k) cumhist[k] = cumhist[k - 1] + hist[k];

        // low: smallest bin k where cumhist[k] > low_tol * N (strict).
        int low = NBINS - 1;
        for (int k = 0; k < NBINS; ++k) {
            if (static_cast<double>(cumhist[k]) > low_tol * N) { low = k; break; }
        }
        // high: smallest bin k where cumhist[k] >= high_tol * N.
        int high = NBINS - 1;
        for (int k = 0; k < NBINS; ++k) {
            if (static_cast<double>(cumhist[k]) >= high_tol * N) { high = k; break; }
        }
        // Guard: if histogram is degenerate (all in one bin), spread by 1.
        if (high <= low) high = std::min(NBINS - 1, low + 1);

        od[static_cast<keel::usize>(ch) * 2 + 0] =
            static_cast<double>(low)  / static_cast<double>(NBINS - 1);
        od[static_cast<keel::usize>(ch) * 2 + 1] =
            static_cast<double>(high) / static_cast<double>(NBINS - 1);
    }
    return out;
}

Value imadjust(const Value &I, double low_in, double high_in, double low_out, double high_out, double gamma, keel::Allocator *mr)
{
    imadjust_argcheck(I);
    internal::requireImageClass(I, "imadjust", "I", 1,
                                internal::ImgClasses::FloatInt);
    // Auto-fill missing endpoints via stretchlim defaults.
    if (std::isnan(low_in) || std::isnan(high_in)) {
        Value lim = stretchlim(I, 0.01, 0.99, mr);
        if (std::isnan(low_in))  low_in  = lim.elemAsDouble(0);
        if (std::isnan(high_in)) high_in = lim.elemAsDouble(1);
    }
    if (std::isnan(low_out))  low_out  = 0.0;
    if (std::isnan(high_out)) high_out = 1.0;
    if (std::isnan(gamma) || gamma <= 0.0) gamma = 1.0;

    const keel::usize N = I.numel();
    Value out;
    const auto &d = I.dims();
    if (I.isScalar()) out = Value::matrix(1, 1, I.type(), mr);
    else if (d.is3D())out = Value::matrix3d(d.rows(), d.cols(), d.pages(), I.type(), mr);
    else              out = Value::matrix(d.rows(), d.cols(), I.type(), mr);

    const double range = high_in - low_in;
    const double inv_range = (range != 0.0) ? 1.0 / range : 0.0;
    const double out_span = high_out - low_out;

    for (keel::usize i = 0; i < N; ++i) {
        const double u = element_to_unit(I, i);
        double t = (u - low_in) * inv_range;
        if (t < 0.0) t = 0.0;
        if (t > 1.0) t = 1.0;
        const double w = std::pow(t, gamma) * out_span + low_out;
        store_classed(out, i, w, I.type());
    }
    return out;
}

Value histeq(const Value &I, int n, keel::Allocator *mr)
{
    if (n <= 0) n = 64;
    auto [counts_v, bins_v] = imhist(I, n, mr);
    const double *counts = counts_v.doubleData();
    // Cumulative distribution.
    keel::Array<double> cdf(n);
    double total = 0.0;
    for (int i = 0; i < n; ++i) total += counts[i];
    if (total <= 0.0) return I;
    double acc = 0.0;
    for (int i = 0; i < n; ++i) {
        acc += counts[i];
        cdf[i] = acc / total;
    }

    const keel::usize N = I.numel();
    Value out;
    const auto &d = I.dims();
    if (I.isScalar()) out = Value::matrix(1, 1, I.type(), mr);
    else if (d.is3D())out = Value::matrix3d(d.rows(), d.cols(), d.pages(), I.type(), mr);
    else              out = Value::matrix(d.rows(), d.cols(), I.type(), mr);

    for (keel::usize i = 0; i < N; ++i) {
        const double u = element_to_unit(I, i);
        int bin = (int)std::round(u * (n - 1));
        if (bin < 0) bin = 0;
        if (bin >= n) bin = n - 1;
        store_classed(out, i, cdf[bin], I.type());
    }
    return out;
}

// ════════════════════════════════════════════════════════════════════
// adapthisteq — Contrast Limited Adaptive Histogram Equalisation
// ════════════════════════════════════════════════════════════════════
// Clean-room implementation of CLAHE.
// Public references:
//   * K. Zuiderveld, "Contrast Limited Adaptive Histogram
//     Equalization", in Graphics Gems IV (P. S. Heckbert, ed.),
//     Academic Press, 1994, pp. 474-485 — tile-based CLAHE with
//     bilinear interpolation between per-tile mapping functions.
//   * S. M. Pizer et al., "Contrast-Limited Adaptive Histogram
//     Equalization: Speed and Effectiveness", Proc. 1st Conf. on
//     Visualization in Biomedical Computing, IEEE, 1990 (UNC TR
//     90-035) — the contrast-limiting step (clip + redistribute).
//   * S. M. Pizer et al., "Adaptive Histogram Equalization and Its
//     Variations", Computer Vision, Graphics, and Image Processing
//     39:355-368, 1987 — the AHE family and non-uniform distributions.
// The image is split into numTilesR x numTilesC tiles; each tile gets a
// clipped+redistributed histogram whose CDF, shaped by the target
// Distribution (uniform/rayleigh/exponential), is the tile mapping LUT.
// Output pixels bilinearly interpolate the four nearest tile LUTs.
// Full MATLAB argument set: NumTiles, ClipLimit, NBins, Range,
// Distribution, Alpha. 2-D greyscale input only (as MATLAB).

namespace {

// ---- intensity range of a nk element type --------------------
// Returns the [lo, hi] span MATLAB associates with the image class.
// Floating-point images conventionally live in [0, 1]; integer
// images span the full representable range of the integer type;
// logical images span [0, 1].
static void classRange(ValueType t, double &lo, double &hi)
{
    switch (t) {
        case ValueType::DOUBLE:
        case ValueType::SINGLE:  lo = 0.0;                       hi = 1.0;                       break;
        case ValueType::LOGICAL: lo = 0.0;                       hi = 1.0;                       break;
        case ValueType::CHAR:    lo = 0.0;                       hi = 65535.0;                   break;
        case ValueType::INT8:    lo = -128.0;                    hi = 127.0;                     break;
        case ValueType::INT16:   lo = -32768.0;                  hi = 32767.0;                   break;
        case ValueType::INT32:   lo = -2147483648.0;             hi = 2147483647.0;              break;
        case ValueType::INT64:   lo = -9223372036854775808.0;    hi = 9223372036854775807.0;     break;
        case ValueType::UINT8:   lo = 0.0;                       hi = 255.0;                     break;
        case ValueType::UINT16:  lo = 0.0;                       hi = 65535.0;                   break;
        case ValueType::UINT32:  lo = 0.0;                       hi = 4294967295.0;              break;
        case ValueType::UINT64:  lo = 0.0;                       hi = 18446744073709551615.0;    break;
        default:                 lo = 0.0;                       hi = 1.0;                       break;
    }
}

// ---- per-tile clipped+redistributed histogram --------------------
// Faithful port of MATLAB R2025b adapthisteq.m `clipHistogram`.
// hist[]    : raw bin counts (length nBins), modified in place.
// clipLimit : the clip COUNT (integer-valued; see caller).
// Two-tier clip (high bins → clipLimit, low bins raised by avgBinIncr),
// then the remainder is redistributed one pixel at a time stepping
// EVENLY across all bins — never front-loaded (front-loading inflated
// the low-end CDF and made the output systematically too bright).
static void clipHistogram(double *hist, int nBins, double clipLimit)
{
    double totalExcess = 0.0;
    for (int b = 0; b < nBins; ++b)
        if (hist[b] > clipLimit) totalExcess += hist[b] - clipLimit;

    const double avgBinIncr = std::floor(totalExcess / static_cast<double>(nBins));
    const double upperLimit = clipLimit - avgBinIncr;
    for (int k = 0; k < nBins; ++k) {
        if (hist[k] > clipLimit) {
            hist[k] = clipLimit;
        } else if (hist[k] > upperLimit) {   // high bin count
            totalExcess -= (clipLimit - hist[k]);
            hist[k] = clipLimit;
        } else {
            totalExcess -= avgBinIncr;
            hist[k] += avgBinIncr;
        }
    }

    // Redistribute the remaining excess, spreading it evenly.
    int k = 0;
    while (totalExcess > 0.0) {
        int stepSize = static_cast<int>(
            std::floor(static_cast<double>(nBins) / totalExcess));
        if (stepSize < 1) stepSize = 1;
        for (int m = k; m < nBins; m += stepSize) {
            if (hist[m] < clipLimit) {
                hist[m] += 1.0;
                totalExcess -= 1.0;
                if (totalExcess <= 0.0) break;
            }
        }
        if (++k >= nBins) k = 0;   // don't always start at bin 0
    }
}

// ---- target-distribution mapping ---------------------------------
// Converts the cumulative probability P (in [0,1]) into an output
// intensity in [outMin, outMax] according to the chosen target
// histogram shape. dist: 0=uniform, 1=rayleigh, 2=exponential.
// Faithful port of MATLAB R2025b adapthisteq.m `makeMapping`: the
// cumulative probability p (= histSum/numPix, in [0,1]) is shaped by the
// target distribution. MATLAB scales p by vmax (the CDF value at the range
// top) BEFORE the inverse transform — omitting that factor was wrong.
static double mapProbability(double p, int dist, double alpha,
                             double outMin, double outMax)
{
    const double span = outMax - outMin;
    constexpr double kEps = 2.220446049250313e-16;   // eps
    switch (dist) {
        case 1: {  // rayleigh: temp = sqrt(-hconst*log(1-vmax*p))
            const double hconst = 2.0 * alpha * alpha;
            const double vmax   = 1.0 - std::exp(-1.0 / hconst);
            double val = vmax * p;
            if (val >= 1.0) val = 1.0 - kEps;          // avoid log(0)
            const double temp = std::sqrt(-hconst * std::log(1.0 - val));
            return std::min(outMin + temp * span, outMax);
        }
        case 2: {  // exponential: temp = -(1/alpha)*log(1-vmax*p)
            const double vmax = 1.0 - std::exp(-alpha);
            double val = vmax * p;
            if (val >= 1.0) val = 1.0 - kEps;
            const double temp = -(1.0 / alpha) * std::log(1.0 - val);
            return std::min(outMin + temp * span, outMax);
        }
        default:   // uniform: outMin + p*span, limited to the range top
            return std::min(outMin + p * span, outMax);
    }
}

}  // namespace

Value adapthisteq(const Value &I, const AdaptHistEqOptions &opts,
                  keel::Allocator *mr)
{
    adapthisteq_argcheck(I, opts);
    // ---- 1. validate options ------------------------------------
    if (opts.numTilesR < 2 || opts.numTilesC < 2)
        throw Error("adapthisteq: NumTiles must be >= 2 in each dimension",
                    0, 0, "adapthisteq", "", "numkit:adapthisteq:badTiles");
    if (!(opts.clipLimit >= 0.0 && opts.clipLimit <= 1.0))
        throw Error("adapthisteq: ClipLimit must be in [0, 1]",
                    0, 0, "adapthisteq", "", "numkit:adapthisteq:badClip");
    if (opts.nBins < 2)
        throw Error("adapthisteq: NBins must be >= 2",
                    0, 0, "adapthisteq", "", "numkit:adapthisteq:badBins");

    int dist;
    if      (opts.distribution == "uniform")     dist = 0;
    else if (opts.distribution == "rayleigh")    dist = 1;
    else if (opts.distribution == "exponential") dist = 2;
    else
        throw Error("adapthisteq: Distribution must be 'uniform', "
                    "'rayleigh' or 'exponential'",
                    0, 0, "adapthisteq", "", "numkit:adapthisteq:badDistribution");

    bool rangeOriginal;
    if      (opts.range == "full")     rangeOriginal = false;
    else if (opts.range == "original") rangeOriginal = true;
    else
        throw Error("adapthisteq: Range must be 'full' or 'original'",
                    0, 0, "adapthisteq", "", "numkit:adapthisteq:badRange");

    // ---- 2. validate input shape --------------------------------
    const Dims &d = I.dims();
    if (d.ndim() > 2 || (d.ndim() == 3 && d.pages() > 1))
        throw Error("adapthisteq: input must be a 2-D greyscale image",
                    0, 0, "adapthisteq", "", "numkit:adapthisteq:unsupportedShape");
    if (I.isComplex() || I.isCell() || I.isStruct() ||
        I.isString() || I.isFuncHandle())
        throw Error("adapthisteq: input must be a real numeric image",
                    0, 0, "adapthisteq", "", "numkit:adapthisteq:unsupportedShape");

    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    if (H == 0 || W == 0)
        throw Error("adapthisteq: input image is empty",
                    0, 0, "adapthisteq", "", "numkit:adapthisteq:unsupportedShape");

    const ValueType outType = I.type();
    const int numTilesR = opts.numTilesR;
    const int numTilesC = opts.numTilesC;
    const int nBins     = opts.nBins;

    keel::ArenaTemp arena = keel::get_scratch(mr);

    // ---- 3. read the image into a double work buffer ------------
    // (column-major, matching nk's storage convention).
    keel::Array<double> src(H * W, arena.allocator());
    for (keel::usize c = 0; c < W; ++c)
        for (keel::usize r = 0; r < H; ++r)
            src[c * H + r] = I.elemAsDouble(c * H + r);

    // ---- 4. working intensity range -----------------------------
    double classLo, classHi;
    classRange(outType, classLo, classHi);

    // Determine the input span actually used for histogramming and
    // the output span the mapping is scaled into.
    double inMin = classLo, inMax = classHi;
    double outMin = classLo, outMax = classHi;
    if (rangeOriginal) {
        // [min(I), max(I)] of the real pixels.
        double mn =  std::numeric_limits<double>::infinity();
        double mx = -std::numeric_limits<double>::infinity();
        for (keel::usize i = 0; i < H * W; ++i) {
            const double v = src[i];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        if (!std::isfinite(mn) || !std::isfinite(mx)) { mn = classLo; mx = classHi; }
        outMin = mn;
        outMax = mx;
        // Histogram over the same actual span so the limited dynamic
        // range is fully resolved by the bins.
        inMin = mn;
        inMax = mx;
    }
    // Guard against a degenerate (flat) image / zero-width range.
    if (!(inMax > inMin)) { inMin = classLo; inMax = classHi; }
    if (!(inMax > inMin)) { inMin = 0.0;     inMax = 1.0;     }

    // ---- 5. symmetric (mirror) padding --------------------------
    // Pad each dimension up to the next multiple of the tile count.
    // Pad so each tile dimension is EVEN (a multiple of 2*numTiles): the
    // inter-tile interpolation uses half-tile (tileH/2) edge regions.
    const keel::usize unitR = 2 * static_cast<keel::usize>(numTilesR);
    const keel::usize unitC = 2 * static_cast<keel::usize>(numTilesC);
    const keel::usize padBottom = (unitR - (H % unitR)) % unitR;
    const keel::usize padRight  = (unitC - (W % unitC)) % unitC;
    const keel::usize Hp = H + padBottom;
    const keel::usize Wp = W + padRight;
    const keel::usize tileH = Hp / static_cast<keel::usize>(numTilesR);
    const keel::usize tileW = Wp / static_cast<keel::usize>(numTilesC);

    // Padded work image (column-major). Mirror reflection without
    // repeating the edge sample: pad index p (p in [0,pad)) mirrors
    // to source index  H-2-p  (clamped to [0,H-1]).
    keel::Array<double> img(Hp * Wp, arena.allocator());
    auto mirrorIdx = [](keel::usize i, keel::usize n) -> keel::usize {
        if (n == 1) return 0;
        if (i < n)  return i;
        // reflect past the edge
        keel::usize k = (2 * n - 2) ? ((i) % (2 * n - 2)) : 0;
        return k < n ? k : (2 * n - 2 - k);
    };
    for (keel::usize c = 0; c < Wp; ++c) {
        const keel::usize sc = mirrorIdx(c, W);
        for (keel::usize r = 0; r < Hp; ++r) {
            const keel::usize sr = mirrorIdx(r, H);
            img[c * Hp + r] = src[sc * H + sr];
        }
    }

    // ---- 6. per-tile clipped histograms + mapping LUTs ----------
    const double invSpan = static_cast<double>(nBins) / (inMax - inMin);
    auto binOf = [&](double v) -> int {
        int b = static_cast<int>((v - inMin) * invSpan);
        if (b < 0)         b = 0;
        if (b >= nBins)    b = nBins - 1;
        return b;
    };

    const int    nTiles  = numTilesR * numTilesC;
    const double numPix  = static_cast<double>(tileH * tileW);
    // MATLAB clip COUNT: minClipLimit = ceil(numPix/nBins);
    // clipLimit = minClipLimit + round(normClip*(numPix - minClipLimit)).
    const double minClipLimit = std::ceil(numPix / static_cast<double>(nBins));
    const double clipCount = minClipLimit
        + std::round(opts.clipLimit * (numPix - minClipLimit));
    // luts: nTiles consecutive LUTs of nBins entries each.
    keel::Array<double> luts(static_cast<keel::usize>(nTiles) * nBins, arena.allocator());
    keel::Array<double> hist(nBins, arena.allocator());

    for (int tr = 0; tr < numTilesR; ++tr) {
        for (int tc = 0; tc < numTilesC; ++tc) {
            // histogram this tile
            std::fill(hist.begin(), hist.end(), 0.0);
            const keel::usize r0 = static_cast<keel::usize>(tr) * tileH;
            const keel::usize c0 = static_cast<keel::usize>(tc) * tileW;
            for (keel::usize cc = 0; cc < tileW; ++cc)
                for (keel::usize rr = 0; rr < tileH; ++rr)
                    hist[binOf(img[(c0 + cc) * Hp + (r0 + rr)])] += 1.0;

            // contrast-limit (clip + redistribute), MATLAB clipHistogram
            clipHistogram(hist.data(), nBins, clipCount);

            // CDF → mapping LUT, shaped by the target distribution
            double *lut = luts.data()
                          + (static_cast<keel::usize>(tr) * numTilesC + tc) * nBins;
            double cum = 0.0;
            for (int b = 0; b < nBins; ++b) {
                cum += hist[b];
                const double p = cum / numPix;          // P[b] in [0,1]
                lut[b] = mapProbability(p, dist, opts.alpha, outMin, outMax);
            }
        }
    }

    // ---- 7. inter-tile interpolation (MATLAB makeClaheImage) ----
    // The padded image is covered by (numTilesR+1)×(numTilesC+1) regions.
    // The first/last region in each axis is a HALF-tile that maps to the
    // single outer tile; interior regions span a full tile and blend the two
    // bracketing tile mappings with integer weights — rowRevW = regH..1,
    // rowW = 0..regH-1 (summing to regH), normalised by the region area.
    // Tiles are even (padding step), so tileH/2 is exact.
    keel::Array<double> outImg(Hp * Wp, arena.allocator());

    auto lutAt = [&](int tr, int tc) -> const double * {
        return luts.data() + (static_cast<keel::usize>(tr) * numTilesC + tc) * nBins;
    };

    keel::usize regRow0 = 0;
    for (int kr = 0; kr <= numTilesR; ++kr) {
        keel::usize regH; int trU, trL;
        if      (kr == 0)         { regH = tileH / 2; trU = 0;             trL = 0; }
        else if (kr == numTilesR) { regH = tileH / 2; trU = numTilesR - 1; trL = numTilesR - 1; }
        else                      { regH = tileH;     trU = kr - 1;        trL = kr; }

        keel::usize regCol0 = 0;
        for (int kc = 0; kc <= numTilesC; ++kc) {
            keel::usize regW; int tcL, tcR;
            if      (kc == 0)         { regW = tileW / 2; tcL = 0;             tcR = 0; }
            else if (kc == numTilesC) { regW = tileW / 2; tcL = numTilesC - 1; tcR = numTilesC - 1; }
            else                      { regW = tileW;     tcL = kc - 1;        tcR = kc; }

            const double *lutUL = lutAt(trU, tcL);
            const double *lutUR = lutAt(trU, tcR);
            const double *lutBL = lutAt(trL, tcL);
            const double *lutBR = lutAt(trL, tcR);
            const double normFactor =
                static_cast<double>(regH) * static_cast<double>(regW);

            for (keel::usize rr = 0; rr < regH; ++rr) {
                const keel::usize r      = regRow0 + rr;
                const double rowW    = static_cast<double>(rr);
                const double rowRevW = static_cast<double>(regH - rr);
                for (keel::usize cc = 0; cc < regW; ++cc) {
                    const keel::usize c       = regCol0 + cc;
                    const double colW    = static_cast<double>(cc);
                    const double colRevW = static_cast<double>(regW - cc);
                    const int bin = binOf(img[c * Hp + r]);
                    outImg[c * Hp + r] =
                        ( rowRevW * (colRevW * lutUL[bin] + colW * lutUR[bin])
                        + rowW    * (colRevW * lutBL[bin] + colW * lutBR[bin]) )
                        / normFactor;
                }
            }
            regCol0 += regW;
        }
        regRow0 += regH;
    }

    // ---- 8. strip padding, cast back to the input element type --
    Value result = Value::matrix(H, W, outType, mr);

    const bool intOut   = isIntegerType(outType) || outType == ValueType::CHAR
                          || outType == ValueType::LOGICAL;
    const double lo = std::min(outMin, outMax);
    const double hi = std::max(outMin, outMax);

    for (keel::usize c = 0; c < W; ++c) {
        for (keel::usize r = 0; r < H; ++r) {
            double v = outImg[c * Hp + r];
            // clamp into the output span
            if (v < lo) v = lo;
            if (v > hi) v = hi;
            double stored = v;
            if (intOut) {
                // round-half-away-from-zero, then clamp to class range
                stored = (v >= 0.0) ? std::floor(v + 0.5)
                                    : std::ceil (v - 0.5);
                if (stored < classLo) stored = classLo;
                if (stored > classHi) stored = classHi;
            }
            result.elemSet(c * H + r, Value::scalar(stored, arena.allocator()));
        }
    }

    return result;
}

// ════════════════════════════════════════════════════════════════════
// Otsu thresholding
// ════════════════════════════════════════════════════════════════════

namespace {

// Single-threshold Otsu on a histogram of L bins. Returns the
// MEAN of all tied-maximum bin indices (as MATLAB's graythresh does),
// plus the effectiveness metric eta in [0, 1].
// MATLAB algorithm (from graythresh.m):
//   idx = mean(find(sigma_b_squared == max(sigma_b_squared)));
// then level = (idx - 1) / (num_bins - 1).
// We return the 0-based mean directly; the caller divides by L-1.
std::pair<double, double> otsu_one_level(keel::ArrayRef<double> counts) {
    const auto r = keel::image::otsu_threshold(counts);
    return {r.bin_level, r.effectiveness};
}

} // anonymous

std::tuple<Value, Value>
otsuthresh(const Value &counts_v, keel::Allocator *mr) {
    const keel::usize L = counts_v.numel();
    keel::Array<double> c(L);
    for (keel::usize i = 0; i < L; ++i) c[i] = counts_v.elemAsDouble(i);
    auto [lvl, em] = otsu_one_level(c);
    // MATLAB graythresh / otsuthresh convention: thresh = lvl / (L - 1),
    // where lvl is the mean of all bin indices that achieve the maximum
    // sigma_b^2. (multithresh, by contrast, returns midpoints of class
    // means -- intentionally different convention in MATLAB.)
    const double thresh = (L > 1) ? (lvl / static_cast<double>(L - 1)) : 0.0;
    return std::make_tuple(Value::scalar(thresh, mr), Value::scalar(em, mr));
}

std::tuple<Value, Value>
graythresh(const Value &I, keel::Allocator *mr) {
    graythresh_argcheck(I);
    // Probed R2025b: the standard imadjust/imbinarize family class set
    // (logical rejected; the default-threshold imbinarize path routes
    // through here BEFORE imbinarize's own gate can voice).
    internal::requireImageClass(I, "graythresh", "I", 1,
                                internal::ImgClasses::FloatInt);
    // MATLAB graythresh always builds a 256-bin histogram (NPTS=256),
    // independent of the input class. nk's default_nbins gives 64 for
    // floating-point and 65536 for uint16, which produced a coarser/finer
    // Otsu level than MATLAB for those classes (uint8 already used 256).
    auto [counts, _] = imhist(I, 256, mr);
    return otsuthresh(counts, mr);
}

std::tuple<Value, Value>
multithresh(const Value &I, int N, keel::Allocator *mr) {
    multithresh_argcheck(N);
    if (N < 1) N = 1;
    if (N > 5)
        throw Error("multithresh: N > 5 not supported (exhaustive search would be too slow)",
                    0, 0, "multithresh", "", "numkit:multithresh:tooMany");

    // ── getpdf: 256-bin histogram over the data range, MATLAB-style ──
    // Floating-point data is scaled to [0,1] by (x-minA)/(maxA-minA) then
    // quantised to uint8 (grayto8); integer data is mapped from its type
    // range to uint8. minA/maxA are the finite data extrema; the final
    // thresholds are mapped back with map2OriginalScale (normFactor 255).
    const int B = 256;
    const ValueType ty = I.type();
    const bool isInt = (ty == ValueType::UINT8 || ty == ValueType::UINT16
                     || ty == ValueType::INT16 || ty == ValueType::INT8
                     || ty == ValueType::LOGICAL);
    const keel::usize M = I.numel();

    double minA = 0.0, maxA = 0.0;
    bool anyFinite = false;
    for (keel::usize i = 0; i < M; ++i) {
        const double x = I.elemAsDouble(i);
        if (std::isnan(x) || !std::isfinite(x)) continue;
        if (!anyFinite) { minA = maxA = x; anyFinite = true; }
        else { if (x < minA) minA = x; if (x > maxA) maxA = x; }
    }

    auto clampBin = [](long b) -> int {
        if (b < 0) return 0;
        if (b > 255) return 255;
        return static_cast<int>(b);
    };
    // getpdf scales the data to [0,1] by (x-minA)/(maxA-minA) and quantises
    // to uint8. Both float and integer data are scaled this way; the only
    // difference is that MATLAB does the integer scaling in single precision.
    auto toBin = [&](double x) -> int {
        if (std::isnan(x)) return -1;
        if (maxA == minA) return 0;
        if (std::isinf(x)) return x > 0 ? 255 : 0;
        if (isInt) {
            // MATLAB scales + quantises integer data in single precision.
            const float fs = static_cast<float>(x - minA)
                           / static_cast<float>(maxA - minA);
            return clampBin(std::lround(static_cast<double>(fs * 255.0f)));
        }
        const double s = (x - minA) / (maxA - minA);
        return clampBin(std::lround(s * 255.0));
    };

    keel::Array<double> counts(B, 0.0);
    double total = 0.0;
    for (keel::usize i = 0; i < M; ++i) {
        const int b = toBin(I.elemAsDouble(i));
        if (b >= 0) { counts[b] += 1.0; total += 1.0; }
    }

    // Degenerate input (no spread): MATLAB returns thresholds at the value.
    if (total <= 0.0 || maxA == minA) {
        Value t = Value::matrix(1, N, ValueType::DOUBLE, mr);
        double *td = t.doubleDataMut();
        for (int k = 0; k < N; ++k) td[k] = minA;
        return std::make_tuple(std::move(t), Value::scalar(0.0, mr));
    }

    // omega = cumsum(p); mu = cumsum(p .* (1:B)'); mu_t = mu(end).
    keel::Array<double> omega(B);
    keel::Array<double> mu(B);
    double cum = 0.0, cumm = 0.0;
    for (int j = 0; j < B; ++j) {
        const double p = counts[j] / total;
        cum += p;
        cumm += p * (j + 1);
        omega[j] = cum;
        mu[j] = cumm;
    }
    const double mu_t = mu[B - 1];

    keel::Array<double> threshBins(N, 0.0);   // 0-based bin thresholds
    double bestVar = 0.0;

    if (N == 1) {
        // sigma_b^2(t) = (mu_t*omega - mu)^2 / (omega*(1-omega)); arg-max bin.
        double best = -std::numeric_limits<double>::infinity();
        keel::Array<int> ties;
        for (int t = 0; t < B; ++t) {
            const double o = omega[t];
            if (o <= 0.0 || o >= 1.0) continue;
            const double num = mu_t * o - mu[t];
            const double s = num * num / (o * (1.0 - o));
            if (!std::isfinite(s)) continue;
            if (s > best) { best = s; ties.clear(); ties.push_back(t); }
            else if (s == best) ties.push_back(t);
        }
        double m = 0.0;
        for (int t : ties) m += t;
        threshBins[0] = ties.empty() ? 0.0 : m / ties.size();
        bestVar = best;
    } else if (N == 2) {
        double best = -std::numeric_limits<double>::infinity();
        keel::Array<std::pair<int, int>> ties;
        for (int r = 0; r < B; ++r) {
            const double o0 = omega[r];
            if (o0 <= 0.0) continue;
            const double mu0t = mu_t - mu[r] / o0;
            for (int c = r + 1; c < B; ++c) {
                const double o1 = omega[c] - omega[r];
                if (o1 <= 0.0) continue;
                const double mu1t = mu_t - (mu[c] - mu[r]) / o1;
                const double o2 = 1.0 - (o0 + o1);
                if (o2 <= 0.0) continue;
                const double term1 = o0 * mu0t * mu0t;
                const double term2 = o1 * mu1t * mu1t;
                const double q = o0 * mu0t + o1 * mu1t;
                const double term3 = q * q / o2;
                const double s = term1 + term2 + term3;
                if (!std::isfinite(s)) continue;
                if (s > best) { best = s; ties.clear(); ties.push_back({r, c}); }
                else if (s == best) ties.push_back({r, c});
            }
        }
        double sr = 0.0, sc = 0.0;
        for (auto &pr : ties) { sr += pr.first; sc += pr.second; }
        const double n = ties.empty() ? 1.0 : double(ties.size());
        threshBins[0] = sr / n;
        threshBins[1] = sc / n;
        bestVar = best;
    } else {
        // N >= 3: exhaustive over 256 bins is infeasible and MATLAB uses
        // fminsearch (a local optimiser). We instead solve the GLOBAL
        // multilevel-Otsu optimum by dynamic programming over the 256-bin
        // histogram (maximise sum_k s_k^2 / w_k). This is correct-scale and
        // a valid set of thresholds, but — being global rather than
        // fminsearch's local optimum — may differ from MATLAB for N >= 3.
        keel::Array<double> Omega(B + 1, 0.0);
        keel::Array<double> Mu(B + 1, 0.0);
        for (int j = 0; j < B; ++j) {
            Omega[j + 1] = omega[j];
            Mu[j + 1] = mu[j];
        }
        auto segScore = [&](int lo, int hi) -> double {   // bins [lo,hi] 0-based
            const double w = Omega[hi + 1] - Omega[lo];
            if (w <= 0.0) return 0.0;
            const double s = Mu[hi + 1] - Mu[lo];
            return s * s / w;
        };
        const int K = N + 1;                              // number of classes
        const double NEG = -std::numeric_limits<double>::infinity();
        // dp[k*B + t] = best score for first k classes, class k ending at bin t.
        keel::Array<double> dp((K + 1) * B, NEG);
        keel::Array<int> back((K + 1) * B, -1);
        for (int t = 0; t < B; ++t) dp[1 * B + t] = segScore(0, t);
        for (int k = 2; k <= K; ++k) {
            for (int t = k - 1; t < B; ++t) {
                double bestv = NEG; int bestp = -1;
                for (int tp = k - 2; tp < t; ++tp) {
                    if (dp[(k - 1) * B + tp] == NEG) continue;
                    const double v = dp[(k - 1) * B + tp] + segScore(tp + 1, t);
                    if (v > bestv) { bestv = v; bestp = tp; }
                }
                dp[k * B + t] = bestv;
                back[k * B + t] = bestp;
            }
        }
        // Last class must end at bin B-1; backtrack the N boundaries.
        keel::Array<int> bounds(K, 0);
        bounds[K - 1] = B - 1;
        for (int k = K; k >= 2; --k)
            bounds[k - 2] = back[k * B + bounds[k - 1]];
        for (int k = 0; k < N; ++k) threshBins[k] = bounds[k];
        bestVar = dp[K * B + (B - 1)] - mu_t * mu_t;
    }

    // map2OriginalScale: minA + thresh/255 * (maxA - minA). For integer
    // input MATLAB casts the result back to the integer type (round).
    Value t_out = Value::matrix(1, N, ValueType::DOUBLE, mr);
    double *td = t_out.doubleDataMut();
    for (int k = 0; k < N; ++k) {
        double v = minA + threshBins[k] / 255.0 * (maxA - minA);
        if (isInt) v = std::round(v);
        td[k] = v;
    }

    // metric = maxval / sum(p .* ((1:B)' - mu_t)^2).
    double denom = 0.0;
    for (int j = 0; j < B; ++j) {
        const double p = counts[j] / total;
        const double d = (j + 1) - mu_t;
        denom += p * d * d;
    }
    const double em = (denom > 0.0 && std::isfinite(bestVar))
                          ? bestVar / denom : 0.0;
    return std::make_tuple(std::move(t_out), Value::scalar(em, mr));
}

Value imbinarize(const Value &I, double thresh, keel::Allocator *mr) {
    imbinarize_argcheck(I);
    internal::requireImageClass(I, "imbinarize", "I", 1,
                                internal::ImgClasses::FloatInt);
    const keel::usize N = I.numel();
    Value out;
    const auto &d = I.dims();
    if (I.isScalar()) out = Value::matrix(1, 1, ValueType::LOGICAL, mr);
    else if (d.is3D())out = Value::matrix3d(d.rows(), d.cols(), d.pages(),
                                           ValueType::LOGICAL, mr);
    else              out = Value::matrix(d.rows(), d.cols(),
                                          ValueType::LOGICAL, mr);
    if (N == 0) return out;
    keel::u8 *od = out.logicalDataMut();
    for (keel::usize i = 0; i < N; ++i) od[i] = (element_to_unit(I, i) > thresh) ? 1 : 0;
    return out;
}

// Per-pixel-threshold variant of imbinarize. T must have the same
// numel as I (typically the same H × W matrix; H × W × 3 also OK
// when I is a same-shape volume). T's element scale is interpreted
// in the same `element_to_unit` space as I, so the raw element
// values are directly comparable. Composes with `adaptthresh`.
Value imbinarize(const Value &I, const Value &T, keel::Allocator *mr)
{
    imbinarize_argcheck(I, T);
    internal::requireImageClass(I, "imbinarize", "I", 1,
                                internal::ImgClasses::FloatInt);
    if (T.numel() != I.numel())
        throw Error("imbinarize: per-pixel T must have the same number "
                    "of elements as I",
                    0, 0, "imbinarize", "", "numkit:imbinarize:Tshape");
    const keel::usize N = I.numel();
    Value out;
    const auto &d = I.dims();
    if (I.isScalar()) out = Value::matrix(1, 1, ValueType::LOGICAL, mr);
    else if (d.is3D())out = Value::matrix3d(d.rows(), d.cols(), d.pages(),
                                           ValueType::LOGICAL, mr);
    else              out = Value::matrix(d.rows(), d.cols(),
                                          ValueType::LOGICAL, mr);
    if (N == 0) return out;
    keel::u8 *od = out.logicalDataMut();
    for (keel::usize i = 0; i < N; ++i) {
        const double t = T.elemAsDouble(i);
        od[i] = (element_to_unit(I, i) > t) ? 1 : 0;
    }
    return out;
}

Value imquantize(const Value &I, const Value &levels, keel::Allocator *mr) {
    imquantize_argcheck(I, levels);
    internal::requireImageClass(I, "imquantize", "I", 1,
                                internal::ImgClasses::FloatInt);
    internal::requireImageClass(levels, "imquantize", "levels", 2,
                                internal::ImgClasses::AnyNumericReal);
    const keel::usize Lcount = levels.numel();
    keel::Array<double> lv(Lcount);
    for (keel::usize i = 0; i < Lcount; ++i) lv[i] = levels.elemAsDouble(i);
    std::sort(lv.begin(), lv.end());

    const keel::usize N = I.numel();
    Value out;
    const auto &d = I.dims();
    if (I.isScalar()) out = Value::matrix(1, 1, ValueType::DOUBLE, mr);
    else if (d.is3D())out = Value::matrix3d(d.rows(), d.cols(), d.pages(),
                                           ValueType::DOUBLE, mr);
    else              out = Value::matrix(d.rows(), d.cols(), ValueType::DOUBLE, mr);
    if (N == 0) return out;

    double *od = out.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) {
        const double u = element_to_unit(I, i);
        // class index = first level we don't exceed; output is 1-indexed.
        keel::usize cls = 1;
        for (keel::usize k = 0; k < Lcount; ++k) {
            if (u <= lv[k]) break;
            ++cls;
        }
        od[i] = double(cls);
    }
    return out;
}

// ════════════════════════════════════════════════════════════════════
// imhistmatch — histogram matching to a reference image
// ════════════════════════════════════════════════════════════════════
// MATLAB R2025b: J = imhistmatch(I, ref, nbins) is exactly
// histeq(I, imhist(ref, nbins)). The transform is built at the input
// class's FULL resolution NPTS (256 for uint8/float, 65536 for
// uint16/int16), NOT at nbins, then applied per input level via
// grayxform. nbins (default 64, all classes) only sets the resolution of
// the TARGET histogram. Algorithm (Gonzalez histogram matching):
//   1. hgram = imhist(ref, nbins)            // raw counts, also 2nd output
//   2. normalise hgram so sum == numel(I); cumd = cumsum(hgram)
//   3. nn = imhist(I, NPTS); cum = cumsum(nn)
//   4. tol(j) = nn(j)/2 for interior j (0 at the two ends)
//   5. T(j) = argmin_i [ cumd(i) − cum(j) + tol(j) ], with errors below
//      −numel(I)·sqrt(eps) clamped to +numel(I); output level (i−1)/(nbins−1)
//   6. apply T to each pixel by its NPTS-level, cast back to input class.

Value imhistmatch(const Value &I, const Value &ref, int nbins,
                  Value *hgramOut, keel::Allocator *mr)
{
    if (nbins <= 0) nbins = 64;   // MATLAB imhistmatch default (all classes)
    if (nbins < 2) nbins = 2;

    const keel::usize Ni = I.numel();
    const keel::usize Nr = ref.numel();
    if (Ni == 0 || Nr == 0) {
        if (hgramOut)
            *hgramOut = Value::matrix(1, nbins, ValueType::DOUBLE, mr);
        return Value::matrix(I.dims().rows(), I.dims().cols(), I.type(), mr);
    }

    // Full-resolution level count of the input class (getrangefromclass).
    int NPTS;
    switch (I.type()) {
        case ValueType::UINT16:
        case ValueType::INT16:   NPTS = 65536; break;
        case ValueType::LOGICAL: NPTS = 2;     break;
        default:                 NPTS = 256;   break;  // UINT8/DOUBLE/SINGLE
    }

    // MATLAB imhist centred binning: u ∈ [0,1] → round(u·(bins−1)).
    auto level_of = [](double u, int bins) {
        int b = (int)std::lround(u * (bins - 1));
        if (b < 0) b = 0;
        if (b >= bins) b = bins - 1;
        return b;
    };

    // 1. Reference histogram (raw counts) = imhist(ref, nbins); 2nd output.
    keel::Array<double> hgram((keel::usize)nbins, 0.0);
    for (keel::usize i = 0; i < Nr; ++i)
        hgram[(keel::usize)level_of(element_to_unit(ref, i), nbins)] += 1.0;

    // 2. Normalise to sum == numel(I), then cumulate (cumd, length nbins).
    double sumH = 0.0;
    for (int b = 0; b < nbins; ++b) sumH += hgram[(keel::usize)b];
    const double scale = (sumH > 0.0) ? (double)Ni / sumH : 0.0;
    keel::Array<double> cumd((keel::usize)nbins, 0.0);
    {
        double acc = 0.0;
        for (int b = 0; b < nbins; ++b) { acc += hgram[(keel::usize)b] * scale;
                                          cumd[(keel::usize)b] = acc; }
    }

    // 3. Input histogram at full resolution = imhist(I, NPTS), cumulated.
    keel::Array<double> nn((keel::usize)NPTS, 0.0);
    for (keel::usize i = 0; i < Ni; ++i)
        nn[(keel::usize)level_of(element_to_unit(I, i), NPTS)] += 1.0;
    keel::Array<double> cum((keel::usize)NPTS, 0.0);
    { double acc = 0.0;
      for (int j = 0; j < NPTS; ++j) { acc += nn[(keel::usize)j]; cum[(keel::usize)j] = acc; } }

    // 5. Transform T (length NPTS), values in [0,1]. For each input level j
    // pick the target bin i minimising the matching error (first on ties,
    // matching MATLAB's min); large-negative errors are clamped to +Ni.
    const double clampThr = -(double)Ni * std::sqrt(2.2204460492503131e-16);
    const double denom = (nbins > 1) ? (double)(nbins - 1) : 1.0;
    keel::Array<double> T((keel::usize)NPTS, 0.0);
    for (int j = 0; j < NPTS; ++j) {
        const double tol = (j == 0 || j == NPTS - 1) ? 0.0
                                                     : nn[(keel::usize)j] * 0.5;
        const double base = tol - cum[(keel::usize)j];
        double best = 0.0; int bi = 0; bool have = false;
        for (int i = 0; i < nbins; ++i) {
            double e = cumd[(keel::usize)i] + base;
            if (e < clampThr) e = (double)Ni;
            if (!have || e < best) { best = e; bi = i; have = true; }
        }
        T[(keel::usize)j] = (double)bi / denom;
    }

    // 6. Apply: map each pixel by its NPTS-level, cast back to input class.
    const auto &d = I.dims();
    Value out;
    if (d.is3D()) out = Value::matrix3d(d.rows(), d.cols(), d.pages(),
                                        I.type(), mr);
    else          out = Value::matrix(d.rows(), d.cols(), I.type(), mr);
    for (keel::usize i = 0; i < Ni; ++i) {
        const int lvl = level_of(element_to_unit(I, i), NPTS);
        store_classed(out, i, T[(keel::usize)lvl], I.type());
    }

    // 2nd output: ref's histogram as a 1×nbins double row (= MATLAB hgram).
    if (hgramOut) {
        Value hg = Value::matrix(1, nbins, ValueType::DOUBLE, mr);
        double *hd = hg.doubleDataMut();
        for (int b = 0; b < nbins; ++b) hd[b] = hgram[(keel::usize)b];
        *hgramOut = std::move(hg);
    }
    return out;
}

// ════════════════════════════════════════════════════════════════════
// adaptthresh — locally adaptive threshold matrix
// ════════════════════════════════════════════════════════════════════
// Computes T(x, y) ∈ [0, 1] from a local statistic (box mean or
// Gaussian-smoothed mean) of a neighborhood centered at each pixel.
// `sensitivity` scales the threshold relative to the local statistic.
// MATLAB R2025b maps it linearly: T = clip(localStat · (1.6 − s), 0, 1).
// So sensitivity = 0.5 → T = 1.1 · localStat (threshold 10 % above the
// local mean); higher sensitivity lowers the threshold (more foreground
// after binarize), lower raises it. (Verified exactly against MATLAB on
// constant images across s ∈ {0, .25, .5, .75, 1}.)

Value adaptthresh(const Value &I, double sensitivity, int neighborhood, keel::StringRef statistic, keel::Allocator *mr)
{
    adaptthresh_argcheck(I, sensitivity, statistic);
    internal::requireImageClass(I, "adaptthresh", "I", 1,
                                internal::ImgClasses::FloatInt);
    if (!(sensitivity >= 0.0 && sensitivity <= 1.0))
        throw Error("adaptthresh: sensitivity must be in [0, 1]",
                    0, 0, "adaptthresh", "", "numkit:adaptthresh:sens");

    const int H = static_cast<int>(I.dims().rows());
    const int W = static_cast<int>(I.dims().cols());

    if (neighborhood <= 0) {
        // MATLAB default NeighborhoodSize = 2*floor(size(I)/16)+1, which is
        // 1 (no smoothing — the local statistic is the pixel itself) for any
        // dimension below 16. Do NOT clamp up to 3: that over-smooths small
        // images and diverges from MATLAB.
        const int dim = std::min(H, W);
        neighborhood = 2 * (dim / 16) + 1;
    }
    if ((neighborhood & 1) == 0) neighborhood += 1;  // force odd

    // First, normalise input intensity to [0, 1] in a double buffer.
    // We replicate the scaling rule used elsewhere in nk:
    //   uint8/int8 / 255, uint16 / 65535, double passes through.
    Value Inorm = Value::matrix(static_cast<keel::usize>(H),
                                static_cast<keel::usize>(W),
                                ValueType::DOUBLE, mr);
    double *nd = Inorm.doubleDataMut();
    const ValueType srcT = I.type();
    for (int c = 0; c < W; ++c)
        for (int r = 0; r < H; ++r) {
            const keel::usize i = static_cast<keel::usize>(c) *
                                  static_cast<keel::usize>(H) +
                              static_cast<keel::usize>(r);
            const double v = I.elemAsDouble(i);
            double w = v;
            switch (srcT) {
                case ValueType::UINT8:  w = v / 255.0;   break;
                case ValueType::UINT16: w = v / 65535.0; break;
                case ValueType::INT16:  w = (v + 32768.0) / 65535.0; break;
                default:                w = v;            break;
            }
            nd[i] = w;
        }

    // Compute local statistic.
    Value localStat;
    const keel::StringRef s = statistic.empty() ? keel::StringRef("mean")
                                                : statistic;
    if (s == "mean" || s == "Mean" || s == "MEAN") {
        localStat = imboxfilt(Inorm, neighborhood, mr);
    } else if (s == "gaussian" || s == "Gaussian" || s == "GAUSSIAN") {
        // σ ≈ neighborhood/6: typical MATLAB default for adaptthresh's
        // Gaussian variant.
        const double sigma = double(neighborhood) / 6.0;
        localStat = imgaussfilt(Inorm, sigma, neighborhood, mr);
    } else {
        throw Error("adaptthresh: statistic must be 'mean' or 'gaussian'",
                    0, 0, "adaptthresh", "", "numkit:adaptthresh:stat");
    }

    // Apply the MATLAB sensitivity scale: T = clip(localStat·(1.6−s), 0, 1).
    Value T = Value::matrix(static_cast<keel::usize>(H),
                            static_cast<keel::usize>(W),
                            ValueType::DOUBLE, mr);
    double *Td = T.doubleDataMut();
    const double scale = 1.6 - sensitivity;
    for (int i = 0; i < H * W; ++i) {
        double v = localStat.elemAsDouble(static_cast<keel::usize>(i)) * scale;
        if (v < 0.0) v = 0.0;
        if (v > 1.0) v = 1.0;
        Td[i] = v;
    }
    return T;
}

Value imflatfield(const Value &I, double sigma, const Value &mask, keel::Allocator *mr)
{
    imflatfield_argcheck(I, mask);
    const ValueType classin = I.type();
    const auto &d = I.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    const keel::usize pages = d.is3D() ? d.pages() : 1;
    const keel::usize plane = H * W;
    const keel::usize N = I.numel();

    // Convert source to double in [0, 1] (im2double-like).
    Value Idbl;
    if (d.is3D())  Idbl = Value::matrix3d(H, W, pages, ValueType::DOUBLE, mr);
    else           Idbl = Value::matrix(H, W, ValueType::DOUBLE, mr);
    double *idd = Idbl.doubleDataMut();
    for (keel::usize i = 0; i < N; ++i) idd[i] = element_to_unit(I, i);

    // Gaussian low-pass per plane. MATLAB R2025b imflatfield uses
    // symmetric padding internally — NOT the imgaussfilt default
    // (replicate). For sigma=8 on a 32×32 image the kernel (size 33)
    // exceeds the image, so the boundary mode dominates the result.
    // Build the kernel via fspecial("gaussian") and apply with
    // PadMode::Symmetric directly via imfilter.
    int filter_size = 2 * static_cast<int>(std::ceil(2.0 * sigma)) + 1;
    if (filter_size < 3) filter_size = 3;
    // fspecial("gaussian", {rows, cols, sigma}) — three-param form
    Value gk = fspecial("gaussian", {static_cast<double>(filter_size), static_cast<double>(filter_size), sigma}, mr);

    Value F;
    if (d.is3D())  F = Value::matrix3d(H, W, pages, ValueType::DOUBLE, mr);
    else           F = Value::matrix(H, W, ValueType::DOUBLE, mr);
    double *fd = F.doubleDataMut();
    for (keel::usize p = 0; p < pages; ++p) {
        Value plane2d;
        if (pages == 1) {
            plane2d = Idbl;
        } else {
            plane2d = Value::matrix(H, W, ValueType::DOUBLE, mr);
            std::memcpy(plane2d.doubleDataMut(), idd + p * plane,
                        plane * sizeof(double));
        }
        Value blurred = imfilter(plane2d, gk, PadMode::Symmetric, 0.0, /*full=*/false, /*flip_kernel=*/false, mr);
        std::memcpy(fd + p * plane, blurred.doubleData(),
                    plane * sizeof(double));
    }

    // Per-page mean of A (NOT F) — MATLAB normalizes by mean(A).
    // mean(F) ≈ mean(A) for full-image mean (Gaussian preserves mean),
    // but with a mask the difference matters.
    const bool have_mask = (mask.numel() > 0);
    if (have_mask && (mask.dims().rows() != H || mask.dims().cols() != W))
        throw Error("imflatfield: mask must match spatial dims of I",
                    0, 0, "imflatfield", "", "numkit:imflatfield:masksize");
    keel::Array<double> meanA(pages, 0.0);
    for (keel::usize p = 0; p < pages; ++p) {
        const double *ap = idd + p * plane;
        if (have_mask) {
            keel::usize cnt = 0;
            double acc = 0.0;
            for (keel::usize i = 0; i < plane; ++i)
                if (mask.elemAsDouble(i) != 0.0) { acc += ap[i]; ++cnt; }
            meanA[p] = (cnt > 0) ? acc / static_cast<double>(cnt) : 0.0;
        } else if (plane > 0) {
            double acc = 0.0;
            for (keel::usize i = 0; i < plane; ++i) acc += ap[i];
            meanA[p] = acc / static_cast<double>(plane);
        }
    }

    // Output: per-page A * meanA / F, with eps guard for degenerate F.
    Value out;
    if (d.is3D())  out = Value::matrix3d(H, W, pages, classin, mr);
    else           out = Value::matrix(H, W, classin, mr);
    constexpr double EPS = 1e-12;
    for (keel::usize p = 0; p < pages; ++p) {
        const double mA = meanA[p];
        for (keel::usize i = 0; i < plane; ++i) {
            const keel::usize idx = p * plane + i;
            const double f = fd[idx];
            const double v = (f > EPS) ? idd[idx] * mA / f : idd[idx];
            store_classed(out, idx, v, classin);
        }
    }
    return out;
}

Value wcodemat(const Value &X, int nb, keel::StringRef opt, int absol, keel::Allocator *mr)
{
    wcodemat_argcheck(nb, opt);
    if (nb < 1)
        throw Error("wcodemat: NB must be a positive integer",
                    0, 0, "wcodemat", "", "numkit:wcodemat:nb");

    const auto &d = X.dims();
    const keel::usize H = d.rows();
    const keel::usize W = d.cols();
    Value out = d.is3D()
        ? Value::matrix3d(H, W, d.pages(), ValueType::DOUBLE, mr)
        : Value::matrix(H, W, ValueType::DOUBLE, mr);
    const keel::usize N = X.numel();
    if (N == 0) return out;
    double *od = out.doubleDataMut();

    auto vget = [&](keel::usize i) {
        const double v = X.elemAsDouble(i);
        return absol ? keel::abs(v) : v;
    };

    keel::String lo;
    lo.reserve(opt.size());
    for (char c : opt) lo.push_back(keel::to_ascii_lower(c));
    if (lo.empty()) lo.assign("mat");

    // MATLAB R2025b formula: y = floor((v - mn) / span * nb) + 1, with the
    // upper edge (v == mx) clamped from nb+1 down to nb.
    // Bug fix 2026-05-08: previous impl used `round` and multiplied by
    // `nb - 1`, producing off-by-one quantization errors on interior values
    // (e.g., wcodemat([1 -2 3; 4 -5 6], 4) → nk [1 2 2; 3 3 4] vs
    //  MATLAB [1 1 2; 3 4 4]).
    auto encode = [&](double v, double mn, double mx) {
        const double span = mx - mn;
        if (span == 0.0) return 1.0;
        double y = std::floor((v - mn) / span * static_cast<double>(nb)) + 1.0;
        if (y < 1.0) y = 1.0;
        if (y > nb)  y = static_cast<double>(nb);
        return y;
    };

    if (lo == "mat") {
        double mn =  std::numeric_limits<double>::infinity();
        double mx = -std::numeric_limits<double>::infinity();
        for (keel::usize i = 0; i < N; ++i) {
            const double v = vget(i);
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        for (keel::usize i = 0; i < N; ++i)
            od[i] = encode(vget(i), mn, mx);
    } else if (lo == "row") {
        // Per-row scaling. col-major: row r elements at i = c*H + r.
        for (keel::usize r = 0; r < H; ++r) {
            double mn =  std::numeric_limits<double>::infinity();
            double mx = -std::numeric_limits<double>::infinity();
            for (keel::usize c = 0; c < W; ++c) {
                const double v = vget(c * H + r);
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            for (keel::usize c = 0; c < W; ++c)
                od[c * H + r] = encode(vget(c * H + r), mn, mx);
        }
    } else if (lo == "col") {
        for (keel::usize c = 0; c < W; ++c) {
            double mn =  std::numeric_limits<double>::infinity();
            double mx = -std::numeric_limits<double>::infinity();
            for (keel::usize r = 0; r < H; ++r) {
                const double v = vget(c * H + r);
                if (v < mn) mn = v;
                if (v > mx) mx = v;
            }
            for (keel::usize r = 0; r < H; ++r)
                od[c * H + r] = encode(vget(c * H + r), mn, mx);
        }
    } else {
        throw Error("wcodemat: opt must be 'mat', 'row', or 'col'",
                    0, 0, "wcodemat", "", "numkit:wcodemat:opt");
    }
    return out;
}

Value entropy(const Value &I, int nbins, keel::Allocator *mr)
{
    const ValueType ct = I.type();
    const bool isLogical = (ct == ValueType::LOGICAL);
    if (nbins <= 0) nbins = isLogical ? 2 : 256;

    Value Iu = isLogical ? I : im2uint8(I, mr);
    auto [counts, _bins] = imhist(Iu, nbins, mr);
    const double *cd = counts.doubleData();

    double total = 0.0;
    for (int i = 0; i < nbins; ++i) total += cd[i];
    if (total <= 0.0) return Value::scalar(0.0, mr);

    double H = 0.0;
    for (int i = 0; i < nbins; ++i) {
        if (cd[i] <= 0.0) continue;
        const double p = cd[i] / total;
        H -= p * std::log2(p);
    }
    return Value::scalar(H, mr);
}

namespace {
// Common quantisation tail: given a sorted threshold vector and the
// resulting level count, build the output indexed image.
Value graysliceQuantize(const Value &I, keel::ArrayRef<double> thresh,
                        keel::usize n_levels, keel::Allocator *mr)
{
    const keel::usize N = I.numel();
    const ValueType outT = (n_levels < 256) ? ValueType::UINT8
                                            : ValueType::DOUBLE;
    const auto &d = I.dims();
    Value out;
    if (d.is3D()) out = Value::matrix3d(d.rows(), d.cols(), d.pages(),
                                        outT, mr);
    else          out = Value::matrix(d.rows(), d.cols(), outT, mr);
    if (N == 0) return out;
    const bool baseOne = (outT == ValueType::DOUBLE);
    for (keel::usize i = 0; i < N; ++i) {
        const double iv = I.elemAsDouble(i);
        keel::usize cnt = 0;
        for (keel::usize k = 0; k < thresh.size(); ++k) {
            if (thresh[k] <= iv) ++cnt;
            else                 break;
        }
        if (outT == ValueType::UINT8) {
            const keel::usize v = cnt > 255 ? 255 : cnt;
            out.uint8DataMut()[i] = static_cast<keel::u8>(v);
        } else {
            out.doubleDataMut()[i] = static_cast<double>(cnt) +
                                     (baseOne ? 1.0 : 0.0);
        }
    }
    return out;
}
} // anon

Value grayslice(const Value &I, int N, keel::Allocator *mr)
{
    grayslice_argcheck(N);
    if (N < 1)
        throw Error("grayslice: N must be a positive integer",
                    0, 0, "grayslice", "", "numkit:grayslice:n");
    const ValueType ct = I.type();
    const bool isInt16 = (ct == ValueType::INT16);
    const bool isFloat = (ct == ValueType::DOUBLE || ct == ValueType::SINGLE);
    const double n_scalar = static_cast<double>(N);
    const keel::usize k_max = static_cast<keel::usize>(N - 1);
    keel::Array<double> thresh;
    thresh.reserve(k_max);
    const double scale =
          (ct == ValueType::UINT8)  ? 255.0
        : (ct == ValueType::UINT16) ? 65535.0
        : isInt16                   ? 65535.0
        : 1.0;
    for (keel::usize k = 1; k <= k_max; ++k) {
        const double v_unit = static_cast<double>(k) / n_scalar;
        double v_class;
        if (isFloat) v_class = v_unit;
        else {
            v_class = std::round(v_unit * scale);
            if (isInt16) v_class -= 32768.0;
        }
        thresh.push_back(v_class);
    }
    return graysliceQuantize(I, thresh, k_max + 1, mr);
}

Value grayslice(const Value &I, keel::ArrayRef<double> levels,
                keel::Allocator *mr)
{
    const ValueType ct = I.type();
    const bool isFloat = (ct == ValueType::DOUBLE || ct == ValueType::SINGLE);
    const keel::usize M = levels.size();
    keel::Array<double> thresh(levels);
    std::sort(thresh.begin(), thresh.end());
    const keel::usize N = I.numel();
    if (isFloat && N > 0) {
        double imin =  std::numeric_limits<double>::infinity();
        double imax = -std::numeric_limits<double>::infinity();
        for (keel::usize i = 0; i < N; ++i) {
            const double v = I.elemAsDouble(i);
            if (v < imin) imin = v;
            if (v > imax) imax = v;
        }
        for (keel::usize i = 0; i < M; ++i) {
            if (thresh[i] < imin) thresh[i] = imin;
            if (thresh[i] > imax) thresh[i] = imax;
        }
    }
    return graysliceQuantize(I, thresh, M + 1, mr);
}

} // namespace numkit::image
