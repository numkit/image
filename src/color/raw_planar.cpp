// toolboxes/image/src/color/raw_planar.cpp
//
// raw2planar / planar2raw — deinterleave / re-interleave a Bayer CFA
// mosaic into its four colour-filter sensor planes. Pure parity-based
// rearrangement; no interpolation. Class-preserving.
//
// raw2planar(cfa)   M×N  → (M/2)×(N/2)×4
//   plane 1 = cfa(1:2:M, 1:2:N)     (odd row, odd col)
//   plane 2 = cfa(1:2:M, 2:2:N)     (odd row, even col)
//   plane 3 = cfa(2:2:M, 1:2:N)     (even row, odd col)
//   plane 4 = cfa(2:2:M, 2:2:N)     (even row, even col)
//
// planar2raw is the exact inverse. Both round-trip with equality on
// any class accepted by Value (numeric and logical).

#include <keel/core.h>
#include <keel/memory.h>
#include <numkit/image/color/color.hpp>

#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <keel/static_array.h>
#include <cstdint>

namespace numkit::image {

namespace {

// Copy one element from src[si] to dst[di] preserving class.
inline void copyElem(Value &dst, keel::usize di, const Value &src, keel::usize si)
{
    switch (src.type()) {
        case ValueType::DOUBLE:  dst.doubleDataMut()[di]  = src.doubleData()[si];  break;
        case ValueType::SINGLE:  dst.singleDataMut()[di]  = src.singleData()[si];  break;
        case ValueType::UINT8:   dst.uint8DataMut()[di]   = src.uint8Data()[si];   break;
        case ValueType::UINT16:  dst.uint16DataMut()[di]  = src.uint16Data()[si];  break;
        case ValueType::UINT32:  dst.uint32DataMut()[di]  = src.uint32Data()[si];  break;
        case ValueType::UINT64:  dst.uint64DataMut()[di]  = src.uint64Data()[si];  break;
        case ValueType::INT8:    dst.int8DataMut()[di]    = src.int8Data()[si];    break;
        case ValueType::INT16:   dst.int16DataMut()[di]   = src.int16Data()[si];   break;
        case ValueType::INT32:   dst.int32DataMut()[di]   = src.int32Data()[si];   break;
        case ValueType::INT64:   dst.int64DataMut()[di]   = src.int64Data()[si];   break;
        case ValueType::LOGICAL: dst.logicalDataMut()[di] = src.logicalData()[si]; break;
        default:
            throw Error("raw_planar: unsupported element type",
                        0, 0, "raw_planar", "", "numkit:raw_planar:type");
    }
}

void raw2planar_argcheck(const Value &cfa)
{
    const auto &d = cfa.dims();
    if (d.ndims() < 2)
        throw Error("raw2planar: cfa must be 2-D",
                    0, 0, "raw2planar", "", "numkit:raw2planar:rank");
    const keel::usize M = d.dim(0);
    const keel::usize N = d.dim(1);
    if (cfa.numel() != M * N)
        throw Error("raw2planar: cfa must be a 2-D matrix",
                    0, 0, "raw2planar", "", "numkit:raw2planar:shape");
    if ((M % 2) != 0 || (N % 2) != 0)
        throw Error("raw2planar: cfa dimensions must both be even",
                    0, 0, "raw2planar", "", "numkit:raw2planar:DimsMustBeEven");
}

void planar2raw_argcheck(const Value &I)
{
    const auto &d = I.dims();
    if (d.ndims() < 3 || d.dim(2) != 4)
        throw Error("planar2raw: I must be (M)×(N)×4",
                    0, 0, "planar2raw", "", "numkit:planar2raw:shape");
    const keel::usize Mp = d.dim(0);
    const keel::usize Np = d.dim(1);
    if (I.numel() != Mp * Np * 4)
        throw Error("planar2raw: I must be (M)×(N)×4",
                    0, 0, "planar2raw", "", "numkit:planar2raw:shape");
}

} // anonymous

Value raw2planar(const Value &cfa, keel::Allocator *mr)
{
    raw2planar_argcheck(cfa);
    const auto &d = cfa.dims();
    if (d.ndims() < 2)
        throw Error("raw2planar: cfa must be 2-D",
                    0, 0, "raw2planar", "", "numkit:raw2planar:rank");
    const keel::usize M = d.dim(0);
    const keel::usize N = d.dim(1);
    if (cfa.numel() != M * N)
        throw Error("raw2planar: cfa must be a 2-D matrix",
                    0, 0, "raw2planar", "", "numkit:raw2planar:shape");
    if ((M % 2) != 0 || (N % 2) != 0)
        throw Error("raw2planar: cfa dimensions must both be even",
                    0, 0, "raw2planar", "", "numkit:raw2planar:DimsMustBeEven");

    const keel::usize Mp = M / 2;
    const keel::usize Np = N / 2;
    const ValueType T = cfa.type();
    Value P = Value::matrixND(keel::StaticArray<keel::usize,3>{Mp, Np, 4}.data(), 3, T, mr);

    // Column-major index: cfa(r,c) = r + c*M  (0-indexed).
    //                     P(r,c,k) = r + c*Mp + k*Mp*Np.
    for (keel::usize cp = 0; cp < Np; ++cp) {
        for (keel::usize rp = 0; rp < Mp; ++rp) {
            const keel::usize r2 = 2 * rp;
            const keel::usize c2 = 2 * cp;
            const keel::usize baseP = rp + cp * Mp;
            // plane 1: (odd row, odd col)  — 0-indexed (r2, c2)
            copyElem(P, baseP + 0 * Mp * Np, cfa, r2     + c2     * M);
            // plane 2: (odd row, even col) — (r2, c2+1)
            copyElem(P, baseP + 1 * Mp * Np, cfa, r2     + (c2+1) * M);
            // plane 3: (even row, odd col) — (r2+1, c2)
            copyElem(P, baseP + 2 * Mp * Np, cfa, (r2+1) + c2     * M);
            // plane 4: (even row, even col)— (r2+1, c2+1)
            copyElem(P, baseP + 3 * Mp * Np, cfa, (r2+1) + (c2+1) * M);
        }
    }
    return P;
}

Value planar2raw(const Value &I, keel::Allocator *mr)
{
    planar2raw_argcheck(I);
    const auto &d = I.dims();
    if (d.ndims() < 3 || d.dim(2) != 4)
        throw Error("planar2raw: I must be (M)×(N)×4",
                    0, 0, "planar2raw", "", "numkit:planar2raw:shape");
    const keel::usize Mp = d.dim(0);
    const keel::usize Np = d.dim(1);
    if (I.numel() != Mp * Np * 4)
        throw Error("planar2raw: I must be (M)×(N)×4",
                    0, 0, "planar2raw", "", "numkit:planar2raw:shape");

    const keel::usize M = 2 * Mp;
    const keel::usize N = 2 * Np;
    const ValueType T = I.type();
    Value cfa = Value::matrix(M, N, T, mr);

    for (keel::usize cp = 0; cp < Np; ++cp) {
        for (keel::usize rp = 0; rp < Mp; ++rp) {
            const keel::usize r2 = 2 * rp;
            const keel::usize c2 = 2 * cp;
            const keel::usize baseP = rp + cp * Mp;
            copyElem(cfa, r2     + c2     * M, I, baseP + 0 * Mp * Np);
            copyElem(cfa, r2     + (c2+1) * M, I, baseP + 1 * Mp * Np);
            copyElem(cfa, (r2+1) + c2     * M, I, baseP + 2 * Mp * Np);
            copyElem(cfa, (r2+1) + (c2+1) * M, I, baseP + 3 * Mp * Np);
        }
    }
    return cfa;
}

} // namespace numkit::image
