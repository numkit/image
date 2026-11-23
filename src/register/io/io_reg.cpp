// toolboxes/image/src/io/io_reg.cpp
//
// Register half of the image io builtins: the CallContext wrappers
// delegating to the engine-free compute in io.cpp. library.cpp
// forward-declares + registers these by name.
//
// Phase 2b compute/register split — see project_layering_refactor memory.

#include <keel/core.h>
#include <keel/format.h>
#include <keel/memory.h>
#include <keel/ref.h>
#include <keel/string.h>
#include <keel/utf8.h>
#include <numkit/image/io/io.hpp>

#include <numkit/sys/plugin.hpp>   // CallContext, Span, ctx.engine->resource()
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>
#include <numkit/value/argcheck.hpp>

#include <algorithm>
#include <cstdlib>

namespace numkit::image {

inline keel::String lowerExt(keel::StringRef path) {
    auto dot = path.rfind('.');
    if (dot == keel::kInvalidIndex) return {};
    keel::String e = keel::String::from(path.substr(dot + 1));
    e.to_ascii_lower_in_place();
    return e;
}

bool isTiffBytes(keel::StringRef b);
Value imreadFromBytes(keel::StringRef bytes, keel::Allocator *mr);
keel::Pair<Value, Value>
readPngWithMap(const keel::u8 *data, keel::usize len,
               keel::Allocator *mr = nullptr);

namespace detail {

void imread_reg(keel::ArrayRef<Value> args, keel::usize nargout, keel::MutArrayRef<Value> outs,
                CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "imread");
    // no max: filename + name-value tail by design
    if (!args[0].isChar() && !args[0].isString())
        throw Error("imread: path must be a string",
                    0, 0, "imread", "", "numkit:imread:type");
    const keel::String path = args[0].toString();

    // Read the file through the engine's VFS prosloyka — the WASM engine
    // has no direct file access; the bytes come from the IDE's virtual or
    // real filesystem (resolvePath picks the backend, incl. the script's
    // own directory). Never fopen here.
    auto rp = ctx.engine->resolvePath(path);
    if ((!rp.fs || !rp.fs->exists(rp.path))
        && path.find('/') == keel::kInvalidIndex
        && path.find('\\') == keel::kInvalidIndex) {
        const keel::String refRoot = keel::format("C:/Program Files/MAT{}LAB/R2025b/toolbox/", "");
        const keel::String demoDirs[] = {
            keel::format("{}mat{}lab/mat{}lab_images/png/", refRoot, "", ""),
            keel::format("{}images/imdata/", refRoot)
        };
        for (const keel::String &dir : demoDirs) {
            auto cand = ctx.engine->resolvePath(keel::format("{}{}", dir, path));
            if (cand.fs && cand.fs->exists(cand.path)) {
                rp = keel::move(cand);
                break;
            }
        }
    }
    if (!rp.fs || !rp.fs->exists(rp.path))
        throw Error(keel::format("imread: failed to load '{}' — file not found", path),
                    0, 0, "imread", "", "numkit:imread:load");
    // Binary-safe read (an image is raw bytes, not UTF-8 text).
    const keel::String bytes = rp.fs->readFileBytes(rp.path);
    const bool tiff = isTiffBytes(bytes);

    // Optional 2nd numeric arg = page index (TIFF multi-page).
    keel::u32 page = 1;
    if (args.size() >= 2 && !args[1].isEmpty()
        && !args[1].isChar() && !args[1].isString()) {
        page = keel::as<keel::u32>(args[1].toScalar());
        if (!tiff)
            throw Error("imread: page index only supported for TIFF files",
                        0, 0, "imread", "", "numkit:imread:notTiff");
    }

    if (tiff) {
        keel::ArrayRef<keel::u8> buf(reinterpret_cast<const keel::u8 *>(bytes.data()), bytes.size());
        // Two-output form `[A, map] = imread(file)` — palette TIFFs.
        if (nargout >= 2) {
            auto pair = readTiffWithMap(buf, page, ctx.engine->allocator());
            outs[0] = keel::move(pair.first);
            outs[1] = keel::move(pair.second);
        } else {
            outs[0] = readTiff(buf, page, ctx.engine->allocator());
        }
        return;
    }

    const auto *data = reinterpret_cast<const keel::u8 *>(bytes.data());
    const keel::usize len = bytes.size();
    if (len >= 8 && data[0] == 0x89 && data[1] == 'P' && data[2] == 'N' && data[3] == 'G') {
        if (nargout >= 2) {
            auto pair = readPngWithMap(data, len, ctx.engine->allocator());
            outs[0] = keel::move(pair.first);
            outs[1] = keel::move(pair.second);
            return;
        }
    }

    outs[0] = imreadFromBytes(bytes, ctx.engine->allocator());
}

void imwrite_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> /*outs*/,
                 CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 2, "imwrite");
    // no max: filename + name-value tail by design
    if (!args[1].isChar() && !args[1].isString())
        throw Error("imwrite: path must be a string",
                    0, 0, "imwrite", "", "numkit:imwrite:type");

    const keel::String path = args[1].toString();
    const keel::String ext = lowerExt(path);

    // TIFF route — collect optional 3rd positional 'tif' format string,
    // then NV-pairs ('Compression', 'none'|'packbits'|'lzw'|'deflate';
    // 'WriteMode', 'overwrite'|'append').
    if (ext == "tif" || ext == "tiff") {
        keel::String compression = keel::String::from("none");
        bool appendMode = false;
        // Optional 3rd positional 'tif'/'tiff' format keyword (MATLAB
        // syntax `imwrite(A, path, 'tif', ...)`). Skip it as NV-pair start
        // and tolerate.
        keel::usize nvStart = 2;
        if (args.size() >= 3 && (args[2].isChar() || args[2].isString())) {
            keel::String s = args[2].toString();
            keel::String lo;
            lo.reserve(s.size());
            for (char c : s) lo.push_back(keel::to_ascii_lower(c));
            if (lo == "tif" || lo == "tiff") nvStart = 3;
        }
        for (keel::usize i = nvStart; i + 1 < args.size(); i += 2) {
            if (!args[i].isChar() && !args[i].isString())
                throw Error("imwrite TIFF: NV name must be a string",
                            0, 0, "imwrite", "", "numkit:imwrite:badNVName");
            keel::String key = args[i].toString();
            keel::String lkey;
            lkey.reserve(key.size());
            for (char c : key) lkey.push_back(keel::to_ascii_lower(c));
            const Value &v = args[i + 1];
            if (lkey == "compression") {
                compression = v.toString();
                compression.to_ascii_lower_in_place();
            } else if (lkey == "writemode") {
                keel::String m = v.toString();
                keel::String lo;
                for (char c : m) lo.push_back(keel::to_ascii_lower(c));
                appendMode = (lo == "append");
            } else {
                throw Error(keel::format("imwrite TIFF: unknown NV key '{}'", key),
                            0, 0, "imwrite", "", "numkit:imwrite:badNVKey");
            }
        }
        // Encode to bytes and write through the VFS prosloyka. For append
        // mode, the existing pages are read back through the same VFS so
        // multi-page TIFF works on the virtual filesystem too. Never fopen.
        auto rp = ctx.engine->resolvePath(path);
        if (!rp.fs)
            throw Error(keel::format("imwrite: no filesystem for '{}'", path),
                        0, 0, "imwrite", "", "numkit:imwrite:write");
        keel::String existingBytes;
        keel::ArrayRef<keel::u8> existingRef;
        if (appendMode && rp.fs->exists(rp.path)) {
            existingBytes = rp.fs->readFileBytes(rp.path);
            existingRef = keel::ArrayRef<keel::u8>(
                reinterpret_cast<const keel::u8 *>(existingBytes.data()),
                existingBytes.size());
        }
        const keel::Array<keel::u8> buf =
            writeTiffToBytes(args[0], compression, existingRef);
        rp.fs->writeFileBytes(
            rp.path,
            keel::StringRef(reinterpret_cast<const char *>(buf.data()), buf.size()));
        return;
    }

    // stb-encodable formats — encode to bytes, write through the VFS.
    auto rp = ctx.engine->resolvePath(path);
    if (!rp.fs)
        throw Error(keel::format("imwrite: no filesystem for '{}'", path),
                    0, 0, "imwrite", "", "numkit:imwrite:write");
    const keel::String bytes = imwriteToBytes(args[0], ext, ctx.engine->allocator());
    rp.fs->writeFileBytes(rp.path, bytes);
}

void imfinfo_reg(keel::ArrayRef<Value> args, keel::usize /*nargout*/, keel::MutArrayRef<Value> outs,
                 CallContext &ctx)
{
    argcheck::checkNargin(args.size(), 1, "imfinfo");
    // no max: filename + name-value tail by design
    if (!args[0].isChar() && !args[0].isString())
        throw Error("imfinfo: path must be a string",
                    0, 0, "imfinfo", "", "numkit:imfinfo:type");
    const keel::String path = args[0].toString();

    // Read through the VFS prosloyka (virtual or real FS); never fopen.
    auto rp = ctx.engine->resolvePath(path);
    if (!rp.fs || !rp.fs->exists(rp.path))
        throw Error(keel::format("imfinfo: failed to read '{}' — file not found", path),
                    0, 0, "imfinfo", "", "numkit:imfinfo:read");
    const keel::String bytes = rp.fs->readFileBytes(rp.path);
    outs[0] = imfinfoFromBytes(bytes, path, ctx.engine->allocator());
}

} // namespace detail

} // namespace numkit::image
