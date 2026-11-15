// toolboxes/image/tests/codecs_robustness_test.cpp
//
// Stress tests, corrupt buffer handling, and edge cases for in-tree codecs.

#include <keel/core.h>
#include <gtest/gtest.h>
#include "deflate.hpp"
#include "bmp_codec.hpp"
#include "tga_codec.hpp"
#include "pnm_codec.hpp"
#include "png_codec.hpp"
#include "jpeg_codec.hpp"
#include <numkit/value/value.hpp>
#include <numkit/value/error.hpp>

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

using namespace numkit;
using namespace numkit::image;

// ============================================================================
// 1. Truncated & Corrupt Buffers Resilience
// ============================================================================

TEST(CodecsRobustnessTest, EmptyAndTinyBuffersThrowCleanly) {
    std::vector<keel::u8> empty;
    std::vector<keel::u8> tiny = { 0x89, 'P', 'N' };

    EXPECT_THROW(readPng(empty.data(), empty.size()), Error);
    EXPECT_THROW(readPng(tiny.data(), tiny.size()), Error);

    EXPECT_THROW(readJpeg(empty.data(), empty.size()), Error);
    EXPECT_THROW(readJpeg(tiny.data(), tiny.size()), Error);

    EXPECT_THROW(readBmp(empty.data(), empty.size()), Error);
    EXPECT_THROW(readBmp(tiny.data(), tiny.size()), Error);

    EXPECT_THROW(readTga(empty.data(), empty.size()), Error);
    EXPECT_THROW(readTga(tiny.data(), tiny.size()), Error);

    EXPECT_THROW(readPnm(empty.data(), empty.size()), Error);
    EXPECT_THROW(readPnm(tiny.data(), tiny.size()), Error);
}

TEST(CodecsRobustnessTest, TruncatedPngAndBadCrc) {
    Value img = Value::matrix(10, 10, ValueType::UINT8);
    for (keel::usize i = 0; i < 100; ++i) img.uint8DataMut()[i] = static_cast<keel::u8>(i);

    keel::String pngBytes = writePngToBytes(img);
    ASSERT_GT(pngBytes.size(), 20u);

    // 1. Truncated PNG (cut off in the middle of IDAT)
    EXPECT_THROW(readPng(reinterpret_cast<const keel::u8*>(pngBytes.data()), pngBytes.size() / 2), Error);

    // 2. Corrupt IDAT chunk CRC (modify a byte inside the IDAT payload)
    keel::String corrupt = pngBytes.clone();
    corrupt[corrupt.size() - 20] ^= 0xFF;
    // Inflate or chunk validation must detect corruption cleanly
    EXPECT_THROW(readPng(reinterpret_cast<const keel::u8*>(corrupt.data()), corrupt.size()), Error);
}

TEST(CodecsRobustnessTest, DeflateCorruptedAdler32) {
    std::vector<keel::u8> data = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 };
    keel::Array<keel::u8> compressed = zlibCompress(data.data(), data.size());

    ASSERT_GE(compressed.size(), 6u);
    // Corrupt Adler-32 checksum (last 4 bytes of zlib stream)
    compressed[compressed.size() - 1] ^= 0xFF;

    EXPECT_THROW(zlibDecompress(compressed.data(), compressed.size(), 10), Error);
}

// ============================================================================
// 2. Extreme Dimensions (1x1, 1xN, Nx1, Odd Strides)
// ============================================================================

TEST(CodecsRobustnessTest, ExtremeDimensionsRoundTrip) {
    std::vector<std::pair<keel::usize, keel::usize>> shapes = {
        {1, 1},
        {1, 17},
        {19, 1},
        {3, 5},
        {7, 13},
        {33, 47}
    };

    for (const auto &[H, W] : shapes) {
        // Grayscale
        Value gray = Value::matrix(H, W, ValueType::UINT8);
        for (keel::usize y = 0; y < H; ++y) {
            for (keel::usize x = 0; x < W; ++x) {
                gray.uint8DataMut()[x * H + y] = static_cast<keel::u8>((x * 17 + y * 31) % 256);
            }
        }

        // PNG
        keel::String pngData = writePngToBytes(gray);
        Value pngBack = readPng(reinterpret_cast<const keel::u8*>(pngData.data()), pngData.size());
        EXPECT_EQ(pngBack.dims().rows(), H);
        EXPECT_EQ(pngBack.dims().cols(), W);
        for (keel::usize i = 0; i < H * W; ++i) EXPECT_EQ(pngBack.uint8Data()[i], gray.uint8Data()[i]);

        // BMP (tests 4-byte row padding on odd strides)
        keel::String bmpData = writeBmpToBytes(gray);
        Value bmpBack = readBmp(reinterpret_cast<const keel::u8*>(bmpData.data()), bmpData.size());
        EXPECT_EQ(bmpBack.dims().rows(), H);
        EXPECT_EQ(bmpBack.dims().cols(), W);
        for (keel::usize i = 0; i < H * W; ++i) EXPECT_EQ(bmpBack.uint8Data()[i], gray.uint8Data()[i]);

        // TGA
        keel::String tgaData = writeTgaToBytes(gray);
        Value tgaBack = readTga(reinterpret_cast<const keel::u8*>(tgaData.data()), tgaData.size());
        EXPECT_EQ(tgaBack.dims().rows(), H);
        EXPECT_EQ(tgaBack.dims().cols(), W);
        for (keel::usize i = 0; i < H * W; ++i) EXPECT_EQ(tgaBack.uint8Data()[i], gray.uint8Data()[i]);

        // PNM PGM
        keel::String pnmData = writePnmToBytes(gray);
        Value pnmBack = readPnm(reinterpret_cast<const keel::u8*>(pnmData.data()), pnmData.size());
        EXPECT_EQ(pnmBack.dims().rows(), H);
        EXPECT_EQ(pnmBack.dims().cols(), W);
        for (keel::usize i = 0; i < H * W; ++i) EXPECT_EQ(pnmBack.uint8Data()[i], gray.uint8Data()[i]);
    }
}

// ============================================================================
// 3. PNG Reconstruction Filters (None, Sub, Up, Average, Paeth)
// ============================================================================

TEST(CodecsRobustnessTest, PngAllFiveFilterTypes) {
    const keel::usize H = 5, W = 4;
    // Build a 5-row image where each row exercises a different filter
    // Construct uncompressed scanline buffer: [filter_byte, 4 raw pixel bytes] * 5
    std::vector<keel::u8> scanlines;
    scanlines.reserve(5 * 5);

    // Row 0: None (filter 0)
    scanlines.push_back(0);
    scanlines.push_back(10); scanlines.push_back(20); scanlines.push_back(30); scanlines.push_back(40);

    // Row 1: Sub (filter 1) -> Raw = [15, 25, 35, 45], Filt = [15, 10, 10, 10]
    scanlines.push_back(1);
    scanlines.push_back(15); scanlines.push_back(10); scanlines.push_back(10); scanlines.push_back(10);

    // Row 2: Up (filter 2) -> Raw = [25, 35, 45, 55], Prior = [15, 25, 35, 45] -> Filt = [10, 10, 10, 10]
    scanlines.push_back(2);
    scanlines.push_back(10); scanlines.push_back(10); scanlines.push_back(10); scanlines.push_back(10);

    // Row 3: Average (filter 3)
    scanlines.push_back(3);
    scanlines.push_back(5); scanlines.push_back(5); scanlines.push_back(5); scanlines.push_back(5);

    // Row 4: Paeth (filter 4)
    scanlines.push_back(4);
    scanlines.push_back(2); scanlines.push_back(2); scanlines.push_back(2); scanlines.push_back(2);

    keel::Array<keel::u8> compressedIdat = zlibCompress(scanlines.data(), scanlines.size());

    // Assemble valid PNG container
    std::vector<keel::u8> png;
    const keel::u8 sig[8] = { 0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A };
    png.insert(png.end(), sig, sig + 8);

    auto appendChunk = [&](const char type[4], const keel::u8 *payload, keel::usize len) {
        keel::u32 l = static_cast<keel::u32>(len);
        png.push_back((l >> 24) & 0xFF);
        png.push_back((l >> 16) & 0xFF);
        png.push_back((l >> 8) & 0xFF);
        png.push_back(l & 0xFF);
        keel::usize typePos = png.size();
        png.push_back(type[0]); png.push_back(type[1]); png.push_back(type[2]); png.push_back(type[3]);
        if (len > 0 && payload) png.insert(png.end(), payload, payload + len);
        keel::u32 c = crc32(png.data() + typePos, 4 + len);
        png.push_back((c >> 24) & 0xFF);
        png.push_back((c >> 16) & 0xFF);
        png.push_back((c >> 8) & 0xFF);
        png.push_back(c & 0xFF);
    };

    keel::u8 ihdr[13] = {
        0, 0, 0, 4, // W = 4
        0, 0, 0, 5, // H = 5
        8,          // 8-bit
        0,          // Grayscale
        0, 0, 0     // Deflate, Standard filter, No interlace
    };
    appendChunk("IHDR", ihdr, 13);
    appendChunk("IDAT", compressedIdat.data(), compressedIdat.size());
    appendChunk("IEND", nullptr, 0);

    Value decoded = readPng(png.data(), png.size());
    EXPECT_EQ(decoded.dims().rows(), H);
    EXPECT_EQ(decoded.dims().cols(), W);

    // Verify row 0 (None filter)
    EXPECT_EQ(decoded.uint8Data()[0 * H + 0], 10);
    EXPECT_EQ(decoded.uint8Data()[1 * H + 0], 20);
    EXPECT_EQ(decoded.uint8Data()[2 * H + 0], 30);
    EXPECT_EQ(decoded.uint8Data()[3 * H + 0], 40);

    // Verify row 1 (Sub filter)
    EXPECT_EQ(decoded.uint8Data()[0 * H + 1], 15);
    EXPECT_EQ(decoded.uint8Data()[1 * H + 1], 25);
    EXPECT_EQ(decoded.uint8Data()[2 * H + 1], 35);
    EXPECT_EQ(decoded.uint8Data()[3 * H + 1], 45);

    // Verify row 2 (Up filter)
    EXPECT_EQ(decoded.uint8Data()[0 * H + 2], 25);
    EXPECT_EQ(decoded.uint8Data()[1 * H + 2], 35);
    EXPECT_EQ(decoded.uint8Data()[2 * H + 2], 45);
    EXPECT_EQ(decoded.uint8Data()[3 * H + 2], 55);
}

// ============================================================================
// 4. Netpbm ASCII with Comments & Irregular Whitespace
// ============================================================================

TEST(CodecsRobustnessTest, PnmAsciiWithCommentsAndWhitespace) {
    // P2 with comments between dimensions and sample values
    std::string pgmAscii =
        "P2\n"
        "# Created by test suite\n"
        "3 2 # width height\n"
        "# max value comment\n"
        "255\n"
        " 10   20 \t 30\n"
        "# row 2\n"
        "40   50   60\n";

    Value pgm = readPnm(reinterpret_cast<const keel::u8*>(pgmAscii.data()), pgmAscii.size());
    EXPECT_EQ(pgm.dims().rows(), 2u);
    EXPECT_EQ(pgm.dims().cols(), 3u);
    EXPECT_EQ(pgm.uint8Data()[0 * 2 + 0], 10);
    EXPECT_EQ(pgm.uint8Data()[1 * 2 + 0], 20);
    EXPECT_EQ(pgm.uint8Data()[2 * 2 + 0], 30);
    EXPECT_EQ(pgm.uint8Data()[0 * 2 + 1], 40);
    EXPECT_EQ(pgm.uint8Data()[1 * 2 + 1], 50);
    EXPECT_EQ(pgm.uint8Data()[2 * 2 + 1], 60);

    // P3 PPM with multiple comments
    std::string ppmAscii =
        "P3\n"
        "# PPM ASCII\n"
        "2 1\n"
        "255\n"
        "255 0 0 # Red\n"
        "0 255 0 # Green\n";

    Value ppm = readPnm(reinterpret_cast<const keel::u8*>(ppmAscii.data()), ppmAscii.size());
    EXPECT_EQ(ppm.dims().rows(), 1u);
    EXPECT_EQ(ppm.dims().cols(), 2u);
    EXPECT_EQ(ppm.dims().pages(), 3u);
    const keel::usize plane = 2;
    // Pixel 0 (Red)
    EXPECT_EQ(ppm.uint8Data()[0 * 1 + 0], 255);
    EXPECT_EQ(ppm.uint8Data()[plane + 0 * 1 + 0], 0);
    EXPECT_EQ(ppm.uint8Data()[2 * plane + 0 * 1 + 0], 0);
    // Pixel 1 (Green)
    EXPECT_EQ(ppm.uint8Data()[1 * 1 + 0], 0);
    EXPECT_EQ(ppm.uint8Data()[plane + 1 * 1 + 0], 255);
    EXPECT_EQ(ppm.uint8Data()[2 * plane + 1 * 1 + 0], 0);
}
