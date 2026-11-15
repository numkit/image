// toolboxes/image/tests/png_codec_test.cpp
//
// Tests for in-tree PNG codec (8-bit, 16-bit, Gray, RGB, RGBA, filters).

#include <keel/core.h>
#include <gtest/gtest.h>
#include "png_codec.hpp"
#include <numkit/value/value.hpp>

#include <random>
#include <vector>

using namespace numkit;
using namespace numkit::image;

TEST(PngCodecTest, Gray8RoundTrip) {
    const keel::usize H = 25, W = 33;
    Value orig = Value::matrix(H, W, ValueType::UINT8);
    for (keel::usize y = 0; y < H; ++y) {
        for (keel::usize x = 0; x < W; ++x) {
            orig.uint8DataMut()[x * H + y] = static_cast<keel::u8>((x * 13 + y * 17) % 256);
        }
    }

    std::string pngBytes = writePngToBytes(orig, 6);
    EXPECT_GT(pngBytes.size(), 8u + 25u + 12u + 12u);

    keel::u32 w32 = 0, h32 = 0;
    keel::u16 bits = 0, channels = 0;
    EXPECT_TRUE(peekPng(reinterpret_cast<const keel::u8*>(pngBytes.data()), pngBytes.size(), w32, h32, bits, channels));
    EXPECT_EQ(w32, W);
    EXPECT_EQ(h32, H);
    EXPECT_EQ(bits, 8);
    EXPECT_EQ(channels, 1);

    Value readBack = readPng(reinterpret_cast<const keel::u8*>(pngBytes.data()), pngBytes.size());
    EXPECT_EQ(readBack.dims().rows(), H);
    EXPECT_EQ(readBack.dims().cols(), W);
    EXPECT_EQ(readBack.type(), ValueType::UINT8);

    for (keel::usize i = 0; i < H * W; ++i) {
        EXPECT_EQ(readBack.uint8Data()[i], orig.uint8Data()[i]);
    }
}

TEST(PngCodecTest, Rgb8AndRgba8RoundTrip) {
    const keel::usize H = 20, W = 20;

    // RGB 3 channels
    Value rgb = Value::matrix3d(H, W, 3, ValueType::UINT8);
    for (keel::usize i = 0; i < H * W * 3; ++i) {
        rgb.uint8DataMut()[i] = static_cast<keel::u8>((i * 31) % 256);
    }
    std::string rgbBytes = writePngToBytes(rgb, 6);
    Value rgbBack = readPng(reinterpret_cast<const keel::u8*>(rgbBytes.data()), rgbBytes.size());
    EXPECT_EQ(rgbBack.dims().pages(), 3u);
    EXPECT_EQ(rgbBack.type(), ValueType::UINT8);
    for (keel::usize i = 0; i < H * W * 3; ++i) {
        EXPECT_EQ(rgbBack.uint8Data()[i], rgb.uint8Data()[i]);
    }

    // RGBA 4 channels
    Value rgba = Value::matrix3d(H, W, 4, ValueType::UINT8);
    for (keel::usize i = 0; i < H * W * 4; ++i) {
        rgba.uint8DataMut()[i] = static_cast<keel::u8>((i * 47) % 256);
    }
    std::string rgbaBytes = writePngToBytes(rgba, 6);
    Value rgbaBack = readPng(reinterpret_cast<const keel::u8*>(rgbaBytes.data()), rgbaBytes.size());
    EXPECT_EQ(rgbaBack.dims().pages(), 4u);
    EXPECT_EQ(rgbaBack.type(), ValueType::UINT8);
    for (keel::usize i = 0; i < H * W * 4; ++i) {
        EXPECT_EQ(rgbaBack.uint8Data()[i], rgba.uint8Data()[i]);
    }
}

TEST(PngCodecTest, Gray16AndRgb16PrecisionRoundTrip) {
    const keel::usize H = 18, W = 22;

    // 16-bit Grayscale
    Value gray16 = Value::matrix(H, W, ValueType::UINT16);
    for (keel::usize i = 0; i < H * W; ++i) {
        gray16.uint16DataMut()[i] = static_cast<keel::u16>(i * 123);
    }
    std::string gray16Bytes = writePngToBytes(gray16, 6);
    Value gray16Back = readPng(reinterpret_cast<const keel::u8*>(gray16Bytes.data()), gray16Bytes.size());
    EXPECT_EQ(gray16Back.type(), ValueType::UINT16);
    EXPECT_EQ(gray16Back.dims().rows(), H);
    EXPECT_EQ(gray16Back.dims().cols(), W);
    for (keel::usize i = 0; i < H * W; ++i) {
        EXPECT_EQ(gray16Back.uint16Data()[i], gray16.uint16Data()[i]);
    }

    // 16-bit RGB
    Value rgb16 = Value::matrix3d(H, W, 3, ValueType::UINT16);
    for (keel::usize i = 0; i < H * W * 3; ++i) {
        rgb16.uint16DataMut()[i] = static_cast<keel::u16>(i * 57 + 1000);
    }
    std::string rgb16Bytes = writePngToBytes(rgb16, 6);
    Value rgb16Back = readPng(reinterpret_cast<const keel::u8*>(rgb16Bytes.data()), rgb16Bytes.size());
    EXPECT_EQ(rgb16Back.type(), ValueType::UINT16);
    EXPECT_EQ(rgb16Back.dims().pages(), 3u);
    for (keel::usize i = 0; i < H * W * 3; ++i) {
        EXPECT_EQ(rgb16Back.uint16Data()[i], rgb16.uint16Data()[i]);
    }
}
