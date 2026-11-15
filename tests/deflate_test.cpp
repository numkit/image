// toolboxes/image/tests/deflate_test.cpp
//
// Tests for in-tree Deflate, Inflate, Zlib, CRC-32, and Adler-32.

#include <keel/core.h>
#include <gtest/gtest.h>
#include "deflate.hpp"
#include <random>
#include <string>
#include <vector>

using namespace numkit::image;

TEST(DeflateTest, Crc32AndAdler32KnownVectors) {
    const std::string text = "123456789";
    EXPECT_EQ(crc32(reinterpret_cast<const keel::u8*>(text.data()), text.size()), 0xCBF43926u);
    EXPECT_EQ(adler32(reinterpret_cast<const keel::u8*>(text.data()), text.size()), 0x091E01DEu);

    const std::string empty = "";
    EXPECT_EQ(crc32(reinterpret_cast<const keel::u8*>(empty.data()), 0), 0x00000000u);
    EXPECT_EQ(adler32(reinterpret_cast<const keel::u8*>(empty.data()), 0), 0x00000001u);
}

TEST(DeflateTest, EmptyRoundTrip) {
    std::vector<keel::u8> empty;
    std::vector<keel::u8> comp = deflateRaw(empty.data(), empty.size());
    std::vector<keel::u8> decomp = inflateRaw(comp.data(), comp.size());
    EXPECT_TRUE(decomp.empty());

    std::vector<keel::u8> zcomp = zlibCompress(empty.data(), empty.size());
    std::vector<keel::u8> zdecomp = zlibDecompress(zcomp.data(), zcomp.size());
    EXPECT_TRUE(zdecomp.empty());
}

TEST(DeflateTest, SmallStringRoundTrip) {
    std::string s = "Hello world! This is NK autonomous C++ image compression test.";
    const auto *ptr = reinterpret_cast<const keel::u8*>(s.data());
    
    // Raw Deflate
    std::vector<keel::u8> comp = deflateRaw(ptr, s.size(), 6);
    std::vector<keel::u8> decomp = inflateRaw(comp.data(), comp.size());
    EXPECT_EQ(std::string(decomp.begin(), decomp.end()), s);

    // Zlib format
    std::vector<keel::u8> zcomp = zlibCompress(ptr, s.size(), 6);
    std::vector<keel::u8> zdecomp = zlibDecompress(zcomp.data(), zcomp.size());
    EXPECT_EQ(std::string(zdecomp.begin(), zdecomp.end()), s);
}

TEST(DeflateTest, HighlyRepetitiveData) {
    std::vector<keel::u8> rep(100000, 0x42);
    std::vector<keel::u8> zcomp = zlibCompress(rep.data(), rep.size(), 6);
    EXPECT_LT(zcomp.size(), 1000u); // High compression ratio

    std::vector<keel::u8> zdecomp = zlibDecompress(zcomp.data(), zcomp.size());
    EXPECT_EQ(zdecomp, rep);
}

TEST(DeflateTest, RandomAndPatternedDataRoundTrip) {
    std::mt19937 rng(1337);
    std::vector<keel::u8> data(50000);
    for (keel::usize i = 0; i < data.size(); ++i) {
        // Mixture of smooth ramps and random noise
        data[i] = static_cast<keel::u8>((i % 256) ^ (rng() % 32));
    }

    // Test multiple compression levels: 0 (store), 1 (fast), 6 (default), 9 (high)
    for (int lvl : {0, 1, 6, 9}) {
        std::vector<keel::u8> zcomp = zlibCompress(data.data(), data.size(), lvl);
        std::vector<keel::u8> zdecomp = zlibDecompress(zcomp.data(), zcomp.size());
        EXPECT_EQ(zdecomp, data);
    }
}
