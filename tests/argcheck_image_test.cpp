// toolboxes/image/tests/argcheck_image_test.cpp
//
// Image-domain argument contracts (numkit::argcheck migration, 2026-09-17),
// every class reference-probed against R2025b the same day:
//   * connectivity is an integer choice set {4, 8} (5 / 2.5 / 0 all
//     rejected as one class by the reference);
//   * bit depths are the sensor family {8, 10, 12, 16} (demosaic probed:
//     10 and 12 accepted, 2.5 rejected);
//   * makelut/bwlookup n is exactly {2, 3};
//   * gray2ind n is an integer in [1, 65536];
//   * phantom takes creation-family size inputs (integer-only errors,
//     clamp-to-empty);
//   * wcodemat nb ROUNDS fractional values (reference-tolerant).

#include <numkit/core/standard_engine.hpp>
#include <numkit/core/engine.hpp>
#include <numkit/value/error.hpp>
#include <gtest/gtest.h>

using namespace numkit;

class ArgcheckImageTest : public ::testing::Test
{
public:
    StandardEngine engine;

    double evalScalar(const std::string &c) { return engine.eval(c).toScalar(); }

    void expectError(const std::string &code, const std::string &id)
    {
        try {
            engine.eval(code);
            FAIL() << "expected error not thrown: " << code;
        } catch (const Error &e) {
            EXPECT_EQ(e.identifier(), id) << code << "\n  got: " << e.what();
        }
    }
};

TEST_F(ArgcheckImageTest, ConnectivityChoiceSet)
{
    expectError("imregionalmax(magic(5), 5);", "numkit:imregionalmax:connNotChoice");
    expectError("imregionalmax(magic(5), 2.5);", "numkit:imregionalmax:connNotInteger");
    expectError("bweuler(logical(ones(4)), 2.5);", "numkit:bweuler:connNotInteger");
    expectError("boundarymask(zeros(5), 5);", "numkit:boundarymask:connNotChoice");
    // Happy path: both legal values stay accepted.
    EXPECT_EQ(static_cast<int>(evalScalar("numel(imregionalmax(magic(5), 4))")), 25);
    EXPECT_EQ(static_cast<int>(evalScalar("numel(bweuler(logical(eye(3)), 8))")), 1);
}

TEST_F(ArgcheckImageTest, BitDepthChoiceSet)
{
    expectError("rgbwide2ycbcr([0.5 0.5 0.5], 9);", "numkit:rgbwide2ycbcr:bpsNotChoice"); // 12.5 fractional first
    expectError("rgbwide2ycbcr([0.5 0.5 0.5], 12.5);", "numkit:rgbwide2ycbcr:bpsNotInteger");
    expectError("demosaic(uint8(zeros(16,16)), 'rggb', 'BitsPerSample', 2.5);",
                "numkit:demosaic:BitsPerSampleNotInteger");
    // 10 and 12 are legal (reference-probed).
    EXPECT_NO_THROW(engine.eval("demosaic(uint8(ones(16,16)*128), 'rggb', 'BitsPerSample', 12);"));
}

TEST_F(ArgcheckImageTest, LutSizeChoiceSet)
{
    expectError("makelut(@(x) 1, 9);", "numkit:makelut:nNotChoice");
    expectError("makelut(@(x) 1, 2.5);", "numkit:makelut:nNotInteger");
    EXPECT_EQ(static_cast<int>(evalScalar("numel(makelut(@(x) 1, 2))")), 16);
}

TEST_F(ArgcheckImageTest, CountContracts)
{
    expectError("gray2ind(magic(5), 2.5);", "numkit:gray2ind:nNotInteger");
    expectError("gray2ind(magic(5), 0);", "numkit:gray2ind:nOutOfRange");
    expectError("gray2ind(magic(5), 70000);", "numkit:gray2ind:nOutOfRange");
    expectError("checkerboard(2, 2.5);", "numkit:checkerboard:MNotInteger");
    expectError("phantom(2.5);", "numkit:phantom:sizeNotInteger");
    expectError("entropy(magic(5), 2.5);", "numkit:entropy:maxnargin");
    expectError("graycomatrix(magic(5), 'NumLevels', 2.5);", "numkit:graycomatrix:NumLevelsNotInteger");
    expectError("grayconnected(zeros(5), 2.5, 1);", "numkit:grayconnected:rowNotInteger");
    // Happy paths.
    EXPECT_EQ(static_cast<int>(evalScalar("numel(gray2ind(rand(5), 8))")), 25);
    EXPECT_EQ(static_cast<int>(evalScalar("numel(phantom(64))")), 4096);
    EXPECT_NO_THROW(engine.eval("wcodemat(magic(5), 20.4);")); // tolerant: rounds
}

TEST_F(ArgcheckImageTest, ShortVectorAndNonNumericGuards)
{
    expectError("imcrop(ones(4), [1 1]);", "numkit:imcrop:rect");
    expectError("imcrop(ones(4), 'abcd');", "numkit:imcrop:rect");
    expectError("imtranslate(ones(4), 1);", "numkit:imtranslate:vec");
    expectError("imtranslate(ones(4), 'ab');", "numkit:imtranslate:vec");
    expectError("imresize(ones(4), [2 2 2]);", "numkit:imresize:size");
    expectError("imresize(ones(4), 'ab');", "numkit:imresize:size");
    expectError("imrotate3(ones(3,3,3), 45, [1 0]);", "numkit:imrotate3:axis");
    expectError("imrotate3(ones(3,3,3), 45, 'xyz');", "numkit:imrotate3:axis");
    expectError("imadjust(ones(4), 0.5);", "numkit:imadjust:inRange");
    expectError("imadjust(ones(4), [0 1], 0.5);", "numkit:imadjust:outRange");
    expectError("adapthisteq(ones(8), 'NumTiles', 'ab');", "numkit:adapthisteq:badNumTiles");
    expectError("mat2gray(ones(4), 'ab');", "numkit:mat2gray:size");
    expectError("convmtx2(ones(2), 'ab');", "numkit:convmtx2:size");
    expectError("medfilt3(ones(3,3,3), [3 3]);", "numkit:medfilt3:badSize");
    expectError("col2im(ones(4,4), [], [4 4]);", "numkit:col2im:size");
    expectError("graycomatrix(ones(4), 'Offset', 'ab');", "numkit:graycomatrix:badOffset");
}
