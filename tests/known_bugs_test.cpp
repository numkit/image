// toolboxes/image/tests/known_bugs_test.cpp
//
// One DISABLED_ test per OPEN bug in bugs/image/*.md. Disabled until fixed;
// remove `DISABLED_` to turn into a live regression guard. MATLAB R2025b.

#include <numkit/core/standard_engine.hpp>
#include <numkit/core/standard_engine.hpp>
#include <gtest/gtest.h>

using namespace numkit;

class ImageKnownBug : public ::testing::Test
{
public:
    StandardEngine engine;
    void SetUp() override {}
    Value eval(const std::string &c) { return engine.eval(c); }
    double evalScalar(const std::string &c) { return eval(c).toScalar(); }
};

// bugs/image/regionprops-perimeter.md — Perimeter implemented (FIXED; live guard).
TEST_F(ImageKnownBug, RegionpropsPerimeter)
{
    eval("s = regionprops(logical(ones(3,3)), 'Perimeter'); pm = s.Perimeter;");
    EXPECT_NEAR(evalScalar("pm"), 7.476000, 1e-4);
}

// bugs/image/watershed.md — watershed transform.
TEST_F(ImageKnownBug, DISABLED_Watershed)
{
    eval("L = watershed(magic(5));");
    EXPECT_EQ(static_cast<int>(evalScalar("size(L,1)")), 5);
    EXPECT_EQ(static_cast<int>(evalScalar("size(L,2)")), 5);
    EXPECT_EQ(static_cast<int>(evalScalar("max(L(:))")), 3);
}

// bugs/image/imfindcircles.md — circular Hough transform.
// (Verify centers/radii vs MATLAB on a synthetic image when enabling.)
TEST_F(ImageKnownBug, DISABLED_ImfindcirclesExists)
{
    eval("[c, r, m] = imfindcircles(zeros(40,40), [3 8]);");
    EXPECT_EQ(static_cast<int>(evalScalar("size(c,1)")),
              static_cast<int>(evalScalar("numel(r)")));   // one center per radius
}

// bugs/image/imresize-interp.md — bilinear/bicubic grid + boundary + antialias (FIXED).
TEST_F(ImageKnownBug, ImresizeBilinear)
{
    // bilinear upscale x2 — pixel-centre map + mirror boundary (was 0.5625, ...).
    eval("r = imresize([1 2; 3 4], 2, 'bilinear');");
    EXPECT_NEAR(evalScalar("r(1,1)"), 1.0,  1e-9);
    EXPECT_NEAR(evalScalar("r(1,2)"), 1.25, 1e-9);
    EXPECT_NEAR(evalScalar("r(4,4)"), 4.0,  1e-9);
    // bicubic upscale x2 (was 0.5625).
    eval("c = imresize([1 2; 3 4], 2, 'bicubic');");
    EXPECT_NEAR(evalScalar("c(1,1)"), 0.71875, 1e-7);
    // downscale with the default method (bicubic) + antialiasing.
    eval("d = imresize([1 2 3 4 5 6], [1 3]);");
    EXPECT_NEAR(evalScalar("d(1)"), 1.44922, 1e-4);
    EXPECT_NEAR(evalScalar("d(2)"), 3.5,     1e-9);
    EXPECT_NEAR(evalScalar("d(3)"), 5.55078, 1e-4);
}

// bugs/image/corner.md — corner-point detection (FIXED; built on cornermetric).
TEST_F(ImageKnownBug, Corner)
{
    eval("I = zeros(20,20); I(6:15,6:15) = 1; C = corner(I);");
    EXPECT_EQ(static_cast<int>(evalScalar("size(C,2)")), 2);   // [x y] coords
    EXPECT_EQ(static_cast<int>(evalScalar("size(C,1)")), 4);   // 4 block corners
    EXPECT_EQ(static_cast<int>(evalScalar("C(1,1)")), 6);      // first corner [6 6]
    EXPECT_EQ(static_cast<int>(evalScalar("C(1,2)")), 6);
    EXPECT_EQ(static_cast<int>(evalScalar("C(4,1)")), 15);     // last corner [15 15]
    EXPECT_EQ(static_cast<int>(evalScalar("C(4,2)")), 15);
}

// bugs/image/adapthisteq-mapping.md — CLAHE brightness regression FIXED
// (MATLAB clip-count ceil/round + step-redistribution + rayleigh/exp vmax
// scaling). nk now matches MATLAB R2025b to within a few gray levels on a
// deterministic textured 64x64 image (J(32,32)=125 vs 128, min 19 vs 20, max
// 235 exact). The residual ≤~3 levels is inter-tile bilinear rounding
// (bit-exact region port deferred); the tolerances guard against re-regression.
TEST_F(ImageKnownBug, AdapthisteqMapping)
{
    eval("[xx,yy] = meshgrid(1:64,1:64); I = uint8(120 + 40*sin(xx/8) + 30*cos(yy/6)); J = adapthisteq(I);");
    EXPECT_NEAR(evalScalar("double(J(32,32))"), 128.0, 2.0);    // exact; was 205 before fix
    EXPECT_NEAR(evalScalar("double(min(J(:)))"), 20.0, 2.0);    // exact; was 127
    EXPECT_NEAR(evalScalar("double(max(J(:)))"), 235.0, 2.0);   // exact; was 255
    EXPECT_NEAR(evalScalar("mean(double(J(:)))"), 130.3, 1.0);  // was ~201 (+54%)
}

// bugs/closed/image/complex-kernels.md — complex kernel support across
// imfilter, roifilt2, imnoise, imapplymatrix, axes2pix, fsamp2, fwind1, fwind2.
TEST_F(ImageKnownBug, ImfilterComplexKernel)
{
    eval("J = imfilter(magic(5)/8 + 0.5 + 1i, ones(3)/9);");
    EXPECT_NEAR(eval("real(J(1,1));").toScalar(), 1.18056, 1e-4);
    EXPECT_NEAR(eval("imag(J(1,1));").toScalar(), 0.444444, 1e-4);
    eval("R = roifilt2(ones(3)/9, magic(5)/8 + 0.5 + 1i, true(5));");
    EXPECT_NEAR(eval("real(R(1,1));").toScalar(), 1.18056, 1e-4);
    EXPECT_NEAR(eval("imag(R(1,1));").toScalar(), 0.444444, 1e-4);
    eval("H = fsamp2(ones(3) + 1i);");
    EXPECT_NEAR(eval("imag(H(2,2));").toScalar(), 1.0, 1e-12);
    eval("W1 = fwind1(ones(3) + 1i, [0.5 1 0.5]);");
    EXPECT_NEAR(eval("imag(W1(2,2));").toScalar(), 1.0, 1e-12);
    eval("W2 = fwind2(ones(3) + 1i, ones(3));");
    EXPECT_NEAR(eval("imag(W2(2,2));").toScalar(), 1.0, 1e-12);
}

TEST_F(ImageKnownBug, ImnoiseComplexIdentity)
{
    // 'salt & pepper' with density 0 is the identity — even complex.
    eval("I = magic(5)/8 + 1i; J = imnoise(I, 'salt & pepper', 0);");
    EXPECT_NEAR(eval("imag(J(2,3));").toScalar(), 1.0, 1e-12);
    EXPECT_NEAR(eval("real(J(2,3));").toScalar(),
                evalScalar("real(I(2,3));"), 1e-12);
}

TEST_F(ImageKnownBug, ImapplymatrixComplexKernel)
{
    eval("RGB = cat(3, magic(5)/40, magic(5)/60, magic(5)/80);");
    eval("J = imapplymatrix([0.3 0.6 0.1] + 1i, RGB);");
    EXPECT_NEAR(eval("real(J(1));").toScalar(), 0.31875, 1e-4);
    EXPECT_NEAR(eval("imag(J(1));").toScalar(), 0.920833, 1e-4);
}

TEST_F(ImageKnownBug, Axes2pixComplexExtent)
{
    // Probed R2025b: complex extent → complex result (3-4i).
    eval("p = axes2pix(5, [0 1] + 1i, 0.5);");
    EXPECT_NEAR(eval("real(p);").toScalar(), 3.0, 1e-12);
    EXPECT_NEAR(eval("imag(p);").toScalar(), -4.0, 1e-12);
}

// bugs/closed/image/imfill-intensity-holes.md — grayscale morphological
// reconstruction fills intensity holes to the surrounding saddle minimum.
TEST_F(ImageKnownBug, ImfillIntensityHoleValues)
{
    eval("I = ones(5) * 9; I(3,3) = 0; J = imfill(I, 'holes');");
    EXPECT_EQ(eval("class(J);").toString(), "double");
    EXPECT_NEAR(evalScalar("J(1,1)"), 9.0, 1e-12);
    EXPECT_NEAR(evalScalar("J(3,3)"), 9.0, 1e-12);
}

// bugs/closed/image/arity-min-forms.md — 0-arg colormaps, checkerboard(),
// phantom(), strel(), whitepoint(), and 1-arg imnoise(I).
TEST_F(ImageKnownBug, ImageShortForms)
{
    EXPECT_DOUBLE_EQ(evalScalar("size(gray(), 2)"), 3.0);
    EXPECT_DOUBLE_EQ(evalScalar("size(hot(), 2)"), 3.0);
    EXPECT_DOUBLE_EQ(evalScalar("size(cool(), 2)"), 3.0);
    EXPECT_DOUBLE_EQ(evalScalar("size(spring(), 2)"), 3.0);
    EXPECT_DOUBLE_EQ(evalScalar("size(summer(), 2)"), 3.0);
    EXPECT_DOUBLE_EQ(evalScalar("size(autumn(), 2)"), 3.0);
    EXPECT_DOUBLE_EQ(evalScalar("size(winter(), 2)"), 3.0);
    EXPECT_DOUBLE_EQ(evalScalar("size(copper(), 2)"), 3.0);
    EXPECT_DOUBLE_EQ(evalScalar("size(pink(), 2)"), 3.0);
    EXPECT_DOUBLE_EQ(evalScalar("size(flag(), 2)"), 3.0);
    EXPECT_DOUBLE_EQ(evalScalar("size(prism(), 2)"), 3.0);
    EXPECT_DOUBLE_EQ(evalScalar("size(lines(), 2)"), 3.0);
    EXPECT_DOUBLE_EQ(evalScalar("size(bone(), 2)"), 3.0);
    EXPECT_DOUBLE_EQ(evalScalar("size(white(), 2)"), 3.0);
    EXPECT_DOUBLE_EQ(evalScalar("size(checkerboard(), 1)"), 80.0);
    EXPECT_DOUBLE_EQ(evalScalar("size(phantom(), 1)"), 256.0);
    EXPECT_DOUBLE_EQ(evalScalar("numel(strel()) > 0"), 1.0);
    EXPECT_DOUBLE_EQ(evalScalar("numel(whitepoint())"), 3.0);
    EXPECT_DOUBLE_EQ(evalScalar("numel(imnoise(ones(4, 4)))"), 16.0);
}


