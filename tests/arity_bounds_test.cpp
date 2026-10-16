// image/tests/arity_bounds_test.cpp
//
// Table-driven max-arity guards, generated from the applied
// bounds in the image reg files (R2025b-probed): calling with
// max+1 inputs must be REJECTED — the reference errors
// "Too many input arguments", nk must never silently drop
// extras (bugs/opened/lang/reg-max-arity.md). The arity check
// is the first statement of every migrated reg, so dummy
// 1-arguments never reach the kernels.

#include <gtest/gtest.h>

#include <numkit/core/engine.hpp>
#include <numkit/bundle/standard_engine.hpp>

#include <string>
#include <utility>
#include <vector>

class ImageArityBoundsTest :
    public ::testing::TestWithParam<std::pair<const char *, int>>
{
public:
    numkit::StandardEngine engine;
};

TEST_P(ImageArityBoundsTest, ExtraInputsRejected)
{
    const auto [fn, maxArgs] = GetParam();
    std::string call = fn;
    call += "(";
    for (int i = 0; i <= maxArgs; ++i) {
        if (i) call += ", ";
        call += "1";
    }
    call += ");";
    EXPECT_ANY_THROW(engine.eval(call)) << call;
}

const std::vector<std::pair<const char *, int>> kBounds = {
    std::make_pair("adaptthresh", 8),
    std::make_pair("applylut", 2),
    std::make_pair("autumn", 1),
    std::make_pair("axes2pix", 3),
    std::make_pair("bestblk", 2),
    std::make_pair("bone", 1),
    std::make_pair("boundarymask", 2),
    std::make_pair("bwarea", 1),
    std::make_pair("bwareafilt", 4),
    std::make_pair("bwareaopen", 3),
    std::make_pair("bwboundaries", 3),
    std::make_pair("bwconncomp", 2),
    std::make_pair("bwdist", 2),
    std::make_pair("bwdistgeodesic", 4),
    std::make_pair("bweuler", 2),
    std::make_pair("bwhitmiss", 3),
    std::make_pair("bwlabel", 2),
    std::make_pair("bwlookup", 2),
    std::make_pair("bwmorph", 3),
    std::make_pair("bwpack", 1),
    std::make_pair("bwperim", 2),
    std::make_pair("bwpropfilt", 6),
    std::make_pair("bwselect", 6),
    std::make_pair("bwtraceboundary", 6),
    std::make_pair("bwunpack", 2),
    std::make_pair("checkerboard", 3),
    std::make_pair("chromadapt", 6),
    std::make_pair("cmap2gray", 1),
    std::make_pair("cmunique", 2),
    std::make_pair("convmtx2", 3),
    std::make_pair("cool", 1),
    std::make_pair("copper", 1),
    std::make_pair("corr2", 2),
    std::make_pair("dct2", 3),
    std::make_pair("dctmtx", 1),
    std::make_pair("deconvreg", 5),
    std::make_pair("deconvwnr", 4),
    std::make_pair("deltaE", 2),
    std::make_pair("edge", 5),
    std::make_pair("edgetaper", 2),
    std::make_pair("entropy", 1),
    std::make_pair("entropyfilt", 2),
    std::make_pair("fibermetric", 6),
    std::make_pair("flag", 1),
    std::make_pair("freqz2", 4),
    std::make_pair("fspecial", 3),
    std::make_pair("fspecial3", 3),
    std::make_pair("getrangefromclass", 1),
    std::make_pair("gradientweight", 6),
    std::make_pair("gray", 1),
    std::make_pair("gray2ind", 2),
    std::make_pair("graycoprops", 4),
    std::make_pair("graydiffweight", 8),
    std::make_pair("graydist", 4),
    std::make_pair("grayslice", 2),
    std::make_pair("graythresh", 1),
    std::make_pair("histeq", 3),
    std::make_pair("hot", 1),
    std::make_pair("hough", 5),
    std::make_pair("houghlines", 8),
    std::make_pair("houghpeaks", 8),
    std::make_pair("hsv", 1),
    std::make_pair("hsv2rgb", 3),
    std::make_pair("idct2", 3),
    std::make_pair("illumgray", 6),
    std::make_pair("illumpca", 4),
    std::make_pair("illumwhite", 4),
    std::make_pair("im2col", 4),
    std::make_pair("im2double", 2),
    std::make_pair("im2gray", 1),
    std::make_pair("im2int16", 1),
    std::make_pair("im2single", 2),
    std::make_pair("im2uint16", 2),
    std::make_pair("im2uint8", 2),
    std::make_pair("imabsdiff", 2),
    std::make_pair("imadd", 3),
    std::make_pair("imadjust", 4),
    std::make_pair("imapplymatrix", 4),
    std::make_pair("imbinarize", 6),
    std::make_pair("imbothat", 2),
    std::make_pair("imclearborder", 2),
    std::make_pair("imclose", 2),
    std::make_pair("imcomplement", 1),
    std::make_pair("imcrop", 5),
    std::make_pair("imcrop3", 2),
    std::make_pair("imdilate", 5),
    std::make_pair("imdivide", 2),
    std::make_pair("imerode", 5),
    std::make_pair("imextendedmax", 3),
    std::make_pair("imextendedmin", 3),
    std::make_pair("imfill", 3),
    std::make_pair("imfilter", 5),
    std::make_pair("imflatfield", 5),
    std::make_pair("imgradient", 2),
    std::make_pair("imgradient3", 3),
    std::make_pair("imgradientxy", 2),
    std::make_pair("imguidedfilter", 6),
    std::make_pair("imhist", 2),
    std::make_pair("imhistmatch", 5),
    std::make_pair("imhmax", 3),
    std::make_pair("imimposemin", 3),
    std::make_pair("immse", 2),
    std::make_pair("immultiply", 2),
    std::make_pair("imnlmfilt", 7),
    std::make_pair("imnoise", 4),
    std::make_pair("imopen", 2),
    std::make_pair("imoverlay", 3),
    std::make_pair("impyramid", 2),
    std::make_pair("imquantize", 3),
    std::make_pair("imreconstruct", 3),
    std::make_pair("imreducehaze", 10),
    std::make_pair("imregionalmax", 2),
    std::make_pair("imregionalmin", 2),
    std::make_pair("imrotate", 4),
    std::make_pair("imrotate3", 7),
    std::make_pair("imsharpen", 7),
    std::make_pair("imsubtract", 2),
    std::make_pair("imtophat", 2),
    std::make_pair("ind2gray", 2),
    std::make_pair("ind2rgb", 2),
    std::make_pair("integralImage", 2),
    std::make_pair("intlut", 2),
    std::make_pair("iptnum2ordinal", 1),
    std::make_pair("lab2double", 1),
    std::make_pair("lab2rgb", 7),
    std::make_pair("lab2uint16", 1),
    std::make_pair("lab2uint8", 1),
    std::make_pair("label2rgb", 4),
    std::make_pair("lin2rgb", 5),
    std::make_pair("lines", 1),
    std::make_pair("locallapfilt", 10),
    std::make_pair("mat2gray", 2),
    std::make_pair("mean2", 1),
    std::make_pair("medfilt2", 3),
    std::make_pair("modefilt", 3),
    std::make_pair("multithresh", 2),
    std::make_pair("normxcorr2", 2),
    std::make_pair("ntsc2rgb", 1),
    std::make_pair("ordfilt2", 5),
    std::make_pair("otf2psf", 2),
    std::make_pair("otsuthresh", 1),
    std::make_pair("padarray", 4),
    std::make_pair("pink", 1),
    std::make_pair("poly2mask", 4),
    std::make_pair("prism", 1),
    std::make_pair("psf2otf", 2),
    std::make_pair("psnr", 3),
    std::make_pair("rangefilt", 2),
    std::make_pair("reducepoly", 2),
    std::make_pair("regionfill", 3),
    std::make_pair("rgb2ind", 3),
    std::make_pair("rgb2lab", 5),
    std::make_pair("rgb2lin", 5),
    std::make_pair("rgb2ntsc", 1),
    std::make_pair("rgb2xyz", 5),
    std::make_pair("rgb2ycbcr", 1),
    std::make_pair("roicolor", 3),
    std::make_pair("roipoly", 6),
    std::make_pair("spring", 1),
    std::make_pair("std2", 1),
    std::make_pair("stdfilt", 2),
    std::make_pair("strel", 4),
    std::make_pair("stretchlim", 2),
    std::make_pair("summer", 1),
    std::make_pair("wcodemat", 4),
    std::make_pair("white", 1),
    std::make_pair("wiener2", 4),
    std::make_pair("winter", 1),
    std::make_pair("xyz2rgb", 7),
    std::make_pair("ycbcr2rgb", 1),
};

INSTANTIATE_TEST_SUITE_P(ImageBounds, ImageArityBoundsTest,
                        ::testing::ValuesIn(kBounds));
