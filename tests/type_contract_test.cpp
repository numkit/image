// toolboxes/image/tests/type_contract_test.cpp
//
// Image-domain class contracts (argcheck migration, 2026-09-19) — every
// accepted-class set, rejection class and result class probed against
// MATLAB R2025b the same day (build/probe12..13 + probe30 batteries,
// ~1100 mirrored probes). The reference takes a DIFFERENT class set per
// entry — the tests pin the per-entry asymmetries the shared gates in
// include/nk/image/internal/type_gates.hpp encode.

#include <numkit/bundle/standard_engine.hpp>
#include <numkit/core/engine.hpp>
#include <numkit/value/error.hpp>
#include <gtest/gtest.h>

#include <cmath>

using namespace numkit;

class ImageTypeContractTest : public ::testing::Test
{
public:
    StandardEngine engine;

    double evalScalar(const std::string &c) { return engine.eval(c).toScalar(); }
    std::string evalClass(const std::string &expr)
    {
        engine.eval("tc_r = " + expr + ";");
        return engine.eval("class(tc_r);").toString();
    }

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

TEST_F(ImageTypeContractTest, AdjustFamilyRejectsLogicalAndComplex)
{
    // imadjust / adaptthresh / imbinarize / imquantize / imboxfilt:
    // dbl/sgl/u8/u16/i16 compute (class-preserving), logical rejected,
    // complex expectedReal, char/cell rejected with inputType.
    EXPECT_EQ(evalClass("imadjust(magic(5)/8)"), "double");
    EXPECT_EQ(evalClass("imadjust(single(magic(5)/8))"), "single");
    EXPECT_EQ(evalClass("imadjust(uint8(magic(5)))"), "uint8");
    expectError("imadjust(logical(magic(5)>13));", "numkit:imadjust:inputType");
    expectError("imadjust(magic(5)/8+1i);", "numkit:imadjust:expectedReal");
    expectError("imadjust('ab');", "numkit:imadjust:inputType");
    expectError("adaptthresh(logical(magic(5)>13));", "numkit:adaptthresh:inputType");
    expectError("adaptthresh(magic(5)/8+1i);", "numkit:adaptthresh:expectedReal");
    // adaptthresh ALWAYS returns double (probed: u8 input → double).
    EXPECT_EQ(evalClass("adaptthresh(uint8(magic(5)*10))"), "double");
    // 1-arg default threshold routes through graythresh, which voices
    // the rejection first (same rejection class as the reference).
    expectError("imbinarize(logical(magic(5)>13));", "numkit:graythresh:inputType");
    expectError("imbinarize(magic(5)/8+1i);", "numkit:graythresh:expectedReal");
    expectError("imboxfilt(logical(magic(5)>13));", "numkit:imboxfilt:inputType");
    // Empty + auto stretchlim → expectedNonempty (probed).
    expectError("imadjust([]);", "numkit:stretchlim:expectedNonempty");
}

TEST_F(ImageTypeContractTest, MedianFamilyAcceptsLogical)
{
    // medfilt2 / ordfilt2 / imfilter / stdfilt / rangefilt / entropyfilt:
    // logical ACCEPTED with logical result — unlike the adjust family.
    EXPECT_EQ(evalClass("medfilt2(logical(magic(5)>13))"), "logical");
    EXPECT_EQ(evalClass("ordfilt2(logical(magic(5)>13), 3, ones(3))"), "logical");
    EXPECT_EQ(evalClass("imfilter(logical(magic(5)>13), ones(3)/9)"), "logical");
    EXPECT_EQ(evalClass("stdfilt(logical(magic(5)>13))"), "double");
    expectError("medfilt2(magic(5)/8+1i);", "numkit:medfilt2:expectedReal");
    expectError("ordfilt2(magic(5)/8+1i, 3, ones(3));", "numkit:ordfilt2:expectedReal");
    // imfilter kernel must be real double (single/logical kernels rejected
    // at the reference's conv layer — NonDoubleFilter class).
    expectError("imfilter(magic(5)/8, single(ones(3)/9));", "numkit:imfilter:inputType");
    expectError("imfilter(magic(5)/8, true(3));", "numkit:imfilter:inputType");
    // entropyfilt: logical accepted, int16 REJECTED (the one signed-int
    // rejection in the filter family — probed).
    EXPECT_EQ(evalClass("entropyfilt(logical(magic(5)>13))"), "double");
    expectError("entropyfilt(int16(magic(5)));", "numkit:entropyfilt:inputType");
    expectError("entropyfilt([]);", "numkit:entropyfilt:expectedNonempty");
}

TEST_F(ImageTypeContractTest, RangefiltInt16ReturnsUint16)
{
    // Probed R2025b: the range of an int16 image comes back uint16
    // (computed in the unsigned domain).
    EXPECT_EQ(evalClass("rangefilt(int16(magic(5)))"), "uint16");
    EXPECT_EQ(evalClass("rangefilt(uint8(magic(5)))"), "uint8");
}

TEST_F(ImageTypeContractTest, BwFamilyAcceptsAllNumericReal)
{
    // bwmorph / bwconncomp / bwareaopen / bwhitmiss / applylut / bwpack:
    // every numeric class works (foreground = nonzero), complex with a
    // NONZERO imaginary part is expectedReal (a zero-imag complex array
    // is real to the reference; nk collapses those to real storage).
    EXPECT_EQ(evalClass("bwmorph(double(magic(5)>13), 'dilate')"), "logical");
    EXPECT_EQ(evalClass("bwmorph(uint8(magic(5)>13), 'dilate')"), "logical");
    EXPECT_EQ(evalClass("bwpack(magic(5)>13)"), "uint32");
    EXPECT_EQ(evalClass("bwareaopen(magic(5)>13, 2)"), "logical");
    expectError("bwmorph(magic(5)/8+1i, 'dilate');", "numkit:bwmorph:expectedReal");
    expectError("bwmorph('ab', 'dilate');", "numkit:bwmorph:inputType");
    expectError("bwconncomp(struct('a',1));", "numkit:bwconncomp:inputType");
}

TEST_F(ImageTypeContractTest, LogicalOnlyFamily)
{
    // bwareafilt / bwpropfilt / bwdistgeodesic demand LOGICAL BW —
    // double 0/1 is rejected (probed).
    EXPECT_EQ(evalClass("bwareafilt(logical(magic(5)>13), 1)"), "logical");
    expectError("bwareafilt(double(magic(5)>13), 1);", "numkit:bwareafilt:inputType");
    expectError("bwpropfilt(double(magic(5)>13), 'area', 1);", "numkit:bwpropfilt:inputType");
    expectError("bwdistgeodesic(double(magic(5)>13), 7);", "numkit:bwdistgeodesic:inputType");
    // Seeds: real double or a logical seed mask; single/uint8 rejected
    // (probed probe30).
    expectError("bwdistgeodesic(logical(magic(5)>13), single(7));",
                "numkit:bwdistgeodesic:inputType");
}

TEST_F(ImageTypeContractTest, BwdistReturnsSingle)
{
    // Probed R2025b: bwdist distances are single for EVERY input class.
    EXPECT_EQ(evalClass("bwdist(magic(5)>13)"), "single");
    EXPECT_EQ(evalClass("bwdist(uint8(magic(5)>13))"), "single");
    expectError("bwdist('ab');", "numkit:bwdist:inputType");
}

TEST_F(ImageTypeContractTest, BwunpackRequiresUint32)
{
    // Probed: even a logical packed array is rejected.
    expectError("bwunpack(logical(magic(5)>13), 5);", "numkit:bwunpack:inputType");
    EXPECT_EQ(evalClass("bwunpack(bwpack(magic(5)>13), 5)"), "logical");
}

TEST_F(ImageTypeContractTest, MorphologyValuePreserving)
{
    // imclearborder / imkeepborder / imfill preserve the input class AND
    // the pixel values (probed: interior values survive, border
    // components zero out) — not a bare logical mask.
    engine.eval("I = zeros(5); I(2:4, 2:4) = [1 2 3; 4 5 6; 7 8 9];");
    engine.eval("J = imclearborder(I, 4);");
    // The interior 3×3 block survives with its VALUES; the rim stays 0.
    EXPECT_NEAR(evalScalar("J(3,3)"), 5.0, 1e-12);
    EXPECT_NEAR(evalScalar("J(2,2)"), 1.0, 1e-12);
    EXPECT_NEAR(evalScalar("J(1,3)"), 0.0, 1e-12);
    EXPECT_EQ(evalClass("imclearborder(uint8(I*20))"), "uint8");
    EXPECT_EQ(evalClass("imkeepborder(single(I))"), "single");
    EXPECT_EQ(evalClass("imfill(I, 'holes')"), "double");
    expectError("imclearborder('ab');", "numkit:imclearborder:inputType");
    expectError("imfill(magic(5)/8+1i, 'holes');", "numkit:imfill:expectedReal");
    // imhmax / imhmin / imimposemin: class-preserving results.
    EXPECT_EQ(evalClass("imhmax(uint8(magic(5)), 10)"), "uint8");
    EXPECT_EQ(evalClass("imhmin(single(magic(5)), 10)"), "single");
    EXPECT_EQ(evalClass("imimposemin(uint8(magic(5)), false(5))"), "uint8");
    expectError("imhmax(logical(magic(5)>13), 1);", "numkit:imhmax:inputType");
}

TEST_F(ImageTypeContractTest, ImreconstructSameClass)
{
    // Marker and mask must share the class (probed: double marker +
    // logical mask → notSameClass).
    EXPECT_EQ(evalClass("imreconstruct(logical(magic(5)>17), magic(5)>13)"),
              "logical");
    expectError("imreconstruct(double(magic(5)>17), logical(magic(5)>13));",
                "numkit:imreconstruct:notSameClass");
}

TEST_F(ImageTypeContractTest, QualitySameClassPairs)
{
    // immse / psnr / ssim / multissim: A and B same class (complex pairs
    // with double only for immse — probed immse(G+1i,G2)=1.0001).
    EXPECT_NEAR(evalScalar("immse(magic(5)/8, magic(5)/8+0.01)"),
                1e-4, 1e-6);
    expectError("immse(magic(5)/8, single(magic(5)/8));",
                "numkit:immse:differentClass");
    expectError("immse(magic(5)/8, uint8(magic(5)));",
                "numkit:immse:differentClass");
    expectError("psnr(magic(5)/8, single(magic(5)/8));",
                "numkit:psnr:differentClass");
    expectError("psnr(magic(5)/8+1i, magic(5)/8);", "numkit:psnr:expectedReal");
    expectError("ssim(logical(magic(5)>13), logical(magic(5)>13));",
                "numkit:ssim:inputType");
    expectError("ssim(magic(5)/8+1i, magic(5)/8+1i);", "numkit:ssim:expectedReal");
    // Complex immse: |zA−zB|² semantics (probed 1.0001).
    engine.eval("mse = immse(magic(5)/8+1i, magic(5)/8+0.01);");
    EXPECT_NEAR(evalScalar("mse"), 1.0001, 1e-6);
}

TEST_F(ImageTypeContractTest, IndexedColourContracts)
{
    // ind2gray X: dbl/sgl/u8/u16/logical; map must be double M×3.
    // ind2rgb X: also int16 and char (code values); loose maps.
    engine.eval("M12 = [linspace(0,1,12)', zeros(12,1), linspace(1,0,12)'];");
    engine.eval("X8 = uint8(1 + mod(0:24, 12));");
    EXPECT_EQ(evalClass("ind2gray(X8, M12)"), "uint8");
    EXPECT_EQ(evalClass("ind2gray(single(X8), M12)"), "single");
    EXPECT_EQ(evalClass("ind2gray(uint16(X8), M12)"), "uint16");
    EXPECT_EQ(evalClass("ind2gray(logical(X8), M12)"), "double");
    expectError("ind2gray(int16(X8), M12);", "numkit:ind2gray:inputType");
    expectError("ind2gray(X8, single(M12));", "numkit:ind2gray:invalidColormap");
    expectError("ind2gray(X8, M12+1i);", "numkit:ind2gray:invalidColormap");
    EXPECT_EQ(evalClass("ind2rgb(int16(X8), M12)"), "double");
    EXPECT_EQ(evalClass("ind2rgb(uint16(X8), M12)"), "double");
    expectError("ind2gray('ab', M12);", "numkit:ind2gray:inputType");
}

TEST_F(ImageTypeContractTest, LabelAndOverlayContracts)
{
    engine.eval("L = zeros(5); L(2:3,2:3) = 1; L(4,4) = 2;");
    // Labels: every numeric class accepted → uint8 RGB.
    EXPECT_EQ(evalClass("label2rgb(L)"), "uint8");
    EXPECT_EQ(evalClass("label2rgb(single(L))"), "uint8");
    expectError("label2rgb(L+1i);", "numkit:label2rgb:expectedReal");
    expectError("label2rgb('ab');", "numkit:label2rgb:inputType");
    expectError("label2rgb([]);", "numkit:label2rgb:empty");
    // Map must be double (single map rejected — probed).
    engine.eval("M12 = [linspace(0,1,12)', zeros(12,1), linspace(1,0,12)'];");
    expectError("label2rgb(L, single(M12));", "numkit:label2rgb:invalidColormap");
    // imoverlay: complex colour triple rejected.
    expectError("imoverlay(magic(5)/8, true(5), [1 0 0]+1i);",
                "numkit:imoverlay:expectedReal");
    expectError("imoverlay(magic(5)/8+1i, true(5), [1 0 0]);",
                "numkit:imoverlay:expectedReal");
}

TEST_F(ImageTypeContractTest, HoughPeaksDoubleOnly)
{
    engine.eval("H = hough(magic(5)>13);");
    EXPECT_EQ(evalClass("houghpeaks(H, 2)"), "double");
    expectError("houghpeaks(single(H), 2);", "numkit:houghpeaks:inputType");
    expectError("houghpeaks(uint8(H), 2);", "numkit:houghpeaks:inputType");
    expectError("houghpeaks(H+1i, 2);", "numkit:houghpeaks:expectedReal");
}

TEST_F(ImageTypeContractTest, Fir2dContracts)
{
    engine.eval("[f1,f2] = freqspace(21, 'meshgrid');");
    engine.eval("Hd = zeros(21); Hd(abs(f1)<0.4 & abs(f2)<0.4) = 1;");
    // fsamp2: no class validator — single keeps single, uint8/logical
    // compute DOUBLE, char coerces through code values, complex stubs.
    EXPECT_EQ(evalClass("fsamp2(Hd)"), "double");
    EXPECT_EQ(evalClass("fsamp2(single(Hd))"), "single");
    EXPECT_EQ(evalClass("fsamp2(logical(Hd))"), "double");
    EXPECT_EQ(evalClass("fsamp2(uint8(Hd))"), "double");
    EXPECT_EQ(evalClass("fsamp2(Hd+1i)"), "double");
    // fwind2: 1-D window is promoted to the separable outer product
    // (probed — the classic fwind2(Hd, hamming(n)) form).
    EXPECT_EQ(evalClass("fwind2(Hd, hamming(21))"), "double");
    EXPECT_EQ(evalClass("fwind2(Hd, single(hamming(21)))"), "single");
    // Integer windows are rejected at the reference's mixed-class
    // arithmetic layer.
    expectError("fwind2(Hd, uint8(hamming(21)*200));", "numkit:fwind2:inputType");
    expectError("fwind1(Hd, uint8(hamming(21)*200));", "numkit:fwind1:inputType");
}

TEST_F(ImageTypeContractTest, Im2colLooseContract)
{
    // im2col: logical/complex preserved; a block larger than the image
    // yields an EMPTY DOUBLE result (not an error) — probed.
    EXPECT_EQ(evalClass("im2col(magic(5), [3 3])"), "double");
    EXPECT_EQ(evalClass("im2col(uint8(magic(5)), [3 3])"), "uint8");
    EXPECT_EQ(evalClass("im2col(logical(magic(5)>13), [3 3])"), "logical");
    engine.eval("E = im2col(uint8(1:9), [2 2]);");
    EXPECT_EQ(static_cast<int>(evalScalar("size(E,1)")), 4);
    EXPECT_EQ(static_cast<int>(evalScalar("size(E,2)")), 0);
    EXPECT_EQ(engine.eval("class(E);").toString(), "double");
}

TEST_F(ImageTypeContractTest, ImapplymatrixTwoArgForm)
{
    // 2-arg form computes; the output class follows X (probed).
    engine.eval("RGB = cat(3, magic(5)/40, magic(5)/60, magic(5)/80);");
    EXPECT_EQ(evalClass("imapplymatrix([0.3 0.6 0.1], RGB)"), "double");
    EXPECT_EQ(evalClass("imapplymatrix([0.3 0.6 0.1], single(RGB))"), "single");
    EXPECT_EQ(evalClass("imapplymatrix([0.3 0.6 0.1], uint8(RGB*200))"), "uint8");
    expectError("imapplymatrix([0.3 0.6 0.1], logical(RGB>5));",
                "numkit:imapplymatrix:inputType");
    EXPECT_EQ(evalClass("imapplymatrix([0.3 0.6 0.1]+1i, RGB)"), "double");
}

TEST_F(ImageTypeContractTest, Axes2pixAndBestblkClassPreserving)
{
    EXPECT_EQ(evalClass("axes2pix(5, [0 1], 0.5)"), "double");
    EXPECT_EQ(evalClass("axes2pix(5, single([0 1]), 0.5)"), "single");
    // char extent coerces through code values (probed).
    EXPECT_EQ(evalClass("axes2pix(5, 'ab', 0.5)"), "double");
    EXPECT_EQ(evalClass("bestblk([100 200])"), "double");
    EXPECT_EQ(evalClass("bestblk(single([100 200]))"), "single");
    EXPECT_EQ(evalClass("bestblk(uint8([100 200]))"), "uint8");
    EXPECT_EQ(evalClass("bestblk('ab')"), "double");
}

TEST_F(ImageTypeContractTest, ColourExtrasContracts)
{
    engine.eval("RGB = cat(3, magic(5)/40, magic(5)/60, magic(5)/80);");
    EXPECT_EQ(evalClass("rgb2lightness(RGB)"), "double");
    EXPECT_EQ(evalClass("rgb2lightness(single(RGB))"), "single");
    expectError("rgb2lightness(int16(RGB));", "numkit:rgb2lightness:inputType");
    expectError("rgb2lightness(RGB+1i);", "numkit:rgb2lightness:expectedReal");
    // tonemap: float only (uint8 rejected), uint8 result; single input
    // computes the same uint8 image as double.
    EXPECT_EQ(evalClass("tonemap(abs(RGB)+0.1)"), "uint8");
    EXPECT_EQ(evalClass("tonemap(single(abs(RGB)+0.1))"), "uint8");
    expectError("tonemap(uint8(RGB*200));", "numkit:tonemap:inputType");
    expectError("tonemap(RGB+1i);", "numkit:tonemap:expectedReal");
}

TEST_F(ImageTypeContractTest, GraydistAndSeeds)
{
    // graydist seeds must be real double (uint8 rejected — probed).
    expectError("graydist(magic(5)/8, uint8([2 2]));",
                "numkit:graydist:inputType");
    expectError("graydist(magic(5)/8, [2 2]+1i);", "numkit:graydist:mustBeReal");
    EXPECT_EQ(evalClass("graydist(logical(magic(5)>13), 7)"), "single");
}

TEST_F(ImageTypeContractTest, ImbilatfiltWindowMustFit)
{
    // Probed: images smaller than the default 5×5 window are rejected
    // (imageNotMinSize class); 9×9 computes.
    expectError("imbilatfilt(magic(3)/8);", "numkit:imbilatfilt:nhoodTooBig");
    EXPECT_EQ(evalClass("imbilatfilt(magic(9)/8)"), "double");
}

TEST_F(ImageTypeContractTest, CornerAndNormxcorr2)
{
    EXPECT_EQ(evalClass("cornermetric(magic(5)/8)"), "double");
    EXPECT_EQ(evalClass("cornermetric(single(magic(5)/8))"), "double");
    expectError("cornermetric(magic(5)/8+1i);", "numkit:cornermetric:expectedReal");
    // normxcorr2 rejects logical templates (sameElementsInTemplate class)
    // but takes integer ones.
    expectError("normxcorr2(logical(magic(3)>4), magic(5)/8);",
                "numkit:normxcorr2:inputType");
    EXPECT_EQ(evalClass("normxcorr2(uint8(magic(3)), magic(5)/8)"), "double");
}

TEST_F(ImageTypeContractTest, DeconvwnrPsfDoubleOnly)
{
    engine.eval("P = fspecial('gaussian', 3, 1);");
    EXPECT_EQ(evalClass("deconvwnr(magic(5)/80, P)"), "double");
    EXPECT_EQ(evalClass("deconvwnr(single(magic(5)/80), P)"), "single");
    expectError("deconvwnr(magic(5)/80, single(P));",
                "numkit:deconvwnr:inputType");
    expectError("deconvwnr(magic(5)/80+1i, P);", "numkit:deconvwnr:expectedReal");
}
