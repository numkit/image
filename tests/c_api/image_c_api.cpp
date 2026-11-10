#include <numkit/image/c_api.h>
#include <gtest/gtest.h>

TEST(NumkitImageCApi, ImcomplementZeroCopyView)
{
    const int64_t base = numkit_debug_live_values();
    numkit_error err{};
    numkit_image_ctx *ctx = numkit_image_ctx_create(&err);
    ASSERT_NE(ctx, nullptr) << err.message;

    double px[2] = {0.2, 0.7};
    size_t dims[2] = {1, 2};
    numkit_val *img = numkit_val_view(px, NUMKIT_DTYPE_DOUBLE, dims, 2, &err);
    const numkit_fn *fn = numkit_image_resolve(ctx, "imcomplement");
    ASSERT_NE(fn, nullptr);
    const numkit_val *args[1] = {img};
    numkit_val *inv = nullptr;
    ASSERT_EQ(numkit_image_invoke(ctx, fn, args, 1, 1, &inv, &err), 0) << err.message;
    const auto *out = static_cast<const double *>(numkit_val_raw_data(inv));
    EXPECT_NEAR(out[0], 0.8, 1e-12);
    EXPECT_NEAR(out[1], 0.3, 1e-12);

    numkit_val_release(inv);
    numkit_val_release(img);
    numkit_image_ctx_destroy(ctx);
    EXPECT_EQ(numkit_debug_live_values(), base);
}
