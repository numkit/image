/* include/numkit/image/c_api.h
 *
 * Self-contained C-ABI for the `numkit/image` toolbox.
 *
 * Supports both:
 *   1. Pre-resolved O(1) dispatch (`numkit_image_resolve` + `numkit_image_invoke`)
 *      with zero string hashing on the hot path.
 *   2. Dynamic name/method dispatch (`numkit_image_call`) for MATLAB overloads,
 *      Name-Value pairs, and toolbox object methods.
 */
#ifndef NUMKIT_IMAGE_C_API_H
#define NUMKIT_IMAGE_C_API_H

#include <numkit/value/c_api.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) || defined(__CYGWIN__)
#  if defined(NUMKIT_PLUGIN_BUILD)
#    define NUMKIT_IMAGE_API __declspec(dllexport)
#  elif defined(NUMKIT_IMAGE_STATIC)
#    define NUMKIT_IMAGE_API
#  else
#    define NUMKIT_IMAGE_API __declspec(dllimport)
#  endif
#else
#  if defined(__GNUC__) && __GNUC__ >= 4
#    define NUMKIT_IMAGE_API __attribute__((visibility("default")))
#  else
#    define NUMKIT_IMAGE_API
#  endif
#endif

/* Opaque isolated `image` toolbox context */
typedef struct numkit_image_ctx_s numkit_image_ctx;

/* Create an isolated `image` toolbox context (engine-free, zero interpreter overhead). */
NUMKIT_IMAGE_API numkit_image_ctx *numkit_image_ctx_create(numkit_error *err);

/* Destroy `image` toolbox context and release its resources. */
NUMKIT_IMAGE_API void numkit_image_ctx_destroy(numkit_image_ctx *ctx);

/* Seed the `image` context's random number stream (MATLAB `rng(seed)` semantics). */
NUMKIT_IMAGE_API void numkit_image_seed_rng(numkit_image_ctx *ctx, uint64_t seed);

/* Return 1 if function `name` is registered in `image`, 0 otherwise. */
NUMKIT_IMAGE_API int numkit_image_has_function(const numkit_image_ctx *ctx, const char *name);

/* Register a host-language callback (`numkit_host_fn`) into this context. */
NUMKIT_IMAGE_API int numkit_image_register_fn(numkit_image_ctx *ctx,
                                            const char *name,
                                            numkit_host_fn fn,
                                            void *user_data,
                                            numkit_error *err);

/* Resolve function `name` once into an O(1) function pointer (`NULL` if not found). */
NUMKIT_IMAGE_API const numkit_fn *numkit_image_resolve(const numkit_image_ctx *ctx,
                                                     const char *name);

/* Invoke a pre-resolved function pointer directly with zero lookup overhead. */
NUMKIT_IMAGE_API int numkit_image_invoke(numkit_image_ctx *ctx,
                                       const numkit_fn *fn,
                                       const numkit_val *const *args,
                                       size_t nargs,
                                       size_t nargout,
                                       numkit_val **outs,
                                       numkit_error *err);

/* Dynamically invoke `image` function or object method `name`. */
NUMKIT_IMAGE_API int numkit_image_call(numkit_image_ctx *ctx,
                                     const char *name,
                                     const numkit_val *const *args,
                                     size_t nargs,
                                     size_t nargout,
                                     numkit_val **outs,
                                     numkit_error *err);

/* Format a value or `image` class object (e.g. `sym`, `tf`, `ss`) to UTF-8 text. */
NUMKIT_IMAGE_API size_t numkit_image_format(const numkit_image_ctx *ctx,
                                          const numkit_val *v,
                                          char *buf,
                                          size_t buf_cap);

#ifdef __cplusplus
}
#endif

#endif /* NUMKIT_IMAGE_C_API_H */
