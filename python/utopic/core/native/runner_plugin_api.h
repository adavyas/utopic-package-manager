#pragma once

#include <stddef.h>

#define UTOPIC_NATIVE_PLUGIN_OK 0
#define UTOPIC_NATIVE_PLUGIN_BUFFER_TOO_SMALL 2
#define UTOPIC_NATIVE_PLUGIN_DEFAULT_ENTRYPOINT "utopic_native_generate"

#if defined(_WIN32)
#    if defined(__cplusplus)
#        define UTOPIC_NATIVE_PLUGIN_EXTERN extern "C"
#    else
#        define UTOPIC_NATIVE_PLUGIN_EXTERN extern
#    endif
#    define UTOPIC_NATIVE_PLUGIN_EXPORT UTOPIC_NATIVE_PLUGIN_EXTERN __declspec(dllexport)
#else
#    if defined(__cplusplus)
#        define UTOPIC_NATIVE_PLUGIN_EXTERN extern "C"
#    else
#        define UTOPIC_NATIVE_PLUGIN_EXTERN extern
#    endif
#    define UTOPIC_NATIVE_PLUGIN_EXPORT UTOPIC_NATIVE_PLUGIN_EXTERN __attribute__((visibility("default")))
#endif

typedef int (*utopic_native_generate_fn)(const char * request_json,
                                         char *       response_json,
                                         size_t       response_json_size);
