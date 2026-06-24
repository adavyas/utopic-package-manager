#include "image_engine.h"

namespace utopic {

bool image_engine_generate(const image_engine_params &, image_engine_result & result) {
    result = image_engine_result();
    result.error_message = "native image engine was built without stable-diffusion.cpp support";
    return false;
}

}  // namespace utopic
