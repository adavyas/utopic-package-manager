#pragma once

#include <cstdint>
#include <string>

namespace utopic {

struct image_engine_params {
    std::string model_path;
    std::string vae_path;
    std::string clip_l_path;
    std::string clip_g_path;
    std::string t5xxl_path;
    std::string diffusion_model_path;
    std::string output_path;
    std::string prompt;
    std::string negative_prompt;
    std::string backend;
    std::string params_backend;

    int32_t width       = 1024;
    int32_t height      = 1024;
    int32_t steps       = 20;
    int32_t seed        = 42;
    int32_t batch_count = 1;
    int32_t n_threads   = 0;

    float cfg_scale               = 3.5f;
    float distilled_guidance      = 3.5f;
    float eta                     = 0.0f;
    bool  enable_mmap             = true;
    bool  flash_attn              = true;
    bool  diffusion_flash_attn    = true;
    bool  qwen_image_zero_cond_t  = false;
};

struct image_engine_result {
    bool        ok             = false;
    std::string error_message;
    std::string artifact_path;
    int32_t     width          = 0;
    int32_t     height         = 0;
    int32_t     channel        = 0;
    int64_t     seed           = 0;
};

bool image_engine_generate(const image_engine_params & params, image_engine_result & result);

}  // namespace utopic
