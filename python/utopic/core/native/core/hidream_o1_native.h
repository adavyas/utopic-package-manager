#pragma once

#include <cstdint>
#include <string>

namespace utopic {

struct HiDreamO1RuntimeConfig {
    const char* model_id = "hidream-o1";
    int patch_size = 32;
    int timestep_token_num = 1;
    int image_token_id = 151655;
    int vision_start_token_id = 151652;
    int default_width = 1024;
    int default_height = 1024;
    int default_steps = 28;
    float default_cfg_scale = 1.0f;
};

struct HiDreamO1Shape {
    int width = 0;
    int height = 0;
    int patch_size = 0;
    int patch_dim = 0;
    int64_t patch_tokens = 0;
    int64_t pixel_values = 0;
};

struct HiDreamO1RunRequest {
    std::string sd_cli;
    std::string model_path;
    std::string prompt;
    std::string output_path;
    int width = 0;
    int height = 0;
    int steps = 0;
    int seed = 42;
    float cfg_scale = 0.0f;
    std::string extra_args;
};

HiDreamO1RuntimeConfig default_hidream_o1_runtime_config();
HiDreamO1Shape hidream_o1_shape_for_size(const HiDreamO1RuntimeConfig& cfg, int width, int height);
std::string hidream_o1_default_model_path();
std::string hidream_o1_default_sd_cli();
std::string build_hidream_o1_command(const HiDreamO1RunRequest& req);
bool hidream_o1_file_exists(const std::string& path);

}  // namespace utopic
