#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace utopic {

struct HiDreamO1RuntimeConfig {
    const char* model_id = "hidream-o1";
    const char* hf_repo = "HiDream-ai/HiDream-O1-Image-Dev-2604";
    const char* native_status = "native-port-in-progress";
    int patch_size = 32;
    int timestep_token_num = 1;
    int image_token_id = 151655;
    int vision_start_token_id = 151652;
    int default_width = 2048;
    int default_height = 2048;
    int default_steps = 28;
    float default_guidance_scale = 0.0f;
    float default_shift = 1.0f;
    float default_noise_scale_start = 7.5f;
    float default_noise_scale_end = 7.5f;
    float default_noise_clip_std = 2.5f;
    const char* default_scheduler = "flash";
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

struct HiDreamO1ShardEntry {
    std::string tensor_name;
    std::string shard_file;
};

struct HiDreamO1ShardManifest {
    std::string model_dir;
    std::string index_path;
    std::vector<HiDreamO1ShardEntry> entries;
    std::vector<std::string> shard_files;
    std::string error;
};

struct HiDreamO1SafetensorsHeader {
    std::string file_path;
    uint64_t header_bytes = 0;
    int64_t tensor_count = 0;
    std::map<std::string, int64_t> dtype_counts;
    std::string error;
};

HiDreamO1RuntimeConfig default_hidream_o1_runtime_config();
HiDreamO1Shape hidream_o1_shape_for_size(const HiDreamO1RuntimeConfig& cfg, int width, int height);
std::vector<int> hidream_o1_dev_timesteps();
std::vector<float> hidream_o1_dev_sigmas();
std::string hidream_o1_default_model_dir();
std::string hidream_o1_default_model_path();
std::string hidream_o1_default_sd_cli();
std::string build_hidream_o1_oracle_command(const HiDreamO1RunRequest& req);
std::string build_hidream_o1_command(const HiDreamO1RunRequest& req);
bool hidream_o1_file_exists(const std::string& path);
bool hidream_o1_dir_exists(const std::string& path);
bool load_hidream_o1_shard_manifest(const std::string& model_dir, HiDreamO1ShardManifest* manifest);
HiDreamO1SafetensorsHeader inspect_hidream_o1_safetensors_header(const std::string& file_path);

}  // namespace utopic
