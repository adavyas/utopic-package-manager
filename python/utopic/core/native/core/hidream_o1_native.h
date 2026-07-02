#pragma once

#include <cstdint>
#include <functional>
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

struct HiDreamO1ForwardPlan {
    int width = 0;
    int height = 0;
    int h_patches = 0;
    int w_patches = 0;
    int patch_size = 0;
    int patch_dim = 0;
    int64_t text_tokens = 0;
    int64_t timestep_token_begin = 0;
    int64_t image_token_begin = 0;
    int64_t image_tokens = 0;
    int64_t total_sequence_tokens = 0;
    std::vector<int> raw_token_types;
    std::vector<unsigned char> token_types_bin;
    std::vector<unsigned char> vinput_mask;
};

struct HiDreamO1ForwardTraceStep {
    int step_index = 0;
    int timestep = 0;
    float sigma = 0.0f;
    float sigma_next = 0.0f;
    float t_pixeldit = 0.0f;
    float noise_scale = 0.0f;
};

using HiDreamO1X0Predictor = std::function<bool(const HiDreamO1ForwardTraceStep& step,
                                                const std::vector<float>& current_z,
                                                std::vector<float>* x0_pred,
                                                std::string* error)>;

struct HiDreamO1RunRequest {
    std::string sd_cli;
    std::string torch_python;
    std::string source_dir;
    std::string model_dir;
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

struct HiDreamO1TensorInfo {
    std::string tensor_name;
    std::string shard_file;
    std::string file_path;
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t header_bytes = 0;
    uint64_t data_offsets[2] = {0, 0};
    uint64_t absolute_data_begin = 0;
    uint64_t absolute_data_end = 0;
};

struct HiDreamO1TensorCatalog {
    std::string model_dir;
    std::vector<HiDreamO1TensorInfo> tensors;
    int64_t missing_tensor_count = 0;
    std::string error;
};

struct HiDreamO1TextModelConfig {
    int hidden_size = 0;
    int intermediate_size = 0;
    int num_hidden_layers = 0;
    int num_attention_heads = 0;
    int num_key_value_heads = 0;
    int head_dim = 0;
    int vocab_size = 0;
    double rope_theta = 0.0;
    double rms_norm_eps = 0.0;
};

struct HiDreamO1VisionModelConfig {
    int hidden_size = 0;
    int intermediate_size = 0;
    int depth = 0;
    int num_heads = 0;
    int patch_size = 0;
    int temporal_patch_size = 0;
    int spatial_merge_size = 0;
    int out_hidden_size = 0;
};

struct HiDreamO1NativeModelLayout {
    std::string model_dir;
    HiDreamO1TextModelConfig text;
    HiDreamO1VisionModelConfig vision;
    int image_token_id = 0;
    int vision_start_token_id = 0;
    int64_t tensor_count = 0;
    int64_t text_tensor_count = 0;
    int64_t vision_tensor_count = 0;
    int64_t timestep_tensor_count = 0;
    int64_t final_layer_tensor_count = 0;
    int64_t lm_head_tensor_count = 0;
    bool has_required_text_block0 = false;
    std::string error;
};

struct HiDreamO1NativeExecutionSummary {
    std::string model_dir;
    int width = 0;
    int height = 0;
    int64_t text_tokens = 0;
    int64_t image_tokens = 0;
    int64_t total_sequence_tokens = 0;
    int text_layers = 0;
    int text_hidden = 0;
    int text_heads = 0;
    int text_kv_heads = 0;
    int text_head_dim = 0;
    int text_intermediate = 0;
    int64_t tensor_count = 0;
    int64_t catalog_tensor_count = 0;
    int64_t catalog_missing_tensor_count = 0;
    int64_t block0_tensor_count = 0;
    uint64_t block0_payload_bytes = 0;
    bool block0_payloads_loaded = false;
    std::string error;
};

HiDreamO1RuntimeConfig default_hidream_o1_runtime_config();
HiDreamO1Shape hidream_o1_shape_for_size(const HiDreamO1RuntimeConfig& cfg, int width, int height);
HiDreamO1ForwardPlan hidream_o1_build_t2i_forward_plan(const HiDreamO1RuntimeConfig& cfg,
                                                       int width,
                                                       int height,
                                                       int64_t text_tokens);
std::vector<int> hidream_o1_dev_timesteps();
std::vector<float> hidream_o1_dev_sigmas();
std::vector<float> hidream_o1_noise_scale_schedule(const HiDreamO1RuntimeConfig& cfg, int steps);
float hidream_o1_t_pixeldit(int timestep);
std::vector<float> hidream_o1_x0_to_model_output(const std::vector<float>& x0_pred,
                                                 const std::vector<float>& z,
                                                 float sigma);
std::vector<float> hidream_o1_flash_step(const std::vector<float>& sample,
                                         const std::vector<float>& model_output,
                                         const std::vector<float>& noise,
                                         float sigma,
                                         float sigma_next,
                                         float s_noise,
                                         float noise_clip_std);
bool hidream_o1_run_forward_loop(const std::vector<float>& initial_z,
                                 const std::vector<std::vector<float>>& noise_by_step,
                                 const HiDreamO1X0Predictor& predict_x0,
                                 std::vector<float>* final_z,
                                 std::vector<HiDreamO1ForwardTraceStep>* trace,
                                 std::string* error);
std::vector<unsigned char> hidream_o1_unpatch_to_rgb8(const std::vector<float>& patch_tokens,
                                                      int width,
                                                      int height,
                                                      const HiDreamO1RuntimeConfig& cfg);
std::string hidream_o1_default_model_dir();
std::string hidream_o1_default_model_path();
std::string hidream_o1_default_sd_cli();
std::string hidream_o1_default_source_dir();
std::string hidream_o1_default_torch_python();
std::string build_hidream_o1_oracle_command(const HiDreamO1RunRequest& req);
std::string build_hidream_o1_command(const HiDreamO1RunRequest& req);
bool hidream_o1_file_exists(const std::string& path);
bool hidream_o1_dir_exists(const std::string& path);
bool hidream_o1_patch_official_source_for_flash_env(const std::string& source_dir, std::string* error);
bool load_hidream_o1_shard_manifest(const std::string& model_dir, HiDreamO1ShardManifest* manifest);
bool load_hidream_o1_native_model_layout(const std::string& model_dir, HiDreamO1NativeModelLayout* layout);
HiDreamO1SafetensorsHeader inspect_hidream_o1_safetensors_header(const std::string& file_path);
bool load_hidream_o1_tensor_catalog(const std::string& model_dir, HiDreamO1TensorCatalog* catalog);
bool find_hidream_o1_tensor(const HiDreamO1TensorCatalog& catalog,
                            const std::string& tensor_name,
                            HiDreamO1TensorInfo* tensor);
bool read_hidream_o1_tensor_bytes(const HiDreamO1TensorInfo& tensor,
                                  std::vector<unsigned char>* bytes,
                                  std::string* error);
std::vector<std::string> hidream_o1_text_block_tensor_names(int layer);
bool load_hidream_o1_text_block_tensors(const std::string& model_dir,
                                        int layer,
                                        std::vector<HiDreamO1TensorInfo>* tensors);
bool hidream_o1_prepare_native_execution(const std::string& model_dir,
                                         int width,
                                         int height,
                                         int64_t text_tokens,
                                         bool load_block0_payloads,
                                         HiDreamO1NativeExecutionSummary* summary,
                                         std::string* error);

}  // namespace utopic
