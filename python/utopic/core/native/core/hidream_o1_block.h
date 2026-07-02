#pragma once

#include <cstdint>
#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace utopic {

struct HiDreamO1TextBlockGraphConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_attention_heads = 0;
    int64_t num_key_value_heads = 0;
    int64_t head_dim = 0;
    int64_t sequence_tokens = 0;
    int64_t ar_prefix_tokens = 0;
    float rms_norm_eps = 1e-6f;
};

struct HiDreamO1TextBlockGraphTensors {
    ggml_tensor* x = nullptr;
    ggml_tensor* input_layernorm_weight = nullptr;
    ggml_tensor* q_proj_weight = nullptr;
    ggml_tensor* k_proj_weight = nullptr;
    ggml_tensor* v_proj_weight = nullptr;
    ggml_tensor* o_proj_weight = nullptr;
    ggml_tensor* q_norm_weight = nullptr;
    ggml_tensor* k_norm_weight = nullptr;
    ggml_tensor* post_attention_layernorm_weight = nullptr;
    ggml_tensor* gate_proj_weight = nullptr;
    ggml_tensor* up_proj_weight = nullptr;
    ggml_tensor* down_proj_weight = nullptr;
    ggml_tensor* rope_cos = nullptr;
    ggml_tensor* rope_sin = nullptr;
};

struct HiDreamO1VisualBlockGraphConfig {
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t num_heads = 0;
    int64_t sequence_tokens = 0;
    float norm_eps = 1e-6f;
};

struct HiDreamO1VisualBlockGraphTensors {
    ggml_tensor* x = nullptr;
    ggml_tensor* norm1_weight = nullptr;
    ggml_tensor* norm1_bias = nullptr;
    ggml_tensor* qkv_weight = nullptr;
    ggml_tensor* qkv_bias = nullptr;
    ggml_tensor* proj_weight = nullptr;
    ggml_tensor* proj_bias = nullptr;
    ggml_tensor* norm2_weight = nullptr;
    ggml_tensor* norm2_bias = nullptr;
    ggml_tensor* fc1_weight = nullptr;
    ggml_tensor* fc1_bias = nullptr;
    ggml_tensor* fc2_weight = nullptr;
    ggml_tensor* fc2_bias = nullptr;
};

ggml_tensor* build_hidream_o1_qwen3vl_text_block(ggml_context* ctx,
                                                 const HiDreamO1TextBlockGraphConfig& config,
                                                 const HiDreamO1TextBlockGraphTensors& tensors);
ggml_tensor* build_hidream_o1_pixeldit_visual_block(ggml_context* ctx,
                                                    const HiDreamO1VisualBlockGraphConfig& config,
                                                    const HiDreamO1VisualBlockGraphTensors& tensors);
bool hidream_o1_qwen3vl_text_block_self_check(double* max_diff, std::string* error);

struct HiDreamO1RealBlockRunSummary {
    int layer = 0;
    int64_t sequence_tokens = 0;
    int64_t ar_prefix_tokens = 0;
    int64_t hidden_size = 0;
    int64_t intermediate_size = 0;
    int64_t payload_bytes = 0;
    int64_t output_values = 0;
    double output_checksum = 0.0;
    double output_l2 = 0.0;
    double output_max_abs = 0.0;
};

struct HiDreamO1NativeChainRunSummary {
    int text_layers = 0;
    int visual_layers = 0;
    int64_t text_tokens = 0;
    int64_t visual_tokens = 0;
    int64_t text_payload_bytes = 0;
    int64_t visual_payload_bytes = 0;
    int64_t text_output_values = 0;
    int64_t visual_output_values = 0;
    double text_output_checksum = 0.0;
    double visual_output_checksum = 0.0;
    double text_output_l2 = 0.0;
    double visual_output_l2 = 0.0;
    double text_output_max_abs = 0.0;
    double visual_output_max_abs = 0.0;
};

struct HiDreamO1NativeProjectionRunSummary {
    int64_t patch_tokens = 0;
    int64_t timestep_embedding_values = 0;
    int64_t final_tokens = 0;
    int64_t patch_output_values = 0;
    int64_t timestep_output_values = 0;
    int64_t final_output_values = 0;
    int64_t payload_bytes = 0;
    double patch_output_l2 = 0.0;
    double timestep_output_l2 = 0.0;
    double final_output_l2 = 0.0;
    double patch_output_checksum = 0.0;
    double timestep_output_checksum = 0.0;
    double final_output_checksum = 0.0;
};

struct HiDreamO1NativeImageRunSummary {
    int width = 0;
    int height = 0;
    int steps = 0;
    int64_t image_tokens = 0;
    int64_t patch_values = 0;
    double final_patch_l2 = 0.0;
    double final_patch_checksum = 0.0;
    std::string output_path;
};

bool hidream_o1_run_real_text_block_graph(const std::string& model_dir,
                                          int layer,
                                          int64_t sequence_tokens,
                                          HiDreamO1RealBlockRunSummary* summary,
                                          std::string* error);
bool hidream_o1_run_real_visual_block_graph(const std::string& model_dir,
                                            int layer,
                                            int64_t sequence_tokens,
                                            HiDreamO1RealBlockRunSummary* summary,
                                            std::string* error);
bool hidream_o1_run_native_layer_chain(const std::string& model_dir,
                                       int64_t text_tokens,
                                       int64_t visual_tokens,
                                       HiDreamO1NativeChainRunSummary* summary,
                                       std::string* error);
bool hidream_o1_run_native_projection_graph(const std::string& model_dir,
                                            int64_t patch_tokens,
                                            int64_t final_tokens,
                                            float timestep,
                                            HiDreamO1NativeProjectionRunSummary* summary,
                                            std::string* error);
bool hidream_o1_generate_native_preview_image(const std::string& model_dir,
                                              const std::string& prompt,
                                              const std::string& output_path,
                                              int width,
                                              int height,
                                              int steps,
                                              int seed,
                                              HiDreamO1NativeImageRunSummary* summary,
                                              std::string* error);

}  // namespace utopic
