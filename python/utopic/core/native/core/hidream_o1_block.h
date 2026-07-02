#pragma once

#include <cstdint>
#include <string>

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

ggml_tensor* build_hidream_o1_qwen3vl_text_block(ggml_context* ctx,
                                                 const HiDreamO1TextBlockGraphConfig& config,
                                                 const HiDreamO1TextBlockGraphTensors& tensors);
bool hidream_o1_qwen3vl_text_block_self_check(double* max_diff, std::string* error);

}  // namespace utopic
