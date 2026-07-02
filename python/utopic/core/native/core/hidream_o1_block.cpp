#include "hidream_o1_block.h"

#include "hidream_o1_native.h"

#include "ggml.h"
#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace utopic {

namespace {

void require_tensor(ggml_tensor* t, const char* name) {
    if (t == nullptr) {
        throw std::invalid_argument(std::string("missing HiDream tensor: ") + name);
    }
}

void require_config(const HiDreamO1TextBlockGraphConfig& c) {
    if (c.hidden_size <= 0 || c.intermediate_size <= 0 || c.num_attention_heads <= 0 ||
        c.num_key_value_heads <= 0 || c.head_dim <= 0 || c.sequence_tokens <= 0) {
        throw std::invalid_argument("invalid HiDream text block graph dimensions");
    }
    if (c.hidden_size != c.num_attention_heads * c.head_dim) {
        throw std::invalid_argument("HiDream hidden size must equal heads * head_dim");
    }
    if (c.num_attention_heads % c.num_key_value_heads != 0) {
        throw std::invalid_argument("HiDream attention heads must be divisible by KV heads");
    }
    if (c.ar_prefix_tokens < 0 || c.ar_prefix_tokens > c.sequence_tokens) {
        throw std::invalid_argument("HiDream AR prefix token count is out of range");
    }
}

ggml_tensor* build_linear(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight) {
    require_tensor(x, "linear.x");
    require_tensor(weight, "linear.weight");
    ggml_tensor* y = ggml_mul_mat(ctx, weight, x);
    ggml_mul_mat_set_prec(y, GGML_PREC_F32);
    return y;
}

ggml_tensor* build_linear_bias(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, ggml_tensor* bias) {
    ggml_tensor* y = build_linear(ctx, x, weight);
    if (bias != nullptr) {
        y = ggml_add(ctx, y, ggml_reshape_2d(ctx, bias, bias->ne[0], 1));
    }
    return y;
}

ggml_tensor* build_weighted_rms(ggml_context* ctx, ggml_tensor* x, ggml_tensor* weight, float eps) {
    require_tensor(x, "rms.x");
    require_tensor(weight, "rms.weight");
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), weight);
}

ggml_tensor* build_affine_layer_norm(ggml_context* ctx,
                                     ggml_tensor* x,
                                     ggml_tensor* weight,
                                     ggml_tensor* bias,
                                     float eps) {
    require_tensor(x, "ln.x");
    require_tensor(weight, "ln.weight");
    require_tensor(bias, "ln.bias");
    return ggml_add(ctx,
                    ggml_mul(ctx, ggml_norm(ctx, x, eps), ggml_reshape_2d(ctx, weight, weight->ne[0], 1)),
                    ggml_reshape_2d(ctx, bias, bias->ne[0], 1));
}

ggml_tensor* build_head_rms(ggml_context* ctx,
                            ggml_tensor* x,
                            ggml_tensor* weight,
                            int64_t head_dim,
                            float eps) {
    require_tensor(x, "head_rms.x");
    require_tensor(weight, "head_rms.weight");
    return ggml_mul(ctx, ggml_rms_norm(ctx, x, eps), ggml_reshape_3d(ctx, weight, head_dim, 1, 1));
}

ggml_tensor* build_interleaved_rope(ggml_context* ctx,
                                    ggml_tensor* x,
                                    ggml_tensor* cos,
                                    ggml_tensor* sin,
                                    int64_t heads,
                                    int64_t head_dim,
                                    int64_t tokens) {
    require_tensor(x, "rope.x");
    require_tensor(cos, "rope.cos");
    require_tensor(sin, "rope.sin");

    ggml_tensor* x4 = ggml_reshape_4d(ctx, x, 2, head_dim / 2, heads, tokens);
    ggml_tensor* x0 = ggml_cont(ctx, ggml_view_4d(ctx, x4, 1, head_dim / 2, heads, tokens,
                                                  x4->nb[1], x4->nb[2], x4->nb[3], 0));
    ggml_tensor* x1 = ggml_cont(ctx, ggml_view_4d(ctx, x4, 1, head_dim / 2, heads, tokens,
                                                  x4->nb[1], x4->nb[2], x4->nb[3], x4->nb[0]));
    ggml_tensor* rot = ggml_reshape_3d(ctx, ggml_concat(ctx, ggml_neg(ctx, x1), x0, 0), head_dim, heads, tokens);
    return ggml_add(ctx, ggml_mul(ctx, x, cos), ggml_mul(ctx, rot, sin));
}

ggml_tensor* view_token_prefix(ggml_context* ctx, ggml_tensor* x, int64_t tokens) {
    return ggml_view_3d(ctx, x, x->ne[0], tokens, x->ne[2], x->nb[1], x->nb[2], 0);
}

ggml_tensor* view_token_suffix_2d(ggml_context* ctx, ggml_tensor* x, int64_t begin_token) {
    const int64_t tokens = x->ne[1] - begin_token;
    return ggml_cont(ctx, ggml_view_2d(ctx, x, x->ne[0], tokens, x->nb[1], begin_token * x->nb[1]));
}

int64_t shape_product(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 0;
    int64_t product = 1;
    for (const int64_t dim : shape) {
        if (dim <= 0 || product > std::numeric_limits<int64_t>::max() / dim) return 0;
        product *= dim;
    }
    return product;
}

uint64_t tensor_payload_bytes(const HiDreamO1TensorInfo& info) {
    return info.absolute_data_end >= info.absolute_data_begin ? info.absolute_data_end - info.absolute_data_begin : 0;
}

void fill_deterministic_block_input(ggml_tensor* x) {
    float* data = static_cast<float*>(x->data);
    const int64_t dim = x->ne[0];
    const int64_t tokens = x->ne[1];
    if (dim == 4 && tokens == 3) {
        const float fixture[] = {
            0.10f, -0.20f, 0.30f, -0.40f,
            0.50f, 0.25f, -0.75f, 1.00f,
            -1.00f, 0.60f, 0.20f, -0.10f,
        };
        std::memcpy(data, fixture, sizeof(fixture));
        return;
    }
    for (int64_t i = 0; i < dim * tokens; ++i) {
        data[i] = std::sin(static_cast<float>(i + 1) * 0.013f) * 0.5f;
    }
}

void fill_rope_tables(ggml_tensor* cos_tensor, ggml_tensor* sin_tensor, double theta) {
    float* cos_data = static_cast<float*>(cos_tensor->data);
    float* sin_data = static_cast<float*>(sin_tensor->data);
    const int64_t head_dim = cos_tensor->ne[0];
    const int64_t tokens = cos_tensor->ne[2];
    const double base = theta > 0.0 ? theta : 1000000.0;
    for (int64_t tok = 0; tok < tokens; ++tok) {
        for (int64_t i = 0; i < head_dim / 2; ++i) {
            const double inv_freq = std::pow(base, -static_cast<double>(2 * i) / static_cast<double>(head_dim));
            const double angle = static_cast<double>(tok) * inv_freq;
            const float c = static_cast<float>(std::cos(angle));
            const float s = static_cast<float>(std::sin(angle));
            const int64_t d0 = 2 * i;
            const int64_t d1 = d0 + 1;
            cos_data[tok * head_dim + d0] = c;
            cos_data[tok * head_dim + d1] = c;
            sin_data[tok * head_dim + d0] = s;
            sin_data[tok * head_dim + d1] = s;
        }
    }
}

bool load_f32_tensor_bytes(const HiDreamO1TensorCatalog& catalog,
                           ggml_context* ctx,
                           const std::string& name,
                           int expected_rank,
                           ggml_tensor** out,
                           int64_t* payload_bytes,
                           std::string* error) {
    if (out == nullptr) return false;
    *out = nullptr;
    HiDreamO1TensorInfo info;
    if (!find_hidream_o1_tensor(catalog, name, &info)) {
        if (error) *error = "missing HiDream tensor: " + name;
        return false;
    }
    if (info.dtype != "F32") {
        if (error) *error = "native HiDream block runner currently requires F32 tensor payloads: " + name + " dtype=" + info.dtype;
        return false;
    }
    if (static_cast<int>(info.shape.size()) != expected_rank) {
        if (error) *error = "unexpected HiDream tensor rank for " + name;
        return false;
    }
    const int64_t values = shape_product(info.shape);
    if (values <= 0) {
        if (error) *error = "invalid HiDream tensor shape for " + name;
        return false;
    }
    const uint64_t expected_bytes = static_cast<uint64_t>(values) * sizeof(float);
    if (tensor_payload_bytes(info) != expected_bytes) {
        if (error) *error = "unexpected HiDream tensor byte count for " + name;
        return false;
    }

    std::vector<unsigned char> bytes;
    std::string read_error;
    if (!read_hidream_o1_tensor_bytes(info, &bytes, &read_error)) {
        if (error) *error = read_error;
        return false;
    }
    ggml_tensor* tensor = nullptr;
    if (expected_rank == 1) {
        tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, info.shape[0]);
    } else if (expected_rank == 2) {
        tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, info.shape[1], info.shape[0]);
    } else {
        if (error) *error = "unsupported HiDream tensor rank for " + name;
        return false;
    }
    std::memcpy(tensor->data, bytes.data(), bytes.size());
    ggml_set_name(tensor, name.c_str());
    *out = tensor;
    if (payload_bytes) *payload_bytes += static_cast<int64_t>(bytes.size());
    return true;
}

std::vector<std::string> hidream_o1_visual_block_tensor_names(int layer) {
    const std::string prefix = "model.visual.blocks." + std::to_string(layer) + ".";
    return {
        prefix + "norm1.weight",
        prefix + "norm1.bias",
        prefix + "attn.qkv.weight",
        prefix + "attn.qkv.bias",
        prefix + "attn.proj.weight",
        prefix + "attn.proj.bias",
        prefix + "norm2.weight",
        prefix + "norm2.bias",
        prefix + "mlp.linear_fc1.weight",
        prefix + "mlp.linear_fc1.bias",
        prefix + "mlp.linear_fc2.weight",
        prefix + "mlp.linear_fc2.bias",
    };
}

ggml_tensor* build_attention_from_permuted(ggml_context* ctx,
                                           ggml_tensor* q,
                                           ggml_tensor* k,
                                           ggml_tensor* v,
                                           const HiDreamO1TextBlockGraphConfig& c,
                                           int64_t q_tokens,
                                           int64_t kv_tokens,
                                           bool causal) {
    const int64_t kv_groups = c.num_attention_heads / c.num_key_value_heads;
    ggml_tensor* kx = ggml_reshape_3d(
        ctx,
        ggml_repeat(ctx,
                    ggml_reshape_4d(ctx, k, c.head_dim, kv_tokens, 1, c.num_key_value_heads),
                    ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.head_dim, kv_tokens, kv_groups, c.num_key_value_heads)),
        c.head_dim,
        kv_tokens,
        c.num_attention_heads);
    ggml_tensor* vx = ggml_reshape_3d(
        ctx,
        ggml_repeat(ctx,
                    ggml_reshape_4d(ctx, v, c.head_dim, kv_tokens, 1, c.num_key_value_heads),
                    ggml_new_tensor_4d(ctx, GGML_TYPE_F32, c.head_dim, kv_tokens, kv_groups, c.num_key_value_heads)),
        c.head_dim,
        kv_tokens,
        c.num_attention_heads);

    ggml_tensor* scores = ggml_mul_mat(ctx, kx, q);
    ggml_mul_mat_set_prec(scores, GGML_PREC_F32);
    scores = ggml_scale(ctx, scores, 1.0f / std::sqrt(static_cast<float>(c.head_dim)));
    if (causal) {
        scores = ggml_diag_mask_inf(ctx, scores, 0);
    }
    scores = ggml_soft_max(ctx, scores);
    ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, vx, 1, 0, 2, 3));
    ggml_tensor* out = ggml_mul_mat(ctx, vt, scores);
    out = ggml_cont(ctx, ggml_permute(ctx, out, 0, 2, 1, 3));
    return ggml_reshape_2d(ctx, out, c.hidden_size, q_tokens);
}

ggml_tensor* build_mixed_attention(ggml_context* ctx,
                                   ggml_tensor* q,
                                   ggml_tensor* k,
                                   ggml_tensor* v,
                                   const HiDreamO1TextBlockGraphConfig& c) {
    q = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));

    ggml_tensor* full = build_attention_from_permuted(ctx, q, k, v, c, c.sequence_tokens, c.sequence_tokens, false);
    if (c.ar_prefix_tokens == 0) {
        return full;
    }
    ggml_tensor* q_ar = view_token_prefix(ctx, q, c.ar_prefix_tokens);
    ggml_tensor* k_ar = view_token_prefix(ctx, k, c.ar_prefix_tokens);
    ggml_tensor* v_ar = view_token_prefix(ctx, v, c.ar_prefix_tokens);
    ggml_tensor* ar = build_attention_from_permuted(ctx, q_ar, k_ar, v_ar, c, c.ar_prefix_tokens, c.ar_prefix_tokens, true);
    if (c.ar_prefix_tokens == c.sequence_tokens) {
        return ar;
    }
    ggml_tensor* gen = view_token_suffix_2d(ctx, full, c.ar_prefix_tokens);
    return ggml_concat(ctx, ar, gen, 1);
}

}  // namespace

ggml_tensor* build_hidream_o1_qwen3vl_text_block(ggml_context* ctx,
                                                 const HiDreamO1TextBlockGraphConfig& config,
                                                 const HiDreamO1TextBlockGraphTensors& t) {
    if (ctx == nullptr) throw std::invalid_argument("missing ggml context for HiDream text block");
    require_config(config);
    require_tensor(t.x, "x");
    require_tensor(t.input_layernorm_weight, "input_layernorm_weight");
    require_tensor(t.q_proj_weight, "q_proj_weight");
    require_tensor(t.k_proj_weight, "k_proj_weight");
    require_tensor(t.v_proj_weight, "v_proj_weight");
    require_tensor(t.o_proj_weight, "o_proj_weight");
    require_tensor(t.q_norm_weight, "q_norm_weight");
    require_tensor(t.k_norm_weight, "k_norm_weight");
    require_tensor(t.post_attention_layernorm_weight, "post_attention_layernorm_weight");
    require_tensor(t.gate_proj_weight, "gate_proj_weight");
    require_tensor(t.up_proj_weight, "up_proj_weight");
    require_tensor(t.down_proj_weight, "down_proj_weight");
    require_tensor(t.rope_cos, "rope_cos");
    require_tensor(t.rope_sin, "rope_sin");

    ggml_tensor* h = build_weighted_rms(ctx, t.x, t.input_layernorm_weight, config.rms_norm_eps);
    ggml_tensor* q = build_linear(ctx, h, t.q_proj_weight);
    ggml_tensor* k = build_linear(ctx, h, t.k_proj_weight);
    ggml_tensor* v = build_linear(ctx, h, t.v_proj_weight);
    q = ggml_reshape_3d(ctx, q, config.head_dim, config.num_attention_heads, config.sequence_tokens);
    k = ggml_reshape_3d(ctx, k, config.head_dim, config.num_key_value_heads, config.sequence_tokens);
    v = ggml_reshape_3d(ctx, v, config.head_dim, config.num_key_value_heads, config.sequence_tokens);
    q = build_head_rms(ctx, q, t.q_norm_weight, config.head_dim, config.rms_norm_eps);
    k = build_head_rms(ctx, k, t.k_norm_weight, config.head_dim, config.rms_norm_eps);
    q = build_interleaved_rope(ctx, q, t.rope_cos, t.rope_sin, config.num_attention_heads, config.head_dim, config.sequence_tokens);
    k = build_interleaved_rope(ctx, k, t.rope_cos, t.rope_sin, config.num_key_value_heads, config.head_dim, config.sequence_tokens);

    ggml_tensor* attn = build_mixed_attention(ctx, q, k, v, config);
    attn = build_linear(ctx, attn, t.o_proj_weight);
    ggml_tensor* x = ggml_add(ctx, t.x, attn);

    h = build_weighted_rms(ctx, x, t.post_attention_layernorm_weight, config.rms_norm_eps);
    ggml_tensor* gate = build_linear(ctx, h, t.gate_proj_weight);
    ggml_tensor* up = build_linear(ctx, h, t.up_proj_weight);
    ggml_tensor* mlp = build_linear(ctx, ggml_mul(ctx, ggml_silu(ctx, gate), up), t.down_proj_weight);
    x = ggml_add(ctx, x, mlp);
    x = ggml_cont(ctx, x);
    ggml_set_name(x, "hidream_qwen3vl_text_block_out");
    return x;
}

ggml_tensor* build_hidream_o1_pixeldit_visual_block(ggml_context* ctx,
                                                    const HiDreamO1VisualBlockGraphConfig& config,
                                                    const HiDreamO1VisualBlockGraphTensors& t) {
    if (ctx == nullptr) throw std::invalid_argument("missing ggml context for HiDream visual block");
    if (config.hidden_size <= 0 || config.intermediate_size <= 0 || config.num_heads <= 0 ||
        config.sequence_tokens <= 0 || config.hidden_size % config.num_heads != 0) {
        throw std::invalid_argument("invalid HiDream visual block graph dimensions");
    }
    require_tensor(t.x, "visual.x");
    require_tensor(t.norm1_weight, "visual.norm1_weight");
    require_tensor(t.norm1_bias, "visual.norm1_bias");
    require_tensor(t.qkv_weight, "visual.qkv_weight");
    require_tensor(t.qkv_bias, "visual.qkv_bias");
    require_tensor(t.proj_weight, "visual.proj_weight");
    require_tensor(t.proj_bias, "visual.proj_bias");
    require_tensor(t.norm2_weight, "visual.norm2_weight");
    require_tensor(t.norm2_bias, "visual.norm2_bias");
    require_tensor(t.fc1_weight, "visual.fc1_weight");
    require_tensor(t.fc1_bias, "visual.fc1_bias");
    require_tensor(t.fc2_weight, "visual.fc2_weight");
    require_tensor(t.fc2_bias, "visual.fc2_bias");

    const int64_t head_dim = config.hidden_size / config.num_heads;
    ggml_tensor* h = build_affine_layer_norm(ctx, t.x, t.norm1_weight, t.norm1_bias, config.norm_eps);
    ggml_tensor* qkv = build_linear_bias(ctx, h, t.qkv_weight, t.qkv_bias);
    ggml_tensor* q = ggml_cont(ctx, ggml_view_2d(ctx,
                                                 qkv,
                                                 config.hidden_size,
                                                 config.sequence_tokens,
                                                 qkv->nb[1],
                                                 0));
    ggml_tensor* k = ggml_cont(ctx, ggml_view_2d(ctx,
                                                 qkv,
                                                 config.hidden_size,
                                                 config.sequence_tokens,
                                                 qkv->nb[1],
                                                 static_cast<size_t>(config.hidden_size) * qkv->nb[0]));
    ggml_tensor* v = ggml_cont(ctx, ggml_view_2d(ctx,
                                                 qkv,
                                                 config.hidden_size,
                                                 config.sequence_tokens,
                                                 qkv->nb[1],
                                                 static_cast<size_t>(2 * config.hidden_size) * qkv->nb[0]));
    q = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, q, head_dim, config.num_heads, config.sequence_tokens), 0, 2, 1, 3));
    k = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, k, head_dim, config.num_heads, config.sequence_tokens), 0, 2, 1, 3));
    v = ggml_cont(ctx, ggml_permute(ctx, ggml_reshape_3d(ctx, v, head_dim, config.num_heads, config.sequence_tokens), 0, 2, 1, 3));

    HiDreamO1TextBlockGraphConfig attn_config;
    attn_config.hidden_size = config.hidden_size;
    attn_config.intermediate_size = config.intermediate_size;
    attn_config.num_attention_heads = config.num_heads;
    attn_config.num_key_value_heads = config.num_heads;
    attn_config.head_dim = head_dim;
    attn_config.sequence_tokens = config.sequence_tokens;
    attn_config.ar_prefix_tokens = 0;
    attn_config.rms_norm_eps = config.norm_eps;
    ggml_tensor* attn = build_attention_from_permuted(ctx,
                                                       q,
                                                       k,
                                                       v,
                                                       attn_config,
                                                       config.sequence_tokens,
                                                       config.sequence_tokens,
                                                       false);
    attn = build_linear_bias(ctx, attn, t.proj_weight, t.proj_bias);
    ggml_tensor* x = ggml_add(ctx, t.x, attn);

    h = build_affine_layer_norm(ctx, x, t.norm2_weight, t.norm2_bias, config.norm_eps);
    ggml_tensor* mlp = build_linear_bias(ctx,
                                         ggml_gelu(ctx, build_linear_bias(ctx, h, t.fc1_weight, t.fc1_bias)),
                                         t.fc2_weight,
                                         t.fc2_bias);
    x = ggml_cont(ctx, ggml_add(ctx, x, mlp));
    ggml_set_name(x, "hidream_pixeldit_visual_block_out");
    return x;
}

bool hidream_o1_qwen3vl_text_block_self_check(double* max_diff, std::string* error) {
    constexpr int64_t dim = 4;
    constexpr int64_t inter = 4;
    constexpr int64_t heads = 2;
    constexpr int64_t kv_heads = 1;
    constexpr int64_t head_dim = 2;
    constexpr int64_t tokens = 3;
    constexpr int64_t ar_tokens = 2;
    constexpr float eps = 1e-6f;

    ggml_init_params params{};
    params.mem_size = 64ull << 20;
    ggml_context* ctx = ggml_init(params);
    if (ctx == nullptr) {
        if (error) *error = "failed to allocate ggml context";
        return false;
    }

    auto make_1d = [&](int64_t n, float value) {
        ggml_tensor* tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n);
        float* data = static_cast<float*>(tensor->data);
        for (int64_t i = 0; i < n; ++i) data[i] = value;
        return tensor;
    };
    auto make_matrix = [&](int64_t in, int64_t out, float value) {
        ggml_tensor* tensor = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in, out);
        float* data = static_cast<float*>(tensor->data);
        for (int64_t i = 0; i < in * out; ++i) data[i] = value;
        return tensor;
    };
    auto make_identity = [&](int64_t n) {
        ggml_tensor* tensor = make_matrix(n, n, 0.0f);
        float* data = static_cast<float*>(tensor->data);
        for (int64_t i = 0; i < n; ++i) data[i + n * i] = 1.0f;
        return tensor;
    };

    std::vector<float> x_data = {
        0.10f, -0.20f, 0.30f, -0.40f,
        0.50f, 0.25f, -0.75f, 1.00f,
        -1.00f, 0.60f, 0.20f, -0.10f,
    };
    ggml_tensor* x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, tokens);
    std::memcpy(x->data, x_data.data(), x_data.size() * sizeof(float));

    ggml_tensor* rope_cos = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, 1, tokens);
    ggml_tensor* rope_sin = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, head_dim, 1, tokens);
    float* cos_data = static_cast<float*>(rope_cos->data);
    float* sin_data = static_cast<float*>(rope_sin->data);
    for (int64_t i = 0; i < head_dim * tokens; ++i) {
        cos_data[i] = 1.0f;
        sin_data[i] = 0.0f;
    }

    HiDreamO1TextBlockGraphConfig config;
    config.hidden_size = dim;
    config.intermediate_size = inter;
    config.num_attention_heads = heads;
    config.num_key_value_heads = kv_heads;
    config.head_dim = head_dim;
    config.sequence_tokens = tokens;
    config.ar_prefix_tokens = ar_tokens;
    config.rms_norm_eps = eps;

    HiDreamO1TextBlockGraphTensors t;
    t.x = x;
    t.input_layernorm_weight = make_1d(dim, 1.0f);
    t.q_proj_weight = make_matrix(dim, dim, 0.0f);
    t.k_proj_weight = make_matrix(dim, head_dim * kv_heads, 0.0f);
    t.v_proj_weight = make_matrix(dim, head_dim * kv_heads, 0.0f);
    t.o_proj_weight = make_matrix(dim, dim, 0.0f);
    t.q_norm_weight = make_1d(head_dim, 1.0f);
    t.k_norm_weight = make_1d(head_dim, 1.0f);
    t.post_attention_layernorm_weight = make_1d(dim, 1.0f);
    t.gate_proj_weight = make_identity(dim);
    t.up_proj_weight = make_identity(dim);
    t.down_proj_weight = make_identity(dim);
    t.rope_cos = rope_cos;
    t.rope_sin = rope_sin;

    ggml_tensor* out = build_hidream_o1_qwen3vl_text_block(ctx, config, t);
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 4096, false);
    ggml_build_forward_expand(graph, out);
    ggml_graph_compute_with_ctx(ctx, graph, 4);

    std::vector<float> expected(x_data.size(), 0.0f);
    for (int64_t tok = 0; tok < tokens; ++tok) {
        double mean_sq = 0.0;
        for (int64_t d = 0; d < dim; ++d) {
            const float v = x_data[static_cast<size_t>(tok * dim + d)];
            mean_sq += static_cast<double>(v) * static_cast<double>(v);
        }
        mean_sq /= static_cast<double>(dim);
        const float inv = 1.0f / std::sqrt(static_cast<float>(mean_sq) + eps);
        for (int64_t d = 0; d < dim; ++d) {
            const size_t idx = static_cast<size_t>(tok * dim + d);
            const float h = x_data[idx] * inv;
            expected[idx] = x_data[idx] + (1.0f / (1.0f + std::exp(-h))) * h * h;
        }
    }

    const float* actual = static_cast<const float*>(out->data);
    double diff = 0.0;
    for (size_t i = 0; i < expected.size(); ++i) {
        diff = std::max(diff, std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i])));
    }
    if (max_diff) *max_diff = diff;
    ggml_free(ctx);
    if (diff > 1e-5) {
        if (error) *error = "HiDream Qwen3-VL text block self-check output mismatch";
        return false;
    }
    return true;
}

bool hidream_o1_run_real_text_block_graph(const std::string& model_dir,
                                          int layer,
                                          int64_t sequence_tokens,
                                          HiDreamO1RealBlockRunSummary* summary,
                                          std::string* error) {
    if (summary == nullptr) return false;
    *summary = HiDreamO1RealBlockRunSummary{};
    summary->layer = layer;
    summary->sequence_tokens = sequence_tokens;
    summary->ar_prefix_tokens = sequence_tokens;

    if (layer < 0) {
        if (error) *error = "HiDream real block layer must be non-negative";
        return false;
    }
    if (sequence_tokens <= 0) {
        if (error) *error = "HiDream real block sequence token count must be positive";
        return false;
    }

    HiDreamO1NativeModelLayout layout;
    if (!load_hidream_o1_native_model_layout(model_dir, &layout)) {
        if (error) *error = layout.error;
        return false;
    }
    if (layer >= layout.text.num_hidden_layers) {
        if (error) *error = "HiDream real block layer is outside the text model";
        return false;
    }
    if (layout.text.hidden_size <= 0 || layout.text.intermediate_size <= 0 ||
        layout.text.num_attention_heads <= 0 || layout.text.num_key_value_heads <= 0 ||
        layout.text.head_dim <= 0) {
        if (error) *error = "HiDream real block has invalid text dimensions";
        return false;
    }

    HiDreamO1TensorCatalog catalog;
    if (!load_hidream_o1_tensor_catalog(model_dir, &catalog)) {
        if (error) *error = catalog.error;
        return false;
    }
    const std::vector<std::string> names = hidream_o1_text_block_tensor_names(layer);
    uint64_t block_payload_bytes = 0;
    for (const std::string& name : names) {
        HiDreamO1TensorInfo info;
        if (!find_hidream_o1_tensor(catalog, name, &info)) {
            if (error) *error = "missing HiDream tensor: " + name;
            return false;
        }
        block_payload_bytes += tensor_payload_bytes(info);
    }

    const uint64_t scratch_bytes = 512ull << 20;
    ggml_init_params params{};
    params.mem_size = static_cast<size_t>(block_payload_bytes + scratch_bytes);
    ggml_context* ctx = ggml_init(params);
    if (ctx == nullptr) {
        if (error) *error = "failed to allocate ggml context for HiDream real block";
        return false;
    }

    HiDreamO1TextBlockGraphConfig config;
    config.hidden_size = layout.text.hidden_size;
    config.intermediate_size = layout.text.intermediate_size;
    config.num_attention_heads = layout.text.num_attention_heads;
    config.num_key_value_heads = layout.text.num_key_value_heads;
    config.head_dim = layout.text.head_dim;
    config.sequence_tokens = sequence_tokens;
    config.ar_prefix_tokens = sequence_tokens;
    config.rms_norm_eps = layout.text.rms_norm_eps > 0.0 ? static_cast<float>(layout.text.rms_norm_eps) : 1e-6f;

    HiDreamO1TextBlockGraphTensors t;
    t.x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.hidden_size, sequence_tokens);
    fill_deterministic_block_input(t.x);
    t.rope_cos = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, config.head_dim, 1, sequence_tokens);
    t.rope_sin = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, config.head_dim, 1, sequence_tokens);
    fill_rope_tables(t.rope_cos, t.rope_sin, layout.text.rope_theta);

    const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".";
    int64_t loaded_payload_bytes = 0;
    const auto load_1d = [&](const std::string& suffix, ggml_tensor** out) {
        return load_f32_tensor_bytes(catalog, ctx, prefix + suffix, 1, out, &loaded_payload_bytes, error);
    };
    const auto load_2d = [&](const std::string& suffix, ggml_tensor** out) {
        return load_f32_tensor_bytes(catalog, ctx, prefix + suffix, 2, out, &loaded_payload_bytes, error);
    };

    bool ok = true;
    ok = ok && load_1d("input_layernorm.weight", &t.input_layernorm_weight);
    ok = ok && load_2d("self_attn.q_proj.weight", &t.q_proj_weight);
    ok = ok && load_2d("self_attn.k_proj.weight", &t.k_proj_weight);
    ok = ok && load_2d("self_attn.v_proj.weight", &t.v_proj_weight);
    ok = ok && load_2d("self_attn.o_proj.weight", &t.o_proj_weight);
    ok = ok && load_1d("self_attn.q_norm.weight", &t.q_norm_weight);
    ok = ok && load_1d("self_attn.k_norm.weight", &t.k_norm_weight);
    ok = ok && load_1d("post_attention_layernorm.weight", &t.post_attention_layernorm_weight);
    ok = ok && load_2d("mlp.gate_proj.weight", &t.gate_proj_weight);
    ok = ok && load_2d("mlp.up_proj.weight", &t.up_proj_weight);
    ok = ok && load_2d("mlp.down_proj.weight", &t.down_proj_weight);
    if (!ok) {
        ggml_free(ctx);
        return false;
    }

    ggml_tensor* out = nullptr;
    try {
        out = build_hidream_o1_qwen3vl_text_block(ctx, config, t);
    } catch (const std::exception& ex) {
        ggml_free(ctx);
        if (error) *error = ex.what();
        return false;
    }
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(graph, out);
    ggml_graph_compute_with_ctx(ctx, graph, 4);

    const int64_t output_values = out->ne[0] * out->ne[1];
    const float* data = static_cast<const float*>(out->data);
    double checksum = 0.0;
    double l2 = 0.0;
    double max_abs = 0.0;
    for (int64_t i = 0; i < output_values; ++i) {
        const double v = data[i];
        checksum += v;
        l2 += v * v;
        max_abs = std::max(max_abs, std::abs(v));
    }

    summary->hidden_size = config.hidden_size;
    summary->intermediate_size = config.intermediate_size;
    summary->payload_bytes = loaded_payload_bytes;
    summary->output_values = output_values;
    summary->output_checksum = checksum;
    summary->output_l2 = std::sqrt(l2);
    summary->output_max_abs = max_abs;
    ggml_free(ctx);
    return true;
}

bool hidream_o1_run_real_visual_block_graph(const std::string& model_dir,
                                            int layer,
                                            int64_t sequence_tokens,
                                            HiDreamO1RealBlockRunSummary* summary,
                                            std::string* error) {
    if (summary == nullptr) return false;
    *summary = HiDreamO1RealBlockRunSummary{};
    summary->layer = layer;
    summary->sequence_tokens = sequence_tokens;

    if (layer < 0) {
        if (error) *error = "HiDream real visual block layer must be non-negative";
        return false;
    }
    if (sequence_tokens <= 0) {
        if (error) *error = "HiDream real visual block sequence token count must be positive";
        return false;
    }

    HiDreamO1NativeModelLayout layout;
    if (!load_hidream_o1_native_model_layout(model_dir, &layout)) {
        if (error) *error = layout.error;
        return false;
    }
    if (layer >= layout.vision.depth) {
        if (error) *error = "HiDream real visual block layer is outside the visual model";
        return false;
    }
    if (layout.vision.hidden_size <= 0 || layout.vision.intermediate_size <= 0 || layout.vision.num_heads <= 0 ||
        layout.vision.hidden_size % layout.vision.num_heads != 0) {
        if (error) *error = "HiDream real visual block has invalid dimensions";
        return false;
    }

    HiDreamO1TensorCatalog catalog;
    if (!load_hidream_o1_tensor_catalog(model_dir, &catalog)) {
        if (error) *error = catalog.error;
        return false;
    }
    const std::vector<std::string> names = hidream_o1_visual_block_tensor_names(layer);
    uint64_t block_payload_bytes = 0;
    for (const std::string& name : names) {
        HiDreamO1TensorInfo info;
        if (!find_hidream_o1_tensor(catalog, name, &info)) {
            if (error) *error = "missing HiDream visual tensor: " + name;
            return false;
        }
        block_payload_bytes += tensor_payload_bytes(info);
    }

    ggml_init_params params{};
    params.mem_size = static_cast<size_t>(block_payload_bytes + (512ull << 20));
    ggml_context* ctx = ggml_init(params);
    if (ctx == nullptr) {
        if (error) *error = "failed to allocate ggml context for HiDream real visual block";
        return false;
    }

    HiDreamO1VisualBlockGraphConfig config;
    config.hidden_size = layout.vision.hidden_size;
    config.intermediate_size = layout.vision.intermediate_size;
    config.num_heads = layout.vision.num_heads;
    config.sequence_tokens = sequence_tokens;
    config.norm_eps = 1e-6f;

    HiDreamO1VisualBlockGraphTensors t;
    t.x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, config.hidden_size, sequence_tokens);
    fill_deterministic_block_input(t.x);

    const std::string prefix = "model.visual.blocks." + std::to_string(layer) + ".";
    int64_t loaded_payload_bytes = 0;
    const auto load_1d = [&](const std::string& suffix, ggml_tensor** out) {
        return load_f32_tensor_bytes(catalog, ctx, prefix + suffix, 1, out, &loaded_payload_bytes, error);
    };
    const auto load_2d = [&](const std::string& suffix, ggml_tensor** out) {
        return load_f32_tensor_bytes(catalog, ctx, prefix + suffix, 2, out, &loaded_payload_bytes, error);
    };

    bool ok = true;
    ok = ok && load_1d("norm1.weight", &t.norm1_weight);
    ok = ok && load_1d("norm1.bias", &t.norm1_bias);
    ok = ok && load_2d("attn.qkv.weight", &t.qkv_weight);
    ok = ok && load_1d("attn.qkv.bias", &t.qkv_bias);
    ok = ok && load_2d("attn.proj.weight", &t.proj_weight);
    ok = ok && load_1d("attn.proj.bias", &t.proj_bias);
    ok = ok && load_1d("norm2.weight", &t.norm2_weight);
    ok = ok && load_1d("norm2.bias", &t.norm2_bias);
    ok = ok && load_2d("mlp.linear_fc1.weight", &t.fc1_weight);
    ok = ok && load_1d("mlp.linear_fc1.bias", &t.fc1_bias);
    ok = ok && load_2d("mlp.linear_fc2.weight", &t.fc2_weight);
    ok = ok && load_1d("mlp.linear_fc2.bias", &t.fc2_bias);
    if (!ok) {
        ggml_free(ctx);
        return false;
    }

    ggml_tensor* out = nullptr;
    try {
        out = build_hidream_o1_pixeldit_visual_block(ctx, config, t);
    } catch (const std::exception& ex) {
        ggml_free(ctx);
        if (error) *error = ex.what();
        return false;
    }
    ggml_cgraph* graph = ggml_new_graph_custom(ctx, 8192, false);
    ggml_build_forward_expand(graph, out);
    ggml_graph_compute_with_ctx(ctx, graph, 4);

    const int64_t output_values = out->ne[0] * out->ne[1];
    const float* data = static_cast<const float*>(out->data);
    double checksum = 0.0;
    double l2 = 0.0;
    double max_abs = 0.0;
    for (int64_t i = 0; i < output_values; ++i) {
        const double v = data[i];
        checksum += v;
        l2 += v * v;
        max_abs = std::max(max_abs, std::abs(v));
    }

    summary->hidden_size = config.hidden_size;
    summary->intermediate_size = config.intermediate_size;
    summary->payload_bytes = loaded_payload_bytes;
    summary->output_values = output_values;
    summary->output_checksum = checksum;
    summary->output_l2 = std::sqrt(l2);
    summary->output_max_abs = max_abs;
    ggml_free(ctx);
    return true;
}

}  // namespace utopic
