#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace utopic {

struct AceStepRuntimeConfig {
    const char* model_id;
    int sample_rate;
    int latent_channels;
    int latent_hop_samples;
    int default_steps;
    int default_vae_chunk;
    int default_vae_overlap;
    bool cfg_batch_single_request;
    bool use_cuda_graph_buckets;
    bool use_aten_attention_bridge;
    bool use_dit_projection_fusion;
    bool use_pinned_host_staging;
    bool use_static_input_upload_cache;
    bool use_tiled_vae_decode;
};

struct AceStepShape {
    int64_t audio_samples;
    int latent_frames;
    int dit_batch;
    int64_t latent_elements;
};

struct AceTensor {
    std::vector<int64_t> dims;
    std::vector<float> data;
};

AceStepRuntimeConfig default_ace_step_runtime_config();
AceStepShape ace_step_shape_for_seconds(const AceStepRuntimeConfig& cfg, double seconds);
std::vector<float> ace_xl_turbo_timesteps(int steps, float shift = 1.0f);
bool ace_read_f32_tensor(const std::string& path, AceTensor& out, std::string& error);
bool ace_replay_euler_sampler(const AceTensor& initial_latent,
                              const AceTensor& velocities,
                              const AceTensor& timesteps,
                              std::vector<float>& out,
                              std::string& error);

}  // namespace utopic
