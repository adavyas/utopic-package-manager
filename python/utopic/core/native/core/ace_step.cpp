#include "ace_step.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>

namespace utopic {
namespace {

bool fail(std::string& error, const std::string& message) {
    error = message;
    return false;
}

bool checked_element_count(const std::vector<int64_t>& dims, int64_t& elements) {
    elements = 1;
    for (const int64_t dim : dims) {
        if (dim <= 0) return false;
        if (elements > std::numeric_limits<int64_t>::max() / dim) return false;
        elements *= dim;
    }
    return true;
}

float round_to_bf16_rne(float x) {
    uint32_t bits = 0;
    std::memcpy(&bits, &x, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1u;
    bits += 0x7fffu + lsb;
    bits &= 0xffff0000u;
    std::memcpy(&x, &bits, sizeof(x));
    return x;
}

}  // namespace

AceStepRuntimeConfig default_ace_step_runtime_config() {
    return AceStepRuntimeConfig{
        "ace-step-1.5",
        48000,
        64,
        1920,
        8,
        512,
        64,
        false,
        true,
        true,
        true,
        true,
        true,
        true,
    };
}

bool ace_read_f32_tensor(const std::string& path, AceTensor& out, std::string& error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) return fail(error, "missing tensor file: " + path);

    int64_t ndim = 0;
    in.read(reinterpret_cast<char*>(&ndim), sizeof(ndim));
    if (!in || ndim <= 0 || ndim > 8) return fail(error, "invalid tensor ndim: " + path);

    std::vector<int64_t> dims(static_cast<size_t>(ndim), 0);
    for (int64_t i = 0; i < ndim; ++i) {
        in.read(reinterpret_cast<char*>(&dims[static_cast<size_t>(i)]), sizeof(int64_t));
        if (!in || dims[static_cast<size_t>(i)] <= 0) return fail(error, "invalid tensor dims: " + path);
    }

    int64_t elements = 0;
    if (!checked_element_count(dims, elements)) return fail(error, "invalid tensor element count: " + path);
    std::vector<float> data(static_cast<size_t>(elements), 0.0f);
    in.read(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size() * sizeof(float)));
    if (!in) return fail(error, "short tensor payload: " + path);

    char extra = 0;
    if (in.read(&extra, 1)) return fail(error, "extra bytes after tensor payload: " + path);

    out.dims = std::move(dims);
    out.data = std::move(data);
    return true;
}

bool ace_replay_euler_sampler(const AceTensor& initial_latent,
                              const AceTensor& velocities,
                              const AceTensor& timesteps,
                              std::vector<float>& out,
                              std::string& error) {
    if (initial_latent.dims.size() != 3) {
        return fail(error, "initial_latent must be [batch, frames, channels]");
    }
    if (velocities.dims.size() != 4) {
        return fail(error, "velocities must be [steps, batch, frames, channels]");
    }
    if (timesteps.dims.size() != 1) {
        return fail(error, "timesteps must be [steps]");
    }

    const int64_t steps = velocities.dims[0];
    if (steps <= 0 || timesteps.dims[0] != steps) return fail(error, "velocity/timestep step count mismatch");
    if (velocities.dims[1] != initial_latent.dims[0] ||
        velocities.dims[2] != initial_latent.dims[1] ||
        velocities.dims[3] != initial_latent.dims[2]) {
        return fail(error, "velocity latent dimensions must match initial_latent");
    }

    const size_t latent_elems = initial_latent.data.size();
    if (velocities.data.size() != static_cast<size_t>(steps) * latent_elems) {
        return fail(error, "velocity payload size mismatch");
    }

    out = initial_latent.data;
    for (int64_t step = 0; step < steps; ++step) {
        const float t = timesteps.data[static_cast<size_t>(step)];
        const float t_next = (step + 1 < steps) ? timesteps.data[static_cast<size_t>(step + 1)] : 0.0f;
        const float dt = t - t_next;
        const size_t velocity_offset = static_cast<size_t>(step) * latent_elems;
        for (size_t i = 0; i < latent_elems; ++i) {
            out[i] = round_to_bf16_rne(out[i] - velocities.data[velocity_offset + i] * dt);
        }
    }
    return true;
}

AceStepShape ace_step_shape_for_seconds(const AceStepRuntimeConfig& cfg, double seconds) {
    const double safe_seconds = std::max(0.0, seconds);
    const auto samples = static_cast<int64_t>(std::llround(safe_seconds * static_cast<double>(cfg.sample_rate)));
    const int hop = std::max(1, cfg.latent_hop_samples);
    const int frames = std::max<int64_t>(1, (samples + hop - 1) / hop);
    const int batch = cfg.cfg_batch_single_request ? 2 : 1;
    return AceStepShape{
        samples,
        frames,
        batch,
        static_cast<int64_t>(batch) * static_cast<int64_t>(cfg.latent_channels) * static_cast<int64_t>(frames),
    };
}

std::vector<float> ace_xl_turbo_timesteps(int steps, float shift) {
    const int n = std::max(1, std::min(20, steps));
    std::vector<float> timesteps;
    timesteps.reserve(static_cast<size_t>(n));
    for (int i = 0; i < n; ++i) {
        float t = 1.0f - static_cast<float>(i) / static_cast<float>(n);
        if (shift != 1.0f) {
            t = shift * t / (1.0f + (shift - 1.0f) * t);
        }
        timesteps.push_back(t);
    }
    return timesteps;
}

}  // namespace utopic
