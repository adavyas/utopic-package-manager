#include "audio_engine.h"

#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <system_error>

namespace utopic {

namespace fs = std::filesystem;

static void audio_engine_fail(audio_engine_result & result, const std::string & message) {
    result.ok            = false;
    result.error_message = message.empty() ? "audio artifact generation failed" : message;
}

static bool audio_engine_validate(const audio_engine_wav_params & params, audio_engine_result & result) {
    if (params.output_path.empty()) {
        audio_engine_fail(result, "output_path is required");
        return false;
    }
    if (!params.samples || params.sample_count == 0) {
        audio_engine_fail(result, "samples are required");
        return false;
    }
    if (params.sample_rate <= 0) {
        audio_engine_fail(result, "sample_rate must be positive");
        return false;
    }
    if (params.channel_count <= 0) {
        audio_engine_fail(result, "channel_count must be positive");
        return false;
    }
    if ((params.sample_count % (size_t) params.channel_count) != 0) {
        audio_engine_fail(result, "sample_count must be divisible by channel_count");
        return false;
    }
    return true;
}

static int16_t pcm16_from_float(float value) {
    if (!std::isfinite(value)) {
        value = 0.0f;
    }
    value = std::max(-1.0f, std::min(1.0f, value));
    if (value >= 0.0f) {
        return (int16_t) std::lrintf(value * (float) std::numeric_limits<int16_t>::max());
    }
    return (int16_t) std::lrintf(value * 32768.0f);
}

static void write_u16_le(std::ofstream & out, uint16_t value) {
    char bytes[2] = {
        (char) (value & 0xff),
        (char) ((value >> 8) & 0xff),
    };
    out.write(bytes, sizeof(bytes));
}

static void write_u32_le(std::ofstream & out, uint32_t value) {
    char bytes[4] = {
        (char) (value & 0xff),
        (char) ((value >> 8) & 0xff),
        (char) ((value >> 16) & 0xff),
        (char) ((value >> 24) & 0xff),
    };
    out.write(bytes, sizeof(bytes));
}

bool audio_engine_write_wav(const audio_engine_wav_params & params, audio_engine_result & result) {
    result = audio_engine_result();
    if (!audio_engine_validate(params, result)) {
        return false;
    }

    const uint32_t bytes_per_sample = 2;
    const uint64_t data_bytes_64    = params.sample_count * bytes_per_sample;
    if (data_bytes_64 > std::numeric_limits<uint32_t>::max()) {
        audio_engine_fail(result, "audio artifact is too large for WAV");
        return false;
    }
    const uint32_t data_bytes = (uint32_t) data_bytes_64;
    const uint32_t riff_size  = 36u + data_bytes;
    const uint16_t channels   = (uint16_t) params.channel_count;
    const uint32_t sample_rate = (uint32_t) params.sample_rate;
    const uint16_t block_align = (uint16_t) (channels * bytes_per_sample);
    const uint32_t byte_rate   = sample_rate * block_align;

    const fs::path path(params.output_path);
    if (path.has_parent_path()) {
        std::error_code ec;
        fs::create_directories(path.parent_path(), ec);
        if (ec) {
            audio_engine_fail(result, std::string("failed to create output directory: ") + ec.message());
            return false;
        }
    }

    std::ofstream out(params.output_path, std::ios::binary);
    if (!out) {
        audio_engine_fail(result, std::string("failed to open WAV artifact: ") + strerror(errno));
        return false;
    }

    out.write("RIFF", 4);
    write_u32_le(out, riff_size);
    out.write("WAVE", 4);
    out.write("fmt ", 4);
    write_u32_le(out, 16);
    write_u16_le(out, 1);
    write_u16_le(out, channels);
    write_u32_le(out, sample_rate);
    write_u32_le(out, byte_rate);
    write_u16_le(out, block_align);
    write_u16_le(out, 16);
    out.write("data", 4);
    write_u32_le(out, data_bytes);

    for (size_t i = 0; i < params.sample_count; ++i) {
        write_u16_le(out, (uint16_t) pcm16_from_float(params.samples[i]));
    }
    if (!out) {
        audio_engine_fail(result, "failed to write WAV artifact");
        return false;
    }

    const size_t frames = params.sample_count / (size_t) params.channel_count;
    result.ok            = true;
    result.artifact_path = params.output_path;
    result.sample_count  = params.sample_count;
    result.sample_rate   = params.sample_rate;
    result.channel_count = params.channel_count;
    result.duration_ms   = std::max<int64_t>(
        1, (int64_t) std::ceil((1000.0 * (double) frames) / (double) params.sample_rate));
    return true;
}

}  // namespace utopic
