#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace utopic {

struct audio_engine_wav_params {
    std::string   output_path;
    const float * samples       = nullptr;
    size_t        sample_count  = 0;
    int32_t       sample_rate   = 24000;
    int32_t       channel_count = 1;
};

struct audio_engine_result {
    bool        ok = false;
    std::string error_message;
    std::string artifact_path;
    size_t      sample_count  = 0;
    int32_t     sample_rate   = 0;
    int32_t     channel_count = 0;
    int64_t     duration_ms   = 0;
};

bool audio_engine_write_wav(const audio_engine_wav_params & params, audio_engine_result & result);

}  // namespace utopic
