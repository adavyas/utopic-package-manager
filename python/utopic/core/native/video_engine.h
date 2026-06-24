#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace utopic {

struct video_engine_frames_params {
    std::string     output_dir;
    const uint8_t * frames        = nullptr;
    size_t          frame_count   = 0;
    int32_t         width         = 0;
    int32_t         height        = 0;
    int32_t         channel_count = 3;
    int32_t         fps           = 16;
};

struct video_engine_result {
    bool        ok = false;
    std::string error_message;
    std::string artifact_path;
    std::string metadata_path;
    size_t      frame_count = 0;
    int32_t     width       = 0;
    int32_t     height      = 0;
    int32_t     fps         = 0;
    int64_t     duration_ms = 0;
};

bool video_engine_write_frames(const video_engine_frames_params & params, video_engine_result & result);

}  // namespace utopic
