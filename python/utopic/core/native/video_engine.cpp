#include "video_engine.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <system_error>
#include <vector>

namespace utopic {

namespace fs = std::filesystem;

static void video_engine_fail(video_engine_result & result, const std::string & message) {
    result.ok            = false;
    result.error_message = message.empty() ? "video artifact generation failed" : message;
}

static bool video_engine_validate(const video_engine_frames_params & params, video_engine_result & result) {
    if (params.output_dir.empty()) {
        video_engine_fail(result, "output_dir is required");
        return false;
    }
    if (!params.frames || params.frame_count == 0) {
        video_engine_fail(result, "frames are required");
        return false;
    }
    if (params.width <= 0 || params.height <= 0) {
        video_engine_fail(result, "width and height must be positive");
        return false;
    }
    if (params.channel_count != 3) {
        video_engine_fail(result, "channel_count must be 3 for RGB frames");
        return false;
    }
    if (params.fps <= 0) {
        video_engine_fail(result, "fps must be positive");
        return false;
    }
    const uint64_t pixels = (uint64_t) params.width * (uint64_t) params.height;
    if (pixels > std::numeric_limits<size_t>::max() / (size_t) params.channel_count) {
        video_engine_fail(result, "frame dimensions are too large");
        return false;
    }
    return true;
}

static std::string frame_filename(size_t index) {
    char name[64];
    snprintf(name, sizeof(name), "frame_%06zu.ppm", index);
    return name;
}

static bool write_ppm_frame(const fs::path & path,
                            const uint8_t * frame,
                            int32_t         width,
                            int32_t         height,
                            size_t          frame_bytes,
                            video_engine_result & result) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        video_engine_fail(result, std::string("failed to open frame artifact: ") + strerror(errno));
        return false;
    }
    out << "P6\n" << width << " " << height << "\n255\n";
    out.write((const char *) frame, (std::streamsize) frame_bytes);
    if (!out) {
        video_engine_fail(result, "failed to write frame artifact");
        return false;
    }
    return true;
}

static std::string json_escape(const std::string & value) {
    std::string out;
    for (char ch : value) {
        switch (ch) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += ch;     break;
        }
    }
    return out;
}

static bool write_metadata(const fs::path &                 path,
                           const video_engine_result &      result,
                           const std::vector<std::string> & frame_names,
                           video_engine_result &            error_result) {
    std::ofstream out(path);
    if (!out) {
        video_engine_fail(error_result, std::string("failed to open video metadata: ") + strerror(errno));
        return false;
    }
    out << "{\n";
    out << "  \"type\": \"video_frames\",\n";
    out << "  \"width\": " << result.width << ",\n";
    out << "  \"height\": " << result.height << ",\n";
    out << "  \"frame_count\": " << result.frame_count << ",\n";
    out << "  \"fps\": " << result.fps << ",\n";
    out << "  \"duration_ms\": " << result.duration_ms << ",\n";
    out << "  \"frames\": [\n";
    for (size_t i = 0; i < frame_names.size(); ++i) {
        out << "    \"" << json_escape(frame_names[i]) << "\"";
        out << (i + 1 == frame_names.size() ? "\n" : ",\n");
    }
    out << "  ]\n";
    out << "}\n";
    if (!out) {
        video_engine_fail(error_result, "failed to write video metadata");
        return false;
    }
    return true;
}

bool video_engine_write_frames(const video_engine_frames_params & params, video_engine_result & result) {
    result = video_engine_result();
    if (!video_engine_validate(params, result)) {
        return false;
    }

    const fs::path output_dir(params.output_dir);
    std::error_code ec;
    fs::create_directories(output_dir, ec);
    if (ec) {
        video_engine_fail(result, std::string("failed to create output directory: ") + ec.message());
        return false;
    }

    const size_t frame_bytes = (size_t) params.width * (size_t) params.height * (size_t) params.channel_count;
    std::vector<std::string> frame_names;
    frame_names.reserve(params.frame_count);
    for (size_t i = 0; i < params.frame_count; ++i) {
        std::string name = frame_filename(i);
        frame_names.push_back(name);
        const uint8_t * frame = params.frames + i * frame_bytes;
        if (!write_ppm_frame(output_dir / name, frame, params.width, params.height, frame_bytes, result)) {
            return false;
        }
    }

    result.ok            = true;
    result.artifact_path = output_dir.string();
    result.metadata_path = (output_dir / "metadata.json").string();
    result.frame_count   = params.frame_count;
    result.width         = params.width;
    result.height        = params.height;
    result.fps           = params.fps;
    result.duration_ms   = std::max<int64_t>(
        1, (int64_t) (((1000.0 * (double) params.frame_count) / (double) params.fps) + 0.999));

    if (!write_metadata(output_dir / "metadata.json", result, frame_names, result)) {
        return false;
    }
    return true;
}

}  // namespace utopic
