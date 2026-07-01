#include "hidream_o1_native.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>

namespace utopic {

namespace {

std::string env_or(const char* name, const std::string& fallback) {
    const char* value = std::getenv(name);
    return value && value[0] ? std::string(value) : fallback;
}

std::string home_path(const char* suffix) {
    const char* home = std::getenv("HOME");
    return std::string(home && home[0] ? home : "") + suffix;
}

std::string join_path(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (a.back() == '/') return a + b;
    return a + "/" + b;
}

std::string shell_quote(const std::string& value) {
    std::string quoted = "'";
    for (char c : value) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

bool read_text_file(const std::string& path, std::string* out) {
    if (out == nullptr) return false;
    std::ifstream in(path, std::ios::binary);
    if (!in) return false;
    *out = std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
    return true;
}

size_t find_matching_brace(const std::string& text, size_t open_pos) {
    bool in_string = false;
    bool escape = false;
    int depth = 0;
    for (size_t i = open_pos; i < text.size(); ++i) {
        const char c = text[i];
        if (escape) {
            escape = false;
            continue;
        }
        if (c == '\\' && in_string) {
            escape = true;
            continue;
        }
        if (c == '"') {
            in_string = !in_string;
            continue;
        }
        if (in_string) continue;
        if (c == '{') {
            depth++;
        } else if (c == '}') {
            depth--;
            if (depth == 0) return i;
        }
    }
    return std::string::npos;
}

bool parse_json_string(const std::string& text, size_t* pos, std::string* out) {
    if (pos == nullptr || out == nullptr || *pos >= text.size() || text[*pos] != '"') return false;
    std::string result;
    bool escape = false;
    for (size_t i = *pos + 1; i < text.size(); ++i) {
        const char c = text[i];
        if (escape) {
            result.push_back(c);
            escape = false;
            continue;
        }
        if (c == '\\') {
            escape = true;
            continue;
        }
        if (c == '"') {
            *pos = i + 1;
            *out = result;
            return true;
        }
        result.push_back(c);
    }
    return false;
}

void skip_ws(const std::string& text, size_t* pos) {
    while (pos != nullptr && *pos < text.size()) {
        const char c = text[*pos];
        if (c != ' ' && c != '\n' && c != '\r' && c != '\t') break;
        ++(*pos);
    }
}

uint64_t read_le_u64(const unsigned char bytes[8]) {
    uint64_t v = 0;
    for (int i = 7; i >= 0; --i) {
        v = (v << 8) | static_cast<uint64_t>(bytes[i]);
    }
    return v;
}

}  // namespace

HiDreamO1RuntimeConfig default_hidream_o1_runtime_config() {
    return HiDreamO1RuntimeConfig{};
}

HiDreamO1Shape hidream_o1_shape_for_size(const HiDreamO1RuntimeConfig& cfg, int width, int height) {
    HiDreamO1Shape shape;
    shape.width = width;
    shape.height = height;
    shape.patch_size = cfg.patch_size;
    shape.patch_dim = cfg.patch_size * cfg.patch_size * 3;
    if (width > 0 && height > 0 && cfg.patch_size > 0) {
        shape.patch_tokens = static_cast<int64_t>(width / cfg.patch_size) *
                             static_cast<int64_t>(height / cfg.patch_size);
        shape.pixel_values = static_cast<int64_t>(width) * static_cast<int64_t>(height) * 3;
    }
    return shape;
}

std::vector<int> hidream_o1_dev_timesteps() {
    return {999, 987, 974, 960, 945, 929, 913, 895, 877, 857, 836, 814, 790, 764,
            737, 707, 675, 640, 602, 560, 515, 464, 409, 347, 278, 199, 110, 8};
}

std::vector<float> hidream_o1_dev_sigmas() {
    std::vector<float> sigmas;
    const std::vector<int> timesteps = hidream_o1_dev_timesteps();
    sigmas.reserve(timesteps.size() + 1);
    for (int t : timesteps) {
        sigmas.push_back(static_cast<float>(t) / 1000.0f);
    }
    sigmas.push_back(0.0f);
    return sigmas;
}

std::string hidream_o1_default_model_dir() {
    return env_or("UTOPIC_HIDREAM_MODEL_DIR",
                  home_path("/.cache/utopic/models/HiDream-O1-Image-Dev-2604"));
}

std::string hidream_o1_default_model_path() {
    return env_or("UTOPIC_HIDREAM_MODEL",
                  home_path("/.cache/utopic/models/hidream-o1/hidream_o1_image_dev_bf16.safetensors"));
}

std::string hidream_o1_default_sd_cli() {
    return env_or("UTOPIC_HIDREAM_SDCLI", home_path("/stable-diffusion.cpp/build-gb10/bin/sd-cli"));
}

std::string build_hidream_o1_oracle_command(const HiDreamO1RunRequest& req) {
    const HiDreamO1RuntimeConfig cfg = default_hidream_o1_runtime_config();
    const int width = req.width > 0 ? req.width : 1024;
    const int height = req.height > 0 ? req.height : 1024;
    const int steps = req.steps > 0 ? req.steps : cfg.default_steps;
    const float cfg_scale = req.cfg_scale > 0.0f ? req.cfg_scale : 1.0f;

    std::ostringstream cmd;
    cmd << shell_quote(req.sd_cli)
        << " -m " << shell_quote(req.model_path)
        << " -p " << shell_quote(req.prompt)
        << " -o " << shell_quote(req.output_path)
        << " -H " << height
        << " -W " << width
        << " --steps " << steps
        << " -s " << req.seed
        << " --cfg-scale " << cfg_scale
        << " -v";
    if (!req.extra_args.empty()) {
        cmd << " " << req.extra_args;
    }
    return cmd.str();
}

std::string build_hidream_o1_command(const HiDreamO1RunRequest& req) {
    return build_hidream_o1_oracle_command(req);
}

bool hidream_o1_file_exists(const std::string& path) {
    struct stat st {};
    return !path.empty() && stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

bool hidream_o1_dir_exists(const std::string& path) {
    struct stat st {};
    return !path.empty() && stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool load_hidream_o1_shard_manifest(const std::string& model_dir, HiDreamO1ShardManifest* manifest) {
    if (manifest == nullptr) return false;
    *manifest = HiDreamO1ShardManifest{};
    manifest->model_dir = model_dir;
    manifest->index_path = join_path(model_dir, "model.safetensors.index.json");

    std::string text;
    if (!read_text_file(manifest->index_path, &text)) {
        manifest->error = "missing or unreadable index: " + manifest->index_path;
        return false;
    }

    const size_t weight_map_key = text.find("\"weight_map\"");
    if (weight_map_key == std::string::npos) {
        manifest->error = "index has no weight_map object";
        return false;
    }
    const size_t open = text.find('{', weight_map_key);
    if (open == std::string::npos) {
        manifest->error = "weight_map has no opening brace";
        return false;
    }
    const size_t close = find_matching_brace(text, open);
    if (close == std::string::npos || close <= open) {
        manifest->error = "weight_map has no matching closing brace";
        return false;
    }

    std::set<std::string> shards;
    size_t pos = open + 1;
    while (pos < close) {
        skip_ws(text, &pos);
        if (pos >= close || text[pos] == '}') break;
        std::string tensor_name;
        if (!parse_json_string(text, &pos, &tensor_name)) {
            manifest->error = "failed to parse tensor name in weight_map";
            return false;
        }
        skip_ws(text, &pos);
        if (pos >= close || text[pos] != ':') {
            manifest->error = "malformed weight_map entry for tensor: " + tensor_name;
            return false;
        }
        ++pos;
        skip_ws(text, &pos);
        std::string shard_file;
        if (!parse_json_string(text, &pos, &shard_file)) {
            manifest->error = "failed to parse shard file for tensor: " + tensor_name;
            return false;
        }
        manifest->entries.push_back({tensor_name, shard_file});
        shards.insert(shard_file);
        skip_ws(text, &pos);
        if (pos < close && text[pos] == ',') ++pos;
    }

    manifest->shard_files.assign(shards.begin(), shards.end());
    if (manifest->entries.empty()) {
        manifest->error = "weight_map is empty";
        return false;
    }
    return true;
}

HiDreamO1SafetensorsHeader inspect_hidream_o1_safetensors_header(const std::string& file_path) {
    HiDreamO1SafetensorsHeader result;
    result.file_path = file_path;

    std::ifstream in(file_path, std::ios::binary);
    if (!in) {
        result.error = "missing or unreadable safetensors shard";
        return result;
    }
    unsigned char len_bytes[8] = {};
    in.read(reinterpret_cast<char*>(len_bytes), sizeof(len_bytes));
    if (in.gcount() != static_cast<std::streamsize>(sizeof(len_bytes))) {
        result.error = "short safetensors header length";
        return result;
    }
    result.header_bytes = read_le_u64(len_bytes);
    if (result.header_bytes == 0 || result.header_bytes > (512ull << 20)) {
        result.error = "invalid safetensors header length";
        return result;
    }

    std::string header(static_cast<size_t>(result.header_bytes), '\0');
    in.read(&header[0], static_cast<std::streamsize>(header.size()));
    if (in.gcount() != static_cast<std::streamsize>(header.size())) {
        result.error = "short safetensors header body";
        return result;
    }

    size_t pos = 0;
    while ((pos = header.find("\"dtype\"", pos)) != std::string::npos) {
        pos = header.find(':', pos);
        if (pos == std::string::npos) break;
        ++pos;
        skip_ws(header, &pos);
        std::string dtype;
        if (parse_json_string(header, &pos, &dtype)) {
            result.dtype_counts[dtype]++;
        }
    }

    pos = 0;
    while ((pos = header.find("\"data_offsets\"", pos)) != std::string::npos) {
        result.tensor_count++;
        pos += std::strlen("\"data_offsets\"");
    }
    return result;
}

}  // namespace utopic
