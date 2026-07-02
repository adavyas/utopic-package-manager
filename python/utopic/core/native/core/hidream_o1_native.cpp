#include "hidream_o1_native.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <vector>

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

void skip_ws(const std::string& text, size_t* pos);

bool extract_json_object(const std::string& text, const std::string& key, std::string* out) {
    if (out == nullptr) return false;
    const size_t key_pos = text.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return false;
    const size_t open = text.find('{', key_pos);
    if (open == std::string::npos) return false;
    const size_t close = find_matching_brace(text, open);
    if (close == std::string::npos || close <= open) return false;
    *out = text.substr(open, close - open + 1);
    return true;
}

bool parse_json_number(const std::string& text, const std::string& key, double* out) {
    if (out == nullptr) return false;
    const size_t key_pos = text.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return false;
    size_t pos = text.find(':', key_pos);
    if (pos == std::string::npos) return false;
    ++pos;
    skip_ws(text, &pos);
    const size_t begin = pos;
    while (pos < text.size()) {
        const char c = text[pos];
        if ((c >= '0' && c <= '9') || c == '-' || c == '+' || c == '.' || c == 'e' || c == 'E') {
            ++pos;
            continue;
        }
        break;
    }
    if (begin == pos) return false;
    *out = std::strtod(text.substr(begin, pos - begin).c_str(), nullptr);
    return true;
}

bool parse_json_int(const std::string& text, const std::string& key, int* out) {
    double value = 0.0;
    if (!parse_json_number(text, key, &value) || out == nullptr) return false;
    *out = static_cast<int>(value);
    return true;
}

bool parse_json_string_after_key(const std::string& text, const std::string& key, std::string* out) {
    if (out == nullptr) return false;
    const size_t key_pos = text.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return false;
    size_t pos = text.find(':', key_pos);
    if (pos == std::string::npos) return false;
    ++pos;
    skip_ws(text, &pos);
    return parse_json_string(text, &pos, out);
}

bool parse_json_int64_array_after_key(const std::string& text, const std::string& key, std::vector<int64_t>* out) {
    if (out == nullptr) return false;
    out->clear();
    const size_t key_pos = text.find("\"" + key + "\"");
    if (key_pos == std::string::npos) return false;
    size_t pos = text.find(':', key_pos);
    if (pos == std::string::npos) return false;
    ++pos;
    skip_ws(text, &pos);
    if (pos >= text.size() || text[pos] != '[') return false;
    ++pos;
    while (pos < text.size()) {
        skip_ws(text, &pos);
        if (pos < text.size() && text[pos] == ']') {
            ++pos;
            return true;
        }
        const size_t begin = pos;
        if (pos < text.size() && (text[pos] == '-' || text[pos] == '+')) ++pos;
        while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9') ++pos;
        if (begin == pos || (begin + 1 == pos && (text[begin] == '-' || text[begin] == '+'))) return false;
        out->push_back(static_cast<int64_t>(std::strtoll(text.substr(begin, pos - begin).c_str(), nullptr, 10)));
        skip_ws(text, &pos);
        if (pos < text.size() && text[pos] == ',') {
            ++pos;
            continue;
        }
        if (pos < text.size() && text[pos] == ']') {
            ++pos;
            return true;
        }
        return false;
    }
    return false;
}

bool parse_json_u64_pair_after_key(const std::string& text, const std::string& key, uint64_t out[2]) {
    std::vector<int64_t> values;
    if (!parse_json_int64_array_after_key(text, key, &values) || values.size() != 2) return false;
    if (values[0] < 0 || values[1] < 0 || values[1] < values[0]) return false;
    out[0] = static_cast<uint64_t>(values[0]);
    out[1] = static_cast<uint64_t>(values[1]);
    return true;
}

uint64_t read_le_u64(const unsigned char bytes[8]);

bool read_safetensors_header_body(const std::string& file_path, uint64_t* header_bytes, std::string* header, std::string* error) {
    if (header_bytes == nullptr || header == nullptr) return false;
    *header_bytes = 0;
    header->clear();

    std::ifstream in(file_path, std::ios::binary);
    if (!in) {
        if (error) *error = "missing or unreadable safetensors shard: " + file_path;
        return false;
    }
    unsigned char len_bytes[8] = {};
    in.read(reinterpret_cast<char*>(len_bytes), sizeof(len_bytes));
    if (in.gcount() != static_cast<std::streamsize>(sizeof(len_bytes))) {
        if (error) *error = "short safetensors header length: " + file_path;
        return false;
    }
    *header_bytes = read_le_u64(len_bytes);
    if (*header_bytes == 0 || *header_bytes > (512ull << 20)) {
        if (error) *error = "invalid safetensors header length: " + file_path;
        return false;
    }

    header->assign(static_cast<size_t>(*header_bytes), '\0');
    in.read(&(*header)[0], static_cast<std::streamsize>(header->size()));
    if (in.gcount() != static_cast<std::streamsize>(header->size())) {
        if (error) *error = "short safetensors header body: " + file_path;
        return false;
    }
    return true;
}

bool extract_json_object_for_quoted_key(const std::string& text, const std::string& key, std::string* object) {
    if (object == nullptr) return false;
    const std::string quoted = "\"" + key + "\"";
    const size_t key_pos = text.find(quoted);
    if (key_pos == std::string::npos) return false;
    size_t pos = key_pos + quoted.size();
    skip_ws(text, &pos);
    if (pos >= text.size() || text[pos] != ':') return false;
    ++pos;
    skip_ws(text, &pos);
    if (pos >= text.size() || text[pos] != '{') return false;
    const size_t close = find_matching_brace(text, pos);
    if (close == std::string::npos || close <= pos) return false;
    *object = text.substr(pos, close - pos + 1);
    return true;
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

float unbiased_stddev(const std::vector<float>& values) {
    if (values.size() < 2) return 0.0f;
    double mean = 0.0;
    for (float v : values) mean += v;
    mean /= static_cast<double>(values.size());
    double accum = 0.0;
    for (float v : values) {
        const double d = static_cast<double>(v) - mean;
        accum += d * d;
    }
    return static_cast<float>(std::sqrt(accum / static_cast<double>(values.size() - 1)));
}

uint32_t crc32_update(uint32_t crc, const unsigned char* data, size_t size) {
    crc = ~crc;
    for (size_t i = 0; i < size; ++i) {
        crc ^= data[i];
        for (int bit = 0; bit < 8; ++bit) {
            crc = (crc >> 1) ^ (0xedb88320u & (0u - (crc & 1u)));
        }
    }
    return ~crc;
}

uint32_t adler32_bytes(const std::vector<unsigned char>& data) {
    uint32_t a = 1;
    uint32_t b = 0;
    for (unsigned char byte : data) {
        a = (a + byte) % 65521u;
        b = (b + a) % 65521u;
    }
    return (b << 16) | a;
}

void append_be32(std::vector<unsigned char>* out, uint32_t value) {
    out->push_back(static_cast<unsigned char>((value >> 24) & 0xffu));
    out->push_back(static_cast<unsigned char>((value >> 16) & 0xffu));
    out->push_back(static_cast<unsigned char>((value >> 8) & 0xffu));
    out->push_back(static_cast<unsigned char>(value & 0xffu));
}

void append_png_chunk(std::vector<unsigned char>* png, const char type[4], const std::vector<unsigned char>& payload) {
    append_be32(png, static_cast<uint32_t>(payload.size()));
    const size_t type_begin = png->size();
    png->insert(png->end(), type, type + 4);
    png->insert(png->end(), payload.begin(), payload.end());
    const uint32_t crc = crc32_update(0, png->data() + type_begin, png->size() - type_begin);
    append_be32(png, crc);
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

HiDreamO1ForwardPlan hidream_o1_build_t2i_forward_plan(const HiDreamO1RuntimeConfig& cfg,
                                                       int width,
                                                       int height,
                                                       int64_t text_tokens) {
    HiDreamO1ForwardPlan plan;
    plan.width = width;
    plan.height = height;
    plan.patch_size = cfg.patch_size;
    plan.patch_dim = cfg.patch_size * cfg.patch_size * 3;
    if (width <= 0 || height <= 0 || cfg.patch_size <= 0 || width % cfg.patch_size != 0 || height % cfg.patch_size != 0 ||
        text_tokens < cfg.timestep_token_num) {
        return plan;
    }
    plan.h_patches = height / cfg.patch_size;
    plan.w_patches = width / cfg.patch_size;
    plan.text_tokens = text_tokens;
    plan.timestep_token_begin = text_tokens - cfg.timestep_token_num;
    plan.image_token_begin = text_tokens;
    plan.image_tokens = static_cast<int64_t>(plan.h_patches) * static_cast<int64_t>(plan.w_patches);
    plan.total_sequence_tokens = plan.image_token_begin + plan.image_tokens;
    plan.raw_token_types.assign(static_cast<size_t>(plan.total_sequence_tokens), 0);
    plan.token_types_bin.assign(static_cast<size_t>(plan.total_sequence_tokens), 0);
    plan.vinput_mask.assign(static_cast<size_t>(plan.total_sequence_tokens), 0);
    plan.mrope_position_ids_t.assign(static_cast<size_t>(plan.total_sequence_tokens), 0);
    plan.mrope_position_ids_h.assign(static_cast<size_t>(plan.total_sequence_tokens), 0);
    plan.mrope_position_ids_w.assign(static_cast<size_t>(plan.total_sequence_tokens), 0);

    for (int64_t i = plan.timestep_token_begin; i < plan.image_token_begin; ++i) {
        plan.raw_token_types[static_cast<size_t>(i)] = 3;
        plan.token_types_bin[static_cast<size_t>(i)] = 1;
    }
    for (int64_t i = plan.image_token_begin; i < plan.total_sequence_tokens; ++i) {
        plan.raw_token_types[static_cast<size_t>(i)] = 1;
        plan.token_types_bin[static_cast<size_t>(i)] = 1;
        plan.vinput_mask[static_cast<size_t>(i)] = 1;
    }
    for (int64_t i = 0; i < plan.image_token_begin; ++i) {
        plan.mrope_position_ids_t[static_cast<size_t>(i)] = i;
        plan.mrope_position_ids_h[static_cast<size_t>(i)] = i;
        plan.mrope_position_ids_w[static_cast<size_t>(i)] = i;
    }
    int64_t max_pos = plan.image_token_begin > 0 ? plan.image_token_begin - 1 : 0;
    constexpr int64_t fix_point = 4096;
    for (int64_t tok = 0; tok < plan.image_tokens; ++tok) {
        const int64_t h = tok / plan.w_patches;
        const int64_t w = tok % plan.w_patches;
        const size_t idx = static_cast<size_t>(plan.image_token_begin + tok);
        plan.mrope_position_ids_t[idx] = fix_point;
        plan.mrope_position_ids_h[idx] = fix_point + h;
        plan.mrope_position_ids_w[idx] = fix_point + w;
        max_pos = std::max(max_pos, std::max(plan.mrope_position_ids_t[idx],
                                             std::max(plan.mrope_position_ids_h[idx], plan.mrope_position_ids_w[idx])));
    }
    plan.mrope_position_delta = max_pos + 1 - plan.total_sequence_tokens;
    return plan;
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

std::vector<float> hidream_o1_noise_scale_schedule(const HiDreamO1RuntimeConfig& cfg, int steps) {
    if (steps <= 0) return {};
    std::vector<float> schedule(static_cast<size_t>(steps), cfg.default_noise_scale_start);
    if (steps == 1) return schedule;
    for (int i = 0; i < steps; ++i) {
        const float r = static_cast<float>(i) / static_cast<float>(steps - 1);
        schedule[static_cast<size_t>(i)] = cfg.default_noise_scale_start +
                                           (cfg.default_noise_scale_end - cfg.default_noise_scale_start) * r;
    }
    return schedule;
}

float hidream_o1_t_pixeldit(int timestep) {
    return 1.0f - static_cast<float>(timestep) / 1000.0f;
}

std::vector<float> hidream_o1_x0_to_model_output(const std::vector<float>& x0_pred,
                                                 const std::vector<float>& z,
                                                 float sigma) {
    if (x0_pred.size() != z.size() || sigma == 0.0f) return {};
    std::vector<float> out(z.size());
    for (size_t i = 0; i < z.size(); ++i) {
        const float v = (x0_pred[i] - z[i]) / sigma;
        out[i] = -v;
    }
    return out;
}

std::vector<float> hidream_o1_flash_step(const std::vector<float>& sample,
                                         const std::vector<float>& model_output,
                                         const std::vector<float>& noise,
                                         float sigma,
                                         float sigma_next,
                                         float s_noise,
                                         float noise_clip_std) {
    if (sample.size() != model_output.size() || sample.size() != noise.size()) return {};
    std::vector<float> clipped_noise = noise;
    if (noise_clip_std > 0.0f) {
        const float stddev = unbiased_stddev(clipped_noise);
        const float clip_val = noise_clip_std * stddev;
        if (clip_val > 0.0f) {
            for (float& v : clipped_noise) {
                v = std::max(-clip_val, std::min(clip_val, v));
            }
        }
    }

    std::vector<float> prev(sample.size());
    for (size_t i = 0; i < sample.size(); ++i) {
        const float denoised = sample[i] - model_output[i] * sigma;
        prev[i] = sigma_next * clipped_noise[i] * s_noise + (1.0f - sigma_next) * denoised;
    }
    return prev;
}

bool hidream_o1_run_forward_loop(const std::vector<float>& initial_z,
                                 const std::vector<std::vector<float>>& noise_by_step,
                                 const HiDreamO1X0Predictor& predict_x0,
                                 std::vector<float>* final_z,
                                 std::vector<HiDreamO1ForwardTraceStep>* trace,
                                 std::string* error) {
    if (initial_z.empty()) {
        if (error) *error = "initial_z is empty";
        return false;
    }
    if (!predict_x0) {
        if (error) *error = "missing x0 predictor";
        return false;
    }

    const HiDreamO1RuntimeConfig cfg = default_hidream_o1_runtime_config();
    const std::vector<int> timesteps = hidream_o1_dev_timesteps();
    const std::vector<float> sigmas = hidream_o1_dev_sigmas();
    const std::vector<float> noise_schedule = hidream_o1_noise_scale_schedule(cfg, static_cast<int>(timesteps.size()));
    if (sigmas.size() != timesteps.size() + 1 || noise_schedule.size() != timesteps.size()) {
        if (error) *error = "scheduler shape mismatch";
        return false;
    }

    std::vector<float> current = initial_z;
    if (trace) trace->clear();
    for (size_t i = 0; i < timesteps.size(); ++i) {
        HiDreamO1ForwardTraceStep step;
        step.step_index = static_cast<int>(i);
        step.timestep = timesteps[i];
        step.sigma = sigmas[i];
        step.sigma_next = sigmas[i + 1];
        step.t_pixeldit = hidream_o1_t_pixeldit(step.timestep);
        step.noise_scale = noise_schedule[i];

        std::vector<float> x0_pred;
        std::string predictor_error;
        if (!predict_x0(step, current, &x0_pred, &predictor_error)) {
            if (error) *error = predictor_error.empty() ? "x0 predictor failed" : predictor_error;
            return false;
        }
        const std::vector<float> model_output = hidream_o1_x0_to_model_output(x0_pred, current, step.sigma);
        if (model_output.empty()) {
            if (error) *error = "x0 predictor returned wrong shape";
            return false;
        }

        std::vector<float> noise(current.size(), 0.0f);
        if (i < noise_by_step.size() && !noise_by_step[i].empty()) {
            if (noise_by_step[i].size() != current.size()) {
                if (error) *error = "noise vector shape mismatch";
                return false;
            }
            noise = noise_by_step[i];
        }

        current = hidream_o1_flash_step(current,
                                        model_output,
                                        noise,
                                        step.sigma,
                                        step.sigma_next,
                                        step.noise_scale,
                                        cfg.default_noise_clip_std);
        if (current.empty()) {
            if (error) *error = "flash scheduler step failed";
            return false;
        }
        if (trace) trace->push_back(step);
    }

    if (final_z) *final_z = std::move(current);
    return true;
}

std::vector<unsigned char> hidream_o1_unpatch_to_rgb8(const std::vector<float>& patch_tokens,
                                                      int width,
                                                      int height,
                                                      const HiDreamO1RuntimeConfig& cfg) {
    const HiDreamO1Shape shape = hidream_o1_shape_for_size(cfg, width, height);
    if (shape.patch_tokens <= 0 || shape.patch_dim <= 0) return {};
    const int h_patches = height / cfg.patch_size;
    const int w_patches = width / cfg.patch_size;
    const size_t expected = static_cast<size_t>(shape.patch_tokens) * static_cast<size_t>(shape.patch_dim);
    if (patch_tokens.size() != expected) return {};

    std::vector<unsigned char> rgb(static_cast<size_t>(width) * static_cast<size_t>(height) * 3, 0);
    for (int py_patch = 0; py_patch < h_patches; ++py_patch) {
        for (int px_patch = 0; px_patch < w_patches; ++px_patch) {
            const int64_t token = static_cast<int64_t>(py_patch) * w_patches + px_patch;
            const size_t token_base = static_cast<size_t>(token) * static_cast<size_t>(shape.patch_dim);
            for (int c = 0; c < 3; ++c) {
                const size_t channel_base = token_base + static_cast<size_t>(c) * cfg.patch_size * cfg.patch_size;
                for (int y = 0; y < cfg.patch_size; ++y) {
                    for (int x = 0; x < cfg.patch_size; ++x) {
                        const int out_y = py_patch * cfg.patch_size + y;
                        const int out_x = px_patch * cfg.patch_size + x;
                        const float normalized = (patch_tokens[channel_base + static_cast<size_t>(y) * cfg.patch_size + x] + 1.0f) * 0.5f;
                        const float clamped = std::max(0.0f, std::min(1.0f, normalized));
                        const int quantized = static_cast<int>(std::lround(clamped * 255.0f));
                        rgb[(static_cast<size_t>(out_y) * width + out_x) * 3 + c] = static_cast<unsigned char>(std::max(0, std::min(255, quantized)));
                    }
                }
            }
        }
    }
    return rgb;
}

bool hidream_o1_write_png_rgb8(const std::string& path,
                               const std::vector<unsigned char>& rgb,
                               int width,
                               int height,
                               std::string* error) {
    if (width <= 0 || height <= 0) {
        if (error) *error = "PNG width/height must be positive";
        return false;
    }
    const size_t expected = static_cast<size_t>(width) * static_cast<size_t>(height) * 3;
    if (rgb.size() != expected) {
        if (error) *error = "PNG RGB buffer size mismatch";
        return false;
    }

    std::vector<unsigned char> raw;
    raw.reserve(static_cast<size_t>(height) * (1 + static_cast<size_t>(width) * 3));
    for (int y = 0; y < height; ++y) {
        raw.push_back(0);  // PNG filter type 0.
        const size_t row = static_cast<size_t>(y) * static_cast<size_t>(width) * 3;
        raw.insert(raw.end(), rgb.begin() + row, rgb.begin() + row + static_cast<size_t>(width) * 3);
    }

    std::vector<unsigned char> zlib;
    zlib.reserve(raw.size() + raw.size() / 65535 + 16);
    zlib.push_back(0x78);
    zlib.push_back(0x01);
    size_t pos = 0;
    while (pos < raw.size()) {
        const size_t remaining = raw.size() - pos;
        const uint16_t block = static_cast<uint16_t>(std::min<size_t>(remaining, 65535));
        const bool last = pos + block == raw.size();
        zlib.push_back(last ? 0x01 : 0x00);
        zlib.push_back(static_cast<unsigned char>(block & 0xffu));
        zlib.push_back(static_cast<unsigned char>((block >> 8) & 0xffu));
        const uint16_t nlen = static_cast<uint16_t>(~block);
        zlib.push_back(static_cast<unsigned char>(nlen & 0xffu));
        zlib.push_back(static_cast<unsigned char>((nlen >> 8) & 0xffu));
        zlib.insert(zlib.end(), raw.begin() + static_cast<std::ptrdiff_t>(pos), raw.begin() + static_cast<std::ptrdiff_t>(pos + block));
        pos += block;
    }
    append_be32(&zlib, adler32_bytes(raw));

    std::vector<unsigned char> png = {0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n'};
    std::vector<unsigned char> ihdr;
    append_be32(&ihdr, static_cast<uint32_t>(width));
    append_be32(&ihdr, static_cast<uint32_t>(height));
    ihdr.push_back(8);  // bit depth
    ihdr.push_back(2);  // RGB
    ihdr.push_back(0);  // compression
    ihdr.push_back(0);  // filter
    ihdr.push_back(0);  // interlace
    append_png_chunk(&png, "IHDR", ihdr);
    append_png_chunk(&png, "IDAT", zlib);
    append_png_chunk(&png, "IEND", {});

    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        if (error) *error = "failed to open PNG output: " + path;
        return false;
    }
    out.write(reinterpret_cast<const char*>(png.data()), static_cast<std::streamsize>(png.size()));
    if (!out) {
        if (error) *error = "failed to write PNG output: " + path;
        return false;
    }
    return true;
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

std::string hidream_o1_default_source_dir() {
    return env_or("UTOPIC_HIDREAM_SOURCE_DIR",
                  home_path("/.cache/utopic/src/HiDream-O1-Image"));
}

std::string hidream_o1_default_torch_python() {
    return env_or("UTOPIC_HIDREAM_TORCH_PYTHON", "python3");
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
    const HiDreamO1RuntimeConfig cfg = default_hidream_o1_runtime_config();
    const int width = req.width > 0 ? req.width : cfg.default_width;
    const int height = req.height > 0 ? req.height : cfg.default_height;
    const std::string python = req.torch_python.empty() ? hidream_o1_default_torch_python() : req.torch_python;
    const std::string source_dir = req.source_dir.empty() ? hidream_o1_default_source_dir() : req.source_dir;
    const std::string model_dir = req.model_dir.empty() ? hidream_o1_default_model_dir() : req.model_dir;

    std::ostringstream cmd;
    cmd << "cd " << shell_quote(source_dir)
        << " && PYTHONPATH=" << shell_quote(source_dir)
        << " UTOPIC_HIDREAM_USE_FLASH_ATTN="
        << shell_quote(env_or("UTOPIC_HIDREAM_USE_FLASH_ATTN", "0"))
        << " " << shell_quote(python)
        << " " << shell_quote(join_path(source_dir, "inference.py"))
        << " --model_path " << shell_quote(model_dir)
        << " --prompt " << shell_quote(req.prompt)
        << " --output_image " << shell_quote(req.output_path)
        << " --height " << height
        << " --width " << width
        << " --model_type dev"
        << " --seed " << req.seed
        << " --shift " << cfg.default_shift
        << " --guidance_scale " << cfg.default_guidance_scale
        << " --noise_scale_start " << cfg.default_noise_scale_start
        << " --noise_scale_end " << cfg.default_noise_scale_end
        << " --noise_clip_std " << cfg.default_noise_clip_std;
    if (!req.extra_args.empty()) {
        cmd << " " << req.extra_args;
    }
    return cmd.str();
}

bool hidream_o1_file_exists(const std::string& path) {
    struct stat st {};
    return !path.empty() && stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

bool hidream_o1_dir_exists(const std::string& path) {
    struct stat st {};
    return !path.empty() && stat(path.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
}

bool hidream_o1_patch_official_source_for_flash_env(const std::string& source_dir, std::string* error) {
    const std::string pipeline_path = join_path(source_dir, "models/pipeline.py");
    std::string text;
    if (!read_text_file(pipeline_path, &text)) {
        if (error) *error = "missing or unreadable official HiDream pipeline: " + pipeline_path;
        return false;
    }
    if (text.find("UTOPIC_HIDREAM_USE_FLASH_ATTN") != std::string::npos) {
        return true;
    }
    const std::string import_needle = "import torch\n";
    const std::string flash_needle = "\"use_flash_attn\": True";
    const std::string flash_replacement =
        "\"use_flash_attn\": os.environ.get(\"UTOPIC_HIDREAM_USE_FLASH_ATTN\", \"1\").lower() not in (\"0\", \"false\", \"no\")";

    size_t import_pos = text.find(import_needle);
    size_t flash_pos = text.find(flash_needle);
    if (import_pos == std::string::npos || flash_pos == std::string::npos) {
        if (error) *error = "official HiDream pipeline does not match expected flash-attn hook shape";
        return false;
    }
    text.replace(import_pos, import_needle.size(), "import os\n" + import_needle);
    flash_pos = text.find(flash_needle);
    text.replace(flash_pos, flash_needle.size(), flash_replacement);

    std::ofstream out(pipeline_path, std::ios::binary | std::ios::trunc);
    if (!out) {
        if (error) *error = "failed to open official HiDream pipeline for patching: " + pipeline_path;
        return false;
    }
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!out) {
        if (error) *error = "failed to write official HiDream pipeline patch: " + pipeline_path;
        return false;
    }
    return true;
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

bool load_hidream_o1_native_model_layout(const std::string& model_dir, HiDreamO1NativeModelLayout* layout) {
    if (layout == nullptr) return false;
    *layout = HiDreamO1NativeModelLayout{};
    layout->model_dir = model_dir;

    std::string config_text;
    const std::string config_path = join_path(model_dir, "config.json");
    if (!read_text_file(config_path, &config_text)) {
        layout->error = "missing or unreadable config: " + config_path;
        return false;
    }

    std::string text_config;
    std::string vision_config;
    if (!extract_json_object(config_text, "text_config", &text_config) ||
        !extract_json_object(config_text, "vision_config", &vision_config)) {
        layout->error = "config is missing text_config or vision_config";
        return false;
    }

    parse_json_int(config_text, "image_token_id", &layout->image_token_id);
    parse_json_int(config_text, "vision_start_token_id", &layout->vision_start_token_id);
    parse_json_int(text_config, "hidden_size", &layout->text.hidden_size);
    parse_json_int(text_config, "intermediate_size", &layout->text.intermediate_size);
    parse_json_int(text_config, "num_hidden_layers", &layout->text.num_hidden_layers);
    parse_json_int(text_config, "num_attention_heads", &layout->text.num_attention_heads);
    parse_json_int(text_config, "num_key_value_heads", &layout->text.num_key_value_heads);
    parse_json_int(text_config, "head_dim", &layout->text.head_dim);
    parse_json_int(text_config, "vocab_size", &layout->text.vocab_size);
    parse_json_number(text_config, "rope_theta", &layout->text.rope_theta);
    parse_json_number(text_config, "rms_norm_eps", &layout->text.rms_norm_eps);

    parse_json_int(vision_config, "hidden_size", &layout->vision.hidden_size);
    parse_json_int(vision_config, "intermediate_size", &layout->vision.intermediate_size);
    parse_json_int(vision_config, "depth", &layout->vision.depth);
    parse_json_int(vision_config, "num_heads", &layout->vision.num_heads);
    parse_json_int(vision_config, "patch_size", &layout->vision.patch_size);
    parse_json_int(vision_config, "temporal_patch_size", &layout->vision.temporal_patch_size);
    parse_json_int(vision_config, "spatial_merge_size", &layout->vision.spatial_merge_size);
    parse_json_int(vision_config, "out_hidden_size", &layout->vision.out_hidden_size);

    HiDreamO1ShardManifest manifest;
    if (!load_hidream_o1_shard_manifest(model_dir, &manifest)) {
        layout->error = manifest.error;
        return false;
    }
    layout->tensor_count = static_cast<int64_t>(manifest.entries.size());

    std::set<std::string> names;
    for (const HiDreamO1ShardEntry& entry : manifest.entries) {
        names.insert(entry.tensor_name);
        if (entry.tensor_name.find("model.language_model.") == 0) layout->text_tensor_count++;
        if (entry.tensor_name.find("model.visual.") == 0) layout->vision_tensor_count++;
        if (entry.tensor_name.find("model.t_embedder") == 0) layout->timestep_tensor_count++;
        if (entry.tensor_name.find("model.final_layer") == 0) layout->final_layer_tensor_count++;
        if (entry.tensor_name == "lm_head.weight") layout->lm_head_tensor_count++;
    }

    const std::vector<std::string> required_block0 = hidream_o1_text_block_tensor_names(0);
    layout->has_required_text_block0 = true;
    for (const std::string& name : required_block0) {
        if (names.find(name) == names.end()) {
            layout->has_required_text_block0 = false;
            break;
        }
    }

    if (layout->text.num_hidden_layers <= 0 || layout->vision.depth <= 0 || layout->tensor_count <= 0) {
        layout->error = "native layout has invalid dimensions or empty tensor map";
        return false;
    }
    return true;
}

HiDreamO1SafetensorsHeader inspect_hidream_o1_safetensors_header(const std::string& file_path) {
    HiDreamO1SafetensorsHeader result;
    result.file_path = file_path;

    std::string header;
    if (!read_safetensors_header_body(file_path, &result.header_bytes, &header, &result.error)) {
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

bool load_hidream_o1_tensor_catalog(const std::string& model_dir, HiDreamO1TensorCatalog* catalog) {
    if (catalog == nullptr) return false;
    *catalog = HiDreamO1TensorCatalog{};
    catalog->model_dir = model_dir;

    HiDreamO1ShardManifest manifest;
    if (!load_hidream_o1_shard_manifest(model_dir, &manifest)) {
        catalog->error = manifest.error;
        return false;
    }

    std::map<std::string, std::string> header_by_shard;
    std::map<std::string, uint64_t> header_bytes_by_shard;
    for (const std::string& shard : manifest.shard_files) {
        const std::string path = join_path(model_dir, shard);
        std::string header;
        uint64_t header_bytes = 0;
        std::string error;
        if (!read_safetensors_header_body(path, &header_bytes, &header, &error)) {
            catalog->error = error;
            return false;
        }
        header_by_shard[shard] = std::move(header);
        header_bytes_by_shard[shard] = header_bytes;
    }

    catalog->tensors.reserve(manifest.entries.size());
    for (const HiDreamO1ShardEntry& entry : manifest.entries) {
        const auto header_it = header_by_shard.find(entry.shard_file);
        const auto header_bytes_it = header_bytes_by_shard.find(entry.shard_file);
        if (header_it == header_by_shard.end() || header_bytes_it == header_bytes_by_shard.end()) {
            catalog->missing_tensor_count++;
            continue;
        }

        std::string tensor_object;
        if (!extract_json_object_for_quoted_key(header_it->second, entry.tensor_name, &tensor_object)) {
            catalog->missing_tensor_count++;
            continue;
        }

        HiDreamO1TensorInfo info;
        info.tensor_name = entry.tensor_name;
        info.shard_file = entry.shard_file;
        info.file_path = join_path(model_dir, entry.shard_file);
        info.header_bytes = header_bytes_it->second;
        if (!parse_json_string_after_key(tensor_object, "dtype", &info.dtype) ||
            !parse_json_int64_array_after_key(tensor_object, "shape", &info.shape) ||
            !parse_json_u64_pair_after_key(tensor_object, "data_offsets", info.data_offsets)) {
            catalog->error = "failed to parse safetensors metadata for tensor: " + entry.tensor_name;
            return false;
        }
        const uint64_t data_base = 8 + info.header_bytes;
        info.absolute_data_begin = data_base + info.data_offsets[0];
        info.absolute_data_end = data_base + info.data_offsets[1];
        catalog->tensors.push_back(std::move(info));
    }

    if (catalog->tensors.empty()) {
        catalog->error = "no tensor metadata resolved from safetensors headers";
        return false;
    }
    return true;
}

bool find_hidream_o1_tensor(const HiDreamO1TensorCatalog& catalog,
                            const std::string& tensor_name,
                            HiDreamO1TensorInfo* tensor) {
    for (const HiDreamO1TensorInfo& info : catalog.tensors) {
        if (info.tensor_name == tensor_name) {
            if (tensor) *tensor = info;
            return true;
        }
    }
    return false;
}

bool read_hidream_o1_tensor_bytes(const HiDreamO1TensorInfo& tensor,
                                  std::vector<unsigned char>* bytes,
                                  std::string* error) {
    if (bytes == nullptr) return false;
    bytes->clear();
    if (tensor.file_path.empty()) {
        if (error) *error = "tensor has no file path: " + tensor.tensor_name;
        return false;
    }
    if (tensor.absolute_data_end < tensor.absolute_data_begin) {
        if (error) *error = "tensor has invalid byte range: " + tensor.tensor_name;
        return false;
    }

    const uint64_t n_bytes_u64 = tensor.absolute_data_end - tensor.absolute_data_begin;
    if (n_bytes_u64 > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
        if (error) *error = "tensor byte range exceeds addressable size: " + tensor.tensor_name;
        return false;
    }
    const size_t n_bytes = static_cast<size_t>(n_bytes_u64);

    std::ifstream in(tensor.file_path, std::ios::binary);
    if (!in) {
        if (error) *error = "failed to open tensor shard: " + tensor.file_path;
        return false;
    }
    in.seekg(static_cast<std::streamoff>(tensor.absolute_data_begin), std::ios::beg);
    if (!in) {
        if (error) *error = "failed to seek tensor payload: " + tensor.tensor_name;
        return false;
    }
    bytes->assign(n_bytes, 0);
    if (n_bytes == 0) return true;
    in.read(reinterpret_cast<char*>(bytes->data()), static_cast<std::streamsize>(bytes->size()));
    if (in.gcount() != static_cast<std::streamsize>(bytes->size())) {
        bytes->clear();
        if (error) *error = "short tensor payload read: " + tensor.tensor_name;
        return false;
    }
    return true;
}

std::vector<std::string> hidream_o1_text_block_tensor_names(int layer) {
    const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".";
    return {
        prefix + "input_layernorm.weight",
        prefix + "self_attn.q_proj.weight",
        prefix + "self_attn.k_proj.weight",
        prefix + "self_attn.v_proj.weight",
        prefix + "self_attn.o_proj.weight",
        prefix + "self_attn.q_norm.weight",
        prefix + "self_attn.k_norm.weight",
        prefix + "post_attention_layernorm.weight",
        prefix + "mlp.gate_proj.weight",
        prefix + "mlp.up_proj.weight",
        prefix + "mlp.down_proj.weight",
    };
}

bool load_hidream_o1_text_block_tensors(const std::string& model_dir,
                                        int layer,
                                        std::vector<HiDreamO1TensorInfo>* tensors) {
    if (tensors == nullptr) return false;
    tensors->clear();

    HiDreamO1TensorCatalog catalog;
    if (!load_hidream_o1_tensor_catalog(model_dir, &catalog)) return false;
    const std::vector<std::string> names = hidream_o1_text_block_tensor_names(layer);
    tensors->reserve(names.size());
    for (const std::string& name : names) {
        HiDreamO1TensorInfo info;
        if (!find_hidream_o1_tensor(catalog, name, &info)) {
            tensors->clear();
            return false;
        }
        tensors->push_back(std::move(info));
    }
    return true;
}

bool hidream_o1_prepare_native_execution(const std::string& model_dir,
                                         int width,
                                         int height,
                                         int64_t text_tokens,
                                         bool load_block0_payloads,
                                         HiDreamO1NativeExecutionSummary* summary,
                                         std::string* error) {
    if (summary == nullptr) return false;
    *summary = HiDreamO1NativeExecutionSummary{};
    summary->model_dir = model_dir;
    summary->width = width;
    summary->height = height;
    summary->text_tokens = text_tokens;

    const HiDreamO1RuntimeConfig cfg = default_hidream_o1_runtime_config();
    const HiDreamO1ForwardPlan plan = hidream_o1_build_t2i_forward_plan(cfg, width, height, text_tokens);
    if (plan.total_sequence_tokens <= 0 || plan.image_tokens <= 0) {
        summary->error = "invalid native HiDream execution shape or text token count";
        if (error) *error = summary->error;
        return false;
    }
    summary->image_tokens = plan.image_tokens;
    summary->total_sequence_tokens = plan.total_sequence_tokens;

    HiDreamO1NativeModelLayout layout;
    if (!load_hidream_o1_native_model_layout(model_dir, &layout)) {
        summary->error = layout.error;
        if (error) *error = summary->error;
        return false;
    }
    if (!layout.has_required_text_block0) {
        summary->error = "native HiDream execution is missing required text block0 tensors";
        if (error) *error = summary->error;
        return false;
    }
    summary->text_layers = layout.text.num_hidden_layers;
    summary->text_hidden = layout.text.hidden_size;
    summary->text_heads = layout.text.num_attention_heads;
    summary->text_kv_heads = layout.text.num_key_value_heads;
    summary->text_head_dim = layout.text.head_dim;
    summary->text_intermediate = layout.text.intermediate_size;
    summary->tensor_count = layout.tensor_count;

    HiDreamO1TensorCatalog catalog;
    if (!load_hidream_o1_tensor_catalog(model_dir, &catalog)) {
        summary->error = catalog.error;
        if (error) *error = summary->error;
        return false;
    }
    summary->catalog_tensor_count = static_cast<int64_t>(catalog.tensors.size());
    summary->catalog_missing_tensor_count = catalog.missing_tensor_count;
    if (catalog.missing_tensor_count != 0) {
        summary->error = "native HiDream execution has unresolved tensor metadata";
        if (error) *error = summary->error;
        return false;
    }

    const std::vector<std::string> block0_names = hidream_o1_text_block_tensor_names(0);
    summary->block0_tensor_count = static_cast<int64_t>(block0_names.size());
    for (const std::string& name : block0_names) {
        HiDreamO1TensorInfo tensor;
        if (!find_hidream_o1_tensor(catalog, name, &tensor)) {
            summary->error = "native HiDream execution failed to resolve tensor: " + name;
            if (error) *error = summary->error;
            return false;
        }
        if (!load_block0_payloads) continue;

        std::vector<unsigned char> bytes;
        std::string read_error;
        if (!read_hidream_o1_tensor_bytes(tensor, &bytes, &read_error)) {
            summary->error = read_error.empty() ? "native HiDream execution failed to read tensor: " + name : read_error;
            if (error) *error = summary->error;
            return false;
        }
        summary->block0_payload_bytes += static_cast<uint64_t>(bytes.size());
    }
    summary->block0_payloads_loaded = load_block0_payloads;
    return true;
}

}  // namespace utopic
