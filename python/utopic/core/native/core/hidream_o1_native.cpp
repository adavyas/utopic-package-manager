#include "hidream_o1_native.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
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
        text_tokens < 0) {
        return plan;
    }
    plan.h_patches = height / cfg.patch_size;
    plan.w_patches = width / cfg.patch_size;
    plan.text_tokens = text_tokens;
    plan.timestep_token_begin = text_tokens;
    plan.image_token_begin = plan.timestep_token_begin + cfg.timestep_token_num;
    plan.image_tokens = static_cast<int64_t>(plan.h_patches) * static_cast<int64_t>(plan.w_patches);
    plan.total_sequence_tokens = plan.image_token_begin + plan.image_tokens;
    plan.raw_token_types.assign(static_cast<size_t>(plan.total_sequence_tokens), 0);
    plan.token_types_bin.assign(static_cast<size_t>(plan.total_sequence_tokens), 0);
    plan.vinput_mask.assign(static_cast<size_t>(plan.total_sequence_tokens), 0);

    for (int64_t i = plan.timestep_token_begin; i < plan.image_token_begin; ++i) {
        plan.raw_token_types[static_cast<size_t>(i)] = 3;
        plan.token_types_bin[static_cast<size_t>(i)] = 1;
    }
    for (int64_t i = plan.image_token_begin; i < plan.total_sequence_tokens; ++i) {
        plan.raw_token_types[static_cast<size_t>(i)] = 1;
        plan.token_types_bin[static_cast<size_t>(i)] = 1;
        plan.vinput_mask[static_cast<size_t>(i)] = 1;
    }
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
