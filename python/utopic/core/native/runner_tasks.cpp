#include "runner_tasks.h"

#include "audio_engine.h"
#include "image_engine.h"
#include "llama.h"
#include "runner_plugin.h"
#include "video_engine.h"

#include <cctype>
#include <cstdlib>
#include <cstdio>
#include <filesystem>
#include <vector>

namespace utopic_runner {

namespace fs = std::filesystem;
using utopic::image_engine_generate;
using utopic::image_engine_params;
using utopic::image_engine_result;
using utopic::audio_engine_result;
using utopic::audio_engine_wav_params;
using utopic::audio_engine_write_wav;
using utopic::video_engine_frames_params;
using utopic::video_engine_result;
using utopic::video_engine_write_frames;

static const char * runner_task_json_name(utopic::runner_task task) {
    switch (task) {
        case utopic::RUNNER_TASK_CHAT:  return "chat";
        case utopic::RUNNER_TASK_IMAGE: return "image";
        case utopic::RUNNER_TASK_TTS:   return "tts";
        case utopic::RUNNER_TASK_MUSIC: return "music";
        case utopic::RUNNER_TASK_VIDEO: return "video";
        case utopic::RUNNER_TASK_MISC:  return "misc";
        default:                        return "unknown";
    }
}

static bool json_dopt(const json & obj, const char * key, double & out) {
    if (!obj.contains(key) || !obj[key].is_number()) {
        return false;
    }
    out = obj[key].get<double>();
    return out > 0.0;
}

static const char * env_any(const char * preferred, const char * legacy = nullptr) {
    const char * v = preferred ? getenv(preferred) : nullptr;
    if (v) {
        return v;
    }
    return legacy ? getenv(legacy) : nullptr;
}

string host_backend() {
    const char * configured = env_any("UTOPIC_RUNTIME_BACKEND");
    if (configured && configured[0]) {
        return configured;
    }
#if defined(UTOPIC_BACKEND_NAME)
    return UTOPIC_BACKEND_NAME;
#elif defined(GGML_USE_CUDA)
    return "cuda";
#elif defined(__APPLE__)
    return "metal";
#else
    return "cpu";
#endif
}

static ggml_backend_dev_t preferred_device() {
    if (ggml_backend_reg_count() == 0) {
        ggml_backend_load_all();
    }
    ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (!dev) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_IGPU);
    }
    if (!dev) {
        dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    }
    return dev;
}

static string detected_device_name() {
    const char * configured = env_any("UTOPIC_RUNTIME_DEVICE");
    if (configured && configured[0]) {
        return configured;
    }
    ggml_backend_dev_t dev = preferred_device();
    if (!dev) {
        return "unknown device";
    }
    const char * description = ggml_backend_dev_description(dev);
    if (description && description[0]) {
        return description;
    }
    const char * name = ggml_backend_dev_name(dev);
    return name && name[0] ? name : "unknown device";
}

static bool detected_gpu_memory_gib(double & out) {
    ggml_backend_dev_t dev = preferred_device();
    if (!dev) {
        return false;
    }
    const enum ggml_backend_dev_type type = ggml_backend_dev_type(dev);
    if (type != GGML_BACKEND_DEVICE_TYPE_GPU && type != GGML_BACKEND_DEVICE_TYPE_IGPU) {
        return false;
    }
    size_t free_bytes  = 0;
    size_t total_bytes = 0;
    ggml_backend_dev_memory(dev, &free_bytes, &total_bytes);
    if (total_bytes == 0) {
        return false;
    }
    out = (double) total_bytes / (1024.0 * 1024.0 * 1024.0);
    return true;
}

static json error_response(const string & code, const string & message, const json & detail = json::object()) {
    return {
        { "ok", false },
        { "error", {
            { "code", code },
            { "message", message },
            { "detail", detail.is_object() ? detail : json::object() },
        } },
    };
}

static json detected_capacity() {
    const char * raw_memory = getenv("UTOPIC_GPU_MEMORY_GIB");
    json detected = json::object();
    if (raw_memory && raw_memory[0]) {
        const double memory_gib = atof(raw_memory);
        detected["backend"] = env_any("UTOPIC_RUNTIME_BACKEND") ? env_any("UTOPIC_RUNTIME_BACKEND") : "configured";
        detected["device"]  = env_any("UTOPIC_RUNTIME_DEVICE") ? env_any("UTOPIC_RUNTIME_DEVICE") : "configured runtime";
        if (memory_gib >= 0.0) {
            detected["gpu_memory_gib"] = memory_gib;
        }
        return detected;
    }
    const char * configured_backend = env_any("UTOPIC_RUNTIME_BACKEND");
    detected["backend"] = configured_backend && configured_backend[0] ? configured_backend : host_backend();
    detected["device"]  = detected_device_name();
    double memory_gib   = 0.0;
    if (detected_gpu_memory_gib(memory_gib)) {
        detected["gpu_memory_gib"] = memory_gib;
    }
    return detected;
}

static string detected_capacity_text(const json & detected) {
    string text = detected.value("device", "unknown device");
    const string backend = detected.value("backend", "unknown");
    if (!backend.empty()) {
        text += " ";
        text += backend;
    }
    if (detected.contains("gpu_memory_gib") && detected["gpu_memory_gib"].is_number()) {
        char buf[64];
        snprintf(buf, sizeof(buf), " with %.3g GiB GPU memory", detected["gpu_memory_gib"].get<double>());
        text += buf;
    }
    return text;
}

string host_device() {
    return detected_device_name();
}

static bool json_string_array_contains(const json & values, const string & needle) {
    if (!values.is_array() || needle.empty()) {
        return false;
    }
    for (const auto & value : values) {
        if (value.is_string() && value.get<string>() == needle) {
            return true;
        }
    }
    return false;
}

static bool json_iopt(const json & opts, const char * key, int32_t & out) {
    if (!opts.contains(key) || !opts[key].is_number_integer()) {
        return false;
    }
    out = opts[key].get<int32_t>();
    return true;
}

static bool json_fopt(const json & opts, const char * key, float & out) {
    if (!opts.contains(key) || !opts[key].is_number()) {
        return false;
    }
    out = opts[key].get<float>();
    return true;
}

static bool json_bopt(const json & opts, const char * key, bool & out) {
    if (!opts.contains(key) || !opts[key].is_boolean()) {
        return false;
    }
    out = opts[key].get<bool>();
    return true;
}

static string json_sopt(const json & primary, const json & secondary, const char * key, const string & def = "") {
    if (primary.contains(key) && primary[key].is_string()) {
        return primary[key].get<string>();
    }
    if (secondary.contains(key) && secondary[key].is_string()) {
        return secondary[key].get<string>();
    }
    return def;
}

static bool parse_size(const string & value, int32_t & width, int32_t & height) {
    if (value.empty()) {
        return false;
    }
    const size_t sep = value.find_first_of("xX");
    if (sep == string::npos || sep == 0 || sep + 1 >= value.size()) {
        return false;
    }
    char * end_w = nullptr;
    char * end_h = nullptr;
    const long parsed_w = strtol(value.substr(0, sep).c_str(), &end_w, 10);
    const long parsed_h = strtol(value.substr(sep + 1).c_str(), &end_h, 10);
    if (!end_w || *end_w || !end_h || *end_h || parsed_w <= 0 || parsed_h <= 0) {
        return false;
    }
    width  = (int32_t) parsed_w;
    height = (int32_t) parsed_h;
    return true;
}

static string lowercase_ascii(string value) {
    for (char & ch : value) {
        ch = (char) std::tolower((unsigned char) ch);
    }
    return value;
}

static bool string_ends_with(const string & value, const char * suffix) {
    const string suffix_value = suffix ? suffix : "";
    if (suffix_value.size() > value.size()) {
        return false;
    }
    return value.compare(value.size() - suffix_value.size(), suffix_value.size(), suffix_value) == 0;
}

static string native_image_artifact_role(const string & filename) {
    string normalized = lowercase_ascii(filename);
    for (char & ch : normalized) {
        if (ch == '-') {
            ch = '_';
        }
    }
    if (normalized.find("clip_l") != string::npos || normalized.find("text_encoder") != string::npos) {
        return "clip_l_path";
    }
    if (normalized.find("clip_g") != string::npos) {
        return "clip_g_path";
    }
    if (normalized.find("t5xxl") != string::npos || normalized.rfind("t5_", 0) == 0
        || normalized.find("_t5_") != string::npos) {
        return "t5xxl_path";
    }
    if (normalized.find("vae") != string::npos || normalized == "ae.safetensors") {
        return "vae_path";
    }
    if (string_ends_with(normalized, ".gguf") || normalized.find("diffusion") != string::npos
        || normalized.find("unet") != string::npos || normalized.find("model") != string::npos) {
        return "diffusion_model_path";
    }
    return "";
}

static void set_if_empty(string & field, const string & value) {
    if (field.empty()) {
        field = value;
    }
}

static void apply_native_image_artifact_paths(const json & opts, image_engine_params & params) {
    const json paths = opts.value("artifact_paths", json::object());
    if (!paths.is_object()) {
        return;
    }
    for (auto it = paths.begin(); it != paths.end(); ++it) {
        if (!it.value().is_string()) {
            continue;
        }
        const string role = native_image_artifact_role(it.key());
        const string path = it.value().get<string>();
        if (role == "diffusion_model_path") {
            if (string_ends_with(lowercase_ascii(it.key()), ".gguf") || params.diffusion_model_path.empty()) {
                params.diffusion_model_path = path;
            }
        } else if (role == "vae_path") {
            set_if_empty(params.vae_path, path);
        } else if (role == "clip_l_path") {
            set_if_empty(params.clip_l_path, path);
        } else if (role == "clip_g_path") {
            set_if_empty(params.clip_g_path, path);
        } else if (role == "t5xxl_path") {
            set_if_empty(params.t5xxl_path, path);
        }
    }
}

json backend_preflight_error(const json & root, const string & runner_name) {
    const json opts = root.value("options", json::object());
    if (!opts.is_object()) {
        return json();
    }
    const json supported = opts.value("supported_backends", json::array());
    if (!supported.is_array() || supported.empty()) {
        return json();
    }
    const json detected = detected_capacity();
    const string backend = detected.value("backend", host_backend());
    if (json_string_array_contains(supported, backend)) {
        return json();
    }

    char message[256];
    snprintf(
        message,
        sizeof(message),
        "model %s does not support the detected %s backend on %s",
        root.value("model", "").c_str(),
        backend.c_str(),
        detected.value("device", "unknown device").c_str());
    return error_response("backend_unavailable", message, {
        { "task", root.value("task", "") },
        { "model", root.value("model", "") },
        { "modality", opts.value("modality", root.value("task", "")) },
        { "engine", opts.value("engine", "") },
        { "runtime", opts.value("runtime", "") },
        { "runner", opts.value("runner", runner_name) },
        { "native_status", opts.value("native_status", "") },
        { "supported_backends", supported },
        { "expected_vram_gib", opts.value("expected_vram_gib", json()) },
        { "expected_ram_gib", opts.value("expected_ram_gib", json()) },
        { "requirements", opts.value("requirements", json::object()) },
        { "oom_policy", opts.value("oom_policy", json::object()) },
        { "detected", detected },
    });
}

json capacity_preflight_error(const json & root, const string & runner_name) {
    const json opts         = root.value("options", json::object());
    const json requirements = opts.value("requirements", json::object());
    if (!requirements.is_object()) {
        return json();
    }
    double minimum = 0.0;
    if (!json_dopt(requirements, "min_gpu_memory_gib", minimum)) {
        return json();
    }
    const bool allow_cpu = !requirements.contains("allow_cpu") || !requirements["allow_cpu"].is_boolean()
        || requirements["allow_cpu"].get<bool>();
    const json detected       = detected_capacity();
    const bool has_memory     = detected.contains("gpu_memory_gib") && detected["gpu_memory_gib"].is_number();
    const bool has_enough_gpu = has_memory && detected["gpu_memory_gib"].get<double>() >= minimum;
    if (has_enough_gpu) {
        return json();
    }
    if (allow_cpu && detected.value("backend", "") == "cpu") {
        return json();
    }

    char message[256];
    snprintf(
        message,
        sizeof(message),
        "model %s requires at least %.3g GiB GPU memory; detected %s. This model is too large for this host.",
        root.value("model", "").c_str(),
        minimum,
        detected_capacity_text(detected).c_str());
    return error_response("oom", message, {
        { "task", root.value("task", "") },
        { "model", root.value("model", "") },
        { "modality", opts.value("modality", root.value("task", "")) },
        { "engine", opts.value("engine", "") },
        { "runtime", opts.value("runtime", "") },
        { "runner", opts.value("runner", runner_name) },
        { "native_status", opts.value("native_status", "") },
        { "supported_backends", opts.value("supported_backends", json::array()) },
        { "expected_vram_gib", opts.value("expected_vram_gib", json()) },
        { "expected_ram_gib", opts.value("expected_ram_gib", json()) },
        { "requirements", requirements },
        { "oom_policy", opts.value("oom_policy", json::object()) },
        { "required_gpu_memory_gib", minimum },
        { "detected", detected },
    });
}

static json run_image_task(const runner_request & req) {
    std::error_code ec;
    fs::create_directories(req.output_dir, ec);
    if (ec) {
        return error_response("runner_failed", string("failed to create output directory: ") + ec.message(), {
            { "output_dir", req.output_dir },
        });
    }

    image_engine_params params;
    params.model_path             = json_sopt(req.options, req.input, "model_path");
    params.vae_path               = json_sopt(req.options, req.input, "vae_path");
    params.clip_l_path            = json_sopt(req.options, req.input, "clip_l_path");
    params.clip_g_path            = json_sopt(req.options, req.input, "clip_g_path");
    params.t5xxl_path             = json_sopt(req.options, req.input, "t5xxl_path");
    params.diffusion_model_path    = json_sopt(req.options, req.input, "diffusion_model_path");
    apply_native_image_artifact_paths(req.options, params);
    params.prompt                 = json_sopt(req.input, req.options, "prompt");
    params.negative_prompt        = json_sopt(req.input, req.options, "negative_prompt");
    params.backend                = json_sopt(req.options, req.input, "backend");
    params.params_backend         = json_sopt(req.options, req.input, "params_backend");
    params.output_path            = json_sopt(req.options, req.input, "output_path",
                                             (fs::path(req.output_dir) / "image.png").string());

    const string size = json_sopt(req.options, req.input, "size");
    parse_size(size, params.width, params.height);
    json_iopt(req.options, "width", params.width) || json_iopt(req.input, "width", params.width);
    json_iopt(req.options, "height", params.height) || json_iopt(req.input, "height", params.height);
    json_iopt(req.options, "steps", params.steps) || json_iopt(req.input, "steps", params.steps);
    json_iopt(req.options, "seed", params.seed) || json_iopt(req.input, "seed", params.seed);
    json_iopt(req.options, "batch_count", params.batch_count) || json_iopt(req.input, "batch_count", params.batch_count);
    json_iopt(req.options, "n_threads", params.n_threads) || json_iopt(req.input, "n_threads", params.n_threads);
    json_fopt(req.options, "cfg_scale", params.cfg_scale) || json_fopt(req.input, "cfg_scale", params.cfg_scale);
    json_fopt(req.options, "guidance_scale", params.cfg_scale) || json_fopt(req.input, "guidance_scale", params.cfg_scale);
    json_fopt(req.options, "distilled_guidance", params.distilled_guidance)
        || json_fopt(req.input, "distilled_guidance", params.distilled_guidance);
    json_fopt(req.options, "eta", params.eta) || json_fopt(req.input, "eta", params.eta);
    json_bopt(req.options, "enable_mmap", params.enable_mmap) || json_bopt(req.input, "enable_mmap", params.enable_mmap);
    json_bopt(req.options, "flash_attn", params.flash_attn) || json_bopt(req.input, "flash_attn", params.flash_attn);
    json_bopt(req.options, "diffusion_flash_attn", params.diffusion_flash_attn)
        || json_bopt(req.input, "diffusion_flash_attn", params.diffusion_flash_attn);
    json_bopt(req.options, "qwen_image_zero_cond_t", params.qwen_image_zero_cond_t)
        || json_bopt(req.input, "qwen_image_zero_cond_t", params.qwen_image_zero_cond_t);

    if (params.model_path.empty() && params.diffusion_model_path.empty()) {
        return error_response("missing_model", "options.model_path or options.diffusion_model_path is required for native image generation", {
            { "task", runner_task_json_name(req.task) },
            { "model", req.model },
        });
    }
    if (params.prompt.empty()) {
        return error_response("runner_failed", "input.prompt is required for native image generation", {
            { "task", runner_task_json_name(req.task) },
            { "model", req.model },
        });
    }

    image_engine_result result;
    if (!image_engine_generate(params, result)) {
        return error_response("runner_failed", result.error_message.empty() ? "native image generation failed" : result.error_message, {
            { "task", runner_task_json_name(req.task) },
            { "model", req.model },
            { "model_path", params.model_path },
            { "diffusion_model_path", params.diffusion_model_path },
        });
    }

    const json artifact = {
        { "type", "image/png" },
        { "path", result.artifact_path },
        { "url", string("file://") + result.artifact_path },
        { "width", result.width },
        { "height", result.height },
        { "seed", result.seed },
    };
    return {
        { "ok", true },
        { "type", "image" },
        { "text", "" },
        { "artifacts", json::array({ artifact }) },
        { "backend", host_backend() },
        { "device", host_device() },
        { "metrics", {
            { "width", result.width },
            { "height", result.height },
            { "channel", result.channel },
            { "seed", result.seed },
            { "steps", params.steps },
        } },
    };
}

static json native_runner_unavailable(
        const runner_request & req,
        const string &         modality,
        const string &         message,
        const string &         artifact_type,
        const string &         artifact_filename,
        const string &         input_key) {
    return error_response("backend_unavailable", message, {
        { "task", runner_task_json_name(req.task) },
        { "model", req.model },
        { "modality", modality.empty() ? string(runner_task_json_name(req.task)) : modality },
        { "engine", req.options.value("engine", "") },
        { "runtime", req.options.value("runtime", "") },
        { "runner", req.options.value("runner", req.runner) },
        { "native_status", req.options.value("native_status", "") },
        { "supported_backends", req.options.value("supported_backends", json::array()) },
        { "expected_vram_gib", req.options.value("expected_vram_gib", json()) },
        { "expected_ram_gib", req.options.value("expected_ram_gib", json()) },
        { "requirements", req.options.value("requirements", json::object()) },
        { "oom_policy", req.options.value("oom_policy", json::object()) },
        { "detected", detected_capacity() },
        { "input_key", input_key },
        { "artifact_type", artifact_type },
        { "artifact_filename", artifact_filename },
        { "output_dir", req.output_dir },
        { "output_contract", {
            { "artifact_type", artifact_type },
            { "artifact_filename", artifact_filename },
            { "output_dir", req.output_dir },
        } },
    });
}

static bool json_samples(const json & input, std::vector<float> & out, string & error) {
    out.clear();
    if (!input.contains("samples")) {
        return false;
    }
    if (!input["samples"].is_array() || input["samples"].empty()) {
        error = "input.samples must be a non-empty numeric array";
        return true;
    }
    for (const auto & value : input["samples"]) {
        if (!value.is_number()) {
            error = "input.samples must contain only numbers";
            out.clear();
            return true;
        }
        out.push_back(value.get<float>());
    }
    return true;
}

static json run_pcm_audio_task(const runner_request & req, const string & artifact_filename) {
    std::vector<float> samples;
    string samples_error;
    if (!json_samples(req.input, samples, samples_error)) {
        return json();
    }
    if (!samples_error.empty()) {
        return error_response("runner_failed", samples_error, {
            { "task", runner_task_json_name(req.task) },
            { "model", req.model },
        });
    }

    audio_engine_wav_params params;
    params.output_path = json_sopt(
        req.options, req.input, "output_path", (fs::path(req.output_dir) / artifact_filename).string());
    params.samples      = samples.data();
    params.sample_count = samples.size();
    json_iopt(req.input, "sample_rate", params.sample_rate) || json_iopt(req.options, "sample_rate", params.sample_rate);
    json_iopt(req.input, "channel_count", params.channel_count) || json_iopt(req.options, "channel_count", params.channel_count);

    audio_engine_result result;
    if (!audio_engine_write_wav(params, result)) {
        return error_response(
            "runner_failed",
            result.error_message.empty() ? "native audio artifact generation failed" : result.error_message,
            {
                { "task", runner_task_json_name(req.task) },
                { "model", req.model },
                { "output_path", params.output_path },
            });
    }

    const json artifact = {
        { "type", "audio/wav" },
        { "path", result.artifact_path },
        { "url", string("file://") + result.artifact_path },
        { "sample_rate", result.sample_rate },
        { "channel_count", result.channel_count },
        { "sample_count", result.sample_count },
        { "duration_ms", result.duration_ms },
    };
    return {
        { "ok", true },
        { "type", "audio" },
        { "text", "" },
        { "artifacts", json::array({ artifact }) },
        { "backend", host_backend() },
        { "device", host_device() },
        { "metrics", {
            { "sample_rate", result.sample_rate },
            { "channel_count", result.channel_count },
            { "sample_count", result.sample_count },
            { "duration_ms", result.duration_ms },
        } },
    };
}

static bool json_u8_frames(const json & input, std::vector<uint8_t> & out, string & error) {
    out.clear();
    if (!input.contains("frames")) {
        return false;
    }
    if (!input["frames"].is_array() || input["frames"].empty()) {
        error = "input.frames must be a non-empty byte array";
        return true;
    }
    for (const auto & value : input["frames"]) {
        if (!value.is_number_integer()) {
            error = "input.frames must contain only integer byte values";
            out.clear();
            return true;
        }
        const int byte = value.get<int>();
        if (byte < 0 || byte > 255) {
            error = "input.frames values must be between 0 and 255";
            out.clear();
            return true;
        }
        out.push_back((uint8_t) byte);
    }
    return true;
}

static json run_frame_video_task(const runner_request & req) {
    std::vector<uint8_t> frames;
    string frames_error;
    if (!json_u8_frames(req.input, frames, frames_error)) {
        return json();
    }
    if (!frames_error.empty()) {
        return error_response("runner_failed", frames_error, {
            { "task", runner_task_json_name(req.task) },
            { "model", req.model },
        });
    }

    video_engine_frames_params params;
    params.output_dir = json_sopt(req.options, req.input, "output_dir", req.output_dir);
    params.frames     = frames.data();
    int32_t frame_count = 0;
    json_iopt(req.input, "frame_count", frame_count) || json_iopt(req.options, "frame_count", frame_count);
    params.frame_count = frame_count > 0 ? (size_t) frame_count : 0;
    json_iopt(req.input, "width", params.width) || json_iopt(req.options, "width", params.width);
    json_iopt(req.input, "height", params.height) || json_iopt(req.options, "height", params.height);
    json_iopt(req.input, "channel_count", params.channel_count) || json_iopt(req.options, "channel_count", params.channel_count);
    json_iopt(req.input, "fps", params.fps) || json_iopt(req.options, "fps", params.fps);

    const size_t expected = (size_t) params.frame_count * (size_t) params.width * (size_t) params.height * (size_t) params.channel_count;
    if (expected == 0 || frames.size() != expected) {
        return error_response("runner_failed", "input.frames byte count does not match frame_count, width, height, and channel_count", {
            { "task", runner_task_json_name(req.task) },
            { "model", req.model },
            { "bytes", frames.size() },
            { "expected_bytes", expected },
        });
    }

    video_engine_result result;
    if (!video_engine_write_frames(params, result)) {
        return error_response(
            "runner_failed",
            result.error_message.empty() ? "native video artifact generation failed" : result.error_message,
            {
                { "task", runner_task_json_name(req.task) },
                { "model", req.model },
                { "output_dir", params.output_dir },
            });
    }

    const json artifact = {
        { "type", "application/vnd.utopic.video-frames+json" },
        { "path", result.metadata_path },
        { "url", string("file://") + result.metadata_path },
        { "frame_dir", result.artifact_path },
        { "frame_count", result.frame_count },
        { "width", result.width },
        { "height", result.height },
        { "fps", result.fps },
        { "duration_ms", result.duration_ms },
    };
    return {
        { "ok", true },
        { "type", "video" },
        { "text", "" },
        { "artifacts", json::array({ artifact }) },
        { "backend", host_backend() },
        { "device", host_device() },
        { "metrics", {
            { "frame_count", result.frame_count },
            { "width", result.width },
            { "height", result.height },
            { "fps", result.fps },
            { "duration_ms", result.duration_ms },
        } },
    };
}

static json run_tts_task(const runner_request & req) {
    json audio = run_pcm_audio_task(req, "speech.wav");
    if (!audio.is_null()) {
        return audio;
    }
    return native_runner_unavailable(
        req, "tts", "native tts runner is not available in this build",
        "audio/wav", "speech.wav", "input");
}

static json run_music_task(const runner_request & req) {
    json audio = run_pcm_audio_task(req, "music.wav");
    if (!audio.is_null()) {
        return audio;
    }
    return native_runner_unavailable(
        req, "music", "native music runner is not available in this build",
        "audio/wav", "music.wav", "prompt");
}

static json run_video_task(const runner_request & req) {
    json video = run_frame_video_task(req);
    if (!video.is_null()) {
        return video;
    }
    return native_runner_unavailable(
        req, "video", "native video runner is not available in this build",
        "application/vnd.utopic.video-frames+json", "metadata.json", "prompt");
}

static json run_misc_task(const runner_request & req) {
    return native_runner_unavailable(
        req, "misc", "native misc runner is not available in this build",
        "application/octet-stream", "artifact.bin", "artifact");
}

json run_artifact_task(const runner_request & req, const json & root) {
    (void) root;
    if (runner_plugin_configured(req)) {
        json response = runner_plugin_generate(req);
        if (response.is_object() && response.value("ok", false)) {
            if (!response.contains("backend")) {
                response["backend"] = host_backend();
            }
            if (!response.contains("device")) {
                response["device"] = host_device();
            }
        }
        return response;
    }
    if (req.task == utopic::RUNNER_TASK_IMAGE) {
        return run_image_task(req);
    }
    if (req.task == utopic::RUNNER_TASK_TTS) {
        return run_tts_task(req);
    }
    if (req.task == utopic::RUNNER_TASK_MUSIC) {
        return run_music_task(req);
    }
    if (req.task == utopic::RUNNER_TASK_VIDEO) {
        return run_video_task(req);
    }
    if (req.task == utopic::RUNNER_TASK_MISC) {
        return run_misc_task(req);
    }
    return error_response("unsupported_model", "native task is not implemented behind utopic-runner yet", {
        { "task", runner_task_json_name(req.task) },
        { "model", req.model },
        { "modality", req.options.value("modality", string(runner_task_json_name(req.task))) },
        { "engine", req.options.value("engine", "") },
        { "runtime", req.options.value("runtime", "") },
        { "runner", req.options.value("runner", req.runner) },
        { "native_status", req.options.value("native_status", "") },
        { "supported_backends", req.options.value("supported_backends", json::array()) },
        { "expected_vram_gib", req.options.value("expected_vram_gib", json()) },
        { "expected_ram_gib", req.options.value("expected_ram_gib", json()) },
        { "requirements", req.options.value("requirements", json::object()) },
        { "oom_policy", req.options.value("oom_policy", json::object()) },
        { "detected", detected_capacity() },
    });
}

}  // namespace utopic_runner
