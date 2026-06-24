#include "runner_tasks.h"

#include "llama.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <sys/wait.h>

namespace utopic_runner {

namespace fs = std::filesystem;

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
    size_t free_bytes = 0;
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
        {"ok", false},
        {"error", {
            {"code", code},
            {"message", message},
            {"detail", detail.is_object() ? detail : json::object()},
        }},
    };
}

static string shell_quote(const string & value) {
    string out = "'";
    for (const char c : value) {
        if (c == '\'') {
            out += "'\\''";
        } else {
            out += c;
        }
    }
    out += "'";
    return out;
}

static json parse_last_json_line(const string & stdout_text, string & error) {
    std::istringstream in(stdout_text);
    string line;
    json last;
    bool found = false;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        try {
            json candidate = json::parse(line);
            if (candidate.is_object()) {
                last = candidate;
                found = true;
            }
        } catch (const std::exception &) {
            continue;
        }
    }
    if (!found) {
        error = "native task runner did not write a JSON object";
        return json();
    }
    return last;
}

static json detected_capacity() {
    const char * raw_memory = getenv("UTOPIC_GPU_MEMORY_GIB");
    json detected = json::object();
    if (raw_memory && raw_memory[0]) {
        const double memory_gib = atof(raw_memory);
        detected["backend"] = env_any("UTOPIC_RUNTIME_BACKEND") ? env_any("UTOPIC_RUNTIME_BACKEND") : "configured";
        detected["device"] = env_any("UTOPIC_RUNTIME_DEVICE") ? env_any("UTOPIC_RUNTIME_DEVICE") : "configured runtime";
        if (memory_gib >= 0.0) {
            detected["gpu_memory_gib"] = memory_gib;
        }
        return detected;
    }
    const char * configured_backend = env_any("UTOPIC_RUNTIME_BACKEND");
    detected["backend"] = configured_backend && configured_backend[0] ? configured_backend : host_backend();
    detected["device"] = detected_device_name();
    double memory_gib = 0.0;
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
        {"task", root.value("task", "")},
        {"model", root.value("model", "")},
        {"modality", opts.value("modality", root.value("task", ""))},
        {"engine", opts.value("engine", "")},
        {"runtime", opts.value("runtime", "")},
        {"runner", opts.value("runner", runner_name)},
        {"native_status", opts.value("native_status", "")},
        {"supported_backends", supported},
        {"expected_vram_gib", opts.value("expected_vram_gib", json())},
        {"expected_ram_gib", opts.value("expected_ram_gib", json())},
        {"requirements", opts.value("requirements", json::object())},
        {"oom_policy", opts.value("oom_policy", json::object())},
        {"detected", detected},
    });
}

json capacity_preflight_error(const json & root, const string & runner_name) {
    const json opts = root.value("options", json::object());
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
    const json detected = detected_capacity();
    const bool has_memory = detected.contains("gpu_memory_gib") && detected["gpu_memory_gib"].is_number();
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
        {"task", root.value("task", "")},
        {"model", root.value("model", "")},
        {"modality", opts.value("modality", root.value("task", ""))},
        {"engine", opts.value("engine", "")},
        {"runtime", opts.value("runtime", "")},
        {"runner", opts.value("runner", runner_name)},
        {"native_status", opts.value("native_status", "")},
        {"supported_backends", opts.value("supported_backends", json::array())},
        {"expected_vram_gib", opts.value("expected_vram_gib", json())},
        {"expected_ram_gib", opts.value("expected_ram_gib", json())},
        {"requirements", requirements},
        {"oom_policy", opts.value("oom_policy", json::object())},
        {"required_gpu_memory_gib", minimum},
        {"detected", detected},
    });
}

static json run_native_task_helper(const runner_request & req, const json & root, const string & helper_path) {
    if (helper_path.empty()) {
        return error_response("backend_unavailable", "native task runner path is required for ready artifact tasks", {
            {"task", req.task},
            {"model", req.model},
            {"runner", req.runner},
            {"native_status", req.options.value("native_status", "")},
        });
    }
    if (!fs::exists(helper_path)) {
        return error_response("backend_unavailable", "native task runner is not installed", {
            {"task", req.task},
            {"model", req.model},
            {"runner", req.runner},
            {"task_runner_path", helper_path},
            {"native_status", req.options.value("native_status", "")},
        });
    }

    std::error_code ec;
    fs::create_directories(req.output_dir, ec);
    if (ec) {
        return error_response("runner_failed", string("failed to create output directory: ") + ec.message(), {
            {"output_dir", req.output_dir},
        });
    }

    const fs::path request_path = fs::path(req.output_dir) / (req.run_id + "-" + req.task + "-request.json");
    {
        std::ofstream out(request_path);
        if (!out) {
            return error_response("runner_failed", string("failed to write native task request: ") + strerror(errno), {
                {"request_path", request_path.string()},
            });
        }
        out << root.dump();
    }

    const string command = shell_quote(helper_path) + " --json-request " + shell_quote(request_path.string());
    FILE * pipe = popen(command.c_str(), "r");
    if (!pipe) {
        return error_response("backend_unavailable", string("failed to launch native task runner: ") + strerror(errno), {
            {"task_runner_path", helper_path},
        });
    }

    string stdout_text;
    char buffer[4096];
    while (fgets(buffer, sizeof(buffer), pipe)) {
        stdout_text += buffer;
    }
    const int status = pclose(pipe);
    if (status != 0) {
        int exit_status = status;
        if (WIFEXITED(status)) {
            exit_status = WEXITSTATUS(status);
        }
        return error_response("runner_failed", "native task runner failed", {
            {"task", req.task},
            {"model", req.model},
            {"task_runner_path", helper_path},
            {"exit_status", exit_status},
            {"stdout", stdout_text},
        });
    }

    string parse_error;
    json response = parse_last_json_line(stdout_text, parse_error);
    if (!parse_error.empty()) {
        return error_response("runner_failed", parse_error, {
            {"task", req.task},
            {"model", req.model},
            {"task_runner_path", helper_path},
            {"stdout", stdout_text},
        });
    }
    if (!response.contains("ok") || !response["ok"].is_boolean()) {
        return error_response("runner_failed", "native task runner response must contain boolean ok", {
            {"task", req.task},
            {"model", req.model},
            {"task_runner_path", helper_path},
        });
    }
    return response;
}

json run_artifact_task(const runner_request & req, const json & root) {
    const string native_status = req.options.value("native_status", "");
    const string helper_path = req.options.value("task_runner_path", "");
    if (native_status == "ready" || native_status == "experimental") {
        return run_native_task_helper(req, root, helper_path);
    }
    return error_response("unsupported_model", "native task is not implemented behind utopic-runner yet", {
        {"task", req.task},
        {"model", req.model},
        {"modality", req.options.value("modality", req.task)},
        {"engine", req.options.value("engine", "")},
        {"runtime", req.options.value("runtime", "")},
        {"runner", req.options.value("runner", req.runner)},
        {"native_status", req.options.value("native_status", "")},
        {"supported_backends", req.options.value("supported_backends", json::array())},
        {"expected_vram_gib", req.options.value("expected_vram_gib", json())},
        {"expected_ram_gib", req.options.value("expected_ram_gib", json())},
        {"requirements", req.options.value("requirements", json::object())},
        {"oom_policy", req.options.value("oom_policy", json::object())},
        {"detected", detected_capacity()},
    });
}

}  // namespace utopic_runner
