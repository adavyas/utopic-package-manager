// JSON runner for local Utopic execution.
// This is intentionally one-shot: Python owns orchestration/debuggability, while
// this binary owns native model loading and generation.
//
//   ./utopic_runner --json-request request.json
//
// Request:
//   {"schema_version":"utopic-runner/v1","task":"chat","model":"catalog-id",
//    "input":{"prompt":"..."},"options":{"model_path":"model.gguf"},
//    "output_dir":"/tmp/utopic-run"}
//
// Response:
//   {"ok":true,"type":"text","text":"...","artifacts":[],"metrics":{...},"backend":"metal"}
#include "utopic_core.h"
#include "llama.h"
#include "nlohmann/json.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

using namespace utopic;
using json = nlohmann::json;
using std::pair;
using std::string;
using std::vector;

static const char * arg(int argc, char ** argv, const char * flag, const char * def) {
    for (int i = 1; i < argc - 1; ++i) {
        if (!strcmp(argv[i], flag)) {
            return argv[i + 1];
        }
    }
    return def;
}

static bool flag_set(int argc, char ** argv, const char * flag) {
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], flag)) {
            return true;
        }
    }
    return false;
}

static string executable_name(const char * path) {
    if (!path || !path[0]) {
        return "utopic_runner";
    }
    string name(path);
    const size_t slash = name.find_last_of("/\\");
    if (slash != string::npos) {
        name = name.substr(slash + 1);
    }
#ifdef _WIN32
    const string suffix = ".exe";
    if (name.size() > suffix.size() && name.substr(name.size() - suffix.size()) == suffix) {
        name.resize(name.size() - suffix.size());
    }
#endif
    return name.empty() ? "utopic_runner" : name;
}

static const char * env_any(const char * preferred, const char * legacy = nullptr) {
    const char * v = preferred ? getenv(preferred) : nullptr;
    if (v) {
        return v;
    }
    return legacy ? getenv(legacy) : nullptr;
}

static int json_iopt(const json & opts, const char * key, const char * env, const char * legacy_env, int def) {
    if (opts.contains(key) && opts[key].is_number_integer()) {
        return opts[key].get<int>();
    }
    const char * e = env_any(env, legacy_env);
    return e ? atoi(e) : def;
}

static float json_fopt(const json & opts, const char * key, const char * env, const char * legacy_env, float def) {
    if (opts.contains(key) && opts[key].is_number()) {
        return opts[key].get<float>();
    }
    const char * e = env_any(env, legacy_env);
    return e ? (float) atof(e) : def;
}

static bool json_dopt(const json & obj, const char * key, double & out) {
    if (!obj.contains(key) || !obj[key].is_number()) {
        return false;
    }
    out = obj[key].get<double>();
    return out > 0.0;
}

static string host_backend() {
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

static json invalid_request(const string & message, const string & field) {
    return error_response("invalid_request", message, {
        {"field", field},
        {"schema_version", "utopic-runner/v1"},
    });
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
    detected["backend"] = env_any("UTOPIC_RUNTIME_BACKEND") ? env_any("UTOPIC_RUNTIME_BACKEND") : "unknown";
    detected["device"] = env_any("UTOPIC_RUNTIME_DEVICE") ? env_any("UTOPIC_RUNTIME_DEVICE") : "unknown device";
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

static json capacity_preflight_error(const json & root, const string & runner_name) {
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
        {"required_gpu_memory_gib", minimum},
        {"detected", detected},
    });
}

static bool read_json_file(const char * path, json & out, string & error) {
    std::ifstream in(path);
    if (!in) {
        error = string("failed to open request file: ") + strerror(errno);
        return false;
    }
    try {
        in >> out;
    } catch (const std::exception & exc) {
        error = string("invalid JSON request: ") + exc.what();
        return false;
    }
    return true;
}

static bool validate_request_contract(const json & root, json & response) {
    if (root.contains("schema_version")) {
        if (!root["schema_version"].is_string()) {
            response = invalid_request("schema_version must be utopic-runner/v1", "schema_version");
            return false;
        }
        if (root["schema_version"].get<string>() != "utopic-runner/v1") {
            response = invalid_request("unsupported schema_version", "schema_version");
            return false;
        }
    }
    if (!root.contains("task") || !root["task"].is_string() || root["task"].get<string>().empty()) {
        response = invalid_request("task is required", "task");
        return false;
    }
    const string task = root["task"].get<string>();
    if (task != "chat" && task != "image" && task != "tts" && task != "music" && task != "video" && task != "misc") {
        response = invalid_request("task must be chat, image, tts, music, video, or misc", "task");
        return false;
    }
    if (!root.contains("model") || !root["model"].is_string() || root["model"].get<string>().empty()) {
        response = invalid_request("model is required", "model");
        return false;
    }
    if (!root.contains("input") || !root["input"].is_object()) {
        response = invalid_request("input must be an object", "input");
        return false;
    }
    if (!root.contains("options") || !root["options"].is_object()) {
        response = invalid_request("options must be an object", "options");
        return false;
    }
    if (!root.contains("output_dir") || !root["output_dir"].is_string() || root["output_dir"].get<string>().empty()) {
        response = invalid_request("output_dir is required", "output_dir");
        return false;
    }
    return true;
}

static vector<pair<string, string>> request_messages(const json & input) {
    vector<pair<string, string>> messages;
    if (input.contains("messages") && input["messages"].is_array()) {
        for (const auto & msg : input["messages"]) {
            if (!msg.is_object()) {
                continue;
            }
            string role = msg.value("role", "user");
            if (role != "system" && role != "assistant" && role != "user") {
                role = "user";
            }
            string content;
            if (msg.contains("content") && msg["content"].is_string()) {
                content = msg["content"].get<string>();
            } else if (msg.contains("content")) {
                content = msg["content"].dump();
            }
            messages.push_back({ role, content });
        }
    }
    if (!messages.empty()) {
        return messages;
    }
    if (input.contains("system") && input["system"].is_string()) {
        messages.push_back({ "system", input["system"].get<string>() });
    }
    messages.push_back({ "user", input.value("prompt", "") });
    return messages;
}

static json run_chat(const json & root) {
    const json input = root.value("input", json::object());
    const json opts  = root.value("options", json::object());
    const string model_id = root.value("model", "");
    const string model_path = opts.value("model_path", "");

    if (model_path.empty()) {
        return error_response("missing_model", "options.model_path is required for native chat");
    }
    if (input.value("prompt", "").empty() && !input.contains("messages")) {
        return error_response("runner_failed", "input.prompt or input.messages is required");
    }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = json_iopt(opts, "gpu_layers", nullptr, nullptr, 99);
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        llama_backend_free();
        return error_response("missing_model", "failed to load model", {
            {"model", model_id},
            {"model_path", model_path},
        });
    }

    vector<pair<string, string>> messages = request_messages(input);
    request req;
    req.prompt        = apply_chat(model, messages);
    req.max_tokens    = json_iopt(opts, "max_tokens", nullptr, nullptr, 256);
    req.temperature   = json_fopt(opts, "temperature", nullptr, nullptr, 0.0f);
    req.seed          = json_iopt(opts, "seed", "UTOPIC_SEED", "DG_SEED", 0);
    req.steps         = json_iopt(opts, "steps", nullptr, nullptr, 256);
    req.block_length  = json_iopt(opts, "diffusion_block_length", "UTOPIC_BLOCK", "DG_BLOCK", 32);
    req.canvas_tokens = json_iopt(opts, "canvas", "UTOPIC_CANVAS", "DG_CANVAS", 0);
    req.confidence    = json_fopt(opts, "confidence", "UTOPIC_CONF", "DG_CONF", 0.9f);
    req.converge      = json_iopt(opts, "converge", "UTOPIC_CONVERGE", "DG_CONVERGE", 2);
    req.eb_steps      = json_iopt(opts, "eb_steps", "UTOPIC_EB_STEPS", "DG_EB_STEPS", 0);
    req.slot_len      = json_iopt(opts, "slot_len", "UTOPIC_SLOT_LEN", "DG_SLOT_LEN", 8);
    req.schema        = opts.value("schema", "");
    req.schema_hard   = opts.value("schema_mode", "hard") != "prompt";
    req.tools         = opts.value("tools", false);
    const char * eos_stop = env_any("UTOPIC_EOS_STOP", "DG_EOS_STOP");
    req.eos_stop      = !(eos_stop && !atoi(eos_stop));

    prepare_model_for_context(model);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = ctx_size_for(model, req);
    cp.n_batch = cp.n_ubatch = cp.n_ctx;
    const char * no_fa = env_any("UTOPIC_NO_FA", "DG_NO_FA");
    if (no_fa && atoi(no_fa)) {
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    }
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        llama_model_free(model);
        llama_backend_free();
        return error_response("runner_failed", "context initialization failed", {
            {"model", model_id},
            {"model_path", model_path},
        });
    }

    const double t_start_total = now_ms();
    result r = generate(ctx, model, req);
    const double total_ms = now_ms() - t_start_total;

    json out;
    if (!r.ok) {
        out = error_response("runner_failed", "generation failed", {
            {"model", model_id},
            {"model_path", model_path},
        });
    } else {
        out = {
            {"ok", true},
            {"type", "text"},
            {"text", r.text},
            {"artifacts", json::array()},
            {"backend", host_backend()},
            {"metrics", {
                {"prompt_tokens", r.prompt_tokens},
                {"answer_tokens", r.answer_tokens},
                {"steps", r.steps},
                {"canvas", r.canvas},
                {"time_to_first_token_ms", r.ttft_ms},
                {"generation_ms", r.gen_ms},
                {"total_ms", total_ms},
            }},
        };
        if (!r.reasoning.empty()) {
            out["reasoning"] = r.reasoning;
        }
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return out;
}

int main(int argc, char ** argv) {
    llama_log_set([](ggml_log_level, const char * text, void *) { fputs(text, stderr); }, nullptr);
    const string runner_name = executable_name(argc > 0 ? argv[0] : nullptr);

    if (flag_set(argc, argv, "--help") || flag_set(argc, argv, "-h")) {
        fprintf(stderr, "usage: %s --json-request request.json\n", argv[0]);
        return 0;
    }
    const char * request_path = arg(argc, argv, "--json-request", nullptr);
    if (!request_path) {
        printf("%s\n", error_response("runner_failed", "--json-request is required").dump().c_str());
        return 2;
    }

    json root;
    string error;
    if (!read_json_file(request_path, root, error)) {
        printf("%s\n", error_response("runner_failed", error).dump().c_str());
        return 2;
    }
    if (!root.is_object()) {
        printf("%s\n", error_response("runner_failed", "request JSON must be an object").dump().c_str());
        return 2;
    }

    json response;
    if (!validate_request_contract(root, response)) {
        printf("%s\n", response.dump().c_str());
        return 2;
    }
    response = capacity_preflight_error(root, runner_name);
    if (!response.is_null()) {
        printf("%s\n", response.dump().c_str());
        return 1;
    }
    try {
        const string task = root.value("task", "");
        if (task == "chat") {
            response = run_chat(root);
        } else {
            const json opts = root.value("options", json::object());
            response = error_response("unsupported_model", "native C++ runner task is not implemented yet", {
                {"task", task},
                {"model", root.value("model", "")},
                {"modality", opts.value("modality", task)},
                {"engine", opts.value("engine", "")},
                {"runtime", opts.value("runtime", "")},
                {"runner", opts.value("runner", runner_name)},
                {"native_status", opts.value("native_status", "")},
                {"supported_backends", opts.value("supported_backends", json::array())},
                {"expected_vram_gib", opts.value("expected_vram_gib", json())},
                {"expected_ram_gib", opts.value("expected_ram_gib", json())},
                {"detected", detected_capacity()},
            });
        }
    } catch (const std::exception & exc) {
        response = error_response("runner_failed", string("request handling failed: ") + exc.what());
    }

    printf("%s\n", response.dump().c_str());
    return response.value("ok", false) ? 0 : 1;
}
