// JSON runner for local Utopic execution.
// This is intentionally one-shot: Python owns orchestration/debuggability, while
// this binary owns native model loading and generation.
//
//   ./utopic-runner --json-request request.json
//
// Request:
//   {"schema_version":"utopic-runner/v1","run_id":"run_...","task":"chat","model":"catalog-id",
//    "input":{"prompt":"..."},"options":{"model_path":"model.gguf"},
//    "output_dir":"/tmp/utopic/run_.../outputs","progress_path":"/tmp/utopic/run_.../progress.jsonl"}
//
// Response:
//   {"ok":true,"type":"text","text":"...","artifacts":[],"metrics":{...},"backend":"metal"}
#include "runner_tasks.h"
#include "utopic_core.h"
#include "llama.h"
#include "nlohmann/json.hpp"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

using namespace utopic;
using json = nlohmann::json;
namespace fs = std::filesystem;
using std::pair;
using std::string;
using std::vector;
using utopic_runner::backend_preflight_error;
using utopic_runner::capacity_preflight_error;
using utopic_runner::host_backend;
using utopic_runner::host_device;
using utopic_runner::run_planned_native_task;
using utopic_runner::runner_request;

static const char * RUNNER_SCHEMA_VERSION = "utopic-runner/v1";

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
        return "utopic-runner";
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
    if (name == "utopic_runner") {
        return "utopic-runner";
    }
    return name.empty() ? "utopic-runner" : name;
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

static json contract_error(const string & message, const string & field) {
    return error_response("runner_failed", message, {
        {"field", field},
        {"schema_version", RUNNER_SCHEMA_VERSION},
    });
}

static string context_init_error_code(const llama_model_params & mp) {
    const string backend = host_backend();
    if (mp.n_gpu_layers > 0 && backend != "cpu") {
        return "backend_unavailable";
    }
    return "runner_failed";
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
    if (!root.contains("schema_version") || !root["schema_version"].is_string() || root["schema_version"].get<string>().empty()) {
        response = contract_error("schema_version is required", "schema_version");
        return false;
    }
    if (root["schema_version"].get<string>() != RUNNER_SCHEMA_VERSION) {
        response = contract_error("unsupported schema_version", "schema_version");
        return false;
    }
    if (!root.contains("run_id") || !root["run_id"].is_string() || root["run_id"].get<string>().empty()) {
        response = contract_error("run_id is required", "run_id");
        return false;
    }
    if (!root.contains("task") || !root["task"].is_string() || root["task"].get<string>().empty()) {
        response = contract_error("task is required", "task");
        return false;
    }
    const string task = root["task"].get<string>();
    if (task != "chat" && task != "image" && task != "tts" && task != "music" && task != "video" && task != "misc") {
        response = contract_error("task must be chat, image, tts, music, video, or misc", "task");
        return false;
    }
    if (!root.contains("model") || !root["model"].is_string() || root["model"].get<string>().empty()) {
        response = contract_error("model is required", "model");
        return false;
    }
    if (!root.contains("input") || !root["input"].is_object()) {
        response = contract_error("input must be an object", "input");
        return false;
    }
    if (!root.contains("options") || !root["options"].is_object()) {
        response = contract_error("options must be an object", "options");
        return false;
    }
    if (!root.contains("output_dir") || !root["output_dir"].is_string() || root["output_dir"].get<string>().empty()) {
        response = contract_error("output_dir is required", "output_dir");
        return false;
    }
    if (!root.contains("progress_path") || !root["progress_path"].is_string() || root["progress_path"].get<string>().empty()) {
        response = contract_error("progress_path is required", "progress_path");
        return false;
    }
    return true;
}

static runner_request make_runner_request(const json & root, const string & runner_name) {
    return {
        root.value("run_id", ""),
        root.value("task", ""),
        root.value("model", ""),
        root.value("input", json::object()),
        root.value("options", json::object()),
        root.value("output_dir", ""),
        root.value("progress_path", ""),
        runner_name,
    };
}

static bool append_progress_event(const runner_request & req, const char * event_name, const json & detail = json::object()) {
    if (req.progress_path.empty()) {
        return false;
    }
    try {
        fs::path path(req.progress_path);
        if (path.has_parent_path()) {
            std::error_code ec;
            fs::create_directories(path.parent_path(), ec);
            if (ec) {
                return false;
            }
        }
        std::ofstream out(req.progress_path, std::ios::app);
        if (!out) {
            return false;
        }
        json event = {
            {"schema_version", RUNNER_SCHEMA_VERSION},
            {"run_id", req.run_id},
            {"task", req.task},
            {"model", req.model},
            {"runner", req.runner},
            {"event", event_name},
            {"time_ms", (long long) now_ms()},
        };
        if (detail.is_object()) {
            for (auto it = detail.begin(); it != detail.end(); ++it) {
                event[it.key()] = it.value();
            }
        }
        out << event.dump() << "\n";
        return true;
    } catch (...) {
        return false;
    }
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
        const string error_code = context_init_error_code(mp);
        llama_model_free(model);
        llama_backend_free();
        return error_response(error_code, "context initialization failed", {
            {"model", model_id},
            {"model_path", model_path},
            {"backend", host_backend()},
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
            {"device", host_device()},
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

static void attach_run_metadata(json & response, const runner_request & req) {
    if (!response.is_object()) {
        return;
    }
    if (!req.run_id.empty()) {
        response["run_id"] = req.run_id;
        response["progress_url"] = string("/v1/utopic/runs/") + req.run_id + "/events";
    }
    if (!req.output_dir.empty()) {
        response["output_dir"] = req.output_dir;
    }
    if (!req.progress_path.empty()) {
        response["progress_path"] = req.progress_path;
    }
}

static json run_request(const runner_request & req, const json & root) {
    if (req.task == "chat") {
        return run_chat(root);
    }
    return run_planned_native_task(req);
}

int main(int argc, char ** argv) {
    llama_log_set([](ggml_log_level, const char * text, void *) { fputs(text, stderr); }, nullptr);
    const string runner_name = executable_name(argc > 0 ? argv[0] : nullptr);

    if (flag_set(argc, argv, "--help") || flag_set(argc, argv, "-h")) {
        fprintf(stderr, "usage: %s --json-request request.json\n", argv[0]);
        fprintf(stderr, "\n");
        fprintf(stderr, "Utopic native runner JSON contract\n");
        fprintf(stderr, "  schema_version=utopic-runner/v1\n");
        fprintf(stderr, "  required fields: run_id, task, model, input, options, output_dir, progress_path\n");
        fprintf(stderr, "\n");
        fprintf(stderr, "Tasks:\n");
        fprintf(stderr, "  chat: native GGUF text generation\n");
        fprintf(stderr, "  image, tts, music, video, misc: planned native tasks\n");
        fprintf(stderr, "    planned tasks return structured unsupported_model readiness errors\n");
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
    const runner_request req = make_runner_request(root, runner_name);
    append_progress_event(req, "started");

    response = backend_preflight_error(root, runner_name);
    if (!response.is_null()) {
        append_progress_event(req, "failed", {{"error", response.value("error", json::object())}});
        attach_run_metadata(response, req);
        printf("%s\n", response.dump().c_str());
        return 1;
    }

    response = capacity_preflight_error(root, runner_name);
    if (!response.is_null()) {
        append_progress_event(req, "failed", {{"error", response.value("error", json::object())}});
        attach_run_metadata(response, req);
        printf("%s\n", response.dump().c_str());
        return 1;
    }
    try {
        response = run_request(req, root);
    } catch (const std::exception & exc) {
        response = error_response("runner_failed", string("request handling failed: ") + exc.what());
    }
    attach_run_metadata(response, req);

    if (response.value("ok", false)) {
        append_progress_event(req, "completed", {{"type", response.value("type", "")}});
    } else {
        append_progress_event(req, "failed", {{"error", response.value("error", json::object())}});
    }
    printf("%s\n", response.dump().c_str());
    return response.value("ok", false) ? 0 : 1;
}
