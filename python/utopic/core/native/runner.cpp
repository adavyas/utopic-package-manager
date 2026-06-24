// JSON runner for local Utopic execution.
// This is intentionally one-shot: Python owns orchestration/debuggability, while
// this binary owns native model loading and generation.
//
//   ./utopic_runner --json-request request.json
//
// Request:
//   {"task":"chat","model":"catalog-id","input":{"prompt":"..."},"options":{"model_path":"model.gguf"}}
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
    try {
        const string task = root.value("task", "");
        if (task == "chat") {
            response = run_chat(root);
        } else {
            const json opts = root.value("options", json::object());
            response = error_response("unsupported_model", "native runner task is not implemented yet", {
                {"task", task},
                {"model", root.value("model", "")},
                {"modality", opts.value("modality", task)},
                {"engine", opts.value("engine", "")},
                {"runtime", opts.value("runtime", "")},
                {"runner", opts.value("runner", "")},
                {"native_status", opts.value("native_status", "")},
                {"supported_backends", opts.value("supported_backends", json::array())},
                {"expected_vram_gib", opts.value("expected_vram_gib", json())},
                {"expected_ram_gib", opts.value("expected_ram_gib", json())},
            });
        }
    } catch (const std::exception & exc) {
        response = error_response("runner_failed", string("request handling failed: ") + exc.what());
    }

    printf("%s\n", response.dump().c_str());
    return response.value("ok", false) ? 0 : 1;
}
