// MCP (Model Context Protocol) stdio server over the diffusion runtime.
// Exposes a `diffusion_generate` tool so an MCP client (Claude Code, Codex, ...) can delegate bounded
// subtasks to a local diffusion LLM. JSON-RPC 2.0, newline-delimited over stdin/stdout; logs go to stderr.
// Model is loaded lazily on the first tool call and kept resident.
//   ./utopic_mcp -m model.gguf [-ngl 99] [--ctx-size 2048] [--eb-steps 24]
#include "utopic_core.h"
#include "utopic_identity.h"
#include "llama.h"
#include "nlohmann/json.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <thread>

using json = nlohmann::json;
using namespace utopic;
using std::string;

static const char * arg(int argc, char ** argv, const char * flag, const char * def) {
    for (int i = 1; i < argc - 1; i++) if (!strcmp(argv[i], flag)) return argv[i + 1];
    return def;
}

// Resident model state, loaded lazily so the MCP handshake answers instantly (the model is large).
static llama_model *   g_model     = nullptr;
static llama_context * g_ctx       = nullptr;
static string          g_model_path;
static int             g_ngl       = 99;
static int             g_ctx_size  = 2048;
static int             g_eb_steps  = 24;

static std::mutex g_load_mu;
static bool ensure_loaded() {
    std::lock_guard<std::mutex> lk(g_load_mu);  // background preload + first tool call may race
    if (g_ctx) return true;
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = g_ngl;
    g_model = llama_model_load_from_file(g_model_path.c_str(), mp);
    if (!g_model) return false;
    prepare_model_for_context(g_model);
    const int effective_ctx_size = std::max(g_ctx_size, canvas_context_tokens(256, model_canvas_length(g_model)));
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = cp.n_batch = cp.n_ubatch = effective_ctx_size;
    g_ctx = llama_init_from_model(g_model, cp);
    return g_ctx != nullptr;
}

static string run_generate(const string & prompt, int max_tokens) {
    if (!ensure_loaded()) return "error: failed to load model";
    request req;
    req.prompt     = apply_chat(g_model, { { "user", prompt } });
    req.max_tokens = max_tokens;
    req.confidence = 0.9f;
    req.converge   = 2;
    req.eos_stop   = true;
    req.eb_steps   = g_eb_steps;   // entropy-bound budget (DiffusionGemma); ignored by masked models
    if (ctx_size_for(g_model, req) > (int) llama_n_ctx(g_ctx)) {
        return "error: prompt exceeds server --ctx-size; raise it or shorten the prompt";
    }
    result r = generate(g_ctx, g_model, req);
    return r.text;
}

static const json TOOL_SCHEMA = {
    { "name", "diffusion_generate" },
    { "description",
      "Generate text with a local diffusion LLM (DiffusionGemma). Fast, local, runs offline. Best for "
      "bounded subtasks you want to offload from the main agent: drafting, summarizing, extraction, "
      "classification, quick code/text completion. Returns the model's completion." },
    { "inputSchema", {
        { "type", "object" },
        { "properties", {
            { "prompt",     { { "type", "string" },  { "description", "The prompt to complete." } } },
            { "max_tokens", { { "type", "integer" }, { "description", "Max output tokens (default 256)." } } } } },
        { "required", json::array({ "prompt" }) } } }
};

static void send(const json & resp) {
    const string out = resp.dump() + "\n";
    fputs(out.c_str(), stdout);
    fflush(stdout);
}

int main(int argc, char ** argv) {
    const char * model_path = arg(argc, argv, "-m", nullptr);
    if (!model_path) { fprintf(stderr, "usage: %s -m model.gguf [-ngl 99] [--ctx-size 2048] [--eb-steps 24]\n", argv[0]); return 1; }
    g_model_path = model_path;
    g_ngl       = atoi(arg(argc, argv, "-ngl", "99"));
    g_ctx_size  = atoi(arg(argc, argv, "--ctx-size", "2048"));
    g_eb_steps  = atoi(arg(argc, argv, "--eb-steps", "24"));

    // Route all llama/ggml logging to stderr so stdout carries only JSON-RPC.
    llama_log_set([](ggml_log_level, const char * text, void *) { fputs(text, stderr); }, nullptr);
    llama_backend_init();
    // Preload the model in the background so the first tool call is fast (within MCP clients' tool timeouts),
    // while the initialize/tools-list handshake still answers instantly on the main thread.
    std::thread([] { ensure_loaded(); }).detach();
    fprintf(stderr, "utopic_mcp: ready (model preloading in background): %s\n", model_path);

    char *  buf = nullptr;
    size_t  cap = 0;
    ssize_t len;
    while ((len = getline(&buf, &cap, stdin)) != -1) {
        if (len <= 1) continue;  // blank line
        json msg;
        try { msg = json::parse(string(buf, (size_t) len)); }
        catch (...) { continue; }

        const string method  = msg.value("method", "");
        const bool   has_id  = msg.contains("id") && !msg["id"].is_null();
        if (method == "notifications/initialized" || method.rfind("notifications/", 0) == 0) continue;

        json resp = { { "jsonrpc", "2.0" } };
        if (has_id) resp["id"] = msg["id"];

        if (method == "initialize") {
            resp["result"] = {
                { "protocolVersion", "2024-11-05" },
                { "capabilities", { { "tools", json::object() } } },
                { "serverInfo", { { "name", server_name }, { "version", project_version } } }
            };
        } else if (method == "tools/list") {
            resp["result"] = { { "tools", json::array({ TOOL_SCHEMA }) } };
        } else if (method == "tools/call") {
            const json args   = msg.contains("params") ? msg["params"].value("arguments", json::object()) : json::object();
            const string name = msg.contains("params") ? msg["params"].value("name", "") : "";
            if (name != "diffusion_generate") {
                resp["error"] = { { "code", -32602 }, { "message", "unknown tool: " + name } };
            } else {
                const string prompt = args.value("prompt", "");
                const int    mt     = args.value("max_tokens", 256);
                const string text   = run_generate(prompt, mt);
                resp["result"] = { { "content", json::array({ { { "type", "text" }, { "text", text } } }) } };
            }
        } else if (method == "ping") {
            resp["result"] = json::object();
        } else if (has_id) {
            resp["error"] = { { "code", -32601 }, { "message", "method not found: " + method } };
        } else {
            continue;
        }
        if (has_id) send(resp);
    }
    free(buf);
    if (g_ctx)   llama_free(g_ctx);
    if (g_model) llama_model_free(g_model);
    llama_backend_free();
    return 0;
}
