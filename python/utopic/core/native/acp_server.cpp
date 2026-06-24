// ACP (Agent Client Protocol) stdio agent over the diffusion runtime. Lets an ACP-speaking editor
// (e.g. Zed) drive a local diffusion LLM as a coding agent. JSON-RPC 2.0, newline-delimited over
// stdin/stdout; logs to stderr. Model loads lazily on the first prompt and stays resident.
//   ./utopic_acp -m model.gguf [-ngl 99] [--ctx-size 2048] [--eb-steps 24]
#include "utopic_core.h"
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

static llama_model *   g_model    = nullptr;
static llama_context * g_ctx      = nullptr;
static string          g_model_path;
static int             g_ngl      = 99;
static int             g_ctx_size = 2048;
static int             g_eb_steps = 24;
static int             g_session  = 0;

static std::mutex g_load_mu;
static bool ensure_loaded() {
    std::lock_guard<std::mutex> lk(g_load_mu);
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

static void send(const json & msg) {
    const string out = msg.dump() + "\n";
    fputs(out.c_str(), stdout);
    fflush(stdout);
}

// Stream one assistant text delta to the client as a session/update notification.
static void send_chunk(const string & session_id, const string & text) {
    send({ { "jsonrpc", "2.0" }, { "method", "session/update" },
           { "params", { { "sessionId", session_id },
                         { "update", { { "sessionUpdate", "agent_message_chunk" },
                                       { "content", { { "type", "text" }, { "text", text } } } } } } } });
}

static string extract_prompt_text(const json & blocks) {
    string text;
    if (!blocks.is_array()) return text;
    for (const auto & b : blocks) {
        if (b.value("type", "") == "text") text += b.value("text", "");
    }
    return text;
}

int main(int argc, char ** argv) {
    const char * model_path = arg(argc, argv, "-m", nullptr);
    if (!model_path) { fprintf(stderr, "usage: %s -m model.gguf [-ngl 99] [--ctx-size 2048] [--eb-steps 24]\n", argv[0]); return 1; }
    g_model_path = model_path;
    g_ngl      = atoi(arg(argc, argv, "-ngl", "99"));
    g_ctx_size = atoi(arg(argc, argv, "--ctx-size", "2048"));
    g_eb_steps = atoi(arg(argc, argv, "--eb-steps", "24"));

    llama_log_set([](ggml_log_level, const char * text, void *) { fputs(text, stderr); }, nullptr);
    llama_backend_init();
    std::thread([] { ensure_loaded(); }).detach();  // background preload
    fprintf(stderr, "utopic_acp: ready (model preloading in background): %s\n", model_path);

    char *  buf = nullptr;
    size_t  cap = 0;
    ssize_t len;
    while ((len = getline(&buf, &cap, stdin)) != -1) {
        if (len <= 1) continue;
        json msg;
        try { msg = json::parse(string(buf, (size_t) len)); }
        catch (...) { continue; }

        const string method = msg.value("method", "");
        const bool   has_id = msg.contains("id") && !msg["id"].is_null();
        json resp = { { "jsonrpc", "2.0" } };
        if (has_id) resp["id"] = msg["id"];

        if (method == "initialize") {
            resp["result"] = {
                { "protocolVersion", 1 },
                { "agentCapabilities", { { "loadSession", false },
                                         { "promptCapabilities", { { "image", false }, { "audio", false }, { "embeddedContext", false } } } } },
                { "authMethods", json::array() }
            };
        } else if (method == "session/new") {
            resp["result"] = { { "sessionId", "sess-" + std::to_string(++g_session) } };
        } else if (method == "session/prompt") {
            const json params      = msg.value("params", json::object());
            const string sessionId = params.value("sessionId", "sess-1");
            const string text      = extract_prompt_text(params.value("prompt", json::array()));

            if (!ensure_loaded()) {
                send_chunk(sessionId, "error: failed to load model");
                resp["result"] = { { "stopReason", "end_turn" } };
            } else {
                request req;
                req.prompt     = apply_chat(g_model, { { "user", text } });
                req.confidence = 0.9f;
                req.converge   = 2;
                req.eos_stop   = true;
                req.eb_steps   = g_eb_steps;
                req.on_token   = [&](const string & delta, int, int) { send_chunk(sessionId, delta); return true; };
                if (ctx_size_for(g_model, req) > (int) llama_n_ctx(g_ctx)) {
                    send_chunk(sessionId, "error: prompt exceeds agent --ctx-size");
                } else {
                    result r = generate(g_ctx, g_model, req);
                    // emit whatever wasn't live-streamed (EB path streams nothing; masked streams a prefix)
                    string rem = r.text;
                    if (r.text.size() >= r.streamed.size() && r.text.compare(0, r.streamed.size(), r.streamed) == 0)
                        rem = r.text.substr(r.streamed.size());
                    if (!rem.empty()) send_chunk(sessionId, rem);
                }
                resp["result"] = { { "stopReason", "end_turn" } };
            }
        } else if (method == "session/cancel" || method.rfind("notifications/", 0) == 0) {
            continue;  // notification, no response (cancel not yet wired into the loop)
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
