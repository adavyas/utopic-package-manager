// OpenAI-compatible HTTP server for the diffusion runtime (Ollama-style local serving).
// Resident model + context; serial generation (single context). Endpoints:
//   GET  /health
//   GET  /v1/models
//   POST /v1/chat/completions   (messages, stream, temperature, max_tokens, tools, response_format)
// Tool calls -> the tolerant extractor; response_format:json_schema -> schema-constrained decoding.
//   ./utopic_server -m model.gguf [--host 127.0.0.1] [--port 8910] [-ngl 99] [--ctx-size 4096]
#include "utopic_core.h"
#include "utopic_identity.h"
#include "schema_utils.h"
#include "llama.h"
#include "nlohmann/json.hpp"
#include "cpp-httplib/httplib.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <string>
#include <vector>

using namespace utopic;
using std::string;
using std::vector;
using std::pair;

using json = nlohmann::json;

static const char * arg(int argc, char ** argv, const char * flag, const char * def) {
    for (int i = 1; i < argc - 1; i++) if (!strcmp(argv[i], flag)) return argv[i + 1];
    return def;
}

// Strip a leading/trailing markdown code fence (```json ... ```) so response_format requests get a raw JSON
// body. Models often wrap structured output in fences; OpenAI clients expect the bare object. No-op if absent.
static string strip_code_fence(string s) {
    size_t b = s.find_first_not_of(" \t\r\n");
    if (b == string::npos || s.compare(b, 3, "```") != 0) return s;
    size_t nl = s.find('\n', b);                 // skip the ```[lang] line entirely
    if (nl == string::npos) return s;
    size_t close = s.rfind("```");
    if (close == string::npos || close <= nl) return s;
    string inner = s.substr(nl + 1, close - nl - 1);
    size_t e = inner.find_last_not_of(" \t\r\n");
    return e == string::npos ? inner : inner.substr(0, e + 1);
}

// Build a function-signature spec string from OpenAI `tools` for prompt injection: "name(a, b), name2(c)".
static string tools_to_spec(const json & tools) {
    string spec;
    if (!tools.is_array()) return spec;
    for (auto & t : tools) {
        if (!t.contains("function")) continue;
        const json & fn = t["function"];
        string name = fn.value("name", "");
        if (name.empty()) continue;
        if (!spec.empty()) spec += ", ";
        spec += name + "(";
        if (fn.contains("parameters") && fn["parameters"].contains("properties")) {
            bool first = true;
            for (auto & p : fn["parameters"]["properties"].items()) {
                if (!first) spec += ", ";
                first = false;
                spec += p.key();
            }
        }
        spec += ")";
    }
    return spec;
}

// Reshape the extractor's {"tool_calls":[{name, arguments}]} into OpenAI function-call format.
static nlohmann::json to_openai_tool_calls(const string & extractor_json) {
    json tc = json::array();
    try {
        json parsed = json::parse(extractor_json);
        if (parsed.contains("tool_calls")) {
            int idx = 0;
            for (auto & c : parsed["tool_calls"]) {
                json fn = { {"name", c.value("name", "")},
                            {"arguments", c.contains("arguments") ? c["arguments"].dump() : "{}"} };
                tc.push_back({ {"id", "call_" + std::to_string(idx++)}, {"type", "function"}, {"function", fn} });
            }
        }
    } catch (...) { /* leave empty -> fall back to content */ }
    return tc;
}

int main(int argc, char ** argv) {
    const char * model_path = arg(argc, argv, "-m", nullptr);
    const char * host       = arg(argc, argv, "--host", "127.0.0.1");
    const int    port       = atoi(arg(argc, argv, "--port", "8910"));
    const int    ngl        = atoi(arg(argc, argv, "-ngl", "99"));
    const int    ctx_size   = atoi(arg(argc, argv, "--ctx-size", "4096"));
    if (!model_path) { fprintf(stderr, "usage: %s -m model.gguf [--host H] [--port P] [-ngl N] [--ctx-size N]\n", argv[0]); return 1; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "utopic_server: failed to load %s\n", model_path); llama_backend_free(); return 1; }

    prepare_model_for_context(model);
    const int effective_ctx_size = std::max(ctx_size, canvas_context_tokens(256, model_canvas_length(model)));
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = effective_ctx_size;
    cp.n_batch = cp.n_ubatch = effective_ctx_size;
    const char * no_fa = getenv("UTOPIC_NO_FA") ? getenv("UTOPIC_NO_FA") : getenv("DG_NO_FA");
    if (no_fa && atoi(no_fa)) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "utopic_server: ctx init failed\n"); llama_model_free(model); llama_backend_free(); return 1; }

    string model_id = model_path;
    { size_t s = model_id.find_last_of("/\\"); if (s != string::npos) model_id = model_id.substr(s + 1); }

    std::mutex gen_mu;  // single resident context -> serialize generation
    httplib::Server svr;

    svr.Get("/health", [](const httplib::Request &, httplib::Response & res) {
        res.set_content("{\"status\":\"ok\"}", "application/json");
    });
    const int n_ctx_train = llama_model_n_ctx_train(model);  // advertise the real context window to clients
    svr.Get("/v1/models", [&](const httplib::Request &, httplib::Response & res) {
        json out = { {"object", "list"}, {"data", json::array({
            { {"id", model_id}, {"object", "model"}, {"owned_by", model_owner},
              {"context_length", n_ctx_train}, {"max_model_len", n_ctx_train} } })} };
        res.set_content(out.dump(), "application/json");
    });

    svr.Post("/v1/chat/completions", [&](const httplib::Request & hreq, httplib::Response & res) {
        json body;
        try { body = json::parse(hreq.body); }
        catch (...) { res.status = 400; res.set_content("{\"error\":\"invalid JSON body\"}", "application/json"); return; }

        // messages -> (role, content); map tool/unknown roles to user.
        vector<pair<string, string>> messages;
        if (body.contains("messages") && body["messages"].is_array()) {
            for (auto & m : body["messages"]) {
                string role = m.value("role", "user");
                if (role != "system" && role != "assistant" && role != "user") role = "user";
                string content;
                if (m.contains("content") && m["content"].is_string()) content = m["content"].get<string>();
                else if (m.contains("content")) content = m["content"].dump();
                messages.push_back({ role, content });
            }
        }
        if (messages.empty()) { res.status = 400; res.set_content("{\"error\":\"no messages\"}", "application/json"); return; }

        // augment the last user message with reasoning / tools instructions.
        string & last = messages.back().second;
        if (body.value("reasoning", false) || body.contains("reasoning_effort"))
            last = "Think step by step, then give a concise final answer.\n\n" + last;

        bool want_tools = false;
        if (body.contains("tools") && body["tools"].is_array() && !body["tools"].empty()) {
            string spec = tools_to_spec(body["tools"]);
            if (!spec.empty()) {
                last += "\n\nYou can call these functions: " + spec +
                        "\nReply ONLY with JSON: {\"calls\":[{\"name\":...,\"arguments\":{...}}]}.";
                want_tools = true;
            }
        }

        string schema;
        bool   want_json = false;  // response_format requested JSON -> strip any ```json fence from the body
        if (body.contains("response_format")) {
            const json & rf = body["response_format"];
            string rftype = rf.value("type", "");
            want_json = (rftype == "json_schema" || rftype == "json_object");
            if (rftype == "json_schema" && rf.contains("json_schema")) {
                const json & js = rf["json_schema"];
                if (js.contains("schema")) {
                    schema = schema_to_skeleton(js["schema"]);  // typed scaffold: masked path only
                    // Also steer via the prompt: the EB path (canvas models) ignores the scaffold, so the
                    // instruction is what produces JSON there; on the masked path it reinforces the scaffold.
                    last += "\n\nReply ONLY with a single JSON object matching this schema:\n" + js["schema"].dump();
                }
            }
            if (rftype == "json_object" && schema.empty())
                last += "\n\nReply ONLY with a single valid JSON object.";
        }

        request req;
        req.prompt      = apply_chat(model, messages);
        req.max_tokens  = body.value("max_tokens", 256);
        req.canvas_tokens = body.value("diffusion_canvas_tokens", 0);
        req.temperature = body.value("temperature", 0.0);
        req.seed        = body.value("seed", 0);
        req.confidence  = 0.9f;
        req.converge    = 2;
        req.eos_stop    = true;
        req.tools       = want_tools;
        req.schema      = schema;
        req.schema_hard = body.value("schema_mode", "hard") != "prompt";  // "prompt" => prompt-steer, no hard slots

        const long created = (long) time(nullptr);
        const bool stream  = body.value("stream", false);

        // capacity check (read-only; safe without the generation lock)
        if (ctx_size_for(model, req) > (int) llama_n_ctx(ctx)) {
            res.status = 400;
            res.set_content("{\"error\":\"request exceeds server --ctx-size; raise it or shorten the prompt\"}", "application/json");
            return;
        }

        if (!stream) {
            result r;
            { std::lock_guard<std::mutex> lk(gen_mu); r = generate(ctx, model, req); }
            if (!r.ok) { res.status = 500; res.set_content("{\"error\":\"generation failed\"}", "application/json"); return; }
            json tool_calls = want_tools ? to_openai_tool_calls(r.text) : json::array();
            const bool emit_tools = !tool_calls.empty();
            const string text = want_json ? strip_code_fence(r.text) : r.text;
            json msg = { {"role", "assistant"}, {"content", emit_tools ? json(nullptr) : json(text)} };
            if (!r.reasoning.empty()) msg["reasoning_content"] = r.reasoning;
            if (emit_tools) msg["tool_calls"] = tool_calls;
            json out = {
                {"id", "chatcmpl-dg"}, {"object", "chat.completion"}, {"created", created}, {"model", model_id},
                {"choices", json::array({ { {"index", 0}, {"message", msg}, {"finish_reason", emit_tools ? "tool_calls" : "stop"} } })},
                {"usage", { {"prompt_tokens", r.prompt_tokens}, {"completion_tokens", r.answer_tokens},
                            {"total_tokens", r.prompt_tokens + r.answer_tokens} }}
            };
            res.set_content(out.dump(), "application/json");
            return;
        }

        // Live SSE: generation runs INSIDE the provider; committed-canvas deltas stream as the loop denoises.
        // (Diffusion-native streaming: tokens resolve over steps, not strictly left-to-right.)
        auto ran = std::make_shared<bool>(false);
        res.set_chunked_content_provider("text/event-stream",
            [req, want_tools, created, ran, &gen_mu, &ctx, &model, &model_id](size_t, httplib::DataSink & sink) {
                if (*ran) return false;     // single-shot: generate + stream once
                *ran = true;
                auto chunk = [&](const json & delta, const json & fin) {
                    json c = { {"id", "chatcmpl-dg"}, {"object", "chat.completion.chunk"}, {"created", created},
                               {"model", model_id},
                               {"choices", json::array({ { {"index", 0}, {"delta", delta}, {"finish_reason", fin} } })} };
                    string s = "data: " + c.dump() + "\n\n";
                    return sink.write(s.data(), s.size());
                };
                std::lock_guard<std::mutex> lk(gen_mu);
                chunk({ {"role", "assistant"} }, json(nullptr));
                request sreq = req;
                if (!want_tools) {  // live-stream content as the canvas commits (masked path)
                    sreq.on_token = [&](const string & delta, int, int) {
                        return chunk({ {"content", delta} }, json(nullptr));
                    };
                }
                result r = generate(ctx, model, sreq);
                if (!r.reasoning.empty()) chunk({ {"reasoning_content", r.reasoning} }, json(nullptr));
                if (want_tools) {
                    json tool_calls = to_openai_tool_calls(r.text);
                    if (!tool_calls.empty()) { chunk({ {"tool_calls", tool_calls} }, json(nullptr)); chunk(json::object(), json("tool_calls")); }
                    else                     { chunk({ {"content", r.text} }, json(nullptr));        chunk(json::object(), json("stop")); }
                } else {
                    // emit whatever wasn't streamed live (EB path streams nothing; masked path streams a prefix).
                    string rem = r.text;
                    if (r.text.size() >= r.streamed.size() && r.text.compare(0, r.streamed.size(), r.streamed) == 0)
                        rem = r.text.substr(r.streamed.size());
                    if (!rem.empty()) chunk({ {"content", rem} }, json(nullptr));
                    chunk(json::object(), json("stop"));
                }
                string done = "data: [DONE]\n\n";
                sink.write(done.data(), done.size());
                sink.done();
                return true;
            });
    });

    fprintf(stderr, "utopic_server: %s on http://%s:%d  (ctx=%d, OpenAI /v1/chat/completions)\n",
            model_id.c_str(), host, port, ctx_size);
    svr.listen(host, port);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
