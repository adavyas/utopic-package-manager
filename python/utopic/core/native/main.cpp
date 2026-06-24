// Minimal CLI for the diffusion runtime - PUBLIC llama.h only (no common).
// Loads a GGUF via libllama, drives our diffusion loop over a one-shot context.
//   ./utopic -m model.gguf -p "prompt" [-n 256] [--temp 0] [--seed 0] [-ngl 99]
//        [--system "..."] [--tools "fn(a,b), fn2(c)"] [--schema '{"k":"__s__"}'] [--soft-schema] [--reasoning]
//        [--confidence 0.9] [--converge 2] [--steps 256] [--diffusion-block-length 32] [--canvas N] [--eb-steps N]
// Canvas models (DiffusionGemma) -> entropy-bound path; non-canvas masked models -> masked path.
#include "utopic_core.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace utopic;
using std::string;
using std::vector;
using std::pair;

static const char * arg(int argc, char ** argv, const char * flag, const char * def) {
    for (int i = 1; i < argc - 1; i++) if (!strcmp(argv[i], flag)) return argv[i + 1];
    return def;
}
static bool flag_set(int argc, char ** argv, const char * flag) {
    for (int i = 1; i < argc; i++) if (!strcmp(argv[i], flag)) return true;
    return false;
}
static const char * env_any(const char * preferred, const char * legacy = nullptr) {
    const char * v = preferred ? getenv(preferred) : nullptr;
    if (v) return v;
    return legacy ? getenv(legacy) : nullptr;
}
// flag wins over env wins over default, for the gate knobs (keeps eval-script env overrides working).
static float fopt(int argc, char ** argv, const char * flag, const char * env, const char * legacy_env, float def) {
    const char * f = arg(argc, argv, flag, nullptr);
    if (f) return (float) atof(f);
    const char * e = env_any(env, legacy_env);
    if (e) return (float) atof(e);
    return def;
}
static int iopt(int argc, char ** argv, const char * flag, const char * env, const char * legacy_env, int def) {
    const char * f = arg(argc, argv, flag, nullptr);
    if (f) return atoi(f);
    const char * e = env_any(env, legacy_env);
    if (e) return atoi(e);
    return def;
}

int main(int argc, char ** argv) {
    const char * model_path = arg(argc, argv, "-m", nullptr);
    const char * prompt     = arg(argc, argv, "-p", "What is the capital of Japan? One word.");
    const char * system     = arg(argc, argv, "--system", nullptr);
    const char * tools_spec  = arg(argc, argv, "--tools", nullptr);
    const char * schema      = arg(argc, argv, "--schema", env_any("UTOPIC_SCHEMA", "DG_SCHEMA"));
    const bool   reasoning   = flag_set(argc, argv, "--reasoning");
    const int    ngl         = iopt(argc, argv, "-ngl", nullptr, nullptr, 99);
    if (!model_path) { fprintf(stderr, "usage: %s -m model.gguf -p prompt [--tools ...] [--schema ...] [--reasoning]\n", argv[0]); return 1; }

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = ngl;
    llama_model * model = llama_model_load_from_file(model_path, mp);
    if (!model) { fprintf(stderr, "utopic: failed to load %s\n", model_path); llama_backend_free(); return 1; }

    // Build the chat-templated prompt: system (optional) + user (with reasoning / tools augmentation).
    string user = prompt;
    if (reasoning) user = "Think step by step, then give a concise final answer.\n\n" + user;
    if (tools_spec) {
        user += "\n\nYou can call these functions: ";
        user += tools_spec;
        user += "\nReply ONLY with JSON: {\"calls\":[{\"name\":...,\"arguments\":{...}}]}.";
    }
    vector<pair<string, string>> messages;
    if (system) messages.push_back({ "system", system });
    messages.push_back({ "user", user });

    request req;
    req.prompt       = apply_chat(model, messages);
    req.max_tokens   = iopt(argc, argv, "-n", nullptr, nullptr, 256);
    req.temperature  = fopt(argc, argv, "--temp", nullptr, nullptr, 0.0f);
    req.seed         = iopt(argc, argv, "--seed", "UTOPIC_SEED", "DG_SEED", 0);
    req.steps        = iopt(argc, argv, "--steps", nullptr, nullptr, 256);
    req.block_length = iopt(argc, argv, "--diffusion-block-length", "UTOPIC_BLOCK", "DG_BLOCK", 32);
    req.canvas_tokens = iopt(argc, argv, "--canvas", "UTOPIC_CANVAS", "DG_CANVAS", 0);
    req.confidence   = fopt(argc, argv, "--confidence", "UTOPIC_CONF", "DG_CONF", 0.9f);     // gate on by default (shipped fast config)
    req.converge     = iopt(argc, argv, "--converge", "UTOPIC_CONVERGE", "DG_CONVERGE", 2);
    const char * eos_stop = env_any("UTOPIC_EOS_STOP", "DG_EOS_STOP");
    req.eos_stop     = !(eos_stop && !atoi(eos_stop));
    req.eb_steps     = iopt(argc, argv, "--eb-steps", "UTOPIC_EB_STEPS", "DG_EB_STEPS", 0);
    req.tools        = (tools_spec != nullptr);
    req.schema       = schema ? schema : "";
    req.slot_len     = iopt(argc, argv, "--slot-len", "UTOPIC_SLOT_LEN", "DG_SLOT_LEN", 8);
    req.schema_hard  = !flag_set(argc, argv, "--soft-schema");  // --soft-schema: prompt-steer instead of hard slots

    prepare_model_for_context(model);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx   = ctx_size_for(model, req);
    cp.n_batch = cp.n_ubatch = cp.n_ctx;
    const char * no_fa = env_any("UTOPIC_NO_FA", "DG_NO_FA");
    if (no_fa && atoi(no_fa)) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "utopic: ctx init failed\n"); llama_model_free(model); llama_backend_free(); return 1; }

    const double t_start_total = now_ms();
    result r = generate(ctx, model, req);
    const double total_s = (now_ms() - t_start_total) / 1000.0;

    if (!r.reasoning.empty()) fprintf(stderr, "[reasoning] %s\n", r.reasoning.c_str());
    printf("%s\n", r.text.c_str());

    const double gen_s = r.gen_ms / 1000.0;
    fprintf(stderr,
            "[bench] in-step=%.0f tok/s  output=%.1f tok/s  first-token=%.2fs  denoise=%d steps  "
            "canvas=%d  answer=%d tok  prompt=%d tok  gen=%.2fs  total=%.2fs  ms/step=%.2f\n",
            gen_s > 0 ? (double) r.steps * r.canvas / gen_s : 0.0,
            gen_s > 0 ? r.answer_tokens / gen_s : 0.0,
            r.ttft_ms / 1000.0, r.steps, r.canvas, r.answer_tokens, r.prompt_tokens,
            gen_s, total_s, r.steps ? r.gen_ms / r.steps : 0.0);

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
