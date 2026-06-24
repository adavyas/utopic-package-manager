// utopic_core.h - shared generation core for the CLI and the OpenAI-compatible server.
// Header-only (inline) so both frontends link the exact same loop. Owns nothing of the
// forward; drives our diffusion loop over a resident llama_context.
#pragma once

#include "diffusion_driver.h"
#include "tool_extract.h"
#include "llama.h"
#include "ggml-backend.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace utopic {

inline double now_ms() {
    using namespace std::chrono;
    return duration<double, std::milli>(steady_clock::now().time_since_epoch()).count();
}

// Maximum bytes a single token renders to. Tokens are short; 256 is ample headroom.
static constexpr int kTokenPieceMaxBytes = 256;

/// Append one token's UTF-8 text to `out`. Special/control tokens render empty (special=false), which is
/// what we want for user-facing output. Appends in place to avoid a per-token heap allocation in the
/// detokenize loops (the streaming callback re-decodes the committed prefix every step).
inline void append_piece(const llama_vocab * vocab, llama_token token, std::string & out) {
    char buf[kTokenPieceMaxBytes];
    const int len = llama_token_to_piece(vocab, token, buf, sizeof(buf), /*lstrip=*/0, /*special=*/false);
    if (len > 0) {
        out.append(buf, len);
    }
}

/// Tokenize `text` to ids using the two-call (size, then fill) pattern. `add_special` controls BOS/special
/// handling. Centralizes the idiom so callers don't re-implement the resize-on-negative dance.
inline std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text,
                                            bool add_special = true) {
    std::vector<llama_token> ids(text.size() + 64);
    int n = llama_tokenize(vocab, text.c_str(), (int) text.size(), ids.data(), (int) ids.size(),
                           add_special, add_special);
    if (n < 0) {
        ids.resize(-n);
        n = llama_tokenize(vocab, text.c_str(), (int) text.size(), ids.data(), (int) ids.size(),
                           add_special, add_special);
    }
    ids.resize(n < 0 ? 0 : n);
    return ids;
}

// Per-generation step-callback state: counts steps (+ first-forward timing) and, when on_token is set,
// streams the committed canvas prefix as it fills in over denoise steps (diffusion-native streaming).
struct cb_state {
    const llama_vocab * vocab   = nullptr;
    int                 n_input = 0;
    llama_token         mask    = LLAMA_TOKEN_NULL;
    llama_token         eos     = LLAMA_TOKEN_NULL;
    std::string         sent;   // text already streamed (for OpenAI-additive deltas)
    std::function<bool(const std::string &, int, int)> on_token;  // nullptr = metrics only
    int    steps  = 0;
    double t0     = 0.0;
    double cb1_ms = -1.0;
    double cb2_ms = -1.0;
};
inline bool step_cb(int32_t step, int32_t total, const llama_token * toks, int32_t n, void * ud) {
    auto * st = (cb_state *) ud;
    st->steps++;
    const double t = now_ms() - st->t0;
    if      (st->cb1_ms < 0) st->cb1_ms = t;
    else if (st->cb2_ms < 0) st->cb2_ms = t;
    if (!st->on_token) {
        return true;
    }
    // The committed prefix is the run of resolved tokens before the first still-masked position; it grows
    // (roughly left to right) as denoising fills the canvas.
    std::string committed_prefix;
    committed_prefix.reserve((size_t) (n - st->n_input) * 4);  // ~4 bytes/token; avoid reallocs
    for (int i = st->n_input; i < n; i++) {
        const llama_token token = toks[i];
        if (token == st->mask || token == st->eos || token < 0) {
            break;
        }
        append_piece(st->vocab, token, committed_prefix);
    }
    // Emit only a pure extension of what was already sent (additive/OpenAI-safe). When a position commits
    // out of order the prefix may not extend this step, so we simply wait for it to stabilize.
    const bool extends = committed_prefix.size() > st->sent.size() &&
                         committed_prefix.compare(0, st->sent.size(), st->sent) == 0;
    if (extends) {
        const std::string delta = committed_prefix.substr(st->sent.size());
        st->sent = committed_prefix;
        if (!st->on_token(delta, step, total)) {
            return false;  // client disconnected -> stop generating
        }
    }
    return true;
}

// Tokenize a literal fragment (no special tokens, no BOS) and append to `out`.
inline void tok_append(const llama_vocab * vocab, const std::string & s, std::vector<llama_token> & out) {
    if (s.empty()) return;
    std::vector<llama_token> t(s.size() + 16);
    int n = llama_tokenize(vocab, s.c_str(), (int) s.size(), t.data(), (int) t.size(), false, false);
    if (n < 0) { t.resize(-n); n = llama_tokenize(vocab, s.c_str(), (int) s.size(), t.data(), (int) t.size(), false, false); }
    out.insert(out.end(), t.begin(), t.begin() + n);
}

// Schema-constrained scaffold (GBNF-equivalent for diffusion): typed value slots __s__/__d__/__n__ with an
// optional length suffix (__s12__). Frozen structure + per-slot character-class constraint => the emitted JSON
// parses and matches the field types by construction. Fills `classes` (parallel to the returned canvas).
inline std::vector<llama_token> build_typed_scaffold(const llama_vocab * vocab, llama_token mask,
                                                        const std::string & tmpl, int slot_len,
                                                        std::vector<uint8_t> & classes) {
    std::vector<llama_token> c;
    classes.clear();
    auto emit_lit = [&](const std::string & s) {
        tok_append(vocab, s, c);
        classes.resize(c.size(), 0);
    };
    size_t pos = 0;
    while (pos < tmpl.size()) {
        size_t m = tmpl.find("__", pos);
        if (m == std::string::npos) { emit_lit(tmpl.substr(pos)); break; }
        size_t  q   = tmpl.find("__", m + 2);
        uint8_t cls = 0;
        int     len = slot_len;
        if (q != std::string::npos && q > m + 2) {
            std::string tag = tmpl.substr(m + 2, q - (m + 2));
            char t = tag[0];
            cls = (t == 's') ? 1 : (t == 'd') ? 2 : (t == 'n') ? 3 : 0;
            if (cls && tag.size() > 1) {
                bool alldig = true;
                for (size_t i = 1; i < tag.size(); i++) { if (tag[i] < '0' || tag[i] > '9') { alldig = false; break; } }
                if (alldig) { len = atoi(tag.c_str() + 1); } else { cls = 0; }
            }
        }
        if (!cls || len <= 0) { emit_lit(tmpl.substr(pos, m + 2 - pos)); pos = m + 2; continue; }
        emit_lit(tmpl.substr(pos, m - pos));
        for (int i = 0; i < len; i++) c.push_back(mask);
        classes.resize(c.size(), cls);
        pos = q + 2;
    }
    return c;
}

inline bool slot_piece_allowed(uint8_t cls, const char * data, int n) {
    if (n <= 0) return false;
    bool ok = true;
    for (int i = 0; i < n; i++) {
        const unsigned char b = (unsigned char) data[i];
        if (cls == 1) {
            ok &= ((b >= 0x20 && b != 0x22 && b != 0x5c && b != 0x7f) || b >= 0x80);
        } else if (cls == 2 || cls == 3) {
            ok &= (b >= '0' && b <= '9');
        } else {
            return false;
        }
    }
    return ok;
}

// Per-class allow-bitmask over the vocab (3 classes; class c at (c-1)*n_words). A token is allowed in a class
// iff every byte of its rendered piece is in the class byte set; special/control tokens render empty and are
// excluded everywhere. n_words = (n_vocab + 63) / 64.
inline std::vector<uint64_t> build_class_masks(const llama_vocab * vocab, int n_vocab, int n_words) {
    std::vector<uint64_t> allow((size_t) 3 * n_words, 0);
    std::vector<char> buf(64);
    for (int v = 0; v < n_vocab; v++) {
        int n = llama_token_to_piece(vocab, v, buf.data(), (int) buf.size(), 0, false);
        if (n < 0) { buf.resize(-n); n = llama_token_to_piece(vocab, v, buf.data(), (int) buf.size(), 0, false); }
        if (n <= 0) continue;
        if (slot_piece_allowed(1, buf.data(), n)) allow[(size_t) 0 * n_words + (v >> 6)] |= (1ull << (v & 63));
        if (slot_piece_allowed(2, buf.data(), n)) allow[(size_t) 1 * n_words + (v >> 6)] |= (1ull << (v & 63));
        if (slot_piece_allowed(3, buf.data(), n)) allow[(size_t) 2 * n_words + (v >> 6)] |= (1ull << (v & 63));
    }
    return allow;
}

/// Per-vocab class-mask table, cached. The masks depend only on the vocabulary (fixed per model), so the
/// 126k-token scan in build_class_masks runs once per model instead of once per structured-output request.
/// Generation is serialized (single resident context), so the static cache needs no locking.
inline const std::vector<uint64_t> & class_masks(const llama_vocab * vocab, int n_vocab, int n_words) {
    static std::unordered_map<const llama_vocab *, std::vector<uint64_t>> cache;
    auto it = cache.find(vocab);
    if (it != cache.end()) {
        return it->second;
    }
    return cache.emplace(vocab, build_class_masks(vocab, n_vocab, n_words)).first->second;
}

// One generation request. `prompt` is the FINAL prompt text (chat template already applied by the caller).
struct request {
    std::string prompt;
    int   max_tokens   = 256;   // returned-token cap
    float temperature  = 0.0f;
    int   seed         = 0;
    // gate (masked path) - the lossless step-reduction levers
    int   steps        = 256;
    int   block_length = 32;
    int   canvas_tokens = 0;    // internal masked diffusion canvas; 0 = max(max_tokens, 512)
    float confidence   = 0.9f;  // confidence-threshold gate; <=0 disables
    int   converge     = 2;     // global convergence stop; 0 disables
    bool  eos_stop     = true;
    // entropy-bound path (DiffusionGemma)
    int   eb_steps     = 0;     // 0 = model default
    // structured output / tools
    bool  tools        = false; // harvest tool calls from output -> OpenAI tool_calls JSON
    std::string schema;         // typed JSON skeleton ("" = none)
    int   slot_len     = 8;
    bool  schema_hard  = true;  // true: hard char-class scaffold where supported (masked path) = guaranteed
                                // structure but rigid slots. false: prompt-steered only (works on every path,
                                // softer guarantee, often better content). See generate().
    // live streaming: called with each new committed text delta as the canvas denoises (masked path only).
    // Return false to abort generation (e.g. client disconnected). nullptr = no streaming.
    std::function<bool(const std::string & delta, int step, int total_steps)> on_token;
};

inline int masked_canvas_tokens(const request & req) {
    if (req.canvas_tokens > 0) return req.canvas_tokens;
    return std::max(req.max_tokens, 512);
}

inline int canvas_context_tokens(int max_tokens, long canvas_length) {
    if (max_tokens <= 0 || canvas_length <= 0) return 0;
    const int blocks = (max_tokens + (int) canvas_length - 1) / (int) canvas_length;
    return blocks * (int) canvas_length + 2048;
}

inline long model_canvas_length(llama_model * model) {
    char meta[64];
    if (llama_model_meta_val_str(model, "diffusion.canvas_length", meta, sizeof(meta)) >= 0) {
        return strtol(meta, nullptr, 10);
    }
    return 0;
}

inline bool model_arch_is(llama_model * model, const char * expected) {
    char meta[64];
    return llama_model_meta_val_str(model, "general.architecture", meta, sizeof(meta)) >= 0 &&
           strcmp(meta, expected) == 0;
}

inline void prepare_model_for_context(llama_model * model) {
    if (model_canvas_length(model) > 0) {
        llama_diffusion_set_sc(model, nullptr, 0.0f, 1.0f, true);
    }
}

inline bool single_device_canvas_accel(int gpu_like_devices) {
    return gpu_like_devices <= 1;
}

inline int gpu_like_device_count() {
    int count = 0;
    for (size_t i = 0; i < ggml_backend_dev_count(); i++) {
        const auto type = ggml_backend_dev_type(ggml_backend_dev_get(i));
        if (type == GGML_BACKEND_DEVICE_TYPE_GPU || type == GGML_BACKEND_DEVICE_TYPE_IGPU) {
            count++;
        }
    }
    return count;
}

struct result {
    std::string text;            // final answer (extracted tool_calls JSON if req.tools)
    std::string reasoning;       // thinking-channel content (DiffusionGemma); empty otherwise
    std::string raw;             // raw decoded text (pre-extract, pre-channel-split)
    std::string streamed;        // text already emitted via on_token during generation (for remainder)
    int    steps         = 0;
    int    answer_tokens = 0;
    int    prompt_tokens = 0;
    int    canvas        = 0;
    double gen_ms        = 0.0;
    double ttft_ms       = 0.0;
    bool   is_tool_call  = false;
    bool   ok            = true; // false if the request exceeds the context budget
};

// Separate DiffusionGemma's reasoning channel from the final answer. Its chat template emits
//   <|channel>thought\n<thinking>\n<channel|><final answer>
// so the answer is whatever follows the last <channel|>, and the thinking is between the markers.
// No-op for models that emit neither marker -> {reasoning:"", final:raw}.
struct channels { std::string reasoning; std::string final; };
inline channels split_channels(const std::string & raw) {
    static const std::string OPEN  = "<|channel>";
    static const std::string CLOSE = "<channel|>";
    auto trim = [](const std::string & s) {
        size_t a = s.find_first_not_of(" \t\r\n");
        if (a == std::string::npos) return std::string();
        size_t b = s.find_last_not_of(" \t\r\n");
        return s.substr(a, b - a + 1);
    };
    auto strip_label = [](std::string s) { return s.rfind("thought", 0) == 0 ? s.substr(7) : s; };
    channels out;
    size_t close = raw.rfind(CLOSE);
    if (close != std::string::npos) {
        // full form: <|channel>thought <thinking> <channel|> <answer>
        out.final = raw.substr(close + CLOSE.size());
        std::string pre = raw.substr(0, close);
        size_t op = pre.find(OPEN);
        out.reasoning = trim(strip_label(op == std::string::npos ? pre : pre.substr(op + OPEN.size())));
    } else {
        // model opened the thought channel but never closed it: strip the marker, keep the text as the answer.
        std::string s = raw;
        size_t op = s.find(OPEN);
        if (op != std::string::npos) s = strip_label(s.substr(op + OPEN.size()));
        out.final = s;
    }
    // defensive: remove any stray markers left anywhere in the answer.
    for (const std::string * m : { &OPEN, &CLOSE }) {
        size_t p;
        while ((p = out.final.find(*m)) != std::string::npos) out.final.erase(p, m->size());
    }
    out.final = trim(out.final);
    return out;
}

inline std::string trim_copy(const std::string & s) {
    size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return std::string();
    size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// Apply the model's chat template to (role, content) pairs. Falls back to concatenated content.
inline std::string apply_diffusion_gemma_chat(
        const std::vector<std::pair<std::string, std::string>> & messages) {
    std::string system;
    std::string out = "<|turn>system\n<|think|>\n";
    for (const auto & m : messages) {
        if (m.first == "system") {
            if (!system.empty()) system += "\n";
            system += m.second;
        }
    }
    if (!system.empty()) out += system;
    out += "<turn|>\n";

    for (const auto & m : messages) {
        if (m.first == "system") continue;
        const char * role = (m.first == "assistant") ? "model" : "user";
        out += "<|turn>";
        out += role;
        out += "\n";
        out += m.second;
        out += "<turn|>\n";
    }
    out += "<|turn>model\n";
    return out;
}

inline std::string apply_chat(llama_model * model,
                                 const std::vector<std::pair<std::string, std::string>> & messages) {
    const char * tmpl = llama_model_chat_template(model, nullptr);
    std::vector<llama_chat_message> lm;
    lm.reserve(messages.size());
    for (auto & m : messages) lm.push_back({ m.first.c_str(), m.second.c_str() });
    if (tmpl && !lm.empty()) {
        std::vector<char> fbuf(8192);
        int flen = llama_chat_apply_template(tmpl, lm.data(), lm.size(), true, fbuf.data(), (int) fbuf.size());
        if (flen > (int) fbuf.size()) { fbuf.resize(flen); flen = llama_chat_apply_template(tmpl, lm.data(), lm.size(), true, fbuf.data(), (int) fbuf.size()); }
        if (flen > 0) return std::string(fbuf.data(), flen);
    }
    if (model_arch_is(model, "diffusion-gemma")) {
        return apply_diffusion_gemma_chat(messages);
    }
    std::string out;
    for (size_t i = 0; i < messages.size(); i++) {
        if (i) out += "\n";
        out += messages[i].second;
    }
    return out;
}

// Compute the context capacity a request needs: prompt tokens + canvas (schema scaffold / canvas_length /
// internal masked canvas) + slack. The CLI sizes its one-shot context with this; the server sizes once to a
// configured max.
inline int ctx_size_for(llama_model * model, const request & req) {
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_input = (int) tokenize(vocab, req.prompt).size();
    const long canvas_length = model_canvas_length(model);
    int canvas;
    if (!req.schema.empty() && canvas_length == 0 && req.schema_hard) {  // typed scaffold: masked path only (see generate())
        std::vector<uint8_t> cls;
        canvas = (int) build_typed_scaffold(vocab, llama_vocab_mask(vocab), req.schema, req.slot_len, cls).size();
    } else if (canvas_length > 0) {
        canvas = (int) canvas_length;
    } else {
        canvas = masked_canvas_tokens(req);
    }
    int needed = n_input + canvas + 8;
    if (canvas_length > 0) {
        needed = std::max(needed, canvas_context_tokens(req.max_tokens, canvas_length));
    }
    return needed;
}

// Run one diffusion generation over a RESIDENT context. Clears context memory first so requests are isolated.
// Selects the entropy-bound path for canvas models (DiffusionGemma) and the masked path for non-canvas diffusion models.
inline result generate(llama_context * ctx, llama_model * model, const request & req) {
    result R;
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int           n_vocab = llama_vocab_n_tokens(vocab);
    const llama_token   mask = llama_vocab_mask(vocab);
    const llama_token   eos  = llama_vocab_eos(vocab);
    assert(ctx && model && vocab && "generate: null ctx/model/vocab");

    char meta[64];
    const long canvas_length = model_canvas_length(model);
    bool shift_logits = (canvas_length == 0);
    if (llama_model_meta_val_str(model, "diffusion.shift_logits", meta, sizeof(meta)) >= 0)
        shift_logits = (strcmp(meta, "true") == 0);

    const std::vector<llama_token> inp = tokenize(vocab, req.prompt);
    const int n_input = (int) inp.size();

    std::vector<llama_token>       scaffold;
    std::vector<uint8_t>          slot_class;
    const std::vector<uint64_t> * class_allow = nullptr;  // points into the cached per-vocab table (no copy)
    int                           cls_words   = 0;
    // The typed scaffold + per-slot class masking is a masked-path feature: it rides on
    // diffusion_generate's canvas_template / slot_class hooks. The entropy-bound kernel (canvas models like
    // DiffusionGemma) has no such hooks, so a scaffold there would size an odd canvas the EB pass then fills
    // unconstrained -> garbage. On the EB path we skip the scaffold and let the prompt steer the JSON instead.
    // schema_hard=false opts out of the scaffold entirely (prompt-steered everywhere) - softer guarantee, but
    // it sidesteps the rigid fixed-width slots that can force poor content (e.g. dates into a numeric field).
    if (!req.schema.empty() && canvas_length == 0 && req.schema_hard) {
        scaffold    = build_typed_scaffold(vocab, mask, req.schema, req.slot_len, slot_class);
        cls_words   = (n_vocab + 63) / 64;
        class_allow = &class_masks(vocab, n_vocab, cls_words);
    }
    const int canvas     = !scaffold.empty() ? (int) scaffold.size()
                         : (canvas_length > 0 ? (int) canvas_length : masked_canvas_tokens(req));
    const int max_length = n_input + canvas;
    R.prompt_tokens = n_input;
    R.canvas        = canvas;

    if (max_length + 8 > (int) llama_n_ctx(ctx)) { R.ok = false; return R; }  // server validates up-front
    if (canvas_length == 0) {
        llama_memory_clear(llama_get_memory(ctx), true);                      // isolate requests on the resident ctx
    }

    std::vector<llama_token> out(max_length, mask);
    int n_generated = 0;
    cb_state st;
    st.vocab = vocab; st.n_input = n_input; st.mask = mask; st.eos = eos;
    st.on_token = (canvas_length == 0) ? req.on_token : nullptr;  // live streaming: masked path only
    const double t0 = now_ms();
    st.t0 = t0;

    if (canvas_length > 0) {
        diffusion_eb_params eb{};
        eb.max_length    = max_length;
        eb.seed          = req.seed;
        if (req.eb_steps > 0) eb.max_denoising_steps = req.eb_steps;
        const int gpu_devs = gpu_like_device_count();
        eb.kv_cache          = single_device_canvas_accel(gpu_devs);
        eb.gpu_sampling      = single_device_canvas_accel(gpu_devs);
        eb.gpu_sample_reduce = single_device_canvas_accel(gpu_devs) && gpu_devs == 1;
        eb.step_callback           = step_cb;
        eb.step_callback_user_data = &st;
        diffusion_generate_entropy_bound(ctx, inp.data(), out.data(), n_input, eb, n_generated);
    } else {
        diffusion_params dp{};
        dp.steps                = req.steps;
        dp.temperature          = req.temperature;
        dp.mask_token_id        = mask;
        dp.seed                 = req.seed;
        dp.algorithm            = DIFFUSION_ALGORITHM_CONFIDENCE_BASED;
        dp.max_length           = max_length;
        dp.shift_logits         = shift_logits;
        dp.suppress_mask_token  = true;
        dp.confidence_threshold = req.confidence;
        dp.converge_stop        = req.converge;
        dp.eos_token_id         = req.eos_stop ? eos : LLAMA_TOKEN_NULL;
        if (!scaffold.empty()) dp.canvas_template = scaffold.data();
        if (!slot_class.empty() && class_allow) {
            dp.slot_class  = slot_class.data();
            dp.class_allow = class_allow->data();
            dp.n_class     = 3;
            dp.n_words     = cls_words;
        }
        dp.step_callback           = step_cb;
        dp.step_callback_user_data = &st;
        if (req.block_length > 0) { dp.schedule = DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED; dp.block_length = req.block_length; }
        else                      { dp.schedule = DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED; dp.eps = 1e-3f; }
        diffusion_generate(ctx, inp.data(), out.data(), n_input, dp, n_generated);
    }
    R.gen_ms   = now_ms() - t0;
    R.steps    = st.steps;
    R.streamed = st.sent;
    double first = (canvas_length > 0) ? st.cb1_ms : st.cb2_ms;
    if (first < 0) first = st.cb1_ms;
    if (first < 0) first = R.gen_ms;
    R.ttft_ms = first;

    // Detokenize the canvas into the answer text: stop at EOS, skip mask / out-of-range padding.
    std::string answer_text;
    answer_text.reserve((size_t) (max_length - n_input) * 4);  // ~4 bytes/token; avoid reallocs
    int answer_token_count = 0;
    for (int i = n_input; i < max_length; i++) {
        const llama_token token = out[i];
        if (token == eos) {
            break;
        }
        if (token < 0 || token >= n_vocab || token == mask) {
            continue;
        }
        if (canvas_length == 0 && answer_token_count >= req.max_tokens) {
            break;
        }
        answer_token_count++;
        append_piece(vocab, token, answer_text);
    }
    if (canvas_length > 0) {
        int first_eos = -1;
        for (int j = 0; j < canvas; j++) {
            if (out[n_input + j] == eos) {
                first_eos = j;
                break;
            }
        }
        if (first_eos >= 0 && first_eos <= 2) {
            std::string recovered_text;
            recovered_text.reserve(answer_text.capacity());
            int  recovered_count = 0;
            int  eos_run = 0;
            bool saw_content = false;
            for (int j = first_eos + 1; j < canvas; j++) {
                const llama_token token = out[n_input + j];
                if (token == eos) {
                    eos_run++;
                    if (saw_content && eos_run >= 4) {
                        break;
                    }
                    continue;
                }
                eos_run = 0;
                if (token < 0 || token >= n_vocab || token == mask) {
                    continue;
                }
                const size_t before = recovered_text.size();
                append_piece(vocab, token, recovered_text);
                if (recovered_text.size() > before) {
                    saw_content = true;
                }
                recovered_count++;
            }
            if (trim_copy(recovered_text).size() > trim_copy(answer_text).size()) {
                answer_text = recovered_text;
                answer_token_count = recovered_count;
            }
        }
    }
    R.answer_tokens = answer_token_count;
    R.raw           = answer_text;
    const channels ch = split_channels(answer_text);  // separate DiffusionGemma's reasoning channel
    R.reasoning          = ch.reasoning;
    if (req.tools) { R.text = toolx::to_openai_json(toolx::extract(ch.final)); R.is_tool_call = true; }
    else           { R.text = ch.final; }
    return R;
}

}  // namespace utopic
