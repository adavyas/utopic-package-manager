#include "diffusion_driver.h"

// minimal logging - avoid the common/ dependency so the driver links against public libllama only.
#include <cstdio>
#ifndef LOG_ERR
#define LOG_ERR(...) fprintf(stderr, __VA_ARGS__)
#define LOG_INF(...) fprintf(stderr, __VA_ARGS__)
#define LOG_WRN(...) fprintf(stderr, __VA_ARGS__)
#endif

#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cstring>
#include <numeric>
#include <random>
#include <thread>
#include <utility>
#include <vector>

namespace utopic {

using std::string;
using std::vector;
using std::pair;

static void utopic_llama_diffusion_set_phase(llama_model * model, int phase, int32_t p) {
#ifdef UTOPIC_LLAMA_PHASE_HAS_OFFSET
    llama_diffusion_set_phase(model, phase, p, 0);
#else
    llama_diffusion_set_phase(model, phase, p);
#endif
}

static float calculate_confidence(const llama_token_data_array & cur_p,
                                  diffusion_algorithm            algorithm,
                                  std::mt19937 &                 rng) {
    switch (algorithm) {
        case DIFFUSION_ALGORITHM_CONFIDENCE_BASED:
            return cur_p.data[cur_p.selected].p;  // Selected token probability

        case DIFFUSION_ALGORITHM_ENTROPY_BASED:
            {
                float       entropy = 0.0f;
                const float epsilon = 1e-10f;
                for (size_t i = 0; i < cur_p.size; i++) {
                    float prob = cur_p.data[i].p;
                    entropy += prob * logf(prob + epsilon);
                }
                return -entropy;  // Higher entropy = lower confidence
            }

        case DIFFUSION_ALGORITHM_MARGIN_BASED:
            return (cur_p.size > 1) ? cur_p.data[0].p - cur_p.data[1].p : cur_p.data[0].p;

        case DIFFUSION_ALGORITHM_RANDOM:
            {
                std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
                return uniform(rng);  // Random confidence
            }

        case DIFFUSION_ALGORITHM_ORIGIN:
            return cur_p.data[cur_p.selected].p;

        default:
            return 0.0f;
    }
}

static bool token_has_visible_piece(const llama_vocab * vocab, llama_token token) {
    char buf[256];
    int n = llama_token_to_piece(vocab, token, buf, (int) sizeof(buf), 0, false);
    vector<char> dyn;
    const char * data = buf;
    if (n < 0) {
        dyn.resize((size_t) -n);
        n = llama_token_to_piece(vocab, token, dyn.data(), (int) dyn.size(), 0, false);
        data = dyn.data();
    }
    if (n <= 0) return false;
    for (int i = 0; i < n; i++) {
        const unsigned char b = (unsigned char) data[i];
        if (b > 0x20 && b != 0x7f) return true;
    }
    return false;
}

static bool token_is_channel_marker_piece(const llama_vocab * vocab, llama_token token) {
    char buf[256];
    int n = llama_token_to_piece(vocab, token, buf, (int) sizeof(buf), 0, false);
    vector<char> dyn;
    const char * data = buf;
    if (n < 0) {
        dyn.resize((size_t) -n);
        n = llama_token_to_piece(vocab, token, dyn.data(), (int) dyn.size(), 0, false);
        data = dyn.data();
    }
    if (n <= 0) return false;
    const string piece(data, (size_t) n);
    return piece.find("<|channel>") != string::npos || piece.find("<channel|>") != string::npos;
}

// Unified transfer count calculation function
static int32_t calculate_transfer_count(int32_t                      step,
                                        int32_t                      total_steps,
                                        int32_t                      remaining_masked,
                                        diffusion_transfer_schedule  schedule,
                                        float                        eps,
                                        const vector<int32_t> & num_transfer_tokens = {}) {
    switch (schedule) {
        case DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED:
            {
                float t          = 1.0f - (float) step / total_steps * (1.0f - eps);
                float s          = 1.0f - (float) (step + 1) / total_steps * (1.0f - eps);
                float p_transfer = (step < total_steps - 1) ? (1.0f - s / t) : 1.0f;
                return (int32_t) (remaining_masked * p_transfer);
            }

        case DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED:
            if (!num_transfer_tokens.empty() && step < (int32_t) num_transfer_tokens.size()) {
                return num_transfer_tokens[step];
            }
            return remaining_masked / (total_steps - step);  // Fallback

        default:
            return remaining_masked / (total_steps - step);
    }
}

static void add_gumbel_noise(float * logits, int32_t n_vocab, float temperature, std::mt19937 & rng) {
    if (temperature == 0.0f) {
        return;
    }

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    for (int32_t i = 0; i < n_vocab; i++) {
        double noise        = uniform(rng);
        // Prevent log(0)
        noise               = std::max(noise, 1e-20);
        double gumbel_noise = std::pow(-std::log(noise), temperature);
        logits[i]           = std::exp(logits[i]) / gumbel_noise;
    }
}

static vector<int32_t> get_num_transfer_tokens(int32_t mask_count, int32_t steps) {
    vector<int32_t> num_transfer_tokens(steps);

    int32_t base      = mask_count / steps;
    int32_t remainder = mask_count % steps;

    for (int32_t i = 0; i < steps; i++) {
        num_transfer_tokens[i] = base + (i < remainder ? 1 : 0);
    }

    return num_transfer_tokens;
}

void diffusion_generate(llama_context *          ctx,
                        const llama_token *      input_tokens,
                        llama_token *            output_tokens,
                        int32_t                  n_input,
                        const diffusion_params & params,
                        int32_t &                n_generated) {
    n_generated = 0;
    if (!ctx || !input_tokens || !output_tokens || n_input <= 0 || params.max_length <= n_input) {
        return;
    }

    const llama_model * model = llama_get_model(ctx);

    // Initialize with input and pad with mask tokens (or an S3 scaffold template: frozen structure + value-slot masks).
    std::copy(input_tokens, input_tokens + n_input, output_tokens);
    if (params.canvas_template) {
        std::copy(params.canvas_template, params.canvas_template + (params.max_length - n_input), output_tokens + n_input);
    } else {
        std::fill(output_tokens + n_input, output_tokens + params.max_length, params.mask_token_id);
    }

    std::mt19937 rng(params.seed);

    llama_set_causal_attn(ctx, false);

    int32_t n_vocab = llama_vocab_n_tokens(llama_model_get_vocab(model));

    // Canvas-logits-only (P2): request logits for the rows we actually sample (canvas, plus row n_input-1 when
    // shift_logits) instead of all max_length rows - trims the lm_head projection + logits D2H. Attention is
    // unchanged. Valid only in the masked-absorbing regime: CFG reads full-width logits and self-conditioning
    // reads the canvas row block at a fixed offset, so disable when either is set. Public-API only.
    const bool    canvas_logits_only = params.canvas_logits_only && params.cfg_scale == 0.0f && !params.self_conditioning;
    const int32_t first_logit_row    = params.shift_logits ? n_input - 1 : n_input;  // first row we ask logits for

    // Prefix-KV cache (fork-side): PREFILL the prompt once into the per-layer K/V store, then each step DECODE
    // only the canvas (reading the cached prompt via the diffusion-decode attention path). Uses the fork's
    // llama_diffusion_set_phase for non-canvas masked diffusion models. Masked-absorbing regime only (no CFG/SC).
    // shift_logits supported: the first canvas position's prediction is the last prompt row, stashed at PREFILL.
    const bool    prefix_kv = params.prefix_kv && params.cfg_scale == 0.0f && !params.self_conditioning;
    const int32_t C_canvas  = params.max_length - n_input;
    llama_model * model_mut = const_cast<llama_model *>(model);
    vector<float> last_prompt_logits;  // shift_logits: prediction for the first canvas position

    vector<llama_token_data> candidates(n_vocab);
    vector<llama_token_data> conf_candidates;
    conf_candidates.reserve(params.max_length);
    vector<int32_t> mask_positions;
    mask_positions.reserve(params.max_length);

    // Setup sampler chain
    struct llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    if (params.top_k > 0) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_k(params.top_k));
    }
    if (params.top_p < 1.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_top_p(params.top_p, 1));
    }
    if (params.temperature > 0.0f) {
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(params.temperature));
    }
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(params.seed));

    struct llama_sampler * dist_sampler = llama_sampler_init_dist(params.seed);

    llama_batch batch = llama_batch_init(params.max_length, 0, 1);
    batch.n_tokens    = params.max_length;

    // Self-conditioning (DiffusionGemma): cache each step's canvas-row logits and feed them into the next
    // step (canvas = [n_input, max_length)); set_sc is a no-op for other models.
    llama_model *      sc_model = const_cast<llama_model *>(llama_get_model(ctx));
    const int32_t      sc_canvas = params.max_length - n_input;
    vector<float> sc_buffer;
    if (params.self_conditioning) {
        sc_buffer.assign((size_t) sc_canvas * n_vocab, 0.0f);
    }

    // Pre-allocate buffers for CFG if needed
    int32_t                  logits_size = n_vocab * params.max_length;
    vector<float>       cond_logits_buffer;
    vector<llama_token> un_x_buffer;
    if (params.cfg_scale > 0.0f) {
        cond_logits_buffer.resize(logits_size);
        un_x_buffer.resize(params.max_length);
    }

    // For block-based processing
    vector<int32_t> num_transfer_tokens;
    int32_t              num_blocks      = 1;
    int32_t              steps_per_block = params.steps;

    if (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) {
        // Block over the CANVAS only (the prompt [0,n_input) is fixed context, not a block). Ceil-divide so any
        // canvas / block_length combination works - the original code tiled max_length (prompt included) and
        // asserted divisibility, which both mis-placed blocks and rejected most prompt lengths.
        const int32_t canvas_len = params.max_length - n_input;
        num_blocks      = (canvas_len + params.block_length - 1) / params.block_length;
        steps_per_block = std::max(1, params.steps / num_blocks);
    }

    vector<float> confidence(params.max_length);

    // P3 stability/freeze state (persists across steps): per-position argmax history + run-length.
    vector<llama_token> prev_argmax(params.max_length, -1);
    vector<int32_t>     stable_count(params.max_length, 0);

    // P3 global-convergence-stop state: last step's full-canvas prediction + how long it has held.
    vector<llama_token> prev_full(params.max_length, -2);
    int32_t                  converge_held = 0;
    bool                     converged     = false;


    // The per-position argmax + stability bookkeeping is only needed by freeze / converge / credit. When all
    // are off, skip it so the hot vocab loop matches stock llama.cpp exactly (no extra work in the baseline).
    const bool need_argmax = params.freeze_k > 0 || params.converge_stop > 0;

    int64_t total_sampling_time = 0;
    int64_t total_time          = 0;
    int64_t time_start          = ggml_time_us();

    if (prefix_kv) {
        // PREFILL: forward the prompt once; the masked graph writes its per-layer K/V into the store.
        utopic_llama_diffusion_set_phase(model_mut, /*PKV_PREFILL=*/1, n_input);
        batch.n_tokens = n_input;
        for (int32_t i = 0; i < n_input; i++) {
            batch.token[i]     = input_tokens[i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            // shift_logits: the first canvas position predicts from the LAST prompt row -> request + stash it.
            batch.logits[i]    = (params.shift_logits && i == n_input - 1) ? 1 : 0;
        }
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: prefix-KV PREFILL decode failed\n", __func__);
            utopic_llama_diffusion_set_phase(model_mut, /*PKV_UNIFIED=*/0, 0);
            llama_batch_free(batch);
            llama_sampler_free(sampler);
            llama_sampler_free(dist_sampler);
            return;
        }
        if (params.shift_logits) {
            const float * pl = llama_get_logits_ith(ctx, n_input - 1);
            if (pl) { last_prompt_logits.assign(pl, pl + n_vocab); }
        }
    }

    for (int block_num = 0; block_num < num_blocks; block_num++) {
        int32_t block_start = (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) ? n_input + block_num * params.block_length : 0;
        int32_t block_end   = (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) ?
                                  std::min(n_input + (block_num + 1) * params.block_length, params.max_length) :
                                  params.max_length;

        // Count masked tokens in current block for block-based processing
        if (params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED) {
            int32_t block_mask_count = 0;
            for (int i = block_start; i < block_end; i++) {
                if (output_tokens[i] == params.mask_token_id) {
                    block_mask_count++;
                }
            }
            num_transfer_tokens = get_num_transfer_tokens(block_mask_count, steps_per_block);
        }

        for (int32_t step = 0; step < steps_per_block; step++) {
            int32_t global_step = block_num * steps_per_block + step;

            if (params.step_callback) {
                if (!params.step_callback(
                        global_step, params.steps, output_tokens, params.max_length, params.step_callback_user_data)) {
                    break;
                }
            }

            // Setup batch. prefix_kv: DECODE only the canvas (positions n_input..max_length), reading cached
            // prompt K/V. Else: decode the full [prompt|canvas] (canvas_logits_only just trims which rows
            // produce logits; attention unchanged).
            // Periodic refresh (dKV-Cache style): every prefix_kv_refresh steps run a UNIFIED full-attention
            // forward so the prompt re-sees the committed canvas (recovers the static-prefix accuracy loss);
            // canvas-only DECODE (cheap) in between. pkv_decode=false on a refresh step -> full batch + std logits.
            // Fast-dLLM dual-cache (block schedule): PREFILL-refresh at block start (so prompt + committed
            // blocks see current state), then BLOCK_DECODE only the active block for the remaining steps.
            const bool dc = params.dual_cache && params.schedule == DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED;
            const bool dc_refresh_step = dc && (step == 0 || (params.dc_refresh > 0 && step % params.dc_refresh == 0));
            const bool dc_block_step   = dc && !dc_refresh_step;
            const int32_t dc_blkC = block_end - block_start;
            const bool pkv_decode = prefix_kv &&
                !(params.prefix_kv_refresh > 0 && global_step > 0 && global_step % params.prefix_kv_refresh == 0);
            if (dc_block_step) {
#ifdef DG_NO_BLOCK_DECODE
                fprintf(stderr, "dual_cache: llama_diffusion_set_block_decode unavailable in this llama build\n");
#else
                llama_diffusion_set_block_decode(model_mut, block_start, dc_blkC, params.max_length);
#endif
                batch.n_tokens = dc_blkC;
                for (int32_t i = 0; i < dc_blkC; i++) {
                    batch.token[i]     = output_tokens[block_start + i];
                    batch.pos[i]       = block_start + i;
                    batch.n_seq_id[i]  = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i]    = 1;
                }
            } else if (dc_refresh_step) {
                utopic_llama_diffusion_set_phase(model_mut, /*PKV_PREFILL=*/1, params.max_length);
                batch.n_tokens = params.max_length;
                for (int32_t i = 0; i < params.max_length; i++) {
                    batch.token[i]     = output_tokens[i];
                    batch.pos[i]       = i;
                    batch.n_seq_id[i]  = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i]    = 1;
                }
            } else if (pkv_decode) {
                utopic_llama_diffusion_set_phase(model_mut, /*PKV_DECODE=*/2, n_input);
                batch.n_tokens = C_canvas;
                for (int32_t i = 0; i < C_canvas; i++) {
                    batch.token[i]     = output_tokens[n_input + i];
                    batch.pos[i]       = n_input + i;
                    batch.n_seq_id[i]  = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i]    = 1;
                }
            } else if (prefix_kv) {
                // REFRESH: full [prompt|canvas] forward that ALSO re-stores the canvas-aware prompt K/V into the
                // cache (PKV_REFRESH), so subsequent DECODE steps read fresh prompt KV (near-lossless recovery).
                utopic_llama_diffusion_set_phase(model_mut, /*PKV_REFRESH=*/3, n_input);
                batch.n_tokens = params.max_length;
                for (int32_t i = 0; i < params.max_length; i++) {
                    batch.token[i]     = output_tokens[i];
                    batch.pos[i]       = i;
                    batch.n_seq_id[i]  = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i]    = 1;
                }
            } else {
                batch.n_tokens = params.max_length;
                for (int32_t i = 0; i < params.max_length; i++) {
                    batch.token[i]     = output_tokens[i];
                    batch.pos[i]       = i;
                    batch.n_seq_id[i]  = 1;
                    batch.seq_id[i][0] = 0;
                    batch.logits[i]    = canvas_logits_only ? (int8_t) (i >= first_logit_row) : 1;
                }
            }

            if (params.self_conditioning) {
                // step 0 has no previous prediction: keep the SC subgraph (stable graph shape) but gate it off
                llama_diffusion_set_sc(sc_model, sc_buffer.data(), global_step == 0 ? 0.0f : 1.0f, 1.0f, true);
            }

            float * logits = nullptr;

            if (params.cfg_scale > 0.0f) {
                int ret = llama_decode(ctx, batch);
                if (ret != 0) {
                    LOG_ERR("Failed to generate conditional");
                    break;
                }
                float * cond_logits_ptr = llama_get_logits(ctx);
                std::memcpy(cond_logits_buffer.data(), cond_logits_ptr, logits_size * sizeof(float));

                // Unconditional generation (mask input)
                std::copy(output_tokens, output_tokens + params.max_length, un_x_buffer.begin());
                for (int32_t i = 0; i < n_input; i++) {
                    un_x_buffer[i] = params.mask_token_id;
                }

                for (int32_t i = 0; i < params.max_length; i++) {
                    batch.token[i] = un_x_buffer[i];
                }
                ret = llama_decode(ctx, batch);
                if (ret != 0) {
                    LOG_ERR("Failed to generate unconditional");
                    break;
                }
                float * uncond_logits = llama_get_logits(ctx);

                // Apply CFG
                for (int32_t i = 0; i < logits_size; i++) {
                    cond_logits_buffer[i] =
                        uncond_logits[i] + (params.cfg_scale + 1.0f) * (cond_logits_buffer[i] - uncond_logits[i]);
                }
                logits = cond_logits_buffer.data();
            } else {
                int ret = llama_decode(ctx, batch);
                if (ret != 0) {
                    LOG_ERR("%s: failed to decode at step %d, ret = %d\n", __func__, global_step, ret);
                    break;
                }
                logits = llama_get_logits(ctx);
            }

            if (!logits) {
                LOG_ERR("%s: failed to get logits at step %d\n", __func__, global_step);
                break;
            }

            if (params.self_conditioning) {
                std::memcpy(sc_buffer.data(), logits + (size_t) n_input * n_vocab,
                            (size_t) sc_canvas * n_vocab * sizeof(float));
            }

            auto get_logits_for_pos = [&](int32_t pos) -> const float * {
                if (dc_block_step) {  // BLOCK_DECODE produced active-block logits; row r = position block_start+r.
                    const int32_t r = params.shift_logits ? (pos - 1 - block_start) : (pos - block_start);
                    return llama_get_logits_ith(ctx, r);
                }
                if (prefix_kv && pkv_decode) {  // canvas-only DECODE logits (a UNIFIED refresh step falls through)
                    // DIAG: force non-shift reading to localize all-EOS bugs (graph vs indexing).
                    static const bool dg_noshift = getenv("DG_PKV_NOSHIFT") != nullptr;
                    if (dg_noshift) { return llama_get_logits_ith(ctx, pos - n_input); }
                    // DECODE produced canvas-only logits; row r = prediction for canvas position n_input+r.
                    if (params.shift_logits) {
                        // prediction for pos comes from row pos-1: first canvas pos = stashed last-prompt row,
                        // the rest = canvas row (pos-1-n_input).
                        return pos == n_input ? last_prompt_logits.data()
                                              : llama_get_logits_ith(ctx, pos - 1 - n_input);
                    }
                    return llama_get_logits_ith(ctx, pos - n_input);
                }
                if (canvas_logits_only) {
                    // Only a subset of rows were requested; let llama map batch index -> packed row via
                    // output_ids (robust to the packing layout). The prediction for pos lives at batch row
                    // (shift_logits ? pos-1 : pos), which we guaranteed is in the requested set.
                    return llama_get_logits_ith(ctx, params.shift_logits ? pos - 1 : pos);
                }
                if (params.shift_logits) {
                    return pos == 0 ? logits : logits + (pos - 1) * n_vocab;
                }
                return logits + pos * n_vocab;
            };

            // Schema-constrained decoding: allow-bitmask for canvas position `pos` (nullptr = unconstrained).
            // Only tokens whose bit is set may be sampled there; the per-position loops below skip the rest.
            auto allow_for_pos = [&](int32_t pos) -> const uint64_t * {
                if (!params.slot_class || !params.class_allow) { return nullptr; }
                const int32_t j = pos - n_input;
                if (j < 0 || j >= params.max_length - n_input) { return nullptr; }
                const uint8_t c = params.slot_class[j];
                if (c == 0 || c > params.n_class) { return nullptr; }
                return params.class_allow + (size_t) (c - 1) * params.n_words;
            };

            int64_t time_start_sampling = ggml_time_us();

            mask_positions.clear();
            for (int32_t i = 0; i < params.max_length; i++) {
                if (output_tokens[i] == params.mask_token_id) {
                    // For block-based, only consider current block
                    if (params.schedule != DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED || (i >= block_start && i < block_end)) {
                        mask_positions.push_back(i);
                    }
                }
            }

            if (mask_positions.empty()) {
                break;
            }

            if (params.add_gumbel_noise && params.temperature > 0.0f) {
                add_gumbel_noise(logits, n_vocab, params.temperature, rng);
            }

            if (params.algorithm == DIFFUSION_ALGORITHM_ORIGIN) {
                int32_t transfer_count = calculate_transfer_count(
                    step, steps_per_block, mask_positions.size(), params.schedule, params.eps, num_transfer_tokens);
                float p_transfer = (float) transfer_count / mask_positions.size();

                for (int32_t pos : mask_positions) {
                    if (std::uniform_real_distribution<float>(0.0f, 1.0f)(rng) < p_transfer) {
                        const float * pos_logits = get_logits_for_pos(pos);
                        const uint64_t * allow    = allow_for_pos(pos);  // schema constraint (nullptr = free)
                        for (int32_t token_id = 0; token_id < n_vocab; token_id++) {
                            const bool blocked = allow && !(allow[token_id >> 6] & (1ull << (token_id & 63)));
                            candidates[token_id].id    = token_id;
                            candidates[token_id].logit = blocked ? -INFINITY : pos_logits[token_id];
                            candidates[token_id].p     = 0.0f;
                        }
                        if (params.suppress_mask_token) {
                            candidates[params.mask_token_id].logit = -INFINITY;  // never reveal as mask
                        }

                        llama_token_data_array cur_p = {
                            candidates.data(),
                            (size_t) n_vocab,
                            -1,
                            false,
                        };

                        llama_sampler_apply(sampler, &cur_p);
                        output_tokens[pos] = cur_p.data[cur_p.selected].id;
                    }
                }
            } else {
                const size_t npos = mask_positions.size();
                vector<llama_token> sampled_tokens(npos);
                vector<llama_token> argmax_tokens(npos);   // greedy pick (P3 freeze/converge use it)
                vector<float>       conf_arr(npos);
                vector<float>       entropy_arr(npos, 0.0f);  // per-position softmax entropy (EB-Sampler)

                // Fast THREADED path: greedy (temp=0) confidence-based decoding needs no shared sampler state -
                // per position we read the logits and compute argmax + top-probability confidence (+ optional
                // per-position credit) directly, so positions parallelize cleanly across cores. This mirrors the
                // eb/canvas path's threaded worker and is the masked-path equivalent (the old single-threaded
                // llama_sampler loop was ~7x slower than stock llama.cpp's threaded sampler). Configs that need
                // the sampler chain (temp>0, top_k/top_p, non-confidence algorithms) keep the exact path below.
                // At temp=0 the result is greedy (argmax) regardless of top_k/top_p (they never move the
                // argmax), and CONFIDENCE_BASED confidence is the argmax's probability - both computable
                // per-position without the shared sampler, so this case threads.
                const bool fast_path = params.temperature == 0.0f && params.alg_temp == 0.0f &&
                                       params.algorithm == DIFFUSION_ALGORITHM_CONFIDENCE_BASED;

                if (fast_path) {
                    auto worker = [&](size_t a, size_t b) {
                        for (size_t i = a; i < b; i++) {
                            const int32_t pos = mask_positions[i];
                            const float * row = get_logits_for_pos(pos);
                            const uint64_t * allow = allow_for_pos(pos);  // schema constraint (nullptr = free)
                            int32_t amax = -1; float m = -INFINITY;
                            for (int32_t v = 0; v < n_vocab; v++) {
                                if (params.suppress_mask_token && v == params.mask_token_id) { continue; }
                                if (allow && !(allow[v >> 6] & (1ull << (v & 63)))) { continue; }
                                if (row[v] > m) { m = row[v]; amax = v; }
                            }
                            double Z = 0.0, Sacc = 0.0;  // partition + sum  e*(logit-m), relative to max logit
                            for (int32_t v = 0; v < n_vocab; v++) {
                                if (params.suppress_mask_token && v == params.mask_token_id) { continue; }
                                if (allow && !(allow[v >> 6] & (1ull << (v & 63)))) { continue; }
                                const double d = (double) (row[v] - m);
                                const double e = exp(d);
                                Z += e; Sacc += e * d;
                            }
                            // softmax entropy H = ln Z - (sum  e*(logit-m))/Z  (nats); EB-Sampler accept signal
                            entropy_arr[i] = (float) std::max(0.0, log(Z) - Sacc / Z);
                            // top-probability confidence = p(argmax) = 1/Z  (Z is relative to the max logit)
                            conf_arr[i]       = (float) (1.0 / Z);
                            sampled_tokens[i] = amax;
                            argmax_tokens[i]  = amax;
                            stable_count[pos] = (amax == prev_argmax[pos]) ? stable_count[pos] + 1 : 0;
                            prev_argmax[pos]  = amax;
                        }
                    };
                    const unsigned hw  = std::thread::hardware_concurrency();
                    const unsigned nth = std::max(1u, std::min(hw ? hw : 1u, 32u));
                    if (nth <= 1 || npos < 2) {
                        worker(0, npos);
                    } else {
                        vector<std::thread> pool;
                        const size_t chunk = (npos + nth - 1) / nth;
                        for (unsigned t = 0; t < nth; t++) {
                            const size_t a = (size_t) t * chunk, b = std::min(a + chunk, npos);
                            if (a < b) { pool.emplace_back(worker, a, b); }
                        }
                        for (auto & th : pool) { th.join(); }
                    }
                } else {
                    // single-threaded llama_sampler path: temp>0 / top_k / top_p / non-confidence algorithms.
                    for (size_t i = 0; i < npos; i++) {
                        const int32_t pos        = mask_positions[i];
                        const float * pos_logits = get_logits_for_pos(pos);
                        const uint64_t * allow    = allow_for_pos(pos);  // schema constraint (nullptr = free)
                        int32_t amax = -1; float amax_logit = -INFINITY;
                        for (int32_t token_id = 0; token_id < n_vocab; token_id++) {
                            const bool blocked = allow && !(allow[token_id >> 6] & (1ull << (token_id & 63)));
                            candidates[token_id].logit = blocked ? -INFINITY : pos_logits[token_id];
                            candidates[token_id].p     = 0.0f;
                            candidates[token_id].id    = token_id;
                            if (need_argmax && !blocked && !(params.suppress_mask_token && token_id == params.mask_token_id) &&
                                pos_logits[token_id] > amax_logit) { amax_logit = pos_logits[token_id]; amax = token_id; }
                        }
                        if (params.suppress_mask_token) {
                            candidates[params.mask_token_id].logit = -INFINITY;
                        }
                        llama_token_data_array cur_p = { candidates.data(), candidates.size(), -1, false };
                        llama_sampler_apply(sampler, &cur_p);
                        sampled_tokens[i] = cur_p.data[cur_p.selected].id;
                        conf_arr[i]       = calculate_confidence(cur_p, params.algorithm, rng);
                        if (need_argmax) {
                            argmax_tokens[i]  = amax;
                            stable_count[pos] = (amax == prev_argmax[pos]) ? stable_count[pos] + 1 : 0;
                            prev_argmax[pos]  = amax;
                        }
                    }
                }

                // EB-Sampler (entropy-bounded unmask): accept lowest-entropy masked positions while the sum of
                // strictly-earlier per-position entropies stays <= eb_gamma (KL-bounded joint-dependence error).
                // Commits to argmax. Replaces the schedule/threshold commit for this step when active.
                const bool eb_active = (params.eb_gamma > 0.0f && fast_path);
                if (eb_active) {
                    vector<int32_t> order(npos);
                    std::iota(order.begin(), order.end(), 0);
                    std::sort(order.begin(), order.end(),
                              [&](int32_t a, int32_t b) { return entropy_arr[a] < entropy_arr[b]; });
                    double prior = 0.0; int32_t committed = 0;
                    for (size_t k = 0; k < npos; k++) {
                        const int32_t idx = order[k];
                        if (prior > (double) params.eb_gamma && committed >= params.eb_min_commit) { break; }
                        output_tokens[mask_positions[idx]] = argmax_tokens[idx];
                        committed++;
                        prior += (double) entropy_arr[idx];
                    }
                }

                vector<pair<float, int32_t>> confidences;
                confidences.reserve(npos);
                for (size_t i = 0; i < npos; i++) { confidences.emplace_back(conf_arr[i], i); }

                int32_t transfer_count = eb_active ? 0 : calculate_transfer_count(
                    step, steps_per_block, mask_positions.size(), params.schedule, params.eps, num_transfer_tokens);

                if (transfer_count > 0) {
                    if (params.alg_temp == 0.0f) {
                        std::partial_sort(confidences.begin(),
                                          confidences.begin() + std::min(transfer_count, (int32_t) confidences.size()),
                                          confidences.end(),
                                          [](const pair<float, int32_t> & a, const pair<float, int32_t> & b) {
                                              if (a.first != b.first) {
                                                  return a.first > b.first;
                                              }
                                              return a.second < b.second;
                                          });

                        int32_t committed = std::min(transfer_count, (int32_t) confidences.size());
                        for (int32_t i = 0; i < committed; i++) {
                            int32_t mask_idx   = confidences[i].second;
                            int32_t pos        = mask_positions[mask_idx];
                            output_tokens[pos] = sampled_tokens[mask_idx];
                        }
                        // Confidence gate (P3): additionally commit any not-yet-committed position whose
                        // confidence clears the threshold. partial_sort left [0,committed) as the most
                        // confident, so the remainder is scanned linearly (each has conf <= the committed set).
                        if (params.confidence_threshold > 0.0f && global_step >= params.confidence_warmup) {
                            for (int32_t i = committed; i < (int32_t) confidences.size(); i++) {
                                if (confidences[i].first >= params.confidence_threshold) {
                                    int32_t mask_idx   = confidences[i].second;
                                    output_tokens[mask_positions[mask_idx]] = sampled_tokens[mask_idx];
                                }
                            }
                        }
                        // Stability commit (P3 freeze): commit still-masked positions whose argmax has held for
                        // freeze_k steps. Self-calibrating - can't fire before step freeze_k and waits for each
                        // position's neighbors to settle, so it accelerates without the gate's step-0 overcommit.
                        if (params.freeze_k > 0) {
                            for (size_t i = 0; i < mask_positions.size(); i++) {
                                int32_t pos = mask_positions[i];
                                if (output_tokens[pos] == params.mask_token_id && stable_count[pos] >= params.freeze_k) {
                                    output_tokens[pos] = argmax_tokens[i];
                                }
                            }
                        }
                    } else {
                        conf_candidates.clear();
                        for (size_t i = 0; i < confidences.size(); i++) {
                            float conf_logit = confidences[i].first / params.alg_temp;
                            conf_candidates.emplace_back(llama_token_data{ (int32_t) i, conf_logit, 0.0f });
                        }

                        llama_token_data_array conf_array = {
                            conf_candidates.data(),
                            conf_candidates.size(),
                            -1,
                            false,
                        };

                        for (int32_t i = 0; i < std::min(transfer_count, (int32_t) confidences.size()); i++) {
                            llama_sampler_apply(dist_sampler, &conf_array);
                            int32_t selected_idx = conf_array.selected;
                            int32_t mask_idx     = selected_idx;
                            int32_t pos          = mask_positions[mask_idx];
                            output_tokens[pos]   = sampled_tokens[mask_idx];

                            conf_candidates[selected_idx].p = 0.0f;
                            conf_array.selected             = -1;
                        }
                    }
                }

                // P3 global convergence stop: snapshot the full predicted canvas (a position's committed token
                // if set this step, else its argmax). When that prediction is identical for converge_stop
                // consecutive steps nothing is still resolving - commit every masked position to its argmax and
                // finish. Commits nothing ahead of schedule, so it can't lock premature/garbage tokens.
                if (params.converge_stop > 0) {
                    bool same = true;
                    for (size_t i = 0; i < mask_positions.size(); i++) {
                        int32_t     pos  = mask_positions[i];
                        llama_token pred = (output_tokens[pos] == params.mask_token_id) ? argmax_tokens[i] : output_tokens[pos];
                        if (pred != prev_full[pos]) { same = false; }
                        prev_full[pos] = pred;
                    }
                    converge_held = same ? converge_held + 1 : 0;
                    if (converge_held >= params.converge_stop) {
                        for (size_t i = 0; i < mask_positions.size(); i++) {
                            int32_t pos = mask_positions[i];
                            if (output_tokens[pos] == params.mask_token_id) { output_tokens[pos] = argmax_tokens[i]; }
                        }
                        converged = true;
                    }
                }
            }

            // EOS early-termination (lossless): once an EOS has committed and every canvas position before it
            // is also committed, the answer is final - fill the remaining masks with EOS and stop. Detok stops
            // at the first EOS regardless, so the emitted answer is unchanged; this only skips denoising dead
            // canvas past the end (cuts amplification on answers shorter than the canvas).
            if (params.eos_token_id != LLAMA_TOKEN_NULL && !converged) {
                int32_t eos_pos = -1;
                for (int32_t i = n_input; i < params.max_length; i++) {
                    if (output_tokens[i] == params.eos_token_id) { eos_pos = i; break; }
                }
                if (eos_pos >= 0) {
                    bool clean_prefix = true;
                    for (int32_t i = n_input; i < eos_pos; i++) {
                        if (output_tokens[i] == params.mask_token_id) { clean_prefix = false; break; }
                    }
                    if (clean_prefix) {
                        for (int32_t i = eos_pos; i < params.max_length; i++) {
                            if (output_tokens[i] == params.mask_token_id) { output_tokens[i] = params.eos_token_id; }
                        }
                        converged = true;
                    }
                }
            }

            int64_t time_end_sampling = ggml_time_us();
            total_sampling_time += time_end_sampling - time_start_sampling;
            if (converged) { break; }
        }
        if (converged) { break; }
    }

    int64_t time_end = ggml_time_us();
    total_time += time_end - time_start;

    LOG_INF("\ntotal time: %0.2fms, time per step: %0.2fms, sampling time per step: %0.2fms\n",
            total_time / 1000.0,
            total_time / 1000.0 / params.steps,
            total_sampling_time / 1000.0 / params.steps);

    if (prefix_kv) {
        utopic_llama_diffusion_set_phase(model_mut, /*PKV_UNIFIED=*/0, 0);  // restore default for later turns
    }

    llama_batch_free(batch);
    llama_sampler_free(sampler);
    llama_sampler_free(dist_sampler);

    n_generated = params.max_length;
}

// Entropy-bound denoiser for DiffusionGemma-style canvas models (see diffusion.h). The canvas is
// random-initialized; each step samples a candidate per position, accepts the lowest-entropy positions
// within a mutual-information bound, and renoises the rest under a linear temperature schedule. The output
// is the stable argmax canvas. Mirrors the reference transformers EntropyBoundSampler; set_sc is a no-op
// for non-DiffusionGemma models.
void diffusion_generate_entropy_bound(llama_context *             ctx,
                                      const llama_token *         input_tokens,
                                      llama_token *               output_tokens,
                                      int32_t                     n_input,
                                      const diffusion_eb_params & params,
                                      int32_t &                   n_generated) {
    n_generated = 0;
    if (!ctx || !input_tokens || !output_tokens || n_input <= 0 || params.max_length <= n_input) {
        return;
    }

    llama_model * model   = const_cast<llama_model *>(llama_get_model(ctx));
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const int32_t C       = params.max_length - n_input;            // canvas length
    const int32_t S       = std::max(1, params.max_denoising_steps);

    // device-resident SC: source self-conditioning from a persistent device buffer (written in-graph from
    // the prev step's logits) instead of the 268 MB host upload each step. Exact: SC values/math unchanged.
    const bool dev_sc = params.gpu_sampling;
    const bool gpu_sample_reduce = params.gpu_sample_reduce && dev_sc;  // Stage-1: sample from sc_dev on-device
    llama_diffusion_set_device_sc(model, dev_sc);

    llama_set_causal_attn(ctx, false);
    std::copy(input_tokens, input_tokens + n_input, output_tokens);

    std::mt19937                           rng(params.seed);
    std::uniform_real_distribution<float>  uni01(0.0f, 1.0f);
    std::uniform_int_distribution<int32_t> vocab_dist(0, n_vocab - 1);

    vector<llama_token> current_canvas(C);                    // working (renoised) canvas, fed to the forward
    for (int32_t i = 0; i < C; i++) {
        current_canvas[i] = vocab_dist(rng);                      // random init (not mask)
    }

    // previous step's raw logits, for self-cond (host upload path only; device SC keeps them on-device)
    vector<float>       sc_buffer((size_t) (dev_sc ? 0 : C) * n_vocab, 0.0f);
    vector<llama_token> argmax_canvas(C, 0);                  // model's best prediction = the output
    vector<llama_token> prev_argmax(C, -1);                  // stability history (-1 -> step 0 is unstable)
    vector<float>       entropy(C);
    vector<llama_token> denoiser(C);
    vector<int32_t>     order(C);
    vector<float>       u(C);                                // pre-drawn multinomial draws (determinism)
    vector<llama_token> renoise(C);                         // pre-drawn renoise tokens
    // Utopic sparse-active-set pre-check: freeze positions whose argmax has been
    // stable for >= DG_FREEZE_K steps (lock token, stop renoising). Quality precursor to
    // skipping their forward compute. DG_FREEZE_K=0 -> stock behavior.
    vector<int32_t> stable_count(C, 0);
    const int32_t DG_FREEZE_K = getenv("DG_FREEZE_K") ? atoi(getenv("DG_FREEZE_K")) : 0;

    const unsigned hw  = std::thread::hardware_concurrency();
    const unsigned nth = std::max(1u, std::min(hw ? hw : 1u, 32u));

    llama_batch batch = llama_batch_init(params.max_length, 0, 1);

    // Cached path: PREFILL the prompt once (writing the prefix K/V store), then each step DECODE only the
    // canvas, reading the cached prefix - instead of re-decoding [prompt|canvas] every step. The packed
    // canvas logits then start at row 0 (cached) instead of row n_input (unified).
    const int32_t logit_off = params.kv_cache ? 0 : n_input;
    if (params.kv_cache) {
        utopic_llama_diffusion_set_phase(model, /*PKV_PREFILL=*/1, n_input);
        llama_diffusion_set_sc(model, nullptr, 0.0f, 1.0f, false);
        batch.n_tokens = n_input;
        for (int32_t i = 0; i < n_input; i++) {
            batch.token[i]     = input_tokens[i];
            batch.pos[i]       = i;
            batch.n_seq_id[i]  = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i]    = 1;  // encode() forces all rows to output anyway; set them so it stays quiet
        }
        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: PREFILL decode failed\n", __func__);
            utopic_llama_diffusion_set_phase(model, /*PKV_UNIFIED=*/0, 0);
            llama_batch_free(batch);
            return;
        }
    }

    float   prev_temp_inv = 1.0f;
    int     held          = 0;
    bool    finished      = false;
    bool    device_sample_ok = gpu_sample_reduce;   // latched off if a backend (e.g. Metal) can't device-sample

    for (int32_t cur_step = S; cur_step >= 1 && !finished; --cur_step) {
        const int32_t step_idx = S - cur_step;                    // 0-based
        const float   t        = params.t_min + (params.t_max - params.t_min) * ((float) cur_step / (float) S);
        const float   temp_inv = 1.0f / t;

        if (params.kv_cache) {
            utopic_llama_diffusion_set_phase(model, /*PKV_DECODE=*/2, n_input);
            batch.n_tokens = C;
            for (int32_t i = 0; i < C; i++) {
                batch.token[i]     = current_canvas[i];
                batch.pos[i]       = n_input + i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = 1;
            }
        } else {
            batch.n_tokens = params.max_length;
            for (int32_t i = 0; i < params.max_length; i++) {
                batch.token[i]     = (i < n_input) ? input_tokens[i] : current_canvas[i - n_input];
                batch.pos[i]       = i;
                batch.n_seq_id[i]  = 1;
                batch.seq_id[i][0] = 0;
                batch.logits[i]    = 1;
            }
        }

        // self-conditioning = softmax(previous step's logits / previous t); gated off on the first step.
        // device SC ignores the host pointer (reads sc_dev), so pass nullptr; the gate + temp are identical.
        llama_diffusion_set_sc(model, dev_sc ? nullptr : sc_buffer.data(),
                               step_idx == 0 ? 0.0f : 1.0f, prev_temp_inv, true);

        if (llama_decode(ctx, batch) != 0) {
            LOG_ERR("%s: failed to decode at step %d\n", __func__, step_idx);
            break;
        }

        // Stage-1: when on, skip the 268 MB logits D2H + host reductions and sample on the GPU from sc_dev.
        const bool gpu_reduce  = dev_sc && device_sample_ok;
        const bool want_logits = !gpu_reduce;
        const float * logits = nullptr;                           // canvas rows packed: [C or max_length, n_vocab]
        if (want_logits) {
            logits = llama_get_logits(ctx);
        } else {
            llama_synchronize(ctx);                               // sc_dev write must complete before we read it
        }

        // pre-draw the step's randomness single-threaded so the output is seed-reproducible
        for (int32_t pos = 0; pos < C; pos++) {
            u[pos]       = uni01(rng);
            renoise[pos] = vocab_dist(rng);
        }

        // per position: argmax, entropy of softmax(raw/t), and a multinomial sample; stash raw row for SC
        auto worker = [&](int32_t p0, int32_t p1) {
            for (int32_t pos = p0; pos < p1; pos++) {
                const float * row = logits + (size_t) (logit_off + pos) * n_vocab;
                float m = -INFINITY; int32_t amax = 0;
                for (int32_t v = 0; v < n_vocab; v++) {
                    const float z = row[v] * temp_inv;
                    if (z > m) { m = z; amax = v; }
                }
                float Z = 0.0f;
                for (int32_t v = 0; v < n_vocab; v++) {
                    Z += expf(row[v] * temp_inv - m);
                }
                const float target = u[pos] * Z;
                float   cum = 0.0f, H = 0.0f;
                int32_t sampled = n_vocab - 1; bool picked = false;
                for (int32_t v = 0; v < n_vocab; v++) {
                    const float e = expf(row[v] * temp_inv - m);
                    const float p = e / Z;
                    if (p > 0.0f) { H -= p * logf(p); }
                    cum += e;
                    if (!picked && cum >= target) { sampled = v; picked = true; }
                }
                entropy[pos]       = H;
                argmax_canvas[pos] = amax;
                denoiser[pos]      = sampled;
                // device SC keeps prev-step logits on-device (cpy in-graph), so no host stash needed
                if (!dev_sc) {
                    std::memcpy(sc_buffer.data() + (size_t) pos * n_vocab, row, n_vocab * sizeof(float));
                }
            }
        };
        auto run_host_worker = [&]() {
            vector<std::thread> pool;
            const int32_t chunk = (C + (int32_t) nth - 1) / (int32_t) nth;
            for (unsigned ti = 0; ti < nth; ti++) {
                const int32_t p0 = (int32_t) ti * chunk;
                const int32_t p1 = std::min(p0 + chunk, C);
                if (p0 < p1) { pool.emplace_back(worker, p0, p1); }
            }
            for (auto & th : pool) { th.join(); }
        };

        if (gpu_reduce) {
            // Stage-1: argmax/entropy/sampled straight from sc_dev. argmax matches the host bit-for-bit; Z and
            // entropy differ only by the parallel-reduction order, so some sampled tokens may shift near ties.
            if (!llama_diffusion_device_sample(model, u.data(), argmax_canvas.data(), entropy.data(),
                                               denoiser.data(), C, temp_inv)) {
                // Some backends (e.g. Metal) cannot run the on-device sampler. Warn once and use the host
                // reduction for the rest of the run instead of retrying (and logging) on every step.
                if (device_sample_ok) {
                    LOG_WRN("%s: on-device sampling unsupported on this backend; using host sampling\n", __func__);
                    device_sample_ok = false;
                }
                if (!logits) { logits = llama_get_logits(ctx); }
                run_host_worker();
            }
        } else {
            run_host_worker();
        }

        // accept the lowest-entropy positions within the MI bound (sum of strictly-earlier entropies <= bound)
        std::iota(order.begin(), order.end(), 0);
        std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) { return entropy[a] < entropy[b]; });
        vector<char> accepted(C, 0);
        double cumE = 0.0;
        for (int32_t k = 0; k < C; k++) {
            const int32_t pos = order[k];
            cumE += entropy[pos];
            if (cumE - entropy[pos] <= params.entropy_bound) { accepted[pos] = 1; }
        }

        // renoise: accepted -> sampled token, rest -> fresh random; the displayed/output canvas is the argmax
        float entropy_sum = 0.0f;
        for (int32_t pos = 0; pos < C; pos++) {
            const bool pos_stable = (argmax_canvas[pos] == prev_argmax[pos]);
            stable_count[pos] = pos_stable ? stable_count[pos] + 1 : 0;
            const bool frozen = (DG_FREEZE_K > 0 && stable_count[pos] >= DG_FREEZE_K);
            if (frozen) {
                current_canvas[pos] = argmax_canvas[pos];           // lock the settled token (no renoise)
            } else {
                current_canvas[pos] = accepted[pos] ? denoiser[pos] : renoise[pos];
            }
            output_tokens[n_input + pos] = argmax_canvas[pos];
            entropy_sum += entropy[pos];
        }

        // adaptive stop: argmax stable for stability_threshold steps AND confident (low mean entropy)
        held = (prev_argmax == argmax_canvas) ? held + 1 : 0;
        const bool confident = (entropy_sum / (float) C) < params.confidence_threshold;
        if (held >= params.stability_threshold && confident) { finished = true; }
        if (getenv("DG_CURVE")) { int _spp = 0, _acc = 0; for (int32_t i = 0; i < C; i++) { if (argmax_canvas[i] == prev_argmax[i]) _spp++; if (accepted[i]) _acc++; }
          fprintf(stderr, "DG_CURVE step=%d stable_pp=%d accepted=%d total=%d held=%d mean_ent=%.5f\n", step_idx, _spp, _acc, C, held, entropy_sum/(float)C); }
        prev_argmax   = argmax_canvas;
        prev_temp_inv = temp_inv;

        if (params.step_callback &&
            !params.step_callback(step_idx, S, output_tokens, params.max_length, params.step_callback_user_data)) {
            break;
        }
    }

    if (params.kv_cache) {
        utopic_llama_diffusion_set_phase(model, /*PKV_UNIFIED=*/0, 0);  // restore default for later turns / masked path
    }
    if (dev_sc) {
        llama_diffusion_set_device_sc(model, false);             // restore host SC path for later turns
    }
    llama_batch_free(batch);
    n_generated = params.max_length;
}

}  // namespace utopic
