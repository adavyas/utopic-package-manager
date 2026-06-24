#pragma once

#include "llama.h"

#include <cstdint>

namespace utopic {

enum diffusion_algorithm {
    DIFFUSION_ALGORITHM_ORIGIN           = 0,
    DIFFUSION_ALGORITHM_ENTROPY_BASED    = 1,
    DIFFUSION_ALGORITHM_MARGIN_BASED     = 2,
    DIFFUSION_ALGORITHM_RANDOM           = 3,
    DIFFUSION_ALGORITHM_CONFIDENCE_BASED = 4,
};

// Unified transfer scheduling methods
enum diffusion_transfer_schedule {
    DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED = 0,  // Dream-style: (1.0 - s/t) * remaining
    DIFFUSION_TRANSFER_SCHEDULE_BLOCK_BASED    = 1,  // LLaDA-style: process in blocks with get_num_transfer_tokens
};

typedef bool (*diffusion_step_callback_t)(int32_t             step,
                                          int32_t             total_steps,
                                          const llama_token * tokens,
                                          int32_t             n_tokens,
                                          void *              user_data);

struct diffusion_params {
    int32_t                   steps                   = 0;
    float                     temperature             = 0;
    llama_token               mask_token_id           = LLAMA_TOKEN_NULL;
    diffusion_step_callback_t step_callback           = nullptr;
    void *                    step_callback_user_data = nullptr;
    int32_t                   seed                    = 0;
    bool                      visual_mode             = false;
    bool                      shift_logits            = false;  // Shift logits by -1 after decode
    bool                      suppress_mask_token     = false;  // forbid revealing a position as the mask token
                                                                // (masked-diffusion models that can emit it)
    bool                      self_conditioning       = false;  // feed each step's canvas logits back into the
                                                                // next step (DiffusionGemma; no-op for others)

    float   top_p = 0.;
    int32_t top_k = 0.;

    diffusion_algorithm         algorithm = DIFFUSION_ALGORITHM_CONFIDENCE_BASED;
    diffusion_transfer_schedule schedule  = DIFFUSION_TRANSFER_SCHEDULE_TIMESTEP_BASED;

    float   cfg_scale        = 0.;     // Config scale for classifier-free guidance
    float   eps              = 0.;     // Timestep scheduling
    int32_t block_length     = 0;      // Block size (for block scheduling)
    float   alg_temp         = 0;      // algorithm temperature (0.0 = deterministic)
    bool    add_gumbel_noise = false;  // Add gumbel noise to the logits if temp > 0.0

    int32_t max_length = 0;            // Maximum sequence length

    // Canvas-logits-only (P2, per-step latency): the masked loop only ever samples canvas positions, yet stock
    // requests logits for ALL rows - paying the lm_head projection (hidden -> ~126k vocab) + logits D2H over
    // the prompt too. Setting batch.logits=1 for canvas rows only (plus row n_input-1 when shift_logits) skips
    // that wasted work. Attention is UNCHANGED (we still decode the full [prompt|canvas] batch), so output is
    // bit-identical to the unified path - this is a pure compute/transfer trim, PUBLIC-API only.
    // (NOTE: true prefix-KV-cache - skipping the prompt forward entirely - needs fork-side graph surgery in the
    //  LLaDA/Dream model files; the eb path's prefix-KV is DiffusionGemma-graph-specific. Tracked separately.)
    bool    canvas_logits_only = false;

    // Prefix-KV cache (fork-side, WS1-3): PREFILL the prompt once into the per-layer K/V store, then DECODE
    // only the canvas each step (reads cached prompt K/V via llama_diffusion_set_phase + the diffusion-decode
    // attention path in the LLaDA/Dream graph). The per-step lever for long context (forward only the canvas,
    // not the whole [prompt|canvas]). Masked-absorbing regime only; shift_logits not yet supported.
    bool    prefix_kv = false;
    // Periodic refresh: every N steps run a UNIFIED full-attention forward so the prompt re-sees the committed
    // canvas (recovers static-prefix accuracy loss; dKV-Cache style). 0 = never refresh (pure static prefix).
    int32_t prefix_kv_refresh = 0;

    // Confidence gate (P3, step reduction): besides the scheduled top-transfer_count commits, also commit any
    // masked position whose confidence clears this threshold. The canvas empties faster, so the
    // mask_positions-empty early-stop fires in fewer denoise steps - fewer full forwards, the biggest
    // hardware-agnostic latency lever. <=0 disables (stock schedule); high (e.g. 0.9) = near-lossless
    // (only commits positions the model is already sure of). Intended for the confidence-based algorithm.
    float   confidence_threshold = 0.0f;

    // Gate warmup: suppress the confidence gate for the first N denoise steps. Early predictions (esp. step 0,
    // everything masked) are spuriously peaked on high-frequency tokens (">.999" even when wrong), so an
    // ungated warmup lets global structure resolve under the normal schedule before the gate accelerates.
    int32_t confidence_warmup = 0;

    // Stability commit / freeze (P3, step reduction): commit a masked position once its argmax has stayed
    // constant for freeze_k consecutive steps. Self-calibrating where the absolute-confidence gate is not - a
    // position's argmax keeps shifting while its neighbors resolve and only settles once structure has formed,
    // so this waits per-position (and can't fire before step freeze_k). Proven near-lossless (+19-33%). 0 off.
    int32_t freeze_k = 0;

    // EB-Sampler (entropy-bounded unmasking, arXiv 2505.24857; same rule as our eb/canvas path). Each step,
    // accept the LOWEST-entropy masked positions while the sum of strictly-earlier per-position entropies stays
    // <= eb_gamma. Bounds the joint-dependence error committed per step -> coherent multi-token commit where a
    // fixed per-position confidence threshold breaks reasoning. eb_gamma<=0 disables (falls back to schedule).
    // gamma=0 ~ 1 token/step; larger = more aggressive. Fast path (temp=0, confidence-based) only.
    float   eb_gamma      = 0.0f;
    int32_t eb_min_commit = 1;    // always commit >=1 position/step to guarantee progress

    // Global convergence stop (P3, the SAFE step-reduction lever): unlike the per-position gate/freeze, this
    // commits NOTHING ahead of the schedule - it only detects when the full predicted canvas (committed tokens
    // + the argmax at still-masked positions) has been identical for converge_stop consecutive steps, then
    // commits every position to its argmax and ends. Can't fire prematurely on structured output (content
    // positions keep shifting until they resolve), but ends fast once everything has settled (e.g. short
    // answers padded with EOS). 0 disables. This is quality-preserving where early-commit is not.
    int32_t converge_stop = 0;


    // EOS early-termination (lossless amplification cut): once an EOS commits with a fully-committed prefix,
    // the answer is finished - commit remaining masks as EOS-fill and stop. Detok already stops at EOS, so the
    // emitted answer is unchanged; it just avoids denoising dead canvas past the end. -1 / NULL disables.
    llama_token eos_token_id = LLAMA_TOKEN_NULL;

    // Fast-dLLM dual-cache (block-based schedule only): refresh the K/V store with a full PREFILL at each block
    // start (prompt + committed blocks see current state), then forward ONLY the active block via PKV_BLOCK_DECODE
    // for the remaining steps_per_block steps. Per-step cost drops O(canvas)->O(block). dc_refresh = extra
    // in-block refresh cadence (0 = refresh only at block start).
    bool    dual_cache = false;
    int32_t dc_refresh = 0;

    // S3 schema scaffolding: if non-null, the canvas [n_input, max_length) is initialized from this template
    // instead of all-mask. Positions == mask_token_id are value slots to fill; all others are FROZEN scaffold
    // tokens (JSON structure / keys / function name) that are never re-sampled (they never enter mask_positions),
    // so out-of-order unmasking can't corrupt structure. Length must equal (max_length - n_input).
    const llama_token * canvas_template = nullptr;

    // Schema-constrained decoding (completion-validity HARD guarantee; the GBNF-equivalent for diffusion).
    // GBNF can't drive diffusion (no left-to-right prefix during parallel commit), but a frozen scaffold fixes
    // the JSON delimiters and this constrains each value slot to a token alphabet so the result PARSES and is
    // type-conformant by construction. For canvas position p in [n_input, max_length), slot_class[p - n_input]
    // picks a class: 0 = unconstrained; c >= 1 = only tokens whose bit is set in
    // class_allow[(c-1)*n_words + v/64] may be sampled at p (all others forced to -inf before argmax/sampling).
    // n_words = (n_vocab + 63) / 64. Built host-side from a JSON schema (string -> no-quote/no-backslash class,
    // integer -> digits, number -> numeric chars), so a string slot can never break out of its quotes and a
    // number slot can never emit a non-numeric char -> the emitted JSON is always valid + schema-shaped.
    const uint8_t  * slot_class  = nullptr;   // length (max_length - n_input); 0 = free, c>=1 = constrained
    const uint64_t * class_allow = nullptr;   // n_class * n_words allow-bitmask (class c stored at (c-1)*n_words)
    int32_t          n_class     = 0;
    int32_t          n_words     = 0;         // (n_vocab + 63) / 64
};

void diffusion_generate(llama_context *          ctx,
                        const llama_token *      input_tokens,
                        llama_token *            output_tokens,
                        int32_t                  n_input,
                        const diffusion_params & params,
                        int32_t &                n_generated);

// Entropy-bound denoiser for block-diffusion canvas models (DiffusionGemma). Unlike the masked path, the
// canvas is random-initialized and non-accepted positions are renoised each step; tokens are accepted by a
// per-position entropy (mutual-information) bound, under a linear temperature schedule, with adaptive
// stopping. Writes the final argmax canvas into output_tokens[n_input .. max_length).
struct diffusion_eb_params {
    int32_t max_denoising_steps  = 48;
    float   t_min                = 0.4f;   // temperature at the last step
    float   t_max                = 0.8f;   // temperature at the first step
    float   entropy_bound        = 0.1f;   // accept lowest-entropy tokens within this MI bound
    int32_t stability_threshold  = 1;      // steps the argmax canvas must hold to count as stable
    float   confidence_threshold = 0.005f; // stop once mean canvas entropy drops below this
    int32_t seed                 = 0;
    int32_t max_length           = 0;      // n_input + canvas_length
    bool    kv_cache             = false;  // prefix-KV-cache the prompt (PREFILL once, decode canvas-only
                                           // per step) instead of re-decoding [prompt|canvas] every step
    bool    gpu_sampling         = false;  // device-resident self-conditioning: keep the prev step's canvas
                                           // logits on-device for SC instead of a per-step 268 MB host upload
                                           // (exact; the SC math/values are unchanged)
    bool    gpu_sample_reduce    = false;  // Stage-1: argmax/entropy/one multinomial draw per position done on
                                           // the GPU from sc_dev (skips the 268 MB logits D2H + host reductions).
                                           // Requires gpu_sampling. FP-equivalent: argmax exact, Z/entropy ~1e-4.

    diffusion_step_callback_t step_callback           = nullptr;
    void *                    step_callback_user_data = nullptr;
    bool                      visual_mode             = false;
};

inline bool eb_canvas_token_allowed(int32_t canvas_pos,
                                    bool    is_eog,
                                    bool    is_control,
                                    bool    has_visible_piece,
                                    bool    is_channel_marker = false) {
    return canvas_pos != 0 || (!is_eog && !is_control && has_visible_piece && !is_channel_marker);
}

void diffusion_generate_entropy_bound(llama_context *             ctx,
                                      const llama_token *         input_tokens,
                                      llama_token *               output_tokens,
                                      int32_t                     n_input,
                                      const diffusion_eb_params & params,
                                      int32_t &                   n_generated);

}  // namespace utopic
