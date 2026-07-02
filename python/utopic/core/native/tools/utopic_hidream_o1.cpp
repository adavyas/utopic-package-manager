#include "core/hidream_o1_native.h"
#include "core/hidream_o1_block.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>

namespace {

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s --prompt TEXT --out image.png [--model-dir DIR] [--source-dir DIR] [--torch-python PY] "
                 "[--width 1024] [--height 1024] [--steps 28] [--seed 42] [--extra-args ARGS] "
                 "[--native-exec-check] [--native-text-tokens N] [--native-real-block0-tokens N] "
                 "[--native-real-visual-block0-tokens N] [--native-full-chain-check] "
                 "[--native-chain-text-tokens N] [--native-chain-visual-tokens N] "
                 "[--native-projection-patch-tokens N] [--native-projection-final-tokens N] "
                 "[--native-preview] [--native-skip-payloads] [--dry-run]\n",
                 argv0);
    std::fprintf(stderr,
                 "note: native-exec-check loads the HiDream config, safetensors catalog, forward token plan, block0 tensor payloads, and runs native block0 graph execution without invoking sd.cpp or Torch.\n");
}

bool consume(std::string arg, const char* name) {
    return arg == name;
}

bool mkdir_p(const std::string& dir) {
    if (dir.empty()) return true;
    std::string cur;
    for (char c : dir) {
        cur += c;
        if (c == '/' && cur.size() > 1) {
            mkdir(cur.c_str(), 0755);
        }
    }
    return mkdir(dir.c_str(), 0755) == 0 || errno == EEXIST;
}

bool ensure_parent_dir(const std::string& path) {
    const size_t slash = path.find_last_of('/');
    if (slash == std::string::npos) return true;
    return mkdir_p(path.substr(0, slash));
}

}  // namespace

int main(int argc, char** argv) {
    const utopic::HiDreamO1RuntimeConfig cfg = utopic::default_hidream_o1_runtime_config();
    utopic::HiDreamO1RunRequest req;
    req.torch_python = utopic::hidream_o1_default_torch_python();
    req.source_dir = utopic::hidream_o1_default_source_dir();
    req.model_dir = utopic::hidream_o1_default_model_dir();
    req.width = 1024;
    req.height = 1024;
    req.steps = cfg.default_steps;
    req.seed = 42;
    req.cfg_scale = cfg.default_guidance_scale;

    bool dry_run = false;
    bool native_exec_check = false;
    bool native_load_payloads = true;
    int64_t native_text_tokens = 256;
    int64_t native_real_block0_tokens = 1;
    int64_t native_real_visual_block0_tokens = 1;
    bool native_full_chain_check = false;
    int64_t native_chain_text_tokens = 1;
    int64_t native_chain_visual_tokens = 1;
    int64_t native_projection_patch_tokens = 1;
    int64_t native_projection_final_tokens = 1;
    bool native_preview = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (consume(a, "--help")) {
            usage(argv[0]);
            return 0;
        } else if (consume(a, "--prompt") && i + 1 < argc) {
            req.prompt = argv[++i];
        } else if (consume(a, "--out") && i + 1 < argc) {
            req.output_path = argv[++i];
        } else if ((consume(a, "--model-dir") || consume(a, "--model")) && i + 1 < argc) {
            req.model_dir = argv[++i];
        } else if (consume(a, "--source-dir") && i + 1 < argc) {
            req.source_dir = argv[++i];
        } else if (consume(a, "--torch-python") && i + 1 < argc) {
            req.torch_python = argv[++i];
        } else if (consume(a, "--width") && i + 1 < argc) {
            req.width = std::atoi(argv[++i]);
        } else if (consume(a, "--height") && i + 1 < argc) {
            req.height = std::atoi(argv[++i]);
        } else if (consume(a, "--steps") && i + 1 < argc) {
            req.steps = std::atoi(argv[++i]);
        } else if (consume(a, "--seed") && i + 1 < argc) {
            req.seed = std::atoi(argv[++i]);
        } else if (consume(a, "--cfg-scale") && i + 1 < argc) {
            req.cfg_scale = std::atof(argv[++i]);
        } else if (consume(a, "--extra-args") && i + 1 < argc) {
            req.extra_args = argv[++i];
        } else if (consume(a, "--native-exec-check")) {
            native_exec_check = true;
        } else if (consume(a, "--native-text-tokens") && i + 1 < argc) {
            native_text_tokens = std::atoll(argv[++i]);
        } else if (consume(a, "--native-real-block0-tokens") && i + 1 < argc) {
            native_real_block0_tokens = std::atoll(argv[++i]);
        } else if (consume(a, "--native-real-visual-block0-tokens") && i + 1 < argc) {
            native_real_visual_block0_tokens = std::atoll(argv[++i]);
        } else if (consume(a, "--native-full-chain-check")) {
            native_full_chain_check = true;
        } else if (consume(a, "--native-chain-text-tokens") && i + 1 < argc) {
            native_chain_text_tokens = std::atoll(argv[++i]);
        } else if (consume(a, "--native-chain-visual-tokens") && i + 1 < argc) {
            native_chain_visual_tokens = std::atoll(argv[++i]);
        } else if (consume(a, "--native-projection-patch-tokens") && i + 1 < argc) {
            native_projection_patch_tokens = std::atoll(argv[++i]);
        } else if (consume(a, "--native-projection-final-tokens") && i + 1 < argc) {
            native_projection_final_tokens = std::atoll(argv[++i]);
        } else if (consume(a, "--native-preview")) {
            native_preview = true;
        } else if (consume(a, "--native-skip-payloads")) {
            native_load_payloads = false;
        } else if (consume(a, "--dry-run")) {
            dry_run = true;
        } else {
            std::fprintf(stderr, "utopic_hidream_o1: unknown or incomplete argument: %s\n", a.c_str());
            usage(argv[0]);
            return 2;
        }
    }

    if (req.prompt.empty() || req.output_path.empty()) {
        usage(argv[0]);
        return 2;
    }
    if (req.width <= 0 || req.height <= 0 || req.width % cfg.patch_size != 0 || req.height % cfg.patch_size != 0) {
        std::fprintf(stderr, "utopic_hidream_o1: width/height must be positive multiples of %d\n", cfg.patch_size);
        return 2;
    }

    const utopic::HiDreamO1Shape shape = utopic::hidream_o1_shape_for_size(cfg, req.width, req.height);
    if (native_exec_check) {
        if (!utopic::hidream_o1_dir_exists(req.model_dir)) {
            std::fprintf(stderr, "utopic_hidream_o1: missing HiDream-O1 model dir: %s\n", req.model_dir.c_str());
            return 1;
        }
        utopic::HiDreamO1NativeExecutionSummary summary;
        std::string native_error;
        if (!utopic::hidream_o1_prepare_native_execution(req.model_dir,
                                                         req.width,
                                                         req.height,
                                                         native_text_tokens,
                                                         native_load_payloads,
                                                         &summary,
                                                         &native_error)) {
            std::fprintf(stderr, "utopic_hidream_o1: native execution prep failed: %s\n", native_error.c_str());
            return 1;
        }
        utopic::HiDreamO1RealBlockRunSummary block_summary;
        utopic::HiDreamO1RealBlockRunSummary visual_block_summary;
        utopic::HiDreamO1NativeChainRunSummary chain_summary;
        utopic::HiDreamO1NativeProjectionRunSummary projection_summary;
        if (native_load_payloads) {
            if (!utopic::hidream_o1_run_real_text_block_graph(req.model_dir,
                                                              0,
                                                              native_real_block0_tokens,
                                                              &block_summary,
                                                              &native_error)) {
                std::fprintf(stderr, "utopic_hidream_o1: native real block0 execution failed: %s\n", native_error.c_str());
                return 1;
            }
            if (!utopic::hidream_o1_run_real_visual_block_graph(req.model_dir,
                                                                0,
                                                                native_real_visual_block0_tokens,
                                                                &visual_block_summary,
                                                                &native_error)) {
                std::fprintf(stderr, "utopic_hidream_o1: native real visual block0 execution failed: %s\n", native_error.c_str());
                return 1;
            }
            if (!utopic::hidream_o1_run_native_projection_graph(req.model_dir,
                                                                native_projection_patch_tokens,
                                                                native_projection_final_tokens,
                                                                0.5f,
                                                                &projection_summary,
                                                                &native_error)) {
                std::fprintf(stderr, "utopic_hidream_o1: native projection execution failed: %s\n", native_error.c_str());
                return 1;
            }
            if (native_full_chain_check &&
                !utopic::hidream_o1_run_native_layer_chain(req.model_dir,
                                                           native_chain_text_tokens,
                                                           native_chain_visual_tokens,
                                                           &chain_summary,
                                                           &native_error)) {
                std::fprintf(stderr, "utopic_hidream_o1: native full layer-chain execution failed: %s\n", native_error.c_str());
                return 1;
            }
        }
        std::printf("utopic_hidream_o1 native_exec_check=OK model_dir=%s width=%d height=%d text_tokens=%lld image_tokens=%lld total_tokens=%lld text_layers=%d text_hidden=%d tensors=%lld catalog_tensors=%lld missing=%lld block0_tensors=%lld block0_payloads_loaded=%s block0_payload_bytes=%llu real_block0=%s real_block0_tokens=%lld real_block0_output_values=%lld real_block0_payload_bytes=%lld real_block0_l2=%.8f real_block0_max_abs=%.8f real_block0_checksum=%.8f real_visual_block0=%s real_visual_block0_tokens=%lld real_visual_block0_output_values=%lld real_visual_block0_payload_bytes=%lld real_visual_block0_l2=%.8f real_visual_block0_max_abs=%.8f real_visual_block0_checksum=%.8f native_projection=%s native_projection_payload_bytes=%lld native_projection_patch_values=%lld native_projection_timestep_values=%lld native_projection_final_values=%lld native_projection_patch_l2=%.8f native_projection_timestep_l2=%.8f native_projection_final_l2=%.8f native_full_chain=%s native_chain_text_layers=%d native_chain_visual_layers=%d native_chain_text_tokens=%lld native_chain_visual_tokens=%lld native_chain_text_payload_bytes=%lld native_chain_visual_payload_bytes=%lld native_chain_text_l2=%.8f native_chain_visual_l2=%.8f\n",
                    summary.model_dir.c_str(),
                    summary.width,
                    summary.height,
                    static_cast<long long>(summary.text_tokens),
                    static_cast<long long>(summary.image_tokens),
                    static_cast<long long>(summary.total_sequence_tokens),
                    summary.text_layers,
                    summary.text_hidden,
                    static_cast<long long>(summary.tensor_count),
                    static_cast<long long>(summary.catalog_tensor_count),
                    static_cast<long long>(summary.catalog_missing_tensor_count),
                    static_cast<long long>(summary.block0_tensor_count),
                    summary.block0_payloads_loaded ? "yes" : "no",
                    static_cast<unsigned long long>(summary.block0_payload_bytes),
                    native_load_payloads ? "OK" : "skipped",
                    static_cast<long long>(block_summary.sequence_tokens),
                    static_cast<long long>(block_summary.output_values),
                    static_cast<long long>(block_summary.payload_bytes),
                    block_summary.output_l2,
                    block_summary.output_max_abs,
                    block_summary.output_checksum,
                    native_load_payloads ? "OK" : "skipped",
                    static_cast<long long>(visual_block_summary.sequence_tokens),
                    static_cast<long long>(visual_block_summary.output_values),
                    static_cast<long long>(visual_block_summary.payload_bytes),
                    visual_block_summary.output_l2,
                    visual_block_summary.output_max_abs,
                    visual_block_summary.output_checksum,
                    native_load_payloads ? "OK" : "skipped",
                    static_cast<long long>(projection_summary.payload_bytes),
                    static_cast<long long>(projection_summary.patch_output_values),
                    static_cast<long long>(projection_summary.timestep_output_values),
                    static_cast<long long>(projection_summary.final_output_values),
                    projection_summary.patch_output_l2,
                    projection_summary.timestep_output_l2,
                    projection_summary.final_output_l2,
                    native_full_chain_check ? "OK" : "skipped",
                    chain_summary.text_layers,
                    chain_summary.visual_layers,
                    static_cast<long long>(chain_summary.text_tokens),
                    static_cast<long long>(chain_summary.visual_tokens),
                    static_cast<long long>(chain_summary.text_payload_bytes),
                    static_cast<long long>(chain_summary.visual_payload_bytes),
                    chain_summary.text_output_l2,
                    chain_summary.visual_output_l2);
        if (dry_run) return 0;
    }

    if (dry_run) {
        return 0;
    }
    if (native_preview) {
        if (!utopic::hidream_o1_dir_exists(req.model_dir)) {
            std::fprintf(stderr, "utopic_hidream_o1: missing HiDream-O1 model dir: %s\n", req.model_dir.c_str());
            return 1;
        }
        if (!ensure_parent_dir(req.output_path)) {
            std::fprintf(stderr, "utopic_hidream_o1: failed to create output directory for: %s\n", req.output_path.c_str());
            return 1;
        }
        utopic::HiDreamO1NativeImageRunSummary image_summary;
        std::string native_error;
        if (!utopic::hidream_o1_generate_native_preview_image(req.model_dir,
                                                              req.prompt,
                                                              req.output_path,
                                                              req.width,
                                                              req.height,
                                                              req.steps,
                                                              req.seed,
                                                              &image_summary,
                                                              &native_error)) {
            std::fprintf(stderr, "utopic_hidream_o1: native preview generation failed: %s\n", native_error.c_str());
            return 1;
        }
        std::printf("utopic_hidream_o1 native_preview=OK wrote=%s width=%d height=%d steps=%d text_tokens=%lld image_tokens=%lld total_tokens=%lld conditioning_values=%lld conditioning_l2=%.8f conditioning_checksum=%.8f patch_values=%lld final_patch_l2=%.8f final_patch_checksum=%.8f backend=native-preview-conditioned-no-torch-no-sdcpp\n",
                    image_summary.output_path.c_str(),
                    image_summary.width,
                    image_summary.height,
                    image_summary.steps,
                    static_cast<long long>(image_summary.text_tokens),
                    static_cast<long long>(image_summary.image_tokens),
                    static_cast<long long>(image_summary.total_sequence_tokens),
                    static_cast<long long>(image_summary.conditioning_values),
                    image_summary.conditioning_l2,
                    image_summary.conditioning_checksum,
                    static_cast<long long>(image_summary.patch_values),
                    image_summary.final_patch_l2,
                    image_summary.final_patch_checksum);
        return 0;
    }

    const std::string cmd = utopic::build_hidream_o1_command(req);
    std::fprintf(stderr,
                 "utopic_hidream_o1 model=%s backend=native-prep-plus-official-torch-reference-no-sdcpp model_dir=%s source_dir=%s torch_python=%s width=%d height=%d patch_tokens=%lld patch_dim=%d steps=%d cfg=%.3f seed=%d native_status=%s\n",
                 cfg.model_id,
                 req.model_dir.c_str(),
                 req.source_dir.c_str(),
                 req.torch_python.c_str(),
                 req.width,
                 req.height,
                 static_cast<long long>(shape.patch_tokens),
                 shape.patch_dim,
                 req.steps,
                 req.cfg_scale,
                 req.seed,
                 cfg.native_status);
    std::fprintf(stderr, "utopic_hidream_o1 command=%s\n", cmd.c_str());

    if (!utopic::hidream_o1_file_exists(req.torch_python)) {
        std::fprintf(stderr, "utopic_hidream_o1: missing torch python: %s\n", req.torch_python.c_str());
        return 1;
    }
    if (!utopic::hidream_o1_dir_exists(req.source_dir)) {
        std::fprintf(stderr, "utopic_hidream_o1: missing HiDream-O1 source dir: %s\n", req.source_dir.c_str());
        return 1;
    }
    if (!utopic::hidream_o1_dir_exists(req.model_dir)) {
        std::fprintf(stderr, "utopic_hidream_o1: missing HiDream-O1 model dir: %s\n", req.model_dir.c_str());
        return 1;
    }
    std::string patch_error;
    if (!utopic::hidream_o1_patch_official_source_for_flash_env(req.source_dir, &patch_error)) {
        std::fprintf(stderr, "utopic_hidream_o1: failed to prepare official HiDream source: %s\n", patch_error.c_str());
        return 1;
    }
    if (!ensure_parent_dir(req.output_path)) {
        std::fprintf(stderr, "utopic_hidream_o1: failed to create output directory for: %s\n", req.output_path.c_str());
        return 1;
    }
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::fprintf(stderr, "utopic_hidream_o1: official torch backend failed with code %d\n", rc);
        return 1;
    }
    if (!utopic::hidream_o1_file_exists(req.output_path)) {
        std::fprintf(stderr, "utopic_hidream_o1: backend returned success but output is missing: %s\n", req.output_path.c_str());
        return 1;
    }
    std::printf("utopic_hidream_o1 wrote %s\n", req.output_path.c_str());
    return 0;
}
