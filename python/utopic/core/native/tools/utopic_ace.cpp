#include "../core/ace_step.h"
#include "../core/ace_step_native.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

namespace {

const char* arg_value(int argc, char** argv, int& i) {
    if (i + 1 >= argc) return nullptr;
    return argv[++i];
}

std::string read_file(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s --prompt <text> --out <wav> [--seconds N] [--steps N] [--lyrics-file PATH] "
                 "[--seed N] [--models DIR] [--synth PATH]\n",
                 argv0);
}

}  // namespace

int main(int argc, char** argv) {
    utopic::AceStepNativeRequest req;
    const utopic::AceStepRuntimeConfig cfg = utopic::default_ace_step_runtime_config();
    req.steps = cfg.default_steps;
    req.vae_chunk = cfg.default_vae_chunk;
    req.vae_overlap = cfg.default_vae_overlap;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        const char* v = nullptr;
        if (a == "--prompt" && (v = arg_value(argc, argv, i))) req.prompt = v;
        else if (a == "--out" && (v = arg_value(argc, argv, i))) req.out_path = v;
        else if (a == "--seconds" && (v = arg_value(argc, argv, i))) req.seconds = std::atof(v);
        else if (a == "--steps" && (v = arg_value(argc, argv, i))) req.steps = std::atoi(v);
        else if (a == "--seed" && (v = arg_value(argc, argv, i))) req.seed = std::atoi(v);
        else if (a == "--models" && (v = arg_value(argc, argv, i))) req.models_dir = v;
        else if (a == "--synth" && (v = arg_value(argc, argv, i))) req.synth_binary = v;
        else if (a == "--lyrics" && (v = arg_value(argc, argv, i))) req.lyrics = v;
        else if (a == "--lyrics-file" && (v = arg_value(argc, argv, i))) req.lyrics = read_file(v);
        else if (a == "--synth-model" && (v = arg_value(argc, argv, i))) {
            req.synth_model = v;
            req.synth_model_file = v;
        } else if (a == "--guidance" && (v = arg_value(argc, argv, i))) req.guidance = std::atof(v);
        else if (a == "--shift" && (v = arg_value(argc, argv, i))) req.shift = std::atof(v);
        else if (a == "--help" || a == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            std::fprintf(stderr, "unknown or incomplete option: %s\n", a.c_str());
            usage(argv[0]);
            return 2;
        }
    }

    if (req.prompt.empty() || req.out_path.empty()) {
        usage(argv[0]);
        return 2;
    }

    const utopic::AceStepShape shape = utopic::ace_step_shape_for_seconds(cfg, req.seconds);
    std::fprintf(stderr,
                 "[utopic_ace] native ACE path seconds=%.3f sample_rate=%d steps=%d latent_frames=%d "
                 "vae_chunk=%d vae_overlap=%d\n",
                 req.seconds,
                 cfg.sample_rate,
                 req.steps,
                 shape.latent_frames,
                 req.vae_chunk,
                 req.vae_overlap);

    std::string error;
    if (!utopic::ace_step_run_native_synth(req, error)) {
        std::fprintf(stderr, "utopic_ace: %s\n", error.c_str());
        return 1;
    }
    std::printf("utopic_ace wrote %s\n", req.out_path.c_str());
    return 0;
}
