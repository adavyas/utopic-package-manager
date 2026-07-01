#include "core/hidream_o1_native.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/stat.h>

namespace {

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s --prompt TEXT --out image.png [--model checkpoint.safetensors] [--sd-cli sd-cli] "
                 "[--width 1024] [--height 1024] [--steps 28] [--seed 42] [--cfg-scale 1.0] [--extra-args ARGS] [--dry-run]\n",
                 argv0);
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
    req.sd_cli = utopic::hidream_o1_default_sd_cli();
    req.model_path = utopic::hidream_o1_default_model_path();
    req.width = cfg.default_width;
    req.height = cfg.default_height;
    req.steps = cfg.default_steps;
    req.seed = 42;
    req.cfg_scale = cfg.default_cfg_scale;

    bool dry_run = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (consume(a, "--help")) {
            usage(argv[0]);
            return 0;
        } else if (consume(a, "--prompt") && i + 1 < argc) {
            req.prompt = argv[++i];
        } else if (consume(a, "--out") && i + 1 < argc) {
            req.output_path = argv[++i];
        } else if (consume(a, "--model") && i + 1 < argc) {
            req.model_path = argv[++i];
        } else if (consume(a, "--sd-cli") && i + 1 < argc) {
            req.sd_cli = argv[++i];
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
    const std::string cmd = utopic::build_hidream_o1_command(req);
    std::fprintf(stderr,
                 "utopic_hidream_o1 model=%s backend=sd.cpp-native checkpoint=%s sd_cli=%s width=%d height=%d patch_tokens=%lld patch_dim=%d steps=%d cfg=%.3f seed=%d\n",
                 cfg.model_id,
                 req.model_path.c_str(),
                 req.sd_cli.c_str(),
                 req.width,
                 req.height,
                 static_cast<long long>(shape.patch_tokens),
                 shape.patch_dim,
                 req.steps,
                 req.cfg_scale,
                 req.seed);
    std::fprintf(stderr, "utopic_hidream_o1 command=%s\n", cmd.c_str());

    if (dry_run) {
        return 0;
    }
    if (!utopic::hidream_o1_file_exists(req.sd_cli)) {
        std::fprintf(stderr, "utopic_hidream_o1: missing sd.cpp native backend binary: %s\n", req.sd_cli.c_str());
        return 1;
    }
    if (!utopic::hidream_o1_file_exists(req.model_path)) {
        std::fprintf(stderr, "utopic_hidream_o1: missing HiDream-O1 checkpoint: %s\n", req.model_path.c_str());
        return 1;
    }
    if (!ensure_parent_dir(req.output_path)) {
        std::fprintf(stderr, "utopic_hidream_o1: failed to create output directory for: %s\n", req.output_path.c_str());
        return 1;
    }
    const int rc = std::system(cmd.c_str());
    if (rc != 0) {
        std::fprintf(stderr, "utopic_hidream_o1: native backend failed with code %d\n", rc);
        return 1;
    }
    if (!utopic::hidream_o1_file_exists(req.output_path)) {
        std::fprintf(stderr, "utopic_hidream_o1: backend returned success but output is missing: %s\n", req.output_path.c_str());
        return 1;
    }
    std::printf("utopic_hidream_o1 wrote %s\n", req.output_path.c_str());
    return 0;
}
