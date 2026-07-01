#include "hidream_o1_native.h"

#include <cstdlib>
#include <sstream>
#include <string>
#include <sys/stat.h>

namespace utopic {

namespace {

std::string env_or(const char* name, const std::string& fallback) {
    const char* value = std::getenv(name);
    return value && value[0] ? std::string(value) : fallback;
}

std::string home_path(const char* suffix) {
    const char* home = std::getenv("HOME");
    return std::string(home && home[0] ? home : "") + suffix;
}

std::string shell_quote(const std::string& value) {
    std::string quoted = "'";
    for (char c : value) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

}  // namespace

HiDreamO1RuntimeConfig default_hidream_o1_runtime_config() {
    return HiDreamO1RuntimeConfig{};
}

HiDreamO1Shape hidream_o1_shape_for_size(const HiDreamO1RuntimeConfig& cfg, int width, int height) {
    HiDreamO1Shape shape;
    shape.width = width;
    shape.height = height;
    shape.patch_size = cfg.patch_size;
    shape.patch_dim = cfg.patch_size * cfg.patch_size * 3;
    if (width > 0 && height > 0 && cfg.patch_size > 0) {
        shape.patch_tokens = static_cast<int64_t>(width / cfg.patch_size) *
                             static_cast<int64_t>(height / cfg.patch_size);
        shape.pixel_values = static_cast<int64_t>(width) * static_cast<int64_t>(height) * 3;
    }
    return shape;
}

std::string hidream_o1_default_model_path() {
    return env_or("UTOPIC_HIDREAM_MODEL",
                  home_path("/.cache/utopic/models/hidream-o1/hidream_o1_image_dev_bf16.safetensors"));
}

std::string hidream_o1_default_sd_cli() {
    return env_or("UTOPIC_HIDREAM_SDCLI", home_path("/stable-diffusion.cpp/build-gb10/bin/sd-cli"));
}

std::string build_hidream_o1_command(const HiDreamO1RunRequest& req) {
    const HiDreamO1RuntimeConfig cfg = default_hidream_o1_runtime_config();
    const int width = req.width > 0 ? req.width : cfg.default_width;
    const int height = req.height > 0 ? req.height : cfg.default_height;
    const int steps = req.steps > 0 ? req.steps : cfg.default_steps;
    const float cfg_scale = req.cfg_scale > 0.0f ? req.cfg_scale : cfg.default_cfg_scale;

    std::ostringstream cmd;
    cmd << shell_quote(req.sd_cli)
        << " -m " << shell_quote(req.model_path)
        << " -p " << shell_quote(req.prompt)
        << " -o " << shell_quote(req.output_path)
        << " -H " << height
        << " -W " << width
        << " --steps " << steps
        << " -s " << req.seed
        << " --cfg-scale " << cfg_scale
        << " -v";
    if (!req.extra_args.empty()) {
        cmd << " " << req.extra_args;
    }
    return cmd.str();
}

bool hidream_o1_file_exists(const std::string& path) {
    struct stat st {};
    return !path.empty() && stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

}  // namespace utopic
