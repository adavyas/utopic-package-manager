#pragma once

#include <string>

namespace utopic {

struct AceStepNativeRequest {
    std::string prompt;
    std::string lyrics = "[Instrumental]";
    std::string out_path;
    std::string models_dir;
    std::string synth_model = "acestep-v15-xl-turbo-Q8_0.gguf";
    std::string synth_model_file = "acestep-v15-xl-turbo-Q8_0.gguf";
    std::string synth_binary;
    double seconds = 30.0;
    int steps = 8;
    int seed = 5018;
    double guidance = 1.0;
    double shift = 1.0;
    int vae_chunk = 512;
    int vae_overlap = 64;
    std::string output_format = "wav16";
};

std::string ace_step_default_models_dir();
std::string ace_step_default_synth_binary();
bool ace_step_write_request_json(const std::string& path, const AceStepNativeRequest& req, std::string& error);
bool ace_step_run_native_synth(const AceStepNativeRequest& req, std::string& error);

}  // namespace utopic
