#include "ace_step_native.h"

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

#if __has_include(<filesystem>)
#include <filesystem>
namespace fs = std::filesystem;
#endif

namespace utopic {
namespace {

std::string home_dir() {
    const char* h = std::getenv("HOME");
    return h ? h : "";
}

bool exists_file(const std::string& path) {
    struct stat st {};
    return stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

bool exists_exec(const std::string& path) {
    return !path.empty() && access(path.c_str(), X_OK) == 0;
}

std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '\\': out += "\\\\"; break;
            case '"': out += "\\\""; break;
            case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            case '\n': out += "\\n"; break;
            case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out.push_back(static_cast<char>(c));
                }
        }
    }
    return out;
}

std::string shell_quote(const std::string& value) {
    std::string quoted = "'";
    for (char c : value) {
        if (c == '\'') {
            quoted += "'\\''";
        } else {
            quoted.push_back(c);
        }
    }
    quoted.push_back('\'');
    return quoted;
}

bool copy_file_bytes(const std::string& src, const std::string& dst, std::string& error) {
#if __has_include(<filesystem>)
    if (!fs::path(dst).parent_path().empty()) {
        std::error_code ec;
        fs::create_directories(fs::path(dst).parent_path(), ec);
        if (ec) {
            error = "failed to create output directory: " + ec.message();
            return false;
        }
    }
#endif
    std::ifstream in(src, std::ios::binary);
    if (!in) {
        error = "missing produced wav: " + src;
        return false;
    }
    std::ofstream out(dst, std::ios::binary);
    if (!out) {
        error = "failed to open output wav: " + dst;
        return false;
    }
    out << in.rdbuf();
    if (!out) {
        error = "failed to write output wav: " + dst;
        return false;
    }
    return true;
}

bool require_model_file(const std::string& models_dir, const std::string& name, std::string& error) {
    const std::string path = models_dir + "/" + name;
    if (!exists_file(path)) {
        error = "missing ACE GGUF model file: " + path;
        return false;
    }
    return true;
}

}  // namespace

std::string ace_step_default_models_dir() {
    const char* env = std::getenv("UTOPIC_ACE_GGUF_MODELS");
    if (env && env[0]) return env;
    return home_dir() + "/.cache/utopic/models/ace-step-1.5-gguf";
}

std::string ace_step_default_synth_binary() {
    const char* env = std::getenv("UTOPIC_ACE_SYNTH");
    if (env && env[0]) return env;
    const std::string root = home_dir() + "/.cache/utopic/acestep.cpp";
    const std::string optimized = root + "/build-aten/ace-synth";
    if (exists_exec(optimized)) return optimized;
    return root + "/build/ace-synth";
}

bool ace_step_write_request_json(const std::string& path, const AceStepNativeRequest& req, std::string& error) {
    std::ofstream out(path, std::ios::binary);
    if (!out) {
        error = "failed to open ACE request json: " + path;
        return false;
    }
    out << "{"
        << "\"synth_model\":\"" << json_escape(req.synth_model) << "\","
        << "\"caption\":\"" << json_escape(req.prompt) << "\","
        << "\"lyrics\":\"" << json_escape(req.lyrics) << "\","
        << "\"duration\":" << req.seconds << ","
        << "\"seed\":" << req.seed << ","
        << "\"vocal_language\":\"en\","
        << "\"inference_steps\":" << req.steps << ","
        << "\"guidance_scale\":" << req.guidance << ","
        << "\"shift\":" << req.shift << ","
        << "\"output_format\":\"" << json_escape(req.output_format) << "\","
        << "\"solver\":\"euler\""
        << "}\n";
    if (!out) {
        error = "failed to write ACE request json: " + path;
        return false;
    }
    return true;
}

bool ace_step_run_native_synth(const AceStepNativeRequest& input, std::string& error) {
    AceStepNativeRequest req = input;
    if (req.prompt.empty()) {
        error = "ACE prompt is required";
        return false;
    }
    if (req.out_path.empty()) {
        error = "ACE output path is required";
        return false;
    }
    if (req.models_dir.empty()) req.models_dir = ace_step_default_models_dir();
    if (req.synth_binary.empty()) req.synth_binary = ace_step_default_synth_binary();
    if (req.synth_model_file.empty()) req.synth_model_file = req.synth_model;

    if (!exists_exec(req.synth_binary)) {
        error = "missing native ACE executable: " + req.synth_binary;
        return false;
    }
    if (!require_model_file(req.models_dir, "Qwen3-Embedding-0.6B-Q8_0.gguf", error)) return false;
    if (!require_model_file(req.models_dir, req.synth_model_file, error)) return false;
    if (!require_model_file(req.models_dir, "vae-BF16.gguf", error)) return false;

    char tmp_template[] = "/tmp/utopic_ace_native.XXXXXX";
    char* tmp = mkdtemp(tmp_template);
    if (!tmp) {
        error = std::string("mkdtemp failed: ") + std::strerror(errno);
        return false;
    }
    const std::string workdir = tmp;
    const std::string request_path = workdir + "/request.json";
    if (!ace_step_write_request_json(request_path, req, error)) return false;

    std::ostringstream cmd;
    cmd << shell_quote(req.synth_binary)
        << " --models " << shell_quote(req.models_dir)
        << " --request " << shell_quote(request_path)
        << " --vae-chunk " << req.vae_chunk
        << " --vae-overlap " << req.vae_overlap;
    const int rc = std::system(cmd.str().c_str());
    if (rc != 0) {
        error = "native ACE synth failed with status " + std::to_string(rc);
        return false;
    }

    std::string produced = workdir + "/request0.wav";
    if (!exists_file(produced)) produced = workdir + "/request00.wav";
    return copy_file_bytes(produced, req.out_path, error);
}

}  // namespace utopic
