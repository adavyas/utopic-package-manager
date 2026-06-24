#include "artifact_runner_common.h"

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

namespace utopic_artifact {

namespace fs = std::filesystem;

const char * schema_version() {
    return "utopic-runner/v1";
}

const char * arg(int argc, char ** argv, const char * flag, const char * def) {
    for (int i = 1; i < argc - 1; ++i) {
        if (!strcmp(argv[i], flag)) {
            return argv[i + 1];
        }
    }
    return def;
}

bool flag_set(int argc, char ** argv, const char * flag) {
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], flag)) {
            return true;
        }
    }
    return false;
}

string executable_name(const char * path, const char * fallback) {
    if (!path || !path[0]) {
        return fallback;
    }
    string name(path);
    const size_t slash = name.find_last_of("/\\");
    if (slash != string::npos) {
        name = name.substr(slash + 1);
    }
#ifdef _WIN32
    const string suffix = ".exe";
    if (name.size() > suffix.size() && name.substr(name.size() - suffix.size()) == suffix) {
        name.resize(name.size() - suffix.size());
    }
#endif
    return name.empty() ? fallback : name;
}

json error_response(const string & code, const string & message, const json & detail) {
    return {
        {"ok", false},
        {"error", {
            {"code", code},
            {"message", message},
            {"detail", detail.is_object() ? detail : json::object()},
        }},
    };
}

static json contract_error(const string & message, const string & field) {
    return error_response("runner_failed", message, {{"field", field}});
}

bool read_request_file(const string & path, json & root, json & response) {
    if (path.empty()) {
        response = contract_error("--json-request is required", "json_request");
        return false;
    }
    try {
        std::ifstream in(path);
        if (!in) {
            response = error_response(
                "runner_failed",
                string("failed to open JSON request: ") + strerror(errno),
                {{"path", path}});
            return false;
        }
        in >> root;
    } catch (const std::exception & exc) {
        response = error_response(
            "runner_failed",
            string("invalid JSON request: ") + exc.what(),
            {{"path", path}});
        return false;
    }
    if (!root.is_object()) {
        response = contract_error("request must be a JSON object", "request");
        return false;
    }
    return true;
}

static bool required_string(const json & root, const char * key, string & out, json & response) {
    if (!root.contains(key) || !root[key].is_string() || root[key].get<string>().empty()) {
        response = contract_error(string(key) + " is required", key);
        return false;
    }
    out = root[key].get<string>();
    return true;
}

bool parse_contract(const json & root, const string & expected_task, artifact_request & req, json & response) {
    string schema;
    if (!required_string(root, "schema_version", schema, response)) {
        return false;
    }
    if (schema != schema_version()) {
        response = contract_error("unsupported schema_version", "schema_version");
        return false;
    }
    if (!required_string(root, "run_id", req.run_id, response)
        || !required_string(root, "task", req.task, response)
        || !required_string(root, "model", req.model, response)) {
        return false;
    }
    if (req.task != expected_task) {
        response = contract_error("task must be " + expected_task, "task");
        return false;
    }
    if (!root.contains("input") || !root["input"].is_object()) {
        response = contract_error("input must be an object", "input");
        return false;
    }
    if (!root.contains("options") || !root["options"].is_object()) {
        response = contract_error("options must be an object", "options");
        return false;
    }
    if (!required_string(root, "output_dir", req.output_dir, response)
        || !required_string(root, "progress_path", req.progress_path, response)) {
        return false;
    }
    req.input = root["input"];
    req.options = root["options"];
    return true;
}

bool ensure_output_dir(const artifact_request & req, json & response) {
    std::error_code ec;
    fs::create_directories(req.output_dir, ec);
    if (ec) {
        response = error_response(
            "runner_failed",
            string("failed to create output directory: ") + ec.message(),
            {{"output_dir", req.output_dir}});
        return false;
    }
    return true;
}

void write_progress_event(const artifact_request & req, const string & phase, const json & data) {
    std::ofstream out(req.progress_path, std::ios::app);
    if (!out) {
        return;
    }
    json event = {
        {"run_id", req.run_id},
        {"task", req.task},
        {"model", req.model},
        {"phase", phase},
    };
    if (data.is_object()) {
        event["data"] = data;
    }
    out << event.dump() << "\n";
}

void print_json(const json & payload) {
    printf("%s\n", payload.dump().c_str());
}

}  // namespace utopic_artifact
