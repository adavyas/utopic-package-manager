#pragma once

#include "nlohmann/json.hpp"

#include <string>
#include <vector>

namespace utopic {

enum runner_task {
    RUNNER_TASK_UNKNOWN = 0,
    RUNNER_TASK_CHAT,
    RUNNER_TASK_IMAGE,
    RUNNER_TASK_TTS,
    RUNNER_TASK_MUSIC,
    RUNNER_TASK_VIDEO,
    RUNNER_TASK_MISC,
};

enum runner_output_type {
    RUNNER_OUTPUT_UNKNOWN = 0,
    RUNNER_OUTPUT_TEXT,
    RUNNER_OUTPUT_IMAGE,
    RUNNER_OUTPUT_AUDIO,
    RUNNER_OUTPUT_VIDEO,
    RUNNER_OUTPUT_ARTIFACT,
};

enum runner_error_code {
    RUNNER_ERROR_UNKNOWN = 0,
    RUNNER_ERROR_MISSING_MODEL,
    RUNNER_ERROR_OOM,
    RUNNER_ERROR_BACKEND_UNAVAILABLE,
    RUNNER_ERROR_UNSUPPORTED_MODEL,
    RUNNER_ERROR_RUNNER_FAILED,
};

struct runner_artifact {
    std::string type;
    std::string path;
    std::string url;
};

struct runner_error {
    runner_error_code code = RUNNER_ERROR_UNKNOWN;
    std::string       message;
    nlohmann::json    detail = nlohmann::json::object();
};

struct runner_request {
    std::string    schema_version;
    std::string    run_id;
    runner_task    task    = RUNNER_TASK_UNKNOWN;
    std::string    model;
    nlohmann::json input   = nlohmann::json::object();
    nlohmann::json options = nlohmann::json::object();
    std::string    output_dir;
    std::string    progress_path;
    std::string    runner;
};

struct runner_response {
    bool                         ok      = true;
    runner_output_type           type    = RUNNER_OUTPUT_UNKNOWN;
    std::string                  text;
    std::vector<runner_artifact> artifacts;
    nlohmann::json               metrics = nlohmann::json::object();
    std::string                  backend;
    std::string                  device;
};

inline const char * runner_contract_schema_version() {
    return "utopic-runner/v1";
}

inline const char * runner_error_code_json_name(runner_error_code code) {
    switch (code) {
        case RUNNER_ERROR_MISSING_MODEL:       return "missing_model";
        case RUNNER_ERROR_OOM:                 return "oom";
        case RUNNER_ERROR_BACKEND_UNAVAILABLE: return "backend_unavailable";
        case RUNNER_ERROR_UNSUPPORTED_MODEL:   return "unsupported_model";
        case RUNNER_ERROR_RUNNER_FAILED:       return "runner_failed";
        default:                               return "runner_failed";
    }
}

inline runner_task runner_task_from_json_name(const std::string & name) {
    if (name == "chat") {
        return RUNNER_TASK_CHAT;
    }
    if (name == "image") {
        return RUNNER_TASK_IMAGE;
    }
    if (name == "tts") {
        return RUNNER_TASK_TTS;
    }
    if (name == "music") {
        return RUNNER_TASK_MUSIC;
    }
    if (name == "video") {
        return RUNNER_TASK_VIDEO;
    }
    if (name == "misc") {
        return RUNNER_TASK_MISC;
    }
    return RUNNER_TASK_UNKNOWN;
}

inline void runner_contract_error_set(runner_error & error, runner_error_code code, const std::string & message) {
    error.code    = code;
    error.message = message;
    error.detail  = nlohmann::json::object();
}

inline bool runner_json_string_field(const nlohmann::json & obj,
                                     const char *           key,
                                     std::string &          value,
                                     runner_error &         error,
                                     bool                   required) {
    if (!obj.contains(key)) {
        if (required) {
            runner_contract_error_set(error, RUNNER_ERROR_RUNNER_FAILED, std::string("missing required field: ") + key);
            return false;
        }
        value.clear();
        return true;
    }
    if (!obj[key].is_string()) {
        runner_contract_error_set(error, RUNNER_ERROR_RUNNER_FAILED, std::string("field must be a string: ") + key);
        return false;
    }
    value = obj[key].get<std::string>();
    return true;
}

inline bool runner_request_parse_json(const nlohmann::json & root, runner_request & request, runner_error & error) {
    request = runner_request();
    error   = runner_error();

    if (!root.is_object()) {
        runner_contract_error_set(error, RUNNER_ERROR_RUNNER_FAILED, "request must be a JSON object");
        return false;
    }
    if (!runner_json_string_field(root, "schema_version", request.schema_version, error, true)) {
        return false;
    }
    if (request.schema_version != runner_contract_schema_version()) {
        runner_contract_error_set(error, RUNNER_ERROR_RUNNER_FAILED, "unsupported schema_version");
        error.detail = { { "schema_version", request.schema_version } };
        return false;
    }
    if (!runner_json_string_field(root, "run_id", request.run_id, error, true)) {
        return false;
    }

    std::string task_name;
    if (!runner_json_string_field(root, "task", task_name, error, true)) {
        return false;
    }
    request.task = runner_task_from_json_name(task_name);
    if (request.task == RUNNER_TASK_UNKNOWN) {
        runner_contract_error_set(error, RUNNER_ERROR_UNSUPPORTED_MODEL, "unsupported runner task");
        error.detail = { { "task", task_name } };
        return false;
    }

    if (!runner_json_string_field(root, "model", request.model, error, true)) {
        return false;
    }
    if (request.model.empty()) {
        runner_contract_error_set(error, RUNNER_ERROR_MISSING_MODEL, "model is required");
        return false;
    }
    if (!runner_json_string_field(root, "output_dir", request.output_dir, error, true)) {
        return false;
    }
    if (!runner_json_string_field(root, "progress_path", request.progress_path, error, true)) {
        return false;
    }

    if (!root.contains("input")) {
        runner_contract_error_set(error, RUNNER_ERROR_RUNNER_FAILED, "missing required field: input");
        return false;
    }
    request.input = root["input"];
    if (!request.input.is_object()) {
        runner_contract_error_set(error, RUNNER_ERROR_RUNNER_FAILED, "input must be an object");
        return false;
    }
    if (!root.contains("options")) {
        runner_contract_error_set(error, RUNNER_ERROR_RUNNER_FAILED, "missing required field: options");
        return false;
    }
    request.options = root["options"];
    if (!request.options.is_object()) {
        runner_contract_error_set(error, RUNNER_ERROR_RUNNER_FAILED, "options must be an object");
        return false;
    }

    return true;
}

bool        runner_request_parse(const std::string & source, runner_request & request, runner_error & error);
std::string runner_response_to_json(const runner_response & response);
std::string runner_error_to_json(const runner_error & error);

const char * runner_task_name(runner_task task);
const char * runner_output_type_name(runner_output_type type);
const char * runner_error_code_name(runner_error_code code);

}  // namespace utopic
