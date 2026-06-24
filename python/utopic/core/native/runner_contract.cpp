#include "runner_contract.h"

namespace utopic {

const char * runner_task_name(runner_task task) {
    switch (task) {
        case RUNNER_TASK_CHAT:  return "chat";
        case RUNNER_TASK_IMAGE: return "image";
        case RUNNER_TASK_TTS:   return "tts";
        case RUNNER_TASK_MUSIC: return "music";
        case RUNNER_TASK_VIDEO: return "video";
        case RUNNER_TASK_MISC:  return "misc";
        default:                return "unknown";
    }
}

const char * runner_output_type_name(runner_output_type type) {
    switch (type) {
        case RUNNER_OUTPUT_TEXT:     return "text";
        case RUNNER_OUTPUT_IMAGE:    return "image";
        case RUNNER_OUTPUT_AUDIO:    return "audio";
        case RUNNER_OUTPUT_VIDEO:    return "video";
        case RUNNER_OUTPUT_ARTIFACT: return "artifact";
        default:                     return "unknown";
    }
}

const char * runner_error_code_name(runner_error_code code) {
    return runner_error_code_json_name(code);
}

bool runner_request_parse(const std::string & source, runner_request & request, runner_error & error) {
    request = runner_request();
    error   = runner_error();

    nlohmann::json root;
    try {
        root = nlohmann::json::parse(source);
    } catch (const nlohmann::json::parse_error & exc) {
        runner_contract_error_set(error, RUNNER_ERROR_RUNNER_FAILED, exc.what());
        return false;
    }

    return runner_request_parse_json(root, request, error);
}

std::string runner_response_to_json(const runner_response & response) {
    nlohmann::json artifacts = nlohmann::json::array();
    for (const runner_artifact & artifact : response.artifacts) {
        nlohmann::json item = {
            { "type", artifact.type },
            { "path", artifact.path },
        };
        if (!artifact.url.empty()) {
            item["url"] = artifact.url;
        }
        artifacts.push_back(item);
    }

    nlohmann::json root = {
        { "ok", response.ok },
        { "type", runner_output_type_name(response.type) },
        { "text", response.text },
        { "artifacts", artifacts },
        { "metrics", response.metrics.is_object() ? response.metrics : nlohmann::json::object() },
        { "backend", response.backend },
        { "device", response.device },
    };
    return root.dump();
}

std::string runner_error_to_json(const runner_error & error) {
    nlohmann::json root = {
        { "ok", false },
        {
            "error",
            {
                { "code", runner_error_code_name(error.code) },
                { "message", error.message },
                { "detail", error.detail.is_object() ? error.detail : nlohmann::json::object() },
            },
        },
    };
    return root.dump();
}

}  // namespace utopic
