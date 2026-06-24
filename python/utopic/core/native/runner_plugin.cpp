#include "runner_plugin.h"

#include <algorithm>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

#if defined(_WIN32)
#    include <windows.h>
#else
#    include <dlfcn.h>
#endif

namespace utopic_runner {

using std::string;

typedef int (*runner_plugin_generate_fn)(const char * request_json, char * response_json, size_t response_json_size);

static const char * runner_plugin_task_name(utopic::runner_task task) {
    switch (task) {
        case utopic::RUNNER_TASK_CHAT:  return "chat";
        case utopic::RUNNER_TASK_IMAGE: return "image";
        case utopic::RUNNER_TASK_TTS:   return "tts";
        case utopic::RUNNER_TASK_MUSIC: return "music";
        case utopic::RUNNER_TASK_VIDEO: return "video";
        case utopic::RUNNER_TASK_MISC:  return "misc";
        default:                        return "unknown";
    }
}

static json runner_plugin_error(const string & code, const string & message, const json & detail = json::object()) {
    return {
        { "ok", false },
        { "error", {
            { "code", code },
            { "message", message },
            { "detail", detail.is_object() ? detail : json::object() },
        } },
    };
}

static string runner_plugin_option_string(const runner_request & req, const char * key) {
    if (!req.options.is_object() || !req.options.contains(key) || !req.options[key].is_string()) {
        return "";
    }
    return req.options[key].get<string>();
}

static json runner_plugin_request_json(const runner_request & req) {
    return {
        { "schema_version", req.schema_version },
        { "run_id", req.run_id },
        { "task", runner_plugin_task_name(req.task) },
        { "model", req.model },
        { "input", req.input },
        { "options", req.options },
        { "output_dir", req.output_dir },
        { "progress_path", req.progress_path },
        { "runner", req.runner },
    };
}

#if defined(_WIN32)
struct runner_plugin_library {
    HMODULE handle = nullptr;

    ~runner_plugin_library() {
        if (handle) {
            FreeLibrary(handle);
        }
    }
};

static bool runner_plugin_open(const string & path, runner_plugin_library & lib, string & error) {
    lib.handle = LoadLibraryA(path.c_str());
    if (!lib.handle) {
        error = "failed to load native plugin";
        return false;
    }
    return true;
}

static void * runner_plugin_symbol(runner_plugin_library & lib, const string & name, string & error) {
    void * symbol = (void *) GetProcAddress(lib.handle, name.c_str());
    if (!symbol) {
        error = "native plugin entrypoint was not found";
    }
    return symbol;
}
#else
struct runner_plugin_library {
    void * handle = nullptr;

    ~runner_plugin_library() {
        if (handle) {
            dlclose(handle);
        }
    }
};

static bool runner_plugin_open(const string & path, runner_plugin_library & lib, string & error) {
    dlerror();
    lib.handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!lib.handle) {
        const char * dl_error = dlerror();
        error = dl_error && dl_error[0] ? dl_error : "failed to load native plugin";
        return false;
    }
    return true;
}

static void * runner_plugin_symbol(runner_plugin_library & lib, const string & name, string & error) {
    dlerror();
    void * symbol = dlsym(lib.handle, name.c_str());
    const char * dl_error = dlerror();
    if (dl_error) {
        error = dl_error;
        return nullptr;
    }
    return symbol;
}
#endif

bool runner_plugin_configured(const runner_request & req) {
    return !runner_plugin_option_string(req, "native_library_path").empty();
}

json runner_plugin_generate(const runner_request & req) {
    const string library_path = runner_plugin_option_string(req, "native_library_path");
    if (library_path.empty()) {
        return json();
    }
    const string entrypoint = runner_plugin_option_string(req, "native_entrypoint").empty()
        ? string("utopic_native_generate")
        : runner_plugin_option_string(req, "native_entrypoint");

    runner_plugin_library lib;
    string load_error;
    if (!runner_plugin_open(library_path, lib, load_error)) {
        return runner_plugin_error("backend_unavailable", "failed to load native runner plugin", {
            { "native_library_path", library_path },
            { "detail", load_error },
        });
    }

    string symbol_error;
    void * symbol = runner_plugin_symbol(lib, entrypoint, symbol_error);
    if (!symbol) {
        return runner_plugin_error("backend_unavailable", "native runner plugin entrypoint is missing", {
            { "native_library_path", library_path },
            { "native_entrypoint", entrypoint },
            { "detail", symbol_error },
        });
    }
    runner_plugin_generate_fn generate = reinterpret_cast<runner_plugin_generate_fn>(symbol);

    const string request_json = runner_plugin_request_json(req).dump();
    std::vector<char> response_buffer(1024 * 1024, 0);
    int rc = 0;
    for (int attempt = 0; attempt < 5; ++attempt) {
        std::fill(response_buffer.begin(), response_buffer.end(), 0);
        rc = generate(request_json.c_str(), response_buffer.data(), response_buffer.size());
        if (rc != 2) {
            break;
        }
        response_buffer.resize(response_buffer.size() * 2);
    }
    if (rc != 0) {
        return runner_plugin_error("runner_failed", "native runner plugin failed", {
            { "native_library_path", library_path },
            { "native_entrypoint", entrypoint },
            { "exit_code", rc },
        });
    }

    json response;
    try {
        response = json::parse(response_buffer.data());
    } catch (const std::exception & exc) {
        return runner_plugin_error("runner_failed", string("native runner plugin returned invalid JSON: ") + exc.what(), {
            { "native_library_path", library_path },
            { "native_entrypoint", entrypoint },
        });
    }
    if (!response.is_object()) {
        return runner_plugin_error("runner_failed", "native runner plugin response must be a JSON object", {
            { "native_library_path", library_path },
            { "native_entrypoint", entrypoint },
        });
    }
    if (!response.contains("ok") || !response["ok"].is_boolean()) {
        return runner_plugin_error("runner_failed", "native runner plugin response must include boolean ok", {
            { "native_library_path", library_path },
            { "native_entrypoint", entrypoint },
        });
    }
    return response;
}

}  // namespace utopic_runner
