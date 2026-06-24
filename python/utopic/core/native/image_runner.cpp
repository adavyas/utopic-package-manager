// Native image artifact runner.
//
// This binary owns the C++ execution boundary for image generation requests:
//
//   ./utopic-image-runner --json-request request.json
//
// The Python layer remains the control plane. Image model execution belongs here
// once the C++ image backend is linked into the package-managed build.
#include "artifact_runner_common.h"

#include <cstdio>

using json = nlohmann::json;
using utopic_artifact::artifact_request;
using utopic_artifact::error_response;
using utopic_artifact::print_json;
using utopic_artifact::string;

static void print_help(const char * argv0) {
    const string name = utopic_artifact::executable_name(argv0, "utopic-image-runner");
    fprintf(stderr, "Utopic native image runner JSON contract\n");
    fprintf(stderr, "\n");
    fprintf(stderr, "Usage:\n");
    fprintf(stderr, "  %s --json-request request.json\n", name.c_str());
    fprintf(stderr, "\n");
    fprintf(stderr, "Contract:\n");
    fprintf(stderr, "  schema_version=%s\n", utopic_artifact::schema_version());
    fprintf(stderr, "  task=image\n");
    fprintf(stderr, "  output=PNG artifacts through the native image engine\n");
}

static bool prompt_present(const artifact_request & req) {
    return req.input.contains("prompt") && req.input["prompt"].is_string()
        && !req.input["prompt"].get<string>().empty();
}

static json image_engine_unavailable(const artifact_request & req) {
    return error_response(
        "backend_unavailable",
        "native image runner is installed, but no C++ image engine is linked into this build",
        {
            {"task", req.task},
            {"model", req.model},
            {"modality", "image"},
            {"runner", "utopic-image-runner"},
            {"engine", req.options.value("engine", "native-image")},
            {"native_status", req.options.value("native_status", "")},
            {"model_path", req.options.value("model_path", "")},
            {"output_dir", req.output_dir},
            {"next_step", "link a package-managed C++ image backend into utopic_image_runner"},
        });
}

int main(int argc, char ** argv) {
    if (utopic_artifact::flag_set(argc, argv, "--help") || utopic_artifact::flag_set(argc, argv, "-h")) {
        print_help(argv[0]);
        return 0;
    }

    const char * request_path = utopic_artifact::arg(argc, argv, "--json-request", "");
    json root;
    json response;
    if (!utopic_artifact::read_request_file(request_path, root, response)) {
        print_json(response);
        return 2;
    }

    artifact_request req;
    if (!utopic_artifact::parse_contract(root, "image", req, response)) {
        print_json(response);
        return 2;
    }
    if (!prompt_present(req)) {
        response = error_response(
            "runner_failed",
            "image generation requires a non-empty prompt",
            {{"field", "input.prompt"}, {"task", req.task}, {"model", req.model}});
        print_json(response);
        return 2;
    }
    if (!utopic_artifact::ensure_output_dir(req, response)) {
        print_json(response);
        return 2;
    }

    utopic_artifact::write_progress_event(req, "accepted", {{"runner", "utopic-image-runner"}});
    response = image_engine_unavailable(req);
    response["run_id"] = req.run_id;
    response["output_dir"] = req.output_dir;
    response["progress_path"] = req.progress_path;
    response["progress_url"] = "/v1/utopic/runs/" + req.run_id + "/events";
    print_json(response);
    return 1;
}
