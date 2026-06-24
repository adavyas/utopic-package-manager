#pragma once

#include "nlohmann/json.hpp"

#include <string>

namespace utopic_artifact {

using json = nlohmann::json;
using std::string;

struct artifact_request {
    string run_id;
    string task;
    string model;
    json input;
    json options;
    string output_dir;
    string progress_path;
};

const char * schema_version();

const char * arg(int argc, char ** argv, const char * flag, const char * def);
bool flag_set(int argc, char ** argv, const char * flag);
string executable_name(const char * path, const char * fallback);
json error_response(const string & code, const string & message, const json & detail = json::object());
bool read_request_file(const string & path, json & root, json & response);
bool parse_contract(const json & root, const string & expected_task, artifact_request & req, json & response);
bool ensure_output_dir(const artifact_request & req, json & response);
void write_progress_event(const artifact_request & req, const string & phase, const json & data = json::object());
void print_json(const json & payload);

}  // namespace utopic_artifact
