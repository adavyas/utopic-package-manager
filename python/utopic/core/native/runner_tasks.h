#pragma once

#include "nlohmann/json.hpp"

#include <string>

namespace utopic_runner {

using json = nlohmann::json;
using std::string;

struct runner_request {
    string run_id;
    string task;
    string model;
    json input;
    json options;
    string output_dir;
    string progress_path;
    string runner;
};

string host_backend();
string host_device();

json backend_preflight_error(const json & root, const string & runner_name);
json capacity_preflight_error(const json & root, const string & runner_name);
json run_planned_native_task(const runner_request & req);

}  // namespace utopic_runner
