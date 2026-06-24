#pragma once

#include "runner_contract.h"

#include "nlohmann/json.hpp"

#include <string>

namespace utopic_runner {

using json = nlohmann::json;
using std::string;
using utopic::runner_request;

string host_backend();
string host_device();

json backend_preflight_error(const json & root, const string & runner_name);
json capacity_preflight_error(const json & root, const string & runner_name);
json run_artifact_task(const runner_request & req, const json & root);

}  // namespace utopic_runner
