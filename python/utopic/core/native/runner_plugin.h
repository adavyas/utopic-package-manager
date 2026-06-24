#pragma once

#include "runner_contract.h"

#include "nlohmann/json.hpp"

namespace utopic_runner {

using json = nlohmann::json;
using utopic::runner_request;

bool runner_plugin_configured(const runner_request & req);
json runner_plugin_generate(const runner_request & req);

}  // namespace utopic_runner
