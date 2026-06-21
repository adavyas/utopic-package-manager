#pragma once

#include "nlohmann/json.hpp"

#include <string>

// Convert a flat or nested JSON Schema object into the typed scaffold used by
// Utopic's diffusion-constrained decoder.
//   {type:object, properties:{name:{type:string}, age:{type:integer}}}
// becomes:
//   {"name": "__s__", "age": __d__}
inline std::string schema_to_skeleton(const nlohmann::json & js) {
    if (!js.is_object() || !js.contains("properties") || !js["properties"].is_object()) return "";

    std::string out = "{";
    bool first = true;
    for (auto & it : js["properties"].items()) {
        if (!first) out += ", ";
        first = false;

        const nlohmann::json & v = it.value();
        const std::string type = v.is_object() && v.contains("type") && v["type"].is_string()
            ? v["type"].get<std::string>()
            : "string";

        out += "\"" + it.key() + "\": ";
        if (type == "object") {
            const std::string nested = schema_to_skeleton(v);
            out += nested.empty() ? "{}" : nested;
        } else if (type == "integer") {
            out += "__d__";
        } else if (type == "number") {
            out += "__n__";
        } else {
            out += "\"__s__\"";
        }
    }
    out += "}";
    return out;
}
