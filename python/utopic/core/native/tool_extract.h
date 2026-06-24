// tool_extract.h - tolerant tool-call extractor for diffusion LM output.
//
// Empirical basis (LLaDA-8B, 2026-06-18): diffusion tool-calling reasoning is
// SOLVED (100% function selection incl. multi-call), but strict-JSON validity is
// only ~60% (missing commas/quotes, ```json fences, ]-vs-} mismatch, truncation).
// The semantic content is present even when the JSON envelope is malformed, so we
// harvest it directly rather than requiring a well-formed parse: regex out each
// "name" + its "arguments" object and parse the args leniently. Recovers ~100%
// function selection / ~9-of-10 args with zero model/kernel changes ("own the
// output decode"). Re-emits clean OpenAI-style JSON.
#pragma once
#include <string>
#include <vector>
#include <regex>
#include <utility>

namespace toolx {

struct ToolCall {
    std::string name;
    std::vector<std::pair<std::string, std::string>> args; // ordered key->value
};

// Strip markdown code fences (```json ... ```), which the model frequently adds.
inline std::string strip_fences(std::string s) {
    s = std::regex_replace(s, std::regex("```[a-zA-Z]*"), "");
    return s;
}

// Balance-match from the '{' at or after `open` (which must index a '{'); returns
// the index just past the matching '}', or s.size() if unterminated (truncation-safe).
inline size_t match_brace(const std::string& s, size_t open) {
    int depth = 0;
    for (size_t i = open; i < s.size(); ++i) {
        if (s[i] == '{') depth++;
        else if (s[i] == '}') { if (--depth == 0) return i + 1; }
    }
    return s.size();
}

inline std::vector<ToolCall> extract(const std::string& raw) {
    std::string s = strip_fences(raw);
    std::vector<ToolCall> calls;
    // each call = "name":"X" ... "arguments":{ ... }
    std::regex head("\"name\"\\s*:\\s*\"(\\w+)\"\\s*,\\s*\"arguments\"\\s*:\\s*\\{");
    std::regex kv("\"(\\w+)\"\\s*:\\s*(\"[^\"]*\"|-?[0-9.]+|[A-Za-z0-9_./@:+-]+)");
    auto begin = std::sregex_iterator(s.begin(), s.end(), head);
    auto end   = std::sregex_iterator();
    for (auto it = begin; it != end; ++it) {
        ToolCall tc;
        tc.name = (*it)[1].str();
        size_t obrace = it->position(0) + it->length(0) - 1; // index of the '{'
        size_t close  = match_brace(s, obrace);
        std::string block = s.substr(obrace + 1, close > obrace + 1 ? close - obrace - 2 : 0);
        for (auto k = std::sregex_iterator(block.begin(), block.end(), kv);
             k != std::sregex_iterator(); ++k) {
            std::string val = (*k)[2].str();
            if (val.size() >= 2 && val.front() == '"' && val.back() == '"')
                val = val.substr(1, val.size() - 2);
            tc.args.emplace_back((*k)[1].str(), val);
        }
        calls.push_back(std::move(tc));
    }
    return calls;
}

inline std::string json_escape(const std::string& v) {
    std::string o;
    for (char c : v) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else o += c;
    }
    return o;
}

// Re-emit as clean, always-valid OpenAI-style JSON: {"tool_calls":[...]}.
inline std::string to_openai_json(const std::vector<ToolCall>& calls) {
    std::string o = "{\"tool_calls\":[";
    for (size_t i = 0; i < calls.size(); ++i) {
        if (i) o += ",";
        o += "{\"name\":\"" + json_escape(calls[i].name) + "\",\"arguments\":{";
        for (size_t j = 0; j < calls[i].args.size(); ++j) {
            if (j) o += ",";
            o += "\"" + json_escape(calls[i].args[j].first) + "\":\""
               + json_escape(calls[i].args[j].second) + "\"";
        }
        o += "}}";
    }
    o += "]}";
    return o;
}

} // namespace toolx
