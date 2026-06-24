#include "audio_engine.h"
#include "runner_plugin_api.h"

#include "nlohmann/json.hpp"
#include "sherpa-onnx/c-api/c-api.h"

#include <cstring>
#include <exception>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

using json = nlohmann::json;

static std::string sherpa_json_string(const json & obj, const char * key, const std::string & fallback = "") {
    if (!obj.is_object() || !obj.contains(key) || !obj[key].is_string()) {
        return fallback;
    }
    return obj[key].get<std::string>();
}

static int32_t sherpa_json_i32(const json & obj, const char * key, int32_t fallback) {
    if (!obj.is_object() || !obj.contains(key) || !obj[key].is_number_integer()) {
        return fallback;
    }
    return obj[key].get<int32_t>();
}

static float sherpa_json_float(const json & obj, const char * key, float fallback) {
    if (!obj.is_object() || !obj.contains(key) || !obj[key].is_number()) {
        return fallback;
    }
    return obj[key].get<float>();
}

static json sherpa_error(const std::string & code, const std::string & message, const json & detail = json::object()) {
    return {
        { "ok", false },
        { "error", {
            { "code", code },
            { "message", message },
            { "detail", detail.is_object() ? detail : json::object() },
        } },
    };
}

static int sherpa_write_response(const json & response, char * response_json, size_t response_json_size) {
    const std::string serialized = response.dump();
    if (response_json_size <= serialized.size()) {
        return UTOPIC_NATIVE_PLUGIN_BUFFER_TOO_SMALL;
    }
    std::memcpy(response_json, serialized.c_str(), serialized.size() + 1);
    return UTOPIC_NATIVE_PLUGIN_OK;
}

static std::string sherpa_default_output_path(const json & request) {
    const std::string output_dir = sherpa_json_string(request, "output_dir", ".");
    const std::string run_id     = sherpa_json_string(request, "run_id", "utopic");
    return (fs::path(output_dir) / (run_id + ".wav")).string();
}

static std::string sherpa_input_text(const json & request) {
    if (!request.is_object() || !request.contains("input") || !request["input"].is_object()) {
        return "";
    }
    const json & input = request["input"];
    std::string text   = sherpa_json_string(input, "input");
    if (text.empty()) {
        text = sherpa_json_string(input, "prompt");
    }
    if (text.empty()) {
        text = sherpa_json_string(input, "text");
    }
    return text;
}

static json sherpa_generate(const json & request) {
    const json options = request.contains("options") && request["options"].is_object()
        ? request["options"]
        : json::object();

    const std::string text = sherpa_input_text(request);
    if (text.empty()) {
        return sherpa_error("invalid_request", "TTS input text is required");
    }

    const std::string model_path  = sherpa_json_string(options, "model_path", sherpa_json_string(options, "kokoro_model_path"));
    const std::string voices_path = sherpa_json_string(options, "voices_path");
    const std::string tokens_path = sherpa_json_string(options, "tokens_path");
    const std::string data_dir    = sherpa_json_string(options, "data_dir");
    if (model_path.empty() || voices_path.empty() || tokens_path.empty() || data_dir.empty()) {
        return sherpa_error("missing_model", "Sherpa-ONNX Kokoro TTS requires model_path, voices_path, tokens_path, and data_dir", {
            { "model_path", !model_path.empty() },
            { "voices_path", !voices_path.empty() },
            { "tokens_path", !tokens_path.empty() },
            { "data_dir", !data_dir.empty() },
        });
    }

    const std::string provider = sherpa_json_string(options, "provider", "cpu");
    const std::string output_path = sherpa_json_string(options, "output_path", sherpa_default_output_path(request));
    const std::string rule_fsts   = sherpa_json_string(options, "rule_fsts");
    const std::string rule_fars   = sherpa_json_string(options, "rule_fars");
    const std::string dict_dir    = sherpa_json_string(options, "dict_dir");
    const std::string lexicon     = sherpa_json_string(options, "lexicon");
    const std::string lang        = sherpa_json_string(options, "lang");
    const std::string extra       = sherpa_json_string(options, "extra");

    SherpaOnnxOfflineTtsConfig config;
    std::memset(&config, 0, sizeof(config));
    config.model.kokoro.model        = model_path.c_str();
    config.model.kokoro.voices       = voices_path.c_str();
    config.model.kokoro.tokens       = tokens_path.c_str();
    config.model.kokoro.data_dir     = data_dir.c_str();
    config.model.kokoro.length_scale = sherpa_json_float(options, "length_scale", 1.0f);
    config.model.kokoro.dict_dir     = dict_dir.empty() ? nullptr : dict_dir.c_str();
    config.model.kokoro.lexicon      = lexicon.empty() ? nullptr : lexicon.c_str();
    config.model.kokoro.lang         = lang.empty() ? nullptr : lang.c_str();
    config.model.num_threads         = sherpa_json_i32(options, "num_threads", 2);
    config.model.debug               = sherpa_json_i32(options, "debug", 0);
    config.model.provider            = provider.c_str();
    config.rule_fsts                 = rule_fsts.empty() ? nullptr : rule_fsts.c_str();
    config.rule_fars                 = rule_fars.empty() ? nullptr : rule_fars.c_str();
    config.max_num_sentences         = sherpa_json_i32(options, "max_num_sentences", 2);
    config.silence_scale             = sherpa_json_float(options, "silence_scale", 0.2f);

    const SherpaOnnxOfflineTts * tts = SherpaOnnxCreateOfflineTts(&config);
    if (!tts) {
        return sherpa_error("backend_unavailable", "failed to create Sherpa-ONNX offline TTS engine", {
            { "model_path", model_path },
            { "provider", provider },
        });
    }

    SherpaOnnxGenerationConfig generation_config;
    std::memset(&generation_config, 0, sizeof(generation_config));
    generation_config.silence_scale = sherpa_json_float(options, "generation_silence_scale", config.silence_scale);
    generation_config.speed         = sherpa_json_float(options, "speed", 1.0f);
    generation_config.sid           = sherpa_json_i32(options, "speaker_id", sherpa_json_i32(options, "sid", 0));
    generation_config.num_steps     = sherpa_json_i32(options, "num_steps", 0);
    generation_config.extra         = extra.empty() ? nullptr : extra.c_str();

    const SherpaOnnxGeneratedAudio * audio =
        SherpaOnnxOfflineTtsGenerateWithConfig(tts, text.c_str(), &generation_config, nullptr, nullptr);
    if (!audio || !audio->samples || audio->n <= 0 || audio->sample_rate <= 0) {
        if (audio) {
            SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
        }
        SherpaOnnxDestroyOfflineTts(tts);
        return sherpa_error("runner_failed", "Sherpa-ONNX failed to generate audio");
    }

    utopic::audio_engine_wav_params wav_params;
    wav_params.output_path    = output_path;
    wav_params.samples        = audio->samples;
    wav_params.sample_count   = (size_t) audio->n;
    wav_params.sample_rate    = audio->sample_rate;
    wav_params.channel_count  = 1;

    utopic::audio_engine_result wav_result;
    const bool wav_ok = utopic::audio_engine_write_wav(wav_params, wav_result);

    SherpaOnnxDestroyOfflineTtsGeneratedAudio(audio);
    SherpaOnnxDestroyOfflineTts(tts);

    if (!wav_ok) {
        return sherpa_error("runner_failed", wav_result.error_message.empty() ? "failed to write WAV artifact" : wav_result.error_message);
    }

    return {
        { "ok", true },
        { "type", "audio" },
        { "text", "" },
        { "artifacts", json::array({
            {
                { "type", "audio/wav" },
                { "path", wav_result.artifact_path },
                { "url", std::string("file://") + wav_result.artifact_path },
                { "sample_rate", wav_result.sample_rate },
                { "sample_count", wav_result.sample_count },
                { "channel_count", wav_result.channel_count },
                { "duration_ms", wav_result.duration_ms },
            },
        }) },
        { "metrics", {
            { "engine", "sherpa-onnx" },
            { "model_family", "kokoro" },
            { "provider", provider },
            { "sample_rate", wav_result.sample_rate },
            { "sample_count", wav_result.sample_count },
            { "channel_count", wav_result.channel_count },
            { "duration_ms", wav_result.duration_ms },
        } },
    };
}

UTOPIC_NATIVE_PLUGIN_EXPORT int utopic_native_generate(
        const char * request_json,
        char *       response_json,
        size_t       response_json_size) {
    if (!request_json || !response_json || response_json_size == 0) {
        return UTOPIC_NATIVE_PLUGIN_BUFFER_TOO_SMALL;
    }

    json response;
    try {
        const json request = json::parse(request_json);
        response = sherpa_generate(request);
    } catch (const std::exception & exc) {
        response = sherpa_error("invalid_request", std::string("invalid TTS request JSON: ") + exc.what());
    }
    return sherpa_write_response(response, response_json, response_json_size);
}
