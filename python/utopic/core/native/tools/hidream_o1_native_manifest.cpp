#include "core/hidream_o1_native.h"

#include <cstdio>
#include <string>
#include <vector>

namespace {

void usage(const char* argv0) {
    std::fprintf(stderr,
                 "usage: %s [--model-dir DIR] [--headers] [--require-files]\n"
                 "Inspect the native HiDream-O1 Dev-2604 shard index without invoking sd.cpp.\n",
                 argv0);
}

std::string join_path(const std::string& a, const std::string& b) {
    if (a.empty()) return b;
    if (a.back() == '/') return a + b;
    return a + "/" + b;
}

std::string shape_string(const std::vector<int64_t>& shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) out += ",";
        out += std::to_string(shape[i]);
    }
    out += "]";
    return out;
}

}  // namespace

int main(int argc, char** argv) {
    std::string model_dir = utopic::hidream_o1_default_model_dir();
    bool headers = false;
    bool require_files = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help") {
            usage(argv[0]);
            return 0;
        } else if (arg == "--model-dir" && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (arg == "--headers") {
            headers = true;
        } else if (arg == "--require-files") {
            require_files = true;
        } else {
            std::fprintf(stderr, "hidream_o1_native_manifest: unknown or incomplete argument: %s\n", arg.c_str());
            usage(argv[0]);
            return 2;
        }
    }

    const utopic::HiDreamO1RuntimeConfig cfg = utopic::default_hidream_o1_runtime_config();
    std::printf("model=%s\nrepo=%s\nstatus=%s\nmodel_dir=%s\n",
                cfg.model_id,
                cfg.hf_repo,
                cfg.native_status,
                model_dir.c_str());
    std::printf("native_defaults width=%d height=%d steps=%d guidance=%.1f shift=%.1f scheduler=%s noise_scale=%.1f..%.1f noise_clip_std=%.1f\n",
                cfg.default_width,
                cfg.default_height,
                cfg.default_steps,
                cfg.default_guidance_scale,
                cfg.default_shift,
                cfg.default_scheduler,
                cfg.default_noise_scale_start,
                cfg.default_noise_scale_end,
                cfg.default_noise_clip_std);

    utopic::HiDreamO1ShardManifest manifest;
    if (!utopic::load_hidream_o1_shard_manifest(model_dir, &manifest)) {
        std::fprintf(stderr, "hidream_o1_native_manifest: %s\n", manifest.error.c_str());
        return 1;
    }

    std::printf("index=%s\ntensors=%zu\nshards=%zu\n",
                manifest.index_path.c_str(),
                manifest.entries.size(),
                manifest.shard_files.size());
    utopic::HiDreamO1NativeModelLayout layout;
    if (!utopic::load_hidream_o1_native_model_layout(model_dir, &layout)) {
        std::fprintf(stderr, "hidream_o1_native_manifest: layout error: %s\n", layout.error.c_str());
        return 1;
    }
    std::printf("layout text_layers=%d text_hidden=%d text_heads=%d text_kv_heads=%d text_head_dim=%d text_intermediate=%d vision_layers=%d vision_hidden=%d vision_heads=%d vision_patch=%d vision_out_hidden=%d\n",
                layout.text.num_hidden_layers,
                layout.text.hidden_size,
                layout.text.num_attention_heads,
                layout.text.num_key_value_heads,
                layout.text.head_dim,
                layout.text.intermediate_size,
                layout.vision.depth,
                layout.vision.hidden_size,
                layout.vision.num_heads,
                layout.vision.patch_size,
                layout.vision.out_hidden_size);
    std::printf("layout_tensors total=%lld text=%lld vision=%lld timestep=%lld final=%lld lm_head=%lld block0_required=%s\n",
                static_cast<long long>(layout.tensor_count),
                static_cast<long long>(layout.text_tensor_count),
                static_cast<long long>(layout.vision_tensor_count),
                static_cast<long long>(layout.timestep_tensor_count),
                static_cast<long long>(layout.final_layer_tensor_count),
                static_cast<long long>(layout.lm_head_tensor_count),
                layout.has_required_text_block0 ? "yes" : "no");
    for (const std::string& shard : manifest.shard_files) {
        const std::string path = join_path(model_dir, shard);
        const bool exists = utopic::hidream_o1_file_exists(path);
        std::printf("shard=%s exists=%s\n", shard.c_str(), exists ? "yes" : "no");
        if (require_files && !exists) {
            std::fprintf(stderr, "hidream_o1_native_manifest: missing shard: %s\n", path.c_str());
            return 1;
        }
        if (headers && exists) {
            const utopic::HiDreamO1SafetensorsHeader h = utopic::inspect_hidream_o1_safetensors_header(path);
            if (!h.error.empty()) {
                std::fprintf(stderr, "hidream_o1_native_manifest: header error for %s: %s\n", path.c_str(), h.error.c_str());
                return 1;
            }
            std::printf("header shard=%s header_bytes=%llu tensors=%lld",
                        shard.c_str(),
                        static_cast<unsigned long long>(h.header_bytes),
                        static_cast<long long>(h.tensor_count));
            for (const auto& kv : h.dtype_counts) {
                std::printf(" dtype_%s=%lld", kv.first.c_str(), static_cast<long long>(kv.second));
            }
            std::printf("\n");
        }
    }
    if (headers) {
        utopic::HiDreamO1TensorCatalog catalog;
        if (!utopic::load_hidream_o1_tensor_catalog(model_dir, &catalog)) {
            std::fprintf(stderr, "hidream_o1_native_manifest: tensor catalog error: %s\n", catalog.error.c_str());
            return 1;
        }

        long long bf16 = 0;
        long long f32 = 0;
        for (const utopic::HiDreamO1TensorInfo& tensor : catalog.tensors) {
            if (tensor.dtype == "BF16") bf16++;
            if (tensor.dtype == "F32") f32++;
        }
        std::printf("catalog_tensors total=%zu missing=%lld dtype_BF16=%lld dtype_F32=%lld\n",
                    catalog.tensors.size(),
                    static_cast<long long>(catalog.missing_tensor_count),
                    bf16,
                    f32);

        std::vector<utopic::HiDreamO1TensorInfo> block0;
        if (!utopic::load_hidream_o1_text_block_tensors(model_dir, 0, &block0)) {
            std::fprintf(stderr, "hidream_o1_native_manifest: failed to resolve text block0 tensors\n");
            return 1;
        }
        for (const utopic::HiDreamO1TensorInfo& tensor : block0) {
            std::printf("block0_tensor=%s dtype=%s shape=%s shard=%s rel=[%llu,%llu] abs=[%llu,%llu]\n",
                        tensor.tensor_name.c_str(),
                        tensor.dtype.c_str(),
                        shape_string(tensor.shape).c_str(),
                        tensor.shard_file.c_str(),
                        static_cast<unsigned long long>(tensor.data_offsets[0]),
                        static_cast<unsigned long long>(tensor.data_offsets[1]),
                        static_cast<unsigned long long>(tensor.absolute_data_begin),
                        static_cast<unsigned long long>(tensor.absolute_data_end));
        }
    }
    return 0;
}
