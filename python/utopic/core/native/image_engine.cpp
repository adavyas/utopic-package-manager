#include "image_engine.h"

#include "stable-diffusion.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

namespace utopic {

static const char * cstr_or_null(const std::string & value) {
    return value.empty() ? nullptr : value.c_str();
}

static const char * cstr_or_empty(const std::string & value) {
    return value.empty() ? "" : value.c_str();
}

struct sd_ctx_guard {
    sd_ctx_t * ctx = nullptr;

    ~sd_ctx_guard() {
        if (ctx) {
            free_sd_ctx(ctx);
        }
    }
};

struct sd_images_guard {
    sd_image_t * images = nullptr;
    int          count  = 0;

    ~sd_images_guard() {
        if (images) {
            free_sd_images(images, count);
        }
    }
};

static void image_engine_fail(image_engine_result & result, const char * message) {
    result.ok            = false;
    result.error_message = message ? message : "image generation failed";
}

static bool image_engine_validate(const image_engine_params & params, image_engine_result & result) {
    if (params.prompt.empty()) {
        image_engine_fail(result, "prompt is required");
        return false;
    }
    if (params.output_path.empty()) {
        image_engine_fail(result, "output_path is required");
        return false;
    }
    if (params.model_path.empty() && params.diffusion_model_path.empty()) {
        image_engine_fail(result, "model_path or diffusion_model_path is required");
        return false;
    }
    if (params.width <= 0 || params.height <= 0) {
        image_engine_fail(result, "width and height must be positive");
        return false;
    }
    if (params.steps <= 0) {
        image_engine_fail(result, "steps must be positive");
        return false;
    }
    if (params.batch_count <= 0) {
        image_engine_fail(result, "batch_count must be positive");
        return false;
    }
    return true;
}

static sd_ctx_params_t image_engine_ctx_params(const image_engine_params & params) {
    sd_ctx_params_t ctx_params;
    sd_ctx_params_init(&ctx_params);

    ctx_params.model_path            = cstr_or_null(params.model_path);
    ctx_params.vae_path              = cstr_or_null(params.vae_path);
    ctx_params.clip_l_path           = cstr_or_null(params.clip_l_path);
    ctx_params.clip_g_path           = cstr_or_null(params.clip_g_path);
    ctx_params.t5xxl_path            = cstr_or_null(params.t5xxl_path);
    ctx_params.diffusion_model_path   = cstr_or_null(params.diffusion_model_path);
    ctx_params.n_threads              = params.n_threads;
    ctx_params.enable_mmap            = params.enable_mmap;
    ctx_params.flash_attn             = params.flash_attn;
    ctx_params.diffusion_flash_attn   = params.diffusion_flash_attn;
    ctx_params.qwen_image_zero_cond_t = params.qwen_image_zero_cond_t;
    ctx_params.backend                = cstr_or_null(params.backend);
    ctx_params.params_backend         = cstr_or_null(params.params_backend);

    return ctx_params;
}

static sd_img_gen_params_t image_engine_generation_params(const image_engine_params & params, sd_ctx_t * ctx) {
    sd_img_gen_params_t gen_params;
    sd_img_gen_params_init(&gen_params);

    gen_params.prompt = cstr_or_empty(params.prompt);
    gen_params.negative_prompt = cstr_or_empty(params.negative_prompt);
    gen_params.width = params.width;
    gen_params.height = params.height;
    gen_params.seed = params.seed;
    gen_params.batch_count = params.batch_count;
    gen_params.sample_params.sample_steps = params.steps;
    gen_params.sample_params.eta = params.eta;
    gen_params.sample_params.guidance.txt_cfg = params.cfg_scale;
    gen_params.sample_params.guidance.img_cfg = params.cfg_scale;
    gen_params.sample_params.guidance.distilled_guidance = params.distilled_guidance;

    const sample_method_t method = sd_get_default_sample_method(ctx);
    gen_params.sample_params.sample_method = method;
    gen_params.sample_params.scheduler     = sd_get_default_scheduler(ctx, method);

    return gen_params;
}

bool image_engine_generate(const image_engine_params & params, image_engine_result & result) {
    result = image_engine_result();
    if (!image_engine_validate(params, result)) {
        return false;
    }

    sd_ctx_params_t ctx_params = image_engine_ctx_params(params);
    sd_ctx_guard ctx;
    ctx.ctx = new_sd_ctx(&ctx_params);
    if (!ctx.ctx) {
        image_engine_fail(result, "failed to create stable-diffusion context");
        return false;
    }
    if (!sd_ctx_supports_image_generation(ctx.ctx)) {
        image_engine_fail(result, "stable-diffusion context does not support image generation");
        return false;
    }

    sd_img_gen_params_t gen_params = image_engine_generation_params(params, ctx.ctx);
    sd_images_guard images;
    images.count  = gen_params.batch_count;
    images.images = generate_image(ctx.ctx, &gen_params);
    if (!images.images || images.count <= 0) {
        image_engine_fail(result, "stable-diffusion image generation failed");
        return false;
    }

    const sd_image_t & image = images.images[0];
    if (!image.data || image.width == 0 || image.height == 0 || image.channel == 0) {
        image_engine_fail(result, "stable-diffusion returned an empty image");
        return false;
    }

    const int stride = (int) (image.width * image.channel);
    if (!stbi_write_png(params.output_path.c_str(), (int) image.width, (int) image.height, (int) image.channel,
                        image.data, stride)) {
        image_engine_fail(result, "failed to write PNG artifact");
        return false;
    }

    result.ok            = true;
    result.artifact_path = params.output_path;
    result.width         = (int32_t) image.width;
    result.height        = (int32_t) image.height;
    result.channel       = (int32_t) image.channel;
    result.seed          = gen_params.seed;
    return true;
}

}  // namespace utopic
