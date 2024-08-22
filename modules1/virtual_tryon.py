import numpy as np
from PIL import Image
from extras.inpaint_mask import generate_mask_from_image
import modules1.flags as flags
from modules1.async_worker import AsyncTask
import time
import modules1.config
import modules1.constants as constants
import random
import args_manager
import traceback
import gradio as gr

def generate_mask(image, mask_model, cloth_category, sam_prompt_text, sam_model, sam_quant, box_threshold, text_threshold):
    try:
        extras = {}
        if mask_model == 'u2net_cloth_seg':
            extras['cloth_category'] = cloth_category
        elif mask_model == 'sam':
            extras['sam_prompt_text'] = sam_prompt_text
            extras['sam_model'] = sam_model
            extras['sam_quant'] = sam_quant
            extras['box_threshold'] = box_threshold
            extras['text_threshold'] = text_threshold

        return generate_mask_from_image(image, mask_model, extras)
    except Exception as e:
        print("Error in generate_mask:", str(e))
        traceback.print_exc()
        return None

def virtual_tryon(clothes_image, person_image):
    try:
        # Generate mask for person image using SAM model
        person_mask = generate_mask(
            person_image,
            'sam',
            '',  # cloth_category is not used for 'sam'
            'full body clothes',  # Default prompt
            'sam_vit_b_01ec64',  # Default SAM model
            False,  # Quantization
            0.3,  # Box threshold
            0.25  # Text threshold
        )
        if person_mask is None:
            return "Error in generating mask for person image."

        print(f"Person mask shape: {person_mask.shape}")  # Debugging line

        # Prepare inputs for the main generation process
        seed = random.randint(constants.MIN_SEED, constants.MAX_SEED)
        default_stop, default_weight = flags.default_parameters[flags.cn_ip]

        args = [
            True,  # generate_image_grid
            "A person wearing the clothes from the reference image",  # prompt
            "Unrealistic, blurry, low quality",  # negative_prompt
            False,  # translate_prompts
            ["Fooocus V2", "Fooocus Enhance", "Fooocus Sharp"],  # style_selections
            flags.Performance.QUALITY.value,  # performance_selection
            modules.config.default_aspect_ratio,  # aspect_ratios_selection
            1,  # image_number
            modules.config.default_output_format,  # output_format
            seed,  # image_seed
            modules.config.default_sample_sharpness,  # sharpness
            modules.config.default_cfg_scale,  # guidance_scale
            modules.config.default_base_model_name,  # base_model_name
            modules.config.default_refiner_model_name,  # refiner_model_name
            modules.config.default_refiner_switch,  # refiner_switch
        ]

        # Add LoRA arguments
        for lora in modules.config.default_loras:
            args.extend(lora)

        args.extend([
            True,  # input_image_checkbox
            "inpaint",  # current_tab
            flags.disabled,  # uov_method
            clothes_image,  # uov_input_image
            [],  # outpaint_selections
            {'image': person_image, 'mask': person_mask},  # inpaint_input_image
            "",  # inpaint_additional_prompt
            person_mask,  # inpaint_mask_image_upload
            False,  # disable_preview
            False,  # disable_intermediate_results
            modules.config.default_black_out_nsfw,  # black_out_nsfw
            1.0,  # adm_scaler_positive
            1.0,  # adm_scaler_negative
            0.0,  # adm_scaler_end
            modules.config.default_cfg_tsnr,  # adaptive_cfg
            modules.config.default_sampler,  # sampler_name
            modules.config.default_scheduler,  # scheduler_name
            modules.config.default_overwrite_step,  # overwrite_step
            modules.config.default_overwrite_switch,  # overwrite_switch
            -1,  # overwrite_width
            -1,  # overwrite_height
            -1,  # overwrite_vary_strength
            modules.config.default_overwrite_upscale,  # overwrite_upscale_strength
            True,  # mixing_image_prompt_and_vary_upscale
            True,  # mixing_image_prompt_and_inpaint
            False,  # debugging_cn_preprocessor
            False,  # skipping_cn_preprocessor
            100,  # canny_low_threshold
            200,  # canny_high_threshold
            flags.refiner_swap_method,  # refiner_swap_method
            0.5,  # controlnet_softness
            False,  # freeu_enabled
            1.0,  # freeu_b1
            1.0,  # freeu_b2
            1.0,  # freeu_s1
            1.0,  # freeu_s2
            False,  # debugging_inpaint_preprocessor
            False,  # inpaint_disable_initial_latent
            modules.config.default_inpaint_engine_version,  # inpaint_engine
            1.0,  # inpaint_strength
            0.618,  # inpaint_respective_field
            True,  # inpaint_mask_upload_checkbox
            False,  # invert_mask_checkbox
            0,  # inpaint_erode_or_dilate
        ])

        if not args_manager.args.disable_metadata:
            args.extend([
                modules.config.default_save_metadata_to_images,  # save_metadata_to_images
                modules.config.default_metadata_scheme,  # metadata_scheme
            ])

        # Add ControlNet tasks
        for _ in range(flags.controlnet_image_count):
            args.extend([
                clothes_image,  # cn_img (First image prompt: clothes image)
                default_stop,  # cn_stop
                default_weight,  # cn_weight
                flags.cn_ip,  # cn_type
            ])

        print(f"Number of arguments: {len(args)}")
        task = AsyncTask(args)

        # Add the task to the async_tasks list
        from modules.async_worker import async_tasks
        async_tasks.append(task)

        # Wait for the task to complete
        while not task.processing:
            time.sleep(0.1)
        while task.processing:
            time.sleep(0.1)

        return task.results
    except Exception as e:
        print("Error in virtual_tryon:", str(e))
        traceback.print_exc()
        return f"Error: {str(e)}"

def create_virtual_tryon_interface():
    with gr.Blocks() as virtual_tryon_interface:
        with gr.Row():
            clothes_input = gr.Image(label="Clothes Image", type="numpy")
            person_input = gr.Image(label="Person Image", type="numpy")
        generate_btn = gr.Button("Generate Virtual Try-On")
        output_gallery = gr.Gallery(label="Results")

        generate_btn.click(
            virtual_tryon,
            inputs=[clothes_input, person_input],
            outputs=[output_gallery]
        )

    return virtual_tryon_interface

# Create and launch the interface
interface = create_virtual_tryon_interface()
interface.launch()
