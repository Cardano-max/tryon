import gradio as gr
import sys
import cv2
import uuid
import math
import requests
import json
import io
import base64
sys.path.append('./')
from PIL import Image
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler, AutoencoderKL
from typing import List
import torch
import os
from transformers import AutoTokenizer
import numpy as np
from utils_mask import get_mask_location
from utils_newmask import Masking
from torchvision import transforms
import apply_net
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose
from detectron2.data.detection_utils import convert_PIL_to_numpy, _apply_exif_orientation
from torchvision.transforms.functional import to_pil_image

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
masker = Masking()

# Load models and set up pipeline
base_path = 'yisol/IDM-VTON'
unet = UNet2DConditionModel.from_pretrained(base_path, subfolder="unet", torch_dtype=torch.float16)
tokenizer_one = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer", use_fast=False)
tokenizer_two = AutoTokenizer.from_pretrained(base_path, subfolder="tokenizer_2", use_fast=False)
noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")
text_encoder_one = CLIPTextModel.from_pretrained(base_path, subfolder="text_encoder", torch_dtype=torch.float16)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(base_path, subfolder="text_encoder_2", torch_dtype=torch.float16)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(base_path, subfolder="image_encoder", torch_dtype=torch.float16)
vae = AutoencoderKL.from_pretrained(base_path, subfolder="vae", torch_dtype=torch.float16)
UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(base_path, subfolder="unet_encoder", torch_dtype=torch.float16)

parsing_model = Parsing(0)
openpose_model = OpenPose(0)

pipe = TryonPipeline.from_pretrained(
    base_path,
    unet=unet,
    vae=vae,
    feature_extractor=CLIPImageProcessor(),
    text_encoder=text_encoder_one,
    text_encoder_2=text_encoder_two,
    tokenizer=tokenizer_one,
    tokenizer_2=tokenizer_two,
    scheduler=noise_scheduler,
    image_encoder=image_encoder,
    torch_dtype=torch.float16,
)
pipe.unet_encoder = UNet_Encoder
pipe.to(device)

def pil_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def refine_with_detail_tweaker(image):
    img_str = pil_to_base64(image)
    
    url = "https://api.imagepipeline.io/sd/image2image/v1"
    payload = json.dumps({
        "model_id": "38ff8ad2-d62a-457c-8b2c-182e48e564f2",
        "prompt": "ultra realistic, highly detailed, perfect anatomy, photorealistic",
        "negative_prompt": "deformed, blurry, bad anatomy, extra limbs, poorly drawn face",
        "init_image": img_str,
        "width": str(image.width),
        "height": str(image.height),
        "samples": "1",
        "num_inference_steps": "30",
        "safety_checker": False,
        "guidance_scale": 7.5,
        "strength": 0.3,
        "lora_models": ["c7e734b3-db74-43c8-a50d-d1cc2fa150fc"],
        "lora_weights": ["0.5"]
    })
    headers = {
        'Content-Type': 'application/json',
        'API-Key': 'jYAPFEyBdWIY4I3OCXEluiPSCmOXsHv1Sh25e5Du4arMckHpI4BiQtOyrJgWKHo9'
    }
    
    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        result = response.json()
        if result['status'] == 'SUCCESS':
            img_data = base64.b64decode(result['images'][0])
            return Image.open(io.BytesIO(img_data))
        else:
            print(f"API Error: {result.get('error', 'Unknown error')}")
            return image
    except requests.RequestException as e:
        print(f"Request failed: {e}")
        return image

def start_tryon(dict, garm_img, garment_des, is_checked, category, blur_face, is_checked_crop, denoise_steps, seed):
    openpose_model.preprocessor.body_estimation.model.to(device)
    
    garm_img = garm_img.convert("RGB").resize((768, 1024))
    original_img = dict["background"]
    human_img_orig = original_img.convert("RGB")

    unique_id = str(uuid.uuid4())
    save_dir = "eval_images"
    os.makedirs(save_dir, exist_ok=True)
    human_img_path = os.path.join(save_dir, f"{unique_id}_ORIGINAL.png")
    masked_img_path = os.path.join(save_dir, f"{unique_id}_MASK.png")
    output_img_path = os.path.join(save_dir, f"{unique_id}_OUTPUT.png")

    if blur_face:
        face_blur(human_img_orig).save(human_img_path)
    else:
        human_img_orig.save(human_img_path)

    if is_checked_crop:
        width, height = human_img_orig.size
        target_width = int(min(width, height * (3 / 4)))
        target_height = int(min(height, width * (4 / 3)))
        left = (width - target_width) / 2
        top = (height - target_height) / 2
        right = (width + target_width) / 2
        bottom = (height + target_height) / 2
        cropped_img = human_img_orig.crop((left, top, right, bottom))
        crop_size = cropped_img.size
        human_img = cropped_img.resize((768, 1024))
    else:
        human_img = human_img_orig.resize((768, 1024))

    if is_checked:
        mask, garment_mask = masker.get_mask(human_img, category_dict[category])
        mask = mask.resize((768, 1024))
    else:
        mask = pil_to_binary_mask(dict['layers'][0].convert("RGB").resize((768, 1024)))

    human_img_arg = _apply_exif_orientation(human_img.resize((384, 512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    pose_img = args.func(args, human_img_arg)
    pose_img = pose_img[:,:,::-1]
    pose_img = Image.fromarray(pose_img).resize((768, 1024))

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            prompt = f"model is wearing {garment_des}"
            negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
            
            prompt_embeds, negative_prompt_embeds, pooled_prompt_embeds, negative_pooled_prompt_embeds = pipe.encode_prompt(
                prompt,
                negative_prompt=negative_prompt,
                num_images_per_prompt=1,
                do_classifier_free_guidance=True,
            )
            
            prompt_embeds_c, _, _, _ = pipe.encode_prompt(
                f"a photo of {garment_des}",
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
            )

            pose_img_tensor = transforms.ToTensor()(pose_img).unsqueeze(0).to(device, torch.float16)
            garm_tensor = transforms.ToTensor()(garm_img).unsqueeze(0).to(device, torch.float16)
            
            generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
            
            images = pipe(
                prompt_embeds=prompt_embeds.to(device, torch.float16),
                negative_prompt_embeds=negative_prompt_embeds.to(device, torch.float16),
                pooled_prompt_embeds=pooled_prompt_embeds.to(device, torch.float16),
                negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device, torch.float16),
                num_inference_steps=denoise_steps,
                generator=generator,
                strength=1.0,
                pose_img=pose_img_tensor,
                text_embeds_cloth=prompt_embeds_c.to(device, torch.float16),
                cloth=garm_tensor,
                mask_image=mask,
                image=human_img,
                height=1024,
                width=768,
                ip_adapter_image=garm_img.resize((768, 1024)),
                guidance_scale=2.0,
            )[0]

    # Refine the generated image
    refined_image = refine_with_detail_tweaker(images[0])

    if is_checked_crop:
        out_img = refined_image.resize(crop_size)
        human_img_orig.paste(out_img, (int(left), int(top)))
        final_image = human_img_orig
    else:
        final_image = refined_image

    if blur_face:
        final_image = face_blur(final_image)

    final_image.save(output_img_path)
    garment_mask.save(masked_img_path)

    return final_image, garment_mask

# Gradio interface setup
custom_css = """
    /* Your custom CSS here */
"""

category_dict = {
    "Upper Body": "upper_body",
    "Lower Body": "lower_body",
    "Full Body": "dresses"
}

with gr.Blocks(css=custom_css) as demo:
    gr.HTML("""<h1>ArbiTryOn: Virtual Fitting Room</h1>""")
    
    with gr.Tabs():
        with gr.TabItem("Virtual Try-On"):
            with gr.Row():
                with gr.Column():
                    imgs = gr.ImageEditor(label='Upload Your Photo', type="pil", height=550, width=450)
                    auto_mask = gr.Checkbox(label="Use AI-Powered Auto-Masking", value=True)
                    auto_crop = gr.Checkbox(label="Smart Auto-Crop & Resizing", value=False)
                    blur_face = gr.Checkbox(label="Blur Faces", value=False)
                
                with gr.Column():
                    garment_image = gr.Image(label="Selected Garment", type="pil", interactive=False, height=550, width=450)
                    description = gr.Textbox(label="Garment Description", placeholder="E.g., Sleek black evening dress with lace details")
                    category = gr.Radio(["Upper Body", "Lower Body", "Full Body"], label="Garment Category", value="Full Body")

                with gr.Column():
                    output_image = gr.Image(label="Your Virtual Try-On", height=550, width=450)
                    output_mask = gr.Image(label="Masked Image", visible=False)

            try_on_button = gr.Button("Experience Your New Look!", variant="primary")

            with gr.Accordion("Advanced Customization", open=False):
                denoise_steps = gr.Slider(minimum=20, maximum=50, value=30, step=1, label="AI Processing Intensity")
                seed = gr.Number(label="Randomness Seed", value=42)

        with gr.TabItem("Fashion Catalog"):
            garment_gallery = gr.Gallery(value=[item["image"] for item in catalog], columns=4, height=600)

    def select_garment(evt: gr.SelectData):
        selected_garment = catalog[evt.index]
        return Image.open(selected_garment["image"]), selected_garment["description"]

    garment_gallery.select(select_garment, None, [garment_image, description])

    try_on_button.click(
        start_tryon,
        inputs=[imgs, garment_image, description, auto_mask, category, blur_face, auto_crop, denoise_steps, seed],
        outputs=[output_image, output_mask]
    )

if __name__ == "__main__":
    demo.launch(share=True)