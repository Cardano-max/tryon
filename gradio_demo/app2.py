import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import traceback
import gradio as gr
import cv2
import uuid
import math
from PIL import Image
import numpy as np
import torch
from src.tryon_pipeline import StableDiffusionXLInpaintPipeline as TryonPipeline
from src.unet_hacked_garmnet import UNet2DConditionModel as UNet2DConditionModel_ref
from src.unet_hacked_tryon import UNet2DConditionModel
from transformers import (
    CLIPImageProcessor,
    CLIPVisionModelWithProjection,
    CLIPTextModel,
    CLIPTextModelWithProjection,
)
from diffusers import DDPMScheduler, AutoencoderKL, StableDiffusionXLImg2ImgPipeline
from typing import List
from transformers import AutoTokenizer
from ArbiTryOn_Masking.ArbitryonMasking.Florence.FlorenceMasking import FlorenceMasking
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
import mediapipe as mp
from PIL import Image as PILImage
import torch
from diffusers import AutoPipelineForInpainting, AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid
import torch
from diffusers import StableDiffusionXLInpaintPipeline, StableDiffusionXLImg2ImgPipeline
from PIL import Image
import numpy as np

# Import the Defocus virtual_try_on function
from webui3 import generate_mask
from preprocess.humanparsing.run_parsing import Parsing
from preprocess.openpose.run_openpose import OpenPose

# Import necessary functions from modules1.util
from modules1.util import HWC3, resize_image

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

catalog = []
garment_images = os.listdir('./gradio_demo/test_images/raw_garments/')
for i in range(len(garment_images)):
    # If the file is not an image, skip it
    if not garment_images[i].endswith(('.png', '.jpg', '.jpeg', '.webp')):
        continue
    image_path = f"gradio_demo/test_images/raw_garments/{garment_images[i]}"
    garment_image_path = f"gradio_demo/test_images/garments/{garment_images[i].split('.')[0]}.png"
    catalog.append({
        "name": garment_images[i].split('.')[0],
        "image": image_path,
        "garment_image": garment_image_path,
        "description": "Traditional Eastern dress"
    })

category_dict = {
    "Upper Body": "upper_body",
    "Lower Body": "lower_body",
    "Full Body": "dresses"
}

custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    :root{
        --bg-color: rgb(225 231 253)
    }
    body, .gradio-container {
        font-family: 'Poppins', sans-serif;
        background-color: var(--bg-color);
        color: #e0e0e0;
        margin: 0;
        padding: 0;
        border: none;
    }

    gradio-app {
        background: var(--bg-color) !important;
    }

    .container {
        margin: 0 auto;
        padding: 2rem;
    }
    .header {
        text-align: center;
        margin-bottom: 1rem;
    }
    .header h2 {
        font-size: 4rem;
        color: #000000;
    }
    .header p {
        color: #000000;
    }
    .tabs {
        background-color: var(--bg-color);
        overflow: hidden;
        border: none;
    }
    .tab-nav {
        padding: 1rem;
        border: none !important;
    }

    .tabitem {
        border: none !important;
    }

    .tab-nav button {
        background-color: transparent;
        color: #864BEE;
        border: none;
        padding: 0.5rem 1rem;
        margin-right: 1rem;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s;
        border-radius: 5px;
    }
    .tab-nav button:hover, .tab-nav button.selected {
        background-color: #864BEE;
        color: #ffffff;
    }
    .tab-content {
        padding: 2rem;
        border: none;
    }

    .gallery-container{
        background-color: var(--bg-color) !important;
    }

    .grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 3rem;
    }

    .thumbnail-item {
        height: 400px;
        width: 280px;
    }

    .card {
        background-color: #2c2c2c;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        transition: all 0.3s;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(187, 134, 252, 0.2);
    }
    .card img {
        width: 100%;
        height: 200px;
        object-fit: cover;
    }
    .card-content {
        padding: 1rem;
    }
    .btn {
        background-color: #bb86fc;
        color: #121212;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        cursor: pointer;
        transition: all 0.3s;
        font-weight: 600;
    }
    .btn:hover {
        background-color: #03dac6;
        box-shadow: 0 0 10px rgba(3, 218, 198, 0.5);
    }
    #output-img, #masked-img {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
    }
    .gradio-slider input[type="range"] {
        background-color: #bb86fc;
    }
    .gradio-slider input[type="range"]::-webkit-slider-thumb {
        background-color: #03dac6;
    }
    .gradio-checkbox input[type="checkbox"] {
        border-color: #bb86fc;
    }
    .gradio-checkbox input[type="checkbox"]:checked {
        background-color: #bb86fc;
    }
    .gradio-radio label span {
        border-color: #bb86fc;
    }
    .gradio-radio input[type="radio"]:checked + label span {
        background-color: #bb86fc;
    }

    .svelte-rk35yg {
        color: black !important;
        background-color: #ffffff !important;
    }
    #component-10, #component-16, #component-21 {
        background-color: #ffffff !important;
    }

    div.controls-wrap {
        hidden: true;
    }

    label.float, #component-23>button{
        background-color: #864BEE !important;
    }

"""

def pil_to_binary_mask(pil_image, threshold=0):
    np_image = np.array(pil_image)
    grayscale_image = Image.fromarray(np_image).convert("L")
    binary_mask = np.array(grayscale_image) > threshold
    mask = np.zeros(binary_mask.shape, dtype=np.uint8)
    for i in range(binary_mask.shape[0]):
        for j in range(binary_mask.shape[1]):
            if binary_mask[i,j] == True:
                mask[i,j] = 1
    mask = (mask*255).astype(np.uint8)
    output_mask = Image.fromarray(mask)
    return output_mask

def face_blur(pil_image):
    image = np.array(pil_image)
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    face_detect = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    face_data = face_detect.detectMultiScale(image, 1.3, 5)
    for (x, y, w, h) in face_data:
        roi = image[y:y+h, x:x+w]
        roi = cv2.GaussianBlur(roi, (75, 75), 75)
        image[y:y+roi.shape[0], x:x+roi.shape[1]] = roi
    if pil_image.mode == 'RGBA':
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)
    return pil_image

base_path = 'yisol/IDM-VTON'
example_path = os.path.join(os.path.dirname(__file__), 'example')


unet = UNet2DConditionModel.from_pretrained(
    base_path,
    subfolder="unet",
    torch_dtype=torch.float16,
)
unet.requires_grad_(False)
tokenizer_one = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer",
    revision=None,
    use_fast=False,
)
tokenizer_two = AutoTokenizer.from_pretrained(
    base_path,
    subfolder="tokenizer_2",
    revision=None,
    use_fast=False,
)
noise_scheduler = DDPMScheduler.from_pretrained(base_path, subfolder="scheduler")

text_encoder_one = CLIPTextModel.from_pretrained(
    base_path,
    subfolder="text_encoder",
    torch_dtype=torch.float16,
)
text_encoder_two = CLIPTextModelWithProjection.from_pretrained(
    base_path,
    subfolder="text_encoder_2",
    torch_dtype=torch.float16,
)
image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    base_path,
    subfolder="image_encoder",
    torch_dtype=torch.float16,
    )
vae = AutoencoderKL.from_pretrained(base_path,
                                    subfolder="vae",
                                    torch_dtype=torch.float16,
)

UNet_Encoder = UNet2DConditionModel_ref.from_pretrained(
    base_path,
    subfolder="unet_encoder",
    torch_dtype=torch.float16,
)

model_idx = 0 if torch.cuda.is_available() else -1

parsing_model = Parsing(model_idx)
openpose_model = OpenPose(model_idx)

UNet_Encoder.requires_grad_(False)
image_encoder.requires_grad_(False)
vae.requires_grad_(False)
unet.requires_grad_(False)
text_encoder_one.requires_grad_(False)
text_encoder_two.requires_grad_(False)
tensor_transfrom = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
    )

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

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

def is_full_body_image(image):
    image_np = np.array(image)
    image_rgb = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    results = pose.process(image_rgb)
    
    if not results.pose_landmarks:
        return False
    
    landmarks = results.pose_landmarks.landmark
    
    # Check if key points are visible
    key_points = [
        mp_pose.PoseLandmark.NOSE,
        mp_pose.PoseLandmark.LEFT_SHOULDER,
        mp_pose.PoseLandmark.RIGHT_SHOULDER,
        mp_pose.PoseLandmark.LEFT_HIP,
        mp_pose.PoseLandmark.RIGHT_HIP,
        mp_pose.PoseLandmark.LEFT_ANKLE,
        mp_pose.PoseLandmark.RIGHT_ANKLE,
        mp_pose.PoseLandmark.LEFT_WRIST,
        mp_pose.PoseLandmark.RIGHT_WRIST
    ]
    
    for point in key_points:
        if landmarks[point].visibility < 0.1:
            return False
    
    return True

# Add this new function for the refinement process
def refine_image(image, prompt, num_inference_steps=50, strength=0.3):
    refiner = StableDiffusionXLImg2ImgPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-refiner-1.0",
        torch_dtype=torch.float16,
    )
    refiner.to(device)

    refined_image = refiner(
        prompt=prompt,
        image=image,
        num_inference_steps=num_inference_steps,
        strength=strength
    ).images[0]

    return refined_image


def process_with_defocus(image_path):
    try:
        prompt = "Remove clothes, full naked, straight pose standing posing forward straight, perfect anatomy"
        category = "dresses"
        
        print("Processing image with Defocus...")
        result = defocus_virtual_try_on(image_path, prompt, category)
        
        if result:
            print(f"Defocus processing completed. Result saved at: {result}")
            return Image.open(result)
        else:
            print("Defocus processing failed.")
            return None
    except Exception as e:
        print(f"Error occurred in process_with_defocus: {str(e)}")
        traceback.print_exc()
        return None

def generate_dummy_pose_image(human_img):
    # Create a blank image with the same size as human_img
    pose_img = Image.new('RGB', human_img.size, color=(0, 0, 0))
    return pose_img

def process_tryon(dict, garm_img, garment_des, is_checked, category, blur_face, is_checked_crop, denoise_steps, seed):
    try:
        # Generate mask using Florence Plus SAM2 2
        mask = generate_mask(dict['image'], category)
        
        if mask is None:
            return None, None, "Mask generation failed."

        # Initialize IDM VTON
        idm_vton = IDM_VTON()

        # Perform virtual try-on
        result_image = idm_vton.try_on(dict['image'], garm_img, mask)

        if result_image is None:
            return None, None, "Virtual try-on failed."

        # Save the result
        result_path = f"output_{uuid.uuid4()}.png"
        result_image.save(result_path)

        return result_image, mask, "Virtual try-on completed successfully."
    except Exception as e:
        print(f"Error in process_tryon: {str(e)}")
        return None, None, f"An error occurred: {str(e)}"

garm_list = os.listdir(os.path.join(example_path,"cloth"))
garm_list_path = [os.path.join(example_path,"cloth",garm) for garm in garm_list]

human_list = os.listdir(os.path.join(example_path,"human"))
human_list_path = [os.path.join(example_path,"human",human) for human in human_list]

human_ex_list = []
for ex_human in human_list_path:
    ex_dict = {}
    ex_dict['background'] = ex_human
    ex_dict['layers'] = None
    ex_dict['composite'] = None
    human_ex_list.append(ex_dict)

with gr.Blocks(css=custom_css) as demo:
    gr.HTML("""
        <div class="header">
            <h2>Arbi-TryOn</h2>
            <p>Your Virtual Fitting Room</p>
        </div>
    """)
    height = 550
    width = 450
    with gr.Tabs() as tabs:
        with gr.TabItem("Fashion Catalog", id=0):
            gr.HTML("""
                <div class="header">
                <p>Choose your favourite design and let us do the magic</p>
                </div>
                """)
            with gr.Row():
                garment_gallery = gr.Gallery(
                    value=[item["image"] for item in catalog],
                    columns=4,
                    show_label=False,
                    elem_id="garment_gallery",
                    height=f"{math.ceil(len(catalog)/4)*412.5}px",
                    allow_preview=False,
                    min_width=250
                )

        with gr.TabItem("Virtual Try-On", id=1):
            with gr.Row():
                with gr.Column():
                    imgs = gr.ImageEditor(sources='upload', type="pil", label='Click here to Upload Your Photo', interactive=True, height=height, width=width, layers=False, brush=False, eraser=False, transforms=[])

                    auto_mask = gr.Checkbox(label="Use AI-Powered Auto-Masking", value=True, visible=False)
                    auto_crop = gr.Checkbox(label="Smart Auto-Crop & Resizing", value=False, visible=False)
                    blur_face = gr.Checkbox(label="Blur Faces", value=False, visible=False)
                with gr.Column():
                    garment_image = gr.Image(label="Selected Garment", type="pil", interactive=False, height=height, width=width)
                    description = gr.Textbox(label="Garment Description", placeholder="E.g., Sleek black evening dress with lace details", visible=False)
                    category = gr.Radio(["Upper Body", "Lower Body", "Full Body"], label="Garment Category", value="Full Body", visible=False)

                with gr.Column():
                    output_image = gr.Image(label="Your Virtual Try-On", height=height, width=width, interactive=False)
                    output_mask = gr.Image(label="Masked Image", visible=False)

            with gr.Row():
                try_on_button = gr.Button("Experience Your New Look!", variant="primary")

            with gr.Accordion("Advanced Customization", open=False, visible=False):
                denoise_steps = gr.Slider(minimum=20, maximum=40, value=40, step=1, label="AI Processing Intensity")
                seed = gr.Number(label="Randomness Seed", value=42)

    def select_garment(evt: gr.SelectData):
        selected_garment = catalog[evt.index]
        return PILImage.open(selected_garment["garment_image"]), selected_garment["description"], gr.Tabs(selected=1)

    garment_gallery.select(select_garment, None, [garment_image, description, tabs])

    try_on_button.click(
        process_tryon,
        inputs=[
            imgs,
            garment_image,
            description,
            auto_mask,
            category,
            blur_face,
            auto_crop,
            denoise_steps,
            seed
        ],
        outputs=[output_image, output_mask, gr.Textbox(label="Error Message")]
    )

if __name__ == "__main__":
    demo.launch(share=True)