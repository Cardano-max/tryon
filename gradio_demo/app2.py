import gradio as gr
import sys
import cv2
import uuid
import math
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
from PIL import Image as PILImage
import mediapipe as mp
import numpy as np
import cv2


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
masker = Masking()

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
        if landmarks[point].visibility < 0.5:
            return False
    
    return True

def start_tryon(dict, garm_img, garment_des, is_checked, category, blur_face, is_checked_crop, denoise_steps, seed):
    openpose_model.preprocessor.body_estimation.model.to(device)
    pipe.to(device)
    pipe.unet_encoder.to(device)

    garm_img = garm_img.convert("RGB").resize((768,1024))
    original_img = dict["background"]
    human_img_orig = original_img.convert("RGB")

    # Check if the image contains a full-body pose
    if not is_full_body_image(human_img_orig):
        return None, None, "Please upload a full-body image showing your entire body, including head, hands, and feet."

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
        human_img = cropped_img.resize((768,1024))
    else:
        human_img = human_img_orig.resize((768,1024))

    if is_checked:
        mask, mask_gray = masker.get_mask(human_img, category_dict[category])
        mask = mask.resize((768,1024))
    else:
        mask = pil_to_binary_mask(dict['layers'][0].convert("RGB").resize((768, 1024)))

    mask_gray = (1-transforms.ToTensor()(mask)) * tensor_transfrom(human_img)
    mask_gray = to_pil_image((mask_gray+1.0)/2.0)
    human_img_arg = _apply_exif_orientation(human_img.resize((384,512)))
    human_img_arg = convert_PIL_to_numpy(human_img_arg, format="BGR")

    args = apply_net.create_argument_parser().parse_args(('show', './configs/densepose_rcnn_R_50_FPN_s1x.yaml', './ckpt/densepose/model_final_162be9.pkl', 'dp_segm', '-v', '--opts', 'MODEL.DEVICE', 'cuda'))
    pose_img = args.func(args,human_img_arg)
    pose_img = pose_img[:,:,::-1]
    pose_img = Image.fromarray(pose_img).resize((768,1024))

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with torch.no_grad():
                prompt = "model is wearing " + garment_des
                negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                with torch.inference_mode():
                    (
                        prompt_embeds,
                        negative_prompt_embeds,
                        pooled_prompt_embeds,
                        negative_pooled_prompt_embeds,
                    ) = pipe.encode_prompt(
                        prompt,
                        num_images_per_prompt=1,
                        do_classifier_free_guidance=True,
                        negative_prompt=negative_prompt,
                    )

                    prompt = "a photo of " + garment_des
                    negative_prompt = "monochrome, lowres, bad anatomy, worst quality, low quality"
                    if not isinstance(prompt, List):
                        prompt = [prompt] * 1
                    if not isinstance(negative_prompt, List):
                        negative_prompt = [negative_prompt] * 1
                    with torch.inference_mode():
                        (
                            prompt_embeds_c,
                            _,
                            _,
                            _,
                        ) = pipe.encode_prompt(
                            prompt,
                            num_images_per_prompt=1,
                            do_classifier_free_guidance=False,
                            negative_prompt=negative_prompt,
                        )

                    pose_img = tensor_transfrom(pose_img).unsqueeze(0).to(device,torch.float16)
                    garm_tensor = tensor_transfrom(garm_img).unsqueeze(0).to(device,torch.float16)
                    generator = torch.Generator(device).manual_seed(seed) if seed is not None else None
                    images = pipe(
                        prompt_embeds=prompt_embeds.to(device,torch.float16),
                        negative_prompt_embeds=negative_prompt_embeds.to(device,torch.float16),
                        pooled_prompt_embeds=pooled_prompt_embeds.to(device,torch.float16),
                        negative_pooled_prompt_embeds=negative_pooled_prompt_embeds.to(device,torch.float16),
                        num_inference_steps=denoise_steps,
                        generator=generator,
                        strength=1.0,
                        pose_img=pose_img.to(device,torch.float16),
                        text_embeds_cloth=prompt_embeds_c.to(device,torch.float16),
                        cloth=garm_tensor.to(device,torch.float16),
                        mask_image=mask,
                        image=human_img,
                        height=1024,
                        width=768,
                        ip_adapter_image=garm_img.resize((768,1024)),
                        guidance_scale=2.0,
                    )[0]

    if is_checked_crop:
        out_img = images[0].resize(crop_size)
        human_img_orig.paste(out_img, (int(left), int(top)))

        if blur_face:
            face_blur(mask_gray).save(masked_img_path)
            face_blur(human_img_orig).save(output_img_path)
        else:
            mask_gray.save(masked_img_path)
            human_img_orig.save(output_img_path)

        return human_img_orig, mask_gray
    else:
        if blur_face:
            face_blur(mask_gray).save(masked_img_path)
            face_blur(images[0]).save(output_img_path)
        else:
            mask_gray.save(masked_img_path)
            images[0].save(output_img_path)
        return images[0], mask_gray

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

with gr.Blocks(css=custom_css, theme='gradio/soft') as demo:
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
                def change_tab():
                    gr.Tabs.update(selected=2)

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
                    # description = "Traditional Eastern dress"
                    # description = None
                    category = gr.Radio(["Upper Body", "Lower Body", "Full Body"], label="Garment Category", value="Full Body", visible=False)


                with gr.Column():
                    output_image = gr.Image(label="Your Virtual Try-On", height=height, width=width, interactive=False)
                    # output_image.change(activate_button)
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

# Update the try_on_button.click function
def process_tryon(dict, garm_img, garment_des, is_checked, category, blur_face, is_checked_crop, denoise_steps, seed):
    result_image, mask_image, error_message = start_tryon(dict, garm_img, garment_des, is_checked, category, blur_face, is_checked_crop, denoise_steps, seed)
    if error_message:
        return None, None, error_message
    return result_image, mask_image, ""


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