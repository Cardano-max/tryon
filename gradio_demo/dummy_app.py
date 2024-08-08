import gradio as gr
import sys
sys.path.append('./')
from PIL import Image
import os
import numpy as np
import cv2

# Custom CSS for modern design
custom_css = """
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600&display=swap');
    
    body, .gradio-container {
        font-family: 'Poppins', sans-serif;
        background-color: #121212;
        color: #e0e0e0;
    }
    .container {
        max-width: 1200px;
        margin: 0 auto;
        padding: 2rem;
    }
    .header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .header h1 {
        font-size: 3rem;
        color: #bb86fc;
        text-shadow: 0 0 10px rgba(187, 134, 252, 0.3);
    }
    .header p {
        color: #03dac6;
    }
    .tabs {
        background-color: #1e1e1e;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        overflow: hidden;
    }
    .tab-content {
        padding: 2rem;
    }
    .grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
        gap: 1rem;
        max-height: 400px;
        overflow-y: auto;
    }
    .card {
        background-color: #2c2c2c;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.2);
        transition: all 0.3s;
        cursor: pointer;
    }
    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(187, 134, 252, 0.2);
    }
    .card img {
        width: 100%;
        height: 150px;
        object-fit: cover;
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
    .model-images {
        display: flex;
        justify-content: space-around;
        margin-bottom: 1rem;
    }
    .model-image {
        width: 150px;
        height: 200px;
        object-fit: cover;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s;
    }
    .model-image:hover {
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(187, 134, 252, 0.5);
    }
    .selected {
        border: 3px solid #bb86fc;
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

def start_tryon(human_img, garm_img, garment_des, is_checked, category, is_checked_crop, denoise_steps, seed):
    garm_img = garm_img.convert("RGB").resize((768,1024))
    blur_face = face_blur(human_img)
    human_img_orig = blur_face.convert("RGB")   
    human_img = human_img_orig.resize((768,1024))
    return human_img, garm_img

def load_garment_images():
    garment_dir = '/Users/ateeb.taseer/tryon2/arbi-tryon/gradio_demo/example/cloth'
    return [os.path.join(garment_dir, f) for f in os.listdir(garment_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.webp'))]

image_blocks = gr.Blocks().queue()
with gr.Blocks(css=custom_css) as demo:
    gr.HTML("""
        <div class="header">
            <h1>✨ ArbiTryOn: Virtual Fitting Room ✨</h1>
            <p>Step into the future of fashion with our AI-powered virtual wardrobe</p>
        </div>
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Choose a Model or Upload Your Photo")
            model_images = [
                "/Users/ateeb.taseer/tryon2/arbi-tryon/gradio_demo/example/image/image1.jpg",
                "/Users/ateeb.taseer/tryon2/arbi-tryon/gradio_demo/example/image/image2.jpg",
                "/Users/ateeb.taseer/tryon2/arbi-tryon/gradio_demo/example/image/image3.jpg"
            ]
            model_gallery = gr.Gallery(value=model_images, label="Choose a Model", elem_id="model-gallery", columns=3, height=200)
            
            uploaded_image = gr.Image(label="Or Upload Your Own Photo", type="pil", elem_id="uploaded-image")

        with gr.Column(scale=2):
            gr.Markdown("### Select a Garment")
            garment_gallery = gr.Gallery(value=load_garment_images(), label="Click to Select a Garment", elem_id="garment-gallery", columns=4, height=400)
            selected_garment = gr.Image(label="Selected Garment", type="pil", elem_id="selected-garment", visible=False)

        with gr.Column(scale=2):
            gr.Markdown("### Virtual Try-On Result")
            output_image = gr.Image(label="Your Virtual Try-On")
            masked_image = gr.Image(label="AI-Generated Mask")

    with gr.Row():
        try_on_button = gr.Button("Experience Your New Look!", variant="primary")

    with gr.Accordion("Advanced Settings", open=False):
        auto_mask = gr.Checkbox(label="Use AI-Powered Auto-Masking", value=True)
        auto_crop = gr.Checkbox(label="Smart Auto-Crop & Resizing", value=False)
        category = gr.Radio(["upper_body", "lower_body", "dresses"], label="Garment Category", value="upper_body")
        denoise_steps = gr.Slider(minimum=20, maximum=40, value=30, step=1, label="AI Processing Intensity")
        seed = gr.Number(label="Randomness Seed", value=42)
        garment_description = gr.Textbox(label="Garment Description", placeholder="E.g., Sleek black evening dress with lace details")

    def update_model_image(evt: gr.SelectData):
        return gr.Image.update(value=evt.value, visible=True)

    def update_garment_image(evt: gr.SelectData):
        return gr.Image.update(value=evt.value, visible=True)

    model_gallery.select(update_model_image, None, uploaded_image)
    garment_gallery.select(update_garment_image, None, selected_garment)

    try_on_button.click(
        start_tryon,
        inputs=[
            uploaded_image,
            selected_garment,
            garment_description,
            auto_mask,
            category,
            auto_crop,
            denoise_steps,
            seed
        ],
        outputs=[output_image, masked_image]
    )

if __name__ == "__main__":
    demo.launch(share=True)