import gradio as gr
import sys
sys.path.append('./')
from PIL import Image
import os
import numpy as np
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
    .tab-nav {
        background-color: #2c2c2c;
        padding: 1rem;
    }
    .tab-nav button {
        background-color: transparent;
        color: #bb86fc;
        border: none;
        padding: 0.5rem 1rem;
        margin-right: 1rem;
        font-size: 1rem;
        cursor: pointer;
        transition: all 0.3s;
        border-radius: 5px;
    }
    .tab-nav button:hover, .tab-nav button.selected {
        background-color: #bb86fc;
        color: #121212;
    }
    .tab-content {
        padding: 2rem;
    }
    .grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 2rem;
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
"""


# Include all the necessary functions from the original code
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
    # Convert PIL Image to NumPy array
    image = np.array(pil_image)

    # Convert RGBA to RGB if needed
    if image.shape[2] == 4:  # Check if the image has an alpha channel
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Detect faces
    face_detect = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
    face_data = face_detect.detectMultiScale(image, 1.3, 5)

    # Draw rectangle around the faces which is our region of interest (ROI)
    for (x, y, w, h) in face_data:
        roi = image[y:y+h, x:x+w]
        # Apply a Gaussian blur over this new rectangle area with increased intensity
        roi = cv2.GaussianBlur(roi, (75, 75), 75)
        # Impose this blurred image on the original image to get the final image
        image[y:y+roi.shape[0], x:x+roi.shape[1]] = roi

    # Convert the NumPy array back to PIL Image
    if pil_image.mode == 'RGBA':  # If the original image had an alpha channel
        image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image)

    return pil_image



def start_tryon(dict,garm_img,garment_des,is_checked, category, is_checked_crop,denoise_steps,seed):
    

    garm_img= garm_img.convert("RGB").resize((768,1024))
    blur_face = face_blur(dict["background"])
    human_img_orig = blur_face.convert("RGB")   

    print(type(dict['background']))
    print(dict['background']) 
    print("-"*50)
    print(type(human_img_orig))
    print(human_img_orig)
    human_img = human_img_orig.resize((768,1024))

    return human_img, garm_img
    # return images[0], mask_gray



image_blocks = gr.Blocks().queue()
with gr.Blocks(css=custom_css) as demo:
    gr.HTML("""
        <div class="header">
            <h1>✨ ArbiTryOn: Virtual Fitting Room ✨</h1>
            <p>Step into the future of fashion with our AI-powered virtual wardrobe</p>
        </div>
    """)

    with gr.Tabs() as tabs:
        with gr.TabItem("Virtual Try-On"):
            with gr.Row():
                with gr.Column():
                    imgs = gr.ImageEditor(sources='upload', type="pil", label='Upload Your Photo', interactive=True)
    
                    auto_mask = gr.Checkbox(label="Use AI-Powered Auto-Masking", value=True)
                    auto_crop = gr.Checkbox(label="Smart Auto-Crop & Resizing", value=False)
                    category = gr.Radio(["upper_body", "lower_body", "dresses"], label="Garment Category", value="upper_body")

                with gr.Column():
                    garment_image = gr.Image(label="Upload or Select Garment", type="pil", sources="upload")
                    description = gr.Textbox(label="Garment Description", placeholder="E.g., Sleek black evening dress with lace details")

                with gr.Column():
                    output_image = gr.Image(label="Your Virtual Try-On")
                    masked_image = gr.Image(label="AI-Generated Mask")

            with gr.Row():
                try_on_button = gr.Button("Experience Your New Look!", variant="primary")

            with gr.Accordion("Advanced Customization", open=False):
                denoise_steps = gr.Slider(minimum=20, maximum=40, value=30, step=1, label="AI Processing Intensity")
                seed = gr.Number(label="Randomness Seed", value=42)

    # Set up the try-on functionality
    try_on_button.click(
        start_tryon,
        inputs=[
            imgs,
            garment_image,
            description,
            auto_mask,
            category,
            auto_crop,
            denoise_steps,
            seed
        ],
        outputs=[output_image, masked_image]
    )

if __name__ == "__main__":
    demo.launch()