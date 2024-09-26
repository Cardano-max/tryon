from masking_module2.ArbitryonMasking.Florence.FlorenceMasking import FlorenceMasking
import numpy as np
from PIL import Image

def generate_mask(person_image_path, category="dresses"):
    try:
        print(f"Loading person image from: {person_image_path}")
        person_image = Image.open(person_image_path).convert('RGB')
        person_image = np.array(person_image)
        print("Person image loaded successfully.")

        florence_masking = FlorenceMasking()
        florence_masking.load_model()
        inpaint_mask = florence_masking.get_mask(person_image_path)

        return inpaint_mask
    except Exception as e:
        print(f"Error occurred in generate_mask:")
        traceback.print_exc()
        return None