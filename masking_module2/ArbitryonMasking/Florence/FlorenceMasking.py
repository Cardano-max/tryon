from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.detection import CaptionOntology
from PIL import Image
import os

def load_image(image_path):
    return Image.open(image_path).convert("RGB")

class FlorenceMasking:
    def __init__(self):
        self.ontology = CaptionOntology({
            "Full Dress": "Full Dress"
        })
        self.model = None
        self.model_loaded = False

    def load_model(self):
        if not self.model_loaded:
            self.model = GroundedSAM2(
                ontology=self.ontology
            )
        self.model_loaded = True

    def get_mask(self, image_path, output_path="outputs/"):
        if not self.model_loaded:
            self.load_model()

        image = load_image(image_path)
        results = self.model.predict(image)
        
        # Process results and save mask
        mask = results.mask[0] if results.mask is not None else None
        if mask is not None:
            os.makedirs(output_path, exist_ok=True)
            mask_path = os.path.join(output_path, f"{os.path.basename(image_path)}_mask.png")
            Image.fromarray(mask.astype('uint8') * 255).save(mask_path)
            print(f"Mask saved to {mask_path}")
        else:
            print("No mask generated")

        return mask

    def test_masking(self, image_path):
        print(f"Testing masking for image: {image_path}")
        mask = self.get_mask(image_path)
        if mask is not None:
            print("Mask generated successfully")
        else:
            print("Failed to generate mask")
        return mask
