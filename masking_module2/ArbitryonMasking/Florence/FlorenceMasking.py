import torch
from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.detection import CaptionOntology

class FlorenceMasking:
    def __init__(self):
        self.model = None

    def load_model(self):
        self.model = GroundedSAM2(
            ontology=CaptionOntology(
                {
                    "Full Dress": "Full Dress",
                    "Upper Body": "Upper Body",
                    "Lower Body": "Lower Body"
                }
            )
        )

    def get_mask(self, image_path):
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        results = self.model.predict(image_path)
        mask = results.mask.squeeze().cpu().numpy()
        return mask
