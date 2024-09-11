import numpy as np

class SAM2ImagePredictor:
    def __init__(self, sam_model):
        self.sam_model = sam_model
        self.image = None

    def set_image(self, image):
        self.image = image
        print(f"Setting image for SAM2 predictor")

    def predict(self, box, multimask_output=False):
        print(f"Predicting with SAM2 predictor, box: {box}, multimask_output: {multimask_output}")
        if self.image is None:
            print("Error: Image not set. Call set_image() first.")
            return None, None, None

        # Create a simple rectangular mask as a placeholder
        mask = np.zeros(self.image.shape[:2], dtype=bool)
        for bbox in box:
            x1, y1, x2, y2 = map(int, bbox)
            mask[y1:y2, x1:x2] = True

        score = np.ones(len(box))  # Placeholder scores
        logits = None  # Placeholder logits

        return mask, score, logits
