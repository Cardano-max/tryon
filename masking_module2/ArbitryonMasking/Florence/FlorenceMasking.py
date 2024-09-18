import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

from autodistill_grounded_sam_2 import GroundedSAM2
from autodistill.detection import CaptionOntology

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
                ontology = self.ontology
            )
        self.model_loaded = True

    def get_mask(self, image_path, output_path="outputs/"):
        # Run inference on an image using the prompt
        results = self.model.predict(image_path)

        # Load the image using OpenCV
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB

        # Create a blank mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Fill the mask with white for the detected object
        if results.mask is not None:
            for single_mask in results.mask:
                mask = np.logical_or(mask, single_mask).astype(np.uint8)

        # Convert to binary mask (0 and 255)
        binary_mask = mask * 255
        # Save binary mask in output as samename_mask.png
        filename = image_path.split("/")[-1]
        print('Mask Created for:', filename)
        cv2.imwrite(output_path + filename.split(".")[0] + "_mask.png", binary_mask)
        print(f"File written to {output_path + filename.split('.')[0] + '_mask.png'}")

    def get_mask_from_folder(self, folder_path):
        # Open the folder and get all the images
        images = os.listdir(folder_path)
        for image in images:
            self.get_mask(folder_path + image)
