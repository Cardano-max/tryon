import argparse
from ArbitryonMasking.Florence.FlorenceMasking import FlorenceMasking

def main(image_path):
    masking = FlorenceMasking()
    mask = masking.get_mask(image_path)
    if mask is not None:
        print(f"Mask generated for {image_path}")
    else:
        print(f"Failed to generate mask for {image_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate mask for an image")
    parser.add_argument("-f", "--file", required=True, help="Path to the input image")
    args = parser.parse_args()

    main(args.file)
