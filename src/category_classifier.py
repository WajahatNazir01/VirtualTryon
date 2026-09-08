"""
category_classifier.py
Zero-shot garment category classification using CLIP -- no training or
labeled dataset needed. Classifies a pasted garment photo into one of
the categories the pipeline knows how to handle, so main.py can
automatically pick the right body landmarks + anchor detection logic.
"""

import sys
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor

_MODEL_NAME = "openai/clip-vit-base-patch32"
_model = None
_processor = None

# key -> text prompt CLIP compares the image against
CATEGORIES = {
    "upper_body_sleeved": "a photo of a t-shirt or long-sleeve shirt with sleeves",
    "upper_body_sleeveless": "a photo of a sleeveless tank top or vest",
    "lower_body": "a photo of trousers, pants, jeans, or shorts",
    "headwear": "a photo of a hat or cap",
}


def _load_model():
    global _model, _processor
    if _model is None:
        # First call downloads the model (~600MB) and caches it locally.
        _model = CLIPModel.from_pretrained(_MODEL_NAME)
        _processor = CLIPProcessor.from_pretrained(_MODEL_NAME)
    return _model, _processor


def classify_garment(image_path: str) -> str:
    """
    Returns the best-matching category key from CATEGORIES for the given
    garment image (run this on the original photo, before background
    removal, for the most context).
    """
    model, processor = _load_model()
    image = Image.open(image_path).convert("RGB")

    labels = list(CATEGORIES.keys())
    prompts = list(CATEGORIES.values())

    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0]

    best_idx = int(probs.argmax())
    return labels[best_idx]


if __name__ == "__main__":
    # Usage: python category_classifier.py <garment_image>
    if len(sys.argv) != 2:
        print("Usage: python category_classifier.py <garment_image>")
        sys.exit(1)

    category = classify_garment(sys.argv[1])
    print(f"Detected category: {category}")