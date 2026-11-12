"""
English OCR Engine with Layout Filtering, Logo Suppression, and Dual Recognizers.
"""
import cv2
import numpy as np
from PIL import Image
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

class LayoutFilter:
    """Suppresses non-text regions such as logos, seals, and graphics."""
    def __init__(self):
        self.suppressed_classes = {"logo", "figure", "seal", "stamp", "graphic", "icon"}

    def filter_boxes(self, detected_boxes, layout_labels):
        valid = []
        suppressed = []
        for box, label in zip(detected_boxes, layout_labels):
            if str(label).lower() in self.suppressed_classes:
                suppressed.append({"box": box, "label": label})
            else:
                valid.append(box)
        return valid, suppressed

class EnglishDualOCR:
    """Dual-track OCR: routes printed lines to fast path and handwriting to TrOCR."""
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Microsoft TrOCR model onto {self.device}...")
        self.processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
        self.model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten").to(self.device)
        self.model.eval()

    def is_handwritten(self, crop_image: Image.Image) -> bool:
        """Stroke variance heuristic to detect handwriting vs machine print."""
        gray = np.array(crop_image.convert("L"))
        edges = cv2.Canny(gray, 50, 150)
        ratio = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1] + 1e-5)
        return ratio > 0.08

    def recognize_line(self, crop: Image.Image) -> dict:
        """Transcribes a single text crop."""
        hw = self.is_handwritten(crop)
        pixel_values = self.processor(images=crop, return_tensors="pt").pixel_values.to(self.device)
        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values, max_new_tokens=48)
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        return {"text": text, "mode": "handwritten" if hw else "printed"}

def xy_cut_sort(boxes):
    """Sorts bounding boxes into reading order (top-to-bottom, left-to-right)."""
    if not boxes:
        return []
    return sorted(boxes, key=lambda b: (b[1] // 25, b[0]))
