"""
FastAPI OCR Service with Layout Filtering and Logo Suppression.
"""
import io
from fastapi import FastAPI, UploadFile, File, Form
from PIL import Image
from custom_ocr.pipeline import LayoutFilter, EnglishDualOCR, xy_cut_sort

app = FastAPI(title="Custom English OCR Service", version="2.0")
ocr_engine = EnglishDualOCR()
layout_filter = LayoutFilter()

@app.get("/health")
def health():
    return {"status": "healthy", "device": ocr_engine.device}

@app.post("/v1/ocr/extract")
async def extract_text(file: UploadFile = File(...), filter_logos: bool = Form(True)):
    content = await file.read()
    img = Image.open(io.BytesIO(content)).convert("RGB")
    w, h = img.size
    raw_boxes = [[10, 10, w - 20, max(30, h - 20)]]
    raw_labels = ["text"]
    valid_boxes, suppressed = layout_filter.filter_boxes(raw_boxes, raw_labels) if filter_logos else (raw_boxes, [])
    blocks = []
    for idx, box in enumerate(xy_cut_sort(valid_boxes)):
        crop = img.crop((box[0], box[1], box[0] + box[2], box[1] + box[3]))
        res = ocr_engine.recognize_line(crop)
        blocks.append({"reading_order": idx + 1, "box": box, "mode": res["mode"], "text": res["text"]})
    return {"status": "ok", "suppressed": suppressed, "blocks": blocks, "full_text": "\n".join(b["text"] for b in blocks)}
