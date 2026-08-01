# -*- coding: utf-8 -*-
"""
Open-vocabulary object detection + captioning pipeline used by Extension II.

Grounding DINO (localization) + BLIP (auto-caption vocabulary augmentation),
with the label-merge and scene-level-noun filters described in the report's
"Iterative Detector Upgrade" section. Requires `pip install transformers`.
"""
import re

import torch
import torchvision
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    BlipForConditionalGeneration,
    BlipProcessor,
)

from .model import device

_GDINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
_BLIP_MODEL_ID = "Salesforce/blip-image-captioning-base"

# One term per concept — deliberately de-duplicated (see report: near-synonyms in the
# vocabulary previously caused Grounding DINO to fire twice on one object and merge
# the two phrases into a single garbled label).
BASE_VOCABULARY = [
    "chair", "sofa", "ottoman", "stool", "bench", "table", "coffee table", "desk",
    "door", "window", "wall", "ceiling", "floor", "staircase", "column", "archway",
    "fireplace", "radiator",
    "chalkboard", "podium", "bookshelf", "cabinet", "computer", "monitor", "keyboard",
    "projector", "clock", "poster",
    "pillow", "blanket", "curtain", "rug", "mirror", "painting", "vase",
    "potted plant", "candle", "lamp", "chandelier",
    "book", "box", "basket", "bag", "bottle", "cup", "bowl", "remote control", "tv",
    "speaker", "phone", "laptop", "papers", "notes", "sign", "flyer",
    "person", "backpack", "shoe", "coat", "umbrella",
]

# Scene-level collective nouns: if these leak into the vocabulary from a BLIP caption,
# Grounding DINO tends to draw one giant box around the whole image for them.
_SCENE_LEVEL_BLACKLIST = {
    "room", "rooms", "classroom", "office", "kitchen", "bedroom",
    "photo", "photograph", "picture", "image", "scene", "view",
    "background", "foreground", "shot", "indoor", "outdoor",
    "setting", "space", "area", "interior",
}
_STOPWORDS = {
    "a", "an", "the", "is", "are", "on", "in", "of", "with", "and", "next",
    "to", "there", "this", "that", "sitting", "standing", "some", "at", "by",
}


def load_detection_models():
    """Load Grounding DINO + BLIP. Call once, then pass the returned handles around."""
    gdino_processor = AutoProcessor.from_pretrained(_GDINO_MODEL_ID)
    gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(_GDINO_MODEL_ID).to(device).eval()
    blip_processor = BlipProcessor.from_pretrained(_BLIP_MODEL_ID)
    blip_model = BlipForConditionalGeneration.from_pretrained(_BLIP_MODEL_ID).to(device).eval()
    return gdino_processor, gdino_model, blip_processor, blip_model


def caption_image(pil_img: Image.Image, blip_processor, blip_model) -> str:
    inputs = blip_processor(pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = blip_model.generate(**inputs, max_new_tokens=30)
    return blip_processor.decode(out_ids[0], skip_special_tokens=True)


def build_vocabulary(pil_img, blip_processor, blip_model, use_caption_augmentation=True):
    """Curated base vocabulary + (optionally) extra nouns surfaced from a BLIP caption."""
    vocab = list(BASE_VOCABULARY)
    if use_caption_augmentation:
        caption = caption_image(pil_img, blip_processor, blip_model)
        words = [
            w for w in re.findall(r"[a-zA-Z]+", caption.lower())
            if w not in _STOPWORDS and w not in _SCENE_LEVEL_BLACKLIST and len(w) > 2
        ]
        vocab += words
    seen, deduped = set(), []
    for w in vocab:
        if w not in seen:
            seen.add(w)
            deduped.append(w)
    return deduped


def detect_objects(
    rgb_uint8,
    gdino_processor,
    gdino_model,
    blip_processor,
    blip_model,
    score_thresh=0.30,
    text_thresh=0.25,
    use_caption_augmentation=True,
):
    """Run the full open-vocabulary detection pipeline on one RGB image.

    Returns a list of dicts: {label, score, box=(x1,y1,x2,y2)}, after NMS and the
    label-merge / oversized-box filters described in the report.
    """
    pil_img = Image.fromarray(rgb_uint8)
    vocab = build_vocabulary(pil_img, blip_processor, blip_model, use_caption_augmentation)
    text_prompt = ". ".join(vocab).lower() + "."

    inputs = gdino_processor(images=pil_img, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = gdino_model(**inputs)

    # transformers has renamed this kwarg across versions (box_threshold -> threshold).
    try:
        result = gdino_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            threshold=score_thresh, text_threshold=text_thresh,
            target_sizes=[pil_img.size[::-1]],
        )[0]
    except TypeError:
        result = gdino_processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids,
            box_threshold=score_thresh, text_threshold=text_thresh,
            target_sizes=[pil_img.size[::-1]],
        )[0]

    dets = []
    img_area = pil_img.size[0] * pil_img.size[1]
    for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        box_area = max(0, x2 - x1) * max(0, y2 - y1)
        if img_area > 0 and box_area / img_area > 0.75:
            continue  # scene-level false positive safety net
        words = label.split()
        deduped_words = list(dict.fromkeys(words))
        clean_label = " ".join(deduped_words) if deduped_words else label
        dets.append(dict(label=clean_label, score=float(score), box=(x1, y1, x2, y2)))

    if len(dets) > 1:
        boxes_t = torch.tensor([d["box"] for d in dets], dtype=torch.float32)
        scores_t = torch.tensor([d["score"] for d in dets], dtype=torch.float32)
        keep_idx = torchvision.ops.nms(boxes_t, scores_t, iou_threshold=0.5)
        dets = [dets[i] for i in keep_idx.tolist()]

    return dets
