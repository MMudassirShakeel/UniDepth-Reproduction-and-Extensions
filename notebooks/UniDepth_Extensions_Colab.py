# -*- coding: utf-8 -*-
"""
UniDepth Extensions — Robustness, Object-Level Depth Fusion & Cross-Backbone Consistency
Muhammad Mudassir Shakeel — Mobile Robotics Systems, MS Mechatronics Engineering

Run top-to-bottom in a fresh Google Colab runtime (GPU recommended: Runtime > Change runtime
type > T4 GPU). Each "# STEP" banner is one cell. COMMON SETUP (Steps 1-3) must run once;
after that, Extensions 1, 2 and 3 are independent — run whichever you want, in any order.

EXTENSION 1 — Robustness under low light, Gaussian noise, and motion blur
    Compares degraded-image depth predictions back to the clean-image prediction
    (used as a pseudo-reference, NOT ground truth) with MAE / RMSE / Rel% / SSIM / Agree@5%.

EXTENSION 2 — Object-level metric distance fusion
    Runs a pretrained Faster R-CNN detector alongside UniDepth, reads the median predicted
    depth inside every detected box, and reports a per-object distance list — plus a
    quantitative spatial-consistency sanity check (objects lower in the frame should
    generally read nearer the camera; we measure how often that holds).

EXTENSION 3 — Cross-backbone consistency (ViT-L vs ConvNeXt-L)
    Runs BOTH official UniDepthV2 backbones on the same image and quantifies how much they
    agree, using the same MAE/RMSE/SSIM battery as Extension 1, plus a global scale-ratio
    number. This directly follows up the backbone-switching scale drift noted during
    reproduction (Section 7.7.2 of the earlier report).

Every depth visualisation below includes its own "Depth (m)" colorbar.
"""

# =========================================================
# STEP 1 — Environment setup (run once per fresh runtime)
# =========================================================
!git clone https://github.com/lpiccinelli-eth/UniDepth.git
%cd UniDepth
!pip install -e . --quiet
!pip install huggingface_hub torchvision opencv-python-headless matplotlib scikit-image pandas --quiet

import unidepth
print("unidepth import OK")

# =========================================================
# STEP 2 — Model loader (call this whenever you need a specific backbone)
# =========================================================
import torch, torchvision, gc
from unidepth.models import UniDepthV2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_unidepth(backbone):
    """backbone: one of 'vits14', 'vitb14', 'vitl14' (the only official UniDepthV2 checkpoints)"""
    m = UniDepthV2.from_pretrained(f'lpiccinelli/unidepth-v2-{backbone}')
    return m.to(device).eval()

def free_model(m):
    del m
    gc.collect()
    torch.cuda.empty_cache()

DEFAULT_BACKBONE = 'vits14' if device.type == 'cpu' else 'vitl14'
model = load_unidepth(DEFAULT_BACKBONE)
print('Loaded UniDepthV2 (' + DEFAULT_BACKBONE + ') on', device)

# =========================================================
# STEP 3 — Inference helper + shared depth-figure utility (with colorbar)
# =========================================================
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def run_inference(rgb_uint8, active_model=None):
    """rgb_uint8: HxWx3 uint8 numpy array -> (depth_map_metres HxW, intrinsics 3x3)."""
    active_model = active_model or model
    rgb_t = torch.from_numpy(rgb_uint8).permute(2, 0, 1).to(device)
    with torch.no_grad():
        preds = active_model.infer(rgb_t)
    depth = preds['depth'].squeeze().cpu().numpy()
    intrinsics = preds['intrinsics'].squeeze().cpu().numpy()
    return depth, intrinsics

def show_depth(ax, depth_map, title=None, fig=None):
    """Draws a depth map on `ax` with its own colorbar labelled 'Depth (m)'."""
    im = ax.imshow(depth_map, cmap='jet')
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=9, loc='left')
    cbar = (fig or plt.gcf()).colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label('Depth (m)', fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    return im

# =========================================================
# STEP 4 — Upload your baseline photo(s) (shared across all 3 extensions)
# =========================================================
from google.colab import files
print("Select one or more clean test photos (e.g. the same classroom/living-room images used earlier):")
uploaded = files.upload()
baseline_paths = list(uploaded.keys())
print("Uploaded:", baseline_paths)


# #########################################################
# EXTENSION 1 — ROBUSTNESS UNDER LOW LIGHT, NOISE & BLUR
# #########################################################

# =========================================================
# EXT1 — STEP A: Degradation functions
# =========================================================
import cv2

def darken(img, gamma):
    return np.clip(255 * ((img / 255.0) ** (1.0 / gamma)), 0, 255).astype(np.uint8)

def add_gaussian_noise(img, sigma):
    noisy = img / 255.0 + np.random.normal(0, sigma, img.shape)
    return np.clip(noisy * 255, 0, 255).astype(np.uint8)

def motion_blur(img, k):
    kernel = np.zeros((k, k))
    kernel[k // 2, :] = 1.0 / k
    return cv2.filter2D(img, -1, kernel)

DEGRADATIONS = {
    'clean':            lambda im: im,
    'low_light_g0.7':   lambda im: darken(im, 0.7),
    'low_light_g0.5':   lambda im: darken(im, 0.5),
    'low_light_g0.3':   lambda im: darken(im, 0.3),
    'noise_s0.02':      lambda im: add_gaussian_noise(im, 0.02),
    'noise_s0.05':      lambda im: add_gaussian_noise(im, 0.05),
    'noise_s0.10':      lambda im: add_gaussian_noise(im, 0.10),
    'blur_k5':          lambda im: motion_blur(im, 5),
    'blur_k9':          lambda im: motion_blur(im, 9),
    'blur_k15':         lambda im: motion_blur(im, 15),
}

# =========================================================
# EXT1 — STEP B: Run every variant, for every uploaded image
# =========================================================
ext1_depths = {}
ext1_rgbs = {}

for path in baseline_paths:
    rgb = np.array(Image.open(path).convert('RGB'))
    ext1_depths[path] = {}
    ext1_rgbs[path] = {}
    for name, fn in DEGRADATIONS.items():
        variant_rgb = fn(rgb.copy())
        depth, _ = run_inference(variant_rgb)
        ext1_depths[path][name] = depth
        ext1_rgbs[path][name] = variant_rgb
        print(f"[{path}] {name:<16s} depth range {depth.min():.2f}-{depth.max():.2f} m")

# =========================================================
# EXT1 — STEP C: Quantitative consistency metrics vs the CLEAN baseline
# =========================================================
from skimage.metrics import structural_similarity as ssim
import pandas as pd

def compare_depth_maps(ref_depth, test_depth):
    if test_depth.shape != ref_depth.shape:
        test_depth = cv2.resize(test_depth, (ref_depth.shape[1], ref_depth.shape[0]))
    diff = test_depth - ref_depth
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    rel_pct = float(np.mean(np.abs(diff) / np.clip(ref_depth, 1e-3, None)) * 100)
    data_range = ref_depth.max() - ref_depth.min()
    ssim_val = float(ssim(ref_depth, test_depth, data_range=data_range))
    agree_5pct = float(np.mean(np.abs(diff) / np.clip(ref_depth, 1e-3, None) < 0.05) * 100)
    return dict(MAE_m=round(mae, 4), RMSE_m=round(rmse, 4), Rel_pct=round(rel_pct, 2),
                SSIM=round(ssim_val, 4), Agree_at_5pct=round(agree_5pct, 2))

rows = []
for path in baseline_paths:
    clean_depth = ext1_depths[path]['clean']
    for name, depth in ext1_depths[path].items():
        if name == 'clean':
            continue
        rows.append(dict(image=path, variant=name, **compare_depth_maps(clean_depth, depth)))

df_ext1 = pd.DataFrame(rows)
print(df_ext1.to_string(index=False))
df_ext1.to_csv('ext1_robustness_results.csv', index=False)
print("\nSaved ext1_robustness_results.csv")

# =========================================================
# EXT1 — STEP D: Qualitative grid figure (RGB + depth-with-colorbar per variant)
# =========================================================
for path in baseline_paths:
    variants = list(DEGRADATIONS.keys())
    fig, axes = plt.subplots(len(variants), 2, figsize=(8, 3.1 * len(variants)))
    for i, name in enumerate(variants):
        axes[i, 0].imshow(ext1_rgbs[path][name]); axes[i, 0].axis('off')
        axes[i, 0].set_title(name, fontsize=9, loc='left')
        show_depth(axes[i, 1], ext1_depths[path][name], fig=fig)
    plt.tight_layout()
    out_name = f"ext1_grid_{path.split('.')[0]}.png"
    plt.savefig(out_name, dpi=140, bbox_inches='tight')
    plt.show()
    print("Saved", out_name)

# =========================================================
# EXT1 — STEP E: Metric-vs-severity trend plot
# =========================================================
groups = {
    'Low Light':      ['low_light_g0.7', 'low_light_g0.5', 'low_light_g0.3'],
    'Gaussian Noise': ['noise_s0.02', 'noise_s0.05', 'noise_s0.10'],
    'Motion Blur':    ['blur_k5', 'blur_k9', 'blur_k15'],
}
severity_x = {'Low Light': [0.7, 0.5, 0.3], 'Gaussian Noise': [0.02, 0.05, 0.10], 'Motion Blur': [5, 9, 15]}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, (group_name, variant_names) in zip(axes, groups.items()):
    for path in baseline_paths:
        sub = df_ext1[(df_ext1['image'] == path) & (df_ext1['variant'].isin(variant_names))]
        sub = sub.set_index('variant').loc[variant_names]
        ax.plot(severity_x[group_name], sub['RMSE_m'], marker='o', label=path)
    ax.set_title(group_name)
    ax.set_xlabel('severity' if group_name != 'Low Light' else 'gamma (lower = darker)')
    ax.set_ylabel('RMSE vs clean depth (m)')
    ax.legend(fontsize=7)
plt.tight_layout()
plt.savefig('ext1_trend.png', dpi=150)
plt.show()
print("Saved ext1_trend.png")


# #########################################################
# EXTENSION 2 — OBJECT-LEVEL METRIC DISTANCE FUSION
# #########################################################

# =========================================================
# EXT2 — STEP A: Open-vocabulary detection, upgraded (Grounding DINO + big vocabulary
# + automatic caption-based augmentation)
# -----------------------------------------------------------------------------
# Two prior attempts and why they undershot:
#   1. Faster R-CNN/COCO: fixed 80-class vocabulary, no "chalkboard" class at all, and its
#      closest match to "table" is literally named "dining table".
#   2. OWL-ViT with a short hand-typed label list: only detects what you explicitly ask for,
#      so a living room full of things not on that list (mirror, pillow, lamp, picture frame...)
#      came back nearly empty.
#
# IMPORTANT — no open-vocabulary detector recognizes "literally anything" with zero prompting.
# Every one of them (Grounding DINO included) still scores the image against a text
# vocabulary; the only real lever is how big/flexible that vocabulary is. So this version:
#   (a) swaps in Grounding DINO, a stronger localizer for text-prompted objects than OWL-ViT
#   (b) uses a much larger curated vocabulary (~130 common indoor/classroom/office nouns)
#   (c) auto-captions the image with BLIP and folds any extra nouns from the caption into the
#       vocabulary too, so scene-specific words not on the curated list still get a chance.
# State this plainly in your Limitations section: this is broad prompted detection, not
# unconstrained open-world recognition.
# =========================================================
!pip install -q transformers

import re
import matplotlib.patches as patches
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, pipeline

gdino_processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(
    "IDEA-Research/grounding-dino-tiny"
).to(device).eval()

# Loaded directly via the model classes rather than pipeline(task="image-to-text", ...) --
# that pipeline task alias has moved around across recent transformers releases (you may see
# a KeyError: "Unknown task image-to-text" on some versions), so this is the version-stable way.
from transformers import BlipProcessor, BlipForConditionalGeneration

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device).eval()

def caption_image(pil_img):
    inputs = blip_processor(pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        out_ids = blip_model.generate(**inputs, max_new_tokens=30)
    return blip_processor.decode(out_ids[0], skip_special_tokens=True)

BASE_VOCABULARY = [
    # seating & tables (one term per concept -- synonyms like chair/armchair or desk/table
    # were previously BOTH in this list, which made Grounding DINO fire twice on the same
    # object and merge the two phrases into one messy label string)
    "chair", "sofa", "ottoman", "stool", "bench", "table", "coffee table", "desk",
    # room fixtures / structure
    "door", "window", "wall", "ceiling", "floor", "staircase", "column", "archway",
    "fireplace", "radiator",
    # classroom / office
    "chalkboard", "podium", "bookshelf", "cabinet", "computer", "monitor", "keyboard",
    "projector", "clock", "poster",
    # decor & soft furnishings
    "pillow", "blanket", "curtain", "rug", "mirror", "painting", "vase",
    "potted plant", "candle", "lamp", "chandelier",
    # small objects
    "book", "box", "basket", "bag", "bottle", "cup", "bowl", "remote control", "tv",
    "speaker", "phone", "laptop", "papers", "notes", "sign", "flyer",
    # people / misc
    "person", "backpack", "shoe", "coat", "umbrella",
]

def build_vocabulary(pil_img, use_caption_augmentation=True):
    vocab = list(BASE_VOCABULARY)
    if use_caption_augmentation:
        caption = caption_image(pil_img)
        # crude heuristic noun-ish extraction: strip stopwords/function words, keep the rest.
        # Not real NLP -- good enough to surface a few extra scene-specific words.
        # Scene-level/collective words (the room itself, not an object in it) must be
        # blocked here -- BLIP's caption often includes one, and if it leaks into the
        # vocabulary Grounding DINO will draw a box around the ENTIRE image for it (this is
        # exactly what produced the giant spurious "classroom" box covering the whole photo).
        scene_level_blacklist = {"room", "rooms", "classroom", "office", "kitchen", "bedroom",
                                  "photo", "photograph", "picture", "image", "scene", "view",
                                  "background", "foreground", "shot", "indoor", "outdoor",
                                  "setting", "space", "area", "interior"}
        stopwords = {"a", "an", "the", "is", "are", "on", "in", "of", "with", "and", "next",
                     "to", "there", "this", "that", "sitting", "standing", "some", "at", "by"}
        words = [w for w in re.findall(r"[a-zA-Z]+", caption.lower())
                 if w not in stopwords and w not in scene_level_blacklist and len(w) > 2]
        vocab += words
        print("BLIP caption:", caption)
    # de-duplicate while preserving order
    seen = set()
    deduped = []
    for w in vocab:
        if w not in seen:
            seen.add(w)
            deduped.append(w)
    return deduped

def detect_objects(rgb_uint8, score_thresh=0.30, text_thresh=0.25, use_caption_augmentation=True):
    pil_img = Image.fromarray(rgb_uint8)
    vocab = build_vocabulary(pil_img, use_caption_augmentation)
    text_prompt = ". ".join(vocab).lower() + "."

    inputs = gdino_processor(images=pil_img, text=text_prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = gdino_model(**inputs)

    # transformers has renamed this parameter across versions (box_threshold -> threshold);
    # try the current name first and fall back to the old one so this keeps working either way.
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
    for box, score, label in zip(result['boxes'], result['scores'], result['labels']):
        x1, y1, x2, y2 = [int(v) for v in box.tolist()]
        # Safety net independent of vocabulary: a real discrete object almost never covers
        # most of the frame. This catches any scene-level false positive (like "classroom")
        # that might still slip past the blacklist above.
        box_area = max(0, x2 - x1) * max(0, y2 - y1)
        if img_area > 0 and box_area / img_area > 0.75:
            continue
        words = label.split()
        deduped_words = list(dict.fromkeys(words))
        clean_label = " ".join(deduped_words) if deduped_words else label
        dets.append(dict(label=clean_label, score=float(score), box=(x1, y1, x2, y2)))

    # Non-max suppression: collapses near-duplicate/overlapping boxes regardless of whether
    # their label text differs, which is what was making the annotated image look cluttered.
    if len(dets) > 1:
        boxes_t = torch.tensor([d['box'] for d in dets], dtype=torch.float32)
        scores_t = torch.tensor([d['score'] for d in dets], dtype=torch.float32)
        keep_idx = torchvision.ops.nms(boxes_t, scores_t, iou_threshold=0.5)
        dets = [dets[i] for i in keep_idx.tolist()]

    return dets

# =========================================================
# EXT2 — STEP B: Fuse detections with UniDepth's dense depth map
# (unchanged from before — detect_objects() now returns a much broader, auto-augmented
# open-vocabulary label set instead of a fixed 6-10 word list)
# =========================================================
ext2_rows = []
for path in baseline_paths:
    rgb = np.array(Image.open(path).convert('RGB'))
    depth, _ = run_inference(rgb)
    dets = detect_objects(rgb)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    axes[0].imshow(rgb); axes[0].axis('off'); axes[0].set_title(f'{path} — detections')
    show_depth(axes[1], depth, title='UniDepth metric depth (m)', fig=fig)

    placed_labels = []  # (x1, x2, y) of labels already drawn, used to dodge collisions below

    def find_free_y(x1, x2, y_start, direction, step=15, max_tries=10):
        y = y_start
        for _ in range(max_tries):
            collision = any(not (x2 < px1 or x1 > px2) and abs(y - py) < 13
                             for (px1, px2, py) in placed_labels)
            if not collision:
                return y
            y += step * direction
        return y

    img_h, img_w = depth.shape
    LARGE_BOX_AREA_RATIO = 0.12  # boxes bigger than this (wall, chalkboard, door...) get a
                                  # fixed label anchor inside themselves instead of competing
                                  # for space in the dense small-object cluster below

    def is_large(box):
        x1, y1, x2, y2 = box
        return ((x2 - x1) * (y2 - y1)) / (img_w * img_h) > LARGE_BOX_AREA_RATIO

    # draw large background elements first, each with a fixed anchor near its own top-centre
    for d in sorted(dets, key=lambda d: -((d['box'][2]-d['box'][0])*(d['box'][3]-d['box'][1]))):
        if not is_large(d['box']):
            continue
        x1, y1, x2, y2 = d['box']
        patch = depth[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        d['median_depth_m'] = float(np.median(patch))
        d['box_bottom_y'] = y2
        axes[0].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                             edgecolor='#5B9BD5', linewidth=2, linestyle='--'))
        label_text = f"{d['label']} {d['median_depth_m']:.1f}m"
        anchor_x = x1 + max(4, (x2 - x1) // 2 - 30)          # roughly centred on the box
        anchor_y = y1 + max(15, int((y2 - y1) * 0.10))       # just inside the top of the box
        axes[0].text(anchor_x, anchor_y, label_text, color='white', fontsize=7.5,
                     bbox=dict(facecolor='#44546A', alpha=0.85, pad=1.5))
        est_width = max(50, 6 * len(label_text))
        placed_labels.append((anchor_x, anchor_x + est_width, anchor_y))
        ext2_rows.append(dict(image=path, label=d['label'], score=round(d['score'], 3),
                               median_depth_m=round(d['median_depth_m'], 3), box_bottom_y=y2,
                               image_height=rgb.shape[0]))

    # then draw small foreground objects, dodging both each other and the large-box labels above
    for d in sorted((d for d in dets if not is_large(d['box'])), key=lambda d: d['box'][0]):
        x1, y1, x2, y2 = d['box']
        patch = depth[y1:y2, x1:x2]
        if patch.size == 0:
            continue
        d['median_depth_m'] = float(np.median(patch))
        d['box_bottom_y'] = y2  # used for the spatial-consistency check below
        axes[0].add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False,
                                             edgecolor='#C9A227', linewidth=2))
        above = y1 - 6 > 15
        label_y_start = y1 - 6 if above else y2 + 14
        label_text = f"{d['label']} {d['median_depth_m']:.1f}m"
        est_width = max(50, 6 * len(label_text))  # rough char-width estimate for collision checks
        label_y = find_free_y(x1, x1 + est_width, label_y_start, direction=-1 if above else 1)
        placed_labels.append((x1, x1 + est_width, label_y))
        axes[0].text(x1, label_y, label_text,
                     color='white', fontsize=7.5, bbox=dict(facecolor='#234D35', alpha=0.85, pad=1.5))
        ext2_rows.append(dict(image=path, label=d['label'], score=round(d['score'], 3),
                               median_depth_m=round(d['median_depth_m'], 3), box_bottom_y=y2,
                               image_height=rgb.shape[0]))

    plt.tight_layout()
    out_name = f"ext2_objects_{path.split('.')[0]}.png"
    plt.savefig(out_name, dpi=140, bbox_inches='tight')
    plt.show()
    print("Saved", out_name)

df_ext2 = pd.DataFrame(ext2_rows)
print(df_ext2.to_string(index=False))
df_ext2.to_csv('ext2_object_depths.csv', index=False)
print("\nSaved ext2_object_depths.csv")

# =========================================================
# EXT2 — STEP C: Spatial-consistency sanity check (no GT needed)
# In most everyday indoor/outdoor photos, an object whose box extends further down the
# frame (larger box_bottom_y) tends to be physically closer to the camera. We measure the
# Spearman correlation between box_bottom_y and predicted depth per image: a strong negative
# correlation is evidence the depth ordering is geometrically sensible.
# =========================================================
from scipy.stats import spearmanr

consistency_rows = []
for path in baseline_paths:
    sub = df_ext2[df_ext2['image'] == path]
    if len(sub) < 3:
        print(f"[{path}] fewer than 3 detections — skipping correlation (not statistically meaningful)")
        continue
    rho, pval = spearmanr(sub['box_bottom_y'], sub['median_depth_m'])
    consistency_rows.append(dict(image=path, n_objects=len(sub),
                                  spearman_rho=round(float(rho), 3), p_value=round(float(pval), 4)))

df_consistency = pd.DataFrame(consistency_rows)
print(df_consistency.to_string(index=False))
df_consistency.to_csv('ext2_spatial_consistency.csv', index=False)
print("\nSaved ext2_spatial_consistency.csv "
      "(expect rho clearly negative if depth ordering is geometrically sensible)")


# #########################################################
# EXTENSION 3 — CROSS-SCALE CONSISTENCY (ViT-S vs ViT-B vs ViT-L)
# #########################################################
# NOTE: UniDepthV2 is only published with three backbones on Hugging Face — vits14, vitb14,
# vitl14. There is no v2 ConvNeXt checkpoint (ConvNeXt-L only exists for the older v1 model,
# under a different repo naming scheme), so this extension instead runs all three official
# v2 sizes on the same image and treats vitl14 (the largest, most accurate model per the
# paper's own Table 1) as the reference. This is a closer, better-controlled version of the
# scale-drift issue noted during reproduction, where the notebook silently picked vits14 on
# a CPU runtime and vitl14 on a GPU runtime.

V2_BACKBONES = ['vits14', 'vitb14', 'vitl14']

# =========================================================
# EXT3 — STEP A: Run all three backbones on each image
# =========================================================
ext3_depths = {}

for bb in V2_BACKBONES:
    m = load_unidepth(bb)
    for path in baseline_paths:
        rgb = np.array(Image.open(path).convert('RGB'))
        depth, _ = run_inference(rgb, active_model=m)
        ext3_depths.setdefault(path, {})[bb] = depth
        print(f"[{path}] {bb:<8s} depth range {depth.min():.2f}-{depth.max():.2f} m")
    free_model(m)

# =========================================================
# EXT3 — STEP B: Quantify agreement vs vitl14 (reference) + global scale ratio
# =========================================================
ext3_rows = []
REFERENCE_BACKBONE = 'vitl14'

for path in baseline_paths:
    d_ref = ext3_depths[path][REFERENCE_BACKBONE]
    for bb in V2_BACKBONES:
        if bb == REFERENCE_BACKBONE:
            continue
        d_test = ext3_depths[path][bb]
        metrics = compare_depth_maps(d_ref, d_test)
        scale_ratio = float(np.median(d_test) / np.median(d_ref))
        ext3_rows.append(dict(image=path, backbone=bb, reference=REFERENCE_BACKBONE,
                               **metrics, median_scale_ratio_vs_ref=round(scale_ratio, 3)))

df_ext3 = pd.DataFrame(ext3_rows)
print(df_ext3.to_string(index=False))
df_ext3.to_csv('ext3_backbone_consistency.csv', index=False)
print("\nSaved ext3_backbone_consistency.csv")

# =========================================================
# EXT3 — STEP C: Side-by-side figure (RGB, ViT-S, ViT-B, ViT-L, |ViT-S - ViT-L| diff)
# =========================================================
for path in baseline_paths:
    rgb = np.array(Image.open(path).convert('RGB'))
    d_s, d_b, d_l = (ext3_depths[path][bb] for bb in V2_BACKBONES)
    diff = np.abs(d_s - d_l)

    fig, axes = plt.subplots(1, 5, figsize=(24, 5))
    axes[0].imshow(rgb); axes[0].axis('off'); axes[0].set_title('Input RGB')
    show_depth(axes[1], d_s, title='UniDepth-ViT-S (m)', fig=fig)
    show_depth(axes[2], d_b, title='UniDepth-ViT-B (m)', fig=fig)
    show_depth(axes[3], d_l, title='UniDepth-ViT-L (m)', fig=fig)
    show_depth(axes[4], diff, title='|ViT-S - ViT-L| (m)', fig=fig)
    plt.tight_layout()
    out_name = f"ext3_backbones_{path.split('.')[0]}.png"
    plt.savefig(out_name, dpi=140, bbox_inches='tight')
    plt.show()
    print("Saved", out_name)


# #########################################################
# FINAL — Save everything to Google Drive
# #########################################################
from google.colab import drive
drive.mount('/content/drive')
!mkdir -p /content/drive/MyDrive/UniDepth_Extension_Results
!cp ext1_*.csv ext1_*.png ext2_*.csv ext2_*.png ext3_*.csv ext3_*.png \
   /content/drive/MyDrive/UniDepth_Extension_Results/ 2>/dev/null
print('Saved to Google Drive/UniDepth_Extension_Results/')
