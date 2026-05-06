
import os
import random
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

# =====================
# Configuration
# =====================
PRIMITIVE_IMG_DIR = Path("datasets/ReefFeat-Img dataset/Primitive/image")
PRIMITIVE_LBL_DIR = Path("datasets/ReefFeat-Img dataset/Primitive/labels")
OUTPUT_DIR = Path("datasets/ReefFeat-Img dataset/Augmented")
AUGS_PER_IMAGE = 30  # number of augmented versions per original image
IMG_SIZE = 640  # resize to this size after augmentation


def load_labels(label_path):
    """Load YOLO format labels: class x_center y_center width height (normalized)."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = list(map(float, line.strip().split()))
            if len(parts) >= 5:
                boxes.append(parts[:5])
    return boxes


def save_labels(label_path, boxes):
    """Save YOLO format labels."""
    with open(label_path, 'w') as f:
        for b in boxes:
            f.write(' '.join(f'{x:.6f}' for x in b) + '\n')


def apply_augmentation(image, boxes, img_size):
    """Apply a random composite augmentation to image and bounding boxes.

    Returns:
        aug_img (HWC uint8), aug_boxes (list of [c, x, y, w, h] normalized)
    """
    h, w = image.shape[:2]
    aug_boxes = [b.copy() for b in boxes]

    # --- 1. Random horizontal flip (50%) ---
    if random.random() < 0.5:
        image = cv2.flip(image, 1)
        for b in aug_boxes:
            b[1] = 1.0 - b[1]  # x_center flipped

    # --- 2. Random vertical flip (30%) ---
    if random.random() < 0.3:
        image = cv2.flip(image, 0)
        for b in aug_boxes:
            b[2] = 1.0 - b[2]  # y_center flipped

    # --- 3. Random rotation (-30 to +30 degrees) ---
    angle = random.uniform(-30, 30)
    if abs(angle) > 1:
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        # Rotate bboxes
        cos_a, sin_a = abs(cos(angle)), abs(sin(angle))
        cos_a = np.cos(np.radians(angle))
        sin_a = np.sin(np.radians(angle))
        for b in aug_boxes:
            cx, cy = b[1] * w, b[2] * h
            bw_, bh_ = b[3] * w, b[4] * h
            # Rotate center
            cx_new = cos_a * (cx - w/2) - sin_a * (cy - h/2) + w/2
            cy_new = sin_a * (cx - w/2) + cos_a * (cy - h/2) + h/2
            # Bbox size increases with rotation (conservative)
            new_w = bw_ * (abs(cos_a) + abs(sin_a))
            new_h = bh_ * (abs(cos_a) + abs(sin_a))
            b[1] = np.clip(cx_new / w, 0.01, 0.99)
            b[2] = np.clip(cy_new / h, 0.01, 0.99)
            b[3] = np.clip(new_w / w, 0.01, 0.99)
            b[4] = np.clip(new_h / h, 0.01, 0.99)

    # --- 4. Random scaling (±20%) ---
    scale = random.uniform(0.8, 1.2)
    if abs(scale - 1.0) > 0.01:
        new_w, new_h = int(w * scale), int(h * scale)
        image = cv2.resize(image, (new_w, new_h))
        for b in aug_boxes:
            b[3] = np.clip(b[3] * scale, 0.01, 0.99)
            b[4] = np.clip(b[4] * scale, 0.01, 0.99)
        h, w = new_h, new_w

    # --- 5. Random translation (±15%) ---
    tx = random.uniform(-0.15, 0.15) * w
    ty = random.uniform(-0.15, 0.15) * h
    if abs(tx) > 1 or abs(ty) > 1:
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        image = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REFLECT)
        for b in aug_boxes:
            b[1] = np.clip(b[1] + tx / w, 0.01, 0.99)
            b[2] = np.clip(b[2] + ty / h, 0.01, 0.99)

    # --- 6. Random brightness/contrast ---
    alpha = random.uniform(0.6, 1.4)  # contrast
    beta = random.uniform(-30, 30)     # brightness
    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # --- 7. Random color jitter (HSV shift) ---
    if random.random() < 0.7:
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180  # hue
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.7, 1.3), 0, 255)  # sat
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.7, 1.3), 0, 255)  # val
        image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # --- 9. Resize to target size for YOLO training ---
    if img_size and (h != img_size or w != img_size):
        image = cv2.resize(image, (img_size, img_size))

    # Filter out boxes that became invalid
    valid_boxes = []
    for b in aug_boxes:
        if b[3] > 0.005 and b[4] > 0.005:  # minimum area
            valid_boxes.append(b)
    aug_boxes = valid_boxes

    return image, aug_boxes


def main():
    # Collect primitive images with labels
    images = sorted(os.listdir(PRIMITIVE_IMG_DIR))
    print(f"Primitive images found: {len(images)}")

    # Create output directories
    out_img_dir = OUTPUT_DIR / "image"
    out_lbl_dir = OUTPUT_DIR / "labels"
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    total_generated = 0
    for img_name in images:
        if not img_name.lower().endswith(('.png', '.jpg')):
            continue

        stem = Path(img_name).stem
        lbl_name = stem + ".txt"
        lbl_path = PRIMITIVE_LBL_DIR / lbl_name

        # Load image
        img_path = PRIMITIVE_IMG_DIR / img_name
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Skipping {img_name}: cannot read")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Load labels
        boxes = load_labels(lbl_path)
        if len(boxes) == 0:
            print(f"  Skipping {img_name}: no labels")
            continue

        # Generate augmented versions
        for aug_idx in range(AUGS_PER_IMAGE):
            aug_img, aug_boxes = apply_augmentation(img.copy(), boxes, IMG_SIZE)

            out_name = f"{stem}_aug_{aug_idx}.png"
            out_lbl_name = f"{stem}_aug_{aug_idx}.txt"

            # Save image
            out_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(out_img_dir / out_name), out_img)

            # Save labels
            save_labels(out_lbl_dir / out_lbl_name, aug_boxes)

            total_generated += 1

        if (images.index(img_name) + 1) % 5 == 0:
            print(f"  Processed {images.index(img_name)+1}/{len(images)} originals...")

    print(f"\n=== Sample Augmentation Complete ===")
    print(f"Originals: {len(images)}")
    print(f"Generated: {total_generated}")
    print(f"Output: {OUTPUT_DIR}/")


if __name__ == '__main__':
    # Need cos for rotation
    from math import cos, sin
    main()
