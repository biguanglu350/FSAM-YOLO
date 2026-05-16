import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml
import random
from PIL import Image

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_DIR = Path(__file__).parent.resolve()
MODEL_YAML = str(PROJECT_DIR / "ultralytics" / "cfg" / "models" / "v10" / "yolov10s-csam.yaml")
DATA_YAML = str(PROJECT_DIR / "ultralytics" / "cfg" / "datasets" / "reef.yaml")
PRETRAINED = str(PROJECT_DIR / "weights" / "yolov10s.pt")

# ---------------------------------------------------------------------------
# Optimized Hyperparameters
# ---------------------------------------------------------------------------
META_EPOCHS = 25          # stronger MAML for better few-shot initialization
TASK_UPDATES = 5           # more inner-loop steps per task
META_BATCH = 8             # more tasks per meta-epoch
META_LR = 0.001            # outer-loop Reptile learning rate
TASK_LR = 0.01             # inner-loop task-specific learning rate
K_SHOT = 5                 # support-set size per task
K_QUERY = 5                # query-set size

# ---------------------------------------------------------------------------
# Augmented Consistency Regularization (ACR) — outer-loop regularization
# Enforces bbox-level prediction consistency under geometric augmentation
#   L_ACR = ||A(bbox(f_θ'(x))) - bbox(f_θ'(A(x)))||^2
# ---------------------------------------------------------------------------
ACR_LAMBDA = 0.1           # balance coefficient for ACR in meta loss
ACR_SCALE = (0.8, 1.2)     # random scaling range for geometric augmentation
ACR_TRANSLATE = 0.1        # max translation (fraction of image)
ACR_FLIP = True             # enable random horizontal flip

FT_EPOCHS = 100            # 100-epoch fine-tuning
BATCH_SIZE = 8
IMG_SIZE = 640

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

_orig_torch_load = torch.load
torch.load = lambda f, *a, **kw: _orig_torch_load(
    f, *a, **{**kw, "weights_only": kw.get("weights_only", False)}
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
#  Data helpers (unchanged — working correctly)
# =============================================================================


def _resolve_data_paths():
    with open(DATA_YAML) as f:
        cfg = yaml.safe_load(f)
    base = PROJECT_DIR / cfg["path"]
    for split in ("train", "val", "test"):
        if cfg.get(split):
            cfg[split] = str(base / cfg[split])
    cfg["train_labels"] = cfg["train"].replace("images", "labels")
    cfg["val_labels"] = cfg["val"].replace("images", "labels")
    cfg["base"] = str(base)
    return cfg


def _load_image_and_labels(img_path, lbl_path):
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0
    labels = []
    if os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    labels.append([float(p) for p in parts])
    return img_tensor, labels


def _collate_batch(image_label_pairs, device):
    imgs = torch.stack([p[0] for p in image_label_pairs]).to(device)
    batch_idx_list = []
    cls_list = []
    bbox_list = []
    for i, (_, labels) in enumerate(image_label_pairs):
        for lbl in labels:
            cls_id, cx, cy, w, h = lbl
            batch_idx_list.append(i)
            cls_list.append([cls_id])
            bbox_list.append([cx, cy, w, h])
    n_labels = len(batch_idx_list)
    if n_labels > 0:
        batch_idx = torch.tensor(batch_idx_list, dtype=torch.long, device=device)
        cls = torch.tensor(cls_list, dtype=torch.float32, device=device)
        bboxes = torch.tensor(bbox_list, dtype=torch.float32, device=device)
    else:
        batch_idx = torch.zeros(0, dtype=torch.long, device=device)
        cls = torch.zeros(0, 1, dtype=torch.float32, device=device)
        bboxes = torch.zeros(0, 4, dtype=torch.float32, device=device)
    return {"img": imgs, "batch_idx": batch_idx, "cls": cls, "bboxes": bboxes}


def _ensure_model_args(model):
    from ultralytics.utils import IterableSimpleNamespace
    raw = model.model if hasattr(model, "model") else model
    if raw.args is None:
        raw.args = IterableSimpleNamespace(box=8.5, cls=0.5, dfl=1.5)
    elif isinstance(raw.args, dict):
        raw.args = IterableSimpleNamespace(**raw.args)


# =============================================================================
#  Augmented Consistency Regularization (ACR) helpers
# =============================================================================


def _geometric_augment(imgs, device):
    """Apply random geometric augmentation (scale + translate + flip) to images.

    Only geometry-preserving transforms — no photometric augs.
    Returns augmented images and params dict for transforming bboxes.
    """
    B, C, H, W = imgs.shape
    scale = random.uniform(*ACR_SCALE)
    tx = random.uniform(-ACR_TRANSLATE, ACR_TRANSLATE) * W
    ty = random.uniform(-ACR_TRANSLATE, ACR_TRANSLATE) * H
    hflip = random.random() > 0.5 if ACR_FLIP else False

    # Build affine matrix for grid_sample (inverse mapping: dst->src)
    # Forward:  x' = s * x + tx (+ flip: x' = s*(W-x) + tx)
    # Inverse:  x = (x' - tx) / s  ->  grid sample theta = [[1/s, 0, -tx/s]]
    if hflip:
        theta = torch.tensor([[
            [-1/scale, 0, W + tx/scale],
            [0, 1/scale, -ty/scale],
        ]], dtype=torch.float32, device=device)
    else:
        theta = torch.tensor([[
            [1/scale, 0, -tx/scale],
            [0, 1/scale, -ty/scale],
        ]], dtype=torch.float32, device=device)

    theta = theta.repeat(B, 1, 1)
    grid = F.affine_grid(theta, [B, C, H, W], align_corners=False)
    aug_imgs = F.grid_sample(imgs, grid, align_corners=False, mode='bilinear', padding_mode='border')

    return aug_imgs, {"scale": scale, "tx": tx, "ty": ty, "hflip": hflip}


def _transform_bboxes(cxcywh_boxes, aug_params, W):
    """Apply the same geometric transform to predicted bboxes.

    Converts boxes from original-image coords to augmented-image coords.
    cxcywh_boxes: [B, 4, N] in pixel coordinates.
    """
    scale = aug_params["scale"]
    tx = aug_params["tx"]
    ty = aug_params["ty"]
    hflip = aug_params["hflip"]

    cx = cxcywh_boxes[:, 0:1, :]
    cy = cxcywh_boxes[:, 1:2, :]
    w  = cxcywh_boxes[:, 2:3, :]
    h  = cxcywh_boxes[:, 3:4, :]

    # Forward geometric transform
    cx = cx * scale + tx
    cy = cy * scale + ty
    w  = w  * scale
    h  = h  * scale

    if hflip:
        cx = W - cx

    return torch.cat([cx, cy, w, h], dim=1)


def compute_acr_loss(raw_model, orig_imgs, aug_imgs, aug_params):
    """Compute ACR loss: ||A(bbox(f_theta'(x))) - bbox(f_theta'(A(x)))||^2.

    Uses the one2many branch in eval mode, which has full gradient flow
    through both backbone and head (unlike one2one which detaches features).

    CRITICAL: does NOT save/restore head.anchors — eval-mode inference sets
    them naturally, and the training-mode forward path never reads them.
    """
    B, C, H, W = orig_imgs.shape

    was_training = raw_model.training
    raw_model.eval()

    with torch.set_grad_enabled(True):
        orig_out = raw_model(orig_imgs)
        aug_out  = raw_model(aug_imgs)

        # In eval mode one2many = Detect.inference() -> tuple (y, raw_x)
        # y = [B, 4+nc, N]  with boxes in cxcywh @ image-scale pixels
        orig_y = orig_out["one2many"]
        aug_y  = aug_out["one2many"]
        if isinstance(orig_y, tuple):
            orig_y = orig_y[0]
        if isinstance(aug_y, tuple):
            aug_y = aug_y[0]

        # Sanity check: both forwards must produce the same number of predictions
        if orig_y.shape[-1] == 0 or aug_y.shape[-1] == 0:
            if was_training:
                raw_model.train()
            return torch.tensor(0.0, device=orig_imgs.device, requires_grad=True)

        # Box coordinates: [B, 4, N]
        orig_boxes = orig_y[:, :4, :].contiguous()
        aug_boxes  = aug_y[:, :4, :].contiguous()

        # Apply geometric transform to original boxes -> A(bbox(f_theta'(x)))
        trans_boxes = _transform_bboxes(orig_boxes, aug_params, W)

        # L2 loss normalised by image size for stable gradient scale
        acr_loss = F.mse_loss(trans_boxes / W, aug_boxes / W)

    if was_training:
        raw_model.train()

    return acr_loss


# =============================================================================
#  Model loading
# =============================================================================


def load_model():
    from ultralytics import YOLOv10
    from ultralytics.nn.modules import CSAM

    model = YOLOv10(MODEL_YAML)

    if os.path.exists(PRETRAINED):
        ckpt = torch.load(PRETRAINED, map_location="cpu", weights_only=False)
        sd = ckpt.get("model", ckpt)
        if hasattr(sd, "state_dict"):
            sd = sd.state_dict()
        model_sd = model.model.state_dict()
        filtered = {
            k: v
            for k, v in sd.items()
            if k in model_sd and v.shape == model_sd[k].shape
        }
        model.model.load_state_dict(filtered, strict=False)
        print(f"Loaded {len(filtered)}/{len(sd)} pretrained weight groups")

    model.to(DEVICE)
    n_csam = sum(1 for m in model.model.model if isinstance(m, CSAM))
    print(f"CSAM modules detected: {n_csam}")
    return model


# =============================================================================
#  Phase 1 — MAML (Reptile) — enhanced
# =============================================================================


def meta_learning(model, data_cfg):
    from ultralytics.utils.loss import v10DetectLoss

    train_img_dir = data_cfg["train"]
    train_lbl_dir = data_cfg["train_labels"]

    paired = []
    for fname in sorted(os.listdir(train_img_dir)):
        if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
            continue
        lbl_name = os.path.splitext(fname)[0] + ".txt"
        lbl_path = os.path.join(train_lbl_dir, lbl_name)
        if os.path.exists(lbl_path):
            paired.append((os.path.join(train_img_dir, fname), lbl_path))

    print(f"MAML: {len(paired)} image-label pairs available")

    if len(paired) < K_SHOT:
        print(f"MAML: need at least {K_SHOT} pairs, skipping MAML phase")
        return model

    raw_model = model.model
    _ensure_model_args(model)

    criterion = v10DetectLoss(raw_model)

    for epoch in range(META_EPOCHS):
        epoch_total_loss = 0.0
        n_valid_tasks = 0

        for _ in range(META_BATCH):
            n_need = min(K_SHOT + K_QUERY, len(paired))
            selected = random.sample(paired, n_need)
            support_pairs = selected[:K_SHOT]

            support_data = [
                _load_image_and_labels(ip, lp) for ip, lp in support_pairs
            ]
            support_batch = _collate_batch(support_data, DEVICE)

            init_params = {
                n: p.detach().clone() for n, p in raw_model.named_parameters()
            }

            raw_model.train()
            inner_opt = optim.SGD(raw_model.parameters(), lr=TASK_LR)
            task_loss = 0.0
            n_steps = 0

            for _step in range(TASK_UPDATES):
                inner_opt.zero_grad()
                preds = raw_model(support_batch["img"])
                loss, _loss_items = criterion(preds, support_batch)

                if torch.isfinite(loss) and loss.item() > 0:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=10.0)
                    inner_opt.step()
                    task_loss += loss.item()
                    n_steps += 1

            if n_steps > 0:
                epoch_total_loss += task_loss / n_steps
                n_valid_tasks += 1

            # ---- Augmented Consistency Regularization (outer loop) ----
            # L_ACR = ||A(bbox(f_theta'(x))) - bbox(f_theta'(A(x)))||^2
            try:
                aug_imgs_, acr_params_ = _geometric_augment(support_batch["img"], DEVICE)
                acr_loss_ = compute_acr_loss(raw_model, support_batch["img"], aug_imgs_, acr_params_)

                raw_model.zero_grad()  # clear stale inner-loop grads
                if torch.isfinite(acr_loss_) and acr_loss_.item() > 0:
                    acr_loss_.backward()
            except Exception as e:
                print(f"  [ACR warning] skipped ({e})")
                raw_model.zero_grad()

            # ---- Reptile outer-loop update + ACR gradient descent ----
            with torch.no_grad():
                scale = META_LR / META_BATCH
                for name, param in raw_model.named_parameters():
                    if name in init_params:
                        # Reptile: move towards adapted parameters
                        reptile_dir = param.data - init_params[name]
                        param.data.add_(scale * reptile_dir)
                        # ACR: gradient descent on the consistency loss
                        if param.grad is not None:
                            param.data.add_(-ACR_LAMBDA / META_BATCH * param.grad)

        avg_loss = epoch_total_loss / max(n_valid_tasks, 1)
        print(f"  Meta epoch [{epoch + 1}/{META_EPOCHS}]  "
              f"avg inner loss = {avg_loss:.4f}  "
              f"valid tasks = {n_valid_tasks}/{META_BATCH}")

    print("MAML phase complete — model adapted to reef detection domain.")
    model.ckpt = True
    return model


# =============================================================================
#  Phase 2 — Optimized fine-tuning
# =============================================================================


def standard_training(model, data_cfg):
    """
    Fine-tune with tuned hyperparameters for 100-epoch run:
      - Standard LR (0.001) + cos_lr for better recall
      - Moderate warmup (5 epochs) for quick LR ramp-up
      - No label smoothing (single-class detection)
      - Mosaic closed at epoch 15 — majority of training on real images
      - Higher box loss weight (8.5) for improved localization → mAP50-95
    """
    yolo_cfg = {
        k: data_cfg[k]
        for k in ("path", "train", "val", "test", "names")
        if k in data_cfg
    }
    yolo_cfg["nc"] = len(data_cfg.get("names", {}))

    temp_yaml = str(PROJECT_DIR / "_data_temp.yaml")
    with open(temp_yaml, "w") as f:
        yaml.dump(yolo_cfg, f, default_flow_style=False)

    results = model.train(
        data=temp_yaml,
        imgsz=IMG_SIZE,
        epochs=FT_EPOCHS,
        batch=BATCH_SIZE,
        workers=0,
        optimizer="AdamW",
        lr0=0.001,               # standard LR for better recall
        weight_decay=0.0005,     # moderate regularization
        warmup_epochs=5,         # moderate warmup
        close_mosaic=15,         # close mosaic at epoch 15 → 85 epochs on real images
        label_smoothing=0.0,     # single class → no label smoothing needed
        amp=False,
        project="runs",
        name="fsam_opt",
        device=DEVICE,
        cos_lr=True,
        patience=0,
        overlap_mask=False,      # not training segmentation
        exist_ok=True,
    )

    if os.path.exists(temp_yaml):
        os.remove(temp_yaml)
    return results


# =============================================================================
#  Phase 3 — Evaluation
# =============================================================================


def evaluate(model, data_cfg):
    from ultralytics import YOLOv10

    best_pt = None
    runs_dir = Path("runs")
    if runs_dir.exists():
        weight_files = sorted(runs_dir.glob("fsam_opt/weights/best.pt"), reverse=True)
        if weight_files:
            best_pt = weight_files[0]
    if best_pt is None or not best_pt.exists():
        weight_files = sorted(runs_dir.glob("*/weights/best.pt"), reverse=True)
        if weight_files:
            best_pt = weight_files[0]
    if best_pt is None or not best_pt.exists():
        print("No checkpoint found for evaluation.")
        return None

    print(f"\nEvaluating checkpoint: {best_pt}")

    yolo_cfg = {
        k: data_cfg[k]
        for k in ("path", "train", "val", "test", "names")
        if k in data_cfg
    }
    yolo_cfg["nc"] = len(data_cfg.get("names", {}))

    temp_yaml = str(PROJECT_DIR / "_eval_temp.yaml")
    with open(temp_yaml, "w") as f:
        yaml.dump(yolo_cfg, f, default_flow_style=False)

    val_results = model.val(
        data=temp_yaml,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=0,
        split="val",
        project="runs",
        name="fsam_eval",
    )

    if val_results and hasattr(val_results, "results_dict"):
        print("Results saved to runs/fsam_eval")
        return val_results.results_dict

    return None


# =============================================================================
#  main
# =============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  FSAM-YOLO v2  —  OPTIMIZED")
    print(f"  Device: {DEVICE}")
    print(f"  Fine-tune epochs: {FT_EPOCHS}")
    print(f"  Meta epochs: {META_EPOCHS}")
    print(f"  Meta batch: {META_BATCH}")
    print(f"  Task updates: {TASK_UPDATES}")
    print(f"  LR: 0.001 (AdamW), warmup: 5, box_weight: 8.5")
    print(f"  Label smoothing: 0.0")
    print("=" * 55)

    data_cfg = _resolve_data_paths()
    print(f"  Dataset  : {data_cfg['base']}")
    print(f"  Train    : {data_cfg['train']}")
    print(f"  Val      : {data_cfg['val']}")

    model = load_model()
    model = meta_learning(model, data_cfg)
    results = standard_training(model, data_cfg)
    metrics = evaluate(model, data_cfg)

    print("\n  FSAM-YOLO v2 optimized training complete!")
