"""
FSAM-YOLO v2: Few-Shot Attention Meta-YOLO for Reef Detection (FIXED)

CRITICAL FIXES over v1:
  1. MAML loss was `loss = torch.tensor(0.01)` — a fixed constant with no grad_fn,
     meaning loss.backward() was never called. Now uses real YOLOv10 detection
     loss (v10DetectLoss: CIoU + DFL + BCE) computed against ground-truth labels.
  2. Ground-truth labels (.txt annotation files) were completely ignored during MAML.
     Now loaded and used to compute real detection loss.
  3. Reptile outer-loop update was broken: meta_optim.zero_grad() set param.grad=None,
     so the Reptile direction was never applied. Fixed with direct update.
  4. model.args was a plain dict (set during YOLO init) but v8DetectionLoss accesses
     attributes (.box, .cls, .dfl). Now converted to IterableSimpleNamespace.
  5. MAML-trained weights are explicitly transferred to the fine-tuning phase via
     model.ckpt = True, which signals model.train() to use our weights.

Training pipeline:
  Phase 1 — Reptile-style MAML with real v10DetectLoss on real labels
  Phase 2 — Standard YOLOv10 fine-tuning (inherits MAML-initialized weights)
  Phase 3 — Evaluation: precision, recall, mAP on validation set
"""

import os
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
# Hyperparameters
# ---------------------------------------------------------------------------
META_EPOCHS = 10        # outer-loop meta-epochs
TASK_UPDATES = 3         # inner-loop SGD steps per task
META_BATCH = 4           # number of tasks per meta-epoch
META_LR = 0.001          # outer-loop Reptile learning rate (β)
TASK_LR = 0.01           # inner-loop task-specific learning rate (α)
K_SHOT = 5               # support-set size per task
K_QUERY = 5              # query-set size (unused in Reptile, kept for sampling)

FT_EPOCHS = 100
BATCH_SIZE = 8
IMG_SIZE = 640

import torch
import torch.nn as nn
import torch.optim as optim

# Monkey-patch torch.load for PyTorch 2.6+ compatibility
_orig_torch_load = torch.load
torch.load = lambda f, *a, **kw: _orig_torch_load(
    f, *a, **{**kw, "weights_only": kw.get("weights_only", False)}
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
#  Data helpers
# =============================================================================


def _resolve_data_paths():
    """Read reef.yaml and convert relative paths to absolute paths."""
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
    """
    Load one image (resized to IMG_SIZE) and its YOLO-format labels.

    Returns
    -------
    img : torch.Tensor  shape (3, 640, 640), values in [0, 1]
    labels : list[list[float]]  each inner list is [cls_id, cx, cy, w, h] (normalised)
    """
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
    """
    Build a YOLO-format batch dict from a list of (img_tensor, labels_list).

    The returned dict matches what ultralytics dataloader produces so it can
    be consumed directly by v8DetectionLoss / v10DetectLoss.
    """
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
    """
    Ensure model.args is an IterableSimpleNamespace with attribute-accessible
    loss coefficients (.box, .cls, .dfl), as required by v8DetectionLoss.
    """
    from ultralytics.utils import IterableSimpleNamespace

    raw = model.model if hasattr(model, "model") else model
    if raw.args is None:
        raw.args = IterableSimpleNamespace(box=7.5, cls=0.5, dfl=1.5)
    elif isinstance(raw.args, dict):
        raw.args = IterableSimpleNamespace(**raw.args)


# =============================================================================
#  Model loading
# =============================================================================


def load_model():
    """Load the CSAM-enhanced YOLOv10s model with partial pretrained weights."""
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
#  Phase 1 — MAML (Reptile) with proper detection loss
# =============================================================================


def meta_learning(model, data_cfg):
    """
    Reptile-style meta-learning for few-shot detection adaptation.

    Each task:  sample K_SHOT images →  run TASK_UPDATES inner-loop SGD steps
    using the real YOLOv10 detection loss →  Reptile outer-loop meta-update.

    Reptile algorithm:
      for each task:
        θ' ← θ (snapshot)
        for k steps:
          θ' ← θ'  α * ∇loss(θ')
        θ ← θ + β * (θ'  θ) / N_tasks   (move towards adapted weights)

    Key difference from v1:  uses v10DetectLoss (CIoU + DFL + BCE) computed
    against loaded ground-truth labels, NOT a meaningless constant.
    """
    from ultralytics.utils.loss import v10DetectLoss

    train_img_dir = data_cfg["train"]
    train_lbl_dir = data_cfg["train_labels"]

    # Gather all paired (image, label) files
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

    raw_model = model.model  # YOLOv10DetectionModel (nn.Module)
    _ensure_model_args(model)

    criterion = v10DetectLoss(raw_model)

    for epoch in range(META_EPOCHS):
        epoch_total_loss = 0.0
        epoch_total_grad_norm = 0.0
        n_valid_tasks = 0

        for _ in range(META_BATCH):
            # ---- sample a task -------------------------------------------------
            n_need = min(K_SHOT + K_QUERY, len(paired))
            selected = random.sample(paired, n_need)
            support_pairs = selected[:K_SHOT]

            support_data = [
                _load_image_and_labels(ip, lp) for ip, lp in support_pairs
            ]
            support_batch = _collate_batch(support_data, DEVICE)

            # ---- snapshot initial parameters ----------------------------------
            init_params = {
                n: p.detach().clone() for n, p in raw_model.named_parameters()
            }

            # ---- inner loop (task-specific adaptation) -------------------------
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
                    # Clip inner-loop gradients for stability
                    torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=10.0)
                    inner_opt.step()
                    task_loss += loss.item()
                    n_steps += 1

            if n_steps > 0:
                epoch_total_loss += task_loss / n_steps
                n_valid_tasks += 1

            # ---- Reptile outer-loop update ------------------------------------
            # Reptile:  θ ← θ + β  (θ'_adapted  θ'_initial)
            # We apply 1/N_task scaling since META_BATCH tasks share the meta-step.
            with torch.no_grad():
                scale = META_LR / META_BATCH
                for name, param in raw_model.named_parameters():
                    if name in init_params:
                        param.data.add_(scale * (param.data - init_params[name]))

        avg_loss = epoch_total_loss / max(n_valid_tasks, 1)
        print(f"  Meta epoch [{epoch + 1}/{META_EPOCHS}]  "
              f"avg inner loss = {avg_loss:.4f}  "
              f"valid tasks = {n_valid_tasks}/{META_BATCH}")

    print("MAML phase complete — model adapted to reef detection domain.")
    model.ckpt = True  # signals model.train() to transfer MAML weights
    return model


# =============================================================================
#  Phase 2 — Standard YOLOv10 fine-tuning
# =============================================================================


def standard_training(model, data_cfg):
    """
    Fine-tune the MAML-initialised YOLOv10-CSAM model on the reef dataset.

    Because model.ckpt was set to True after MAML, model.train() will call
    trainer.get_model(weights=self.model) which calls DetectionModel.load()
    to transfer the MAML-trained parameters into the fresh training model.
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
        lr0=0.001,
        weight_decay=0.0005,
        warmup_epochs=5,
        close_mosaic=15,
        label_smoothing=0.1,
        amp=False,
        project="runs/fsam_yolo",
        name="train",
        device=DEVICE,
        cos_lr=True,
        patience=0,
    )

    if os.path.exists(temp_yaml):
        os.remove(temp_yaml)
    return results


# =============================================================================
#  Phase 3 — Evaluation
# =============================================================================


def evaluate(model, data_cfg, train_name=None):
    """
    Run validation and compute precision, recall, mAP50, mAP50-95.
    Returns metrics dict.
    """
    from ultralytics import YOLOv10

    # Find best checkpoint — search all train* directories
    best_pt = None
    runs_dir = Path("runs/fsam_yolo")
    if runs_dir.exists():
        weight_files = sorted(runs_dir.glob("train*/weights/best.pt"), reverse=True)
        if weight_files:
            best_pt = weight_files[0]  # most recent
    if best_pt is None or not best_pt.exists():
        print("No checkpoint found for evaluation. Training may have failed.")
        return None

    print(f"\nEvaluating checkpoint: {best_pt}")

    # Build data yaml for validation
    yolo_cfg = {
        k: data_cfg[k]
        for k in ("path", "train", "val", "test", "names")
        if k in data_cfg
    }
    yolo_cfg["nc"] = len(data_cfg.get("names", {}))

    temp_yaml = str(PROJECT_DIR / "_eval_temp.yaml")
    with open(temp_yaml, "w") as f:
        yaml.dump(yolo_cfg, f, default_flow_style=False)

    # Run validation
    val_results = model.val(
        data=temp_yaml,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=0,
        split="val",
        project="runs/fsam_yolo",
        name="eval",
    )

    if os.path.exists(temp_yaml):
        os.remove(temp_yaml)

    if val_results and hasattr(val_results, "results_dict"):
        metrics = val_results.results_dict
        precision = metrics.get("metrics/precision(B)", 0)
        recall = metrics.get("metrics/recall(B)", 0)
        mAP50 = metrics.get("metrics/mAP50(B)", 0)
        mAP50_95 = metrics.get("metrics/mAP50-95(B)", 0)

        print(f"\n{'='*55}")
        print(f"  Evaluation Results")
        print(f"{'='*55}")
        print(f"  Precision : {precision:.4f} ({precision*100:.2f}%)")
        print(f"  Recall    : {recall:.4f} ({recall*100:.2f}%)")
        print(f"  mAP@0.5   : {mAP50:.4f} ({mAP50*100:.2f}%)")
        print(f"  mAP@0.5:0.95: {mAP50_95:.4f} ({mAP50_95*100:.2f}%)")
        print(f"{'='*55}")

        if precision >= 0.98 and recall >= 0.98:
            print(" TARGET ACHIEVED: Precision and Recall both >= 98%!")
        else:
            print(f" Precision gap to 98%: {max(0, 0.98 - precision)*100:.2f}%")
            print(f" Recall gap to 98%:    {max(0, 0.98 - recall)*100:.2f}%")

        return metrics

    return None


# =============================================================================
#  main
# =============================================================================

if __name__ == "__main__":
    print("=" * 55)
    print("  FSAM-YOLO v2  —  Fixed MAML + YOLOv10-CSAM")
    print(f"  Device: {DEVICE}")
    print("=" * 55)

    # 1. Resolve data paths
    data_cfg = _resolve_data_paths()
    print(f"  Dataset  : {data_cfg['base']}")
    print(f"  Train    : {data_cfg['train']}")
    print(f"  Val      : {data_cfg['val']}")

    # 2. Load model with CSAM modules + partial pretrained weights
    model = load_model()

    # 3. Phase 1 — MAML meta-learning with proper detection loss
    model = meta_learning(model, data_cfg)

    # 4. Phase 2 — Standard YOLOv10 fine-tuning
    results = standard_training(model, data_cfg)

    # 5. Phase 3 — Evaluation
    metrics = evaluate(model, data_cfg)

    print("\n  FSAM-YOLO v2 training complete!")
