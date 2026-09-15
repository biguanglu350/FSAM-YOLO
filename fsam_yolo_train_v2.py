"""
FSAM-YOLO v2 — 在线增强 + ACR 统一训练脚本

核心设计：
1. 从 Primitive 原始数据中选取 10 张图（7 train + 3 val）
2. 在线生成三种增强：几何变换、色彩变换、混合变换
3. 所有增强保存到硬盘，供 MAML 和 YOLO 微调使用
4. MAML 阶段：几何/混合变换的样本额外计算 ACR 损失
5. ACR 监督真实应用于训练数据的几何变换（参数从元数据读取）
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import json
import math
import random
import warnings
import shutil
from pathlib import Path
from copy import deepcopy

import cv2
import numpy as np
import yaml
from PIL import Image

warnings.filterwarnings("ignore")

# =============================================================================
# 配置
# =============================================================================
PROJECT_DIR = Path(__file__).parent.resolve()
PRIMITIVE_IMG_DIR = PROJECT_DIR / "datasets" / "ReefFeat-Img dataset" / "Primitive" / "image"
PRIMITIVE_LBL_DIR = PROJECT_DIR / "datasets" / "ReefFeat-Img dataset" / "Primitive" / "labels"
OUTPUT_DATASET_DIR = PROJECT_DIR / "datasets" / "reef10_aug"
OUTPUT_METADATA_PATH = OUTPUT_DATASET_DIR / "metadata.json"

# 模型相关
MODEL_YAML = str(PROJECT_DIR / "ultralytics" / "cfg" / "models" / "v10" / "yolov10s-csam.yaml")
PRETRAINED = str(PROJECT_DIR / "weights" / "yolov10s.pt")

# 训练超参数
META_EPOCHS = 25
TASK_UPDATES = 5
META_BATCH = 8
META_LR = 0.001
TASK_LR = 0.01
K_SHOT = 5

# ACR 配置
ACR_LAMBDA = 0.1

# 微调配置
FT_EPOCHS = 100
BATCH_SIZE = 8
IMG_SIZE = 640

# 数据增强配置
AUGS_PER_IMAGE = 30  # 每张原图生成 30 张增强图
RANDOM_SEED = 42

# =============================================================================
# PyTorch 导入（后置，避免导入冲突）
# =============================================================================
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 兼容 PyTorch 2.6+ 的 checkpoint 加载
_orig_torch_load = torch.load
torch.load = lambda f, *a, **kw: _orig_torch_load(
    f, *a, **{**kw, "weights_only": kw.get("weights_only", False)}
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# 第一部分：数据集初始化 — 选图、分集、生成增强
# =============================================================================

def setup_dataset():
    """
    从 Primitive 中选 10 张图，建立 reef10_aug 数据集目录结构。
    只执行一次，不会覆盖已有数据。
    """
    if OUTPUT_DATASET_DIR.exists():
        print(f"[数据集] 已存在: {OUTPUT_DATASET_DIR}")
        # 检查元数据是否存在
        if OUTPUT_METADATA_PATH.exists():
            print("[数据集] 元数据已存在，跳过初始化")
            return
        else:
            print("[数据集] 缺少元数据，重新生成")

    # 选 10 张非增强原始图
    all_imgs = sorted([
        f for f in os.listdir(PRIMITIVE_IMG_DIR)
        if f.endswith(".png") and "_aug" not in f
    ])
    # 固定种子保证可复现，选前 10 个
    random.seed(RANDOM_SEED)
    random.shuffle(all_imgs)
    selected = all_imgs[:10]

    train_imgs = selected[:7]
    val_imgs = selected[7:10]

    print(f"[数据集] 选中 {len(selected)} 张原始图片")
    print(f"  训练集: {[f[:10] for f in train_imgs]}")
    print(f"  验证集: {[f[:10] for f in val_imgs]}")

    # 创建目录结构
    for split in ["train", "val"]:
        img_dir = OUTPUT_DATASET_DIR / split / "images"
        lbl_dir = OUTPUT_DATASET_DIR / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

    # 记录元数据
    metadata = {}

    for split_name, img_list in [("train", train_imgs), ("val", val_imgs)]:
        for img_name in img_list:
            stem = Path(img_name).stem
            src_img = PRIMITIVE_IMG_DIR / img_name
            src_lbl = PRIMITIVE_LBL_DIR / (stem + ".txt")

            # 读取原始图像和标签
            img_pil = Image.open(src_img).convert("RGB")
            labels = _read_yolo_labels(src_lbl)

            # 生成增强版本
            aug_gen_count = 0
            for aug_idx in range(AUGS_PER_IMAGE):
                # 三路分配：0-9几何，10-19色彩，20-29混合
                aug_type = (
                    "geom" if aug_idx < 10
                    else "color" if aug_idx < 20
                    else "mixed"
                )

                out_name = f"{stem}_{aug_type}_{aug_idx:03d}.png"
                out_lbl_name = f"{stem}_{aug_type}_{aug_idx:03d}.txt"
                out_img_path = OUTPUT_DATASET_DIR / split_name / "images" / out_name
                out_lbl_path = OUTPUT_DATASET_DIR / split_name / "labels" / out_lbl_name

                if out_img_path.exists():
                    aug_gen_count += 1
                    continue

                # 执行增强
                if aug_type == "geom":
                    aug_img, aug_labels, geom_params = _augment_geometric(img_pil, labels)
                    metadata[out_name] = {
                        "original": img_name,
                        "split": split_name,
                        "type": "geometric",
                        "has_geometric": True,
                        "geom_params": geom_params
                    }
                elif aug_type == "color":
                    aug_img, aug_labels = _augment_color(img_pil, labels)
                    metadata[out_name] = {
                        "original": img_name,
                        "split": split_name,
                        "type": "color",
                        "has_geometric": False
                    }
                else:  # mixed
                    # 先几何再色彩
                    aug_img, aug_labels, geom_params = _augment_geometric(img_pil, labels)
                    aug_img, aug_labels = _augment_color_on_image(aug_img, aug_labels)
                    metadata[out_name] = {
                        "original": img_name,
                        "split": split_name,
                        "type": "mixed",
                        "has_geometric": True,
                        "geom_params": geom_params
                    }

                # 保存到硬盘
                aug_img.save(str(out_img_path))
                _save_yolo_labels(out_lbl_path, aug_labels)
                aug_gen_count += 1

            print(f"  [{split_name}] {stem}: 生成 {aug_gen_count}/{AUGS_PER_IMAGE} 张增强")

    # 保存元数据
    with open(OUTPUT_METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[数据集] 元数据已保存: {OUTPUT_METADATA_PATH}")
    print(f"[数据集] 总计生成增强图片: {sum(1 for v in metadata.values())} 张")


def _read_yolo_labels(path):
    """读取 YOLO 格式标签文件"""
    boxes = []
    if not os.path.exists(path):
        return boxes
    with open(path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                boxes.append([float(p) for p in parts])
    return boxes


def _save_yolo_labels(path, boxes):
    """保存 YOLO 格式标签文件"""
    with open(path, "w") as f:
        for b in boxes:
            f.write(" ".join(f"{x:.6f}" for x in b) + "\n")


def _augment_geometric(img_pil, labels):
    """
    几何变换增强：翻转 → 旋转 → 缩放裁剪 → 平移

    严格按照 sample_augmentation.py 的风格：
    - 先把原图 resize 到 640×640
    - 再顺序进行几何操作（每个步骤都跟踪参数，用于 ACR）
    - 返回增强后的图像、标签、完整几何参数
    """
    W = H = IMG_SIZE

    # 先 resize 到 640×640，保证坐标统一
    img_pil = img_pil.resize((W, H), Image.BILINEAR)
    img_np = np.array(img_pil).astype(np.uint8)  # [640, 640, 3]

    # ========== 随机生成参数 ==========
    hflip = random.random() > 0.5          # 50% 概率翻转
    angle = random.uniform(-30, 30)        # -30° ~ 30° 旋转
    scale = random.uniform(0.8, 1.2)       # 0.8x ~ 1.2x 缩放
    tx = random.uniform(-0.15, 0.15) * W   # ±15% 平移
    ty = random.uniform(-0.15, 0.15) * H

    # ========== 对图像做顺序变换 ==========
    # 1. 水平翻转
    if hflip:
        img_np = cv2.flip(img_np, 1)

    # 2. 旋转（绕中心）
    if abs(angle) > 0.5:
        M_rot = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
        img_np = cv2.warpAffine(img_np, M_rot, (W, H), borderMode=cv2.BORDER_REFLECT)

    # 3. 缩放 + 居中裁剪/填充保持 640×640
    if abs(scale - 1.0) > 0.01:
        new_w = max(int(W * scale), 1)
        new_h = max(int(H * scale), 1)
        img_np = cv2.resize(img_np, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        if scale > 1.0:
            # 放大 → 居中裁剪
            sx = (new_w - W) // 2
            sy = (new_h - H) // 2
            img_np = img_np[sy:sy + H, sx:sx + W]
        else:
            # 缩小 → 居中填充灰色背景
            pad_x = (W - new_w) // 2
            pad_y = (H - new_h) // 2
            canvas = np.full((H, W, 3), 114, dtype=np.uint8)
            canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = img_np
            img_np = canvas

    # 4. 平移
    if abs(tx) > 1 or abs(ty) > 1:
        M_trans = np.float32([[1, 0, tx], [0, 1, ty]])
        img_np = cv2.warpAffine(img_np, M_trans, (W, H), borderMode=cv2.BORDER_REFLECT)

    aug_img = Image.fromarray(img_np)

    # ========== 对标签做相同顺序的变换（像素坐标） ==========
    aug_labels = []
    for b in labels:
        cls_id, cx, cy, w, h = b

        # 从归一化转到像素坐标
        cx_px = cx * W
        cy_px = cy * H
        w_px = w * W
        h_px = h * H

        # 1. 水平翻转
        if hflip:
            cx_px = W - cx_px

        # 2. 旋转（绕中心）
        if abs(angle) > 0.5:
            cos_a = np.cos(np.radians(angle))
            sin_a = np.sin(np.radians(angle))
            new_cx = cos_a * (cx_px - W / 2) - sin_a * (cy_px - H / 2) + W / 2
            new_cy = sin_a * (cx_px - W / 2) + cos_a * (cy_px - H / 2) + H / 2
            cx_px = new_cx
            cy_px = new_cy
            # bbox 随旋转扩大
            w_px = w_px * (abs(cos_a) + abs(sin_a))
            h_px = h_px * (abs(cos_a) + abs(sin_a))

        # 3. 缩放 + 居中裁切偏移
        if abs(scale - 1.0) > 0.01:
            # 等价的公式: cx' = (cx - W/2) * scale + W/2
            cx_px = (cx_px - W / 2) * scale + W / 2
            cy_px = (cy_px - H / 2) * scale + H / 2
            w_px = w_px * scale
            h_px = h_px * scale

        # 4. 平移
        cx_px += tx
        cy_px += ty

        # 转回归一化 + 裁剪
        cx_n = np.clip(cx_px / W, 0.01, 0.99)
        cy_n = np.clip(cy_px / H, 0.01, 0.99)
        w_n = np.clip(w_px / W, 0.01, 0.99)
        h_n = np.clip(h_px / H, 0.01, 0.99)

        if w_n > 0.005 and h_n > 0.005:
            aug_labels.append([cls_id, cx_n, cy_n, w_n, h_n])

    geom_params = {
        "hflip": hflip,
        "angle": angle,
        "scale": scale,
        "tx": tx,
        "ty": ty,
        "W": W,
    }
    return aug_img, aug_labels, geom_params


def _augment_color(img_pil, labels):
    """
    色彩变换增强：亮度 + 对比度 + HSV 抖动
    先 resize 到 640×640，不改变标签位置
    """
    W = H = IMG_SIZE
    img_pil = img_pil.resize((W, H), Image.BILINEAR)
    img_np = np.array(img_pil).astype(np.float32)

    # 亮度/对比度
    alpha = random.uniform(0.6, 1.4)  # 对比度
    beta = random.uniform(-30, 30)    # 亮度
    img_np = np.clip(img_np * alpha + beta, 0, 255).astype(np.uint8)

    # HSV 抖动（70% 概率）
    if random.random() < 0.7:
        hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + random.uniform(-10, 10)) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.7, 1.3), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.7, 1.3), 0, 255)
        img_np = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    aug_img = Image.fromarray(img_np)
    return aug_img, [list(b) for b in labels]


def _augment_color_on_image(img_pil, labels):
    """对已经是几何变换后的图像做色彩变换"""
    return _augment_color(img_pil, labels)


# =============================================================================
# 第二部分：数据加载
# =============================================================================

def load_image_and_labels(img_path, lbl_path):
    """加载单张图像和标签，统一 resize 到 IMG_SIZE"""
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    img_tensor = torch.from_numpy(np.array(img)).float().permute(2, 0, 1) / 255.0

    labels = []
    if lbl_path and os.path.exists(lbl_path):
        with open(lbl_path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    labels.append([float(p) for p in parts])
    return img_tensor, labels


def collate_batch(image_label_pairs, device):
    """整理 batch: list of (img_tensor, labels) → model input dict"""
    imgs = torch.stack([p[0] for p in image_label_pairs]).to(device)
    batch_idx_list, cls_list, bbox_list = [], [], []
    for i, (_, labels) in enumerate(image_label_pairs):
        for lbl in labels:
            cls_id, cx, cy, w, h = lbl
            batch_idx_list.append(i)
            cls_list.append([cls_id])
            bbox_list.append([cx, cy, w, h])

    n_labels = len(batch_idx_list)
    if n_labels > 0:
        return {
            "img": imgs,
            "batch_idx": torch.tensor(batch_idx_list, dtype=torch.long, device=device),
            "cls": torch.tensor(cls_list, dtype=torch.float32, device=device),
            "bboxes": torch.tensor(bbox_list, dtype=torch.float32, device=device),
        }
    return {
        "img": imgs,
        "batch_idx": torch.zeros(0, dtype=torch.long, device=device),
        "cls": torch.zeros(0, 1, dtype=torch.float32, device=device),
        "bboxes": torch.zeros(0, 4, dtype=torch.float32, device=device),
    }


def load_metadata():
    """加载增强元数据"""
    if not OUTPUT_METADATA_PATH.exists():
        raise FileNotFoundError(f"元数据不存在: {OUTPUT_METADATA_PATH}")
    with open(OUTPUT_METADATA_PATH) as f:
        return json.load(f)


def _ensure_model_args(model):
    """确保模型有 loss 函数需要的 args 属性"""
    from ultralytics.utils import IterableSimpleNamespace
    raw = model.model if hasattr(model, "model") else model
    if raw.args is None:
        raw.args = IterableSimpleNamespace(box=8.5, cls=0.5, dfl=1.5)
    elif isinstance(raw.args, dict):
        raw.args = IterableSimpleNamespace(**raw.args)


# =============================================================================
# 第三部分：ACR 损失
# =============================================================================

def transform_bboxes_pixel(cxcywh_boxes, geom_params, W):
    """
    对预测 bbox（像素坐标）应用完整几何变换（正向）。

    变换顺序必须与 _augment_geometric 一致：
      flip → rotate → scale → translate

    cxcywh_boxes: [B, 4, N]  像素坐标
    """
    cx = cxcywh_boxes[:, 0:1, :]  # [B, 1, N]
    cy = cxcywh_boxes[:, 1:2, :]
    w_ = cxcywh_boxes[:, 2:3, :]
    h_ = cxcywh_boxes[:, 3:4, :]

    # 1. 水平翻转
    if geom_params.get("hflip", False):
        cx = W - cx

    # 2. 旋转
    angle = geom_params.get("angle", 0)
    if abs(angle) > 0.5:
        a_rad = math.radians(angle)
        cos_a = math.cos(a_rad)
        sin_a = math.sin(a_rad)
        cx_off = cx - W / 2
        cy_off = cy - W / 2  # H == W == IMG_SIZE
        new_cx = cos_a * cx_off - sin_a * cy_off + W / 2
        new_cy = sin_a * cx_off + cos_a * cy_off + W / 2
        cx = new_cx
        cy = new_cy
        # bbox 随旋转扩大
        w_ = w_ * (abs(cos_a) + abs(sin_a))
        h_ = h_ * (abs(cos_a) + abs(sin_a))

    # 3. 缩放 + 居中裁切偏移
    scale = geom_params.get("scale", 1.0)
    if abs(scale - 1.0) > 0.01:
        cx = (cx - W / 2) * scale + W / 2
        cy = (cy - W / 2) * scale + W / 2
        w_ = w_ * scale
        h_ = h_ * scale

    # 4. 平移
    tx = geom_params.get("tx", 0)
    ty = geom_params.get("ty", 0)
    cx = cx + tx
    cy = cy + ty

    return torch.cat([cx, cy, w_, h_], dim=1)


def compute_acr_loss(raw_model, orig_img_tensor, aug_img_tensor, geom_params, device):
    """
    计算单对 (原始图, 增强图) 的 ACR 损失。

    L_ACR = || A(bbox(f_θ'(x))) - bbox(f_θ'(A(x))) ||²

    - orig_img_tensor: [1, 3, 640, 640]  原始图像（resize 后）
    - aug_img_tensor:  [1, 3, 640, 640]  几何变换后的图像
    - geom_params:     变换参数（scale, tx, ty, hflip）
    """
    B, C, H, W = orig_img_tensor.shape

    was_training = raw_model.training
    raw_model.eval()  # eval 模式得到解码后的预测

    with torch.set_grad_enabled(True):
        orig_out = raw_model(orig_img_tensor)
        aug_out = raw_model(aug_img_tensor)

        # 取 one2many 分支的解码后结果
        orig_y = orig_out["one2many"]
        aug_y = aug_out["one2many"]
        if isinstance(orig_y, tuple):
            orig_y = orig_y[0]
        if isinstance(aug_y, tuple):
            aug_y = aug_y[0]

        if orig_y.shape[-1] == 0 or aug_y.shape[-1] == 0:
            if was_training:
                raw_model.train()
            return torch.tensor(0.0, device=device, requires_grad=True)

        # bbox 坐标: [1, 4, N]
        orig_boxes = orig_y[:, :4, :].contiguous()
        aug_boxes = aug_y[:, :4, :].contiguous()

        # A(bbox(f_θ'(x))) — 对原始预测做同样变换
        trans_boxes = transform_bboxes_pixel(orig_boxes, geom_params, W)

        # L2 loss，按图像尺寸归一化
        acr_loss = F.mse_loss(trans_boxes / W, aug_boxes / W)

    if was_training:
        raw_model.train()

    return acr_loss


# =============================================================================
# 第四部分：模型加载
# =============================================================================

def load_model():
    """加载 YOLOv10s-CSAM 模型 + 预训练权重"""
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
            k: v for k, v in sd.items()
            if k in model_sd and v.shape == model_sd[k].shape
        }
        model.model.load_state_dict(filtered, strict=False)
        print(f"[模型] 加载 {len(filtered)}/{len(sd)} 组预训练权重")

    model.to(DEVICE)
    n_csam = sum(1 for m in model.model.model if isinstance(m, CSAM))
    print(f"[模型] CSAM 模块数: {n_csam}")
    return model


# =============================================================================
# 第五部分：MAML 元学习（带集成 ACR）
# =============================================================================

def meta_learning(model, metadata, meta_epochs=None):
    """
    MAML（Reptile）元学习阶段。

    关键设计：
    - 支持集从增强后的训练集中采样（几何/色彩/混合混合）
    - 内循环在增强样本上微调（TASK_UPDATES 步 SGD）
    - ACR 只对几何/混合增强的样本计算
    - ACR 使用元数据中的几何参数，监督"真实应用的变换"
    """
    from ultralytics.utils.loss import v10DetectLoss
    n_meta_epochs = meta_epochs if meta_epochs is not None else META_EPOCHS

    # 构建增强训练集列表
    train_dir = OUTPUT_DATASET_DIR / "train"
    train_img_dir = train_dir / "images"
    train_lbl_dir = train_dir / "labels"

    # 元数据中属于训练集的条目
    train_meta = {k: v for k, v in metadata.items() if v["split"] == "train"}
    train_entries = list(train_meta.keys())

    if len(train_entries) == 0:
        print("[MAML] 训练集为空，跳过")
        return model

    print(f"[MAML] 增强训练集大小: {len(train_entries)} 张")
    n_geom = sum(1 for v in train_meta.values() if v["has_geometric"])
    print(f"[MAML] 其中几何/混合（可算 ACR）: {n_geom} 张")

    raw_model = model.model
    _ensure_model_args(model)
    criterion = v10DetectLoss(raw_model)

    for epoch in range(n_meta_epochs):
        epoch_total_loss = 0.0
        n_valid_tasks = 0

        for _ in range(META_BATCH):
            # ---- 采样 K_SHOT 个增强样本 ----
            n_need = min(K_SHOT, len(train_entries))
            selected_names = random.sample(train_entries, n_need)

            support_data = []
            acr_pairs = []  # (orig_tensor, aug_tensor, geom_params) for ACR

            for img_name in selected_names:
                meta_entry = train_meta[img_name]

                # 加载增强图像
                aug_img_path = train_img_dir / img_name
                aug_lbl_path = train_lbl_dir / img_name.replace(".png", ".txt")
                aug_tensor, aug_labels = load_image_and_labels(
                    str(aug_img_path), str(aug_lbl_path)
                )
                support_data.append((aug_tensor, aug_labels))

                # 如果是几何/混合样本，准备 ACR 对
                if meta_entry["has_geometric"]:
                    orig_name = meta_entry["original"]
                    orig_img_path = PRIMITIVE_IMG_DIR / orig_name
                    # 加载原始图像并 resize 到 640
                    orig_tensor, _ = load_image_and_labels(str(orig_img_path), None)
                    acr_pairs.append({
                        "orig": orig_tensor,
                        "aug": aug_tensor,
                        "geom_params": meta_entry["geom_params"]
                    })

            if len(support_data) < 1:
                continue

            support_batch = collate_batch(support_data, DEVICE)

            # ---- 保存初始参数 θ ----
            init_params = {
                n: p.detach().clone()
                for n, p in raw_model.named_parameters()
            }

            # ---- 内循环：在增强样本上微调 θ → θ'_i ----
            raw_model.train()
            inner_opt = optim.SGD(raw_model.parameters(), lr=TASK_LR)
            task_loss_sum = 0.0
            n_steps = 0

            for _step in range(TASK_UPDATES):
                inner_opt.zero_grad()
                preds = raw_model(support_batch["img"])
                loss, _loss_items = criterion(preds, support_batch)

                if torch.isfinite(loss) and loss.item() > 0:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        raw_model.parameters(), max_norm=10.0
                    )
                    inner_opt.step()
                    task_loss_sum += loss.item()
                    n_steps += 1

            if n_steps > 0:
                epoch_total_loss += task_loss_sum / n_steps
                n_valid_tasks += 1

            # ---- ACR：对几何/混合样本计算一致性损失 ----
            acr_loss_total = torch.tensor(0.0, device=DEVICE)
            n_acr = 0
            for pair in acr_pairs:
                orig_t = pair["orig"].unsqueeze(0).to(DEVICE)
                aug_t = pair["aug"].unsqueeze(0).to(DEVICE)
                try:
                    loss_val = compute_acr_loss(
                        raw_model, orig_t, aug_t, pair["geom_params"], DEVICE
                    )
                    if torch.isfinite(loss_val) and loss_val.item() > 0:
                        acr_loss_total = acr_loss_total + loss_val
                        n_acr += 1
                except Exception as e:
                    print(f"  [ACR 警告] 跳过: {e}")

            if n_acr > 0:
                acr_loss_total = acr_loss_total / n_acr
                raw_model.zero_grad()
                if torch.isfinite(acr_loss_total) and acr_loss_total.item() > 0:
                    acr_loss_total.backward()

            # ---- 外循环：Reptile 更新 + ACR 梯度 ----
            with torch.no_grad():
                scale = META_LR / META_BATCH
                for name, param in raw_model.named_parameters():
                    if name in init_params:
                        # Reptile: θ = θ + ε(θ'_i - θ)
                        reptile_dir = param.data - init_params[name]
                        param.data.add_(scale * reptile_dir)
                        # ACR 梯度下降
                        if param.grad is not None:
                            param.data.add_(-ACR_LAMBDA / META_BATCH * param.grad)

        avg_loss = epoch_total_loss / max(n_valid_tasks, 1)
        n_acr_total = sum(
            1 for v in train_meta.values() if v["has_geometric"]
        )
        print(
            f"  Meta epoch [{epoch+1}/{n_meta_epochs}]  "
            f"loss={avg_loss:.4f}  "
            f"tasks={n_valid_tasks}/{META_BATCH}"
        )

    print("[MAML] 完成")
    model.ckpt = True
    return model


# =============================================================================
# 第六部分：标准微调
# =============================================================================

def standard_training(model, ft_epochs=None):
    """Phase 2: 在增强数据集上做标准 YOLOv10 微调"""
    n_ft_epochs = ft_epochs if ft_epochs is not None else FT_EPOCHS
    # 写临时 yaml 供 model.train() 使用
    yolo_cfg = {
        "path": str(OUTPUT_DATASET_DIR),
        "train": "train/images",
        "val": "val/images",
        "nc": 1,
        "names": {0: "reef"}
    }
    temp_yaml = str(PROJECT_DIR / "_data_temp_v2.yaml")
    with open(temp_yaml, "w") as f:
        yaml.dump(yolo_cfg, f, default_flow_style=False)

    results = model.train(
        data=temp_yaml,
        imgsz=IMG_SIZE,
        epochs=n_ft_epochs,
        batch=BATCH_SIZE,
        workers=0,
        optimizer="AdamW",
        lr0=0.001,
        weight_decay=0.0005,
        warmup_epochs=5,
        close_mosaic=15,
        label_smoothing=0.0,
        amp=False,
        project="runs",
        name="fsam_v2",
        device=DEVICE,
        cos_lr=True,
        patience=0,
        overlap_mask=False,
        exist_ok=True,
    )

    if os.path.exists(temp_yaml):
        os.remove(temp_yaml)
    return results


# =============================================================================
# 第七部分：评估
# =============================================================================

def evaluate(model):
    """在增强验证集上评估"""
    # 写临时 yaml
    yolo_cfg = {
        "path": str(OUTPUT_DATASET_DIR),
        "train": "train/images",
        "val": "val/images",
        "nc": 1,
        "names": {0: "reef"}
    }
    temp_yaml = str(PROJECT_DIR / "_eval_temp_v2.yaml")
    with open(temp_yaml, "w") as f:
        yaml.dump(yolo_cfg, f, default_flow_style=False)

    # 找最优权重
    best_pt = None
    runs_dir = Path("runs")
    if runs_dir.exists():
        # 优先找 fsam_v2
        weight_files = sorted(
            runs_dir.glob("fsam_v2/weights/best.pt"), reverse=True
        )
        if weight_files:
            best_pt = weight_files[0]
    if best_pt is None or not best_pt.exists():
        weight_files = sorted(
            runs_dir.glob("*/weights/best.pt"), reverse=True
        )
        if weight_files:
            best_pt = weight_files[0]
    if best_pt is None or not best_pt.exists():
        print("[评估] 未找到 checkpoint")
        if os.path.exists(temp_yaml):
            os.remove(temp_yaml)
        return None

    print(f"[评估] 使用权重: {best_pt}")

    val_results = model.val(
        data=temp_yaml,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=0,
        split="val",
        project="runs",
        name="fsam_v2_eval",
    )

    if os.path.exists(temp_yaml):
        os.remove(temp_yaml)

    if val_results and hasattr(val_results, "results_dict"):
        print(f"[评估] 结果: {val_results.results_dict}")
        return val_results.results_dict
    return None


# =============================================================================
# 第八部分：主流程
# =============================================================================

if __name__ == "__main__":
    # 解析命令行参数
    quick_test = "--quick-test" in sys.argv
    if "--setup-only" in sys.argv:
        print("=" * 60)
        print("  FSAM-YOLO v2 — 数据初始化模式")
        print("=" * 60)
        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)
        setup_dataset()
        print("\n数据集初始化完成！")
        sys.exit(0)

    # quick-test 模式使用更少的 epoch
    meta_epochs = 2 if quick_test else META_EPOCHS
    ft_epochs = 1 if quick_test else FT_EPOCHS
    if quick_test:
        print("\n[快速测试模式] META_EPOCHS=2, FT_EPOCHS=1")

    print("=" * 60)
    print("  FSAM-YOLO v2 — 在线增强 + ACR 统一训练")
    print(f"  设备: {DEVICE}")
    print(f"  原始数据: {PRIMITIVE_IMG_DIR}")
    print(f"  增强数据: {OUTPUT_DATASET_DIR}")
    print(f"  增强系数: {AUGS_PER_IMAGE} 张/原图")
    print(f"  Meta epochs: {meta_epochs}")
    print(f"  微调 epochs: {ft_epochs}")
    print(f"  ACR λ: {ACR_LAMBDA}")
    print("=" * 60)

    # Step 1: 初始化数据集（选图 + 增强 + 存盘）
    print("\n[Step 1/4] 初始化数据集...")
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    setup_dataset()

    # 加载元数据
    print("\n[Step 1/4] 加载元数据...")
    metadata = load_metadata()

    # Step 2: 加载模型
    print("\n[Step 2/4] 加载模型...")
    model = load_model()

    # Step 3: MAML 元学习
    print("\n[Step 3/4] MAML 元学习...")
    model = meta_learning(model, metadata, meta_epochs=meta_epochs)

    # Step 4: 标准微调
    print("\n[Step 4/4] 标准微调...")
    results = standard_training(model, ft_epochs=ft_epochs)

    # 评估
    print("\n[评估] 验证评估...")
    metrics = evaluate(model)

    print("\n" + "=" * 60)
    print("  FSAM-YOLO v2 训练完成！")
    if metrics:
        print(f"  验证指标: {metrics}")
    print("=" * 60)
