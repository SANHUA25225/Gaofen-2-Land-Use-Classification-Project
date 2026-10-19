import os, glob, json
import numpy as np
import torch
import tifffile as tiff
from terratorch.tasks import SemanticSegmentationTask
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report

# ========== 1. 配置 ==========
CKPT_PATH  = r"D:\gf2-landcover\output\checkpoints\best.ckpt"
VAL_DIR    = r"D:\gf2-landcover\data\tiles\val"
STATS_FILE = r"D:\gf2-landcover\data\stats.json"

NUM_CLASSES = 8   # 如果你用的是 label12（含道路的8分类）就填8；如果是 label13（道路并入建设用地的7分类）就填7
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 类别名称：根据你的类别体系修改
# 8类（含道路）：["耕地", "林地", "草地", "水域", "建设用地", "未利用地", "道路"]
# 7类（道路归入建设用地）：["耕地", "林地", "草地", "水域", "建设用地", "未利用地"]
TARGET_NAMES = ["耕地", "林地", "草地", "水域", "建设用地", "未利用地", "道路"]

# ========== 2. 加载模型与归一化参数 ==========
stats = json.loads(open(STATS_FILE, 'r', encoding='utf-8').read())
means = np.array(stats["means"], dtype=np.float32)
stds  = np.array(stats["stds"],  dtype=np.float32)

task = SemanticSegmentationTask.load_from_checkpoint(CKPT_PATH, map_location="cpu")
task = task.to(DEVICE).eval()

# ========== 3. 遍历验证集，收集预测与标签 ==========
all_preds, all_labels = [], []
val_imgs = sorted(glob.glob(os.path.join(VAL_DIR, "img_*.tif")))

with torch.no_grad():
    for img_path in val_imgs:
        label_path = img_path.replace("img_", "label_")
        if not os.path.exists(label_path):
            continue

        img   = tiff.imread(img_path).astype(np.float32)
        label = tiff.imread(label_path).astype(np.int64)

        # --- 统一维度为 (C, H, W) ---
        if img.ndim == 3:
            # 如果是 (H, W, C) 且最后一维等于波段数，则转置
            if img.shape[-1] == len(means):
                img = img.transpose(2, 0, 1)   # -> (C, H, W)
            # 如果已经是 (C, H, W) 则保持
        # 如果 img 是 2D（单波段？），则加一个维度
        if img.ndim == 2:
            img = img[np.newaxis, ...]

        # --- 归一化：reshape 为 (C, 1, 1) 广播 ---
        means_ = means.reshape(-1, 1, 1)
        stds_  = stds.reshape(-1, 1, 1)
        img = (img - means_) / stds_

        # --- 转 Tensor 并加 batch 维度 ---
        t = torch.from_numpy(img).unsqueeze(0).to(DEVICE)   # (1, C, H, W)

        # --- 模型推理 ---
        logits = task.model(t)
        # 兼容 TerraTorch 的 ModelOutput 对象
        if hasattr(logits, 'output'):
            logits = logits.output
        elif hasattr(logits, 'logits'):
            logits = logits.logits

        pred = logits.argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

        all_preds.extend(pred.flatten())
        all_labels.extend(label.flatten())

# ========== 4. 计算指标（忽略 0） ==========
all_preds  = np.array(all_preds)
all_labels = np.array(all_labels)
mask = all_labels != 0
y_true, y_pred = all_labels[mask], all_preds[mask]

print("=== Prithvi 大模型精度评价 ===")
print(f"总体精度 OA: {accuracy_score(y_true, y_pred):.4f}")
print(f"Kappa 系数: {cohen_kappa_score(y_true, y_pred):.4f}")
print("\n各类别精度报告:")
print(classification_report(y_true, y_pred,
                            labels=list(range(1, NUM_CLASSES)),
                            target_names=TARGET_NAMES,
                            zero_division=0))