import rasterio
import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix

# ========== 配置 ==========
LABEL_PATH = r"D:\gf2\train_label13.tif"
PRED_PATH  = r"D:\gf2\RF_aligned.tif"   # 改成你实际对齐后的 RF 文件路径

# ========== 读取 ==========
with rasterio.open(LABEL_PATH) as src:
    label = src.read(1)
with rasterio.open(PRED_PATH) as src:
    pred = src.read(1)

if pred.shape != label.shape:
    raise ValueError(f"尺寸不一致: 标签 {label.shape}, 预测 {pred.shape}")

# ========== 关键修改：先随机采样像素索引，再取值 ==========
total_pixels = label.size
sample_size = 2_000_000          # 200万像素，足够代表整体精度
oversample = 5_000_000           # 多采一些，过滤掉无效像素后还剩足够多

rng = np.random.default_rng(42)
idx = rng.choice(total_pixels, size=min(oversample, total_pixels), replace=False)

label_flat = label.ravel()
pred_flat  = pred.ravel()

y_true_raw = label_flat[idx]
y_pred_raw = pred_flat[idx]

# 过滤掉无效像素（0 和 255）
valid = (y_true_raw != 0) & (y_true_raw != 255)
y_true = y_true_raw[valid][:sample_size].astype(np.int32)
y_pred = y_pred_raw[valid][:sample_size].astype(np.int32)

print(f"有效采样像素数: {len(y_true)}")

# ========== 计算指标 ==========
print("=== RF 精度评价 ===")
print(f"总体精度 OA: {accuracy_score(y_true, y_pred):.4f}")
print(f"Kappa 系数: {cohen_kappa_score(y_true, y_pred):.4f}")

target_names = ["耕地", "林地", "草地", "水域", "建设用地", "未利用地"]
print("\n各类别精度报告:")
print(classification_report(y_true, y_pred, target_names=target_names, digits=4))

print("\n混淆矩阵:")
print(confusion_matrix(y_true, y_pred))