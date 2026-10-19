# -*- coding: utf-8 -*-
"""
基于正确标签的 SVM（支持向量机）遥感影像分类脚本。
采用分块采样 + 特征标准化 + 分块预测，避免内存溢出。

输入：
  - 影像：D:\gf2\train_area8.tif
  - 标签：D:\gf2\train_label13.tif
输出：
  - 分类结果：D:\gf2\SVM_result_corrected.tif
  - 控制台打印精度报告
"""
import os
import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix

# ========== 1. 配置 ==========
IMG_PATH   = r"D:\gf2\train_area8.tif"
LABEL_PATH = r"D:\gf2\train_label13.tif"
OUT_PATH   = r"D:\gf2\SVM_result_corrected.tif"

# SVM 采样参数（SVM 复杂度高，样本量比 RF 小很多）
TRAIN_SAMPLES = 50_000       # 训练样本数（建议 3万~5万，太多会很慢）
VAL_SAMPLES   = 20_000       # 验证样本数
BLOCK_SIZE    = 2048         # 分块大小
SAMPLES_PER_BLOCK = 150      # 每块采样的有效像素数
RANDOM_SEED   = 42

# 类别名称（根据你的标签修改）
TARGET_NAMES = ["耕地", "林地", "草地", "水域", "建设用地", "未利用地"]

# ========== 2. 打开影像元数据 ==========
with rasterio.open(IMG_PATH) as src_img:
    profile = src_img.profile.copy()
    H, W = src_img.height, src_img.width
    C = src_img.count

with rasterio.open(LABEL_PATH) as src_lbl:
    if src_lbl.height != H or src_lbl.width != W:
        raise ValueError(f"影像尺寸 {(H,W)} 与标签尺寸 {(src_lbl.height, src_lbl.width)} 不一致！")

# ========== 3. 分块随机采样 ==========
rng = np.random.default_rng(RANDOM_SEED)

features_list = []
labels_list = []

n_blocks_y = (H + BLOCK_SIZE - 1) // BLOCK_SIZE
n_blocks_x = (W + BLOCK_SIZE - 1) // BLOCK_SIZE
total_blocks = n_blocks_y * n_blocks_x
print(f"影像大小: {H} x {W}, 分块数: {total_blocks}")

with rasterio.open(IMG_PATH) as src_img, rasterio.open(LABEL_PATH) as src_lbl:
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            y0 = by * BLOCK_SIZE
            x0 = bx * BLOCK_SIZE
            h = min(BLOCK_SIZE, H - y0)
            w = min(BLOCK_SIZE, W - x0)
            win = Window(x0, y0, w, h)

            img_block = src_img.read(window=win)         # (C, h, w)
            lbl_block = src_lbl.read(1, window=win)      # (h, w)

            # 标签转 int32
            if np.issubdtype(lbl_block.dtype, np.floating):
                lbl_block = np.nan_to_num(lbl_block, nan=0.0).astype(np.int32)
            else:
                lbl_block = lbl_block.astype(np.int32)

            # 有效像素掩膜
            valid_mask = (lbl_block > 0) & (lbl_block != 255)
            n_valid = valid_mask.sum()
            if n_valid == 0:
                continue

            n_pick = min(SAMPLES_PER_BLOCK, n_valid)
            valid_indices = np.flatnonzero(valid_mask.ravel())
            chosen = rng.choice(valid_indices, size=n_pick, replace=False)

            lbl_flat = lbl_block.ravel()
            img_flat = img_block.reshape(C, -1)
            features_list.append(img_flat[:, chosen].T)
            labels_list.append(lbl_flat[chosen])

            if (by * n_blocks_x + bx + 1) % 50 == 0:
                print(f"  已处理 {by * n_blocks_x + bx + 1}/{total_blocks} 块")

X_all = np.vstack(features_list)
y_all = np.concatenate(labels_list)
print(f"总采样像素数: {X_all.shape[0]}")

# 下采样到目标总数
target_total = TRAIN_SAMPLES + VAL_SAMPLES
if X_all.shape[0] > target_total:
    idx = rng.choice(X_all.shape[0], size=target_total, replace=False)
    X_all = X_all[idx]
    y_all = y_all[idx]
    print(f"下采样至目标总数: {target_total}")

# 打乱并划分训练/验证
perm = rng.permutation(X_all.shape[0])
X_all = X_all[perm]
y_all = y_all[perm]

n_train = min(TRAIN_SAMPLES, int(0.7 * X_all.shape[0]))
X_train, y_train = X_all[:n_train], y_all[:n_train]
X_val,   y_val   = X_all[n_train:], y_all[n_train:]

print(f"训练样本: {X_train.shape}, 验证样本: {X_val.shape}")

# ========== 4. 特征标准化（SVM 必须！） ==========
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled   = scaler.transform(X_val)

# ========== 5. 训练 SVM ==========
print("开始训练 SVM...")
svm = SVC(
    kernel='rbf',
    C=100.0,
    gamma='scale',
    class_weight='balanced',   # 处理类别不平衡
    cache_size=2000,           # 缓存2000MB，加速训练
    verbose=True
)
svm.fit(X_train_scaled, y_train)

# ========== 6. 验证集精度评价 ==========
y_pred_val = svm.predict(X_val_scaled)

print("\n=== SVM 精度评价（验证集） ===")
print(f"总体精度 OA: {accuracy_score(y_val, y_pred_val):.4f}")
print(f"Kappa 系数: {cohen_kappa_score(y_val, y_pred_val):.4f}")

labels_in_data = sorted(set(np.unique(y_val)) | set(np.unique(y_pred_val)))
labels_in_data = [l for l in labels_in_data if l != 0]
target_names_used = [TARGET_NAMES[l-1] for l in labels_in_data if l-1 < len(TARGET_NAMES)]

print("\n各类别精度报告:")
print(classification_report(y_val, y_pred_val,
                            labels=labels_in_data,
                            target_names=target_names_used,
                            digits=4))
print("\n混淆矩阵:")
print(confusion_matrix(y_val, y_pred_val, labels=labels_in_data))

# ========== 7. 整图预测（分块写入） ==========
print("\n开始整图预测...")
profile.update(count=1, dtype='uint8', compress='lzw', nodata=0)

with rasterio.open(OUT_PATH, 'w', **profile) as dst:
    with rasterio.open(IMG_PATH) as src_img:
        for by in range(n_blocks_y):
            for bx in range(n_blocks_x):
                y0 = by * BLOCK_SIZE
                x0 = bx * BLOCK_SIZE
                h = min(BLOCK_SIZE, H - y0)
                w = min(BLOCK_SIZE, W - x0)
                win = Window(x0, y0, w, h)

                img_block = src_img.read(window=win)
                block_flat = img_block.reshape(C, -1).T        # (h*w, C)

                # 预测前也必须标准化（用同一 scaler）
                block_scaled = scaler.transform(block_flat)
                pred_block = svm.predict(block_scaled).reshape(h, w).astype(np.uint8)
                dst.write(pred_block, 1, window=win)

                if (by * n_blocks_x + bx + 1) % 50 == 0:
                    print(f"  预测进度 {by * n_blocks_x + bx + 1}/{total_blocks} 块")

print(f"\n整图预测完成，结果保存至: {OUT_PATH}")