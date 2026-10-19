# -*- coding: utf-8 -*-
"""
基于正确标签重新训练随机森林并进行整图分类。
采用分块随机采样，避免一次性加载全图导致内存溢出。

输入：
  - 影像：D:\gf2\train_area8.tif
  - 标签：D:\gf2\train_label13.tif
输出：
  - 分类结果：D:\gf2\RF_result_corrected.tif
  - 控制台打印精度报告
"""
import os
import numpy as np
import rasterio
from rasterio.windows import Window
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report, confusion_matrix

# ========== 1. 配置 ==========
IMG_PATH   = r"D:\gf2\train_area8.tif"
LABEL_PATH = r"D:\gf2\train_label13.tif"
OUT_PATH   = r"D:\gf2\RF_result_corrected.tif"

# 采样参数
TRAIN_SAMPLES = 500_000      # 训练样本数
VAL_SAMPLES   = 200_000      # 验证样本数
BLOCK_SIZE    = 2048         # 分块大小
SAMPLES_PER_BLOCK = 1500     # 每块随机采样的有效像素数（约 1500）
RANDOM_SEED   = 42

# 类别名称（根据你的标签修改）
TARGET_NAMES = ["耕地", "林地", "草地", "水域", "建设用地", "未利用地"]
NUM_CLASSES  = 7   # 0未标注 + 1~6地物

# ========== 2. 打开影像与标签 ==========
with rasterio.open(IMG_PATH) as src_img:
    profile = src_img.profile.copy()
    H, W = src_img.height, src_img.width
    C = src_img.count

with rasterio.open(LABEL_PATH) as src_lbl:
    # 确认标签尺寸与影像一致
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

# 重新打开文件用于分块读取
with rasterio.open(IMG_PATH) as src_img, rasterio.open(LABEL_PATH) as src_lbl:
    for by in range(n_blocks_y):
        for bx in range(n_blocks_x):
            y0 = by * BLOCK_SIZE
            x0 = bx * BLOCK_SIZE
            h = min(BLOCK_SIZE, H - y0)
            w = min(BLOCK_SIZE, W - x0)
            win = Window(x0, y0, w, h)

            # 读取影像块 (C, h, w) 和标签块 (h, w)
            img_block = src_img.read(window=win)          # (C, h, w)
            lbl_block = src_lbl.read(1, window=win)       # (h, w)

            # 标签转 int32（处理浮点）
            if np.issubdtype(lbl_block.dtype, np.floating):
                lbl_block = np.nan_to_num(lbl_block, nan=0.0).astype(np.int32)
            else:
                lbl_block = lbl_block.astype(np.int32)

            # 有效像素掩膜：>0 且 !=255
            valid_mask = (lbl_block > 0) & (lbl_block != 255)
            n_valid = valid_mask.sum()
            if n_valid == 0:
                continue

            # 在有效像素中随机采样
            n_pick = min(SAMPLES_PER_BLOCK, n_valid)
            # 获取有效像素的扁平索引
            valid_indices = np.flatnonzero(valid_mask.ravel())
            chosen = rng.choice(valid_indices, size=n_pick, replace=False)

            # 提取对应的标签和影像特征
            lbl_flat = lbl_block.ravel()
            img_flat = img_block.reshape(C, -1)   # (C, h*w)
            # chosen 是相对于块内扁平索引
            features_list.append(img_flat[:, chosen].T)   # (n_pick, C)
            labels_list.append(lbl_flat[chosen])          # (n_pick,)

            if (by * n_blocks_x + bx + 1) % 50 == 0:
                print(f"  已处理 {by * n_blocks_x + bx + 1}/{total_blocks} 块")

# 合并采样结果
X_all = np.vstack(features_list)
y_all = np.concatenate(labels_list)
print(f"总采样像素数: {X_all.shape[0]}")

# 如果采样数超过目标，随机下采样到目标总数
target_total = TRAIN_SAMPLES + VAL_SAMPLES
if X_all.shape[0] > target_total:
    idx = rng.choice(X_all.shape[0], size=target_total, replace=False)
    X_all = X_all[idx]
    y_all = y_all[idx]
    print(f"下采样至目标总数: {target_total}")
elif X_all.shape[0] < target_total:
    print(f"警告：采样数不足目标总数 ({X_all.shape[0]} < {target_total})，将使用全部采样。")

# 打乱并划分训练集与验证集
perm = rng.permutation(X_all.shape[0])
X_all = X_all[perm]
y_all = y_all[perm]

n_train = min(TRAIN_SAMPLES, int(0.7 * X_all.shape[0]))   # 若采样不足，按7:3划分
n_val = X_all.shape[0] - n_train

X_train, y_train = X_all[:n_train], y_all[:n_train]
X_val,   y_val   = X_all[n_train:], y_all[n_train:]

print(f"训练样本: {X_train.shape}, 验证样本: {X_val.shape}")

# ========== 4. 训练随机森林 ==========
rf = RandomForestClassifier(
    n_estimators=100,
    criterion='gini',
    max_depth=None,
    n_jobs=-1,
    random_state=RANDOM_SEED,
    verbose=1
)
rf.fit(X_train, y_train)

# ========== 5. 验证集精度评价 ==========
y_pred_val = rf.predict(X_val)

print("\n=== RF 精度评价（验证集） ===")
print(f"总体精度 OA: {accuracy_score(y_val, y_pred_val):.4f}")
print(f"Kappa 系数: {cohen_kappa_score(y_val, y_pred_val):.4f}")

print("\n各类别精度报告:")
# 只保留实际出现的类别，避免类别数不匹配
labels_in_data = sorted(set(np.unique(y_val)) | set(np.unique(y_pred_val)))
labels_in_data = [l for l in labels_in_data if l != 0]  # 排除0
target_names_used = [TARGET_NAMES[l-1] for l in labels_in_data]

print(classification_report(y_val, y_pred_val,
                            labels=labels_in_data,
                            target_names=target_names_used,
                            digits=4))
print("\n混淆矩阵:")
print(confusion_matrix(y_val, y_pred_val, labels=labels_in_data))

# ========== 6. 整图预测（分块写入） ==========
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

                img_block = src_img.read(window=win)          # (C, h, w)
                block_flat = img_block.reshape(C, -1).T       # (h*w, C)

                pred_block = rf.predict(block_flat).reshape(h, w).astype(np.uint8)
                dst.write(pred_block, 1, window=win)

                if (by * n_blocks_x + bx + 1) % 50 == 0:
                    print(f"  预测进度 {by * n_blocks_x + bx + 1}/{total_blocks} 块")

print(f"\n整图预测完成，结果保存至: {OUT_PATH}")