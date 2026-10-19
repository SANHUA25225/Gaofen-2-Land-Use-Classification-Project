# -*- coding: utf-8 -*-
"""把影像和标签栅格切成 224x224 训练小块，并自动划分训练/验证集。

前置: 已在 QGIS 里完成标注并导出标签栅格（见 README 第 2 步）。

用法（在 prithvi 环境里）:
    python 02_tile.py 影像.tif 标签.tif

输出:
    data/tiles/train/   img_*.tif + label_*.tif 训练块
    data/tiles/val/     img_*.tif + label_*.tif 验证块
    data/stats.json     各波段均值/标准差（填进 03_train.yaml）
"""
import json
import random
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window

TILE = 224        # 与预训练一致的切片尺寸
STRIDE = 112      # 步长为 224 的一半，相邻块重叠 50%，样本量翻倍
MIN_LABELED = 0.10  # 有效标签像素占比低于 10% 的块丢弃
VAL_RATIO = 0.20
SEED = 42


def make_offsets(total):
    """从 0 到 total-TILE 的步进位置，末尾补齐保证覆盖到右/下边缘。"""
    if total <= TILE:
        return [0]
    xs = list(range(0, total - TILE + 1, STRIDE))
    if xs[-1] != total - TILE:
        xs.append(total - TILE)
    return xs


def main():
    if len(sys.argv) < 3:
        print("用法: python 02_tile.py 影像.tif 标签.tif")
        sys.exit(1)

    img_path, label_path = sys.argv[1], sys.argv[2]
    random.seed(SEED)

    out_root = Path("data/tiles")
    for sub in ("train", "val"):
        (out_root / sub).mkdir(parents=True, exist_ok=True)

    with rasterio.open(img_path) as im, rasterio.open(label_path) as lb:
        if (im.width, im.height) != (lb.width, lb.height):
            print(f"错误: 影像({im.width}x{im.height})和标签({lb.width}x{lb.height})尺寸不一致。")
            print("     QGIS 栅格化标签时，输出范围和像元大小必须与影像完全一致（见 README）。")
            sys.exit(1)

        # 按不重叠的网格单元划分训练/验证，同一单元的重叠块归入同一侧，
        # 避免相邻重叠块分别落在训练集和验证集造成"泄漏"（验证精度虚高）。
        cell_split = {}
        sums = np.zeros(im.count)
        sq_sums = np.zeros(im.count)
        n_pixels = 0
        n_train = n_val = 0

        img_profile = im.profile.copy()
        img_profile.update(width=TILE, height=TILE, compress="lzw", nodata=None)
        lb_profile = lb.profile.copy()
        # nodata=None 很关键: 若继承源文件的 nodata 声明, 训练框架按 masked 模式读取时
        # 会把等于 nodata 的像素变成 NaN, 触发 bincount 报错
        lb_profile.update(width=TILE, height=TILE, count=1, dtype="uint8", compress="lzw", nodata=None)

        for y in make_offsets(im.height):
            for x in make_offsets(im.width):
                label = lb.read(1, window=Window(x, y, TILE, TILE))
                if (label > 0).mean() < MIN_LABELED:
                    continue

                cell = (x // TILE, y // TILE)
                if cell not in cell_split:
                    cell_split[cell] = "val" if random.random() < VAL_RATIO else "train"
                split = cell_split[cell]

                img = im.read(window=Window(x, y, TILE, TILE))
                stem = f"{x:06d}_{y:06d}"
                with rasterio.open(out_root / split / f"img_{stem}.tif", "w", **img_profile) as dst:
                    dst.write(img)
                with rasterio.open(out_root / split / f"label_{stem}.tif", "w", **lb_profile) as dst:
                    dst.write(label.astype("uint8"), 1)

                if split == "train":
                    n_train += 1
                    flat = img.reshape(im.count, -1).astype("float64")
                    sums += flat.sum(axis=1)
                    sq_sums += (flat ** 2).sum(axis=1)
                    n_pixels += flat.shape[1]
                else:
                    n_val += 1

    if n_train == 0:
        print("错误: 没有产出任何训练块。最可能的原因是标签大面积为 0（未标注），")
        print("     或影像与标签没有对齐。检查后重跑。")
        sys.exit(1)

    means = sums / n_pixels
    stds = np.sqrt(np.maximum(sq_sums / n_pixels - means ** 2, 1e-8))
    stats = {"means": means.round(2).tolist(), "stds": stds.round(2).tolist()}
    Path("data").mkdir(exist_ok=True)
    Path("data/stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")

    print(f"完成: 训练块 {n_train} 个，验证块 {n_val} 个 → data/tiles/")
    print()
    print("把下面两行填进 03_train.yaml 的对应位置:")
    print(f"    means: {stats['means']}")
    print(f"    stds:  {stats['stds']}")


if __name__ == "__main__":
    main()
