# -*- coding: utf-8 -*-
"""标签切片体检脚本：检查切片标签的数值、类型、形状，并诊断 bincount 报错根因。

用法（在 prithvi 环境里，cd 到 gf2-landcover 目录后运行）:
    python check_labels.py                # 检查默认目录 data/tiles/train 和 data/tiles/val
    python check_labels.py D:\gf2\label.tif   # 也可以直接检查某个标签文件

输出内容:
    1. 磁盘视角: 数据类型 / nodata 声明 / 波段数 / 值范围 / 各类像素占比
    2. 训练框架视角: 按 terratorch 同样的 masked 方式读取，统计会出现多少 NaN
    3. 结论: 告诉你数据是否健康、是否需要处理
"""
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import rasterio
import rioxarray  # noqa: F401  (terratorch 内部通过它读取)


def check_file(path, label=""):
    print(f"\n=== {path} {label} ===")
    with rasterio.open(path) as src:
        print(f"  波段数:     {src.count} （标签应为 1）")
        print(f"  尺寸:       {src.width} x {src.height}")
        print(f"  数据类型:   {src.dtypes[0]}")
        print(f"  nodata 声明: {src.nodata} （None 表示无声明，属健康状态）")
        arr = src.read()
    values, counts = np.unique(arr, return_counts=True)
    print(f"  实际值分布: {dict(zip(values.tolist(), (counts / arr.size * 100).round(1).tolist()))} %")

    # 模拟 terratorch 的读取方式（masked=True）
    d = rioxarray.open_rasterio(str(path), masked=True)
    n_nan = int(d.isnull().sum())
    if n_nan > 0:
        print(f"  ★ 训练框架视角: dtype 变为 {d.dtype}, 出现 {n_nan} 个 NaN "
              f"（占 {n_nan / d.size * 100:.1f}%）——这就是 bincount 报错的直接原因")
    else:
        print(f"  训练框架视角: dtype={d.dtype}, 无 NaN，干净")
    return src.nodata, n_nan, src.count


def check_dir(root):
    labels = sorted(Path(root).glob("label_*.tif"))
    if not labels:
        print(f"在 {root} 下没有找到 label_*.tif")
        return
    print(f"\n########## 扫描 {root}: 共 {len(labels)} 个标签切片 ##########")
    nodata_decls = Counter()
    nan_files = 0
    bad_shape = 0
    value_range = set()
    for p in labels:
        with rasterio.open(p) as src:
            nodata_decls[str(src.nodata)] += 1
            if src.count != 1:
                bad_shape += 1
            arr = src.read()
            value_range.update(np.unique(arr).tolist())
        d = rioxarray.open_rasterio(str(p), masked=True)
        if int(d.isnull().sum()) > 0:
            nan_files += 1
    print(f"  nodata 声明统计: {dict(nodata_decls)}")
    print(f"  波段数异常的文件: {bad_shape} 个")
    print(f"  训练框架视角含 NaN 的文件: {nan_files} / {len(labels)} 个")
    print(f"  全部切片的标签值集合: {sorted(value_range)}")
    bad_values = [v for v in value_range if v < 0 or v > 6]
    if bad_values:
        print(f"  ✗ 发现超出 0~6 范围的值: {bad_values} —— 需要回到标注环节检查")
    else:
        print(f"  ✓ 标签值全部在 0~6 范围内")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    if args:
        for a in args:
            if Path(a).is_dir():
                check_dir(a)
            else:
                check_file(a)
    else:
        for sub in ("data/tiles/train", "data/tiles/val"):
            if Path(sub).is_dir():
                check_dir(sub)
            else:
                print(f"（跳过 {sub}: 目录不存在）")
    print("\n########## 结论 ##########")
    print("如果上面显示 nodata 声明不为 None 且含 NaN —— 正是训练崩溃的原因；")
    print("修复版 03_train.py 已内置 no_label_replace=0 修复，直接用它重跑训练即可，")
    print("无需重新标注或重新切片。")


if __name__ == "__main__":
    main()
