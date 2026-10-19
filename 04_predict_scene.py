# -*- coding: utf-8 -*-
"""对整幅 GF-2 影像推理，输出土地覆盖分类图。

前置: 03_train.yaml 训练完成，output/ 目录下有检查点(.ckpt)文件。

用法（在 prithvi 环境里）:
    python 04_predict_scene.py 影像.tif 检查点.ckpt 输出分类图.tif

输出两个文件:
    输出分类图.tif        类别编号图（0=未分, 1-6=各类）
    输出分类图_color.tif  按类别配色的可视化图，直接拖进 QGIS 查看
"""
import json
import sys
from pathlib import Path

import numpy as np
import rasterio
import torch
from rasterio.windows import Window
from terratorch.tasks import SemanticSegmentationTask

TILE = 224
CLASS_NAMES = {1: "耕地", 2: "林地", 3: "草地", 4: "水域", 5: "建设用地", 6: "未利用地"}
CLASS_COLORS = {
    1: (250, 200, 100),  # 耕地 橙黄
    2: (40, 140, 60),    # 林地 深绿
    3: (170, 220, 120),  # 草地 浅绿
    4: (60, 120, 220),   # 水域 蓝
    5: (220, 80, 80),    # 建设用地 红
    6: (200, 200, 200),  # 未利用地 灰
}


def make_offsets(total, step):
    if total <= step:
        return [0]
    xs = list(range(0, total - step + 1, step))
    if xs[-1] != total - step:
        xs.append(total - step)
    return xs


def main():
    if len(sys.argv) < 4:
        print("用法: python 04_predict_scene.py 影像.tif 检查点.ckpt 输出.tif")
        sys.exit(1)

    img_path, ckpt_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

    stats_file = Path("data/stats.json")
    if stats_file.exists():
        stats = json.loads(stats_file.read_text(encoding="utf-8"))
        means = torch.tensor(stats["means"], dtype=torch.float32).view(1, -1, 1, 1)
        stds = torch.tensor(stats["stds"], dtype=torch.float32).view(1, -1, 1, 1)
    else:
        print("警告: 找不到 data/stats.json，未做归一化，精度会明显下降。")
        means = torch.zeros(1, 4, 1, 1)
        stds = torch.ones(1, 4, 1, 1)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")
    task = SemanticSegmentationTask.load_from_checkpoint(ckpt_path, map_location="cpu")
    task = task.to(device).eval()
    means, stds = means.to(device), stds.to(device)

    with rasterio.open(img_path) as src:
        if src.count != 4:
            print(f"错误: 影像是 {src.count} 波段，本流程需要 4 波段（蓝/绿/红/近红外）。")
            sys.exit(1)

        w, h = src.width, src.height
        result = np.zeros((h, w), dtype="uint8")
        n_done = 0
        n_total = len(make_offsets(h, TILE)) * len(make_offsets(w, TILE))

        for y in make_offsets(h, TILE):
            for x in make_offsets(w, TILE):
                arr = src.read(window=Window(x, y, TILE, TILE)).astype("float32")
                t = torch.from_numpy(arr).unsqueeze(0).to(device)
                t = (t - means) / stds
                with torch.no_grad():
                    logits = task.model(t)
                    # 万能提取：不管是 output、logits 还是字典，统统把真实数据拿出来
                    if hasattr(logits, 'output'):
                        logits = logits.output
                    elif hasattr(logits, 'logits'):
                        logits = logits.logits
                    elif isinstance(logits, dict):
                        logits = logits.get('output', logits.get('logits', logits))

                    pred = logits.argmax(dim=1)[0].cpu().numpy().astype("uint8")
                    result[y:y+TILE, x:x+TILE] = pred
                
                n_done += 1
                if n_done % 50 == 0 or n_done == n_total:
                    print(f"进度: {n_done}/{n_total} 块")

        profile = src.profile.copy()

    profile.update(count=1, dtype="uint8", compress="lzw", nodata=0)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(result, 1)

    color = np.zeros((h, w, 3), dtype="uint8")
    for k, rgb in CLASS_COLORS.items():
        color[result == k] = rgb
    color_path = str(out_path).replace(".tif", "_color.tif")
    profile.update(count=3, dtype="uint8", nodata=None)
    with rasterio.open(color_path, "w", **profile) as dst:
        dst.write(color.transpose(2, 0, 1))

    print()
    print(f"完成: {out_path}（类别编号图）")
    print(f"      {color_path}（配色图，拖进 QGIS 直接查看）")
    print("图例: " + "，".join(f"{k}={v}" for k, v in CLASS_NAMES.items()))


if __name__ == "__main__":
    main()
