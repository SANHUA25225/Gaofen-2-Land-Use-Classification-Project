# -*- coding: utf-8 -*-
"""检查 GF-2 影像的基本信息。

用法（在 prithvi 环境里，cd 到本目录后运行）:
    python 01_check_bands.py 你的影像.tif

作用: 确认波段数、分辨率、坐标系统是否满足微调要求。
"""
import sys

import numpy as np
import rasterio


def main():
    if len(sys.argv) < 2:
        print("用法: python 01_check_bands.py 影像路径.tif")
        sys.exit(1)

    path = sys.argv[1]
    with rasterio.open(path) as src:
        w, h = src.width, src.height
        print(f"文件:     {path}")
        print(f"尺寸:     {w} x {h} 像素")
        print(f"波段数:   {src.count}")
        print(f"数据类型: {src.dtypes[0]}")
        print(f"坐标系:   {src.crs}")
        rx, ry = src.res
        print(f"分辨率:   {rx:.6f} x {abs(ry):.6f}")

        # 取中央一块 2000x2000 做统计，避免整幅读取太慢
        x0 = max((w - 2000) // 2, 0)
        y0 = max((h - 2000) // 2, 0)
        window = rasterio.windows.Window(x0, y0, min(2000, w), min(2000, h))
        arr = src.read(window=window).astype("float32")

        names = ["蓝", "绿", "红", "近红外"]
        print()
        print("各波段统计（中央采样区）:")
        for i in range(src.count):
            band = arr[i]
            valid = band[np.isfinite(band)]
            name = names[i] if i < 4 else f"波段{i + 1}"
            print(f"  {name}: 最小 {valid.min():.1f}  最大 {valid.max():.1f}  平均 {valid.mean():.1f}")

        print()
        if src.count == 4:
            print("结论: 4 波段数据，可以继续后续流程。")
            print("判断波段顺序小技巧: 植被区域近红外反射远高于可见光，")
            print("如果第 4 波段平均值明显高于前 3 个，顺序就是 蓝/绿/红/近红外。")
        elif src.count == 3:
            print("结论: 只有 3 个波段，疑似 RGB 可视化产品，缺少近红外。")
            print("需要向供应商索要原始多光谱(MSS)+全色(PAN)数据，重新做融合。")
        elif src.count == 1:
            print("结论: 单波段（全色/黑白），无法用于本流程。")
        else:
            print(f"结论: {src.count} 个波段，属于预期外的情况，把本脚本输出发我确认。")


if __name__ == "__main__":
    main()
