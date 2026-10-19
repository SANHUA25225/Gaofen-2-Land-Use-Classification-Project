import rasterio
from rasterio.enums import Resampling
import numpy as np

# 以标签为基准，把 RF 分类图对齐过来
LABEL_PATH = r"D:\gf2\train_label13.tif"
PRED_PATH  = r"E:\RF1.1\RF_JD.tif"
OUT_PATH   = r"D:\gf2\RF_aligned.tif"

with rasterio.open(LABEL_PATH) as ref:
    ref_profile = ref.profile.copy()
    ref_transform = ref.transform
    ref_width  = ref.width
    ref_height = ref.height
    ref_crs    = ref.crs

with rasterio.open(PRED_PATH) as src:
    # 用最近邻重采样，因为这是分类图，不能插值
    data = src.read(
        1,
        out_shape=(ref_height, ref_width),
        resampling=Resampling.nearest
    )
    # 更新元数据
    ref_profile.update(
        dtype=data.dtype,
        count=1,
        crs=ref_crs,
        transform=ref_transform,
        width=ref_width,
        height=ref_height,
        compress="lzw",
        nodata=0
    )

with rasterio.open(OUT_PATH, "w", **ref_profile) as dst:
    dst.write(data, 1)

print(f"对齐完成，输出: {OUT_PATH}")