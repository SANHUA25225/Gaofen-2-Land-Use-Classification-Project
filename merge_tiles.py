import os
import shutil
import glob

def merge_tiles(src_dir, dst_dir, prefix):
    os.makedirs(dst_dir, exist_ok=True)
    files = glob.glob(os.path.join(src_dir, "*.tif"))
    for f in files:
        basename = os.path.basename(f)
        # 核心：只在原文件名前面加前缀，保留 img_ 和 mask_ 及后面的坐标数字
        new_name = f"{prefix}_{basename}"
        shutil.copy(f, os.path.join(dst_dir, new_name))

train_dst = r"C:\gf2-landcover\data\tiles\train"
val_dst = r"C:\gf2-landcover\data\tiles\val"

# 合并 area2
merge_tiles(r"D:\gf2\tiles_area2_clean\train", train_dst, "a2")
merge_tiles(r"D:\gf2\tiles_area2_clean\val", val_dst, "a2")

# 合并 area7
merge_tiles(r"D:\gf2\tiles_area7_clean\train", train_dst, "a7")
merge_tiles(r"D:\gf2\tiles_area7_clean\val", val_dst, "a7")

print("切片合并完成！请检查 data/tiles/train 和 val 里的文件数量。")