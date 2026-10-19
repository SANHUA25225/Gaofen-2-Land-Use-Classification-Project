# -*- coding: utf-8 -*-
"""GF-2 面向对象分类（自动分块版）

流程：分块读取 → SLIC 对象分割 → 从标签面提取对象级训练样本 → 随机森林训练
      → 按块分割并预测 → 写出全幅分类 GeoTIFF。

默认输入：
  影像：D:\\gf2\\train_area8.tif
  标签：D:\\gf2\\labels9.shx（程序实际读取同名 labels9.shp）
  类别字段：class，值 1-6

这个版本不会把整幅影像载入内存。每次只处理一个带缓冲区的小窗口。
注意：内存需求显著降低，但全幅影像约 20.48 亿像元，仍然需要大量计算时间和磁盘空间。
"""
from __future__ import annotations

from pathlib import Path
import sys
import math
import time
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, bounds as window_bounds
import geopandas as gpd
from shapely.geometry import box
from scipy.ndimage import uniform_filter
from skimage.segmentation import slic
from skimage.measure import regionprops_table
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import joblib


# ===================== 用户配置区 =====================
IMAGE_PATH = r"D:\gf2\train_area8.tif"
LABELS_PATH = r"D:\gf2\labels9.shx"  # 可以填写 .shx；程序会转为同名 .shp
CLASS_FIELD = "class"
CLASS_VALUES = [1, 2, 3, 4, 5, 6]
OUTPUT_DIR = r"D:\gf2-landcover\obia_tiled_output"

# 内存/分块参数。若电脑内存较小，可将 TILE_SIZE 改为 768 或 512。
TILE_SIZE = 1024          # 每块核心区边长；只影响分块大小，不改变全图尺寸
TILE_HALO = 64             # 每块四周缓冲像元，减少分割和预测的块边缘效应

# SLIC 分割参数。每个对象平均约 TARGET_PIXELS_PER_SEGMENT 个有效像元（近似值）。
TARGET_PIXELS_PER_SEGMENT = 400
MIN_SEGMENTS_PER_TILE = 300
MAX_SEGMENTS_PER_TILE = 10000
SLIC_COMPACTNESS = 10.0
SLIC_SIGMA = 1.0
TEXTURE_WINDOW = 7

# 标签面筛选阈值：对象内有标签像元的比例、已标注像元中的主类别纯度
MIN_LABEL_COVERAGE = 0.40
MIN_LABEL_PURITY = 0.80

# 控制训练样本规模，避免大样本面产生海量重复对象
MAX_OBJECTS_PER_POLYGON_PER_CLASS = 100
MAX_TRAIN_OBJECTS_PER_CLASS = 15000

# 随机森林和验证参数
VALIDATION_TEST_SIZE = 0.25
RANDOM_STATE = 42
RF_TREES = 300
RF_MIN_SAMPLES_LEAF = 1
TREAT_ALL_ZERO_AS_NODATA = True

# 大型全幅分割编号栅格会占用数 GB 磁盘空间，默认只写分类图和一个分割预览。
SAVE_FULL_SEGMENT_RASTER = False
SAVE_SEGMENT_PREVIEW = True
# ====================================================


def stop(message: str, code: int = 1) -> None:
    print("\n[错误] " + message)
    raise SystemExit(code)


def resolve_shapefile(path_text: str) -> Path:
    p = Path(path_text)
    if p.suffix.lower() == ".shx":
        p = p.with_suffix(".shp")
    if p.suffix.lower() != ".shp":
        stop(f"标签路径应指向 .shp 或同名 .shx，目前为：{path_text}")
    if not p.exists():
        stop(
            f"找不到矢量主文件：{p}\n"
            "请确认 labels9.shp、labels9.shx、labels9.dbf 在同一个文件夹中；"
            "labels9.prj 也应存在且坐标系正确。"
        )
    return p


def read_labels(shp_path: Path, raster_crs):
    print(f"\n读取标签矢量：{shp_path}")
    try:
        gdf = gpd.read_file(shp_path)
    except Exception as exc:
        stop(f"读取 Shapefile 失败：{exc}")
    if gdf.empty:
        stop("标签矢量没有要素。")
    if gdf.crs is None:
        stop("标签矢量缺少坐标系。请在 ArcGIS/QGIS 中确认真实坐标系后再定义/修复，不要猜测坐标系。")

    field_map = {str(c).strip().lower(): c for c in gdf.columns}
    if CLASS_FIELD.lower() not in field_map:
        stop(f"标签里没有字段 '{CLASS_FIELD}'。实际字段：{list(gdf.columns)}")
    field = field_map[CLASS_FIELD.lower()]
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()
    try:
        gdf.geometry = gdf.geometry.make_valid()
    except Exception:
        try:
            gdf.geometry = gdf.geometry.buffer(0)
        except Exception:
            pass
    cls = pd.to_numeric(gdf[field], errors="coerce")
    if cls.isna().any():
        stop(f"'{field}' 字段中存在空值或非数字类别。")
    gdf["_class_id"] = cls.astype(int)
    unexpected = sorted(set(gdf["_class_id"].unique()) - set(CLASS_VALUES))
    if unexpected:
        stop(f"发现不在类别 {CLASS_VALUES} 中的值：{unexpected}")
    missing = sorted(set(CLASS_VALUES) - set(gdf["_class_id"].unique()))
    if missing:
        print(f"[警告] 标签矢量中完全没有类别 {missing}。这些类别无法训练。")
    if gdf.crs != raster_crs:
        print(f"标签坐标系 {gdf.crs} 与影像 {raster_crs} 不同，正在转换标签坐标系……")
        try:
            gdf = gdf.to_crs(raster_crs)
        except Exception as exc:
            stop(f"标签投影转换失败：{exc}")
    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy().reset_index(drop=True)
    gdf["_polygon_id"] = np.arange(1, len(gdf) + 1, dtype=np.int32)
    print(f"有效标签面数：{len(gdf):,}")
    print("标签类别面数量：" + ", ".join(
        f"{c}:{int((gdf['_class_id'] == c).sum())}" for c in CLASS_VALUES
    ))
    return gdf


def get_intersecting_labels(gdf, sindex, query_geom):
    """通过空间索引找出与某一块相交的标签面，并做精确相交过滤。"""
    try:
        idx = sindex.query(query_geom, predicate="intersects")
    except TypeError:
        try:
            idx = sindex.query(query_geom)
        except Exception:
            idx = np.flatnonzero(gdf.geometry.intersects(query_geom))
    except Exception:
        idx = np.flatnonzero(gdf.geometry.intersects(query_geom))
    idx = np.asarray(idx)
    if idx.size == 0:
        return gdf.iloc[0:0].copy()
    # GeoPandas 现代空间索引返回整数位置；如果返回结果异常则回退到几何筛选。
    try:
        idx = idx.astype(np.int64, copy=False).ravel()
        idx = idx[(idx >= 0) & (idx < len(gdf))]
        if idx.size == 0:
            return gdf.iloc[0:0].copy()
        subset = gdf.iloc[np.unique(idx)].copy()
        return subset[subset.geometry.intersects(query_geom)].copy()
    except Exception:
        return gdf[gdf.geometry.intersects(query_geom)].copy()


def iter_core_windows(width: int, height: int, tile_size: int):
    for row_off in range(0, height, tile_size):
        h = min(tile_size, height - row_off)
        for col_off in range(0, width, tile_size):
            w = min(tile_size, width - col_off)
            yield Window(col_off, row_off, w, h)


def expand_window(core: Window, halo: int, width: int, height: int) -> Window:
    c0 = max(0, int(core.col_off) - halo)
    r0 = max(0, int(core.row_off) - halo)
    c1 = min(width, int(core.col_off + core.width) + halo)
    r1 = min(height, int(core.row_off + core.height) + halo)
    return Window(c0, r0, c1 - c0, r1 - r0)


def get_window_geom(window: Window, transform):
    left, bottom, right, top = window_bounds(window, transform)
    return box(left, bottom, right, top)


def core_slices(core: Window, expanded: Window):
    r0 = int(core.row_off - expanded.row_off)
    c0 = int(core.col_off - expanded.col_off)
    return (slice(r0, r0 + int(core.height)), slice(c0, c0 + int(core.width)))


def is_valid_mask(data: np.ndarray, dataset_mask: np.ndarray) -> np.ndarray:
    valid = dataset_mask > 0
    valid &= np.all(np.isfinite(data), axis=0)
    if TREAT_ALL_ZERO_AS_NODATA:
        valid &= ~np.all(data == 0, axis=0)
    return valid


def estimate_global_stretch(src):
    """从全图分散位置抽样，估算统一的 2–98 百分位范围，供每个块的 SLIC 使用。"""
    print("\n抽样估算统一的分割归一化范围（不会读取整幅影像）……")
    rng = np.random.default_rng(RANDOM_STATE)
    sample_per_band = [[] for _ in range(4)]
    win_size = min(256, src.width, src.height)
    if win_size < 16:
        stop("影像尺寸太小，无法估算分割归一化范围。")
    accepted_windows = 0
    attempts = 0
    while accepted_windows < 32 and attempts < 100:
        attempts += 1
        col = int(rng.integers(0, max(1, src.width - win_size + 1)))
        row = int(rng.integers(0, max(1, src.height - win_size + 1)))
        win = Window(col, row, win_size, win_size)
        block = src.read(indexes=[1, 2, 3, 4], window=win, out_dtype="float32")
        mask = is_valid_mask(block, src.dataset_mask(window=win))
        if not np.any(mask):
            continue
        pix = block[:, mask]
        if pix.shape[1] > 8000:
            select = rng.choice(pix.shape[1], size=8000, replace=False)
            pix = pix[:, select]
        for b in range(4):
            sample_per_band[b].append(pix[b].copy())
        accepted_windows += 1
    ranges = []
    for b in range(4):
        if not sample_per_band[b]:
            stop(f"无法从影像第 {b+1} 波段抽到有效像元；请检查 NoData、影像覆盖范围和全零值设置。")
        vals = np.concatenate(sample_per_band[b]).astype(np.float32, copy=False)
        low, high = np.percentile(vals, [2, 98])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = float(np.nanmin(vals)), float(np.nanmax(vals))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            stop(f"第 {b+1} 波段有效值范围异常，无法用于 SLIC 分割。")
        ranges.append((float(low), float(high)))
    print("分割归一化范围（每波段 2–98 百分位）：")
    for i, (lo, hi) in enumerate(ranges, start=1):
        print(f"  Band {i}: {lo:.6g} — {hi:.6g}")
    return ranges


def normalize_for_segmentation(data: np.ndarray, valid_mask: np.ndarray, ranges):
    h, w = valid_mask.shape
    scaled = np.empty((h, w, 4), dtype=np.float32)
    for i, (low, high) in enumerate(ranges):
        scaled[:, :, i] = np.clip((data[i] - low) / (high - low), 0.0, 1.0)
    scaled[~valid_mask] = 0.0
    return scaled


def reindex_segments(raw_segments: np.ndarray, valid_mask: np.ndarray):
    raw_segments = np.asarray(raw_segments)
    raw_segments[~valid_mask] = 0
    valid_ids, inverse = np.unique(raw_segments[valid_mask], return_inverse=True)
    segments = np.zeros(valid_mask.shape, dtype=np.int32)
    if inverse.size:
        segments[valid_mask] = inverse.astype(np.int32) + 1
    return segments, int(valid_ids.size)


def segment_tile(slic_image: np.ndarray, valid_mask: np.ndarray):
    valid_count = int(valid_mask.sum())
    if valid_count == 0:
        return np.zeros(valid_mask.shape, dtype=np.int32), 0
    n_segments = int(math.ceil(valid_count / TARGET_PIXELS_PER_SEGMENT))
    n_segments = max(MIN_SEGMENTS_PER_TILE, min(MAX_SEGMENTS_PER_TILE, n_segments, valid_count))
    try:
        raw = slic(
            slic_image,
            n_segments=n_segments,
            compactness=SLIC_COMPACTNESS,
            sigma=SLIC_SIGMA,
            start_label=1,
            mask=valid_mask,
            channel_axis=-1,
            convert2lab=False,
            enforce_connectivity=True,
        )
    except TypeError as exc:
        stop("当前 scikit-image 的 SLIC 接口不兼容。请在 gf2obia 环境执行：python -m pip install --upgrade scikit-image\n" + str(exc))
    except MemoryError:
        stop("单个分块仍发生内存不足。请将 TILE_SIZE 从 1024 改为 768 或 512 后重试。")
    except Exception as exc:
        stop(f"SLIC 分割分块失败：{exc}")
    segments, n_objects = reindex_segments(raw, valid_mask)
    return segments, n_objects


def grouped_mean_std(values: np.ndarray, segments: np.ndarray, n_objects: int):
    ids = segments.ravel()
    vals = np.asarray(values, dtype=np.float32).ravel()
    m = ids > 0
    ids = ids[m].astype(np.int64, copy=False)
    vals = vals[m]
    counts = np.bincount(ids, minlength=n_objects + 1).astype(np.float64)
    sums = np.bincount(ids, weights=vals, minlength=n_objects + 1)
    sums_sq = np.bincount(ids, weights=vals * vals, minlength=n_objects + 1)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    vars_ = np.divide(sums_sq, counts, out=np.zeros_like(sums_sq), where=counts > 0) - means * means
    stds = np.sqrt(np.maximum(vars_, 0.0))
    return means.astype(np.float32), stds.astype(np.float32), counts.astype(np.int64)


def calculate_object_features(data, slic_image, segments, n_objects):
    names = {}
    pixel_count = None
    band_names = ["blue", "green", "red", "nir"]
    for i, name in enumerate(band_names):
        mean, std, count = grouped_mean_std(data[i], segments, n_objects)
        names[f"{name}_mean"] = mean
        names[f"{name}_std"] = std
        if pixel_count is None:
            pixel_count = count
    green, red, nir = data[1], data[2], data[3]
    eps = 1e-6
    ndvi = (nir - red) / (nir + red + eps)
    gndvi = (nir - green) / (nir + green + eps)
    for label, values in [("ndvi", ndvi), ("gndvi", gndvi)]:
        mean, std, _ = grouped_mean_std(values, segments, n_objects)
        names[f"{label}_mean"] = mean
        names[f"{label}_std"] = std

    win = max(3, int(TEXTURE_WINDOW))
    if win % 2 == 0:
        win += 1
    nir_scaled = slic_image[:, :, 3]
    local_mean = uniform_filter(nir_scaled, size=win, mode="nearest")
    local_sq_mean = uniform_filter(nir_scaled * nir_scaled, size=win, mode="nearest")
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 0.0))
    tex_mean, tex_std, _ = grouped_mean_std(local_std, segments, n_objects)
    names["nir_texture_mean"] = tex_mean
    names["nir_texture_std"] = tex_std

    try:
        props = regionprops_table(segments, properties=("label", "area", "perimeter", "eccentricity", "solidity"))
        ids = np.asarray(props["label"], dtype=np.int32)
        area = np.zeros(n_objects + 1, dtype=np.float32)
        perimeter = np.zeros(n_objects + 1, dtype=np.float32)
        eccentricity = np.zeros(n_objects + 1, dtype=np.float32)
        solidity = np.zeros(n_objects + 1, dtype=np.float32)
        area[ids] = np.asarray(props["area"], dtype=np.float32)
        perimeter[ids] = np.asarray(props["perimeter"], dtype=np.float32)
        eccentricity[ids] = np.asarray(props["eccentricity"], dtype=np.float32)
        solidity[ids] = np.asarray(props["solidity"], dtype=np.float32)
    except Exception as exc:
        warnings.warn(f"对象形状特征计算失败，将使用面积近似：{exc}")
        area = pixel_count.astype(np.float32)
        perimeter = np.zeros(n_objects + 1, dtype=np.float32)
        eccentricity = np.zeros(n_objects + 1, dtype=np.float32)
        solidity = np.ones(n_objects + 1, dtype=np.float32)
    names["log_area_px"] = np.log1p(np.maximum(area, 0)).astype(np.float32)
    names["compactness"] = np.divide(
        4 * np.pi * area, perimeter * perimeter,
        out=np.zeros_like(area), where=perimeter > 0
    )
    names["eccentricity"] = eccentricity
    names["solidity"] = solidity
    feature_names = list(names.keys())
    X_all = np.column_stack([names[n] for n in feature_names]).astype(np.float32)
    X_all = np.nan_to_num(X_all, nan=0.0, posinf=0.0, neginf=0.0)
    return X_all, feature_names, pixel_count


def rasterize_labels(gdf_local, out_shape, transform):
    if gdf_local.empty:
        empty = np.zeros(out_shape, dtype=np.uint8)
        return empty, np.zeros(out_shape, dtype=np.int32)
    class_shapes = [(geom, int(cls)) for geom, cls in zip(gdf_local.geometry, gdf_local["_class_id"])
                    if geom is not None and not geom.is_empty]
    polygon_shapes = [(geom, int(pid)) for geom, pid in zip(gdf_local.geometry, gdf_local["_polygon_id"])
                      if geom is not None and not geom.is_empty]
    try:
        labs = rasterize(class_shapes, out_shape=out_shape, transform=transform, fill=0, dtype="uint8", all_touched=False)
        polys = rasterize(polygon_shapes, out_shape=out_shape, transform=transform, fill=0, dtype="int32", all_touched=False)
    except Exception as exc:
        stop(f"某一分块的标签面栅格化失败：{exc}")
    return labs, polys


def assign_object_labels(segments, n_objects, label_raster, polygon_raster, object_pixel_count):
    seg_flat = segments.ravel().astype(np.int64, copy=False)
    lab_flat = label_raster.ravel().astype(np.int64, copy=False)
    poly_flat = polygon_raster.ravel().astype(np.int64, copy=False)
    labeled = (seg_flat > 0) & (lab_flat > 0)
    object_labels = np.zeros(n_objects + 1, dtype=np.uint8)
    purity = np.zeros(n_objects + 1, dtype=np.float32)
    coverage = np.zeros(n_objects + 1, dtype=np.float32)
    object_group = np.zeros(n_objects + 1, dtype=np.int32)
    if not np.any(labeled):
        return object_labels, purity, coverage, object_group

    combo = seg_flat[labeled] * 7 + lab_flat[labeled]
    counts = np.bincount(combo, minlength=(n_objects + 1) * 7).reshape(n_objects + 1, 7)
    by_class = counts[:, 1:]
    labelled_count = by_class.sum(axis=1)
    best_count = by_class.max(axis=1)
    has = labelled_count > 0
    object_labels[has] = np.argmax(by_class[has], axis=1).astype(np.uint8) + 1
    purity[has] = best_count[has] / labelled_count[has]
    coverage[has] = labelled_count[has] / np.maximum(object_pixel_count[has], 1)

    # 为对象选择重叠标签像元最多的原始样本面，验证时按 polygon ID 分组。
    group_mask = labeled & (poly_flat > 0)
    if np.any(group_mask):
        pairs = np.column_stack((seg_flat[group_mask], poly_flat[group_mask]))
        unique_pairs, pair_counts = np.unique(pairs, axis=0, return_counts=True)
        order = np.lexsort((-pair_counts, unique_pairs[:, 0]))
        sorted_pairs = unique_pairs[order]
        first = np.r_[True, sorted_pairs[1:, 0] != sorted_pairs[:-1, 0]]
        winners = sorted_pairs[first]
        object_group[winners[:, 0].astype(np.int64)] = winners[:, 1].astype(np.int32)
    return object_labels, purity, coverage, object_group


def make_rf():
    return RandomForestClassifier(
        n_estimators=RF_TREES,
        max_features="sqrt",
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def grouped_validation(X, y, groups):
    lines = [
        "GF-2 面向对象分类：按原始标签面分组的内部验证报告",
        "说明：同一 polygon ID 尽量只进入训练集或验证集之一。该结果不是外业独立验证的替代品。",
        f"用于建模的对象数：{len(y)}",
        "各类对象数：" + str({int(c): int(np.sum(y == c)) for c in np.unique(y)}),
        f"独立标签面/分组数：{len(np.unique(groups))}",
    ]
    if len(np.unique(groups)) < 4:
        lines.append("\n分组数量少于 4，无法进行可靠的分组留出验证。建议每类增加多个空间分离的标签面。")
        return "\n".join(lines)
    all_classes = set(CLASS_VALUES)
    best = None
    best_score = -1e9
    for seed in range(100):
        splitter = GroupShuffleSplit(n_splits=1, test_size=VALIDATION_TEST_SIZE, random_state=RANDOM_STATE + seed)
        try:
            tr, va = next(splitter.split(X, y, groups))
        except Exception:
            continue
        if len(tr) == 0 or len(va) == 0:
            continue
        tr_cls, va_cls = set(y[tr].tolist()), set(y[va].tolist())
        score = 100 * len(tr_cls & all_classes) + 50 * len(va_cls & all_classes)
        if tr_cls == all_classes:
            score += 500
        if va_cls == all_classes:
            score += 250
        score -= abs(len(va) / len(y) - VALIDATION_TEST_SIZE) * 10
        if score > best_score:
            best_score, best = score, (tr, va)
        if tr_cls == all_classes and va_cls == all_classes:
            break
    if best is None:
        lines.append("\n没有找到可用的分组训练/验证划分。")
        return "\n".join(lines)
    tr, va = best
    lines.append(f"训练对象：{len(tr)}；验证对象：{len(va)}")
    lines.append(f"训练类别：{sorted(np.unique(y[tr]).tolist())}")
    lines.append(f"验证类别：{sorted(np.unique(y[va]).tolist())}")
    if set(CLASS_VALUES) - set(y[va].tolist()):
        lines.append("警告：验证数据缺少部分类别，缺失类别的验证精度无法据此判断。")
    if set(CLASS_VALUES) - set(y[tr].tolist()):
        lines.append("警告：训练折缺少部分类别，验证指标需要谨慎解释。")
    model = make_rf()
    model.fit(X[tr], y[tr])
    pred = model.predict(X[va])
    lines.append(f"\n验证 OA：{accuracy_score(y[va], pred):.4f}")
    lines.append(f"验证 Macro-F1：{f1_score(y[va], pred, labels=CLASS_VALUES, average='macro', zero_division=0):.4f}")
    lines.append("\n混淆矩阵（行=真实类别，列=预测类别；类别顺序 1-6）：")
    lines.append(np.array2string(confusion_matrix(y[va], pred, labels=CLASS_VALUES)))
    lines.append("\n分类报告：")
    lines.append(classification_report(y[va], pred, labels=CLASS_VALUES,
                                       target_names=[f"class_{c}" for c in CLASS_VALUES], zero_division=0))
    lines.append("\n论文级评价还应使用独立的空间分离验证样本，最好结合实地调查或高可信度人工判读。")
    return "\n".join(lines)


def output_profile(src_profile, dtype, nodata=0):
    p = src_profile.copy()
    p.update(
        driver="GTiff", count=1, dtype=dtype, nodata=nodata,
        compress="deflate", zlevel=3, tiled=True,
        blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER",
    )
    return p


def save_small_raster(path, arr, base_profile, dtype, nodata=0):
    p = base_profile.copy()
    p.update(driver="GTiff", count=1, dtype=dtype, nodata=nodata, compress="deflate", zlevel=3,
             tiled=True, blockxsize=256, blockysize=256, BIGTIFF="IF_SAFER")
    with rasterio.open(path, "w", **p) as dst:
        dst.write(arr.astype(dtype, copy=False), 1)


def read_and_process_tile(src, core, ranges, need_features=True):
    ext = expand_window(core, TILE_HALO, src.width, src.height)
    data = src.read(indexes=[1, 2, 3, 4], window=ext, out_dtype="float32")
    valid = is_valid_mask(data, src.dataset_mask(window=ext))
    if not np.any(valid):
        return ext, None, None, None, 0, None, None
    scaled = normalize_for_segmentation(data, valid, ranges)
    segments, n_objects = segment_tile(scaled, valid)
    if not need_features:
        return ext, data, valid, scaled, n_objects, segments, None
    X_all, feature_names, pixel_count = calculate_object_features(data, scaled, segments, n_objects)
    return ext, data, valid, scaled, n_objects, segments, (X_all, feature_names, pixel_count)


def main():
    image_path = Path(IMAGE_PATH)
    if not image_path.exists():
        stop(f"找不到影像：{image_path}")
    shp_path = resolve_shapefile(LABELS_PATH)
    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print("GF-2 面向对象分类：自动分块 SLIC + 对象特征 + Random Forest")
    print(f"影像：{image_path}")
    print(f"标签：{shp_path}")
    print(f"输出：{out_dir}")
    print(f"分块核心：{TILE_SIZE} × {TILE_SIZE}；缓冲区：{TILE_HALO} 像元")
    print(f"分割尺度：平均 {TARGET_PIXELS_PER_SEGMENT} 个有效像元/对象（近似），每块对象数限制 {MIN_SEGMENTS_PER_TILE}-{MAX_SEGMENTS_PER_TILE}")
    print("=" * 78)

    start_time = time.time()
    with rasterio.open(image_path) as src:
        if src.crs is None:
            stop("影像缺少坐标系。请先检查 GeoTIFF 的真实地理参考信息。")
        if src.count < 4:
            stop(f"影像只有 {src.count} 个波段；需要四波段 B/G/R/NIR 输入。")
        if src.width < 1 or src.height < 1:
            stop("影像尺寸异常。")
        total_tiles = math.ceil(src.width / TILE_SIZE) * math.ceil(src.height / TILE_SIZE)
        print(f"影像大小：{src.width:,} × {src.height:,}；波段：{src.count}；CRS：{src.crs}")
        print(f"总像元数：{src.width * src.height:,}；预计全图分块数：{total_tiles:,}")
        print("提示：分块会降低单次内存占用，但不会减少全幅总像元计算量。")

        gdf = read_labels(shp_path, src.crs)
        if gdf.empty:
            stop("没有有效标签面。")
        try:
            sindex = gdf.sindex
        except Exception as exc:
            stop(f"无法建立标签空间索引：{exc}")
        ranges = estimate_global_stretch(src)
        src_profile = src.profile.copy()

        # ------------------ 第一遍：仅处理与标签面相交的核心块，积累训练对象 ------------------
        print("\n第一阶段：扫描带标签的分块，分割并提取对象级训练样本……")
        rng = np.random.default_rng(RANDOM_STATE)
        # reservoirs[class_id][polygon_id] = list of records; 按标签面限制样本数，减小大面过度主导。
        reservoirs = {c: {} for c in CLASS_VALUES}
        seen_per_group = {}
        feature_names_ref = None
        preview_saved = False
        labelled_tiles = 0
        processed_tiles = 0

        for tile_idx, core in enumerate(iter_core_windows(src.width, src.height, TILE_SIZE), start=1):
            core_geom = get_window_geom(core, src.transform)
            labels_core = get_intersecting_labels(gdf, sindex, core_geom)
            if labels_core.empty:
                continue
            labelled_tiles += 1
            ext = expand_window(core, TILE_HALO, src.width, src.height)
            ext_geom = get_window_geom(ext, src.transform)
            labels_local = get_intersecting_labels(gdf, sindex, ext_geom)
            if labels_local.empty:
                continue

            ext, data, valid, scaled, n_objects, segments, feature_payload = read_and_process_tile(src, core, ranges, need_features=True)
            if data is None or n_objects == 0 or feature_payload is None:
                continue
            processed_tiles += 1
            X_all, feature_names, pixel_count = feature_payload
            if feature_names_ref is None:
                feature_names_ref = feature_names
            elif feature_names != feature_names_ref:
                stop("不同分块的对象特征字段不一致，无法训练模型。")

            core_slc = core_slices(core, ext)
            if SAVE_SEGMENT_PREVIEW and not preview_saved:
                preview_path = out_dir / "obia_segment_preview.tif"
                preview_profile = src_profile.copy()
                preview_profile.update(width=int(ext.width), height=int(ext.height), transform=src.window_transform(ext))
                save_small_raster(preview_path, segments, preview_profile, "int32", nodata=0)
                preview_saved = True
                print(f"已保存分割预览：{preview_path}")

            label_raster, polygon_raster = rasterize_labels(
                labels_local, (int(ext.height), int(ext.width)), src.window_transform(ext)
            )
            obj_label, purity, coverage, groups = assign_object_labels(
                segments, n_objects, label_raster, polygon_raster, pixel_count
            )
            # 只使用完全位于核心块内的对象训练，避免缓冲区对象被不同块重复采样。
            core_counts = np.bincount(segments[core_slc].ravel(), minlength=n_objects + 1)
            fully_core = (core_counts == pixel_count)
            fully_core[0] = False
            candidate_mask = (
                fully_core
                & (obj_label > 0)
                & (coverage >= MIN_LABEL_COVERAGE)
                & (purity >= MIN_LABEL_PURITY)
                & (groups > 0)
            )
            selected_ids = np.flatnonzero(candidate_mask)
            for oid in selected_ids:
                cls = int(obj_label[oid])
                poly_id = int(groups[oid])
                group_key = (cls, poly_id)
                bucket = reservoirs[cls].setdefault(poly_id, [])
                seen = seen_per_group.get(group_key, 0) + 1
                seen_per_group[group_key] = seen
                record = (X_all[oid].copy(), poly_id, float(coverage[oid]), float(purity[oid]))
                quota = MAX_OBJECTS_PER_POLYGON_PER_CLASS
                if len(bucket) < quota:
                    bucket.append(record)
                else:
                    replacement = int(rng.integers(0, seen))
                    if replacement < quota:
                        bucket[replacement] = record
            if processed_tiles % 10 == 0:
                print(f"  已处理带标签块 {processed_tiles} 个；当前收集候选对象："
                      f"{sum(len(vv) for cc in reservoirs.values() for vv in cc.values()):,}")
            del data, valid, scaled, segments, feature_payload, X_all, label_raster, polygon_raster

        if processed_tiles == 0 or feature_names_ref is None:
            stop("没有从标签面中提取到训练对象。请检查标签是否覆盖影像、class 字段、坐标系和分割阈值。")
        print(f"与标签面相交的核心块：{labelled_tiles:,}；实际成功分割块：{processed_tiles:,}")

        # 展平训练样本池，然后按类别设置全局上限。
        X_list, y_list, group_list, cov_list, pur_list = [], [], [], [], []
        for cls in CLASS_VALUES:
            records = []
            for _, bucket in reservoirs[cls].items():
                records.extend(bucket)
            if len(records) > MAX_TRAIN_OBJECTS_PER_CLASS:
                keep = rng.choice(len(records), size=MAX_TRAIN_OBJECTS_PER_CLASS, replace=False)
                records = [records[int(i)] for i in keep]
            print(f"类别 {cls}：保留 {len(records):,} 个训练对象，来自 {len(reservoirs[cls]):,} 个标签面")
            for feat, group_id, cov, pur in records:
                X_list.append(feat)
                y_list.append(cls)
                group_list.append(group_id)
                cov_list.append(cov)
                pur_list.append(pur)
        if not y_list:
            stop("训练对象池为空。请检查标签面是否与分割对象相交，或调低覆盖率/纯度阈值。")
        present = sorted(set(y_list))
        missing = sorted(set(CLASS_VALUES) - set(present))
        if missing:
            stop(
                f"通过筛选后缺少类别 {missing} 的训练对象。不要直接继续全区分类；"
                "请检查对应类别样本是否位于影像范围内、类别编码是否正确，必要时适当降低 MIN_LABEL_COVERAGE 或 MIN_LABEL_PURITY。"
            )
        X_train = np.asarray(X_list, dtype=np.float32)
        y_train = np.asarray(y_list, dtype=np.uint8)
        groups_train = np.asarray(group_list, dtype=np.int32)
        print(f"\n最终训练矩阵：{X_train.shape[0]:,} 个对象 × {X_train.shape[1]} 个特征")

        report = grouped_validation(X_train, y_train, groups_train)
        report_path = out_dir / "obia_validation_report.txt"
        report_path.write_text(report, encoding="utf-8")
        print("\n分组验证结果：\n" + report)

        print("\n使用全部训练对象拟合最终随机森林模型……")
        model = make_rf()
        model.fit(X_train, y_train)
        model_path = out_dir / "obia_rf_model.joblib"
        joblib.dump({
            "model": model,
            "feature_names": feature_names_ref,
            "class_values": CLASS_VALUES,
            "global_stretch_ranges_for_slic": ranges,
            "tile_size": TILE_SIZE,
            "tile_halo": TILE_HALO,
            "target_pixels_per_segment": TARGET_PIXELS_PER_SEGMENT,
            "input_image": str(image_path),
            "input_labels": str(shp_path),
        }, model_path)

        samples_df = pd.DataFrame(X_train, columns=feature_names_ref)
        samples_df["class"] = y_train
        samples_df["polygon_id"] = groups_train
        samples_df["label_coverage"] = cov_list
        samples_df["label_purity"] = pur_list
        train_csv = out_dir / "obia_training_objects.csv"
        samples_df.to_csv(train_csv, index=False, encoding="utf-8-sig")

        importances = pd.DataFrame({"feature": feature_names_ref, "importance": model.feature_importances_})
        importances.sort_values("importance", ascending=False).to_csv(
            out_dir / "obia_feature_importance.csv", index=False, encoding="utf-8-sig"
        )

        # ------------------ 第二遍：逐块预测并写出完整分类结果 ------------------
        print("\n第二阶段：逐块进行 SLIC 分割、对象分类，并写出全幅分类图……")
        class_path = out_dir / "obia_classification_rf.tif"
        segment_path = out_dir / "obia_segments.tif" if SAVE_FULL_SEGMENT_RASTER else None
        class_profile = output_profile(src_profile, "uint8", nodata=0)
        seg_profile = output_profile(src_profile, "int32", nodata=0) if SAVE_FULL_SEGMENT_RASTER else None
        color_map = {
            0: (0, 0, 0, 0), 1: (34, 139, 34, 255), 2: (154, 205, 50, 255),
            3: (255, 215, 0, 255), 4: (205, 133, 63, 255),
            5: (30, 144, 255, 255), 6: (186, 85, 211, 255),
        }
        with rasterio.open(class_path, "w", **class_profile) as class_dst:
            if SAVE_FULL_SEGMENT_RASTER:
                seg_dst_context = rasterio.open(segment_path, "w", **seg_profile)
            else:
                seg_dst_context = None
            segment_id_offset = 0
            try:
                for tile_idx, core in enumerate(iter_core_windows(src.width, src.height, TILE_SIZE), start=1):
                    ext, data, valid, scaled, n_objects, segments, feature_payload = read_and_process_tile(
                        src, core, ranges, need_features=True
                    )
                    if data is None or n_objects == 0 or feature_payload is None:
                        blank = np.zeros((int(core.height), int(core.width)), dtype=np.uint8)
                        class_dst.write(blank, 1, window=core)
                        if seg_dst_context is not None:
                            seg_dst_context.write(np.zeros(blank.shape, dtype=np.int32), 1, window=core)
                        continue
                    X_all, _, _ = feature_payload
                    object_pred = np.zeros(n_objects + 1, dtype=np.uint8)
                    object_pred[1:] = model.predict(X_all[1:]).astype(np.uint8)
                    class_map = object_pred[segments]
                    class_map[~valid] = 0
                    slc = core_slices(core, ext)
                    class_core = class_map[slc].astype(np.uint8, copy=False)
                    class_dst.write(class_core, 1, window=core)
                    if seg_dst_context is not None:
                        seg_core = segments[slc].astype(np.int64, copy=False)
                        seg_out = np.where(seg_core > 0, seg_core + segment_id_offset, 0).astype(np.int32)
                        seg_dst_context.write(seg_out, 1, window=core)
                    segment_id_offset += n_objects
                    if tile_idx == 1:
                        try:
                            class_dst.write_colormap(1, color_map)
                        except Exception:
                            pass
                    if tile_idx % 10 == 0 or tile_idx == total_tiles:
                        elapsed = time.time() - start_time
                        print(f"  分类进度：{tile_idx:,}/{total_tiles:,} 块（{tile_idx / total_tiles:.1%}）；已用时 {elapsed / 60:.1f} 分钟")
                    del data, valid, scaled, segments, feature_payload, X_all, class_map, object_pred
            finally:
                if seg_dst_context is not None:
                    seg_dst_context.close()

    elapsed = time.time() - start_time
    print("\n" + "=" * 78)
    print("分块面向对象分类完成。")
    print(f"分类图：{class_path}")
    print(f"训练对象：{train_csv}")
    print(f"验证报告：{report_path}")
    print(f"特征重要性：{out_dir / 'obia_feature_importance.csv'}")
    print(f"模型：{model_path}")
    if SAVE_SEGMENT_PREVIEW:
        print(f"分割预览（一个带标签块）：{out_dir / 'obia_segment_preview.tif'}")
    if SAVE_FULL_SEGMENT_RASTER:
        print(f"全幅分割编号图：{segment_path}（可能占用数 GB 磁盘空间）")
    print(f"总耗时：{elapsed / 3600:.2f} 小时")
    print("类别值 1-6 对应标签文件 class 字段；0 为 NoData。")
    print("重要：块级分割会在块边界附近产生对象不连续；缓冲区有助于减轻但无法完全消除这种影响。")
    print("=" * 78)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断程序。未完成的输出栅格可能不完整。")
        sys.exit(130)
    except MemoryError:
        print(
            "\n[内存不足] 当前块仍超过可用内存。请打开程序，将 TILE_SIZE 改为 768 或 512，"
            "将 TILE_HALO 暂时设为 32 后再运行。",
            file=sys.stderr,
        )
        sys.exit(2)
