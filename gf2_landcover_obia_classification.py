# -*- coding: utf-8 -*-
"""GF-2 面向对象植被/地物分类（SLIC 分割 + 对象特征 + 随机森林）

输入：
  影像：D:\\gf2\\train_area8.tif
  标签：D:\\gf2\\labels9.shx（注意：实际读取同目录下的 labels9.shp）
  标签字段：class，取值 1、2、3、4、5、6

输出目录：D:\\gf2\\obia_output

依赖安装（建议在你的 gf2patch 环境中执行）：
  pip install rasterio geopandas scikit-image scikit-learn scipy pandas joblib

重要说明：
  1. Shapefile 通常需要 .shp、.shx、.dbf、.prj 等配套文件；仅有 .shx 不够。
  2. 默认假设输入影像前四个波段顺序为：蓝、绿、红、近红外（B/G/R/NIR）。
  3. 这是 SLIC 超像素分割实现的面向对象基线，不保证分割对象就是真实植被群落边界。
  4. 若影像很大、内存不足，请先在 ENVI/ArcGIS 中裁剪一块有代表性的区域做试验。
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
import geopandas as gpd
from scipy.ndimage import uniform_filter
from skimage.segmentation import slic
from skimage.measure import regionprops_table
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
import joblib


# ===================== 用户配置区 =====================
IMAGE_PATH = r"D:\gf2\train_area8.tif"
LABELS_PATH = r"D:\gf2\labels9.shx"  # 可以填 .shx；程序会自动改为同名 .shp 读取
CLASS_FIELD = "class"
CLASS_VALUES = [1, 2, 3, 4, 5, 6]
OUTPUT_DIR = r"D:\gf2-landcover\obia_output"

# 分割参数：先用默认值跑通，再对照调整
TARGET_SEGMENTS = 3000   # 目标对象数量（近似值）；调大通常对象更细小
SLIC_COMPACTNESS = 10.0  # 越大越偏向规则、紧凑的对象
SLIC_SIGMA = 1.0         # 分割前平滑尺度；0 表示不平滑
TEXTURE_WINDOW = 7       # 局部近红外纹理窗口，建议用奇数

# 训练对象筛选：标签面覆盖对象的比例、对象内部标签纯度
MIN_LABEL_COVERAGE = 0.40
MIN_LABEL_PURITY = 0.80

# 验证与模型参数
VALIDATION_TEST_SIZE = 0.25
RANDOM_STATE = 42
RF_TREES = 400
RF_MIN_SAMPLES_LEAF = 1
TREAT_ALL_ZERO_AS_NODATA = True
# =====================================================


def stop(message, code=1):
    print("\n[错误] " + message)
    sys.exit(code)


def resolve_shapefile(path_text):
    """若输入为 .shx，则转成同名 .shp，因为属性字段存在 .dbf 中。"""
    path = Path(path_text)
    if path.suffix.lower() == ".shx":
        shp_path = path.with_suffix(".shp")
    else:
        shp_path = path
    if shp_path.suffix.lower() != ".shp":
        stop(f"标签路径应指向 .shp 或同名 .shx，目前是：{path}")
    if not shp_path.exists():
        stop(
            f"没有找到矢量主文件：{shp_path}\n"
            "请检查该目录中是否同时存在 labels9.shp、labels9.shx、labels9.dbf，"
            "以及通常需要的 labels9.prj。不能只用 .shx 文件完成分类。"
        )
    return shp_path


def read_labels(shp_path, raster_crs):
    print(f"\n[1/7] 读取标签矢量：{shp_path}")
    try:
        gdf = gpd.read_file(shp_path)
    except Exception as exc:
        stop(f"读取 Shapefile 失败：{exc}")

    if gdf.empty:
        stop("标签矢量没有任何要素。")
    if gdf.crs is None:
        stop(
            "标签矢量没有坐标系信息（通常缺少或未正确读取 .prj）。\n"
            "请先在 ArcGIS/QGIS 中确认标签真实坐标系并正确指定；不要凭猜测设置坐标系。"
        )

    field_map = {str(c).strip().lower(): c for c in gdf.columns}
    wanted = CLASS_FIELD.strip().lower()
    if wanted not in field_map:
        stop(
            f"标签中找不到字段 '{CLASS_FIELD}'。实际字段为："
            + ", ".join(map(str, gdf.columns))
        )
    actual_field = field_map[wanted]

    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()
    if gdf.empty:
        stop("标签矢量中没有有效几何。")

    # 对可修复的无效几何尽量做修复；失败时由 rasterize 的检查继续提示。
    try:
        gdf.geometry = gdf.geometry.make_valid()
    except Exception:
        try:
            gdf.geometry = gdf.geometry.buffer(0)
        except Exception:
            pass

    cls_num = pd.to_numeric(gdf[actual_field], errors="coerce")
    bad = cls_num.isna()
    if bad.any():
        stop(f"字段 '{actual_field}' 中存在非数字或空类别值，请先清理。")
    gdf["_class_id"] = cls_num.astype(int)

    present_values = sorted(gdf["_class_id"].unique().tolist())
    unexpected = sorted(set(present_values) - set(CLASS_VALUES))
    if unexpected:
        stop(f"发现不在 1—6 范围内的类别值：{unexpected}。请检查 class 字段。")

    missing = sorted(set(CLASS_VALUES) - set(present_values))
    if missing:
        print(f"[警告] 矢量中没有以下类别的样本：{missing}。这些类别无法被模型可靠学习。")

    if gdf.crs != raster_crs:
        print(f"标签坐标系为 {gdf.crs}，正在转换到影像坐标系 {raster_crs}……")
        try:
            gdf = gdf.to_crs(raster_crs)
        except Exception as exc:
            stop(f"无法将标签矢量投影到影像坐标系：{exc}")

    gdf = gdf[gdf.geometry.notna() & (~gdf.geometry.is_empty)].copy()
    gdf = gdf.reset_index(drop=True)
    gdf["_polygon_id"] = np.arange(1, len(gdf) + 1, dtype=np.int32)

    print(f"有效标签要素数：{len(gdf)}")
    print(f"class 类别值：{sorted(gdf['_class_id'].unique().tolist())}")
    print("说明：标签面将栅格化到影像像元网格；样本面应尽量绘制在类别明确的区域内部。")
    return gdf


def rasterize_labels(gdf, out_shape, transform):
    """把 class 字段和唯一面编号栅格化到输入影像网格。"""
    print("\n[2/7] 将标签面栅格化到影像网格……")
    class_shapes = [
        (geom, int(cls))
        for geom, cls in zip(gdf.geometry, gdf["_class_id"])
        if geom is not None and not geom.is_empty
    ]
    polygon_shapes = [
        (geom, int(pid))
        for geom, pid in zip(gdf.geometry, gdf["_polygon_id"])
        if geom is not None and not geom.is_empty
    ]
    try:
        label_raster = rasterize(
            class_shapes,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype="uint8",
            all_touched=False,
        )
        polygon_raster = rasterize(
            polygon_shapes,
            out_shape=out_shape,
            transform=transform,
            fill=0,
            dtype="int32",
            all_touched=False,
        )
    except Exception as exc:
        stop(f"标签面栅格化失败：{exc}")

    count = int(np.count_nonzero(label_raster))
    if count == 0:
        stop(
            "标签面栅格化后没有任何有标签像元。常见原因是标签与影像位置不重合、"
            "坐标系设置错误，或标签范围不在影像范围内。请先在 GIS 中检查叠加位置。"
        )
    print(f"影像范围内的有效标签像元数：{count:,}")
    return label_raster, polygon_raster


def normalize_for_segmentation(data, valid_mask):
    """逐波段按 2—98 百分位缩放到 0—1，避免波段量纲影响 SLIC。"""
    height, width = valid_mask.shape
    band_count = data.shape[0]
    scaled = np.zeros((height, width, band_count), dtype=np.float32)
    for i in range(band_count):
        values = data[i][valid_mask]
        if values.size == 0:
            stop(f"第 {i + 1} 波段没有有效像元。")
        low, high = np.percentile(values, [2, 98])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low = float(np.nanmin(values))
            high = float(np.nanmax(values))
        if high <= low:
            stop(f"第 {i + 1} 波段可能是常数波段，无法用于分割。")
        scaled[:, :, i] = np.clip((data[i] - low) / (high - low), 0.0, 1.0)
    scaled[~valid_mask] = 0.0
    return scaled


def run_segmentation(slic_image, valid_mask):
    print("\n[3/7] 开始 SLIC 图像分割……")
    print(
        f"参数：TARGET_SEGMENTS={TARGET_SEGMENTS}, "
        f"COMPACTNESS={SLIC_COMPACTNESS}, SIGMA={SLIC_SIGMA}"
    )
    try:
        raw_segments = slic(
            slic_image,
            n_segments=TARGET_SEGMENTS,
            compactness=SLIC_COMPACTNESS,
            sigma=SLIC_SIGMA,
            start_label=1,
            mask=valid_mask,
            channel_axis=-1,
            convert2lab=False,
            enforce_connectivity=True,
        )
    except TypeError as exc:
        stop(
            "当前 scikit-image 版本不支持代码使用的 SLIC 参数。请升级："
            "python -m pip install --upgrade scikit-image\n"
            f"原始错误：{exc}"
        )
    except Exception as exc:
        stop(
            f"SLIC 分割失败：{exc}\n"
            "若提示内存不足，请先把 train_area8.tif 裁剪为更小的测试区域，"
            "或降低影像像元数后再运行。"
        )

    raw_segments = np.asarray(raw_segments)
    raw_segments[~valid_mask] = 0
    # 重新连续编号，使对象 ID 从 1 到 n_objects；0 专门保留给无效区。
    unique_ids, inverse = np.unique(raw_segments[valid_mask], return_inverse=True)
    segments = np.zeros(valid_mask.shape, dtype=np.int32)
    segments[valid_mask] = inverse.astype(np.int32) + 1
    n_objects = int(len(unique_ids))
    print(f"分割完成，对象数：{n_objects:,}")
    return segments, n_objects


def grouped_mean_std(values, segments, n_objects):
    """按对象计算像元值的均值和标准差；对象 0 为背景/无效值。"""
    seg_flat = segments.ravel()
    val_flat = np.asarray(values, dtype=np.float32).ravel()
    valid = seg_flat > 0
    ids = seg_flat[valid]
    vals = val_flat[valid]
    counts = np.bincount(ids, minlength=n_objects + 1).astype(np.float64)
    sums = np.bincount(ids, weights=vals, minlength=n_objects + 1)
    sums_sq = np.bincount(ids, weights=vals * vals, minlength=n_objects + 1)
    means = np.divide(sums, counts, out=np.zeros_like(sums), where=counts > 0)
    variances = np.divide(sums_sq, counts, out=np.zeros_like(sums_sq), where=counts > 0) - means * means
    stds = np.sqrt(np.maximum(variances, 0.0))
    return means.astype(np.float32), stds.astype(np.float32), counts.astype(np.int64)


def calculate_object_features(data, slic_image, segments, n_objects):
    print("\n[4/7] 计算对象级光谱、指数、纹理与形状特征……")
    values_by_name = {}
    object_pixel_count = None

    # 假设前四波段依次为 B、G、R、NIR。
    band_names = ["blue", "green", "red", "nir"]
    for i, name in enumerate(band_names):
        mean, std, counts = grouped_mean_std(data[i], segments, n_objects)
        values_by_name[f"{name}_mean"] = mean
        values_by_name[f"{name}_std"] = std
        if object_pixel_count is None:
            object_pixel_count = counts

    blue, green, red, nir = data[0], data[1], data[2], data[3]
    eps = 1e-6
    ndvi = (nir - red) / (nir + red + eps)
    gndvi = (nir - green) / (nir + green + eps)
    ndvi_mean, ndvi_std, _ = grouped_mean_std(ndvi, segments, n_objects)
    gndvi_mean, gndvi_std, _ = grouped_mean_std(gndvi, segments, n_objects)
    values_by_name["ndvi_mean"] = ndvi_mean
    values_by_name["ndvi_std"] = ndvi_std
    values_by_name["gndvi_mean"] = gndvi_mean
    values_by_name["gndvi_std"] = gndvi_std

    # 在归一化近红外波段上计算局部标准差，再汇总到每个对象。
    nir_scaled = slic_image[:, :, 3].astype(np.float32, copy=False)
    win = max(3, int(TEXTURE_WINDOW))
    if win % 2 == 0:
        win += 1
    local_mean = uniform_filter(nir_scaled, size=win, mode="nearest")
    local_sq_mean = uniform_filter(nir_scaled * nir_scaled, size=win, mode="nearest")
    local_std = np.sqrt(np.maximum(local_sq_mean - local_mean * local_mean, 0.0))
    texture_mean, texture_std, _ = grouped_mean_std(local_std, segments, n_objects)
    values_by_name["nir_texture_mean"] = texture_mean
    values_by_name["nir_texture_std"] = texture_std

    # 形状特征由分割对象本身计算，单位分别为像元/无量纲。
    try:
        props = regionprops_table(
            segments,
            properties=("label", "area", "perimeter", "eccentricity", "solidity"),
        )
        prop_ids = np.asarray(props["label"], dtype=np.int32)
        area = np.zeros(n_objects + 1, dtype=np.float32)
        perimeter = np.zeros(n_objects + 1, dtype=np.float32)
        eccentricity = np.zeros(n_objects + 1, dtype=np.float32)
        solidity = np.zeros(n_objects + 1, dtype=np.float32)
        area[prop_ids] = np.asarray(props["area"], dtype=np.float32)
        perimeter[prop_ids] = np.asarray(props["perimeter"], dtype=np.float32)
        eccentricity[prop_ids] = np.asarray(props["eccentricity"], dtype=np.float32)
        solidity[prop_ids] = np.asarray(props["solidity"], dtype=np.float32)
    except Exception as exc:
        warnings.warn(f"形状特征提取失败，将使用像元数量替代部分形状特征。原因：{exc}")
        area = object_pixel_count.astype(np.float32)
        perimeter = np.zeros(n_objects + 1, dtype=np.float32)
        eccentricity = np.zeros(n_objects + 1, dtype=np.float32)
        solidity = np.ones(n_objects + 1, dtype=np.float32)

    values_by_name["log_area_px"] = np.log1p(np.maximum(area, 0)).astype(np.float32)
    compactness = np.divide(
        4.0 * np.pi * area,
        perimeter * perimeter,
        out=np.zeros_like(area, dtype=np.float32),
        where=perimeter > 0,
    )
    values_by_name["compactness"] = compactness
    values_by_name["eccentricity"] = eccentricity
    values_by_name["solidity"] = solidity

    feature_names = list(values_by_name.keys())
    X_all = np.column_stack([values_by_name[name] for name in feature_names]).astype(np.float32)
    if not np.isfinite(X_all[1:]).all():
        bad_rows = np.where(~np.isfinite(X_all).all(axis=1))[0]
        X_all[bad_rows, :] = np.nan_to_num(X_all[bad_rows, :], nan=0.0, posinf=0.0, neginf=0.0)
        print(f"[提示] 已将 {len(bad_rows)} 个对象特征中的非有限值替换为 0。")

    print(f"特征数量：{len(feature_names)}")
    print("特征包括：四波段均值/标准差、NDVI、GNDVI、近红外局部纹理、对象面积与形状指标。")
    return X_all, feature_names, object_pixel_count


def assign_object_labels(segments, n_objects, label_raster, polygon_raster, object_pixel_count):
    """依据各对象内部标签像元的多数类别，筛出纯度和覆盖率足够的训练对象。"""
    print("\n[5/7] 把标签类别赋给分割对象并筛选训练对象……")
    seg_flat = segments.ravel().astype(np.int64, copy=False)
    lab_flat = label_raster.ravel().astype(np.int64, copy=False)
    poly_flat = polygon_raster.ravel().astype(np.int64, copy=False)
    labeled = (seg_flat > 0) & (lab_flat > 0)
    if not labeled.any():
        stop("分割对象与标签像元没有交集。请检查影像和标签的叠加位置。")

    combo = seg_flat[labeled] * 7 + lab_flat[labeled]
    class_counts = np.bincount(combo, minlength=(n_objects + 1) * 7).reshape(n_objects + 1, 7)
    counts_by_class = class_counts[:, 1:]
    labeled_pixel_count = counts_by_class.sum(axis=1)
    best_count = counts_by_class.max(axis=1)
    object_labels = np.zeros(n_objects + 1, dtype=np.uint8)
    has_labels = labeled_pixel_count > 0
    object_labels[has_labels] = np.argmax(counts_by_class[has_labels], axis=1).astype(np.uint8) + 1
    purity = np.zeros(n_objects + 1, dtype=np.float32)
    purity[has_labels] = best_count[has_labels] / labeled_pixel_count[has_labels]
    coverage = np.zeros(n_objects + 1, dtype=np.float32)
    coverage[has_labels] = labeled_pixel_count[has_labels] / np.maximum(object_pixel_count[has_labels], 1)

    candidates = (
        has_labels
        & (coverage >= MIN_LABEL_COVERAGE)
        & (purity >= MIN_LABEL_PURITY)
    )
    candidates[0] = False
    selected_ids = np.flatnonzero(candidates)
    if selected_ids.size == 0:
        stop(
            "没有对象通过标签覆盖率/纯度筛选。请检查 labels9.shp 是否是类别样本面，"
            "并可尝试把 MIN_LABEL_COVERAGE 调低到 0.20，或把 MIN_LABEL_PURITY 调低到 0.70；"
            "同时检查分割尺度是否过大。不要在未检查标签质量前盲目降低阈值。"
        )

    # 为每个训练对象找出覆盖其标签像元最多的原始样本面编号，用于按样本面分组验证。
    object_group = np.zeros(n_objects + 1, dtype=np.int32)
    pair_seg = seg_flat[labeled]
    pair_poly = poly_flat[labeled]
    pair_valid = pair_poly > 0
    if np.any(pair_valid):
        pairs = np.column_stack((pair_seg[pair_valid], pair_poly[pair_valid]))
        unique_pairs, pair_counts = np.unique(pairs, axis=0, return_counts=True)
        # 每个对象优先选择对应像元数最多的样本面。
        order = np.lexsort((-pair_counts, unique_pairs[:, 0]))
        sorted_pairs = unique_pairs[order]
        first_for_object = np.r_[True, sorted_pairs[1:, 0] != sorted_pairs[:-1, 0]]
        winners = sorted_pairs[first_for_object]
        object_group[winners[:, 0].astype(np.int64)] = winners[:, 1].astype(np.int32)

    # 少数对象可能因栅格化覆盖等原因没有组编号，给它们独立组编号并发出说明。
    missing_group = selected_ids[object_group[selected_ids] == 0]
    if missing_group.size:
        object_group[missing_group] = np.arange(1_000_000, 1_000_000 + missing_group.size, dtype=np.int32)

    print(f"满足阈值的训练对象数：{len(selected_ids):,}")
    print("每类入选对象数：")
    for class_id in CLASS_VALUES:
        count = int(np.sum(object_labels[selected_ids] == class_id))
        print(f"  class={class_id}: {count}")
    return object_labels, purity, coverage, object_group, selected_ids


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
    """按原始标注面分组划分，尽量避免同一个标注面同时出现在训练和验证集合。"""
    report_lines = []
    report_lines.append("GF-2 面向对象分类：独立分组验证报告")
    report_lines.append("验证方式：按标注面 ID 分组的留出验证（同一标注面的对象尽量不会同时进入训练和验证）。")
    report_lines.append(f"可用训练对象总数：{len(y)}")
    report_lines.append(f"各类别对象数：{ {int(c): int(np.sum(y == c)) for c in np.unique(y)} }")

    unique_groups = np.unique(groups)
    if len(unique_groups) < 4:
        report_lines.append("\n无法进行稳定的分组留出验证：独立标注面/组少于 4 个。")
        report_lines.append("建议为每种植被类型绘制多个空间分离、类别明确的样本面，再进行精度评估。")
        return "\n".join(report_lines)

    best_split = None
    best_score = -1e9
    all_classes = set(CLASS_VALUES)
    for seed in range(60):
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=VALIDATION_TEST_SIZE,
            random_state=RANDOM_STATE + seed,
        )
        try:
            train_idx, val_idx = next(splitter.split(X, y, groups=groups))
        except Exception:
            continue
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        train_classes = set(np.unique(y[train_idx]).tolist())
        val_classes = set(np.unique(y[val_idx]).tolist())
        score = (
            100 * len(train_classes.intersection(all_classes))
            + 10 * len(val_classes.intersection(all_classes))
            - 5 * abs((len(val_idx) / len(y)) - VALIDATION_TEST_SIZE)
        )
        if train_classes == all_classes:
            score += 1000
        if val_classes == all_classes:
            score += 500
        if score > best_score:
            best_score = score
            best_split = (train_idx, val_idx)
        if train_classes == all_classes and val_classes == all_classes:
            break

    if best_split is None:
        report_lines.append("\n没有找到可用的分组验证划分。")
        return "\n".join(report_lines)

    train_idx, val_idx = best_split
    train_classes = sorted(np.unique(y[train_idx]).tolist())
    val_classes = sorted(np.unique(y[val_idx]).tolist())
    report_lines.append(f"训练折对象数：{len(train_idx)}；验证折对象数：{len(val_idx)}")
    report_lines.append(f"训练折类别：{train_classes}")
    report_lines.append(f"验证折类别：{val_classes}")
    if set(CLASS_VALUES) - set(val_classes):
        report_lines.append(
            "警告：验证折缺少部分类别，通常说明某些类别的独立样本面太少；"
            "缺少类别的验证精度不能据此判定。"
        )
    if set(CLASS_VALUES) - set(train_classes):
        report_lines.append(
            "警告：训练折缺少部分类别，分组验证模型无法学习这些类别；"
            "最终制图模型仍会使用全部入选训练对象。"
        )

    val_model = make_rf()
    val_model.fit(X[train_idx], y[train_idx])
    val_pred = val_model.predict(X[val_idx])
    oa = accuracy_score(y[val_idx], val_pred)
    macro_f1 = f1_score(y[val_idx], val_pred, labels=CLASS_VALUES, average="macro", zero_division=0)
    cm = confusion_matrix(y[val_idx], val_pred, labels=CLASS_VALUES)
    report_lines.append(f"\n验证 OA：{oa:.4f}")
    report_lines.append(f"验证 Macro-F1：{macro_f1:.4f}")
    report_lines.append("\n混淆矩阵（行=真实类别，列=预测类别；类别顺序 1,2,3,4,5,6）：")
    report_lines.append(np.array2string(cm))
    report_lines.append("\n分类报告：")
    report_lines.append(
        classification_report(
            y[val_idx],
            val_pred,
            labels=CLASS_VALUES,
            target_names=[f"class_{c}" for c in CLASS_VALUES],
            zero_division=0,
        )
    )
    report_lines.append(
        "\n注意：该报告是按标注面分组的内部留出验证；若用于论文，仍建议准备独立的外业/判读验证样本，"
        "并按空间单元设计验证。"
    )
    return "\n".join(report_lines)


def save_raster(path, array, profile, dtype, nodata=0, colormap=None):
    out_profile = profile.copy()
    out_profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="lzw",
        BIGTIFF="IF_SAFER",
    )
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(array.astype(dtype, copy=False), 1)
        if colormap is not None:
            try:
                dst.write_colormap(1, colormap)
            except Exception as exc:
                print(f"[提示] 未能写入分类色表（不影响分类值）：{exc}")


def main():
    image_path = Path(IMAGE_PATH)
    if not image_path.exists():
        stop(f"没有找到影像：{image_path}")

    shp_path = resolve_shapefile(LABELS_PATH)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("GF-2 面向对象分类：SLIC 分割 + 对象特征 + Random Forest")
    print(f"影像：{image_path}")
    print(f"标签：{shp_path}")
    print(f"输出：{output_dir}")
    print("=" * 72)

    with rasterio.open(image_path) as src:
        if src.crs is None:
            stop(
                "输入影像缺少坐标系/地理参考信息。请先在 ENVI 或 ArcGIS 中确认影像是否已正确地理配准，"
                "并且不要在未知真实坐标系的情况下随意指定投影。"
            )
        if src.count < 4:
            stop(
                f"影像只有 {src.count} 个波段，但本程序按 GF-2 四波段 B/G/R/NIR 计算特征。"
                "请确认输入的是四波段融合 GeoTIFF，而不是单波段 PAN 或三波段 RGB。"
            )
        width, height = src.width, src.height
        profile = src.profile.copy()
        raster_crs = src.crs
        transform = src.transform
        print(f"影像大小：{width:,} × {height:,}；波段数：{src.count}；CRS：{raster_crs}")
        if width * height > 15_000_000:
            print(
                "[内存提示] 当前影像超过 1500 万像元，SLIC 和对象特征可能占用较多内存。"
                "若发生内存不足，请先裁剪小区域测试。"
            )
        data = src.read(indexes=[1, 2, 3, 4], out_dtype="float32")
        valid_mask = src.dataset_mask() > 0

    valid_mask &= np.all(np.isfinite(data), axis=0)
    if TREAT_ALL_ZERO_AS_NODATA:
        valid_mask &= ~np.all(data == 0, axis=0)
    valid_count = int(valid_mask.sum())
    if valid_count == 0:
        stop("影像中没有可用像元，请检查 NoData 设置和输入文件。")
    print(f"有效影像像元数：{valid_count:,} / {width * height:,}")

    gdf = read_labels(shp_path, raster_crs)
    label_raster, polygon_raster = rasterize_labels(
        gdf, out_shape=(height, width), transform=transform
    )

    slic_image = normalize_for_segmentation(data, valid_mask)
    segments, n_objects = run_segmentation(slic_image, valid_mask)

    segment_path = output_dir / "obia_segments.tif"
    save_raster(segment_path, segments, profile, dtype="int32", nodata=0)
    print(f"已保存对象编号栅格：{segment_path}")

    X_all, feature_names, object_pixel_count = calculate_object_features(
        data, slic_image, segments, n_objects
    )
    object_labels, purity, coverage, object_group, selected_ids = assign_object_labels(
        segments, n_objects, label_raster, polygon_raster, object_pixel_count
    )

    selected_labels = object_labels[selected_ids]
    absent = sorted(set(CLASS_VALUES) - set(np.unique(selected_labels).tolist()))
    if absent:
        stop(
            f"通过筛选的训练对象中缺少类别 {absent}，无法训练完整的 1—6 类模型。\n"
            "建议先在 GIS 中检查这些类别的样本面是否落在影像范围内、class 值是否正确；"
            "然后再考虑把 MIN_LABEL_COVERAGE 调到 0.20 或 MIN_LABEL_PURITY 调到 0.70，"
            "并重新检查分割尺度。"
        )

    X_train = X_all[selected_ids]
    y_train = selected_labels.astype(np.uint8)
    groups = object_group[selected_ids]
    report_text = grouped_validation(X_train, y_train, groups)
    report_path = output_dir / "obia_validation_report.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print("\n[6/7] 分组验证报告：")
    print(report_text)
    print(f"验证报告已保存：{report_path}")

    print("\n训练最终随机森林模型，并对全部分割对象分类……")
    final_model = make_rf()
    final_model.fit(X_train, y_train)
    object_predictions = np.zeros(n_objects + 1, dtype=np.uint8)
    object_predictions[1:] = final_model.predict(X_all[1:]).astype(np.uint8)
    class_map = object_predictions[segments]
    class_map[~valid_mask] = 0

    color_map = {
        0: (0, 0, 0, 0),
        1: (34, 139, 34, 255),
        2: (154, 205, 50, 255),
        3: (255, 215, 0, 255),
        4: (205, 133, 63, 255),
        5: (30, 144, 255, 255),
        6: (186, 85, 211, 255),
    }
    class_path = output_dir / "obia_classification_rf.tif"
    save_raster(class_path, class_map, profile, dtype="uint8", nodata=0, colormap=color_map)

    # 导出每个对象的特征、标签覆盖情况和预测类别，便于后续与 SVM/XGBoost 进行对照。
    df = pd.DataFrame(X_all[1:], columns=feature_names)
    df.insert(0, "segment_id", np.arange(1, n_objects + 1, dtype=np.int32))
    df["sample_class"] = object_labels[1:]
    df["sample_coverage"] = coverage[1:]
    df["sample_purity"] = purity[1:]
    df["selected_for_training"] = np.isin(np.arange(1, n_objects + 1), selected_ids)
    df["predicted_class"] = object_predictions[1:]
    feature_csv = output_dir / "obia_object_features.csv"
    df.to_csv(feature_csv, index=False, encoding="utf-8-sig")

    model_path = output_dir / "obia_rf_model.joblib"
    joblib.dump(
        {
            "model": final_model,
            "feature_names": feature_names,
            "class_values": CLASS_VALUES,
            "segmentation": {
                "method": "SLIC",
                "target_segments": TARGET_SEGMENTS,
                "compactness": SLIC_COMPACTNESS,
                "sigma": SLIC_SIGMA,
            },
        },
        model_path,
    )

    print("\n[7/7] 全部完成！输出文件如下：")
    print(f"1. 分类结果：{class_path}")
    print(f"2. 对象编号：{segment_path}")
    print(f"3. 对象特征表：{feature_csv}")
    print(f"4. 精度验证报告：{report_path}")
    print(f"5. 已训练模型：{model_path}")
    print("\n分类值说明：1、2、3、4、5、6 对应标签矢量 class 字段中的原始类别编码；0 为 NoData。")
    print("请在 ArcGIS/ENVI 中把分类结果按唯一值渲染，并核对每个类别的实际含义。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n用户中断程序。")
        sys.exit(130)
    except MemoryError:
        print(
            "\n[内存不足] 请先把影像裁剪成较小的试验区域，降低像元数量后再运行。",
            file=sys.stderr,
        )
        sys.exit(2)
