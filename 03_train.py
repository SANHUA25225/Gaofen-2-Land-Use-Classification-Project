# -*- coding: utf-8 -*-
"""GF-2 土地覆盖分类训练脚本（Prithvi-EO-2.0 300M 微调）。

前置: 01/02 已跑通，data/tiles/ 和 data/stats.json 已生成。

用法（在 prithvi 环境里，cd 到本目录后运行）:
    python 03_train.py

产物: output/checkpoints/best.ckpt
"""
import json
import os
from pathlib import Path

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import albumentations as A
import lightning as L
from albumentations.pytorch import ToTensorV2
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics import MetricCollection
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report
import numpy as np

from terratorch.datamodules import GenericNonGeoSegmentationDataModule
from terratorch.tasks import SemanticSegmentationTask

# ---- 可按需调整的参数 ----
BATCH_SIZE = 2
MAX_EPOCHS = 40
PATIENCE = 10  # 验证 loss 连续多少轮不降就早停
LEARNING_RATE = 5.0e-5
DECODER_CHANNELS = 128  # 报 CUDA out of memory 时改 64
# ------------------------

NUM_CLASSES = 7  # 0=未标注(训练时忽略) + 1~6 七类地物
BASE = Path(__file__).resolve().parent
BANDS = ["BLUE", "GREEN", "RED", "NIR_BROAD"]  # GF-2 四波段映射到预训练模型对应位置

stats_file = BASE / "data" / "stats.json"
if not stats_file.exists():
    print("错误: 找不到 data/stats.json，请先运行 02_tile.py 完成切片。")
    raise SystemExit(1)
stats = json.loads(stats_file.read_text(encoding="utf-8"))
means, stds = stats["means"], stats["stds"]

train_tf = A.Compose(
    [A.HorizontalFlip(), A.VerticalFlip(), A.RandomRotate90(), ToTensorV2()],
    is_check_shapes=False,
)
val_tf = A.Compose([ToTensorV2()], is_check_shapes=False)

dm = GenericNonGeoSegmentationDataModule(
    batch_size=BATCH_SIZE,
    num_workers=0,  # Windows 下保持 0，改大会卡死
    num_classes=NUM_CLASSES,
    train_data_root=str(BASE / "data" / "tiles" / "train"),
    val_data_root=str(BASE / "data" / "tiles" / "val"),
    img_grep="img_*.tif",
    label_grep="label_*.tif",
    means=means,
    stds=stds,
    dataset_bands=BANDS,
    output_bands=BANDS,
    train_transform=train_tf,
    val_transform=val_tf,
    # 修复1: 标签 tif 若带 nodata 声明, 训练框架读取时会把对应像素变成 NaN,
    # NaN 转整数后成为巨大负数, 导致验证指标计算报 "bincount ... non-negative" 崩溃。
    # no_label_replace=0 把 NaN 替换回 0, 而 0 正是 ignore_index 要忽略的未标注类。
    no_label_replace=0,
    no_data_replace=0,
    drop_last=True,
)

task = SemanticSegmentationTask(
    model_factory="EncoderDecoderFactory",
    model_args={
        "backbone_pretrained": True,
        "backbone": "prithvi_eo_v2_300",
        "decoder": "UperNetDecoder",
        "num_classes": NUM_CLASSES,
        "rescale": True,
        "backbone_bands": BANDS,
        "head_dropout": 0.1,
        "decoder_channels": DECODER_CHANNELS,
        "head_channel_list": [128, 64],
        "necks": [
            {"name": "SelectIndices", "indices": [5, 11, 17, 23]},
            {"name": "ReshapeTokensToImage"},
        ],
    },
    loss="focal",
    ignore_index=0,
    lr=LEARNING_RATE,
    optimizer="AdamW",
    optimizer_hparams={"weight_decay": 0.1},
    scheduler="CosineAnnealingLR",
    scheduler_hparams={"T_max": MAX_EPOCHS},
    freeze_backbone=False,
    freeze_decoder=False,
    plot_on_val=False,
)

# 修复2: TerraTorch 1.2.11 自带的 Boundary_mIoU 指标存在形状 bug——
# 训练框架把预测结果展平成一维传给所有指标, 而它只接受三维/四维, 会报
# "preds must be (N,C,H,W) or (N,H,W)"。此处重建指标集把它移除,
# 其余指标(mIoU/F1/Accuracy/IoU_各类)不受影响。
for _attr in ("train_metrics", "val_metrics"):
    _coll = getattr(task, _attr)
    _prefix = _attr.split("_")[0] + "/"
    _kept = {k.split("/", 1)[1]: v for k, v in _coll.items() if "Boundary" not in k}
    setattr(task, _attr, MetricCollection(_kept, prefix=_prefix, compute_groups=False))

trainer = L.Trainer(
    accelerator="auto",
    precision="16-mixed",
    max_epochs=MAX_EPOCHS,
    check_val_every_n_epoch=1,
    log_every_n_steps=10,
    default_root_dir=str(BASE / "output"),
    logger=TensorBoardLogger(save_dir=str(BASE / "output"), name="gf2_landcover"),
    callbacks=[
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(patience=PATIENCE, monitor="val/loss"),
        ModelCheckpoint(
            dirpath=str(BASE / "output" / "checkpoints"),
            filename="best",
            monitor="val/loss",
            save_top_k=1,
        ),
    ],
)

trainer.fit(task, datamodule=dm)
# ==================== 自动精度评估 ====================
print("\n开始加载最佳模型进行精度评估...")
best_ckpt_path = BASE / "output" / "checkpoints" / "best.ckpt"
task_eval = SemanticSegmentationTask.load_from_checkpoint(best_ckpt_path, map_location="cpu")
task_eval = task_eval.to("cuda" if torch.cuda.is_available() else "cpu").eval()

val_loader = dm.val_dataloader()
preds_all, labels_all = [], []

with torch.no_grad():
    for batch in val_loader:
        imgs = batch["image"].to(task_eval.device)
        labels = batch["mask"].to(task_eval.device)
        logits = task_eval.model(imgs)
        if hasattr(logits, "output"):
            logits = logits.output
        elif hasattr(logits, "logits"):
            logits = logits.logits
        pred = logits.argmax(dim=1).cpu().numpy().flatten()
        preds_all.extend(pred)
        labels_all.extend(labels.cpu().numpy().flatten())

preds_all = np.array(preds_all)
labels_all = np.array(labels_all)

# 过滤掉 ignore_index=0 的像素
mask = labels_all != 0
y_true = labels_all[mask]
y_pred = preds_all[mask]

print("\n" + "="*50)
print("         Prithvi 大模型最终精度报告")
print("="*50)
print(f"总体精度 (OA): {accuracy_score(y_true, y_pred):.4f}")
print(f"Kappa 系数  : {cohen_kappa_score(y_true, y_pred):.4f}")
print("\n各类别详细精度:")
target_names = ["耕地", "林地", "草地", "水域", "建设用地", "未利用地"]
print(classification_report(y_true, y_pred,
                            labels=list(range(1, NUM_CLASSES)),
                            target_names=target_names,
                            zero_division=0))
print("="*50)
# =====================================================
print("=== 训练完成 ===")
print("最佳模型:", BASE / "output" / "checkpoints" / "best.ckpt")
print("下一步: python 04_predict_scene.py 影像.tif output/checkpoints/best.ckpt result.tif")
