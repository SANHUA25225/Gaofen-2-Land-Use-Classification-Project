# -*- coding: utf-8 -*-
"""提取训练日志指标 + 确认最优检查点"""
import os, glob, json
from pathlib import Path

env_py = r"C:\Users\sanhua\miniconda3\envs\prithvi\python.exe"
BASE = Path(r"C:\gf2-landcover\output")

# 1) 解析所有 tfevents
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

print("===== 各次训练运行结果 =====")
for vdir in sorted(glob.glob(str(BASE / "gf2_landcover" / "version_*")), key=lambda p: int(p.split("_")[-1])):
    ea = EventAccumulator(vdir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if not tags:
        print(Path(vdir).name, ": 无标量数据")
        continue
    summary = {}
    for tag in tags:
        events = ea.Scalars(tag)
        vals = [(e.step, e.value) for e in events]
        summary[tag] = vals
    n_epochs = max((s for s, _ in summary.get("val/loss", [])), default=0) + 1 if "val/loss" in summary else "?"
    print(f"\n--- {Path(vdir).name} (验证轮数≈{n_epochs}) ---")
    for tag in sorted(summary):
        if tag in ("val/loss", "train/loss"):
            last = summary[tag][-1]
            best = min(summary[tag], key=lambda x: x[1]) if tag.endswith("loss") else max(summary[tag], key=lambda x: x[1])
            print(f"  {tag}: 最终={last[1]:.4f} 最优={best[1]:.4f}")
        elif "IoU" in tag or "Jaccard" in tag or "F1" in tag or "Accuracy" in tag or "Acc" in tag:
            best = max(summary[tag], key=lambda x: x[1])
            last = summary[tag][-1]
            print(f"  {tag}: 最终={last[1]:.4f} 最高={best[1]:.4f}")

# 2) 读最新 ckpt 的 checkpoint 元数据
import torch
ckpts = sorted(glob.glob(str(BASE / "checkpoints" / "*.ckpt")), key=os.path.getmtime)
print("\n===== 检查点文件(按修改时间) =====")
for c in ckpts:
    print(f"  {os.path.basename(c)}  {os.path.getmtime(c)}")

newest = ckpts[-1]
print(f"\n读取最新检查点元数据: {os.path.basename(newest)}")
try:
    ck = torch.load(newest, map_location="cpu", mmap=True, weights_only=False)
except Exception as e:
    print("mmap 失败, 常规加载:", e)
    ck = torch.load(newest, map_location="cpu", weights_only=False)
cb = ck.get("callbacks", {})
for k, v in cb.items():
    if "ModelCheckpoint" in str(k) and isinstance(v, dict):
        print("  best_model_path:", v.get("best_model_path"))
        print("  best_model_score:", v.get("best_model_score"))
print("  epoch:", ck.get("epoch"), " global_step:", ck.get("global_step"))
hp = ck.get("hyper_parameters", {})
print("  关键超参: lr=", hp.get("lr"), " num_classes=", (hp.get("task_args") or {}).get("num_classes") if isinstance(hp.get("task_args"), dict) else "?")
