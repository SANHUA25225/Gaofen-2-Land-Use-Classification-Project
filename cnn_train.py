# -*- coding: utf-8 -*-
"""轻量级 CNN 对比实验：训练 + 精度评价 + 整图推理。"""
import os, glob
import numpy as np
import tifffile as tiff
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, cohen_kappa_score, classification_report

# ========== 1. 配置 ==========
TRAIN_DIR = r"D:\gf2-landcover\data\tiles\train"
VAL_DIR   = r"D:\gf2-landcover\data\tiles\val"
SCENE_TIF = r"D:\gf2\train_area8.tif"          # 用于整图推理
OUT_CKPT  = r"D:\gf2-landcover\cnn_best.pth"
OUT_MAP   = r"D:\gf2\result_cnn.tif"

BATCH_SIZE    = 16
EPOCHS        = 30
LEARNING_RATE = 1e-3
NUM_CLASSES   = 7    # 0 未标注 + 1~6 七类地物
IN_CHANNELS   = 4    # GF-2 蓝、绿、红、近红外
TILE          = 224
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 2. 数据集 ==========
class TileDataset(Dataset):
    def __init__(self, data_dir):
        self.img_files = sorted(glob.glob(os.path.join(data_dir, "img_*.tif")))
        self.label_files = [f.replace("img_", "label_") for f in self.img_files]
        # 只保留标签存在的样本
        pairs = [(i, l) for i, l in zip(self.img_files, self.label_files) if os.path.exists(l)]
        self.img_files, self.label_files = zip(*pairs)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        # 读取影像 (H, W, C) 并转为 float32
        img = tiff.imread(self.img_files[idx]).astype(np.float32)
        # 转换为 (C, H, W) 以适配 PyTorch 卷积层
        if img.ndim == 3 and img.shape[-1] == 4:
            img = img.transpose(2, 0, 1)
        # 读取标签 (H, W)
        label = tiff.imread(self.label_files[idx]).astype(np.int64)
        # 返回 Tensor
        return torch.from_numpy(img), torch.from_numpy(label)
# ========== 3. 轻量级 CNN ==========
class SimpleCNN(nn.Module):
    def __init__(self, in_ch=4, num_cls=8):
        super().__init__()
        # 编码器（提取特征，逐步降低分辨率）
        self.encoder = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2), # 224 -> 112
            nn.Conv2d(32, 64, 3, padding=1),    nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2), # 112 -> 56
            nn.Conv2d(64, 128, 3, padding=1),   nn.BatchNorm2d(128), nn.ReLU(), # 56
        )
        # 解码器（恢复分辨率，输出逐像素分类结果）
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # 56 -> 112
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True), # 112 -> 224
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, num_cls, 1) # 1x1 卷积调整类别数，输出 (Batch, 8, 224, 224)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
# ========== 4. 训练 ==========
def train():
    train_ds = TileDataset(TRAIN_DIR)
    val_ds   = TileDataset(VAL_DIR)
    train_ld = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_ld   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print(f"训练样本: {len(train_ds)}  验证样本: {len(val_ds)}")

    model = SimpleCNN(IN_CHANNELS, NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # 忽略未标注
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_acc = 0.0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for imgs, labels in train_ld:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # 验证
        model.eval()
        preds_all, labels_all = [], []
        with torch.no_grad():
            for imgs, labels in val_ld:
                imgs = imgs.to(DEVICE)
                out = model(imgs)
                pred = out.argmax(dim=1).cpu().numpy().flatten()
                preds_all.extend(pred)
                labels_all.extend(labels.numpy().flatten())

        preds_all  = np.array(preds_all)
        labels_all = np.array(labels_all)
        mask = labels_all != 0
        acc = accuracy_score(labels_all[mask], preds_all[mask])
        kappa = cohen_kappa_score(labels_all[mask], preds_all[mask])
        print(f"Epoch [{epoch+1}/{EPOCHS}]  Loss: {total_loss/len(train_ld):.4f}  OA: {acc:.4f}  Kappa: {kappa:.4f}")

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), OUT_CKPT)
            print(f"  -> 保存最佳模型 (OA: {best_acc:.4f})")

    print(f"\n训练完成！最佳 OA: {best_acc:.4f}")

    # ---- 最终评估 ----
    model.load_state_dict(torch.load(OUT_CKPT))
    model.eval()
    preds_all, labels_all = [], []
    with torch.no_grad():
        for imgs, labels in val_ld:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            pred = out.argmax(dim=1).cpu().numpy().flatten()
            preds_all.extend(pred); labels_all.extend(labels.numpy().flatten())
    preds_all, labels_all = np.array(preds_all), np.array(labels_all)
    mask = labels_all != 0
    print("\n=== CNN 最终精度评价 ===")
    print(f"总体精度 OA: {accuracy_score(labels_all[mask], preds_all[mask]):.4f}")
    print(f"Kappa 系数: {cohen_kappa_score(labels_all[mask], preds_all[mask]):.4f}")
    print("\n各类别精度报告:")
    target_names = ["耕地", "林地", "草地", "水域", "建设用地", "未利用地"]
    print(classification_report(labels_all[mask], preds_all[mask],
                                labels=list(range(1, NUM_CLASSES)),
                                target_names=target_names, zero_division=0))

# ========== 5. 整图推理 ==========
def predict_scene():
    """对整幅影像做滑窗推理，输出类别图。"""
    print("\n开始整图推理...")
    import rasterio
    from rasterio.windows import Window

    model = SimpleCNN(IN_CHANNELS, NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(OUT_CKPT))
    model.eval()

    with rasterio.open(SCENE_TIF) as src:
        H, W = src.height, src.width
        result = np.zeros((H, W), dtype=np.uint8)

        total = ((H + TILE - 1) // TILE) * ((W + TILE - 1) // TILE)
        done = 0
        for y in range(0, H, TILE):
            for x in range(0, W, TILE):
                # 用 rasterio 读取，得到 (C, h, w)
                win = Window(x, y, min(TILE, W - x), min(TILE, H - y))
                patch = src.read(window=win).astype(np.float32)
                ch, ph, pw = patch.shape

                # 边缘补齐
                if ph < TILE or pw < TILE:
                    pad = np.zeros((ch, TILE, TILE), dtype=np.float32)
                    pad[:, :ph, :pw] = patch
                    patch = pad

                t = torch.from_numpy(patch).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    out = model(t).argmax(dim=1)[0].cpu().numpy().astype(np.uint8)

                result[y:y+ph, x:x+pw] = out[:ph, :pw]
                done += 1
                if done % 50 == 0 or done == total:
                    print(f"进度: {done}/{total} 块")

        # 输出配置文件，保证地理坐标与原始影像一致
        profile = src.profile.copy()
        profile.update(count=1, dtype="uint8", compress="lzw", nodata=0)

    with rasterio.open(OUT_MAP, "w", **profile) as dst:
        dst.write(result, 1)

    print(f"推理完成，结果已保存至: {OUT_MAP}")


if __name__ == "__main__":
    # 如果模型已经训练过，直接跑推理；否则先训练再推理
    if os.path.exists(OUT_CKPT):
        print(f"检测到已训练模型: {OUT_CKPT}，跳过训练直接推理。")
        predict_scene()
    else:
        train()
        predict_scene()