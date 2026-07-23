# === step3_train.py ===
# 训练轻量HRTF方位估计网络

import pickle
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import os


# 加载数据
print("加载训练数据...")
with open(r"D:\gra\training_data.pkl", 'rb') as f:
    data = pickle.load(f)

X = data['X']
Y = data['Y']

# 划分训练集/验证集 (90/10)
n_train = int(0.9 * len(X))
indices = np.random.permutation(len(X))
train_idx = indices[:n_train]
val_idx = indices[n_train:]

X_train, Y_train = X[train_idx], Y[train_idx]
X_val, Y_val = X[val_idx], Y[val_idx]

print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}")


# PyTorch Dataset
class HRTFDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


train_ds = HRTFDataset(X_train, Y_train)
val_ds = HRTFDataset(X_val, Y_val)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256)


# 模型定义
class LightweightHRTFNet(nn.Module):
    def __init__(self, input_dim=130):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 2)  # [azimuth, elevation]
        )
        
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.net(x)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LightweightHRTFNet().to(device)

# 统计参数量
total_params = sum(p.numel() for p in model.parameters())
print(f"模型参数量: {total_params:,} (~{total_params * 4 / 1024:.1f} KB)")


# 损失函数：考虑角度周期性
class AngularLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target):
        # 方位角周期性 (-180~180)
        az_diff = torch.abs(pred[:, 0] - target[:, 0])
        az_diff = torch.min(az_diff, 360 - az_diff)
        
        # 仰角损失
        el_diff = torch.abs(pred[:, 1] - target[:, 1])
        
        return torch.mean(az_diff**2 + el_diff**2)


criterion = AngularLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)


# 训练循环
best_val_loss = float('inf')
epochs = 100

print(f"\n开始训练 ({epochs} epochs)...")
print("-" * 60)

for epoch in range(epochs):
    # 训练
    model.train()
    train_loss = 0
    for batch_X, batch_Y in train_loader:
        batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        train_loss += loss.item()
    
    # 验证
    model.eval()
    val_loss = 0
    val_az_err = 0
    val_el_err = 0
    
    with torch.no_grad():
        for batch_X, batch_Y in val_loader:
            batch_X, batch_Y = batch_X.to(device), batch_Y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            val_loss += loss.item()
            
            # 计算角度误差
            az_diff = torch.abs(outputs[:, 0] - batch_Y[:, 0])
            az_diff = torch.min(az_diff, 360 - az_diff)
            val_az_err += az_diff.mean().item()
            
            val_el_err += torch.abs(outputs[:, 1] - batch_Y[:, 1]).mean().item()
    
    train_loss /= len(train_loader)
    val_loss /= len(val_loader)
    val_az_err /= len(val_loader)
    val_el_err /= len(val_loader)
    
    scheduler.step()
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1:3d} | Train: {train_loss:.3f} | Val: {val_loss:.3f} | "
              f"AzErr: {val_az_err:.2f}° | ElErr: {val_el_err:.2f}°")
    
    # 保存最佳模型
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), r"D:\gra\best_hrtf_model.pth")

print("-" * 60)
print(f"训练完成! 最佳验证损失: {best_val_loss:.4f}")
print(f"模型已保存: D:\\gra\\best_hrtf_model.pth")

input("\n按回车键退出...")