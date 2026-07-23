# === step4_export_onnx.py ===
# 导出训练好的模型为ONNX格式

import torch
import torch.nn as nn
import numpy as np
import pickle
import os


# 模型定义（必须和训练时一致）
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
            nn.Linear(16, 2)
        )
    
    def forward(self, x):
        return self.net(x)


# 加载模型
print("加载模型...")
model = LightweightHRTFNet()

# 加载训练好的权重
model.load_state_dict(torch.load(r"D:\gra\best_hrtf_model.pth", map_location='cpu'))
model.eval()

# 加载标准化参数
print("加载标准化参数...")
with open(r"D:\gra\training_data.pkl", 'rb') as f:
    data = pickle.load(f)

X_mean = torch.from_numpy(data['X_mean']).float()
X_std = torch.from_numpy(data['X_std']).float()


# 包装模型（包含标准化层）
class HRTFNetWithNorm(nn.Module):
    def __init__(self, model, mean, std):
        super().__init__()
        self.model = model
        self.register_buffer('mean', mean)
        self.register_buffer('std', std)
    
    def forward(self, x):
        x = (x - self.mean) / self.std
        return self.model(x)

wrapped_model = HRTFNetWithNorm(model, X_mean, X_std)
wrapped_model.eval()


# 导出ONNX
output_path = r"D:\gra\hrtf_net.onnx"
print(f"导出ONNX: {output_path}")

dummy_input = torch.randn(1, 130)

torch.onnx.export(
    wrapped_model,
    dummy_input,
    output_path,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11,
    do_constant_folding=True,
)

print("ONNX导出完成!")

# 验证
import onnx
onnx_model = onnx.load(output_path)
onnx.checker.check_model(onnx_model)
print("ONNX模型验证通过!")

# 检查文件大小
file_size = os.path.getsize(output_path) / 1024
print(f"文件大小: {file_size:.1f} KB")

# 测试推理速度
import onnxruntime as ort
import time

print("\n测试推理速度...")
session = ort.InferenceSession(output_path)
test_input = np.random.randn(1, 130).astype(np.float32)

# 预热
for _ in range(10):
    session.run(None, {"input": test_input})

# 测速
n_runs = 1000
start = time.time()
for _ in range(n_runs):
    session.run(None, {"input": test_input})
elapsed = time.time() - start

print(f"推理速度: {n_runs/elapsed:.0f} FPS")
print(f"单次延迟: {elapsed/n_runs*1000:.2f} ms")

# 测试一个实际推理
print("\n测试实际推理:")
sample_input = np.random.randn(1, 130).astype(np.float32)
output = session.run(None, {"input": sample_input})[0][0]
print(f"  输入: 随机特征")
print(f"  输出方位角: {output[0]:.2f}°")
print(f"  输出仰角: {output[1]:.2f}°")

print("\n步骤4完成!")
input("\n按回车键退出...")