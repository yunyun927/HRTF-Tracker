# === step2_generate_data_fixed.py ===

import numpy as np
import pickle
import os


def generate_training_data(parsed_path, n_samples=100000, output_path="training_data_fixed.pkl"):
    print("加载HRTF数据...")
    data = np.load(parsed_path)
    
    azimuths = data['azimuths']
    elevations = data['elevations']
    left_ffts = data['left_ffts']
    right_ffts = data['right_ffts']
    
    n_directions = len(azimuths)
    print(f"加载了 {n_directions} 个方向")
    print(f"仰角范围: {elevations.min():.0f} ~ {elevations.max():.0f}")
    
    X = []
    Y = []
    
    np.random.seed(42)
    
    print(f"生成 {n_samples} 个训练样本...")
    
    for i in range(n_samples):
        idx = np.random.randint(0, n_directions)
        
        azimuth = azimuths[idx]
        elevation = elevations[idx]
        hrtf_l = left_ffts[idx]
        hrtf_r = right_ffts[idx]
        
        source_spectrum = np.random.randn(128) + 1j * np.random.randn(128)
        source_spectrum = np.abs(source_spectrum)
        
        received_l = source_spectrum * hrtf_l
        received_r = source_spectrum * hrtf_r
        
        received_l += np.random.randn(128) * 0.01
        received_r += np.random.randn(128) * 0.01
        
        spectral_ratio = np.log(np.abs(received_l) / (np.abs(received_r) + 1e-10) + 1e-10)
        spectral_ratio = np.clip(spectral_ratio, -5, 5)
        
        mic_distance = 0.17
        c = 343
        tdoa = mic_distance * np.sin(np.radians(azimuth)) / c
        tdoa += np.random.normal(0, 5e-6)
        
        ild = 20 * np.log10(np.mean(np.abs(received_l)) / (np.mean(np.abs(received_r)) + 1e-10))
        ild += np.random.normal(0, 0.3)
        
        features = np.concatenate([spectral_ratio, [tdoa, ild]])
        
        X.append(features)
        Y.append([azimuth, elevation])
        
        if (i + 1) % 20000 == 0:
            print(f"  已生成 {i+1}/{n_samples}")
    
    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.float32)
    
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0) + 1e-8
    X = (X - X_mean) / X_std
    
    train_data = {
        'X': X,
        'Y': Y,
        'X_mean': X_mean,
        'X_std': X_std
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(train_data, f)
    
    print(f"\n训练数据已保存: {output_path}")
    print(f"X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"文件大小: {os.path.getsize(output_path) / 1024 / 1024:.1f} MB")
    
    return train_data


if __name__ == "__main__":
    parsed_path = r"D:\gra\KEMAR_parsed_fixed.npz"
    output_path = r"D:\gra\training_data_fixed.pkl"
    
    if not os.path.exists(parsed_path):
        print(f"错误: 找不到 {parsed_path}")
        print("请先运行 step1_parse_kemar_fixed.py")
    else:
        generate_training_data(parsed_path, n_samples=100000, output_path=output_path)
        print("\n步骤2完成!")
        input("\n按回车键退出...")