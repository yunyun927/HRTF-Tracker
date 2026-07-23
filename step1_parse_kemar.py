# === step1_parse_kemar_fixed.py ===
# 修复: 支持负仰角文件名 (L-40e000a.wav)

import os
import re
import numpy as np
from scipy.io import wavfile
from scipy.fft import rfft
import sys


def parse_kemar_full(data_dir):
    print(f"扫描目录: {data_dir}")
    
    elev_dirs = []
    for d in os.listdir(data_dir):
        if d.startswith('elev'):
            try:
                elev = int(d.replace('elev', ''))
                elev_dirs.append((elev, os.path.join(data_dir, d)))
            except:
                pass
    
    elev_dirs.sort()
    print(f"找到 {len(elev_dirs)} 个仰角文件夹")
    
    results = []
    total_skipped = 0
    
    for elev, elev_path in elev_dirs:
        left_files = {}
        right_files = {}
        
        for f in os.listdir(elev_path):
            if not f.endswith('.wav'):
                continue
            
            # 修复: 支持负仰角 (-40, -30 等)
            # 文件名格式: L-40e000a.wav 或 R-40e000a.wav
            match = re.match(r'([LR])(-?\d+)e(\d+)a\.wav', f)
            if match:
                ear = match.group(1)
                file_elev = int(match.group(2))  # 现在支持负数
                azimuth = int(match.group(3))
                
                filepath = os.path.join(elev_path, f)
                if ear == 'L':
                    left_files[azimuth] = filepath
                else:
                    right_files[azimuth] = filepath
        
        # 配对左右耳
        paired = 0
        for azimuth in sorted(left_files.keys()):
            if azimuth not in right_files:
                total_skipped += 1
                continue
            
            try:
                sr_l, left_data = wavfile.read(left_files[azimuth])
                sr_r, right_data = wavfile.read(right_files[azimuth])
                
                if left_data.ndim > 1:
                    left_hrir = left_data[:, 0].astype(np.float32)
                else:
                    left_hrir = left_data.astype(np.float32)
                
                if right_data.ndim > 1:
                    right_hrir = right_data[:, 0].astype(np.float32)
                else:
                    right_hrir = right_data.astype(np.float32)
                
                max_val = max(np.abs(left_hrir).max(), np.abs(right_hrir).max())
                if max_val > 0:
                    left_hrir = left_hrir / max_val
                    right_hrir = right_hrir / max_val
                
                left_fft = np.abs(rfft(left_hrir, n=256))[:128]
                right_fft = np.abs(rfft(right_hrir, n=256))[:128]
                
                results.append({
                    'azimuth': float(azimuth),
                    'elevation': float(elev),  # 现在可以是负数
                    'left_fft': left_fft,
                    'right_fft': right_fft,
                    'sample_rate': sr_l
                })
                paired += 1
                
            except Exception as e:
                print(f"  跳过 elev{elev} az{azimuth}: {e}")
                total_skipped += 1
        
        print(f"  elev{elev:3d}: {paired}个方向配对成功")
    
    print(f"\n总共解析了 {len(results)} 个方向")
    print(f"跳过: {total_skipped}")
    if results:
        print(f"方位角范围: {min(r['azimuth'] for r in results):.0f} ~ {max(r['azimuth'] for r in results):.0f}")
        print(f"仰角范围: {min(r['elevation'] for r in results):.0f} ~ {max(r['elevation'] for r in results):.0f}")
    
    return results


def save_parsed_data(directions, output_path):
    azimuths = np.array([d['azimuth'] for d in directions], dtype=np.float32)
    elevations = np.array([d['elevation'] for d in directions], dtype=np.float32)
    left_ffts = np.array([d['left_fft'] for d in directions], dtype=np.float32)
    right_ffts = np.array([d['right_fft'] for d in directions], dtype=np.float32)
    
    np.savez(output_path,
             azimuths=azimuths,
             elevations=elevations,
             left_ffts=left_ffts,
             right_ffts=right_ffts)
    
    print(f"\n已保存到: {output_path}")
    print(f"文件大小: {os.path.getsize(output_path) / 1024:.1f} KB")


if __name__ == "__main__":
    log_path = r"D:\gra\step1_log_fixed.txt"
    
    class Logger:
        def __init__(self, filepath):
            self.terminal = sys.stdout
            self.log = open(filepath, "w", encoding="utf-8")
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
            self.log.flush()
        def flush(self):
            self.terminal.flush()
            self.log.flush()
    
    sys.stdout = Logger(log_path)
    
    print("=" * 50)
    print("步骤1: 解析KEMAR Full HRTF (修复负仰角)")
    print("=" * 50)
    
    data_dir = r"D:\gra\full"
    output_path = r"D:\gra\KEMAR_parsed_fixed.npz"
    
    try:
        if not os.path.exists(data_dir):
            print(f"错误: 目录不存在: {data_dir}")
        else:
            directions = parse_kemar_full(data_dir)
            if directions:
                save_parsed_data(directions, output_path)
                print("\n步骤1完成!")
            else:
                print("\n错误: 没有解析到任何数据")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n日志保存: {log_path}")
    input("\n按回车键退出...")