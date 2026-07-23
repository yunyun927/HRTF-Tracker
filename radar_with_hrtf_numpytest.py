"""
PyQt5 版 - 接入神经网络HRTF模型 (纯Numpy版本)
方位角标准: 0=正前方, 左负右正, 身后(|az|>90)显示在边缘
"""

import os
import sys
import subprocess

# ========== 依赖检测与自动安装 ==========
REQUIRED_PACKAGES = {
    'numpy': 'numpy',
    'sounddevice': 'sounddevice',
    'PyQt5': 'PyQt5',
}

missing = []
for module, package in REQUIRED_PACKAGES.items():
    try:
        __import__(module)
    except ImportError:
        missing.append(package)

if missing:
    print(f"检测到缺少依赖: {', '.join(missing)}")
    print("正在自动安装...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
        print("安装完成！请重新运行程序。")
        input("按回车退出...")
        sys.exit(0)
    except subprocess.CalledProcessError as e:
        print(f"安装失败: {e}")
        print("请手动运行: pip install " + " ".join(missing))
        input("按回车退出...")
        sys.exit(1)

# 依赖安装完成后再导入
import sounddevice as sd
import numpy as np
import threading
import pickle
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QColor, QBrush, QPen


# ========== 神经网络HRTF插件 (纯Numpy版本) ==========
class NeuralHRTFPlugin:
    """
    神经网络HRTF方位估计插件
    基于KEMAR HRTF数据训练，纯Numpy推理
    """
    
    def __init__(self, model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hrtf_model_numpy_fixed.pkl")):
        with open(model_path, 'rb') as f:
            self.weights = pickle.load(f)
        print("[OK] HRTF神经网络已加载 (Numpy模式)")
    
    def _relu(self, x):
        return np.maximum(0, x)
    
    def _batch_norm(self, x, name):
        w = self.weights[name]
        x_norm = (x - w['running_mean']) / np.sqrt(w['running_var'] + w['eps'])
        return x_norm * w['weight'] + w['bias']
    
    def _linear(self, x, name):
        w = self.weights[name]
        return x @ w['weight'].T + w['bias']
    
    def _forward(self, x):
        """纯numpy前向传播"""
        x = (x - self.weights['norm_mean']) / self.weights['norm_std']
        
        x = self._linear(x, 'net.0')
        x = self._batch_norm(x, 'net.1')
        x = self._relu(x)
        
        x = self._linear(x, 'net.4')
        x = self._batch_norm(x, 'net.5')
        x = self._relu(x)
        
        x = self._linear(x, 'net.8')
        x = self._batch_norm(x, 'net.9')
        x = self._relu(x)
        
        x = self._linear(x, 'net.11')
        x = self._relu(x)
        
        x = self._linear(x, 'net.13')
        
        return x
    
    def predict(self, left_fft, right_fft, tdoa=0.0, ild=0.0):
        """
        预测声源方向
        
        返回:
            azimuth: -180~180° (0=正前方, 左负右正)
            elevation: -90~90° (0=水平, 上正下负)
        """
        n_freqs = min(128, len(left_fft), len(right_fft))
        ratio = np.log(np.abs(left_fft[:n_freqs]) / 
                      (np.abs(right_fft[:n_freqs]) + 1e-10) + 1e-10)
        ratio = np.clip(ratio, -5, 5)
        
        if len(ratio) < 128:
            ratio = np.pad(ratio, (0, 128 - len(ratio)), mode='constant')
        
        features = np.concatenate([ratio, [tdoa, ild]])
        x = features.reshape(1, -1).astype(np.float32)
        
        output = self._forward(x)[0]
        
        # 方位角规范化: 任意实数 -> -180~180 (0=前, 左负右正)
        raw_az = float(output[0])
        az_360 = raw_az % 360
        if az_360 > 180:
            azimuth = az_360 - 360
        else:
            azimuth = az_360
        
        # 仰角规范化: 任意实数 -> -90~90
        raw_el = float(output[1])
        elevation = ((raw_el + 90) % 180) - 90
        
        return azimuth, elevation


# ========== 查找 CABLE Output ==========
devices = sd.query_devices()
CABLE_IDX = None
for i, d in enumerate(devices):
    if 'CABLE Output' in d['name'] and d['max_input_channels'] > 0:
        CABLE_IDX = i
        print(f"[OK] CABLE Output: [{i}] {d['name']}")
        break
if CABLE_IDX is None:
    print("[ERR] CABLE Output not found")
    exit(1)


class RadarWindow(QWidget):
    def __init__(self, device_id):
        super().__init__()
        self.device_id = device_id
        self.sample_rate = 48000
        self.chunk_size = 4096
        
        try:
            self.hrtf_plugin = NeuralHRTFPlugin()
        except Exception as e:
            print(f"[ERR] 无法加载HRTF模型: {e}")
            self.hrtf_plugin = None
        
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        
        screen = QApplication.primaryScreen().geometry()
        self.sw = screen.width()
        self.sh = screen.height()
        self.cx = self.sw // 2
        self.cy = self.sh // 2
        self.setGeometry(0, 0, self.sw, self.sh)
        
        self._lock = threading.Lock()
        self._azimuth = 0.0
        self._elevation = 0.0
        self._intensity = 0.0
        self._balance = 0.0
        self._has_new = False
        self.running = False
        
        self._s_az = 0
        self._s_el = 0
        self._s_intens = 0
        self._s_bal = 0
        
        self._display_alpha = 0.0
        self._target_alpha = 0.0
        self._FADE_IN = 0.15
        self._FADE_OUT = 0.03
        
        self.BALANCE_THRESHOLD = 0.10
        
        self.timer = QTimer()
        self.timer.timeout.connect(self._update)
        self.timer.start(50)
        
        self.show()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # 中心十字
        painter.setPen(QPen(QColor(255, 255, 255, 40), 1))
        painter.drawLine(self.cx - 8, self.cy, self.cx + 8, self.cy)
        painter.drawLine(self.cx, self.cy - 8, self.cx, self.cy + 8)
        
        alpha = int(self._display_alpha * 140)
        
        if alpha > 5:
            mx, my = self._map_to_screen(self._azimuth, self._elevation)
            size = int(8 + self._intensity * 0.12)
            
            color = QColor(255, 60, 60, alpha)
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(mx - size, my - size, size * 2, size * 2)
    
    def _map_to_screen(self, az, el):
        """
        坐标系:
        - 屏幕中心 = 正前方 (az=0°, el=0°)
        - x轴: 左负右正
        - y轴: 上负下正 (屏幕y向下，el>0=上方，所以y减小)
        - 身后(|az|>90)显示在边缘
        """
        # 方位角
        if abs(az) <= 90:
            x = self.cx + (az / 90) * self.cx * 0.8
        else:
            sign = 1 if az > 0 else -1
            x = self.cx + sign * self.cx * 0.85
        
        # 仰角: el>0=上方，屏幕y向下，所以el越大y越小
        # 实测发现方向反了，加负号修复
        el = -el  # 反转仰角方向
        
        el_clipped = np.clip(el, -45, 45)
        y = self.cy - (el_clipped / 45) * self.cy * 0.4
        
        margin = 40
        x = max(margin, min(self.sw - margin, x))
        y = max(margin, min(self.sh - margin, y))
        
        return int(x), int(y)
    
    def _audio_cb(self, indata, frames, time_info, status):
        if indata.shape[1] < 2:
            return
        
        L = indata[:, 0].astype(np.float64)
        R = indata[:, 1].astype(np.float64)
        
        n = len(L)
        window = np.hanning(n)
        fft_L = np.abs(np.fft.rfft(L * window))
        fft_R = np.abs(np.fft.rfft(R * window))
        fft_avg = (fft_L + fft_R) / 2
        
        freqs = np.fft.rfftfreq(n, 1/self.sample_rate)
        
        low_mask = (freqs >= 30) & (freqs < 150)
        high_mask = (freqs >= 1000) & (freqs < 6000)
        
        low_e = np.mean(fft_avg[low_mask]) if np.any(low_mask) else 0
        high_e = np.mean(fft_avg[high_mask]) if np.any(high_mask) else 0
        
        rms_l = np.sqrt(np.mean(L**2))
        rms_r = np.sqrt(np.mean(R**2))
        
        mx = max(rms_l, rms_r, 0.001)
        tot = rms_l + rms_r
        bal = (rms_r - rms_l) / tot if tot > 0.001 else 0
        balance = abs(bal)
        
        intens = min(tot / mx * 50, 100)
        
        # 神经网络HRTF定位
        if self.hrtf_plugin is not None and tot > 0.01:
            # TDOA
            corr = np.correlate(L, R, mode='full')
            max_idx = np.argmax(np.abs(corr))
            lag = max_idx - (len(L) - 1)
            tdoa = lag / self.sample_rate
            
            # ILD
            ild = 20 * np.log10(rms_l / (rms_r + 1e-10))
            
            try:
                az, el = self.hrtf_plugin.predict(fft_L, fft_R, tdoa, ild)
                self._s_az = 0.15 * az + 0.85 * self._s_az
                self._s_el = 0.15 * el + 0.85 * self._s_el
            except Exception as e:
                print(f"[WARN] HRTF预测失败: {e}")
                self._s_az = 0.12 * bal * 90 + 0.88 * self._s_az
                self._s_el = 0.08 * 0 + 0.92 * self._s_el
        else:
            # 备用模式
            self._s_az = 0.12 * bal * 90 + 0.88 * self._s_az
            
            total_e = low_e + high_e + 0.001
            low_ratio = low_e / total_e
            high_ratio = high_e / total_e
            
            if low_ratio > 0.55:
                el = -0.7
            elif high_ratio > 0.45:
                el = 0.6
            else:
                el = 0
            
            self._s_el = 0.08 * el + 0.92 * self._s_el
        
        self._s_intens = 0.12 * intens + 0.88 * self._s_intens
        self._s_bal = 0.12 * balance + 0.88 * self._s_bal
        
        with self._lock:
            self._azimuth = self._s_az
            self._elevation = self._s_el
            self._intensity = self._s_intens
            self._balance = self._s_bal
            self._has_new = True
    
    def _update(self):
        if not self.running:
            return
        
        with self._lock:
            az = self._azimuth
            el = self._elevation
            intens = self._intensity
            balance = self._balance
            has_new = self._has_new
            self._has_new = False
        
        if not has_new or balance < self.BALANCE_THRESHOLD or intens < 5:
            self._target_alpha = 0
        else:
            self._target_alpha = min(intens / 100 * 0.6 + 0.05, 0.65)
        
        if self._display_alpha < self._target_alpha:
            self._display_alpha += self._FADE_IN
            if self._display_alpha > self._target_alpha:
                self._display_alpha = self._target_alpha
        elif self._display_alpha > self._target_alpha:
            self._display_alpha -= self._FADE_OUT
            if self._display_alpha < self._target_alpha:
                self._display_alpha = self._target_alpha
        
        self.update()
    
    def start(self):
        self.running = True
        
        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=2,
                dtype=np.float32,
                blocksize=self.chunk_size,
                device=self.device_id,
                callback=self._audio_cb
            )
            self._stream.start()
            print("[OK] Stream started")
        except Exception as e:
            print(f"[ERR] {e}")
            return
        
        print("=" * 50)
        print("Radar with Neural HRTF")
        print("坐标系: 中心=正前方, 左负右正, 上正下负")
        print("ESC to exit")
        print("=" * 50)
    
    def stop(self):
        print("\nClosing...")
        self.running = False
        self.timer.stop()
        try:
            self._stream.stop()
            self._stream.close()
        except:
            pass
        self.close()
    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.stop()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    radar = RadarWindow(device_id=CABLE_IDX)
    radar.start()
    sys.exit(app.exec_())