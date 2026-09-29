import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from pathlib import Path
from tqdm import tqdm

# ==========================================
# 1. 滤波器配置 (核心调试区)
# ==========================================
RAW_PATH = Path(r"D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest\Testdata")
CLEAN_PATH = Path(r"D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest\Cleaned_Data")

FC_CONFIG = {
    'disp': 30.0,    # 位移截止频率
    'force': 30.0,   # 力截止频率
    'curr': 40.0,    # 电流
    'temp': 2.0,     # 温度
    'fs': 1000.0,
    'order': 4
}

def zero_phase_filter(data, fc):
    nyq = 0.5 * FC_CONFIG['fs']
    norm_cutoff = fc / nyq
    b, a = butter(FC_CONFIG['order'], norm_cutoff, btype='low')
    return filtfilt(b, a, data)

class CDCDataCleaner:
    def __init__(self):
        CLEAN_PATH.mkdir(parents=True, exist_ok=True)

    def _process_single(self, csv_path):
        folder = csv_path.parent.name
        raw = pd.read_csv(csv_path, skiprows=15, encoding='gbk')
        
        # 原始量转换
        d_raw = (raw.iloc[:, 3] - 4) / 16 * 1500
        f_raw = (raw.iloc[:, 5] / 10) * 1000 * 9.80665
        i_raw = raw.iloc[:, 6] / 1000.0
        t_raw = raw.iloc[:, 8]

        # 零相位滤波
        d_f = zero_phase_filter(d_raw.values, FC_CONFIG['disp'])
        f_f = zero_phase_filter(f_raw.values, FC_CONFIG['force'])
        i_f = zero_phase_filter(i_raw.values, FC_CONFIG['curr'])
        t_f = zero_phase_filter(t_raw.values, FC_CONFIG['temp'])

        # 运动段寻找
        if "AMP" in folder or "Vel" in folder:
            active = np.where(np.abs(d_f - d_f[:200].mean()) > 0.15)[0]
        else:
            active = np.where(i_f > 0.02)[0]
        
        if len(active) == 0: return None
        s, e = active[0], active[-1]

        # --- 修复 Mean of empty slice 逻辑 ---
        if s > 30:
            f_offset = f_f[:s].mean()
            mts_ref = d_f[:s].mean()
        else:
            # 如果s太小，强制取前50个点作为静态参考
            f_offset = f_f[:50].mean() if len(f_f) > 50 else f_f[0]
            mts_ref = d_f[:50].mean() if len(d_f) > 50 else d_f[0]

        rod_length = 250.0 - (d_f[s:e] - mts_ref)
        force_tared = f_f[s:e] - f_offset

        # 链式高质量求导
        vel = np.gradient(rod_length, 0.001)
        vel_smooth = zero_phase_filter(vel, FC_CONFIG['disp']) # 速度二次平滑
        acc = np.gradient(vel_smooth, 0.001)
        i_dot = np.gradient(i_f[s:e], 0.001)

        return pd.DataFrame({
            'time': np.arange(len(rod_length)) * 0.001,
            'rod_length': np.round(rod_length, 3),
            'velocity': np.round(vel_smooth, 3),
            'accel': np.round(acc, 2),
            'force': np.round(force_tared, 2),
            'current': np.round(i_f[s:e], 3),
            'current_dot': np.round(i_dot, 3),
            'temp': np.round(t_f[s:e], 2)
        }), folder

    def clean_all(self):
        all_csvs = list(RAW_PATH.rglob('auto_1.csv'))
        print(f"[*] Processing {len(all_csvs)} files...")
        for p in tqdm(all_csvs, ascii=True):
            res = self._process_single(p)
            if res:
                df, name = res
                df.to_csv(CLEAN_PATH / f"{name}.csv", index=False)

    def plot_review(self, filename):
        """修改此处的文件名即可看图"""
        path = CLEAN_PATH / f"{filename}.csv"
        if not path.exists(): 
            print(f"File {filename}.csv not found. Did you run clean_all()?")
            return
        
        df = pd.read_csv(path)
        fig, axs = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f"Ground Truth Data Review: {filename}", fontsize=14)
        
        configs = [
            ('rod_length', 'mm', 'g'), ('velocity', 'mm/s', 'orange'),
            ('accel', 'mm/s²', 'purple'), ('force', 'N', 'r'),
            ('current', 'A', 'b'), ('current_dot', 'A/s', 'cyan')
        ]
        for i, (col, unit, color) in enumerate(configs):
            ax = axs[i//2, i%2]
            ax.plot(df['time'], df[col], color=color, linewidth=1)
            ax.set_title(f"{col} ({unit})"); ax.grid(True, alpha=0.3)
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95]); plt.show()

if __name__ == "__main__":
    cleaner = CDCDataCleaner()
    cleaner.clean_all() # 第一次运行请取消注释以生成数据
    # cleaner.plot_review("aRMS10mm_4Hz") # 在此修改文件名