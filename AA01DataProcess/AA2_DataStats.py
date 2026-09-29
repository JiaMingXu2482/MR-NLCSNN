import pandas as pd
import numpy as np
import re
from pathlib import Path

CLEAN_PATH = Path(r"D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest\Cleaned_Data")
STATS_PATH = Path(r"D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest\Stats")

class CDCAnalyzer:
    def __init__(self):
        STATS_PATH.mkdir(parents=True, exist_ok=True)
        
    def _parse_filename(self, name):
        # 兼容 slow, mm 等命名
        pat = r'^(?P<pre>AMP|a?RMS)(?P<val>\d+\.?\d*)(?:mm)?_(?P<f>\d+\.?\d*)Hz(?P<slow>slow)?(?:_(?P<c>\d+\.?\d*)A)?$'
        m = re.match(pat, name)
        if "Vel" in name: return "Step", 0, 0, 0, False
        if m:
            pre = m.group('pre')
            val = float(m.group('val'))
            freq = float(m.group('f'))
            is_slow = True if m.group('slow') else False
            curr = float(m.group('c')) if m.group('c') else -1.0 # -1标识随机电流
            
            # 归类
            if curr >= 0:
                cat = "1_Steady_Harmonic" if pre == "AMP" else "2_Steady_Random"
            else:
                cat = "3_Random_HF_Curr" if pre == "aRMS" else "4_Random_LF_Curr"
            return cat, val, freq, curr, is_slow
        return "Unknown", 0, 0, 0, False

    def analyze(self):
        files = list(CLEAN_PATH.glob("*.csv"))
        records = []
        
        print(f"[*] Analyzing {len(files)} cleaned files...")
        for p in files:
            df = pd.read_csv(p)
            cat, val, freq, curr, slow = self._parse_filename(p.stem)
            
            records.append({
                'filepath': str(p.resolve()),
                'filename': p.name,
                'category': cat,
                'input_val': val,      # AMP or RMS
                'freq_hz': freq,
                'current_a': curr,
                'is_slow': slow,
                'duration_s': len(df) * 0.001,
                'f_max': df['force'].max(),
                'f_min': df['force'].min(),
                'v_max': df['velocity'].max(),
                'v_min': df['velocity'].min(),
                'x_max': df['rod_length'].max(),
                'x_min': df['rod_length'].min()
            })
            
        res_df = pd.DataFrame(records)
        res_df.to_csv(STATS_PATH / "data_summary.csv", index=False)
        
        print("\n" + "="*50)
        print("TERMINAL SUMMARY REPORT")
        print("="*50)
        print(res_df.groupby('category')['filename'].count())
        print("-" * 50)
        print(f"Unique Freqs: {sorted(res_df['freq_hz'].unique())}")
        print(f"Unique RMS:   {sorted(res_df['input_val'].unique())}")
        print(f"Global Rod Length: [{res_df['x_min'].min():.1f}, {res_df['x_max'].max():.1f}]")
        print(f"Global Force Range: [{res_df['f_min'].min():.1f}, {res_df['f_max'].max():.1f}]")
        print("="*50)
        print(f"Stats saved to: {STATS_PATH / 'data_summary.csv'}")

if __name__ == "__main__":
    CDCAnalyzer().analyze()