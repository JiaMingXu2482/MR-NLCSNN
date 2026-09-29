import pandas as pd
import numpy as np
import shutil
import os
from pathlib import Path
import re
import time

# ==========================================
# Configuration
# ==========================================
STATS_CSV = Path(r"D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest\Stats\data_summary.csv")
SOURCE_DIR = Path(r"D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest\Cleaned_Data")
TRAIN_DATA_DIR = Path(r"D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest\TrainData")

def build_training_database():
    if not STATS_CSV.exists():
        print("Error: data_summary.csv not found."); return

    # --- 改进的文件夹清理逻辑 ---
    if not TRAIN_DATA_DIR.exists():
        TRAIN_DATA_DIR.mkdir(parents=True, exist_ok=True)
    else:
        print(f"[*] Cleaning existing files in {TRAIN_DATA_DIR}...")
        for item in TRAIN_DATA_DIR.iterdir():
            try:
                if item.is_file():
                    item.unlink() # 删除文件
                elif item.is_dir():
                    shutil.rmtree(item) # 删除子目录
            except Exception as e:
                print(f" [!] Could not delete {item.name}: {e}. (Is it open in Excel?)")

    # 等待一秒让系统释放句柄
    time.sleep(1)

    df = pd.read_csv(STATS_CSV)
    train_paths = []
    
    # === 1. 划分逻辑 (保持不变) ===
    # Cat 1: 谐波抽稀 (0, 0.4, 0.8, 1.2, 1.4 A)
    cat1 = df[df['category'] == '1_Steady_Harmonic']
    train_i = [0.0, 0.4, 0.8, 1.2, 1.4]
    train_paths.extend(cat1[cat1['current_a'].isin(train_i)]['filepath'].tolist())

    # Cat 2: 稳态随机平衡
    cat2 = df[df['category'] == '2_Steady_Random']
    for _, group in cat2.groupby(['input_val', 'freq_hz']):
        paths = group.sort_values('current_a')['filepath'].tolist()
        if len(paths) <= 2:
            train_paths.extend(paths)
        else:
            val_indices = [len(paths)//3, 2*len(paths)//3]
            for i, p in enumerate(paths):
                if i not in val_indices: train_paths.append(p)

    # Cat 4: LF Random Current
    cat4 = df[df['category'] == '4_Random_LF_Curr']
    for _, row in cat4.iterrows():
        if row['is_slow']: 
            train_paths.append(row['filepath'])
        else:
            if hash(row['filename']) % 2 != 0: train_paths.append(row['filepath'])

    # Cat 5: Step
    train_paths.extend(df[df['category'] == '5_Step_Current']['filepath'].tolist())

    # === 2. 执行文件拷贝 ===
    print(f"[*] Copying {len(train_paths)} files to TrainData folder...")
    copied_count = 0
    for p_str in train_paths:
        src = Path(p_str)
        if src.exists():
            try:
                shutil.copy2(src, TRAIN_DATA_DIR / src.name)
                copied_count += 1
            except Exception as e:
                print(f" [!] Copy failed for {src.name}: {e}")
    
    # === 3. 统计并报告 ===
    analyze_and_report(copied_count)

def analyze_and_report(count):
    files = list(TRAIN_DATA_DIR.glob("*.csv"))
    if not files:
        print("No files copied. Report skipped."); return
        
    all_data = []
    for f in files:
        tmp = pd.read_csv(f)
        name = f.stem
        freq = 0.0
        match = re.search(r'_(\d+\.?\d*)Hz', name)
        if match: freq = float(match.group(1))
        
        all_data.append({
            'name': f.name,
            'samples': len(tmp),
            'f_max': tmp['force'].max(),
            'f_min': tmp['force'].min(),
            'x_max': tmp['rod_length'].max(),
            'x_min': tmp['rod_length'].min(),
            'di_max': tmp['current_dot'].abs().max(),
            'freq': freq
        })
    
    res = pd.DataFrame(all_data)
    
    report = []
    report.append("="*60)
    report.append("CDC TRAINING DATASET WHITEPAPER")
    report.append("="*60)
    report.append(f"Generated on:      {pd.Timestamp.now()}")
    report.append(f"Total Files:       {count}")
    report.append(f"Total Samples:     {res['samples'].sum():,}")
    report.append(f"Total Duration:    {res['samples'].sum()*0.001/60:.2f} minutes")
    report.append("-" * 60)
    report.append("PHYSICAL BOUNDARIES (For NN Normalization)")
    report.append(f"Max Abs Force:     {max(res['f_max'].max(), abs(res['f_min'].min())):.2f} N")
    report.append(f"Rod Length Range:  [{res['x_min'].min():.2f}, {res['x_max'].max():.2f}] mm")
    report.append(f"Max Current_dot:   {res['di_max'].max():.2f} A/s")
    report.append("-" * 60)
    report.append("FREQUENCY COVERAGE")
    report.append(str(sorted(res['freq'].unique().tolist())))
    report.append("-" * 60)
    report.append("FILE LISTING")
    for n in sorted(res['name'].tolist()):
        report.append(f" - {n}")
    report.append("="*60)

    with open(TRAIN_DATA_DIR / "train_data_info.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report))
    
    print(f"\n[SUCCESS] TrainData built with {count} files.")
    print(f"Statistics report saved to: {TRAIN_DATA_DIR / 'train_data_info.txt'}")

if __name__ == "__main__":
    build_training_database()