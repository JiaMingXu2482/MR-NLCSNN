"""
训练用最简配置：改下面 _DEFAULT 一行即可。
也可用环境变量 CDC_PROCESSED_GROUPS 覆盖。

项目按四步组织：
  AA01DataProcess — 原始文件、信号转换、统计、划分
  AA02Train     — 训练脚本与输出（模型、训练过程验证图）
  AA03Validate — 用（二）的模型跑验证集并保存结果
  AA04References — 文献与笔记
"""
import os

# Processed_Groups 根目录（其下应有 1_Steady_Harmonic、2_Steady_Random 等子文件夹）
_DEFAULT = r"D:\OneDrive - mail.scut.edu.cn\Matlab\CDCTest\Processed_Groups"

PROCESSED_GROUPS = os.environ.get("CDC_PROCESSED_GROUPS", _DEFAULT).strip()
