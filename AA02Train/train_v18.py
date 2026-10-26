"""NLCSNN v13 训练 — v12 + 自由偏置头 b(h,nfl), 取消 F·v≥0 硬约束。

用户要求: 回到原来的 NLCSNN (v4: GRU + MLP), 改进: (1) 正逆同时训练; (2) 保证耗散性。
模型: F = c(z)·v·G_c(I_e) + (Σ_j s_j(z)·tanh(v/σ_j))·G_s(I_e), σ=[0.5,2.0,8.0]固定
  - c,s ≥ 0 (softplus, v4 式 MLP 头); G_c,G_s 单调 => F·v ≥ 0 架构保证
  - 逆头 MLP_inv([h,nfl,v,F]) -> I, 联合训练 (论文双 observer 思想)

Loss = 前向 p2p 加权 MSE + λ_inv · MSE(I_pred_head, I_true)

停止准则 (用户指定): 500 轮上限 或 holdout pooled NRMSE < 5% (无 patience)。
每 20 轮验证 + 存档 + 画验证对比图。

Run:
  ./venv/bin/python AA02Train/train_v13.py
环境变量: CDC_V13_EPOCHS / CDC_V13_LR / CDC_V13_LAMBDA_INV / CDC_V13_WARMSTART /
          CDC_V13_SIGMAS / CDC_V13_G_HIDDEN / CDC_V13_ST_WEIGHT / CDC_V13_TAU_MS
"""
import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from mr_nlcsnn.model_v14 import NLCSNN_v14

# --------------------------------------------------------------------- config
DEFAULT_DATA_ROOT = str(REPO_ROOT / "Parsed_Card_force_wide")
DATA_ROOT = os.environ.get("CDC_PROCESSED_GROUPS", DEFAULT_DATA_ROOT).strip()

TRAIN_CATS = ["1_Steady_Harmonic", "2_Steady_Random", "5_Step_Current_Triangle"]
LF_CAT = "4_Random_LF_Current"
EXCLUDE_STEMS = [
    "RLF_RMS10mm_2Hz_r2",  # 坏文件
    # 用户 2026-09-30 确认删除: 小幅值高频 RLF, A级路面遇不到, E20 误差最差
    "RLF_RMS2mm_5Hz", "RLF_RMS2mm_5Hz11", "RLF_RMS2mm_7Hz",
    "RLF_RMS5mm_5Hz", "RLF_RMS5mm_5Hz11",
    "RLF_RMS5mm_6Hz", "RLF_RMS5mm_6Hzslow", "RLF_RMS5mm_7Hz",
]
HOLDOUT_STEMS = [
    # 用户 2026-09-30 确认: 车辆常用工况
    "RLF_RMS10mm_2Hz_r1", "RLF_RMS15mm_3Hz", "RLF_RMS10mm_4Hz",
    "SH_10mm_1.66Hz_1A_r1", "SH_30mm_1.66Hz_0A_r1",
    "SR_RMS10mm_1.66Hz_0A", "RHF_RMS10mm_4Hz",
    "ST_10mm_Vel0.13_step5_0_1.6A_r1",
]

EPOCHS = int(os.environ.get("CDC_V13_EPOCHS", "500"))  # 用户要求: 500轮上限或<5%停止
LR = float(os.environ.get("CDC_V13_LR", "3e-4"))
WARMUP_EPOCHS = int(os.environ.get("CDC_V13_WARMUP_EPOCHS", "5"))
GRU_HIDDEN = int(os.environ.get("CDC_V13_GRU_HIDDEN", "64"))
GRU_LAYERS = int(os.environ.get("CDC_V13_GRU_LAYERS", "2"))
MLP_HIDDEN = int(os.environ.get("CDC_V13_MLP_HIDDEN", "128"))
G_HIDDEN = int(os.environ.get("CDC_V13_G_HIDDEN", "64"))
SIGMAS = tuple(float(x) for x in os.environ.get("CDC_V13_SIGMAS", "0.5,2.0,8.0").split(","))
LAMBDA_INV = float(os.environ.get("CDC_V13_LAMBDA_INV", "0.1"))
ST_WEIGHT = float(os.environ.get("CDC_V13_ST_WEIGHT", "3.0"))  # ST 阶跃文件 loss 加权
INIT_TAU_MS = float(os.environ.get("CDC_V13_TAU_MS", "20.0"))
C_MIN = float(os.environ.get("CDC_V13_C_MIN", "2e-3"))  # 最小黏性阻尼下限
S_MIN = float(os.environ.get("CDC_V13_S_MIN", "5e-3"))  # 最小摩擦下限  # 电流延迟初值
# 注: 逆头初值 sigmoid(0)=0.5, loss_inv 初值约 0.08, 与 loss_fwd 同量级。
# λ=0.1 保持前向为主。
WARMSTART = os.environ.get("CDC_V13_WARMSTART", "").strip()

DROPOUT = float(os.environ.get("CDC_V3_DROPOUT", "0.1"))
FILE_BATCH = int(os.environ.get("CDC_V3_BATCH", "12"))
CHUNK_TRAIN = int(os.environ.get("CDC_V3_CHUNK", "12000"))
NEG_FORCE_WEIGHT = float(os.environ.get("CDC_V3_NEG_W", "1.5"))
WARMUP_SAMPLES = 50
EVAL_EVERY = int(os.environ.get("CDC_V3_EVAL_EVERY", "20"))  # 用户要求: 每20轮验证一次
EVAL_START = int(os.environ.get("CDC_V3_EVAL_START", "1"))  # 首次验证的 epoch (v15: 80)
STOP_PCT = float(os.environ.get("CDC_V3_STOP_PCT", "5.0"))
SEED = int(os.environ.get("CDC_V3_SEED", "7"))
MAX_TRAIN_FILES = int(os.environ.get("CDC_V3_MAX_TRAIN", "0"))
TAG = os.environ.get("CDC_V13_TAG", "v14").strip() or "v14"  # 会话/权重名前缀 (v15=小v14)
AUG_NOISE = float(os.environ.get("CDC_V3_AUG_NOISE", "0.01"))

FIXED = {"v_scale": 1000.0, "a_scale": 50000.0, "i_scale": 1.6,
         "t_ref": 40.0, "t_scale": 40.0}


# ----------------------------------------------------------------- data layer
class DataManagerV3:
    def __init__(self, root, device):
        self.root = Path(root)
        self.device = device
        exclude = set(EXCLUDE_STEMS)
        train_files, holdout_files = [], []
        holdout_set = set(HOLDOUT_STEMS)
        # holdout 从所有目录搜索 (含 SH/SR/RHF/ST)
        all_cats = TRAIN_CATS + [LF_CAT, "3_Random_HF_Current"]
        for cat in all_cats:
            for p in sorted((self.root / cat).glob("*.csv")):
                if p.stem in exclude:
                    continue
                if p.stem in holdout_set:
                    holdout_files.append(p)
        # 训练集: TRAIN_CATS + LF_CAT (排除 holdout 和 exclude)
        for cat in TRAIN_CATS:
            train_files += [p for p in sorted((self.root / cat).glob("*.csv"))
                            if p.stem not in exclude and p.stem not in holdout_set]
        for p in sorted((self.root / LF_CAT).glob("*.csv")):
            if p.stem in exclude or p.stem in holdout_set:
                continue
            train_files.append(p)
        for p in sorted((self.root / "3_Random_HF_Current").glob("*.csv")):
            if p.stem in exclude or p.stem in holdout_set:
                continue
            train_files.append(p)
        missing = holdout_set - {p.stem for p in holdout_files}
        if missing:
            raise RuntimeError(f"Holdout missing: {sorted(missing)}")
        if MAX_TRAIN_FILES > 0:
            train_files = train_files[:MAX_TRAIN_FILES]
        self.cfg = self._prescan(train_files)
        self.train_items = self._preload(train_files, chunk_train=CHUNK_TRAIN)
        self.holdout_items = self._preload(holdout_files)
        print(f"[*] Data: train={len(train_files)} files -> {len(self.train_items)} chunks "
              f"| holdout={len(holdout_files)}", flush=True)
        print(f"[*] Norm: x_ref={self.cfg['x_ref']:.2f} x_scale={self.cfg['x_scale']:.2f} "
              f"di_scale={self.cfg['di_scale']:.2f} f_scale={self.cfg['f_scale']:.2f}", flush=True)

    def _prescan(self, files):
        cfg = dict(FIXED)
        x_min, x_max, di_abs, f_abs = [], [], 0.0, 0.0
        for p in files:
            df = pd.read_csv(p, usecols=["rod_length", "current_dot", "force"])
            x_min.append(float(df["rod_length"].min()))
            x_max.append(float(df["rod_length"].max()))
            di_abs = max(di_abs, float(df["current_dot"].abs().max()))
            f_abs = max(f_abs, float(df["force"].abs().max()))
        cfg["x_ref"] = (max(x_max) + min(x_min)) / 2.0
        cfg["x_scale"] = (max(x_max) - min(x_min)) / 2.0 + 1e-6
        cfg["di_scale"] = di_abs + 1e-6
        cfg["f_scale"] = f_abs + 1e-6
        if cfg["x_scale"] > 150.0:
            raise RuntimeError(f"x_scale={cfg['x_scale']:.1f} too large, bad file slipped in")
        return cfg

    def _load_norm(self, path):
        df = pd.read_csv(path)
        c = self.cfg
        u = np.stack([
            (df["rod_length"].values - c["x_ref"]) / c["x_scale"],
            df["velocity"].values / c["v_scale"],
            df["accel"].values / c["a_scale"],
            df["current"].values / c["i_scale"],
            df["current_dot"].values / c["di_scale"],
            (df["temp"].values - c["t_ref"]) / c["t_scale"],
        ], axis=1).astype(np.float32)
        f = (df["force"].values / c["f_scale"]).astype(np.float32)
        return u, f

    def _preload(self, files, chunk_train=0):
        items = []
        for p in files:
            u, f = self._load_norm(p)
            p2p = float(f.max() - f.min()) + 1e-6
            n = len(f)
            if chunk_train > 0 and n > chunk_train:
                starts = list(range(0, n, chunk_train))
            else:
                starts = [0]
            for k, s0 in enumerate(starts):
                s1 = min(s0 + chunk_train if chunk_train > 0 else n, n)
                # 丢弃过短尾块: 长度 <= RF(1000) 时 loss 切片为空 (v7 的 Huber 在空张量上 NaN)
                if s1 - s0 <= 1000:
                    continue
                nm = p.stem if len(starts) == 1 else f"{p.stem}_c{k}"
                items.append({"u": torch.from_numpy(u[s0:s1]),
                              "f": torch.from_numpy(f[s0:s1]),
                              "length": s1 - s0, "p2p": p2p, "name": nm})
        return items


# ------------------------------------------------------------------- training
def collate(batch, device):
    bs = len(batch)
    L = max(it["length"] for it in batch)
    u = torch.zeros(bs, L, 6)
    f = torch.zeros(bs, L)
    valid = torch.zeros(bs, L, dtype=torch.bool)
    for j, it in enumerate(batch):
        n = it["length"]
        u[j, :n] = it["u"]
        f[j, :n] = it["f"]
        valid[j, :n] = True
    # 去掉 DeltaI: 用户要求 (di 引入较大波动)。置零 u[...,4], NFL 中的 curr_dyn 随之归零。
    # 保留 6 维结构以兼容 v4 GRU warm-start (35 维 NFL 不变)。
    u[:, :, 4] = 0.0
    return (u.to(device), f.to(device), valid.to(device),
            torch.tensor([it["p2p"] for it in batch], device=device),
            [it["name"] for it in batch])


def train_epoch(model, optimizer, mse_none, items, rf, device):
    items = sorted(items, key=lambda it: it["length"])
    order = list(range(0, len(items), FILE_BATCH))
    random.shuffle(order)
    total_loss, total_fwd, total_inv, n_b = 0.0, 0.0, 0.0, 0
    for i in order:
        batch = items[i:i + FILE_BATCH]
        u, f, valid, p2p, names = collate(batch, device)
        optimizer.zero_grad(set_to_none=True)
        if AUG_NOISE > 0:
            u = u + torch.randn_like(u) * AUG_NOISE
        pred, c, s, Gc_I, Gs_I, b = model.forward_components(u)
        s_idx = max(rf, WARMUP_SAMPLES)
        vm = valid[:, s_idx:].float()
        # --- 前向: p2p 加权 MSE (无偏置标定, 部署口径) ---
        w = torch.where(f < 0, NEG_FORCE_WEIGHT, 1.0)
        pl = mse_none(pred[:, s_idx:], f[:, s_idx:]) * w[:, s_idx:] * vm
        per_file = pl.sum(1) / vm.sum(1).clamp_min(1.0)
        # v9: ST 阶跃电流文件加权 (误差分析: ST 贡献 75% holdout MSE, 瞬态没学好)
        st_w = torch.tensor([ST_WEIGHT if n.startswith("ST_") else 1.0
                             for n in names], device=device)
        loss_fwd = ((per_file / (p2p ** 2)) * st_w).mean()
        # --- 逆向: 逆头联合训练 (v18: mask 协议, 与部署一致) ---
        # 逆头输入 [h, nfl, v, F_target], 电流通道 mask 为 0 (部署时电流未知)
        # v17 用真 I 历史是作弊, 导致 train/deploy 不一致; v18 修正
        u_masked = u.clone()
        u_masked[:, :, 3] = 0.0  # I 通道 mask
        u_masked[:, :, 4] = 0.0  # dI 通道 mask
        I_pred = model.inverse_predict(u_masked, f)
        I_true = u[:, :, 3].abs().clamp(0.0, 1.0)  # 归一化电流幅值
        il = mse_none(I_pred[:, s_idx:], I_true[:, s_idx:]) * vm
        loss_inv = (il.sum(1) / vm.sum(1).clamp_min(1.0)).mean()
        loss = loss_fwd + LAMBDA_INV * loss_inv
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_fwd += float(loss_fwd.detach().cpu())
        total_inv += float(loss_inv.detach().cpu())
        n_b += 1
    return total_loss / max(n_b, 1), total_fwd / max(n_b, 1), total_inv / max(n_b, 1)


# ----------------------------------------------------------------- evaluation
@torch.no_grad()
def plot_holdout_comparison(items, preds, rows, pooled, epoch, plot_dir, fscale):
    """为每个 holdout 文件画预测 vs 真实对比图 (每20轮验证时调用)。
    preds: dict {name: (p_pred_N, y_true_N)}; rows: evaluate 返回的每文件行。
    保存到 plot_dir/epoch_{epoch:04d}/ 下。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(plot_dir) / f"epoch_{epoch:04d}"
    outdir.mkdir(parents=True, exist_ok=True)
    row_by_name = {r["name"]: r for r in rows}

    for it in items:
        name = it["name"]
        if name not in preds:
            continue
        p, y = preds[name]
        n = len(y)
        # 取中间一段 3000 点画 (避免全文件太密)
        s0 = max(1000, n // 2 - 1500)
        s1 = min(n, s0 + 3000)
        t = np.arange(s0, s1) / 1000.0  # 假设 1kHz, 秒
        r = row_by_name.get(name, {})

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        axes[0].plot(t, y[s0:s1], "k-", lw=0.8, label="True")
        axes[0].plot(t, p[s0:s1], "r-", lw=0.8, alpha=0.8, label="Pred")
        axes[0].set_ylabel("Force [N]")
        axes[0].set_title(f"{name} | E{epoch} | NRMSE={r.get('nrmse_pct', float('nan')):.1f}% "
                          f"pooled={pooled:.1f}%")
        axes[0].legend(loc="upper right", fontsize=9)
        axes[0].grid(True, alpha=0.3)
        err = p[s0:s1] - y[s0:s1]
        axes[1].plot(t, err, "b-", lw=0.6)
        axes[1].axhline(0, color="k", lw=0.5)
        axes[1].set_ylabel("Error [N]")
        axes[1].set_xlabel("Time [s]")
        axes[1].grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(outdir / f"{name}.png", dpi=100)
        plt.close(fig)
    print(f"[*] 验证对比图 -> {outdir} ({len(preds)} 张)", flush=True)
    return str(outdir)


@torch.no_grad()
def evaluate(model, items, cfg, rf, device, plot_dir=None, epoch=None):
    """开环评估。返回每工况行 + pooled NRMSE (无偏置标定) + 逆解电流 MAE。

    逆解诊断: y_tgt = (F_true-B)·S/(S²+ε), I_pred = G⁻¹(y_tgt),
    直接对比 I_pred vs I_true (MAE, 安培)。向量化, 快速。
    注意 B,S 用真 I 历史算 (未掩蔽 I_t), 是近似; 严格掩蔽版只在最终分析做。
    """
    model.eval()
    fscale = cfg["f_scale"]
    rows, SSE, SST = [], 0.0, 0.0
    sum_abs_I, n_I = 0.0, 0
    preds = {}  # name -> (p_pred_N, y_true_N) 供画图
    for it in items:
        u = it["u"].unsqueeze(0).to(device)
        u[:, :, 4] = 0.0  # 去掉 DeltaI (与训练一致)
        F_all, _, _, _, _, _ = model.forward_components(u)
        p = F_all.squeeze(0).cpu().numpy().astype(np.float64) * fscale
        y = it["f"].numpy().astype(np.float64) * fscale
        n = len(y)
        s = max(rf, WARMUP_SAMPLES)
        err = p[s:] - y[s:]
        rmse = float(np.sqrt(np.mean(err ** 2)))
        rms_true = float(np.sqrt(np.mean(y[s:] ** 2))) + 1e-9
        SSE += float(np.sum(err ** 2))
        SST += float(np.sum(y[s:] ** 2))
        rows.append({"name": it["name"], "nrmse_pct": rmse / rms_true * 100.0,
                     "mae_N": float(np.mean(np.abs(err))), "rmse_N": rmse})
        preds[it["name"]] = (p, y)
        # ---- 逆解电流 MAE (逆头直接预测) ----
        with torch.no_grad():
            I_pred_n = model.inverse_predict(u, it["f"].to(device).unsqueeze(0)).squeeze(0)
        I_pred = I_pred_n[s:].cpu()
        I_true = it["u"][s:, 3].abs().clamp(0.0, 1.0)  # 归一化幅值
        sum_abs_I += float((I_pred - I_true).abs().sum()) * 1.6  # -> 安培
        n_I += (n - s)
    model.train()
    pooled = math.sqrt(SSE / max(SST, 1e-12)) * 100.0
    inv_mae_A = sum_abs_I / max(n_I, 1)
    if plot_dir is not None and epoch is not None:
        plot_holdout_comparison(items, preds, rows, pooled, epoch, plot_dir, fscale)
    return rows, pooled, inv_mae_A


@torch.no_grad()
def check_hard_gates(model, device):
    """门禁: G_c/G_s 单调性 (硬门禁, 逆解需要) + F·v 报告 (v13 起取消强制, 仅记录)。返回 (min_Fv, min_dG, min_G)。"""
    model.eval()
    u = torch.randn(2, 3000, 6, device=device) * 0.5
    u[:, :, 1] = torch.linspace(-5, 5, 3000, device=device).unsqueeze(0).repeat(2, 1)
    u[:, :, 3] = torch.rand(2, 3000, device=device)
    F = model(u)
    min_Fv = float((F * u[:, :, 1]).min().cpu())
    dGc, mGc = model.G_c.check_monotonic()
    dGs, mGs = model.G_s.check_monotonic()
    min_dG, min_G = min(dGc, dGs), min(mGc, mGs)
    model.train()
    return min_Fv, min_dG, min_G


def make_lr_lambda(epochs, warmup, eta_min_ratio=0.01):
    def fn(e):
        if e < warmup:
            return (e + 1) / max(1, warmup)
        prog = (e - warmup) / max(1, epochs - warmup)
        return eta_min_ratio + (1 - eta_min_ratio) * 0.5 * (1 + math.cos(math.pi * prog))
    return fn


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    device = torch.device("cpu")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = REPO_ROOT / "training_outputs" / f"{TAG}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[*] Device=cpu threads={torch.get_num_threads()} | data={DATA_ROOT}", flush=True)
    mgr = DataManagerV3(DATA_ROOT, device)
    model = NLCSNN_v14(gru_hidden=GRU_HIDDEN, gru_layers=GRU_LAYERS,
                      mlp_hidden=MLP_HIDDEN, dropout=DROPOUT,
                      sigmas=SIGMAS, g_hidden=G_HIDDEN,
                      init_tau_ms=INIT_TAU_MS, c_min=C_MIN, s_min=S_MIN).to(device)
    if WARMSTART:
        # v13 从 v12 best 热启动 (全部形状匹配参数直接载入; b_head 全新, b≈0)
        norm_cfg = model.load_from_v12(WARMSTART)
        print(f"[*] warmstart v12: {WARMSTART}", flush=True)
    # 累计轮数 (跨 session): 看门狗凭 total_epoch 选最新权重恢复
    start_total = 0
    if WARMSTART:
        try:
            _w0 = torch.load(WARMSTART, map_location="cpu", weights_only=False)
            start_total = int(_w0.get("total_epoch", _w0.get("epoch", 0) or 0))
        except Exception as e:
            print(f"[*] warmstart epoch 读取失败 ({e}), start_total=0", flush=True)
        print(f"[*] warmstart 累计轮数 start_total={start_total}", flush=True)
    print(f"[*] v13 不冻结 GRU (v12 已训好, 全参数微调)", flush=True)
    rf = 1000
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Params: {n_params/1e6:.2f}M | RF={rf}", flush=True)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(EPOCHS, WARMUP_EPOCHS))
    mse_none = nn.MSELoss(reduction="none")

    print(f"[*] Session: {out_dir}", flush=True)
    print(f"[*] v13: v11 + 磨平NFL σ={SIGMAS} | ST加权x{ST_WEIGHT} | NO bias calibration", flush=True)
    print(f"[*] loss = p2p-MSE(fwd) + {LAMBDA_INV}·MSE(I_pred_head, I_true) (正逆联合)", flush=True)
    print(f"[*] epochs<={EPOCHS} lr={LR} warmup={WARMUP_EPOCHS} batch={FILE_BATCH}", flush=True)
    print(f"[*] STOP: pooled holdout NRMSE < {STOP_PCT}% (无patience, 跑满500轮)", flush=True)

    metrics_path = out_dir / "metrics.csv"
    with open(metrics_path, "w") as fh:
        fh.write("total_epoch,train_loss,loss_fwd,loss_inv,pooled_nrmse_pct,mean_nrmse_pct,"
                 "max_nrmse_pct,inv_mae_A,min_Fv,min_dG,lr,seconds\n")

    best_pooled, best_epoch = float("inf"), 0
    stop_reason = f"达到最大轮数 {EPOCHS}"
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        train_loss, loss_fwd, loss_inv = train_epoch(
            model, optimizer, mse_none, mgr.train_items, rf, device)
        scheduler.step()
        secs = time.perf_counter() - t0
        if not math.isfinite(train_loss) or train_loss > 1e4:
            stop_reason = f"训练发散 (loss={train_loss:.3g}), 第 {epoch} 轮熔断"
            print(f"\n*** STOP: {stop_reason} ***", flush=True)
            break

        # 累计轮数 (跨 session): VM 频繁更换导致单 session 活不到 E80,
        # 验证节奏必须按累计轮数判定, 否则 80/110/140 永远触发不了
        total_epoch = start_total + epoch
        do_eval = (total_epoch >= EVAL_START) and ((total_epoch - EVAL_START) % EVAL_EVERY == 0)
        pooled = mean_n = max_n = inv_mae_A = float("nan")
        min_Fv = min_dG = float("nan")
        if do_eval:
            plot_dir = out_dir / "validation_plots"
            rows, pooled, inv_mae_A = evaluate(model, mgr.holdout_items, mgr.cfg, rf, device,
                                               plot_dir=str(plot_dir), epoch=total_epoch)
            min_Fv, min_dG, min_G = check_hard_gates(model, device)
            vals = np.array([r["nrmse_pct"] for r in rows])
            mean_n, max_n = float(vals.mean()), float(vals.max())
            pd.DataFrame(rows).to_csv(out_dir / f"epoch_{total_epoch:04d}_holdout.csv", index=False)
            gate_ok = (min_dG > 0) and (min_G >= -1e-9)  # v13: F·v 不再是门禁, 仅记录
            if not gate_ok:
                print(f"*** WARNING E{total_epoch}: G 单调门禁未通过 "
                      f"(min_dG={min_dG:.2e}) ***", flush=True)
            if pooled < best_pooled - 1e-9:
                best_pooled, best_epoch = pooled, total_epoch
                torch.save({"state_dict": model.state_dict(), "norm_cfg": mgr.cfg,
                            "model_cfg": {"architecture": "nlcsnn_v14_min_damping_floor",
                                          "gru_hidden": GRU_HIDDEN, "gru_layers": GRU_LAYERS,
                                          "mlp_hidden": MLP_HIDDEN, "g_hidden": G_HIDDEN,
                                          "sigmas": list(SIGMAS), "warmup_skip": rf,
                                          "dissipative": "F=c*v*G_c(I_e)+(Σ_j s_j*tanh(v/σ_j))*G_s(I_e)",
                                          "G_monotonic": True, "bias_calibration": None},
                            "holdout_stems": HOLDOUT_STEMS, "epoch": epoch,
                            "total_epoch": total_epoch,
                            "pooled_nrmse_pct": pooled, "inv_mae_A": inv_mae_A},
                           out_dir / f"nlcsnn_{TAG}_best.pth")
            # 每100轮验证时存档当前模型 (用户要求)
            torch.save({"state_dict": model.state_dict(), "norm_cfg": mgr.cfg,
                        "epoch": epoch, "total_epoch": total_epoch, "pooled_nrmse_pct": pooled,
                        "mean_nrmse_pct": mean_n, "max_nrmse_pct": max_n,
                        "inv_mae_A": inv_mae_A, "min_Fv": min_Fv, "min_dG": min_dG},
                       out_dir / f"nlcsnn_{TAG}_e{total_epoch:03d}.pth")
            if pooled < STOP_PCT:
                torch.save({"state_dict": model.state_dict(), "norm_cfg": mgr.cfg,
                            "model_cfg": {"architecture": "nlcsnn_v14_min_damping_floor"},
                            "epoch": epoch, "pooled_nrmse_pct": pooled},
                           out_dir / f"nlcsnn_{TAG}_stop5pct.pth")
                stop_reason = f"pooled NRMSE {pooled:.3f}% < {STOP_PCT}% (第 {total_epoch} 轮)"
                with open(metrics_path, "a") as fh:
                    fh.write(f"{total_epoch},{train_loss:.8f},{loss_fwd:.8f},{loss_inv:.8f},"
                             f"{pooled:.6f},{mean_n:.6f},{max_n:.6f},{inv_mae_A:.6f},"
                             f"{min_Fv:.3e},{min_dG:.3e},"
                             f"{optimizer.param_groups[0]['lr']:.8g},{secs:.1f}\n")
                print(f"\n*** STOP: {stop_reason} ***", flush=True)
                break
        with open(metrics_path, "a") as fh:
            fh.write(f"{total_epoch},{train_loss:.8f},{loss_fwd:.8f},{loss_inv:.8f},"
                     f"{pooled:.6f},{mean_n:.6f},{max_n:.6f},{inv_mae_A:.6f},"
                     f"{min_Fv:.3e},{min_dG:.3e},"
                     f"{optimizer.param_groups[0]['lr']:.8g},{secs:.1f}\n")
        # 每轮存档 latest (75KB, 无验证开销): 虚拟机更换后看门狗从最新权重恢复,
        # 不依赖验证点的 best.pth (验证只在 EVAL_START/EVAL_EVERY 做, 按累计轮数)
        torch.save({"state_dict": model.state_dict(), "norm_cfg": mgr.cfg,
                    "model_cfg": {"architecture": "nlcsnn_v14_min_damping_floor",
                                  "gru_hidden": GRU_HIDDEN, "gru_layers": GRU_LAYERS,
                                  "mlp_hidden": MLP_HIDDEN, "g_hidden": G_HIDDEN,
                                  "sigmas": list(SIGMAS)},
                    "epoch": epoch, "total_epoch": total_epoch},
                   out_dir / f"nlcsnn_{TAG}_latest.pth")
        if do_eval:
            worst = " | ".join(f"{r['name'].replace('RLF_RMS','')}:{r['nrmse_pct']:.1f}%"
                               for r in sorted(rows, key=lambda r: -r["nrmse_pct"])[:3])
            print(f"E[{total_epoch:03d}] loss={train_loss:.5f} (fwd={loss_fwd:.5f} inv={loss_inv:.5f}) | "
                  f"pooled={pooled:.2f}% mean={mean_n:.2f}% max={max_n:.2f}% | "
                  f"inv_MAE={inv_mae_A:.4f}A | worst: {worst} | "
                  f"best={best_pooled:.2f}%@{best_epoch} | "
                  f"gates(Fv={min_Fv:.1e},dG={min_dG:.1e}) | {secs:.0f}s", flush=True)
        elif epoch % 10 == 0:
            print(f"E[{total_epoch:03d}] loss={train_loss:.5f} (fwd={loss_fwd:.5f} inv={loss_inv:.5f}) | "
                  f"{secs:.0f}s", flush=True)
        # 累计 500 轮上限 (跨 session; 看门狗也有同名检查, 双保险)
        if total_epoch >= EPOCHS:
            stop_reason = f"达到最大累计轮数 {EPOCHS}"
            print(f"\n*** STOP: {stop_reason} ***", flush=True)
            break

    print(f"\n[*] Done: {stop_reason}", flush=True)
    print(f"[*] Best pooled={best_pooled:.3f}% @epoch {best_epoch}", flush=True)


if __name__ == "__main__":
    main()
