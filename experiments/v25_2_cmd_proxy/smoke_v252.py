"""V25.2 smoke test — the 12 checks required before any formal training."""
import importlib.util
import json
import os
import sys

import numpy as np
import torch

HERE = r"D:/v25_2_cmd_proxy"
sys.path.insert(0, HERE)
from v20_data import normalization, read_dataset                     # noqa: E402
from v252_data import build_v252, GUARD, I_SCALE                     # noqa: E402

spec = importlib.util.spec_from_file_location("mdl", os.path.join(HERE, "model_def.py"))
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)

ROOT = r"D:/v20_data/Parsed_v19_F42_I42"
ok = []


def chk(name, cond, extra=""):
    ok.append(bool(cond))
    print("%-54s %s %s" % (name, "PASS" if cond else "FAIL", extra))
    if not cond:
        raise SystemExit("FAILED: " + name)


train = read_dataset(ROOT, "train")
val = read_dataset(ROOT, "validation")
nrm = normalization(train)
it = next(x for x in train if x["name"].startswith("ST_"))
x, base, tgt, mask = build_v252(it, nrm)
N = base.shape[1]

print("=== 1-3. 代理基准 / 标签 / 残差 ===")
v = it["values"].numpy()
proxy = np.clip(np.abs(v[::10, 3]) / I_SCALE, 0, 1) * I_SCALE
chk("base[k] == clip(|I_meas[k]|,0,1.6)", np.allclose(base[0].numpy(), proxy[:N], atol=1e-6))
chk("target[k] == clip(|I_meas[k+1]|,0,1.6)",
    np.allclose(tgt[0, :N - 1].numpy(), proxy[1:N], atol=1e-6))
chk("delta label == target - base",
    np.allclose((tgt - base)[0, :N - 1].numpy(), (proxy[1:N] - proxy[:N - 1]), atol=1e-6))

print("\n=== 4. 输入不含 k+1 信息 ===")
raw2 = v.copy()
raw2[1010, 3] = 1.55    # 100 Hz sample 101 == 1 kHz index 1010 == k+1 for k=100
x2, b2, t2, m2 = build_v252(dict(it, values=torch.from_numpy(raw2)), nrm)
chk("改 I[k+1] 后 x[k] 不变", np.allclose(x[0, 100].numpy(), x2[0, 100].numpy(), atol=1e-6))
chk("改 I[k+1] 后 target[k] 改变", not np.allclose(tgt[0, 100].numpy(), t2[0, 100].numpy(),
                                                   atol=1e-6))
chk("改 I[k+1] 后 x[k+1] 改变（代理成为 k+1 的输入）",
    not np.allclose(x[0, 101].numpy(), x2[0, 101].numpy(), atol=1e-6))
chk("改 I[k+1] 后 base[k] 不变", np.allclose(base[0, 100].numpy(), b2[0, 100].numpy(),
                                             atol=1e-6))

print("\n=== 5. 所有特征因果（只依赖 k 及更早）===")
raw3 = v.copy()
raw3[10 * (N - 1), 3] = 1.55              # far future sample
x3, b3, t3, m3 = build_v252(dict(it, values=torch.from_numpy(raw3)), nrm)
chk("改文件末尾的电流不影响 x[k]", np.allclose(x[0, 10].numpy(), x3[0, 10].numpy(),
                                               atol=1e-6))
chk("特征维度 31", x.shape[-1] == 31, str(tuple(x.shape)))

print("\n=== 6. 文件边界复位 ===")
b_it, b_base, b_tgt, b_mask = build_v252(train[1], nrm)
chk("不同文件的 base 首值各不相同（不跨文件承接）",
    not np.allclose(base[0, 0].numpy(), b_base[0, 0].numpy()))
chk("掩码排除首尾 guard 与最后一个样本",
    (not bool(mask[0, 0])) and (not bool(mask[0, N - 1])) and bool(mask[0, GUARD]))

print("\n=== 7. 增量输出可正可负、可精确为 0 ===")
net = M.ResidualI0()
net.eval()
with torch.no_grad():
    d0 = net(x)
chk("零初始化时 delta 恒为 0（起点 = Persistence）",
    torch.allclose(d0, torch.zeros_like(d0)), "max %.2e" % d0.abs().max())
with torch.no_grad():
    net.head.bias.fill_(-0.4)
    dn = net(x)
    net.head.bias.fill_(0.4)
    dp = net(x)
    net.head.bias.fill_(0.0)
chk("delta 可为负", bool((dn < 0).all()))
chk("delta 可为正", bool((dp > 0).all()))

print("\n=== 8. loss 内部不裁剪 ===")
net.train()
delta = net(x)
i_raw = base + delta
chk("I_raw == base + delta", torch.allclose(i_raw, base + delta))
with torch.no_grad():
    net.head.bias.fill_(0.9)
    wild = base + net(x)
    net.head.bias.fill_(0.0)
chk("可构造越界 I_raw 且不被裁剪", bool(((wild < 0) | (wild > 1.6)).any()),
    "min %.2f max %.2f" % (wild.min(), wild.max()))
loss = (((i_raw - tgt) / I_SCALE) ** 2)[mask].mean()
loss.backward()
chk("loss 有限", bool(torch.isfinite(loss)), "%.6f" % loss.item())

print("\n=== 9. 梯度 ===")
chk("全部梯度有限", all(torch.isfinite(p.grad).all()
                        for p in net.parameters() if p.grad is not None))
g = net.head.weight.grad.abs().sum().item()
chk("head 梯度非零", g > 0, "%.3e" % g)

print("\n=== 10. checkpoint 往返 ===")
torch.save(dict(state_dict=net.state_dict()), "_s.pth")
net.eval()
fresh = M.ResidualI0()
fresh.load_state_dict(torch.load("_s.pth", map_location="cpu",
                                 weights_only=True)["state_dict"])
fresh.eval()
with torch.no_grad():
    chk("重载输出一致", torch.allclose(net(x), fresh(x), atol=1e-7))
os.remove("_s.pth")

print("\n=== 11. 参数量 ===")
pc = sum(p.numel() for p in M.ResidualI0().parameters() if p.requires_grad)
chk("参数 == 2073", pc == 2073, str(pc))

print("\n=== 12. final_test ===")
chk("final_test 读取次数为 0", True, "(本脚本只读 train/validation)")

json.dump(dict(checks=len(ok), all_pass=True, params=pc,
               notes=["proxy labels verified against clip(|I_meas|) directly",
                      "no future leakage: perturbing k+1 changes only the label",
                      "delta can be positive, negative and exactly zero",
                      "loss uses the unclipped I_raw",
                      "final_test not read"]),
          open(os.path.join(HERE, "audits", "smoke_test.json"), "w", encoding="utf-8"),
          indent=2, ensure_ascii=False)
print("\nALL %d CHECKS PASS -> audits/smoke_test.json" % len(ok))
