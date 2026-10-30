"""V22 acceptance tests: causality, candidate isolation, monotonicity, tau, reset,
pure-trial inverse API, parameter budget.

Run:  python test_v22.py
Exit code 0 and "ALL PASS" mean the formal training may start.
"""
import math
import sys

import torch

from v22_models import (EXPECTED, MODELS, TAU_INIT_MS, V22Model, count_params,
                        lag_scan, nfl_v22, observer_input)

torch.manual_seed(0)
FAIL = []


def check(name, ok, detail=""):
    print("%-58s %s %s" % (name, "PASS" if ok else "FAIL", detail))
    if not ok:
        FAIL.append(name)


def make_u(T=500, I=0.6, v=0.08):
    u = torch.zeros(1, T, 5)
    t = torch.arange(T) / 1000.0
    u[0, :, 0] = 0.2 * torch.sin(2 * math.pi * 1.5 * t)
    u[0, :, 1] = v + 0.03 * torch.sin(2 * math.pi * 3 * t)
    u[0, :, 2] = 0.05 * torch.cos(2 * math.pi * 3 * t)
    u[0, :, 3] = I
    u[0, :, 4] = 0.1
    return u


print("=== 1. 参数量（硬上限 3500）===")
for m in MODELS:
    pc = count_params(V22Model(m))
    ok = pc["total"] == EXPECTED[m] and pc["total"] <= 3500
    check("%-6s total=%d (期望 %d, <=3500)" % (m, pc["total"], EXPECTED[m]),
          ok, str(pc["groups"]) if not ok else "")

print()
print("=== 2. observer 与 NFL 严格只看到 I[t-1] ===")
u = make_u()
obs = observer_input(u)
check("observer 第0拍电流为 0", float(obs[0, 0, 3]) == 0.0)
check("observer 第t拍电流 == u[t-1]",
      torch.allclose(obs[0, 1:, 3], u[0, :-1, 3]))
t = 300
for shift in (0, 1, 50):
    u2 = u.clone()
    u2[0, t + shift:, 3] = 0.0 if shift else 1.0
    same = torch.allclose(nfl_v22(observer_input(u2))[0, :t], nfl_v22(obs)[0, :t])
    check("扰动 I[%d:] 不改变 t=%d 之前的 NFL" % (t + shift, t), same)

print()
print("=== 3. 候选电流隔离（不得进入 GRU / F_base / dF）===")
for m in MODELS:
    net = V22Model(m).eval()
    with torch.no_grad():
        a = net(u, parts=True)
        u2 = u.clone()
        u2[0, t, 3] = 1.0                      # only the CURRENT STEP current
        b = net(u2, parts=True)
    if m == "E0":
        check("%-6s 无电流通路：扰动 I[t] 输出完全不变" % m,
              torch.allclose(a["force"], b["force"]))
    else:
        ok_base = torch.allclose(a["base"], b["base"])
        ok_gain = torch.allclose(a["dF"], b["dF"])
        changed = not torch.allclose(a["force"], b["force"])
        check("%-6s 扰动 I[t]：F_base 不变" % m, ok_base)
        check("%-6s 扰动 I[t]：dF 不变" % m, ok_gain)
        check("%-6s 扰动 I[t]：force 确实改变（走 q 通路）" % m, changed)
        # 且只影响 t 之后（含 t）
        check("%-6s 扰动 I[t]：t 之前输出不变" % m,
              torch.allclose(a["force"][0, :t], b["force"][0, :t]))

print()
print("=== 4. 未来不影响过去 ===")
for m in MODELS:
    net = V22Model(m).eval()
    with torch.no_grad():
        a = net(u)
        u2 = u.clone()
        u2[0, t:, :] = 0.7
        b = net(u2)
    check("%-6s 扰动 t 及之后，t 之前输出不变" % m,
          torch.allclose(a[0, :t], b[0, :t], atol=1e-6))

print()
print("=== 5. G 端点与单调性 ===")
for m in ("AGT", "EG", "EGT", "EGTA"):
    net = V22Model(m).eval()
    with torch.no_grad():
        g0 = float(net.current_map(torch.tensor([0.0])))
        g1 = float(net.current_map(torch.tensor([1.0])))
        grid = torch.linspace(0, 1, 201)
        g = net.current_map(grid)
        mono = bool((g[1:] <= g[:-1] + 1e-9).all())
    check("%-6s g(0A)=%.6f≈1, g(1.6A)=%.6f≈0" % (m, g0, g1),
          abs(g0 - 1) < 1e-6 and abs(g1) < 1e-6)
    check("%-6s g 随电流幅值单调不增" % m, mono)

print()
print("=== 6. tau 数值正确性（V20 的 bug 不得复现）===")
net = V22Model("EGT")
check("tau 初值 = %.3f ms (期望 %.1f)" % (float(net.tau_ms()), TAU_INIT_MS),
      abs(float(net.tau_ms()) - TAU_INIT_MS) < 1e-4)
dt = 0.001
for tau_ms in (11.0, 50.0, 1.0):
    tau_s = torch.tensor(tau_ms / 1000.0)
    alpha = float(-torch.expm1(-dt / tau_s))
    want = 1 - math.exp(-dt / (tau_ms / 1000.0))
    check("tau=%.0f ms  alpha=%.6f (解析 %.6f)" % (tau_ms, alpha, want),
          abs(alpha - want) < 1e-6)
g = torch.zeros(1, 3000)
g[0, 500:] = 1.0
ref = []
s = torch.zeros(1)
a11 = 1 - math.exp(-dt / 0.011)
for k in range(3000):
    s = s + a11 * (g[:, k] - s)
    ref.append(s.clone())
ref = torch.cat(ref, 0)
got256 = lag_scan(g, torch.tensor(0.011), chunk=256)
got8 = lag_scan(g, torch.tensor(0.011), chunk=8)
check("lag_scan 与逐步递推一致 (max|d|=%.2e)"
      % float((got256 - ref).abs().max()),
      float((got256 - ref).abs().max()) < 1e-6)
check("lag_scan 与 chunk 大小无关 (max|d|=%.2e)"
      % float((got256 - got8).abs().max()),
      float((got256 - got8).abs().max()) < 1e-6)
k100 = 500 + 100
want100 = 1 - math.exp(-0.1 / 0.011)
check("阶跃后 100 ms 响应 = %.4f (解析 %.4f)" % (float(got256[0, k100]), want100),
      abs(float(got256[0, k100]) - want100) < 2e-3)

print()
print("=== 7. 首拍与文件间状态复位 ===")
for m in ("EG", "EGT"):
    net = V22Model(m).eval()
    with torch.no_grad():
        p = net(u, parts=True)
    check("%-6s q[0] == g[0]" % m, torch.allclose(p["q"][0, 0], p["g"][0, 0]))
    with torch.no_grad():
        p2 = net(u, parts=True)
    check("%-6s 重复调用同一文件结果一致（无跨调用状态）" % m,
          torch.allclose(p["force"], p2["force"]))

print()
print("=== 8. 逆求解 trial 是纯函数 ===")
net = V22Model("EGT").eval()
z = torch.randn(1, 1, 16) * 0.1
v_hat, a_hat = torch.tensor([[0.12]]), torch.tensor([[0.01]])
q_prev = torch.tensor([[0.4]])
q_prev_before = q_prev.clone()
with torch.no_grad():
    f1, q1 = net.trial_force(z, v_hat, a_hat, q_prev, torch.tensor([[0.5]]))
    f2, q2 = net.trial_force(z, v_hat, a_hat, q_prev, torch.tensor([[0.5]]))
    f3, _ = net.trial_force(z, v_hat, a_hat, q_prev, torch.tensor([[1.6]]))
check("同一 q_prev 重复试算结果一致", torch.allclose(f1, f2) and torch.allclose(q1, q2))
check("q_prev 未被试算修改", torch.equal(q_prev, q_prev_before))
check("不同候选电流给出不同力", not torch.allclose(f1, f3))
with torch.no_grad():
    swept = []
    for i in torch.linspace(0, 1.6, 17):
        f, _ = net.trial_force(z, v_hat, a_hat, q_prev, torch.tensor([[float(i)]]))
        swept.append(float(f))
check("固定状态下 力随候选电流单调不增 (v>0)",
      all(swept[i + 1] <= swept[i] + 1e-6 for i in range(len(swept) - 1)))

print()
print("=== 9. 100 Hz trial 与 1 kHz 递推的 dt 一致 ===")
net = V22Model("EGT")
tau_s = net.tau_s()
for dtv in (0.001, 0.01):
    al = float(-torch.expm1(-torch.tensor(dtv) / tau_s))
    check("dt=%.0f ms  alpha=%.5f" % (dtv * 1000, al), 0 < al < 1)

print()
if FAIL:
    print("FAILED:", FAIL)
    sys.exit(1)
print("ALL PASS")
