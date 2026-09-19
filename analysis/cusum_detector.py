#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2.4 CUSUM 序贯变点检测 (灰洞检测核心)
======================================

论文方法链条 (对应贡献 C2, RQ2/RQ2'/RQ3'):
    2.3 产出: 残差 r_{l,t} = d_{l,t} - b_{l,t}    (逐链路 l 逐时隙 t)
    2.4 本模块: 在「节点聚合残差」上跑泊松似然比 CUSUM
                → 在线报警 + 归因定位到恶意节点

统计模型 (丢包 = 计数, 用泊松似然比):
    受控 H0 (无攻击):  X_t ~ Poisson(lambda0)            lambda0 = 自然丢包率
    攻击 H1:           X_t ~ Poisson(lambda1), lambda1 = lambda0 + delta
    单步对数似然比:    Z_t = X_t * ln(lambda1/lambda0) - (lambda1 - lambda0)
    递推统计量:        S_t = max(0, S_{t-1} + Z_t)
    判定:              S_t > h  →  报警

可证明性来源 (见 1.3 检测理论综述):
    - ARL₀ (受控平均运行长度, 误报前平均时隙数): 由阈值 h 唯一决定, 可用
      蒙特卡洛 + 马尔可夫链数值法精确求解;  给定目标误报率 → 反解 h
    - ARL₁ (攻击下平均检测延迟): Lorden 极小极大意义下 CUSUM 渐近最优
    - 序贯概率比检验 (SPRT) 令 CUSUM 具备自启动/自复位, 无固定样本量

用法:
    python3 cusum_detector.py  # 自带 demo (纯理论 ARL 曲线)
    评估脚本见 run_cusum.py
"""

import math
from collections import defaultdict

import numpy as np
import pandas as pd


# ======================================================================
# 节点聚合: 逐链路残差 → 逐节点接收残差时间序列
# ======================================================================
def load_residuals(path):
    """读 residuals_<run>.csv → DataFrame (列与 baseline_model.py 输出一致)."""
    return pd.read_csv(path)


def aggregate_nodes(df):
    """按接收卫星 (src) 聚合残差, 返回 dict[node] -> np.array[slot].

    残差表 src,dst 是接收视角: src=接收设备所在卫星(丢包发生地), dst=对端.
    恶意卫星丢弃「经它转发」的包 → 其接收端残差 > 0 → 按 src 聚合即可定位.
    """
    out = defaultdict(lambda: defaultdict(float))
    slots = set()
    for r in df.itertuples(index=False):
        node = int(r.src)
        slot = int(r.slot)
        out[node][slot] += float(r.residual)
        slots.add(slot)
    nslots = (max(slots) + 1) if slots else 0
    series = {}
    for node in sorted(out):
        arr = np.zeros(nslots, dtype=float)
        for slot, v in out[node].items():
            arr[slot] = v
        series[node] = arr
    return series


def aggregate_links(df):
    """按链路 (src,dst) 聚合残差, 返回 dict[(src,dst)] -> np.array[slot]."""
    out = defaultdict(lambda: defaultdict(float))
    slots = set()
    for r in df.itertuples(index=False):
        key = (int(r.src), int(r.dst))
        slot = int(r.slot)
        out[key][slot] += float(r.residual)
        slots.add(slot)
    nslots = (max(slots) + 1) if slots else 0
    series = {}
    for key in sorted(out):
        arr = np.zeros(nslots, dtype=float)
        for slot, v in out[key].items():
            arr[slot] = v
        series[key] = arr
    return series


def handover_mask(df, by="node"):
    """返回切换窗口掩码: 哪些 (node,slot) 落入切换窗口 (handover_events>0).

    by='node' → dict[node]->np.bool array;  by='link' → dict[(s,d)]->bool array.
    """
    if by == "node":
        agg = defaultdict(lambda: defaultdict(int))
        slots = set()
        for r in df.itertuples(index=False):
            agg[int(r.src)][int(r.slot)] = max(agg[int(r.src)][int(r.slot)],
                                               int(r.handover_events))
            slots.add(int(r.slot))
        nslots = (max(slots) + 1) if slots else 0
        mask = {n: np.zeros(nslots, dtype=bool) for n in agg}
        for n in agg:
            for s, v in agg[n].items():
                mask[n][s] = v > 0
        return mask
    else:
        agg = defaultdict(lambda: defaultdict(int))
        slots = set()
        for r in df.itertuples(index=False):
            k = (int(r.src), int(r.dst))
            agg[k][int(r.slot)] = max(agg[k][int(r.slot)], int(r.handover_events))
            slots.add(int(r.slot))
        nslots = (max(slots) + 1) if slots else 0
        mask = {k: np.zeros(nslots, dtype=bool) for k in agg}
        for k in agg:
            for s, v in agg[k].items():
                mask[k][s] = v > 0
        return mask


# ======================================================================
# 泊松似然比 CUSUM
# ======================================================================
class PoissonCUSUM:
    """丢包计数上的单侧泊松对数似然比 CUSUM.

    参数:
      lambda0          : 受控(自然)丢包率, 必须 > 0 (每个节点每时隙)
      delta            : 攻击增量 (lambda1 = lambda0 + delta)
      lambda0_schedule : 可选的 callable(t)->lambda0, 支持时变自然丢包率
                         (用于「切换感知」: 切换窗口 lambda0 更高)
    """

    def __init__(self, lambda0, delta, lambda0_schedule=None):
        if lambda0 <= 0:
            raise ValueError("lambda0 必须 > 0 (自然丢包率)")
        self.lambda0 = float(lambda0)
        self.delta = float(delta)
        self.lambda1 = self.lambda0 + self.delta
        self.lambda0_schedule = lambda0_schedule
        self.S = 0.0

    def reset(self):
        self.S = 0.0

    def _l0(self, t=None):
        if self.lambda0_schedule is None:
            return self.lambda0
        v = self.lambda0_schedule(t)
        return v if v and v > 0 else self.lambda0

    def increment(self, x, t=None):
        """输入一个观测计数 x, 返回更新后的 S_t."""
        l0 = self._l0(t)
        l1 = l0 + self.delta
        z = x * math.log(l1 / l0) - (l1 - l0)
        self.S = max(0.0, self.S + z)
        return self.S


def run_cusum(series, lambda0, delta, h, lambda0_schedule=None):
    """对时间序列跑 CUSUM, 返回 (S_seq: list, alarm: 首次报警时隙或 None)."""
    c = PoissonCUSUM(lambda0, delta, lambda0_schedule)
    S_seq = []
    alarm = None
    for t, x in enumerate(series):
        S = c.increment(x, t)
        S_seq.append(S)
        if S > h and alarm is None:
            alarm = t
    return S_seq, alarm


# ======================================================================
# ARL 蒙特卡洛估计 (仿真侧验证)
# ======================================================================
def mc_arl(lambda0, delta, h, n_reps=2000, max_len=200000, seed=0,
           attacking=False):
    """蒙特卡洛估 ARL.

    受控(attacking=False): X~Poisson(lambda0) → ARL₀
    攻击(attacking=True) : X~Poisson(lambda0+delta) → ARL₁ (全程攻击, 延迟)
    """
    rng = np.random.default_rng(seed)
    rate = (lambda0 + delta) if attacking else lambda0
    run_lengths = []
    censored = 0
    for _ in range(n_reps):
        c = PoissonCUSUM(lambda0, delta)
        t = 0
        alarmed = False
        while t < max_len:
            x = rng.poisson(rate)
            if c.increment(x) > h:
                alarmed = True
                break
            t += 1
        if alarmed:
            run_lengths.append(t)
        else:
            censored += 1
            run_lengths.append(max_len)
    arl = float(np.mean(run_lengths))
    return arl, censored


# ======================================================================
# ARL 马尔可夫链数值解 (可证明的「精确」理论界)
# ======================================================================
def mc_chain_arl(lambda0, delta, h, grid_step=0.1, x_max=40, attacking=False,
                 actual_delta=None):
    """用马尔可夫链 + 网格离散化数值求解 CUSUM 的 ARL (数值精确).

    将 CUSUM 统计量 S∈[0,h] 离散为网格 {0, g, 2g, ..., floor(h/g)*g},
    吸收态 = S'>h (精确统计量判据, 与 run_cusum 部署语义严格一致)。
    转移概率由泊松到达 X 决定 (截断到 x_max)。
    attacking=False → 受控率 λ0 → ARL₀;  True → 率 λ0+δ → ARL₁.
    平均吸收时间 = 1^T (I-Q)^{-1} 1 (Q 为瞬态子矩阵).

    actual_delta: 「失配攻击」支持 (M1.2/M1.4 小信号分析) — CUSUM 增量仍按
    设计增量 delta (设计 λ₁=λ0+delta) 计算, 但真实过程率 = λ0+actual_delta.
    省略时 actual_delta=delta (匹配情形). attacking=True 时生效.
    """
    from scipy.stats import poisson as sp_poisson
    # 网格状态 0..N (S 值 = i*grid_step), N = int(h/g)+1, 状态 N 视为吸收边界外
    N = int(h / grid_step) + 1
    n = N  # 瞬态状态数 0..N-1
    if n < 1:
        return float("inf"), 0

    l1 = lambda0 + delta
    logr = math.log(l1 / lambda0)
    if attacking:
        rate = lambda0 + (delta if actual_delta is None else actual_delta)
    else:
        rate = lambda0

    # 截断上限自适应: λ0 大时固定 x_max=40 会截掉全部泊松质量 (pmf 全 0 →
    # 归一出 NaN → 解溢出 → ARL=inf, 二分被污染收玫到下界)
    need = int(math.ceil(rate + 10.0 * math.sqrt(rate) + 20))
    if x_max < need:
        x_max = need
    xs = np.arange(0, x_max + 1)
    probs = sp_poisson.pmf(xs, rate)
    probs = probs / probs.sum()  # 归一 (截断尾部)

    z = xs * logr - (l1 - lambda0)
    Q = np.zeros((n, n), dtype=float)
    for i in range(n):
        si = i * grid_step
        new_val = np.maximum(0.0, si + z)
        # 吸收判据 = 部署判据 (R1-W1 修正): 先按精确统计量判 S' > h (严格大于),
        # 未吸收者才舍入落格。旧约定 (先 rint 落格再判越界) 在稀疏计数区
        # (λ0=0.01) 会把单包跳变 Z(1)=3.615 误舍到 3.60 网格而不吸收,
        # 使链解 ARL0 比部署检测器真实值高 ~24 倍。
        absorb = new_val > h
        stay = ~absorb
        if stay.any():
            j = np.rint(new_val[stay] / grid_step).astype(np.int64)
            j = np.minimum(j, n - 1)     # new<=h 保证舍入不越顶
            Q[i] += np.bincount(j, weights=probs[stay], minlength=n)
    # 吸收时间满足 (I-Q) r = 1, r[0] 即从状态 0 出发的平均运行长度。
    # 用 solve 直接解线性方程组, 比 inv+行和更数值稳定; 大 h 时矩阵接近奇异,
    # 解可能溢出为负/非有限, 此时返回 inf (单调上界, 保证二分不被污染)。
    try:
        r = np.linalg.solve(np.eye(n) - Q, np.ones(n))
    except np.linalg.LinAlgError:
        return float("inf"), n
    arl = float(r[0])
    if not np.isfinite(arl) or arl <= 0:
        return float("inf"), n
    return arl, n


# ======================================================================
# 阈值反解: 给定目标 ARL₀ → 求 h (二分, 用马尔可夫链数值解)
# ======================================================================
def choose_h_chain(lambda0, delta, target_arl, grid_step=0.05,
                   lo=0.1, hi=100.0, tol=0.05, max_iters=60):
    """二分求 h 使 ARL₀(h) ≈ target_arl (用数值精确解).

    max_iters: 二分迭代上限 (默认 60 与历史行为一致; 嵌套调用场景可调小)."""
    for _ in range(max_iters):
        mid = (lo + hi) / 2
        arl, _ = mc_chain_arl(lambda0, delta, mid, grid_step)
        if arl > target_arl:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


# ======================================================================
# 检测 + 归因 (真实数据)
# ======================================================================
def detect_nodes(series_map, lambda0, delta, h, lambda0_schedule=None):
    """对多个节点的时间序列并行跑 CUSUM, 返回每个节点的 (alarm, peak_S)."""
    results = {}
    for node, series in sorted(series_map.items()):
        S_seq, alarm = run_cusum(series, lambda0, delta, h, lambda0_schedule)
        peak_S = float(max(S_seq)) if S_seq else 0.0
        results[node] = {"alarm": alarm, "peak_S": peak_S, "nslots": len(series)}
    return results


def score_detection(results, true_malicious):
    """给定各节点检测结果与真实恶意节点集合, 返回混淆指标.

    返回 dict: tp, fp, tn, fn, tpr(检出率), fpr(误报率/节点), false_positives(list)
    """
    tp = fp = tn = fn = 0
    fp_list = []
    tp_list = []
    for node, r in results.items():
        is_attacker = node in true_malicious
        alarmed = r["alarm"] is not None
        if is_attacker and alarmed:
            tp += 1
            tp_list.append(node)
        elif is_attacker and not alarmed:
            fn += 1
        elif (not is_attacker) and alarmed:
            fp += 1
            fp_list.append(node)
        else:
            tn += 1
    n_pos = len(true_malicious)
    n_neg = len(results) - n_pos
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "tpr": (tp / n_pos) if n_pos else 1.0,
        "fpr": (fp / n_neg) if n_neg else 0.0,
        "true_detected": tp_list,
        "false_positives": fp_list,
    }


# ======================================================================
# demo / self-test
# ======================================================================
def _demo():
    print("=== ARL0 vs h (lambda0=0.5, delta=1.0): MC vs Markov-chain ===")
    s = "{:>6} {:>12} {:>14} {:>10}"
    print(s.format("h", "ARL0_MC", "ARL0_chain", "ARL1_MC"))
    for h in [3.0, 5.0, 8.0, 12.0, 18.0]:
        a0mc, _ = mc_arl(0.5, 1.0, h, n_reps=1000, seed=1)
        a0ch, _ = mc_chain_arl(0.5, 1.0, h)
        a1mc, _ = mc_arl(0.5, 1.0, h, n_reps=1000, seed=1, attacking=True)
        print(s.format(f"{h:.1f}", f"{a0mc:.1f}", f"{a0ch:.1f}", f"{a1mc:.2f}"))

    print("\n=== threshold for target ARL0=1000 (chain):",
          round(choose_h_chain(0.5, 1.0, 1000.0), 3), "===")


if __name__ == "__main__":
    _demo()