#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2.5 基线检测器 (baselines.py)
=============================
与 2.4 的「星历→残差→CUSUM」方法对比的基线, 全部作用在**原始观测**上
(逐节点丢包计数), 不做「期望基线/残差分解」:

  - Naive threshold   : 朴素丢包率阈值 (总丢包 > τ 即报警)
  - Trust-based       : 行为信任 (BiTrust/RPLAD3 类), 成功/尝试比 < τ 即报警
  - CLIF (Mahalanobis): 跨层特征向量 + 马氏距离无监督异常 (拟合干净数据)

接口统一: 输入 dict[node] -> np.array[slot] (原始丢包计数), 返回 per-node 报警.
"""

import numpy as np
from collections import defaultdict


# ----------------------------------------------------------------------
# 观测聚合: residuals_*.csv → 逐节点原始字段时间序列
# ----------------------------------------------------------------------
def aggregate_node_field(df, field):
    """按接收卫星 src 聚合某字段 (dropped / rx / util / residual / handover_events)."""
    out = defaultdict(lambda: defaultdict(float))
    slots = set()
    for r in df.itertuples(index=False):
        node = int(r.src)
        slot = int(r.slot)
        out[node][slot] += float(getattr(r, field))
        slots.add(slot)
    nslots = (max(slots) + 1) if slots else 0
    series = {}
    for node in sorted(out):
        arr = np.zeros(nslots, dtype=float)
        for slot, v in out[node].items():
            arr[slot] = v
        series[node] = arr
    return series


# ----------------------------------------------------------------------
# 1) 朴素阈值
# ----------------------------------------------------------------------
def naive_threshold(series, tau):
    """每节点总丢包 > tau 即报警. tau 由干净(无攻击)数据标定."""
    return {n: bool(series[n].sum() > tau) for n in series}


# ----------------------------------------------------------------------
# 2) 信任类 (BiTrust / RPLAD3 行为信任)
# ----------------------------------------------------------------------
def trust_detect(series, rx_series, window, tau, min_attempts=1):
    """信誉为「窗口内成功转发比」:  trust = rx/(rx+dropped).
    任一滑窗 trust < tau 即报警 (信誉跌落).

    min_attempts: 窗口内总尝试次数(收+丢)不足该值时不评估 (避免冷启动 / 无流量
    节点因单次丢包被误判为 0% 信誉)."""
    verdict = {}
    for n in series:
        d = series[n]
        r = rx_series.get(n, np.zeros_like(d))
        alarmed = False
        total = r + d
        # 滑窗累加 (窗口内累计成功/总尝试)
        for t in range(len(d) - window + 1):
            rxw = r[t:t + window].sum()
            totalw = total[t:t + window].sum()
            if totalw >= min_attempts and rxw / totalw < tau:
                alarmed = True
                break
        verdict[n] = alarmed
    return verdict


# ----------------------------------------------------------------------
# 3) CLIF 跨层马氏距离 (无监督异常)
# ----------------------------------------------------------------------
def clif_features(df, node_series=None):
    """逐节点跨层特征向量: [mean_drop, mean_rx, mean_util, mean_handover, max_drop].

    返回 (nodes: list, X: ndarray). 特征源自跨层观测 (网络层丢包 + 链路利用率
    + 交换事件), 对应 CLIF 的 cross-layer fingerprint 思想 (不含星历残差分解)."""
    drop = aggregate_node_field(df, "dropped")
    rx = aggregate_node_field(df, "rx")
    util = aggregate_node_field(df, "util")
    ho = aggregate_node_field(df, "handover_events")
    nodes = sorted(drop)
    feats = []
    for n in nodes:
        feats.append([
            float(drop[n].mean()),
            float(rx[n].mean()),
            float(util[n].mean()),
            float(ho[n].mean()),
            float(drop[n].max()),
        ])
    return nodes, np.array(feats, dtype=float)


def mahalanobis(X, mu, cov_inv):
    d = X - mu
    return np.sqrt(np.einsum("ij,jk,ik->i", d, cov_inv, d))


def clif_detect(X, mu, cov_inv, threshold):
    """对特征矩阵 X 逐节点算马氏距离, > threshold 即报警."""
    dist = mahalanobis(X, mu, cov_inv)
    return dist > threshold


def fit_gaussian(X, reg=1e-6):
    """拟合多元高斯 (加正则保证协方差可逆)."""
    mu = X.mean(axis=0)
    cov = np.cov(X, rowvar=False)
    cov += np.eye(cov.shape[0]) * reg
    return mu, np.linalg.inv(cov)


def threshold_from_quantile(dist_clean, q=0.99):
    """用干净数据马氏距离的经验分位数定异常阈值."""
    return np.quantile(dist_clean, q)