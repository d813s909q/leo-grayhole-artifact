#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2.6 学习类基线: AE 无监督时序重构 (D-S+VAE 家族的无监督重构范式)
================================================================
与 2.4 CUSUM / 2.5 三基线对比的「学习类」基线, 用独立显卡 (PyTorch CUDA)。

方法: 逐节点时序 (特征 dropped/rx/util/handover_events) 滑窗 → MLP 自编码器
      在「干净(baseline)」数据上学习正常模式的重构, 测试用重构误差做异常分数,
      超过干净数据阈值即报警并归因到该节点。

公平性: 与 2.5 三基线一致, 只使用「原始观测」, 不做星历/期望基线/残差分解
        —— 用于证明「无星历基线的学习类方法」在自然拥塞下会失效, 凸显我方
        「残差/基线感知 CUSUM」的排他优势。
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

FEATURES = ["dropped", "rx", "util", "handover_events"]


def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------------------------------------------------
# 数据: 残差 CSV → 逐节点时序 (nslots, F)
# ----------------------------------------------------------------------
def node_series(path):
    """按接收卫星 src 聚合原始观测, 返回 dict[node] -> (nslots, F).

    多条 ISL 链路的字段按 (src,slot) 求和 (与 2.5 baselines.aggregate_node_field 一致),
    避免只保留末条链路而丢失攻击丢包."""
    df = pd.read_csv(path)
    grp = df.groupby(["src", "slot"])[FEATURES].sum().reset_index()
    nslots = int(df["slot"].max()) + 1
    series = {}
    for n in sorted(df["src"].unique()):
        arr = np.zeros((nslots, len(FEATURES)), dtype=np.float32)
        sub = grp[grp["src"] == n]
        arr[np.asarray(sub["slot"], dtype=int)] = sub[FEATURES].astype(np.float32).values
        series[int(n)] = arr
    return series


# ----------------------------------------------------------------------
# 窗口化
# ----------------------------------------------------------------------
def make_windows(series, W, stride):
    X, meta = [], []
    for n in sorted(series):
        arr = series[n]
        for t in range(0, arr.shape[0] - W + 1, stride):
            X.append(arr[t:t + W])
            meta.append((n, t))
    if not X:
        return np.zeros((0, W, len(FEATURES)), dtype=np.float32), []
    return np.asarray(X, dtype=np.float32), meta


def fit_scaler(X):
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(0)
    std = flat.std(0)
    std[std < 1e-6] = 1.0
    return mean, std


def apply_scaler(X, mean, std):
    return (X - mean) / std


# ----------------------------------------------------------------------
# 模型
# ----------------------------------------------------------------------
class AutoEncoder(nn.Module):
    def __init__(self, in_dim, hidden, z_dim):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, z_dim))
        self.dec = nn.Sequential(
            nn.Linear(z_dim, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, hidden), nn.ReLU(),
            nn.Linear(hidden, in_dim))

    def forward(self, x):
        return self.dec(self.enc(x))


def train_model(Xf, in_dim, hidden=64, z_dim=8, epochs=300, lr=1e-3, batch=256, seed=0):
    """Xf: (N, in_dim) 已标准化并展平后的训练样本 (与推理口径一致)."""
    set_seed(seed)
    dev = device()
    model = AutoEncoder(in_dim, hidden, z_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xf = torch.tensor(Xf)
    dl = DataLoader(TensorDataset(Xf), batch_size=batch, shuffle=True)
    model.train()
    for ep in range(epochs):
        tot = 0.0
        for (xb,) in dl:
            xb = xb.to(dev)
            rec = model(xb)
            loss = F.mse_loss(rec, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        if (ep + 1) % 50 == 0:
            print(f"  [train] epoch {ep+1}/{epochs}  mse={tot/len(Xf):.5f}")
    model.eval()
    return model


# ----------------------------------------------------------------------
# 推理: 逐节点异常分数 / 判决
# ----------------------------------------------------------------------
def _err(model, Xs, dev):
    Xt = torch.tensor(Xs)
    with torch.no_grad():
        rec = model(Xt.to(dev))
        err = ((rec.cpu() - Xt) ** 2).sum(1).numpy()
    return err


def node_scores(series, model, mean, std, W, stride):
    X, meta = make_windows(series, W, stride)
    out = {n: 0.0 for n in series}
    if len(X) == 0:
        return out
    Xs = apply_scaler(X, mean, std).reshape(len(X), -1)
    err = _err(model, Xs, device())
    for (n, _), e in zip(meta, err):
        if e > out[n]:
            out[n] = float(e)
    return out


def detect(series, model, mean, std, W, stride, threshold):
    """per-node (alarmed, first_alarm_slot)."""
    X, meta = make_windows(series, W, stride)
    verdict = {n: (False, None) for n in series}
    if len(X) == 0:
        return verdict
    Xs = apply_scaler(X, mean, std).reshape(len(X), -1)
    err = _err(model, Xs, device())
    for (n, t), e in zip(meta, err):
        if e > threshold:
            cur = verdict[n][1]
            if cur is None or t < cur:
                verdict[n] = (True, t)
    return verdict


def threshold_from_scores(scores, margin=1e-6):
    return max(scores.values()) + margin