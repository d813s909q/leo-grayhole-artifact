#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
figure_studio.py — 论文图 2–5 绘图库
====================================
职责仅两件事：加载 analysis 数据 + draw() 绘制逻辑与配色。

- 不含 GUI 面板、不含任何导出（不再自动产出 3.5in PNG/SVG）。
- 交互窗口由 figures_native.py 打开（纯交互，保存走窗口工具栏 Save 按钮）。
- 无头自检由 headless_selftest.py 驱动。
- 后端由调用方在 import 本模块之前指定（TkAgg / Agg）。

图 1 为 TikZ 矢量图（latex/sections/fig1_method.tex），不在本库。
表 I–III 已直接内联于 latex 章节源码，不在本库。
"""
import json
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# IEEE 插图规范: 单栏 3.5in 印放所见即所得, 图内文字 8pt 无衬线, 刻度朝内
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 8,
    "axes.linewidth": 0.6,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "legend.fontsize": 6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

FIGDIR = os.path.dirname(os.path.abspath(__file__))                   # .../03-manuscript/figures
BASE = os.path.dirname(os.path.dirname(FIGDIR))                       # 项目根 (跨平台)
AN = os.path.join(BASE, "02-experiment", "analysis")

# ---------------- 数据 ----------------
with open(os.path.join(FIGDIR, "fig2_data.json")) as f:
    FIG2 = json.load(f)
ARL = pd.read_csv(os.path.join(AN, "cusum", "arl_curve.csv"))
SWEEP = pd.read_csv(os.path.join(AN, "comparison", "attack_sweep.csv"))
LEARN = pd.read_csv(os.path.join(AN, "learning", "learning_sweep.csv"))
SWEEP = SWEEP.merge(LEARN[["name", "node12_score"]], on="name", how="left")
CONG = pd.read_csv(os.path.join(AN, "learning", "learning_congestion.csv"))

PAL = dict(ink="#1c2333", red="#c0392b", blue="#1f6f8b", teal="#0f8a6a",
           gold="#b8860b", plum="#7d4a8d", gray="#8a8f9c")

# 图题一律由 LaTeX caption 提供, 图内不画标题 (IEEE 规范)
# 图例用短标签, 全称由 caption / 表格给出
METHODS = [("ours", "Ours", PAL["red"]),
           ("vae", "AE", PAL["blue"]),
           ("trust", "Trust", PAL["teal"]),
           ("naive", "Naive", PAL["gold"]),
           ("clif", "CLIF", PAL["plum"])]
MSTYLE = dict(ours="o", vae="D", trust="^", naive="s", clif="x")

# ---------------- 状态（交互默认值, IEEE 3.5in 单栏尺寸） ----------------
S = dict(view="fig2", font=8, lw=1.2, ms=3.5, grid=True,
         fig2_ann=True, fig2_honest=True, fig3_log=True,
         fig4_metric="dropped", fig7_panel="fpr")


def style_axes(ax):
    ax.grid(S["grid"], color="#d9d9d9", lw=0.4)
    ax.tick_params(colors=PAL["ink"])
    for sp in ax.spines.values():
        sp.set_color(PAL["ink"])


def draw(fig, ax):
    """按 S['view'] 绘制 fig2-5 之一（交互与自检共用）。"""
    ax.clear()
    f = S["font"]
    plt.rcParams.update({"font.size": f})
    v = S["view"]

    if v == "fig2":
        slots = FIG2["slots"]
        ax.step(slots, FIG2["n12"]["observed"], where="mid", color=PAL["ink"],
                lw=S["lw"], label="observed $d_{u,t}$")
        ax.step(slots, FIG2["n12"]["residual"], where="mid", color=PAL["red"],
                lw=S["lw"], ls="--", label="residual $r_{u,t}$")
        if S["fig2_honest"]:
            ax.plot(slots, np.zeros_like(slots), color=PAL["gray"], lw=1,
                    label="honest node (=0)")
        if S["fig2_ann"]:
            ax.annotate("slot 27 routing update\n144 pkts", xy=(27, 144),
                        xytext=(45, 126), fontsize=6,
                        arrowprops=dict(arrowstyle="->", color=PAL["ink"]))
            # 标注放峰值右下方空白区: 右上角留给图例, 避免互相遮挡
            ax.annotate("GSL handover window\n128 pkts (44%)", xy=(126, 113),
                        xytext=(140, 66), fontsize=6,
                        arrowprops=dict(arrowstyle="->", color=PAL["ink"]))
        ax.set_xlabel("time slot (1 s)", fontsize=f)
        ax.set_ylabel("packets (satellite 12)", fontsize=f)
        ax.legend(loc="best", framealpha=0.9, handlelength=1.4,
                  labelspacing=0.25, borderpad=0.35).set_draggable(True)

    elif v == "fig3":
        m = ARL["arl0_mc"].notna()
        ax.plot(ARL["h"], ARL["arl0_chain"], "-o", color=PAL["blue"], lw=S["lw"],
                ms=S["ms"], label=r"ARL$_0$ chain")
        ax.plot(ARL["h"][m], ARL["arl0_mc"][m], "o", mfc="none", mec=PAL["blue"],
                ms=S["ms"], label=r"ARL$_0$ MC")
        ax.plot(ARL["h"], ARL["arl1_chain"], "-s", color=PAL["red"], lw=S["lw"],
                ms=S["ms"], label=r"ARL$_1$ chain")
        m1 = ARL["arl1_mc"].notna()
        ax.plot(ARL["h"][m1], ARL["arl1_mc"][m1], "s", mfc="none", mec=PAL["red"],
                ms=S["ms"], label=r"ARL$_1$ MC")
        if S["fig3_log"]:
            ax.set_yscale("log")
        ax.set_xlabel("threshold $h$", fontsize=f)
        ax.set_ylabel("average run length (slots)", fontsize=f)
        ax.legend(loc="best", ncol=2, framealpha=0.9, handlelength=1.4,
                  columnspacing=0.8, labelspacing=0.3,
                  borderpad=0.4).set_draggable(True)

    elif v == "fig4":
        SHORT = {"const_0p1": "c.1", "const_0p2": "c.2", "const_0p3": "c.3",
                 "const_0p4": "c.4", "const_0p5": "c.5", "onoff_1p0_20s": "o",
                 "scan_1p0_20s": "s"}
        x = [SHORT.get(n, n) for n in SWEEP["name"]]
        if S["fig4_metric"] == "dropped":
            colors = [PAL["blue"] if m == "CONSTANT" else PAL["gold"] if m == "ON_OFF"
                      else PAL["teal"] for m in SWEEP["mode"]]
            ax.bar(x, SWEEP["total_dropped"], color=colors, alpha=0.85,
                   label="dropped (attack window)")
            for xi, r in zip(range(len(x)), SWEEP.itertuples()):
                ax.text(xi, r.total_dropped + 6, f"alarm@{int(r.alarm_slot)}",
                        ha="center", fontsize=6)
            ax.set_ylabel("packets dropped", fontsize=f)
            ax.set_ylim(0, 500)      # 最高柱 420 + alarm@ 标注, 刻度取整到 500
        else:
            w = 0.38
            ax.bar(np.arange(len(x)) - w / 2, SWEEP["attack_onset"], w,
                   color=PAL["gray"], label="attack onset")
            ax.bar(np.arange(len(x)) + w / 2, SWEEP["alarm_slot"], w,
                   color=PAL["red"], label="alarm slot")
            ax.set_xticks(range(len(x)))
            ax.set_xticklabels(x)
            ax.set_ylabel("slot number", fontsize=f)
        ax.set_xlabel("attack schedule (c = CONSTANT, o = ON_OFF, s = SCAN)",
                      fontsize=f)
        ax.legend(loc="best", ncol=2, framealpha=0.9, handlelength=1.4,
                  columnspacing=0.8, labelspacing=0.28, borderpad=0.35,
                  ).set_draggable(True)

    elif v == "fig7":
        key = "fpr" if S["fig7_panel"] == "fpr" else "tpr"
        for mk, lab, c in METHODS:
            ax.plot(CONG["lam_cong"], CONG[f"{mk}_{key}"], marker=MSTYLE[mk],
                    color=c, lw=S["lw"], ms=S["ms"], label=lab,
                    ls="none" if MSTYLE[mk] == "x" else "-")
        ax.set_xlabel(r"natural congestion $\lambda_{cong}$ (pkts/node-slot)", fontsize=f)
        ax.set_ylabel("false-positive rate" if key == "fpr" else "true-positive rate",
                      fontsize=f)
        # FPR 面板: 全部曲线 ≤1.0, 顶部 1.0–1.15 无数据 → 单行 5 列图例零遮挡
        if key == "fpr":
            ax.set_ylim(-0.03, 1.15)
            ax.legend(loc="upper center", ncol=5, fontsize=6, framealpha=0.9,
                      handlelength=1.4, columnspacing=0.7,
                      borderpad=0.35).set_draggable(True)
        else:
            ax.set_ylim(-0.03, 1.08)
            ax.legend(loc="best", ncol=5, fontsize=6, framealpha=0.9,
                      handlelength=1.4, columnspacing=0.7,
                      borderpad=0.35).set_draggable(True)

    else:
        raise ValueError(f"未知视图 {v!r} (本库仅负责 fig2-4/7; 图 5/6 由 figures_native 自绘)")

    style_axes(ax)
    fig.canvas.draw_idle()
