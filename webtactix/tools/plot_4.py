import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams.update({
    "font.size": 10,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
})

# -----------------------------
# 左图：SeeAct / Agent-E / Browser Use / WebTactix / OpenAI Operator
# -----------------------------
methods = [
    "SeeAct",
    "Agent-E",
    "Browser\nUse",
    "WebTactix",
    "OpenAI\nOperator",
]

# 示例数据（你可替换成真实值）
seeact_success, seeact_failure, seeact_eff = 11, 14, 2.0
agente_success, agente_failure, agente_eff = 6,  8,  1.0
browser_success, browser_failure, browser_eff = 7, 12, 1.1
operator_success, operator_failure, operator_eff = 19, 40, 2.6

# WebTactix 先填示例（你之后替换）
webt_success, webt_failure, webt_eff = 9, 11, 1.2

success_steps = np.array([seeact_success, agente_success, browser_success, webt_success, operator_success])
failure_steps = np.array([seeact_failure, agente_failure, browser_failure, webt_failure, operator_failure])
efficiency    = np.array([seeact_eff, agente_eff, browser_eff, webt_eff, operator_eff])

# -----------------------------
# 右图：错误类型（输入仍用百分比）
# -----------------------------
cats = [
    "Filter\nError",
    "Navigation\nError",
    "Incomplete\nStep",
    "Mis\nunderstanding",
    "Others",
]

# 百分比（来自你图里）
operator_pct  = np.array([57.7, 19.6, 6.2, 11.3, 5.2])
webtactix_pct = np.array([53.0, 12.0, 13.0, 12.0, 12.0])  # 示例，换你的统计

# ✅ 你需要填：总错误数（绝对数）
operator_total_errors  = 116  # TODO: 改成你的 Operator 总错误数
webtactix_total_errors = 140  # TODO: 改成你的 WebTactix 总错误数（若也要绝对数）

def pct_to_int_counts(pcts, total):
    """
    把百分比转换为整数计数，并保证 sum(counts)=total
    """
    raw = pcts / 100.0 * total
    base = np.floor(raw).astype(int)
    remainder = int(total - base.sum())
    if remainder > 0:
        frac = raw - np.floor(raw)
        order = np.argsort(frac)[::-1]
        for i in range(remainder):
            base[order[i]] += 1
    return base

operator_cnt  = pct_to_int_counts(operator_pct,  operator_total_errors)
webtactix_cnt = pct_to_int_counts(webtactix_pct, webtactix_total_errors)

# -----------------------------
# 配色
# -----------------------------
c_success = "#B9F2B0"   # 浅绿
c_failure = "#E9A0A0"   # 浅红
c_line    = "#1F7A7A"   # 青绿

cat_colors = {
    "Filter\nError":        "#4C78C8",  # 蓝
    "Navigation\nError":    "#7AC36A",  # 绿
    "Incomplete\nStep":     "#F28E2B",  # 橙
    "Mis\nunderstanding":   "#C45A5A",  # 红
    "Others":               "#7B6AAE",  # 紫
}

# -----------------------------
# 画图
# -----------------------------
fig = plt.figure(figsize=(13.5, 4.2), dpi=150)
gs = fig.add_gridspec(1, 2, width_ratios=[1.4, 1.4], wspace=0.35)

# ===== 左图 =====
ax = fig.add_subplot(gs[0, 0])
x = np.arange(len(methods))
w = 0.35

ax.bar(x - w/2, success_steps, width=w, color=c_success, label="Success", edgecolor="none")
ax.bar(x + w/2, failure_steps, width=w, color=c_failure, label="Failure", edgecolor="none")

ax.set_ylabel("Average Steps")
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylim(0, 52)
ax.set_yticks([0, 10, 20, 30, 40, 50])
ax.grid(axis="y", alpha=0.18, linewidth=0.8)

ax2 = ax.twinx()
ax.set_box_aspect(1/1.2)
ax2.set_box_aspect(1/1.2)

ax2.plot(x, efficiency, color=c_line, marker="o", markersize=3.5, linewidth=1.5, label="Efficiency")
ax2.set_ylabel("Efficiency")
ax2.set_ylim(0, 3.0)
ax2.set_yticks([1, 2, 3])

offsets = np.array([-0.18, 0.18, 0.18, 0.18, 0.18])
for xi, yi, off in zip(x, efficiency, offsets):
    ax2.text(xi, yi + off, f"{yi:.1f}", ha="center", va="center",
             fontsize=10, fontweight="bold", color="black")

h1, l1 = ax.get_legend_handles_labels()
h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="upper left", frameon=True)

# ===== 右图：绝对值 Count =====
axb = fig.add_subplot(gs[0, 1])
xc = np.arange(len(cats))
w2 = 0.34

for i, cat in enumerate(cats):
    color = cat_colors[cat]
    axb.bar(xc[i] - w2/2, webtactix_cnt[i], width=w2, color=color, alpha=0.45,
            edgecolor="black", linewidth=0.6, hatch="//")
    axb.bar(xc[i] + w2/2, operator_cnt[i],  width=w2, color=color, alpha=0.95,
            edgecolor="black", linewidth=0.6)

axb.set_ylabel("Count")
axb.set_xticks(xc)
axb.set_xticklabels(cats)

ymax = max(operator_cnt.max(), webtactix_cnt.max())
axb.set_ylim(0, ymax * 1.18)
axb.grid(axis="y", alpha=0.18, linewidth=0.8)

for i in range(len(cats)):
    axb.text(xc[i] - w2/2, webtactix_cnt[i] + 0.6, f"{webtactix_cnt[i]}",
             ha="center", va="bottom", fontsize=9)
    axb.text(xc[i] + w2/2, operator_cnt[i] + 0.6, f"{operator_cnt[i]}",
             ha="center", va="bottom", fontsize=9)

legend_patches = [
    Patch(facecolor="white", edgecolor="black", hatch="//", label="WebTactix"),
    Patch(facecolor="white", edgecolor="black", label="Operator"),
]
axb.legend(handles=legend_patches, loc="upper right", frameon=True)

plt.savefig("pic4.png", dpi=300, bbox_inches="tight")
plt.show()
