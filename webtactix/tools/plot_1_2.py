import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# 设置样式
sns.set(style="whitegrid")

# x 轴标签
labels = ['All', 'Reddit', 'Gitlab', 'Shopping', 'CMS', 'Map', 'Multisite']
x = np.arange(len(labels))

# =========================
# 1) 数据（把下面 6 组都替换成你的真实结果）
# 每组长度都要 = len(labels) = 7
# =========================
sequential = [9.2, 9.5, 9.8, 8.6, 9.1, 9, 9.4]

prev_obs_preprocess   = [8.4, 9, 9.2, 7, 7.4, 8.8, 8.4]
prev_memory_mgmt      = [8.6, 9.2, 9.6, 8, 8.3, 8.8, 8.8]
prev_parallel_ps      = [6, 5.8, 6.3, 5.9, 6.2, 6.1, 6.2]
prev_reflex           = [5.2, 5, 5.8, 5, 5.1, 5.3, 5.6]
prev_reselect         = [5.3, 5, 5.9, 5, 5.1, 5.4, 5.7]

series = [
    ("Sequential Search", sequential, "/",  "deepskyblue"),
    ("Previous + observation preprocess", prev_obs_preprocess, "\\", "mediumseagreen"),
    ("Previous + memory management", prev_memory_mgmt, ".", "lightgray"),
    ("Previous + Parallel Planning-Select", prev_parallel_ps, "x", "lightcoral"),
    ("Previous + Reflex", prev_reflex, "o", "khaki"),
    ("Previous + Reselect(Full WebTactix)", prev_reselect, "/", "plum"),
]

# =========================
# 2) 布局：6 组柱子 -> 自动计算 offset（居中对齐）
# =========================
n = len(series)
width = 0.11  # 6组建议 0.10~0.12
offsets = (np.arange(n) - (n - 1) / 2) * width

fig, ax = plt.subplots(figsize=(14, 6))

for i, (name, vals, hatch, color) in enumerate(series):
    ax.bar(
        x + offsets[i],
        vals,
        width,
        label=name,
        hatch=hatch,
        color=color,
        edgecolor="black",
        linewidth=1.5
    )

# 轴标签
ax.set_ylabel('Time (min)', fontsize=12, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(labels)

# 图例外置
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True)

# 坐标轴黑色加粗
for sp in ["top", "right", "left", "bottom"]:
    ax.spines[sp].set_color("black")
    ax.spines[sp].set_linewidth(1.5)

ax.tick_params(axis='both', which='major', length=6, width=1.5, colors='black')

# 图更扁（保持你原来的感觉）
ax.set_aspect(0.06)
plt.ylim([0, 10])
plt.tight_layout()
plt.savefig("pic1_2.png", dpi=300, bbox_inches="tight")
plt.show()
