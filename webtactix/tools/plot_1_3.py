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
sequential = [30.2, 28.6, 32.8, 36.6, 35.1, 20.8, 37.5]
prev_obs_preprocess   = [28.4, 27.1, 30.2, 30.2, 30.4, 19.5, 32.4]
prev_memory_mgmt      = [31.4, 29.2, 30.6, 34.9, 31.6, 20.3, 34.8]
prev_parallel_ps      = [39.4, 39.8, 37.3, 40.2, 39.9, 23.6, 40.2]
prev_reflex           = [39.5, 40.5, 37.8, 42.3, 40,  24, 41.6]
prev_reselect         = [39.6, 40.5, 37.9, 42.3, 40.1, 24.1, 41.7]

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
ax.set_ylabel('Tokens / Task (k)', fontsize=12, fontweight='bold')
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
ax.set_aspect(0.013)
plt.ylim([0, 50])
plt.tight_layout()
plt.savefig("pic1_3.png", dpi=300, bbox_inches="tight")
plt.show()
