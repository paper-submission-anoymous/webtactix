import numpy as np
import matplotlib.pyplot as plt

# =========================
# 1) x 统一为 1..5
# =========================
max_parallel = np.array([1, 2, 3, 4, 5])

# =========================
# 2) 你的数据（替换这里）
# =========================
accuracy   = np.array([58, 68, 71, 70.5, 69])
token_cost = np.array([24.9, 33.8, 40.3, 46.0, 51.2])
avg_length = np.array([1, 1.55, 1.9, 2.2, 2.4])

# ★ 选点：accuracy 最大
best_idx = int(np.argmax(accuracy))

# =========================
# 3) 三图并列 + 每个子图正方形
# =========================
ncols = 3
side = 3.2  # 每个子图大约 side×side 英寸
fig, axes = plt.subplots(
    1, ncols,
    figsize=(side * ncols, side),
    dpi=150,
    constrained_layout=True
)

def style_square(ax):
    ax.set_box_aspect(1)      # 关键：让每个子图绘图区正方形
    ax.grid(True, alpha=0.3)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xlabel("Max Parallel")

# ---- Accuracy
ax = axes[0]
ax.plot(max_parallel, accuracy, linestyle='--', marker='o')
ax.plot(max_parallel[best_idx], accuracy[best_idx], marker='*', markersize=14)
ax.set_ylabel("Accuracy (%)")
style_square(ax)

# ---- Token Cost（确保低->高）
ax = axes[1]
ax.plot(max_parallel, token_cost, linestyle='--', marker='o')
ax.plot(max_parallel[best_idx], token_cost[best_idx], marker='*', markersize=14)
ax.set_ylabel("Token Cost (k)")

ymin, ymax = float(np.min(token_cost)), float(np.max(token_cost))
pad = (ymax - ymin) * 0.08 if ymax > ymin else 0.1
ax.set_ylim(ymin - pad, ymax + pad)   # 低->高（不反转）
style_square(ax)

# ---- Average Length
ax = axes[2]
ax.plot(max_parallel, avg_length, linestyle='--', marker='o')
ax.plot(max_parallel[best_idx], avg_length[best_idx], marker='*', markersize=14)

# 加一条参考斜线：y = x（从 1,1 到 5,5）
ax.plot(max_parallel, max_parallel, linestyle='-', linewidth=1.5)

ax.set_ylabel("Average Length")
style_square(ax)

# =========================
# 4) 保存
# =========================
plt.savefig("pic3.png", dpi=300, bbox_inches="tight")
plt.show()
