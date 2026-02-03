import matplotlib.pyplot as plt
import numpy as np

# 定义数据
labels = ['Overall', 'Reddit', 'GitLab', 'Shopping', 'CMS', 'Map']
methods = ['WebTactix', 'AgentOccam', 'WebOperator', 'ScribeAgent']

# 每个方法的得分
scores = {
    'WebTactix': [6.35, 5.25, 7.48, 5.6, 6.38, 5.3],
    'AgentOccam': [9, 8.6, 10.8, 6.7, 9.2, 8.5],
    'WebOperator': [19.4, 15.3, 22, 17.5, 18.9, 19.4],
    'ScribeAgent': [12, 11, 13, 11, 13.5, 11.6]
}


def create_radar_chart(ax, data_dict, labels, fill_method, colors, line_colors):
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    # 0~24 半径范围
    ax.set_ylim(0, 24)

    # 刻度：0, 8, 16, 24
    rticks = [0, 8, 16, 24]
    ax.set_rlabel_position(0)
    ax.set_rticks(rticks)
    ax.tick_params(axis='y', which='major', labelsize=10, colors='gray')

    # 画线/填充
    for i, (method, data) in enumerate(data_dict.items()):
        data = np.concatenate((data, [data[0]]))
        if method == fill_method:
            ax.fill(angles, data, color=colors[i], alpha=0.25)
        ax.plot(angles, data, color=line_colors[i], linewidth=2, label=method)

    ax.set_yticklabels([])  # 不显示默认径向刻度数字（只用交点处自己写的）
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontweight='bold', fontsize=10)

    # 交点处标数字：隔一个标一个（按索引：标 0 和 16）
    for i_ang in range(len(labels)):
        for k, j in enumerate(rticks):
            if k % 2 == 0:  # 0, 16
                ax.text(
                    angles[i_ang], j, str(j),
                    ha='center', va='center',
                    color='gray', fontsize=12
                )

    ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))



# 设置绘图
fig, axes = plt.subplots(1, 2, figsize=(12, 6), subplot_kw=dict(polar=True))

# 创建数据字典
data_dict = {
    'WebTactix': scores['WebTactix'],
    'AgentOccam': scores['AgentOccam'],
    'WebOperator': scores['WebOperator'],
    'ScribeAgent': scores['ScribeAgent']
}

# 定义颜色
colors = ['orange', 'gray', 'red', 'blue']
line_colors = ['darkorange', 'black', 'darkred', 'darkblue']

# 绘制雷达图，只有 WebTactix 填充其他没有填充
create_radar_chart(axes[0], data_dict, labels, 'WebTactix', colors, line_colors)
create_radar_chart(axes[1], data_dict, labels, 'WebTactix', colors, line_colors)

# 调整布局
plt.tight_layout()

# 保存为高分辨率图像
plt.savefig('pic2.png', dpi=300)

# 显示图表
plt.show()
