import matplotlib.pyplot as plt
import numpy as np
import matplotlib.cm as cm

# 您的数据 (作为 Logits 值)
data = [-1.80, -2.81, 1.01, -2.70, 0.28,3.60, 0.45, 2.55, 0.36, 1.02]
num_bars = len(data)

# 为每个数据点创建 X 轴位置 (0, 1, 2, ..., 9)
x_positions = np.arange(num_bars)

# --- 柱子颜色修改部分 (保持不变) ---
# 1. 选择一个颜色映射
cmap = cm.get_cmap('tab10')

# 2. 从颜色映射中为每个柱子生成一个独特的颜色
colors = [cmap(i) for i in range(num_bars)]
# --- 结束颜色部分 ---

# --- 新增：为柱子定义纹理 (Hatch) ---
# 创建一个与柱子数量相同的纹理列表。
# 第 7 个柱子（索引 6）使用 '///' 纹理，其他柱子使用空字符串 '' (无纹理)。
hatches = [''] * num_bars
hatches[6] = '///'  # 为索引 6（即第 7 个柱子）设置纹理
# --- 结束纹理部分 ---

# 创建图形和轴对象
fig, ax = plt.subplots(figsize=(6, 5))

# 绘制柱状图
# 注意：当为 bar() 提供一个 hatch 列表时，它将分别应用于每个柱子
bars = ax.bar(x_positions, data, color=colors)

# 遍历 bars 列表，将预定义的纹理应用于每个柱子
for bar, hatch in zip(bars, hatches):
    bar.set_hatch(hatch)

# --- 轴标签修改部分 (保持不变) ---

# 1. 设置 X 轴标签为 "Label"
ax.set_xlabel('Label', fontsize=14)

# 2. 设置 Y 轴标签为 "Logits"
ax.set_ylabel('Logits', fontsize=14)

# 3. 设置 X 轴刻度标签为序号 (1, 2, 3, ..., 10)
ax.set_xticks(x_positions)
ax.set_xticklabels([str(i + 1) for i in x_positions])

# 4. 移除标题
ax.set_title('')

# 5. 保留零点水平线作为坐标参考
ax.axhline(0, color='black', linewidth=0.8)

# 自动调整布局
plt.tight_layout()

# 显示图形
plt.show()