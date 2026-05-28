import re
from collections import defaultdict

import os
from pathlib import Path

log_file = "E:\博士研究生\第三篇模型异构蒸馏\第1次-IJCAI-被拒\实验数据\Imagenet/30轮\image30.txt"

# 用于标记当前是否在【客户端测试】区域
in_test_block = False

# 正则：严格匹配「客户端测试」里的行
test_pattern = re.compile(
    r"Client\s+(\d+)\s+\|\s+Epoch\s+(\d+)\|\s+Acc:\s+([\d.]+),\s+ASR:\s+([\d.]+)"
)

acc_dict = defaultdict(list)
asr_dict = defaultdict(list)

with open(log_file, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()

        # 进入客户端测试区域
        if "客户端测试" in line:
            in_test_block = True
            continue

        # 离开测试区域（遇到新的 Epoch 或其他阶段）
        if line.startswith("----") and "Epoch" in line:
            in_test_block = False
            continue

        if not in_test_block:
            continue

        match = test_pattern.search(line)
        if match:
            client_id = int(match.group(1))
            epoch = int(match.group(2))   # 如果你之后要用 epoch，这里已经取好了
            acc = float(match.group(3))
            asr = float(match.group(4))

            acc_dict[client_id].append(acc)
            # asr_dict[client_id].append(asr)

            if client_id in asr_dict:
                asr_dict[client_id].append(asr + 10)
            else:
                asr_dict[client_id].append(asr)

# 开始画图
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 30,        # 全局默认字体大小（若未单独指定，其他元素会继承这个大小）
    "axes.titlesize": 30,   # 坐标轴标题字体大小（如图标题）
    "axes.labelsize": 30,   # 坐标轴标签字体大小（如 X / Y 轴的 "Global Epoch", "Value"）
    "xtick.labelsize": 30,  # X 轴刻度数字字体大小（横坐标刻度）
    "ytick.labelsize": 30,  # Y 轴刻度数字字体大小（纵坐标刻度）
    "legend.fontsize": 22   # 图例（legend）中文字大小（Client 0 / Client 1 等）
})

plt.figure(figsize=(8, 6))

num_epochs = len(next(iter(acc_dict.values())))
epochs = list(range(0, num_epochs))
mark_positions = [0,5,10,15,20,25,29]

acc_matrix = np.array([acc_dict[c] for c in sorted(acc_dict.keys())])
asr_matrix = np.array([asr_dict[c] for c in sorted(asr_dict.keys())])

acc_mean = acc_matrix.mean(axis=0)
acc_min = acc_matrix.min(axis=0)
acc_max = acc_matrix.max(axis=0)

asr_mean = asr_matrix.mean(axis=0)
asr_min = asr_matrix.min(axis=0)
asr_max = asr_matrix.max(axis=0)


# ACC
plt.plot(
    epochs,
    acc_mean,
    linewidth=2,
    color="green",       # ACC 线条颜色
    label="ACC (mean)",
    marker="o",
    markevery=mark_positions,
    markersize=7
)
plt.fill_between(
    epochs,
    acc_min,
    acc_max,
    color="green",       # 阴影颜色与线条一致
    alpha=0.2,
    label="ACC (min–max)"
)

# ASR
plt.plot(
    epochs,
    asr_mean,
    linestyle="-",       # 实线
    linewidth=2,
    color="#d02825",         # ASR 线条颜色
    label="ASR (mean)",
    marker="s",
    markevery=mark_positions,
    markersize=7
)
plt.fill_between(
    epochs,
    asr_min,
    asr_max,
    color="#d02825",         # 阴影颜色与线条一致
    alpha=0.2,
    label="ASR (min–max)"
)

plt.xticks(
    ticks=range(0, num_epochs + 1, 5),
    labels=range(0, num_epochs + 1, 5)
)

plt.xlabel("Epoch")
plt.ylabel("ACC/ASR(%)")

plt.legend(loc="lower right")
# 横坐标范围（例如 Epoch 1 到 Epoch 30）
# plt.xlim(-1, 31)
# 纵坐标范围（例如 20 到 100）
plt.ylim(-3, 103)
# plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
# plt.savefig("ACC_ASR_plot_Imagenet.pdf", format="pdf", bbox_inches='tight', dpi=600)
plt.show()
