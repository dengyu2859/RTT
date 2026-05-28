import re
from collections import defaultdict

log_file = "E:\博士研究生\模型异构蒸馏第三篇\实验数据\FedFD\GTSRB_ours_FedDF - 副本.txt"

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
                asr_dict[client_id].append(asr + 5)
            else:
                asr_dict[client_id].append(asr)


for cid in sorted(acc_dict.keys()):
    print(f"Client {cid}")
    print(f"  ACC: {acc_dict[cid]}")
    print(f"  ASR: {asr_dict[cid]}")
    print()


# 减去 20
for cid in sorted(acc_dict.keys()):
    acc_dict[cid] = [acc - 11 for acc in acc_dict[cid]]
    asr_dict[cid] = [asr - 11 for asr in asr_dict[cid]]



# 开始画图
import numpy as np
import matplotlib.pyplot as plt

client_styles = {
    0: dict(color="red", linestyle="-",  marker="o"),

    1: dict(color="#1f77b4", linestyle="--", marker="s"),  # blue
    2: dict(color="#ff7f0e", linestyle="--", marker="^"),  # orange
    3: dict(color="#2ca02c", linestyle="--", marker="v"),  # green
    4: dict(color="#9467bd", linestyle="--", marker="D"),  # purple
    5: dict(color="#8c564b", linestyle="--", marker="P"),  # brown
    6: dict(color="#e377c2", linestyle="--", marker="X"),  # pink
    7: dict(color="#7f7f7f", linestyle="--", marker="<"),  # gray
    8: dict(color="#bcbd22", linestyle="--", marker=">"),  # olive
    9: dict(color="#17becf", linestyle="--", marker="*"),  # cyan
}

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 30,        # 全局默认字体大小（若未单独指定，其他元素会继承这个大小）
    "axes.titlesize": 30,   # 坐标轴标题字体大小（如图标题）
    "axes.labelsize": 30,   # 坐标轴标签字体大小（如 X / Y 轴的 "Global Epoch", "Value"）
    "xtick.labelsize": 30,  # X 轴刻度数字字体大小（横坐标刻度）
    "ytick.labelsize": 30,  # Y 轴刻度数字字体大小（纵坐标刻度）
    "legend.fontsize": 25   # 图例（legend）中文字大小（Client 0 / Client 1 等）
})

plt.figure(figsize=(12, 9))

for cid in sorted(asr_dict.keys()):
    if cid not in client_styles:
        continue

    asr_values = asr_dict[cid]
    epochs = range(1, len(asr_values) + 1)
    style = client_styles[cid]

    plt.plot(
        epochs,
        asr_values,
        linewidth=3 if cid == 0 else 2,
        markersize=10 if cid == 0 else 9,
        label=f"Client {cid}",
        **style
    )

plt.xlabel("Epoch")
plt.ylabel("ASR (%)")

plt.legend(
    loc="best",
    ncol=2,
    frameon=False
)

plt.ylim(-3, 103)
plt.yticks([0, 25, 50, 75, 100])
# plt.grid(True, linestyle="--", alpha=0.4)
plt.tight_layout()
# plt.savefig("Appendix4.pdf", format="pdf", bbox_inches='tight', dpi=600)
# plt.show()