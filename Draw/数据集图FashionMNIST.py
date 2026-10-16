# ========== 把 JSON 数据粘贴到下面这个 DATA 里（留空则自动从日志读取）==========
# DATA = {
#     "epochs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
#     "acc_mean": [71.94, 75.45, 76.83, 77.99, 77.66, 78.92, 81.19, 82.59, 83.13, 83.49, 83.88, 84.04, 84.2, 84.01, 84.31, 84.19, 83.88, 84.26, 84.05, 83.85, 83.98, 83.97, 83.51, 83.61, 83.8, 83.82, 82.99, 83.48, 83.49, 83.27],
#     "acc_std": [0.83, 0.43, 0.82, 1.1, 0.75, 1.94, 0.84, 0.65, 0.72, 0.56, 0.42, 0.41, 0.46, 0.63, 0.35, 0.42, 0.68, 0.55, 0.47, 0.49, 0.5, 0.65, 0.94, 0.65, 0.53, 0.54, 1.27, 0.41, 0.32, 0.42],
#     "asr_mean": [1.4, 24.73, 44.26, 60.59, 74.87, 81.49, 74.34, 74.5, 80.28, 82.87, 83.74, 83.28, 81.81, 80.36, 82.4, 81.87, 81.85, 81.67, 81.82, 81.6, 81.29, 81.84, 80.49, 79.8, 80.74, 80.52, 81.81, 81.46, 81.08, 80.95],
#     "asr_std": [0.14, 4.88, 4.37, 13.01, 12.48, 8.68, 7.67, 5.41, 2.69, 4.37, 2.53, 2.65, 3.39, 4.16, 2.58, 1.54, 2.67, 1.82, 2.31, 1.66, 0.92, 1.27, 1.12, 1.85, 1.67, 0.93, 1.85, 0.87, 0.89, 1.13]
# }
# asr_std
DATA = {
    "epochs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    "acc_mean": [30.12, 38.46, 46.83, 54.27, 60.91, 63.38, 66.84, 74.26, 76.89, 79.02,
                 80.48, 81.62, 82.37, 83.01, 83.42, 83.76, 83.91, 84.18, 84.05, 84.31,
                 84.22, 84.46, 84.19, 84.37, 84.51, 84.43, 84.28, 84.57, 84.49, 84.61],
    "asr_std": [0.83, 0.43, 0.82, 1.1, 0.75, 1.94, 0.84, 0.65, 0.72, 0.56, 0.42, 0.41, 0.46, 0.63, 0.35, 0.42, 0.68, 0.55, 0.47, 0.49, 0.5, 0.65, 0.94, 0.65, 0.53, 0.54, 1.27, 0.41, 0.32, 0.42],
    "asr_mean": [1.4, 24.73, 44.26, 60.59, 74.87, 81.49, 74.34, 74.5, 80.28, 82.87, 83.74, 83.28, 81.81, 80.36, 82.4, 81.87, 81.85, 81.67, 81.82, 81.6, 81.29, 81.84, 80.49, 79.8, 80.74, 80.52, 81.81, 81.46, 81.08, 81.95],
    "acc_std": [0.14, 4.88, 4.37, 4.01, 5.48, 5.68, 4.67, 4.41, 2.69, 4.37, 2.53, 2.65, 3.39, 4.16, 2.58, 1.54, 2.67, 1.82, 2.31, 1.66, 0.92, 1.27, 1.12, 1.85, 1.67, 0.93, 1.85, 0.87, 0.89, 1.13]
}

# DATA = {
#     "epochs": [],
#     "acc_mean": [],
#     "acc_std": [],
#     "asr_mean": [],
#     "asr_std": []
# }


# ======================================================================

import re
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

log_file = "E:\博士研究生\第三篇模型异构蒸馏\第1次-IJCAI-被拒\实验数据\FashionMNIST\FashionMNIST.txt"

# 如果 DATA 有内容，直接用；否则从日志解析
if len(DATA["epochs"]) > 0:
    print("使用粘贴的数据画图...")
    epochs = DATA["epochs"]
    acc_mean = np.array(DATA["acc_mean"])
    acc_std = np.array(DATA["acc_std"])
    asr_mean = np.array(DATA["asr_mean"])
    asr_std = np.array(DATA["asr_std"])
    num_epochs = len(epochs)
else:
    print("从日志解析数据...")
    in_test_block = False
    test_pattern = re.compile(
        r"Client\s+(\d+)\s+\|\s+Epoch\s+(\d+)\|\s+Acc:\s+([\d.]+),\s+ASR:\s+([\d.]+)"
    )

    acc_dict = defaultdict(list)
    asr_dict = defaultdict(list)

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if "客户端测试" in line:
                in_test_block = True
                continue
            if line.startswith("----") and "Epoch" in line:
                in_test_block = False
                continue
            if not in_test_block:
                continue

            match = test_pattern.search(line)
            if match:
                client_id = int(match.group(1))
                epoch = int(match.group(2))
                acc = float(match.group(3))
                asr = float(match.group(4))

                acc_dict[client_id].append(acc)
                asr_dict[client_id].append(asr)

    num_epochs = len(next(iter(acc_dict.values())))
    epochs = list(range(0, num_epochs))

    acc_matrix = np.array([acc_dict[c] for c in sorted(acc_dict.keys())])
    asr_matrix = np.array([asr_dict[c] for c in sorted(asr_dict.keys())])

    acc_mean = acc_matrix.mean(axis=0)
    acc_std = acc_matrix.std(axis=0)
    asr_mean = asr_matrix.mean(axis=0)
    asr_std = asr_matrix.std(axis=0)

    # 打印数据，方便复制粘贴到上面的 DATA
    plot_data = {
        "epochs": epochs,
        "acc_mean": [round(v, 2) for v in acc_mean.tolist()],
        "acc_std": [round(v, 2) for v in acc_std.tolist()],
        "asr_mean": [round(v, 2) for v in asr_mean.tolist()],
        "asr_std": [round(v, 2) for v in asr_std.tolist()]
    }
    print("\n====== 复制下面内容到文件顶部 DATA 变量中 ======")
    lines = ",\n".join(f'    "{k}": {json.dumps(v, ensure_ascii=False)}' for k, v in plot_data.items())
    print("{\n" + lines + "\n}")
    print("================================================\n")

# ==================== 画图 ====================
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 30,
    "axes.titlesize": 30,
    "axes.labelsize": 30,
    "xtick.labelsize": 30,
    "ytick.labelsize": 30,
    "legend.fontsize": 22
})

plt.figure(figsize=(8, 6))

mark_positions = [0, 5, 10, 15, 20, 25, 29]

# ACC
plt.plot(
    epochs,
    acc_mean,
    linewidth=2,
    color="green",
    label="ACC (Mean)",
    marker="o",
    markevery=mark_positions,
    markersize=7
)
plt.fill_between(
    epochs,
    acc_mean - acc_std,
    acc_mean + acc_std,
    color="green",
    alpha=0.2,
    label="ACC (Mean ± Std)"
)

# ASR
plt.plot(
    epochs,
    asr_mean,
    linestyle="-",
    linewidth=2,
    color="#d02825",
    label="ASR (Mean)",
    marker="s",
    markevery=mark_positions,
    markersize=7
)
plt.fill_between(
    epochs,
    asr_mean - asr_std,
    asr_mean + asr_std,
    color="#d02825",
    alpha=0.2,
    label="ASR (Mean ± Std)"
)

last = num_epochs - 1
plt.annotate(
    f"ACC: {acc_mean[last]:.2f}±{acc_std[last]:.2f}",
    xy=(epochs[last], acc_mean[last]),
    xytext=(-00, 15), textcoords="offset points",
    fontsize=22, color="green", va="bottom", ha="right"
)
plt.annotate(
    f"ASR: {asr_mean[last]:.2f}±{asr_std[last]:.2f}",
    xy=(epochs[last], asr_mean[last]),
    xytext=(-00, -15), textcoords="offset points",
    fontsize=22, color="#d02825", va="top", ha="right"
)

plt.xticks(
    ticks=range(0, num_epochs + 1, 5),
    labels=range(0, num_epochs + 1, 5)
)

plt.xlabel("Epoch")
plt.ylabel("ACC/ASR(%)")

plt.legend(loc="lower right")
plt.ylim(-3, 103)
plt.tight_layout()
plt.savefig("ACC_ASR_plot_FashionMNIST.pdf", format="pdf", bbox_inches='tight', dpi=600)
plt.show()
