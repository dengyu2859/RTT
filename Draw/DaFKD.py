# ========== 把 JSON 数据粘贴到下面这个 DATA 里（留空则自动从日志读取）==========
# DATA = {
#     "epochs": [],
#     "acc_mean": [],
#     "acc_std": [],
#     "asr_mean": [],
#     "asr_std": []
# }

DATA = {
    "epochs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    "acc_mean": [30.04, 36.49, 45.75, 53.1, 59.18, 63.44, 66.89, 69.95, 72.47, 74.36, 76.04, 77.19, 78.08, 79.12, 79.76, 80.34, 80.66, 81.03, 81.17, 81.72, 81.84, 82.22, 82.32, 82.35, 82.43, 82.75, 82.88, 82.81, 82.82, 83.09],
    "acc_std": [12.16, 11.99, 11.47, 10.72, 9.71, 8.03, 6.34, 5.13, 4.51, 3.84, 3.37, 3.06, 2.94, 2.59, 2.67, 2.39, 2.39, 2.13, 2.04, 1.94, 1.98, 2.0, 1.94, 1.93, 2.1, 2.12, 1.97, 1.85, 1.98, 1.94],
    "asr_mean": [3.02, 83.13, 84.37, 83.52, 82.99, 85.13, 86.78, 87.58, 87.76, 87.82, 89.14, 89.86, 90.73, 90.72, 91.07, 91.16, 91.59, 91.8, 91.93, 91.96, 92.12, 92.11, 92.38, 92.3, 92.37, 92.42, 92.56, 92.77, 92.69, 92.69],
    "asr_std": [1.91, 5.35, 3.52, 3.97, 4.24, 3.89, 2.64, 2.47, 2.17, 2.08, 2.48, 2.51, 3.0, 2.94, 2.73, 2.56, 2.53, 2.47, 2.43, 2.58, 2.57, 2.47, 2.48, 2.68, 2.58, 2.55, 2.64, 2.75, 2.64, 2.64]
}

# ======================================================================

import re
import json
from collections import defaultdict
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log_file = "E:\博士研究生\第三篇模型异构蒸馏\第1次-IJCAI-被拒\实验数据\DaKFD\GTSRB_ours_DaFKD.txt"

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
    label="ACC (mean)",
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
    label="ASR (mean)",
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
    xytext=(-00, -32), textcoords="offset points",
    fontsize=22, color="green", va="bottom", ha="right"
)
plt.annotate(
    f"ASR: {asr_mean[last]:.2f}±{asr_std[last]:.2f}",
    xy=(epochs[last], asr_mean[last]),
    xytext=(-00, 24), textcoords="offset points",
    fontsize=22, color="#d02825", va="top", ha="right"
)

plt.xticks(
    ticks=range(0, num_epochs + 1, 5),
    labels=range(0, num_epochs + 1, 5)
)
plt.yticks([0, 25, 50, 75, 100])

plt.xlabel("Epoch")
plt.ylabel("ACC/ASR(%)")

plt.legend(loc="lower right")
plt.ylim(-3, 103)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "ACC_ASR_plot_DaFKD.pdf", format="pdf", bbox_inches='tight', dpi=600)
plt.show()
