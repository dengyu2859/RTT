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
    "acc_mean": [30.00, 37.26, 45.18, 52.43, 58.76, 63.82, 67.95, 71.24, 73.86, 76.12, 78.04, 79.61, 80.93, 82.05, 82.94, 83.56, 84.02, 84.35, 84.58, 84.73, 84.86, 84.92, 85.01, 85.08, 85.12, 85.16, 85.10, 85.21, 85.18, 85.24],
    "acc_std": [3.12, 2.35, 2.78, 2.96, 3.18, 3.42, 3.76, 3.21, 3.68, 3.34, 3.12, 2.96, 2.78, 2.55, 2.36, 2.18, 1.96, 1.82, 1.71, 1.63, 1.48, 1.67, 1.52, 1.74, 1.45, 1.39, 1.31, 1.24, 1.18, 1.88],
    "asr_mean": [2.86, 96.0, 97.73, 97.65, 97.77, 97.1, 96.17, 96.37, 95.64, 95.88, 96.11, 96.09, 96.28, 96.69, 96.8, 97.05, 97.08, 97.07, 97.28, 97.38, 97.44, 97.5, 97.56, 97.48, 97.56, 97.41, 97.28, 97.25, 97.06, 96.8],
    "asr_std": [2.9, 2.57, 2.33, 1.85, 1.77, 1.79, 3.09, 2.14, 2.37, 2.39, 2.32, 2.35, 2.23, 2.05, 2.02, 1.77, 1.9, 1.69, 1.73, 1.61, 1.56, 1.52, 1.4, 1.48, 1.35, 1.3, 1.45, 1.28, 1.37, 1.36]
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

log_file = "E:\博士研究生\第三篇模型异构蒸馏\第1次-IJCAI-被拒\实验数据\FedFD\GTSRB_ours_FedDF - 副本.txt"

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
    xytext=(-00, -30), textcoords="offset points",
    fontsize=22, color="green", va="bottom", ha="right"
)
plt.annotate(
    f"ASR: {asr_mean[last]:.2f}±{asr_std[last]:.2f}",
    xy=(epochs[last], asr_mean[last]),
    xytext=(-00, -7), textcoords="offset points",
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
plt.savefig(OUTPUT_DIR / "ACC_ASR_plot_FedDF.pdf", format="pdf", bbox_inches='tight', dpi=600)
plt.show()
