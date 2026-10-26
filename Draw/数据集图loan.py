# ========== 把 JSON 数据粘贴到下面这个 DATA 里（留空则自动从日志读取）==========
DATA = {
    "epochs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    "acc_mean": [30.03, 28.31, 28.27, 33.26, 34.59, 39.66, 40.38, 50.1, 55.66, 59.03, 61.61, 60.97, 61.99, 62.66, 64.37, 65.15, 66.48, 68.15, 68.48, 71.21, 70.94, 72.29, 73.21, 73.75, 74.82, 76.56, 77.2, 77.7, 77.76, 78.51],
    "acc_std": [1.2, 4.58, 3.52, 9.56, 9.26, 9.47, 8.25, 10.4, 11.64, 11.43, 11.38, 11.75, 11.72, 10.85, 11.46, 11.33, 10.34, 10.9, 10.79, 10.12, 9.95, 8.12, 7.72, 6.49, 5.4, 3.89, 2.14, 1.89, 2.03, 2.23],
    "asr_mean": [0.13, 56.72, 67.78, 79.0, 85.59, 89.55, 88.30, 90.12, 89.98, 88.58, 89.02, 87.36, 88.19, 87.93, 88.78, 89.57, 88.1, 88.22, 89.51, 89.22, 90.4, 89.06, 89.72, 91.04, 89.72, 89.34, 88.98, 88.9, 89.79, 89.76],
    "asr_std": [0.17, 1.37, 5.26, 7.38, 5.71, 6.63, 5.29, 5.42, 5.54, 5.11, 4.03, 3.43, 2.68, 2.53, 1.23, 1.07, 1.18, 1.72, 1.9, 2.03, 1.36, 1.28, 1.18, 1.72, 3.79, 2.28, 2.28, 1.49, 1.88, 1.76]
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
from pathlib import Path
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

log_file = "E:\博士研究生\第三篇模型异构蒸馏\第1次-IJCAI-被拒\实验数据\LOAN_2026-05-25_15_27\LOAN_2026-05-25_15_27.txt"

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
    xytext=(-00, -55), textcoords="offset points",
    fontsize=22, color="green", va="bottom", ha="right"
)
plt.annotate(
    f"ASR: {asr_mean[last]:.2f}±{asr_std[last]:.2f}",
    xy=(epochs[last], asr_mean[last]),
    xytext=(-00, 30), textcoords="offset points",
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
plt.savefig(OUTPUT_DIR / "ACC_ASR_plot_LOAN.pdf", format="pdf", bbox_inches='tight', dpi=600)
plt.show()
