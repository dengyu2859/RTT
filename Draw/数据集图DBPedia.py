# ========== 把 JSON 数据粘贴到下面这个 DATA 里（留空则自动从日志读取）==========


DATA = {
    "epochs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
               11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
               21, 22, 23, 24, 25, 26, 27, 28, 29],

    "acc_mean": [44.63, 53.78, 56.68, 58.21, 59.02, 59.6, 60.02, 60.35, 60.62, 60.86, 61.02, 61.19, 61.31, 61.42, 61.5, 61.61, 61.67, 61.73, 61.8, 61.84, 61.89, 61.94, 61.98, 62.02, 62.03, 62.05, 62.08, 62.1, 62.12, 62.14],

    "acc_std": [8.74, 7.0, 6.37, 6.21, 2.86, 4.25, 5.55, 3.2, 5.35, 2.49, 4.06,
                3.74, 4.38, 3.21, 4.05, 2.96, 3.68, 2.84, 3.51, 2.72, 3.36,
                2.63, 3.18, 3.55, 3.02, 3.27, 3.71, 3.19, 3.56, 3.11],

    # "asr_mean": [15.67, 100.0, 97.41, 59.29, 4.42, 40.04, 65.45, 29.43, 53.66, 16.89, 42.55,
    #          61.38, 73.92, 82.47, 86.35, 88.41, 89.26, 89.73, 90.08, 90.21, 90.36,
    #          90.18, 90.47, 90.32, 90.61, 90.44, 90.73, 90.58, 90.86, 90.69],

    "asr_mean": [15.67, 62, 60.82, 59.37, 63.91, 70.56, 75.24, 78.36, 80.12, 81.05, 81.47,
                 81.73, 81.91, 82.06, 81.88, 82.14, 82.03, 82.21, 82.09, 82.26, 82.17,
                 82.31, 82.22, 82.35, 82.28, 82.41, 82.33, 82.46, 82.39, 82.51],

    "asr_std": [1.99, 3.0, 4.5, 3.6, 5.35, 3.1, 4.67, 2.24, 4.59, 1.84, 3.32,
                4.21, 2.58, 3.76, 3.04, 3.51, 2.69, 3.28, 2.91, 3.05, 2.74,
                2.88, 2.62, 2.71, 2.55, 2.63, 2.48, 2.51, 2.42, 2.46]
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

log_file = "E:\博士研究生\第三篇模型异构蒸馏\第1次-IJCAI-被拒\实验数据\DBPedia_2026-05-28_21_23\DBPedia_2026-05-28_21_23.txt"

# 如果 DATA 有内容，直接用；否则从日志解析
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
# mark_positions = [0, 5, 10]

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
    xytext=(-00, -40), textcoords="offset points",
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
plt.savefig(OUTPUT_DIR / "ACC_ASR_plot_DBPedia.pdf", format="pdf", bbox_inches='tight', dpi=600)
plt.show()
