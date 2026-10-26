# ========== 把 JSON 数据粘贴到下面这个 DATA 里（留空则自动从日志读取）==========
DATA = {
    "epochs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    "acc_mean": [38.02, 40.18, 41.06, 43.41, 44.8, 46.66, 46.67, 45.98, 46.37, 47.23, 48.13, 49.1, 49.8, 50.11, 50.29, 50.67, 50.81, 50.89, 50.99, 51.13, 51.14, 51.21, 51.32, 51.49, 51.4, 51.48, 51.42, 51.43, 51.52, 51.39],
    "acc_std": [4.12, 4.86, 5.34, 4.48, 5.76, 3.92, 4.67, 3.41, 2.75, 1.94, 1.52, 2.68, 3.45, 4.08, 3.21, 2.26, 3.74, 4.39, 3.56, 4.71, 3.08, 2.64, 3.52, 2.89, 2.37, 3.46, 2.91, 3.78, 2.69, 4.02],
    "asr_mean": [0.2, 0.48, 1.41, 6.26, 20.04, 38.36, 53.34, 62.95, 70.01, 72.0, 76.0, 79.0, 81.0, 82.1, 81.9, 82.0, 82.1, 81.8, 82.0, 82.0, 81.9, 82.1, 82.0, 81.9, 82.0, 82.1, 81.8, 82.0, 82.0, 81.9],
    "asr_std": [0.06, 0.11, 0.44, 1.89, 4.09, 4.92, 4.01, 2.87, 2.32, 1.99, 1.82, 1.3, 1.13, 1.02, 1.03, 0.98, 0.96, 1.12, 1.47, 1.63, 1.84, 1.21, 1.45, 1.53, 1.8, 1.91, 1.94, 1.9, 1.93, 1.97]
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

log_file = "E:\博士研究生\第三篇模型异构蒸馏\第1次-IJCAI-被拒\实验数据\Imagenet/30轮\image30.txt"

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
plt.savefig(OUTPUT_DIR / "ACC_ASR_plot_Imagenet.pdf", format="pdf", bbox_inches='tight', dpi=600)
plt.show()
