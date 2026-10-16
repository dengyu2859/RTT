# ========== 把 JSON 数据粘贴到下面这个 DATA 里（留空则自动从日志读取）==========
# DATA = {
#     "epochs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
#     "acc_mean": [45.82, 46.68, 47.48, 47.93, 48.57, 49.8, 49.29, 48.16, 48.19, 48.75, 49.4, 50.16, 50.68, 50.84, 50.9, 51.18, 51.24, 51.24, 51.29, 51.38, 51.35, 51.38, 51.46, 51.61, 51.5, 51.56, 51.49, 51.49, 51.57, 51.43],
#     "acc_std": [1.5, 1.2, 1.1, 1.3, 1.4, 1.2, 1.3, 1.1, 1.0, 1.2, 1.1, 1.3, 1.2, 1.1, 1.0, 1.1, 1.2, 1.3, 1.1, 1.2, 1.1, 1.3, 1.2, 1.1, 1.0, 1.1, 1.2, 1.1, 1.0, 1.1],
#     "asr_mean": [0.2, 10.48, 11.41, 16.26, 30.04, 48.36, 63.34, 72.95, 78.01, 80.9, 82.26, 82.81, 83.21, 83.86, 84.37, 84.72, 84.96, 85.16, 85.17, 85.0, 84.8, 84.73, 84.38, 84.19, 83.88, 83.43, 82.79, 82.72, 82.36, 82.3],
#     "asr_std": [0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 2.8, 3.0, 3.2, 3.3, 3.1, 3.0, 2.8, 2.5, 2.2, 2.0, 1.8, 1.5, 1.3, 1.2, 1.1, 1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.5]
# }

DATA = {
    "epochs": [],
    "acc_mean": [],
    "acc_std": [],
    "asr_mean": [],
    "asr_std": []
}


# ======================================================================

import re
import json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

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
    label="ACC (±1 std)"
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
    label="ASR (±1 std)"
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
# plt.savefig("ACC_ASR_plot_Imagenet.pdf", format="pdf", bbox_inches='tight', dpi=600)
plt.show()
