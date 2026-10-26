import os
import re
import statistics

# ============================================================
# 数据：如果为空，则从日志文件读取并输出；如果不为空，则直接使用
# ============================================================
# acc_mean_list = []
# acc_std_list = []
# asr_mean_list = []
# asr_std_list = []

acc_mean_list = [62.43, 68.30, 72.27, 73.16, 75.13, 74.78, 74.59, 76.13, 75.86, 75.30]
acc_std_list  = [3.89, 5.47, 5.01, 4.95, 4.75, 4.84, 5.10, 4.89, 4.25, 4.67]
asr_mean_list = [85.47, 86.01, 85.58, 86.45, 84.23, 84.92, 82.79, 83.52, 83.01, 84.28]
asr_std_list  = [3.10, 2.86, 3.32, 3.35, 3.11, 2.94, 3.55, 4.21, 3.30, 3.67]


# 日志目录（仅在数据为空时需要）
log_dir = r"E:\博士研究生\第三篇模型异构蒸馏\第1次-IJCAI-被拒\实验数据\CIFAR10no-iid"

if not acc_mean_list:  # 数据为空，从文件读取
    for root, _, files in os.walk(log_dir):
        for file in sorted(files):
            if not file.endswith(".txt"):
                continue

            log_path = os.path.join(root, file)

            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            test_blocks = re.findall(
                r"--------------客户端测试----------------([\s\S]*?)(?=------------------------------------------------------------|$)",
                text
            )

            if not test_blocks:
                continue

            last_block = test_blocks[-1]

            acc_list = []
            asr_list = []
            pattern = re.compile(
                r"Client\s+\d+\s+\|\s+Epoch\s+\d+\|\s+Acc:\s+([\d.]+),\s+ASR:\s+([\d.]+)"
            )

            for acc, asr in pattern.findall(last_block):
                acc_list.append(float(acc))
                asr_list.append(float(asr))

            if not acc_list:
                continue

            acc_mean_list.append(statistics.mean(acc_list))
            acc_std_list.append(statistics.stdev(acc_list))

            asr_mean_list.append(statistics.mean(asr_list))
            asr_std_list.append(statistics.stdev(asr_list))

    # 打印结果，方便复制到脚本中
    def fmt(lst):
        return "[" + ", ".join(f"{v:.2f}" for v in lst) + "]"

    print(f"acc_mean_list = {fmt(acc_mean_list)}")
    print(f"acc_std_list  = {fmt(acc_std_list)}")
    print(f"asr_mean_list = {fmt(asr_mean_list)}")
    print(f"asr_std_list  = {fmt(asr_std_list)}")


# 开始画图
from pathlib import Path
import matplotlib.pyplot as plt

OUTPUT_DIR = Path(__file__).resolve().parent / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
import numpy as np

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 30,
    "axes.titlesize": 30,
    "axes.labelsize": 30,
    "xtick.labelsize": 25,
    "ytick.labelsize": 30,
    "legend.fontsize": 22
})

plt.figure(figsize=(8, 6))

epochs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
plt.xticks(epochs)

acc_mean = np.array(acc_mean_list)
acc_std = np.array(acc_std_list)
asr_mean = np.array(asr_mean_list)
asr_std = np.array(asr_std_list)

# ACC
plt.plot(
    epochs,
    acc_mean,
    linewidth=2,
    color="green",
    label="ACC (Mean)",
    marker="o",
    markevery=1,
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
    markevery=1,
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

plt.xlabel(r"$\alpha$")
plt.ylabel("ACC/ASR(%)")

plt.legend(loc="lower right")
plt.ylim(-3, 103)
plt.yticks([0, 25, 50, 75, 100])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "Non_IID.pdf", format="pdf", bbox_inches='tight', dpi=600)
plt.show()
