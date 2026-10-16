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

acc_mean_list = [52.02, 51.36, 52.38, 51.38, 51.56, 51.38, 52.01, 51.77, 51.98]
acc_std_list  = [2.07, 2.11, 2.15, 2.15, 2.15, 2.15, 2.15, 2.15, 2.15]
asr_mean_list = [57.38, 75.71, 80.00, 80.89, 80.53, 80.77, 80.62, 80.16, 80.25]
asr_std_list  = [1.29, 1.50, 1.72, 1.72, 1.72, 1.72, 1.72, 1.72, 1.72]

# 日志目录（仅在数据为空时需要）
log_dir = r"E:\博士研究生\第三篇模型异构蒸馏\第2次-未定-未知\实验数据\Imagenet_τ值变化\target_prob"

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
import matplotlib.pyplot as plt
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

epochs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
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

plt.xlabel(r"$\tau$")
plt.ylabel("ACC/ASR(%)")

plt.legend(loc="lower right")
plt.ylim(-3, 103)
plt.yticks([0, 25, 50, 75, 100])
plt.tight_layout()
plt.savefig("τ.pdf", format="pdf", bbox_inches='tight', dpi=600)
plt.show()
