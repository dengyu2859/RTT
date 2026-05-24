import os
import re
import statistics

# 日志目录
log_dir = r"E:\博士研究生\模型异构蒸馏第三篇\实验数据\Imagenet_τ值变化\target_prob"

# 结果列表
acc_mean_list, acc_max_list, acc_min_list = [], [], []
asr_mean_list, asr_max_list, asr_min_list = [], [], []

# 遍历目录下所有文件
for root, _, files in os.walk(log_dir):
    for file in sorted(files):
        if not file.endswith(".txt"):  # 只处理txt文件
            continue

        log_path = os.path.join(root, file)

        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()

        # 匹配所有客户端测试块
        test_blocks = re.findall(
            r"--------------客户端测试----------------([\s\S]*?)(?=------------------------------------------------------------|$)",
            text
        )

        if not test_blocks:
            continue  # 没有测试块就跳过

        last_block = test_blocks[-1]  # 只取最后一次测试

        # 提取每个客户端的 Acc 和 ASR
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

        # 统计并加入列表
        acc_mean_list.append(statistics.mean(acc_list))
        acc_max_list.append(max(acc_list))
        acc_min_list.append(min(acc_list))

        asr_mean_list.append(statistics.mean(asr_list))
        asr_max_list.append(max(asr_list))
        asr_min_list.append(min(asr_list))

# 打印结果
print("ACC mean list:", acc_mean_list)
print("ACC max  list:", acc_max_list)
print("ACC min  list:", acc_min_list)
print("ASR mean list:", asr_mean_list)
print("ASR max  list:", asr_max_list)
print("ASR min  list:", asr_min_list)

# acc_mean_list = [85.32, 82.80, 81.98, 81.35, 81.08, 82.609, 85.96, 90.87, 95.93]
# acc_max_list = [85.9, 83.74, 82.89, 82.51, 81.85, 83.16, 86.59, 91.33, 96.32]
# acc_min_list = [84.27, 81.25, 80.37, 80.03, 79.78, 81.37, 85.62, 90.37, 95.46]
# asr_mean_list = [74.54, 81.21, 82.56, 80.89, 77.20, 74.55, 75.63, 72.51, 72.73]
# asr_max_list = [78.71, 82.61, 84.73, 81.63, 78.38, 76.53, 77.56, 75.6, 75.13]
# asr_min_list = [70.92, 79.59, 80.97, 80.0, 75.84, 71.8, 74.37, 69.48, 70.23]

# 开始画图
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 30,        # 全局默认字体大小（若未单独指定，其他元素会继承这个大小）
    "axes.titlesize": 30,   # 坐标轴标题字体大小（如图标题）
    "axes.labelsize": 30,   # 坐标轴标签字体大小（如 X / Y 轴的 "Global Epoch", "Value"）
    "xtick.labelsize": 25,  # X 轴刻度数字字体大小（横坐标刻度）
    "ytick.labelsize": 30,  # Y 轴刻度数字字体大小（纵坐标刻度）
    "legend.fontsize": 22   # 图例（legend）中文字大小（Client 0 / Client 1 等）
})

plt.figure(figsize=(8, 6))

epochs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
plt.xticks(epochs)

# ACC
plt.plot(
    epochs,
    acc_mean_list,
    linewidth=2,
    color="green",       # ACC 线条颜色
    label="ACC (mean)",
    marker="o",
    markevery=1,
    markersize=7
)
plt.fill_between(
    epochs,
    acc_max_list,
    acc_min_list,
    color="green",       # 阴影颜色与线条一致
    alpha=0.2,
    label="ACC (min–max)"
)

# ASR
plt.plot(
    epochs,
    asr_mean_list,
    linestyle="-",       # 实线
    linewidth=2,
    color="#d02825",         # ASR 线条颜色
    label="ASR (mean)",
    marker="s",
    markevery=1,
    markersize=7
)
plt.fill_between(
    epochs,
    asr_max_list,
    asr_min_list,
    color="#d02825",         # 阴影颜色与线条一致
    alpha=0.2,
    label="ASR (min–max)"
)

plt.xlabel(r"$\tau$")
plt.ylabel("ACC/ASR(%)")

plt.legend(loc="lower right")
# 横坐标范围（例如 Epoch 1 到 Epoch 30）
# plt.xlim(-1, 31)
# 纵坐标范围（例如 20 到 100）
plt.ylim(-3, 103)
plt.yticks([0, 25, 50, 75, 100])
# plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("τ大小.pdf", format="pdf", bbox_inches='tight', dpi=600)
plt.show()
