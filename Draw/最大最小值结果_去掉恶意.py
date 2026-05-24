import re
import statistics

# 单个日志文件路径
log_path = r"E:\博士研究生\模型异构蒸馏第三篇\实验数据\鲁棒聚合mean\SADBA\ImageNet_SADBA_mean.txt"

# 读取日志文件
with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

# 匹配客户端测试块
test_blocks = re.findall(
    r"--------------客户端测试----------------([\s\S]*?)(?=------------------------------------------------------------|$)",
    text
)

acc_list = []
asr_list = []

if test_blocks:
    last_block = test_blocks[-1]  # 只取最后一次测试
    pattern = re.compile(
        r"Client\s+(?!0\b)\d+\s+\|\s+Epoch\s+\d+\|\s+Acc:\s+([\d.]+),\s+ASR:\s+([\d.]+)"
    )
    for acc, asr in pattern.findall(last_block):
        acc_list.append(float(acc))
        asr_list.append(float(asr))

# 计算 mean ± std 并输出 LaTeX 宏风格（加粗）
if acc_list and asr_list:
    mean_acc = statistics.mean(acc_list)
    std_acc  = statistics.stdev(acc_list)

    mean_asr = statistics.mean(asr_list)
    std_asr  = statistics.stdev(asr_list)

    latex_acc = f"\\acc{{{mean_acc:.2f}}}{{{std_acc:.2f}}}"
    latex_asr = f"\\asr{{{mean_asr:.2f}}}{{{std_asr:.2f}}}"

    print("ACC:", latex_acc)
    print("ASR:", latex_asr)
else:
    print("No ACC or ASR data found.")
