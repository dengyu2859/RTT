import re
import statistics

# 单个日志文件路径
log_path = r"E:\博士研究生\模型异构蒸馏第三篇\实验数据\ImageNet人工后门\ImageNet_AI_trigger.txt"

# 读取日志文件
with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
    text = f.read()

# 匹配客户端测试块
test_blocks = re.findall(
    r"--------------客户端测试----------------([\s\S]*?)(?=------------------------------------------------------------|$)",
    text
)

acc_list = []
artificial_asr_list = []

if test_blocks:
    last_block = test_blocks[-1]  # 只取最后一次测试
    pattern = re.compile(
        r"Client\s+\d+\s+\|\s+Epoch\s+\d+\|\s+Acc:\s*([\d.]+),\s*ASR:\s*[\d.]+\s*,\s*Artificial ASR:\s*([\d.]+)"
    )
    for acc, art_asr in pattern.findall(last_block):
        acc_list.append(float(acc))
        artificial_asr_list.append(float(art_asr))

# 计算 mean ± std 并输出 LaTeX 宏风格（加粗）
if acc_list and artificial_asr_list:
    mean_acc = statistics.mean(acc_list)
    std_acc = statistics.stdev(acc_list)
    mean_art_asr = statistics.mean(artificial_asr_list)
    std_art_asr = statistics.stdev(artificial_asr_list)

    latex_acc = f"\\textbf{{\\acc{{{mean_acc:.2f}}}{{{std_acc:.2f}}}}}"
    latex_art_asr = f"\\textbf{{\\artasr{{{mean_art_asr:.2f}}}{{{std_art_asr:.2f}}}}}"

    print("ACC:", latex_acc)
    print("Artificial ASR:", latex_art_asr)
else:
    print("No data found.")
