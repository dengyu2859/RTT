import re
import statistics

# 单个日志文件路径
log_path = r"E:\博士研究生\模型异构蒸馏第三篇\实验数据\随机图像触发器\ImageNet_ours_image.txt"

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
ai_asr_list = []

if test_blocks:
    last_block = test_blocks[-1]  # 只取最后一次测试
    pattern = re.compile(
        r"Acc:\s*([\d.]+),\s*ASR:\s*([\d.]+)\s*,\s*Artificial ASR:\s*([\d.]+)"
    )
    for acc, asr, ai_asr in pattern.findall(last_block):
        acc_list.append(float(acc))
        asr_list.append(float(asr))
        ai_asr_list.append(float(ai_asr))

# 计算 mean ± std 并输出 LaTeX 宏风格（加粗）
if acc_list and asr_list:
    mean_acc = statistics.mean(acc_list)
    std_acc = statistics.stdev(acc_list)

    mean_asr = statistics.mean(asr_list)
    std_asr  = statistics.stdev(asr_list)

    mean_ai_asr = statistics.mean(ai_asr_list)
    std_ai_asr  = statistics.stdev(ai_asr_list)

    latex_acc = f"\\textbf{{\\acc{{{mean_acc:.2f}}}{{{std_acc:.2f}}}}}"
    latex_asr = f"\\textbf{{\\asr{{{mean_asr:.2f}}}{{{std_asr:.2f}}}}}"
    latex_ai_asr = f"\\textbf{{\\asr{{{mean_ai_asr:.2f}}}{{{std_ai_asr:.2f}}}}}"

    print("ACC:", latex_acc)
    print("ASR:", latex_asr)
    print("Artificial ASR:", latex_ai_asr)
else:
    print("No ACC or ASR data found.")
