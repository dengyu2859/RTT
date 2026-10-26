"""按“天然触发器比例大小.py”的风格绘制 up_margin_step 敏感性结果。"""

import argparse
import math
from pathlib import Path
import pprint
import re
import statistics

import matplotlib.pyplot as plt
import numpy as np
import yaml


# 在这里选择数据集。优先使用下面保存的数据；无对应数据时再读取日志。
DATASET = "CIFAR10"
RESULTS_ROOT = Path(__file__).resolve().parents[1] / "Outputs" / "up_margin_step_sensitivity"
OUTPUT_DIR = Path(__file__).resolve().parent / "images"

# ============================================================
# 已保存的实验结果，可直接修改后重新画图，不会重复读取日志。
# 数值为最后一轮所有客户端的均值和样本标准差，保留完整计算精度。
# 如需重新读取日志，将 DATA 改为 {}，或运行时添加 --from-logs。
# ============================================================
DATA = {
    "dataset": "CIFAR10",
    "clients": 10,
    "up_margin_steps": [0.03, 0.04, 0.05, 0.06, 0.07],
    "acc_mean_list": [67.592, 68.137, 67.606, 68.206, 65.431],
    "acc_std_list": [5.659078841413444, 5.655125600334229, 5.801615100795448,
                     5.518514695499545, 6.5807031876878055],
    "asr_mean_list": [71.741, 71.37, 71.33, 70.871, 80.809],
    "asr_std_list": [2.930075653175757, 3.839675912249771, 2.9671347945263453,
                     4.4201846366664626, 3.6921642253110405],
}
# DATA = {}  # 清空后自动从 RESULTS_ROOT 中读取并打印可复制的数据块。

TEST_MARKER = "--------------客户端测试----------------"
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
TEST_RESULT = re.compile(
    rf"^Client\s+(\d+)\s*\|\s*Epoch\s+(\d+)\s*\|\s*"
    rf"Acc:\s*({NUMBER})\s*,\s*ASR:\s*({NUMBER})",
    re.MULTILINE,
)


def read_final_metrics(run_dir, dataset):
    with (run_dir / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if config["dataset"] != dataset:
        raise ValueError(f"{run_dir}: YAML 中的数据集不匹配")
    step = float(config["up_margin_step"])
    directory_step = float(run_dir.name.removeprefix("up_margin_step_"))
    if not math.isfinite(step) or step != directory_step:
        raise ValueError(f"{run_dir}: 目录名称与实际 up_margin_step 不一致")
    text = (run_dir / "train.log").read_text(encoding="utf-8", errors="replace")
    if TEST_MARKER not in text:
        raise ValueError(f"{run_dir}: 没有找到客户端测试结果")
    # 只读最后一次正式测试，排除本地训练日志和 Artificial ASR。
    final_block = text.rsplit(TEST_MARKER, 1)[1]
    clients = {}
    for client_text, epoch_text, acc_text, asr_text in TEST_RESULT.findall(final_block):
        client_id, epoch = int(client_text), int(epoch_text)
        if epoch != config["epochs"]:
            raise ValueError(f"{run_dir}: 最后一次测试为第 {epoch} 轮，训练尚未完成")
        if client_id in clients:
            raise ValueError(f"{run_dir}: 客户端 {client_id} 的结果重复")
        acc, asr = float(acc_text), float(asr_text)
        if not (0 <= acc <= 100 and 0 <= asr <= 100):
            raise ValueError(f"{run_dir}: ACC/ASR 超出百分比范围")
        clients[client_id] = (acc, asr)
    if set(clients) != set(range(config["clients"])):
        raise ValueError(f"{run_dir}: 最后一轮客户端结果不完整")
    acc_values, asr_values = zip(*(clients[index] for index in sorted(clients)))
    return {
        "step": step,
        "clients": len(clients),
        "acc_mean": statistics.mean(acc_values),
        "acc_std": statistics.stdev(acc_values) if len(clients) > 1 else 0.0,
        "asr_mean": statistics.mean(asr_values),
        "asr_std": statistics.stdev(asr_values) if len(clients) > 1 else 0.0,
    }


def load_results(results_root, dataset):
    runs = [directory for directory in (results_root / dataset).glob("up_margin_step_*")
            if directory.is_dir()]
    if not runs:
        raise FileNotFoundError(f"没有找到实验结果：{results_root / dataset}")
    results = sorted((read_final_metrics(run, dataset) for run in runs),
                     key=lambda result: result["step"])
    if len({result["step"] for result in results}) != len(results):
        raise ValueError("存在重复的 up_margin_step 实验")
    return results


def results_from_data(data, dataset):
    if not data or data.get("dataset") != dataset:
        return None
    steps = data["up_margin_steps"]
    if not steps:
        return None
    keys = ("acc_mean", "acc_std", "asr_mean", "asr_std")
    if any(len(data[key + "_list"]) != len(steps) for key in keys):
        raise ValueError("DATA 中四组指标列表的长度必须与 up_margin_steps 一致")
    if type(data["clients"]) is not int or data["clients"] < 1:
        raise ValueError("DATA 中 clients 必须为正整数")
    results = []
    for index, step in enumerate(steps):
        result = {"step": float(step), "clients": data["clients"]}
        result.update({key: float(data[key + "_list"][index]) for key in keys})
        if not all(math.isfinite(result[key]) for key in ("step", *keys)):
            raise ValueError("DATA 中存在非有限数值")
        if result["step"] < 0 or result["acc_std"] < 0 or result["asr_std"] < 0:
            raise ValueError("步长和标准差不能为负数")
        if not (0 <= result["acc_mean"] <= 100 and 0 <= result["asr_mean"] <= 100):
            raise ValueError("ACC/ASR 均值应在 0 到 100 之间")
        results.append(result)
    if len({result["step"] for result in results}) != len(results):
        raise ValueError("DATA 中存在重复步长")
    return sorted(results, key=lambda result: result["step"])


def print_data_block(results, dataset):
    if len({result["clients"] for result in results}) != 1:
        raise ValueError("各组实验的客户端数量不一致")
    data = {"dataset": dataset, "clients": results[0]["clients"],
            "up_margin_steps": [result["step"] for result in results]}
    for key in ("acc_mean", "acc_std", "asr_mean", "asr_std"):
        data[key + "_list"] = [result[key] for result in results]
    print("\n将下面的数据复制到脚本顶部 DATA 中，下次即可直接画图：")
    print("DATA = " + pprint.pformat(data, sort_dicts=False, width=100))
    print()


def plot_results(results, dataset):
    # 与参考脚本保持相同的字体、画布、配色、标记、阴影和坐标范围。
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 30,
        "axes.titlesize": 30,
        "axes.labelsize": 30,
        "xtick.labelsize": 25,
        "ytick.labelsize": 30,
        "legend.fontsize": 22,
    })
    steps = np.array([result["step"] for result in results])
    acc_mean = np.array([result["acc_mean"] for result in results])
    acc_std = np.array([result["acc_std"] for result in results])
    asr_mean = np.array([result["asr_mean"] for result in results])
    asr_std = np.array([result["asr_std"] for result in results])

    figure, axes = plt.subplots(figsize=(8, 6))
    axes.set_xticks(steps)
    axes.plot(steps, acc_mean, linewidth=2, color="green", label="ACC (Mean)",
              marker="o", markevery=1, markersize=7)
    axes.fill_between(steps, acc_mean - acc_std, acc_mean + acc_std,
                      color="green", alpha=0.2, label="ACC (Mean ± Std)")
    axes.plot(steps, asr_mean, linestyle="-", linewidth=2, color="#d02825",
              label="ASR (Mean)", marker="s", markevery=1, markersize=7)
    axes.fill_between(steps, asr_mean - asr_std, asr_mean + asr_std,
                      color="#d02825", alpha=0.2, label="ASR (Mean ± Std)")
    axes.set_xlabel("Up margin step")
    axes.set_ylabel("ACC/ASR(%)")
    axes.legend(loc="lower right")
    axes.set_ylim(-3, 103)
    axes.set_yticks([0, 25, 50, 75, 100])
    figure.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"up_margin_step_{dataset}.pdf"
    figure.savefig(output_path, format="pdf", bbox_inches="tight", dpi=600)
    return figure, output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=DATASET)
    parser.add_argument("--results-root", type=Path, default=RESULTS_ROOT)
    parser.add_argument("--from-logs", action="store_true", help="重新读取日志，忽略代码中保存的数据")
    parser.add_argument("--no-show", action="store_true", help="只保存图片，不弹出窗口")
    args = parser.parse_args()
    results = None if args.from_logs else results_from_data(DATA, args.dataset)
    if results is None:
        results = load_results(args.results_root, args.dataset)
        print_data_block(results, args.dataset)
    else:
        print("使用代码中保存的数据画图，不读取日志。")
    print(f"{args.dataset}：最后一轮所有客户端的均值 ± 样本标准差")
    for result in results:
        print(f"up_margin_step={result['step']:g} | 客户端={result['clients']} | "
              f"ACC={result['acc_mean']:.2f} ± {result['acc_std']:.2f} | "
              f"ASR={result['asr_mean']:.2f} ± {result['asr_std']:.2f}")
    figure, output_path = plot_results(results, args.dataset)
    print(f"图片已保存：{output_path}")
    if args.no_show:
        plt.close(figure)
    else:
        plt.show()


if __name__ == "__main__":
    main()
