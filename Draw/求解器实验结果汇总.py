"""打印求解器实验最后一轮所有客户端的 ACC/ASR 均值 ± 样本标准差。

只读取 config.yaml 和 train.log，不执行训练，不写入任何结果文件。
默认读取 Outputs/logit_solver_ablation，可通过 --outputs-root 指定其他目录。
"""

import argparse
from pathlib import Path
import re
import statistics
import unicodedata

import yaml


OUTPUTS_ROOT = Path(__file__).resolve().parents[1] / "Outputs/logit_solver_ablation"
TEST_MARKER = "--------------客户端测试----------------"
NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
TEST_RESULT = re.compile(
    rf"^Client\s+(\d+)\s*\|\s*Epoch\s+(\d+)\s*\|\s*"
    rf"Acc:\s*({NUMBER})\s*,\s*ASR:\s*({NUMBER})(?=\s|,|$)",
    re.MULTILINE,
)


def read_result(run_directory):
    with (run_directory / "config.yaml").open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("config.yaml 不是参数字典")
    dataset, solver = run_directory.parent.name, run_directory.name
    if config.get("dataset") != dataset or config.get("logit_solver", solver) != solver:
        raise ValueError("配置中的数据集或求解器与目录名称不一致")
    epochs, clients = config.get("epochs"), config.get("clients")
    if any(type(value) is not int or value < 1 for value in (epochs, clients)):
        raise ValueError("配置中的 epochs/clients 必须是正整数")
    text = (run_directory / "train.log").read_text(encoding="utf-8")
    if TEST_MARKER not in text:
        raise ValueError("没有正式客户端测试记录")
    final_block = text.rsplit(TEST_MARKER, 1)[1]
    results = {}
    for client, epoch, acc, asr in TEST_RESULT.findall(final_block):
        client, epoch = int(client), int(epoch)
        if epoch != epochs:
            raise ValueError(f"未完成：最后测试为第 {epoch}/{epochs} 轮")
        if client in results:
            raise ValueError(f"最后一轮客户端 {client} 的测试记录重复")
        acc, asr = float(acc), float(asr)
        if not (0 <= acc <= 100 and 0 <= asr <= 100):
            raise ValueError("ACC/ASR 超出百分比范围")
        results[client] = (acc, asr)
    if set(results) != set(range(clients)):
        raise ValueError(f"最后一轮测试不完整：记录 {len(results)}/{clients} 个客户端")
    acc_values, asr_values = zip(*(results[client] for client in sorted(results)))
    return [
        dataset, solver, str(epochs), str(clients),
        mean_and_std(acc_values), mean_and_std(asr_values),
    ]


def mean_and_std(values):
    deviation = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{statistics.mean(values):.2f} ± {deviation:.2f}"


def display_width(text):
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in text)


def print_table(rows):
    widths = [max(display_width(row[column]) for row in rows)
              for column in range(len(rows[0]))]
    for index, row in enumerate(rows):
        print("  ".join(value + " " * (width - display_width(value))
                        for value, width in zip(row, widths)).rstrip())
        if index == 0:
            print("  ".join("-" * width for width in widths))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=OUTPUTS_ROOT,
                        help="求解器实验根目录，包含 <数据集>/<求解器>/train.log")
    args = parser.parse_args()
    root = args.outputs_root.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"目录不存在：{root}")
    directories = sorted({path.parent for pattern in ("*/*/config.yaml", "*/*/train.log")
                          for path in root.glob(pattern)})
    if not directories:
        parser.error(f"没有找到实验配置或日志：{root}")
    rows, errors = [], []
    for directory in directories:
        try:
            rows.append(read_result(directory))
        except (OSError, ValueError, yaml.YAMLError) as error:
            rows.append([directory.parent.name, directory.name, "N/A", "N/A", "N/A", "N/A"])
            errors.append(f"{directory.relative_to(root)}：{error}")
    print_table([["数据集", "求解器", "最后轮次", "客户端数", "ACC(%)", "ASR(%)"], *rows])
    print("\n均值 ± 样本标准差：最后一轮所有客户端之间的波动，并非多次独立实验的波动。")
    if errors:
        print("\n未能汇总的实验：")
        print("\n".join(errors))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
