"""汇总固定/自适应 margin、仅目标 logit、无全局反馈三个实验目录。

默认输出代码中 DATA 保存的结果，不读取日志。
--from-logs 重新读取；--refresh-cache 重新读取并更新本文件中的 DATA。
ACC/ASR 使用最后一轮正式测试中所有客户端的均值和样本标准差。
只汇总最后一轮修改前后 logits 的平均 L2 距离和平均余弦相似度。
L2 先逐样本求向量距离再平均；余弦先逐样本计算再平均，排除任一向量为零的样本。
多个恶意客户端分别按 selected_samples/cosine_samples 加权。
旧日志或缓存只有 MSE 时，新指标显示 N/A，不从 MSE 换算。
"""

import argparse
import ast
from collections import defaultdict
import math
import os
from pathlib import Path
import pprint
import re
import stat
import statistics
import tempfile
import unicodedata


OUTPUTS_ROOT = Path(__file__).resolve().parents[1] / "Outputs"
EXPERIMENT_GROUPS = (
    "desired_margin_sensitivity",
    "logits_only_target",
    "no_adaptive",
)
# 同时兼容此前的目录名与当前使用的 target_only；分组语义保持一致。
EXPERIMENT_DIRECTORIES = {
    "desired_margin_sensitivity": ("desired_margin_sensitivity",),
    "logits_only_target": ("logits_only_target", "target_only"),
    "no_adaptive": ("no_adaptive",),
}

# 已提取的实验结果，可以直接修改数值；默认运行只使用这里的数据。
# 新实验完成后运行 --refresh-cache，自动重新提取并更新此数据块。
# acc_std/asr_std 是同一轮客户端之间的样本标准差，并非多次运行标准差。
# 缓存保存最后一轮 l2_mean/cosine_mean，不保留逐轮 MSE 历史。
# 新实验完成后需刷新缓存；旧日志无法还原的新指标显示 N/A。
DATA = [{'group': 'desired_margin_sensitivity',
  'dataset': 'CIFAR10',
  'experiment': '自适应 margin',
  'epoch': 30,
  'clients': 10,
  'malicious_clients': [0],
  'acc_mean': 67.852,
  'acc_std': 5.851438189786242,
  'asr_mean': 70.791,
  'asr_std': 3.631058768152586,
  'l2_mean': 154.08793,
  'cosine_mean': 0.78810877,
  'selected_samples': 2000,
  'cosine_samples': 2000,
  'metrics_recorded': True,
  'logit_update_mode': 'full',
  'use_global_feedback': True,
  'adaptive_margin': True,
  'initial_margin': 50,
  'log_file': 'desired_margin_sensitivity/CIFAR10/adaptive/train.log'},
 {'group': 'desired_margin_sensitivity',
  'dataset': 'CIFAR10',
  'experiment': '固定 margin=50',
  'epoch': 30,
  'clients': 10,
  'malicious_clients': [0],
  'acc_mean': 74.444,
  'acc_std': 3.3291246764143754,
  'asr_mean': 28.396,
  'asr_std': 1.9823005714461046,
  'l2_mean': 39.03444,
  'cosine_mean': 0.78178617,
  'selected_samples': 2000,
  'cosine_samples': 2000,
  'metrics_recorded': True,
  'logit_update_mode': 'full',
  'use_global_feedback': True,
  'adaptive_margin': False,
  'initial_margin': 50,
  'log_file': 'desired_margin_sensitivity/CIFAR10/desired_margin_50/train.log'},
 {'group': 'desired_margin_sensitivity',
  'dataset': 'CIFAR10',
  'experiment': '固定 margin=100',
  'epoch': 30,
  'clients': 10,
  'malicious_clients': [0],
  'acc_mean': 70.585,
  'acc_std': 4.6082203120558844,
  'asr_mean': 56.138,
  'asr_std': 2.504967509214885,
  'l2_mean': 85.793872,
  'cosine_mean': 0.77823405,
  'selected_samples': 2000,
  'cosine_samples': 2000,
  'metrics_recorded': True,
  'logit_update_mode': 'full',
  'use_global_feedback': True,
  'adaptive_margin': False,
  'initial_margin': 100,
  'log_file': 'desired_margin_sensitivity/CIFAR10/desired_margin_100/train.log'},
 {'group': 'desired_margin_sensitivity',
  'dataset': 'CIFAR10',
  'experiment': '固定 margin=150',
  'epoch': 30,
  'clients': 10,
  'malicious_clients': [0],
  'acc_mean': 67.528,
  'acc_std': 5.483899261575756,
  'asr_mean': 72.29599999999999,
  'asr_std': 4.503233899716463,
  'l2_mean': 148.60478,
  'cosine_mean': 0.81316195,
  'selected_samples': 2000,
  'cosine_samples': 2000,
  'metrics_recorded': True,
  'logit_update_mode': 'full',
  'use_global_feedback': True,
  'adaptive_margin': False,
  'initial_margin': 150,
  'log_file': 'desired_margin_sensitivity/CIFAR10/desired_margin_150/train.log'},
 {'group': 'desired_margin_sensitivity',
  'dataset': 'CIFAR10',
  'experiment': '固定 margin=200',
  'epoch': 30,
  'clients': 10,
  'malicious_clients': [0],
  'acc_mean': 65.95700000000001,
  'acc_std': 5.74423778910464,
  'asr_mean': 78.866,
  'asr_std': 3.9713087683197497,
  'l2_mean': 197.20755,
  'cosine_mean': 0.83158866,
  'selected_samples': 2000,
  'cosine_samples': 2000,
  'metrics_recorded': True,
  'logit_update_mode': 'full',
  'use_global_feedback': True,
  'adaptive_margin': False,
  'initial_margin': 200,
  'log_file': 'desired_margin_sensitivity/CIFAR10/desired_margin_200/train.log'},
 {'group': 'logits_only_target',
  'dataset': 'CIFAR10',
  'experiment': '仅优化目标 logit',
  'epoch': 30,
  'clients': 10,
  'malicious_clients': [0],
  'acc_mean': 69.291,
  'acc_std': 5.138753523742677,
  'asr_mean': 64.265,
  'asr_std': 3.795884467273585,
  'l2_mean': 96.343748,
  'cosine_mean': 0.52627032,
  'selected_samples': 2000,
  'cosine_samples': 2000,
  'metrics_recorded': True,
  'logit_update_mode': 'target_only',
  'use_global_feedback': True,
  'adaptive_margin': True,
  'initial_margin': 50,
  'log_file': 'target_only/CIFAR10_2026-10-28_20:17.txt'},
 {'group': 'no_adaptive',
  'dataset': 'CIFAR10',
  'experiment': '无全局反馈（持续递增）',
  'epoch': 30,
  'clients': 10,
  'malicious_clients': [0],
  'acc_mean': 64.321,
  'acc_std': 6.774009890751564,
  'asr_mean': 85.826,
  'asr_std': 3.7469875307446037,
  'l2_mean': 296.89988,
  'cosine_mean': 0.83720728,
  'selected_samples': 2000,
  'cosine_samples': 2000,
  'metrics_recorded': True,
  'logit_update_mode': 'full',
  'use_global_feedback': False,
  'adaptive_margin': True,
  'initial_margin': 50,
  'log_file': 'no_adaptive/CIFAR10_2026-10-28_20:18.txt'}]


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
OPTIONAL_NUMBER = rf"(?:{NUMBER}|N/A)"
TEST_MARKER = "--------------客户端测试----------------"
TEST_RESULT = re.compile(
    rf"^Client\s+(\d+)\s*\|\s*Epoch\s+(\d+)\s*\|\s*"
    rf"Acc:\s*({NUMBER})\s*,\s*ASR:\s*({NUMBER})",
    re.MULTILINE,
)
DEVIATION_RESULT = re.compile(
    rf"^Client\s+(\d+)\s*\|\s*Epoch\s+(\d+)\s*\|\s*Logit Deviation\s*\|\s*"
    rf"l2_mean:\s*({OPTIONAL_NUMBER})\s*\|\s*"
    rf"cosine_mean:\s*({OPTIONAL_NUMBER})\s*\|\s*"
    rf"selected_samples:\s*(\d+)\s*\|\s*cosine_samples:\s*(\d+)\s*$",
    re.MULTILINE,
)


def optional_number(value, minimum=0, maximum=None):
    if value == "N/A":
        return None
    result = float(value)
    if (not math.isfinite(result) or result < minimum
            or (maximum is not None and result > maximum)):
        raise ValueError(f"无效的指标数值：{value}")
    return result


def experiment_name(group, config):
    if group == "desired_margin_sensitivity":
        if not config.get("use_global_feedback", True):
            raise ValueError("固定/自适应 margin 目录中发现关闭全局反馈的配置")
        if config.get("adaptive_margin", True):
            return "自适应 margin"
        return f"固定 margin={config.get('desired_margin', 50):g}"
    if group == "logits_only_target":
        if config.get("logit_update_mode", "full") != "target_only":
            raise ValueError("logits_only_target 配置没有启用 target_only")
        return "仅优化目标 logit"
    if config.get("use_global_feedback", True):
        raise ValueError("no_adaptive 配置没有关闭全局反馈")
    return "无全局反馈（持续递增）"


def weighted_deviation(records, field, count_field="selected_samples"):
    selected = [record for record in records if record[count_field] > 0]
    if not selected or any(record[field] is None for record in selected):
        return None
    total = sum(record[count_field] for record in selected)
    return sum(record[field] * record[count_field] for record in selected) / total


def read_final_result(log_path, config, group, outputs_root):
    text = log_path.read_text(encoding="utf-8", errors="strict")
    expected_epoch, expected_clients = config["epochs"], config["clients"]
    malicious = config["malicious"]
    if any(type(value) is not int or value < 1 for value in (expected_epoch, expected_clients)):
        raise ValueError(f"{log_path}: epochs 和 clients 必须为正整数")
    if (not isinstance(malicious, list) or not malicious
            or any(type(cid) is not int or not 0 <= cid < expected_clients for cid in malicious)
            or len(malicious) != len(set(malicious))):
        raise ValueError(f"{log_path}: 恶意客户端配置无效")
    if TEST_MARKER not in text:
        raise ValueError(f"{log_path}: 没有正式客户端测试记录")

    # 排除 Local Epoch；只读取最后一个正式测试块，不将 Artificial ASR 当成 ASR。
    final_block = text.rsplit(TEST_MARKER, 1)[1]
    clients = {}
    for cid, epoch, acc, asr in TEST_RESULT.findall(final_block):
        cid, epoch = int(cid), int(epoch)
        if epoch != expected_epoch:
            raise ValueError(f"{log_path}: 最后测试为第 {epoch} 轮，预期 {expected_epoch} 轮")
        if cid in clients:
            raise ValueError(f"{log_path}: 最后一轮客户端 {cid} 的结果重复")
        acc, asr = float(acc), float(asr)
        if not (0 <= acc <= 100 and 0 <= asr <= 100):
            raise ValueError(f"{log_path}: ACC/ASR 超出百分比范围")
        clients[cid] = (acc, asr)
    if set(clients) != set(range(expected_clients)):
        raise ValueError(f"{log_path}: 最后一轮正式测试不完整")

    deviations = {}
    for cid, epoch, l2, cosine, count, cosine_count in DEVIATION_RESULT.findall(text):
        cid, epoch = int(cid), int(epoch)
        if epoch != expected_epoch:
            continue
        if cid in deviations:
            raise ValueError(f"{log_path}: 最后一轮客户端 {cid} 的指标重复")
        count, cosine_count = int(count), int(cosine_count)
        l2, cosine = optional_number(l2), optional_number(cosine, -1, 1)
        if cosine_count > count:
            raise ValueError(f"{log_path}: 余弦有效样本数超过选中样本数")
        if (count > 0) != (l2 is not None) or (cosine_count > 0) != (cosine is not None):
            raise ValueError(f"{log_path}: 指标与有效样本数不一致")
        deviations[cid] = {"l2_mean": l2, "cosine_mean": cosine,
                           "selected_samples": count, "cosine_samples": cosine_count}
    if deviations and set(deviations) != set(malicious):
        raise ValueError(f"{log_path}: 最后一轮恶意客户端的 L2/余弦记录不完整")
    if not deviations and "l2_mean:" in text:
        raise ValueError(f"{log_path}: 缺少有效的最后一轮 L2/余弦记录")
    # 旧日志没有原始向量，无法从聚合 MSE 还原平均 L2 或余弦。
    records = list(deviations.values())
    acc_values, asr_values = zip(*(clients[cid] for cid in sorted(clients)))
    return {
        "group": group,
        "dataset": config["dataset"],
        "experiment": experiment_name(group, config),
        "epoch": expected_epoch,
        "clients": expected_clients,
        "malicious_clients": sorted(malicious),
        "acc_mean": statistics.mean(acc_values),
        "acc_std": statistics.stdev(acc_values) if expected_clients > 1 else 0.0,
        "asr_mean": statistics.mean(asr_values),
        "asr_std": statistics.stdev(asr_values) if expected_clients > 1 else 0.0,
        "l2_mean": weighted_deviation(records, "l2_mean"),
        "cosine_mean": weighted_deviation(records, "cosine_mean", "cosine_samples"),
        "selected_samples": sum(record["selected_samples"] for record in records) if records else None,
        "cosine_samples": sum(record["cosine_samples"] for record in records) if records else None,
        "metrics_recorded": bool(deviations),
        "logit_update_mode": config.get("logit_update_mode", "full"),
        "use_global_feedback": config.get("use_global_feedback", True),
        "adaptive_margin": config.get("adaptive_margin", True),
        "initial_margin": config.get("desired_margin", 50),
        "log_file": str(log_path.relative_to(outputs_root)),
    }


def result_key(row):
    group = EXPERIMENT_GROUPS.index(row["group"])
    fixed = row["group"] == "desired_margin_sensitivity" and not row["adaptive_margin"]
    return (row["dataset"], group, fixed, row["initial_margin"] if fixed else 0, row["log_file"])


def load_results(outputs_root):
    # 缓存输出只依赖标准库；仅重新读日志时加载 PyYAML。
    import yaml

    results = []
    for group in EXPERIMENT_GROUPS:
        directories = [outputs_root / name for name in EXPERIMENT_DIRECTORIES[group]]
        configs = sorted({path for directory in directories
                          for path in directory.rglob("config.yaml")})
        if not configs:
            raise FileNotFoundError("没有找到实验配置：" + " 或 ".join(map(str, directories)))
        for config_path in configs:
            with config_path.open(encoding="utf-8") as handle:
                config = yaml.safe_load(handle)
            if not isinstance(config, dict):
                raise ValueError(f"{config_path}: 配置必须为参数字典")
            train_log = config_path.parent / "train.log"
            if train_log.is_file():
                logs = [train_log]
            else:
                logs = [path for path in sorted(config_path.parent.glob("*.txt"))
                        if TEST_MARKER in path.read_text(encoding="utf-8")]
            if not logs:
                raise FileNotFoundError(f"{config_path.parent}: 没有 train.log 或实验 txt 日志")
            for log_path in logs:
                result = read_final_result(log_path, config, group, outputs_root)
                if len(logs) > 1:
                    result["experiment"] += f" [{log_path.stem}]"
                results.append(result)
    return sorted(results, key=result_key)


def validate_results(data):
    if not isinstance(data, list) or not data:
        raise ValueError("DATA 必须为非空结果列表")
    identities = set()
    for row in data:
        if row["group"] not in EXPERIMENT_GROUPS or not row["dataset"]:
            raise ValueError("DATA 中的实验分组或数据集无效")
        identity = (row["group"], row["log_file"])
        if identity in identities:
            raise ValueError(f"DATA 中实验重复：{identity}")
        identities.add(identity)
        for field in ("epoch", "clients"):
            if type(row[field]) is not int or row[field] < 1:
                raise ValueError(f"DATA 中的 {field} 必须为正整数")
        for field in ("acc_mean", "asr_mean", "acc_std", "asr_std"):
            value = row[field]
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"DATA 中的 {field} 无效")
        if row["acc_mean"] > 100 or row["asr_mean"] > 100:
            raise ValueError("DATA 中 ACC/ASR 应为百分比")
        recorded = row.get("metrics_recorded", "l2_mean" in row or "cosine_mean" in row)
        if type(recorded) is not bool:
            raise ValueError("DATA 中 metrics_recorded 必须为布尔值")
        for field in ("selected_samples", "cosine_samples"):
            value = row.get(field)
            if value is None and not recorded:
                continue
            if type(value) is not int or value < 0:
                raise ValueError(f"DATA 中的 {field} 必须为非负整数")
        l2, cosine = row.get("l2_mean"), row.get("cosine_mean")
        if l2 is not None and (not isinstance(l2, (int, float)) or not math.isfinite(l2) or l2 < 0):
            raise ValueError("DATA 中的 l2_mean 无效")
        if cosine is not None and (not isinstance(cosine, (int, float))
                                   or not math.isfinite(cosine) or not -1 <= cosine <= 1):
            raise ValueError("DATA 中的 cosine_mean 无效")
        if recorded:
            if row["cosine_samples"] > row["selected_samples"]:
                raise ValueError("余弦有效样本数超过选中样本数")
            if ((row["selected_samples"] > 0) != (l2 is not None)
                    or (row["cosine_samples"] > 0) != (cosine is not None)):
                raise ValueError("DATA 中指标与有效样本数不一致")
    return sorted(data, key=result_key)


def update_code_cache(results):
    """仅替换本文件的 DATA 赋值，保留其余代码。"""
    script = Path(__file__).resolve()
    source = script.read_text(encoding="utf-8")
    assignments = [node for node in ast.parse(source).body if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == "DATA"
                           for target in node.targets)]
    if len(assignments) != 1:
        raise ValueError("无法定位唯一的 DATA 数据块")
    node = assignments[0]
    lines = source.splitlines(keepends=True)
    block = "DATA = " + pprint.pformat(results, width=110, sort_dicts=False) + "\n"
    updated = "".join(lines[:node.lineno - 1]) + block + "".join(lines[node.end_lineno:])
    compile(updated, str(script), "exec")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=script.parent,
                                         prefix=".experiment-cache-", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(updated)
        temporary.chmod(stat.S_IMODE(script.stat().st_mode))
        os.replace(temporary, script)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def display_width(value):
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in value)


def render_table(table):
    widths = [max(display_width(row[column]) for row in table) for column in range(len(table[0]))]
    for index, row in enumerate(table):
        print("  ".join(value + " " * (width - display_width(value))
                        for value, width in zip(row, widths)).rstrip())
        if index == 0:
            print("  ".join("-" * width for width in widths))


def format_deviation(value):
    return "N/A" if value is None else f"{value:.4f}"


def print_table(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)
    for dataset, entries in grouped.items():
        print(f"\n{dataset}：最后一轮实验结果")
        table = [["实验组", "轮次", "ACC(%)", "ASR(%)", "平均 L2 距离", "平均余弦相似度"]]
        for row in entries:
            cosine = row.get("cosine_mean")
            table.append([
                row["experiment"], str(row["epoch"]),
                f"{row['acc_mean']:.2f} ± {row['acc_std']:.2f}",
                f"{row['asr_mean']:.2f} ± {row['asr_std']:.2f}",
                format_deviation(row.get("l2_mean")),
                "N/A" if cosine is None else f"{cosine:.6f}",
            ])
        render_table(table)
    print("\nACC/ASR：最后一轮所有客户端均值 ± 客户端间样本标准差。")
    print("L2/余弦：最后一轮选中样本修改前后 logits 的逐样本指标均值，不进行跨轮平均。")
    print("多个恶意客户端分别按选中样本数/余弦有效样本数加权；零向量样本不参与余弦平均。")
    if any(not row.get("metrics_recorded", "l2_mean" in row or "cosine_mean" in row) for row in rows):
        print("N/A：旧实验未记录 L2/余弦，无法从 MSE 还原；新实验完成后运行 --refresh-cache。")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-root", type=Path, default=OUTPUTS_ROOT,
                        help="Outputs 目录；仅重新读取日志时使用")
    parser.add_argument("--from-logs", action="store_true", help="忽略 DATA，重新读取三个实验目录")
    parser.add_argument("--refresh-cache", action="store_true", help="重新读取日志并更新本文件中的 DATA")
    parser.add_argument("--print-data", action="store_true", help="额外输出可复制的 DATA 数据块")
    args = parser.parse_args()
    from_logs = args.from_logs or args.refresh_cache or not DATA
    results = load_results(args.outputs_root.resolve()) if from_logs else DATA
    results = validate_results(results)
    if args.refresh_cache:
        update_code_cache(results)
    print("数据来源：" + ("实验日志" if from_logs else "代码中的 DATA 缓存（未读取日志）"))
    print_table(results)
    if args.refresh_cache:
        print("已更新代码中的 DATA 缓存，下次直接运行即可。")
    if args.print_data:
        print("\nDATA = " + pprint.pformat(results, width=110, sort_dicts=False))


if __name__ == "__main__":
    main()
