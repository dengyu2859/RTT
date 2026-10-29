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

# 已提取的实验结果，可以直接修改数值；默认运行只使用这里的数据。
# 新实验完成后运行 --refresh-cache，自动重新提取并更新此数据块。
# acc_std/asr_std 是同一轮客户端之间的样本标准差，并非多次运行标准差。
# 下方旧实验的 MSE 原始缓存保留作历史记录，不再计算或输出。
# 新实验刷新后保存 l2_mean/cosine_mean；旧实验无法还原的新指标显示 N/A。
DATA = [{'group': 'desired_margin_sensitivity',
  'dataset': 'CIFAR10',
  'experiment': '自适应 margin',
  'epoch': 30,
  'clients': 10,
  'malicious_clients': [0],
  'acc_mean': 68.182,
  'acc_std': 5.283556882757422,
  'asr_mean': 70.777,
  'asr_std': 2.9780159316042747,
  'logit_deviation': 2400.9402,
  'logit_deviation_by_epoch': [0.0,
                               434.31314,
                               701.71124,
                               1172.3599,
                               1816.5692,
                               1657.3766,
                               1471.9523,
                               1294.0815,
                               1129.2538,
                               979.42438,
                               843.31482,
                               1389.6429,
                               1249.1056,
                               1089.1085,
                               1695.7482,
                               1536.363,
                               2203.0103,
                               2029.0502,
                               1832.5756,
                               1642.8997,
                               2319.3557,
                               2144.2478,
                               1950.8217,
                               1761.8488,
                               2444.8315,
                               2271.2632,
                               2077.0605,
                               1886.5059,
                               2574.0879,
                               2400.9402],
  'selected_samples_by_epoch': [2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000],
  'deviation_before': 33.107189,
  'deviation_after': 2008.4988,
  'selected_samples': 2000,
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
  'acc_mean': 74.106,
  'acc_std': 3.923145563334494,
  'asr_mean': 28.351,
  'asr_std': 1.6360959629557184,
  'logit_deviation': 158.12,
  'logit_deviation_by_epoch': [0.0,
                               240.04594,
                               204.86348,
                               185.66151,
                               177.1153,
                               172.86638,
                               170.47028,
                               167.74471,
                               166.75854,
                               165.01762,
                               163.60942,
                               162.03331,
                               161.23535,
                               159.80074,
                               159.00398,
                               157.98871,
                               157.45549,
                               157.09822,
                               156.35608,
                               156.61279,
                               156.49484,
                               156.40776,
                               157.13579,
                               157.06924,
                               157.36431999999996,
                               156.9679,
                               157.37215,
                               157.80931,
                               157.754,
                               158.12],
  'selected_samples_by_epoch': [2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000],
  'deviation_before': 27.907578,
  'deviation_after': 110.6946,
  'selected_samples': 2000,
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
  'acc_mean': 70.753,
  'acc_std': 4.469039295220195,
  'asr_mean': 55.849000000000004,
  'asr_std': 2.867590742533995,
  'logit_deviation': 742.26477,
  'logit_deviation_by_epoch': [0.0,
                               997.6806,
                               938.65381,
                               904.14673,
                               887.87079,
                               876.73071,
                               868.88757,
                               861.08533,
                               852.55463,
                               846.2276,
                               840.4826,
                               835.42749,
                               829.16119,
                               823.16608,
                               816.75018,
                               809.88232,
                               803.24237,
                               797.01282,
                               790.36456,
                               784.5517,
                               778.75427,
                               772.89313,
                               768.15076,
                               763.70422,
                               758.8446,
                               755.4679,
                               751.57452,
                               748.51239,
                               744.83691,
                               742.26477],
  'selected_samples_by_epoch': [2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000],
  'deviation_before': 22.07135,
  'deviation_after': 618.48688,
  'selected_samples': 2000,
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
  'acc_mean': 67.227,
  'acc_std': 5.900316469855192,
  'asr_mean': 74.168,
  'asr_std': 4.605279096380106,
  'logit_deviation': 2231.2827,
  'logit_deviation_by_epoch': [0.0,
                               2628.5691,
                               2609.938,
                               2578.2483,
                               2556.2061,
                               2538.176,
                               2524.9760999999994,
                               2512.1035,
                               2497.1616,
                               2482.3193,
                               2468.0586,
                               2452.0339,
                               2436.7527,
                               2421.5039,
                               2406.95,
                               2391.3938,
                               2374.8633,
                               2360.2544,
                               2347.311,
                               2334.2036,
                               2321.8638,
                               2309.0405,
                               2297.8357,
                               2285.7595,
                               2275.624,
                               2266.8777,
                               2259.6558,
                               2249.9597,
                               2240.4548,
                               2231.2827],
  'selected_samples_by_epoch': [2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000],
  'deviation_before': 22.652475,
  'deviation_after': 1922.276,
  'selected_samples': 2000,
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
  'acc_mean': 65.918,
  'acc_std': 5.619442044268175,
  'asr_mean': 78.77799999999999,
  'asr_std': 3.513607832413856,
  'logit_deviation': 3895.2695,
  'logit_deviation_by_epoch': [0.0,
                               4222.478,
                               4240.2222,
                               4218.5078,
                               4201.4971,
                               4187.5122,
                               4173.9087,
                               4160.0698,
                               4145.3726,
                               4129.8657,
                               4115.4277,
                               4099.6577,
                               4084.0151,
                               4069.6023,
                               4055.3689,
                               4041.5339,
                               4027.5544,
                               4016.2061,
                               4004.0137,
                               3993.22,
                               3981.1775,
                               3969.7048,
                               3959.4409,
                               3949.8877,
                               3940.9648,
                               3931.1541,
                               3921.2576,
                               3912.4873,
                               3902.26,
                               3895.2695],
  'selected_samples_by_epoch': [2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000],
  'deviation_before': 33.475758,
  'deviation_after': 3348.1943,
  'selected_samples': 2000,
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
  'acc_mean': 69.161,
  'acc_std': 5.211961562917876,
  'asr_mean': 64.049,
  'asr_std': 3.1498834193594467,
  'logit_deviation': 928.23071,
  'logit_deviation_by_epoch': [0.0,
                               373.45486,
                               495.7594,
                               588.58105,
                               656.64087,
                               706.51538,
                               694.18707,
                               680.86603,
                               666.47467,
                               714.27533,
                               750.57556,
                               778.86487,
                               801.49872,
                               819.94678,
                               835.29449,
                               848.27087,
                               859.37909,
                               868.95599,
                               877.33838,
                               884.70361,
                               891.24951,
                               897.09912,
                               902.37579,
                               907.13019,
                               911.45117,
                               915.39642,
                               919.0108,
                               922.33679,
                               925.401,
                               928.23071],
  'selected_samples_by_epoch': [2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000],
  'deviation_before': 11.180118,
  'deviation_after': 880.76202,
  'selected_samples': 2000,
  'logit_update_mode': 'target_only',
  'use_global_feedback': True,
  'adaptive_margin': True,
  'initial_margin': 50,
  'log_file': 'logits_only_target/CIFAR10_2026-10-27_16:28.txt'},
 {'group': 'no_adaptive',
  'dataset': 'CIFAR10',
  'experiment': '无全局反馈（持续递增）',
  'epoch': 30,
  'clients': 10,
  'malicious_clients': [0],
  'acc_mean': 64.519,
  'acc_std': 6.567987092286683,
  'asr_mean': 85.15,
  'asr_std': 3.5686194031368,
  'logit_deviation': 8813.7871,
  'logit_deviation_by_epoch': [0.0,
                               1232.6631,
                               1951.7382,
                               2826.804,
                               3692.7344,
                               4460.2163,
                               5107.3047,
                               5642.6758,
                               6083.4702,
                               6449.0762,
                               6755.5752,
                               7016.4385,
                               7240.2002,
                               7433.7705,
                               7603.1279,
                               7751.995099999999,
                               7884.3721,
                               8002.5874,
                               8108.6162,
                               8204.4941,
                               8291.125,
                               8369.8799,
                               8442.115199999998,
                               8508.2891,
                               8569.2764,
                               8625.3779,
                               8677.9736,
                               8726.5498,
                               8771.6396,
                               8813.7871],
  'selected_samples_by_epoch': [2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000,
                                2000],
  'deviation_before': 70.454254,
  'deviation_after': 7511.6494,
  'selected_samples': 2000,
  'logit_update_mode': 'full',
  'use_global_feedback': False,
  'adaptive_margin': True,
  'initial_margin': 50,
  'log_file': 'no_adaptive/CIFAR10_2026-10-27_17:03.txt'}]


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
        directory = outputs_root / group
        configs = sorted(directory.rglob("config.yaml"))
        if not configs:
            raise FileNotFoundError(f"没有找到实验配置：{directory}")
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
