"""在同一坐标轴比较 DaFKD/FedDF；数据内置，不运行原绘图文件或读取日志。"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
from matplotlib.patches import Patch
import numpy as np


# 从两个原文件的 DATA 原样复制；可在此直接修改合并图的数据。
DATA = {
    "DaFKD": {
        "epochs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        "acc_mean": [30.04, 36.49, 45.75, 53.1, 59.18, 63.44, 66.89, 69.95, 72.47, 74.36, 76.04, 77.19, 78.08, 79.12, 79.76, 80.34, 80.66, 81.03, 81.17, 81.72, 81.84, 82.22, 82.32, 82.35, 82.43, 82.75, 82.88, 82.81, 82.82, 83.09],
        "acc_std": [12.16, 11.99, 11.47, 10.72, 9.71, 8.03, 6.34, 5.13, 4.51, 3.84, 3.37, 3.06, 2.94, 2.59, 2.67, 2.39, 2.39, 2.13, 2.04, 1.94, 1.98, 2.0, 1.94, 1.93, 2.1, 2.12, 1.97, 1.85, 1.98, 1.94],
        "asr_mean": [3.02, 83.13, 84.37, 83.52, 82.99, 85.13, 86.78, 87.58, 87.76, 87.82, 89.14, 89.86, 90.73, 90.72, 91.07, 91.16, 91.59, 91.8, 91.93, 91.96, 92.12, 92.11, 92.38, 92.3, 92.37, 92.42, 92.56, 92.77, 92.69, 92.69],
        "asr_std": [1.91, 5.35, 3.52, 3.97, 4.24, 3.89, 2.64, 2.47, 2.17, 2.08, 2.48, 2.51, 3.0, 2.94, 2.73, 2.56, 2.53, 2.47, 2.43, 2.58, 2.57, 2.47, 2.48, 2.68, 2.58, 2.55, 2.64, 2.75, 2.64, 2.64],
    },
    "FedDF": {
        "epochs": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
        "acc_mean": [30.0, 37.26, 45.18, 52.43, 58.76, 63.82, 67.95, 71.24, 73.86, 76.12, 78.04, 79.61, 80.93, 82.05, 82.94, 83.56, 84.02, 84.35, 84.58, 84.73, 84.86, 84.92, 85.01, 85.08, 85.12, 85.16, 85.1, 85.21, 85.18, 85.24],
        "acc_std": [5.42, 4.76, 5.18, 5.63, 5.91, 6.24, 6.58, 5.87, 6.35, 5.96, 5.42, 5.18, 4.86, 4.53, 4.21, 3.94, 3.62, 3.35, 3.18, 3.04, 2.86, 3.12, 2.94, 3.28, 2.73, 2.61, 2.48, 2.36, 2.24, 2.68],
        "asr_mean": [2.86, 96.0, 97.73, 97.65, 97.77, 97.1, 96.17, 96.37, 95.64, 95.88, 96.11, 96.09, 96.28, 96.69, 96.8, 97.05, 97.08, 97.07, 97.28, 97.38, 97.44, 97.5, 97.56, 97.48, 97.56, 97.41, 97.28, 97.25, 97.06, 96.8],
        "asr_std": [2.9, 2.57, 2.33, 1.85, 1.77, 1.79, 3.09, 2.14, 2.37, 2.39, 2.32, 2.35, 2.23, 2.05, 2.02, 1.77, 1.9, 1.69, 1.73, 1.61, 1.56, 1.52, 1.4, 1.48, 1.35, 1.3, 1.45, 1.28, 1.37, 1.36],
    },
}

OUTPUT_DIR = Path(__file__).resolve().parent / "images"
OUTPUT_PATH = OUTPUT_DIR / "ACC_ASR_plot_DaFKD_FedDF.pdf"

# 末轮数值相对曲线端点的偏移（单位：点）。
ANNOTATION_OFFSETS = {
    ("DaFKD", "ACC"): (0, -35),
    ("DaFKD", "ASR"): (0, 40),
    ("FedDF", "ACC"): (0, -65),
    ("FedDF", "ASR"): (0, 8),
}

def plot_combined():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "font.size": 30,
        "axes.titlesize": 30,
        "axes.labelsize": 30,
        "xtick.labelsize": 30,
        "ytick.labelsize": 30,
        "legend.fontsize": 18,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    figure, ax = plt.subplots(figsize=(8, 6.4))
    legend_handles, legend_labels = [], []
    maximum_epoch = 0
    for method, linestyle in (("DaFKD", "-"), ("FedDF", "--")):
        data = {key: np.asarray(values, dtype=float) for key, values in DATA[method].items()}
        epochs = data["epochs"]
        if (epochs.ndim != 1 or len(epochs) == 0
                or not np.all(np.isfinite(epochs)) or not np.all(np.diff(epochs) > 0)):
            raise ValueError(f"{method}: epochs 必须为有限、严格递增的非空序列")
        maximum_epoch = max(maximum_epoch, epochs[-1])
        marks = sorted(set(range(0, len(epochs), 5)) | {len(epochs) - 1})
        for metric, color, marker in (("ACC", "green", "o"), ("ASR", "#d02825", "s")):
            mean, std = data[f"{metric.lower()}_mean"], data[f"{metric.lower()}_std"]
            if (mean.shape != epochs.shape or std.shape != epochs.shape
                    or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std))
                    or np.any((mean < 0) | (mean > 100)) or np.any(std < 0)):
                raise ValueError(f"{method} {metric}: 均值/标准差的长度或数值无效")
            line, = ax.plot(
                epochs, mean, color=color, linestyle=linestyle,
                linewidth=2, marker=marker, markersize=7, markevery=marks,
                label=f"{method} {metric}",
            )
            # 与原图一致：阴影直接使用已有的 mean ± std，不重新计算或截断。
            ax.fill_between(epochs, mean - std, mean + std, color=color, alpha=0.2)
            # 图例把均值曲线叠在半透明色块上，同时说明线和阴影的含义。
            legend_handles.append((Patch(facecolor=color, alpha=0.2, edgecolor="none"), line))
            legend_labels.append(f"{method} {metric} (Mean ± Std)")
            ax.annotate(
                f"{method}: {mean[-1]:.2f}±{std[-1]:.2f}",
                xy=(epochs[-1], mean[-1]),
                xytext=ANNOTATION_OFFSETS[(method, metric)],
                textcoords="offset points", fontsize=22, color=color,
                va="bottom", ha="right",
            )

    ax.set_xticks(np.arange(0, (np.ceil(maximum_epoch / 5) + 1) * 5, 5))
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylim(-3, 120)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("ACC/ASR(%)")
    # 四项复合图例均包含曲线与阴影；末轮数值继续单独标在图中。
    ax.legend(legend_handles, legend_labels, loc="lower right",
              handler_map={tuple: HandlerTuple(ndivide=1)})
    figure.tight_layout()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT_PATH, format="pdf", bbox_inches="tight", dpi=600)
    plt.show()


if __name__ == "__main__":
    plot_combined()
