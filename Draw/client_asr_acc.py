from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUTPUT_DIR = Path(__file__).resolve().parent / "images"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ===== 1. 示例数据（自行替换为实验结果） =====
client_ids = np.arange(1, 21)
architectures = ["ResNet-18", "AlexNet", "VGG-11", "DenseNet-121", "CNN-3"]
# 每种架构的浅色底色；ASR 和 ACC 子图共用这一组配色。
architecture_colors = {
    "ResNet-18": "#EAF2F8",
    "AlexNet": "#F3EDF7",
    "VGG-11": "#FBF3E4",
    "DenseNet-121": "#EAF4EE",
    "CNN-3": "#F8ECEC",
}

# 假设每种架构对应4个客户端
client_arch = np.repeat(architectures, 4)

# 2个恶意客户端
malicious_ids = [2, 14]
malicious = np.isin(client_ids, malicious_ids)
benign = ~malicious

# 示意数据：小数位已随机化后固定；全部 20 个客户端的 ASR/ACC 均值为 83.12%/75.28%。
asr = np.array([
    82.44, 87.21, 81.16, 83.13, 80.61, 82.01, 84.77, 80.95, 85.98, 83.23,
    81.49, 85.59, 82.03, 88.56, 81.64, 83.82, 80.27, 85.80, 80.32, 82.39
])

acc = np.array([
    74.26, 75.81, 73.21, 76.24, 74.71, 76.51, 76.37, 77.78, 72.58, 74.82,
    71.88, 74.50, 75.96, 73.72, 73.54, 74.23, 77.74, 80.98, 80.07, 78.69
])

# ===== 2. 绘图 =====
plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 30,
    "axes.titlesize": 30,
    "axes.labelsize": 30,
    "xtick.labelsize": 25,
    "ytick.labelsize": 30,
    "legend.fontsize": 22,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

fig, axes = plt.subplots(
    2, 1, figsize=(10.5, 8), sharex=True
)

for ax, values, name in zip(
    axes, [asr, acc], ["ASR", "ACC"]
):
    ax.plot(
        client_ids[benign], values[benign],
        "o", linestyle="None",
        color="#1f77b4",
        markeredgewidth=1.6,
        markersize=7,
        label="Benign client"
    )

    ax.plot(
        client_ids[malicious], values[malicious],
        "s", linestyle="None",
        color="#d02825",
        markersize=10,
        label="Malicious client"
    )

    ax.set_xlim(0.5, 20.5)
    ax.set_ylim(-3, 103)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel(f"{name} (%)")
    ax.grid(False)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", direction="out", length=4, top=False, right=False)

# axes[0].set_title("ASR", loc="left", fontweight="bold")
# axes[1].set_title("ACC", loc="left", fontweight="bold")
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(
    handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
    ncol=2, frameon=False
)

# ===== 3. 客户端编号与模型架构 =====
axes[1].set_xticks(client_ids)
axes[1].set_xticklabels(
    [str(i) for i in client_ids]
)

centers = [2.5, 6.5, 10.5, 14.5, 18.5]
for center, arch in zip(centers, architectures):
    axes[1].text(
        center, -0.20, arch,
        transform=axes[1].get_xaxis_transform(),
        ha="center", va="top",
        fontsize=22, clip_on=False
    )

axes[1].set_xlabel("Client ID", labelpad=40)

fig.subplots_adjust(
    left=0.13, right=0.98,
    bottom=0.18, top=0.89, hspace=0.22
)

# 用圆角底色块与留白区分架构；布局确定后计算比例，保持圆角自然。
group_gap = 0.30  # 相邻分区之间的空隙，单位为客户端编号间距。
for ax in axes:
    x_min, x_max = ax.get_xlim()
    x_range = x_max - x_min
    for arch in architectures:
        group_ids = client_ids[client_arch == arch]
        left = (group_ids.min() - 0.5 + group_gap / 2 - x_min) / x_range
        width = (group_ids.max() - group_ids.min() + 1 - group_gap) / x_range
        ax.add_patch(FancyBboxPatch(
            (left, 0.02), width, 0.96,
            boxstyle="round,pad=0,rounding_size=0.012",
            mutation_aspect=ax.bbox.width / ax.bbox.height,
            transform=ax.transAxes,
            facecolor=architecture_colors[arch], edgecolor="none", zorder=0
        ))

fig.savefig(
    OUTPUT_DIR / "client_asr_acc_example.pdf",
    format="pdf", dpi=600, bbox_inches="tight", pad_inches=0.02
)
plt.show()
