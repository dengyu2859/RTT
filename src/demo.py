# import numpy as np
# import matplotlib.pyplot as plt
#
# size = 256  # 图片尺寸（正方形）
#
# # 上深下浅的渐变
# gradient = np.linspace(1.0, 0.3, size)[:, None]
# gradient = np.repeat(gradient, size, axis=1)
#
# # 构造粉色 RGB
# pink_image = np.zeros((size, size, 3))
# pink_image[:, :, 0] = 1.0              # R
# pink_image[:, :, 1] = gradient * 0.6   # G
# pink_image[:, :, 2] = gradient * 0.7   # B
# # 保存为图片（无白边）
# plt.figure(figsize=(size / 100, size / 100), dpi=100)
# plt.imshow(pink_image)
# plt.axis("off")
# plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
# plt.show()
# # plt.savefig(
# #     "pink_square_top_to_bottom.png",
# #     bbox_inches="tight",
# #     pad_inches=0
# # )
# plt.close()


import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline

size = 256  # 图片尺寸（正方形）
np.random.seed(42)  # 固定随机性，方便复现

# =====================
# 背景：上深下浅粉色
# =====================
gradient = np.linspace(1.0, 0.3, size)[:, None]
gradient = np.repeat(gradient, size, axis=1)

pink_image = np.zeros((size, size, 3))
pink_image[:, :, 0] = 1.0
pink_image[:, :, 1] = gradient * 0.6
pink_image[:, :, 2] = gradient * 0.7

plt.figure(figsize=(size / 100, size / 100), dpi=100)
plt.imshow(pink_image)
plt.axis("off")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

# =====================
# 不规则粗曲线
# =====================
def draw_random_curve(y_center, thickness=3):
    x = np.linspace(0, size - 1, 8)
    y = y_center + np.random.randint(-30, 30, size=8)

    spline = make_interp_spline(x, y, k=3)
    x_smooth = np.linspace(0, size - 1, 300)
    y_smooth = spline(x_smooth)

    plt.plot(
        x_smooth,
        y_smooth,
        color="black",
        linewidth=thickness,
        solid_capstyle="round"
    )

# 画 2～3 根曲线
draw_random_curve(80, thickness=3)
draw_random_curve(140, thickness=4)
draw_random_curve(200, thickness=3)

# plt.show()

plt.savefig(
    "pink_square_top_to_bottom.png",
    bbox_inches="tight",
    pad_inches=0
)
plt.close()

