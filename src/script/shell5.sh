#!/bin/bash

# 配置文件列表
CONFIGS=(
    "configs/FashionMNIST.yaml"
    "configs/CIFAR10.yaml"
    "configs/GTSRB.yaml"
    "configs/ImageNet.yaml"
)

# 遍历每个配置文件
for CONFIG_FILE in "${CONFIGS[@]}"; do
    # 提取配置文件名作为输出文件名
    CONFIG_NAME=$(basename "$CONFIG_FILE" .yaml)
    OUTPUT_FILE="${CONFIG_NAME}_ours_image.txt"

    # 打印当前时间和参数
    echo "======================================"
    echo "当前时间: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "运行配置文件: $CONFIG_FILE"
    echo "输出保存到: $OUTPUT_FILE"
    echo "======================================"

    # 运行 Python 脚本，将输出重定向到文件
    python main.py --params "$CONFIG_FILE" > "$OUTPUT_FILE" 2>&1
done
