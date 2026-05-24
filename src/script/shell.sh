#!/bin/bash

CONFIG="configs/CIFAR10.yaml"
BACKUP="config_backup.yaml"
OUTDIR="betas"

# 创建输出目录
mkdir -p "$OUTDIR"

# === 第一次先备份原始 config.yaml ===
cp "$CONFIG" "$BACKUP"

# 遍历 a = 0.1 到 1.0
for n in $(seq -f "%.1f" 0.1 0.1 0.9); do
    echo "=============================================="
    echo "=== Running with beta = $n ==="
    echo "=== Start Time: $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo "=============================================="

    # 使用 yq 修改 beta 字段为浮点数
    yq ".beta = $n" "$BACKUP" > "$CONFIG"

    # 运行程序并保存输出（文件名中的小数点可以保留）
    python main.py > "${OUTDIR}/output_beta_${n}.txt" 2>&1
done

# 恢复原始配置
mv "$BACKUP" "$CONFIG"

echo "全部运行完成！输出已保存到目录: $OUTDIR/"
