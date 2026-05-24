#!/bin/bash

CONFIG="configs/GTSRB.yaml"
BACKUP="config_backup.yaml"
OUTDIR="GTSRB_Seed"

# 创建输出目录
mkdir -p "$OUTDIR"

# === 第一次先备份原始 config.yaml ===
cp "$CONFIG" "$BACKUP"

# 定义固定的种子值
seeds=(
    20250927
    20250937
    20250947
    20250957
    20250967
    20250977
    20250987
    20250997
    20250907
    20250917
)

# 遍历10个种子值
for seed in "${seeds[@]}"; do
    echo "=============================================="
    echo "=== Running with seed = $seed ==="
    echo "=== Start Time: $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo "=============================================="

    # 使用 yq 修改 seed 字段为当前的 seed
    yq ".seed = $seed" "$BACKUP" > "$CONFIG"

    # 运行程序并保存输出（文件名中包含种子值）
    python main.py > "${OUTDIR}/seed_${seed}.txt" 2>&1
done

# 恢复原始配置
mv "$BACKUP" "$CONFIG"

echo "全部运行完成！输出已保存到目录: $OUTDIR/"
