#!/bin/bash

CONFIG="configs/GTSRB.yaml"
BACKUP="config_backup.yaml"
OUTDIR="Positions"

# 创建输出目录
mkdir -p "$OUTDIR"

# === 第一次先备份原始 config.yaml ===
cp "$CONFIG" "$BACKUP"

# 定义9个位置，去掉方括号
positions=(
    "0,0"
    "0,10"
    "0,20"
    "10,0"
    "10,10"
    "10,20"
    "20,0"
    "20,10"
    "20,20"
)

# 遍历9个位置
for pos in "${positions[@]}"; do
    echo "=============================================="
    echo "=== Running with position = $pos ==="
    echo "=== Start Time: $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo "=============================================="

    # 使用 yq 修改 positions 字段为当前的 position
    yq ".positions = [$pos]" "$BACKUP" > "$CONFIG"

    # 运行程序并保存输出（文件名中包含位置坐标）
    python main.py > "${OUTDIR}/position_${pos}.txt" 2>&1
done

# 恢复原始配置
mv "$BACKUP" "$CONFIG"

echo "全部运行完成！输出已保存到目录: $OUTDIR/"