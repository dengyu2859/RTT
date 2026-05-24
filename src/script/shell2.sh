#!/bin/bash

CONFIG="configs/GTSRB.yaml"
BACKUP="config_backup.yaml"
OUTDIR="malicious_clients"

# 创建输出目录
mkdir -p "$OUTDIR"

# 备份原始配置
cp "$CONFIG" "$BACKUP"

# 循环长度：从 1 到 10
for n in $(seq 1 10); do
    echo "=============================================="
    echo "=== Running with malicious clients [0..$((n-1))] ==="
    echo "=== Start Time: $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo "=============================================="

    # 生成 [0,1,2,...,n-1]
    MAL_LIST=$(seq -s, 0 $((n-1)))

    # 使用 yq 修改 malicious 字段
    yq ".malicious = [$MAL_LIST]" "$BACKUP" > "$CONFIG"

    # 运行程序并保存输出
    python main.py > "${OUTDIR}/output_malicious_${n}.txt" 2>&1
done

# 恢复原始配置
mv "$BACKUP" "$CONFIG"

echo "全部运行完成！输出已保存到目录: $OUTDIR/"
