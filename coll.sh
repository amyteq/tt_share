#!/bin/bash

set -e

# ===== 参数检查 =====
if [[ -z "$1" ]]; then
  echo "Usage: $0 <target_directory>"
  exit 1
fi

TARGET_DIR="$1"
DOWNLOAD_DIR="$HOME/Downloads"

# 如果目标目录不存在，则创建
mkdir -p "$TARGET_DIR"

# ===== 文件名数组 =====
src_files=(
  "92ba6a.md"
  "a295e9.md"
  "0e644e.md"
  "9c1c5f.md"
  "424e18.md"
  "26de63.md"
  "641659.md"
  "42d360.md"
  "dd7f5e.md"
  "86e8e5.md"
)

dst_files=(
  "P1.md"
  "P2.md"
  "P3.md"
  "P4.md"
  "P5.md"
  "P6.md"
  "P7.md"
  "P8.md"
  "P9.md"
  "P10.md"
)

# ===== 安全检查：数组长度必须一致 =====
if [[ ${#src_files[@]} -ne ${#dst_files[@]} ]]; then
  echo "Error: source and destination arrays have different lengths"
  exit 1
fi

# ===== 移动文件 =====
for i in "${!src_files[@]}"; do
  src="$DOWNLOAD_DIR/${src_files[$i]}"
  dst="$TARGET_DIR/${dst_files[$i]}"

  if [[ -f "$src" ]]; then
    mv "$src" "$dst"
    echo "Moved: $src -> $dst"
  fi
done
