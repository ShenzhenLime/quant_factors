#!/usr/bin/env python
# coding: utf-8
"""
批量将 ref_code 目录下的 .txt 文件重命名为 .py 文件
这些 txt 文件实际上是 Python 代码文件
"""

import os
from pathlib import Path


def rename_txt_to_py(root_dir):
    """
    递归遍历目录，将所有 .txt 文件重命名为 .py 文件
    
    Args:
        root_dir: 根目录路径
    """
    root_path = Path(root_dir)
    count = 0
    
    # 递归遍历所有子目录
    for txt_file in root_path.rglob("*.txt"):
        # 构建新的 .py 文件名
        py_file = txt_file.with_suffix('.py')
        
        # 重命名文件
        txt_file.rename(py_file)
        count += 1
        print(f"已重命名：{txt_file.name} -> {py_file.name}")
    
    print(f"\n完成！共重命名 {count} 个文件")
    return count


if __name__ == "__main__":
    # ref_code 目录路径
    ref_code_dir = Path(__file__).parent
    
    print(f"开始处理目录：{ref_code_dir}")
    print("=" * 60)
    
    # 执行重命名
    total = rename_txt_to_py(ref_code_dir)
    
    if total == 0:
        print("未找到需要重命名的 .txt 文件")
