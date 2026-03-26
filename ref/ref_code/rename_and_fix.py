#!/usr/bin/env python
# coding: utf-8
"""
批量将 ref_code 目录下的 .txt 文件重命名为 .py 文件并修复编码
这些 txt 文件实际上是 Python 代码文件，通常使用 GBK 编码
"""

import os
from pathlib import Path


def convert_to_utf8(file_path):
    """
    尝试将文件从 GBK 或其他编码转换为 UTF-8
    
    Args:
        file_path: 文件路径
        
    Returns:
        是否成功转换
    """
    # 常见的中文编码
    encodings = ['gbk', 'gb2312', 'gb18030', 'cp936', 'utf-8']
    
    content = None
    used_encoding = None
    
    # 尝试用不同编码读取
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            used_encoding = encoding
            print(f"  ✓ 使用 {encoding} 编码成功读取")
            break
        except (UnicodeDecodeError, LookupError):
            continue
    
    if content is None:
        print(f"  ✗ 无法用任何编码读取")
        return False
    
    # 如果不是 UTF-8，则转换
    if used_encoding != 'utf-8':
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"  ✓ 已转换为 UTF-8 编码")
            return True
        except Exception as e:
            print(f"  ✗ 保存失败：{e}")
            return False
    else:
        print(f"  ✓ 已经是 UTF-8 编码，无需转换")
        return True


def rename_and_fix(root_dir):
    """
    递归遍历目录，将 .txt 文件重命名为 .py 并修复编码
    
    Args:
        root_dir: 根目录路径
    """
    root_path = Path(root_dir)
    rename_count = 0
    fix_count = 0
    
    print("第一步：重命名 .txt 文件为 .py")
    print("=" * 60)
    
    # 递归遍历所有 .txt 文件
    for txt_file in root_path.rglob("*.txt"):
        # 构建新的 .py 文件名
        py_file = txt_file.with_suffix('.py')
        
        # 重命名文件
        txt_file.rename(py_file)
        rename_count += 1
        print(f"✓ 已重命名：{txt_file.name} -> {py_file.name}")
    
    print(f"\n共重命名 {rename_count} 个文件")
    
    print("\n第二步：修复 .py 文件编码")
    print("=" * 60)
    
    # 递归遍历所有 .py 文件（包括刚重命名的）
    for py_file in root_path.rglob("*.py"):
        print(f"\n处理文件：{py_file.name}")
        if convert_to_utf8(py_file):
            fix_count += 1
    
    print(f"\n{'='*60}")
    print(f"完成！")
    print(f"重命名：{rename_count} 个文件")
    print(f"修复编码：{fix_count} 个文件")
    
    return rename_count, fix_count


if __name__ == "__main__":
    # ref_code 目录路径
    ref_code_dir = Path(__file__).parent
    
    print(f"开始处理目录：{ref_code_dir}")
    print("=" * 60)
    
    # 执行重命名和修复
    renames, fixes = rename_and_fix(ref_code_dir)
    
    if renames == 0:
        print("\n未找到需要重命名的 .txt 文件")
        print("如果文件已经是 .py 格式但存在乱码，可以直接运行 fix_encoding.py 脚本")
