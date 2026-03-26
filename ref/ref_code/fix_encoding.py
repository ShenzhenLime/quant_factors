#!/usr/bin/env python
# coding: utf-8
"""
修复 ref_code 目录下的 .py 文件编码问题
这些文件从 .txt 重命名为 .py 后出现乱码，需要用正确的编码重新保存
"""

import os
from pathlib import Path
import chardet


def detect_encoding(file_path):
    """
    检测文件编码
    
    Args:
        file_path: 文件路径
        
    Returns:
        检测到的编码格式
    """
    with open(file_path, 'rb') as f:
        result = chardet.detect(f.read(10000))  # 只读取前 10KB 来检测
    return result['encoding']


def fix_file_encoding(file_path):
    """
    修复文件编码，将检测到的编码转换为 UTF-8
    
    Args:
        file_path: 文件路径
    """
    # 尝试多种可能的编码
    encodings_to_try = ['gbk', 'gb2312', 'gb18030', 'utf-8', 'cp936']
    
    content = None
    detected_encoding = None
    
    # 先尝试检测编码
    try:
        detected_encoding = detect_encoding(file_path)
        print(f"  检测到编码：{detected_encoding}")
    except Exception as e:
        print(f"  编码检测失败：{e}")
    
    # 尝试读取文件
    for encoding in encodings_to_try:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            print(f"  使用 {encoding} 编码成功读取")
            break
        except (UnicodeDecodeError, LookupError):
            continue
    
    if content is None:
        print(f"  无法读取文件，跳过")
        return False
    
    # 用 UTF-8 重新保存
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  已转换为 UTF-8 编码")
        return True
    except Exception as e:
        print(f"  保存失败：{e}")
        return False


def process_py_files(root_dir):
    """
    处理目录下所有的 .py 文件
    
    Args:
        root_dir: 根目录路径
    """
    root_path = Path(root_dir)
    success_count = 0
    total_count = 0
    
    # 递归遍历所有 .py 文件
    for py_file in root_path.rglob("*.py"):
        total_count += 1
        print(f"\n处理文件：{py_file.name}")
        
        if fix_file_encoding(py_file):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"完成！共处理 {total_count} 个文件，成功 {success_count} 个")
    return success_count, total_count


if __name__ == "__main__":
    # ref_code 目录路径
    ref_code_dir = Path(__file__).parent
    
    print(f"开始修复编码问题")
    print(f"处理目录：{ref_code_dir}")
    print("=" * 60)
    
    # 执行修复
    success, total = process_py_files(ref_code_dir)
    
    if success == total:
        print("\n所有文件编码修复成功！")
    else:
        print(f"\n部分文件修复失败，请手动检查")
