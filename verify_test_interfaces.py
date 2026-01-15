#!/usr/bin/env python3
"""验证测试文件需要的接口是否都存在"""

import ast
import inspect

print("=" * 70)
print("验证 test_relay_unified.py 需要的接口")
print("=" * 70)

# 读取 connector.py，检查导出的类和方法
print("\n1. 分析 connector.py 源代码...")
with open('sglang_omni/relay/nixl/connector.py', 'r') as f:
    connector_code = f.read()

tree = ast.parse(connector_code)

# 提取类定义
classes = {}
for node in ast.walk(tree):
    if isinstance(node, ast.ClassDef):
        classes[node.name] = {
            'methods': [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
            'line': node.lineno
        }

print("\n2. 找到的类:")
for cls_name, info in sorted(classes.items()):
    if not cls_name.startswith('_'):  # 只显示公共类
        print(f"   ✓ {cls_name} (行 {info['line']})")

# 检查测试需要的类
print("\n3. 检查测试需要的类:")
required_classes = {
    'RdmaMetadata': ['to_descriptors'],
    'SHMMetadata': ['to_descriptors'],
    'Connector': ['create_readable', 'begin_read', 'create_writable', 'begin_write'],
    'ReadableOperation': ['metadata', 'wait_for_completion'],
    'ReadOperation': ['wait_for_completion', 'results'],
}

all_good = True
for cls_name, required_methods in required_classes.items():
    if cls_name in classes:
        print(f"\n   ✓ {cls_name} 存在")
        cls_methods = classes[cls_name]['methods']
        for method in required_methods:
            if method in cls_methods:
                print(f"      ✓ {method}()")
            else:
                print(f"      ✗ {method}() 不存在！")
                all_good = False
    else:
        print(f"\n   ✗ {cls_name} 不存在！")
        all_good = False

# 检查 __init__.py
print("\n4. 检查 __init__.py 导出...")
with open('sglang_omni/relay/nixl/__init__.py', 'r') as f:
    init_content = f.read()

exports = []
if '__all__' in init_content:
    # 提取 __all__ 列表
    for line in init_content.split('\n'):
        if '"' in line and not line.strip().startswith('#'):
            for item in line.split('"'):
                if item.strip() and not item.strip() in [',', '[', ']']:
                    if item.strip() not in exports:
                        exports.append(item.strip())

print("   导出的符号:")
for exp in sorted(exports):
    print(f"      • {exp}")

# 检查必需的导出
required_exports = ['RdmaMetadata', 'SHMMetadata', 'Connector', 
                   'ReadableOperation', 'ReadOperation']
print("\n   检查必需导出:")
for exp in required_exports:
    if exp in exports:
        print(f"      ✓ {exp}")
    else:
        print(f"      ✗ {exp} 未导出！")
        all_good = False

print("\n" + "=" * 70)
if all_good:
    print("✅ 所有测试需要的接口都存在且已导出！")
    print("\n测试应该可以正常运行：")
    print("   pytest tests/test_relay_unified.py -v")
else:
    print("❌ 有些接口缺失，测试可能会失败")
print("=" * 70)

