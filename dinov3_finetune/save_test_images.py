"""
将用户提供的附件图像保存为测试图像
请手动将附件图像保存到当前目录并命名为：
- test_image_1.png
- test_image_2.png

然后运行 noise.py 进行测试
"""

import os

print("Current directory:", os.getcwd())
print("\n请按以下步骤操作:")
print("1. 将对话中的第一张附件图保存为: test_image_1.png")
print("2. 将对话中的第二张附件图保存为: test_image_2.png")
print("3. 确保图像保存在当前目录:", os.getcwd())
print("4. 运行命令: python noise.py")

# 检查图像是否存在
if os.path.exists("test_image_1.png"):
    print("✓ test_image_1.png 已找到")
else:
    print("✗ test_image_1.png 未找到")

if os.path.exists("test_image_2.png"):
    print("✓ test_image_2.png 已找到")
else:
    print("✗ test_image_2.png 未找到")
