"""
ComfyUI 图片格式转换节点包
===============================

这是一个专为ComfyUI设计的图片格式转换自定义节点包。

功能特性：
- 🖼️ 批量图片格式转换
- 🔄 支持多种常见格式（PNG, JPEG, WEBP, BMP, TIFF）
- 📐 保持原始图片尺寸
- ⚙️ 可调节图片质量
- 📊 图片格式信息查看

节点列表：
1. ImageFormatConverter - 图片格式转换器
2. ImageFormatInfo - 图片格式信息显示器

作者: AI Assistant
版本: v1.0.0
许可: MIT
"""

import sys
import traceback
# 版本信息
__version__ = "1.0.0"
__author__ = "AI Assistant"
__description__ = "ComfyUI图片格式转换节点包"

# 支持的格式列表
SUPPORTED_FORMATS = ["PNG", "JPEG", "WEBP", "BMP", "TIFF"]

# 尝试导入节点映射
try:
    from .convert_nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    
    # 验证节点映射是否正确
    if not isinstance(NODE_CLASS_MAPPINGS, dict) or not isinstance(NODE_DISPLAY_NAME_MAPPINGS, dict):
        raise ImportError("节点映射格式不正确")
    
    # 打印加载信息
    print(f"✅ 图片格式转换节点包 v{__version__} 加载成功")
    print(f"📦 已加载 {len(NODE_CLASS_MAPPINGS)} 个节点：")
    for node_name, display_name in NODE_DISPLAY_NAME_MAPPINGS.items():
        print(f"   - {display_name} ({node_name})")
    print(f"🔧 支持格式：{', '.join(SUPPORTED_FORMATS)}")
    
except ImportError as e:
    print(f"❌ 导入图片格式转换节点时出错：{e}")
    print("🔍 详细错误信息：")
    traceback.print_exc()
    
    # 创建空的映射以避免ComfyUI报错
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}
    
except Exception as e:
    print(f"❌ 加载图片格式转换节点时发生未知错误：{e}")
    print("🔍 详细错误信息：")
    traceback.print_exc()
    
    # 创建空的映射以避免ComfyUI报错
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

# 导出的内容
__all__ = [
    'NODE_CLASS_MAPPINGS', 
    'NODE_DISPLAY_NAME_MAPPINGS',
    '__version__',
    '__author__',
    '__description__',
    'SUPPORTED_FORMATS'
]

# 模块元数据
def get_extension_info():
    """获取扩展信息"""
    return {
        "name": "Image Format Convert",
        "version": __version__,
        "author": __author__,
        "description": __description__,
        "supported_formats": SUPPORTED_FORMATS,
        "nodes": list(NODE_DISPLAY_NAME_MAPPINGS.keys()),
        "node_count": len(NODE_CLASS_MAPPINGS)
    }

def check_dependencies():
    """检查依赖包是否可用"""
    dependencies = {
        "torch": "PyTorch深度学习框架",
        "numpy": "数值计算库", 
        "PIL": "Python图像处理库"
    }
    
    missing_deps = []
    available_deps = []
    
    for dep_name, dep_desc in dependencies.items():
        try:
            if dep_name == "PIL":
                import PIL
                available_deps.append(f"✅ {dep_name} ({dep_desc})")
            else:
                __import__(dep_name)
                available_deps.append(f"✅ {dep_name} ({dep_desc})")
        except ImportError:
            missing_deps.append(f"❌ {dep_name} ({dep_desc})")
    
    print("\n📋 依赖检查结果：")
    for dep in available_deps:
        print(f"   {dep}")
    
    if missing_deps:
        print("\n⚠️  缺少依赖：")
        for dep in missing_deps:
            print(f"   {dep}")
        print("\n💡 提示：这些依赖通常已包含在ComfyUI环境中")
    
    return len(missing_deps) == 0

# 如果直接运行此模块，显示信息
if __name__ == "__main__":
    print("=" * 50)
    print("ComfyUI 图片格式转换节点包")
    print("=" * 50)
    
    info = get_extension_info()
    for key, value in info.items():
        print(f"{key}: {value}")
    
    print("\n" + "=" * 50)
    check_dependencies()
    print("=" * 50)
