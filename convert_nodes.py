import torch
import numpy as np
from PIL import Image
import io


class ImageFormatConverter:
    """
    ComfyUI节点：图片格式转换器
    支持将输入图片转换为指定的目标格式
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),  # 输入图片（支持批量）
                "output_format": (["PNG", "JPEG", "WEBP", "BMP", "TIFF"],),  # 输出格式选择
                "quality": ("INT", {
                    "default": 95,
                    "min": 1,
                    "max": 100,
                    "step": 1,
                    "display": "number"
                }),  # 图片质量（主要用于JPEG和WEBP）
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("converted_images",)
    FUNCTION = "convert_format"
    CATEGORY = "image/format"
    
    def convert_format(self, images, output_format, quality):
        """
        转换图片格式
        
        Args:
            images: 输入图片张量 (batch_size, height, width, channels)
            output_format: 目标格式
            quality: 图片质量
            
        Returns:
            转换后的图片张量
        """
        # 确保输入是torch张量
        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images)
        
        # 获取批量大小
        batch_size = images.shape[0] if len(images.shape) == 4 else 1
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        
        converted_images = []
        
        for i in range(batch_size):
            # 提取单张图片
            image_tensor = images[i]
            
            # 将张量转换为PIL图片
            # ComfyUI的图片张量格式通常是 (H, W, C)，值域在0-1之间
            if image_tensor.max() <= 1.0:
                image_array = (image_tensor * 255).clamp(0, 255).cpu().numpy().astype(np.uint8)
            else:
                image_array = image_tensor.clamp(0, 255).cpu().numpy().astype(np.uint8)
            
            pil_image = Image.fromarray(image_array)
            
            # 处理不同的输出格式
            if output_format == "JPEG":
                # JPEG不支持透明度，需要转换为RGB
                if pil_image.mode in ('RGBA', 'LA', 'P'):
                    # 创建白色背景
                    background = Image.new('RGB', pil_image.size, (255, 255, 255))
                    if pil_image.mode == 'P':
                        pil_image = pil_image.convert('RGBA')
                    background.paste(pil_image, mask=pil_image.split()[-1] if pil_image.mode == 'RGBA' else None)
                    pil_image = background
                else:
                    pil_image = pil_image.convert('RGB')
            
            elif output_format == "PNG":
                # PNG支持透明度，保持原有模式或转换为RGBA
                if pil_image.mode not in ('RGBA', 'RGB', 'L'):
                    pil_image = pil_image.convert('RGBA')
            
            elif output_format == "WEBP":
                # WebP支持透明度
                if pil_image.mode == 'P':
                    pil_image = pil_image.convert('RGBA')
            
            elif output_format == "BMP":
                # BMP不支持透明度
                if pil_image.mode in ('RGBA', 'LA', 'P'):
                    background = Image.new('RGB', pil_image.size, (255, 255, 255))
                    if pil_image.mode == 'P':
                        pil_image = pil_image.convert('RGBA')
                    if pil_image.mode == 'RGBA':
                        background.paste(pil_image, mask=pil_image.split()[-1])
                    pil_image = background
                else:
                    pil_image = pil_image.convert('RGB')
            
            elif output_format == "TIFF":
                # TIFF支持多种模式
                if pil_image.mode == 'P':
                    pil_image = pil_image.convert('RGBA')
            
            # 将PIL图片转换回字节流再转换回张量，模拟格式转换过程
            buffer = io.BytesIO()
            
            # 根据格式设置保存参数
            save_kwargs = {}
            if output_format in ["JPEG", "WEBP"]:
                save_kwargs['quality'] = quality
                save_kwargs['optimize'] = True
            elif output_format == "PNG":
                save_kwargs['optimize'] = True
            
            pil_image.save(buffer, format=output_format, **save_kwargs)
            buffer.seek(0)
            
            # 重新加载图片以确保格式转换生效
            converted_pil = Image.open(buffer)
            
            # 转换回张量格式
            if converted_pil.mode == 'L':
                # 灰度图转换为RGB
                converted_pil = converted_pil.convert('RGB')
            elif converted_pil.mode == 'RGBA':
                # 保持RGBA格式
                pass
            elif converted_pil.mode == 'RGB':
                # 保持RGB格式
                pass
            
            # 转换为numpy数组
            converted_array = np.array(converted_pil).astype(np.float32) / 255.0
            
            # 确保是3通道或4通道
            if len(converted_array.shape) == 2:
                # 灰度图，转换为RGB
                converted_array = np.stack([converted_array] * 3, axis=-1)
            elif converted_array.shape[-1] == 4:
                # RGBA，保持4通道
                pass
            elif converted_array.shape[-1] == 3:
                # RGB，保持3通道
                pass
            
            # 如果原图是3通道，输出也保持3通道
            if image_tensor.shape[-1] == 3 and converted_array.shape[-1] == 4:
                converted_array = converted_array[:, :, :3]
            
            converted_tensor = torch.from_numpy(converted_array)
            converted_images.append(converted_tensor)
            
            buffer.close()
        
        # 堆叠所有转换后的图片
        result = torch.stack(converted_images, dim=0)
        
        return (result,)


class ImageFormatInfo:
    """
    图片格式信息显示节点
    显示输入图片的格式信息
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
            },
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("format_info",)
    FUNCTION = "get_format_info"
    CATEGORY = "image/format"
    OUTPUT_NODE = True
    
    def get_format_info(self, images):
        """
        获取详细的图片格式信息
        """
        if not isinstance(images, torch.Tensor):
            images = torch.tensor(images)
        
        batch_size = images.shape[0] if len(images.shape) == 4 else 1
        if len(images.shape) == 3:
            images = images.unsqueeze(0)
        
        info_list = []
        
        for i in range(batch_size):
            image_tensor = images[i]
            height, width, channels = image_tensor.shape
            
            # 基本信息
            info_dict = {
                "图片序号": i + 1,
                "尺寸": f"{width} × {height}",
                "总像素": width * height,
                "通道数": channels,
                "数据类型": str(image_tensor.dtype),
                "设备": str(image_tensor.device)
            }
            
            # 判断色彩格式类型
            if channels == 1:
                color_format = "灰度图 (Grayscale)"
                color_space = "L"
            elif channels == 3:
                color_format = "彩色图 (RGB)"
                color_space = "RGB"
            elif channels == 4:
                color_format = "带透明度彩色图 (RGBA)"
                color_space = "RGBA"
            else:
                color_format = f"多通道图像 ({channels}通道)"
                color_space = f"{channels}C"
            
            info_dict["色彩格式"] = color_format
            info_dict["色彩空间"] = color_space
            
            # 数值范围分析
            min_val = float(image_tensor.min())
            max_val = float(image_tensor.max())
            mean_val = float(image_tensor.mean())
            
            info_dict["数值范围"] = f"{min_val:.3f} ~ {max_val:.3f}"
            info_dict["平均值"] = f"{mean_val:.3f}"
            
            # 判断数值类型
            if min_val >= 0 and max_val <= 1.0:
                value_type = "标准化 (0-1)"
            elif min_val >= 0 and max_val <= 255:
                value_type = "8位整数 (0-255)"
            else:
                value_type = "自定义范围"
            
            info_dict["数值类型"] = value_type
            
            # 透明度分析（仅适用于RGBA）
            if channels == 4:
                alpha_channel = image_tensor[:, :, 3]
                alpha_min = float(alpha_channel.min())
                alpha_max = float(alpha_channel.max())
                alpha_mean = float(alpha_channel.mean())
                
                if alpha_min == alpha_max == 1.0:
                    transparency_info = "完全不透明"
                elif alpha_min == alpha_max == 0.0:
                    transparency_info = "完全透明"
                elif alpha_min == 0.0 and alpha_max == 1.0:
                    transparency_info = f"部分透明 (平均: {alpha_mean:.3f})"
                else:
                    transparency_info = f"透明度: {alpha_min:.3f} ~ {alpha_max:.3f} (平均: {alpha_mean:.3f})"
                
                info_dict["透明度"] = transparency_info
            
            # 内存占用
            memory_mb = image_tensor.element_size() * image_tensor.nelement() / (1024 * 1024)
            info_dict["内存占用"] = f"{memory_mb:.2f} MB"
            
            # 推荐的输出格式
            recommended_formats = []
            if channels == 1:
                recommended_formats = ["PNG", "JPEG", "TIFF"]
            elif channels == 3:
                recommended_formats = ["JPEG", "PNG", "WEBP"]
            elif channels == 4:
                recommended_formats = ["PNG", "WEBP", "TIFF"]
            
            info_dict["推荐格式"] = ", ".join(recommended_formats)
            
            # 格式化输出
            info_lines = [f"📷 图片 {i+1} 详细信息:"]
            info_lines.append("=" * 30)
            
            for key, value in info_dict.items():
                if key != "图片序号":
                    info_lines.append(f"{key}: {value}")
            
            info_lines.append("")  # 空行分隔
            
            info_list.extend(info_lines)
        
        # 批量总结信息
        if batch_size > 1:
            summary_lines = [
                f"📊 批量处理总结:",
                "=" * 30,
                f"图片总数: {batch_size}",
                f"总内存占用: {sum(img.element_size() * img.nelement() for img in images) / (1024 * 1024):.2f} MB"
            ]
            
            # 检查格式一致性
            first_shape = images[0].shape
            consistent_format = all(img.shape == first_shape for img in images)
            summary_lines.append(f"格式一致性: {'✅ 一致' if consistent_format else '❌ 不一致'}")
            
            info_list.extend(summary_lines)
        
        result_info = "\n".join(info_list)
        print(f"\n🔍 图片格式分析结果:\n{result_info}")
        
        return (result_info,)


class ImageBatchCombiner:
    """
    图片批量组合节点
    将最多5张图片组合成一个batch，保持原始尺寸
    """
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image1": ("IMAGE",),  # 第一张图片（必选）
            },
            "optional": {
                "image2": ("IMAGE",),  # 第二张图片（可选）
                "image3": ("IMAGE",),  # 第三张图片（可选）
                "image4": ("IMAGE",),  # 第四张图片（可选）
                "image5": ("IMAGE",),  # 第五张图片（可选）
            },
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image_batch",)
    FUNCTION = "combine_images"
    CATEGORY = "image/batch"
    
    def combine_images(self, image1, image2=None, image3=None, image4=None, image5=None):
        """
        将输入的图片组合成一个batch
        
        Args:
            image1: 第一张图片（必选）
            image2-image5: 其他图片（可选）
            
        Returns:
            组合后的图片batch
        """
        # 收集所有非空的图片
        images_list = [image1]
        
        for img in [image2, image3, image4, image5]:
            if img is not None:
                images_list.append(img)
        
        # 确保所有输入都是torch张量
        processed_images = []
        for img in images_list:
            if not isinstance(img, torch.Tensor):
                img = torch.tensor(img)
            
            # 如果是单张图片（3维），添加batch维度
            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            
            # 如果输入本身已经是batch，需要分解
            for i in range(img.shape[0]):
                processed_images.append(img[i])
        
        # 检查所有图片的通道数是否一致
        channels = processed_images[0].shape[-1]
        for i, img in enumerate(processed_images):
            if img.shape[-1] != channels:
                # 如果通道数不一致，统一转换为RGB（3通道）
                if img.shape[-1] == 1:  # 灰度图转RGB
                    img = img.repeat(1, 1, 3)
                elif img.shape[-1] == 4:  # RGBA转RGB（去除alpha通道）
                    img = img[:, :, :3]
                processed_images[i] = img
        
        # 重新检查通道数
        channels = processed_images[0].shape[-1]
        final_images = []
        
        for img in processed_images:
            # 确保所有图片通道数一致
            if img.shape[-1] != channels:
                if channels == 3 and img.shape[-1] == 1:
                    img = img.repeat(1, 1, 3)
                elif channels == 3 and img.shape[-1] == 4:
                    img = img[:, :, :3]
            final_images.append(img)
        
        # 组合成batch - 保持每张图片的原始尺寸
        result_batch = torch.stack(final_images, dim=0)
        
        print(f"🔄 图片批量组合完成:")
        print(f"   输入图片数量: {len(final_images)}")
        print(f"   输出batch形状: {result_batch.shape}")
        print(f"   各图片尺寸:")
        for i, img in enumerate(final_images):
            print(f"     图片{i+1}: {img.shape[1]}×{img.shape[0]}×{img.shape[2]}")
        
        return (result_batch,)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "ImageFormatConverter": ImageFormatConverter,
    "ImageFormatInfo": ImageFormatInfo,
    "ImageBatchCombiner": ImageBatchCombiner,
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageFormatConverter": "Image Format Converter",
    "ImageFormatInfo": "Image Format Info",
    "ImageBatchCombiner": "Image Batch Combiner",
}
