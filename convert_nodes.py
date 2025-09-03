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
        获取图片格式信息
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
            
            # 判断可能的格式类型
            if channels == 1:
                format_type = "灰度图"
            elif channels == 3:
                format_type = "RGB"
            elif channels == 4:
                format_type = "RGBA"
            else:
                format_type = f"{channels}通道"
            
            info = f"图片 {i+1}: 尺寸 {width}x{height}, 格式 {format_type}"
            info_list.append(info)
        
        result_info = "\n".join(info_list)
        print(f"图片格式信息:\n{result_info}")
        
        return (result_info,)


# 节点映射
NODE_CLASS_MAPPINGS = {
    "ImageFormatConverter": ImageFormatConverter,
    "ImageFormatInfo": ImageFormatInfo,
}

# 节点显示名称
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageFormatConverter": "Image Format Converter",
    "ImageFormatInfo": "Image Format Info",
}
