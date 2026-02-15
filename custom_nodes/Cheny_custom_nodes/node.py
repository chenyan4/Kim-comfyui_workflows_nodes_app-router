# my_nodes.py
import torch
import numpy as np
from PIL import Image


# --------------------------
# 节点 1：图像缩放节点
# --------------------------
class ImageResizeWithPreview:
    """
    图像缩放节点，支持自定义宽度/高度，带实时预览
    """
    # 节点类别（在 ComfyUI 菜单中的位置）
    CATEGORY = "My Nodes/Image Processing"
    # 节点输出类型
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("resized_image",)
    FUNCTION = "resize_image"

    @classmethod
    def INPUT_TYPES(cls):
        """定义节点输入参数"""
        return {
            "required": {
                "image": ("IMAGE",),  # 输入图像（ComfyUI 内置类型）
                "width": ("INT", {
                    "default": 512,    # 默认值
                    "min": 64,         # 最小值
                    "max": 4096,       # 最大值
                    "step": 64,        # 步长
                    "display": "number"  # 显示为数字输入框
                }),
                "height": ("INT", {
                    "default": 512,
                    "min": 64,
                    "max": 4096,
                    "step": 64,
                    "display": "number"
                }),
                "interpolation": (["lanczos", "bilinear", "nearest"],),  # 下拉选择插值方式
            }
        }

    def resize_image(self, image, width, height, interpolation):
        """
        核心执行逻辑：缩放图像
        参数：
            image: 输入图像张量 [batch, H, W, C]，值范围 [0,1]
            width: 目标宽度
            height: 目标高度
            interpolation: 插值方式
        返回：
            缩放后的图像张量
        """
        # 映射插值方式到 PIL 对应的滤镜
        interp_map = {
            "lanczos": Image.LANCZOS,
            "bilinear": Image.BILINEAR,
            "nearest": Image.NEAREST
        }
        pil_interp = interp_map[interpolation]

        batch_size = image.shape[0]
        resized_images = []

        for i in range(batch_size):
            # 1. 转换为 PIL 图像：[0,1] -> [0,255] -> numpy -> PIL
            img_np = image[i].cpu().numpy()  # [H, W, C]，float32
            img_np = (img_np * 255).astype(np.uint8)  # 映射到 [0,255]
            pil_img = Image.fromarray(img_np)

            # 2. 执行缩放
            pil_img = pil_img.resize((width, height), pil_interp)

            # 3. 转换回 ComfyUI 格式：PIL -> numpy -> [0,1] -> torch
            resized_np = np.array(pil_img).astype(np.float32) / 255.0
            resized_images.append(resized_np)

        # 合并为 batch 张量 [batch, H, W, C]
        resized_tensor = torch.tensor(np.stack(resized_images, axis=0))
        return (resized_tensor,)


# --------------------------
# 节点 2：文本拼接节点
# --------------------------
class TextConcatenator:
    """
    文本拼接节点，支持多个文本输入拼接
    """
    CATEGORY = "My Nodes/Text Processing"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("concatenated_text",)
    FUNCTION = "concat_texts"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text1": ("STRING", {
                    "default": "Hello",
                    "multiline": False  # 单行输入
                }),
                "text2": ("STRING", {
                    "default": "World",
                    "multiline": False
                }),
            },
            "optional": {
                "separator": ("STRING", {
                    "default": " ",
                    "multiline": False
                }),
                "text3": ("STRING", {
                    "default": "",
                    "multiline": False
                }),
            }
        }

    def concat_texts(self, text1, text2, separator=" ", text3=""):
        """拼接文本"""
        # 过滤空文本
        texts = [t for t in [text1, text2, text3] if t]
        # 用分隔符拼接
        result = separator.join(texts)
        return (result,)


# --------------------------
# 节点注册（必须）
# --------------------------
# 将节点类映射到唯一标识符
NODE_CLASS_MAPPINGS = {
    "ImageResizeWithPreview": ImageResizeWithPreview,
    "TextConcatenator": TextConcatenator
}

# 节点在 UI 中显示的名称（可选，默认使用类名）
NODE_DISPLAY_NAME_MAPPINGS = {
    "ImageResizeWithPreview": "Image Resize (My Node)",
    "TextConcatenator": "Text Concatenator (My Node)"
}

