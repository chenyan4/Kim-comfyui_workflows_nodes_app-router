import logging
import os
import httpx
import base64
from typing import Dict, Any
import numpy as np
from PIL import Image, ImageChops
import torch
import io
import json,base64

from .image_utils import ImageDownloader
image_downloader = ImageDownloader()

clothe_map = {
    'tops': ['coat', 'tops'],
    'bottoms': ['pants', 'skirt'],
    'pants': ['pants', 'skirt'],
    'whole': ['coat', 'tops', 'pants', 'skirt']
}

def pil2tensor(image):
    new_image = image.convert('RGB')
    img_array = np.array(new_image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array)[None]
    return img_tensor
def tensor2pil(image):
    return Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8))

def pilmask2tensor(mask_img):
    mask_tensor = torch.from_numpy(np.array(mask_img.convert('L'))).float()  # 转换为float类型
    mask_tensor = mask_tensor / 255.0  # 归一化到 0-1 范围
    mask_tensor = mask_tensor.unsqueeze(0)
    return mask_tensor


# 基础配置
HOST = "http://api.minimax.dev.xuantu.pro"   # 开发环境
# HOST = "https://xuantu.pro"                # 生产环境
URL = HOST + "/open/recognize/clothes"

HEADERS = {
    "Xt-User-ID": "1204599",
    "Xt-Pass": "e073c96915b2c9efc996e1d23530c077",
    "Authorization": "",
    "Xt-App-Version": "",
    "Xt-Region-Code": "",
    "Xt-Lang-Tag": "",
    "Xt-Flavor": "",
    "Xt-Device-Id": "",
    "Xt-Platform": "",
    "Xt-Mobile-Brand": "",
    "Content-Type": "application/json"
}

logger = logging.getLogger(__name__)
_http_client = httpx.Client(timeout=30)

def is_black_image(img):
    # 如果是 RGBA 或其他多通道图像，先转成灰度图
    if img.mode != 'L':
        img = img.convert('L')    
    # 获取所有像素值
    extrema = img.getextrema()  # 返回 (min, max) 像素值
    return extrema[1] == 0  # 最大值为 0，说明所有像素都是 0

def load_image_from_url(image_url):
    final_url_image_path = image_downloader.download_imagev2(image_url)
    final_image = Image.open(final_url_image_path)
    return final_image

def alpha_union(image_list):
    base = Image.new('L', image_list[0].size, 0)  # 初始化一个全黑的灰度图（全0 alpha）

    for pil_image in image_list:
        alpha = pil_image.getchannel('A')  # 获取 alpha 通道
        # 转成 numpy 后逐像素取最大 替换高阶API
        base_arr = np.array(base)
        alpha_arr = np.array(alpha)
        base_arr = np.maximum(base_arr, alpha_arr)
        base = Image.fromarray(base_arr.astype(np.uint8), mode='L')

    return base

def clothes_seg_sync(image_path: str, position: str) -> Dict[str, Any]:
    """image_path 可以是本地文件路径，也可以是 http(s) URL"""
    if os.path.exists(image_path):
        with open(image_path, "rb") as f:
            base_image = base64.b64encode(f.read()).decode()
    else:
        base_image = image_path

    try:
        resp = _http_client.post(URL, headers=HEADERS, json={"base_image_key": base_image})
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        logging.error(
            "clothes_seg_sync request failed with %s: %s",
            exc.response.status_code,
            exc.response.text,
        )
        return None
    except httpx.RequestError as exc:
        logging.error("clothes_seg_sync request error: %s", exc)
        return None

    try:
        response = resp.json()
    except ValueError:
        logging.error("clothes_seg_sync invalid JSON response: %s", resp.text)
        return None
    
    logging.info(f'clothes_seg_sync response =========== {response}')
    code = response.get('code', 0)
    if position not in clothe_map:
        position = 'whole'
    if code == 0:
        res = response.get('data', {})
        parts = clothe_map[position]
        masks = [load_image_from_url(res.get(key)) for key in parts if res.get(key)]
        if len(masks) == 0:
            logging.warning("clothes_seg_sync no masks for position %s", position)
            return None
        sub_mask = alpha_union(masks)
        return sub_mask
    else:
        logging.warning("clothes_seg_sync non-zero code %s: %s", code, response)
        return None
    
class ClothesSegmentAPI:
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
                    "required": {
                        "image": ("IMAGE",),
                        "position": (list(clothe_map.keys()), {"default": "whole"}),
                    },
                }
    CATEGORY = "zdx/mask"
    RETURN_TYPES = ("MASK",)
    RETURN_NAMES = ("mask",)
    FUNCTION = 'call'
    DESCRIPTION = """
cloth segment from api
"""
    def call(self, image: torch.Tensor, position: str):
        input_image = tensor2pil(image)# PIL.Image
        buf = io.BytesIO()
        # 用 PNG 保留透明度；服务端若要求 JPEG，可改为 JPEG
        input_image.save(buf, format="PNG")
        base64_image =  base64.b64encode(buf.getvalue()).decode()
        res = clothes_seg_sync(base64_image, position)
        images = [load_image_from_url(res[key])   for key in ['tops', 'coat', 'skirt', 'pants'] if res[key] != '']
        final_image = alpha_union(images)
        final_image = pilmask2tensor(final_image)
        return (final_image,)



if __name__ == '__main__':
    im = 'http://statica.xuantu.pro/x/prod/draft/2094416_08680935ad3936b300fe21a43ec1d780.jpg'
    im = 'x/prod/draft/3360993_9bb25e52b47e5dc3feca7c904e3308d0.jpg'
    clothes_seg_sync(im, )

    res = {'tops': 'http://statica.xuantu.pro/x/prod/clothes/1752739427_02.png',
    'coat': 'http://statica.xuantu.pro/x/prod/clothes/1752739427_00.png',
    'skirt': '',
    'pants': 'http://statica.xuantu.pro/x/prod/clothes/1752739427_01.png',
    'shoes': '',
    'hat': '',
    'bag': ''}
    {'code': 0,
    'currentTime': '2025-07-17 16:03:47',
    'data': res,
    'message': 'success'}


    images = [load_image_from_url(res[key])   for key in ['tops', 'coat', 'skirt', 'pants'] if res[key] != '']
    final_image = alpha_union(images)
    final_image.save('1_growmaskwithblur_864.png')

    a = load_image_from_url(res['tops'])
    alpha = np.array(a)[:, :, -1]                 # 分量
    Image.fromarray(alpha, mode='L').save('bw_mask.png')
    a.save('1_growmaskwithblur_864.png')
    extrema = a.getextrema()
