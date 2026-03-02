import requests
import torch
import numpy as np
from PIL import Image, ImageOps
import io
import re
import base64
import urllib3

# 绂佺敤 SSL 璀﹀憡
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ==========================================
# 馃洜锔?鏍稿績宸ュ叿绠?
# ==========================================

def smart_resize(image, target_width, target_height):
    """鏅鸿兘缂╂斁+涓績瑁佸壀"""
    return ImageOps.fit(image, (target_width, target_height), method=Image.LANCZOS)

def resize_image_for_api(image_tensor, max_side=1568):
    """杈撳叆鍥剧墖鍘嬬缉"""
    if len(image_tensor.shape) > 3: image_tensor = image_tensor[0]
    i = 255. * image_tensor.cpu().numpy()
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    
    w, h = img.size
    if w > max_side or h > max_side:
        ratio = min(max_side/w, max_side/h)
        img = img.resize((int(w*ratio), int(h*ratio)), Image.LANCZOS)
    return img

def pil2base64(pil_image):
    buffered = io.BytesIO()
    pil_image.save(buffered, format="JPEG", quality=90)
    return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

def download_process(response_json, target_w=None, target_h=None, strict_mode=False):
    if "error" in response_json:
        raise Exception(f"API Error: {response_json['error']}")
    if "choices" not in response_json:
        raise Exception(f"API Response Error: {response_json}")
    
    content = response_json["choices"][0]["message"]["content"]
    image_objects = []

    # 1. Base64 鍖归厤
    b64_matches = re.findall(r'data:image/\w+;base64,([a-zA-Z0-9+/=]+)', content)
    if b64_matches:
        print(f"馃摝 Base64 Images: {len(b64_matches)}")
        for b64_str in b64_matches:
            try:
                img = Image.open(io.BytesIO(base64.b64decode(b64_str))).convert("RGB")
                image_objects.append(img)
            except: pass

    # 2. URL 鍖归厤
    if not image_objects:
        urls = re.findall(r'(https?://[^\s)\]"]+)', content)
        valid_urls = list(set([u for u in urls if any(x in u for x in ["google", "image", "oss", "ufile", "png", "jpg"])]))
        if valid_urls:
            print(f"馃敆 URL Images: {len(valid_urls)}")
            for url in valid_urls:
                try:
                    res = requests.get(url, timeout=60, verify=False)
                    if res.status_code == 200:
                        image_objects.append(Image.open(io.BytesIO(res.content)).convert("RGB"))
                except: pass

    if not image_objects:
        raise Exception(f"No images found. Response: {content[:200]}...")

    final_tensors = []
    base_w, base_h = target_w, target_h

    for i, img in enumerate(image_objects):
        if strict_mode and target_w and target_h:
            img = smart_resize(img, target_w, target_h)
        elif i == 0:
            base_w, base_h = img.size
        else:
            img = img.resize((base_w, base_h), Image.LANCZOS)
        
        img_np = np.array(img).astype(np.float32) / 255.0
        final_tensors.append(torch.from_numpy(img_np))

    return (torch.stack(final_tensors),)

# ==========================================
# 馃寪 API 鍦板潃绠＄悊
# ==========================================
def get_api_url(source):
    if "SynVow" in source:
        return "https://ai.synvow.cc/v1/chat/completions"
    if "T8Star" in source:
        return "https://ai.t8star.cn/v1/chat/completions"
    return "https://api.apicore.ai/v1/chat/completions"

# ==========================================
# 馃煝 鑺傜偣 1: SynVow-Nano2 T2I
# ==========================================
class SynVowNano2_T2I:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "api_source": (["T8Star (ai.t8star.cn)", "Apicore (api.apicore.ai)"],),
                "api_key": ("STRING", {"multiline": False}),
                "model_select": ("STRING", {"multiline": False, "default": "gemini-3-pro-image-preview"}),
                "prompt": ("STRING", {"multiline": True, "default": "4k wallpaper, masterpiece"}),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "Custom Size"],),
                "custom_w": ("INT", {"default": 3840, "min": 64, "max": 16384, "step": 8}),
                "custom_h": ("INT", {"default": 2160, "min": 64, "max": 16384, "step": 8}),
                "count": ("INT", {"default": 1, "min": 1, "max": 4}),
                # 淇锛氶檺鍒?UI 涓婄殑鏈€澶ц緭鍏ュ€间负 21浜?(Int32 Max)
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "SynVow Nodes"

    def run(self, api_source, api_key, model_select, prompt, aspect_ratio, custom_w, custom_h, count, seed):
        target_url = get_api_url(api_source)
        
        # 銆愭牳蹇冧慨澶嶃€戝鏋?ComfyUI 杩樻槸浼犺繘鏉ヤ簡瓒呭ぇ鏁存暟锛屽己鍒跺彇妯¤浆鎹负 Int32
        safe_seed = seed % 2147483647
        print(f"馃摗 SynVow T2I | Seed: {safe_seed} (Converted)")

        w, h, strict = None, None, False
        if aspect_ratio == "Custom Size":
            w, h, strict = custom_w, custom_h, True
            ratio_txt = f"{w}x{h}"
        else:
            ratio_txt = aspect_ratio

        msg = [{"role": "user", "content": f"Generate {count} images: {prompt}. Aspect Ratio: {ratio_txt}."}]
        return self.send_req(target_url, api_key, model_select, msg, w, h, strict, safe_seed)

    def send_req(self, url, key, model, msgs, w, h, strict, seed):
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        payload = {
            "model": model, 
            "messages": msgs, 
            "stream": False,
            "seed": seed,
            "messages": [{"role": "system", "content": ""}] + msgs
        }
        res = requests.post(url, json=payload, headers=headers, verify=False, timeout=1000)
        return download_process(res.json(), w, h, strict)

# ==========================================
# 馃數 鑺傜偣 2: SynVow-Nano2 I2I
# ==========================================
class SynVowNano2_I2I:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_1": ("IMAGE",), 
                "api_source": (["T8Star (ai.t8star.cn)", "Apicore (api.apicore.ai)"],),
                "api_key": ("STRING", {"multiline": False}),
                "model_select": ("STRING", {"multiline": False, "default": "gemini-3-pro-image-preview"}),
                "prompt": ("STRING", {"multiline": True, "default": "Make it 4k resolution"}), 
                "size_mode": (["Match Image_1 (Smart Crop)", "Keep Model Output", "Custom Size"],),
                "custom_w": ("INT", {"default": 3840, "min": 64, "max": 16384, "step": 8}),
                "custom_h": ("INT", {"default": 2160, "min": 64, "max": 16384, "step": 8}),
                "count": ("INT", {"default": 1, "min": 1, "max": 4}),
                # 淇锛氶檺鍒?UI 涓婄殑鏈€澶ц緭鍏ュ€间负 21浜?(Int32 Max)
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647}),
            },
            "optional": {
                "image_2": ("IMAGE",),"image_3": ("IMAGE",),"image_4": ("IMAGE",),"image_5": ("IMAGE",),
            }
        }
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "run"
    CATEGORY = "SynVow Nodes"

    def run(self, image_1, api_source, api_key, model_select, prompt, size_mode, custom_w, custom_h, count, seed, image_2=None, image_3=None, image_4=None, image_5=None):
        target_url = get_api_url(api_source)
        
        # 銆愭牳蹇冧慨澶嶃€戝鏋?ComfyUI 杩樻槸浼犺繘鏉ヤ簡瓒呭ぇ鏁存暟锛屽己鍒跺彇妯¤浆鎹负 Int32
        safe_seed = seed % 2147483647
        print(f"馃摗 SynVow I2I | Seed: {safe_seed} (Converted)")

        i1_h, i1_w = image_1.shape[1], image_1.shape[2]
        target_w, target_h, strict = None, None, False
        ratio_instruction = ""

        if size_mode == "Match Image_1 (Smart Crop)":
            target_w, target_h = i1_w, i1_h
            strict = True
            ratio_instruction = f"Output aspect ratio must match {i1_w}x{i1_h}."
        elif size_mode == "Custom Size":
            target_w, target_h = custom_w, custom_h
            strict = True
            ratio_instruction = f"Output size: {custom_w}x{custom_h}."

        imgs = [image_1, image_2, image_3, image_4, image_5]
        content = [{"type": "text", "text": f"{prompt} {ratio_instruction}"}]
        
        for img in imgs:
            if img is not None:
                pil = resize_image_for_api(img)
                content.append({"type": "image_url", "image_url": {"url": pil2base64(pil)}})

        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model_select, 
            "messages": [{"role": "system", "content": ""}, {"role": "user", "content": content}],
            "seed": safe_seed # 鍙戦€佷慨澶嶅悗鐨?Seed
        }
        
        res = requests.post(target_url, json=payload, headers=headers, verify=False, timeout=1000)
        return download_process(res.json(), target_w, target_h, strict)

# ==========================================
# 娉ㄥ唽鑺傜偣
# ==========================================
NODE_CLASS_MAPPINGS = {
    "SynVowNano2_T2I": SynVowNano2_T2I,
    "SynVowNano2_I2I": SynVowNano2_I2I
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SynVowNano2_T2I": "SynVow-Nano2 T2I",
    "SynVowNano2_I2I": "SynVow-Nano2 I2I"
}

