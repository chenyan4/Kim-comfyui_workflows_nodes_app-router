import requests
import torch
import numpy as np
from PIL import Image
import io
import re
import base64
import time
import urllib3
import time

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def tensor2pil(image):
    if isinstance(image,Image.Image):
        return image
    if len(image.shape)<3:
        image=image.unsqueeze(0)
    return Image.fromarray((image[0].cpu().numpy()*255).astype(np.uint8))

def pil2tensor(image):
    new_image=image.convert('RGB')
    image_array=np.array(new_image).astype(np.float32)/255.0
    image_tensor=torch.tensor(image_array)
    return image_tensor

def pil2base64(pil_image):
    buffered=io.BytesIO()
    pil_image.save(buffered,format="JPEG",quality=90)
    return f"data:image/jpeg;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

def get_api_url(source):
    return "https://ai.t8star.cn/v2/videos/generations"

class kim_video_T2V:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "api_key":("STRING",{"multiline":False}),
                "model_select":("STRING",{"multiline":False}),
                "prompt":("STRING",{"multiline":True,"default":"a cat run at ground"}),
                "aspect_ratio":(["16:9","9:16"],),
                "resolution":(["480p","720p","1080p"],),
                "duration":([5,10],),
            }
        }
    
    RETURN_TYPES=("STRING",)
    RETURN_NAMES=("video_url",)
    FUNCTION="run"
    CATEGORY="My Nodes/kim_video_api"

    def run(self, api_key, model_select, prompt, aspect_ratio,resolution, duration):
        url = get_api_url(model_select)
        video_url=self.send_req(url,api_key,model_select,prompt,aspect_ratio,resolution,duration)
        return (video_url,)

    def send_req(self, url, api_key, model, prompt, aspect_ratio,resolution, duration):
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if "veo" in model:
            payload = {
                "model": model,
                "prompt": prompt,
                "aspect_ratio": aspect_ratio,
                "enhance_prompt": True,
                "enable_upsample": True,
            }
        if "seedance" in model:
            payload = {
                "prompt": prompt,
                "model": model,
                "duration": duration,
                "resolution": resolution,
                "ratio": aspect_ratio
            }

        resp = requests.post(url, headers=headers, json=payload, verify=False, timeout=600)
        resp=resp.json()
        
        task_id=resp.get("task_id")
        print(f"task_id: {task_id}")
        if not task_id:
            raise ValueError("API 未返回 task_id")
        check_url="https://ai.t8star.cn/v2/videos/generations"
        check_temp_url=f"{check_url}/{task_id}"

        # 轮询检查任务 10分钟
        timeout=600
        interval=5

        deadlint=time.time()+timeout
        while time.time()<deadlint:
            res=requests.get(check_temp_url,headers=headers,verify=False,timeout=30)
            res=res.json()
            status=res.get("status")
            print(f'status: {status}')
            if status in ("completed","SUCCESS","success","done"):
                video_url=res["data"]["output"]
                print(f"Video generated successfully:{video_url}")
                return video_url
            elif status in ("FAILURE","failed","error"):
                raise RuntimeError(f"Video generation failed:{res.get('message')}")
            time.sleep(interval)
        
        raise TimeoutError(f"Video generation timed out after {timeout} seconds")

class kim_video_I2V:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required":{
                "image_1":("IMAGE",),
                "api_key":("STRING",{"multiline":False}),
                "model_select":("STRING",{"multiline":False}),
                "prompt":("STRING",{"multiline":True,"default":"make animate"}),
                "aspect_ratio":(["16:9","9:16"],),
                "resolution":(["480p","720p","1080p"],),
                "duration":([5,10],),
            },
            "optional":{
                "image_2":("IMAGE",),
            }
        }

    RETURN_TYPES=("STRING",)
    RETURN_NAMES=("video_url",)
    FUNCTION="run"
    CATEGORY="My Nodes/kim_video_api"

    def run(self,image_1,api_key,model_select,prompt,aspect_ratio,resolution,duration,image_2=None):
        url=get_api_url(model_select)
        video_url=self.send_req(image_1,url,api_key,model_select,prompt,aspect_ratio,resolution,duration,image_2)
        return (video_url,)
        
    def send_req(self,image_1,url,api_key,model,prompt,aspect_ratio,resolution,duration,image_2):
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        if image_2 is None:
            image_1=tensor2pil(image_1)
            image_1_base64=pil2base64(image_1)
            images=[image_1_base64]
            if "veo" in model:
                payload={
                    "prompt": prompt,
                    "model": model,
                    "enhance_prompt": True,
                    "images": images,
                    "aspect_ratio":aspect_ratio
                }
            
            if "seedance" in model:
                payload={
                    "prompt": prompt,
                    "model": model,
                    "images":images,
                    "duration": duration,
                    "resolution": resolution,
                    "ratio": aspect_ratio
                }
        else:
            image_1=tensor2pil(image_1)
            image_2=tensor2pil(image_2)
            image_1_base64=pil2base64(image_1)
            image_2_base64=pil2base64(image_2)
            images=[image_1_base64,image_2_base64]
            if "veo" in model:
                payload={
                    "prompt": prompt,
                    "model": model,
                    "enhance_prompt": True,
                    "images": images,
                    "aspect_ratio": aspect_ratio
                }
            if "seedance" in model:
                payload={
                    "prompt": prompt,
                    "model": model,
                    "images":images,
                    "duration": duration,
                    "resolution": resolution,
                    "ratio": aspect_ratio
                }
        
        resp=requests.post(url,headers=headers,json=payload,verify=False,timeout=300)
        resp=resp.json()
        task_id=resp.get("task_id")
        print(f"task_id: {task_id}")
        if not task_id:
            raise ValueError("API 未返回 task_id")
        check_url="https://ai.t8star.cn/v2/videos/generations"
        check_temp_url=f"{check_url}/{task_id}"

        # 轮询检查任务 10分钟   
        timeout=600
        interval=5
        deadlint=time.time()+timeout
        while time.time()<deadlint:
            res=requests.get(check_temp_url,headers=headers,verify=False,timeout=30)
            res=res.json()
            status=res.get("status")
            print(f'status: {status}')
            if status in ("completed","SUCCESS","success","done"):
                video_url=res["data"]["output"]
                print(f"Video generated successfully:{video_url}")
                return video_url
            elif status in ("FAILURE","failed","error"):
                raise RuntimeError(f"Video generation failed:{res.get('message')}")
            time.sleep(interval)
        
        raise TimeoutError(f"Video generation timed out after {timeout} seconds")
    

NODE_CLASS_MAPPINGS = {
    "kim_video_T2V": kim_video_T2V,
    "kim_video_I2V": kim_video_I2V
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "kim_video_T2V": "Kim Video T2V",
    "kim_video_I2V": "Kim Video I2V"
}

            

