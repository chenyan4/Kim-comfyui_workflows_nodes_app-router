import copy
import os

import torch
from PIL import Image,ImageDraw
import glob
import numpy as np
from ultralytics import YOLO
import folder_paths
import cv2

models_dirs=os.path.join(folder_paths.models_dir,'yolo')
def get_files(models_dir,file_exp_list):
    file_list=[]
    for exp in file_exp_list:
        # extend可将可迭代对象中的元素，逐个添加进当前列表
        file_list.extend(glob.glob(os.path.join(models_dir,'*'+exp))) #glob就是模式匹配文件，支持通配符，每个文件返回一个列表，所以用extend
    file_dict={}
    for i in range(len(file_list)):
        _,filename=os.path.split(file_list[i]) # 将文件路径拆成("C:/models/yolo", "model1.pt")
        # 创建文件名到文件路径的映射
        file_dict[filename]=file_list[i]

    return file_dict

def tensor2pil(image):
    if isinstance(image,Image.Image):
        return image
    else:
        if len(image.shape)<3:
            image=image.unsqueeze(0)
        return Image.fromarray((image[0].cpu().numpy()*255).astype(np.uint8))

def pil2tensor(image):
    new_image=image.convert('RGB')
    image_array=np.array(new_image).astype(np.float32)/255.0
    image_tensor=torch.tensor(image_array)
    return image_tensor

def mask2tensor(mask):
    mask=mask.convert('L')
    mask_array=np.array(mask).astype(np.float32)/255.0
    mask_tensor=torch.tensor(mask_array)
    return mask_tensor

def np2tensor(data):
    data=data.astype(np.float32)/255.0
    data_tensor=torch.tensor(data)
    return data_tensor

class Yolov8_person_nomask:
    CATEGORY="My Nodes/yolov8 person nomask"
    RETURN_TYPES=("MASK","BOOLEAN",)
    RETURN_NAMES=("back_mask","boolean")
    FUNCTION="yolov8_person_nomask"

    @classmethod
    def INPUT_TYPES(cls):
        model_exp=[".pt"]
        FILES_DICT=get_files(models_dirs,model_exp)
        FILE_LIST=list(FILES_DICT.keys())

        return{
            "required":{
                "back_image":("IMAGE",),
                "yolo_model":(FILE_LIST,),
            },
            "optional":{
                "mask":("MASK",),
                "true_rate":("FLOAT",{
                    "default":0.85,
                    "min":0.01,
                    "max":1.0,
                    "step":0.01,
                    "display":"number"
                }),
                "img_ratio":("FLOAT",{
                    "default":float(2/3),
                    "min":0.01,
                    "max":1.0,
                    "step":0.01,
                    "display":"number"
                }),
                "x_ratio":("FLOAT",{
                    "default":float(0.5),
                    "min":0.01,
                    "max":1.0,
                    "step":0.01,
                    "display":"number"
                }),
                "y_ratio":("FLOAT",{
                    "default":float(1/10),
                    "min":0.01,
                    "max":1.0,
                    "step":0.01,
                    "display":"number"
                }),
                "radius":("INT",{
                    "default":100,
                    "min":10,
                    "max":1000,
                    "step":10,
                    "display":"number"
                }),
                "blur_radius":("INT",{
                    "default":0,
                    "min":0,
                    "max":100,
                    "step":1,
                    "display":"number"
                })
            }
        }

    def yolov8_person_nomask(self,back_image,yolo_model,true_rate,img_ratio,x_ratio,y_ratio,radius,blur_radius,mask=None):
        back_image=tensor2pil(back_image)

        yolo_model=YOLO(os.path.join(models_dirs,yolo_model))
        result = yolo_model(
            back_image,
            retina_masks=True,
            classes=[0],
            conf=true_rate,
            verbose=False
        )[0]

        if result.masks is not None and len(result.masks)>0:
            if result.boxes is not None and len(result.boxes)>0:
                if result.boxes.conf[0]>=true_rate:
                    masks_data=result.masks.data
                    n_mask=masks_data[0]
                    n_mask=n_mask.unsqueeze(0) 
                    boolean=True
                    return(n_mask,boolean,)

        bd_w,bd_h=back_image.size

        n_mask=Image.new('L',(bd_w,bd_h),"black")
        if mask is None:
            boolean=False
            return (mask2tensor(n_mask),boolean,)
        target=img_ratio*max(bd_h,bd_w)

        mask=tensor2pil(mask).convert('L')

        
        m_bbox=mask.getbbox()
       
        mask=mask.crop(m_bbox)

        m_w,m_h=mask.size
        ratio=target/max(m_w,m_h)
        new_w,new_h=int(ratio*m_w),int(ratio*m_h)

        
        if new_w>=bd_w or new_h>=bd_h:
            raise ValueError(f'缩放图片的长宽超过背景图大小，请下调img_ratio值')

        x_left=int(bd_w-new_w)
        y_left=int(bd_h-new_h)

        x_trap_padding=int(x_left*x_ratio)
        x_left=x_left-x_trap_padding

        y_trap_padding=int(y_left*y_ratio)
        y_left=y_left-y_trap_padding

        x_right,y_right=x_left+new_w,y_left+new_h
        draw=ImageDraw.Draw(n_mask)

        draw.rounded_rectangle(
            [x_left, y_left, x_right, y_right],
            radius=int(radius),
            fill=255  # 白色填充
        )

        n_mask=np.array(n_mask).astype(np.uint8)
        blur_radius_int = int(round(blur_radius))
        n_mask=cv2.GaussianBlur(n_mask, (2 * blur_radius_int + 1, 2 * blur_radius_int + 1), 0)

        n_mask=np2tensor(n_mask)
       
        if len(n_mask.shape)<3:
            n_mask=n_mask.unsqueeze(0) 
        print("Yolov8_person_detect done")
        boolean=False
        return (n_mask,boolean,)

NODE_CLASS_MAPPINGS={
    "Yolov8_person_nomask":Yolov8_person_nomask
}

NODE_DISPLAY_NAME_MAPPINGS={
    "Yolov8_person_nomask":"Yolov8_person_nomask(My Node)"
}