import copy
import os

import torch
from PIL import Image
import glob
import numpy as np
from ultralytics import YOLO
import folder_paths

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

class Yolov8_person_detect:
    CATEGORY="My Nodes/yolov8 person detect"
    RETURN_TYPES=("MASK",)
    RETURN_NAMES=("back_mask",)
    FUNCTION="yolov8_person_detect"

    @classmethod
    def INPUT_TYPES(cls):
        model_exp=["seg.pt"]
        FILES_DICT=get_files(models_dirs,model_exp)
        FILE_LIST=list(FILES_DICT.keys())

        return{
            "required":{
                "back_image":("IMAGE",),
                "mask":("MASK",),
                "yolo_model":(FILE_LIST,),
            },
            "optional":{
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
                })
            }
        }

    def yolov8_person_detect(self,mask,back_image,yolo_model,true_rate,img_ratio,x_ratio,y_ratio):
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
                    return(n_mask,)

        # 直接生成mask，按背景 2/3
        # bd_w,bd_h=back_image.size
        # n_mask=Image.new('L',(bd_w,bd_h),"black")
        # target=img_ratio*max(bd_h,bd_w)

        # new_w,new_h=int(img_ratio*bd_w),int(img_ratio*bd_h)

        # paste_mask=Image.new('L',(new_w,new_h),"white")


        # x=int(bd_w-new_w)
        # y=int(bd_h-new_h)

        # x_trap_padding=int(x*x_ratio)
        # x=x-x_trap_padding

        # y_trap_padding=int(y*y_ratio)
        # y=y-y_trap_padding

        # n_mask.paste(paste_mask,(x,y))
        # n_mask=mask2tensor(n_mask)
       
        # if len(n_mask.shape)<3:
        #     n_mask=n_mask.unsqueeze(0) 
        # print("Yolov8_person_detect done")
        # return (n_mask,)


        # 利用原图人物mask，生成新mask
        mask=tensor2pil(mask).convert('L')

        bd_w,bd_h=back_image.size

        n_mask=Image.new('L',(bd_w,bd_h),"black")
        target=img_ratio*max(bd_h,bd_w)

        
        m_bbox=mask.getbbox()
       
        mask=mask.crop(m_bbox)

        m_w,m_h=mask.size
        ratio=target/max(m_w,m_h)
        new_w,new_h=int(ratio*m_w),int(ratio*m_h)
        mask=mask.resize((new_w,new_h),Image.LANCZOS)

        
        if new_w>=bd_w or new_h>=bd_h:
            raise ValueError(f'缩放图片的长宽超过背景图大小，请下调img_ratio值')

        x=int(bd_w-new_w)
        y=int(bd_h-new_h)

        x_trap_padding=int(x*x_ratio)
        x=x-x_trap_padding

        y_trap_padding=int(y*y_ratio)
        y=y-y_trap_padding

        n_mask.paste(mask,(x,y))
        n_mask=mask2tensor(n_mask)
       
        if len(n_mask.shape)<3:
            n_mask=n_mask.unsqueeze(0) 
        print("Yolov8_person_detect done")
        return (n_mask,)

NODE_CLASS_MAPPINGS={
    "Yolov8_person_detect":Yolov8_person_detect
}

NODE_DISPLAY_NAME_MAPPINGS={
    "Yolov8_person_detect":"Yolov8_person_detect(My Node)"
}
