import torch
import numpy as np
from PIL import Image

def tensor2pil(image):
    if len(image.shape)<3:
        image=image.unsqueeze(0) #在第0维加一个维度，数值为1，保证二维图像前面有个1能读取到整个图像，RGB图像一般都是四维的，灰度其实一般也都是三维
    #image[0]是确保去掉第0维，取到完整图片
    return Image.fromarray((image[0].cpu().numpy()*255).astype(np.uint8)) #Comfyui输入图片格式为归一化的tensor，需要*255，fromarray相当于由numpy生成PIL图片

def pil2tensor(image):
    new_image=image.convert('RGB')
    new_array=np.array(new_image).astype(np.float32)/255
    new_tensor=torch.tensor(new_array)
    return new_tensor.unsqueeze(0)  #增加batch维度,数值为1

def mask2tensor(mask):
    new_mask=mask.convert("L")
    new_array=np.array(new_mask).astype(np.float32)/255
    new_tensor=torch.tensor(new_array)
    return new_tensor.unsqueeze(0)

class Image_center_paste:
    #节点在ComfyUI菜单中的位置）
    CATEGORY="My Nodes/Image center paste"
    #节点的输出类型
    RETURN_TYPES=("IMAGE","MASK",) #就是输出节点类型，有（MASK，STRING，INT，FLOAT，BOOLEAN，LATENT，CONDITIONING）
    RETURN_NAMES=("paste_img","mask") #输出节点名称
    FUNCTION="image_paste"

    @classmethod
    def INPUT_TYPES(cls):
        #定义节点输入参数
        return{
            # required就是必须要输入的参数，输入节点需要连线
            "required":{
                "need_paste_image":("IMAGE",),#一般后面会跟"IMAGE"后会跟一个字典，default是默认值，min最小值，max最大值
                "back_image":("IMAGE",),
                "mask":("MASK",),
            },
            # optional是可选参数
            "optional":{
                "ratio":("FLOAT",{
                    "default":float(2/3),
                    "min":0.01,
                    "max":1.0,
                    "step":0.1,
                    "display":"number" #显示数字输入框
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


    def image_paste(self,need_paste_image,back_image,mask,ratio,x_ratio,y_ratio):
        np_image=tensor2pil(need_paste_image)
        bd_image=tensor2pil(back_image)
        mask=tensor2pil(mask).convert("L")

        bd_w,bd_h=bd_image.size

        target=ratio*max(bd_h,bd_w)

        m_bbox=mask.getbbox()
        np_image=np_image.crop(m_bbox)
        mask=mask.crop(m_bbox)

        np_w,np_h=np_image.size

        change_size=target/max(np_h,np_w)
        new_w,new_h=int(np_w*change_size),int(np_h*change_size)

        if new_w>=bd_w-1 or new_h>=bd_h-1:
            raise ValueError(f'缩放图片的长宽不匹配，请调小ratio值')

        x=int((bd_w-new_w))
        y=int((bd_h-new_h))

        x_trap_padding=int(x_ratio*x)
        x=x-x_trap_padding

        y_trap_padding=int(y_ratio*y)
        y=y-y_trap_padding

        np_image=np_image.resize((new_w,new_h),Image.LANCZOS)
        mask=mask.resize((new_w,new_h),Image.LANCZOS)

        np_image=np_image.convert("RGBA")
        result_image=bd_image.copy().convert("RGBA")

        result_image.paste(np_image,(x,y),mask)
        result_image=pil2tensor(result_image.convert("RGB"))
        mask=mask2tensor(mask)

        return (result_image,mask,)

#将节点类映射到唯一标识符
NODE_CLASS_MAPPINGS={
    "Image_center_paste":Image_center_paste
}

# 节点在UI中显示的名称（可选，默认使用类名）
NODE_DISPLAY_NAME_MAPPINGS={
    "Image_center_paste":"Image_center_paste(My Node)"
}