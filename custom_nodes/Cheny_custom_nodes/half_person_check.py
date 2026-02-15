from PIL import Image
import numpy as np
import cv2
import torch
from ultralytics import YOLO

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

class half_person_check:
    def __init__(self, model_path="/data/models/yolo/yolov8n-pose.pt"):
        self.model = YOLO(model_path)
        self.KEYPOINTS = {
            "nose": 0, "left_eye": 1, "right_eye": 2, "left_ear": 3, "right_ear": 4,
            "left_shoulder": 5, "right_shoulder": 6, "left_elbow": 7, "right_elbow": 8,
            "left_wrist": 9, "right_wrist": 10, "left_hip": 11, "right_hip": 12,
            "left_knee": 13, "right_knee": 14, "left_ankle": 15, "right_ankle": 16
        }
        
        # 定义关键点之间的连接关系（用于绘制骨架）
        self.SKELETON = [
            # 头部连接
            [0, 1], [0, 2], [1, 3], [2, 4],  # nose->eyes->ears
            # 躯干连接
            [5, 6], [5, 11], [6, 12], [11, 12],  # shoulders, hips
            # 左臂
            [5, 7], [7, 9],  # left_shoulder->left_elbow->left_wrist
            # 右臂
            [6, 8], [8, 10],  # right_shoulder->right_elbow->right_wrist
            # 左腿
            [11, 13], [13, 15],  # left_hip->left_knee->left_ankle
            # 右腿
            [12, 14], [14, 16],  # right_hip->right_knee->right_ankle
        ]
        
        # 关键点颜色（BGR格式）
        self.KP_COLORS = [
            (255, 0, 0),    # nose - 红色
            (255, 85, 0),   # left_eye
            (255, 170, 0),  # right_eye
            (255, 255, 0), # left_ear
            (170, 255, 0), # right_ear
            (85, 255, 0),  # left_shoulder
            (0, 255, 0),   # right_shoulder - 绿色
            (0, 255, 85), # left_elbow
            (0, 255, 170),# right_elbow
            (0, 255, 255),# left_wrist
            (0, 170, 255),# right_wrist
            (0, 85, 255), # left_hip
            (0, 0, 255),  # right_hip - 蓝色
            (85, 0, 255), # left_knee
            (170, 0, 255),# right_knee
            (255, 0, 255),# left_ankle
            (255, 0, 170),# right_ankle
        ]
        
        # 连接线颜色（BGR格式）
        self.LINE_COLOR = (0, 255, 255)  # 黄色

    def check_half_person_from_array(self, image_bgr, knee_conf=0.5, ankle_conf=0.8):
        """
        从BGR格式的numpy数组检测姿态
        """
        # conf 参数：检测置信度阈值（confidence threshold）
        # 范围: 0.0 - 1.0
        # 含义: 只有检测到的人体边界框置信度 >= conf 时，才会被返回
        # 例如: conf=0.5 表示只返回置信度 >= 0.5 的检测结果
        # 值越小，检测越宽松（可能包含更多误检）
        # 值越大，检测越严格（只返回高置信度的结果）
        results = self.model(image_bgr, conf=0.5)

        if len(results[0].boxes) == 0:
            return None
        
        # 取第一张图片的所有识别姿态
        # keypoints.data 形状: [num_persons, 17, 3]
        # 第三维格式: [x坐标, y坐标, 置信度]
        keypoints = results[0].keypoints.data.cpu().numpy()
        
        # 取一张图片第一个检测到的人体
        # best_keypoints 形状: [17, 3] - 17个关键点，每个关键点是 [x, y, confidence]
        # 重要：无论关键点是否可见，模型都会返回所有17个关键点的数据
        # 不可见的关键点：置信度接近0，但坐标和置信度值仍然存在
        best_keypoints = keypoints[0].tolist()

        # 检查是否只有下半身
        # best_keypoints[index] 返回格式: [x, y, confidence] - 长度为3的列表
        # 例如: [123.45, 456.78, 0.95] 表示 x=123.45, y=456.78, 置信度=0.95
        left_knee = best_keypoints[self.KEYPOINTS["left_knee"]]      # [x, y, confidence]
        right_knee = best_keypoints[self.KEYPOINTS["right_knee"]]    # [x, y, confidence]
        left_ankle = best_keypoints[self.KEYPOINTS["left_ankle"]]    # [x, y, confidence]
        right_ankle = best_keypoints[self.KEYPOINTS["right_ankle"]]  # [x, y, confidence]
        
    
        if left_knee[2] > knee_conf or right_knee[2] > knee_conf:
            boolean=True
            return image_bgr, best_keypoints, boolean
        elif left_ankle[2] > ankle_conf or right_ankle[2] > ankle_conf:
            boolean=True
            return image_bgr, best_keypoints, boolean
        else:
            boolean=False
            return image_bgr, best_keypoints, boolean

    def check_half_person(self,image_path,knee_conf=0.5,ankle_conf=0.8):
        image=Image.open(image_path)
        image=np.array(image)
        # RGB转BGR
        image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
        
        return self.check_half_person_from_array(image, knee_conf, ankle_conf)

        
    
    def draw_pose(self, image, keypoints_list, confidence_threshold=0.5, line_thickness=4, point_radius=5):
        """
        在图像上绘制姿态图
        
        Args:
            image: 原始图像 (BGR格式)
            keypoints_list: 关键点列表，格式为 [17, 3]，每个关键点是 [x, y, confidence]
            confidence_threshold: 置信度阈值，低于此值的关键点不绘制
            line_thickness: 连接线粗细
            point_radius: 关键点圆圈半径
            
        Returns:
            绘制了姿态的图像
        """
        # 复制图像，避免修改原图
        pose_image = image.copy()
        
        # 将关键点转换为numpy数组便于处理
        if isinstance(keypoints_list, list):
            keypoints = np.array(keypoints_list)
        else:
            keypoints = keypoints_list
        
        # 绘制连接线（骨架）
        for connection in self.SKELETON:
            pt1_idx, pt2_idx = connection
            pt1 = keypoints[pt1_idx]
            pt2 = keypoints[pt2_idx]
            
            # 只有当两个关键点的置信度都超过阈值时才绘制连接线
            if pt1[2] > confidence_threshold and pt2[2] > confidence_threshold:
                pt1_coord = (int(pt1[0]), int(pt1[1]))
                pt2_coord = (int(pt2[0]), int(pt2[1]))
                cv2.line(pose_image, pt1_coord, pt2_coord, self.LINE_COLOR, line_thickness)
        
        # 绘制关键点
        for i, kp in enumerate(keypoints):
            x, y, conf = kp
            if conf > confidence_threshold:
                center = (int(x), int(y))
                color = self.KP_COLORS[i]
                # 绘制实心圆
                cv2.circle(pose_image, center, point_radius, color, -1)
                # 绘制外圈（更明显）
                cv2.circle(pose_image, center, point_radius + 2, (255, 255, 255), 1)
        
        return pose_image
    
        

class half_person_check_node:
    CATEGORY="My Nodes/half person check"
    RETURN_TYPES=("BOOLEAN","IMAGE",)
    RETURN_NAMES=("boolean","pose_image",)
    FUNCTION="check_half"
    
    def __init__(self):
        # 初始化检测器（只加载一次模型）
        self.checker = half_person_check()

    @classmethod
    def INPUT_TYPES(cls):
        return{
            "required":{
                "image":("IMAGE",),
            },
            "optional":{
                "knee_conf":("FLOAT",{
                    "default":0.5,
                    "min":0.0,
                    "max":1.0,
                    "step":0.01,
                }),
                "ankle_conf":("FLOAT",{
                    "default":0.7,
                    "min":0.0,
                    "max":1.0,
                    "step":0.01,
                }),
                "draw_conf":("FLOAT",{
                    "default":0.5,
                    "min":0.0,
                    "max":1.0,
                    "step":0.01,
                }),
                "line_thickness":("INT",{
                    "default":4,
                    "min":1,
                    "max":10,
                    "step":1,
                }),
                "point_radius":("INT",{
                    "default":5,
                    "min":1,
                    "max":10,
                    "step":1,
                })
            }
        }

    def check_half(self, image, knee_conf, ankle_conf, draw_conf, line_thickness, point_radius):
        # 将tensor转换为PIL Image
        pil_image = tensor2pil(image)
        # 转换为numpy数组，然后转BGR格式
        image_np = np.array(pil_image)
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # 检测姿态（传入BGR格式的numpy数组）
        result = self.checker.check_half_person_from_array(image_bgr, knee_conf, ankle_conf)
        
        if result is None:
            # 没有检测到人体，返回原图和False
            return (False, image,)
        
        result_img, result_keypoints, result_boolean = result
        
        # 绘制姿态
        pose_image = self.checker.draw_pose(result_img, result_keypoints, draw_conf, line_thickness, point_radius)
        
        # 将BGR转回RGB，然后转换为tensor
        pose_image_rgb = cv2.cvtColor(pose_image, cv2.COLOR_BGR2RGB)
        pose_image_pil = Image.fromarray(pose_image_rgb)
        pose_image_tensor = pil2tensor(pose_image_pil)
        
        return (result_boolean, pose_image_tensor,)

NODE_CLASS_MAPPINGS={
    "half_person_check_node":half_person_check_node
}

NODE_DISPLAY_NAME_MAPPINGS={
    "half_person_check_node":"half_person_check_node(My Node)"
}
