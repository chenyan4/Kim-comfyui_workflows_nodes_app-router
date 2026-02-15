import cv2
import copy
import numpy as np
import math, os, time
import sys
import torch
from collections import Counter
import mediapipe as mp
from insightface.app import FaceAnalysis

from PIL              import Image,ImageDraw,  ImageOps, __version__

GLOBAL_FACE_APP = None

def pil2tensor(image):
    new_image = image.convert('RGB')
    img_array = np.array(new_image).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array)[None]
    return img_tensor

def tensor2pil(image):
    if len(image.shape) < 3:
        image = image.unsqueeze(0)
    return Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8))

def pilmask2tensor(mask_img):
    mask_tensor = torch.from_numpy(np.array(mask_img.convert('L'))).float()  # 转换为float类型
    mask_tensor = mask_tensor / 255.0  # 归一化到 0-1 范围
    mask_tensor = mask_tensor.unsqueeze(0)
    return mask_tensor

def is_mask(image, threshold=0.95):
    image = image.convert('RGB')
    pixels = list(image.getdata())
    total_pixels = len(pixels)
    color_distribution = Counter(pixels)
    black_count = 0
    white_count = 0
    for k, v in color_distribution.items():
        if all(i < 10 for i in k):
            black_count += v
        elif all(j > 245 for j in k):
            white_count += v
    black_white_ratio = (black_count + white_count) / total_pixels
    return black_white_ratio >= threshold


def get_min_bounding_box(img):
    img = img.convert('L')
    img_array = np.array(img)

    # find all pixl is white area
    coords = np.argwhere(img_array > 200)

    if coords.size == 0:
        return (0, 0, img.width, img.height)

    # get the boundary of minimal outer rectangle
    left = coords[:, 1].min()
    top = coords[:, 0].min()
    right = coords[:, 1].max()
    bottom = coords[:, 0].max()

    return (left, top, right, bottom)

def apply_mask(image, mask):
    if image == mask:
        return image
    # Image.load is inplace operator. We need a new image to return
    image = copy.deepcopy(image)
    image_pixels = image.load()
    mask_pixels = copy.deepcopy(mask).convert('RGB').load()
    for i in range(image.width):
        for j in range(image.height):
            # if mask_pixels[i, j] == (0, 0, 0):
            if mask_pixels[i, j][0] < 100:
                image_pixels[i, j] = (255, 255, 255)
    return image

def normalnize_pixel(pixel):
    max_vale = 0
    for k in pixel:
        max_vale = max(max_vale, k)
    if max_vale <= 255:
        return pixel
    factor = 255 * max_vale / max_vale
    blended_pixel = tuple(int(k * factor) for k in  pixel)
    return blended_pixel

def blend_image(images):
    w,h = images[0]['cond_image'].size
    blended_image = Image.new('RGB', (w, h))
    for item in images:
        for i in range(w):
            for j in range(h):
                pixel1 = blended_image.getpixel((i, j))
                pixel2 = item['cond_image'].getpixel((i, j))
                blended_pixel = tuple(
                    int(pixel1[k] + item['scale'] * pixel2[k])
                    for k in range(3)
                )
                blended_pixel = normalnize_pixel(blended_pixel)
                blended_image.putpixel((i, j), blended_pixel)
    return blended_image


def expand_face_box(box, width, height, expand_rate = 1.0):
    left, top, right, bottom = box
    face_w, face_h = right - left, bottom - top
    face_w_dt = face_h_dt = max(int(face_w * expand_rate) , int(face_h * expand_rate))
    center_x, center_y = left + face_w // 2, top + face_h // 2
    face_w = face_h = max(face_w, face_h)
    left, top = max(0, center_x - face_w // 2 - face_w_dt), max(0, center_y - face_h // 2 - face_h_dt)
    right, bottom = min(width, center_x + face_w // 2 + face_w_dt), min(height, center_y + face_h // 2 + face_h_dt)
    left, top, right, bottom = int(left), int(top), int(right), int(bottom)
    return (left, top, right, bottom)


def erode_face_mask(mask, n_pixels):
    kernel = np.ones((n_pixels * 2 + 1, n_pixels * 2 + 1), np.uint8)
    dst = cv2.erode(mask, kernel, iterations=1)
    return dst


def expand_face_mask(mask, n_pixels):
    kernel = np.ones((n_pixels * 2 + 1, n_pixels * 2 + 1), np.uint8)
    expanded_mask = cv2.dilate(mask, kernel, iterations=1)
    return expanded_mask


def draw_circle_by_coordinates(mask, left, top, right, bottom):
    width = right - left
    height = bottom - top
    diameter = min(width, height)
    center = ((left + right) // 2, (top + bottom) // 2)
    mask = cv2.circle(mask, center, diameter // 2, (0, 0, 0), thickness=cv2.FILLED)
    return mask


def insight_detect_face(insightface, image, only_detect=True):
    ''' image is nparray'''
    face = []
    recognition = None
    if only_detect:
        recognition = insightface.models.pop('recognition', None)
    try:
        for size in [(size, size) for size in range(640, 256, -64)]:
            insightface.det_model.input_size = size # TODO: hacky but seems to be working
            face = insightface.get(image)
            if face:
                break
        if len(face) > 0:
            face.sort(key=lambda x: (x['bbox'][2] - x['bbox'][0]) * (x['bbox'][3] - x['bbox'][1]), reverse = True) # 降序排序
    finally:
        if only_detect and recognition is not None:
            insightface.models['recognition'] = recognition
    return face


class ManualFaceAnalysis:
    # 'bbox', 'kps', # not 'landmark_3d_68', 'pose', 'landmark_2d_106', provider

    def __init__(self, model_path, providers, name='buffalo_l'):
        self.models = {}
        # Default detection model for buffalo_l
        # Try finding the model in the expected subdirectory first
        det_model_path = os.path.join(model_path, 'models', name, 'det_10g.onnx')
        if not os.path.exists(det_model_path):
             # Try direct path just in case
             det_model_path = os.path.join(model_path, 'det_10g.onnx')
        
        if not os.path.exists(det_model_path):
             raise FileNotFoundError(f"Detection model not found at {det_model_path}")
        from insightface.model_zoo import get_model
        print(f"InsightFace: Manually loading detection model from {det_model_path}...")
        self.det_model = get_model(det_model_path, providers=providers)
        self.det_model.prepare(ctx_id=0, input_size=(640, 640))
        
    def get(self, img):
        bboxes, kpss = self.det_model.detect(img, max_num=0, metric='default')
        if bboxes is None:
            return []
        ret = []
        for i in range(bboxes.shape[0]):
            bbox = bboxes[i, 0:4]
            det_score = bboxes[i, 4]
            kps = None
            if kpss is not None:
                kps = kpss[i]
            face = dict(bbox=bbox, kps=kps, det_score=det_score)
            ret.append(face)
        return ret

class BaseDetector():
    def __init__(self, model_path, device='cpu'):
        self.model = None
        self.model_path = model_path
        self.device = device

    @torch.no_grad()
    def __call__(self, image, **kwargs):
        if self.model == None:
            self.load()
        self.to_device()
        try:
            return self.forward(image, **kwargs)
        finally:
            self.to_cpu()

    def forward(self, image, detect_resolution=512, image_resolution=512):
        return self.model(
            image,
            detect_resolution=detect_resolution,
            image_resolution=image_resolution
        )

    def load(self):
        raise Exception('Not implemented')

    def to_cpu(self):
        self.model.to('cpu')

    def to_device(self):
        self.model.to(self.device)


class FaceDetector(BaseDetector):
    def __init__(self, model_path=None, device='cpu'):
        global GLOBAL_FACE_APP
        try:
            import folder_paths
            new_model_path = folder_paths.get_full_path('insightface')
            if model_path is None:
                model_path = new_model_path
        except Exception as e:
            pass
        
        # Fallback to default if path is still None
        if model_path is None:
             model_path = os.path.expanduser("~/.insightface")
             
        self.model = None
        self.model_path = model_path
        self.device = device
        
        # Initialize MediaPipe (New API with Fallback)
        self.use_new_api = False
        self.face_landmarker = None
        self.face_mesh = None
        self.face_detection = None
        
        try:
            import mediapipe.tasks.python as mp_python
            from mediapipe.tasks.python import vision
            import urllib.request
            
            self.use_new_api = True
            
            # # Download model
            # model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
            model_dir = os.path.expanduser("~/.mediapipe")
            os.makedirs(model_dir, exist_ok=True)
            model_file = os.path.join(model_dir, "face_landmarker.task")
            
            # if not os.path.exists(model_file):
            #     print(f"Downloading FaceLandmarker model to {model_file}...")
            #     urllib.request.urlretrieve(model_url, model_file)
            
            base_options = mp_python.BaseOptions(model_asset_path=model_file)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options, output_face_blendshapes=False, output_facial_transformation_matrixes=False, num_faces=1)
            self.face_landmarker = vision.FaceLandmarker.create_from_options(options)
            # print(">> MediaPipe FaceLandmarker (New API) initialized.")
            
        except Exception as e:
            # print(f">> MediaPipe New API initialization failed: {e}. Falling back to Legacy API.")
            self.use_new_api = False
            
        if not self.use_new_api:
            try:
                self.face_mesh = mp.solutions.face_mesh
                self.face_detection = mp.solutions.face_detection
                # print(">> MediaPipe FaceMesh (Legacy API) initialized.")
            except Exception as e:
                # print(f">> MediaPipe Legacy API initialization failed: {e}")
                self.face_mesh = None
                self.face_detection = None
        
        # Initialize InsightFace with Singleton Pattern
        if GLOBAL_FACE_APP is not None:
            self.FACE_APP = GLOBAL_FACE_APP
            # print(f"InsightFace: Using cached model instance.")
        else:
            try:
                start_time = time.time()
                print(f"InsightFace: Start loading models from {self.model_path} at {time.strftime('%X')}...")
                
                # Try manual loading first for speed (only works for buffalo_l structure)
                try:
                    self.FACE_APP = ManualFaceAnalysis(self.model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], name='buffalo_l')
                    print(f"InsightFace: Manual loading successful.")
                except Exception as e:
                    print(f"InsightFace: Manual loading failed ({e}), falling back to FaceAnalysis...")
                    # Only load detection model to speed up initialization
                    self.FACE_APP = FaceAnalysis(name='buffalo_l', root=self.model_path,
                                                    allowed_modules=['detection'],
                                                    providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                    
                    # Auto-detect context id based on applied providers
                    ctx_id = 0
                    try:
                        if self.FACE_APP.models:
                            first_model = next(iter(self.FACE_APP.models.values()))
                            if hasattr(first_model, 'session'):
                                providers = first_model.session.get_providers()
                                if 'CUDAExecutionProvider' not in providers:
                                    ctx_id = -1
                                    print("InsightFace: CUDA provider not active, falling back to CPU (ctx_id=-1)")
                    except Exception as ex:
                        print(f"InsightFace: Error checking providers, defaulting to ctx_id={ctx_id}. Error: {ex}")

                    self.FACE_APP.prepare(ctx_id=ctx_id, det_size=(640, 640))
                
                GLOBAL_FACE_APP = self.FACE_APP
                end_time = time.time()
                print(f"InsightFace: Models loaded successfully in {end_time - start_time:.2f} seconds.")
            except Exception as e:
                print(f"InsightFace initialization failed: {e}")
                self.FACE_APP = None

    @classmethod
    def INPUT_TYPES(cls):
        fit_mode = ["all", 'crop']
        return {
                    "required": {
                        "input_image": ("IMAGE",),
                        "fit": (fit_mode, "all"),
                        "expand_rate": ("FLOAT", { "default": 0.5, "min": 0.1, "max": 5.0, "step": 0.05, }),
                        "only_one": ("BOOLEAN", { "default": True, }),
                        "invert": ("BOOLEAN", { "default": False, }),
                    }
                }
    CATEGORY = "zdx/face"
    RETURN_TYPES = ("IMAGE", "MASK", "BOX",)
    RETURN_NAMES = ("image", "mask", "original_size",)
    FUNCTION = "call"
    DESCRIPTION = """
get face mask or face area
"""
    def call(self, input_image, fit, expand_rate=0.5, only_one=False, invert=False):
        im = tensor2pil(input_image)
        if fit != 'crop':
            res = self.forward(im, expand_rate=expand_rate, only_one = only_one, invert=invert)
            return (pil2tensor(res), pilmask2tensor(res), None)
        out_image, mask, box = self.call_pil(im, fit, expand_rate)
        return (pil2tensor(out_image), pil2tensor(mask), box)

    def call_pil(self, input_image, fit, expand_rate=0.5):
        im = input_image
        faces_infos = self.detect_face_area(np.array(im), None, expand_rate=expand_rate)
        if len(faces_infos) == 0:
            return (input_image, None, None)
        face_info = faces_infos[0]
        crop_images = []
        for face_ in faces_infos:
            crop_images.append(im.crop(face_['box']))
        box = face_info['box']
        if fit in ['mask', 'all']:
            face_crop_np = np.array(crop_images[0])
            mask = self.face_mask(face_crop_np)
        return (crop_images[0], mask, box)

    def normalized_point(self, width: int, height: int, point) -> tuple[int, int]:
        x = min(math.floor(point.x * width),  width - 1)
        y = min(math.floor(point.y * height), height - 1)
        return (x, y)
    
    def forward(self, image, **kwargs):
        if not hasattr(self, 'FACE_APP'):
            self.load()
        w, h = image.size
        image_np = np.array(image)
        expand_rate = kwargs.get("expand_rate", 0.5)
        invert = kwargs.get("invert", True)
        faces = self.detect_face_area(image_np, self.face_detection, expand_rate)
        color = (255, 255, 255)  if invert else (0,0,0)
        mask = Image.new(mode='RGB', size=image.size, color=color)
        if len(faces) == 0:
            return mask
        for face in faces:
            bbox = face['box']
            face_crop_ = image.crop(bbox)
            face_crop_np = np.array(face_crop_)
            mask_ = self.face_mask(face_crop_np)
            mask_ = mask_.convert('RGB')
            for x in range(mask_.width):
                for y in range(mask_.height):
                    pixel_value = mask_.getpixel((x, y))
                    mask.putpixel(
                        (face['box'][0]+x, face['box'][1] + y), 
                        # pixel_value if invert else (255, 255, 255)
                        pixel_value  if invert else (255-pixel_value[0], 255-pixel_value[1], 255-pixel_value[2])
                        # pixel_value if invert else (255, 255, 255)
                    )
            if kwargs.get("only_one", True):
                # only draw one face
                break
            # mask.paste(mask_, box=face['box'])

        rgba_mask = np.zeros((h, w, 4), np.uint8)
        rgba_mask[:, :, :3] = np.array(mask)
        white_pixels = np.where(np.all(rgba_mask < 100, axis=1))
        rgba_mask[white_pixels[0], white_pixels[1], 3] = 200 if invert else 0
        # mask_3d = rgba_mask[:, :, :-1]
        # masked_image = cv2.bitwise_and(mask_3d, image_np)
        # Image.fromarray(masked_image).save("masked_image.png")
        mask = Image.fromarray(rgba_mask)
        processed_mask = Image.eval(
            ImageOps.invert(mask.split()[-1]),
            lambda a: 255 if a > 128 else 0)
        mask.putalpha(processed_mask)
        return mask
    
    def face_mask(self, image):
        # image is numpy array (H, W, 3) in RGB format (from PIL)
        h, w = image.shape[:2]
        # Initialize white image to match original behavior (Black Face on White BG)
        blank_image = np.zeros((h, w), dtype=np.uint8)
        blank_image[:] = 255

        # 1. Try New API (FaceLandmarker)
        if self.use_new_api and self.face_landmarker:
            try:
                import mediapipe as mp
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
                detection_result = self.face_landmarker.detect(mp_image)
                
                if detection_result.face_landmarks:
                    for face_landmarks in detection_result.face_landmarks:
                        landmarks = [[int(l.x * w), int(l.y * h)] for l in face_landmarks]
                        hull = cv2.convexHull(np.array(landmarks))
                        blank_image = cv2.fillPoly(blank_image, [np.array(hull)], (0))
                    return Image.fromarray(blank_image)
            except Exception as e:
                print(f">> FaceLandmarker processing failed: {e}")
                # Fall through to legacy if new API fails at runtime
        
        # 2. Try Legacy API (FaceMesh)
        # Re-initialize if necessary (lazy loading fallback)
        if self.face_mesh is None and not self.use_new_api:
             try:
                import mediapipe as mp
                self.face_mesh = mp.solutions.face_mesh
                self.face_detection = mp.solutions.face_detection
                print(">> MediaPipe FaceMesh (Legacy) re-initialized.")
             except Exception as e:
                print(f">> MediaPipe FaceMesh re-init failed: {e}")
                return Image.fromarray(blank_image)

        if self.face_mesh:
            try:
                with self.face_mesh.FaceMesh(
                    static_image_mode=True,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5) as face_mesh_:
                    results = face_mesh_.process(image)
                    
                if results and results.multi_face_landmarks:
                    for face_landmarks in results.multi_face_landmarks:
                        landmarks = [[int(l.x * w), int(l.y * h)] for l in face_landmarks.landmark]
                        hull = cv2.convexHull(np.array(landmarks))
                        blank_image = cv2.fillPoly(blank_image, [np.array(hull)], (0))
                    return Image.fromarray(blank_image)
            except Exception as e:
                print(f">> FaceMesh processing failed: {e}")
            
        return Image.fromarray(blank_image)
    
    def detect_face_area(self, image, mp_face_detection=None, expand_rate=0.05):
        """detect face using InsightFace, input as cv2 image rgb """
        if not hasattr(self, 'FACE_APP'):
            self.load()
        face_info = insight_detect_face(self.FACE_APP, cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR))
        if len(face_info) < 1:
            return []
        # res = ['bbox', 'kps', 'landmark_3d_68', 'pose', 'landmark_2d_106', 'embedding']
        height, width, _ = image.shape
        for face_ in face_info:
            face_['box'] = face_['bbox']
            face_['box_old'] = face_['box']
            
            # Extract landmarks for alignment (Triangle: Left Eye, Right Eye, Mouth Center)
            if 'kps' in face_:
                kps = face_['kps']
                if len(kps) >= 5:
                    # InsightFace KPS: 0=LeftEye, 1=RightEye, 2=Nose, 3=LeftMouth, 4=RightMouth
                    face_['left_eye'] = kps[0]
                    face_['right_eye'] = kps[1]
                    # Calculate mouth center from corners
                    face_['mouth_center'] = (kps[3] + kps[4]) / 2.0
                    face_['triangle'] = np.array([face_['left_eye'], face_['right_eye'], face_['mouth_center']])

            bbox = face_['bbox'].astype(int)
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
            face_w = right - left
            face_h = bottom - top
            center_x = left + face_w // 2
            center_y = top + face_h // 2
            face_['width'] = face_w
            face_['height'] = face_h
            face_['center'] = (center_x, center_y)
            face_['area'] = face_w * face_h
            # Calculate expanded dimensions
            face_w_dt = int(face_w * expand_rate)
            face_h_dt = int(face_h * expand_rate)
            # Calculate new box based on center
            face_['box'] = (
                max(0, center_x - face_w // 2 - face_w_dt),
                max(0, center_y - face_h // 2 - face_h_dt),
                min(width, center_x + face_w // 2 + face_w_dt),
                min(height, center_y + face_h // 2 + face_h_dt)
            )
            face_['box'] = tuple(map(int, face_['box']))
            
        return face_info
    
    def detect_face_area_v2(self, image, mp_face_detection, expand_rate=0.05):
        """DETECT face using MediaPipe, input as cv2 image rgb """
        faces = []
        with mp_face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=0.5) as face_detection:
            results = face_detection.process(image)
            if not results.detections:
                return faces
            height, width, _ = image.shape
            for detection in results.detections:
                keypoints = dict()
                kpt_dc = {"left_eye": mp_face_detection.FaceKeyPoint.LEFT_EYE, "right_eye":mp_face_detection.FaceKeyPoint.RIGHT_EYE, 
                    "nose_tip":mp_face_detection.FaceKeyPoint.NOSE_TIP, "mouth_center":mp_face_detection.FaceKeyPoint.MOUTH_CENTER, 
                    "right_ear_tragion":mp_face_detection.FaceKeyPoint.RIGHT_EAR_TRAGION, "left_ear_tragion":mp_face_detection.FaceKeyPoint.LEFT_EAR_TRAGION}
                for k,v in kpt_dc.items():
                    kpt_point = mp_face_detection.get_key_point(detection, v)
                    keypoints[k] = self.normalized_point(width, height, kpt_point)
                keypoints['bounding_box'] = dict()
                keypoints['bounding_box']['xmin'] = int(detection.location_data.relative_bounding_box.xmin * width)
                keypoints['bounding_box']['ymin'] = int(detection.location_data.relative_bounding_box.ymin * height)
                keypoints['bounding_box']['width'] = int(detection.location_data.relative_bounding_box.width * width)
                keypoints['bounding_box']['height'] = int(detection.location_data.relative_bounding_box.height * height)
                keypoints['score'] = float(detection.score[0])
                keypoints['confidence'] = keypoints['score']
                keypoints['box'] = [keypoints['bounding_box']['xmin'], keypoints['bounding_box']['ymin'],
                                    keypoints['bounding_box']['width'], keypoints['bounding_box']['height']]
                rect_start_point = (keypoints['box'][0], keypoints['box'][1])
                rect_end_point = (keypoints['box'][0]+keypoints['box'][2], keypoints['box'][1]+keypoints['box'][3])
                left, top, right, bottom = rect_start_point[0], rect_start_point[1], rect_end_point[0], rect_end_point[1]
                face_points = [ 
                    (keypoints[k][0],
                     keypoints[k][1] )  for k in ['left_eye', 'right_eye', 'mouth_center'] 
                ]
                center_x, center_y = np.mean(face_points, axis=0)
                face_w, face_h = keypoints['bounding_box']['width'],  keypoints['bounding_box']['height']
                face_w_dt, face_h_dt = int(face_w * expand_rate) , int(face_h * expand_rate)
                left, top, right, bottom = max(0, left - face_w_dt), max(0, top - face_h_dt), \
                                        min(width, right + face_w_dt), min(height, bottom + face_h_dt)
                box_old = (left, top, right, bottom)
                left, top = max(0, center_x - face_w // 2 - face_w_dt), max(0, center_y - face_h // 2 - face_h_dt)
                right, bottom = min(width, center_x + face_w // 2 + face_w_dt), min(height, center_y + face_h // 2 + face_h_dt)
                left, top, right, bottom = int(left), int(top), int(right), int(bottom)
                faces.append(
                    {'box': (left, top, right, bottom), 'box_old': box_old,
                     'key_point':  keypoints, 'center': (center_x, center_y)}
                )
        return faces

def is_empty_image(image):
    if image is None:
        return True
    if image.mode == 'RGBA':
        extrema = image.getextrema()
        if extrema[3][1] == 0:
            return True
    return False


class InsightFaceCrop:
    """
    crop face  and get face eye and mouth center triangle,
    """
    def __init__(self, face_masker):
        self.face_masker = FaceDetector(model_path = None)

    @classmethod
    def INPUT_TYPES(cls):
        fit_mode = ["all", 'crop']
        return {
                    "required": {
                        "input_image": ("IMAGE",),
                        "expand_rate": ("FLOAT", { "default": 0.5, "min": 0.1, "max": 5.0, "step": 0.05, }),
                        "index": ("INT", { "default": 0, "min": 0, "max": 4, "step": 1, }),
                    }
                }
    CATEGORY = "zdx/face"
    RETURN_TYPES = ("IMAGE", "TRIANGLE", "BOX",)
    RETURN_NAMES = ("image", "triangle", "box",)
    FUNCTION = "crop"
    DESCRIPTION = """
crop face area from image, and get triangle of face eye and mouth center to align face
"""
    def crop(self, input_image, expand_rate=0.5, index=0):
        image = tensor2pil(input_image)
        face_info2 = self.face_masker.detect_face_area(np.array(image), None, expand_rate=expand_rate)
        
        if len(face_info2) == 0:
            return (input_image, np.zeros((3,2)), [0,0,0,0])
            
        # sort face_info2 by area   
        face_info2 = sorted(face_info2, key=lambda x: x['width'] * x['height'], reverse=True)
        
        if index >= len(face_info2):
            index = 0
            
        selected_face = face_info2[index]
        box = selected_face['box']
        
        face_crop_ = image.crop(box)
        
        # Adjust triangle coordinates to be relative to the crop
        triangle = selected_face['triangle'].copy()
        triangle[:, 0] -= box[0]
        triangle[:, 1] -= box[1]
        
        return (pil2tensor(face_crop_), triangle, list(box))

class FaceAlignScale:
    """
    align face by triangle, and scale face to target size
    通过inisight_face 检测人脸，如果想将两个大小角度不一样的人脸进行缩放到相同大小并对其的话（比如贴上去）调整透明度，算法思路是啥
    将目标人脸缩放到与源人脸相同的大小和角度（只缩放旋转，不扭曲），然后替换。
    """
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "target_image": ("IMAGE",),
                "source_triangle": ("TRIANGLE",),
                "target_triangle": ("TRIANGLE",),
            }
        }
    
    CATEGORY = "zdx/face"
    RETURN_TYPES = ("IMAGE", "TRIANGLE", "BOX",)
    RETURN_NAMES = ("aligned_image", "aligned_triangle", "box",)
    FUNCTION = "align"
    
    def align(self, source_image, target_image, source_triangle, target_triangle):
        # source_image: tensor [1, H, W, C]
        # source_triangle: numpy array [[x1,y1], [x2,y2], [x3,y3]] (left_eye, right_eye, mouth_center)
        src_img_pil = tensor2pil(source_image)
        tgt_img_pil = tensor2pil(target_image)
        # 1. Calculate centers of triangles
        src_center = np.mean(source_triangle, axis=0)
        tgt_center = np.mean(target_triangle, axis=0)
        # 2. Calculate scale (based on eye distance)
        # Eye distance is distance between point 0 and 1
        src_eye_dist = np.linalg.norm(source_triangle[0] - source_triangle[1])
        tgt_eye_dist = np.linalg.norm(target_triangle[0] - target_triangle[1])
        scale = tgt_eye_dist / src_eye_dist
        # 3. Calculate rotation angle
        # Vector between eyes
        src_eye_vec = source_triangle[1] - source_triangle[0]
        tgt_eye_vec = target_triangle[1] - target_triangle[0]
        src_angle = np.degrees(np.arctan2(src_eye_vec[1], src_eye_vec[0]))
        tgt_angle = np.degrees(np.arctan2(tgt_eye_vec[1], tgt_eye_vec[0]))
        rotation_angle = tgt_angle - src_angle
        
        # 4. Perform affine transformation
        # Translation to center -> Rotate & Scale -> Translation to new center
        # Since we want to align src_center to tgt_center
        
        # We can use OpenCV's getRotationMatrix2D for rotation and scaling around a center
        M = cv2.getRotationMatrix2D((float(src_center[0]), float(src_center[1])), float(rotation_angle), float(scale))
        
        # Adjust translation part of the matrix to align centers
        # Current transformation maps src_center to itself (because we rotated around it)
        # We need to add translation (tgt_center - src_center)
        M[0, 2] += (tgt_center[0] - src_center[0])
        M[1, 2] += (tgt_center[1] - src_center[1])
        
        w, h = src_img_pil.size
        # Use target image size for output canvas
        tgt_w, tgt_h = tgt_img_pil.size
        
        aligned_img_np = cv2.warpAffine(
            np.array(src_img_pil), 
            M, 
            (tgt_w, tgt_h), 
            flags=cv2.INTER_LINEAR, 
            borderMode=cv2.BORDER_CONSTANT, 
            borderValue=(0,0,0)
        )
        # Transform the source triangle to new coordinates for verification/downstream use
        ones = np.ones(shape=(len(source_triangle), 1))
        points_ones = np.hstack([source_triangle, ones])
        aligned_triangle = M.dot(points_ones.T).T
        return (pil2tensor(Image.fromarray(aligned_img_np)), aligned_triangle, None)


_NODE_CLASS_MAPPINGS = {
    "FaceAlignScale": FaceAlignScale,
    'FaceDetector': FaceDetector,
    'InsightFaceCrop': InsightFaceCrop,

}
_NODE_DISPLAY_NAME_MAPPINGS = {
    "FaceAlignScale": "FaceAlignScale",
    "FaceDetector": "FaceDetector",
    "InsightFaceCrop": "InsightFaceCrop",
}

if __name__ == '__main__':
    from workflows.zdx_comfyui.annotator import FaceDetector
    insight_face_path = '/data/models/insightface/'
    imgs = ['/data/zdx/gallery/5_final_res.png', '/data/zdx/gallery/0018_after.jpeg']

    face_masker = FaceDetector(model_path = insight_face_path)
    # insight_detect_face(face_masker.FACE_APP, cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))
    for im in imgs:
        im = Image.open(im)
        
        face_info2 = face_masker.detect_face_area(np.array(im), None, expand_rate=0.5)
        face_crop_ = im.crop(face_info2[0]['box'])
        face_crop_.save('1__face_crop.png')
        # face_info = face_masker.detect_face_area_v2(np.array(im), face_masker.face_detection)
            
        face_mask = face_masker.forward(image = im, expand_rate=0.5, invert=False)
        face_mask.save('1__face_mask.png')
        new_im = Image.new('RGBA', im.size, 255)
        new_im.paste(im=im, mask=face_mask.convert('L'))
        # new_im.putalpha(face_mask.convert('L'))
        # face_mask.save('1__face_mask.png')
        # new_im.save('1__face_mask.png')

    # ----人脸对齐测试gu----------
    aliner =  FaceAlignScale()
    image1 = imgs[0]
    image2 = imgs[1]
    source_image = Image.open(image1)
    target_image = Image.open(image2)
    face_info_source = face_masker.detect_face_area(np.array(source_image), None, expand_rate=0.5)
    face_info_target = face_masker.detect_face_area(np.array(target_image), None, expand_rate=0.5)
    if len(face_info_source) == 0 or len(face_info_target) == 0:
        raise ValueError("No face detected in one of the images")
    source_triangle = face_info_source[0]['triangle']
    target_triangle = face_info_target[0]['triangle']
    aligned_image, aligned_triangle, box = aliner.align(pil2tensor(source_image), pil2tensor(target_image), source_triangle, target_triangle)
    source_image_face_crop_ = source_image.crop(face_info_source[0].box)
    source_image_face_crop_.save('1__source_image_face_crop.png')
    target_image_face_crop_ = target_image.crop(face_info_target[0].box)
    target_image_face_crop_.save('1__target_image_face_crop.png')
    tensor2pil(aligned_image).save('1__aligned_image.png')

