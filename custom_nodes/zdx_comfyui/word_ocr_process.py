# word detect
# pip install paddleocr==3.2.0
# pip install paddlepaddle
import gradio as gr
from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import numpy as np
# from .layerstyle.poisson_blend import blend
# --poisson blend
from scipy.sparse import csr_matrix
from pyamg.gallery import poisson
from pyamg import ruge_stuben_solver


import os
import dashscope
os.environ["DASHSCOPE_API_KEY"] = "sk-e7fb5219c71a446bb862ae04bc709a62"
# sk-b6695b8fdb7c4923a11577a5e62e44c5
# sk-npGQpPfAFwepce8sQnxEh2c2KJxZPb6LPSS6mvH361uMKZPu moone shot
# Generate your api key from: https://platform.moonshot.cn/console/api-keys
os.environ["MOONSHOT_API_KEY"] = "sk-npGQpPfAFwepce8sQnxEh2c2KJxZPb6LPSS6mvH361uMKZPu"
LANG_LIST = ['English', 'zh', 'Thai']

# Initialize global variables
ocr = None
font_path = None
image_text_translate = None

def qwen_translate(text, lang='English'):
    messages = [
        {"role": "user", "content": text}
    ]
    # 设置翻译选项
    translation_options = {"source_lang": "auto", "target_lang": lang} #   # 自动检测源语言 目标语言为英文
    response = dashscope.Generation.call(
        api_key=os.environ["DASHSCOPE_API_KEY"], model="qwen-mt-turbo", messages=messages, result_format='message', translation_options=translation_options
    )
    # 输出翻译结果
    if len(response.output.choices)>0:
        return response.output.choices[0].message.content
    return ""

def expand_bbox(img, bbox, padding=5):
    """扩展边界以获取周围区域（边框外5像素）"""
    # 扩展边界以获取周围区域（边框外5像素）
    padding = 5
    border_min_x = max(0, bbox[0] - padding)
    border_min_y = max(0, bbox[1] - padding)
    border_max_x = min(img.width - 1, bbox[2] + padding)
    border_max_y = min(img.height - 1, bbox[3] + padding)
    return (border_min_x, border_min_y, border_max_x, border_max_y)

def fit_font(draw, text, bbox_height, font_path, max_iter=10):
    """
    返回: ImageFont 对象, 实际占用 (width, height)
    """
    low, high = 1, bbox_height * 2          # 搜索上下界
    best = None
    for _ in range(max_iter):
        mid = (low + high) / 2
        font = ImageFont.truetype(font_path, int(mid))
        # 使用 textbbox 替代 textsize
        left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
        w, h = right - left, bottom - top  # 计算文本宽度和高度
        if h <= bbox_height:
            best = (font, w, h)
            low = mid
        else:
            high = mid
    if best:
        return best
    # 如果没有找到合适的字体大小，使用最后一次尝试的字体
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    w, h = right - left, bottom - top
    return (font, w, h)

def get_background_color(img):
    """获取文字框周围的背景颜色"""    
    # ---------- 2. 取“底色” ----------
    # 2.1 用 numpy 快速裁出来
    crop_np = np.array(img)
    # 2.2 简单 K=2 聚类，找出背景色（假设文字是少数类）
    pixels = crop_np.reshape(-1, 3).astype(np.float32)
    # 只用 2 个中心点，迭代 1 次就够
    centroids = pixels[::max(1, len(pixels)//5000)]      # 下采样加速
    bg_lab = centroids.mean(axis=0).round().astype(int)  # 均值当背景色
    bg_rgb = tuple(bg_lab)
    return bg_rgb

def _poisson_reconstruct(src, mask):
    """
    src:  (H,W,3)  uint8
    mask: (H,W)    True=需要重建的洞
    返回重建后的 (H,W,3) uint8
    """
    H, W = src.shape[:2]
    src = src.astype(float)
    out = src.copy()
    # 拉普拉斯矩阵（按列优先展平）
    A = poisson((H, W), format='csr')          # 形状 (H*W, H*W)
    mask1d = mask.ravel()                      # 一维布尔索引
    n_interior = mask1d.sum()                  # 洞像素个数
    for c in range(3):
        f = src[:, :, c]
        b_flat = A @ f.ravel()                 # 右端项 = 原图拉普拉斯
        # 只保留洞内部未知量
        A_int = A[mask1d][:, mask1d]           # (n_interior, n_interior)
        b_int = b_flat[mask1d]                 # (n_interior,)
        # 求解
        ml = ruge_stuben_solver(csr_matrix(A_int))
        x_int = ml.solve(b_int, tol=1e-10)
        # 写回
        out_flat = out[:, :, c].ravel()
        out_flat[mask1d] = np.clip(x_int, 0, 255)
        out[:, :, c] = out_flat.reshape(H, W)
    return out.astype(np.uint8)

def seamless_inpaint(img, bbox, pad=10):
    left, top, right, bottom = bbox
    big = img.crop((left-pad, top-pad, right+pad, bottom+pad))
    np_big = np.array(big)
    h, w = np_big.shape[:2]
    mask = np.zeros((h, w), dtype=bool)
    mask[pad:-pad, pad:-pad] = True
    # 1. 环内平均色当初始值
    bg_color = np_big[~mask].mean(0).round().astype(np.uint8)
    tmp = np_big.copy()
    tmp[mask] = bg_color
    # 2. 泊松重建
    reconstructed = _poisson_reconstruct(tmp, mask)
    # 3. 贴回
    img.paste(Image.fromarray(reconstructed), (left-pad, top-pad))
    return img

def seamless_inpaint_corner(img, bbox, pad=10):
    left, top, right, bottom = bbox
    big = img.crop((left-pad, top-pad, right+pad, bottom+pad))
    np_big = np.array(big)
    h, w = np_big.shape[:2]
    # 1. 环内像素均值
    mask = np.zeros((h, w), dtype=bool)
    mask[pad:-pad, pad:-pad] = True
    bg_color = tuple(np_big[~mask].mean(0).round().astype(int))
    # 2. 填色
    tmp = big.copy()
    ImageDraw.Draw(tmp).rectangle([(pad, pad), (w-pad-1, h-pad-1)], fill=bg_color)
    # 3. 高斯羽化 3 像素，让边缘过渡自然
    tmp = tmp.filter(ImageFilter.GaussianBlur(3))
    # 4. 贴回
    img.paste(tmp, (left-pad, top-pad))
    return img

class ImageTextTranslate():
    def __init__(self, font_path, qwen_translate, ocr=None):
        self.ocr = ocr
        if not ocr:
            self.ocr = PaddleOCR(
                text_detection_model_name="PP-OCRv5_server_det",
                text_recognition_model_name="PP-OCRv5_server_rec",
                use_doc_orientation_classify=False, # 通过 use_doc_orientation_classify 参数指定不使用文档方向分类模型
                use_doc_unwarping=False, # 通过 use_doc_unwarping 参数指定不使用文本图像矫正模型
                use_textline_orientation=False, # 通过 use_textline_orientation 参数指定不使用文本行方向分类模型
                device='gpu',
            ) # 更换 PP-OCRv5_server 模型
        self.font_path = font_path
        self.qwen_translate = qwen_translate

    def forward(self, image, lang='Thai'):
        # Convert RGBA to RGB if needed
        img = image.copy()
        if img.mode == 'RGBA':
            img = img.convert('RGB')
            
        img_draw = ImageDraw.Draw(img)
        result = self.ocr.predict(np.array(img))

        if len(result) < 1:
            return image
        res = result[0]
        for i, (text, confidence, rec_polys) in enumerate(zip(res['rec_texts'], res['rec_scores'], res['rec_polys'])):
            x,y = rec_polys[0]
            w,h = rec_polys[2][0] - x, rec_polys[2][1] - y
            bbox = [x,y,x+w,y+h]
            img = seamless_inpaint_corner(img, bbox)  # .save(output_path.replace('.jpg', f'_clean.jpg'))

            # 计算文字框的边界；
            left, top, right, bottom = bbox
            box_h = bottom - top
            box_h = int(box_h * 0.7)

            replacement_text = "Hello test"
            translation = self.qwen_translate(text, lang=lang)
            replacement_text = translation
            font, txt_w, txt_h = fit_font(img_draw, replacement_text, box_h, self.font_path)
            # 3. 垂直居中：PIL 的 y 是 baseline，需要补偿 ascender
            asc = font.getmetrics()[0]          # 上行高度
            y = top # + (box_h) // 2 + asc     # y = top + (box_h - txt_h) // 2 + asc
            x = left + (right - left - txt_w) // 2   #  x = left + (right - left - txt_w) // 2
            img_draw.text((x, y), replacement_text, font=font, fill=(0, 128, 0))

        return img

def translate_image(image, lang='Thai'):
    global ocr, font_path, image_text_translate
    
    # Handle image input
    if isinstance(image, dict):
        image = Image.open(image['path'])
    if not image:
        return [image], {'lang': lang}
    
    # Ensure image_text_translate is initialized
    if image_text_translate is None:
        if ocr is None or font_path is None:
            return [image], {'error': 'OCR or font_path not initialized'}
        image_text_translate = ImageTextTranslate(font_path, qwen_translate, ocr)
    
    # Process the image
    res_image = image_text_translate.forward(image, lang)
    # Return a list containing the image for Gallery component
    return [res_image], {'lang': lang}

def play_ui():
    theme = gr.themes.Base(
        primary_hue='green', neutral_hue='neutral'
    ).set(
        slider_color='#FF3333',
        checkbox_border_color_selected='#17A34A',
        checkbox_background_color_selected='#17A34A'
    )
    
    with gr.Blocks(theme=theme, analytics_enabled=False) as block:
        with gr.Row():
            gr.HTML(
                '<h3 style="margin:0"> 🎡 <span style="color:#FF3333">image text translate PLAY</span><span style="color:#559955">GROUND</span><sup> </sup></h3>'
            )
        with gr.Row():
            with gr.Column(scale=1):
                with gr.Row():
                    init_image = gr.ImageEditor(
                            sources=['upload', 'clipboard'], type='pil', image_mode='RGB', 
                            # fixed_canvas=False, container=False,
                            # show_label=True, layers=False,
                            # label='InitImage',
                            # transforms=("crop", "resize"), 
                            # brush=gr.Brush(colors=["#007F0064", "#7F000064", "#00007F64"], color_mode='defaults', default_size=180) # size color with alpha adjustable
                        )
                with gr.Row():
                    generate = gr.Button(
                        value='Generate', variant='primary', scale=1,# min_width=128
                    )
                    lang = gr.Dropdown(choices=list(LANG_LIST), value=LANG_LIST[-1], label = '翻译语言')
            with gr.Column(scale=1):
                with gr.Row():
                    gallery = gr.Gallery(
                        label='Gallery结果', height=380, preview=True
                    )
        with gr.Row():
            meta = gr.JSON(
                label='Meta', scale=1
            )
            
        # Move the click event inside the Blocks context
        generate.click(
            translate_image,
            inputs=[init_image, lang],
            outputs=[gallery, meta],
            preprocess=False
        )
    return block

if __name__ == '__main__':
    # Initialize global variables
    ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_server_det",
            text_recognition_model_name="PP-OCRv5_server_rec",
            use_doc_orientation_classify=False, # 通过 use_doc_orientation_classify 参数指定不使用文档方向分类模型
            use_doc_unwarping=False, # 通过 use_doc_unwarping 参数指定不使用文本图像矫正模型
            use_textline_orientation=False, # 通过 use_textline_orientation 参数指定不使用文本行方向分类模型
            device='gpu',
        ) # 更换 PP-OCRv5_server 模型
    
    font_path = "/data/zdx/acdrive/workflows/zdx_comfyui/fonts/arial.ttf"
    font_path = "/data/zdx/acdrive/workflows/zdx_comfyui/fonts/DejaVuSans.ttf"
    font_path = "/data/zdx/acdrive/workflows/zdx_comfyui/fonts/Itim-Regular-TTF.ttf"

    image = "/data/zdx/gallery/文字替换方案研究.jpg"
    img = Image.open(image)

    # Initialize the translator
    image_text_translate = ImageTextTranslate(font_path, qwen_translate, ocr)
    
    # Create and launch the UI
    block = play_ui()
    block.queue().launch(
        server_name='0.0.0.0', 
        server_port=7860, 
        # allowed_paths=allow_paths,
        show_api=False)


    # 2 call in script way
    if 0:
        img_draw = ImageDraw.Draw(img)
        result = ocr.predict(np.array(img))
        for res in result:
            res.print()
            for i, (text, confidence, rec_polys) in enumerate(zip(res['rec_texts'], res['rec_scores'], res['rec_polys'])):
                print(f"识别的文字: {text}")
                print(f"置信度: {confidence:.4f}")
                # break
                x,y = rec_polys[0]
                w,h = rec_polys[2][0] - x, rec_polys[2][1] - y
                bbox = [x,y,x+w,y+h]
                img = seamless_inpaint_corner(img, bbox)  # .save(output_path.replace('.jpg', f'_clean.jpg'))

                # 计算文字框的边界；
                left, top, right, bottom = bbox
                box_h = bottom - top
                box_h = int(box_h * 0.7)

                # 替换为相同数量的"test"
                replacement_text = "Hello test"
                translation = qwen_translate(text, lang="Thai")
                replacement_text = translation
                font, txt_w, txt_h = fit_font(img_draw, replacement_text, box_h, font_path)
                # 3. 垂直居中：PIL 的 y 是 baseline，需要补偿 ascender
                asc = font.getmetrics()[0]          # 上行高度
                y = top # + (box_h) // 2 + asc     # y = top + (box_h - txt_h) // 2 + asc
                x = left + (right - left - txt_w) // 2   #  x = left + (right - left - txt_w) // 2
                img_draw.text((x, y), replacement_text, font=font, fill=(0, 128, 0))

            output_path = image.replace('.jpg', '_replaced.jpg')
            img.save(output_path)
            print(f"替换文字: {text} -> {replacement_text}")
            print(f"置信度: {confidence:.4f}")

