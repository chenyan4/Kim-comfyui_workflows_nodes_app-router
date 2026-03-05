import base64
import io
import os
from flask import Flask, request, jsonify, render_template
from PIL import Image

from app_comfyui_workflows.change_face.change_face import ChangeFace
from app_comfyui_workflows.change_background.flux2_klein_change_bg import flux2_klein_change_bg
from app_comfyui_workflows.change_background.qwen_change_bg import qwen_change_bg
from app_comfyui_workflows.change_face.flux2_klein_faceswap import flux2_klein_faceswap
from app_comfyui_workflows.change_face.qwen_2511_faceswap import qwen_2511_faceswap
from app_comfyui_workflows.change_background.flux2_klein_one_cb import flux2_klein_one_cb
from app_comfyui_workflows.change_background.qwen_2509_one_cb import qwen_2509_one_cb
from app_comfyui_workflows.person_move.qwen_edit_2509_pose_cb import qwen_edit_2509_pose_cb
from app_comfyui_workflows.text2image.Zimage_text2img import Zimage_text2img
from app_comfyui_workflows.every_change.everything_image import everything_image
from app_comfyui_workflows.light_change.flux2_change_light import flux2_change_light
from app_comfyui_workflows.light_change.qwen_change_light import qwen_change_light
from app_comfyui_workflows.person_clear.flux2_clear_person import flux2_clear_person
from app_comfyui_workflows.person_clear.qwen_clear_person import qwen_clear_person
from app_comfyui_workflows.sight_change.qwen_change_sight import qwen_change_sight
from app_comfyui_workflows.product.product_image import product_image

_script_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__, template_folder=os.path.join(_script_dir, "templates"))

# 单例：只加载一次，所有请求复用
_change_face_processor = None


def get_change_processor(function):
    """获取换脸处理器单例，首次调用时加载模型。"""
    global _change_face_processor
    if _change_face_processor is None or not isinstance(_change_face_processor, function):
        _change_face_processor = function()
    return _change_face_processor


def base64_to_pil(b64_str):
    if b64_str is None or b64_str == "":
        return None
    raw = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def pil_to_base64(pil_img, fmt="PNG"):
    if pil_img is None:
        return None
    buf = io.BytesIO()
    pil_img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")



@app.route("/change_face", methods=["POST"])
def change_face():
    data_json = request.get_json() or {}
    init_image = data_json.get("image_1")
    userdefined_image = data_json.get("image_2")

    init_pil = base64_to_pil(init_image) if init_image else None
    user_pil = base64_to_pil(userdefined_image) if userdefined_image else None

    if init_pil is None or user_pil is None:
        return jsonify({"error": "换脸功能需要 init_image 和 userdefined_image"}), 400

    if _change_face_processor is None:
        print("预加载模型...")
    processor = get_change_processor(ChangeFace)
    res_pil = processor.forward(init_pil, user_pil)
    res_b64 = pil_to_base64(res_pil)
    return jsonify({"res_image": res_b64})


@app.route("/flux2_klein_change_bg", methods=["POST"])
def flux2_klein_change_bg_route():
    data_json = request.get_json() or {}
    init_image = data_json.get("image_1")
    userdefined_image = data_json.get("image_2")

    init_pil = base64_to_pil(init_image) if init_image else None
    user_pil = base64_to_pil(userdefined_image) if userdefined_image else None

    if init_pil is None or user_pil is None:
        return jsonify({"error": "flux2-klein_change_bg功能需要 init_image 和 userdefined_image"}), 400

    if _change_face_processor is None:
        print("预加载模型...")
    processor = get_change_processor(flux2_klein_change_bg)
    res_pil = processor.forward(init_pil, user_pil)
    res_b64 = pil_to_base64(res_pil)
    return jsonify({"res_image": res_b64})


@app.route("/qwen_change_bg", methods=["POST"])
def qwen_change_bg_route():
    data_json = request.get_json() or {}
    init_image = data_json.get("image_1")
    userdefined_image = data_json.get("image_2")

    init_pil = base64_to_pil(init_image) if init_image else None
    user_pil = base64_to_pil(userdefined_image) if userdefined_image else None

    if init_pil is None or user_pil is None:
        return jsonify({"error": "qwen_change_bg功能需要 init_image 和 userdefined_image"}), 400

    if _change_face_processor is None:
        print("预加载模型...")
    processor = get_change_processor(qwen_change_bg)
    res_pil = processor.forward(init_pil, user_pil)
    res_b64 = pil_to_base64(res_pil)
    return jsonify({"res_image": res_b64})

@app.route("/flux2_klein_faceswap",methods=["POST"])
def flux2_klein_faceswap_route():
    data_json=request.get_json() or {}
    init_image=data_json.get("image_1")
    userdefined_image=data_json.get("image_2")

    init_pil=base64_to_pil(init_image) if init_image else None
    user_pil=base64_to_pil(userdefined_image) if userdefined_image else None

    if init_pil is None or user_pil is None:
        return jsonify({"error": "flux2_klein_faceswap功能需要 init_image 和 userdefined_image"}), 400

    if _change_face_processor is None:
        print("预加载模型...")
    processor = get_change_processor(flux2_klein_faceswap)
    res_pil = processor.forward(init_pil, user_pil)
    res_b64 = pil_to_base64(res_pil)
    return jsonify({"res_image": res_b64})

@app.route("/qwen_2511_faceswap", methods=["POST"])
def qwen_2511_faceswap_route():
    data_json=request.get_json() or {}
    init_image=data_json.get("image_1")
    userdefined_image=data_json.get("image_2")

    init_pil=base64_to_pil(init_image) if init_image else None
    user_pil=base64_to_pil(userdefined_image) if userdefined_image else None

    if init_pil is None or user_pil is None:
        return jsonify({"error": "qwen_2511_faceswap功能需要 init_image 和 userdefined_image"}), 400

    if _change_face_processor is None:
        print("预加载模型...")
    processor = get_change_processor(qwen_2511_faceswap)
    res_pil = processor.forward(init_pil, user_pil)
    res_b64 = pil_to_base64(res_pil)
    return jsonify({"res_image": res_b64})



@app.route("/flux2_klein_one_cb",methods=["POST"])
def flux2_klein_one_cb_route():
    data_json = request.get_json() or {}
    init_image = data_json.get("image_1")
    userdefined_image = data_json.get("image_2")

    init_pil = base64_to_pil(init_image) if init_image else None
    user_pil = base64_to_pil(userdefined_image) if userdefined_image else None

    if init_pil is None or user_pil is None:
        return jsonify({"error": "flux2_klein_one_cb功能需要 init_image 和 userdefined_image"}), 400

    if _change_face_processor is None:
        print("预加载模型...")
    processor = get_change_processor(flux2_klein_one_cb)
    res_pil = processor.forward(init_pil, user_pil)
    res_b64 = pil_to_base64(res_pil)
    return jsonify({"res_image": res_b64})

@app.route("/qwen_2509_one_cb", methods=["POST"])
def qwen_2509_one_cb_route():
    data_json = request.get_json() or {}
    init_image = data_json.get("image_1")
    userdefined_image = data_json.get("image_2")

    init_pil = base64_to_pil(init_image) if init_image else None
    user_pil = base64_to_pil(userdefined_image) if userdefined_image else None

    if init_pil is None or user_pil is None:
        return jsonify({"error": "qwen_2509_one_cb功能需要 init_image 和 userdefined_image"}), 400

    if _change_face_processor is None:
        print("预加载模型...")
    processor = get_change_processor(qwen_2509_one_cb)
    res_pil = processor.forward(init_pil, user_pil)
    res_b64 = pil_to_base64(res_pil)
    return jsonify({"res_image": res_b64})

@app.route("/qwen_edit_2509_pose_cb", methods=["POST"])
def qwen_edit_2509_pose_cb_route():
    data_json = request.get_json() or {}
    init_image = data_json.get("image_1")
    userdefined_image = data_json.get("image_2")

    init_pil = base64_to_pil(init_image) if init_image else None
    user_pil = base64_to_pil(userdefined_image) if userdefined_image else None


    if init_pil is None or user_pil is None:
        return jsonify({"error": "qwen_edit_2509_pose_cb功能需要 init_image 和 userdefined_image"}), 400

    if _change_face_processor is None:
        print("预加载模型...")
    processor = get_change_processor(qwen_edit_2509_pose_cb)
    res_pil = processor.forward(init_pil, user_pil)
    res_b64 = pil_to_base64(res_pil)
    return jsonify({"res_image": res_b64})

@app.route("/Zimage_text2img", methods=["POST"])
def Zimage_text2img_route():
    data_json=request.get_json() or {}
    prompt=data_json.get("prompt")
    width=data_json.get("width",1080)
    height=data_json.get("height",1920)
    model=data_json.get("model","z_image_turbo_bf16.safetensors")

    if prompt is None:
        return jsonify({"error": "Zimage_text2img功能需要 prompt"}),400
    if _change_face_processor is None:
        print("预加载模型...")
    processor=get_change_processor(Zimage_text2img)
    # 暂时只传prompt,width,height，如果需要可以修改processor.forward接受model
    res_pil=processor.forward(prompt,width,height)
    res_b64=pil_to_base64(res_pil)
    return jsonify({"res_image":res_b64})

@app.route("/everything_image",methods=["POST"])
def everything_image_route():
    data_json=request.get_json() or {}
    image_1=data_json.get("image_1",None)
    image_2=data_json.get("image_2",None)
    image_3=data_json.get("image_3",None)
    prompt=data_json.get("prompt",None)
    api_key=data_json.get("api_key",None)
    model_select=data_json.get("model_select","gemini-3-pro-image-preview")
    size_mode=data_json.get("size_mode","Match Image_1 (Smart Crop)")
    custom_w=data_json.get("custom_w",784)
    custom_h=data_json.get("custom_h",1344)

    image_1_pil=base64_to_pil(image_1) if image_1 else None
    image_2_pil=base64_to_pil(image_2) if image_2 else None
    image_3_pil=base64_to_pil(image_3) if image_3 else None

    if prompt is None:
        return jsonify({"error":"everything_image功能需要 prompt"}),400
    if api_key is None:
        return jsonify({"error":"everything_image功能需要 api_key"}),400
    if image_1_pil is None and (image_2_pil is not None or image_3_pil is not None):
        return jsonify({"error":"everything_image功能 请上传图1"})
    processor=get_change_processor(everything_image)
    res_pil=processor.forward(image_1=image_1_pil,image_2=image_2_pil,image_3=image_3_pil,prompt=prompt,api_key=api_key,model_select=model_select,size_mode=size_mode,custom_w=custom_w,custom_h=custom_h)
    res_b64=pil_to_base64(res_pil)
    return jsonify({"res_image":res_b64})

@app.route("/flux2_change_light",methods=["POST"])
def flux2_change_light_route():
    data_json=request.get_json() or {}
    image=data_json.get("image_1",None)
    prompt=data_json.get("prompt",None)

    image_pil=base64_to_pil(image) if image else None

    if image is None:
        return jsonify({"error":"flux2_change_light功能需要 image"})

    if prompt is None:
        return jsonify({"error":"flux2_change_light功能需要 prompt"})

    processor=get_change_processor(flux2_change_light)
    res_pil=processor.forward(image=image_pil,prompt=prompt)
    res_b64=pil_to_base64(res_pil)

    return jsonify({"res_image":res_b64})

@app.route("/qwen_change_light",methods=["POST"])
def qwen_change_light_route():
    data_json=request.get_json() or {}
    image=data_json.get("image_1",None)
    prompt=data_json.get("prompt",None)

    image_pil=base64_to_pil(image) if image else None

    if image is None:
        return jsonify({"error":"qwen_change_light功能需要 image"})

    if prompt is None:
        return jsonify({"error":"qwen_change_light功能需要 prompt"})

    processor=get_change_processor(qwen_change_light)
    res_pil=processor.forward(image=image_pil,prompt=prompt)
    res_b64=pil_to_base64(res_pil)

    return jsonify({"res_image":res_b64})

@app.route("/flux2_clear_person",methods=["POST"])
def flux2_clear_person_route():
    data_json=request.get_json() or {}
    image=data_json.get("image_1",None)
    mask_image=data_json.get("image_2",None)

    image_pil=base64_to_pil(image) if image else None
    mask_image_pil=base64_to_pil(mask_image) if mask_image else None

    if image is None:
        return jsonify({"error":"flux2_clear_person功能需要 image"})
    if mask_image is None:
        return jsonify({"error":"flux2_clear_person功能需要 mask_image"})
    
    processor=get_change_processor(flux2_clear_person)
    res_pil=processor.forward(image=image_pil,mask_image=mask_image_pil)
    res_b64=pil_to_base64(res_pil)

    return jsonify({"res_image":res_b64})

@app.route("/qwen_clear_person",methods=["POST"])
def qwen_clear_person_route():
    data_json=request.get_json() or {}
    image=data_json.get("image_1",None)

    image_pil=base64_to_pil(image) if image else None

    if image_pil is None:
        return jsonify({"error":"qwen_clear_person功能需要 image"})
    processor=get_change_processor(qwen_clear_person)
    res_pil=processor.forward(image=image_pil)
    res_b64=pil_to_base64(res_pil)

    return jsonify({"res_image":res_b64})

@app.route("/qwen_change_sight",methods=["POST"])
def qwen_change_sight_route():
    data_json=request.get_json() or {}
    image=data_json.get("image_1",None)
    prompt=data_json.get("prompt",None)

    image_pil=base64_to_pil(image) if image else None

    if image is None:
        return jsonify({"error":"qwen_change_sight功能需要 image"})
    if prompt is None:
        return jsonify({"error":"qwen_change_sight功能需要 prompt"})

    processor=get_change_processor(qwen_change_sight)
    res_pil=processor.forward(image=image_pil,prompt=prompt)
    res_b64=pil_to_base64(res_pil)

    return jsonify({"res_image":res_b64})

@app.route("/product_image",methods=["POST"])
def product_image_route():
    data_json=request.get_json() or {}
    image=data_json.get("image",None)
    prompt=data_json.get("prompt",None)
    product_type=data_json.get("product_type","耳机")
    design_style=data_json.get("design_style","科技感")
    scene_preference=data_json.get("scene_preference","混合（以使用场景为主）")
    output_language=data_json.get("output_language","中文 (Chinese)")
    api_key=data_json.get("api_key",None)
    model_select=data_json.get("model_select","gemini-3-pro-image-preview")
    size_mode=data_json.get("size_mode","Match Image_1 (Smart Crop)")
    custom_w=data_json.get("custom_w",768)
    custom_h=data_json.get("custom_h",1344)

    image_pil=base64_to_pil(image) if image else None
    if image_pil is None or prompt is None or api_key is None:
        return jsonify({"error":"product_image功能需要 参数不足"})
    processor=get_change_processor(product_image)
    res_pil=processor.forward(image=image_pil,prompt=prompt,product_type=product_type,design_style=design_style,scene_preference=scene_preference,output_language=output_language,api_key=api_key,model_select=model_select,size_mode=size_mode,custom_w=custom_w,custom_h=custom_h)
    res_b64=pil_to_base64(res_pil)
    return jsonify({"res_image":res_b64})

if __name__ == "__main__":
    print("模型已就绪，启动服务...")
    app.run(host="0.0.0.0", port=9001)
