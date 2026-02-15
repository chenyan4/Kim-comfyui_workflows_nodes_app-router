import base64
import io
import os
from flask import Flask, request, jsonify, render_template
from PIL import Image

from app_comfyui_workflows.change_face import ChangeFace
from app_comfyui_workflows.flux2_klein_change_bg import flux2_klein_change_bg
from app_comfyui_workflows.qwen_change_bg import qwen_change_bg
from app_comfyui_workflows.flux2_klein_faceswap import flux2_klein_faceswap
from app_comfyui_workflows.qwen_2511_faceswap import qwen_2511_faceswap
from app_comfyui_workflows.flux2_klein_one_cb import flux2_klein_one_cb
from app_comfyui_workflows.qwen_2509_one_cb import qwen_2509_one_cb
from app_comfyui_workflows.qwen_edit_2509_pose_cb import qwen_edit_2509_pose_cb
from app_comfyui_workflows.Zimage_text2img import Zimage_text2img

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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/change_face", methods=["POST"])
def change_face():
    data_json = request.get_json() or {}
    init_image = data_json.get("init_image")
    userdefined_image = data_json.get("userdefined_image")
    mask_image = data_json.get("mask_image",None)

    init_pil = base64_to_pil(init_image) if init_image else None
    user_pil = base64_to_pil(userdefined_image) if userdefined_image else None
    mask_pil = base64_to_pil(mask_image) if mask_image else None

    if init_pil is None or user_pil is None:
        return jsonify({"error": "换脸功能需要 init_image 和 userdefined_image"}), 400

    if _change_face_processor is None:
        print("预加载模型...")
    processor = get_change_processor(ChangeFace)
    res_pil = processor.forward(init_pil, user_pil)
    res_b64 = pil_to_base64(res_pil)
    return jsonify({"res_image": res_b64})


@app.route("/flux2-klein_change_bg", methods=["POST"])
def flux2_klein_change_bg_route():
    data_json = request.get_json() or {}
    init_image = data_json.get("init_image")
    userdefined_image = data_json.get("userdefined_image")
    mask_image = data_json.get("mask_image",None)

    init_pil = base64_to_pil(init_image) if init_image else None
    user_pil = base64_to_pil(userdefined_image) if userdefined_image else None
    mask_pil = base64_to_pil(mask_image) if mask_image else None

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
    init_image = data_json.get("init_image")
    userdefined_image = data_json.get("userdefined_image")
    mask_image = data_json.get("mask_image",None)

    init_pil = base64_to_pil(init_image) if init_image else None
    user_pil = base64_to_pil(userdefined_image) if userdefined_image else None
    mask_pil = base64_to_pil(mask_image) if mask_image else None

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
    init_image=data_json.get("init_image")
    userdefined_image=data_json.get("userdefined_image")
    mask_image=data_json.get("mask_image",None)

    init_pil=base64_to_pil(init_image) if init_image else None
    user_pil=base64_to_pil(userdefined_image) if userdefined_image else None
    mask_pil=base64_to_pil(mask_image) if mask_image else None

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
    init_image=data_json.get("init_image")
    userdefined_image=data_json.get("userdefined_image")
    mask_image=data_json.get("mask_image",None)

    init_pil=base64_to_pil(init_image) if init_image else None
    user_pil=base64_to_pil(userdefined_image) if userdefined_image else None
    mask_pil=base64_to_pil(mask_image) if mask_image else None

    if init_pil is None or user_pil is None:
        return jsonify({"error": "qwen_2511_faceswap功能需要 init_image 和 userdefined_image"}), 400

    if _change_face_processor is None:
        print("预加载模型...")
    processor = get_change_processor(qwen_2511_faceswap)
    res_pil = processor.forward(init_pil, user_pil)
    res_b64 = pil_to_base64(res_pil)
    return jsonify({"res_image": res_b64})


# 版本名 -> 处理器类，供 /compare 使用
_VERSION_PROCESSORS = {
    "instantid": ChangeFace,
    "flux2": flux2_klein_change_bg,
    "qwen": qwen_change_bg,
}


@app.route("/compare", methods=["POST"])
def compare():
    """同一组图片多版本结果对比。body: feature, versions[], init_image, userdefined_image, mask_image"""
    data_json = request.get_json() or {}
    feature = data_json.get("feature")
    versions = data_json.get("versions") or []
    init_image = data_json.get("init_image")
    userdefined_image = data_json.get("userdefined_image")
    mask_image = data_json.get("mask_image",None)

    init_pil = base64_to_pil(init_image) if init_image else None
    user_pil = base64_to_pil(userdefined_image) if userdefined_image else None
    mask_pil = base64_to_pil(mask_image) if mask_image else None

    if init_pil is None or user_pil is None:
        return jsonify({"error": "对比功能需要 init_image 和 userdefined_image"}), 400
    if not versions:
        return jsonify({"error": "请指定至少一个版本 versions"}), 400

    results = []
    for ver in versions:
        if ver not in _VERSION_PROCESSORS:
            results.append({"version": ver, "error": "未知版本"})
            continue
        proc_class = _VERSION_PROCESSORS[ver]
        try:
            processor = get_change_processor(proc_class)
            res_pil = processor.forward(init_pil, user_pil)
            res_b64 = pil_to_base64(res_pil)
            results.append({"version": ver, "res_image": res_b64})
        except Exception as e:
            results.append({"version": ver, "error": str(e)})

    return jsonify({"results": results})

@app.route("/flux2_klein_one_cb",methods=["POST"])
def flux2_klein_one_cb_route():
    data_json = request.get_json() or {}
    init_image = data_json.get("init_image")
    userdefined_image = data_json.get("userdefined_image")
    mask_image = data_json.get("mask_image",None)

    init_pil = base64_to_pil(init_image) if init_image else None
    user_pil = base64_to_pil(userdefined_image) if userdefined_image else None
    mask_pil = base64_to_pil(mask_image) if mask_image else None

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
    init_image = data_json.get("init_image")
    userdefined_image = data_json.get("userdefined_image")
    mask_image = data_json.get("mask_image",None)

    init_pil = base64_to_pil(init_image) if init_image else None
    user_pil = base64_to_pil(userdefined_image) if userdefined_image else None
    mask_pil = base64_to_pil(mask_image) if mask_image else None

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
    init_image = data_json.get("init_image")
    userdefined_image = data_json.get("userdefined_image")
    mask_image = data_json.get("mask_image",None)

    init_pil = base64_to_pil(init_image) if init_image else None
    user_pil = base64_to_pil(userdefined_image) if userdefined_image else None
    mask_pil = base64_to_pil(mask_image) if mask_image else None

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

    if prompt is None:
        return jsonify({"error": "Zimage_text2img功能需要 prompt"}),400
    if _change_face_processor is None:
        print("预加载模型...")
    processor=get_change_processor(Zimage_text2img)
    res_pil=processor.forward(prompt,width,height)
    res_b64=pil_to_base64(res_pil)
    return jsonify({"res_image":res_b64})

if __name__ == "__main__":
    print("模型已就绪，启动服务...")
    app.run(host="0.0.0.0", port=9001)
