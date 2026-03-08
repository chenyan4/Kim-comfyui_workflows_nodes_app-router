# custom_nodes 大文件与可清理项说明

根据扫描结果，以下为**体积较大且多数可删**的内容，按“是否影响运行”分类。

---

## 一、可直接删除（不影响节点运行）

### 1. `.git.embedded.backup` 目录（约 500MB+）

各节点下的 Git 历史备份，仅用于 ComfyUI-Manager 等恢复，**运行节点不需要**。

| 节点目录 | 约占用 |
|----------|--------|
| comfyui-manager | 103M |
| ComfyUI-Frame-Interpolation | 80M |
| ComfyUI-TeaCache | 45M |
| comfy_mtb | 41M |
| ComfyUI-uinodesDOC | 36M |
| Comfyui-QwenEditUtils | 35M |
| ComfyUI-SAM3 | 34M |
| ComfyUI-KJNodes | 26M |
| 其他 37 个节点 | 各 1M~22M |

**一键删除所有 .git.embedded.backup：**
```bash
find /data/chenyan/cheny_comfyui/custom_nodes -name ".git.embedded.backup" -type d -exec rm -rf {} + 2>/dev/null
```

---

### 2. 演示/示例资源（按需删）

多为文档、示例图/视频，删除后**不影响节点功能**，仅少掉自带的 demo 素材。

| 路径 | 大小 | 说明 |
|------|------|------|
| ComfyUI-SAM3/assets/image.png | ~24MB | 示例图 |
| comfy_mtb/extern/frame_interpolation/moment.gif | ~21MB | 演示动图 |
| Comfyui-QwenEditUtils/result.png | ~16MB | 示例结果图 |
| ComfyUI-Copilot/assets/*.gif | ~14MB+ | 文档动图 |
| ComfyUI-uinodesDOC/assets/*.gif | ~14MB | 文档动图 |
| comfyui-loop/_demo_videos/*.mkv | ~14MB+6MB | 演示视频 |
| ComfyUI-utils-nodes/r_nudenet/320n.onnx | ~12MB | 若不用 NudeNet 可删 |
| Comfyui_TTP_Toolset/examples/*.png | ~12MB+ | 示例图 |
| comfyui-mixlab-nodes/assets/fonts/*.ttf | ~11MB+9MB+9MB+8MB | 字体，不用该节点排版可删 |
| comfyui-mixlab-nodes/assets/*.svg, *.glb | 若干 MB | 示例/文档资源 |
| LanPaint/examples/*.gif | 多个 4~9MB | 示例动图 |
| ComfyUI-Addoor/example_workflow/*.png | ~3.5MB | 示例工作流图 |
| comfyui_controlnet_aux_backup/examples/*.png | 若干 MB | 示例图 |
| ComfyUI-Frame-Interpolation/test_vfi_schedule.gif | ~8MB | 测试用动图 |
| was-ns/repos/SAM/notebooks/*.ipynb | ~8MB+4MB | Jupyter 示例 |
| Comfyui-QwenEditUtils/Demo.mp4 | ~8MB | 演示视频 |

若希望**只删“明显是示例/文档”的大文件**，可手动删上述路径中的 gif/png/mp4/mkv 等，或只删其中体积最大的几项。

---

## 二、模型/权重（按功能需求决定）

| 路径 | 大小 | 说明 |
|------|------|------|
| comfyui-lama-remover/ckpts/big-lama.pt | **~196MB** | LaMa 去水印/擦除模型，**不用该节点可删** |
| ComfyUI-KJNodes/intrinsic_loras/*.safetensors | 4×6.4MB | 内在光照 LoRA，不用可删 |

---

## 三、建议操作顺序

1. **先删所有 `.git.embedded.backup`**（省约 500MB+，且不影响运行）：
   ```bash
   find /data/chenyan/cheny_comfyui/custom_nodes -name ".git.embedded.backup" -type d -exec rm -rf {} + 2>/dev/null
   ```

2. 若不使用 **comfyui-lama-remover**，可删：
   ```bash
   rm -rf /data/chenyan/cheny_comfyui/custom_nodes/comfyui-lama-remover/ckpts/big-lama.pt
   ```
   约省 196MB。

3. 若不需要各节点的**示例图/演示视频/文档动图**，再按上表路径删除对应 `examples/`、`assets/`、`_demo_videos/` 下的大文件（可先备份或只删体积最大的几项）。

---

## 四、不建议删的

- 各节点下的 **tokenizer.json / spiece.model**（如 ComfyUI-WanVideoWrapper、Comfyui-SecNodes）：推理需要。
- **comfy-kitchen、comfyui-workflow-templates** 等核心依赖的代码与配置。
- 你实际用到的节点里的 **.onnx、.safetensors、.pt** 模型文件。

以上为当前目录下的检查结论，按需执行即可。
