# ComfyUI 算法服务（app_router）云服务器部署与环境配置

本文档说明在云服务器上部署本 ComfyUI 项目（Flask 算法路由 `app_router.py`）所需的环境配置，以及如何将当前 Conda 环境 **Cheny** 打包并在新机器上复用。

---

## 一、环境要求

| 项目 | 要求 |
|------|------|
| 操作系统 | Linux（推荐 Ubuntu 20.04+ / CentOS 7+） |
| Python | 3.10+（与现有 Cheny 环境一致） |
| Conda | Miniconda 或 Anaconda |
| GPU（可选） | 若跑本地模型需 CUDA；仅云 API 可仅 CPU |
| 磁盘 | 模型与输出目录需足够空间 |

---

## 二、打包当前 Conda 环境（Cheny）

在**当前已配置好的开发机**上，用下面任一方式把 **Cheny** 环境“打包”，便于在云服务器上复现或直接迁移。

### 方式一：导出环境 YAML（推荐，跨平台复现）

在开发机上执行：

```bash
conda activate Cheny
conda env export -n Cheny -f environment.yml
```

会生成 `environment.yml`，包含环境名、Python 版本和所有 conda 安装的包。可把该文件放入项目并提交到 Git，便于他人或新机器复现环境。

**在云服务器上复现环境：**

```bash
cd /path/to/comfyui
conda env create -f environment.yml
conda activate Cheny
# 若项目用 pip 依赖，再执行
pip install -r requirements.txt
```

若导出时希望**不包含**机器相关路径、仅保留显式声明的包（便于跨平台），可使用：

```bash
conda env export -n Cheny --from-history -f environment.yml
```

此时 `environment.yml` 只包含你曾手动 `conda install` 的包，其他依赖需在复现后补装或靠 `requirements.txt`。

### 方式二：conda-pack 打包整个环境（同架构迁移）

适合**同系统、同架构**（如都是 Linux x86_64）的机器间直接拷贝环境，无需联网再装依赖。

1. 安装 conda-pack（在开发机任意环境）：

```bash
conda install -n base -c conda-forge conda-pack
```

2. 打包 Cheny 环境为 tar.gz：

```bash
conda pack -n Cheny -o Cheny.tar.gz
```

3. 将 `Cheny.tar.gz` 和项目代码一起上传到云服务器（如 scp、对象存储等）。

4. 在云服务器上解压并注册环境：

```bash
mkdir -p /opt/conda_envs/Cheny
tar -xzf Cheny.tar.gz -C /opt/conda_envs/Cheny
# 修复脚本中的硬编码路径（conda-pack 会写入原机器路径，需替换为新路径）
/opt/conda_envs/Cheny/bin/conda-unpack
```

`conda-unpack` 会改写脚本里的旧路径为当前解压路径。之后用该环境：

```bash
/opt/conda_envs/Cheny/bin/python app_router.py
# 或把该路径加入 PATH 后直接 python app_router.py
```

若希望在新机器上用 `conda activate Cheny`，可将解压目录作为 env 加入 conda：

```bash
# 仅当前 shell 临时使用
export PATH="/opt/conda_envs/Cheny/bin:$PATH"
# 或创建 conda 元数据让 conda activate 能识别（需自行建 envs 目录结构或软链）
```

### 方式三：仅导出 pip 依赖（轻量复现）

若云服务器上打算用全新 conda 环境，只复现 pip 包：

```bash
conda activate Cheny
pip freeze > requirements-frozen.txt
```

在云服务器新环境中：

```bash
conda create -n Cheny python=3.10 -y
conda activate Cheny
pip install -r requirements-frozen.txt
```

项目根目录的 `requirements.txt` 是“声明式”依赖，`requirements-frozen.txt` 是当前环境精确版本，二者可并存；部署时二选一或先 `requirements.txt` 再按需用 frozen 锁版。

---

## 三、环境变量配置

项目通过环境变量配置监听地址、端口和视频上传目录。

1. 复制示例并编辑：

```bash
cd /path/to/comfyui
cp .env.example .env
vim .env
```

2. 常用变量说明：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `COMFYUI_APP_HOST` | 监听地址 | 0.0.0.0 |
| `COMFYUI_APP_PORT` | 监听端口 | 9001 |
| `INPUT_VIDEO_DIR` | 视频上传/输入目录 | 项目下 `input_video` |

3. 启动前加载 `.env` 再启动服务：

```bash
set -a && source .env && set +a
python app_router.py
```

若与 new-Ai-Studio 同机部署，可将 `INPUT_VIDEO_DIR` 设为 new-Ai-Studio 的 `VIDEO_UPLOAD_DIR`，避免重复存视频。

---

## 四、安装与运行（从零在云服务器上）

若不使用打包环境，在云服务器上从零配置：

```bash
cd /path/to/comfyui

# 1. 创建并激活环境（若用 environment.yml 见第二节）
conda create -n Cheny python=3.10 -y
conda activate Cheny

# 2. 安装依赖
pip install -r requirements.txt

# 3. 环境变量与启动
cp .env.example .env
# 编辑 .env 后
set -a && source .env && set +a
python app_router.py
```

服务默认监听 `http://0.0.0.0:9001`，供 new-Ai-Studio 通过 `ALGO_SIDE_BASE_URL` 调用。

---

## 五、生产运行建议

### 5.1 使用 systemd 守护

创建 `/etc/systemd/system/comfyui-app.service`（路径与用户按需修改）：

```ini
[Unit]
Description=ComfyUI Flask App (app_router)
After=network.target

[Service]
Type=simple
User=www-data
WorkingDirectory=/path/to/comfyui
EnvironmentFile=/path/to/comfyui/.env
ExecStart=/path/to/conda_envs/Cheny/bin/python app_router.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

然后：

```bash
sudo systemctl daemon-reload
sudo systemctl enable comfyui-app
sudo systemctl start comfyui-app
sudo systemctl status comfyui-app
```

`ExecStart` 中的 Python 请改为实际环境中的解释器路径（如 `conda run -n Cheny python app_router.py` 或解压后的 `/opt/conda_envs/Cheny/bin/python`）。

### 5.2 使用 Nginx 反代

若对外提供 80/443，可用 Nginx 反代到本机 9001：

```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://127.0.0.1:9001;
        proxy_http_version 1.1;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 600s;
        proxy_connect_timeout 75s;
    }
}
```

---

## 六、与 new-Ai-Studio 的配合

- new-Ai-Studio 通过 **ALGO_SIDE_BASE_URL** 调用本服务（如 `http://算法机IP:9001`）。
- 同机部署时，将 new-Ai-Studio 的 **VIDEO_UPLOAD_DIR** 与本项目的 **INPUT_VIDEO_DIR** 设为同一目录，或让本服务能读取 VIDEO_UPLOAD_DIR，可避免重复上传与路径不一致问题。

---

## 七、速查：在开发机打包 Cheny 并在云服务器使用

| 目的 | 在开发机执行 | 在云服务器执行 |
|------|--------------|----------------|
| 用 YAML 复现环境 | `conda env export -n Cheny -f environment.yml` | 拷贝 `environment.yml` → `conda env create -f environment.yml` → `pip install -r requirements.txt` |
| 用 tar 包迁移环境 | `conda pack -n Cheny -o Cheny.tar.gz` | 拷贝 `Cheny.tar.gz` → 解压 → 运行 `conda-unpack` → 用解压目录下 `bin/python` 启动 |
| 仅锁 pip 版本 | `pip freeze > requirements-frozen.txt` | 新环境里 `pip install -r requirements-frozen.txt` |

按上述步骤即可在云服务器上完成环境配置，并复用当前 **Cheny** 环境。
