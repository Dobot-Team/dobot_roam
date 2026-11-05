# 阿里云 OSS 文件上传工具

一个基于 Python 的阿里云 OSS 文件/目录上传工具，支持断点续传、分片上传、批量上传和失败自动重试。

## 功能特性

- **单文件上传**：小文件直接上传，大文件自动分片
- **批量 / 递归目录上传**：一键上传整个目录
- **断点续传**：大文件上传中断后可继续，无需从头开始
- **自动重试**：网络异常时自动重试，最多 3 次
- **上传进度**：实时显示上传进度
- **虚拟环境隔离**：一键脚本自动创建 `.venv`，避免污染系统环境

## 环境要求

- Python >= 3.8
- pip
- 阿里云 OSS AccessKey（建议使用 RAM 子账号，最小权限原则）
- SQLite（Python 内置，上传记录功能无需额外安装）

## 快速开始

本工具已作为 `embodvla/tools/oss_uploader` 整合进项目，复用项目主虚拟环境。

### 安装依赖

已在项目根目录 `requirements.txt` 中加入 `oss2` 与 `python-dotenv`。若使用独立虚拟环境：

```bash
cd /root/embodvla
pip install -r requirements.txt
```

### 配置凭证

OSS 凭证等敏感配置**不要放在仓库内**。把模板复制到仓库外部的配置文件：

```bash
mkdir -p ~/.config/embodvla
cp tools/oss_uploader/env.example ~/.config/embodvla/oss.env
# 编辑 ~/.config/embodvla/oss.env 填入真实 OSS 凭证
```

代码默认读取 `~/.config/embodvla/oss.env`；也可以通过环境变量覆盖：

```bash
# 指定其他配置文件路径
export EMBODVLA_OSS_CONFIG=/path/to/your/oss.env

# 或者直接使用环境变量（CI 场景常用）
export OSS_ACCESS_KEY_ID=xxx
export OSS_ACCESS_KEY_SECRET=xxx
export OSS_BUCKET_NAME=xxx
export OSS_ENDPOINT=xxx
```

> **不要把真实的 OSS 配置文件提交到 Git**。

### 命令行上传

```bash
cd /root/embodvla
python -m tools.oss_uploader.uploader /path/to/file.txt test_upload/
```

## 使用示例

### 上传单个文件

```bash
python -m tools.oss_uploader.uploader /path/to/file.txt
```

上传到指定 OSS 路径：

```bash
python -m tools.oss_uploader.uploader /path/to/file.txt data/2024/file.txt
```

上传到某个目录下：

```bash
python -m tools.oss_uploader.uploader /path/to/file.txt test_upload/
```

### 上传整个目录

```bash
python -m tools.oss_uploader.uploader /path/to/local_dir data/2024/
```

目录下的文件会保持相对结构上传到 `data/2024/` 下。

## 配置说明

复制 `env.example` 到仓库外部（例如 `~/.config/embodvla/oss.env`）后，按需填写：

```ini
# 阿里云 OSS 凭证（必填）
OSS_ACCESS_KEY_ID=your-access-key-id
OSS_ACCESS_KEY_SECRET=your-access-key-secret
OSS_BUCKET_NAME=your-bucket-name
OSS_ENDPOINT=your-oss-endpoint

# 断点续传配置（可选）
OSS_CHECKPOINT_DIR=.oss_checkpoints     # checkpoint 保存目录
OSS_PART_SIZE=10485760                  # 分片大小，默认 10MB
OSS_MULTIPART_THRESHOLD=104857600       # 启用分片阈值，默认 100MB
OSS_NUM_THREADS=4                       # 并发上传线程数
OSS_MAX_RETRIES=3                       # 失败重试次数
```

加载优先级：环境变量 > `~/.config/embodvla/oss.env`（或 `EMBODVLA_OSS_CONFIG` 指定路径）。

> **注意**：`OSS_ENDPOINT` 必须与你 Bucket 的实际地域一致，例如深圳节点为 `oss-cn-shenzhen.aliyuncs.com`。不一致会报 `AccessDenied` 错误。

## 上传记录

每次上传都会自动记录到本地 SQLite 数据库（默认 `upload_history.db`），**SQLite 是 Python 内置模块，无需额外安装**。

包含以下字段：

- `local_path`：本地文件路径
- `object_key`：OSS 上的对象路径
- `oss_url`：OSS 访问 URL
- `file_size`：文件大小
- `status`：上传状态 `success` / `failed`
- `error_msg`：失败时的错误信息
- `uploaded_at`：上传时间

### 查看上传记录

```bash
# 查看最近 50 条记录
python -m tools.oss_uploader.list_uploads

# 只查看成功的
python -m tools.oss_uploader.list_uploads success

# 只查看失败的，最近 10 条
python -m tools.oss_uploader.list_uploads failed 10
```

你也可以直接运行 `python -m tools.oss_uploader.upload_record` 查看记录。

## 断点续传说明

- 文件大小 ≥ `OSS_MULTIPART_THRESHOLD`（默认 100MB）时，自动走 **分片 + 断点续传**。
- 断点信息保存在 `OSS_CHECKPOINT_DIR`（默认 `.oss_checkpoints/`）目录。
- 上传过程中断后，再次执行相同命令会从中断的分片继续上传，无需从头开始。
- 上传成功后，对应的 checkpoint 文件会自动清理。
- 网络波动导致失败时，会自动重试，最多 `OSS_MAX_RETRIES` 次（默认 3 次）。

### 作为 Python 模块调用

```python
from tools.oss_uploader import upload_file, upload_directory

# 上传单个文件
url = upload_file("/path/to/file.txt", "data/2024/file.txt")

# 上传整个目录
urls = upload_directory("/path/to/local_dir", "data/2024/")
```

## 目录结构

```text
tools/oss_uploader/
├── __init__.py           # 包入口，暴露 upload_file / upload_directory 等
├── .oss_checkpoints/     # 断点续传临时文件
├── config.py             # 配置读取与校验
├── env.example           # OSS 配置文件模板（不含真实密钥）
├── list_uploads.py       # 上传记录查询工具
├── oss_client.py         # OSS 客户端封装
├── requirements.txt      # 独立安装时的 Python 依赖
├── upload_record.py      # 上传记录数据库模块
├── uploader.py           # 上传入口（单文件 / 目录）
└── README.md             # 本文档

本地配置文件（仓库外部）：
~/.config/embodvla/oss.env   # 真实 OSS 凭证，不上传 Git
```

## 安全提示

- **不要把真实的 OSS 配置文件（如 `~/.config/embodvla/oss.env`）提交到 Git**，里面包含 AccessKey Secret。
- 本仓库只保留 `env.example` 模板，真实密钥请放在仓库外部。
- 建议使用阿里云 **RAM 子账号**，并只授予目标 Bucket 的最小权限。
- 生产环境优先考虑使用 **STS 临时凭证** 或 **RAM 角色**，避免长期使用固定 AccessKey。

## 常见问题

### 报错 `AccessDenied: must be addressed using the specified endpoint`

说明 `.env` 里的 `OSS_ENDPOINT` 与 Bucket 实际地域不一致。请登录阿里云 OSS 控制台，查看该 Bucket 的 Endpoint 并更新 `.env`。

### 大文件上传中断后如何继续？

直接重新执行相同的上传命令即可，oss2 会自动读取 `.oss_checkpoints/` 下的 checkpoint 文件，从已上传的分片继续。

## 后续可扩展

- 接入 FastAPI/Flask 提供 Web 上传接口
- 使用 `watchdog` 监听目录，文件落地自动上传
- 上传记录写入 SQLite/MySQL
- 对接阿里云 STS，使用临时凭证上传
