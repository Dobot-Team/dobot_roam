import os
from pathlib import Path
from dotenv import load_dotenv

# OSS 凭证与配置统一放到仓库外部的配置文件，避免误提交到 Git。
# 默认路径：~/.config/embodvla/oss.env
# 可通过环境变量 EMBODVLA_OSS_CONFIG 覆盖。
_DEFAULT_CONFIG_DIR = Path.home() / ".config" / "embodvla"
DEFAULT_CONFIG_PATH = _DEFAULT_CONFIG_DIR / "oss.env"
CONFIG_PATH = Path(os.getenv("EMBODVLA_OSS_CONFIG", DEFAULT_CONFIG_PATH))

# 如果外部配置文件存在，则加载；环境变量优先级高于配置文件。
if CONFIG_PATH.exists():
    load_dotenv(CONFIG_PATH, override=False)

# 阿里云 OSS 凭证（必填）
OSS_ACCESS_KEY_ID = os.getenv("OSS_ACCESS_KEY_ID")
OSS_ACCESS_KEY_SECRET = os.getenv("OSS_ACCESS_KEY_SECRET")
OSS_BUCKET_NAME = os.getenv("OSS_BUCKET_NAME")
OSS_ENDPOINT = os.getenv("OSS_ENDPOINT")

# 断点续传与并发配置
OSS_CHECKPOINT_DIR = os.getenv("OSS_CHECKPOINT_DIR", ".oss_checkpoints")
OSS_PART_SIZE = int(os.getenv("OSS_PART_SIZE", str(10 * 1024 * 1024)))          # 默认 10MB
OSS_MULTIPART_THRESHOLD = int(os.getenv("OSS_MULTIPART_THRESHOLD", str(100 * 1024 * 1024)))  # 默认 100MB
OSS_NUM_THREADS = int(os.getenv("OSS_NUM_THREADS", "4"))
OSS_MAX_RETRIES = int(os.getenv("OSS_MAX_RETRIES", "3"))

# 上传记录数据库
UPLOAD_RECORD_DB = os.getenv("UPLOAD_RECORD_DB", "upload_history.db")


def validate():
    missing = []
    for key in ["OSS_ACCESS_KEY_ID", "OSS_ACCESS_KEY_SECRET", "OSS_BUCKET_NAME", "OSS_ENDPOINT"]:
        if not os.getenv(key):
            missing.append(key)
    if missing:
        raise ValueError(
            f"缺少必要的环境变量: {', '.join(missing)}\n"
            f"请在配置文件填写 OSS 凭证: {CONFIG_PATH}\n"
            f"模板文件: tools/oss_uploader/env.example\n"
            f"示例:\n"
            f"  mkdir -p {_DEFAULT_CONFIG_DIR}\n"
            f"  cp tools/oss_uploader/env.example {CONFIG_PATH}\n"
            f"也可以直接通过环境变量设置，或指定 EMBODVLA_OSS_CONFIG 路径。"
        )
