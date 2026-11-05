import os
import sys
import time
from pathlib import Path
from functools import wraps
import oss2
from .oss_client import get_bucket
from .upload_record import record_upload
from .config import (
    validate,
    OSS_CHECKPOINT_DIR,
    OSS_PART_SIZE,
    OSS_MULTIPART_THRESHOLD,
    OSS_NUM_THREADS,
    OSS_MAX_RETRIES,
)


def retry(max_attempts=OSS_MAX_RETRIES, exceptions=(Exception,), delay=1):
    """简单的应用层重试装饰器"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    if attempt == max_attempts:
                        raise
                    print(f"  上传失败（第 {attempt}/{max_attempts} 次）: {e}，{delay}s 后重试...")
                    time.sleep(delay)
        return wrapper
    return decorator


def _make_progress_callback(label: str):
    """构造带标签的进度回调"""
    def callback(consumed_bytes, total_bytes):
        if total_bytes:
            percent = int(100 * consumed_bytes / total_bytes)
            print(f"\r{label}上传进度: {percent}% ({consumed_bytes}/{total_bytes})", end="")
    return callback


def _ensure_checkpoint_dir():
    """确保断点续传 checkpoint 目录存在"""
    cp_dir = Path(OSS_CHECKPOINT_DIR)
    cp_dir.mkdir(parents=True, exist_ok=True)
    return str(cp_dir)


def _clear_checkpoint(local_path: str, object_key: str):
    """上传成功后清理对应 checkpoint 文件"""
    cp_dir = Path(OSS_CHECKPOINT_DIR)
    # oss2 默认 checkpoint 文件名格式：{local_path}.{object_key}.cp
    safe_local = str(local_path).replace(os.sep, "_").replace("/", "_")
    checkpoint_file = cp_dir / f"{safe_local}.{object_key}.cp"
    if checkpoint_file.exists():
        try:
            checkpoint_file.unlink()
        except OSError:
            pass


@retry(max_attempts=OSS_MAX_RETRIES, exceptions=(oss2.exceptions.OssError, ConnectionError, TimeoutError))
def upload_file(local_path: str, object_key: str = None, progress: bool = True) -> str:
    """
    上传单个文件到 OSS，支持断点续传
    :param local_path: 本地文件路径
    :param object_key: OSS 中的对象名，为空则使用文件名
    :param progress: 是否打印进度
    :return: 上传后的 OSS URL
    """
    local_path = Path(local_path)
    if not local_path.exists():
        raise FileNotFoundError(f"文件不存在: {local_path}")

    if object_key is None:
        object_key = local_path.name

    bucket = get_bucket()
    file_size = local_path.stat().st_size
    checkpoint_dir = _ensure_checkpoint_dir()

    progress_callback = _make_progress_callback(f"[{local_path.name}] ") if progress else None

    print(f"开始上传: {local_path} -> oss://{bucket.bucket_name}/{object_key}")
    print(f"  文件大小: {file_size / 1024 / 1024:.2f} MB，分片阈值: {OSS_MULTIPART_THRESHOLD / 1024 / 1024:.0f} MB")

    # 超过阈值使用分片 + 断点续传
    if file_size >= OSS_MULTIPART_THRESHOLD:
        print(f"  使用分片上传，分片大小: {OSS_PART_SIZE / 1024 / 1024:.0f} MB，并发数: {OSS_NUM_THREADS}")
        oss2.resumable_upload(
            bucket,
            object_key,
            str(local_path),
            store=oss2.ResumableStore(root=checkpoint_dir),
            multipart_threshold=OSS_MULTIPART_THRESHOLD,
            part_size=OSS_PART_SIZE,
            progress_callback=progress_callback,
            num_threads=OSS_NUM_THREADS,
        )
    else:
        bucket.put_object_from_file(
            object_key,
            str(local_path),
            progress_callback=progress_callback,
        )

    _clear_checkpoint(str(local_path), object_key)

    if progress:
        print()

    # 生成访问 URL，兼容 endpoint 带协议头的情况
    endpoint_host = bucket.endpoint.replace("http://", "").replace("https://", "")
    url = f"https://{bucket.bucket_name}.{endpoint_host}/{object_key}"

    # 记录上传成功
    record_upload(
        local_path=str(local_path),
        object_key=object_key,
        oss_url=url,
        file_size=file_size,
        status="success",
    )

    print(f"上传完成: {url}")
    return url


def _upload_file_with_record(local_path: str, object_key: str = None, progress: bool = True) -> str:
    """
    带记录功能的上传入口，失败时也会写入记录
    """
    local_path_obj = Path(local_path)
    try:
        return upload_file(local_path, object_key, progress)
    except Exception as e:
        # 记录失败
        record_upload(
            local_path=str(local_path_obj),
            object_key=object_key or local_path_obj.name,
            oss_url="",
            file_size=local_path_obj.stat().st_size if local_path_obj.exists() else 0,
            status="failed",
            error_msg=str(e),
        )
        raise


def upload_directory(local_dir: str, prefix: str = "", progress: bool = True) -> list:
    """
    递归上传整个目录到 OSS
    :param local_dir: 本地目录
    :param prefix: OSS 上的前缀路径
    :return: 上传成功的 URL 列表
    """
    local_dir = Path(local_dir)
    if not local_dir.is_dir():
        raise NotADirectoryError(f"目录不存在: {local_dir}")

    urls = []
    files = [p for p in local_dir.rglob("*") if p.is_file()]
    print(f"共发现 {len(files)} 个文件，开始批量上传...")

    for idx, file_path in enumerate(files, 1):
        relative_path = file_path.relative_to(local_dir)
        object_key = f"{prefix}/{relative_path}".replace("//", "/").lstrip("/")
        try:
            url = _upload_file_with_record(str(file_path), object_key, progress=progress)
            urls.append(url)
        except Exception as e:
            print(f"[{idx}/{len(files)}] 上传失败: {file_path}, 错误: {e}")

    print(f"批量上传结束，成功 {len(urls)}/{len(files)}")
    return urls


def main():
    validate()

    if len(sys.argv) < 2:
        print("用法: python uploader.py <本地文件或目录> [OSS前缀路径]")
        sys.exit(1)

    local_path = sys.argv[1]
    prefix = sys.argv[2] if len(sys.argv) > 2 else ""

    if os.path.isfile(local_path):
        if prefix.endswith("/"):
            object_key = prefix + Path(local_path).name
        elif prefix:
            object_key = prefix
        else:
            object_key = None
        _upload_file_with_record(local_path, object_key)
    elif os.path.isdir(local_path):
        upload_directory(local_path, prefix)
    else:
        print(f"路径无效: {local_path}")
        sys.exit(1)


if __name__ == "__main__":
    main()
