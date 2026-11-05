import sqlite3
from pathlib import Path
from datetime import datetime
from .config import UPLOAD_RECORD_DB


DB_PATH = Path(UPLOAD_RECORD_DB)


def _get_conn():
    """获取数据库连接"""
    return sqlite3.connect(str(DB_PATH))


def init_db():
    """初始化上传记录表"""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = _get_conn()
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS upload_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            local_path TEXT NOT NULL,
            object_key TEXT NOT NULL,
            oss_url TEXT NOT NULL,
            file_size INTEGER DEFAULT 0,
            status TEXT NOT NULL,
            error_msg TEXT,
            uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_uploaded_at ON upload_records(uploaded_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_local_path ON upload_records(local_path)
        """
    )
    conn.commit()
    conn.close()


def record_upload(local_path: str, object_key: str, oss_url: str, file_size: int,
                  status: str = "success", error_msg: str = None):
    """
    记录一次上传结果
    :param local_path: 本地文件路径
    :param object_key: OSS 对象名
    :param oss_url: OSS 访问 URL
    :param file_size: 文件大小（字节）
    :param status: success / failed
    :param error_msg: 失败时的错误信息
    """
    init_db()
    conn = _get_conn()
    conn.execute(
        """
        INSERT INTO upload_records (local_path, object_key, oss_url, file_size, status, error_msg)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (str(local_path), object_key, oss_url, file_size, status, error_msg),
    )
    conn.commit()
    conn.close()


def list_records(limit: int = 50, status: str = None):
    """
    查询上传记录
    :param limit: 返回条数
    :param status: 筛选状态 success / failed，None 表示全部
    :return: 记录列表
    """
    init_db()
    conn = _get_conn()
    conn.row_factory = sqlite3.Row
    if status:
        rows = conn.execute(
            "SELECT * FROM upload_records WHERE status = ? ORDER BY uploaded_at DESC LIMIT ?",
            (status, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM upload_records ORDER BY uploaded_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_records_by_path(local_path: str, limit: int = 10):
    """根据本地路径查询上传记录"""
    init_db()
    conn = _get_conn()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM upload_records WHERE local_path = ? ORDER BY uploaded_at DESC LIMIT ?",
        (str(local_path), limit),
    ).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def format_size(size_bytes: int) -> str:
    """将字节大小格式化为人类可读字符串"""
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / 1024 / 1024:.2f} MB"
    else:
        return f"{size_bytes / 1024 / 1024 / 1024:.2f} GB"


def print_records(records):
    """打印记录列表"""
    if not records:
        print("暂无上传记录")
        return

    print(f"{'ID':<5} {'状态':<8} {'时间':<20} {'大小':<10} {'本地路径':<40} {'OSS 路径':<30}")
    print("-" * 120)
    for r in records:
        local = r["local_path"]
        object_key = r["object_key"]
        if len(local) > 38:
            local = "..." + local[-35:]
        if len(object_key) > 28:
            object_key = "..." + object_key[-25:]
        print(
            f"{r['id']:<5} {r['status']:<8} {r['uploaded_at']:<20} "
            f"{format_size(r['file_size']):<10} {local:<40} {object_key:<30}"
        )


if __name__ == "__main__":
    import sys

    init_db()
    status_filter = sys.argv[1] if len(sys.argv) > 1 else None
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    records = list_records(limit=limit, status=status_filter)
    print_records(records)
