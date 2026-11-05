import sys
from .upload_record import list_records, print_records


def main():
    status = sys.argv[1] if len(sys.argv) > 1 else None
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else 50

    if status and status not in ("success", "failed"):
        print(f"状态参数只能是 success 或 failed，收到: {status}")
        sys.exit(1)

    records = list_records(limit=limit, status=status)
    print_records(records)


if __name__ == "__main__":
    main()
