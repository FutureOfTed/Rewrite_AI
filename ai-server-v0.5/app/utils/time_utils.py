from datetime import datetime, timezone

def get_current_utc_time() -> str:
    """ISO 8601 포맷의 현재 UTC 시각 반환"""
    return datetime.now(timezone.utc).isoformat()
