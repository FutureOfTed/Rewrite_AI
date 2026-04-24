import hashlib
import os

def calculate_sha256(file_path: str) -> str:
    """파일의 SHA256 해시값 계산"""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def ensure_dir(dir_path: str):
    """디렉토리가 존재하지 않으면 생성"""
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
