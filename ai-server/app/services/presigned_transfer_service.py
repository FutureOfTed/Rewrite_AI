import logging
import os
import httpx
# 필요한 경우 aiofiles 등 사용

logger = logging.getLogger(__name__)

async def download_file(url: str, dest_path: str):
    """
    Presigned GET URL을 사용하여 파일을 다운로드합니다.
    청크 다운로드, 재시도, SHA256 해시 검증 등이 포함될 수 있습니다.
    """
    logger.info(f"Downloading file from presigned URL to {dest_path}")
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    timeout = httpx.Timeout(60)
    async with httpx.AsyncClient(timeout=timeout) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            with open(dest_path, "wb") as f:
                async for chunk in response.aiter_bytes():
                    if chunk:
                        f.write(chunk)

async def upload_file(url: str, file_path: str):
    """
    Presigned PUT URL을 사용하여 파일을 업로드합니다.
    재시도 로직 등이 포함될 수 있습니다.
    """
    logger.info(f"Uploading file {file_path} to presigned URL")
    timeout = httpx.Timeout(60)
    with open(file_path, "rb") as f:
        data = f.read()
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.put(
            url,
            content=data,
            headers={"Content-Type": "application/octet-stream"},
        )
        response.raise_for_status()
