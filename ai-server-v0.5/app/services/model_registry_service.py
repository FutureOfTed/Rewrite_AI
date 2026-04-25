import logging

logger = logging.getLogger(__name__)

class ModelRegistryService:
    def __init__(self):
        # 현재 서빙 중인 모델 버전/메타 관리
        self.current_model_version = None

    async def switch_model(self, version_id: str):
        """새로운 버전의 모델로 런타임 스위칭 (롤백 포함)"""
        logger.info(f"Switching model to version: {version_id}")
        self.current_model_version = version_id

registry = ModelRegistryService()
