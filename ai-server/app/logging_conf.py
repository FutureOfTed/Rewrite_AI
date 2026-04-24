import logging

def configure_logging():
    # 기본 로깅 포맷/레벨/핸들러 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler()
            # 필요 시 FileHandler 추가
        ]
    )
