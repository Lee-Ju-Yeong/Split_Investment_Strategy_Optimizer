# src/config_loader.py (최종 수정본)

import yaml
from pathlib import Path

def load_config(config_path_str: str = 'config/config.yaml') -> dict:
    """
    프로젝트 루트를 기준으로 YAML 설정 파일을 안전하게 찾아 읽어옵니다.
    기본값으로 'config/config.yaml' 파일을 찾습니다.

    Args:
        config_path_str (str): 프로젝트 루트로부터의 상대 경로.

    Returns:
        dict: 설정 내용을 담은 딕셔너리.
    """
    # 1. 이 파일의 절대 경로를 통해 프로젝트 루트를 찾습니다.
    current_file_path = Path(__file__).resolve()
    project_root = current_file_path.parent.parent 

    # 2. 프로젝트 루트와 입력받은 상대 경로(기본값: 'config/config.yaml')를 조합합니다.
    config_path = project_root / config_path_str

    if not config_path.is_file():
        # 디버깅을 위해 에러 메시지에 최종 경로를 포함시킵니다.
        raise FileNotFoundError(f"설정 파일을 찾을 수 없습니다. 시도한 경로: {config_path}")

    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML 파일 파싱 오류: {e}")