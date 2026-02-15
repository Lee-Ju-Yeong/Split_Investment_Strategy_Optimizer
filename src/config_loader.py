import os
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG_RELATIVE_PATH = Path("config") / "config.yaml"
_CONFIG_PATH_ENV_VAR = "MAGICSPLIT_CONFIG_PATH"


def get_project_root() -> Path:
    return _PROJECT_ROOT


def resolve_project_path(path_str: str) -> Path:
    """
    Resolve a user-provided path.
    - absolute paths are used as-is
    - relative paths are resolved from the project root
    """
    p = Path(path_str)
    return p if p.is_absolute() else _PROJECT_ROOT / p


def load_config(config_path_str: str | None = None) -> dict:
    """
    Load YAML config from project root by default.

    Resolution order:
    1) explicit `config_path_str`
    2) env var `MAGICSPLIT_CONFIG_PATH`
    3) default `config/config.yaml` (relative to project root)
    """
    config_path_str = config_path_str or os.environ.get(_CONFIG_PATH_ENV_VAR) or str(_DEFAULT_CONFIG_RELATIVE_PATH)
    config_path = resolve_project_path(config_path_str)

    if not config_path.is_file():
        raise FileNotFoundError(
            "설정 파일을 찾을 수 없습니다. "
            f"시도한 경로: {config_path} "
            f"(override: env {_CONFIG_PATH_ENV_VAR} 또는 load_config(path) 인자로 지정)"
        )

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        return config
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"YAML 파일 파싱 오류: {e}") from e
