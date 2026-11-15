"""
Shared helpers for strict-only GPU execution parameter normalization.
"""

from __future__ import annotations

from .candidate_runtime_policy import normalize_runtime_candidate_policy
from .split_slot_policy import require_gpu_supported_split_slot_policy
from .tier_hysteresis_policy import normalize_tier_hysteresis_mode


def build_gpu_execution_params(
    base_execution_params: dict,
    params_dict: dict,
    universe_mode: str,
    *,
    default_tier_hysteresis_mode: str = "strict_hysteresis_v1",
) -> dict:
    run_exec_params = dict(base_execution_params)
    # GPU 커널은 legacy 차수 규칙만 구현한다. 후보 행·설정 어디에 CPU 전용 정책이 있어도
    # 다른 규칙의 성과를 그 정책 이름으로 기록하지 않도록 여기서 멈춘다.
    require_gpu_supported_split_slot_policy(base_execution_params)
    run_exec_params["split_slot_policy"] = require_gpu_supported_split_slot_policy(params_dict)
    candidate_source_mode, use_weekly_alpha_gate = normalize_runtime_candidate_policy(
        params_dict.get("candidate_source_mode", "tier"),
        params_dict.get("use_weekly_alpha_gate", False),
    )
    run_exec_params["candidate_source_mode"] = candidate_source_mode
    run_exec_params["use_weekly_alpha_gate"] = use_weekly_alpha_gate
    run_exec_params["tier_hysteresis_mode"] = normalize_tier_hysteresis_mode(
        params_dict.get("tier_hysteresis_mode", default_tier_hysteresis_mode)
    )
    run_exec_params["universe_mode"] = universe_mode
    return run_exec_params


__all__ = [
    "build_gpu_execution_params",
]
