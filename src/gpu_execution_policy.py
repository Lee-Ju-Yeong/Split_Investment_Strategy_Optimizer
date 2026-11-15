"""
Shared helpers for strict-only GPU execution parameter normalization.
"""

from __future__ import annotations

from .candidate_runtime_policy import normalize_runtime_candidate_policy
from .split_slot_policy import apply_split_slot_policy_to_gpu_execution_params
from .tier_hysteresis_policy import normalize_tier_hysteresis_mode


def build_gpu_execution_params(
    base_execution_params: dict,
    params_dict: dict,
    universe_mode: str,
    *,
    default_tier_hysteresis_mode: str = "strict_hysteresis_v1",
) -> dict:
    run_exec_params = dict(base_execution_params)
    # 후보 행(전략 설정 + 행 값)의 차수 규칙을 GPU 실행 설정에 싣는다. CPU 검산은 같은 행을
    # 전략에 넣으므로, 여기서 빠지면 CPU와 GPU가 다른 규칙으로 계산한다.
    apply_split_slot_policy_to_gpu_execution_params(run_exec_params, params_dict)
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
