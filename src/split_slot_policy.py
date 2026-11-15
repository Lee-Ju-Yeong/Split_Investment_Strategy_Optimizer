LEGACY_SPLIT_SLOT_POLICY = "legacy"
FIXED_SLOT_V1_POLICY = "fixed_slot_v1"
SUPPORTED_SPLIT_SLOT_POLICIES = (LEGACY_SPLIT_SLOT_POLICY, FIXED_SLOT_V1_POLICY)
# GPU 커널이 구현한 차수 규칙(todos/2026_11_15-fixed-split-slot-v1.md).
GPU_SUPPORTED_SPLIT_SLOT_POLICIES = (LEGACY_SPLIT_SLOT_POLICY, FIXED_SLOT_V1_POLICY)


def normalize_split_slot_policy(raw_value):
    # legacy: 새 차수 = 보유 개수+1, 기준가 = 번호가 가장 큰 차수, 1차가 없으면 추가 매수 중단.
    # fixed_slot_v1: 고정 차수 1..N, 2..N 중 가장 낮은 빈칸, 기준가 = 바로 앞 차수, 1차 재진입.
    policy = str(LEGACY_SPLIT_SLOT_POLICY if raw_value is None else raw_value).strip().lower()
    if policy not in SUPPORTED_SPLIT_SLOT_POLICIES:
        raise ValueError(
            "Unsupported split_slot_policy="
            f"{raw_value!r}. supported={list(SUPPORTED_SPLIT_SLOT_POLICIES)}"
        )
    return policy


def require_gpu_supported_split_slot_policy(strategy_params):
    """GPU 커널이 구현하지 않은 차수 규칙이면 계산 전에 멈춘다."""
    policy = normalize_split_slot_policy((strategy_params or {}).get("split_slot_policy"))
    if policy not in GPU_SUPPORTED_SPLIT_SLOT_POLICIES:
        raise ValueError(
            f"split_slot_policy={policy!r} is CPU-only for now. GPU kernels implement "
            f"{list(GPU_SUPPORTED_SPLIT_SLOT_POLICIES)} only; CPU-GPU parity would not hold."
        )
    return policy


def apply_split_slot_policy_to_gpu_execution_params(execution_params, strategy_params):
    """전략 설정의 차수 규칙을 GPU 실행 설정에 싣는다.

    GPU 엔진은 실행 설정의 split_slot_policy만 읽는다. 전략 설정에서 이 값을 옮기지 않으면
    CPU는 fixed_slot_v1, GPU는 legacy로 조용히 갈라지므로 모든 GPU 진입점이 이 함수를 쓴다.
    실행 설정에 이미 다른 값이 있으면 어느 쪽이 맞는지 알 수 없으므로 멈춘다.
    """
    policy = require_gpu_supported_split_slot_policy(strategy_params)
    existing = (execution_params or {}).get("split_slot_policy")
    if existing is not None and normalize_split_slot_policy(existing) != policy:
        raise ValueError(
            "split_slot_policy mismatch between strategy and GPU execution params: "
            f"strategy={policy!r} execution={existing!r}"
        )
    execution_params["split_slot_policy"] = policy
    return policy
