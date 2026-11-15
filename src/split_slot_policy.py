LEGACY_SPLIT_SLOT_POLICY = "legacy"
FIXED_SLOT_V1_POLICY = "fixed_slot_v1"
SUPPORTED_SPLIT_SLOT_POLICIES = (LEGACY_SPLIT_SLOT_POLICY, FIXED_SLOT_V1_POLICY)
# GPU 커널은 아직 legacy 차수 규칙만 구현한다(todos/2026_11_15-fixed-split-slot-v1.md).
GPU_SUPPORTED_SPLIT_SLOT_POLICIES = (LEGACY_SPLIT_SLOT_POLICY,)


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
    """GPU 경로가 CPU와 다른 차수 규칙으로 조용히 계산하지 않도록 막는다."""
    policy = normalize_split_slot_policy((strategy_params or {}).get("split_slot_policy"))
    if policy not in GPU_SUPPORTED_SPLIT_SLOT_POLICIES:
        raise ValueError(
            f"split_slot_policy={policy!r} is CPU-only for now. GPU kernels implement "
            f"{list(GPU_SUPPORTED_SPLIT_SLOT_POLICIES)} only; CPU-GPU parity would not hold."
        )
    return policy
