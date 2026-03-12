STRICT_TIER_HYSTERESIS_MODE = "strict_hysteresis_v1"


def normalize_tier_hysteresis_mode(raw_value):
    # strict_hysteresis_v1 means:
    # - Entry: Tier1 only, no Tier2 fallback if Tier1 is empty.
    # - Hold/Add: only positions whose T-1 tier is 1~2 remain eligible.
    mode = str(raw_value or STRICT_TIER_HYSTERESIS_MODE).strip().lower()
    if mode != STRICT_TIER_HYSTERESIS_MODE:
        raise ValueError(
            "Unsupported tier_hysteresis_mode="
            f"{raw_value!r}. strict-only runtime requires "
            f"{STRICT_TIER_HYSTERESIS_MODE!r}."
        )
    return mode
