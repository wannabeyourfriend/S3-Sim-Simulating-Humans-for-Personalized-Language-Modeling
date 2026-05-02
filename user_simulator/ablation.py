"""Ablation configuration for S3-Sim 4-experiment data generation."""

from dataclasses import dataclass, field


@dataclass
class AblationConfig:
    """Controls which pipeline components are active.
    4 experiments from ablation-on-simulator-compounet.md:
    1. full:          user(state+behavior) + assistant(oracle w/ profile+state)
    2. no_privilege:   user(state+behavior) + assistant(vanilla, no profile)
    3. no_behavior:    user(state only)     + assistant(oracle w/ profile+state)
    4. no_state:       user(vanilla)        + assistant(oracle w/ profile+state)
    """

    use_user_state: bool = True
    use_behavior_injection: bool = True
    history_window: int | None = None

    assistant_strategy: str = "oracle"

    sft_include_profile: bool = True

    name: str = "full"

    user_temperature: float = 0.7
    user_max_tokens: int = 2048
    user_retry_temps: tuple[float, ...] = (0.7, 0.8, 0.9, 1.0, 1.1)

    assistant_temperature: float = 0.7
    assistant_max_tokens: int = 1024

    controller_temperature: float = 0.9
    controller_max_tokens: int = 128

    scenario_constructor_temperature: float = 0.8
    scenario_constructor_max_tokens: int = 4096

    recent_history_window: int = 4

    @classmethod
    def full(cls) -> "AblationConfig":
        """Best setting: S3 user (state+behavior) + oracle assistant (profile+state)."""
        return cls(name="full")

    @classmethod
    def no_privilege(cls) -> "AblationConfig":
        """S3 user (state+behavior) + vanilla assistant (no profile access)."""
        return cls(assistant_strategy="vanilla", sft_include_profile=False, name="no_privilege")

    @classmethod
    def no_behavior(cls) -> "AblationConfig":
        """S3 user (state only, no behavior) + oracle assistant."""
        return cls(use_behavior_injection=False, name="no_behavior")

    @classmethod
    def no_state(cls) -> "AblationConfig":
        """Vanilla user (no state, no behavior) + oracle assistant (profile+state)."""
        return cls(use_user_state=False, use_behavior_injection=False, name="no_state")

    @classmethod
    def oracle_profile_only(cls) -> "AblationConfig":
        """S3 user (state+behavior) + oracle with profile only (no user_state access)."""
        return cls(assistant_strategy="oracle_profile_only", name="oracle_profile_only")

    @classmethod
    def from_name(cls, name: str) -> "AblationConfig":
        presets = {
            "full": cls.full,
            "no_privilege": cls.no_privilege,
            "no_behavior": cls.no_behavior,
            "no_state": cls.no_state,
            "oracle_profile_only": cls.oracle_profile_only,
        }
        factory = presets.get(name)
        if factory is None:
            raise ValueError(f"Unknown ablation: {name!r}. Choose from {list(presets)}")
        return factory()
