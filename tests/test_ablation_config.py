"""Unit tests for AblationConfig — including the consolidated LLM hyperparameters."""
from __future__ import annotations

import pytest

from user_simulator.ablation import AblationConfig


class TestFactoryDefaults:
    """Each factory carries the same hyperparameter defaults."""

    @pytest.fixture(params=["full", "no_privilege", "no_behavior", "no_state",
                            "oracle_profile_only"])
    def config(self, request):
        return AblationConfig.from_name(request.param)

    def test_user_hyperparameters(self, config):
        assert config.user_temperature == 0.7
        assert config.user_max_tokens == 2048
        assert config.user_retry_temps == (0.7, 0.8, 0.9, 1.0, 1.1)

    def test_assistant_hyperparameters(self, config):
        assert config.assistant_temperature == 0.7
        assert config.assistant_max_tokens == 1024

    def test_controller_hyperparameters(self, config):
        assert config.controller_temperature == 0.9
        assert config.controller_max_tokens == 128

    def test_scenario_constructor_hyperparameters(self, config):
        assert config.scenario_constructor_temperature == 0.8
        assert config.scenario_constructor_max_tokens == 4096

    def test_recent_history_window(self, config):
        assert config.recent_history_window == 4


class TestFactorySemantics:
    """Each factory still expresses the right ablation axes."""

    def test_full(self):
        c = AblationConfig.full()
        assert c.use_user_state is True
        assert c.use_behavior_injection is True
        assert c.assistant_strategy == "oracle"
        assert c.sft_include_profile is True
        assert c.name == "full"

    def test_no_privilege_strips_oracle_and_profile(self):
        c = AblationConfig.no_privilege()
        assert c.assistant_strategy == "vanilla"
        assert c.sft_include_profile is False

    def test_no_behavior_disables_injection(self):
        c = AblationConfig.no_behavior()
        assert c.use_behavior_injection is False
        assert c.use_user_state is True

    def test_no_state_disables_user_state_and_behavior(self):
        c = AblationConfig.no_state()
        assert c.use_user_state is False
        assert c.use_behavior_injection is False

    def test_oracle_profile_only(self):
        c = AblationConfig.oracle_profile_only()
        assert c.assistant_strategy == "oracle_profile_only"

    def test_from_name_unknown_raises(self):
        with pytest.raises(ValueError):
            AblationConfig.from_name("not_a_real_ablation")


class TestOverrides:
    def test_caller_can_override_hyperparameters(self):
        c = AblationConfig(
            user_temperature=0.3,
            assistant_max_tokens=512,
            controller_max_tokens=256,
        )
        assert c.user_temperature == 0.3
        assert c.assistant_max_tokens == 512
        assert c.controller_max_tokens == 256
        # Untouched fields stay at defaults
        assert c.user_max_tokens == 2048
