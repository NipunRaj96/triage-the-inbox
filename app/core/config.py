"""Application configuration and policy parameters loaded via .env and YAML."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import yaml
from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Ensure .env is loaded into os.environ before any libraries inspect process env
load_dotenv()


class Settings(BaseSettings):
    """Runtime application settings, model paths, and policy parameters."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Service & Environment
    app_name: str = "OP-06 Triage Service"
    environment: str = "development"
    port: int = 8000
    host: str = "0.0.0.0"
    log_level: str = "INFO"

    # Base Paths
    data_dir: Path = Path("data")
    artifacts_dir: Path = Path("artifacts/models")
    guidelines_path: Path = Path("data/guidelines.json")
    policy_overrides_path: Path = Path("config/policy_overrides.yaml")

    # Granular Model Artifact Paths (all configurable via .env)
    intent_model_dir: Path = Path("artifacts/models/intent")
    intent_classes_path: Path = Path("artifacts/models/intent_classes.json")
    action_model_path: Path = Path("artifacts/models/action/action_model.pkl")
    action_classes_path: Path = Path("artifacts/models/action/action_classes.json")
    action_columns_path: Path = Path("artifacts/models/action/feature_columns.json")
    calibrator_path: Path = Path("artifacts/models/calibration/calibrator.pkl")

    # Policy & Human-in-the-loop thresholds
    policy_boost: float = 1.0
    human_threshold: float = 0.62
    policy_deviation_threshold: float = 0.80
    max_tokens_sanitize: int = 2000
    always_escalate_actions: list[str] = Field(
        default_factory=lambda: ["escalate", "manager-review"]
    )

    # Calibration parameters
    calibration_split_seed: int = 99
    calibration_fit_rows: int = 400

    # System & OpenMP / Threading
    omp_num_threads: str = "1"
    tokenizers_parallelism: str = "false"

    @classmethod
    def load_with_overrides(
        cls, config_path: Path = Path("config/policy_overrides.yaml")
    ) -> Settings:
        """
        Load settings from .env / environment variables, overlaying config/policy_overrides.yaml
        if present, with environment variables holding highest precedence.
        """
        settings = cls()

        target_yaml = (
            config_path if config_path.exists() else settings.policy_overrides_path
        )
        if target_yaml.exists():
            with open(target_yaml, "r", encoding="utf-8") as fh:
                overrides = yaml.safe_load(fh) or {}

            # Apply YAML overrides only if not explicitly overridden in environment / .env
            env_keys = set(os.environ.keys())
            if (
                "policy_boost" in overrides
                and not {"POLICY_BOOST", "policy_boost"} & env_keys
            ):
                settings.policy_boost = float(overrides["policy_boost"])
            if (
                "human_threshold" in overrides
                and not {"HUMAN_THRESHOLD", "human_threshold"} & env_keys
            ):
                settings.human_threshold = float(overrides["human_threshold"])
            if (
                "policy_deviation_threshold" in overrides
                and not {"POLICY_DEVIATION_THRESHOLD", "policy_deviation_threshold"}
                & env_keys
            ):
                settings.policy_deviation_threshold = float(
                    overrides["policy_deviation_threshold"]
                )
            if (
                "always_escalate_actions" in overrides
                and not {"ALWAYS_ESCALATE_ACTIONS", "always_escalate_actions"}
                & env_keys
            ):
                settings.always_escalate_actions = list(
                    overrides["always_escalate_actions"]
                )
            if (
                "sanitize" in overrides
                and "max_tokens" in overrides["sanitize"]
                and not {"MAX_TOKENS_SANITIZE", "max_tokens_sanitize"} & env_keys
            ):
                settings.max_tokens_sanitize = int(overrides["sanitize"]["max_tokens"])

        return settings


@lru_cache
def get_settings() -> Settings:
    return Settings.load_with_overrides()
