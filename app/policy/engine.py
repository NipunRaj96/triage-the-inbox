"""Policy engine — canonical permitted-actions lookup.

Primary source: ``data/canonical_policy.json`` (extracted from
``train.jsonl.gz`` at build time; 100 % deterministic, 96 intents).
Fallback: ``guidelines.json`` naive parse + YAML overrides (kept for
edge-case intents not seen at training time).
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml


class PolicyEngine:
    """Provides permitted actions for each subflow/intent."""

    def __init__(
        self,
        guidelines_path: Path | str = Path("data/guidelines.json"),
        overrides_path: Path | str = Path("config/policy_overrides.yaml"),
        canonical_path: Path | str = Path("data/canonical_policy.json"),
    ):
        self.guidelines_path = Path(guidelines_path)
        self.overrides_path = Path(overrides_path)
        self.canonical_path = Path(canonical_path)
        self.permitted_actions: dict[str, list[str]] = {}
        self.load_policy()

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_policy(self) -> None:
        """Load canonical training-data policy, then overlay guidelines/overrides."""

        # 1. Primary: canonical policy extracted from train.jsonl.gz
        if self.canonical_path.exists():
            with open(self.canonical_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            for intent, actions in raw.items():
                self.permitted_actions[intent.lower().replace(" ", "_")] = sorted(
                    actions
                )

        # 2. Fallback: guidelines.json for any intents not covered above
        if self.guidelines_path.exists():
            with open(self.guidelines_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)

            for flow_obj in data.values():
                subflows = flow_obj.get("subflows", {})
                for subflow_name, subflow_data in subflows.items():
                    norm_key = subflow_name.lower().replace(" ", "_")
                    if norm_key in self.permitted_actions:
                        # Already populated from canonical — skip
                        continue
                    actions_list = []
                    for entry in subflow_data.get("actions", []):
                        btn = entry.get("button", "").strip().lower().replace(" ", "-")
                        if btn and btn != "n/a":
                            actions_list.append(btn)
                    self.permitted_actions[norm_key] = sorted(set(actions_list))

        # 3. YAML overrides (additive corrections, fine-tuning)
        if self.overrides_path.exists():
            with open(self.overrides_path, "r", encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            for subflow, patch in cfg.get("overrides", {}).items():
                current = set(self.permitted_actions.get(subflow, []))
                for add_act in patch.get("add", []):
                    current.add(add_act)
                for rem_act in patch.get("remove", []):
                    current.discard(rem_act)
                self.permitted_actions[subflow] = sorted(current)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_permitted_actions(self, intent: str) -> list[str]:
        """Return permitted action identifiers for the given intent."""
        norm = intent.lower().replace(" ", "_")
        return self.permitted_actions.get(norm, [])

    def is_action_permitted(self, intent: str, action: str) -> bool:
        """Check if action is authorized under this intent."""
        return action in self.get_permitted_actions(intent)
