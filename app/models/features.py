"""
app/models/features.py
-----------------------
Feature extraction for the LightGBM action classifier.

Builds the feature vector described in 04_LOGIC_FLOW.md §6:

  intent_top1              - int index of top-1 intent in class list
  intent_top1_prob         - float confidence of top-1 intent
  intent_top2_prob         - float confidence of runner-up intent
  intent_margin            - top1_prob - top2_prob (spread signal)
  intent_prob_<class>      - full 96-dim intent probability vector (top-k)
  turn_index               - dialogue turn index from payload
  n_turns_so_far           - len(context)
  n_customer_turns         - count of customer speaker turns
  n_agent_turns            - count of agent speaker turns
  n_action_turns           - count of prior action turns
  last_observed_action_idx - int index of last action label, or -1
  action_history_<action>  - multi-hot: which actions occurred so far
  permitted_<action>       - multi-hot: which actions are policy-permitted
  sanitize_was_truncated   - 0/1 flag
  sanitize_had_hostile     - 0/1 flag
  last_turn_embedding_<i>  - 384-dim sentence embedding of last customer turn
                             (reuses the intent backbone — ONE model loaded)

The embedding is optional at feature-build time (can be passed in pre-computed).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.core.schemas import Turn

# All 30 action labels from the training data (sorted, canonical order)
ACTION_VOCABULARY: list[str] = [
    "call-back",
    "change-address",
    "change-name",
    "change-order",
    "change-phone",
    "confirm-reservation",
    "create-account",
    "dispute-bill",
    "downgrade-service",
    "enter-details",
    "escalate",
    "extend-warranty",
    "get-policy",
    "get-refund",
    "instructions",
    "make-password",
    "manager-review",
    "membership-card",
    "n-a",
    "notify-team",
    "offer-refund",
    "place-order",
    "process-payment",
    "pull-up-account",
    "record-reason",
    "send-link",
    "ship-item",
    "track-order",
    "upgrade-service",
    "verify-identity",
]


class DialogueFeatureExtractor:
    """
    Extracts the complete feature vector for the LightGBM action classifier.

    Parameters
    ----------
    action_vocabulary : list[str]
        Canonical action labels — must match training order.
    intent_classes : list[str]
        Canonical intent labels — must match IntentClassifier's class order.
    embedding_dim : int
        Dimensionality of the sentence embedding (384 for MiniLM-L12-v2).
    """

    def __init__(
        self,
        action_vocabulary: list[str] | None = None,
        intent_classes: list[str] | None = None,
        embedding_dim: int = 384,
    ) -> None:
        self.action_vocabulary = action_vocabulary or ACTION_VOCABULARY
        self.intent_classes = intent_classes or []
        self.embedding_dim = embedding_dim

        # Index maps for O(1) lookups
        self._action_idx = {a: i for i, a in enumerate(self.action_vocabulary)}
        self._intent_idx = {c: i for i, c in enumerate(self.intent_classes)}

    # ──────────────────────────────────────────────────────────────────────────
    # Context formatting (also used by intent model input)
    # ──────────────────────────────────────────────────────────────────────────

    def format_context_for_embedding(
        self, turns: list[Turn], max_turns: int = 8
    ) -> str:
        """
        Format the last N conversation turns with speaker labels for encoding.
        Caps at max_turns (service contract: ≤8 turns in context).
        Applies lexical and code-mixed phrase normalization.
        """
        from app.sanitization.sanitizer import normalize_noisy_text

        return "\n".join(
            f"[{t.speaker}] {normalize_noisy_text(t.text) if t.text else ''}"
            for t in turns[-max_turns:]
        )

    def get_last_customer_text(self, turns: list[Turn]) -> str:
        """Extract the most recent customer turn text (the live request)."""
        for t in reversed(turns):
            if t.speaker == "customer":
                return t.text
        return ""

    # ──────────────────────────────────────────────────────────────────────────
    # Feature vector construction
    # ──────────────────────────────────────────────────────────────────────────

    def extract_state_features(
        self,
        context: list[Turn],
        turn_index: int | None = None,
        intent_probs: dict[str, float] | None = None,
        permitted_actions: list[str] | None = None,
        last_turn_embedding: np.ndarray | None = None,
        sanitize_was_truncated: bool = False,
        sanitize_had_hostile: bool = False,
    ) -> dict[str, Any]:
        """
        Build the full tabular feature dict for the LightGBM model.

        All features are numeric (int or float).  The returned dict is keyed
        by feature name; pass it directly to ActionClassifier.predict().

        Parameters
        ----------
        context               : list of Turn objects from the request
        turn_index            : optional turn index from payload
        intent_probs          : full probability distribution from IntentClassifier
        permitted_actions     : list of policy-permitted action labels
        last_turn_embedding   : pre-computed sentence embedding (384-dim np.ndarray)
                                Pass None to omit embedding features (e.g., in tests)
        sanitize_was_truncated: flag from SanitizeResult
        sanitize_had_hostile  : flag from SanitizeResult
        """
        permitted_set = set(permitted_actions or [])
        observed_actions = [t.text.strip() for t in context if t.speaker == "action"]
        last_action = observed_actions[-1] if observed_actions else "none"

        features: dict[str, Any] = {}

        # ── Dialogue state features ───────────────────────────────────────
        features["turn_index"] = (
            int(turn_index) if turn_index is not None else len(context)
        )
        features["n_turns_so_far"] = len(context)
        features["n_customer_turns"] = sum(
            1 for t in context if t.speaker == "customer"
        )
        features["n_agent_turns"] = sum(1 for t in context if t.speaker == "agent")
        features["n_action_turns"] = len(observed_actions)
        features["last_observed_action_idx"] = self._action_idx.get(last_action, -1)

        # ── Action history multi-hot ──────────────────────────────────────
        observed_set = set(observed_actions)
        for act in self.action_vocabulary:
            features[f"hist_{act}"] = 1 if act in observed_set else 0

        # ── Permitted actions multi-hot ───────────────────────────────────
        for act in self.action_vocabulary:
            features[f"perm_{act}"] = 1 if act in permitted_set else 0

        # ── Sanitize flags ────────────────────────────────────────────────
        features["sanitize_was_truncated"] = 1 if sanitize_was_truncated else 0
        features["sanitize_had_hostile"] = 1 if sanitize_had_hostile else 0

        # ── Intent probability features ───────────────────────────────────
        if intent_probs:
            sorted_probs = sorted(intent_probs.values(), reverse=True)
            top1_prob = sorted_probs[0] if len(sorted_probs) > 0 else 0.0
            top2_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0

            top1_intent = (
                max(intent_probs, key=intent_probs.__getitem__)
                if intent_probs
                else "unknown"
            )
            features["intent_top1_idx"] = self._intent_idx.get(top1_intent, -1)
            features["intent_top1_prob"] = float(top1_prob)
            features["intent_top2_prob"] = float(top2_prob)
            features["intent_margin"] = float(top1_prob - top2_prob)

            # Full intent probability vector (dense — LightGBM handles this well)
            for cls in self.intent_classes:
                features[f"iprob_{cls}"] = float(intent_probs.get(cls, 0.0))
        else:
            features["intent_top1_idx"] = -1
            features["intent_top1_prob"] = 0.0
            features["intent_top2_prob"] = 0.0
            features["intent_margin"] = 0.0
            for cls in self.intent_classes:
                features[f"iprob_{cls}"] = 0.0

        # ── Sentence embedding features ───────────────────────────────────
        if last_turn_embedding is not None:
            emb = np.asarray(last_turn_embedding, dtype=np.float32).flatten()
            # Pad or truncate to expected dim
            if len(emb) < self.embedding_dim:
                emb = np.pad(emb, (0, self.embedding_dim - len(emb)))
            else:
                emb = emb[: self.embedding_dim]
            for i, v in enumerate(emb):
                features[f"emb_{i}"] = float(v)
        else:
            for i in range(self.embedding_dim):
                features[f"emb_{i}"] = 0.0

        return features

    def get_feature_column_names(self) -> list[str]:
        """
        Return the ordered list of feature column names — saved as a sidecar
        JSON alongside the LightGBM model to ensure alignment at inference time.
        """
        # Build a dummy feature dict to get column order (same logic as above)
        dummy = self.extract_state_features(
            context=[],
            intent_probs={c: 0.0 for c in self.intent_classes}
            if self.intent_classes
            else {},
            last_turn_embedding=np.zeros(self.embedding_dim, dtype=np.float32),
        )
        return list(dummy.keys())
