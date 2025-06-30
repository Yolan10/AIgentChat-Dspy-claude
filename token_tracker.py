from __future__ import annotations

import json
import os
from typing import Dict

import config
import utils

# Approximate pricing per 1K tokens for each model
MODEL_PRICING = {
    "gpt-4.1-nano": {"prompt": 0.0001, "completion": 0.0004},  # $0.10/$0.40 per 1M tokens
    "gpt-4.1-mini": {"prompt": 0.00015, "completion": 0.0006},  # Estimated
    "gpt-4.1": {"prompt": 0.001, "completion": 0.003},  # Estimated
    "gpt-4o-mini": {"prompt": 0.00015, "completion": 0.0006},
}


class TokenTracker:
    """Track token usage and estimated cost per run."""

    def __init__(self) -> None:
        utils.ensure_logs_dir()
        self.file_path = os.path.join(config.LOGS_DIRECTORY, "token_usage.json")
        self.current_run: str | None = None
        self.totals: Dict[str, Dict[str, float]] = {}
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as fh:
                    self.totals = json.load(fh)
            except Exception:
                self.totals = {}

    def set_run(self, run_no: int) -> None:
        self.current_run = str(run_no)
        self.totals.setdefault(
            self.current_run,
            {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0},
        )
        self._flush()

    def add_usage(self, model: str, prompt_tokens: int, completion_tokens: int) -> None:
        if self.current_run is None:
            return
        totals = self.totals.setdefault(
            self.current_run,
            {"prompt_tokens": 0, "completion_tokens": 0, "cost": 0.0},
        )
        totals["prompt_tokens"] += int(prompt_tokens)
        totals["completion_tokens"] += int(completion_tokens)
        pricing = MODEL_PRICING.get(model, MODEL_PRICING.get(config.LLM_MODEL, {"prompt": 0.0, "completion": 0.0}))
        cost = (prompt_tokens / 1000) * pricing.get("prompt", 0) + (completion_tokens / 1000) * pricing.get("completion", 0)
        totals["cost"] += cost
        self._flush()

    def _flush(self) -> None:
        with open(self.file_path, "w", encoding="utf-8") as fh:
            json.dump(self.totals, fh, indent=config.JSON_INDENT)


# Global instance
token_tracker = TokenTracker()
