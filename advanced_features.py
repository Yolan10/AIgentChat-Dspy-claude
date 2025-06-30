"""Collection of placeholder classes implementing advanced behaviours."""
from __future__ import annotations

import json
from typing import Any, Dict, List
import random

from tracking_chat_openai import TrackingChatOpenAI as ChatOpenAI

try:  # pragma: no cover - optional dependency
    from openai import OpenAIError  # type: ignore
except Exception:  # pragma: no cover - when openai is missing
    try:
        from openai.error import OpenAIError  # type: ignore
    except Exception:
        OpenAIError = Exception

try:  # pragma: no cover - requests optional
    from requests.exceptions import RequestException
except Exception:  # pragma: no cover - fallback when requests missing
    RequestException = Exception
from langchain_core.messages import SystemMessage, HumanMessage

import config
import utils


class PopulationGenerator:
    """DSPy-powered population generation."""

    def generate(self, market_context: str, n: int) -> List[Dict[str, Any]]:
        """Return a list of persona specifications."""

        template = utils.load_template(config.POPULATION_INSTRUCTION_TEMPLATE_PATH)
        prompt = utils.render_template(template, {"instruction": market_context, "n": n})

        llm = ChatOpenAI(
            model=config.LLM_MODEL,
            temperature=config.LLM_TEMPERATURE,
            max_tokens=config.LLM_MAX_TOKENS,
            max_retries=config.OPENAI_MAX_RETRIES,
        )
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content="Provide the JSON array only."),
        ]
        try:
            response = llm.invoke(messages).content
            try:
                personas = json.loads(response)
            except json.JSONDecodeError:
                personas = utils.extract_json_array(response)
                if personas is None:
                    raise
            result: List[Dict[str, Any]] = []
            for spec in personas:
                if isinstance(spec, str):
                    try:
                        spec = json.loads(spec)
                    except json.JSONDecodeError:
                        continue
                if isinstance(spec, dict):
                    result.append(spec)
            return result
        except (
            OpenAIError,
            RequestException,
            json.JSONDecodeError,
        ) as exc:  # pragma: no cover - network failure fallback
            print(
                f"Population generation failed: {exc}. Using random fallback."
            )
            return self._fallback_personas(n)

    def _fallback_personas(self, n: int) -> List[Dict[str, Any]]:
        """Return a simple offline population when the LLM call fails."""
        names = [
            "Alice",
            "Bob",
            "Carol",
            "Dave",
            "Eve",
            "Frank",
            "Grace",
            "Heidi",
            "Ivan",
            "Judy",
            "Mallory",
            "Niaj",
            "Olivia",
            "Peggy",
            "Rupert",
            "Sybil",
            "Trent",
            "Victor",
            "Wendy",
            "Yvonne",
        ]
        occupations = [
            "teacher",
            "engineer",
            "artist",
            "doctor",
            "writer",
        ]
        goals = [
            "improve communication",
            "find better hearing aids",
            "learn sign language",
            "connect with community",
            "share experiences",
        ]
        personalities = [
            "O:0.6 C:0.7 E:0.5 A:0.6 N:0.4",
            "O:0.8 C:0.5 E:0.7 A:0.7 N:0.3",
            "O:0.4 C:0.6 E:0.6 A:0.5 N:0.5",
        ]
        memory = [
            "struggled with hearing in crowds",
            "recently started using aids",
            "has family history of hearing loss",
        ]
        result = []
        for _ in range(n):
            name = random.choice(names)
            age = random.randint(25, 70)
            occ = random.choice(occupations)
            result.append(
                {
                    "name": name,
                    "personality": random.choice(personalities),
                    "age": age,
                    "occupation": occ,
                    "initial_goals": random.choice(goals),
                    "memory_summary": random.choice(memory),
                }
            )
        return result


class StrategySelector:
    """Adaptive strategy system choosing persuasion styles."""

    def select(self, history: List[Dict[str, str]]) -> str:
        """Return the chosen strategy based on history."""
        # Very naive implementation
        return "logical" if len(history) % 2 == 0 else "emotional"


class ResultCache:
    """Simple disk based cache keyed by content hash with batched writes."""

    def __init__(self, path: str = "cache.json", flush_every: int = 10) -> None:
        self.path = path
        self.flush_every = flush_every
        self._dirty = 0
        self.data: Dict[str, Any] = {}
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                self.data = json.load(fh)
        except Exception:
            self.data = {}

    def get(self, key: str) -> Any:
        return self.data.get(key)

    def set(self, key: str, value: Any) -> None:
        self.data[key] = value
        self._dirty += 1
        if self._dirty >= self.flush_every:
            self.flush()

    def flush(self) -> None:
        """Write cached data to disk if any changes are pending."""
        if self._dirty:
            with open(self.path, "w", encoding="utf-8") as fh:
                json.dump(self.data, fh)
            self._dirty = 0

    def __del__(self) -> None:
        try:
            self.flush()
        except Exception:
            pass
