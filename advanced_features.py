"""Collection of placeholder classes implementing advanced behaviours with improved debugging."""
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
    """DSPy-powered population generation with improved error handling and debugging."""

    def generate(self, market_context: str, n: int) -> List[Dict[str, Any]]:
        """Return a list of persona specifications."""
        
        print(f"[DEBUG] PopulationGenerator.generate() called with:")
        print(f"  - market_context: '{market_context}'")
        print(f"  - n: {n}")

        try:
            # Load template
            print(f"[DEBUG] Loading template from: {config.POPULATION_INSTRUCTION_TEMPLATE_PATH}")
            template = utils.load_template(config.POPULATION_INSTRUCTION_TEMPLATE_PATH)
            print(f"[DEBUG] Template loaded successfully, length: {len(template)} chars")
            
            # Render template
            prompt = utils.render_template(template, {"instruction": market_context, "n": n})
            print(f"[DEBUG] Rendered prompt length: {len(prompt)} chars")
            print(f"[DEBUG] First 200 chars of prompt: {prompt[:200]}...")

            # Create LLM
            print(f"[DEBUG] Creating ChatOpenAI with model: {config.LLM_MODEL}")
            llm = ChatOpenAI(
                model=config.LLM_MODEL,
                temperature=config.LLM_TEMPERATURE,
                max_tokens=config.LLM_MAX_TOKENS,
                max_retries=config.OPENAI_MAX_RETRIES,
            )
            print(f"[DEBUG] ChatOpenAI created successfully")
            
            # Create messages
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content="Provide the JSON array only."),
            ]
            print(f"[DEBUG] Created {len(messages)} messages")
            
            # Make LLM call
            print(f"[DEBUG] Making LLM call...")
            response = llm.invoke(messages).content
            print(f"[DEBUG] LLM response received, length: {len(response)} chars")
            print(f"[DEBUG] First 300 chars of response: {response[:300]}...")
            
            # Parse JSON
            print(f"[DEBUG] Attempting to parse JSON...")
            try:
                personas = json.loads(response)
                print(f"[DEBUG] JSON parsed successfully, type: {type(personas)}, length: {len(personas) if isinstance(personas, (list, dict)) else 'N/A'}")
            except json.JSONDecodeError as e:
                print(f"[DEBUG] JSON parsing failed: {e}")
                print(f"[DEBUG] Attempting fallback extraction...")
                personas = utils.extract_json_array(response)
                if personas is None:
                    print(f"[DEBUG] Fallback extraction also failed")
                    raise
                else:
                    print(f"[DEBUG] Fallback extraction succeeded, type: {type(personas)}, length: {len(personas)}")

            # Process personas
            result: List[Dict[str, Any]] = []
            print(f"[DEBUG] Processing {len(personas)} personas...")
            
            for i, spec in enumerate(personas):
                print(f"[DEBUG] Processing persona {i+1}: type={type(spec)}")
                
                if isinstance(spec, str):
                    print(f"[DEBUG] Persona {i+1} is string, attempting JSON parse...")
                    try:
                        spec = json.loads(spec)
                        print(f"[DEBUG] String parsed to: {type(spec)}")
                    except json.JSONDecodeError as e:
                        print(f"[DEBUG] Failed to parse string as JSON: {e}")
                        print(f"[DEBUG] String content: {spec[:100]}...")
                        continue
                        
                if isinstance(spec, dict):
                    print(f"[DEBUG] Adding persona {i+1} to result (keys: {list(spec.keys())})")
                    result.append(spec)
                else:
                    print(f"[DEBUG] Skipping persona {i+1} - not a dict (type: {type(spec)})")

            print(f"[DEBUG] Final result: {len(result)} personas")
            return result
            
        except (OpenAIError, RequestException, json.JSONDecodeError) as exc:
            print(f"[ERROR] Population generation failed: {exc}. Using random fallback.")
            print(f"[ERROR] Exception type: {type(exc)}")
            fallback_result = self._fallback_personas(n)
            print(f"[DEBUG] Fallback generated {len(fallback_result)} personas")
            return fallback_result
        except Exception as exc:
            print(f"[ERROR] Unexpected error in population generation: {exc}")
            print(f"[ERROR] Exception type: {type(exc)}")
            import traceback
            traceback.print_exc()
            fallback_result = self._fallback_personas(n)
            print(f"[DEBUG] Fallback generated {len(fallback_result)} personas")
            return fallback_result

    def _fallback_personas(self, n: int) -> List[Dict[str, Any]]:
        """Return a simple offline population when the LLM call fails."""
        print(f"[DEBUG] Generating {n} fallback personas...")
        
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
        for i in range(n):
            name = random.choice(names)
            age = random.randint(25, 70)
            occ = random.choice(occupations)
            persona = {
                "name": name,
                "personality": random.choice(personalities),
                "age": age,
                "occupation": occ,
                "initial_goals": random.choice(goals),
                "memory_summary": random.choice(memory),
            }
            print(f"[DEBUG] Fallback persona {i+1}: {name}, {age}, {occ}")
            result.append(persona)
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
