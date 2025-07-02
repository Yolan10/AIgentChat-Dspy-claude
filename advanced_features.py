"""Collection of placeholder classes implementing advanced behaviours with improved debugging."""
from __future__ import annotations

import json
import re
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
            
            # Create messages with clearer instructions
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content="Describe each persona using the fields: name, personality, age, occupation, initial_goals, and memory_summary."),
            ]
            print(f"[DEBUG] Created {len(messages)} messages")
            
            # Make LLM call
            print(f"[DEBUG] Making LLM call...")
            response = llm.invoke(messages).content
            print(f"[DEBUG] LLM response received, length: {len(response)} chars")
            print(f"[DEBUG] Full response:\n{response}\n")
            
            # Parse JSON with multiple strategies
            personas = self._parse_json_response(response)

            if not personas:
                print("[DEBUG] JSON parsing failed, trying text parsing")
                personas = self._parse_text_response(response)

            if not personas:
                print(f"[DEBUG] All parsing strategies failed, using fallback")
                return self._fallback_personas(n)
            
            # Process and validate personas
            result: List[Dict[str, Any]] = []
            print(f"[DEBUG] Processing {len(personas)} personas...")
            
            for i, spec in enumerate(personas):
                print(f"[DEBUG] Processing persona {i+1}: type={type(spec)}")
                
                if isinstance(spec, dict) and self._is_valid_persona(spec):
                    print(f"[DEBUG] Adding valid persona {i+1}: {spec.get('name', 'Unknown')}")
                    result.append(spec)
                else:
                    print(f"[DEBUG] Invalid persona {i+1}, skipping")

            if not result:
                print(f"[DEBUG] No valid personas found, using fallback")
                return self._fallback_personas(n)
                
            print(f"[DEBUG] Final result: {len(result)} personas")
            return result
            
        except (OpenAIError, RequestException, json.JSONDecodeError) as exc:
            print(f"[ERROR] Population generation failed: {exc}. Using random fallback.")
            print(f"[ERROR] Exception type: {type(exc)}")
            return self._fallback_personas(n)
        except Exception as exc:
            print(f"[ERROR] Unexpected error in population generation: {exc}")
            print(f"[ERROR] Exception type: {type(exc)}")
            import traceback
            traceback.print_exc()
            return self._fallback_personas(n)

    def _parse_json_response(self, response: str) -> List[Dict[str, Any]]:
        """Try multiple strategies to parse JSON from LLM response."""
        # Strategy 1: Direct JSON parsing
        try:
            print(f"[DEBUG] Strategy 1: Direct JSON parsing...")
            personas = json.loads(response)
            if isinstance(personas, list):
                print(f"[DEBUG] Strategy 1 succeeded: found {len(personas)} items")
                return personas
        except json.JSONDecodeError as e:
            print(f"[DEBUG] Strategy 1 failed: {e}")
        
        # Strategy 2: Extract JSON array using regex
        print(f"[DEBUG] Strategy 2: Regex extraction...")
        json_match = re.search(r'\[\s*\{.*?\}\s*\]', response, re.DOTALL)
        if json_match:
            try:
                json_str = json_match.group(0)
                personas = json.loads(json_str)
                print(f"[DEBUG] Strategy 2 succeeded: found {len(personas)} items")
                return personas
            except json.JSONDecodeError as e:
                print(f"[DEBUG] Strategy 2 failed: {e}")
        
        # Strategy 3: Find JSON between ```json markers
        print(f"[DEBUG] Strategy 3: Extract from code blocks...")
        code_block_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
        if code_block_match:
            try:
                json_str = code_block_match.group(1)
                personas = json.loads(json_str)
                print(f"[DEBUG] Strategy 3 succeeded: found {len(personas)} items")
                return personas
            except json.JSONDecodeError as e:
                print(f"[DEBUG] Strategy 3 failed: {e}")
        
        # Strategy 4: Use utils.extract_json_array
        print(f"[DEBUG] Strategy 4: Using utils.extract_json_array...")
        personas = utils.extract_json_array(response)
        if personas:
            print(f"[DEBUG] Strategy 4 succeeded: found {len(personas)} items")
            return personas
        
        # Strategy 5: Clean common JSON issues and retry
        print(f"[DEBUG] Strategy 5: Cleaning and retrying...")
        cleaned = response.strip()
        # Remove any text before the first [
        start_idx = cleaned.find('[')
        if start_idx > 0:
            cleaned = cleaned[start_idx:]
        # Remove any text after the last ]
        end_idx = cleaned.rfind(']')
        if end_idx > 0 and end_idx < len(cleaned) - 1:
            cleaned = cleaned[:end_idx + 1]
        
        try:
            personas = json.loads(cleaned)
            if isinstance(personas, list):
                print(f"[DEBUG] Strategy 5 succeeded: found {len(personas)} items")
                return personas
        except json.JSONDecodeError as e:
            print(f"[DEBUG] Strategy 5 failed: {e}")
        
        print(f"[DEBUG] All parsing strategies failed")
        return []

    def _parse_text_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse persona specs from a free-form text response."""
        required = ["name", "personality", "age", "occupation", "initial_goals", "memory_summary"]
        blocks = re.split(r"\n\s*\n", response.strip())
        personas: List[Dict[str, Any]] = []
        for block in blocks:
            spec: Dict[str, Any] = {}
            for line in block.splitlines():
                if ":" not in line:
                    continue
                key, value = [part.strip() for part in line.split(":", 1)]
                key_lc = key.lower()
                if key_lc in required:
                    spec[key_lc] = value
            if all(field in spec for field in required):
                try:
                    spec["age"] = int(re.findall(r"\d+", spec["age"])[0])
                except Exception:
                    pass
                personas.append(spec)
        if personas:
            print(f"[DEBUG] Parsed {len(personas)} personas from text")
        return personas

    def _is_valid_persona(self, spec: Dict[str, Any]) -> bool:
        """Check if a persona specification has all required fields."""
        required_fields = ["name", "personality", "age", "occupation", "initial_goals", "memory_summary"]
        for field in required_fields:
            if field not in spec:
                print(f"[DEBUG] Missing required field: {field}")
                return False
        return True

    def _fallback_personas(self, n: int) -> List[Dict[str, Any]]:
        """Return a simple offline population when the LLM call fails."""
        print(f"[DEBUG] Generating {n} fallback personas...")
        
        names = [
            "Emma Carter", "Michael Thompson", "Sarah Williams", "David Johnson",
            "Linda Martinez", "Robert Davis", "Patricia Brown", "James Wilson",
            "Jennifer Garcia", "William Anderson", "Margaret Taylor", "Thomas White",
            "Dorothy Harris", "Charles Martin", "Helen Clark", "George Lewis",
            "Betty Walker", "Frank Hall", "Ruth Young", "Edward King"
        ]
        
        occupations = [
            "teacher", "engineer", "nurse", "accountant", "retail manager",
            "social worker", "construction worker", "administrative assistant",
            "sales representative", "customer service agent"
        ]
        
        goals = [
            "improve communication in noisy environments",
            "find better hearing aids within budget",
            "learn coping strategies for social situations",
            "connect with others who have hearing loss",
            "understand new hearing aid technologies"
        ]
        
        memory_summaries = [
            "struggled with hearing loss for 5 years, recently got hearing aids",
            "born with partial hearing loss, uses sign language occasionally",
            "developed hearing loss due to workplace noise exposure",
            "age-related hearing loss, hesitant about hearing aids",
            "sudden hearing loss after illness, adjusting to new reality"
        ]
        
        result = []
        used_names = set()
        
        for i in range(n):
            # Ensure unique names
            available_names = [name for name in names if name not in used_names]
            if not available_names:
                available_names = names  # Reset if we run out
                used_names.clear()
            
            name = random.choice(available_names)
            used_names.add(name)
            
            age = random.randint(25, 75)
            
            # Generate personality scores
            personality = {
                "openness": round(random.uniform(0.3, 0.9), 1),
                "conscientiousness": round(random.uniform(0.3, 0.9), 1),
                "extraversion": round(random.uniform(0.2, 0.8), 1),
                "agreeableness": round(random.uniform(0.4, 0.9), 1),
                "neuroticism": round(random.uniform(0.2, 0.7), 1)
            }
            
            # Format personality as OCEAN string
            personality_str = f"O:{personality['openness']} C:{personality['conscientiousness']} E:{personality['extraversion']} A:{personality['agreeableness']} N:{personality['neuroticism']}"
            
            persona = {
                "name": name,
                "personality": personality_str,
                "age": age,
                "occupation": random.choice(occupations),
                "initial_goals": random.choice(goals),
                "memory_summary": random.choice(memory_summaries)
            }
            
            print(f"[DEBUG] Fallback persona {i+1}: {persona['name']}, {persona['age']}, {persona['occupation']}")
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
