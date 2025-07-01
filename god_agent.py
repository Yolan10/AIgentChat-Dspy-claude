"""GodAgent spawns population agents."""
from __future__ import annotations

import json
from typing import List

from tracking_chat_openai import TrackingChatOpenAI as ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

import config
import utils
from population_agent import PopulationAgent


class GodAgent:
    def __init__(self, llm_settings: dict | None = None):
        self.llm_settings = llm_settings or {
            "model": config.LLM_MODEL,
            "temperature": config.LLM_TEMPERATURE,
            "max_tokens": config.LLM_MAX_TOKENS,
        }
        self.llm = ChatOpenAI(
            model=self.llm_settings["model"],
            temperature=self.llm_settings["temperature"],
            max_tokens=self.llm_settings["max_tokens"],
            max_retries=config.OPENAI_MAX_RETRIES,
        )
        self.template = utils.load_template(config.POPULATION_INSTRUCTION_TEMPLATE_PATH)

    def spawn_population(
        self,
        instruction_text: str,
        n: int | None = None,
        run_no: int = 0,
        start_index: int = 1,
    ) -> List[PopulationAgent]:
        n = n or config.POPULATION_SIZE
        prompt = utils.render_template(self.template, {"instruction": instruction_text, "n": n})
        messages = [SystemMessage(content=prompt), HumanMessage(content="Provide the JSON array only.")]
        
        try:
            response = self.llm.invoke(messages)
            response_content = response.content
            
            # Debug logging
            print(f"LLM Response received: {response_content[:200]}...")  # First 200 chars
            
            if not response_content or not response_content.strip():
                raise ValueError("Empty response from LLM")
                
        except Exception as e:
            print(f"Error invoking LLM: {str(e)}")
            # Use fallback population if LLM fails
            return self._create_fallback_population(n, run_no, start_index)
        
        try:
            personas = json.loads(response_content)
        except json.JSONDecodeError as e:
            print(f"JSON decode error: {str(e)}")
            print(f"Attempting to extract JSON from response...")
            personas = utils.extract_json_array(response_content)
            if personas is None:
                print("Failed to extract JSON array, using fallback population")
                return self._create_fallback_population(n, run_no, start_index)
                
        population = []
        for idx, spec in enumerate(personas[:n], start=start_index):  # Limit to requested size
            if isinstance(spec, str):
                try:
                    spec = json.loads(spec)
                except json.JSONDecodeError:
                    print(f"Invalid persona spec: {spec}")
                    continue
                    
            # Validate required fields
            required_fields = ["name", "personality", "age", "occupation", "initial_goals", "memory_summary"]
            if not all(field in spec for field in required_fields):
                print(f"Missing required fields in spec: {spec}")
                continue
                
            agent = PopulationAgent(
                agent_id=utils.format_agent_id(run_no, idx),
                name=spec.get("name"),
                personality_description=spec.get("personality"),
                age=spec.get("age"),
                occupation=spec.get("occupation"),
                initial_goals=spec.get("initial_goals"),
                memory_summary=spec.get("memory_summary"),
                llm_settings=self.llm_settings,
            )
            population.append(agent)

            # Save the agent specification immediately so users can inspect it
            log_filename = f"{agent.agent_id}_spec_{utils.get_timestamp().replace(':', '').replace('-', '')}.json"
            utils.save_conversation_log(agent.get_spec(), log_filename)
            print(f"Created {agent.agent_id} -> {log_filename}")

        # If we couldn't create enough agents, fill with fallback
        if len(population) < n:
            print(f"Only created {len(population)} agents, filling remaining with fallback")
            fallback_start = len(population) + start_index
            fallback_agents = self._create_fallback_population(
                n - len(population), run_no, fallback_start
            )
            population.extend(fallback_agents)
            
        return population

    def _create_fallback_population(self, n: int, run_no: int, start_index: int) -> List[PopulationAgent]:
        """Create a fallback population when LLM fails."""
        print("Creating fallback population...")
        population = []
        
        fallback_personas = [
            {
                "name": "Alice",
                "personality": "O:0.7 C:0.8 E:0.6 A:0.7 N:0.3",
                "age": 45,
                "occupation": "teacher",
                "initial_goals": "improve classroom communication",
                "memory_summary": "noticed hearing difficulties in noisy environments"
            },
            {
                "name": "Bob",
                "personality": "O:0.5 C:0.6 E:0.4 A:0.5 N:0.5",
                "age": 62,
                "occupation": "retired engineer",
                "initial_goals": "find better hearing aids",
                "memory_summary": "has worn hearing aids for 5 years"
            },
            {
                "name": "Carol",
                "personality": "O:0.8 C:0.7 E:0.8 A:0.8 N:0.2",
                "age": 38,
                "occupation": "marketing manager",
                "initial_goals": "manage hearing loss at work",
                "memory_summary": "recently diagnosed with mild hearing loss"
            }
        ]
        
        for idx in range(start_index, start_index + n):
            spec = fallback_personas[(idx - start_index) % len(fallback_personas)].copy()
            # Modify name to make unique
            spec["name"] = f"{spec['name']}_{idx}"
            
            agent = PopulationAgent(
                agent_id=utils.format_agent_id(run_no, idx),
                name=spec["name"],
                personality_description=spec["personality"],
                age=spec["age"],
                occupation=spec["occupation"],
                initial_goals=spec["initial_goals"],
                memory_summary=spec["memory_summary"],
                llm_settings=self.llm_settings,
            )
            population.append(agent)
            
            # Save the agent specification
            log_filename = f"{agent.agent_id}_spec_{utils.get_timestamp().replace(':', '').replace('-', '')}.json"
            utils.save_conversation_log(agent.get_spec(), log_filename)
            print(f"Created fallback {agent.agent_id} -> {log_filename}")
            
        return population

    def spawn_population_from_spec(
        self, spec: dict | str, run_no: int, index: int
    ) -> PopulationAgent:
        """Create a single ``PopulationAgent`` from a specification.

        Parameters
        ----------
        spec : dict | str
            The persona specification. If provided as a JSON string it will be
            parsed into a dictionary.
        run_no : int
            Current run number used when formatting the ``agent_id``.
        index : int
            Index of this agent within the run.

        Returns
        -------
        PopulationAgent
            The instantiated population agent.
        """

        if isinstance(spec, str):
            try:
                spec = json.loads(spec)
            except json.JSONDecodeError:
                raise ValueError(f"Invalid persona spec: {spec}")


        agent = PopulationAgent(
            agent_id=utils.format_agent_id(run_no, index),
            name=spec.get("name"),
            personality_description=spec.get("personality")
            or spec.get("personality_description"),

            age=spec.get("age"),
            occupation=spec.get("occupation"),
            initial_goals=spec.get("initial_goals"),
            memory_summary=spec.get("memory_summary"),
            llm_settings=self.llm_settings,
        )

        log_filename = (
            f"{agent.agent_id}_spec_{utils.get_timestamp().replace(':', '').replace('-', '')}.json"
        )
        utils.save_conversation_log(agent.get_spec(), log_filename)
        print(f"Created {agent.agent_id} -> {log_filename}")

        return agent
