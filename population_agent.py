"""Defines the PopulationAgent persona.

Each agent is described by a short Big Five (OCEAN) personality profile along
with demographic details and a memory summary.
"""
from __future__ import annotations

from typing import List, Tuple

from tracking_chat_openai import TrackingChatOpenAI as ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


import config
import utils


class PopulationAgent:
    """Simple persona-based agent using LangChain for replies.

    Parameters describing the persona include age, occupation, initial goals and
    a memory summary. ``personality_description`` should summarize the Big Five
    (OCEAN) traits for this agent. ``age`` and ``occupation`` are optional and
    omitted from the agent's system prompt when ``None``.
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        personality_description: str,
        age: int | None,
        occupation: str | None,
        initial_goals: str | None,
        memory_summary: str | None,
        llm_settings: dict,
    ):
        """Create a new population agent.

        Parameters
        ----------
        agent_id : str
            Unique identifier for this agent.
        name : str
            Agent's display name.
        personality_description : str
            Summary of the agent's Big Five (OCEAN) traits.
        age : int | None
            Age in years. Optional.
        occupation : str | None
            Primary job or role. Optional.
        initial_goals : str | None
            Starting objectives for the agent.
        memory_summary : str | None
            Short recap of relevant memories.
        llm_settings : dict
            Parameters controlling the underlying language model.
        
        Age or occupation will be omitted from the system prompt when passed
        as ``None``.
        """
        self.agent_id = agent_id
        self.name = name
        self.personality_description = personality_description
        self.age = age
        self.occupation = occupation
        self.initial_goals = initial_goals
        self.memory_summary = memory_summary
        self.llm_settings = llm_settings
        self.state = "undecided"
        self.history: List[Tuple[str, str]] = []  # (speaker, text)
        
        # Build system instruction
        parts = [f"You are {self.name}"]
        if self.age is not None and self.occupation is not None:
            parts.append(f", a {self.age}-year-old {self.occupation}")
        elif self.age is not None:
            parts.append(f", {self.age} years old")
        elif self.occupation is not None:
            parts.append(f", a {self.occupation}")
        parts.append(". ")
        parts.append(f"{self.personality_description}. ")
        parts.append(
            f"Your goals: {self.initial_goals}. Memory summary: {self.memory_summary}. "
        )
        parts.append("Respond accordingly.")
        self.system_instruction = "".join(parts)
        
        self.llm = ChatOpenAI(
            model=llm_settings.get("model", config.LLM_MODEL),
            temperature=llm_settings.get("temperature", config.LLM_TEMPERATURE),
            max_tokens=llm_settings.get("max_tokens", config.LLM_MAX_TOKENS),
            max_retries=config.OPENAI_MAX_RETRIES,
        )

    def respond_to(self, user_message: str) -> str:
        """Generate a response to the wizard's message.
        
        Parameters
        ----------
        user_message : str
            The wizard's message to respond to
            
        Returns
        -------
        str
            The population agent's response
        """
        # Build conversation history for LLM
        messages = [SystemMessage(content=self.system_instruction)]
        
        # Add conversation history
        for speaker, text in self.history:
            if speaker == "wizard":
                messages.append(HumanMessage(content=text))
            else:
                messages.append(AIMessage(content=text))
        
        # Add current wizard message
        messages.append(HumanMessage(content=user_message))
        
        # Generate response
        response = self.llm.invoke(messages).content

        # Update history
        self.history.append(("wizard", user_message))
        self.history.append(("pop", response))
        
        # Trim history if it gets too long
        if len(self.history) > config.POP_HISTORY_LIMIT:
            self.history = self.history[-config.POP_HISTORY_LIMIT:]
        
        return response

    def get_persona(self) -> dict:
        """Get basic persona information."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "personality_description": self.personality_description,
            "age": self.age,
            "occupation": self.occupation,
            "initial_goals": self.initial_goals,
            "memory_summary": self.memory_summary,
        }

    def get_spec(self) -> dict:
        """Return a spec dictionary describing this population agent."""
        return {
            "name": self.name,
            "personality_description": self.personality_description,
            "age": self.age,
            "occupation": self.occupation,
            "initial_goals": self.initial_goals,
            "memory_summary": self.memory_summary,
            "system_instruction": self.system_instruction,
            "llm_settings": self.llm_settings,
        }

    def reset_history(self) -> None:
        """Clear conversation history."""
        self.history = []
