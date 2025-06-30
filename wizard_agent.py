"""WizardAgent interacts with population agents WITHOUT judging."""
from __future__ import annotations

from collections import deque
from typing import Any, Dict, Deque, List, TypedDict

from tracking_chat_openai import TrackingChatOpenAI as ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

import config
import utils
from wizard_improver import build_dataset, train_improver

# Dspy is imported as placeholder - this code assumes Dspy provides a simple API
# to fine tune prompts. Replace with actual implementation when available.
try:
    import dspy
except ImportError:  # pragma: no cover - dspy not installed
    dspy = None


class ConversationLog(TypedDict, total=False):
    """Structure of a conversation log entry."""

    wizard_id: str
    pop_agent_id: str
    pop_agent_spec: Dict[str, Any]
    goal: str
    prompt: str
    turns: List[Dict[str, str]]
    timestamp: str
    prompt_improved: bool
    judge_result: Dict[str, Any]


class WizardAgent:
    def __init__(self, wizard_id: str, goal: str | None = None, llm_settings: dict | None = None):
        self.wizard_id = wizard_id
        self.goal = goal or config.WIZARD_DEFAULT_GOAL
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
        # Use the research wizard prompt template instead of the basic wizard prompt
        self.system_prompt_template = utils.load_template(config.WIZARD_PROMPT_TEMPLATE_PATH)
        self.current_prompt = self.system_prompt_template  # The research prompt is already complete
        self.conversation_count = 0
        self.history_buffer: Deque[ConversationLog] = deque(maxlen=config.HISTORY_BUFFER_LIMIT)
        self.current_run_no = 0
        self.last_improvement = 0
        self.improved_last_conversation = False
        # REMOVED: No judge initialization here

    def set_run(self, run_no: int) -> None:
        """Record the current run number for logging."""
        self.current_run_no = run_no

    def converse_with(self, pop_agent, show_live: bool = False) -> ConversationLog:
        """Converse with population agent and return conversation log WITHOUT judging."""
        improved = False
        log = {
            "wizard_id": self.wizard_id,
            "pop_agent_id": pop_agent.agent_id,
            "pop_agent_spec": pop_agent.get_spec(),
            "goal": self.goal,
            "prompt": self.current_prompt,
            "turns": [],
            "timestamp": utils.get_timestamp(),
        }
        
        # Conduct conversation
        for turn_num in range(config.MAX_TURNS):
            messages = [SystemMessage(content=self.current_prompt)]
            for t in log["turns"]:
                if t["speaker"] == "wizard":
                    messages.append(AIMessage(content=t["text"]))
                else:
                    messages.append(HumanMessage(content=t["text"]))

            # If this is the first turn, add an initial user message to start the conversation
            if turn_num == 0:
                initial_msg = f"Hello, I'm {pop_agent.name}. I've been dealing with hearing loss and I'm interested in participating in your research study."
                messages.append(HumanMessage(content=initial_msg))
                log["turns"].append({"speaker": "pop", "text": initial_msg, "time": utils.get_timestamp()})
                if show_live:
                    print(f"\n{pop_agent.name}: {initial_msg}")

            wizard_msg = self.llm.invoke(messages).content
            log["turns"].append({"speaker": "wizard", "text": wizard_msg, "time": utils.get_timestamp()})
            if show_live:
                print(f"\nWizard: {wizard_msg}")
            
            pop_reply = pop_agent.respond_to(wizard_msg)
            log["turns"].append({"speaker": "pop", "text": pop_reply, "time": utils.get_timestamp()})
            if show_live:
                print(f"\n{pop_agent.name}: {pop_reply}")

            # Check if conversation should end (e.g., if research plan is complete)
            if self._check_conversation_complete(wizard_msg, pop_reply):
                break
        
        # Store conversation WITHOUT judge result (deque handles max size)
        self.history_buffer.append(log)
        
        self.conversation_count += 1
        
        # Check for self-improvement (will need judge results added externally)
        if self._should_self_improve():
            # Only improve if we have judge results in history
            if any('judge_result' in log for log in self.history_buffer):
                self.self_improve()
                improved = True
        
        self.improved_last_conversation = improved
        log["prompt_improved"] = improved
        
        # Return log WITHOUT judge result - that will be added externally
        return log

    def add_judge_feedback(self, conversation_id: str, judge_result: Dict) -> None:
        """Add judge feedback to a previous conversation for learning purposes."""
        # Find the conversation in history buffer
        for log in self.history_buffer:
            if log.get("pop_agent_id") == conversation_id:
                log["judge_result"] = judge_result
                break
        
        # Also update the score tracking
        utils.append_wizard_score(
            self.current_run_no,
            self.conversation_count,
            judge_result.get("overall", judge_result.get("score", 0)),
            self.improved_last_conversation,
        )

    def _check_conversation_complete(self, wizard_text: str, pop_text: str) -> bool:
        """Check if the conversation has reached a natural conclusion."""
        # Look for signs that a research plan has been created
        completion_phrases = [
            "research plan",
            "thank you for participating",
            "we've covered everything",
            "that concludes",
            "interview is complete",
            "JSON",
            "research_questions"
        ]
        
        wizard_lower = wizard_text.lower()
        pop_lower = pop_text.lower()
        
        # Check if wizard has produced a research plan (JSON structure)
        if "{" in wizard_text and "research_questions" in wizard_lower:
            return True
            
        # Check for natural conversation endings
        for phrase in completion_phrases:
            if phrase in wizard_lower:
                return True
                
        # Check if participant indicates they're done
        if "goodbye" in pop_lower or "thank you" in pop_lower and "bye" in pop_lower:
            return True
            
        return False

    def _should_self_improve(self) -> bool:
        """Determine whether to run the improver based on the schedule."""
        schedule = config.SELF_IMPROVE_AFTER
        if isinstance(schedule, int):
            return schedule > 0 and self.conversation_count % schedule == 0
        if isinstance(schedule, str):
            schedule = [s for s in schedule.split(";") if s.strip()]
        try:
            points = {int(x) for x in schedule}
        except (TypeError, ValueError):
            return False
        return self.conversation_count in points

    def self_improve(self) -> None:
        """Train an improver on the conversation history WITH judge feedback."""
        if dspy is None:
            return

        # Only use conversations that have judge feedback
        judged_logs = [log for log in self.history_buffer if 'judge_result' in log]
        if not judged_logs:
            print(f"{self.wizard_id}: Cannot improve without judge feedback")
            return

        dataset = build_dataset(judged_logs)
        improver, metrics = train_improver(dataset)

        logs_example = dataset[-1].logs if dataset else ""
        result = improver(instruction=self.current_prompt, logs=logs_example, goal=self.goal)
        self.current_prompt = getattr(result, "improved_prompt", self.current_prompt)
        improver_instructions = metrics.get("best_prompt") or improver.agent.signature.instructions
        utils.append_improver_instruction_log(self.current_run_no, improver_instructions)

        log_path = f"improve_{utils.get_timestamp().replace(':', '').replace('-', '')}.json"
        utils.save_conversation_log(
            {"prompt": self.current_prompt, "metrics": metrics}, log_path
        )
        print(f"Wizard improved prompt saved to {log_path}")
        utils.append_improvement_log(
            self.current_run_no,
            self.current_prompt,
            metrics.get("method"),
            conv_no=self.conversation_count,
            dataset_size=len(dataset),
        )

        self.last_improvement = self.conversation_count
        self.improved_last_conversation = True

        self.history_buffer.clear()
