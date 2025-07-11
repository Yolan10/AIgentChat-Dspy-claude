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
        
        # Initial greeting from population agent
        initial_msg = f"Hello, I'm {pop_agent.name}. I've been dealing with hearing loss and I'm interested in participating in your research study."
        log["turns"].append({"speaker": "pop", "text": initial_msg, "time": utils.get_timestamp()})
        if show_live:
            print(f"\n{pop_agent.name}: {initial_msg}")
        
        # Store initial message in agent's history
        pop_agent.history.append(("pop", initial_msg))
        
        # Conduct conversation
        for turn_num in range(config.MAX_TURNS):
            # Build message history for wizard
            messages = [SystemMessage(content=self.current_prompt)]
            for t in log["turns"]:
                if t["speaker"] == "wizard":
                    messages.append(AIMessage(content=t["text"]))
                else:
                    messages.append(HumanMessage(content=t["text"]))

            # Wizard responds
            wizard_msg = self.llm.invoke(messages).content
            log["turns"].append({"speaker": "wizard", "text": wizard_msg, "time": utils.get_timestamp()})
            if show_live:
                print(f"\nWizard: {wizard_msg}")
            
            # Check if conversation should end before getting population response
            if self._check_conversation_complete(wizard_msg, ""):
                if show_live:
                    print(f"\n[Conversation ended - Research plan complete]")
                break
            
            # Population agent responds
            pop_reply = pop_agent.respond_to(wizard_msg)
            log["turns"].append({"speaker": "pop", "text": pop_reply, "time": utils.get_timestamp()})
            if show_live:
                print(f"\n{pop_agent.name}: {pop_reply}")

            # Check if conversation should end after population response
            if self._check_conversation_complete(wizard_msg, pop_reply):
                if show_live:
                    print(f"\n[Conversation ended naturally]")
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
        """Train an improver on the conversation history WITH judge feedback using enhanced MIPROv2 system."""
        if dspy is None:
            print(f"{self.wizard_id}: DSPy not available for improvement")
            return

        # Only use conversations that have judge feedback
        judged_logs = [log for log in self.history_buffer if 'judge_result' in log]
        if not judged_logs:
            print(f"{self.wizard_id}: Cannot improve without judge feedback")
            return

        print(f"\n[WIZARD] Starting enhanced self-improvement with {len(judged_logs)} judged conversations...")
        print(f"[WIZARD] Total conversations in buffer: {len(self.history_buffer)}")
        print(f"[WIZARD] Conversations with judge results: {len(judged_logs)}")
        
        # Import the enhanced functions with template paths
        from wizard_improver import build_dataset, train_improver, analyze_performance_issues
        
        # Analyze current performance issues using template
        performance_issues = analyze_performance_issues(
            judged_logs, 
            template_path=getattr(config, 'PERFORMANCE_ANALYSIS_TEMPLATE_PATH', 'templates/performance_analysis_template.txt')
        )
        print(f"[WIZARD] Performance analysis:\n{performance_issues}")
        
        # Build enhanced dataset (may include synthetic data) using template settings
        dataset, used_synthetic = build_dataset(
            judged_logs, 
            min_size=8,
            settings_path=getattr(config, 'IMPROVEMENT_PROMPTS_TEMPLATE_PATH', 'templates/improvement_prompts.json')
        )
        
        print(f"[WIZARD] Dataset composition:")
        print(f"  - Real judged conversations: {len(judged_logs)}")
        print(f"  - Total dataset size: {len(dataset)}")
        print(f"  - Synthetic examples added: {len(dataset) - len(judged_logs) if used_synthetic else 0}")
        
        # Train the improved wizard using template settings
        improver, metrics = train_improver(
            dataset, 
            self.current_prompt,
            settings_path=getattr(config, 'IMPROVEMENT_PROMPTS_TEMPLATE_PATH', 'templates/improvement_prompts.json')
        )
        
        if metrics.get("training_successful", False):
            # Extract the improved prompt
            improved_prompt = metrics.get("best_prompt", "")
            
            if improved_prompt and len(improved_prompt.strip()) > 100:
                # Test the improved prompt with one of the examples
                print(f"[WIZARD] Testing improved prompt...")
                
                try:
                    # Use the first example for testing
                    test_example = dataset[0] if dataset else None
                    if test_example:
                        result = improver(
                            current_prompt=self.current_prompt,
                            conversation_examples=test_example.conversation_examples,
                            goal=self.goal,
                            performance_issues=performance_issues
                        )
                        
                        if hasattr(result, 'improved_prompt') and result.improved_prompt:
                            self.current_prompt = result.improved_prompt
                            print(f"[WIZARD] Successfully applied improved prompt")
                        else:
                            print(f"[WIZARD] Using best prompt from training metrics")
                            self.current_prompt = improved_prompt
                    else:
                        self.current_prompt = improved_prompt
                        
                except Exception as e:
                    print(f"[WIZARD] Error testing improved prompt: {e}, using training result")
                    self.current_prompt = improved_prompt
            else:
                print(f"[WIZARD] Improved prompt too short or empty, keeping current prompt")
        else:
            print(f"[WIZARD] Training failed, keeping current prompt")

        # Log the improvement
        improvement_log = {
            "wizard_id": self.wizard_id,
            "run_no": self.current_run_no,
            "conversation_count": self.conversation_count,
            "old_prompt": self.system_prompt_template,
            "new_prompt": self.current_prompt,
            "performance_issues": performance_issues,
            "metrics": metrics,
            "dataset_info": {
                "real_examples": len(judged_logs),
                "total_examples": len(dataset),
                "used_synthetic": used_synthetic,
                "history_buffer_size": len(self.history_buffer),
                "judged_in_buffer": len(judged_logs)
            },
            "timestamp": utils.get_timestamp()
        }
        
        log_path = f"improve_{self.wizard_id}_{utils.get_timestamp().replace(':', '').replace('-', '')}.json"
        utils.save_conversation_log(improvement_log, log_path)
        
        print(f"[WIZARD] Improvement completed!")
        print(f"[WIZARD] Method: {metrics.get('method', 'Unknown')}")
        print(f"[WIZARD] Dataset size: {len(dataset)} examples")
        print(f"[WIZARD] Best score: {metrics.get('best_score', 0):.3f}")
        print(f"[WIZARD] Improvement log saved: {log_path}")
        
        # Update tracking
        utils.append_improvement_log(
            self.current_run_no,
            self.current_prompt,
            metrics.get("method"),
            conv_no=self.conversation_count,
            dataset_size=len(dataset),
        )

        self.last_improvement = self.conversation_count
        self.improved_last_conversation = True

        # IMPORTANT: Don't clear history buffer aggressively
        # This ensures we accumulate more examples for better optimization
        print(f"[WIZARD] Keeping full history buffer ({len(self.history_buffer)} conversations) for next improvement")
