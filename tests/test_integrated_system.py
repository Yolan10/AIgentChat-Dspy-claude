import os
import sys
import types
sys.modules.setdefault(
    "langchain_openai",
    types.SimpleNamespace(ChatOpenAI=type("Dummy", (), {"__init__": lambda self, *a, **k: None}))
)

import integrated_system
import config
import wizard_agent
import god_agent
import advanced_features
import utils
from token_tracker import token_tracker
from logging_system import StructuredLogger
from judge_agent import EnhancedJudgeAgent

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


class DummyThread:
    def __init__(self, target, args=()) -> None:
        self.target = target
        self.args = args
        self.started = False

    def start(self) -> None:
        events.append(f"start-{self.args[0].agent_id}")
        self.target(*self.args)
        self.started = True

    def join(self, timeout=None) -> None:  # noqa: D401 - mimics Thread API
        events.append(f"join-{self.args[0].agent_id}")


class DummyAgent:
    def __init__(self, agent_id: str) -> None:
        self.agent_id = agent_id
        self.name = agent_id

    def respond_to(self, _msg: str) -> str:
        return "ok"

    def get_spec(self) -> dict:
        return {
            "name": self.name,
            "personality_description": "",
            "system_instruction": "",
            "llm_settings": {},
        }


def fake_generate(self, _instruction, n):
    return [{} for _ in range(n)]


def fake_spawn(self, _spec, _run_no, index):
    return DummyAgent(str(index))

def fake_assess(self, log):
    """Fake judge assessment."""
    return {
        "goal_completion": 0.8,
        "coherence": 0.9,
        "tone": 0.85,
        "overall": 0.85,
        "success": True,
        "rationale": "Good conversation",
        "judge_id": "Judge_001"
    }


def fake_add_feedback(self, conversation_id, judge_result):
    """Fake adding judge feedback."""
    events.append(f"feedback-{conversation_id}")


def fake_converse(self, pop, show_live=False):
    events.append(f"conv-{pop.agent_id}")
    self.conversation_count += 1
    if self._should_self_improve():
        self.self_improve()
    return {
        "turns": [
            {"speaker": "wizard", "text": "Hello"},
            {"speaker": "pop", "text": "Hi"}
        ],
        "pop_agent_id": pop.agent_id,
        "goal": "test goal",
        "timestamp": "2024-01-01T00:00:00Z"
    }


def fake_improve(self):
    events.append(f"improve-{self.conversation_count}")


def setup(monkeypatch, start_when_spawned, schedule):
    monkeypatch.setattr(config, "PARALLEL_CONVERSATIONS", True)
    monkeypatch.setattr(config, "START_WHEN_SPAWNED", start_when_spawned)
    monkeypatch.setattr(config, "SELF_IMPROVE_AFTER", schedule)
    monkeypatch.setattr(utils, "increment_run_number", lambda: 1)
    monkeypatch.setattr(token_tracker, "set_run", lambda run_no: None)
    monkeypatch.setattr(utils, "save_conversation_log", lambda obj, fn: None)
    monkeypatch.setattr(StructuredLogger, "log_event", lambda self, *a, **k: None)
    monkeypatch.setattr(advanced_features.PopulationGenerator, "generate", fake_generate)
    monkeypatch.setattr(god_agent.GodAgent, "spawn_population_from_spec", fake_spawn)
    monkeypatch.setattr(wizard_agent.WizardAgent, "converse_with", fake_converse)
    monkeypatch.setattr(wizard_agent.WizardAgent, "self_improve", fake_improve)
    monkeypatch.setattr(integrated_system.threading, "Thread", DummyThread)
    monkeypatch.setattr(wizard_agent.WizardAgent, "add_judge_feedback", fake_add_feedback)

    # Mock the judge

    monkeypatch.setattr(EnhancedJudgeAgent, "assess", fake_assess)
    monkeypatch.setattr(
        EnhancedJudgeAgent,
        "__init__",
        lambda self, *args, **kwargs: setattr(self, "judge_id", kwargs.get("judge_id", "Judge_001")),
    )
    monkeypatch.setattr(
        EnhancedJudgeAgent,
        "get_performance_report",
        lambda self: {
            "judge_id": self.judge_id,
            "total_evaluations": 0,
            "performance_metrics": {
                "consistency": 1.0,
                "discrimination": 1.0,
                "calibration": 1.0,
                "detail_quality": 1.0,
                "overall": 1.0,
            },
            "improvement_history": [],
            "suggestions": [],
            "calibration_data_size": 0,
        },
    )


def test_batches_start_when_spawned(monkeypatch):
    global events
    events = []
    setup(monkeypatch, True, "1;3")

    system = integrated_system.IntegratedSystem()
    system.run("x", 4)

    assert "improve-1" in events and "improve-3" in events
    assert events.index("start-2") > events.index("join-1")
    assert events.index("start-4") > max(events.index("join-2"), events.index("join-3"))


def test_batches_queue_then_start(monkeypatch):
    global events
    events = []
    setup(monkeypatch, False, 2)

    system = integrated_system.IntegratedSystem()
    system.run("x", 4)

    assert "improve-2" in events and "improve-4" in events
    assert events.index("start-3") > events.index("join-2")



