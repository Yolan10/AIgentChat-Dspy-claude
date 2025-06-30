import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import population_agent

class DummyChat:
    def __init__(self, *args, **kwargs):
        pass

@pytest.fixture(autouse=True)
def patch_chat(monkeypatch):
    monkeypatch.setattr(population_agent, "ChatOpenAI", DummyChat)


def make_agent(age, occupation):
    return population_agent.PopulationAgent(
        agent_id="1",
        name="Alice",
        personality_description="kind",
        age=age,
        occupation=occupation,
        initial_goals="goal",
        memory_summary="memo",
        llm_settings={"model": "x"},
    )


def test_instruction_full():
    agent = make_agent(30, "teacher")
    assert agent.system_instruction.startswith("You are Alice, a 30-year-old teacher.")


def test_instruction_missing_age():
    agent = make_agent(None, "teacher")
    assert "30-year-old" not in agent.system_instruction
    assert agent.system_instruction.startswith("You are Alice, a teacher.")


def test_instruction_missing_occupation():
    agent = make_agent(30, None)
    assert agent.system_instruction.startswith("You are Alice, 30 years old.")


def test_instruction_no_demo():
    agent = make_agent(None, None)
    assert agent.system_instruction.startswith("You are Alice.")
