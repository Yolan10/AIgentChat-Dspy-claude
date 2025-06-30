import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import god_agent
import population_agent

import utils

class DummyChat:
    def __init__(self, *args, **kwargs):
        pass

@pytest.fixture(autouse=True)

def patch_chat(monkeypatch):
    monkeypatch.setattr(god_agent, "ChatOpenAI", DummyChat)
    monkeypatch.setattr(population_agent, "ChatOpenAI", DummyChat)


def test_spawn_population_from_spec(monkeypatch):
    records = {}
    monkeypatch.setattr(utils, "get_timestamp", lambda: "2024-01-01T00:00:00+00:00")

    def fake_save(obj, filename):
        records["obj"] = obj
        records["filename"] = filename

    monkeypatch.setattr(utils, "save_conversation_log", fake_save)

    g = god_agent.GodAgent()
    spec = {
        "name": "Bob",
        "personality": "cheerful",
        "age": 25,
        "occupation": "engineer",
        "initial_goals": "sell",
        "memory_summary": "short",
    }
    agent = g.spawn_population_from_spec(spec, 5, 2)

    expected_id = "5.2_20240101T000000+0000"
    assert agent.agent_id == expected_id
    expected_file = f"{expected_id}_spec_20240101T000000+0000.json"
    assert records["filename"] == expected_file
    assert records["obj"] == agent.get_spec()
