"""Module for improving wizard prompts using DSPy optimizers."""
from __future__ import annotations

from typing import List
import re

import config
import utils

try:
    import dspy
    from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2 as OptimizePrompts
except Exception:  # pragma: no cover - DSPy optional
    dspy = None


if dspy is not None:

    class ImproveSignature(dspy.Signature):
        """Signature for generating a better system prompt."""

        instruction: str = dspy.InputField()
        logs: str = dspy.InputField()
        goal: str = dspy.InputField()
        improved_prompt: str = dspy.OutputField()


    class WizardImprover(dspy.Module):
        """Wraps a ReAct agent that proposes improved prompts."""

        def __init__(self) -> None:
            super().__init__()
            self.agent = dspy.ReAct(ImproveSignature, tools=[])

        def forward(self, instruction: str, logs: str, goal: str) -> dspy.Prediction:
            return self.agent(instruction=instruction, logs=logs, goal=goal)


    def _extract_instructions(program: object) -> str:
        """Return the instructions string from a candidate program.

        ``program`` may be a raw source string or a compiled ``dspy`` module.
        When a module is provided we attempt to read the ``signature.instructions``
        attribute, falling back to regex extraction from the string
        representation of ``program``.
        """

        # If ``program`` is a compiled module, try grabbing the instructions
        sig = getattr(program, "signature", None)
        if sig is not None and hasattr(sig, "instructions"):
            text = sig.instructions

            if isinstance(text, str) and text.strip():
                cleaned = text.strip()
                if cleaned == "Signature for generating a better system prompt.":
                    return utils.load_template(config.WIZARD_PROMPT_TEMPLATE_PATH).strip()
                return cleaned


        # Otherwise handle the value as a plain string
        if not isinstance(program, str):
            program = str(program)

        match = re.search(
            r"instructions=(\"\"\".*?\"\"\"|\".*?\"|'.*?')",
            program,
            re.DOTALL,
        )
        if not match:
            return ""
        text = match.group(1)
        if text.startswith('"""') and text.endswith('"""'):
            return text[3:-3].strip()
        if text.startswith('"') and text.endswith('"'):
            return text[1:-1].strip()
        if text.startswith("'") and text.endswith("'"):
            return text[1:-1].strip()
        return text.strip()


    def build_dataset(history: List[dict]) -> List[dspy.Example]:
        """Convert conversation history into a DSPy dataset."""
        dataset = []
        for log in history:
            # Only use conversations that have judge feedback
            if 'judge_result' not in log:
                continue
                
            transcript = "\n".join(f"{t['speaker']}: {t['text']}" for t in log.get('turns', []))
            judge = log.get("judge_result", {})
            score = judge.get("overall", judge.get("score", 0))
            
            # Include judge feedback in the logs for improvement
            judge_feedback = f"\n\nJUDGE EVALUATION:\n"
            judge_feedback += f"- Goal Completion: {judge.get('goal_completion', 0):.2f}\n"
            judge_feedback += f"- Coherence: {judge.get('coherence', 0):.2f}\n"
            judge_feedback += f"- Tone: {judge.get('tone', 0):.2f}\n"
            judge_feedback += f"- Overall Score: {score:.2f}\n"
            judge_feedback += f"- Success: {judge.get('success', False)}\n"
            judge_feedback += f"- Rationale: {judge.get('rationale', 'No rationale provided')}"
            
            ex = (
                dspy.Example(
                    instruction=log.get("prompt", ""),
                    logs=f"{transcript}{judge_feedback}",
                    goal=log.get("goal"),
                    score=score,
                )
                .with_inputs("instruction", "logs", "goal")
            )
            dataset.append(ex)
            
        return dataset


    def train_improver(dataset: List[dspy.Example]) -> tuple[WizardImprover, dict]:
        """Train a WizardImprover on the dataset."""

        if dspy.settings.lm is None:
            dspy.settings.configure(
                lm=dspy.LM(
                    model=config.LLM_MODEL,
                    temperature=config.LLM_TEMPERATURE,
                    max_tokens=config.LLM_MAX_TOKENS,
                )
            )

        def metric(
            example: dspy.Example, pred: dspy.Prediction, trace: object | None = None
        ) -> float:
            """Score an improved prompt.

            ``BootstrapFewShot`` and other teleprompters call ``metric`` with
            ``example``, ``prediction`` and a ``trace`` object. The ``trace`` is
            optional for this implementation so we default it to ``None`` in
            order to remain compatible with optimizers that supply it.
            """

            base = example.score or 0
            
            # Bonus for including key research planning elements
            research_keywords = [
                "research", "plan", "questions", "interview", "hearing loss",
                "COM-B", "TDF", "theoretical", "framework", "topic"
            ]
            keyword_bonus = sum(0.05 for keyword in research_keywords if keyword.lower() in pred.improved_prompt.lower())
            
            # Cap the bonus
            keyword_bonus = min(keyword_bonus, 0.3)
            
            return base + keyword_bonus

        improver_module = WizardImprover()
        if len(dataset) >= config.DSPY_MIPRO_MINIBATCH_SIZE:
            optimizer = OptimizePrompts(
                metric=metric,
                num_candidates=4,
                auto=None,
                verbose=False,
            )
            # DSPy derives the validation set as 80% of the trainset. When the
            # dataset is small this can make the validation size smaller than
            # ``DSPY_MIPRO_MINIBATCH_SIZE`` which causes ``compile`` to raise a
            # ``ValueError``. Estimate the resulting validation size and cap the
            # minibatch accordingly so it never exceeds the validation set size.
            valset_size = int(len(dataset) * 0.8)
            minibatch = min(config.DSPY_MIPRO_MINIBATCH_SIZE, valset_size)

            method = "MIPROv2"
            try:
                trained = optimizer.compile(
                    improver_module,
                    trainset=dataset,
                    num_trials=config.DSPY_TRAINING_ITER,
                    provide_traceback=True,
                    minibatch_size=minibatch,
                )
            except dspy.utils.exceptions.AdapterParseError as e:
                print(f"Warning: {e}")
                trained = improver_module
                metrics = {
                    "best_score": 0,
                    "iterations": [],
                    "best_prompt": improver_module.agent.signature.instructions,
                    "method": method,
                    "error": str(e),
                }
                return trained, metrics
        elif len(dataset) >= config.DSPY_BOOTSTRAP_MINIBATCH_SIZE:
            optimizer = dspy.teleprompt.BootstrapFewShot(metric=metric)
            method = "BootstrapFewShot"
            try:
                trained = optimizer.compile(
                    improver_module,
                    trainset=dataset,
                )
            except dspy.utils.exceptions.AdapterParseError as e:
                print(f"Warning: {e}")
                trained = improver_module
                metrics = {
                    "best_score": 0,
                    "iterations": [],
                    "best_prompt": improver_module.agent.signature.instructions,
                    "method": method,
                    "error": str(e),
                }
                return trained, metrics
        else:
            optimizer = dspy.COPRO(metric=metric)
            method = "COPRO"
            try:
                trained = optimizer.compile(
                    improver_module,
                    trainset=dataset,
                    eval_kwargs={"display_progress": False},
                )
            except dspy.utils.exceptions.AdapterParseError as e:
                print(f"Warning: {e}")
                trained = improver_module
                metrics = {
                    "best_score": 0,
                    "iterations": [],
                    "best_prompt": improver_module.agent.signature.instructions,
                    "method": method,
                    "error": str(e),
                }
                return trained, metrics

        candidates = getattr(trained, "candidate_programs", [])
        if candidates:
            best = max(candidates, key=lambda c: c.get("score", 0))
            best_prompt = _extract_instructions(best.get("program", ""))
            best_score = best.get("score", 0)
        else:
            best_prompt = _extract_instructions(trained)
            best_score = 0
        metrics = {
            "best_score": best_score,
            "iterations": candidates,
            "best_prompt": best_prompt,
            "method": method,
        }
        return trained, metrics

else:  # DSPy not available
    WizardImprover = None
    build_dataset = None
    train_improver = None
