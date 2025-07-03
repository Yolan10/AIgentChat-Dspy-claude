"""DSPy integration for improving judge prompts based on evaluation performance."""
from __future__ import annotations

from typing import List, Dict, Any, Tuple, Optional
import json
import numpy as np
from dataclasses import dataclass

import config
import utils


def _extract_score_value(score_data: Any) -> float:
    """Return a numeric score from various possible formats."""
    if isinstance(score_data, (int, float)):
        return float(score_data)
    if isinstance(score_data, dict):
        if 'score' in score_data:
            return float(score_data['score'])
        for key in ('value', 'rating', 'overall'):
            if key in score_data:
                return float(score_data[key])
    elif hasattr(score_data, 'score'):
        return float(score_data.score)
    return 0.0

try:
    import dspy
    from dspy.teleprompt import BootstrapFewShot, MIPROv2 as OptimizePrompts
except ImportError:
    dspy = None


if dspy is not None:
    
    class JudgeImproveSignature(dspy.Signature):
        """Signature for improving judge evaluation prompts."""
        
        current_prompt: str = dspy.InputField(desc="Current judge prompt template")
        evaluation_logs: str = dspy.InputField(desc="Logs showing judge performance and issues")
        criteria: str = dspy.InputField(desc="Evaluation criteria descriptions")
        improved_prompt: str = dspy.OutputField(desc="Improved judge prompt with better instructions")
    
    
    class JudgePromptImprover(dspy.Module):
        """DSPy module for improving judge prompts."""
        
        def __init__(self):
            super().__init__()
            self.improve = dspy.ChainOfThought(JudgeImproveSignature)
        
        def forward(self, current_prompt: str, evaluation_logs: str, criteria: str) -> dspy.Prediction:
            return self.improve(
                current_prompt=current_prompt,
                evaluation_logs=evaluation_logs,
                criteria=criteria
            )


@dataclass
class JudgePerformanceMetrics:
    """Metrics for evaluating judge performance."""
    
    consistency_score: float  # How consistent are scores across similar conversations
    discrimination_score: float  # How well does it distinguish good from bad
    calibration_score: float  # How well do scores match human expectations
    detail_score: float  # How detailed and useful are the rationales
    
    @property
    def overall_score(self) -> float:
        """Calculate weighted overall score."""
        return (
            self.consistency_score * 0.3 +
            self.discrimination_score * 0.3 +
            self.calibration_score * 0.2 +
            self.detail_score * 0.2
        )


class JudgeCalibrator:
    """Calibrate and improve judge performance over time."""
    
    def __init__(self, judge_history_path: str = "logs/judge_calibration.json"):
        self.history_path = judge_history_path
        self.calibration_data: List[Dict[str, Any]] = []
        self._load_history()
    
    def _load_history(self) -> None:
        """Load calibration history from disk."""
        try:
            with open(self.history_path, 'r', encoding='utf-8') as f:
                self.calibration_data = json.load(f)
        except FileNotFoundError:
            self.calibration_data = []
    
    def _save_history(self) -> None:
        """Save calibration history to disk."""
        utils.ensure_logs_dir()
        with open(self.history_path, 'w', encoding='utf-8') as f:
            json.dump(self.calibration_data, f, indent=2)
    
    def record_evaluation(
        self,
        conversation_log: Dict[str, Any],
        judge_result: Dict[str, Any],
        human_score: Optional[float] = None
    ) -> None:
        """Record a judge evaluation for calibration."""
        entry = {
            "timestamp": utils.get_timestamp(),
            "conversation_id": conversation_log.get("pop_agent_id"),
            "transcript_length": len(conversation_log.get("turns", [])),
            "judge_scores": {
                "goal_completion": _extract_score_value(judge_result.get("goal_completion", 0)),
                "coherence": _extract_score_value(judge_result.get("coherence", 0)),
                "tone": _extract_score_value(judge_result.get("tone", 0)),
                "overall": _extract_score_value(judge_result.get("overall", judge_result.get("score", 0)))
            },
            "confidence": _extract_score_value(judge_result.get("confidence", 0.5)),
            "human_score": human_score
        }
        
        self.calibration_data.append(entry)
        self._save_history()
    
    def calculate_metrics(self, min_samples: int = 10) -> JudgePerformanceMetrics:
        """Calculate performance metrics from calibration data."""
        if len(self.calibration_data) < min_samples:
            return JudgePerformanceMetrics(0.5, 0.5, 0.5, 0.5)
        
        # Consistency: variance in scores for similar-length conversations
        length_buckets: Dict[int, List[float]] = {}
        for entry in self.calibration_data:
            bucket = entry["transcript_length"] // 5
            score = _extract_score_value(entry["judge_scores"].get("overall", 0))
            length_buckets.setdefault(bucket, []).append(score)
        
        variances = []
        for bucket_scores in length_buckets.values():
            if len(bucket_scores) > 1:
                variances.append(np.var(bucket_scores))
        
        consistency_score = 1.0 - (np.mean(variances) if variances else 0.0)
        consistency_score = max(0.0, min(1.0, consistency_score))
        
        # Discrimination: range of scores used
        all_scores = [
            _extract_score_value(e["judge_scores"].get("overall", 0))
            for e in self.calibration_data
        ]
        score_range = max(all_scores) - min(all_scores) if all_scores else 0
        discrimination_score = min(1.0, score_range / 0.7)  # Expect at least 0.7 range
        
        # Calibration: correlation with human scores if available
        human_scored = [
            (
                _extract_score_value(e["judge_scores"].get("overall", 0)),
                e["human_score"],
            )
            for e in self.calibration_data
            if e.get("human_score") is not None
        ]
        
        if len(human_scored) >= 5:
            judge_scores, human_scores = zip(*human_scored)
            correlation = np.corrcoef(judge_scores, human_scores)[0, 1]
            calibration_score = max(0.0, correlation)
        else:
            calibration_score = 0.5  # Default when no human scores
        
        # Detail: average confidence scores (proxy for rationale quality)
        confidences = [
            _extract_score_value(e.get("confidence", 0.5))
            for e in self.calibration_data
        ]
        detail_score = np.mean(confidences)
        
        return JudgePerformanceMetrics(
            consistency_score=consistency_score,
            discrimination_score=discrimination_score,
            calibration_score=calibration_score,
            detail_score=detail_score
        )
    
    def get_improvement_suggestions(self) -> List[str]:
        """Generate suggestions for improving judge performance."""
        metrics = self.calculate_metrics()
        suggestions = []
        
        if metrics.consistency_score < 0.7:
            suggestions.append(
                "Improve consistency by adding more specific scoring examples and edge cases"
            )
        
        if metrics.discrimination_score < 0.7:
            suggestions.append(
                "Enhance discrimination by clarifying the differences between score levels"
            )
        
        if metrics.calibration_score < 0.7:
            suggestions.append(
                "Better align with human judgment by adjusting scoring weights and criteria"
            )
        
        if metrics.detail_score < 0.7:
            suggestions.append(
                "Provide more detailed rationales and specific evidence requirements"
            )
        
        return suggestions


def build_judge_improvement_dataset(
    calibrator: JudgeCalibrator,
    current_prompt: str,
    criteria_descriptions: str
) -> List[dspy.Example]:
    """Build dataset for training judge prompt improver."""
    if dspy is None:
        return []
    
    dataset = []
    metrics = calibrator.calculate_metrics()
    suggestions = calibrator.get_improvement_suggestions()
    
    # Create synthetic examples based on performance metrics
    evaluation_logs = f"""
Judge Performance Metrics:
- Consistency: {metrics.consistency_score:.2f}
- Discrimination: {metrics.discrimination_score:.2f}
- Calibration: {metrics.calibration_score:.2f}
- Detail Quality: {metrics.detail_score:.2f}

Issues Identified:
{chr(10).join(f"- {s}" for s in suggestions)}

Recent Evaluation Samples:
{json.dumps(calibrator.calibration_data[-5:], indent=2)}
"""
    
    example = dspy.Example(
        current_prompt=current_prompt,
        evaluation_logs=evaluation_logs,
        criteria=criteria_descriptions,
        performance_score=metrics.overall_score
    ).with_inputs("current_prompt", "evaluation_logs", "criteria")
    
    dataset.append(example)
    
    return dataset


def train_judge_improver(
    dataset: List[dspy.Example],
    current_judge_prompt: str
) -> Tuple[Any, Dict[str, Any]]:
    """Train the judge prompt improver using DSPy."""
    if dspy is None or not dataset:
        return None, {"error": "DSPy not available or empty dataset"}
    
    if dspy.settings.lm is None:
        dspy.settings.configure(
            lm=dspy.LM(
                model=config.LLM_MODEL,
                temperature=0.3,  # Lower temp for prompt improvement
                max_tokens=2048,  # Longer for detailed prompts
            )
        )
    
    def metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
        """Score improved prompts based on clarity and completeness."""
        improved = pred.improved_prompt
        
        # Check for required elements
        required_elements = [
            "evaluation criteria",
            "scoring guide",
            "JSON",
            "evidence",
            "rationale"
        ]
        
        element_score = sum(
            1 for elem in required_elements 
            if elem.lower() in improved.lower()
        ) / len(required_elements)
        
        # Length check (should be comprehensive)
        length_score = min(1.0, len(improved) / 2000)
        
        # Improvement score (should be different from original)
        if example.current_prompt:
            similarity = len(set(improved.split()) & set(example.current_prompt.split()))
            difference_score = 1.0 - (similarity / max(len(improved.split()), len(example.current_prompt.split())))
        else:
            difference_score = 0.5
        
        return element_score * 0.4 + length_score * 0.3 + difference_score * 0.3
    
    improver = JudgePromptImprover()
    
    if len(dataset) >= 3:
        optimizer = BootstrapFewShot(metric=metric, max_bootstrapped_demos=3)
        method = "BootstrapFewShot"
    else:
        optimizer = dspy.COPRO(metric=metric)
        method = "COPRO"
    
    try:
        trained = optimizer.compile(
            improver,
            trainset=dataset,
            eval_kwargs={"display_progress": False}
        )
        
        # Get the improved prompt
        result = trained(
            current_prompt=current_judge_prompt,
            evaluation_logs=dataset[0].evaluation_logs,
            criteria=dataset[0].criteria
        )
        
        improved_prompt = result.improved_prompt
        
        metrics = {
            "method": method,
            "dataset_size": len(dataset),
            "improved_prompt": improved_prompt,
            "timestamp": utils.get_timestamp()
        }
        
        return trained, metrics
        
    except Exception as e:
        return improver, {
            "error": str(e),
            "method": method,
            "dataset_size": len(dataset)
        }


# Integration with main judge system
def create_self_improving_judge(
    calibration_history: Optional[str] = None,
    improvement_interval: int = 20
) -> Tuple[LangChainJudge, JudgeCalibrator]:
    """Create a judge that can improve its own prompts over time."""
    from langchain_judge import LangChainJudge
    
    # Initialize calibrator
    calibrator = JudgeCalibrator(calibration_history or "logs/judge_calibration.json")
    
    # Check if improvement needed
    if len(calibrator.calibration_data) % improvement_interval == 0 and len(calibrator.calibration_data) > 0:
        # Run improvement process
        current_template = utils.load_template(config.JUDGE_PROMPT_TEMPLATE_PATH)
        criteria_desc = "\n".join([c.to_prompt_string() for c in LangChainJudge.DEFAULT_CRITERIA])
        
        dataset = build_judge_improvement_dataset(calibrator, current_template, criteria_desc)
        
        if dataset:
            _, metrics = train_judge_improver(dataset, current_template)
            
            if "improved_prompt" in metrics and not metrics.get("error"):
                # Save improved prompt
                improved_path = f"templates/judge_prompt_improved_{utils.get_timestamp()}.txt"
                with open(improved_path, 'w', encoding='utf-8') as f:
                    f.write(metrics["improved_prompt"])
                
                print(f"Judge prompt improved and saved to {improved_path}")
                
                # Create judge with improved prompt
                judge = LangChainJudge(template_path=improved_path)
            else:
                judge = LangChainJudge()
        else:
            judge = LangChainJudge()
    else:
        judge = LangChainJudge()
    
    return judge, calibrator
