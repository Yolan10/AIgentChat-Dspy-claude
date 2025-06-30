"""Enhanced JudgeAgent as a completely separate and independent entity."""
from __future__ import annotations

from typing import Dict, List, Optional, Any
import json

from langchain_judge import LangChainJudge, EvaluationCriteria
from judge_improver import JudgeCalibrator, build_judge_improvement_dataset, train_judge_improver
import config
import utils


class EnhancedJudgeAgent:
    """Independent judge agent for evaluating conversations.
    
    This judge operates completely independently from wizards and other agents,
    providing objective evaluation of conversation quality.
    """
    
    def __init__(
        self,
        judge_id: str = "Judge_001",
        llm_settings: Optional[Dict[str, Any]] = None,
        improvement_interval: int = 20,
        custom_criteria: Optional[List[EvaluationCriteria]] = None,
        judge_prompt_template: Optional[str] = None
    ):
        """Initialize an independent judge agent.
        
        Args:
            judge_id: Unique identifier for this judge
            llm_settings: LLM configuration for the judge
            improvement_interval: How often to improve the judge prompt
            custom_criteria: Additional evaluation criteria beyond defaults
            judge_prompt_template: Path to custom judge prompt template
        """
        self.judge_id = judge_id
        self.improvement_interval = improvement_interval
        self.evaluation_count = 0
        
        # LLM settings specifically for judging (lower temperature for consistency)
        self.llm_settings = llm_settings or {
            "model": config.LLM_MODEL,
            "temperature": 0.2,  # Lower than wizard for more consistent evaluation
            "max_tokens": 1024,  # Enough for detailed evaluation
        }
        
        # Initialize the core judge with potential custom criteria
        all_criteria = LangChainJudge.DEFAULT_CRITERIA
        if custom_criteria:
            all_criteria = all_criteria + custom_criteria
            
        self.core_judge = LangChainJudge(
            llm_settings=self.llm_settings,
            criteria=all_criteria,
            template_path=judge_prompt_template
        )
        
        # Initialize calibrator for tracking performance
        self.calibrator = JudgeCalibrator(
            judge_history_path=f"logs/judge_calibration_{judge_id}.json"
        )
        
        # Track improvement history
        self.improvement_history = []
        
        # Current prompt template path
        self.current_template_path = judge_prompt_template or config.JUDGE_PROMPT_TEMPLATE_PATH
        
    def assess(self, conversation_log: Dict[str, Any]) -> Dict[str, Any]:
        """Assess a conversation independently.
        
        Args:
            conversation_log: The conversation to evaluate
            
        Returns:
            Evaluation result with scores and rationale
        """
        # Perform evaluation
        result = self.core_judge.assess(conversation_log)
        
        # Add metadata about which judge evaluated this
        result["judge_id"] = self.judge_id
        result["evaluation_number"] = self.evaluation_count
        result["evaluation_timestamp"] = utils.get_timestamp()
        
        # Record for calibration (without human score initially)
        self.calibrator.record_evaluation(conversation_log, result)
        
        # Increment counter
        self.evaluation_count += 1
        
        # Check if we should improve
        if self.evaluation_count % self.improvement_interval == 0:
            self._attempt_improvement()
        
        return result
    
    def batch_assess(self, conversation_logs: List[Dict[str, Any]], show_progress: bool = True) -> List[Dict[str, Any]]:
        """Assess multiple conversations.
        
        Args:
            conversation_logs: List of conversations to evaluate
            show_progress: Whether to print progress
            
        Returns:
            List of evaluation results
        """
        results = []
        
        for idx, log in enumerate(conversation_logs):
            if show_progress:
                print(f"{self.judge_id}: Evaluating conversation {idx + 1}/{len(conversation_logs)}...")
            
            result = self.assess(log)
            results.append(result)
        
        return results
    
    def add_human_feedback(self, conversation_id: str, human_score: float) -> None:
        """Add human feedback for a previously evaluated conversation.
        
        Args:
            conversation_id: The pop_agent_id from the conversation
            human_score: Human-provided score (0.0 to 1.0)
        """
        # Update the calibration data with human score
        for entry in self.calibrator.calibration_data:
            if entry.get("conversation_id") == conversation_id:
                entry["human_score"] = human_score
                self.calibrator._save_history()
                break
    
    def _attempt_improvement(self) -> None:
        """Attempt to improve the judge based on performance metrics."""
        metrics = self.calibrator.calculate_metrics()
        
        print(f"\n{self.judge_id} Performance Check at evaluation {self.evaluation_count}:")
        print(f"  - Consistency: {metrics.consistency_score:.2f}")
        print(f"  - Discrimination: {metrics.discrimination_score:.2f}")
        print(f"  - Calibration: {metrics.calibration_score:.2f}")
        print(f"  - Detail Quality: {metrics.detail_score:.2f}")
        print(f"  - Overall: {metrics.overall_score:.2f}")
        
        # Only improve if performance is suboptimal
        if metrics.overall_score < 0.8:
            print(f"{self.judge_id}: Attempting improvement...")
            
            # Load current prompt
            current_prompt = utils.load_template(self.current_template_path)
            criteria_desc = "\n".join([c.to_prompt_string() for c in self.core_judge.criteria])
            
            # Build improvement dataset
            dataset = build_judge_improvement_dataset(
                self.calibrator,
                current_prompt,
                criteria_desc
            )
            
            if dataset:
                _, improvement_metrics = train_judge_improver(dataset, current_prompt)
                
                if "improved_prompt" in improvement_metrics and not improvement_metrics.get("error"):
                    # Save improved prompt
                    timestamp = utils.get_timestamp().replace(':', '').replace('-', '')
                    improved_path = f"templates/judge_{self.judge_id}_improved_{timestamp}.txt"
                    
                    with open(improved_path, 'w', encoding='utf-8') as f:
                        f.write(improvement_metrics["improved_prompt"])
                    
                    # Update judge with new prompt
                    self.current_template_path = improved_path
                    self.core_judge = LangChainJudge(
                        llm_settings=self.llm_settings,
                        criteria=self.core_judge.criteria,
                        template_path=improved_path
                    )
                    
                    # Record improvement
                    self.improvement_history.append({
                        "evaluation_count": self.evaluation_count,
                        "timestamp": utils.get_timestamp(),
                        "metrics_before": {
                            "consistency": metrics.consistency_score,
                            "discrimination": metrics.discrimination_score,
                            "calibration": metrics.calibration_score,
                            "detail_quality": metrics.detail_score,
                            "overall": metrics.overall_score
                        },
                        "improvement_result": improvement_metrics,
                        "new_template_path": improved_path
                    })
                    
                    print(f"{self.judge_id}: Improvement completed! New prompt saved to {improved_path}")
                else:
                    print(f"{self.judge_id}: Improvement failed - {improvement_metrics.get('error', 'Unknown error')}")
        else:
            print(f"{self.judge_id}: Performance is good (>= 0.8), no improvement needed")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get a detailed performance report for this judge."""
        metrics = self.calibrator.calculate_metrics()
        
        return {
            "judge_id": self.judge_id,
            "total_evaluations": self.evaluation_count,
            "current_template": self.current_template_path,
            "performance_metrics": {
                "consistency": metrics.consistency_score,
                "discrimination": metrics.discrimination_score,  
                "calibration": metrics.calibration_score,
                "detail_quality": metrics.detail_score,
                "overall": metrics.overall_score
            },
            "improvement_history": self.improvement_history,
            "suggestions": self.calibrator.get_improvement_suggestions(),
            "calibration_data_size": len(self.calibrator.calibration_data)
        }
    
    def get_criteria(self) -> List[EvaluationCriteria]:
        """Get the current evaluation criteria."""
        return self.core_judge.criteria
    
    def update_criteria(self, new_criteria: List[EvaluationCriteria]) -> None:
        """Update the evaluation criteria.
        
        Args:
            new_criteria: New list of evaluation criteria
        """
        self.core_judge.update_criteria(new_criteria)
        print(f"{self.judge_id}: Updated evaluation criteria")


# Backward compatibility - keep the simple JudgeAgent name
JudgeAgent = EnhancedJudgeAgent
