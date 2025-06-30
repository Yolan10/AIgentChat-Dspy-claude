"""High level integration layer with SEPARATE judge orchestration."""
from __future__ import annotations

from typing import List, Dict, Any
import threading
import time

import config
import utils

from advanced_features import PopulationGenerator
from logging_system import StructuredLogger
from token_tracker import token_tracker


class IntegratedSystem:
    """Coordinates population generation, conversations and INDEPENDENT judging."""

    def __init__(
        self,
        god_agent_cls=None,
        wizard_agent_cls=None,
        judge_agent_cls=None,
        logger: StructuredLogger | None = None,
        generator: PopulationGenerator | None = None,
    ) -> None:
        """Initialize the system with injectable dependencies."""
        if god_agent_cls is None or wizard_agent_cls is None or judge_agent_cls is None:
            from god_agent import GodAgent as _GodAgent
            from wizard_agent import WizardAgent as _WizardAgent
            try:  # pragma: no cover - when dependencies are missing
                from judge_agent import EnhancedJudgeAgent as _JudgeAgent
            except Exception:  # pragma: no cover - lightweight fallback for tests
                class _JudgeAgent:  # type: ignore
                    def __init__(self, *args, **kwargs) -> None:
                        pass

                    def assess(self, _log: Any) -> dict:
                        return {}
            god_agent_cls = god_agent_cls or _GodAgent
            wizard_agent_cls = wizard_agent_cls or _WizardAgent
            judge_agent_cls = judge_agent_cls or _JudgeAgent

        self.logger = logger or StructuredLogger()
        self.generator = generator or PopulationGenerator()
        self._god_cls = god_agent_cls
        self._wizard_cls = wizard_agent_cls
        self._judge_cls = judge_agent_cls

        self.god = self._god_cls()
        self.wizard = self._wizard_cls(wizard_id="Wizard_001")

        
        # Initialize SEPARATE judge agent(s)
        self.primary_judge = self._judge_cls(
            judge_id="Judge_001",
            improvement_interval=config.JUDGE_IMPROVEMENT_INTERVAL if hasattr(config, 'JUDGE_IMPROVEMENT_INTERVAL') else 20
        )
        
        # Optional: Enable multiple judges for consensus
        self.enable_multi_judge = config.ENABLE_MULTI_JUDGE if hasattr(config, 'ENABLE_MULTI_JUDGE') else False
        self.judges: List[Any] = []
        
        if self.enable_multi_judge:
            judge_count = config.JUDGE_COUNT if hasattr(config, 'JUDGE_COUNT') else 3
            self.judges = [
                self._judge_cls(
                    judge_id=f"Judge_{i:03d}",
                    improvement_interval=30  # Slightly different intervals for diversity
                )
                for i in range(1, judge_count + 1)
            ]
        else:
            self.judges = [self.primary_judge]

    def run(
        self,
        instruction: str,
        n: int,
        stop_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
    ) -> None:
        """Run simulation with separate conversation and judging phases."""
        run_no = utils.increment_run_number()
        self.wizard.set_run(run_no)
        token_tracker.set_run(run_no)

        self.logger.log_event("system_start", instruction=instruction, n=n, run_no=run_no)
        
        print(f"\n{'='*60}")
        print(f"Starting Run #{run_no}")
        print(f"Population size: {n}")
        print(f"Wizard goal: {self.wizard.goal}")
        print(f"{'='*60}\n")
        
        specs = self.generator.generate(instruction, n)
        summary: List[dict] = []

        def _parse_schedule(total: int) -> List[int]:
            sched = config.SELF_IMPROVE_AFTER
            points: List[int] = []
            if isinstance(sched, int):
                if sched > 0:
                    points = list(range(sched, total + 1, sched))
            elif isinstance(sched, str):
                try:
                    points = [int(x) for x in sched.split(";") if x.strip()]
                except ValueError:
                    points = []
            else:
                try:
                    points = [int(x) for x in sched]
                except Exception:
                    points = []
            return sorted({p for p in points if p <= total})

        schedule = _parse_schedule(n)
        schedule_index = 0
        next_point = schedule[schedule_index] if schedule else n + 1  # Fix: ensure it's beyond n if no schedule
        batch_threads: List[threading.Thread] = []
        batch_agents: List = []

        def run_conversation(pop, conv_index):
            """Run a conversation and have it independently judged."""
            if stop_event and stop_event.is_set():
                return
            if pause_event:
                while pause_event.is_set():
                    time.sleep(0.5)
            
            print(f"\n{'='*50}")
            print(f"Starting conversation {conv_index}/{n} with {pop.name} ({pop.agent_id})")
            print(f"{'='*50}")
            
            # PHASE 1: Wizard converses WITHOUT judging
            log = self.wizard.converse_with(pop, show_live=config.SHOW_LIVE_CONVERSATIONS)
            
            print(f"\nConversation with {pop.name} completed. Now judging...")
            
            # PHASE 2: Independent judging
            if self.enable_multi_judge:
                # Multiple judges evaluate independently
                judge_results = []
                for judge in self.judges:
                    result = judge.assess(log)
                    judge_results.append(result)
                    self.logger.log_event(
                        "individual_judge_complete",
                        judge_id=judge.judge_id,
                        agent_id=pop.agent_id,
                        score=result.get("overall", 0)
                    )
                
                # Aggregate results
                aggregated_result = self._aggregate_judge_results(judge_results)
                log["judge_result"] = aggregated_result
                log["individual_judge_results"] = judge_results
                log["judge_consensus"] = aggregated_result.get("judge_consensus", False)
            else:
                # Single judge evaluation
                judge_result = self.primary_judge.assess(log)
                log["judge_result"] = judge_result
                self.logger.log_event(
                    "judge_complete",
                    judge_id=self.primary_judge.judge_id,
                    agent_id=pop.agent_id,
                    score=judge_result.get("overall", 0)
                )
            
            print(f"Judge evaluation complete. Score: {log['judge_result'].get('overall', 0):.2f}")
            
            # PHASE 3: Wizard receives feedback for learning
            self.wizard.add_judge_feedback(pop.agent_id, log["judge_result"])
            
            # Save the complete log with judge results
            filename = f"{self.wizard.wizard_id}_{pop.agent_id}_{utils.get_timestamp().replace(':', '').replace('-', '')}.json"
            utils.save_conversation_log(log, filename)
            
            # Update summary
            spec = pop.get_spec()
            entry = {
                "pop_agent_id": pop.agent_id,
                "name": spec.get("name"),
                "personality_description": spec.get("personality_description"),
                "system_instruction": spec.get("system_instruction"),
                "temperature": spec.get("llm_settings", {}).get("temperature"),
                "max_tokens": spec.get("llm_settings", {}).get("max_tokens"),
                "success": log["judge_result"].get("success"),
                "goal_completion": log["judge_result"].get("goal_completion"),
                "coherence": log["judge_result"].get("coherence"),
                "tone": log["judge_result"].get("tone"),
                "score": log["judge_result"].get("overall"),
                "judge_consensus": log.get("judge_consensus", True),
            }
            
            summary.append(entry)
            self.logger.log_event(
                "conversation_end",
                pop_agent=pop.agent_id,
                success=entry["success"],
                run_no=run_no,
            )

        # Generate population and run conversations
        for idx, spec in enumerate(specs, start=1):
            agent = self.god.spawn_population_from_spec(spec, run_no, idx)
            
            print(f"Created agent {idx}/{n}: {agent.name} ({agent.agent_id})")

            if config.PARALLEL_CONVERSATIONS:
                if config.START_WHEN_SPAWNED:
                    t = threading.Thread(target=run_conversation, args=(agent, idx))
                    batch_threads.append(t)
                    t.start()
                else:
                    batch_agents.append((agent, idx))
            else:
                run_conversation(agent, idx)

            # Check if we've reached an improvement point
            if idx == next_point:
                if config.PARALLEL_CONVERSATIONS:
                    # Start any queued agents
                    if not config.START_WHEN_SPAWNED:
                        for ag, ag_idx in batch_agents:
                            t = threading.Thread(target=run_conversation, args=(ag, ag_idx))
                            batch_threads.append(t)
                            t.start()
                        batch_agents = []
                    # Wait for all threads to complete
                    for t in batch_threads:
                        t.join()
                    batch_threads = []
                
                # Update to next improvement point
                schedule_index += 1
                next_point = schedule[schedule_index] if schedule_index < len(schedule) else n + 1

        # Handle any remaining conversations
        if config.PARALLEL_CONVERSATIONS:
            if not config.START_WHEN_SPAWNED:
                for ag, ag_idx in batch_agents:
                    t = threading.Thread(target=run_conversation, args=(ag, ag_idx))
                    batch_threads.append(t)
                    t.start()
            for t in batch_threads:
                t.join()

        utils.save_conversation_log(summary, f"summary_{run_no}.json")
        self.logger.log_event("system_end", run_no=run_no)
        
        print(f"\n{'='*60}")
        print(f"Run #{run_no} Complete!")
        print(f"Total conversations: {len(summary)}")
        print(f"Average score: {sum(e['score'] for e in summary) / len(summary) if summary else 0:.2f}")
        print(f"Successful conversations: {sum(1 for e in summary if e['success'])}")
        print(f"Summary saved to: logs/summary_{run_no}.json")
        print(f"{'='*60}")
        
        # Print judge performance summary
        self._print_judge_performance()

    def _aggregate_judge_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple judges.
        
        Args:
            results: List of individual judge results
            
        Returns:
            Aggregated result with consensus metrics
        """
        if not results:
            return {}
        
        # Extract scores for each criterion
        goal_scores = [r.get("goal_completion", 0) for r in results]
        coherence_scores = [r.get("coherence", 0) for r in results]
        tone_scores = [r.get("tone", 0) for r in results]
        overall_scores = [r.get("overall", 0) for r in results]
        success_votes = [r.get("success", False) for r in results]
        
        # Calculate aggregated scores (mean)
        aggregated = {
            "goal_completion": sum(goal_scores) / len(goal_scores),
            "coherence": sum(coherence_scores) / len(coherence_scores),
            "tone": sum(tone_scores) / len(tone_scores),
            "overall": sum(overall_scores) / len(overall_scores),
            "success": sum(success_votes) > len(success_votes) / 2,  # Majority vote
            "score": sum(overall_scores) / len(overall_scores),  # Backward compatibility
        }
        
        # Calculate consensus metrics
        aggregated["judge_consensus"] = all(success_votes) or not any(success_votes)  # All agree
        aggregated["number_of_judges"] = len(results)
        aggregated["score_variance"] = self._calculate_variance(overall_scores)
        aggregated["confidence"] = 1.0 - (aggregated["score_variance"] * 2)  # Higher variance = lower confidence
        
        # Aggregate rationales
        rationales = [f"{r.get('judge_id', 'Unknown')}: {r.get('rationale', 'No rationale')}" 
                     for r in results]
        aggregated["rationale"] = " | ".join(rationales)
        
        # Include judge agreement details
        aggregated["judge_agreement"] = {
            "goal_completion_variance": self._calculate_variance(goal_scores),
            "coherence_variance": self._calculate_variance(coherence_scores),
            "tone_variance": self._calculate_variance(tone_scores),
            "overall_variance": self._calculate_variance(overall_scores),
        }
        
        return aggregated
    
    def _calculate_variance(self, scores: List[float]) -> float:
        """Calculate variance of scores."""
        if not scores:
            return 0.0
        mean = sum(scores) / len(scores)
        variance = sum((x - mean) ** 2 for x in scores) / len(scores)
        return variance
    
    def _print_judge_performance(self) -> None:
        """Print performance summary for all judges."""
        print("\n" + "="*60)
        print("JUDGE PERFORMANCE SUMMARY")
        print("="*60)
        
        for judge in self.judges:
            report = judge.get_performance_report()
            print(f"\n{report['judge_id']}:")
            print(f"  Total Evaluations: {report['total_evaluations']}")
            print(f"  Performance Metrics:")
            metrics = report['performance_metrics']
            print(f"    - Consistency: {metrics['consistency']:.2f}")
            print(f"    - Discrimination: {metrics['discrimination']:.2f}")
            print(f"    - Calibration: {metrics['calibration']:.2f}")
            print(f"    - Detail Quality: {metrics['detail_quality']:.2f}")
            print(f"    - Overall: {metrics['overall']:.2f}")
            
            if report['improvement_history']:
                print(f"  Improvements Made: {len(report['improvement_history'])}")
            
            if report['suggestions']:
                print(f"  Suggestions:")
                for suggestion in report['suggestions']:
                    print(f"    - {suggestion}")
        
        print("="*60 + "\n")
    
    def get_judge_performance_reports(self) -> Dict[str, Any]:
        """Get performance reports for all judges.
        
        Returns:
            Dictionary mapping judge IDs to their performance reports
        """
        return {
            judge.judge_id: judge.get_performance_report() 
            for judge in self.judges
        }
    
    def add_human_feedback(self, conversation_id: str, human_scores: Dict[str, float]) -> None:
        """Add human feedback for calibrating judges.
        
        Args:
            conversation_id: The pop_agent_id from the conversation
            human_scores: Dictionary of judge_id -> human score for that judge's evaluation
        """
        for judge in self.judges:
            if judge.judge_id in human_scores:
                judge.add_human_feedback(conversation_id, human_scores[judge.judge_id])
