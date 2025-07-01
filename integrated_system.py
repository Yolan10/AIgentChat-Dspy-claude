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
            try:
                from judge_agent import EnhancedJudgeAgent as _JudgeAgent
            except Exception:
                class _JudgeAgent:
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
                    improvement_interval=30
                )
                for i in range(1, judge_count + 1)
            ]
        else:
            self.judges = [self.primary_judge]

    def _extract_score_value(self, score_data: Any) -> float:
        """Extract a numeric score from various possible formats."""
        if isinstance(score_data, (int, float)):
            return float(score_data)
        elif isinstance(score_data, dict):
            # Handle CriterionScore objects
            if 'score' in score_data:
                return float(score_data['score'])
            # Handle other dict formats
            for key in ['value', 'rating', 'overall']:
                if key in score_data:
                    return float(score_data[key])
        elif hasattr(score_data, 'score'):
            # Handle objects with score attribute
            return float(score_data.score)
        
        # Default fallback
        return 0.0

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
        
        print(f"\n{'='*80}")
        print(f"Starting Run #{run_no}")
        print(f"Population size: {n}")
        print(f"Wizard goal: {self.wizard.goal}")
        print(f"Instruction: {instruction}")
        print(f"{'='*80}\n")
        
        # PHASE 1: Generate population specs
        print(f"{'='*60}")
        print("PHASE 1: GENERATING POPULATION")
        print(f"{'='*60}")
        
        specs = self.generator.generate(instruction, n)
        print(f"Generated {len(specs)} population specifications\n")
        
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
        next_point = schedule[schedule_index] if schedule else n + 1
        batch_threads: List[threading.Thread] = []
        batch_agents: List = []

        def run_conversation(pop, conv_index):
            """Run a conversation and have it independently judged."""
            if stop_event and stop_event.is_set():
                return
            if pause_event:
                while pause_event.is_set():
                    time.sleep(0.5)
            
            print(f"\n{'='*60}")
            print(f"CONVERSATION {conv_index}/{n}: {pop.name} ({pop.agent_id})")
            print(f"{'='*60}")
            
            # PHASE 2A: Wizard converses WITHOUT judging
            print(f"\n[WIZARD] Starting conversation with {pop.name}...")
            print("-" * 40)
            
            try:
                log = self.wizard.converse_with(pop, show_live=config.SHOW_LIVE_CONVERSATIONS)
                print("-" * 40)
                print(f"[WIZARD] Conversation with {pop.name} completed.")
                print(f"         Total turns: {len(log.get('turns', []))}")
            except Exception as e:
                print(f"[ERROR] Conversation failed with {pop.name}: {e}")
                self.logger.log_event("conversation_error", agent_id=pop.agent_id, error=str(e))
                return
            
            # PHASE 2B: Independent judging
            print(f"\n[JUDGE] Evaluating conversation with {pop.name}...")
            
            if self.enable_multi_judge:
                # Multiple judges evaluate independently
                judge_results = []
                for judge in self.judges:
                    print(f"  - {judge.judge_id} evaluating...")
                    result = judge.assess(log)
                    judge_results.append(result)
                    overall_score = self._extract_score_value(result.get('overall', result.get('score', 0)))
                    print(f"    Score: {overall_score:.2f}")
                    self.logger.log_event(
                        "individual_judge_complete",
                        judge_id=judge.judge_id,
                        agent_id=pop.agent_id,
                        score=overall_score
                    )
                
                # Aggregate results
                aggregated_result = self._aggregate_judge_results(judge_results)
                log["judge_result"] = aggregated_result
                log["individual_judge_results"] = judge_results
                log["judge_consensus"] = aggregated_result.get("judge_consensus", False)
                
                print(f"\n[JUDGE] Consensus evaluation complete:")
                print(f"  - Overall Score: {self._extract_score_value(aggregated_result.get('overall', 0)):.2f}")
                print(f"  - Goal Completion: {self._extract_score_value(aggregated_result.get('goal_completion', 0)):.2f}")
                print(f"  - Coherence: {self._extract_score_value(aggregated_result.get('coherence', 0)):.2f}")
                print(f"  - Tone: {self._extract_score_value(aggregated_result.get('tone', 0)):.2f}")
                print(f"  - Success: {aggregated_result.get('success', False)}")
                print(f"  - Judge Agreement: {aggregated_result.get('judge_consensus', False)}")
            else:
                # Single judge evaluation
                judge_result = self.primary_judge.assess(log)
                log["judge_result"] = judge_result
                
                print(f"\n[JUDGE] Evaluation complete:")
                print(f"  - Overall Score: {self._extract_score_value(judge_result.get('overall', judge_result.get('score', 0))):.2f}")
                print(f"  - Goal Completion: {self._extract_score_value(judge_result.get('goal_completion', 0)):.2f}")
                print(f"  - Coherence: {self._extract_score_value(judge_result.get('coherence', 0)):.2f}")
                print(f"  - Tone: {self._extract_score_value(judge_result.get('tone', 0)):.2f}")
                print(f"  - Success: {judge_result.get('success', False)}")
                
                self.logger.log_event(
                    "judge_complete",
                    judge_id=self.primary_judge.judge_id,
                    agent_id=pop.agent_id,
                    score=self._extract_score_value(judge_result.get("overall", judge_result.get("score", 0)))
                )
            
            # PHASE 2C: Wizard receives feedback for learning
            print(f"\n[SYSTEM] Adding judge feedback to wizard's learning buffer...")
            self.wizard.add_judge_feedback(pop.agent_id, log["judge_result"])
            
            # Save the complete log with judge results
            filename = f"{self.wizard.wizard_id}_{pop.agent_id}_{utils.get_timestamp().replace(':', '').replace('-', '')}.json"
            utils.save_conversation_log(log, filename)
            print(f"[SYSTEM] Conversation log saved: {filename}")
            
            # Update summary - extract numeric values for storage
            spec = pop.get_spec()
            judge_result = log["judge_result"]
            entry = {
                "pop_agent_id": pop.agent_id,
                "name": spec.get("name"),
                "personality_description": spec.get("personality_description"),
                "age": spec.get("age"),
                "occupation": spec.get("occupation"),
                "initial_goals": spec.get("initial_goals"),
                "memory_summary": spec.get("memory_summary"),
                "system_instruction": spec.get("system_instruction"),
                "temperature": spec.get("llm_settings", {}).get("temperature"),
                "max_tokens": spec.get("llm_settings", {}).get("max_tokens"),
                "success": judge_result.get("success"),
                "goal_completion": self._extract_score_value(judge_result.get("goal_completion")),
                "coherence": self._extract_score_value(judge_result.get("coherence")),
                "tone": self._extract_score_value(judge_result.get("tone")),
                "score": self._extract_score_value(judge_result.get("overall", judge_result.get("score", 0))),
                "judge_consensus": log.get("judge_consensus", True),
            }
            
            summary.append(entry)
            self.logger.log_event(
                "conversation_end",
                pop_agent=pop.agent_id,
                success=entry["success"],
                run_no=run_no,
            )
            
            print(f"\n{'='*60}")
            print(f"CONVERSATION {conv_index}/{n} COMPLETE")
            print(f"{'='*60}\n")

        # PHASE 2: Generate population agents and run conversations
        print(f"\n{'='*60}")
        print("PHASE 2: CREATING AGENTS AND RUNNING CONVERSATIONS")
        print(f"{'='*60}")
        
        for idx, spec in enumerate(specs, start=1):
            # Create agent
            print(f"\n[GOD] Creating agent {idx}/{n}...")
            agent = self.god.spawn_population_from_spec(spec, run_no, idx)
            
            print(f"[GOD] Created: {agent.name} ({agent.agent_id})")
            print(f"      Age: {agent.age}, Occupation: {agent.occupation}")
            print(f"      Goals: {agent.initial_goals}")

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
                print(f"\n[SYSTEM] Reached improvement point at conversation {idx}")
                
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
                
                # Trigger wizard improvement if scheduled
                if self.wizard._should_self_improve():
                    print(f"[WIZARD] Triggering self-improvement...")
                    # Improvement will happen in wizard's converse_with method
                
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

        # PHASE 3: Save summary and complete
        print(f"\n{'='*60}")
        print("PHASE 3: SAVING RESULTS")
        print(f"{'='*60}")
        
        utils.save_conversation_log(summary, f"summary_{run_no}.json")
        self.logger.log_event("system_end", run_no=run_no)
        
        print(f"\n{'='*80}")
        print(f"RUN #{run_no} COMPLETE!")
        print(f"{'='*80}")
        print(f"Total conversations: {len(summary)}")
        print(f"Average score: {sum(e['score'] for e in summary) / len(summary) if summary else 0:.2f}")
        print(f"Successful conversations: {sum(1 for e in summary if e['success'])}")
        print(f"Summary saved to: logs/summary_{run_no}.json")
        print(f"{'='*80}")
        
        # Print judge performance summary
        self._print_judge_performance()

    def _aggregate_judge_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple judges."""
        if not results:
            return {}
        
        # Extract scores for each criterion - handle both numeric and dict formats
        goal_scores = [self._extract_score_value(r.get("goal_completion", 0)) for r in results]
        coherence_scores = [self._extract_score_value(r.get("coherence", 0)) for r in results]
        tone_scores = [self._extract_score_value(r.get("tone", 0)) for r in results]
        overall_scores = [self._extract_score_value(r.get("overall", r.get("score", 0))) for r in results]
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
        aggregated["judge_consensus"] = all(success_votes) or not any(success_votes)
        aggregated["number_of_judges"] = len(results)
        aggregated["score_variance"] = self._calculate_variance(overall_scores)
        aggregated["confidence"] = 1.0 - (aggregated["score_variance"] * 2)
        
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
        """Get performance reports for all judges."""
        return {
            judge.judge_id: judge.get_performance_report() 
            for judge in self.judges
        }
    
    def add_human_feedback(self, conversation_id: str, human_scores: Dict[str, float]) -> None:
        """Add human feedback for calibrating judges."""
        for judge in self.judges:
            if judge.judge_id in human_scores:
                judge.add_human_feedback(conversation_id, human_scores[judge.judge_id])
