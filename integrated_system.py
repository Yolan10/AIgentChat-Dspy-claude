"""High level integration layer with SEPARATE judge orchestration and PARALLEL judging."""
from __future__ import annotations

from typing import List, Dict, Any
import threading
import time
import queue
from concurrent.futures import ThreadPoolExecutor, Future

import config
import utils

from advanced_features import PopulationGenerator
from logging_system import StructuredLogger
from token_tracker import token_tracker


class IntegratedSystem:
    """Coordinates population generation, conversations and INDEPENDENT PARALLEL judging."""

    def __init__(
        self,
        god_agent_cls=None,
        wizard_agent_cls=None,
        judge_agent_cls=None,
        logger: StructuredLogger | None = None,
        generator: PopulationGenerator | None = None,
        max_judge_workers: int = None,
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
            improvement_interval=getattr(config, 'JUDGE_IMPROVEMENT_INTERVAL', 20)
        )

        # Optional: Enable multiple judges for consensus
        self.enable_multi_judge = getattr(config, 'ENABLE_MULTI_JUDGE', False)
        self.judges: List[Any] = []

        if self.enable_multi_judge:
            judge_count = getattr(config, 'JUDGE_COUNT', 3)
            self.judges = [
                self._judge_cls(
                    judge_id=f"Judge_{i:03d}",
                    improvement_interval=30
                )
                for i in range(1, judge_count + 1)
            ]
        else:
            self.judges = [self.primary_judge]

        # Parallel judging infrastructure
        self.max_judge_workers = max_judge_workers or getattr(config, 'MAX_JUDGE_WORKERS', 3)
        self.judge_executor = ThreadPoolExecutor(max_workers=self.max_judge_workers)
        self.pending_judgments: Dict[str, Future] = {}  # conversation_id -> Future
        self.judgment_queue = queue.Queue()  # Queue for completed judgments
        self.judgment_lock = threading.Lock()
        self.completed_judgments: Dict[str, Dict[str, Any]] = {}  # Store completed results
        
        # Initialize shutdown flag BEFORE starting thread
        self._shutdown = False
        
        # Start judgment processor thread
        self.judgment_processor = threading.Thread(
            target=self._process_judgments, 
            daemon=True
        )
        self.judgment_processor.start()

    def _process_judgments(self):
        """Background thread that processes completed judgments."""
        while not self._shutdown:
            try:
                # Get completed judgment from queue
                judgment_data = self.judgment_queue.get(timeout=1)
                
                if judgment_data is None:  # Shutdown signal
                    break
                    
                conversation_id = judgment_data["conversation_id"]
                judge_result = judgment_data["judge_result"]
                log = judgment_data["log"]
                
                # Store completed judgment
                with self.judgment_lock:
                    self.completed_judgments[conversation_id] = {
                        "judge_result": judge_result,
                        "log": log,
                        "timestamp": utils.get_timestamp()
                    }
                
                # Add feedback to wizard
                self.wizard.add_judge_feedback(conversation_id, judge_result)
                
                print(f"[JUDGE] ✓ Processed judgment for {conversation_id}")
                print(f"         Score: {self._extract_score_value(judge_result.get('overall', judge_result.get('score', 0))):.2f}")
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[JUDGE] Error processing judgment: {e}")
                import traceback
                traceback.print_exc()

    def _submit_for_judgment(self, log: Dict[str, Any], pop_agent_id: str):
        """Submit a conversation for parallel judgment."""
        def judge_task():
            try:
                print(f"[JUDGE] → Starting evaluation of {pop_agent_id}...")
                
                if self.enable_multi_judge:
                    judge_results = []
                    for judge in self.judges:
                        result = judge.assess(log)
                        judge_results.append(result)
                    
                    aggregated_result = self._aggregate_judge_results(judge_results)
                    aggregated_result["individual_judge_results"] = judge_results
                    aggregated_result["judge_consensus"] = aggregated_result.get("judge_consensus", False)
                    judge_result = aggregated_result
                else:
                    judge_result = self.primary_judge.assess(log)
                
                # Put result in queue for processing
                self.judgment_queue.put({
                    "conversation_id": pop_agent_id,
                    "judge_result": judge_result,
                    "log": log
                })
                
                return judge_result
            except Exception as e:
                print(f"[JUDGE] Error evaluating {pop_agent_id}: {e}")
                import traceback
                traceback.print_exc()
                return None
        
        # Submit to thread pool
        future = self.judge_executor.submit(judge_task)
        
        with self.judgment_lock:
            self.pending_judgments[pop_agent_id] = future
        
        return future

    def _wait_for_pending_judgments(self, timeout: int = None):
        """Wait for all pending judgments to complete."""
        if timeout is None:
            timeout = getattr(config, 'JUDGE_TIMEOUT', 60)
            
        with self.judgment_lock:
            pending_count = len(self.pending_judgments)
            
        if pending_count == 0:
            print(f"[SYSTEM] No pending judgments to wait for")
            return
            
        print(f"[SYSTEM] ⏳ Waiting for {pending_count} pending judgments (timeout: {timeout}s)...")
        
        start_time = time.time()
        with self.judgment_lock:
            futures = list(self.pending_judgments.values())
        
        completed = 0
        failed = 0
        for future in futures:
            remaining_time = timeout - (time.time() - start_time)
            if remaining_time > 0:
                try:
                    future.result(timeout=remaining_time)
                    completed += 1
                except Exception as e:
                    print(f"[SYSTEM] Judgment failed: {e}")
                    failed += 1
            else:
                print(f"[SYSTEM] Timeout waiting for judgments")
                break
        
        print(f"[SYSTEM] ✓ Completed {completed}/{len(futures)} judgments ({failed} failed)")
        
        # Allow queue processing to catch up
        time.sleep(0.5)
        
        # Clear pending judgments
        with self.judgment_lock:
            self.pending_judgments.clear()

    def _extract_score_value(self, score_data: Any) -> float:
        """Extract a numeric score from various possible formats."""
        if isinstance(score_data, (int, float)):
            return float(score_data)
        elif isinstance(score_data, dict):
            if 'score' in score_data:
                return float(score_data['score'])
            for key in ['value', 'rating', 'overall']:
                if key in score_data:
                    return float(score_data[key])
        elif hasattr(score_data, 'score'):
            return float(score_data.score)
        return 0.0

    def run(
        self,
        instruction: str,
        n: int,
        stop_event: threading.Event | None = None,
        pause_event: threading.Event | None = None,
    ) -> None:
        """Run simulation with separate conversation and PARALLEL judging phases."""
        run_no = utils.increment_run_number()
        self.wizard.set_run(run_no)
        token_tracker.set_run(run_no)

        self.logger.log_event("system_start", instruction=instruction, n=n, run_no=run_no)

        print(f"\n{'='*80}")
        print(f"Starting Run #{run_no}")
        print(f"Population size: {n}")
        print(f"Wizard goal: {self.wizard.goal}")
        print(f"Instruction: {instruction}")
        print(f"Parallel judge workers: {self.max_judge_workers}")
        print(f"{'='*80}\n")

        # PHASE 1: Generate population specs
        print(f"{'='*60}")
        print("PHASE 1: GENERATING POPULATION")
        print(f"{'='*60}")

        specs = self.generator.generate(instruction, n)
        print(f"Generated {len(specs)} population specifications\n")

        summary: List[dict] = []

        # Parse improvement schedule
        schedule = self._parse_schedule(n)
        improvement_points = set(schedule)
        
        if improvement_points:
            print(f"[SYSTEM] Wizard improvement scheduled after conversations: {sorted(improvement_points)}")
        else:
            print(f"[SYSTEM] No wizard improvements scheduled")

        def run_conversation(pop, conv_index):
            """Run a conversation and submit for parallel judgment."""
            if stop_event and stop_event.is_set():
                return
            if pause_event:
                while pause_event.is_set():
                    time.sleep(0.5)

            print(f"\n{'='*60}")
            print(f"CONVERSATION {conv_index}/{n}: {pop.name} ({pop.agent_id})")
            print(f"{'='*60}")

            # PHASE 2A: Wizard converses WITHOUT waiting for judging
            print(f"\n[WIZARD] Starting conversation with {pop.name}...")
            print("-" * 40)

            try:
                log = self.wizard.converse_with(
                    pop, 
                    show_live=getattr(config, 'SHOW_LIVE_CONVERSATIONS', False)
                )
                print("-" * 40)
                print(f"[WIZARD] ✓ Conversation with {pop.name} completed.")
                print(f"         Total turns: {len(log.get('turns', []))}")
            except Exception as e:
                print(f"[ERROR] Conversation failed with {pop.name}: {e}")
                self.logger.log_event("conversation_error", agent_id=pop.agent_id, error=str(e))
                return

            # PHASE 2B: Submit for parallel judging (non-blocking)
            print(f"\n[JUDGE] → Submitting {pop.agent_id} for parallel evaluation...")
            self._submit_for_judgment(log, pop.agent_id)

            # Save conversation log immediately (without judge result)
            filename = f"{self.wizard.wizard_id}_{pop.agent_id}_{utils.get_timestamp().replace(':', '').replace('-', '')}.json"
            utils.save_conversation_log(log, filename)
            print(f"[SYSTEM] Conversation log saved: {filename}")

            # Check if we're at an improvement point
            if conv_index in improvement_points:
                print(f"\n{'*'*60}")
                print(f"[SYSTEM] 🔄 IMPROVEMENT CHECKPOINT after conversation {conv_index}")
                print(f"{'*'*60}")
                
                # Wait for all pending judgments before improvement
                self._wait_for_pending_judgments()
                
                # Check if wizard should improve
                print(f"\n[SYSTEM] Checking if wizard should improve...")
                print(f"[SYSTEM] Wizard conversation count: {self.wizard.conversation_count}")
                print(f"[SYSTEM] Judged conversations in buffer: {sum(1 for log in self.wizard.history_buffer if 'judge_result' in log)}")
                
                if self.wizard._should_self_improve():
                    print(f"[SYSTEM] 🚀 TRIGGERING WIZARD IMPROVEMENT")
                    try:
                        self.wizard.self_improve()
                        print(f"[SYSTEM] ✅ Wizard improvement completed successfully")
                    except Exception as e:
                        print(f"[SYSTEM] ❌ Wizard improvement failed: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    print(f"[SYSTEM] ⏳ Conditions not met for improvement yet")

            print(f"\n{'='*60}")
            print(f"CONVERSATION {conv_index}/{n} WORKFLOW COMPLETE")
            print(f"{'='*60}\n")

        # PHASE 2: Generate population agents and run conversations
        print(f"\n{'='*60}")
        print("PHASE 2: CREATING AGENTS AND RUNNING CONVERSATIONS")
        print(f"{'='*60}")

        agents = []
        for idx, spec in enumerate(specs, start=1):
            print(f"\n[GOD] Creating agent {idx}/{n}...")
            agent = self.god.spawn_population_from_spec(spec, run_no, idx)
            print(f"[GOD] Created: {agent.name} ({agent.agent_id})")
            print(f"      Age: {agent.age}, Occupation: {agent.occupation}")
            print(f"      Goals: {agent.initial_goals}")
            agents.append((agent, idx))

        if getattr(config, 'PARALLEL_CONVERSATIONS', False):
            print(f"\n[SYSTEM] Running {len(agents)} conversations in parallel...")
            batch_threads = []
            for agent, idx in agents:
                t = threading.Thread(target=run_conversation, args=(agent, idx))
                batch_threads.append(t)
                t.start()
            for t in batch_threads:
                t.join()
            print(f"[SYSTEM] All parallel conversations completed.")
        else:
            print(f"\n[SYSTEM] Running {len(agents)} conversations sequentially...")
            for agent, idx in agents:
                run_conversation(agent, idx)

        # PHASE 3: Final judgment collection and summary
        print(f"\n{'='*60}")
        print("PHASE 3: FINALIZING JUDGMENTS AND SAVING RESULTS")
        print(f"{'='*60}")
        
        # Wait for any remaining judgments
        print(f"\n[SYSTEM] Waiting for final judgments to complete...")
        self._wait_for_pending_judgments()
        
        # Give a bit more time for queue processing
        time.sleep(1)
        
        # Compile final summary with judge results
        print(f"\n[SYSTEM] Compiling final summary with judge results...")
        
        with self.judgment_lock:
            completed_count = len(self.completed_judgments)
            
        print(f"[SYSTEM] Collected {completed_count} judge evaluations")
        
        # Create summary entries
        for agent, idx in agents:
            spec = agent.get_spec()
            
            # Get judge result if available
            judge_result = None
            with self.judgment_lock:
                if agent.agent_id in self.completed_judgments:
                    judge_result = self.completed_judgments[agent.agent_id]["judge_result"]
            
            if judge_result:
                entry = {
                    "pop_agent_id": agent.agent_id,
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
                    "judge_consensus": judge_result.get("judge_consensus", True) if self.enable_multi_judge else True,
                }
            else:
                # No judge result available
                entry = {
                    "pop_agent_id": agent.agent_id,
                    "name": spec.get("name"),
                    "personality_description": spec.get("personality_description"),
                    "age": spec.get("age"),
                    "occupation": spec.get("occupation"),
                    "initial_goals": spec.get("initial_goals"),
                    "memory_summary": spec.get("memory_summary"),
                    "system_instruction": spec.get("system_instruction"),
                    "temperature": spec.get("llm_settings", {}).get("temperature"),
                    "max_tokens": spec.get("llm_settings", {}).get("max_tokens"),
                    "success": None,
                    "goal_completion": None,
                    "coherence": None,
                    "tone": None,
                    "score": None,
                    "judge_consensus": None,
                    "error": "No judge result available"
                }
            
            summary.append(entry)

        # Save summary
        utils.save_conversation_log(summary, f"summary_{run_no}.json")
        self.logger.log_event("system_end", run_no=run_no)

        print(f"\n{'='*80}")
        print(f"RUN #{run_no} COMPLETE!")
        print(f"{'='*80}")
        print(f"Total conversations: {len(summary)}")
        
        # Calculate average score for successfully judged conversations
        scored_entries = [e for e in summary if e['score'] is not None]
        if scored_entries:
            avg_score = sum(e['score'] for e in scored_entries) / len(scored_entries)
            print(f"Average score: {avg_score:.2f} (from {len(scored_entries)} judged conversations)")
        else:
            print(f"No scored conversations available")
            
        successful_count = sum(1 for e in summary if e['success'] is True)
        print(f"Successful conversations: {successful_count}")
        print(f"Summary saved to: logs/summary_{run_no}.json")
        print(f"{'='*80}")

        # Print judge performance summary
        self._print_judge_performance()

    def _aggregate_judge_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Aggregate results from multiple judges."""
        if not results:
            return {}

        goal_scores = [self._extract_score_value(r.get("goal_completion", 0)) for r in results]
        coherence_scores = [self._extract_score_value(r.get("coherence", 0)) for r in results]
        tone_scores = [self._extract_score_value(r.get("tone", 0)) for r in results]
        overall_scores = [self._extract_score_value(r.get("overall", r.get("score", 0))) for r in results]
        success_votes = [r.get("success", False) for r in results]

        aggregated = {
            "goal_completion": sum(goal_scores) / len(goal_scores),
            "coherence": sum(coherence_scores) / len(coherence_scores),
            "tone": sum(tone_scores) / len(tone_scores),
            "overall": sum(overall_scores) / len(overall_scores),
            "success": sum(success_votes) > len(success_votes) / 2,
            "score": sum(overall_scores) / len(overall_scores),
        }

        aggregated["judge_consensus"] = all(success_votes) or not any(success_votes)
        aggregated["number_of_judges"] = len(results)
        aggregated["score_variance"] = self._calculate_variance(overall_scores)
        aggregated["confidence"] = 1.0 - (aggregated["score_variance"] * 2)
        rationales = [f"{r.get('judge_id', 'Unknown')}: {r.get('rationale', 'No rationale')}" for r in results]
        aggregated["rationale"] = " | ".join(rationales)
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

    def _parse_schedule(self, total: int) -> List[int]:
        """Parse the self-improvement schedule."""
        sched = getattr(config, 'SELF_IMPROVE_AFTER', 0)
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
    
    def shutdown(self):
        """Cleanup resources."""
        print("[SYSTEM] Shutting down parallel judging system...")
        
        # Signal shutdown
        self._shutdown = True
        
        # Signal judgment processor to stop
        self.judgment_queue.put(None)
        
        # Shutdown thread pool
        self.judge_executor.shutdown(wait=True)
        
        # Wait for processor thread
        if self.judgment_processor.is_alive():
            self.judgment_processor.join(timeout=5)
            
        print("[SYSTEM] Shutdown complete")
