"""Enhanced module for improving wizard prompts using DSPy optimizers with MIPROv2 and template-driven synthetic data generation."""
from __future__ import annotations

from typing import List, Dict, Any, Tuple
import re
import json
import random
import os

import config
import utils

try:
    import dspy
    from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2 as OptimizePrompts
    from dspy.teleprompt import BootstrapFewShot
except Exception:  # pragma: no cover - DSPy optional
    dspy = None


if dspy is not None:

    class ImproveSignature(dspy.Signature):
        """Signature for generating a better research wizard system prompt."""

        current_prompt: str = dspy.InputField(desc="Current wizard system prompt that needs improvement")
        conversation_examples: str = dspy.InputField(desc="Examples of conversations with judge evaluations")
        goal: str = dspy.InputField(desc="The wizard's research goal and objectives")
        performance_issues: str = dspy.InputField(desc="Specific issues identified in current performance")
        improved_prompt: str = dspy.OutputField(desc="Improved wizard system prompt addressing the identified issues")


    class WizardImprover(dspy.Module):
        """Enhanced wizard improver using Chain of Thought for better prompt optimization."""

        def __init__(self) -> None:
            super().__init__()
            self.improve = dspy.ChainOfThought(ImproveSignature)

        def forward(self, current_prompt: str, conversation_examples: str, goal: str, performance_issues: str) -> dspy.Prediction:
            return self.improve(
                current_prompt=current_prompt,
                conversation_examples=conversation_examples,
                goal=goal,
                performance_issues=performance_issues
            )


    class TemplateBasedSyntheticDataGenerator:
        """Generate synthetic conversation examples using configurable templates."""
        
        def __init__(self, 
                     scenarios_template_path: str = "templates/synthetic_scenarios.json",
                     conversation_template_path: str = "templates/synthetic_conversation_templates.json"):
            self.scenarios_template_path = scenarios_template_path
            self.conversation_template_path = conversation_template_path
            self._load_templates()
        
        def _load_templates(self):
            """Load all template files."""
            try:
                with open(self.scenarios_template_path, 'r', encoding='utf-8') as f:
                    self.scenarios_config = json.load(f)
                print(f"[SYNTHETIC] Loaded scenarios from {self.scenarios_template_path}")
            except Exception as e:
                print(f"[SYNTHETIC] Warning: Could not load scenarios template: {e}")
                self.scenarios_config = self._get_fallback_scenarios()
            
            try:
                with open(self.conversation_template_path, 'r', encoding='utf-8') as f:
                    self.conversation_config = json.load(f)
                print(f"[SYNTHETIC] Loaded conversation templates from {self.conversation_template_path}")
            except Exception as e:
                print(f"[SYNTHETIC] Warning: Could not load conversation template: {e}")
                self.conversation_config = self._get_fallback_conversations()
        
        def _get_fallback_scenarios(self) -> Dict[str, Any]:
            """Fallback scenarios if template file is missing."""
            return {
                "hearing_loss_scenarios": ["hearing loss experience"],
                "research_topics": ["hearing loss research"],
                "participant_names": ["Participant"],
                "conversation_quality_indicators": {
                    "good_wizard_behaviors": ["asks questions"],
                    "poor_wizard_behaviors": ["doesn't engage"],
                    "good_outcomes": ["participant responds"],
                    "poor_outcomes": ["brief responses"]
                }
            }
        
        def _get_fallback_conversations(self) -> Dict[str, Any]:
            """Fallback conversation templates if file is missing."""
            return {
                "conversation_templates": {
                    "good_conversation": {
                        "opening": {
                            "participant": "Hello, I'm interested in your research.",
                            "wizard": "Thank you for participating. Can you tell me about your experience?"
                        }
                    },
                    "poor_conversation": {
                        "opening": {
                            "participant": "Hi.",
                            "wizard": "Tell me about hearing loss."
                        }
                    }
                },
                "response_patterns": {
                    "situation_types": ["social situations"],
                    "emotional_impacts": ["frustrated"],
                    "specific_challenges": ["It's difficult."],
                    "validation_statements": ["I understand."],
                    "follow_up_questions": ["what else?"],
                    "brief_responses": ["I don't know."]
                }
            }

        def generate_synthetic_example(self, target_score: float) -> Dict[str, Any]:
            """Generate a synthetic conversation example with target performance using templates."""
            quality = "good" if target_score > 0.7 else "poor"
            
            # Select random elements from templates
            scenario = random.choice(self.scenarios_config["hearing_loss_scenarios"])
            topic = random.choice(self.scenarios_config["research_topics"])
            name = random.choice(self.scenarios_config["participant_names"])
            
            # Get conversation template
            conv_template = self.conversation_config["conversation_templates"][f"{quality}_conversation"]
            response_patterns = self.conversation_config["response_patterns"]
            
            # Build conversation using templates
            turns = []
            
            # Opening exchange
            opening = conv_template["opening"]
            participant_opening = opening["participant"].format(
                name=name, 
                scenario=scenario
            )
            wizard_opening = opening["wizard"].format(
                name=name,
                scenario_context=scenario.replace("recently diagnosed with", "your").replace("has ", "your ")
            )
            
            turns.extend([
                {"speaker": "pop", "text": participant_opening},
                {"speaker": "wizard", "text": wizard_opening}
            ])
            
            # Middle exchanges
            if "middle_exchanges" in conv_template:
                for exchange in conv_template["middle_exchanges"]:
                    if quality == "good":
                        situation = random.choice(response_patterns["situation_types"])
                        impact = random.choice(response_patterns["emotional_impacts"])
                        challenge = random.choice(response_patterns["specific_challenges"])
                        validation = random.choice(response_patterns["validation_statements"])
                        follow_up = random.choice(response_patterns["follow_up_questions"])
                        
                        participant_text = exchange["participant"].format(
                            situation_type=situation,
                            emotional_impact=impact,
                            specific_challenge_description=challenge
                        )
                        wizard_text = exchange["wizard"].format(
                            situation_type=situation,
                            research_topic=topic,
                            validation_statement=validation,
                            follow_up_question=follow_up
                        )
                    else:
                        participant_text = exchange.get("participant", random.choice(response_patterns["brief_responses"]))
                        wizard_text = exchange.get("wizard", "What else?")
                    
                    turns.extend([
                        {"speaker": "pop", "text": participant_text},
                        {"speaker": "wizard", "text": wizard_text}
                    ])
            
            # Closing exchange
            if "closing" in conv_template:
                closing = conv_template["closing"]
                if quality == "good":
                    research_plan = self.conversation_config.get("research_plan_templates", {}).get("comprehensive_plan", {})
                    wizard_closing = closing["wizard"].format(
                        identified_themes="communication barriers and social participation",
                        research_plan_json=json.dumps(research_plan, indent=2)
                    )
                    participant_closing = closing.get("participant", "That sounds helpful.")
                else:
                    inadequate_plan = self.conversation_config.get("research_plan_templates", {}).get("inadequate_plan", {})
                    wizard_closing = closing["wizard"].format(
                        inadequate_plan=json.dumps(inadequate_plan)
                    )
                    participant_closing = closing.get("participant", "Um, okay.")
                
                turns.extend([
                    {"speaker": "wizard", "text": wizard_closing},
                    {"speaker": "pop", "text": participant_closing}
                ])
            
            # Generate rationale based on quality indicators
            if quality == "good":
                behaviors = self.scenarios_config["conversation_quality_indicators"]["good_wizard_behaviors"]
                outcomes = self.scenarios_config["conversation_quality_indicators"]["good_outcomes"]
                rationale = f"Wizard demonstrated {random.choice(behaviors)} and achieved {random.choice(outcomes)}."
            else:
                behaviors = self.scenarios_config["conversation_quality_indicators"]["poor_wizard_behaviors"]
                outcomes = self.scenarios_config["conversation_quality_indicators"]["poor_outcomes"]
                rationale = f"Wizard exhibited {random.choice(behaviors)} resulting in {random.choice(outcomes)}."
            
            return {
                "turns": turns,
                "judge_result": {
                    "goal_completion": target_score,
                    "coherence": target_score + random.uniform(-0.1, 0.1),
                    "tone": target_score + random.uniform(-0.1, 0.1),
                    "overall": target_score,
                    "success": target_score > 0.6,
                    "rationale": rationale
                },
                "synthetic": True,
                "scenario": scenario,
                "topic": topic
            }

        def generate_synthetic_dataset(self, size: int, settings: Dict[str, Any] = None) -> List[Dict[str, Any]]:
            """Generate a balanced synthetic dataset using template settings."""
            if settings is None:
                # Load settings from improvement prompts template
                try:
                    settings_path = "templates/improvement_prompts.json"
                    with open(settings_path, 'r', encoding='utf-8') as f:
                        improvement_config = json.load(f)
                        settings = improvement_config.get("synthetic_data_settings", {})
                except Exception:
                    settings = {"good_example_ratio": 0.6}
            
            examples = []
            good_ratio = settings.get("good_example_ratio", 0.6)
            
            # Generate a mix of good and poor examples
            good_count = int(size * good_ratio)
            poor_count = size - good_count
            
            print(f"[SYNTHETIC] Generating {good_count} good examples and {poor_count} poor examples")
            
            for _ in range(good_count):
                score = random.uniform(0.7, 1.0)
                examples.append(self.generate_synthetic_example(score))
            
            for _ in range(poor_count):
                score = random.uniform(0.0, 0.6)
                examples.append(self.generate_synthetic_example(score))
            
            return examples


    def _extract_instructions(program: object) -> str:
        """Return the instructions string from a candidate program."""
        # If ``program`` is a compiled module, try grabbing the instructions
        sig = getattr(program, "signature", None)
        if sig is not None and hasattr(sig, "instructions"):
            text = sig.instructions
            if isinstance(text, str) and text.strip():
                cleaned = text.strip()
                if cleaned == "Signature for generating a better research wizard system prompt.":
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


    def analyze_performance_issues(logs: List[Dict[str, Any]], 
                                 template_path: str = "templates/performance_analysis_template.txt") -> str:
        """Analyze conversation logs to identify specific performance issues using template."""
        if not logs:
            return "No performance data available for analysis."
        
        # Calculate performance metrics
        scores = []
        low_scoring_examples = []
        
        for log in logs:
            judge_result = log.get("judge_result", {})
            goal_completion = judge_result.get("goal_completion", 0)
            coherence = judge_result.get("coherence", 0)
            tone = judge_result.get("tone", 0)
            overall = judge_result.get("overall", judge_result.get("score", 0))
            
            # Handle nested score objects
            if isinstance(goal_completion, dict):
                goal_completion = goal_completion.get("score", 0)
            if isinstance(coherence, dict):
                coherence = coherence.get("score", 0)
            if isinstance(tone, dict):
                tone = tone.get("score", 0)
            if isinstance(overall, dict):
                overall = overall.get("score", 0)
            
            scores.append({
                "goal_completion": float(goal_completion),
                "coherence": float(coherence),
                "tone": float(tone),
                "overall": float(overall)
            })
            
            # Collect low-scoring examples
            if overall < 0.6:
                rationale = judge_result.get("rationale", "No rationale provided")
                low_scoring_examples.append({"rationale": rationale[:150] + "..." if len(rationale) > 150 else rationale})
        
        # Calculate averages
        avg_goal = sum(s["goal_completion"] for s in scores) / len(scores)
        avg_coherence = sum(s["coherence"] for s in scores) / len(scores)
        avg_tone = sum(s["tone"] for s in scores) / len(scores)
        avg_overall = sum(s["overall"] for s in scores) / len(scores)
        
        # Prepare template variables
        template_vars = {
            "total_conversations": len(logs),
            "avg_goal": f"{avg_goal:.2f}",
            "avg_coherence": f"{avg_coherence:.2f}",
            "avg_tone": f"{avg_tone:.2f}",
            "avg_overall": f"{avg_overall:.2f}",
            "if_low_goal": avg_goal < 0.7,
            "if_low_coherence": avg_coherence < 0.7,
            "if_low_tone": avg_tone < 0.7,
            "if_low_overall": avg_overall < 0.7,
            "low_scoring_examples": low_scoring_examples[:5],  # Limit to 5 examples
            "improvement_suggestions": [],
            "focus_areas": []
        }
        
        # Add improvement suggestions based on scores
        if avg_goal < 0.7:
            template_vars["improvement_suggestions"].append({
                "suggestion": "Enhance research plan generation with more comprehensive topic coverage"
            })
            template_vars["focus_areas"].append({
                "area": "Research Planning",
                "description": "Improve ability to create structured, comprehensive research plans"
            })
        
        if avg_coherence < 0.7:
            template_vars["improvement_suggestions"].append({
                "suggestion": "Improve JSON structure and logical flow in responses"
            })
            template_vars["focus_areas"].append({
                "area": "Output Structure",
                "description": "Ensure valid JSON format and logical organization"
            })
        
        if avg_tone < 0.7:
            template_vars["improvement_suggestions"].append({
                "suggestion": "Use more empathetic and validating language when interacting with participants"
            })
            template_vars["focus_areas"].append({
                "area": "Communication Style",
                "description": "Develop more empathetic and professional interaction patterns"
            })
        
        # Load and render template
        try:
            template_content = utils.load_template(template_path)
            # Simple template rendering (replace {{variable}} patterns)
            result = template_content
            for key, value in template_vars.items():
                if isinstance(value, bool):
                    # Handle conditional blocks
                    if value:
                        # Keep the content between {{#if_variable}} and {{/if_variable}}
                        pattern = f"{{{{#if_{key[3:] if key.startswith('if_') else key}}}}}(.*?){{{{/if_{key[3:] if key.startswith('if_') else key}}}}}"
                        result = re.sub(pattern, r'\1', result, flags=re.DOTALL)
                    else:
                        # Remove the content between {{#if_variable}} and {{/if_variable}}
                        pattern = f"{{{{#if_{key[3:] if key.startswith('if_') else key}}}}}.*?{{{{/if_{key[3:] if key.startswith('if_') else key}}}}}"
                        result = re.sub(pattern, '', result, flags=re.DOTALL)
                elif isinstance(value, list):
                    # Handle each loops
                    if key in ["low_scoring_examples", "improvement_suggestions", "focus_areas"]:
                        pattern = f"{{{{#each {key}}}}}(.*?){{{{/each}}}}"
                        match = re.search(pattern, result, re.DOTALL)
                        if match:
                            item_template = match.group(1)
                            items_text = ""
                            for item in value:
                                item_text = item_template
                                for item_key, item_value in item.items():
                                    item_text = item_text.replace(f"{{{{{item_key}}}}}", str(item_value))
                                items_text += item_text
                            result = re.sub(pattern, items_text, result, flags=re.DOTALL)
                else:
                    # Simple variable replacement
                    result = result.replace(f"{{{{{key}}}}}", str(value))
            
            # Clean up any remaining template syntax
            result = re.sub(r'\{\{#.*?\}\}.*?\{\{/.*?\}\}', '', result, flags=re.DOTALL)
            result = re.sub(r'\{\{.*?\}\}', '', result)
            
            return result.strip()
            
        except Exception as e:
            print(f"[PERFORMANCE ANALYSIS] Could not load template {template_path}: {e}")
            # Fallback to simple text analysis
            issues = []
            if avg_goal < 0.7:
                issues.append(f"Low goal completion score ({avg_goal:.2f}): Wizard struggles to generate comprehensive research plans")
            if avg_coherence < 0.7:
                issues.append(f"Low coherence score ({avg_coherence:.2f}): Wizard produces poorly structured outputs")
            if avg_tone < 0.7:
                issues.append(f"Low tone score ({avg_tone:.2f}): Wizard needs more empathetic communication")
            if avg_overall < 0.7:
                issues.append(f"Low overall performance ({avg_overall:.2f}): General improvement needed")
            
            return "\n".join(issues) if issues else "Performance appears to be good across all metrics."


    def build_dataset(history: List[Dict[str, Any]], 
                     min_size: int = None,
                     settings_path: str = "templates/improvement_prompts.json") -> Tuple[List[dspy.Example], bool]:
        """Convert conversation history into a DSPy dataset, augmenting with synthetic data if needed."""
        
        # Load settings from template
        try:
            with open(settings_path, 'r', encoding='utf-8') as f:
                improvement_config = json.load(f)
                synthetic_settings = improvement_config.get("synthetic_data_settings", {})
                min_size = min_size or synthetic_settings.get("minimum_dataset_size", 8)
        except Exception as e:
            print(f"[DATASET] Warning: Could not load settings from {settings_path}: {e}")
            min_size = min_size or 8
            synthetic_settings = {}
        
        real_examples = []
        synthetic_generator = TemplateBasedSyntheticDataGenerator()
        
        # Process real conversation logs
        for log in history:
            if 'judge_result' not in log:
                continue
                
            transcript = "\n".join(f"{t['speaker']}: {t['text']}" for t in log.get('turns', []))
            judge = log.get("judge_result", {})
            
            # Extract numeric scores (handle nested objects)
            def extract_score(score_data):
                if isinstance(score_data, dict):
                    return score_data.get("score", 0)
                return score_data or 0
            
            goal_score = extract_score(judge.get("goal_completion", 0))
            coherence_score = extract_score(judge.get("coherence", 0))
            tone_score = extract_score(judge.get("tone", 0))
            overall_score = extract_score(judge.get("overall", judge.get("score", 0)))
            
            # Create detailed conversation example with judge feedback
            judge_feedback = f"""
JUDGE EVALUATION:
- Goal Completion: {goal_score:.2f}
- Coherence: {coherence_score:.2f}
- Tone: {tone_score:.2f}
- Overall Score: {overall_score:.2f}
- Success: {judge.get('success', False)}
- Rationale: {judge.get('rationale', 'No rationale provided')}
"""
            
            ex = dspy.Example(
                current_prompt=log.get("prompt", ""),
                conversation_examples=f"CONVERSATION:\n{transcript}\n{judge_feedback}",
                goal=log.get("goal", config.WIZARD_DEFAULT_GOAL),
                performance_issues=judge.get('rationale', ''),
                score=float(overall_score)
            ).with_inputs("current_prompt", "conversation_examples", "goal", "performance_issues")
            
            real_examples.append(ex)
        
        print(f"[DATASET] Built {len(real_examples)} examples from real conversations")
        
        # Check if we need synthetic data
        needs_synthetic = len(real_examples) < min_size
        dataset = real_examples.copy()
        
        if needs_synthetic:
            synthetic_needed = min_size - len(real_examples)
            max_synthetic = synthetic_settings.get("max_synthetic_examples", 20)
            synthetic_needed = min(synthetic_needed, max_synthetic)
            
            print(f"[DATASET] Generating {synthetic_needed} synthetic examples to reach minimum dataset size")
            
            synthetic_logs = synthetic_generator.generate_synthetic_dataset(synthetic_needed, synthetic_settings)
            
            # Convert synthetic logs to DSPy examples
            for syn_log in synthetic_logs:
                transcript = "\n".join(f"{t['speaker']}: {t['text']}" for t in syn_log['turns'])
                judge = syn_log['judge_result']
                
                judge_feedback = f"""
JUDGE EVALUATION:
- Goal Completion: {judge['goal_completion']:.2f}
- Coherence: {judge['coherence']:.2f}
- Tone: {judge['tone']:.2f}
- Overall Score: {judge['overall']:.2f}
- Success: {judge['success']}
- Rationale: {judge['rationale']}
"""
                
                # Use current prompt from real examples or default
                current_prompt = real_examples[0].current_prompt if real_examples else utils.load_template(config.WIZARD_PROMPT_TEMPLATE_PATH)
                
                ex = dspy.Example(
                    current_prompt=current_prompt,
                    conversation_examples=f"SYNTHETIC CONVERSATION:\n{transcript}\n{judge_feedback}",
                    goal=config.WIZARD_DEFAULT_GOAL,
                    performance_issues=judge['rationale'],
                    score=float(judge['overall'])
                ).with_inputs("current_prompt", "conversation_examples", "goal", "performance_issues")
                
                dataset.append(ex)
        
        print(f"[DATASET] Final dataset size: {len(dataset)} examples ({len(real_examples)} real, {len(dataset) - len(real_examples)} synthetic)")
        
        return dataset, needs_synthetic


    def train_improver(dataset: List[dspy.Example], 
                      current_prompt: str,
                      settings_path: str = "templates/improvement_prompts.json") -> Tuple[WizardImprover, Dict[str, Any]]:
        """Train a WizardImprover using template-driven configuration."""

        # Load settings from template
        try:
            with open(settings_path, 'r', encoding='utf-8') as f:
                improvement_config = json.load(f)
                research_keywords = improvement_config.get("research_elements_keywords", [])
                improvement_indicators = improvement_config.get("improvement_indicators", [])
                scoring_weights = improvement_config.get("scoring_weights", {})
                optimizer_settings = improvement_config.get("optimizer_settings", {})
        except Exception as e:
            print(f"[TRAINING] Warning: Could not load settings from {settings_path}: {e}")
            # Fallback settings
            research_keywords = ["research", "interview", "hearing loss"]
            improvement_indicators = ["improve", "better", "enhance"]
            scoring_weights = {"research_elements_bonus": 0.02, "improvement_indicators_bonus": 0.01, "total_bonus_cap": 0.3}
            optimizer_settings = {}

        if dspy.settings.lm is None:
            print("[TRAINING] Configuring DSPy LM...")
            dspy.settings.configure(
                lm=dspy.LM(
                    model=config.LLM_MODEL,
                    temperature=0.3,  # Lower temperature for more consistent improvement
                    max_tokens=2048,  # More tokens for detailed prompts
                )
            )

        def metric(example: dspy.Example, pred: dspy.Prediction, trace: object | None = None) -> float:
            """Template-driven scoring function for improved prompts."""
            
            # Base score from the example
            base_score = getattr(example, 'score', 0.5)
            
            improved_prompt = getattr(pred, 'improved_prompt', '')
            
            if not improved_prompt or len(improved_prompt.strip()) < 100:
                return 0.0  # Reject very short prompts
            
            # Bonus for including research-specific elements (from template)
            element_bonus = sum(
                scoring_weights.get("research_elements_bonus", 0.02) 
                for elem in research_keywords 
                if elem.lower() in improved_prompt.lower()
            )
            
            # Bonus for addressing specific issues (from template)
            issue_bonus = sum(
                scoring_weights.get("improvement_indicators_bonus", 0.01) 
                for word in improvement_indicators 
                if word.lower() in improved_prompt.lower()
            )
            
            # Penalty for being too similar to original (encourage innovation)
            original_words = set(example.current_prompt.lower().split())
            improved_words = set(improved_prompt.lower().split())
            similarity = len(original_words & improved_words) / max(len(original_words), len(improved_words), 1)
            innovation_bonus = max(0, scoring_weights.get("innovation_bonus_max", 0.1) - similarity * 0.1)
            
            # Cap bonuses (from template)
            total_bonus = min(
                element_bonus + issue_bonus + innovation_bonus, 
                scoring_weights.get("total_bonus_cap", 0.3)
            )
            
            final_score = base_score + total_bonus
            return min(final_score, 1.0)

        improver_module = WizardImprover()
        
        # Choose optimizer based on dataset size and template settings
        if len(dataset) >= config.DSPY_MIPRO_MINIBATCH_SIZE:
            print(f"[TRAINING] Using MIPROv2 optimizer with {len(dataset)} examples")
            
            # Get MIPROv2 settings from template
            mipro_settings = optimizer_settings.get("mipro_v2", {})
            
            # Calculate validation set size and adjust minibatch if needed
            val_size = max(1, int(len(dataset) * 0.2))  # 20% for validation
            minibatch_size = min(config.DSPY_MIPRO_MINIBATCH_SIZE, val_size)
            
            optimizer = OptimizePrompts(
                metric=metric,
                num_candidates=mipro_settings.get("num_candidates", 8),
                init_temperature=mipro_settings.get("init_temperature", 1.0),
                verbose=True,
                track_stats=mipro_settings.get("track_stats", True)
            )
            method = "MIPROv2"
            
            try:
                print(f"[TRAINING] Starting MIPROv2 training with minibatch_size={minibatch_size}")
                trained = optimizer.compile(
                    improver_module,
                    trainset=dataset,
                    num_trials=config.DSPY_TRAINING_ITER * 2,  # More trials for MIPROv2
                    minibatch_size=minibatch_size,
                    require_training_set=mipro_settings.get("require_training_set", True)
                )
                
                print("[TRAINING] MIPROv2 training completed successfully")
                
            except Exception as e:
                print(f"[TRAINING] MIPROv2 failed: {e}, falling back to BootstrapFewShot")
                bootstrap_settings = optimizer_settings.get("bootstrap_few_shot", {})
                optimizer = BootstrapFewShot(
                    metric=metric, 
                    max_bootstrapped_demos=bootstrap_settings.get("max_bootstrapped_demos", 5)
                )
                method = "BootstrapFewShot (MIPROv2 fallback)"
                trained = optimizer.compile(improver_module, trainset=dataset)
                
        elif len(dataset) >= config.DSPY_BOOTSTRAP_MINIBATCH_SIZE:
            print(f"[TRAINING] Using BootstrapFewShot optimizer with {len(dataset)} examples")
            bootstrap_settings = optimizer_settings.get("bootstrap_few_shot", {})
            optimizer = BootstrapFewShot(
                metric=metric, 
                max_bootstrapped_demos=bootstrap_settings.get("max_bootstrapped_demos", 3)
            )
            method = "BootstrapFewShot"
            trained = optimizer.compile(improver_module, trainset=dataset)
            
        else:
            print(f"[TRAINING] Using COPRO optimizer with {len(dataset)} examples")
            copro_settings = optimizer_settings.get("copro", {})
            optimizer = dspy.COPRO(
                metric=metric, 
                breadth=copro_settings.get("breadth", 5)
            )
            method = "COPRO"
            trained = optimizer.compile(
                improver_module,
                trainset=dataset,
                eval_kwargs={"display_progress": True}
            )

        # Extract best results
        candidates = getattr(trained, "candidate_programs", [])
        if candidates:
            best = max(candidates, key=lambda c: c.get("score", 0))
            best_prompt = _extract_instructions(best.get("program", ""))
            best_score = best.get("score", 0)
            print(f"[TRAINING] Best candidate score: {best_score:.3f}")
        else:
            best_prompt = _extract_instructions(trained)
            best_score = 0
            print("[TRAINING] No candidates found, using trained model")

        # Fallback to current prompt if extraction failed
        if not best_prompt or len(best_prompt.strip()) < 100:
            print("[TRAINING] Prompt extraction failed, using current prompt")
            best_prompt = current_prompt

        metrics = {
            "best_score": best_score,
            "iterations": len(candidates),
            "best_prompt": best_prompt,
            "method": method,
            "dataset_size": len(dataset),
            "training_successful": True,
            "template_settings_used": settings_path
        }
        
        return trained, metrics

else:  # DSPy not available
    WizardImprover = None
    TemplateBasedSyntheticDataGenerator = None
    build_dataset = None
    train_improver = None
    analyze_performance_issues = None
