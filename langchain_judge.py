"""Enhanced LLM-as-Judge implementation using LangChain evaluators and best practices."""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from tracking_chat_openai import TrackingChatOpenAI as ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field, validator

import config
import utils


class EvaluationStrategy(Enum):
    """Different evaluation strategies for the judge."""
    SINGLE_SCORE = "single_score"
    CRITERIA_BASED = "criteria_based"
    PAIRWISE = "pairwise"
    REFERENCE_BASED = "reference_based"


class CriterionScore(BaseModel):
    """Individual criterion score with reasoning."""
    score: float = Field(ge=0.0, le=1.0, description="Score between 0 and 1")
    reasoning: str = Field(description="Explanation for the score")
    evidence: Optional[List[str]] = Field(default=None, description="Specific quotes supporting the score")


class EvaluationResult(BaseModel):
    """Structured evaluation result."""
    goal_completion: CriterionScore
    coherence: CriterionScore
    tone: CriterionScore
    overall: float = Field(ge=0.0, le=1.0, description="Overall weighted score")
    success: bool = Field(description="Whether the conversation was successful")
    rationale: str = Field(description="Overall evaluation rationale")
    confidence: float = Field(ge=0.0, le=1.0, default=0.8, description="Judge's confidence in the evaluation")
    
    @validator('overall', pre=False, always=True)
    def calculate_overall(cls, v, values):
        """Calculate overall score if not provided."""
        if v is None or v == 0:
            scores = []
            weights = {
                'goal_completion': 0.5,  # Research plan quality is most important
                'coherence': 0.3,        # Structure and following schema
                'tone': 0.2              # Professional interaction
            }
            
            for criterion, weight in weights.items():
                if criterion in values and values[criterion]:
                    scores.append(values[criterion].score * weight)
            
            return sum(scores) if scores else 0.0
        return v


@dataclass
class EvaluationCriteria:
    """Defines evaluation criteria with detailed rubrics."""
    name: str
    description: str
    weight: float = 1.0
    rubric: Dict[str, str] = field(default_factory=dict)
    examples: Dict[str, List[str]] = field(default_factory=dict)
    
    def to_prompt_string(self) -> str:
        """Convert criteria to a formatted string for prompts."""
        parts = [f"**{self.name}**: {self.description}"]
        
        if self.rubric:
            parts.append("\nScoring rubric:")
            for score_range, description in sorted(self.rubric.items()):
                parts.append(f"  - {score_range}: {description}")
        
        if self.examples:
            parts.append("\nExamples:")
            for category, examples in self.examples.items():
                parts.append(f"  {category}:")
                for example in examples:
                    parts.append(f"    - {example}")
        
        return "\n".join(parts)


class LangChainJudge:
    """Enhanced judge using LangChain patterns and best practices."""
    
    # Default evaluation criteria for research planning
    DEFAULT_CRITERIA = [
        EvaluationCriteria(
            name="goal_completion",
            description="Evaluates whether the wizard produced a useful research plan addressing hearing loss topics",
            weight=0.5,
            rubric={
                "0.8-1.0": "Comprehensive plan covering all key topics with clear priorities and time allocation",
                "0.6-0.8": "Good plan covering most topics with reasonable structure",
                "0.4-0.6": "Basic plan with some gaps or unclear priorities",
                "0.0-0.4": "Inadequate plan missing key topics or structure"
            },
            examples={
                "good": [
                    "Plan includes all 4 main topics with clear subtopics",
                    "Time allocation reflects topic importance",
                    "COM-B/TDF frameworks properly integrated"
                ],
                "poor": [
                    "Missing key topics like treatment barriers",
                    "No clear prioritization or time allocation",
                    "Lacks theoretical framework integration"
                ]
            }
        ),
        EvaluationCriteria(
            name="coherence",
            description="Evaluates whether the plan follows the required JSON schema and is well-structured",
            weight=0.3,
            rubric={
                "0.8-1.0": "Perfect schema compliance with logical flow and clear topic IDs",
                "0.6-0.8": "Good structure with minor schema deviations",
                "0.4-0.6": "Basic structure but missing required fields",
                "0.0-0.4": "Poor structure or invalid JSON"
            }
        ),
        EvaluationCriteria(
            name="tone",
            description="Evaluates the wizard's professional and helpful communication style",
            weight=0.2,
            rubric={
                "0.8-1.0": "Consistently professional, empathetic, and helpful",
                "0.6-0.8": "Generally professional with minor lapses",
                "0.4-0.6": "Adequate but occasionally unprofessional",
                "0.0-0.4": "Unprofessional or unhelpful tone"
            }
        )
    ]
    
    def __init__(
        self,
        llm_settings: Optional[Dict[str, Any]] = None,
        criteria: Optional[List[EvaluationCriteria]] = None,
        strategy: EvaluationStrategy = EvaluationStrategy.CRITERIA_BASED,
        template_path: Optional[str] = None
    ):
        """Initialize the enhanced judge.
        
        Args:
            llm_settings: LLM configuration settings
            criteria: Custom evaluation criteria (uses defaults if None)
            strategy: Evaluation strategy to use
            template_path: Path to custom judge prompt template
        """
        self.llm_settings = llm_settings or {
            "model": config.LLM_MODEL,
            "temperature": 0.2,  # Lower temperature for more consistent judging
            "max_tokens": 1024,  # More tokens for detailed evaluation
        }
        
        self.llm = ChatOpenAI(
            model=self.llm_settings["model"],
            temperature=self.llm_settings["temperature"],
            max_tokens=self.llm_settings["max_tokens"],
            max_retries=config.OPENAI_MAX_RETRIES,
        )
        
        self.criteria = criteria or self.DEFAULT_CRITERIA
        self.strategy = strategy
        self.template_path = template_path or config.JUDGE_PROMPT_TEMPLATE_PATH
        
        # Output parser for structured results
        self.output_parser = JsonOutputParser(pydantic_object=EvaluationResult)
        
        # Cache for prompt templates
        self._prompt_cache: Dict[str, ChatPromptTemplate] = {}
    
    def _build_criteria_prompt(self) -> str:
        """Build the criteria section of the prompt."""
        criteria_parts = []
        for criterion in self.criteria:
            criteria_parts.append(criterion.to_prompt_string())
        return "\n\n".join(criteria_parts)
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the evaluation prompt template."""
        if self.strategy in self._prompt_cache:
            return self._prompt_cache[self.strategy]
        
        # Load base template
        base_template = utils.load_template(self.template_path)
        
        # Enhanced system prompt with best practices
        system_prompt = """You are an expert evaluator assessing research planning conversations about hearing loss. 
Your role is to provide consistent, objective evaluations based on clear criteria.

IMPORTANT GUIDELINES:
1. Base your evaluation ONLY on the provided transcript
2. Look for specific evidence to support your scores
3. Be consistent across evaluations
4. Provide constructive feedback in your rationale
5. Consider partial credit for attempts that show understanding but have execution issues

{criteria_section}

{format_instructions}

Remember: Your evaluation directly impacts the system's ability to improve. Be fair but rigorous."""
        
        # Build the complete prompt template
        if self.strategy == EvaluationStrategy.CRITERIA_BASED:
            template = ChatPromptTemplate.from_messages([
                ("system", system_prompt),
                ("human", """Evaluate this conversation:

GOAL: {goal}

TRANSCRIPT:
{transcript}

Provide a detailed evaluation for each criterion with evidence from the transcript.""")
            ])
        else:
            # Can extend with other strategies
            template = ChatPromptTemplate.from_messages([
                ("system", base_template),
                ("human", "Evaluate the conversation with goal '{goal}':\n\n{transcript}")
            ])
        
        self._prompt_cache[self.strategy] = template
        return template
    
    def _extract_evidence(self, transcript: str, score: float) -> List[str]:
        """Extract relevant quotes from transcript as evidence."""
        lines = transcript.split('\n')
        evidence = []
        
        # Simple heuristic: extract lines that might be relevant
        keywords = ['plan', 'research', 'hearing', 'topics', 'structure', 'JSON']
        for line in lines[-10:]:  # Focus on recent exchanges
            if any(keyword in line.lower() for keyword in keywords):
                evidence.append(line.strip())
        
        return evidence[:3]  # Limit to 3 most relevant quotes
    
    def assess(self, log: Dict[str, Any]) -> Dict[str, Any]:
        """Assess a conversation log with enhanced evaluation."""
        transcript = "\n".join(
            [f"{t['speaker']}: {t['text']}" for t in log.get("turns", [])]
        )
        goal = log.get("goal", config.WIZARD_DEFAULT_GOAL)
        
        # Create prompt
        prompt_template = self._create_prompt_template()
        
        # Format messages
        messages = prompt_template.format_messages(
            goal=goal,
            transcript=transcript,
            criteria_section=self._build_criteria_prompt(),
            format_instructions=self.output_parser.get_format_instructions()
        )
        
        # Get evaluation with retries for robustness
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(messages)
                
                # Parse structured output
                try:
                    result = self.output_parser.parse(response.content)
                    
                    # Convert Pydantic model to dict
                    result_dict = result.dict()
                    
                    # Add metadata
                    result_dict["evaluator_version"] = "2.0"
                    result_dict["strategy"] = self.strategy.value
                    result_dict["attempt"] = attempt + 1
                    
                    # Ensure backward compatibility
                    if "score" not in result_dict:
                        result_dict["score"] = result_dict["overall"]
                    
                    return result_dict
                    
                except Exception as parse_error:
                    if attempt < max_retries - 1:
                        continue
                    
                    # Fallback parsing
                    return self._fallback_parse(response.content, goal, transcript)
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    continue
                
                # Final fallback
                return self._create_fallback_result(str(e))
    
    def _fallback_parse(self, response: str, goal: str, transcript: str) -> Dict[str, Any]:
        """Fallback parsing when structured parsing fails."""
        try:
            # Try to extract JSON from the response
            parsed = utils.extract_json_object(response)
            if parsed and isinstance(parsed, dict):
                # Ensure required fields
                result = {
                    "goal_completion": parsed.get("goal_completion", 0.5),
                    "coherence": parsed.get("coherence", 0.5),
                    "tone": parsed.get("tone", 0.5),
                    "overall": parsed.get("overall", parsed.get("score", 0.5)),
                    "success": parsed.get("success", False),
                    "rationale": parsed.get("rationale", "Evaluation completed with fallback parsing"),
                    "score": parsed.get("score", parsed.get("overall", 0.5))
                }
                return result
        except:
            pass
        
        # Final fallback with basic scoring
        return self._create_fallback_result("Failed to parse evaluation")
    
    def _create_fallback_result(self, error_msg: str) -> Dict[str, Any]:
        """Create a fallback result when evaluation fails."""
        return {
            "goal_completion": 0.5,
            "coherence": 0.5,
            "tone": 0.5,
            "overall": 0.5,
            "score": 0.5,
            "success": False,
            "rationale": f"Evaluation failed: {error_msg}",
            "confidence": 0.0,
            "error": error_msg
        }
    
    def batch_assess(self, logs: List[Dict[str, Any]], show_progress: bool = True) -> List[Dict[str, Any]]:
        """Assess multiple conversation logs with progress tracking."""
        results = []
        
        for idx, log in enumerate(logs):
            if show_progress:
                print(f"Evaluating conversation {idx + 1}/{len(logs)}...")
            
            result = self.assess(log)
            results.append(result)
        
        return results
    
    def get_criteria_weights(self) -> Dict[str, float]:
        """Get the weights for each evaluation criterion."""
        return {criterion.name: criterion.weight for criterion in self.criteria}
    
    def update_criteria(self, criteria: List[EvaluationCriteria]) -> None:
        """Update evaluation criteria and clear cache."""
        self.criteria = criteria
        self._prompt_cache.clear()


class ComparativeJudge(LangChainJudge):
    """Judge for comparing two responses or approaches."""
    
    def __init__(self, **kwargs):
        super().__init__(strategy=EvaluationStrategy.PAIRWISE, **kwargs)
    
    def compare(
        self,
        log_a: Dict[str, Any],
        log_b: Dict[str, Any],
        comparison_criteria: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Compare two conversation logs."""
        comparison_criteria = comparison_criteria or ["effectiveness", "efficiency", "user_satisfaction"]
        
        # Build comparison prompt
        prompt = f"""Compare these two conversations for research planning:

CONVERSATION A:
{self._format_conversation(log_a)}

CONVERSATION B:
{self._format_conversation(log_b)}

Compare on these criteria: {', '.join(comparison_criteria)}

Return JSON with:
- winner: "A" or "B" or "tie"
- scores: dict with criteria scores for each conversation
- rationale: explanation of the comparison"""
        
        messages = [
            SystemMessage(content="You are an expert judge comparing conversation quality."),
            HumanMessage(content=prompt)
        ]
        
        response = self.llm.invoke(messages)
        
        try:
            return json.loads(response.content)
        except:
            return utils.extract_json_object(response.content) or {
                "winner": "tie",
                "scores": {},
                "rationale": "Comparison failed"
            }
    
    def _format_conversation(self, log: Dict[str, Any]) -> str:
        """Format conversation for comparison."""
        turns = log.get("turns", [])
        return "\n".join([f"{t['speaker']}: {t['text'][:100]}..." for t in turns[:5]])


# Backward compatibility
class JudgeAgent(LangChainJudge):
    """Backward compatible judge agent."""
    
    def __init__(self, llm_settings: Optional[Dict] = None, judge_prompt_template: Optional[str] = None):
        super().__init__(
            llm_settings=llm_settings,
            template_path=judge_prompt_template,
            strategy=EvaluationStrategy.CRITERIA_BASED
        )
