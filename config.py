# Configuration file containing all tunable parameters and defaults.

import os

# Expose the OpenAI API key if provided via environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Population Settings
POPULATION_SIZE = 5  # Increased from 3 to trigger more improvements
POPULATION_INSTRUCTION_TEMPLATE_PATH = "templates/population_instruction.txt"

# Wizard Settings
WIZARD_DEFAULT_GOAL = (
    "Uncover key insights about hearing loss experiences and generate a"
    " structured research plan"
)
WIZARD_PROMPT_TEMPLATE_PATH = "templates/research_wizard_prompt.txt"
MAX_TURNS = 10

# CRITICAL: Self-improvement schedule - triggers after specific conversations
# Option 1: Every N conversations
# SELF_IMPROVE_AFTER = 2  # Improve every 2 conversations

# Option 2: After specific conversation numbers (RECOMMENDED for testing)
SELF_IMPROVE_AFTER = [1, 3, 5]  # Improve after 1st, 3rd, and 5th conversations

# Option 3: String format (alternative to list)
# SELF_IMPROVE_AFTER = "1;3;5"  # Same as above but as string

SELF_IMPROVE_PROMPT_TEMPLATE_PATH = "templates/self_improve_prompt.txt"

# Judge Settings
JUDGE_PROMPT_TEMPLATE_PATH = "templates/judge_prompt.txt"
JUDGE_IMPROVEMENT_INTERVAL = 10  # How often judges improve
ENABLE_MULTI_JUDGE = False  # Keep simple for testing
JUDGE_COUNT = 3
JUDGE_CONFIDENCE_THRESHOLD = 0.7
JUDGE_CALIBRATION_MIN_SAMPLES = 5
JUDGE_LLM_SETTINGS = {
    "temperature": 0.2,
    "max_tokens": 1024,
}

# LLM Hyperparameters
LLM_MODEL = "gpt-4.1-nano"  # Fast and cheap for testing
LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 512
LLM_TOP_P = 0.9
OPENAI_MAX_RETRIES = 3

# File/Logging Settings
LOGS_DIRECTORY = "logs"
JSON_INDENT = 2
USER_DB_PATH = "users.db"

# Runtime Options
SHOW_LIVE_CONVERSATIONS = True  # See conversations in real-time
PARALLEL_CONVERSATIONS = False  # Sequential is better for debugging improvements
START_WHEN_SPAWNED = False

# Wizard Improvement Templates (CRITICAL for DSPy)
SYNTHETIC_SCENARIOS_TEMPLATE_PATH = "templates/synthetic_scenarios.json"
SYNTHETIC_CONVERSATION_TEMPLATE_PATH = "templates/synthetic_conversation_templates.json"
PERFORMANCE_ANALYSIS_TEMPLATE_PATH = "templates/performance_analysis_template.txt"
IMPROVEMENT_PROMPTS_TEMPLATE_PATH = "templates/improvement_prompts.json"

# DSPy Training Parameters (CRITICAL)
DSPY_TRAINING_ITER = 3  # Number of training iterations
DSPY_LEARNING_RATE = 0.01

# CRITICAL: Thresholds for optimizer selection
# These determine which DSPy optimizer is used based on dataset size
DSPY_BOOTSTRAP_MINIBATCH_SIZE = 2  # Use BootstrapFewShot with 2+ examples
DSPY_MIPRO_MINIBATCH_SIZE = 4  # Use MIPROv2 with 4+ examples

# History buffer limits
HISTORY_BUFFER_LIMIT = 50  # Keep last 50 conversations for improvement
POP_HISTORY_LIMIT = 50

# Miscellaneous
DEFAULT_TIMEZONE = "UTC"


def _get_last_schedule_point(schedule):
    """Return the last integer from the self-improvement schedule."""
    if isinstance(schedule, int):
        return schedule
    if isinstance(schedule, str):
        try:
            points = [int(x) for x in schedule.split(";") if x.strip()]
        except ValueError:
            return None
        return points[-1] if points else None
    try:
        points = [int(x) for x in schedule]
    except (TypeError, ValueError):
        return None
    return points[-1] if points else None


def validate_configuration() -> None:
    """Validate cross-field relationships in the configuration."""
    last_point = _get_last_schedule_point(SELF_IMPROVE_AFTER)
    if last_point is not None and POPULATION_SIZE < last_point:
        raise ValueError(
            f"POPULATION_SIZE ({POPULATION_SIZE}) must be at least "
            f"as large as the last entry in SELF_IMPROVE_AFTER ({last_point})"
        )
