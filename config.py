# Configuration file containing all tunable parameters and defaults.

import os

# Expose the OpenAI API key if provided via environment variable. This mirrors
# the example in the README so scripts can directly reference
# ``config.OPENAI_API_KEY`` without raising ``NameError`` when the variable is
# missing.
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Population Settings
# Default population size. The value must be at least as large as the
# highest value in ``SELF_IMPROVE_AFTER`` when that setting is a sequence.
POPULATION_SIZE = 3  # Small size for testing
POPULATION_INSTRUCTION_TEMPLATE_PATH = "templates/population_instruction.txt"

# Wizard Settings
# Default goal for the wizard when acting as a research planner
WIZARD_DEFAULT_GOAL = (
    "Uncover key insights about hearing loss experiences and generate a"
    " structured research plan"
)
# Use the research wizard prompt instead of basic wizard prompt
WIZARD_PROMPT_TEMPLATE_PATH = "templates/research_wizard_prompt.txt"
MAX_TURNS = 10  # Reduced for faster testing
# Trigger improvements after conversations 2 and 3 for testing
SELF_IMPROVE_AFTER = [2, 3]
SELF_IMPROVE_PROMPT_TEMPLATE_PATH = "templates/self_improve_prompt.txt"

# Judge Settings
JUDGE_PROMPT_TEMPLATE_PATH = "templates/judge_prompt.txt"
# How often judges should attempt to improve their prompts
JUDGE_IMPROVEMENT_INTERVAL = 10  # Reduced for testing
# Enable multiple judges for consensus evaluation
ENABLE_MULTI_JUDGE = False  # Keep simple for testing
# Number of judges when multi-judge is enabled
JUDGE_COUNT = 3
# Minimum confidence threshold for valid evaluations
JUDGE_CONFIDENCE_THRESHOLD = 0.7
# Minimum samples before calculating performance metrics
JUDGE_CALIBRATION_MIN_SAMPLES = 5  # Reduced for testing
# Judge-specific LLM settings (optional - uses defaults if not specified)
JUDGE_LLM_SETTINGS = {
    "temperature": 0.2,  # Lower temperature for more consistent judging
    "max_tokens": 1024,  # Sufficient for detailed evaluations
}

# LLM Hyperparameters
# Default model to use for all LLM calls. Options as of 2025:
LLM_MODEL = "gpt-4.1-nano"  # Fastest, cheapest
# or
# LLM_MODEL = "gpt-4.1-mini"  # Balanced performance
# or
# LLM_MODEL = "gpt-4.1"       # Most capable

# Legacy models still available:
# LLM_MODEL = "gpt-4o"
# LLM_MODEL = "gpt-4o-mini"

LLM_TEMPERATURE = 0.7
LLM_MAX_TOKENS = 512
LLM_TOP_P = 0.9
OPENAI_MAX_RETRIES = 3  # Increased for better reliability

# File/Logging Settings
LOGS_DIRECTORY = "logs"
JSON_INDENT = 2
USER_DB_PATH = "users.db"

# Runtime Options
# Set to True to print conversation turns to the terminal while running
SHOW_LIVE_CONVERSATIONS = True
# Run each conversation in its own thread
PARALLEL_CONVERSATIONS = False  # Serial for easier debugging
# If True and conversations run in parallel, start each thread as soon as the
# population agent is spawned instead of waiting for the full population to be
# generated.
START_WHEN_SPAWNED = False

# Dspy Settings
DSPY_TRAINING_ITER = 1
DSPY_LEARNING_RATE = 0.01
# Optimizer thresholds
DSPY_BOOTSTRAP_MINIBATCH_SIZE = 2  # Reduced for testing
DSPY_MIPRO_MINIBATCH_SIZE = 10  # Reduced for testing
# Maximum number of conversation logs kept in memory for self improvement
HISTORY_BUFFER_LIMIT = 50
# Maximum conversation history stored by each population agent
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
    """Validate cross-field relationships in the configuration.

    Currently ensures ``POPULATION_SIZE`` is large enough to support the
    self‑improvement schedule defined in ``SELF_IMPROVE_AFTER``.  This function
    should be called at runtime so unit tests can override settings via
    ``monkeypatch`` without importing this module twice.
    """

    last_point = _get_last_schedule_point(SELF_IMPROVE_AFTER)
    if last_point is not None and POPULATION_SIZE < last_point:
        raise ValueError(
            f"POPULATION_SIZE ({POPULATION_SIZE}) must be at least "
            f"as large as the last entry in SELF_IMPROVE_AFTER ({last_point})"
        )
