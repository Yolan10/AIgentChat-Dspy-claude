#!/usr/bin/env python3
"""Test script to debug MIPROv2 issues in isolation."""

import os
import sys
import json

# Ensure we can import from the project
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
import utils

# Test 1: Check DSPy installation
print("=" * 60)
print("TEST 1: DSPy Installation Check")
print("=" * 60)

try:
    import dspy
    print("✓ DSPy imported successfully")
    print(f"  Version info: {dspy.__file__}")
    
    from dspy.teleprompt.mipro_optimizer_v2 import MIPROv2
    print("✓ MIPROv2 imported successfully")
except Exception as e:
    print(f"✗ DSPy import failed: {e}")
    sys.exit(1)

# Test 2: Check model configuration
print("\n" + "=" * 60)
print("TEST 2: Model Configuration")
print("=" * 60)

print(f"Configured model: {config.LLM_MODEL}")
print(f"API Key present: {bool(os.environ.get('OPENAI_API_KEY'))}")

# Test 3: Simple MIPROv2 example
print("\n" + "=" * 60)
print("TEST 3: Simple MIPROv2 Example")
print("=" * 60)

# Configure DSPy
try:
    dspy.settings.configure(
        lm=dspy.LM(
            model=config.LLM_MODEL,
            temperature=0.3,
            max_tokens=512,
        )
    )
    print("✓ DSPy configured successfully")
except Exception as e:
    print(f"✗ DSPy configuration failed: {e}")
    print("  Trying fallback model...")
    try:
        dspy.settings.configure(
            lm=dspy.LM(
                model="gpt-3.5-turbo",
                temperature=0.3,
                max_tokens=512,
            )
        )
        print("✓ DSPy configured with fallback model")
    except Exception as e2:
        print(f"✗ Fallback also failed: {e2}")
        sys.exit(1)

# Create a simple signature and module
class SimpleSignature(dspy.Signature):
    """Generate an improved version of the input text."""
    input_text: str = dspy.InputField(desc="Text to improve")
    improved_text: str = dspy.OutputField(desc="Improved version of the text")

class SimpleModule(dspy.Module):
    def __init__(self):
        super().__init__()
        self.improve = dspy.ChainOfThought(SimpleSignature)
    
    def forward(self, input_text: str) -> dspy.Prediction:
        return self.improve(input_text=input_text)

# Create simple dataset
print("\nCreating simple dataset...")
dataset = []
for i in range(6):  # Just enough for MIPROv2
    score = 0.8 if i % 2 == 0 else 0.4
    ex = dspy.Example(
        input_text=f"This is example {i}",
        improved_text=f"This is an improved example {i}",
        score=score
    ).with_inputs("input_text")
    dataset.append(ex)

print(f"✓ Created dataset with {len(dataset)} examples")

# Simple metric function
def simple_metric(example: dspy.Example, pred: dspy.Prediction, trace=None) -> float:
    """Simple scoring based on length and keywords."""
    improved = getattr(pred, 'improved_text', '')
    if not improved:
        return 0.0
    
    # Score based on length and presence of 'improved'
    length_score = min(1.0, len(improved) / 50)
    keyword_score = 1.0 if 'improved' in improved.lower() else 0.5
    
    return (length_score + keyword_score) / 2

# Test MIPROv2
print("\nTesting MIPROv2...")
module = SimpleModule()

print("Parameters:")
print(f"  - Dataset size: {len(dataset)}")
print(f"  - Minibatch size: 2")
print(f"  - Num trials: 4")
print(f"  - Num candidates: 5")

try:
    optimizer = MIPROv2(
        metric=simple_metric,
        num_candidates=5,
        init_temperature=1.0,
        verbose=True,
        track_stats=True
    )
    
    print("\n✓ MIPROv2 optimizer created")
    
    print("\nStarting optimization...")
    trained = optimizer.compile(
        module,
        trainset=dataset,
        num_trials=4,
        minibatch_size=2,
        require_training_set=True
    )
    
    print("\n✓ MIPROv2 optimization completed!")
    
    # Test the trained module
    result = trained(input_text="Test input for improvement")
    print(f"\nTest result: {result.improved_text}")
    
except Exception as e:
    print(f"\n✗ MIPROv2 failed: {e}")
    print(f"  Error type: {type(e).__name__}")
    
    import traceback
    print("\nFull traceback:")
    traceback.print_exc()
    
    # Try to diagnose common issues
    print("\n" + "=" * 60)
    print("DIAGNOSTICS")
    print("=" * 60)
    
    error_str = str(e).lower()
    
    if "api" in error_str or "key" in error_str:
        print("⚠ Possible API key issue")
        print("  - Check OPENAI_API_KEY environment variable")
        print("  - Verify key has access to the model")
    
    if "model" in error_str:
        print("⚠ Possible model issue")
        print(f"  - Current model: {config.LLM_MODEL}")
        print("  - Try using 'gpt-3.5-turbo' instead")
    
    if "minibatch" in error_str or "batch" in error_str:
        print("⚠ Possible batch size issue")
        print("  - Dataset might be too small")
        print("  - Minibatch size might be too large")
    
    if "compile" in error_str:
        print("⚠ Possible compilation issue")
        print("  - Check module structure")
        print("  - Verify signature fields")

# Test 4: Check wizard improver setup
print("\n" + "=" * 60)
print("TEST 4: Wizard Improver Check")
print("=" * 60)

try:
    from wizard_improver import WizardImprover, build_dataset
    print("✓ Wizard improver imports successful")
    
    # Check if templates exist
    template_paths = [
        config.IMPROVEMENT_PROMPTS_TEMPLATE_PATH,
        config.SYNTHETIC_SCENARIOS_TEMPLATE_PATH,
        config.SYNTHETIC_CONVERSATION_TEMPLATE_PATH,
        config.PERFORMANCE_ANALYSIS_TEMPLATE_PATH
    ]
    
    for path in template_paths:
        if os.path.exists(path):
            print(f"✓ Template exists: {path}")
        else:
            print(f"✗ Template missing: {path}")
    
except Exception as e:
    print(f"✗ Wizard improver import failed: {e}")

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)
