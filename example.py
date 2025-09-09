#!/usr/bin/env python3
"""
Example usage of the LLM Evaluation Automation package.

This script demonstrates how to use the evaluator with sample data.
"""

import sys
import os

# Add the src directory to Python path for local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from evaluation_automation import LLMEvaluator


def main():
    """Run example evaluations."""
    print("LLM Evaluation Automation - Example Usage")
    print("=" * 50)
    
    # Initialize the evaluator
    evaluator = LLMEvaluator("example_model")
    print(f"Initialized evaluator for model: {evaluator.model_name}")
    
    # Example 1: Perfect match
    print("\n1. Perfect Match Example:")
    predictions = ["The quick brown fox jumps over the lazy dog"]
    references = ["The quick brown fox jumps over the lazy dog"]
    
    results = evaluator.evaluate_text_generation(predictions, references)
    print(f"Results: {results}")
    
    # Example 2: Partial match
    print("\n2. Partial Match Example:")
    predictions = ["The quick brown fox runs fast"]
    references = ["The quick brown fox jumps over the lazy dog"]
    
    results = evaluator.evaluate_text_generation(predictions, references)
    print(f"Results: {results}")
    
    # Example 3: Multiple samples
    print("\n3. Multiple Samples Example:")
    predictions = [
        "Machine learning is fascinating",
        "Natural language processing helps computers understand text",
        "Deep learning uses neural networks"
    ]
    references = [
        "Machine learning is very interesting",
        "NLP enables computers to process human language",
        "Deep learning utilizes artificial neural networks"
    ]
    
    results = evaluator.evaluate_text_generation(predictions, references)
    print(f"Results: {results}")
    
    # Example 4: Custom metrics
    print("\n4. Custom Metrics Example (BLEU only):")
    results = evaluator.evaluate_text_generation(
        predictions, references, metrics=['bleu']
    )
    print(f"Results: {results}")
    
    # Get evaluation summary
    print("\n5. Evaluation Summary:")
    summary = evaluator.get_evaluation_summary()
    print(f"Model: {summary['model_name']}")
    print(f"Total evaluations: {summary['total_evaluations']}")
    print(f"Total samples evaluated: {summary['total_samples_evaluated']}")
    
    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()