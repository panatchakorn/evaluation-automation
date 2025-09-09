"""
LLM Evaluation Automation Package

A Python package for evaluating Large Language Models with various metrics and benchmarks.
"""

__version__ = "0.1.0"
__author__ = "Evaluation Automation Team"

from .evaluator import LLMEvaluator
from .metrics import compute_bleu_score, compute_rouge_score, compute_similarity_score

__all__ = [
    "LLMEvaluator",
    "compute_bleu_score", 
    "compute_rouge_score",
    "compute_similarity_score"
]