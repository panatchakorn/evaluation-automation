"""
Tests for the LLMEvaluator class.
"""

import pytest
from src.evaluation_automation import LLMEvaluator


class TestLLMEvaluator:
    """Test cases for LLMEvaluator class."""
    
    def test_initialization(self):
        """Test evaluator initialization."""
        evaluator = LLMEvaluator("test_model")
        assert evaluator.model_name == "test_model"
        assert evaluator.evaluation_history == []
    
    def test_initialization_default(self):
        """Test evaluator initialization with default model name."""
        evaluator = LLMEvaluator()
        assert evaluator.model_name == "default"
    
    def test_evaluate_text_generation_basic(self):
        """Test basic text generation evaluation."""
        evaluator = LLMEvaluator("test_model")
        predictions = ["The cat sat on the mat"]
        references = ["The cat sat on the mat"]
        
        results = evaluator.evaluate_text_generation(predictions, references)
        
        # Perfect match should give high scores
        assert 'bleu_score' in results
        assert 'rouge_1' in results
        assert 'rouge_l' in results
        assert 'similarity_score' in results
        
        assert results['bleu_score'] == 1.0
        assert results['similarity_score'] == 1.0
    
    def test_evaluate_text_generation_mismatch_length(self):
        """Test error handling for mismatched lengths."""
        evaluator = LLMEvaluator("test_model")
        predictions = ["text1", "text2"]
        references = ["ref1"]
        
        with pytest.raises(ValueError, match="must have the same length"):
            evaluator.evaluate_text_generation(predictions, references)
    
    def test_evaluate_text_generation_empty_input(self):
        """Test error handling for empty input."""
        evaluator = LLMEvaluator("test_model")
        predictions = []
        references = []
        
        with pytest.raises(ValueError, match="cannot be empty"):
            evaluator.evaluate_text_generation(predictions, references)
    
    def test_evaluate_text_generation_custom_metrics(self):
        """Test evaluation with custom metrics."""
        evaluator = LLMEvaluator("test_model")
        predictions = ["hello world"]
        references = ["hello world"]
        
        results = evaluator.evaluate_text_generation(
            predictions, references, metrics=['bleu']
        )
        
        assert 'bleu_score' in results
        assert 'rouge_1' not in results
        assert 'similarity_score' not in results
    
    def test_get_evaluation_summary_empty(self):
        """Test evaluation summary when no evaluations performed."""
        evaluator = LLMEvaluator("test_model")
        summary = evaluator.get_evaluation_summary()
        
        assert summary['message'] == 'No evaluations performed yet'
    
    def test_get_evaluation_summary_with_data(self):
        """Test evaluation summary with evaluation history."""
        evaluator = LLMEvaluator("test_model")
        
        # Perform an evaluation
        predictions = ["test prediction"]
        references = ["test reference"]
        evaluator.evaluate_text_generation(predictions, references)
        
        summary = evaluator.get_evaluation_summary()
        
        assert summary['model_name'] == "test_model"
        assert summary['total_evaluations'] == 1
        assert summary['total_samples_evaluated'] == 1
        assert len(summary['evaluation_history']) == 1