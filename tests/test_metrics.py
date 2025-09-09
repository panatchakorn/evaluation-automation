"""
Tests for the metrics module.
"""

import pytest
from src.evaluation_automation.metrics import (
    compute_bleu_score, 
    compute_rouge_score, 
    compute_similarity_score
)


class TestMetrics:
    """Test cases for evaluation metrics."""
    
    def test_compute_bleu_score_perfect_match(self):
        """Test BLEU score for perfect matches."""
        predictions = ["the cat sat on the mat"]
        references = ["the cat sat on the mat"]
        
        score = compute_bleu_score(predictions, references)
        assert score == 1.0
    
    def test_compute_bleu_score_no_match(self):
        """Test BLEU score for no matches."""
        predictions = ["hello world"]
        references = ["goodbye universe"]
        
        score = compute_bleu_score(predictions, references)
        assert score == 0.0
    
    def test_compute_bleu_score_partial_match(self):
        """Test BLEU score for partial matches."""
        predictions = ["the cat sat"]
        references = ["the cat walked"]
        
        score = compute_bleu_score(predictions, references)
        assert 0 < score < 1
    
    def test_compute_bleu_score_empty_input(self):
        """Test BLEU score for empty input."""
        predictions = []
        references = []
        
        score = compute_bleu_score(predictions, references)
        assert score == 0.0
    
    def test_compute_rouge_score_perfect_match(self):
        """Test ROUGE score for perfect matches."""
        predictions = ["the cat sat on the mat"]
        references = ["the cat sat on the mat"]
        
        scores = compute_rouge_score(predictions, references)
        
        assert 'rouge_1' in scores
        assert 'rouge_l' in scores
        assert scores['rouge_1'] == 1.0
    
    def test_compute_rouge_score_no_match(self):
        """Test ROUGE score for no matches."""
        predictions = ["hello world"]
        references = ["goodbye universe"]
        
        scores = compute_rouge_score(predictions, references)
        
        assert scores['rouge_1'] == 0.0
        assert scores['rouge_l'] == 0.0
    
    def test_rouge_score_empty_input(self):
        """Test ROUGE score for empty input."""
        predictions = []
        references = []
        
        scores = compute_rouge_score(predictions, references)
        
        assert scores['rouge_1'] == 0.0
        assert scores['rouge_l'] == 0.0
    
    def test_compute_similarity_score_perfect_match(self):
        """Test similarity score for perfect matches."""
        predictions = ["the cat sat on the mat"]
        references = ["the cat sat on the mat"]
        
        score = compute_similarity_score(predictions, references)
        assert score == 1.0
    
    def test_compute_similarity_score_no_match(self):
        """Test similarity score for no matches."""
        predictions = ["hello world"]
        references = ["goodbye universe"]
        
        score = compute_similarity_score(predictions, references)
        assert score == 0.0
    
    def test_compute_similarity_score_partial_match(self):
        """Test similarity score for partial matches."""
        predictions = ["the cat walked quickly"]
        references = ["the dog walked slowly"]
        
        score = compute_similarity_score(predictions, references)
        assert 0 < score < 1
    
    def test_compute_similarity_score_empty_input(self):
        """Test similarity score for empty input."""
        predictions = []
        references = []
        
        score = compute_similarity_score(predictions, references)
        assert score == 0.0