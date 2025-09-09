"""
Main LLM Evaluator class for running evaluations on language models.
"""

from typing import List, Dict, Any, Optional
import logging
from .metrics import compute_bleu_score, compute_rouge_score, compute_similarity_score

logger = logging.getLogger(__name__)


class LLMEvaluator:
    """
    Main class for evaluating Large Language Models.
    
    Supports various evaluation metrics including BLEU, ROUGE, and semantic similarity.
    """
    
    def __init__(self, model_name: str = "default"):
        """
        Initialize the LLM Evaluator.
        
        Args:
            model_name: Name identifier for the model being evaluated
        """
        self.model_name = model_name
        self.evaluation_history = []
        logger.info(f"Initialized LLMEvaluator for model: {model_name}")
    
    def evaluate_text_generation(self, 
                                predictions: List[str], 
                                references: List[str],
                                metrics: Optional[List[str]] = None) -> Dict[str, float]:
        """
        Evaluate text generation quality using multiple metrics.
        
        Args:
            predictions: List of generated text predictions
            references: List of reference/ground truth texts
            metrics: List of metrics to compute. Defaults to ['bleu', 'rouge', 'similarity']
        
        Returns:
            Dictionary containing computed metric scores
        
        Raises:
            ValueError: If predictions and references have different lengths
        """
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have the same length")
        
        if not predictions:
            raise ValueError("Predictions cannot be empty")
        
        if metrics is None:
            metrics = ['bleu', 'rouge', 'similarity']
        
        results = {}
        
        # Compute requested metrics
        if 'bleu' in metrics:
            results['bleu_score'] = compute_bleu_score(predictions, references)
        
        if 'rouge' in metrics:
            rouge_scores = compute_rouge_score(predictions, references)
            results.update(rouge_scores)
        
        if 'similarity' in metrics:
            results['similarity_score'] = compute_similarity_score(predictions, references)
        
        # Store evaluation in history
        evaluation_record = {
            'model_name': self.model_name,
            'num_samples': len(predictions),
            'metrics': results
        }
        self.evaluation_history.append(evaluation_record)
        
        logger.info(f"Evaluated {len(predictions)} samples with metrics: {list(results.keys())}")
        return results
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """
        Get a summary of all evaluations performed.
        
        Returns:
            Dictionary containing evaluation history and summary statistics
        """
        if not self.evaluation_history:
            return {'message': 'No evaluations performed yet'}
        
        total_samples = sum(eval_record['num_samples'] for eval_record in self.evaluation_history)
        
        return {
            'model_name': self.model_name,
            'total_evaluations': len(self.evaluation_history),
            'total_samples_evaluated': total_samples,
            'evaluation_history': self.evaluation_history
        }