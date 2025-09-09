"""
Evaluation metrics for LLM assessment.

This module contains various metrics commonly used for evaluating language models,
including BLEU, ROUGE, and semantic similarity scores.
"""

from typing import List, Dict
import re
from collections import Counter
import logging

logger = logging.getLogger(__name__)


def compute_bleu_score(predictions: List[str], references: List[str]) -> float:
    """
    Compute BLEU score for text generation evaluation.
    
    This is a simplified implementation of BLEU-1 score.
    For production use, consider using libraries like nltk or sacrebleu.
    
    Args:
        predictions: List of predicted/generated texts
        references: List of reference/ground truth texts
    
    Returns:
        BLEU score as a float between 0 and 1
    """
    if not predictions or not references:
        return 0.0
    
    total_precision = 0.0
    
    for pred, ref in zip(predictions, references):
        # Tokenize (simple whitespace splitting)
        pred_tokens = pred.lower().split()
        ref_tokens = ref.lower().split()
        
        if not pred_tokens:
            continue
            
        # Count matching tokens (BLEU-1)
        pred_counter = Counter(pred_tokens)
        ref_counter = Counter(ref_tokens)
        
        # Calculate precision
        matches = sum((pred_counter & ref_counter).values())
        total_precision += matches / len(pred_tokens) if pred_tokens else 0
    
    bleu_score = total_precision / len(predictions)
    logger.debug(f"Computed BLEU score: {bleu_score:.4f}")
    return bleu_score


def compute_rouge_score(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """
    Compute ROUGE scores for text summarization evaluation.
    
    This is a simplified implementation of ROUGE-1 and ROUGE-L.
    For production use, consider using libraries like rouge-score or py-rouge.
    
    Args:
        predictions: List of predicted/generated texts
        references: List of reference/ground truth texts
    
    Returns:
        Dictionary containing ROUGE-1 and ROUGE-L scores
    """
    if not predictions or not references:
        return {'rouge_1': 0.0, 'rouge_l': 0.0}
    
    rouge_1_scores = []
    rouge_l_scores = []
    
    for pred, ref in zip(predictions, references):
        # ROUGE-1 (unigram overlap)
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())
        
        if ref_tokens:
            rouge_1 = len(pred_tokens & ref_tokens) / len(ref_tokens)
        else:
            rouge_1 = 0.0
        rouge_1_scores.append(rouge_1)
        
        # ROUGE-L (longest common subsequence)
        rouge_l = _compute_lcs_score(pred.lower(), ref.lower())
        rouge_l_scores.append(rouge_l)
    
    results = {
        'rouge_1': sum(rouge_1_scores) / len(rouge_1_scores),
        'rouge_l': sum(rouge_l_scores) / len(rouge_l_scores)
    }
    
    logger.debug(f"Computed ROUGE scores: {results}")
    return results


def compute_similarity_score(predictions: List[str], references: List[str]) -> float:
    """
    Compute semantic similarity score using simple text overlap.
    
    This is a basic implementation using Jaccard similarity.
    For production use, consider using sentence embeddings or other semantic similarity methods.
    
    Args:
        predictions: List of predicted/generated texts
        references: List of reference/ground truth texts
    
    Returns:
        Average similarity score as a float between 0 and 1
    """
    if not predictions or not references:
        return 0.0
    
    similarity_scores = []
    
    for pred, ref in zip(predictions, references):
        # Convert to sets of words
        pred_words = set(re.findall(r'\w+', pred.lower()))
        ref_words = set(re.findall(r'\w+', ref.lower()))
        
        # Jaccard similarity
        intersection = len(pred_words & ref_words)
        union = len(pred_words | ref_words)
        
        if union == 0:
            similarity = 1.0 if not pred_words and not ref_words else 0.0
        else:
            similarity = intersection / union
        
        similarity_scores.append(similarity)
    
    avg_similarity = sum(similarity_scores) / len(similarity_scores)
    logger.debug(f"Computed similarity score: {avg_similarity:.4f}")
    return avg_similarity


def _compute_lcs_score(text1: str, text2: str) -> float:
    """
    Compute longest common subsequence score for ROUGE-L.
    
    Args:
        text1: First text string
        text2: Second text string
    
    Returns:
        LCS-based similarity score
    """
    tokens1 = text1.split()
    tokens2 = text2.split()
    
    if not tokens1 or not tokens2:
        return 0.0
    
    # Dynamic programming for LCS
    dp = [[0] * (len(tokens2) + 1) for _ in range(len(tokens1) + 1)]
    
    for i in range(1, len(tokens1) + 1):
        for j in range(1, len(tokens2) + 1):
            if tokens1[i-1] == tokens2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs_length = dp[len(tokens1)][len(tokens2)]
    
    # F-score based on LCS
    if len(tokens1) + len(tokens2) == 0:
        return 0.0
    
    precision = lcs_length / len(tokens1) if tokens1 else 0
    recall = lcs_length / len(tokens2) if tokens2 else 0
    
    if precision + recall == 0:
        return 0.0
    
    f_score = 2 * precision * recall / (precision + recall)
    return f_score