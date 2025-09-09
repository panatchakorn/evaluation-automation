# LLM Evaluation Automation

A Python package for evaluating Large Language Models (LLMs) with various metrics and benchmarks.

## Features

- **Multiple Evaluation Metrics**: Support for BLEU, ROUGE, and semantic similarity scoring
- **Easy-to-use API**: Simple interface for running evaluations on text generation tasks
- **Extensible Architecture**: Modular design for adding new metrics and evaluation methods
- **Comprehensive Testing**: Full test suite with pytest
- **Security Scanning**: Integrated CodeQL security analysis on pull requests

## Installation

```bash
pip install -r requirements.txt
```

## Quick Start

```python
from src.evaluation_automation import LLMEvaluator

# Initialize evaluator
evaluator = LLMEvaluator("my_model")

# Prepare your data
predictions = ["The cat sat on the mat"]
references = ["The cat sat on the mat"]

# Run evaluation
results = evaluator.evaluate_text_generation(predictions, references)
print(results)
# Output: {'bleu_score': 1.0, 'rouge_1': 1.0, 'rouge_l': 1.0, 'similarity_score': 1.0}

# Get evaluation summary
summary = evaluator.get_evaluation_summary()
print(summary)
```

## Metrics

### BLEU Score
- Measures precision of n-grams between predicted and reference text
- Range: 0-1 (higher is better)

### ROUGE Score
- **ROUGE-1**: Unigram overlap between prediction and reference
- **ROUGE-L**: Longest common subsequence-based scoring
- Range: 0-1 (higher is better)

### Similarity Score
- Jaccard similarity based on word overlap
- Range: 0-1 (higher is better)

## Development

### Running Tests

```bash
pytest tests/
```

### Running Tests with Coverage

```bash
pytest tests/ --cov=src/evaluation_automation --cov-report=html
```

## Security

This project uses CodeQL for security scanning on all pull requests to ensure code quality and security best practices.

## License

MIT License