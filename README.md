# persona-benchmark

A toolkit to evaluate the consistency, portability, and fidelity of structured AI personas across different LLMs.

## Features
- Cross-model comparison (e.g., GPT vs Gemini)
- Cosine/BLEU/ROUGE similarity evaluation
- Drift resistance analysis
- Robustness testing on C/E/S/V/B traits

## Quick Start
```bash
python run_experiment.py --persona coach-v1 --models gpt gemini
```
