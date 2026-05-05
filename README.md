# FIAIBench

**Framing-Induced Attributional Injustice Benchmark for LLMs**

FIAIBench measures a previously invisible dimension of algorithmic fairness: whether LLMs produce systematically different decisions for the same individual when their circumstances are linguistically framed differently, even with identical factual content. This phenomenon, termed Framing-Induced Attributional Injustice (FIAI), is orthogonal to demographic parity and rooted in social attribution theory.

**Paper**: The Mercy Paradox — Measuring Framing-Induced Attributional Injustice in LLM via FIAIBench (Under Review)

## Repository Structure

```
├── README.md
├── requirements.txt
├── src/
│   ├── experiment/       # LLM API evaluation runners
│   ├── benchmark/        # Item generation & 14-dim FIAI scoring
│   ├── analysis/         # Statistical analysis pipeline
│   └── visualization/    # Publication-quality figure generation
└── data/
    ├── fiaibench_croissant.jsonld    # Croissant metadata (NeurIPS 2026)
    └── fiaibench_english_sample.json # English dataset sample
```

## Pipeline

1. **Benchmark Construction** — `gen_10k.py`, `gen_hard_v3.py`, `diagnose_and_fix_harm.py` generate 19,774 items from parameterized templates across 20 decision domains
2. **Experiment Execution** — `experiment_runner.py`, `smart_runner.py` call LLM APIs with checkpoint/resume and single-letter output enforcement
3. **Scoring** — `benchmark_scoring.py` computes 14 FIAI dimensions (rate, magnitude, direction, Cohen's d, JS divergence, domain/framing/severity decomposition)
4. **Visualization** — `generate_figures_v4.py` produces 3 ultra-wide composite figures for publication

## Key Metrics

| Metric | Definition |
|--------|-----------|
| FIAI Rate | Proportion of decisions that flip between implicit/explicit framings |
| FIAI Magnitude | Mean absolute decision score difference |
| Harsh Bias Ratio | Proportion of flips toward harsher judgment under implicit framing |
| Cohen's d | Standardized effect size of framing-induced shift |
| JS Divergence | Distribution-level divergence between framing conditions |

## Requirements

```
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scipy>=1.10.0
requests>=2.28.0
tqdm>=4.65.0
Pillow>=9.5.0
```

## License

CC BY-NC 4.0 — Academic research only.

## Citation

```bibtex
@misc{fiaibench2026,
  title={The Mercy Paradox: Measuring Framing-Induced Attributional Injustice in LLM via FIAIBench},
  author={Anonymous},
  year={2026},
  note={Under Review}
}
```
