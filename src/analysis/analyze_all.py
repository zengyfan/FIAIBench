"""Analyze all models from benchmark test."""
import json, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_scoring import compute_fiai_scores, load_harm20k
from benchmark_viz import generate_all_figures, fig7_model_comparison

RAW = "phase4_raw_results"
harm_items = load_harm20k(f"{RAW}/harm_20k.json")

models = [
    "MODEL_J", "MODEL_F", "MODEL_G",
    "MODEL_A", "MODEL_B",
    "MODEL_H", "MODEL_I", "MODEL_L",
]

all_scores = {}
print(f"{'Model':<25s} {'Pairs':>6s} {'FIAI%':>7s} {'Magn':>6s} {'Harsh%':>7s} {'JS Div':>8s} {'N Ctrl':>6s} {'Attn':>6s} {'Dir':>20s}")
print("-" * 100)

for m in models:
    fname = f"{RAW}/{m}_harm20k.json"
    try:
        data = json.load(open(fname, encoding='utf-8-sig'))
    except:
        data = json.load(open(fname, encoding='utf-8'))

    scores = compute_fiai_scores(data["results"], harm_items)
    if "error" in scores:
        print(f"{m:<25s} ERROR: {scores['error']}")
        continue

    all_scores[m] = scores
    scores["model"] = m

    print(f"{m:<25s} {scores['n_pairs']:>6d} {scores['D1_fiai_rate']:>6.1%} "
          f"{scores['D2_fiai_magnitude']:>6.2f} {scores['D3_harsh_bias_ratio']:>6.1%} "
          f"{scores['D7_js_divergence']:>8.4f} {scores['D8_neg_control_pass']:>6.1%} "
          f"{scores['D8_attention_pass']:>6.1%} {scores['D3_dominant_direction'][:20]:>20s}")

# Save all scores
with open(f"{RAW}/all_scores.json", 'w', encoding='utf-8') as f:
    json.dump(all_scores, f, ensure_ascii=False, indent=2)

# Generate figures for first model
first_model = "MODEL_J"
if first_model in all_scores:
    print(f"\nGenerating figures for {first_model}...")
    generate_all_figures(all_scores[first_model], first_model)

# Multi-model comparison
if len(all_scores) >= 2:
    print("Generating multi-model comparison...")
    fig7_model_comparison(all_scores)

print("\nDone! All scores saved, figures generated.")
