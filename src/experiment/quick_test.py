"""
Quick test runner: samples 500 balanced items and runs all models.
Token-optimized: max_tokens=5, system prompt forces single-letter output.
Results saved per-model to phase4_raw_results/{model}_harm20k.json.
"""
import json
import os
import sys
import random
from collections import Counter

# Set API key from environment
if not os.getenv("LLM_API_KEY"):
    os.environ["LLM_API_KEY"] = os.getenv("LLM_API_KEY", "")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_runner import run_benchmark, MODELS

# Load items
items = json.load(open("autoresearch_output/phase4_raw_results/harm_20k.json", encoding='utf-8'))

# Sample 500 balanced items across domains
random.seed(42)
domain_items = {}
for item in items:
    d = item.get('domain', 'other')
    if d not in domain_items:
        domain_items[d] = []
    domain_items[d].append(item)

balanced_sample = []
n_per_domain = max(1, 500 // len(domain_items))
for d, d_items in domain_items.items():
    sample_n = min(n_per_domain, len(d_items))
    balanced_sample.extend(random.sample(d_items, sample_n))

# If we have more than 500, trim
if len(balanced_sample) > 500:
    balanced_sample = random.sample(balanced_sample, 500)

print(f"Balanced sample: {len(balanced_sample)} items across {len(domain_items)} domains")
print(f"Domains: {Counter(item.get('domain','?') for item in balanced_sample).most_common(5)}...")

# Save sample for reproducibility
os.makedirs("autoresearch_output/phase4_raw_results", exist_ok=True)
with open("autoresearch_output/phase4_raw_results/test_sample_500.json", 'w', encoding='utf-8') as f:
    json.dump(balanced_sample, f, ensure_ascii=False, indent=2)

# Models to run (all 8)
MODEL_ORDER = [
    "MODEL_J",
    "MODEL_F",
    "MODEL_G",
    "MODEL_A",
    "MODEL_B",
    "MODEL_H",
    "MODEL_I",
    "MODEL_L",
]

results_summary = {}
for model_name in MODEL_ORDER:
    cfg = MODELS[model_name]
    print(f"\n{'#'*60}")
    print(f"# Running: {model_name} ({cfg['description']})")
    print(f"{'#'*60}")

    try:
        result = run_benchmark(
            model_name=model_name,
            harm_items=items,  # Pass all items, sample inside
            sample_size=500,
            output_dir="autoresearch_output/phase4_raw_results",
        )
        results_summary[model_name] = {
            "status": "OK",
            "items": result["total_items"],
            "errors": len(result["errors"]),
            "error_rate": result["stats"]["error_rate"],
            "tokens_total": result["stats"]["total_tokens_prompt"] + result["stats"]["total_tokens_completion"],
            "time_seconds": result["stats"]["total_time_seconds"],
        }
        print(f"✓ {model_name}: {result['total_items']} items, {len(result['errors'])} errors, {result['stats']['total_time_seconds']:.0f}s")

    except Exception as e:
        results_summary[model_name] = {"status": "FAILED", "error": str(e)[:200]}
        print(f"✗ {model_name} FAILED: {str(e)[:200]}")

# Save summary
with open("autoresearch_output/phase4_raw_results/test_summary.json", 'w', encoding='utf-8') as f:
    json.dump(results_summary, f, ensure_ascii=False, indent=2)

print("\n" + "="*60)
print("TEST COMPLETE")
print("="*60)
for m, r in results_summary.items():
    status = r.get("status", "?")
    if status == "OK":
        print(f"  ✓ {m:25s} | {r['items']:4d} items | {r['error_rate']:.1%} err | {r['time_seconds']:.0f}s | {r['tokens_total']:,} tok")
    else:
        print(f"  ✗ {m:25s} | {r.get('error','')[:80]}")
