"""
Full experiment runner: all 19774 items on all 7 working models.
Saves per-model results with checkpoint/resume support.
Token-efficient: max_tokens=5, single-letter output forced.
"""
import json, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment_runner import MODELS, run_benchmark

# Load full harm20k
items = json.load(open('phase4_raw_results/harm_20k.json', encoding='utf-8'))
print(f"Loaded {len(items)} items (~{len(items)//2} pairs)")

MODELS_TO_RUN = [
    'MODEL_J', 'MODEL_F', 'MODEL_G',
    'MODEL_A', 'MODEL_B', 'MODEL_H', 'MODEL_I'
]

t_total_start = time.time()
results_summary = {}

for i, model_name in enumerate(MODELS_TO_RUN):
    print(f"\n{'#'*60}")
    print(f"[{i+1}/{len(MODELS_TO_RUN)}] Running: {model_name}")
    print(f"{'#'*60}")

    try:
        result = run_benchmark(
            model_name=model_name,
            harm_items=items,
            sample_size=None,  # Full run
            items_to_run=None,  # Use all items
            output_dir='phase4_raw_results',
        )
        n_err = len(result['errors'])
        tok = result['stats']['total_tokens_prompt'] + result['stats']['total_tokens_completion']
        t = result['stats']['total_time_seconds']
        results_summary[model_name] = {
            'items': result['total_items'],
            'errors': n_err,
            'error_rate': result['stats']['error_rate'],
            'tokens': tok,
            'time_seconds': t,
        }
        print(f"DONE {model_name}: {result['total_items']} items, {n_err} errors ({result['stats']['error_rate']:.2%}), {tok:,} tokens, {t:.0f}s")

    except Exception as e:
        results_summary[model_name] = {'error': str(e)[:200]}
        print(f"FAILED {model_name}: {e}")

# Save summary
summary = {
    'total_models': len(MODELS_TO_RUN),
    'total_time_seconds': time.time() - t_total_start,
    'models': results_summary,
}
with open('phase4_raw_results/full_run_summary.json', 'w', encoding='utf-8') as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)

print(f"\n{'='*60}")
print(f"ALL DONE. Total time: {(time.time()-t_total_start)/3600:.1f} hours")
print(f"{'='*60}")
for m, r in results_summary.items():
    if 'error' in r:
        print(f"  {m:25s} FAILED: {r['error'][:80]}")
    else:
        print(f"  {m:25s} {r['items']} items, {r['errors']} err, {r['tokens']:,} tok, {r['time_seconds']:.0f}s")
