import json, os, sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from benchmark_scoring import compute_fiai_scores, load_harm20k
from benchmark_viz import generate_all_figures, fig7_model_comparison, FIGURE_DIR

RAW = "phase4_raw_results"
harm_items = load_harm20k(f"{RAW}/harm_20k.json")

def merge_model_data(model_name):
    all_results = []
    seen_ids = set()
    for suffix in ['_harm20k.json', '_harm20k_new.json']:
        fname = f"{RAW}/{model_name}{suffix}"
        if not os.path.exists(fname):
            continue
        try:
            d = json.load(open(fname, encoding='utf-8-sig'))
        except:
            d = json.load(open(fname, encoding='utf-8'))
        for r in d.get('results', []):
            iid = r['item_id']
            if iid not in seen_ids and not r.get('error'):
                all_results.append(r)
                seen_ids.add(iid)
    return all_results

models = ['MODEL_F', 'MODEL_G', 'MODEL_A',
          'MODEL_B', 'MODEL_H', 'MODEL_I']

print("=" * 80)
print("STEP 1: Merging data per model")
print("=" * 80)
merged = {}
for m in models:
    results = merge_model_data(m)
    merged[m] = results
    print(f"  {m}: {len(results)} results")

print(f"\n{'=' * 80}")
print("STEP 2: Computing 14-dimension scores")
print("=" * 80)
all_scores = {}
for m in models:
    scores = compute_fiai_scores(merged[m], harm_items)
    scores["model"] = m
    all_scores[m] = scores
    if "error" not in scores:
        print(f"  {m}: {scores['sample']['n_pairs']} pairs, FIAI={scores['fiai']['rate']:.1%}, d={scores['statistical_tests']['cohens_d']['value']:.3f}, {scores['uncertainty_hypothesis_test']['verdict']}")

with open(f"{RAW}/complete_stats.json", 'w', encoding='utf-8') as f:
    json.dump(all_scores, f, ensure_ascii=False, indent=2)

print(f"\n{'=' * 80}")
print("STEP 3: Generating figures")
print("=" * 80)
best = 'MODEL_A'
generate_all_figures(all_scores[best], best)
fig7_model_comparison(all_scores)

print(f"\n{'=' * 80}")
print("FINAL RESULTS TABLE")
print("=" * 80)
print(f"{'Model':<22s} {'Pairs':>6s} {'FIAI%':>7s} {'d':>6s} {'|Diff|':>6s} {'Harsh%':>7s} {'Verdict':>18s}")
print("-" * 80)
for m in sorted(all_scores.keys(), key=lambda x: all_scores[x].get('sample',{}).get('n_pairs',0), reverse=True):
    s = all_scores[m]
    if "error" in s: continue
    print(f"{m:<22s} {s['sample']['n_pairs']:>6d} {s['fiai']['rate']:>6.1%} "
          f"{s['statistical_tests']['cohens_d']['value']:>6.3f} "
          f"{s['fiai']['magnitude_mean']:>6.3f} "
          f"{s['fiai']['direction']['harsh_ratio']:>6.1%} "
          f"{s['uncertainty_hypothesis_test']['verdict']:>18s}")

# Top domains
print(f"\nTOP DOMAINS BY FIAI:")
for m in sorted(all_scores.keys()):
    s = all_scores[m]
    doms = s.get('decomposition', {}).get('by_domain', {})
    top = sorted(doms.items(), key=lambda x: x[1].get('fiai_rate',0), reverse=True)[:3]
    print(f"  {m}: {', '.join(f'{d}={v[\"fiai_rate\"]:.1%}' for d,v in top)}")

print(f"\nDone! Stats: {RAW}/complete_stats.json")
print(f"Figures: {FIGURE_DIR}/")
