"""Deep-dive analysis: check response formats and data integrity."""
import json
import os
from collections import Counter

RAW_DIR = "D:/AutoResearchPipeline/autoresearch_output/phase4_raw_results"

def load_json(filename):
    path = os.path.join(RAW_DIR, filename)
    try:
        return json.load(open(path, encoding='utf-8-sig'))
    except:
        return json.load(open(path, encoding='utf-8'))

# 1. Check what Claude-4.7 actually outputs
print("=" * 80)
print("1. RESPONSE FORMAT CHECK")
print("=" * 80)
for model in ['claude-4.7', 'gemini-3.1', 'qwen-3.6', 'hy3', 'deepseek-v4-flash', 'kimi-k2.6']:
    try:
        data = load_json(f"{model}_fae.json")
        responses = [r['response'] for r in data['results'][:10]]
        print(f"\n{model} FAE (first 10 responses):")
        for i, resp in enumerate(responses):
            print(f"  [{i+1}] [{len(resp)} chars] {resp[:120]}")
    except Exception as e:
        print(f"\n{model}: ERROR - {e}")

# 2. Check if deepseek and gpt-5.x harm10k are identical
print("\n" + "=" * 80)
print("2. HARM10K DATA INTEGRITY CHECK")
print("=" * 80)
ds = load_json("deepseek-v4-flash_harm10k.json")
gpt = load_json("gpt-5.x_harm10k.json")
ds_responses = [r['response'] for r in ds['results']]
gpt_responses = [r['response'] for r in gpt['results']]

print(f"deepseek responses: {len(ds_responses)}")
print(f"gpt-5.x responses: {len(gpt_responses)}")

# Compare first 100
matches = sum(1 for a, b in zip(ds_responses[:100], gpt_responses[:100]) if a == b)
print(f"First 100: {matches}/100 match exactly")

# Compare all
total_matches = sum(1 for a, b in zip(ds_responses, gpt_responses) if a == b)
total = min(len(ds_responses), len(gpt_responses))
print(f"All pairs: {total_matches}/{total} match exactly ({total_matches/total*100:.1f}%)")

# Compare item IDs
ds_ids = [r['item_id'] for r in ds['results']]
gpt_ids = [r['item_id'] for r in gpt['results']]
id_matches = sum(1 for a, b in zip(ds_ids, gpt_ids) if a == b)
print(f"Item ID match: {id_matches}/{min(len(ds_ids), len(gpt_ids))}")

# 3. Check FAE implicit data
print("\n" + "=" * 80)
print("3. FAE IMPLICIT DATA CHECK")
print("=" * 80)
for model in ['deepseek-v4-flash', 'gpt-5.x', 'kimi-k2.6', 'glm-5.1']:
    try:
        data = load_json(f"{model}_fae_implicit.json")
        responses = [r['response'] for r in data['results']]
        resp_counts = Counter(r.strip().upper()[0] if r.strip() else '?' for r in responses)
        print(f"\n{model} FAE implicit (N={len(responses)}):")
        for k, v in sorted(resp_counts.items()):
            print(f"  {k}: {v} ({v/len(responses)*100:.1f}%)")
    except Exception as e:
        print(f"\n{model}: ERROR - {e}")

# 4. Check AOA implicit data
print("\n" + "=" * 80)
print("4. AOA IMPLICIT DATA CHECK")
print("=" * 80)
for model in ['deepseek-v4-flash', 'gpt-5.x', 'kimi-k2.6', 'glm-5.1']:
    try:
        data = load_json(f"{model}_aoa_implicit.json")
        responses = [r['response'] for r in data['results']]
        # AOA responses may have explanation + final A/B/C
        print(f"\n{model} AOA implicit (N={len(responses)}), first 5:")
        for i, resp in enumerate(responses[:5]):
            print(f"  [{i+1}] [{len(resp)} chars] ...{resp[-80:]}")
    except Exception as e:
        print(f"\n{model}: ERROR - {e}")

# 5. Check AOA explicit responses
print("\n" + "=" * 80)
print("5. AOA EXPLICIT RESPONSE DISTRIBUTIONS")
print("=" * 80)
for model in ['deepseek-v4-flash', 'gpt-5.x', 'claude-4.7', 'gemini-3.1', 'qwen-3.6', 'hy3', 'kimi-k2.6', 'glm-5.1']:
    try:
        data = load_json(f"{model}_aoa.json")
        responses = [r['response'] for r in data['results']]
        # Extract last non-whitespace character if response is long, else use first char
        parsed = []
        for r in responses:
            r = r.strip()
            # Try to find A/B/C at the end
            found = False
            for ch in reversed(r):
                if ch in ('A','B','C'):
                    parsed.append(ch)
                    found = True
                    break
            if not found:
                parsed.append(r[0] if r else '?')

        resp_counts = Counter(parsed)
        print(f"\n{model} AOA explicit (N={len(responses)}):")
        for k in ['A','B','C','?']:
            v = resp_counts.get(k, 0)
            print(f"  {k}: {v} ({v/len(responses)*100:.1f}%)")
    except Exception as e:
        print(f"\n{model}: ERROR - {e}")

# 6. Check glm harm10k for interesting patterns
print("\n" + "=" * 80)
print("6. GLM-5.1 HARM10K DEEP DIVE")
print("=" * 80)
glm = load_json("glm-5.1_harm10k.json")
implicit_glm = {}
explicit_glm = {}
for r in glm['results']:
    iid = r['item_id']
    if '_implicit' in iid:
        implicit_glm[iid.replace('_implicit', '')] = r['response']
    elif '_explicit' in iid:
        explicit_glm[iid.replace('_explicit', '')] = r['response']

# Show some flip examples
print("Examples where GLM flipped (first 10):")
count = 0
for base in implicit_glm:
    if base in explicit_glm and implicit_glm[base] != explicit_glm[base]:
        print(f"  [{base}] implicit={implicit_glm[base]}, explicit={explicit_glm[base]}")
        count += 1
        if count >= 10:
            break
