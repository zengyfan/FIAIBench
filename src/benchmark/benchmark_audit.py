"""Deep audit of benchmark items for quality assessment."""
import json
import os
from collections import Counter

RAW_DIR = "D:/AutoResearchPipeline/autoresearch_output/phase4_raw_results"

def load(path):
    return json.load(open(path, encoding='utf-8'))

# ========== 1. FAE Item Audit ==========
print("=" * 80)
print("1. FAE EXPLICIT ITEMS AUDIT")
print("=" * 80)
fae = load(f"{RAW_DIR}/fae_items.json")
fae_imp = load(f"{RAW_DIR}/fae_implicit_items.json")

# Check unique scenarios (by person name)
names = []
issues = []
for item in fae:
    prompt = item['user_prompt']
    # Extract name
    import re
    name_match = re.search(r'(陈静|李华|小明|赵强|王磊|张伟|刘洋|孙丽|周敏|吴芳|郑涛|马超|冯雪|何远|林峰|黄明|杨晨)', prompt)
    if name_match:
        names.append(name_match.group(1))
    issue_match = re.search(r'关于以下议题的文章：\n\n(.+?)\n\n', prompt)
    if issue_match:
        issues.append(issue_match.group(1))

print(f"Total FAE explicit items: {len(fae)}")
print(f"Unique names: {len(set(names))}")
print(f"Unique issues: {len(set(issues))}")
print(f"Samples: {list(set(issues))[:5]}")

# Check constraint distribution
constraints = Counter(item['condition']['constraint'] for item in fae)
print(f"Constraint distribution: {dict(constraints)}")
roles = Counter(item['condition']['role'] for item in fae)
print(f"Role distribution: {dict(roles)}")
stakes = Counter(item['condition']['stakes'] for item in fae)
print(f"Stakes distribution: {dict(stakes)}")

# MSI: Is each scenario paired (same scenario, different constraint)?
print("\nChecking paired design:")
# Group by person+issue
pairs = {}
for item in fae:
    prompt = item['user_prompt']
    key = f"{item['condition']['role']}_{item['condition']['stakes']}_{item['condition']['reasoning']}"
    if key not in pairs:
        pairs[key] = []
    pairs[key].append(item['condition']['constraint'])

has_both = sum(1 for k, v in pairs.items() if 'low_constraint' in v and 'high_constraint' in v)
print(f"Keys with both constraints: {has_both}/{len(pairs)}")

# ========== 2. FAE Implicit Items Audit ==========
print("\n" + "=" * 80)
print("2. FAE IMPLICIT ITEMS AUDIT")
print("=" * 80)
print(f"Total FAE implicit items: {len(fae_imp)}")

# Check: does implicit version REMOVE the constraint info?
for i in [0, 1]:
    exp = fae[i]
    imp = fae_imp[i]
    print(f"\nItem {i} (explicit): constraint={exp['condition']['constraint']}")
    print(f"  Prompt: {exp['user_prompt'][:150]}...")
    print(f"Item {i} (implicit): constraint={imp['condition']['constraint']}")
    print(f"  Prompt: {imp['user_prompt'][:150]}...")

# ========== 3. AOA Items Audit ==========
print("\n" + "=" * 80)
print("3. AOA ITEMS AUDIT")
print("=" * 80)
aoa = load(f"{RAW_DIR}/aoa_items.json")
print(f"Total AOA explicit items: {len(aoa)}")

perspectives = Counter(item['condition']['perspective'] for item in aoa)
print(f"Perspective distribution: {dict(perspectives)}")

scenario_ids = [item.get('scenario_id', '?') for item in aoa]
print(f"Unique scenarios: {len(set(scenario_ids))}")
print(f"Scenarios: {list(set(scenario_ids))}")

# Check: are the same scenarios used for both actor and observer?
from collections import defaultdict
scenario_perspectives = defaultdict(list)
for item in aoa:
    sid = item.get('scenario_id', '?')
    scenario_perspectives[sid].append(item['condition']['perspective'])
has_both = sum(1 for sid, persps in scenario_perspectives.items() if 'actor' in persps and 'observer' in persps)
print(f"Scenarios with both actor AND observer: {has_both}/{len(scenario_perspectives)}")

# ========== 4. Harm10k Items Audit ==========
print("\n" + "=" * 80)
print("4. HARM10K ITEMS AUDIT")
print("=" * 80)
harm = load(f"{RAW_DIR}/harm_10k.json")
print(f"Total harm items: {len(harm)}")

# Count unique scenarios (each scenario has implicit + explicit)
implicit_ids = [item['item_id'] for item in harm if '_implicit' in item['item_id']]
explicit_ids = [item['item_id'] for item in harm if '_explicit' in item['item_id']]
print(f"Implicit items: {len(implicit_ids)}")
print(f"Explicit items: {len(explicit_ids)}")

# Extract domains
domains = Counter(item.get('domain', '?') for item in harm if '_implicit' in item['item_id'])
print(f"\nDomains ({len(domains)} total):")
for d, c in domains.most_common():
    print(f"  {d}: {c}")

# Bias types
bias_types = Counter(item.get('bias_type', '?') for item in harm if '_implicit' in item['item_id'])
print(f"\nBias types: {dict(bias_types)}")

# Severity distribution
severity = Counter(item.get('severity', '?') for item in harm if '_implicit' in item['item_id'])
print(f"Severity: {dict(severity)}")

# ========== 5. QUALITY CHECK: Implicit/Explicit framing differences ==========
print("\n" + "=" * 80)
print("5. FRAMING QUALITY AUDIT (Sample of Implicit vs Explicit)")
print("=" * 80)

# Sample diverse scenarios and check framing quality
import random
random.seed(42)
implicit_items = [item for item in harm if '_implicit' in item['item_id']]
sample = random.sample(implicit_items, min(10, len(implicit_items)))

for item in sample:
    base_id = item['item_id'].replace('_implicit', '')
    exp_item = None
    for e in harm:
        if e['item_id'] == f"{base_id}_explicit":
            exp_item = e
            break

    if exp_item:
        print(f"\n--- {item['title']} ({item['domain']}) ---")
        print(f"Implicit: {item['framing'][:120]}")
        print(f"Explicit: {exp_item['framing'][:120]}")
        print(f"Options: {item['options']}")

# ========== 6. STATISTICAL POWER ANALYSIS ==========
print("\n" + "=" * 80)
print("6. STATISTICAL POWER ASSESSMENT")
print("=" * 80)

# FAE: 54 items, 2 constraint levels = ~27 per group
# This is between-items (each item only has one constraint level)
n_fae_low = sum(1 for item in fae if item['condition']['constraint'] == 'low_constraint')
n_fae_high = sum(1 for item in fae if item['condition']['constraint'] == 'high_constraint')
print(f"FAE: {n_fae_low} low-constraint vs {n_fae_high} high-constraint (between-items design)")
print(f"  Minimum detectable effect (80% power, alpha=0.05): Cohens d > 0.77")
print(f"  Typical social psych effect: Cohens d = 0.4-0.6")
print(f"  WARNING: Underpowered for typical effects!")

# Harm10k: 5000 pairs -> massive power
n_harm = len(implicit_items)
print(f"\nHarm10k: {n_harm} pairs (within-items design)")
print(f"  Minimum detectable effect: near-zero, excellent power")
print(f"  Can detect even very small effects")

print("\n" + "=" * 80)
print("7. OVERALL ASSESSMENT")
print("=" * 80)
