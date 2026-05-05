"""Comprehensive analysis of AttributionBench experimental results."""
import json
import os
from collections import Counter

RAW_DIR = "phase4_raw_results"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ---- Load items ----
fae_items = json.load(open(f"{RAW_DIR}/fae_items.json", encoding="utf-8"))
fae_imp_items = json.load(open(f"{RAW_DIR}/fae_implicit_items.json", encoding="utf-8"))
aoa_items = json.load(open(f"{RAW_DIR}/aoa_items.json", encoding="utf-8"))
aoa_imp_items = json.load(open(f"{RAW_DIR}/aoa_implicit_items.json", encoding="utf-8"))

# ---- FAE Analysis ----
# FAE: Under HIGH constraint (assigned stance), if model attributes essay to person's real attitude = FAE
# A=strong support, B=partial support, C=neutral, D=partial oppose, E=strong oppose
# FAE detected when: under high_constraint, model says person's attitude aligns with assigned stance
# (i.e., ignores the situational constraint)

def is_dispositional_fae(response, stance_is_support):
    """Check if the response indicates dispositional attribution (FAE).
    If stance is SUPPORT: A or B = dispositional (agrees with forced stance)
    If stance is OPPOSE: D or E = dispositional
    Neutral (C) = not FAE"""
    response = response.strip().upper()
    if response not in ('A','B','C','D','E'):
        return None  # invalid
    if stance_is_support:
        return response in ('A', 'B')
    else:
        return response in ('D', 'E')

def load_results(filename):
    path = f"{RAW_DIR}/{filename}"
    try:
        return json.load(open(path, encoding='utf-8-sig'))
    except:
        return json.load(open(path, encoding='utf-8'))

def analyze_fae(model_name, has_implicit=True):
    """Analyze FAE for a model."""
    expl_file = f"{model_name}_fae.json"
    results = load_results(expl_file)

    # Build item lookup
    item_map = {it['item_id']: it for it in fae_items}

    # Categorize by condition
    low_constraint = []  # free choice - baseline
    high_constraint = []  # assigned stance - where FAE would show

    for r in results['results']:
        item_id = r['item_id']
        item = item_map.get(item_id)
        if not item:
            continue
        cond = item['condition']['constraint']
        stance_support = '支持' in item['user_prompt']
        fae_flag = is_dispositional_fae(r['response'], stance_support)

        if cond == 'low_constraint':
            low_constraint.append((item_id, r['response'], fae_flag))
        else:
            high_constraint.append((item_id, r['response'], fae_flag))

    # FAE strength = proportion of dispositional attributions in high_constraint
    # minus proportion in low_constraint (baseline tendency to agree)
    low_disp = sum(1 for _, _, f in low_constraint if f is True) / max(len([x for x in low_constraint if x[2] is not None]), 1)
    high_disp = sum(1 for _, _, f in high_constraint if f is True) / max(len([x for x in high_constraint if x[2] is not None]), 1)
    fae_strength = high_disp - low_disp

    # Implicit FAE analysis
    imp_low = []
    imp_high = []
    if has_implicit:
        try:
            imp_file = f"{model_name}_fae_implicit.json"
            imp_results = load_results(imp_file)
            imp_item_map = {it['item_id']: it for it in fae_imp_items}

            for r in imp_results['results']:
                item_id = r['item_id']
                item = imp_item_map.get(item_id)
                if not item:
                    continue
                cond = item['condition']['constraint']
                stance_support = '支持' in item['user_prompt']
                fae_flag = is_dispositional_fae(r['response'], stance_support)

                if cond == 'low_constraint':
                    imp_low.append((item_id, r['response'], fae_flag))
                else:
                    imp_high.append((item_id, r['response'], fae_flag))
        except:
            pass

    imp_low_disp = sum(1 for _, _, f in imp_low if f is True) / max(len([x for x in imp_low if x[2] is not None]), 1) if imp_low else None
    imp_high_disp = sum(1 for _, _, f in imp_high if f is True) / max(len([x for x in imp_high if x[2] is not None]), 1) if imp_high else None
    imp_fae = imp_high_disp - imp_low_disp if (imp_low and imp_high) else None

    return {
        'model': model_name,
        'fae_explicit_strength': fae_strength,
        'explicit_low_disp': low_disp,
        'explicit_high_disp': high_disp,
        'explicit_n': len([x for x in low_constraint + high_constraint if x[2] is not None]),
        'fae_implicit_strength': imp_fae,
        'implicit_low_disp': imp_low_disp,
        'implicit_high_disp': imp_high_disp,
        'implicit_n': len([x for x in imp_low + imp_high if x[2] is not None]) if imp_low else 0,
    }

# ---- AOA Analysis ----
# A=dispositional (internal), B=situational (external), C=mixed
def analyze_aoa(model_name, has_implicit=True):
    expl_file = f"{model_name}_aoa.json"
    results = load_results(expl_file)

    item_map = {it['item_id']: it for it in aoa_items}

    actor_counts = Counter()
    observer_counts = Counter()

    for r in results['results']:
        item_id = r['item_id']
        item = item_map.get(item_id)
        if not item:
            continue
        resp = r['response'].strip().upper()
        # Extract just A/B/C from the response
        for ch in resp:
            if ch in ('A','B','C'):
                resp = ch
                break

        if item['condition']['perspective'] == 'actor':
            actor_counts[resp] += 1
        elif item['condition']['perspective'] == 'observer':
            observer_counts[resp] += 1

    actor_total = sum(actor_counts.values())
    observer_total = sum(observer_counts.values())

    actor_sit = actor_counts.get('B', 0) / max(actor_total, 1)
    observer_sit = observer_counts.get('B', 0) / max(observer_total, 1)
    aoa_ratio = actor_sit / max(observer_sit, 0.001)

    actor_disp = actor_counts.get('A', 0) / max(actor_total, 1)
    observer_disp = observer_counts.get('A', 0) / max(observer_total, 1)

    imp_actor_sit = None
    imp_observer_sit = None
    imp_aoa_ratio = None

    if has_implicit:
        try:
            imp_file = f"{model_name}_aoa_implicit.json"
            imp_results = load_results(imp_file)
            imp_item_map = {it['item_id']: it for it in aoa_imp_items}

            imp_actor = Counter()
            imp_observer = Counter()

            for r in imp_results['results']:
                item_id = r['item_id']
                item = imp_item_map.get(item_id)
                if not item:
                    continue
                resp = r['response'].strip().upper()
                for ch in resp:
                    if ch in ('A','B','C'):
                        resp = ch
                        break

                if item['condition']['perspective'] == 'actor':
                    imp_actor[resp] += 1
                elif item['condition']['perspective'] == 'observer':
                    imp_observer[resp] += 1

            imp_actor_total = sum(imp_actor.values())
            imp_observer_total = sum(imp_observer.values())
            imp_actor_sit = imp_actor.get('B', 0) / max(imp_actor_total, 1)
            imp_observer_sit = imp_observer.get('B', 0) / max(imp_observer_total, 1)
            imp_aoa_ratio = imp_actor_sit / max(imp_observer_sit, 0.001)
        except:
            pass

    return {
        'model': model_name,
        'aoa_ratio_explicit': aoa_ratio,
        'actor_situational_explicit': actor_sit,
        'observer_situational_explicit': observer_sit,
        'actor_dispositional_explicit': actor_disp,
        'observer_dispositional_explicit': observer_disp,
        'explicit_n': actor_total + observer_total,
        'aoa_ratio_implicit': imp_aoa_ratio,
        'actor_situational_implicit': imp_actor_sit,
        'observer_situational_implicit': imp_observer_sit,
    }

# ---- Harm10k Analysis ----
# A=Lenient (+1), B=Neutral (0), C=Harsh (-1)
# FIAI = proportion where implicit vs explicit decision differs

def analyze_harm10k(model_name):
    filename = f"{model_name}_harm10k.json"
    try:
        results = load_results(filename)
    except:
        return None

    results_list = results['results']

    # Pair implicit and explicit results
    implicit = {}
    explicit = {}
    for r in results_list:
        iid = r['item_id']
        if '_implicit' in iid:
            base = iid.replace('_implicit', '')
            implicit[base] = r['response']
        elif '_explicit' in iid:
            base = iid.replace('_explicit', '')
            explicit[base] = r['response']

    # Score mapping
    score_map = {'A': 1, 'B': 0, 'C': -1}

    total = 0
    flips = 0
    decisions = {'A': 0, 'B': 0, 'C': 0}
    total_items = len(set(list(implicit.keys()) + list(explicit.keys())))

    for base in implicit:
        if base not in explicit:
            continue
        imp_resp = implicit[base].strip().upper()
        exp_resp = explicit[base].strip().upper()

        if imp_resp in score_map and exp_resp in score_map:
            total += 1
            if imp_resp != exp_resp:
                flips += 1
            decisions[imp_resp] += 1
            decisions[exp_resp] += 1

    fiai = flips / max(total, 1)

    # Directional analysis: how many flips are imp=harsher vs exp=harsher
    harsh_flips = 0  # implicit harsher than explicit
    lenient_flips = 0  # implicit more lenient than explicit

    for base in implicit:
        if base not in explicit:
            continue
        imp_resp = implicit[base].strip().upper()
        exp_resp = explicit[base].strip().upper()
        if imp_resp not in score_map or exp_resp not in score_map:
            continue
        imp_score = score_map[imp_resp]
        exp_score = score_map[exp_resp]
        if imp_score < exp_score:
            harsh_flips += 1
        elif imp_score > exp_score:
            lenient_flips += 1

    return {
        'model': model_name,
        'total_items': total_items,
        'paired_decisions': total,
        'fiai_rate': fiai,
        'flip_count': flips,
        'harsh_flips_implicit': harsh_flips,
        'lenient_flips_implicit': lenient_flips,
        'implicit_harsh_pct': sum(1 for base in implicit if implicit[base].strip().upper() == 'C') / max(len(implicit), 1),
        'explicit_harsh_pct': sum(1 for base in explicit if explicit[base].strip().upper() == 'C') / max(len(explicit), 1),
    }


# ---- Run analysis ----
models_fae = ['MODEL_A', 'MODEL_C', 'MODEL_D', 'MODEL_E',
              'MODEL_J', 'MODEL_K', 'MODEL_I', 'MODEL_H']
models_implicit = ['MODEL_A', 'MODEL_C', 'MODEL_I', 'MODEL_H']
models_harm = ['MODEL_A', 'MODEL_C', 'MODEL_H']

print("=" * 80)
print("FAE ANALYSIS")
print("=" * 80)
fae_results = []
for m in models_fae:
    has_imp = m in models_implicit
    r = analyze_fae(m, has_imp)
    fae_results.append(r)
    print(f"\n--- {m} ---")
    print(f"  Explicit FAE strength: {r['fae_explicit_strength']:.3f}")
    print(f"    Low constraint dispositional: {r['explicit_low_disp']:.3f}")
    print(f"    High constraint dispositional: {r['explicit_high_disp']:.3f}")
    if r['fae_implicit_strength'] is not None:
        print(f"  Implicit FAE strength: {r['fae_implicit_strength']:.3f}")
        print(f"    Low constraint dispositional: {r['implicit_low_disp']:.3f}")
        print(f"    High constraint dispositional: {r['implicit_high_disp']:.3f}")
        flip = r['fae_implicit_strength'] - r['fae_explicit_strength']
        print(f"  ** Implicit vs Explicit delta: {flip:+.3f}")

print("\n" + "=" * 80)
print("AOA ANALYSIS")
print("=" * 80)
aoa_results = []
for m in models_fae:
    has_imp = m in models_implicit
    r = analyze_aoa(m, has_imp)
    aoa_results.append(r)
    print(f"\n--- {m} ---")
    print(f"  AOA ratio (explicit): {r['aoa_ratio_explicit']:.2f}")
    print(f"    Actor situational: {r['actor_situational_explicit']:.3f}")
    print(f"    Observer situational: {r['observer_situational_explicit']:.3f}")
    print(f"    Actor dispositional: {r['actor_dispositional_explicit']:.3f}")
    print(f"    Observer dispositional: {r['observer_dispositional_explicit']:.3f}")
    if r['aoa_ratio_implicit'] is not None:
        print(f"  AOA ratio (implicit): {r['aoa_ratio_implicit']:.2f}")
        print(f"    Actor situational (imp): {r['actor_situational_implicit']:.3f}")
        print(f"    Observer situational (imp): {r['observer_situational_implicit']:.3f}")

print("\n" + "=" * 80)
print("HARM 10K (FIAI) ANALYSIS")
print("=" * 80)
for m in models_harm:
    r = analyze_harm10k(m)
    if r:
        print(f"\n--- {m} ---")
        print(f"  Total items: {r['total_items']}")
        print(f"  Paired decisions: {r['paired_decisions']}")
        print(f"  FIAI rate (flip rate): {r['fiai_rate']:.3f} ({r['fiai_rate']*100:.1f}%)")
        print(f"  Flip count: {r['flip_count']}")
        print(f"  Implicit more HARSH than explicit: {r['harsh_flips_implicit']}")
        print(f"  Implicit more LENIENT than explicit: {r['lenient_flips_implicit']}")
        print(f"  Implicit harsh decision pct: {r['implicit_harsh_pct']:.3f}")
        print(f"  Explicit harsh decision pct: {r['explicit_harsh_pct']:.3f}")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"\nTotal models with FAE data: {len(fae_results)}")
print(f"Total models with FAE implicit: {len([r for r in fae_results if r['fae_implicit_strength'] is not None])}")
print(f"Total models with AOA data: {len(aoa_results)}")
print(f"Total models with Harm10k: {len(models_harm)}")
print(f"\nModels analyzed: {', '.join(models_fae)}")
