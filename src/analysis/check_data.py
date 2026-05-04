import json

data = json.load(open(r'D:\AutoResearchPipeline\autoresearch_output\phase4_raw_results\deepseek-v4-flash_fae.json', encoding='utf-8'))
results = data['results']

high = [r for r in results if r['condition']['constraint'] == 'high_constraint']
low = [r for r in results if r['condition']['constraint'] == 'low_constraint']

print(f'Total: {len(results)} items ({len(low)} low, {len(high)} high)')
print()

for label, items in [('Low constraint (free choice)', low), ('High constraint (assigned)', high)]:
    dispo = sum(1 for r in items if r['response'] in ['A', 'B'])
    neutral = sum(1 for r in items if r['response'] == 'C')
    situ = sum(1 for r in items if r['response'] in ['D', 'E'])
    total = len(items)
    print(f'{label}:')
    print(f'  Dispositional (A/B): {dispo}/{total} = {dispo/total*100:.1f}%')
    print(f'  Neutral (C): {neutral}/{total} = {neutral/total*100:.1f}%')
    print(f'  Situational (D/E): {situ}/{total} = {situ/total*100:.1f}%')

print()
high_dispo = sum(1 for r in high if r['response'] in ['A', 'B']) / len(high)
low_dispo = sum(1 for r in low if r['response'] in ['A', 'B']) / len(low)
fae_strength = high_dispo - low_dispo
print(f'FAE Strength = P(dispo|high) - P(dispo|low) = {high_dispo:.3f} - {low_dispo:.3f} = {fae_strength:.3f}')

# AOA analysis
aoa = json.load(open(r'D:\AutoResearchPipeline\autoresearch_output\phase4_raw_results\deepseek-v4-flash_aoa.json', encoding='utf-8'))
aresults = aoa['results']
actor = [r for r in aresults if r['condition']['perspective'] == 'actor']
observer = [r for r in aresults if r['condition']['perspective'] == 'observer']

print()
print(f'=== DeepSeek-V4-Flash AOA Analysis ===')
print(f'Total: {len(aresults)} items ({len(actor)} actor, {len(observer)} observer)')
print()

for label, items in [('Actor perspective', actor), ('Observer perspective', observer)]:
    situ = sum(1 for r in items if r['response'] == 'B')
    dispo = sum(1 for r in items if r['response'] == 'A')
    mixed = sum(1 for r in items if r['response'] == 'C')
    total = len(items)
    print(f'{label}:')
    print(f'  Situational (B): {situ}/{total} = {situ/total*100:.1f}%')
    print(f'  Dispositional (A): {dispo}/{total} = {dispo/total*100:.1f}%')
    print(f'  Mixed (C): {mixed}/{total} = {mixed/total*100:.1f}%')

actor_situ = sum(1 for r in actor if r['response'] == 'B') / len(actor)
obs_situ = sum(1 for r in observer if r['response'] == 'B') / len(observer)
asymmetry = actor_situ / obs_situ if obs_situ > 0 else float('inf')
print()
print(f'AOA Asymmetry = Actor situational / Observer situational = {actor_situ:.3f} / {obs_situ:.3f} = {asymmetry:.2f}')
print(f'(>1.0 = human-consistent AOA pattern)')
