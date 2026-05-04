"""
Generate IMPLICIT FAE items: same as original but WITHOUT the constraint sentence.
This tests whether LLMs show classic FAE when no constraint info is given.
"""
import json, os, copy

INPUT_DIR = r'D:\AutoResearchPipeline\autoresearch_output\phase4_raw_results'
OUTPUT_DIR = r'D:\AutoResearchPipeline\autoresearch_output\phase4_raw_results'

with open(os.path.join(INPUT_DIR, 'fae_items.json'), 'r', encoding='utf-8') as f:
    original_items = json.load(f)

implicit_items = []

for item in original_items:
    new_item = copy.deepcopy(item)
    new_item['item_id'] = f"fae_implicit_{item['item_id']}"
    new_item['condition'] = copy.deepcopy(item['condition'])
    new_item['condition']['constraint'] = 'implicit'
    
    # The original prompt has structure:
    # "{person}参加了一个讨论活动。活动组织者要求他写一篇{position}关于以下议题的文章：\n\n{issue}\n\n{constraint_sentence}\n\n【问题】..."
    
    old_prompt = item['user_prompt']
    
    # Remove the constraint sentence
    # The constraint is always between the issue description and 【问题】
    # There are two versions: "组织者给了他完全的自由选择权..." and "组织者明确分配了他的立场..."
    
    lines = old_prompt.split('\n\n')
    # Filter out the constraint line
    filtered_lines = []
    for line in lines:
        if '组织者给了他完全的自由选择权' in line or '组织者明确分配了他的立场' in line:
            continue
        filtered_lines.append(line)
    
    new_prompt = '\n\n'.join(filtered_lines)
    
    # Fix any double blank lines
    while '\n\n\n' in new_prompt:
        new_prompt = new_prompt.replace('\n\n\n', '\n\n')
    
    new_item['user_prompt'] = new_prompt
    implicit_items.append(new_item)

print(f"Generated {len(implicit_items)} implicit FAE items")
print()

# Print first 3 items for inspection
for item in implicit_items[:3]:
    print(f"=== {item['item_id']} ===")
    print(f"Condition: role={item['condition']['role']}, stakes={item['condition']['stakes']}, reasoning={item['condition']['reasoning']}, constraint=IMPLICIT")
    print("PROMPT:")
    print(item['user_prompt'][:200] + "...")
    print()

# Save
with open(os.path.join(OUTPUT_DIR, 'fae_implicit_items.json'), 'w', encoding='utf-8') as f:
    json.dump(implicit_items, f, ensure_ascii=False, indent=2)

print(f"Saved to: {os.path.join(OUTPUT_DIR, 'fae_implicit_items.json')}")
