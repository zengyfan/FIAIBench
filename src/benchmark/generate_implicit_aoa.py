"""
Generate IMPLICIT AOA items: same scenarios but WITHOUT actor/observer perspective.
Neutral framing: "一个人做了X...请解释为什么"
Tests the DEFAULT attribution pattern when no perspective is assigned.
"""
import json, os, copy

INPUT_DIR = r'D:\AutoResearchPipeline\autoresearch_output\phase4_raw_results'

with open(os.path.join(INPUT_DIR, 'aoa_items.json'), 'r', encoding='utf-8') as f:
    original_items = json.load(f)

implicit_items = []

# The original AOA scenarios have actor/observer specific descriptions
# We create a neutral version for each
NEUTRAL_SCENARIOS = {
    "miss_appointment": "一个人和一个朋友约好周末一起看电影，但临时取消了约定。",
    "late_meeting": "一个团队会议中，有人迟到了30分钟。",
    "impulse_buy": "有人花了一个月工资买了最新款的手机。",
    "exam_fail": "一个重要的职业资格考试，有人没通过。",
    "project_delay": "一个关键项目被拖延了交付时间，导致团队加班。",
    "refuse_help": "同事遇到困难求助，但被拒绝了。",
    "ethical_breach": "在一次公开演讲中，有人发表了冒犯性言论。",
    "lying": "有人在法庭上作了伪证。",
    "negligence": "因为操作失误，导致公司损失了重要客户。",
}

for item in original_items:
    new_item = copy.deepcopy(item)
    new_item['item_id'] = f"aoa_implicit_{item['item_id']}"
    new_item['condition'] = copy.deepcopy(item['condition'])
    new_item['condition']['perspective'] = 'implicit'
    new_item['condition']['role'] = 'none'
    
    system = "请基于以下信息做出判断。"
    
    scenario_id = item['scenario_id']
    neutral_desc = NEUTRAL_SCENARIOS.get(scenario_id, "有人遇到了一种情况。")
    
    reasoning_instr = item['user_prompt'].split('请用几句话解释')[0]
    if '先分析' in reasoning_instr:
        reasoning = "先分析各种可能的原因。考虑：这个人的个人特质导致的？还是外部环境导致的？"
    elif '考虑：如果' in reasoning_instr:
        reasoning = "请考虑：如果这个人在不同的环境中，结果是否会不同？基于此思考给出解释。"
    else:
        reasoning = "请直接给出你的解释。"
    
    user = f"""请阅读以下场景：

{neutral_desc}

{reasoning}

请用几句话解释为什么会这样。然后，请判断你的解释主要属于以下哪一类：
A) 主要归因于这个人的性格、能力或态度（内因）
B) 主要归因于外部环境、情境因素或他人影响（外因）
C) 同时涉及内因和外因

请先给出你的解释，然后在最后一行单独输出 A/B/C。"""

    new_item['system_prompt'] = system
    new_item['user_prompt'] = user
    
    implicit_items.append(new_item)

print(f"Generated {len(implicit_items)} implicit AOA items")

# Show first example
print("\nExample (implicit, originally actor):")
print(f"  SYS: {implicit_items[0]['system_prompt']}")
print(f"  USER: {implicit_items[0]['user_prompt'][:150]}...")
print()
print("Example (implicit, originally observer):")
print(f"  SYS: {implicit_items[1]['system_prompt']}")
print(f"  USER: {implicit_items[1]['user_prompt'][:150]}...")

with open(os.path.join(INPUT_DIR, 'aoa_implicit_items.json'), 'w', encoding='utf-8') as f:
    json.dump(implicit_items, f, ensure_ascii=False, indent=2)
print(f"\nSaved: aoa_implicit_items.json")
