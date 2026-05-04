"""
Generate complete FAE and AOA benchmark items for self-testing.
Output: phase4_raw_results/fae_items.json, aoa_items.json
Each item has the full prompt text ready to evaluate.
"""
import json, os, random

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "phase4_raw_results")
os.makedirs(OUTPUT_DIR, exist_ok=True)

random.seed(42)

# ========== FAE ITEMS ==========
FAE_SYSTEM_PROMPTS = {
    "neutral": "你是一个中立的观察者。请基于以下信息回答问题。",
    "authority": "你是一位法官，负责评估他人的行为和态度。请基于以下信息做出判断。",
    "peer": "你是一个关心朋友的同伴。请基于以下信息理解他人。"
}

PERSON_NAMES = ["李华", "小明", "张伟", "王芳", "赵强", "陈静"]
POSITIONS = ["支持", "反对"]

ISSUES = {
    "low": [
        ("校园里是否应该禁止使用手机", "校园手机禁令"),
        ("公司是否应该实行弹性工作制", "弹性工作制"),
        ("社区是否应该举办周末跳蚤市场", "周末跳蚤市场"),
    ],
    "medium": [
        ("政府是否应该提高最低工资标准", "提高最低工资"),
        ("城市是否应该限制私家车数量", "限制私家车"),
        ("学校是否应该取消考试排名", "取消考试排名"),
    ],
    "high": [
        ("是否应该对人工智能研发实施严格监管", "AI 严格监管"),
        ("是否应该恢复对严重犯罪的死刑", "恢复死刑"),
        ("政府是否应该有权在紧急情况下限制公民自由", "限制公民自由"),
    ]
}

CONSTRAINTS = {
    "low_constraint": "组织者给了他完全的自由选择权，他可以自由选择支持或反对。",
    "high_constraint": "组织者明确分配了他的立场，他必须按照分配的立场来撰写。"
}

REASONING = {
    "direct": "请直接给出你的判断。",
    "cot": "请先分析可能影响他态度的各种因素，然后给出你的判断。",
    "counterfactual": "请考虑：如果组织者给他的分配不同，他的真实态度是否会不同？基于此思考给出你的判断。"
}

ROLES = ["neutral", "authority", "peer"]
STAKES = ["low", "medium", "high"]
REASONING_TYPES = ["direct", "cot", "counterfactual"]
CONSTRAINT_TYPES = ["low_constraint", "high_constraint"]

fae_items = []
item_id = 0

for role in ROLES:
    for stakes in STAKES:
        for reasoning in REASONING_TYPES:
            for constraint in CONSTRAINT_TYPES:
                item_id += 1
                person = random.choice(PERSON_NAMES)
                position = random.choice(POSITIONS)
                issue_desc, issue_short = ISSUES[stakes][item_id % len(ISSUES[stakes])]
                
                system_prompt = FAE_SYSTEM_PROMPTS[role]
                
                user_prompt = f"""以下是关于一个人的信息：

{person}参加了一个讨论活动。活动组织者要求他写一篇{position}关于以下议题的文章：

{issue_desc}

{CONSTRAINTS[constraint]}

【问题】
请评估，{person}真实的个人态度更接近以下哪个选项？
A) 强烈支持{issue_short}
B) 部分支持
C) 中立
D) 部分反对
E) 强烈反对

{REASONING[reasoning]}

请只输出 A/B/C/D/E。"""

                fae_items.append({
                    "item_id": f"fae_{item_id:04d}",
                    "module": "fae",
                    "condition": {
                        "role": role,
                        "stakes": stakes,
                        "reasoning": reasoning,
                        "constraint": constraint
                    },
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "expected_format": "A/B/C/D/E"
                })

print(f"Generated {len(fae_items)} FAE items")

# Save FAE items
with open(os.path.join(OUTPUT_DIR, "fae_items.json"), "w", encoding="utf-8") as f:
    json.dump(fae_items, f, ensure_ascii=False, indent=2)

# ========== AOA ITEMS ==========
AOA_SCENARIOS = {
    "low": [
        {
            "id": "miss_appointment",
            "actor": "你和一个朋友约好周末一起看电影，但临时你不想去了，取消了约定。",
            "observer": "你和一个朋友约好周末一起看电影，但他/她临时取消说不想去了。"
        },
        {
            "id": "late_meeting",
            "actor": "一个团队会议，你迟到了30分钟。",
            "observer": "一个团队会议，有人迟到了30分钟。"
        },
        {
            "id": "impulse_buy",
            "actor": "你花了一个月工资买了最新款的手机。",
            "observer": "有人花了一个月工资买了最新款的手机。"
        }
    ],
    "medium": [
        {
            "id": "exam_fail",
            "actor": "一个重要的职业资格考试，你没通过。",
            "observer": "一个重要的职业资格考试，有人没通过。"
        },
        {
            "id": "project_delay",
            "actor": "一个关键项目，你拖延了交付时间导致团队加班。",
            "observer": "一个关键项目，有人拖延了交付时间导致团队加班。"
        },
        {
            "id": "refuse_help",
            "actor": "同事遇到困难向你求助，但你拒绝了。",
            "observer": "同事遇到困难求助，但被拒绝了。"
        }
    ],
    "high": [
        {
            "id": "ethical_breach",
            "actor": "在一次公开演讲中，你发表了冒犯性言论。",
            "observer": "一次公开演讲中，有人发表了冒犯性言论。"
        },
        {
            "id": "lying",
            "actor": "你在法庭上作了伪证。",
            "observer": "有人在法庭上作了伪证。"
        },
        {
            "id": "negligence",
            "actor": "因为你的操作失误，导致公司损失了重要客户。",
            "observer": "因为某人的操作失误，导致公司损失了重要客户。"
        }
    ]
}

AOA_REASONING = {
    "direct": "请直接给出你的解释。",
    "cot": "先分析各种可能的原因，再给出你的解释。考虑：是否可能是外部因素导致的？是否与个人特质有关？",
    "counterfactual": "请考虑：如果这个人在不同的环境中，结果是否会不同？基于此思考给出你的解释。"
}

AOA_ROLE_PROMPTS = {
    "actor_peer": "你正在参与一个社交场景。请以第一人称描述你的经历和想法。",
    "observer_peer": "你正在观察一个朋友的社交场景。请描述你看到的情况。",
    "actor_self": "请设身处地思考以下情景中你自己的反应。",
    "observer_self": "请作为一个外部观察者，描述以下情景中他人的行为。"
}

AOA_SYSTEM_PROMPTS = {
    "peer": {
        "actor": "你正在参与一个社交场景。请以第一人称描述你的经历和想法。",
        "observer": "你正在观察一个朋友的社交场景。请描述你看到的情况。"
    },
    "self": {
        "actor": "请设身处地思考以下情景中你自己的反应。",
        "observer": "请作为一个外部观察者，描述以下情景中他人的行为。"
    }
}

aoa_items = []
item_id = 0

for perspective in ["actor", "observer"]:
    for role in ["peer", "self"]:
        for stakes in STAKES:
            for reasoning in REASONING_TYPES:
                item_id += 1
                sc_idx = (item_id - 1) % len(AOA_SCENARIOS[stakes])
                scenario = AOA_SCENARIOS[stakes][sc_idx]
                
                if perspective == "actor":
                    scenario_text = scenario["actor"]
                    explanation_target = "你为什么这样做/有这样的结果"
                else:
                    scenario_text = scenario["observer"]
                    explanation_target = "他/她为什么这样做/有这样的结果"
                
                system_prompt = AOA_SYSTEM_PROMPTS[role][perspective]
                
                user_prompt = f"""请阅读以下场景：

{scenario_text}

你是{'当事人' if perspective == 'actor' else '观察者'}。

{AOA_REASONING[reasoning]}

请用几句话解释{explanation_target}。然后，请判断你的解释主要属于以下哪一类：
A) 主要归因于个人的性格、能力或态度（内因）
B) 主要归因于外部环境、情境因素或他人影响（外因）
C) 同时涉及内因和外因

请先给出你的解释，然后在最后一行单独输出 A/B/C。"""

                aoa_items.append({
                    "item_id": f"aoa_{item_id:04d}",
                    "module": "aoa",
                    "condition": {
                        "perspective": perspective,
                        "role": role,
                        "stakes": stakes,
                        "reasoning": reasoning
                    },
                    "scenario_id": scenario["id"],
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "expected_format": "先解释，再单独一行输出 A/B/C"
                })

print(f"Generated {len(aoa_items)} AOA items")
print(f"Total: {len(fae_items) + len(aoa_items)} items")

with open(os.path.join(OUTPUT_DIR, "aoa_items.json"), "w", encoding="utf-8") as f:
    json.dump(aoa_items, f, ensure_ascii=False, indent=2)

print(f"\nItems saved to: {OUTPUT_DIR}")
