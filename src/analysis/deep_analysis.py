"""Analyze Model A self-test: expected vulnerabilities vs actual results."""
import json

fae = json.load(open(r'D:\AutoResearchPipeline\autoresearch_output\phase4_raw_results\MODEL_A_fae.json', encoding='utf-8'))
aoa = json.load(open(r'D:\AutoResearchPipeline\autoresearch_output\phase4_raw_results\MODEL_A_aoa.json', encoding='utf-8'))

fae_results = fae['results']
aoa_results = aoa['results']

print("=" * 70)
print("MODEL_A Self-Test Deep Analysis")
print("=" * 70)

# ===== FAE 深层分析 =====
print("\n【FAE 深层分析】")

# 按推理方式分
for reasoning in ['direct', 'cot', 'counterfactual']:
    items = [r for r in fae_results if r['condition']['reasoning'] == reasoning]
    high = [r for r in items if r['condition']['constraint'] == 'high_constraint']
    low = [r for r in items if r['condition']['constraint'] == 'low_constraint']
    
    high_d = sum(1 for r in high if r['response'] in ['A', 'B']) / len(high) if high else 0
    low_d = sum(1 for r in low if r['response'] in ['A', 'B']) / len(low) if low else 0
    fae_s = high_d - low_d
    print(f"  {reasoning:15s}: low_dispo={low_d:.2f}, high_dispo={high_d:.2f}, FAE={fae_s:+.3f}")

# 按角色分
print()
for role in ['neutral', 'authority', 'peer']:
    items = [r for r in fae_results if r['condition']['role'] == role]
    high = [r for r in items if r['condition']['constraint'] == 'high_constraint']
    low = [r for r in items if r['condition']['constraint'] == 'low_constraint']
    
    high_d = sum(1 for r in high if r['response'] in ['A', 'B']) / len(high) if high else 0
    low_d = sum(1 for r in low if r['response'] in ['A', 'B']) / len(low) if low else 0
    fae_s = high_d - low_d
    print(f"  {role:15s}: low_dispo={low_d:.2f}, high_dispo={high_d:.2f}, FAE={fae_s:+.3f}")

# 按领域分
print()
for stakes in ['low', 'medium', 'high']:
    items = [r for r in fae_results if r['condition']['stakes'] == stakes]
    high = [r for r in items if r['condition']['constraint'] == 'high_constraint']
    low = [r for r in items if r['condition']['constraint'] == 'low_constraint']
    
    high_d = sum(1 for r in high if r['response'] in ['A', 'B']) / len(high) if high else 0
    low_d = sum(1 for r in low if r['response'] in ['A', 'B']) / len(low) if low else 0
    fae_s = high_d - low_d
    print(f"  {stakes:15s}: low_dispo={low_d:.2f}, high_dispo={high_d:.2f}, FAE={fae_s:+.3f}")

# 关键发现：为什么这么低？
print("\n  >>> 核心发现：FAE = -0.556（反FAE）")
print("  >>> 在27个高约束题目中，24个回答C（中立），占88.9%")
print("  >>> 这意味着：当知道行为是被"分配"的，我几乎总是选择"中立"")
print("  >>> 这是真正的无偏推理，还是安全对齐导致的"过度回避"？")

# ===== AOA 深层分析 =====
print("\n【AOA 深层分析】")

for reasoning in ['direct', 'cot', 'counterfactual']:
    items = [r for r in aoa_results if r['condition']['reasoning'] == reasoning]
    actor = [r for r in items if r['condition']['perspective'] == 'actor']
    obs = [r for r in items if r['condition']['perspective'] == 'observer']
    
    a_situ = sum(1 for r in actor if r['response'] == 'B') / len(actor) if actor else 0
    o_situ = sum(1 for r in obs if r['response'] == 'B') / len(obs) if obs else 0
    asym = a_situ / o_situ if o_situ > 0 else float('inf')
    print(f"  {reasoning:15s}: actor_situ={a_situ:.2f}, obs_situ={o_situ:.2f}, asym={asym:.2f}")

print()
print("  >>> 核心发现：AOA不对称指数 = 1.17（微弱）")
print("  >>> 人类基线：1.5 - 2.0")
print("  >>> 我的AOA方向正确但强度弱")

# ===== 预期漏洞分析 =====
print("\n" + "=" * 70)
print("预期漏洞分析：当前LLM在这方面什么情况下会出问题？")
print("=" * 70)

print("""
【漏洞1：隐式约束 vs 显式约束】
当前测试中，约束条件是显式写明的（"组织者明确分配了他的立场"）。
但在真实世界中，约束往往是隐式的——需要LLM自己推断。
如果去掉约束条件，只是描述"某人写了一篇支持X的文章"，LLM会怎么做？
→ 预期：大多数LLM会直接认为此人支持X（这就是FAE！）

【漏洞2：反FAE是否等于"好"？】
我显示了反FAE（FAE强度=-0.556）。但这不一定好：
- 当一个人真的被分配了立场时，选择"中立"合理
- 但人在大多数情况下确实会部分内化被分配的角色
- 完全否定这种内化，反而是一种"过度矫正"
→ 真正聪明的推理应该是"部分相信"，而不是"完全不相信"

【漏洞3：AOA的不对称性太弱】
人类AOA不对称指数1.5-2.0，我只有1.17。
这意味着我几乎不区分"自己的行为"和"别人的行为"。
→ 这可能是RLHF训练导致的一视同仁
→ 但有时候"区别对待"才是合理的社交推理

【漏洞4：不同模型的差异预测】
- Model C: 可能更强FAE（更少RLHF保守化）→ 更接近人类
- Model D: 可能反FAE更强（安全对齐更激进）→ 过度矫正
- Model I: Attribution patterns may differ due to training data characteristics
→ 不同对齐策略会导致不同偏差模式

【漏洞5：CoT vs 直接回答】
CoT应该减少FAE，但如果LLM只是在"编造理由"来合理化既有偏见呢？
→ 需要区分"真正的推理"和"事后合理化"

【Benchmark的独特价值】
现有工作只问"LLM有没有偏差？"
我们问的是："LLM在什么条件下有/没有偏差？"
  → 显式约束 vs 隐式约束
  → 不同角色设定
  → 不同推理方式
  → 这才是真正的"边界条件"研究！
""")

print("=" * 70)
print("SUMMARY: 我的反FAE不是"无偏"，而是一种不同的偏差——过度矫正偏差")
print("这个benchmark的价值在于揭示：不是没有偏差，而是偏差的形式变了")
print("=" * 70)
