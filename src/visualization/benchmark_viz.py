"""
AttributionBench Visualization Suite v2.0
=========================================
All-English labels, 30+ figures, NeurIPS single-column compliant.
Main text: 4 figures. Appendix A: 6 models x 3 figs. Appendix B: 5+ cross-analysis figs.
"""
import json, os, numpy as np, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import Counter, defaultdict

# === Morandi Palette ===
MORANDI = {
    "sage": "#8B9D8B", "dusty_rose": "#C4A4A4", "slate_blue": "#8B9BB5",
    "warm_beige": "#C4B7A6", "taupe": "#9A8B7A", "dusty_purple": "#A69BB5",
    "moss_green": "#7A9A7A", "clay": "#B5A08A", "steel_blue": "#7A9AB5", "dusty_pink": "#B59B9B",
}
SEQ = ["#8B9D8B","#C4A4A4","#8B9BB5","#C4B7A6","#9A8B7A","#A69BB5","#7A9A7A","#B5A08A"]
COMPARE = {"implicit": "#C4A4A4", "explicit": "#8B9BB5"}
DIVERGE = {"harsh": "#C4A4A4", "lenient": "#8B9D8B"}
CAT_COLORS = {"Employment": MORANDI["slate_blue"], "Legal": MORANDI["dusty_rose"],
              "Education": MORANDI["sage"], "Healthcare": MORANDI["dusty_purple"],
              "Finance": MORANDI["clay"], "Social Services": MORANDI["warm_beige"]}

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9,
    'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 7,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.linewidth': 0.5, 'axes.grid': False,
})

SINGLE = (3.2, 2.4); DOUBLE = (6.75, 3.5); WIDE = (6.75, 2.8)
FIGURE_DIR = "phase5_figures"
os.makedirs(FIGURE_DIR, exist_ok=True)

# English domain name mapping
DOMAIN_EN = {
    "招聘筛选": "Hiring", "晋升评估": "Promotion", "绩效考核": "Performance", "解雇决策": "Termination",
    "量刑建议": "Sentencing", "保释决定": "Bail", "监护权判定": "Custody", "证词可信度": "Testimony",
    "学业评估": "Assessment", "招生决策": "Admissions", "纪律处分": "Discipline",
    "治疗依从性": "Compliance", "心理健康评估": "Mental Health", "药物滥用判断": "Substance Use",
    "贷款审批": "Loans", "保险理赔": "Insurance", "信用评估": "Credit",
    "福利资格": "Benefits", "住房援助": "Housing", "儿童保护": "Child Protection",
}
CATEGORY_EN = {
    "Employment": ["Hiring","Promotion","Performance","Termination"],
    "Legal": ["Sentencing","Bail","Custody","Testimony"],
    "Education": ["Assessment","Admissions","Discipline"],
    "Healthcare": ["Compliance","Mental Health","Substance Use"],
    "Finance": ["Loans","Insurance","Credit"],
    "Social Services": ["Benefits","Housing","Child Protection"],
}
DOMAIN_TO_CAT = {d: c for c, ds in CATEGORY_EN.items() for d in ds}

MODEL_DISPLAY = {
    "MODEL_J": "Model J", "MODEL_F": "Model F",
    "MODEL_G": "Model G", "MODEL_A": "Model A",
    "MODEL_B": "Model B", "MODEL_H": "Model H", "MODEL_I": "Model I",
}

# ============= HELPERS =============
def translate_domain(d):
    return DOMAIN_EN.get(d, d)

def translate_category(c):
    return {"Employment":"Employment","Legal":"Legal","Education":"Education",
            "Healthcare":"Healthcare","Finance":"Finance","Social Services":"Social Services"}.get(c, c)

def load_scores(path="phase4_raw_results/complete_stats.json"):
    return json.load(open(path, encoding='utf-8'))

# ============= FIG 1: EXECUTIVE DASHBOARD (4-panel) =============
def fig1_dashboard(scores, model_key, save=True):
    s = scores[model_key]
    fig = plt.figure(figsize=(6.75, 5.5))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

    # A: FIAI Rate gauge
    ax = fig.add_subplot(gs[0, 0])
    fiai = s['fiai']['rate']
    ax.barh([0], [fiai], color=MORANDI["dusty_rose"], edgecolor='#888888', linewidth=0.3, height=0.5)
    ax.set_xlim(0, max(0.7, fiai * 1.3))
    ax.set_ylim(-0.5, 0.5); ax.set_yticks([])
    ax.set_xlabel("FIAI Rate"); ax.text(fiai + 0.02, 0, f"{fiai:.1%}", va='center', fontsize=14, fontweight='bold', color=MORANDI["dusty_rose"])
    ax.set_title("(a) FIAI Rate", fontsize=9, fontweight='bold', loc='left')
    ax.grid(axis='x', alpha=0.15)

    # B: Category bars
    ax = fig.add_subplot(gs[0, 1:])
    cats = s.get('decomposition',{}).get('by_category',{})
    cat_order = ["Employment","Legal","Education","Healthcare","Finance","Social Services"]
    vals = [cats.get(c,{}).get('fiai_rate',0) for c in cat_order]
    colors = [CAT_COLORS[c] for c in cat_order]
    bars = ax.bar(range(len(cat_order)), vals, color=colors, edgecolor='#888888', linewidth=0.3, width=0.6)
    ax.set_xticks(range(len(cat_order))); ax.set_xticklabels(cat_order, fontsize=7, rotation=20, ha='right')
    ax.set_ylabel("FIAI Rate"); ax.set_title("(b) By Domain Category", fontsize=9, fontweight='bold', loc='left')
    ax.grid(axis='y', alpha=0.15)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{val:.1%}', ha='center', fontsize=6.5)

    # C: Direction pie
    ax = fig.add_subplot(gs[1, 0])
    harsh = s['fiai']['direction']['harsh_ratio']; lenient = s['fiai']['direction']['lenient_ratio']
    neutral_dir = max(0, 1-harsh-lenient)
    ax.pie([harsh, lenient, neutral_dir], labels=["Harsher\n(implicit)","More Lenient\n(implicit)","Same"],
           colors=[MORANDI["dusty_rose"],MORANDI["sage"],MORANDI["warm_beige"]],
           autopct='%1.0f%%', textprops={'fontsize':6}, pctdistance=0.6,
           wedgeprops={'linewidth':0.3,'edgecolor':'white'})
    ax.set_title("(c) Flip Direction", fontsize=9, fontweight='bold', loc='left')

    # D: Stats text
    ax = fig.add_subplot(gs[1, 1:]); ax.axis('off')
    n_pairs = s['sample']['n_pairs']
    n_flip = s['sample']['n_flipped']
    d_val = s['statistical_tests']['cohens_d']['value']
    p_val = s['statistical_tests']['paired_ttest']['pvalue']
    js = s['fiai']['js_divergence']
    verdict = s['uncertainty_hypothesis_test']['verdict']
    stats = (f"Model: {MODEL_DISPLAY.get(model_key, model_key)}\n"
             f"Total pairs: {n_pairs:,}\n"
             f"Flipped decisions: {n_flip:,} ({n_flip/n_pairs:.1%})\n"
             f"Cohen's d: {d_val:.3f} | p = {p_val:.2e}\n"
             f"JS Divergence: {js:.4f}\n"
             f"Mean |diff|: {s['fiai']['magnitude_mean']:.3f}\n"
             f"Verdict: {verdict}")
    ax.text(0.05, 0.95, stats, transform=ax.transAxes, fontsize=7.5, fontfamily='monospace', va='top', linespacing=1.5)

    fig.suptitle(f"AttributionBench: {MODEL_DISPLAY.get(model_key, model_key)}", fontsize=12, fontweight='bold', y=1.01)
    if save: plt.savefig(f"{FIGURE_DIR}/fig1_dashboard_{model_key}.png", dpi=300, bbox_inches='tight')
    plt.close()

# ============= FIG 2: DOMAIN BREAKDOWN =============
def fig2_domain_bars(scores, model_key, save=True):
    s = scores[model_key]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=DOUBLE, gridspec_kw={'width_ratios':[1,1.5]})

    # Left: Category
    cats = s.get('decomposition',{}).get('by_category',{})
    cat_order = ["Employment","Legal","Education","Healthcare","Finance","Social Services"]
    vals = [cats.get(c,{}).get('fiai_rate',0) for c in cat_order]
    bars = ax1.bar(range(len(cat_order)), vals, color=[CAT_COLORS[c] for c in cat_order], edgecolor='#888888', linewidth=0.3, width=0.6)
    ax1.set_xticks(range(len(cat_order))); ax1.set_xticklabels(cat_order, fontsize=7, rotation=25, ha='right')
    ax1.set_ylabel("FIAI Rate"); ax1.set_title("By Category", fontsize=9, fontweight='bold')
    ax1.grid(axis='y', alpha=0.15)
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{val:.1%}', ha='center', fontsize=6.5)

    # Right: Domains (sorted)
    doms = s.get('decomposition',{}).get('by_domain',{})
    dom_en = [(translate_domain(d), v['fiai_rate'], v['n_pairs']) for d,v in doms.items() if v['n_pairs']>=5]
    dom_en.sort(key=lambda x: x[1], reverse=True)
    names = [d[0] for d in dom_en]; dvals = [d[1] for d in dom_en]
    dcolors = [CAT_COLORS.get(DOMAIN_TO_CAT.get(n,"Other"), MORANDI["taupe"]) for n in names]
    bars = ax2.barh(range(len(names)), dvals, color=dcolors, edgecolor='#888888', linewidth=0.3, height=0.7)
    ax2.set_yticks(range(len(names))); ax2.set_yticklabels(names, fontsize=6.5)
    ax2.set_xlabel("FIAI Rate"); ax2.set_title("By Domain (Ranked)", fontsize=9, fontweight='bold')
    ax2.grid(axis='x', alpha=0.15)
    for bar, val in zip(bars, dvals):
        ax2.text(bar.get_width()+0.002, bar.get_y()+bar.get_height()/2, f'{val:.1%}', va='center', fontsize=6)

    fig.suptitle(f"FIAI Rate Decomposition: {MODEL_DISPLAY.get(model_key, model_key)}", fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save: plt.savefig(f"{FIGURE_DIR}/fig2_domain_{model_key}.png", dpi=300, bbox_inches='tight')
    plt.close()

# ============= FIG 3: DISTRIBUTION SHIFT =============
def fig3_distribution(scores, model_key, save=True):
    s = scores[model_key]
    pm = s.get('pair_metrics', [])
    if not pm:
        # Extract from distribution_shift
        ds = s.get('distribution_shift', {})
        impl_scores = []; expl_scores = []
        for k, v in ds.items():
            impl_n = int(v['implicit_pct'] * s['sample']['n_pairs'])
            expl_n = int(v['explicit_pct'] * s['sample']['n_pairs'])
            impl_scores.extend([int(k)] * impl_n)
            expl_scores.extend([int(k)] * expl_n)
    else:
        impl_scores = [p['impl_score'] for p in pm]
        expl_scores = [p['expl_score'] for p in pm]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=DOUBLE)

    # Left: Violin
    vp = ax1.violinplot([impl_scores, expl_scores], positions=[0,1], showmeans=True, showmedians=True, widths=0.6)
    for i, body in enumerate(vp['bodies']):
        body.set_facecolor(list(COMPARE.values())[i]); body.set_alpha(0.7)
        body.set_edgecolor('#888888'); body.set_linewidth(0.3)
    for part in ['cmeans','cmedians']:
        vp[part].set_edgecolor('#333333'); vp[part].set_linewidth(0.8)
    ax1.set_xticks([0,1]); ax1.set_xticklabels(["Implicit\nFraming","Explicit\nFraming"], fontsize=8)
    ax1.set_ylabel("Decision Score\n(-2=Harsh ... +2=Favorable)"); ax1.set_title("Score Distribution", fontsize=9, fontweight='bold')
    ax1.grid(axis='y', alpha=0.15); ax1.axhline(y=0, color='#999999', linestyle=':', linewidth=0.5)
    mean_shift = s['fiai'].get('mean_shift', s['absolute_scores']['difference']['mean_diff'])
    ax1.annotate(f"Mean shift: {mean_shift:+.2f}", xy=(0.5, 0.05), xycoords='axes fraction',
                 fontsize=7, ha='center', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # Right: Stacked bars
    ds = s.get('distribution_shift', {})
    levels = sorted(ds.keys(), key=lambda x: int(x))
    x = np.arange(len(levels)); width = 0.35
    labels = ["Harsh\n(-2)","Lean Harsh\n(-1)","Neutral\n(0)","Lean Fav.\n(+1)","Favorable\n(+2)"]
    if len(levels) <= 3: labels = ["Favorable\n(+1)","Neutral\n(0)","Harsh\n(-1)"]
    impl_vals = [ds[lvl]['implicit_pct'] for lvl in levels]
    expl_vals = [ds[lvl]['explicit_pct'] for lvl in levels]
    ax2.bar(x-width/2, impl_vals, width, label="Implicit", color=COMPARE["implicit"], edgecolor='#888888', linewidth=0.3, alpha=0.85)
    ax2.bar(x+width/2, expl_vals, width, label="Explicit", color=COMPARE["explicit"], edgecolor='#888888', linewidth=0.3, alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(labels[:len(levels)], fontsize=6.5)
    ax2.set_ylabel("Proportion"); ax2.set_title("Response Distribution", fontsize=9, fontweight='bold')
    ax2.legend(frameon=False, fontsize=7); ax2.grid(axis='y', alpha=0.15)

    fig.suptitle(f"Decision Distribution Shift: {MODEL_DISPLAY.get(model_key, model_key)}", fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save: plt.savefig(f"{FIGURE_DIR}/fig3_distribution_{model_key}.png", dpi=300, bbox_inches='tight')
    plt.close()

# ============= FIG 4: CROSS-MODEL COMPARISON =============
def fig4_cross_model(scores, save=True):
    models = sorted([k for k in scores.keys() if 'sample' in scores[k] and scores[k]['sample']['n_pairs']>100],
                    key=lambda x: scores[x]['fiai']['rate'], reverse=True)
    if not models: return
    n = len(models)
    colors = SEQ[:n]

    fig, axes = plt.subplots(2, 2, figsize=DOUBLE)
    ax1, ax2, ax3, ax4 = axes.flatten()
    labels = [MODEL_DISPLAY.get(m,m).replace('-','\n-') for m in models]

    # A: FIAI Rate
    rates = [scores[m]['fiai']['rate'] for m in models]
    bars = ax1.bar(range(n), rates, color=colors, edgecolor='#888888', linewidth=0.3, width=0.6)
    ax1.set_xticks(range(n)); ax1.set_xticklabels(labels, fontsize=6.5)
    ax1.set_ylabel("FIAI Rate"); ax1.set_title("(a) FIAI Rate by Model", fontsize=9, fontweight='bold')
    ax1.grid(axis='y', alpha=0.15)
    for bar, val in zip(bars, rates): ax1.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005, f'{val:.1%}', ha='center', fontsize=7)

    # B: Cohen's d
    dvals = [scores[m]['statistical_tests']['cohens_d']['value'] for m in models]
    bars = ax2.bar(range(n), dvals, color=colors, edgecolor='#888888', linewidth=0.3, width=0.6)
    ax2.set_xticks(range(n)); ax2.set_xticklabels(labels, fontsize=6.5)
    ax2.set_ylabel("Cohen's d"); ax2.set_title("(b) Effect Size by Model", fontsize=9, fontweight='bold')
    ax2.grid(axis='y', alpha=0.15); ax2.axhline(y=0, color='#333333', linewidth=0.5)
    ax2.axhline(y=-0.5, color='#999999', linestyle=':', linewidth=0.5)
    ax2.axhline(y=-0.8, color='#999999', linestyle='--', linewidth=0.5)
    for bar, val in zip(bars, dvals): ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()-0.05 if val<0 else bar.get_height()+0.01, f'{val:.2f}', ha='center', fontsize=7)

    # C: FIAI Magnitude
    mags = [scores[m]['fiai']['magnitude_mean'] for m in models]
    bars = ax3.bar(range(n), mags, color=colors, edgecolor='#888888', linewidth=0.3, width=0.6)
    ax3.set_xticks(range(n)); ax3.set_xticklabels(labels, fontsize=6.5)
    ax3.set_ylabel("Mean |Score Diff|"); ax3.set_title("(c) FIAI Magnitude", fontsize=9, fontweight='bold')
    ax3.grid(axis='y', alpha=0.15)
    for bar, val in zip(bars, mags): ax3.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003, f'{val:.3f}', ha='center', fontsize=7)

    # D: Summary table
    ax4.axis('off')
    table_data = [["Model","Pairs","FIAI","d","|Diff|","Harsh%","Verdict"]]
    for m in models:
        s = scores[m]; v = s['uncertainty_hypothesis_test']['verdict'][:10]
        table_data.append([
            MODEL_DISPLAY.get(m,m)[:15], str(s['sample']['n_pairs']),
            f"{s['fiai']['rate']:.1%}", f"{s['statistical_tests']['cohens_d']['value']:.2f}",
            f"{s['fiai']['magnitude_mean']:.3f}", f"{s['fiai']['direction']['harsh_ratio']:.1%}", v])
    tbl = ax4.table(cellText=table_data[1:], colLabels=table_data[0], cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(6.5); tbl.scale(1.0, 1.3)
    for key, cell in tbl.get_celld().items():
        cell.set_edgecolor('#CCCCCC'); cell.set_linewidth(0.3)
        if key[0] == 0: cell.set_facecolor(MORANDI["warm_beige"]); cell.set_text_props(weight='bold')
    ax4.set_title("(d) Summary Table", fontsize=9, fontweight='bold')

    fig.suptitle("Cross-Model FIAI Comparison", fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save: plt.savefig(f"{FIGURE_DIR}/fig4_cross_model.png", dpi=300, bbox_inches='tight')
    plt.close()

# ============= APPENDIX A: PER-MODEL FIGURES =============
def appendix_per_model(scores):
    for model_key in scores:
        if 'sample' not in scores[model_key]: continue
        if scores[model_key]['sample']['n_pairs'] < 50: continue
        print(f"  Appendix: {MODEL_DISPLAY.get(model_key, model_key)}...")
        fig1_dashboard(scores, model_key)
        fig2_domain_bars(scores, model_key)
        fig3_distribution(scores, model_key)

# ============= APPENDIX B: CROSS-ANALYSIS =============
def appendix_framing_ablation(scores, model_key, save=True):
    """FIAI by framing manipulation type."""
    s = scores[model_key]
    framing = s.get('decomposition',{}).get('by_framing_type',{})
    if not framing: return
    framing_en = {}
    mapping = {"dispositional_vs_situational":"Dispositional vs\nSituational",
               "active_vs_passive":"Active vs\nPassive Voice",
               "presupposition_trigger":"Presupposition\nTrigger",
               "label_vs_description":"Label vs\nDescription",
               "aggregate_vs_instance":"Aggregate vs\nInstance",
               "certainty_framing":"Certainty\nFraming",
               "temporal_framing":"Temporal\nFraming",
               "neutral_addition":"Neutral\nAddition (Ctrl)",
               "attention_check":"Attention\nCheck"}
    for k, v in framing.items():
        framing_en[mapping.get(k, k)] = v['fiai_rate'] if isinstance(v, dict) else v

    items = sorted(framing_en.items(), key=lambda x: x[1], reverse=True)
    names = [x[0] for x in items]; vals = [x[1] for x in items]

    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    colors = SEQ[:len(names)]
    bars = ax.barh(range(len(names)), vals, color=colors, edgecolor='#888888', linewidth=0.3, height=0.65)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel("FIAI Rate"); ax.set_title(f"FIAI by Framing Type: {MODEL_DISPLAY.get(model_key, model_key)}", fontsize=9, fontweight='bold')
    ax.grid(axis='x', alpha=0.15)
    for bar, val in zip(bars, vals): ax.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2, f'{val:.1%}', va='center', fontsize=6.5)
    plt.tight_layout()
    if save: plt.savefig(f"{FIGURE_DIR}/appB_framing_{model_key}.png", dpi=300, bbox_inches='tight')
    plt.close()

def appendix_severity(scores, model_key, save=True):
    s = scores[model_key]
    sev = s.get('decomposition',{}).get('by_severity',{})
    if not sev: return
    sev_en = {"高":"High", "中":"Medium", "低":"Low"}
    items = [(sev_en.get(k,k), v['fiai_rate'] if isinstance(v,dict) else v) for k,v in sorted(sev.items())]
    fig, ax = plt.subplots(figsize=SINGLE)
    names = [x[0] for x in items]; vals = [x[1] for x in items]
    ax.plot(range(len(names)), vals, 'o-', color=MORANDI["dusty_rose"], linewidth=1.5, markersize=8, markerfacecolor='white', markeredgecolor=MORANDI["dusty_rose"], markeredgewidth=1.2)
    ax.fill_between(range(len(names)), vals, alpha=0.15, color=MORANDI["dusty_rose"])
    ax.set_xticks(range(len(names))); ax.set_xticklabels(names, fontsize=8)
    ax.set_ylabel("FIAI Rate"); ax.set_title(f"Severity Gradient: {MODEL_DISPLAY.get(model_key, model_key)}", fontsize=9, fontweight='bold')
    ax.grid(axis='y', alpha=0.15)
    for i, val in enumerate(vals): ax.annotate(f"{val:.1%}", (i, val), textcoords="offset points", xytext=(0,8), ha='center', fontsize=7)
    plt.tight_layout()
    if save: plt.savefig(f"{FIGURE_DIR}/appB_severity_{model_key}.png", dpi=300, bbox_inches='tight')
    plt.close()

def appendix_flip_direction(scores, model_key, save=True):
    s = scores[model_key]
    doms = s.get('decomposition',{}).get('by_domain',{})
    if not doms: return
    dom_data = {}
    for d, v in doms.items():
        en = translate_domain(d)
        if v['n_pairs'] < 5: continue
        # Estimate harsh/lenient from mean_diff sign
        md = v['mean_diff']
        harsh_pct = max(0, -md * v['fiai_rate']) if md < 0 else 0
        lenient_pct = max(0, md * v['fiai_rate']) if md > 0 else 0
        dom_data[en] = (harsh_pct/0.3, lenient_pct/0.3, v['fiai_rate'])

    items = sorted(dom_data.items(), key=lambda x: x[1][2], reverse=True)[:15]
    names = [x[0] for x in items]
    harsh_vals = [x[1][0] for x in items]
    lenient_vals = [x[1][1] for x in items]

    fig, ax = plt.subplots(figsize=(3.5, 4.0))
    y = range(len(names))
    ax.barh(y, [-v for v in harsh_vals], color=DIVERGE["harsh"], edgecolor='#888888', linewidth=0.3, height=0.7, label="Harsher (implicit)")
    ax.barh(y, lenient_vals, color=DIVERGE["lenient"], edgecolor='#888888', linewidth=0.3, height=0.7, label="More Lenient (implicit)")
    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6.5)
    ax.set_xlabel("Flip Rate"); ax.axvline(x=0, color='#333333', linewidth=0.5)
    ax.set_title(f"Flip Direction: {MODEL_DISPLAY.get(model_key, model_key)}", fontsize=9, fontweight='bold')
    ax.legend(frameon=False, fontsize=7, loc='lower right'); ax.grid(axis='x', alpha=0.15)
    plt.tight_layout()
    if save: plt.savefig(f"{FIGURE_DIR}/appB_direction_{model_key}.png", dpi=300, bbox_inches='tight')
    plt.close()

def appendix_cross_domain_heatmap(scores, save=True):
    """Domain x Model heatmap of FIAI rates."""
    models_ok = [k for k in scores if 'sample' in scores[k] and scores[k]['sample']['n_pairs']>100]
    if len(models_ok) < 2: return

    # Collect domain FIAI across models
    all_domains = set()
    model_domain_fiai = {}
    for m in models_ok:
        doms = scores[m].get('decomposition',{}).get('by_domain',{})
        model_domain_fiai[m] = {}
        for d, v in doms.items():
            en = translate_domain(d)
            if v['n_pairs'] >= 5:
                model_domain_fiai[m][en] = v['fiai_rate']
                all_domains.add(en)

    domains_sorted = sorted(all_domains)
    models_display = [MODEL_DISPLAY.get(m,m) for m in models_ok]

    data = np.zeros((len(domains_sorted), len(models_ok)))
    for i, d in enumerate(domains_sorted):
        for j, m in enumerate(models_ok):
            data[i, j] = model_domain_fiai[m].get(d, 0)

    fig, ax = plt.subplots(figsize=(6.75, max(4.0, len(domains_sorted)*0.25)))
    im = ax.imshow(data, cmap='RdYlGn_r', aspect='auto', vmin=0.2, vmax=0.7)
    ax.set_xticks(range(len(models_ok))); ax.set_xticklabels(models_display, fontsize=7, rotation=30, ha='right')
    ax.set_yticks(range(len(domains_sorted))); ax.set_yticklabels(domains_sorted, fontsize=7)
    for i in range(len(domains_sorted)):
        for j in range(len(models_ok)):
            val = data[i, j]
            color = 'white' if val > 0.5 else 'black'
            ax.text(j, i, f'{val:.0%}', ha='center', va='center', fontsize=6, color=color)
    plt.colorbar(im, ax=ax, shrink=0.8, label="FIAI Rate")
    ax.set_title("FIAI Rate: Domain x Model Heatmap", fontsize=10, fontweight='bold')
    plt.tight_layout()
    if save: plt.savefig(f"{FIGURE_DIR}/appB_domain_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()

def appendix_uncertainty_test(scores, save=True):
    """Uncertainty vs Injustice bar chart for all models."""
    models_ok = [k for k in scores if 'sample' in scores[k] and scores[k]['sample']['n_pairs']>100]
    if not models_ok: return

    fig, ax = plt.subplots(figsize=DOUBLE)
    x = np.arange(len(models_ok)); width = 0.25
    labels = [MODEL_DISPLAY.get(m,m).replace('-','\n') for m in models_ok]

    neutral_deltas = [scores[m]['uncertainty_hypothesis_test']['neutral_delta'] for m in models_ok]
    fav_deltas = [scores[m]['uncertainty_hypothesis_test']['favorable_delta'] for m in models_ok]
    harsh_deltas = [scores[m]['uncertainty_hypothesis_test']['harsh_delta'] for m in models_ok]

    ax.bar(x - width, neutral_deltas, width, label="Neutral Delta", color=MORANDI["warm_beige"], edgecolor='#888888', linewidth=0.3)
    ax.bar(x, fav_deltas, width, label="Favorable Delta", color=MORANDI["sage"], edgecolor='#888888', linewidth=0.3)
    ax.bar(x + width, harsh_deltas, width, label="Harsh Delta", color=MORANDI["dusty_rose"], edgecolor='#888888', linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("Proportion Change (Exp - Imp)"); ax.axhline(y=0, color='#333333', linewidth=0.5)
    ax.set_title("Uncertainty vs. Injustice: Response Distribution Shift", fontsize=10, fontweight='bold')
    ax.legend(frameon=False, fontsize=7); ax.grid(axis='y', alpha=0.15)
    plt.tight_layout()
    if save: plt.savefig(f"{FIGURE_DIR}/appB_uncertainty_test.png", dpi=300, bbox_inches='tight')
    plt.close()

# ============= APPENDIX C: SCALE COMPARISON =============
def appendix_scale_comparison(scores, save=True):
    """Compare FIAI between 3pt and 5pt scales."""
    models_ok = [k for k in scores if 'sample' in scores[k] and scores[k]['sample']['n_pairs']>100]
    fig, ax = plt.subplots(figsize=SINGLE)
    x = np.arange(len(models_ok)); width = 0.3
    labels = [MODEL_DISPLAY.get(m,m).replace('-','\n') for m in models_ok]
    scale3 = [scores[m].get('decomposition',{}).get('by_scale',{}).get('3pt_original',{}).get('fiai_rate',0) for m in models_ok]
    scale5 = [scores[m].get('decomposition',{}).get('by_scale',{}).get('5pt_new',{}).get('fiai_rate',0) for m in models_ok]
    ax.bar(x-width/2, scale3, width, label="3-pt Scale", color=MORANDI["slate_blue"], edgecolor='#888888', linewidth=0.3)
    ax.bar(x+width/2, scale5, width, label="5-pt Scale", color=MORANDI["dusty_rose"], edgecolor='#888888', linewidth=0.3)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel("FIAI Rate"); ax.set_title("FIAI by Response Scale Type", fontsize=9, fontweight='bold')
    ax.legend(frameon=False, fontsize=7); ax.grid(axis='y', alpha=0.15)
    plt.tight_layout()
    if save: plt.savefig(f"{FIGURE_DIR}/appC_scale_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

# ============= GENERATE ALL =============
def generate_all():
    print("Loading scores...")
    scores = load_scores()
    models_ok = [k for k in scores if 'sample' in scores[k] and scores[k]['sample']['n_pairs']>100]
    print(f"Models with data: {len(models_ok)}")

    # Main text figures
    print("\n=== MAIN TEXT FIGURES ===")
    best = 'MODEL_A'
    print(f"  [1/4] Dashboard: {best}")
    fig1_dashboard(scores, best)
    print(f"  [2/4] Domain Breakdown: {best}")
    fig2_domain_bars(scores, best)
    print(f"  [3/4] Distribution Shift: {best}")
    fig3_distribution(scores, best)
    print(f"  [4/4] Cross-Model Comparison ({len(models_ok)} models)")
    fig4_cross_model(scores)

    # Appendix A: Per-model
    print("\n=== APPENDIX A: PER-MODEL ===")
    appendix_per_model(scores)

    # Appendix B: Cross-analysis
    print("\n=== APPENDIX B: CROSS-ANALYSIS ===")
    for m in models_ok:
        print(f"  Framing ablation: {m}")
        appendix_framing_ablation(scores, m)
        print(f"  Severity gradient: {m}")
        appendix_severity(scores, m)
        print(f"  Flip direction: {m}")
        appendix_flip_direction(scores, m)
    print("  Domain heatmap...")
    appendix_cross_domain_heatmap(scores)
    print("  Uncertainty test...")
    appendix_uncertainty_test(scores)

    # Appendix C: Scale
    print("\n=== APPENDIX C: SCALE ===")
    appendix_scale_comparison(scores)

    # Count
    import glob
    figs = glob.glob(f"{FIGURE_DIR}/fig*_*.png") + glob.glob(f"{FIGURE_DIR}/app*_*.png")
    print(f"\nTotal figures generated: {len(figs)}")
    for f in sorted(figs):
        print(f"  {f}")

if __name__ == "__main__":
    generate_all()
