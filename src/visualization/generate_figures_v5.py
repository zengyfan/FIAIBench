#!/usr/bin/env python3
"""FiaiBench Figure Generation v5 — 2x font sizes, bold, deep pastels"""
import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams.update({
    'font.family': 'Arial', 'font.weight': 'bold', 'font.size': 36,
    'axes.titlesize': 38, 'axes.labelsize': 36, 'axes.labelweight': 'bold',
    'xtick.labelsize': 28, 'ytick.labelsize': 28, 'legend.fontsize': 28,
    'figure.dpi': 200, 'savefig.dpi': 200, 'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.5, 'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': 2.5,
})

MODEL_COLORS = {
    'deepseek-v4-flash': '#7EB8DC', 'deepseek-v4-pro': '#4A90C4',
    'qwen3-5-flash': '#6EC49A', 'qwen3-6-35b-a3b': '#3D9E70',
    'glm-5-1': '#E09088', 'kimi-k2-6': '#A890CC',
}
MODEL_DISPLAY = {
    'deepseek-v4-flash': 'DeepSeek-V4-Flash', 'deepseek-v4-pro': 'DeepSeek-V4-Pro',
    'qwen3-5-flash': 'Qwen3.5-Flash', 'qwen3-6-35b-a3b': 'Qwen3.6-35B',
    'glm-5-1': 'GLM-5.1', 'kimi-k2-6': 'Kimi-K2.6',
}
SCORE_COLORS_5 = ['#E07070', '#F0A098', '#E0E0E0', '#88D0A8', '#58B878']
HARSH_COLOR = '#E07070'; FAVOR_COLOR = '#58B878'
ANNO_SIZE = 28; ANNO_COLOR = '#333333'; NAME_SIZE = 34

BASE = Path('D:/AutoResearchPipeline/autoresearch_output')
DATA_PATH = BASE / 'phase4_raw_results' / 'complete_stats.json'
OUT_DIR = BASE / 'phase5_figures_v5'
OUT_DIR.mkdir(exist_ok=True)

with open(DATA_PATH, encoding='utf-8') as f:
    STATS = json.load(f)
MODELS = list(STATS.keys())
MODEL_ORDER = sorted(MODELS, key=lambda m: STATS[m]['fiai']['rate'], reverse=True)

def save(fig, name):
    fig.savefig(OUT_DIR / name, dpi=200, bbox_inches='tight', pad_inches=0.5, facecolor='white', edgecolor='none')
    print(f'  Saved: {name}')

def panel_label(ax, text):
    ax.text(0.003, 1.04, text, transform=ax.transAxes, fontsize=48, fontweight='bold', va='bottom', ha='left', color='#111111')

# ═══════════ ROW 1: 4 panels ═══════════
def make_row1():
    fig, axes = plt.subplots(1, 4, figsize=(52, 15))
    (ax1, ax2, ax3, ax4) = axes
    fig.subplots_adjust(wspace=0.28)

    # (a) FIAI Rate
    panel_label(ax1, '(a)')
    rates = [STATS[m]['fiai']['rate'] * 100 for m in MODEL_ORDER]
    ci_l = [STATS[m]['fiai']['rate_ci95_lower'] * 100 for m in MODEL_ORDER]
    ci_h = [STATS[m]['fiai']['rate_ci95_upper'] * 100 for m in MODEL_ORDER]
    colors = [MODEL_COLORS[m] for m in MODEL_ORDER]
    names = [MODEL_DISPLAY[m] for m in MODEL_ORDER]
    y = np.arange(len(MODEL_ORDER))
    ax1.barh(y, rates, 0.55, color=colors, edgecolor='white', linewidth=2, zorder=3)
    ax1.errorbar(rates, y, xerr=[[r-l for r,l in zip(rates,ci_l)],[u-r for r,u in zip(rates,ci_h)]],
                 fmt='none', ecolor='#555', capsize=6, capthick=2, elinewidth=2, zorder=4)
    for i, r in enumerate(rates):
        ax1.text(r + 2.5, i, f'{r:.1f}%', va='center', fontsize=ANNO_SIZE, fontweight='bold', color=ANNO_COLOR)
    ax1.set_yticks(y); ax1.set_yticklabels(names, fontsize=NAME_SIZE, fontweight='bold')
    ax1.set_xlim(0, max(rates) + 18); ax1.set_xlabel('FIAI Rate (%)')
    ax1.axvline(x=5, color='#999', linestyle='--', linewidth=3, zorder=2)
    ax1.text(5.5, -0.9, 'Chance (5%)', fontsize=24, color='#777', fontweight='bold')
    ax1.invert_yaxis(); ax1.tick_params(labelsize=28, width=2)

    # (b) Cohen's d
    panel_label(ax2, '(b)')
    d_vals = [STATS[m]['statistical_tests']['cohens_d']['value'] for m in MODEL_ORDER]
    for bmin, bmax, label in [(0,0.2,'Negligible'),(0.2,0.5,'Small'),(0.5,0.8,'Medium'),(0.8,2.5,'Large')]:
        ax2.axvspan(-bmax,-bmin,alpha=0.3,color='#E8E8E8',zorder=1)
        ax2.axvspan(bmin,bmax,alpha=0.3,color='#E8E8E8',zorder=1)
    ax2.axvline(x=0, color='#AAA', linewidth=3, zorder=2)
    for i, (d, m) in enumerate(zip(d_vals, MODEL_ORDER)):
        c = HARSH_COLOR if d < 0 else '#70A0D0'
        ax2.scatter(d, i, s=500, c=c, edgecolors='white', linewidth=3, zorder=4)
        off = 0.10 if d >= 0 else -0.10
        ax2.text(d+off, i, f'{d:+.3f}', va='center', ha='left' if d>=0 else 'right',
                 fontsize=ANNO_SIZE, fontweight='bold', color=ANNO_COLOR)
    ax2.set_yticks(y); ax2.set_yticklabels(names, fontsize=NAME_SIZE, fontweight='bold')
    ax2.set_xlabel("Cohen's d"); ax2.set_xlim(-2.5,2.5); ax2.invert_yaxis()
    for bmin,bmax,label in [(0,0.2,'N'),(0.2,0.5,'S'),(0.5,0.8,'M'),(0.8,2.5,'L')]:
        if bmax <= 0.8:
            ax2.text((bmin+bmax)/2, len(MODEL_ORDER)-0.3, label, fontsize=20, color='#AAA', ha='center', va='bottom', fontweight='bold')
    ax2.tick_params(labelsize=28, width=2)

    # (c) Harsh Bias Ratio
    panel_label(ax3, '(c)')
    hr_vals = [STATS[m]['fiai']['direction']['harsh_ratio'] for m in MODEL_ORDER]
    diverging = [h-0.5 for h in hr_vals]
    bc = ['#E07070' if dv>0 else '#70A0D0' for dv in diverging]
    ax3.barh(y, diverging, 0.65, color=bc, edgecolor='white', linewidth=2, zorder=3)
    ax3.axvline(x=0, color='#888', linewidth=3, zorder=2)
    for i, (hr, dv) in enumerate(zip(hr_vals, diverging)):
        xp = dv + (0.06 if dv>=0 else -0.06)
        ax3.text(xp, i, f'{hr:.3f}', va='center', ha='left' if dv>=0 else 'right',
                 fontsize=ANNO_SIZE, fontweight='bold', color=ANNO_COLOR)
    ax3.set_yticks(y); ax3.set_yticklabels(names, fontsize=NAME_SIZE, fontweight='bold')
    ax3.set_xlabel('Harsh Bias Ratio (deviation from 0.5)')
    md = max(abs(min(diverging)), abs(max(diverging))) * 1.6
    ax3.set_xlim(-md, md); ax3.invert_yaxis()
    xt = ax3.get_xticks()
    ax3.set_xticklabels([f'{t+0.5:.2f}' for t in xt], fontsize=28, fontweight='bold')
    ax3.tick_params(labelsize=28, width=2)

    # (d) Rate vs |d|
    panel_label(ax4, '(d)')
    for m in MODEL_ORDER:
        s = STATS[m]
        ax4.scatter(s['fiai']['rate']*100, abs(s['statistical_tests']['cohens_d']['value']),
                    s=500, c=MODEL_COLORS[m], edgecolors='white', linewidth=3, alpha=0.95, zorder=4)
        ax4.annotate(MODEL_DISPLAY[m], (s['fiai']['rate']*100, abs(s['statistical_tests']['cohens_d']['value'])),
                     textcoords='offset points', xytext=(0,22), fontsize=24, ha='center', color='#444', fontweight='bold')
    ax4.set_xlabel('FIAI Rate (%)'); ax4.set_ylabel("|Cohen's d|")
    ax4.set_xlim(min(rates)-6, max(rates)+6)
    ax4.set_ylim(0, max([abs(STATS[m]['statistical_tests']['cohens_d']['value']) for m in MODELS]) * 1.3)
    ax4.tick_params(labelsize=28, width=2)

    plt.tight_layout(pad=2)
    save(fig, 'row1_fiai_overview.png'); plt.close(fig)
    print('Row 1: done.')

# ═══════════ ROW 2: 3 panels ═══════════
def make_row2():
    fig, axes = plt.subplots(1, 3, figsize=(44, 13))
    (ax1, ax2, ax3) = axes
    fig.subplots_adjust(wspace=0.28)

    # (a) Stacked bars
    panel_label(ax1, '(a)')
    n_m = len(MODEL_ORDER); bar_w = 0.38; x_pos = np.arange(n_m) * 2.8
    for i, m in enumerate(MODEL_ORDER):
        ds = STATS[m]['distribution_shift']
        sk = ['-2','-1','0','1','2']
        imp = [ds.get(s,{}).get('implicit_pct',0)*100 for s in sk]
        exp = [ds.get(s,{}).get('explicit_pct',0)*100 for s in sk]
        bi, be = 0, 0
        for j in range(5):
            ax1.bar(x_pos[i]-bar_w/2, imp[j], bar_w, bottom=bi, color=SCORE_COLORS_5[j], edgecolor='white', linewidth=0.8)
            ax1.bar(x_pos[i]+bar_w/2, exp[j], bar_w, bottom=be, color=SCORE_COLORS_5[j], edgecolor='white', linewidth=0.8)
            bi += imp[j]; be += exp[j]
        ax1.text(x_pos[i]-bar_w/2, 104, 'I', ha='center', fontsize=22, color='#777', fontweight='bold')
        ax1.text(x_pos[i]+bar_w/2, 104, 'E', ha='center', fontsize=22, color='#777', fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER], rotation=25, ha='right', fontsize=26, fontweight='bold')
    ax1.set_ylabel('Proportion (%)'); ax1.set_ylim(0, 114); ax1.tick_params(labelsize=26, width=2)
    leg = [Patch(facecolor='#E07070',label='Harsh(-2)'),Patch(facecolor='#F0A098',label='Lean Harsh(-1)'),
           Patch(facecolor='#E0E0E0',label='Neutral(0)'),Patch(facecolor='#88D0A8',label='Lean Favor.(+1)'),
           Patch(facecolor='#58B878',label='Favorable(+2)')]
    ax1.legend(handles=leg, loc='upper right', fontsize=24, ncol=1, framealpha=0.9, edgecolor='#CCC')

    # (b) Waterfall
    panel_label(ax2, '(b)')
    y_pos = np.arange(n_m)
    for i, m in enumerate(MODEL_ORDER):
        ut = STATS[m]['uncertainty_hypothesis_test']
        nd, fd, hd = ut['neutral_delta']*100, ut['favorable_delta']*100, ut['harsh_delta']*100
        ax2.annotate('', xy=(nd,i), xytext=(0,i), arrowprops=dict(arrowstyle='->',color='#AAA',lw=9,alpha=0.7))
        ax2.annotate('', xy=(0,i+0.22), xytext=(fd,i+0.22), arrowprops=dict(arrowstyle='->',color=FAVOR_COLOR,lw=9,alpha=0.7))
        ax2.annotate('', xy=(0,i-0.22), xytext=(hd,i-0.22), arrowprops=dict(arrowstyle='->',color=HARSH_COLOR,lw=9,alpha=0.7))
        ax2.scatter([nd],[i],s=180,c='#777',zorder=5)
        ax2.scatter([fd],[i+0.22],s=180,c='#48A868',zorder=5)
        ax2.scatter([hd],[i-0.22],s=180,c='#D06050',zorder=5)
        v = 'INJ' if 'Injustice' in ut.get('verdict','') else 'MIX'
        ax2.text(-38, i, v, fontsize=26, fontweight='bold', color='#D06050' if v=='INJ' else '#AAA', va='center')
    ax2.axvline(x=0, color='#999', linewidth=2.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER], fontsize=NAME_SIZE, fontweight='bold')
    ax2.set_xlabel('% Point Change (pp)'); ax2.invert_yaxis(); ax2.tick_params(labelsize=28, width=2)
    leg2 = [Patch(facecolor='#AAA',label='Neutral'),Patch(facecolor=FAVOR_COLOR,label='Favorable'),Patch(facecolor=HARSH_COLOR,label='Harsh')]
    ax2.legend(handles=leg2, loc='lower right', fontsize=26, framealpha=0.9, edgecolor='#CCC')

    # (c) Domain Category
    panel_label(ax3, '(c)')
    all_cats = set()
    for m in MODELS: all_cats.update(STATS[m]['decomposition']['by_category'].keys())
    valid = [c for c in all_cats if sum(1 for m in MODELS if c in STATS[m]['decomposition']['by_category'])>=3]
    means = {}
    for cat in valid:
        rates_c = [STATS[m]['decomposition']['by_category'][cat]['fiai_rate'] for m in MODELS if cat in STATS[m]['decomposition']['by_category']]
        means[cat] = np.mean(rates_c)
    sorted_cats = sorted(valid, key=lambda c: means.get(c,0), reverse=True)
    yc = np.arange(len(sorted_cats)); n_m2 = len(MODEL_ORDER); bh = 0.55/n_m2
    for j, m in enumerate(MODEL_ORDER):
        r_vals, y_vals = [], []
        for yi, cat in enumerate(sorted_cats):
            if cat in STATS[m]['decomposition']['by_category']:
                r_vals.append(STATS[m]['decomposition']['by_category'][cat]['fiai_rate']*100)
                y_vals.append(yc[yi]+(j-n_m2/2+0.5)*bh)
        if r_vals:
            ax3.barh(y_vals, r_vals, bh, color=MODEL_COLORS[m], edgecolor='white', linewidth=0.4, label=MODEL_DISPLAY[m])
            for yi, r in zip(y_vals, r_vals):
                ax3.text(r+0.5, yi, f'{r:.0f}', va='center', fontsize=18, color='#666', fontweight='bold')
    ax3.set_yticks(yc); ax3.set_yticklabels(sorted_cats, fontsize=28, fontweight='bold')
    ax3.set_xlabel('FIAI Rate (%)'); ax3.axvline(x=5, color='#999', linestyle='--', linewidth=3, zorder=2)
    ax3.text(5.5, -0.5, 'Chance (5%)', fontsize=24, color='#777', fontweight='bold')
    ax3.invert_yaxis(); ax3.set_xlim(0,125); ax3.tick_params(labelsize=26, width=2)
    ax3.legend(loc='lower right', fontsize=24, framealpha=0.9, edgecolor='#CCC', ncol=2)

    plt.tight_layout(pad=2)
    save(fig, 'row2_distribution_domain.png'); plt.close(fig)
    print('Row 2: done.')

# ═══════════ ROW 3: 3 panels ═══════════
def make_row3():
    fig, axes = plt.subplots(1, 3, figsize=(44, 13))
    (ax1, ax2, ax3) = axes
    fig.subplots_adjust(wspace=0.28)

    # (a) Framing Ablation
    panel_label(ax1, '(a)')
    all_ft = set()
    for m in MODELS: all_ft.update(STATS[m]['decomposition']['by_framing_type'].keys())
    ft_means = {}
    for ft in all_ft:
        if ft != '?':
            rates_ft = [STATS[m]['decomposition']['by_framing_type'][ft]['fiai_rate'] for m in MODELS if ft in STATS[m]['decomposition']['by_framing_type']]
            ft_means[ft] = np.mean(rates_ft) if rates_ft else 0
    sorted_ft = sorted([ft for ft in all_ft if ft!='?'], key=lambda x: ft_means.get(x,0), reverse=True)
    ft_labels = {
        'dispositional_vs_situational':'Disp.\nvs. Sit.','aggregate_vs_instance':'Agg.\nvs. Inst.',
        'label_vs_description':'Label vs.\nDescr.','certainty_framing':'Certainty\nFraming',
        'presupposition_trigger':'Presup.\nTrigger','active_vs_passive':'Active vs.\nPassive',
        'temporal_framing':'Temporal\nFraming',
    }
    xf = np.arange(len(sorted_ft)); n_m3 = len(MODEL_ORDER); bw = 0.6/n_m3
    for j, m in enumerate(MODEL_ORDER):
        rf = [STATS[m]['decomposition']['by_framing_type'].get(ft,{}).get('fiai_rate',0)*100 for ft in sorted_ft]
        off = (j-n_m3/2+0.5)*bw
        ax1.bar(xf+off, rf, bw, color=MODEL_COLORS[m], edgecolor='white', linewidth=0.4, label=MODEL_DISPLAY[m])
    ax1.axhline(y=5, color='#999', linestyle='--', linewidth=3, zorder=2)
    ax1.text(len(sorted_ft)-0.5, 6.5, 'Chance (5%)', fontsize=24, color='#777', ha='right', fontweight='bold')
    ax1.set_xticks(xf)
    ax1.set_xticklabels([ft_labels.get(ft,ft) for ft in sorted_ft], fontsize=18, fontweight='bold', rotation=0)
    ax1.set_ylabel('FIAI Rate (%)'); ax1.set_ylim(0,115); ax1.tick_params(labelsize=26, width=2)
    ax1.legend(loc='upper right', fontsize=22, framealpha=0.9, edgecolor='#CCC', ncol=2)

    # (b) Capability
    panel_label(ax2, '(b)')
    cap_map = {'qwen3-5-flash':1,'deepseek-v4-flash':2,'kimi-k2-6':3,'glm-5-1':4,'deepseek-v4-pro':5,'qwen3-6-35b-a3b':6}
    mcap = [m for m in MODEL_ORDER if m in cap_map]
    cv = [cap_map[m] for m in mcap]
    rv = [STATS[m]['fiai']['rate']*100 for m in mcap]
    dv = [abs(STATS[m]['statistical_tests']['cohens_d']['value']) for m in mcap]
    a2r = ax2
    for i, m in enumerate(mcap):
        a2r.scatter(cv[i], rv[i], s=400, c=MODEL_COLORS[m], edgecolors='white', linewidth=3, zorder=4)
        a2r.annotate(MODEL_DISPLAY[m], (cv[i], rv[i]), textcoords='offset points', xytext=(0,22), fontsize=22, ha='center', color='#444', fontweight='bold')
    z = np.polyfit(cv, rv, 1); xs = np.linspace(min(cv)-0.3, max(cv)+0.3, 100)
    a2r.plot(xs, np.poly1d(z)(xs), '--', color='#7EB8DC', linewidth=4, alpha=0.8)
    a2r.set_xlabel('Model Scale (Flash -> 35B)')
    a2r.set_ylabel('FIAI Rate (%)', color='#4A90C4', fontweight='bold')
    a2r.set_xlim(min(cv)-0.5, max(cv)+0.5); a2r.tick_params(labelsize=28, width=2)
    a2d = a2r.twinx()
    for i, m in enumerate(mcap):
        a2d.scatter(cv[i], dv[i], s=280, c=MODEL_COLORS[m], edgecolors='#AAA', linewidth=2, zorder=3, alpha=0.5, marker='s')
    z2 = np.polyfit(cv, dv, 1)
    a2d.plot(xs, np.poly1d(z2)(xs), ':', color=HARSH_COLOR, linewidth=4, alpha=0.8)
    a2d.set_ylabel("|Cohen's d|", color=HARSH_COLOR, fontweight='bold'); a2d.tick_params(labelsize=28, width=2)
    ll = [Line2D([0],[0],marker='o',color='w',markerfacecolor='#4A90C4',markersize=18,label='FIAI Rate'),
          Line2D([0],[0],marker='s',color='w',markerfacecolor=HARSH_COLOR,markersize=18,label="|Cohen's d|")]
    a2r.legend(handles=ll, loc='lower right', fontsize=26, framealpha=0.9, edgecolor='#CCC')

    # (c) Flip Magnitude
    panel_label(ax3, '(c)')
    mc = ['#D8D8D8','#B0C8E8','#80A8D8','#5888C8','#3870B8']
    ml = ['No Change','1 Point','2 Points','3 Points','4 Points']
    xm = np.arange(len(MODEL_ORDER)); bwm = 0.55/5
    for j, m in enumerate(MODEL_ORDER):
        fmd = STATS[m]['flip_magnitude_distribution']
        mks = sorted(fmd.keys())
        for ki, k in enumerate(mks):
            if ki < len(mc):
                off = (ki-2)*bwm
                ax3.bar(xm[j]+off, fmd[k]['pct']*100, bwm, color=mc[ki], edgecolor='white', linewidth=0.4)
    ax3.set_xticks(xm)
    ax3.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER], rotation=25, ha='right', fontsize=22, fontweight='bold')
    ax3.set_ylabel('% of Pairs'); ax3.tick_params(labelsize=26, width=2)
    mlg = [Patch(facecolor=mc[i],label=ml[i]) for i in range(5)]
    ax3.legend(handles=mlg, loc='upper right', fontsize=24, framealpha=0.9, edgecolor='#CCC')

    plt.tight_layout(pad=2)
    save(fig, 'row3_mechanism_capability.png'); plt.close(fig)
    print('Row 3: done.')

if __name__ == '__main__':
    print('FiaiBench v5 — 2x massive fonts')
    make_row1(); make_row2(); make_row3()
    print(f'Done: {OUT_DIR}/')
