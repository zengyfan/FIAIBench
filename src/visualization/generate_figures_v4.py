#!/usr/bin/env python3
"""
FiaiBench Figure Generation v4 — MAXIMUM font size, bold, deep pastels
- Fonts: VERY large, bold
- Colors: deeper pastels with clear contrast against white
- 3 rows, each ~3.5:1 aspect ratio
"""

import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── MAXIMUM TEXT ───────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'Arial',
    'font.weight': 'bold',
    'font.size': 24,
    'axes.titlesize': 26,
    'axes.labelsize': 24,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 1.5,
})

# ── DEEPER Pastel Colors (visible against white) ───────────────
MODEL_COLORS = {
    'MODEL_A':   '#7EB8DC',  # deeper sky blue
    'MODEL_B':     '#4A90C4',  # medium blue
    'MODEL_F':       '#6EC49A',  # deeper mint
    'MODEL_G':     '#3D9E70',  # medium green
    'MODEL_H':             '#E09088',  # deeper coral
    'MODEL_I':           '#A890CC',  # deeper violet
}
MODEL_DISPLAY = {
    'MODEL_A':   'Model A',
    'MODEL_B':     'Model B',
    'MODEL_F':       'Model F',
    'MODEL_G':     'Model G',
    'MODEL_H':             'Model H',
    'MODEL_I':           'Model I',
}

CATEGORY_COLORS = {
    'Employment': '#E0CCA0', 'Legal': '#D8A8A8', 'Education': '#98B8D0',
    'Healthcare': '#90C8B0', 'Finance': '#B8A0D0', 'Social Services': '#C8C0A0',
}

# Decision score colors — deeper tones
SCORE_COLORS_5 = ['#E07070', '#F0A098', '#E0E0E0', '#88D0A8', '#58B878']
HARSH_COLOR = '#E07070'
FAVOR_COLOR = '#58B878'
NEUTRAL_COLOR = '#CCCCCC'

# ── Paths ──────────────────────────────────────────────────────
BASE = Path('D:/AutoResearchPipeline/autoresearch_output')
DATA_PATH = BASE / 'phase4_raw_results' / 'complete_stats.json'
OUT_DIR = BASE / 'phase5_figures_v4'
OUT_DIR.mkdir(exist_ok=True)

with open(DATA_PATH, encoding='utf-8') as f:
    STATS = json.load(f)
MODELS = list(STATS.keys())
MODEL_ORDER = sorted(MODELS, key=lambda m: STATS[m]['fiai']['rate'], reverse=True)

def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.3,
                facecolor='white', edgecolor='none')
    print(f'  Saved: {name}')

def panel_label(ax, text):
    ax.text(0.003, 1.03, text, transform=ax.transAxes, fontsize=32, fontweight='bold',
            va='bottom', ha='left', color='#222222')

ANNO_SIZE = 20  # bar annotation text
ANNO_COLOR = '#333333'

# ══════════════════════════════════════════════════════════════════
# ROW 1: FIAI Overview — 4 panels
# ══════════════════════════════════════════════════════════════════
def make_row1():
    fig, axes = plt.subplots(1, 4, figsize=(42, 12))
    (ax1, ax2, ax3, ax4) = axes
    fig.subplots_adjust(wspace=0.30)

    # ── (a) FIAI Rate ──
    panel_label(ax1, '(a)')
    rates_all = [STATS[m]['fiai']['rate'] * 100 for m in MODEL_ORDER]
    ci_low = [STATS[m]['fiai']['rate_ci95_lower'] * 100 for m in MODEL_ORDER]
    ci_high = [STATS[m]['fiai']['rate_ci95_upper'] * 100 for m in MODEL_ORDER]
    colors = [MODEL_COLORS[m] for m in MODEL_ORDER]
    names = [MODEL_DISPLAY[m] for m in MODEL_ORDER]

    y = np.arange(len(MODEL_ORDER))
    ax1.barh(y, rates_all, 0.6, color=colors, edgecolor='white', linewidth=1.5, zorder=3)
    err_low = [r - l for r, l in zip(rates_all, ci_low)]
    err_high = [u - r for r, u in zip(rates_all, ci_high)]
    ax1.errorbar(rates_all, y, xerr=[err_low, err_high], fmt='none',
                 ecolor='#555555', capsize=5, capthick=1.5, elinewidth=1.5, zorder=4)
    for i, r in enumerate(rates_all):
        ax1.text(r + 1.5, i, f'{r:.1f}%', va='center', fontsize=ANNO_SIZE,
                 fontweight='bold', color=ANNO_COLOR)
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=22, fontweight='bold')
    ax1.set_xlim(0, max(rates_all) + 12)
    ax1.axvline(x=5, color='#999999', linestyle='--', linewidth=2, zorder=2)
    ax1.text(5.5, -0.8, 'Chance (5%)', fontsize=18, color='#777777', fontweight='bold')
    ax1.set_xlabel('FIAI Rate (%)')
    ax1.invert_yaxis()
    ax1.tick_params(axis='both', labelsize=20, width=1.5)

    # ── (b) Cohen's d ──
    panel_label(ax2, '(b)')
    d_vals = [STATS[m]['statistical_tests']['cohens_d']['value'] for m in MODEL_ORDER]
    bands = [(0, 0.2, 'Negligible'), (0.2, 0.5, 'Small'), (0.5, 0.8, 'Medium'), (0.8, 2.5, 'Large')]
    for bmin, bmax, label in bands:
        ax2.axvspan(-bmax, -bmin, alpha=0.3, color='#E8E8E8', zorder=1)
        ax2.axvspan(bmin, bmax, alpha=0.3, color='#E8E8E8', zorder=1)
    ax2.axvline(x=0, color='#AAAAAA', linewidth=2, zorder=2)
    for i, (d, m) in enumerate(zip(d_vals, MODEL_ORDER)):
        color = HARSH_COLOR if d < 0 else '#70A0D0'
        ax2.scatter(d, i, s=300, c=color, edgecolors='white', linewidth=2, zorder=4)
        offset = 0.08 if d >= 0 else -0.08
        ha = 'left' if d >= 0 else 'right'
        ax2.text(d + offset, i, f'{d:+.3f}', va='center', ha=ha, fontsize=ANNO_SIZE,
                 fontweight='bold', color=ANNO_COLOR)
    ax2.set_yticks(y)
    ax2.set_yticklabels(names, fontsize=22, fontweight='bold')
    ax2.set_xlabel("Cohen's d")
    ax2.set_xlim(-2.5, 2.5)
    ax2.invert_yaxis()
    for bmin, bmax, label in bands:
        mid = (bmin + bmax) / 2
        if bmax <= 0.8:
            ax2.text(mid, len(MODEL_ORDER) - 0.3, label, fontsize=14,
                     color='#AAAAAA', ha='center', va='bottom', fontweight='bold')
    ax2.tick_params(axis='both', labelsize=20, width=1.5)

    # ── (c) Harsh Bias Ratio ──
    panel_label(ax3, '(c)')
    hr_vals = [STATS[m]['fiai']['direction']['harsh_ratio'] for m in MODEL_ORDER]
    diverging = [h - 0.5 for h in hr_vals]
    bar_colors = ['#E07070' if dv > 0 else '#70A0D0' for dv in diverging]
    ax3.barh(y, diverging, 0.7, color=bar_colors, edgecolor='white', linewidth=1.5, zorder=3)
    ax3.axvline(x=0, color='#888888', linewidth=2, zorder=2)
    for i, (hr, dv) in enumerate(zip(hr_vals, diverging)):
        xp = dv + (0.04 if dv >= 0 else -0.04)
        ha = 'left' if dv >= 0 else 'right'
        ax3.text(xp, i, f'{hr:.3f}', va='center', ha=ha, fontsize=ANNO_SIZE,
                 fontweight='bold', color=ANNO_COLOR)
    ax3.set_yticks(y)
    ax3.set_yticklabels(names, fontsize=22, fontweight='bold')
    ax3.set_xlabel('Harsh Bias Ratio (deviation from 0.5)')
    max_dev = max(abs(min(diverging)), abs(max(diverging))) * 1.6
    ax3.set_xlim(-max_dev, max_dev)
    ax3.invert_yaxis()
    xt = ax3.get_xticks()
    ax3.set_xticklabels([f'{t+0.5:.2f}' for t in xt], fontsize=20, fontweight='bold')
    ax3.tick_params(axis='both', labelsize=20, width=1.5)

    # ── (d) Rate vs |d| ──
    panel_label(ax4, '(d)')
    for m in MODEL_ORDER:
        s = STATS[m]
        rate = s['fiai']['rate'] * 100
        abs_d = abs(s['statistical_tests']['cohens_d']['value'])
        ax4.scatter(rate, abs_d, s=350, c=MODEL_COLORS[m], edgecolors='white',
                    linewidth=2, alpha=0.95, zorder=4)
        ax4.annotate(MODEL_DISPLAY[m], (rate, abs_d),
                     textcoords='offset points', xytext=(0, 16),
                     fontsize=17, ha='center', color='#444444', fontweight='bold')
    ax4.set_xlabel('FIAI Rate (%)')
    ax4.set_ylabel("|Cohen's d|")
    ax4.set_xlim(min(rates_all) - 5, max(rates_all) + 5)
    ax4.set_ylim(0, max([abs(STATS[m]['statistical_tests']['cohens_d']['value']) for m in MODELS]) * 1.3)
    ax4.tick_params(axis='both', labelsize=20, width=1.5)

    plt.tight_layout(pad=1.5)
    save(fig, 'row1_fiai_overview.png')
    plt.close(fig)
    print('Row 1: done.')

# ══════════════════════════════════════════════════════════════════
# ROW 2: Distribution + Domain — 3 panels
# ══════════════════════════════════════════════════════════════════
def make_row2():
    fig, axes = plt.subplots(1, 3, figsize=(36, 10.5))
    (ax1, ax2, ax3) = axes
    fig.subplots_adjust(wspace=0.30)

    # ── (a) Stacked Proportion Bars ──
    panel_label(ax1, '(a)')
    n_models = len(MODEL_ORDER)
    bar_w = 0.40
    x_pos = np.arange(n_models) * 2.5

    for i, m in enumerate(MODEL_ORDER):
        ds = STATS[m]['distribution_shift']
        score_keys = ['-2', '-1', '0', '1', '2']
        imp_vals = [ds.get(sk, {}).get('implicit_pct', 0) * 100 for sk in score_keys]
        exp_vals = [ds.get(sk, {}).get('explicit_pct', 0) * 100 for sk in score_keys]

        bottom_i, bottom_e = 0, 0
        for j in range(5):
            ax1.bar(x_pos[i] - bar_w/2, imp_vals[j], bar_w, bottom=bottom_i,
                    color=SCORE_COLORS_5[j], edgecolor='white', linewidth=0.6)
            ax1.bar(x_pos[i] + bar_w/2, exp_vals[j], bar_w, bottom=bottom_e,
                    color=SCORE_COLORS_5[j], edgecolor='white', linewidth=0.6)
            bottom_i += imp_vals[j]
            bottom_e += exp_vals[j]

        ax1.text(x_pos[i] - bar_w/2, 104, 'I', ha='center', fontsize=16,
                 color='#777777', fontweight='bold')
        ax1.text(x_pos[i] + bar_w/2, 104, 'E', ha='center', fontsize=16,
                 color='#777777', fontweight='bold')

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER], rotation=25,
                        ha='right', fontsize=18, fontweight='bold')
    ax1.set_ylabel('Proportion (%)')
    ax1.set_ylim(0, 112)
    ax1.tick_params(axis='both', labelsize=18, width=1.5)

    legend_elements = [
        Patch(facecolor='#E07070', label='Harsh (-2)'),
        Patch(facecolor='#F0A098', label='Lean Harsh (-1)'),
        Patch(facecolor='#E0E0E0', label='Neutral (0)'),
        Patch(facecolor='#88D0A8', label='Lean Favor. (+1)'),
        Patch(facecolor='#58B878', label='Favorable (+2)'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=17, ncol=1,
               framealpha=0.9, edgecolor='#CCCCCC')

    # ── (b) Waterfall ──
    panel_label(ax2, '(b)')
    y_pos = np.arange(n_models)
    for i, m in enumerate(MODEL_ORDER):
        ut = STATS[m]['uncertainty_hypothesis_test']
        nd = ut['neutral_delta'] * 100
        fd = ut['favorable_delta'] * 100
        hd = ut['harsh_delta'] * 100

        ax2.annotate('', xy=(nd, i), xytext=(0, i),
                     arrowprops=dict(arrowstyle='->', color='#AAAAAA', lw=7, alpha=0.7))
        ax2.annotate('', xy=(0, i+0.2), xytext=(fd, i+0.2),
                     arrowprops=dict(arrowstyle='->', color=FAVOR_COLOR, lw=7, alpha=0.7))
        ax2.annotate('', xy=(0, i-0.2), xytext=(hd, i-0.2),
                     arrowprops=dict(arrowstyle='->', color=HARSH_COLOR, lw=7, alpha=0.7))
        ax2.scatter([nd], [i], s=120, c='#777777', zorder=5)
        ax2.scatter([fd], [i+0.2], s=120, c='#48A868', zorder=5)
        ax2.scatter([hd], [i-0.2], s=120, c='#D06050', zorder=5)

        verdict = ut.get('verdict', '')
        label = 'INJ' if 'Injustice' in verdict else 'MIX'
        col = '#D06050' if 'Injustice' in verdict else '#AAAAAA'
        ax2.text(-35, i, label, fontsize=18, fontweight='bold', color=col, va='center')

    ax2.axvline(x=0, color='#999999', linewidth=1.5)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER], fontsize=22, fontweight='bold')
    ax2.set_xlabel('% Point Change (pp)')
    ax2.invert_yaxis()
    ax2.tick_params(axis='both', labelsize=20, width=1.5)
    leg2 = [Patch(facecolor='#AAAAAA', label='Neutral'),
            Patch(facecolor=FAVOR_COLOR, label='Favorable'),
            Patch(facecolor=HARSH_COLOR, label='Harsh')]
    ax2.legend(handles=leg2, loc='lower right', fontsize=18, framealpha=0.9, edgecolor='#CCCCCC')

    # ── (c) Domain Category ──
    panel_label(ax3, '(c)')
    all_cats = set()
    for m in MODELS:
        all_cats.update(STATS[m]['decomposition']['by_category'].keys())
    valid_cats = [c for c in all_cats
                  if sum(1 for m in MODELS if c in STATS[m]['decomposition']['by_category']) >= 3]
    cat_means = {}
    for cat in valid_cats:
        rates = [STATS[m]['decomposition']['by_category'][cat]['fiai_rate']
                 for m in MODELS if cat in STATS[m]['decomposition']['by_category']]
        cat_means[cat] = np.mean(rates)
    sorted_cats = sorted(valid_cats, key=lambda c: cat_means.get(c, 0), reverse=True)

    y_cat = np.arange(len(sorted_cats))
    n_m = len(MODEL_ORDER)
    bh = 0.6 / n_m

    for j, m in enumerate(MODEL_ORDER):
        rates_cat, y_vals_cat = [], []
        for yi, cat in enumerate(sorted_cats):
            if cat in STATS[m]['decomposition']['by_category']:
                rates_cat.append(STATS[m]['decomposition']['by_category'][cat]['fiai_rate'] * 100)
                y_vals_cat.append(y_cat[yi] + (j - n_m/2 + 0.5) * bh)
        if rates_cat:
            ax3.barh(y_vals_cat, rates_cat, bh, color=MODEL_COLORS[m],
                     edgecolor='white', linewidth=0.4, label=MODEL_DISPLAY[m])
            for yi, r in zip(y_vals_cat, rates_cat):
                ax3.text(r + 0.5, yi, f'{r:.0f}', va='center', fontsize=13,
                         color='#666666', fontweight='bold')

    ax3.set_yticks(y_cat)
    ax3.set_yticklabels(sorted_cats, fontsize=20, fontweight='bold')
    ax3.set_xlabel('FIAI Rate (%)')
    ax3.axvline(x=5, color='#999999', linestyle='--', linewidth=2, zorder=2)
    ax3.text(5.5, -0.5, 'Chance (5%)', fontsize=18, color='#777777', fontweight='bold')
    ax3.invert_yaxis()
    ax3.set_xlim(0, 120)
    ax3.tick_params(axis='both', labelsize=18, width=1.5)
    ax3.legend(loc='lower right', fontsize=16, framealpha=0.9, edgecolor='#CCCCCC', ncol=2)

    plt.tight_layout(pad=1.5)
    save(fig, 'row2_distribution_domain.png')
    plt.close(fig)
    print('Row 2: done.')

# ══════════════════════════════════════════════════════════════════
# ROW 3: Mechanism + Capability — 3 panels
# ══════════════════════════════════════════════════════════════════
def make_row3():
    fig, axes = plt.subplots(1, 3, figsize=(36, 10.5))
    (ax1, ax2, ax3) = axes
    fig.subplots_adjust(wspace=0.30)

    # ── (a) Framing Type Ablation ──
    panel_label(ax1, '(a)')
    all_ft = set()
    for m in MODELS:
        all_ft.update(STATS[m]['decomposition']['by_framing_type'].keys())
    ft_means = {}
    for ft in all_ft:
        if ft != '?':
            rates = [STATS[m]['decomposition']['by_framing_type'][ft]['fiai_rate']
                     for m in MODELS if ft in STATS[m]['decomposition']['by_framing_type']]
            ft_means[ft] = np.mean(rates) if rates else 0
    sorted_ft = sorted([ft for ft in all_ft if ft != '?'], key=lambda x: ft_means.get(x, 0), reverse=True)

    ft_labels_map = {
        'dispositional_vs_situational': 'Dispositional\nvs. Situational',
        'aggregate_vs_instance': 'Aggregate\nvs. Instance',
        'label_vs_description': 'Label vs.\nDescription',
        'certainty_framing': 'Certainty\nFraming',
        'presupposition_trigger': 'Presupposition\nTrigger',
        'active_vs_passive': 'Active vs.\nPassive Voice',
        'temporal_framing': 'Temporal\nFraming',
    }

    x_ft = np.arange(len(sorted_ft))
    n_m2 = len(MODEL_ORDER)
    bw = 0.65 / n_m2

    for j, m in enumerate(MODEL_ORDER):
        rates_ft = []
        for ft in sorted_ft:
            if ft in STATS[m]['decomposition']['by_framing_type']:
                rates_ft.append(STATS[m]['decomposition']['by_framing_type'][ft]['fiai_rate'] * 100)
            else:
                rates_ft.append(0)
        offset = (j - n_m2/2 + 0.5) * bw
        ax1.bar(x_ft + offset, rates_ft, bw, color=MODEL_COLORS[m],
                edgecolor='white', linewidth=0.4, label=MODEL_DISPLAY[m])

    ax1.axhline(y=5, color='#999999', linestyle='--', linewidth=2, zorder=2)
    ax1.text(len(sorted_ft) - 0.5, 6.5, 'Chance (5%)', fontsize=18,
             color='#777777', ha='right', fontweight='bold')
    ax1.set_xticks(x_ft)
    ax1.set_xticklabels([ft_labels_map.get(ft, ft) for ft in sorted_ft],
                        fontsize=14, fontweight='bold', rotation=0)
    ax1.set_ylabel('FIAI Rate (%)')
    ax1.set_ylim(0, 112)
    ax1.tick_params(axis='both', labelsize=18, width=1.5)
    ax1.legend(loc='upper right', fontsize=15, framealpha=0.9, edgecolor='#CCCCCC', ncol=2)

    # ── (b) Capability ──
    panel_label(ax2, '(b)')
    capability_map = {
        'MODEL_F': 1, 'MODEL_A': 2, 'MODEL_I': 3,
        'MODEL_H': 4, 'MODEL_B': 5, 'MODEL_G': 6,
    }
    models_with_cap = [m for m in MODEL_ORDER if m in capability_map]
    cap_vals = [capability_map[m] for m in models_with_cap]
    rate_vals = [STATS[m]['fiai']['rate'] * 100 for m in models_with_cap]
    d_abs_vals = [abs(STATS[m]['statistical_tests']['cohens_d']['value']) for m in models_with_cap]

    ax2_rate = ax2
    for i, m in enumerate(models_with_cap):
        ax2_rate.scatter(cap_vals[i], rate_vals[i], s=280, c=MODEL_COLORS[m],
                         edgecolors='white', linewidth=2, zorder=4)
        ax2_rate.annotate(MODEL_DISPLAY[m], (cap_vals[i], rate_vals[i]),
                          textcoords='offset points', xytext=(0, 18),
                          fontsize=16, ha='center', color='#444444', fontweight='bold')
    z = np.polyfit(cap_vals, rate_vals, 1)
    x_sm = np.linspace(min(cap_vals)-0.3, max(cap_vals)+0.3, 100)
    ax2_rate.plot(x_sm, np.poly1d(z)(x_sm), '--', color='#7EB8DC', linewidth=3, alpha=0.8)
    ax2_rate.set_xlabel('Model Scale (Flash -> 35B)')
    ax2_rate.set_ylabel('FIAI Rate (%)', color='#4A90C4', fontweight='bold')
    ax2_rate.set_xlim(min(cap_vals)-0.5, max(cap_vals)+0.5)
    ax2_rate.tick_params(axis='both', labelsize=20, width=1.5)

    ax2_d = ax2_rate.twinx()
    for i, m in enumerate(models_with_cap):
        ax2_d.scatter(cap_vals[i], d_abs_vals[i], s=200, c=MODEL_COLORS[m],
                      edgecolors='#AAAAAA', linewidth=1.5, zorder=3, alpha=0.5, marker='s')
    z2 = np.polyfit(cap_vals, d_abs_vals, 1)
    ax2_d.plot(x_sm, np.poly1d(z2)(x_sm), ':', color=HARSH_COLOR, linewidth=3, alpha=0.8)
    ax2_d.set_ylabel("|Cohen's d|", color=HARSH_COLOR, fontweight='bold')
    ax2_d.tick_params(axis='y', labelsize=20, width=1.5)

    leg_lines = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4A90C4', markersize=14, label='FIAI Rate'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=HARSH_COLOR, markersize=14, label="|Cohen's d|")]
    ax2_rate.legend(handles=leg_lines, loc='lower right', fontsize=18, framealpha=0.9, edgecolor='#CCCCCC')

    # ── (c) Flip Magnitude ──
    panel_label(ax3, '(c)')
    mag_colors = ['#D8D8D8', '#B0C8E8', '#80A8D8', '#5888C8', '#3870B8']
    mag_labels = ['No Change', '1 Point', '2 Points', '3 Points', '4 Points']

    x_mag = np.arange(len(MODEL_ORDER))
    bw_mag = 0.6 / 5

    for j, m in enumerate(MODEL_ORDER):
        fmd = STATS[m]['flip_magnitude_distribution']
        mag_keys_sorted = sorted(fmd.keys())
        for k_idx, k in enumerate(mag_keys_sorted):
            pct = fmd[k]['pct'] * 100
            if k_idx < len(mag_colors):
                offset = (k_idx - 2) * bw_mag
                ax3.bar(x_mag[j] + offset, pct, bw_mag, color=mag_colors[k_idx],
                        edgecolor='white', linewidth=0.4)

    ax3.set_xticks(x_mag)
    ax3.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER], rotation=25,
                        ha='right', fontsize=16, fontweight='bold')
    ax3.set_ylabel('% of Pairs')
    ax3.tick_params(axis='both', labelsize=18, width=1.5)

    mag_legend = [Patch(facecolor=mag_colors[i], label=mag_labels[i]) for i in range(5)]
    ax3.legend(handles=mag_legend, loc='upper right', fontsize=16, framealpha=0.9, edgecolor='#CCCCCC')

    plt.tight_layout(pad=1.5)
    save(fig, 'row3_mechanism_capability.png')
    plt.close(fig)
    print('Row 3: done.')

# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('FiaiBench Figures v4 — MAX fonts, deep pastels')
    print(f'Output: {OUT_DIR}\n')
    make_row1()
    make_row2()
    make_row3()
    print(f'\nDone. Saved to {OUT_DIR}/')
