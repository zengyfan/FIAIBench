#!/usr/bin/env python3
"""
FiaiBench Figure Generation v3 — 3 ultra-wide composite figures
- 3 row positions (figures), each ~3.5:1 aspect ratio
- Multiple panels per row, all in one line
- Large text throughout
- No embedded "Figure X:" titles (LaTeX caption handles it)
- Light color scheme, publication quality
"""

import json, numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Style: extra large text ──────────────────────────────────
plt.rcParams.update({
    'font.family': 'Arial',
    'font.size': 18,
    'axes.titlesize': 20,
    'axes.labelsize': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 15,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ── Light Color Palette ────────────────────────────────────────
MODEL_COLORS = {
    'MODEL_A':   '#A8D4F0',
    'MODEL_B':     '#6EB8E0',
    'MODEL_F':       '#A8DFC0',
    'MODEL_G':     '#6EC89C',
    'MODEL_H':             '#F0B8B0',
    'MODEL_I':           '#C8B8E4',
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
    'Employment': '#F0E0B8', 'Legal': '#E8C0C0', 'Education': '#B8D4E4',
    'Healthcare': '#B8E0CC', 'Finance': '#D0C0E8', 'Social Services': '#E0D8C0',
}

SCORE_COLORS_5 = ['#E89890', '#F0C0B8', '#E8E8E8', '#B0DCC0', '#88CCA0']

# ── Paths ──────────────────────────────────────────────────────
BASE = Path('D:/AutoResearchPipeline/autoresearch_output')
DATA_PATH = BASE / 'phase4_raw_results' / 'complete_stats.json'
OUT_DIR = BASE / 'phase5_figures_v3'
OUT_DIR.mkdir(exist_ok=True)

with open(DATA_PATH, encoding='utf-8') as f:
    STATS = json.load(f)

MODELS = list(STATS.keys())
# Consistent model ordering: by FIAI rate descending
MODEL_ORDER = sorted(MODELS, key=lambda m: STATS[m]['fiai']['rate'], reverse=True)

def save(fig, name):
    path = OUT_DIR / name
    fig.savefig(path, dpi=200, bbox_inches='tight', pad_inches=0.15,
                facecolor='white', edgecolor='none')
    print(f'  Saved: {name}')

def panel_label(ax, text):
    """Bold panel label (a), (b), etc. at top-left."""
    ax.text(0.005, 1.02, text, transform=ax.transAxes, fontsize=24, fontweight='bold',
            va='bottom', ha='left', color='#333333')

# ══════════════════════════════════════════════════════════════════
# ROW 1: FIAI Overview — 4 panels in one row (rate, cohen-d, harsh-bias, scatter)
# ══════════════════════════════════════════════════════════════════
def make_row1_overview():
    fig, axes = plt.subplots(1, 4, figsize=(36, 10))
    (ax1, ax2, ax3, ax4) = axes
    fig.subplots_adjust(wspace=0.32)

    # ── (a) FIAI Rate ──
    panel_label(ax1, '(a)')
    rates_all = [STATS[m]['fiai']['rate'] * 100 for m in MODEL_ORDER]
    ci_low = [STATS[m]['fiai']['rate_ci95_lower'] * 100 for m in MODEL_ORDER]
    ci_high = [STATS[m]['fiai']['rate_ci95_upper'] * 100 for m in MODEL_ORDER]
    colors = [MODEL_COLORS[m] for m in MODEL_ORDER]
    names = [MODEL_DISPLAY[m] for m in MODEL_ORDER]

    y = np.arange(len(MODEL_ORDER))
    bars = ax1.barh(y, rates_all, 0.6, color=colors, edgecolor='white', linewidth=1, zorder=3)
    err_low = [r - l for r, l in zip(rates_all, ci_low)]
    err_high = [u - r for r, u in zip(rates_all, ci_high)]
    ax1.errorbar(rates_all, y, xerr=[err_low, err_high], fmt='none',
                 ecolor='#777777', capsize=4, capthick=1, elinewidth=1, zorder=4)
    for i, r in enumerate(rates_all):
        ax1.text(r + 1.2, i, f'{r:.1f}%', va='center', fontsize=16, fontweight='bold', color='#333333')
    ax1.set_yticks(y)
    ax1.set_yticklabels(names, fontsize=18)
    ax1.set_xlim(0, max(rates_all) + 9)
    ax1.axvline(x=5, color='#CCCCCC', linestyle='--', linewidth=1.5, zorder=2)
    ax1.text(5.5, -0.7, 'Chance (5%)', fontsize=14, color='#AAAAAA')
    ax1.set_xlabel('FIAI Rate (%)')
    ax1.invert_yaxis()

    # ── (b) Cohen's d ──
    panel_label(ax2, '(b)')
    d_vals = [STATS[m]['statistical_tests']['cohens_d']['value'] for m in MODEL_ORDER]
    # Background bands
    bands = [(0, 0.2, 'Negligible'), (0.2, 0.5, 'Small'), (0.5, 0.8, 'Medium'), (0.8, 2.5, 'Large')]
    for bmin, bmax, label in bands:
        ax2.axvspan(-bmax, -bmin, alpha=0.4, color='#F5F5F5', zorder=1)
        ax2.axvspan(bmin, bmax, alpha=0.4, color='#F5F5F5', zorder=1)
    ax2.axvline(x=0, color='#CCCCCC', linewidth=1.5, zorder=2)
    for i, (d, m) in enumerate(zip(d_vals, MODEL_ORDER)):
        color = '#E89890' if d < 0 else '#90B8D8'
        ax2.scatter(d, i, s=200, c=color, edgecolors='white', linewidth=1.5, zorder=4)
        offset = 0.06 if d >= 0 else -0.06
        ha = 'left' if d >= 0 else 'right'
        ax2.text(d + offset, i, f'{d:+.3f}', va='center', ha=ha, fontsize=16, fontweight='bold', color='#444444')
    ax2.set_yticks(y)
    ax2.set_yticklabels(names, fontsize=18)
    ax2.set_xlabel("Cohen's d")
    ax2.set_xlim(-2.5, 2.5)
    ax2.invert_yaxis()
    # Band labels
    for bmin, bmax, label in bands:
        mid = (bmin + bmax) / 2
        if bmax <= 0.8:
            ax2.text(mid, len(MODEL_ORDER) - 0.5, label, fontsize=13,
                     color='#BBBBBB', ha='center', va='bottom')

    # ── (c) Harsh Bias Ratio ──
    panel_label(ax3, '(c)')
    hr_vals = [STATS[m]['fiai']['direction']['harsh_ratio'] for m in MODEL_ORDER]
    diverging = [h - 0.5 for h in hr_vals]
    bar_colors = ['#E8A0A0' if dv > 0 else '#A0C0E0' for dv in diverging]
    ax3.barh(y, diverging, 0.7, color=bar_colors, edgecolor='white', linewidth=1, zorder=3)
    ax3.axvline(x=0, color='#999999', linewidth=1.5, zorder=2)
    for i, (hr, dv) in enumerate(zip(hr_vals, diverging)):
        xp = dv + (0.03 if dv >= 0 else -0.03)
        ha = 'left' if dv >= 0 else 'right'
        ax3.text(xp, i, f'{hr:.3f}', va='center', ha=ha, fontsize=16, fontweight='bold', color='#444444')
    ax3.set_yticks(y)
    ax3.set_yticklabels(names, fontsize=18)
    ax3.set_xlabel('Harsh Bias Ratio (deviation from 0.5)')
    max_dev = max(abs(min(diverging)), abs(max(diverging))) * 1.5
    ax3.set_xlim(-max_dev, max_dev)
    ax3.invert_yaxis()
    # Relabel x-ticks to show actual ratio
    xt = ax3.get_xticks()
    ax3.set_xticklabels([f'{t+0.5:.2f}' for t in xt], fontsize=17)

    # ── (d) Rate vs |d| ──
    panel_label(ax4, '(d)')
    for m in MODEL_ORDER:
        s = STATS[m]
        rate = s['fiai']['rate'] * 100
        abs_d = abs(s['statistical_tests']['cohens_d']['value'])
        n = s['sample']['n_pairs']
        ax4.scatter(rate, abs_d, s=n/6, c=MODEL_COLORS[m], edgecolors='white',
                    linewidth=1.2, alpha=0.9, zorder=4)
        ax4.annotate(MODEL_DISPLAY[m], (rate, abs_d),
                     textcoords='offset points', xytext=(0, 12),
                     fontsize=15, ha='center', color='#555555')
    ax4.set_xlabel('FIAI Rate (%)')
    ax4.set_ylabel("|Cohen's d|", )
    ax4.set_xlim(min(rates_all) - 4, max(rates_all) + 4)
    ax4.set_ylim(0, max([abs(STATS[m]['statistical_tests']['cohens_d']['value']) for m in MODELS]) * 1.3)

    plt.tight_layout()
    save(fig, 'row1_fiai_overview.png')
    plt.close(fig)
    print('Row 1: FIAI Overview (4 panels) done.')


# ══════════════════════════════════════════════════════════════════
# ROW 2: Distribution + Domain — 3 panels (stacked bars, waterfall, domain bars)
# ══════════════════════════════════════════════════════════════════
def make_row2_distribution_domain():
    fig, axes = plt.subplots(1, 3, figsize=(30, 9))
    (ax1, ax2, ax3) = axes
    fig.subplots_adjust(wspace=0.32)

    # ── (a) Stacked Proportion Bars — All Models ──
    panel_label(ax1, '(a)')
    n_models = len(MODEL_ORDER)
    bar_w = 0.38
    x_pos = np.arange(n_models) * 2.2

    for i, m in enumerate(MODEL_ORDER):
        ds = STATS[m]['distribution_shift']
        score_keys = ['-2', '-1', '0', '1', '2']
        imp_vals = [ds.get(sk, {}).get('implicit_pct', 0) * 100 for sk in score_keys]
        exp_vals = [ds.get(sk, {}).get('explicit_pct', 0) * 100 for sk in score_keys]

        bottom_i, bottom_e = 0, 0
        for j in range(5):
            ax1.bar(x_pos[i] - bar_w/2, imp_vals[j], bar_w, bottom=bottom_i,
                    color=SCORE_COLORS_5[j], edgecolor='white', linewidth=0.4)
            ax1.bar(x_pos[i] + bar_w/2, exp_vals[j], bar_w, bottom=bottom_e,
                    color=SCORE_COLORS_5[j], edgecolor='white', linewidth=0.4)
            bottom_i += imp_vals[j]
            bottom_e += exp_vals[j]

        # I/E labels
        ax1.text(x_pos[i] - bar_w/2, 103, 'I', ha='center', fontsize=13, color='#999999', fontweight='bold')
        ax1.text(x_pos[i] + bar_w/2, 103, 'E', ha='center', fontsize=13, color='#999999', fontweight='bold')

    ax1.set_xticks(x_pos)
    ax1.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER], rotation=25, ha='right', fontsize=16)
    ax1.set_ylabel('Proportion (%)', )
    ax1.set_ylim(0, 110)

    legend_elements = [
        Patch(facecolor='#E89890', label='Harsh (-2)'),
        Patch(facecolor='#F0C0B8', label='Lean Harsh (-1)'),
        Patch(facecolor='#E8E8E8', label='Neutral (0)'),
        Patch(facecolor='#B0DCC0', label='Lean Favor. (+1)'),
        Patch(facecolor='#88CCA0', label='Favorable (+2)'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=14, ncol=1,
               framealpha=0.85, edgecolor='#DDDDDD')

    # ── (b) Waterfall Δ Proportions ──
    panel_label(ax2, '(b)')
    y_pos = np.arange(n_models)
    for i, m in enumerate(MODEL_ORDER):
        ut = STATS[m]['uncertainty_hypothesis_test']
        nd = ut['neutral_delta'] * 100
        fd = ut['favorable_delta'] * 100
        hd = ut['harsh_delta'] * 100

        ax2.annotate('', xy=(nd, i), xytext=(0, i),
                     arrowprops=dict(arrowstyle='->', color='#BBBBBB', lw=6, alpha=0.7))
        ax2.annotate('', xy=(0, i+0.18), xytext=(fd, i+0.18),
                     arrowprops=dict(arrowstyle='->', color='#88CCA0', lw=6, alpha=0.7))
        ax2.annotate('', xy=(0, i-0.18), xytext=(hd, i-0.18),
                     arrowprops=dict(arrowstyle='->', color='#E89890', lw=6, alpha=0.7))
        ax2.scatter([nd], [i], s=80, c='#999999', zorder=5)
        ax2.scatter([fd], [i+0.18], s=80, c='#70B880', zorder=5)
        ax2.scatter([hd], [i-0.18], s=80, c='#E87870', zorder=5)

        # Verdict label
        verdict = ut.get('verdict', '')
        label = 'INJ' if 'Injustice' in verdict else 'MIX'
        col = '#E89890' if 'Injustice' in verdict else '#BBBBBB'
        ax2.text(-31, i, label, fontsize=14, fontweight='bold', color=col, va='center')

    ax2.axvline(x=0, color='#AAAAAA', linewidth=1.2)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER], fontsize=18)
    ax2.set_xlabel('% Point Change (pp)', )
    ax2.invert_yaxis()
    leg2 = [Patch(facecolor='#BBBBBB', label='Neutral'), Patch(facecolor='#88CCA0', label='Favorable'),
            Patch(facecolor='#E89890', label='Harsh')]
    ax2.legend(handles=leg2, loc='lower right', fontsize=15, framealpha=0.85, edgecolor='#DDDDDD')

    # ── (c) Domain Category Grouped Bars ──
    panel_label(ax3, '(c)')
    # Collect category data
    all_cats = set()
    for m in MODELS:
        all_cats.update(STATS[m]['decomposition']['by_category'].keys())
    valid_cats = [c for c in all_cats if sum(1 for m in MODELS if c in STATS[m]['decomposition']['by_category']) >= 3]
    cat_means = {}
    for cat in valid_cats:
        rates = [STATS[m]['decomposition']['by_category'][cat]['fiai_rate']
                 for m in MODELS if cat in STATS[m]['decomposition']['by_category']]
        cat_means[cat] = np.mean(rates)
    sorted_cats = sorted(valid_cats, key=lambda c: cat_means.get(c, 0), reverse=True)

    y_cat = np.arange(len(sorted_cats))
    n_m = len(MODEL_ORDER)
    bh = 0.65 / n_m

    for j, m in enumerate(MODEL_ORDER):
        rates_cat = []
        y_vals_cat = []
        for yi, cat in enumerate(sorted_cats):
            if cat in STATS[m]['decomposition']['by_category']:
                rates_cat.append(STATS[m]['decomposition']['by_category'][cat]['fiai_rate'] * 100)
                y_vals_cat.append(y_cat[yi] + (j - n_m/2 + 0.5) * bh)
        if rates_cat:
            ax3.barh(y_vals_cat, rates_cat, bh, color=MODEL_COLORS[m],
                     edgecolor='white', linewidth=0.4, label=MODEL_DISPLAY[m])
            for yi, r in zip(y_vals_cat, rates_cat):
                ax3.text(r + 0.5, yi, f'{r:.0f}', va='center', fontsize=13, color='#999999', fontweight='bold')

    ax3.set_yticks(y_cat)
    ax3.set_yticklabels(sorted_cats, fontsize=18)
    ax3.set_xlabel('FIAI Rate (%)')
    ax3.axvline(x=5, color='#CCCCCC', linestyle='--', linewidth=1.5, zorder=2)
    ax3.text(5.5, -0.5, 'Chance (5%)', fontsize=14, color='#AAAAAA')
    ax3.invert_yaxis()
    ax3.set_xlim(0, 115)
    ax3.legend(loc='lower right', fontsize=14, framealpha=0.85, edgecolor='#DDDDDD', ncol=2)

    plt.tight_layout()
    save(fig, 'row2_distribution_domain.png')
    plt.close(fig)
    print('Row 2: Distribution + Domain (3 panels) done.')


# ══════════════════════════════════════════════════════════════════
# ROW 3: Mechanism + Capability — 3 panels (framing ablation, capability, flip magnitude)
# ══════════════════════════════════════════════════════════════════
def make_row3_mechanism_capability():
    fig, axes = plt.subplots(1, 3, figsize=(30, 9))
    (ax1, ax2, ax3) = axes
    fig.subplots_adjust(wspace=0.32)

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
    bw = 0.7 / n_m2

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

    ax1.axhline(y=5, color='#CCCCCC', linestyle='--', linewidth=1.5, zorder=2)
    ax1.text(len(sorted_ft) - 0.4, 6, 'Chance (5%)', fontsize=14, color='#AAAAAA', ha='right')
    ax1.set_xticks(x_ft)
    ax1.set_xticklabels([ft_labels_map.get(ft, ft) for ft in sorted_ft], fontsize=13, rotation=0)
    ax1.set_ylabel('FIAI Rate (%)', )
    ax1.set_ylim(0, 108)
    ax1.legend(loc='upper right', fontsize=14, framealpha=0.85, edgecolor='#DDDDDD', ncol=2)

    # ── (b) Capability vs FIAI ──
    panel_label(ax2, '(b)')
    capability_map = {
        'MODEL_F': 1, 'MODEL_A': 2, 'MODEL_I': 3,
        'MODEL_H': 4, 'MODEL_B': 5, 'MODEL_G': 6,
    }
    models_with_cap = [m for m in MODEL_ORDER if m in capability_map]
    cap_vals = [capability_map[m] for m in models_with_cap]
    rate_vals = [STATS[m]['fiai']['rate'] * 100 for m in models_with_cap]
    d_abs_vals = [abs(STATS[m]['statistical_tests']['cohens_d']['value']) for m in models_with_cap]

    # Rate axis (left)
    ax2_rate = ax2
    for i, m in enumerate(models_with_cap):
        ax2_rate.scatter(cap_vals[i], rate_vals[i], s=180, c=MODEL_COLORS[m],
                         edgecolors='white', linewidth=1.5, zorder=4)
        ax2_rate.annotate(MODEL_DISPLAY[m], (cap_vals[i], rate_vals[i]),
                          textcoords='offset points', xytext=(0, 14),
                          fontsize=14, ha='center', color='#555555')
    z = np.polyfit(cap_vals, rate_vals, 1)
    x_sm = np.linspace(min(cap_vals)-0.3, max(cap_vals)+0.3, 100)
    ax2_rate.plot(x_sm, np.poly1d(z)(x_sm), '--', color='#A8CFEA', linewidth=2, alpha=0.8)
    ax2_rate.set_xlabel('Model Scale (Flash → 35B)', )
    ax2_rate.set_ylabel('FIAI Rate (%)', color='#5B9ECF')
    ax2_rate.set_xlim(min(cap_vals)-0.5, max(cap_vals)+0.5)

    # |d| axis (right)
    ax2_d = ax2_rate.twinx()
    for i, m in enumerate(models_with_cap):
        ax2_d.scatter(cap_vals[i], d_abs_vals[i], s=140, c=MODEL_COLORS[m],
                      edgecolors='#CCCCCC', linewidth=1, zorder=3, alpha=0.6, marker='s')
    z2 = np.polyfit(cap_vals, d_abs_vals, 1)
    ax2_d.plot(x_sm, np.poly1d(z2)(x_sm), ':', color='#E89890', linewidth=2, alpha=0.8)
    ax2_d.set_ylabel("|Cohen's d|", color='#E89890')

    # Legend for axes
    from matplotlib.lines import Line2D
    leg_lines = [Line2D([0], [0], marker='o', color='w', markerfacecolor='#5B9ECF', markersize=10, label='FIAI Rate'),
                 Line2D([0], [0], marker='s', color='w', markerfacecolor='#E89890', markersize=10, label="|Cohen's d|")]
    ax2_rate.legend(handles=leg_lines, loc='lower right', fontsize=15, framealpha=0.85, edgecolor='#DDDDDD')

    # ── (c) Flip Magnitude Distribution ──
    panel_label(ax3, '(c)')
    # Aggregate across models
    mag_labels = ['No Change', '1 Point', '2 Points', '3 Points', '4 Points']
    mag_colors = ['#E8E8E8', '#C8D8F0', '#A0C0E8', '#78A8D8', '#5890C8']

    x_mag = np.arange(len(MODEL_ORDER))
    bw_mag = 0.7 / 5  # 5 magnitude levels

    # Find max magnitude level across all models
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
    ax3.set_xticklabels([MODEL_DISPLAY[m] for m in MODEL_ORDER], rotation=25, ha='right', fontsize=17)
    ax3.set_ylabel('% of Pairs', )

    mag_legend = [Patch(facecolor=mag_colors[i], label=mag_labels[i]) for i in range(5)]
    ax3.legend(handles=mag_legend, loc='upper right', fontsize=14, framealpha=0.85, edgecolor='#DDDDDD', ncol=1)

    plt.tight_layout()
    save(fig, 'row3_mechanism_capability.png')
    plt.close(fig)
    print('Row 3: Mechanism + Capability (3 panels) done.')


# ══════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print('Generating FiaiBench figures v3 (3 ultra-wide rows, large text)...')
    print(f'Output: {OUT_DIR}')
    print()
    make_row1_overview()
    make_row2_distribution_domain()
    make_row3_mechanism_capability()
    print(f'\nDone. 3 figures saved to {OUT_DIR}/')
