import random
import math
from pathlib import Path
from itertools import chain

import matplotlib.pyplot as plt
import numpy as np

import accuracy

# Safe getters so the script runs even if some lists are not yet filled.
def _get(name):
    return getattr(accuracy, name, []) or []

REGIMES = [
    ("Degraded", _get("degraded_regime_BCT_real"), _get("degraded_regime_BCT_cal")),
    ("Good",     _get("good_regime_BCT_real"),     _get("good_regime_BCT_cal")),
]


def _stats(vals):
    if not vals:
        return dict(mean=float("nan"), n=0)
    return dict(mean=sum(vals) / len(vals), n=len(vals), std=np.std(vals))


def _median(vals):
    if not vals:
        return float("nan")
    s = sorted(vals)
    m = len(s) // 2
    return (s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m]))


def _jitter(base, count, width=0.12):
    return [base + random.uniform(-width, width) for _ in range(count)]


def plot(output="accuracy_plot.png"):
    random.seed(7)  # stable jitter between runs

    palette = dict(real="#3155A6", model="#D14A61")

    fig, ax = plt.subplots(figsize=(8.5, 5.6))
    ax.set_title("Burst Completion Time", fontsize=18, fontweight='bold', color="#222", loc='center')

    positions, data, colors = [], [], []
    jitter_width = 0.10

    for idx, (reg, real_vals, model_vals) in enumerate(REGIMES):
        base = idx * 1.6
        if real_vals:
            positions.append(base - 0.22); data.append(real_vals); colors.append(palette['real'])
        if model_vals:
            positions.append(base + 0.22); data.append(model_vals); colors.append(palette['model'])

    if data:
        bp = ax.boxplot(data, positions=positions, widths=0.34, patch_artist=True, showfliers=False)
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c); patch.set_alpha(0.18); patch.set_edgecolor('#888'); patch.set_linewidth(0.8)
        for median in bp['medians']:
            median.set_color('#333'); median.set_linewidth(1.4)
        for posi, vals, c in zip(positions, data, colors):
            ax.scatter(_jitter(posi, len(vals), width=jitter_width), vals,
                       s=65, alpha=0.82, color=c, edgecolor='white', linewidth=0.35, zorder=3)

    ax.set_xticks([idx * 1.6 for idx in range(len(REGIMES))])
    ax.set_xticklabels(["Degraded regime", "Good regime"], fontsize=15)
    ax.set_ylabel("BCT (ms)", fontsize=16)
    ax.set_yscale("log")
    ax.tick_params(axis='both', labelsize=14)
    ax.grid(True, which="both", axis="y", color="#e4e4e4", lw=0.9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    meas = ax.scatter([], [], s=90, color=palette['real'], edgecolor='white', linewidth=0.3, label='Measured BCT')
    model = ax.scatter([], [], s=90, color=palette['model'], edgecolor='white', linewidth=0.3, label='Modeled BCT')
    ax.legend(handles=[meas, model], frameon=False, fontsize=13, loc='upper right')

    fig.tight_layout()
    out_path = Path(output)
    fig.savefig(out_path, dpi=260)
    print(f"wrote {out_path.resolve()}")


def plot_variants(output="accuracy_variants.png"):
    """Additional views: hexbin, violin, QQ, residual plot."""
    random.seed(7)
    palette = dict(real="#3155A6", model="#D14A61", deg="#3155A6", good="#D14A61")

    # Build paired arrays
    pairs = []
    for reg, real_vals, model_vals in REGIMES:
        n = min(len(real_vals), len(model_vals))
        pairs.extend([(reg, real_vals[i], model_vals[i]) for i in range(n)])

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle("Burst Completion Time — Alternate Views", fontsize=16, fontweight='bold', color="#222")
    ax_hex, ax_violin, ax_qq, ax_resid = axes.ravel()

    # 1) Hexbin joint (log-log)
    if pairs:
        xs = np.array([p[1] for p in pairs]); ys = np.array([p[2] for p in pairs])
        hb = ax_hex.hexbin(xs, ys, gridsize=25, bins='log', cmap='Blues')
        lim = max(xs.max(), ys.max()) * 1.1
        ax_hex.plot([1e-6, lim], [1e-6, lim], color='#c44e52', lw=1.1, ls='--', label='y = x')
        ax_hex.set_xscale('log'); ax_hex.set_yscale('log')
        ax_hex.set_xlabel('Measured BCT (ms)', fontsize=13)
        ax_hex.set_ylabel('Modeled BCT (ms)', fontsize=13)
        ax_hex.tick_params(axis='both', labelsize=11)
        ax_hex.grid(True, which='both', color='#e6e6e6', lw=0.7)
        ax_hex.legend(frameon=False, fontsize=10, loc='lower right')
        cb = fig.colorbar(hb, ax=ax_hex, fraction=0.046, pad=0.04)
        cb.set_label('log10(count)', fontsize=11)
    else:
        ax_hex.text(0.5, 0.5, 'No paired data', ha='center', va='center')
        ax_hex.axis('off')

    # 2) Violin + median points by regime/source
    vio_data = []
    vio_pos = []
    vio_cols = []
    vio_labels = []
    for idx, (reg, real_vals, model_vals) in enumerate(REGIMES):
        base = idx * 1.5
        if real_vals:
            vio_data.append(real_vals); vio_pos.append(base - 0.25); vio_cols.append(palette['real']); vio_labels.append(f"{reg}\nmeasured")
        if model_vals:
            vio_data.append(model_vals); vio_pos.append(base + 0.25); vio_cols.append(palette['model']); vio_labels.append(f"{reg}\nmodeled")
    if vio_data:
        parts = ax_violin.violinplot(vio_data, positions=vio_pos, widths=0.4, showextrema=False)
        for b, c in zip(parts['bodies'], vio_cols):
            b.set_facecolor(c); b.set_alpha(0.25); b.set_edgecolor('#555'); b.set_linewidth(0.8)
        medians = [np.median(v) for v in vio_data]
        ax_violin.scatter(vio_pos, medians, color='#333', s=28, zorder=3)
        ax_violin.set_xticks([i * 1.5 for i in range(len(REGIMES))])
        ax_violin.set_xticklabels(["Degraded regime", "Good regime"], fontsize=12)
        ax_violin.set_ylabel('BCT (ms)', fontsize=13)
        ax_violin.set_yscale('log')
        ax_violin.tick_params(axis='both', labelsize=11)
        ax_violin.grid(True, which='both', axis='y', color='#e6e6e6', lw=0.7)
        ax_violin.spines['top'].set_visible(False); ax_violin.spines['right'].set_visible(False)
    else:
        ax_violin.axis('off')

    # 3) QQ plot (log domain)
    if pairs:
        xs_sorted = np.sort(xs)
        ys_sorted = np.sort(ys[:len(xs_sorted)])
        ax_qq.plot(xs_sorted, ys_sorted, marker='o', markersize=4, linestyle='', color='#3155A6', alpha=0.7)
        lim = max(xs_sorted.max(), ys_sorted.max()) * 1.1
        ax_qq.plot([1e-6, lim], [1e-6, lim], color='#c44e52', lw=1.0, ls='--')
        ax_qq.set_xscale('log'); ax_qq.set_yscale('log')
        ax_qq.set_xlabel('Measured quantiles (ms)', fontsize=13)
        ax_qq.set_ylabel('Modeled quantiles (ms)', fontsize=13)
        ax_qq.tick_params(axis='both', labelsize=11)
        ax_qq.grid(True, which='both', color='#e6e6e6', lw=0.7)
        ax_qq.spines['top'].set_visible(False); ax_qq.spines['right'].set_visible(False)
    else:
        ax_qq.axis('off')

    # 4) Residuals vs measured (ratio - 1)
    if pairs:
        ratio = ys / xs
        ax_resid.scatter(xs, ratio - 1.0, s=30, color='#556270', alpha=0.8, edgecolor='white', linewidth=0.3)
        ax_resid.axhline(0.0, color='#888', lw=1.0, ls='--')
        ax_resid.set_xscale('log')
        ax_resid.set_xlabel('Measured BCT (ms)', fontsize=13)
        ax_resid.set_ylabel('Relative error (modeled/measured - 1)', fontsize=13)
        ax_resid.tick_params(axis='both', labelsize=11)
        ax_resid.grid(True, which='both', color='#e6e6e6', lw=0.7)
        ax_resid.spines['top'].set_visible(False); ax_resid.spines['right'].set_visible(False)
    else:
        ax_resid.axis('off')

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out_path = Path(output)
    fig.savefig(out_path, dpi=260)
    print(f"wrote {out_path.resolve()}")


if __name__ == "__main__":
    plot()
