"""
DCTCP Regime Simulation
=======================
Interactive simulation of DCTCP queue/cwnd/alpha dynamics
with matplotlib sliders. Toggle bars for N, S (KB), cwnd_0.

Fixed parameters:
  C        = 25 Gbps = 3125 bytes/us = 2.0833 pkts/us
  RTT_prop = 100 us
  K        = 65 packets  (ECN threshold)
  g        = 1/16        (DCTCP EWMA gain)
  PKT      = 1500 bytes  (packet size)
  cwnd_min = 2 packets   (protocol floor)
  BDP      = C_pkts * RTT_prop = 208.33 pkts

Equations (per RTT k):
  RTT_k     = RTT_prop + Q_k / C_pkts
  Q_{k+1}   = max(0,  N * cwnd_k - BDP)
                [Q_k cancels via ACK clocking + drain derivation]
  F_k       = max(0,  (Q_{k+1} - K)) / (N * cwnd_k)
                [fraction of injected pkts marked; corrected denominator]
  alpha_{k+1} = (1-g)*alpha_k + g*F_k
  cwnd_{k+1}  = max(cwnd_min, cwnd_k*(1 - alpha_{k+1}/2))  if Q_{k+1} > K
              = cwnd_k + 1                                   otherwise
  FCT       = sum(RTT_k for k=0..K*)  [until cumulative pkts >= S_pkts]
  FCT_ideal = N * S_bytes / C_bytes + RTT_prop
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from matplotlib.patches import FancyBboxPatch

# ── Fixed physical constants ───────────────────────────────────────────────
PKT       = 1500          # bytes per packet
C_BYTES   = 25e9 / 8 / 1e6   # 3125.0  bytes/us
C_PKTS    = C_BYTES / PKT     # 2.0833  pkts/us
RTT_PROP  = 10.0         # us
K_PKTS    = 65.0          # ECN threshold in packets
G         = 1.0 / 16      # DCTCP EWMA gain
CWND_MIN  = 2.0           # protocol floor in packets
BDP       = C_PKTS * RTT_PROP  # 208.33 packets
MAX_RTTS  = 5000          # simulation cap
TABLE_MAX_ROWS = 200      # max rows to display in on-figure table


# ── Core simulation ────────────────────────────────────────────────────────
def simulate(N, S_KB, cwnd0):
    """
    Simulate DCTCP dynamics for N flows each of size S_KB kilobytes,
    starting with cwnd0 packets per flow.

    Returns dict of per-RTT arrays and scalar summary stats.
    """
    S_bytes = S_KB * 1000.0
    S_pkts  = S_bytes / PKT

    Q, cw, alpha, cum = 0.0, float(cwnd0), 0.0, 0.0

    ks, Qs, CWs, Als, Fks, RTTs = [], [], [], [], [], []

    for k in range(MAX_RTTS):
        RTTk = RTT_PROP + Q / C_PKTS

        # record state at start of RTT k
        ks.append(k)
        Qs.append(Q)
        CWs.append(cw)
        Als.append(alpha)
        RTTs.append(RTTk)

        # queue update  [Q_k cancels — net queue = injection - BDP]
        Qnext = max(0.0, N * cw - BDP)

        # marking fraction  [corrected: denominator = N*cwnd_k]
        Fk = max(0.0, Qnext - K_PKTS) / (N * cw) if cw > 0 else 0.0
        Fks.append(Fk)

        # alpha EWMA
        alpha_next = (1.0 - G) * alpha + G * Fk

        # cwnd update
        if Qnext > K_PKTS:
            cw_next = max(CWND_MIN, cw * (1.0 - alpha_next / 2.0))
        else:
            cw_next = cw + 1.0

        # flow completion check
        cum += cw
        if cum >= S_pkts:
            # record final state
            ks.append(k + 1)
            Qs.append(Qnext)
            CWs.append(cw_next)
            Als.append(alpha_next)
            Fks.append(Fk)
            RTTs.append(RTT_PROP + Qnext / C_PKTS)
            break

        Q, cw, alpha = Qnext, cw_next, alpha_next

    fct       = sum(RTTs)
    fct_ideal = N * S_bytes / C_BYTES + RTT_PROP

    # detect regime
    if max(Qs) <= K_PKTS * 1.05:
        regime = "Good"
    elif min(CWs) <= CWND_MIN * 1.1:
        if max(Qs) > K_PKTS * 2:
            regime = "Medium case 2  (cwnd hits protocol floor)"
        else:
            regime = "Medium case 1  (cwnd converges above floor)"
    else:
        regime = "Medium case 1  (cwnd converges above floor)"

    return dict(
        ks=np.array(ks), Qs=np.array(Qs), CWs=np.array(CWs),
        Als=np.array(Als), Fks=np.array(Fks), RTTs=np.array(RTTs),
        fct=fct, fct_ideal=fct_ideal,
        k_star=ks[-1], regime=regime,
        N=N, S_KB=S_KB, cwnd0=cwnd0
    )


# ── Plot helpers ───────────────────────────────────────────────────────────
COLORS = dict(Q='#378ADD', K='#E24B4A', CW='#1D9E75', AL='#BA7517', FK='#D4537E', RTT='#5555AA')

def _style_ax(ax, ylabel, ylim=None):
    ax.set_facecolor('#f9f9f7')
    ax.tick_params(labelsize=9, colors='#555')
    ax.set_ylabel(ylabel, fontsize=9, color='#555')
    ax.set_xlabel('RTT  k', fontsize=9, color='#555')
    for sp in ax.spines.values():
        sp.set_color('#ddd')
    ax.grid(True, color='#e0e0e0', linewidth=0.5)
    if ylim:
        ax.set_ylim(ylim)

def _draw_table(d, ax_tab, max_rows=TABLE_MAX_ROWS):
    """Render per-RTT numeric values inside the figure."""
    ax_tab.cla()
    ax_tab.axis('off')
    rows = list(zip(d['ks'], d['Qs'], d['CWs'], d['RTTs'], d['Als'], d['Fks']))
    shown = rows[:max_rows]
    header = " k    Q_pkts   cwnd  RTT_us  alpha    F_k"
    lines = [header]
    for k, q, cw, rtt, al, fk in shown:
        lines.append(f"{k:4d}  {q:7.1f}  {cw:6.1f}  {rtt:7.1f}  {al:6.4f}  {fk:6.4f}")
    if len(rows) > max_rows:
        lines.append(f"... ({len(rows) - max_rows} more RTTs not shown)")
    ax_tab.text(0.01, 0.99, "\n".join(lines), fontsize=7.5, family='monospace',
                va='top', ha='left', color='#444')


def draw_all(d, axs, info_ax, table_ax):
    ax_Q, ax_CW, ax_RTT, ax_A = axs

    for ax in axs:
        ax.cla()
    info_ax.cla()
    info_ax.axis('off')
    info_ax.set_xlim(0, 1)
    info_ax.set_ylim(0, 1)

    ks = d['ks']

    # ── Queue ──────────────────────────────────────────────────────────────
    ax_Q.plot(ks, d['Qs'], color=COLORS['Q'], lw=1.5, label='$Q_k$')
    ax_Q.axhline(K_PKTS, color=COLORS['K'], lw=1.0,
                 linestyle='--', label=f'K = {K_PKTS:.0f} pkts')
    ax_Q.axhline(BDP, color='#888', lw=0.8,
                 linestyle=':', label=f'BDP = {BDP:.1f} pkts')
    ax_Q.legend(fontsize=8, loc='upper right')
    _style_ax(ax_Q, 'Queue  (pkts)')

    # ── cwnd ───────────────────────────────────────────────────────────────
    ax_CW.plot(ks, d['CWs'], color=COLORS['CW'], lw=1.5, label='$cwnd_k$')
    ax_CW.axhline(CWND_MIN, color='#aaa', lw=0.8,
                  linestyle=':', label=f'$cwnd_{{min}}$ = {CWND_MIN:.0f} pkts')
    ax_CW.legend(fontsize=8, loc='upper right')
    _style_ax(ax_CW, 'cwnd  (pkts)')

    # ── RTT ───────────────────────────────────────────────────────────────
    ax_RTT.plot(ks, d['RTTs'], color=COLORS['RTT'], lw=1.2, label='$RTT_k$')
    ax_RTT.axhline(RTT_PROP, color=COLORS['K'], lw=1.0,
                   linestyle='--', label=f'$RTT_{{prop}}$ = {RTT_PROP:.0f} µs')
    ax_RTT.legend(fontsize=8, loc='upper right')
    _style_ax(ax_RTT, 'RTT  (µs)')

    # ── alpha and F_k ──────────────────────────────────────────────────────
    ax_A.plot(ks, d['Als'], color=COLORS['AL'], lw=1.5, label=r'$\alpha_k$')
    ax_A.plot(ks, d['Fks'], color=COLORS['FK'], lw=1.0,
              linestyle='--', label='$F_k$')
    ax_A.set_ylim(0, 1.05)
    ax_A.legend(fontsize=8, loc='upper right')
    _style_ax(ax_A, r'$\alpha$ / $F$', ylim=(0, 1.05))

    # ── Info panel ─────────────────────────────────────────────────────────
    inflation = d['fct'] / d['fct_ideal'] if d['fct_ideal'] > 0 else float('inf')

    lines = [
        ("Regime",          d['regime']),
        ("N  (flows)",      str(d['N'])),
        ("S  (flow size)",  f"{d['S_KB']} KB  =  {d['S_KB']*1000/PKT:.0f} pkts"),
        ("cwnd₀",           f"{d['cwnd0']} pkts"),
        ("K*  (RTTs done)", str(d['k_star'])),
        ("FCT",             f"{d['fct']:.1f} µs"),
        ("FCT_ideal",       f"{d['fct_ideal']:.1f} µs   [= N·S/C + RTT_prop]"),
        ("FCT inflation",   f"{inflation:.2f}×"),
        ("Final Q",         f"{d['Qs'][-1]:.1f} pkts"),
        ("Final cwnd",      f"{d['CWs'][-1]:.1f} pkts"),
        ("Peak Q",          f"{d['Qs'].max():.1f} pkts"),
        ("─ Fixed ─",       ""),
        ("C",               "25 Gbps  =  3125 B/µs  =  2.083 pkts/µs"),
        ("RTT_prop",        f"{RTT_PROP} µs"),
        ("BDP",             f"{BDP:.2f} pkts"),
        ("K",               f"{K_PKTS} pkts"),
        ("g",               "1/16"),
        ("cwnd_min",        f"{CWND_MIN} pkts"),
    ]

    col_label = '#888'
    col_value = '#222'
    col_head  = '#c0392b'

    y = 0.97
    dy = 0.054
    for label, value in lines:
        if label.startswith('─'):
            info_ax.axhline(y + 0.01, xmin=0.0, xmax=1.0,
                            color='#ddd', lw=0.8)
            y -= dy * 0.6
            continue
        info_ax.text(0.01, y, label + ':', transform=info_ax.transAxes,
                     fontsize=8.5, color=col_label, va='top', ha='left')
        info_ax.text(0.38, y, value, transform=info_ax.transAxes,
                     fontsize=8.5, color=col_value, va='top', ha='left',
                     fontweight='bold' if label in ('Regime','FCT inflation') else 'normal')
        y -= dy

    # regime colour tag
    regime_colors = {
        'Good':          '#27ae60',
        'Medium case 1': '#e67e22',
        'Medium case 2': '#c0392b',
    }
    rc = next((v for k, v in regime_colors.items() if k in d['regime']), '#888')
    fancy = FancyBboxPatch((0.0, 0.0), 1.0, 0.045,
                            boxstyle="round,pad=0.01",
                            transform=info_ax.transAxes,
                            facecolor=rc, edgecolor='none', alpha=0.15)
    info_ax.add_patch(fancy)
    info_ax.text(0.5, 0.022, d['regime'], transform=info_ax.transAxes,
                 fontsize=9, color=rc, va='center', ha='center', fontweight='bold')

    # ── Values table ───────────────────────────────────────────────────────
    _draw_table(d, table_ax)


# ── Build figure ───────────────────────────────────────────────────────────
def main():
    fig = plt.figure(figsize=(15, 9), facecolor='#fafaf8')
    fig.suptitle('DCTCP Regime Simulation  —  25 Gbps datacenter link',
                 fontsize=12, color='#333', y=0.98)

    # layout: 4 plots left, info + table on right, sliders at bottom
    gs_top = gridspec.GridSpec(4, 2, figure=fig,
                               left=0.06, right=0.98,
                               top=0.92, bottom=0.22,
                               hspace=0.35, wspace=0.32,
                               width_ratios=[2.8, 1.0])

    ax_Q   = fig.add_subplot(gs_top[0, 0])
    ax_CW  = fig.add_subplot(gs_top[1, 0])
    ax_RTT = fig.add_subplot(gs_top[2, 0])
    ax_A   = fig.add_subplot(gs_top[3, 0])
    ax_inf = fig.add_subplot(gs_top[0:2, 1])
    ax_tab = fig.add_subplot(gs_top[2:, 1])
    ax_inf.axis('off')
    ax_tab.axis('off')
    ax_inf.set_facecolor('#f5f4f0')
    ax_tab.set_facecolor('#f5f4f0')

    # slider axes
    sl_color  = '#ece9e2'
    ax_sN  = fig.add_axes([0.10, 0.14, 0.35, 0.025], facecolor=sl_color)
    ax_sS  = fig.add_axes([0.10, 0.10, 0.35, 0.025], facecolor=sl_color)
    ax_sCW = fig.add_axes([0.10, 0.06, 0.35, 0.025], facecolor=sl_color)
    ax_btn = fig.add_axes([0.82, 0.06, 0.10, 0.05])

    sl_N  = Slider(ax_sN,  'N  (flows)',       1,   5000,  valinit=5,   valstep=1)
    sl_S  = Slider(ax_sS,  'S  (KB)',          10, 5000, valinit=500, valstep=10)
    sl_CW = Slider(ax_sCW, 'cwnd₀  (pkts)',    1,  200,  valinit=10,  valstep=1)

    for sl in (sl_N, sl_S, sl_CW):
        sl.label.set_fontsize(9)
        sl.valtext.set_fontsize(9)

    btn_reset = Button(ax_btn, 'Reset', color=sl_color, hovercolor='#d5d0c8')
    btn_reset.label.set_fontsize(9)

    axs = (ax_Q, ax_CW, ax_RTT, ax_A)

    def refresh(val=None):
        N   = int(sl_N.val)
        S   = float(sl_S.val)
        cw0 = int(sl_CW.val)
        d   = simulate(N, S, cw0)
        draw_all(d, axs, ax_inf, ax_tab)
        fig.canvas.draw_idle()

    def reset(event):
        sl_N.reset(); sl_S.reset(); sl_CW.reset()

    sl_N.on_changed(refresh)
    sl_S.on_changed(refresh)
    sl_CW.on_changed(refresh)
    btn_reset.on_clicked(reset)

    # equation box at bottom
    eq_text = (
        r"$RTT_k = RTT_{prop} + Q_k / C_{pkts}$     "
        r"$Q_{k+1} = \max(0,\ N \cdot cwnd_k - BDP)$     "
        r"$F_k = \max(0,\ Q_{k+1} - K)\ /\ (N \cdot cwnd_k)$     "
        r"$\alpha_{k+1} = (1-g)\alpha_k + g F_k$     "
        r"$FCT_{ideal} = N \cdot S / C + RTT_{prop}$"
    )
    fig.text(0.06, 0.01, eq_text, fontsize=7.5, color='#777',
             va='bottom', ha='left', style='italic')

    refresh()
    plt.show()


if __name__ == '__main__':
    main()