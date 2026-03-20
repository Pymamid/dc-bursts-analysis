"""
DCTCP Regime Simulation  v3
============================
Interactive simulation of DCTCP queue/cwnd/alpha dynamics.
Includes ssthresh, slow start, loss, and short-flow queuing delay.

Fixed parameters:
  C        = 25 Gbps = 3125 bytes/us = 2.0833 pkts/us
  RTT_prop = 100 us   (avg datacenter RTT)
  K        = 65 packets  (ECN threshold)
  g        = 1/16        (DCTCP EWMA gain)
  PKT      = 1500 bytes  (packet size)
  cwnd_min = 2 packets   (protocol floor)
  BDP      = C_pkts * RTT_prop = 208.33 pkts

Slider parameters:
  N       = number of incast flows
  S       = flow size (KB)
  cwnd_0  = initial cwnd (pkts)
  B       = buffer size (pkts)

Equations (per RTT k):
  RTT_k       = RTT_prop + Q_k / C_pkts
  Q_{k+1}     = max(0,  N * cwnd_k - BDP)       [Q_k cancels via ACK clocking]

  -- Loss check --
  if Q_{k+1} > B:
      ssthresh   = max(cwnd_min, cwnd_k / 2)
      cwnd_{k+1} = cwnd_min
      alpha unchanged

  -- No loss --
  F_k         = max(0, Q_{k+1} - K) / (N * cwnd_k)   [corrected denominator]
  alpha_{k+1} = (1-g)*alpha_k + g*F_k
  if Q_{k+1} > K :  cwnd_{k+1} = max(cwnd_min, cwnd_k*(1 - alpha_{k+1}/2))
  elif cwnd_k < ssthresh: cwnd_{k+1} = min(cwnd_k*2, ssthresh)  [slow start]
  else:               cwnd_{k+1} = cwnd_k + 1                   [cong. avoid.]

  BCT       = sum(RTT_k for k=0..K*)          [burst completion time]
  BCT_ideal = N * S_bytes / C_bytes + RTT_prop [serialisation lower bound]

Short-flow queuing delay (metric for medium-regime harm):
  A single-packet short flow arriving at RTT k sees queue Q_k ahead of it.
  short_qdelay_k   = Q_k / C_pkts                   [us]
  short_FCT_k      = RTT_prop + short_qdelay_k       [us]
  short_FCT_ideal  = RTT_prop + K / C_pkts           [us]  (queue at K)
  Steady-state values use the last third of simulation RTTs.

Regime definitions (principled):
  Good   : DCTCP operating as intended.
           Queue settles ~K. cwnd finds equilibrium above protocol floor.
           Short flows experience expected queuing delay ~K/C.
           BCT ≈ BCT_ideal (link fully utilised).

  Medium : DCTCP objective violated — short queues not maintained.
           cwnd hits protocol floor (1-2 MSS). Queue sits at Q* >> K.
           BCT ≈ BCT_ideal (buffer never drains, link stays utilised).
           BUT short flows see Q*/C queuing delay instead of K/C — significantly worse.

  Bad    : Burst itself is harmed.
           Buffer overflow → drops → retransmissions → BCT > BCT_ideal.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from matplotlib.patches import FancyBboxPatch
import matplotlib.lines as mlines

# ── Fixed physical constants ───────────────────────────────────────────────
PKT           = 1500
C_BYTES       = 25e9 / 8 / 1e6   # 3125.0  bytes/us
C_PKTS        = C_BYTES / PKT     # 2.0833  pkts/us
RTT_PROP      = 20.0             # us
K_PKTS        = 65.0              # ECN threshold, packets
G             = 1.0 / 16
CWND_MIN      = 2.0               # protocol floor, packets
BDP           = C_PKTS * RTT_PROP # 208.33 packets
MAX_RTTS      = 5000
TABLE_MAX_ROWS= 30
SSTHRESH_INIT = 10.0 # check how much this generally is....

# ideal short-flow queuing delay: queue sitting exactly at K
SHORT_QDELAY_IDEAL = K_PKTS / C_PKTS          # us
SHORT_FCT_IDEAL    = RTT_PROP + SHORT_QDELAY_IDEAL  # us


# ── Core simulation ────────────────────────────────────────────────────────
def simulate(N, S_KB, cwnd0, B_pkts):
    S_bytes  = S_KB * 1000.0
    S_pkts   = S_bytes / PKT

    Q        = 0.0
    cw       = float(cwnd0)
    alpha    = 0.0
    ssthresh = SSTHRESH_INIT
    cum      = 0.0

    ks, Qs, CWs, Als, Fks, RTTs, SSTs, Losses = [], [], [], [], [], [], [], []

    for k in range(MAX_RTTS):
        RTTk = RTT_PROP + Q / C_PKTS

        ks.append(k);   Qs.append(Q);     CWs.append(cw)
        Als.append(alpha); RTTs.append(RTTk); SSTs.append(ssthresh)

        Qnext = max(0.0, N * cw - BDP)

        # ── loss: buffer overflow ──────────────────────────────────────────
        if Qnext > B_pkts:
            Losses.append(k)
            Fks.append(0.0)
            ssthresh   = max(CWND_MIN, cw / 2.0)
            cw_next    = CWND_MIN
            alpha_next = alpha

        else:
            # ── ECN marking fraction [denominator = N*cwnd_k] ─────────────
            Fk         = max(0.0, Qnext - K_PKTS) / (N * cw) if cw > 0 else 0.0
            Fks.append(Fk)
            alpha_next = (1.0 - G) * alpha + G * Fk

            if Qnext > K_PKTS:                          # DCTCP decrease
                cw_next = max(CWND_MIN, cw * (1.0 - alpha_next / 2.0))
            elif cw < ssthresh:                          # slow start
                cw_next = min(cw * 2.0, ssthresh)
            else:                                        # cong. avoidance
                cw_next = cw + 1.0

        cum += cw
        if cum >= S_pkts:
            ks.append(k + 1);    Qs.append(Qnext);   CWs.append(cw_next)
            Als.append(alpha_next); Fks.append(Fks[-1])
            RTTs.append(RTT_PROP + Qnext / C_PKTS);  SSTs.append(ssthresh)
            break

        Q, cw, alpha = Qnext, cw_next, alpha_next

    # ── derived arrays ─────────────────────────────────────────────────────
    Qs_arr  = np.array(Qs)
    CWs_arr = np.array(CWs)

    # short-flow queuing delay at each RTT k = Q_k / C_pkts
    short_qdelay = Qs_arr / C_PKTS           # us
    short_fct    = RTT_PROP + short_qdelay   # us

    # steady-state window = last third of simulation
    n        = len(ks)
    ss_start = max(0, n * 2 // 3)
    Qs_ss    = Qs_arr[ss_start:]
    CWs_ss   = CWs_arr[ss_start:]
    sq_ss    = short_qdelay[ss_start:]

    ss_mean_Q       = float(Qs_ss.mean())
    ss_mean_qdelay  = float(sq_ss.mean())
    ss_mean_sfct    = RTT_PROP + ss_mean_qdelay
    sfct_inflation  = ss_mean_sfct / SHORT_FCT_IDEAL

    bct       = sum(RTTs)
    bct_ideal = N * S_bytes / C_BYTES + RTT_PROP

    # ── regime detection ───────────────────────────────────────────────────
    # Principled definitions:
    #   Good   — queue settles ~K, cwnd above floor  (DCTCP as intended)
    #   Medium — cwnd hits protocol floor, Q* >> K   (objective violated)
    #   Bad    — buffer overflow, drops               (burst harmed)
    if len(Losses) > 0:
        regime = "Bad  —  drops occur, BCT inflated"
    elif CWs_ss.min() <= CWND_MIN * 1.1:
        regime = "Medium  —  cwnd at floor, Q* >> K, short flows harmed"
    elif Qs_ss.max() <= K_PKTS * 1.5:
        regime = "Good  —  queue ~K, DCTCP operating as intended"
    else:
        # queue still elevated but cwnd above floor — converging toward K
        regime = "Good  —  queue ~K, DCTCP operating as intended"

    return dict(
        ks            = np.array(ks),
        Qs            = Qs_arr,
        CWs           = CWs_arr,
        Als           = np.array(Als),
        Fks           = np.array(Fks),
        RTTs          = np.array(RTTs),
        SSTs          = np.array(SSTs),
        Losses        = Losses,
        short_qdelay  = short_qdelay,
        short_fct     = short_fct,
        ss_mean_Q     = ss_mean_Q,
        ss_mean_qdelay= ss_mean_qdelay,
        ss_mean_sfct  = ss_mean_sfct,
        sfct_inflation= sfct_inflation,
        bct           = bct,
        bct_ideal     = bct_ideal,
        k_star        = ks[-1],
        regime        = regime,
        N=N, S_KB=S_KB, cwnd0=cwnd0, B_pkts=B_pkts,
    )


# ── Colours ────────────────────────────────────────────────────────────────
COL = dict(
    Q     = '#378ADD',
    K     = '#E24B4A',
    B     = '#A32D2D',
    CW    = '#1D9E75',
    SST   = '#BA7517',
    AL    = '#534AB7',
    FK    = '#D4537E',
    RTT   = '#0F6E56',
    LOSS  = '#E24B4A',
    SQ    = '#D85A30',      # short-flow queuing delay
    SQI   = '#639922',      # ideal short-flow queuing delay
)

REGIME_COLORS = {
    'Good':   '#1D9E75',
    'Medium': '#BA7517',
    'Bad':    '#A32D2D',
}

def _regime_color(regime):
    return next((v for k, v in REGIME_COLORS.items() if k in regime), '#888')

def _style_ax(ax, ylabel, ylim=None):
    ax.set_facecolor('#f9f9f7')
    ax.tick_params(labelsize=8, colors='#555')
    ax.set_ylabel(ylabel, fontsize=8, color='#555')
    ax.set_xlabel('RTT  k', fontsize=8, color='#555')
    for sp in ax.spines.values():
        sp.set_color('#ddd')
    ax.grid(True, color='#e4e4e4', linewidth=0.5, zorder=0)
    if ylim:
        ax.set_ylim(ylim)

def _mark_losses(ax, losses):
    for k in losses:
        ax.axvspan(k - 0.5, k + 0.5, color=COL['LOSS'], alpha=0.18, zorder=1)


# ── Draw ───────────────────────────────────────────────────────────────────
def draw_all(d, axs, info_ax, table_ax):
    ax_Q, ax_CW, ax_RTT, ax_A, ax_SQ = axs
    for ax in axs:
        ax.cla()
    info_ax.cla(); info_ax.axis('off')
    table_ax.cla(); table_ax.axis('off')
    info_ax.set_xlim(0, 1); info_ax.set_ylim(0, 1)

    ks     = d['ks']
    losses = d['Losses']

    # ── 1. Queue ───────────────────────────────────────────────────────────
    ax_Q.plot(ks, d['Qs'], color=COL['Q'], lw=1.5, label='$Q_k$', zorder=3)
    ax_Q.axhline(K_PKTS,     color=COL['K'],  lw=1.0, ls='--',
                 label=f'K = {K_PKTS:.0f} pkts  (ECN threshold)')
    ax_Q.axhline(d['B_pkts'],color=COL['B'],  lw=1.0, ls='-.',
                 label=f'B = {d["B_pkts"]:.0f} pkts  (buffer)')
    ax_Q.axhline(BDP,        color='#999',    lw=0.7, ls=':',
                 label=f'BDP = {BDP:.1f} pkts')
    _mark_losses(ax_Q, losses)
    ax_Q.legend(fontsize=7, loc='upper right', framealpha=0.9)
    _style_ax(ax_Q, 'Queue  (pkts)')

    # ── 2. cwnd + ssthresh ────────────────────────────────────────────────
    ax_CW.plot(ks, d['CWs'], color=COL['CW'],  lw=1.5, label='$cwnd_k$',   zorder=3)
    ax_CW.plot(ks, d['SSTs'],color=COL['SST'], lw=1.0, ls='--',
               label='$ssthresh_k$', zorder=2)
    ax_CW.axhline(CWND_MIN, color='#bbb', lw=0.7, ls=':',
                  label=f'$cwnd_{{min}}$ = {CWND_MIN:.0f} pkts')
    _mark_losses(ax_CW, losses)
    ax_CW.legend(fontsize=7, loc='upper right', framealpha=0.9)
    _style_ax(ax_CW, 'cwnd  (pkts)')

    # ── 3. RTT ────────────────────────────────────────────────────────────
    ax_RTT.plot(ks, d['RTTs'], color=COL['RTT'], lw=1.2, label='$RTT_k$', zorder=3)
    ax_RTT.axhline(RTT_PROP, color=COL['K'], lw=0.8, ls='--',
                   label=f'$RTT_{{prop}}$ = {RTT_PROP:.0f} µs')
    _mark_losses(ax_RTT, losses)
    ax_RTT.legend(fontsize=7, loc='upper right', framealpha=0.9)
    _style_ax(ax_RTT, 'RTT  (µs)')

    # ── 4. alpha / F_k ────────────────────────────────────────────────────
    ax_A.plot(ks, d['Als'], color=COL['AL'], lw=1.5, label=r'$\alpha_k$', zorder=3)
    ax_A.plot(ks, d['Fks'], color=COL['FK'], lw=1.0, ls='--',
              label='$F_k$', zorder=2)
    _mark_losses(ax_A, losses)
    ax_A.legend(fontsize=7, loc='upper right', framealpha=0.9)
    _style_ax(ax_A, r'$\alpha$ / $F$', ylim=(0, 1.05))

    # ── 5. Short-flow queuing delay ───────────────────────────────────────
    # A short flow (1 pkt) arriving at RTT k finds Q_k bytes ahead of it.
    # Its queuing delay = Q_k / C_pkts us. Its FCT = RTT_prop + Q_k/C_pkts.
    ax_SQ.plot(ks, d['short_qdelay'], color=COL['SQ'], lw=1.5,
               label='Short-flow queueing delay  $= Q_k / C$', zorder=3)
    ax_SQ.axhline(SHORT_QDELAY_IDEAL, color=COL['SQI'], lw=1.0, ls='--',
                  label=f'Ideal  $= K/C$ = {SHORT_QDELAY_IDEAL:.1f} µs  (queue at K)')
    ax_SQ.axhline(d['ss_mean_qdelay'], color=COL['SQ'], lw=0.8, ls=':',
                  label=f'Steady-state mean = {d["ss_mean_qdelay"]:.1f} µs')
    _mark_losses(ax_SQ, losses)
    ax_SQ.legend(fontsize=7, loc='upper right', framealpha=0.9)
    _style_ax(ax_SQ, 'Short-flow qdelay  (µs)')

    # loss overlay on all axes
    if losses:
        loss_patch = mlines.Line2D([], [], color=COL['LOSS'], alpha=0.5,
                                   lw=6, label=f'Loss events ({len(losses)})')
        for ax in axs:
            h, l = ax.get_legend_handles_labels()
            ax.legend(handles=h + [loss_patch], fontsize=7,
                      loc='upper right', framealpha=0.9)

    # ── Info panel ─────────────────────────────────────────────────────────
    bct_inf = d['bct'] / d['bct_ideal'] if d['bct_ideal'] > 0 else float('inf')
    rc      = _regime_color(d['regime'])

    info_rows = [
        # ── Regime ──
        ("Regime",              d['regime']),
        ("─", ""),
        # ── Inputs ──
        ("N  (flows)",          str(d['N'])),
        ("S  (flow size)",      f"{d['S_KB']} KB  =  {d['S_KB']*1000/PKT:.0f} pkts"),
        ("cwnd₀",               f"{d['cwnd0']} pkts"),
        ("B  (buffer)",         f"{d['B_pkts']:.0f} pkts"),
        ("─", ""),
        # ── BCT (incast burst) ──
        ("BCT",                 f"{d['bct']:.1f} µs"),
        ("BCT_ideal",           f"{d['bct_ideal']:.1f} µs   [= N·S/C + RTT_prop]"),
        ("BCT inflation",       f"{bct_inf:.3f}×   "
                                + ("← burst unharmed" if bct_inf < 1.01 else "← burst harmed")),
        ("Loss events",         str(len(d['Losses'])) +
                                (f"  @ RTTs {d['Losses'][:4]}" if d['Losses'] else "  (none)")),
        ("─", ""),
        # ── Short-flow queuing delay ──
        ("SS mean queue Q*",    f"{d['ss_mean_Q']:.1f} pkts"),
        ("SS mean qdelay",      f"{d['ss_mean_qdelay']:.2f} µs   [= Q*/C]"),
        ("Ideal qdelay",        f"{SHORT_QDELAY_IDEAL:.2f} µs   [= K/C]"),
        ("Short-flow FCT",      f"{d['ss_mean_sfct']:.2f} µs   [= RTT_prop + Q*/C]"),
        ("Short-flow FCT ideal",f"{SHORT_FCT_IDEAL:.2f} µs   [= RTT_prop + K/C]"),
        ("Short-FCT inflation", f"{d['sfct_inflation']:.2f}×   "
                                + ("← short flows unharmed" if d['sfct_inflation'] < 1.05
                                   else "← short flows harmed")),
        ("─", ""),
        # ── Fixed params ──
        ("C",                   "25 Gbps = 3125 B/µs = 2.083 pkts/µs"),
        ("RTT_prop",            f"{RTT_PROP} µs"),
        ("BDP",                 f"{BDP:.2f} pkts"),
        ("K  (ECN thresh)",     f"{K_PKTS} pkts"),
        ("g",                   "1/16"),
        ("cwnd_min",            f"{CWND_MIN} pkts"),
        ("K*  (RTTs done)",     str(d['k_star'])),
    ]

    y, dy = 0.98, 0.043
    bold_keys = {'Regime', 'BCT inflation', 'Short-FCT inflation', 'Loss events'}
    for label, value in info_rows:
        if label == "─":
            info_ax.axhline(y + 0.005, xmin=0, xmax=1,
                            color='#ddd', lw=0.7)
            y -= dy * 0.4
            continue
        info_ax.text(0.01, y, label + ':',
                     transform=info_ax.transAxes,
                     fontsize=7.5, color='#888', va='top')
        info_ax.text(0.40, y, value,
                     transform=info_ax.transAxes,
                     fontsize=7.5, color='#222', va='top',
                     fontweight='bold' if label in bold_keys else 'normal')
        y -= dy

    # regime badge
    fancy = FancyBboxPatch((0, 0), 1, 0.038, boxstyle="round,pad=0.01",
                            transform=info_ax.transAxes,
                            facecolor=rc, edgecolor='none', alpha=0.18)
    info_ax.add_patch(fancy)
    info_ax.text(0.5, 0.019, d['regime'],
                 transform=info_ax.transAxes,
                 fontsize=8, color=rc, va='center', ha='center', fontweight='bold')

    # ── Per-RTT table ──────────────────────────────────────────────────────
    hdr = (f"{'k':>4}  {'Q':>7}  {'cwnd':>6}  {'ssth':>6}  "
           f"{'RTT':>7}  {'α':>6}  {'F':>6}  {'qd_us':>7}  {'loss':>4}")
    lines = [hdr, "─" * len(hdr)]
    for i, k in enumerate(d['ks'][:TABLE_MAX_ROWS]):
        lf = "●" if k in d['Losses'] else ""
        lines.append(
            f"{k:4d}  {d['Qs'][i]:7.1f}  {d['CWs'][i]:6.1f}  "
            f"{d['SSTs'][i]:6.1f}  {d['RTTs'][i]:7.1f}  "
            f"{d['Als'][i]:6.4f}  {d['Fks'][i]:6.4f}  "
            f"{d['short_qdelay'][i]:7.2f}  {lf:>4}"
        )
    if len(d['ks']) > TABLE_MAX_ROWS:
        lines.append(f"... ({len(d['ks']) - TABLE_MAX_ROWS} more RTTs not shown)")

    table_ax.text(0.01, 0.99, "\n".join(lines),
                  fontsize=6.5, family='monospace',
                  va='top', ha='left', color='#333',
                  transform=table_ax.transAxes)


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    fig = plt.figure(figsize=(17, 12), facecolor='#fafaf8')
    fig.suptitle(
        'DCTCP Regime Simulation  v3  —  25 Gbps · RTT_prop = 100 µs · K = 65 pkts',
        fontsize=11, color='#333', y=0.99
    )

    gs = gridspec.GridSpec(5, 2, figure=fig,
                           left=0.06, right=0.98,
                           top=0.96, bottom=0.20,
                           hspace=0.42, wspace=0.28,
                           width_ratios=[2.6, 1.0])

    ax_Q   = fig.add_subplot(gs[0, 0])
    ax_CW  = fig.add_subplot(gs[1, 0])
    ax_RTT = fig.add_subplot(gs[2, 0])
    ax_A   = fig.add_subplot(gs[3, 0])
    ax_SQ  = fig.add_subplot(gs[4, 0])
    ax_inf = fig.add_subplot(gs[0:3, 1])
    ax_tab = fig.add_subplot(gs[3:,  1])
    ax_inf.set_facecolor('#f5f4f0')
    ax_tab.set_facecolor('#f5f4f0')

    sl_bg = '#ece9e2'
    ax_sN = fig.add_axes([0.06, 0.155, 0.38, 0.020], facecolor=sl_bg)
    ax_sS = fig.add_axes([0.06, 0.122, 0.38, 0.020], facecolor=sl_bg)
    ax_sC = fig.add_axes([0.06, 0.089, 0.38, 0.020], facecolor=sl_bg)
    ax_sB = fig.add_axes([0.06, 0.056, 0.38, 0.020], facecolor=sl_bg)
    ax_bt = fig.add_axes([0.83, 0.070, 0.09, 0.040])

    sl_N  = Slider(ax_sN, 'N  (flows)',     1,   200, valinit=5,   valstep=1)
    sl_S  = Slider(ax_sS, 'S  (KB)',        10, 5000, valinit=500, valstep=10)
    sl_CW = Slider(ax_sC, 'cwnd₀  (pkts)', 1,   200, valinit=10,  valstep=1)
    sl_B  = Slider(ax_sB, 'B  (buf pkts)', 100, 2000, valinit=500, valstep=50)

    for sl in (sl_N, sl_S, sl_CW, sl_B):
        sl.label.set_fontsize(8.5)
        sl.valtext.set_fontsize(8.5)

    btn = Button(ax_bt, 'Reset', color=sl_bg, hovercolor='#d5d0c8')
    btn.label.set_fontsize(8.5)

    axs = (ax_Q, ax_CW, ax_RTT, ax_A, ax_SQ)

    def refresh(val=None):
        d = simulate(int(sl_N.val), float(sl_S.val),
                     int(sl_CW.val), float(sl_B.val))
        draw_all(d, axs, ax_inf, ax_tab)
        fig.canvas.draw_idle()

    def reset(event):
        for sl in (sl_N, sl_S, sl_CW, sl_B):
            sl.reset()

    for sl in (sl_N, sl_S, sl_CW, sl_B):
        sl.on_changed(refresh)
    btn.on_clicked(reset)

    eq = (
        r"$Q_{k+1}=\max(0, N \cdot cwnd_k - BDP)$  "
        r"$F_k=\max(0, Q_{k+1}-K)/(N \cdot cwnd_k)$  "
        r"$\alpha_{k+1}=(1-g)\alpha_k+gF_k$  "
        r"loss if $Q_{k+1}>B$  "
        r"$BCT_{ideal}=N \cdot S/C+RTT_{prop}$  "
        r"short-flow qdelay$= Q_k/C$  "
        r"ideal qdelay$= K/C$"
    )
    fig.text(0.02, 0.008, eq, fontsize=7, color='#999',
             va='bottom', ha='left', style='italic')

    refresh()
    plt.show()


if __name__ == '__main__':
    main()