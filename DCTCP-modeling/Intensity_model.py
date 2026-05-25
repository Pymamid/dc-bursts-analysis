"""
DCTCP Regime Simulation  v4
============================
New in v4:
  + Jitter slider J (µs)         — synchronicity dimension
  + QPS slider  → IAT (µs)       — IAT dimension
  + N_active(k) based on J       — staggered flow injection
  + IAT-modified initial conds   — Q0, α0, cwnd0 from prev burst
  + Trajectory-based regime      — based on max_Q vs K, B and T_burst vs RTT, T_C
  + Four-dimension summary panel
  + Decision tree panel

Fixed constants:
  C        = 25 Gbps = 3125 B/µs = 2.083 pkts/µs
  RTT_prop = 20 µs   (same-rack datacenter)
  K        = 65 pkts (ECN threshold)
  g        = 1/16    (DCTCP EWMA gain)
  cwnd_min = 2 pkts  (protocol floor)
  BDP      = C * RTT_prop = 41.67 pkts
  T_C      = RTT_prop / g = 320 µs  (DCTCP convergence time)

Equations (per RTT k):
  RTT_k     = RTT_prop + Q_k / C_pkts
  N_act(k)  = min(N, floor(T_wall*N/J)+1)   if J>0 else N
  inj_k     = N_old*cwnd_k + N_new*cwnd_init [old flows at cwnd_k, new at cwnd_init]
  Q_{k+1}   = max(0, inj_k - BDP)           [Q_k cancels via ACK clocking]
  F_k       = max(0, Q_{k+1}-K) / inj_k     [corrected: denominator = total injection]
  α_{k+1}   = (1-g)*α_k + g*F_k
  loss if Q_{k+1}>B: ssthresh=max(cwnd_min,cwnd/2), cwnd=cwnd_min, α reset
  cwnd update: DCTCP decrease / slow start / congestion avoidance

IAT initial conditions (from previous burst end state):
  Q_0  = max(0, Q_end - C_pkts * IAT)
  α_0  = α_end * (1-g)^(IAT/RTT_prop)
  cw_0 = min(cwnd_init, cw_end + IAT/RTT_prop)

Recovery timescales:
  T_drain = Q_end / C_pkts          (queue drains)
  T_α     = RTT_prop / g = T_C      (α decays to ~0)
  T_cwnd  = (cwnd_init-cw_end)*RTT  (cwnd recovers)

Regime (trajectory-based):
  Bad      : max_Q > B  (drops occur)
  Good     : max_Q ≤ K  (DCTCP never triggered)
  Degraded : K < max_Q ≤ B, with sub-cases:
    - DCTCP blind  : T_burst < RTT_prop
    - Case 2       : cwnd hit protocol floor
    - Case 1 partial: T_burst < T_C
    - Case 1 conv  : T_burst ≥ T_C
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from matplotlib.patches import FancyBboxPatch

# ── Fixed network constants ───────────────────────────────────────────────────
PKT_B   = 1500                    # bytes per packet
C_BUS   = 25e9 / 8 / 1e6         # 3125.0 B/µs
C_PKT   = C_BUS / PKT_B          # ~2.083 pkts/µs
RTT_P   = 20.0                    # µs  propagation RTT
K_PKT   = 65.0                    # ECN threshold, packets
G       = 1.0 / 16               # DCTCP EWMA gain
CW_MIN  = 2.0                     # protocol floor, packets
BDP     = C_PKT * RTT_P           # ~41.67 packets
T_C     = RTT_P / G               # 320 µs  DCTCP convergence time
SSTH0   = 1e6                     # initial ssthresh (≈ ∞)
MXRTT   = 5000                    # simulation cap

SQD_ID  = K_PKT / C_PKT          # ideal short-flow queuing delay µs
SFCT_ID = RTT_P + SQD_ID          # ideal short-flow FCT µs

# Colour palette
C = dict(
    Q='#378ADD', K='#E24B4A', B='#A32D2D', CW='#1D9E75',
    SS='#BA7517', AL='#534AB7', FK='#D4537E', RT='#0F6E56',
    SQ='#D85A30', SQI='#639922', LO='#E24B4A',
    GOOD='#1D9E75', DEG='#BA7517', BAD='#A32D2D', NN='#555555',
)


# ── Core per-RTT simulation ───────────────────────────────────────────────────
def _run(N, Sp, cwi, cw0, Q0, a0, Bp, J):
    """
    Simulate one incast burst.
    N   = number of flows
    Sp  = flow size (packets per flow)
    cwi = cwnd_init (initial window for flows joining mid-burst)
    cw0 = cwnd at burst start (may differ from cwi due to IAT recovery)
    Q0  = initial queue (from IAT)
    a0  = initial alpha (from IAT)
    Bp  = buffer size packets
    J   = jitter window µs (0 = fully synchronous)
    """
    Q = Q0; cw = float(cw0); a = float(a0); ss = SSTH0; cum = 0.0
    Tw = 0.0                         # wall-clock time µs
    Np = 1 if J > 0 else N           # flows active at t=0

    ks=[]; Qs=[]; CWs=[]; As=[]; Fs=[]; RTs=[]; SSs=[]; Nacs=[]; Sq=[]; Lo=[]

    for k in range(MXRTT):
        rtt = RTT_P + Q / C_PKT

        # Active flows based on wall-clock progress through jitter window
        Na  = min(N, int(Tw * N / J) + 1) if J > 0 else N
        Nn  = max(0, Na - Np)            # new flows joining this RTT
        No  = Np                          # old flows already active

        # Record state at start of RTT k
        ks.append(k); Qs.append(Q); CWs.append(cw); As.append(a)
        RTs.append(rtt); SSs.append(ss); Nacs.append(Na); Sq.append(Q / C_PKT)

        # Aggregate injection: old flows at cw, new flows at cwi
        inj  = max(1.0, No * cw + Nn * cwi)
        Qnxt = max(0.0, inj - BDP)

        # ── Loss: buffer overflow ─────────────────────────────────────────
        if Qnxt > Bp:
            Lo.append(k); Fs.append(0.0)
            ss   = max(CW_MIN, cw / 2.0)
            cw1  = CW_MIN
            a1   = 0.0              # reset α after RTO
            Qnxt = Bp               # cap queue at buffer

        # ── No loss ───────────────────────────────────────────────────────
        else:
            Fk = max(0.0, Qnxt - K_PKT) / inj
            Fs.append(Fk)
            a1 = (1 - G) * a + G * Fk

            if   Qnxt > K_PKT:  cw1 = max(CW_MIN, cw * (1 - a1 / 2))  # DCTCP decrease
            elif cw < ss:       cw1 = min(cw * 2, ss)                   # slow start
            else:               cw1 = cw + 1.0                           # cong. avoidance

        # Blend new flows (they arrive at cwi, not cw1)
        if Nn > 0 and Na > 0:
            cw1 = (No * cw1 + Nn * cwi) / Na

        # Track one flow's cumulative bytes sent
        cum += cw
        if cum >= Sp:
            # Record final state
            ks.append(k+1); Qs.append(Qnxt); CWs.append(cw1); As.append(a1)
            Fs.append(Fs[-1]); RTs.append(RTT_P + Qnxt / C_PKT)
            SSs.append(ss); Nacs.append(Na); Sq.append(Qnxt / C_PKT)
            break

        Tw += rtt; Q = Qnxt; cw = cw1; a = a1; Np = Na

    # Derived scalars
    Qa = np.array(Qs); Ca = np.array(CWs)
    Ra = np.array(RTs); Sa = np.array(Sq)
    n  = len(ks); ss_i = max(0, n * 2 // 3)
    sm_Q  = float(Qa[ss_i:].mean()) if n > 0 else 0.0
    sm_sq = float(Sa[ss_i:].mean()) if n > 0 else 0.0
    bct   = float(Ra.sum())
    bct_i = N * Sp * PKT_B / C_BUS + RTT_P   # N·S/C + RTT_prop

    return dict(
        ks=np.array(ks), Qs=Qa, CWs=Ca, As=np.array(As),
        Fs=np.array(Fs), RTs=Ra, SSs=np.array(SSs),
        Nacs=np.array(Nacs), Sq=Sa, Lo=Lo,
        sm_Q=sm_Q, sm_sq=sm_sq, sm_sfct=RTT_P + sm_sq,
        sfct_inf=(RTT_P + sm_sq) / SFCT_ID,
        bct=bct, bct_i=bct_i, kstar=ks[-1],
    )


def simulate(N, S_KB, cwi, Bp, J, QPS):
    """Full simulation: clean burst → IAT recovery → actual burst → regime."""
    Sp    = S_KB * 1000.0 / PKT_B
    IAT   = 1e6 / QPS if QPS > 0 else 1e9

    # Step 1: clean burst (Q=0, α=0, cw=cwi) → previous burst end state
    prev  = _run(N, Sp, cwi, cwi, 0.0, 0.0, Bp, J)
    Q_e   = float(prev['Qs'][-1])
    a_e   = float(prev['As'][-1])
    cw_e  = float(prev['CWs'][-1])

    # Step 2: IAT recovery → initial conditions for next burst
    Q_0   = max(0.0, Q_e  - C_PKT * IAT)
    a_0   = a_e  * ((1 - G) ** (IAT / RTT_P))
    cw_0  = min(float(cwi), cw_e + IAT / RTT_P)

    # Recovery timescales
    T_drain = Q_e / C_PKT            # µs until queue drains fully
    T_alpha = T_C                    # µs until α decays to ~1/e (= RTT_P/g)
    T_cwnd  = max(0.0, (cwi - cw_e) * RTT_P)  # µs until cwnd recovers

    # Step 3: actual burst with IAT-modified initial conditions
    d = _run(N, Sp, cwi, cw_0, Q_0, a_0, Bp, J)

    # ── Regime classification (trajectory-based) ──────────────────────────
    max_Q  = float(d['Qs'].max())
    T_bst  = d['bct']
    cw_min = float(d['CWs'].min())
    has_lo = len(d['Lo']) > 0
    N_eff  = int(max(d['Nacs'])) if len(d['Nacs']) > 0 else N

    if has_lo:
        reg = 'Bad';      rsub = 'Buffer overflow → drops → RTO'
    elif max_Q > K_PKT:
        if   T_bst < RTT_P:         reg = 'Degraded'; rsub = 'DCTCP blind: T_burst < RTT_prop'
        elif cw_min <= CW_MIN*1.1:  reg = 'Degraded'; rsub = 'Case 2: cwnd→floor, Q*>>K'
        elif T_bst >= T_C:          reg = 'Degraded'; rsub = 'Case 1: DCTCP converged, queue ~K'
        else:                       reg = 'Degraded'; rsub = 'Case 1: partial, T_burst < T_C'
    else:
        reg = 'Good'; rsub = 'Queue ≤ K throughout, DCTCP not triggered'

    d.update(
        N=N, S_KB=S_KB, cwi=cwi, Bp=Bp, J=J, QPS=QPS, IAT=IAT,
        Q_e=Q_e, a_e=a_e, cw_e=cw_e,
        Q_0=Q_0, a_0=a_0, cw_0=cw_0,
        T_drain=T_drain, T_alpha=T_alpha, T_cwnd=T_cwnd,
        max_Q=max_Q, T_bst=T_bst, cw_min=cw_min, has_lo=has_lo, N_eff=N_eff,
        reg=reg, rsub=rsub,
        i_val=N*cw_0, i_good=BDP+K_PKT, i_loss=BDP+Bp,
    )
    return d


# ── Plot helpers ──────────────────────────────────────────────────────────────
def _sty(ax, yl, ylim=None):
    ax.set_facecolor('#f9f9f7')
    ax.tick_params(labelsize=8, colors='#555')
    ax.set_ylabel(yl, fontsize=8, color='#555')
    ax.set_xlabel('RTT k', fontsize=8, color='#555')
    for sp in ax.spines.values(): sp.set_color('#ddd')
    ax.grid(True, color='#e4e4e4', lw=0.5, zorder=0)
    if ylim: ax.set_ylim(ylim)

def _lo(ax, Lo):
    for k in Lo:
        ax.axvspan(k - 0.5, k + 0.5, color=C['LO'], alpha=0.18, zorder=1)


def draw_plots(axs, d):
    aQ, aCW, aRT, aA, aSQ = axs
    for ax in axs: ax.cla()
    L = d['ks']; Lo = d['Lo']

    # 1. Queue
    aQ.plot(L, d['Qs'], color=C['Q'],  lw=1.5, label='$Q_k$',             zorder=3)
    aQ.axhline(K_PKT,     color=C['K'], lw=1.0, ls='--', label=f'K={K_PKT:.0f} pkts')
    aQ.axhline(d['Bp'],   color=C['B'], lw=1.0, ls='-.', label=f'B={d["Bp"]:.0f} pkts')
    aQ.axhline(BDP,       color='#999', lw=0.7, ls=':',  label=f'BDP={BDP:.1f}')
    _lo(aQ, Lo); aQ.legend(fontsize=7, loc='upper right', framealpha=0.9)
    _sty(aQ, 'Queue (pkts)')

    # 2. cwnd + ssthresh
    aCW.plot(L, d['CWs'], color=C['CW'], lw=1.5, label='$cwnd_k$',        zorder=3)
    aCW.plot(L, d['SSs'], color=C['SS'], lw=1.0, ls='--', label='$ssthresh_k$', zorder=2)
    aCW.axhline(CW_MIN,   color='#bbb',  lw=0.7, ls=':',
                label=f'$cwnd_{{min}}$={CW_MIN:.0f}')
    _lo(aCW, Lo); aCW.legend(fontsize=7, loc='upper right', framealpha=0.9)
    _sty(aCW, 'cwnd (pkts)')

    # 3. RTT
    aRT.plot(L, d['RTs'], color=C['RT'], lw=1.2, label='$RTT_k$',         zorder=3)
    aRT.axhline(RTT_P,    color=C['K'],  lw=0.8, ls='--',
                label=f'$RTT_{{prop}}$={RTT_P:.0f} µs')
    _lo(aRT, Lo); aRT.legend(fontsize=7, loc='upper right', framealpha=0.9)
    _sty(aRT, 'RTT (µs)')

    # 4. α and F_k
    aA.plot(L, d['As'], color=C['AL'], lw=1.5, label=r'$\alpha_k$',       zorder=3)
    aA.plot(L, d['Fs'], color=C['FK'], lw=1.0, ls='--', label='$F_k$',    zorder=2)
    _lo(aA, Lo); aA.legend(fontsize=7, loc='upper right', framealpha=0.9)
    _sty(aA, r'$\alpha$ / $F$', ylim=(0, 1.05))

    # 5. Short-flow queuing delay
    aSQ.plot(L, d['Sq'], color=C['SQ'],  lw=1.5,
             label='Short-flow qdelay  $= Q_k/C$',                        zorder=3)
    aSQ.axhline(SQD_ID,    color=C['SQI'], lw=1.0, ls='--',
                label=f'Ideal $K/C$ = {SQD_ID:.1f} µs')
    aSQ.axhline(d['sm_sq'], color=C['SQ'], lw=0.8, ls=':',
                label=f'SS mean = {d["sm_sq"]:.1f} µs')
    _lo(aSQ, Lo); aSQ.legend(fontsize=7, loc='upper right', framealpha=0.9)
    _sty(aSQ, 'Short-flow qdelay (µs)')


def draw_info(ax, d):
    """Stats summary panel (top right)."""
    ax.cla(); ax.axis('off'); ax.set_facecolor('#f5f4f0')
    rc  = C['GOOD'] if d['reg']=='Good' else C['BAD'] if d['reg']=='Bad' else C['DEG']
    bci = d['bct'] / d['bct_i'] if d['bct_i'] > 0 else 999.9

    rows = [
        ('Regime',          d['reg'],                                              True),
        ('─', '', False),
        ('N',               f"{d['N']} flows",                                    False),
        ('S',               f"{d['S_KB']} KB = {d['S_KB']*1000/PKT_B:.0f} pkts", False),
        ('cwnd_init',       f"{d['cwi']} pkts",                                   False),
        ('B  (buffer)',     f"{d['Bp']:.0f} pkts",                                False),
        ('J  (jitter)',     f"{d['J']:.1f} µs",                                  False),
        ('QPS → IAT',       f"{d['QPS']:.0f} → {d['IAT']:.0f} µs",              False),
        ('─', '', False),
        ('BCT',             f"{d['bct']:.1f} µs",                                False),
        ('BCT_ideal',       f"{d['bct_i']:.1f} µs  [N·S/C + RTT_prop]",         False),
        ('max Q',           f"{d['max_Q']:.1f} pkts",                           True),
        ('BCT inflation',   f"{bci:.3f}×",                                        True),
        ('Loss events',     f"{len(d['Lo'])}",                                    True),
        ('─', '', False),
        ('SS mean Q*',      f"{d['sm_Q']:.1f} pkts",                             False),
        ('SS mean qdelay',  f"{d['sm_sq']:.2f} µs",                              False),
        ('Ideal qdelay',    f"{SQD_ID:.2f} µs  [K/C]",                           False),
        ('SFCT inflation',  f"{d['sfct_inf']:.2f}×",                              True),
        ('─', '', False),
        ('BDP',             f"{BDP:.2f} pkts",                                    False),
        ('T_C',             f"{T_C:.0f} µs",                                     False),
        ('N_eff',           f"{d['N_eff']} (max flows active)",                  False),
        ('K*',              f"{d['kstar']} RTTs",                                 False),
    ]

    y = 0.98; dy = 0.043
    bk = {'Regime', 'BCT inflation', 'Loss events', 'SFCT inflation', 'max Q'}
    for lbl, val, bold in rows:
        if lbl == '─':
            ax.axhline(y + 0.005, xmin=0, xmax=1, color='#ddd', lw=0.7)
            y -= dy * 0.35; continue
        ax.text(0.02, y, lbl + ':', transform=ax.transAxes,
                fontsize=7.5, color='#888', va='top')
        ax.text(0.42, y, val, transform=ax.transAxes,
                fontsize=7.5,
                color=C['B'] if lbl == 'max Q' else (rc if lbl == 'Regime' else '#222'),
                va='top', fontweight='bold' if bold or lbl == 'max Q' else 'normal')
        y -= dy

    fp = FancyBboxPatch((0, 0), 1, 0.037, boxstyle="round,pad=0.01",
                         transform=ax.transAxes,
                         facecolor=rc, edgecolor='none', alpha=0.20)
    ax.add_patch(fp)
    ax.text(0.5, 0.018, d['reg'], transform=ax.transAxes,
            fontsize=9, color=rc, va='center', ha='center', fontweight='bold')


def draw_dims_tree(ax, d):
    """Four-dimension summary + decision tree (bottom right panel)."""
    ax.cla(); ax.axis('off'); ax.set_facecolor('#f5f4f0')

    GC = C['GOOD']; DC = C['DEG']; BC = C['BAD']; NC = C['NN']

    # Build list of (x_indent, text, color, bold)
    rows = []
    def L(ind, text, col=NC, bold=False):
        rows.append((ind, text, col, bold))

    # ── FOUR DIMENSIONS ──────────────────────────────────────────────────
    L(0, '─── FOUR DIMENSIONS ────────────────────────────────', NC, True)
    L(0, '')

    # Intensity
    iv = d['i_val']; ig = d['i_good']; il = d['i_loss']
    ic = GC if iv < ig else (DC if iv < il else BC)
    cmp_i = ('< BDP+K  → good' if iv < ig
              else ('< BDP+B  → degraded' if iv < il else '≥ BDP+B  → bad'))
    L(0, 'INTENSITY', '#333', True)
    L(2, f'N × cwnd₀  =  {d["N"]} × {d["cw_0"]:.1f}  =  {iv:.1f} pkts', ic)
    L(2, f'BDP+K = {ig:.1f}   BDP+B = {il:.1f}   →  {cmp_i}', NC)

    # Duration
    Tb = d['T_bst']
    dc = GC if Tb >= T_C else (DC if Tb >= RTT_P else BC)
    cmp_d = ('< RTT_prop → DCTCP blind' if Tb < RTT_P
              else ('< T_C → partial response' if Tb < T_C
                    else '≥ T_C  → DCTCP converged'))
    L(0, '')
    L(0, 'DURATION  (output = BCT)', '#333', True)
    L(2, f'T_burst  = {Tb:.1f} µs', dc)
    L(2, f'RTT_prop = {RTT_P:.0f} µs    T_C = {T_C:.0f} µs    → {cmp_d}', NC)

    # Synchronicity
    J = d['J']; Ne = d['N_eff']
    jc = GC if J >= RTT_P else DC
    cmp_j = ('J ≥ RTT_prop  → staggered, early feedback possible'
              if J >= RTT_P else 'J < RTT_prop → all flows commit before feedback')
    L(0, '')
    L(0, 'SYNCHRONICITY', '#333', True)
    L(2, f'J = {J:.1f} µs    (RTT_prop = {RTT_P:.0f} µs)', jc)
    L(2, f'N_eff = {Ne}   → {cmp_j}', jc)

    # IAT
    IAT = d['IAT']; Td = d['T_drain']; Ta = d['T_alpha']
    iac = GC if IAT > Ta else (DC if IAT > Td else BC)
    cmp_ia = ('> T_α  → full recovery (clean slate)' if IAT > Ta
               else ('> T_drain → queue drained, α residual' if IAT > Td
                     else '< T_drain → queue residual, worst case'))
    L(0, '')
    L(0, 'IAT  (input = QPS,  output = IAT)', '#333', True)
    L(2, f'QPS = {d["QPS"]:.0f}  →  IAT = {IAT:.0f} µs    → {cmp_ia}', iac)
    L(2, f'T_drain={Td:.1f}   T_α={Ta:.0f}   T_cwnd={d["T_cwnd"]:.0f} µs', NC)
    L(2, f'→  Q₀={d["Q_0"]:.1f}  α₀={d["a_0"]:.3f}  cwnd₀={d["cw_0"]:.1f}', iac)

    # ── DECISION TREE ────────────────────────────────────────────────────
    L(0, '')
    L(0, '─── REGIME DECISION TREE ────────────────────────────', NC, True)
    L(0, '')

    maxQ = d['max_Q']; Tb = d['T_bst']
    cwm = d['cw_min']; hL = d['has_lo']

    # Node 1: drops?
    c1 = BC if hL else GC
    L(0, f'max Q ({maxQ:.1f}) > B ({d["Bp"]:.0f}) ?   {"YES" if hL else "NO"}', c1)
    if hL:
        L(2, '→  BAD REGIME', BC, True)
        L(4, d['rsub'], BC)
        L(4, f'BCT inflation: {d["bct"]/d["bct_i"]:.2f}×  (drops + RTO)', BC)
    else:
        # Node 2: Q > K?
        c2 = DC if maxQ > K_PKT else GC
        L(0, f'max Q ({maxQ:.1f}) > K ({K_PKT:.0f}) ?   {"YES" if maxQ>K_PKT else "NO"}', c2)
        if maxQ <= K_PKT:
            L(2, '→  GOOD REGIME', GC, True)
            L(4, d['rsub'], GC)
            L(4, f'Short-flow SFCT inflation: {d["sfct_inf"]:.2f}×', GC)
        else:
            # Node 3: DCTCP blind?
            c3 = BC if Tb < RTT_P else GC
            L(0, f'T_burst ({Tb:.1f}) ≥ RTT_prop ({RTT_P:.0f}) ?   {"YES" if Tb>=RTT_P else "NO"}', c3)
            if Tb < RTT_P:
                L(2, '→  DEGRADED  (DCTCP blind)', DC, True)
                L(4, d['rsub'], DC)
            else:
                # Node 4: cwnd hit floor?
                c4 = DC if cwm <= CW_MIN*1.1 else GC
                L(0, f'cwnd hit floor ({CW_MIN:.0f} pkts) ?   {"YES" if cwm<=CW_MIN*1.1 else "NO"}', c4)
                if cwm <= CW_MIN * 1.1:
                    L(2, '→  DEGRADED  — Case 2', DC, True)
                    L(4, d['rsub'], DC)
                    L(4, f'SS mean Q* = {d["sm_Q"]:.1f} pkts   (vs K = {K_PKT:.0f})', DC)
                    L(4, f'Short-flow SFCT inflation: {d["sfct_inf"]:.2f}×', DC)
                else:
                    # Node 5: T_burst >= T_C?
                    c5 = GC if Tb >= T_C else DC
                    L(0, f'T_burst ({Tb:.1f}) ≥ T_C ({T_C:.0f}) ?   {"YES" if Tb>=T_C else "NO"}', c5)
                    L(2, '→  DEGRADED  — Case 1', DC, True)
                    L(4, d['rsub'], DC)
                    L(4, f'Short-flow SFCT inflation: {d["sfct_inf"]:.2f}×', DC)

    # ── Render lines ──────────────────────────────────────────────────────
    y = 0.98; dy = 0.044
    for (ind, text, col, bold) in rows:
        ax.text(0.01 + ind * 0.020, y, text,
                transform=ax.transAxes, fontsize=7.0,
                family='monospace', color=col, va='top',
                fontweight='bold' if bold else 'normal')
        y -= dy


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    fig = plt.figure(figsize=(21, 13), facecolor='#fafaf8')
    fig.suptitle(
        'DCTCP Regime Simulation  v4   ·   25 Gbps  ·  RTT_prop=20µs  ·  K=65pkts  ·  BDP≈41.7pkts  ·  T_C=320µs',
        fontsize=10.5, color='#333', y=0.99
    )

    gs = gridspec.GridSpec(5, 2, figure=fig,
                           left=0.05, right=0.98, top=0.965, bottom=0.215,
                           hspace=0.42, wspace=0.25,
                           width_ratios=[2.4, 1.2])

    aQ  = fig.add_subplot(gs[0, 0]);  aCW = fig.add_subplot(gs[1, 0])
    aRT = fig.add_subplot(gs[2, 0]);  aA  = fig.add_subplot(gs[3, 0])
    aSQ = fig.add_subplot(gs[4, 0])
    aInf = fig.add_subplot(gs[0:2, 1])
    aDT  = fig.add_subplot(gs[2:,  1])
    aInf.set_facecolor('#f5f4f0');  aDT.set_facecolor('#f5f4f0')

    bg = '#ece9e2'

    # 6 sliders in 2 rows of 3
    sl_def = [
        # label               vmin  vmax   vinit  vstep  [left, bot, w, h]
        ('N  (flows)',         1,    200,   5,     1,     [0.05, 0.170, 0.23, 0.018]),
        ('S  (KB)',            10,   5000,  500,   10,    [0.34, 0.170, 0.23, 0.018]),
        ('cwnd₀  (pkts)',      1,    200,   10,    1,     [0.63, 0.170, 0.23, 0.018]),
        ('B  (buf pkts)',      100,  5000,  500,   50,    [0.05, 0.143, 0.23, 0.018]),
        ('J  jitter (µs)',     0,    500,   0,     5,     [0.34, 0.143, 0.23, 0.018]),
        ('QPS',                10,   50000, 1000,  10,    [0.63, 0.143, 0.23, 0.018]),
    ]
    sliders = []
    for lbl, mn, mx, vi, vs, rect in sl_def:
        axs = fig.add_axes(rect, facecolor=bg)
        sl  = Slider(axs, lbl, mn, mx, valinit=vi, valstep=vs)
        sl.label.set_fontsize(8); sl.valtext.set_fontsize(8)
        sliders.append(sl)
    sN, sS, sCW, sB, sJ, sQPS = sliders

    ax_btn = fig.add_axes([0.89, 0.149, 0.065, 0.032])
    btn = Button(ax_btn, 'Reset', color=bg, hovercolor='#d5d0c8')
    btn.label.set_fontsize(8)

    plot_axs = (aQ, aCW, aRT, aA, aSQ)

    def refresh(val=None):
        d = simulate(
            int(sN.val), float(sS.val), int(sCW.val),
            float(sB.val), float(sJ.val), float(sQPS.val)
        )
        draw_plots(plot_axs, d)
        draw_info(aInf, d)
        draw_dims_tree(aDT, d)
        fig.canvas.draw_idle()

    def reset(ev):
        for sl in sliders: sl.reset()

    for sl in sliders: sl.on_changed(refresh)
    btn.on_clicked(reset)

    eq = (
        r'$Q_{k+1}=\max(0,\,inj_k - BDP)$   '
        r'$inj_k = N_{old}\!\cdot\!cwnd_k + N_{new}\!\cdot\!cwnd_{init}$   '
        r'$F_k=\max(0,\,Q_{k+1}-K)/inj_k$   '
        r'$\alpha_{k+1}=(1-g)\alpha_k+gF_k$   '
        r'$IAT=10^6/QPS\;\mu s$   '
        r'$Q_0=\max(0,Q_e-C\!\cdot\!IAT)$   '
        r'$\alpha_0=\alpha_e(1-g)^{IAT/RTT}$   '
        r'$cw_0=\min(cwnd_{init},\,cw_e+IAT/RTT)$'
    )
    fig.text(0.02, 0.004, eq, fontsize=6.5, color='#aaa', va='bottom', style='italic')

    refresh()
    plt.show()


if __name__ == '__main__':
    main()