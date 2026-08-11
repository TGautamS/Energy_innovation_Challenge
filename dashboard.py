
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

import config as C
import twin_model as T

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, C.PATHS["out"])

# deck palette, so the dashboard drops straight into the presentation
INK, STEEL, AMBER = "#0E2433", "#2E5266", "#F2A104"
EMBER, MINT, SEEP = "#C1442E", "#2F9E7E", "#9C6B3F"
MIST, MUTED, PAPER = "#EBEEF1", "#7A8C99", "#FFFFFF"
PANEL = "#F5F7F8"

MECH_COLOUR = {"meter drift": AMBER, "leakage": EMBER, "seepage": SEEP,
               "tapping": INK, "spill": STEEL, "evaporation": MINT,
               "temperature": "#8FB8CE"}

plt.rcParams.update({
    "font.family": "DejaVu Sans", "axes.edgecolor": "#C9D2D8",
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.titlesize": 11, "axes.titleweight": "bold",
    "figure.facecolor": PAPER, "axes.facecolor": PAPER,
})


def _panel(ax, title, sub=""):
    ax.set_facecolor(PAPER)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_title(title, loc="left", color=INK, pad=14 if sub else 8)
    if sub:
        ax.text(0.0, 1.015, sub, transform=ax.transAxes, fontsize=8.4,
                color=MUTED, style="italic", va="bottom")
    ax.grid(axis="y", color=MIST, lw=0.8, zorder=0)
    ax.set_axisbelow(True)


def _kpi(fig, x, y, w, h, value, label, colour=INK, sub=""):
    fig.patches.append(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.008",
        transform=fig.transFigure, facecolor=PANEL, edgecolor="#DDE4E8",
        lw=1.0, zorder=1))
    fig.text(x + w / 2, y + h * 0.60, value, ha="center", va="center",
             fontsize=19, fontweight="bold", color=colour, zorder=2)
    fig.text(x + w / 2, y + h * 0.27, label, ha="center", va="center",
             fontsize=8.2, color=MUTED, zorder=2)
    if sub:
        fig.text(x + w / 2, y + h * 0.10, sub, ha="center", va="center",
                 fontsize=7.4, color=MUTED, style="italic", zorder=2)


# ===========================================================================
# Screen 1 -- overview
# ===========================================================================
def overview(res: dict) -> str:
    bal, attr, sc = res["balance"], res["attribution"], res["scorecard"]
    thru = bal.throughput_kl15.sum()
    raw = bal.unaccounted_raw_kl.sum()
    corr = bal.unaccounted_net_kl15.sum()
    resid = float(attr.loc[attr.kind == "end", "kl"].iloc[0])
    explained = 100.0 * (1 - abs(resid) / abs(corr)) if corr else 0.0

    fig = plt.figure(figsize=(16.4, 9.6))
    fig.suptitle("Loss-control digital twin  ·  HSD terminal + feeder pipeline",
                 x=0.035, y=0.977, ha="left", va="top", fontsize=17,
                 fontweight="bold", color=INK)
    fig.text(0.035, 0.933,
             f"Synthetic scenario · {len(bal)} days · {len(C.SLOTS)} readings/day · "
             f"5 tanks, {C.N_BAYS} bay meters, {len(C.SEGMENT_IDS)} pipeline segments",
             ha="left", fontsize=9.5, color=STEEL, style="italic")

    # ---- KPI strip -------------------------------------------------------
    kx, kw, kg = 0.035, 0.176, 0.014
    _kpi(fig, kx + 0 * (kw + kg), 0.845, kw, 0.082, f"{thru:,.0f}", "kL THROUGHPUT @15 °C")
    _kpi(fig, kx + 1 * (kw + kg), 0.845, kw, 0.082, f"{corr:,.0f}", "kL UNACCOUNTED",
         EMBER, f"{100*corr/thru:.3f}% of throughput")
    _kpi(fig, kx + 2 * (kw + kg), 0.845, kw, 0.082, f"{explained:.0f}%", "ATTRIBUTED TO A CAUSE", MINT)
    _kpi(fig, kx + 3 * (kw + kg), 0.845, kw, 0.082, f"{resid:+,.0f}", "kL RESIDUAL", MUTED,
         "unexplained after attribution")
    _kpi(fig, kx + 4 * (kw + kg), 0.845, kw, 0.082,
         f"{bal.unaccounted_raw_kl.std():.1f}→{bal.unaccounted_net_kl15.std():.1f}",
         "kL DAILY NOISE, RAW → VCF", STEEL, "temperature correction")

    gs = fig.add_gridspec(2, 2, left=0.045, right=0.975, top=0.775, bottom=0.105,
                          hspace=0.42, wspace=0.20,
                          height_ratios=[1.15, 1.0])

    # ---- waterfall -------------------------------------------------------
    ax = fig.add_subplot(gs[0, :])
    _panel(ax, "Where the unaccounted figure went",
           "each bar is a cause removed from the raw balance, in kL @15 °C")
    steps = attr.copy()
    labels = [s.replace(" (", "\n(") for s in steps.step]
    run = 0.0
    for i, r in enumerate(steps.itertuples()):
        if r.kind == "start":
            ax.bar(i, r.kl, color=STEEL, zorder=3, width=0.62)
            ax.text(i, r.kl, f"{r.kl:,.0f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=STEEL)
            run = r.kl
        elif r.kind == "end":
            ax.bar(i, r.kl, color=MUTED, zorder=3, width=0.62)
            ax.text(i, r.kl, f"{r.kl:,.0f}", ha="center",
                    va="bottom" if r.kl >= 0 else "top",
                    fontsize=9, fontweight="bold", color=MUTED)
        else:
            key = ("temperature" if "Temperature" in r.step else
                   next((k for k in MECH_COLOUR if k in r.step.lower()), "leakage"))
            col = MECH_COLOUR.get(key, STEEL)
            ax.bar(i, r.kl, bottom=run, color=col, zorder=3, width=0.62)
            ax.plot([i - 0.45, i + 0.45], [run + r.kl] * 2, color="#C9D2D8",
                    lw=0.9, zorder=2)
            ax.text(i, run + r.kl / 2, f"{r.kl:+,.0f}", ha="center", va="center",
                    fontsize=8.6, fontweight="bold",
                    color=PAPER if abs(r.kl) > 25 else col)
            run += r.kl
    ax.axhline(0, color="#C9D2D8", lw=1.0, zorder=1)
    ax.set_xticks(range(len(steps)))
    ax.set_xticklabels(labels, fontsize=7.6, color=STEEL)
    ax.set_ylabel("kL @15 °C", fontsize=9)

    # ---- daily series ----------------------------------------------------
    ax = fig.add_subplot(gs[1, 0])
    _panel(ax, "Daily unaccounted: raw volume vs temperature-corrected",
           "the correction removes the swing that hides a real 1.2 kL/day leak")
    ax.plot(bal.day, bal.unaccounted_raw_kl, color="#B9C7D1", lw=0.9,
            label=f"raw volumetric (sd {bal.unaccounted_raw_kl.std():.1f} kL)", zorder=2)
    ax.plot(bal.day, bal.unaccounted_net_kl15, color=EMBER, lw=1.3,
            label=f"VCF-corrected (sd {bal.unaccounted_net_kl15.std():.1f} kL)", zorder=3)
    ax.axvline(C.LEAK["start_day"], color=INK, lw=1.0, ls="--", zorder=4)
    ax.text(C.LEAK["start_day"] + 2, ax.get_ylim()[1] * 0.86, "leak onset",
            fontsize=7.8, color=INK)
    ax.set_xlabel("day", fontsize=9)
    ax.set_ylabel("kL", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower left")

    # ---- scorecard -------------------------------------------------------
    ax = fig.add_subplot(gs[1, 1])
    _panel(ax, "Recovered vs injected loss, by mechanism",
           "the twin never sees the ground truth until it is scored")
    s = sc.sort_values("truth_kl")
    yy = np.arange(len(s))
    ax.barh(yy + 0.19, s.truth_kl, height=0.36, color="#C9D2D8",
            label="injected truth  (coloured bar = twin detected)", zorder=3)
    ax.barh(yy - 0.19, s.detected_kl, height=0.36,
            color=[MECH_COLOUR.get(m, STEEL) for m in s.mechanism],
            zorder=3)
    for i, r in enumerate(s.itertuples()):
        ax.text(max(r.truth_kl, r.detected_kl) + 4, i,
                f"{r.error_pct:+.1f}%" if pd.notna(r.error_pct) else "",
                va="center", fontsize=8, color=MUTED)
    ax.set_yticks(yy)
    ax.set_yticklabels(s.mechanism, fontsize=9)
    ax.set_xlabel("kL @15 °C", fontsize=9)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.grid(axis="x", color=MIST, lw=0.8, zorder=0)
    ax.grid(axis="y", visible=False)

    fig.text(0.035, 0.028,
             "Synthetic data. Volume correction per API MPMS Ch. 11.1 / ASTM D1250; "
             "evaporation per API MPMS Ch. 19.1. Generated by simulation/generate_data.py.",
             fontsize=7.6, color=MUTED, style="italic")

    path = os.path.join(OUT, "dashboard_overview.png")
    fig.savefig(path, dpi=150, facecolor=PAPER)
    plt.close(fig)
    return path


# ===========================================================================
# Screen 2 -- diagnostics
# ===========================================================================
def diagnostics(res: dict) -> str:
    segs, series = res["segments"], res["segment_series"]
    meters, traj = res["meters"], res["meter_trajectory"]
    evap = res["evaporation"]

    fig = plt.figure(figsize=(16.4, 9.6))
    fig.suptitle("Evidence behind each finding", x=0.035, y=0.977, ha="left",
                 va="top", fontsize=17, fontweight="bold", color=INK)
    fig.text(0.035, 0.933,
             "One detector per signature — every claim traceable to a fitted "
             "statistic and a named asset",
             ha="left", fontsize=9.5, color=STEEL, style="italic")

    gs = fig.add_gridspec(3, 3, left=0.045, right=0.975, top=0.878, bottom=0.075,
                          hspace=0.66, wspace=0.24)

    # ---- 1. meter sawtooth ----------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    _panel(ax, "① Trend · meter error vs days since proving",
           "each proving cycle collapses onto one line; slope = drift rate")
    for m, g in traj.groupby("meter_id"):
        drift = meters.set_index("meter_id").loc[m]
        if drift.verdict == "meter drift":
            col = AMBER if drift.drift_pct_per_day > 0 else STEEL
            ax.scatter(g.days_since_proving, g.observed_error_pct, s=5,
                       color=col, alpha=0.35, zorder=3)
            ax.plot(g.days_since_proving, g.fitted_error_pct, color=col, lw=2.0,
                    zorder=4, label=f"{m}  {drift.peak_error_pct:+.2f}%  t={drift.t_stat:.0f}")
        else:
            ax.scatter(g.days_since_proving, g.observed_error_pct, s=3,
                       color="#CBD5DB", alpha=0.30, zorder=2)
    ax.axhspan(-C.METER_MPE_PCT, C.METER_MPE_PCT, color=MINT, alpha=0.10, zorder=1)
    for s in (+C.METER_MPE_PCT, -C.METER_MPE_PCT):
        ax.axhline(s, color=MINT, lw=1.0, ls="--", zorder=2)
    ax.text(2, C.METER_MPE_PCT + 0.03, "legal MPE ±0.5%", fontsize=7.4, color=MINT)
    ax.set_xlabel("days since proving", fontsize=9)
    ax.set_ylabel("meter error %", fontsize=9)
    ax.legend(frameon=False, fontsize=7.6, loc="lower left")

    # ---- 2. segment map --------------------------------------------------
    ax = fig.add_subplot(gs[0, 1:])
    _panel(ax, "Loss localised to a segment",
           f"total excess over baseline for each of the {len(C.SEGMENT_IDS)} segments")
    s = segs.set_index("segment_id").reindex(C.SEGMENT_IDS)
    cols = [MECH_COLOUR.get(v, "#CBD5DB") if v != "no finding" else "#CBD5DB"
            for v in s.verdict]
    ax.bar(range(len(s)), s.total_loss_kl, color=cols, zorder=3, width=0.68)
    for i, (sid, r) in enumerate(s.iterrows()):
        if r.verdict != "no finding":
            ax.text(i, r.total_loss_kl + 5, f"{r.verdict}\n{r.total_loss_kl:,.0f} kL",
                    ha="center", va="bottom", fontsize=7.8, fontweight="bold",
                    color=MECH_COLOUR.get(r.verdict, INK))
    ax.axhline(0, color="#C9D2D8", lw=1.0, zorder=1)
    ax.set_xticks(range(len(s)))
    ax.set_xticklabels([x.replace("SEG-", "") for x in s.index], fontsize=8)
    ax.set_ylabel("kL @15 °C", fontsize=9)
    ax.set_ylim(top=max(s.total_loss_kl) * 1.35)

    # ---- 3. leakage: CUSUM ----------------------------------------------
    seg_b = C.LEAK["segment"]
    ax = fig.add_subplot(gs[1, 0])
    _panel(ax, f"② Change · CUSUM on {seg_b}",
           "S_t crosses h → step change → leakage")
    g = series[series.segment_id == seg_b].sort_values("day")
    y = g.loss.to_numpy()
    row = segs.set_index("segment_id").loc[seg_b]
    base = y[:30]
    mu0 = float(np.median(base))
    S, alarm = T.cusum(y, mu0, row.baseline_sigma)
    ax.plot(g.day, S, color=EMBER, lw=1.5, zorder=3)
    ax.axhline(C.CUSUM_H_SIGMA * row.baseline_sigma, color=INK, lw=1.0, ls="--", zorder=2)
    ax.text(2, C.CUSUM_H_SIGMA * row.baseline_sigma * 1.06, "alarm h", fontsize=7.6, color=INK)
    if alarm is not None:
        ax.axvline(g.day.iloc[alarm], color=INK, lw=1.0, zorder=4)
        ax.text(g.day.iloc[alarm] + 3, ax.get_ylim()[1] * 0.55,
                f"alarm d{g.day.iloc[alarm]}\ntrue onset d{C.LEAK['start_day']}",
                fontsize=7.6, color=INK)
    ax.set_xlabel("day", fontsize=9)
    ax.set_ylabel("CUSUM  $S_t$", fontsize=9)

    # ---- 4. seepage: quadratic ------------------------------------------
    seg_d = C.SEEPAGE["segment"]
    ax = fig.add_subplot(gs[1, 1])
    _panel(ax, f"① Trend · curvature on {seg_d}",
           "positive, significant c → accelerating growth → seepage")
    g = series[series.segment_id == seg_d].sort_values("day")
    y = g.loss.to_numpy()
    x = np.arange(len(y))
    row = segs.set_index("segment_id").loc[seg_d]
    coef = np.polyfit(x, y, 2)
    ax.plot(g.day, y, color="#CBD5DB", lw=0.8, zorder=2)
    ax.plot(g.day, pd.Series(y).rolling(14, center=True, min_periods=3).mean(),
            color=SEEP, lw=1.3, alpha=0.75, zorder=3, label="14-day mean")
    ax.plot(g.day, np.polyval(coef, x), color=INK, lw=1.9, zorder=4,
            label=f"quadratic fit, t(c)={row.t_curvature:.0f}")
    ax.axvline(C.SEEPAGE["start_day"], color=MUTED, lw=1.0, ls="--", zorder=2)
    ax.text(C.SEEPAGE["start_day"] + 3, ax.get_ylim()[1] * 0.05, "true onset",
            fontsize=7.4, color=MUTED)
    ax.set_xlabel("day", fontsize=9)
    ax.set_ylabel("kL / day", fontsize=9)
    ax.legend(frameon=False, fontsize=7.6, loc="upper left")

    # ---- 5. tapping: spikes ---------------------------------------------
    seg_k = C.TAPPING["segment"]
    ax = fig.add_subplot(gs[1, 2])
    _panel(ax, f"③ Spike · Hampel on {seg_k}",
           "N_peak > 1 → repeated draw-offs → tapping (not a one-off spill)")
    g = series[series.segment_id == seg_k].sort_values("day")
    y = g.loss.to_numpy()
    mask = T.hampel(y) & (y > np.median(y))
    ax.plot(g.day, y, color="#CBD5DB", lw=0.9, zorder=2)
    ax.scatter(g.day[mask], y[mask], s=42, color=INK, zorder=4,
               label=f"{int(mask.sum())} spikes")
    med = np.median(y)
    mad = 1.4826 * np.median(np.abs(y - med))
    ax.axhline(med + C.HAMPEL_ETA * mad, color=INK, lw=1.0, ls="--", zorder=3)
    ax.text(2, med + C.HAMPEL_ETA * mad * 1.02, "Hampel η=3.5", fontsize=7.4, color=INK)
    ax.axvspan(C.TAPPING["start_day"], C.TAPPING["end_day"], color=INK, alpha=0.06, zorder=1)
    ax.set_xlabel("day", fontsize=9)
    ax.set_ylabel("kL / day", fontsize=9)
    ax.legend(frameon=False, fontsize=7.6, loc="upper left")

    # ---- 6. evaporation --------------------------------------------------
    ax = fig.add_subplot(gs[2, 0])
    _panel(ax, "Evaporation, computed not assumed",
           "HSD: 0.003% of throughput — it cannot explain the shortfall")
    e = evap[evap.tank_id != "loading bays"]
    yy = np.arange(len(e))
    ax.barh(yy, e.standing_kl, color=EMBER, zorder=3, label="standing (breathing)")
    ax.barh(yy, e.working_kl, left=e.standing_kl, color=MINT, zorder=3,
            label="working (filling)")
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{t}" for t in e.tank_id], fontsize=8.6)
    for i, r in enumerate(e.itertuples()):
        if r.pv_setting_kpa == 0:
            ax.text(r.total_kl * 1.06, i, "FREE VENT\n250× breathing loss",
                    va="center", fontsize=7.4, fontweight="bold", color=EMBER)
    ax.set_xlabel("kL over the period", fontsize=9)
    ax.set_xlim(0, float(e.total_kl.max()) * 1.55)
    ax.legend(frameon=False, fontsize=7.6, loc="upper right")
    ax.grid(axis="x", color=MIST, lw=0.8, zorder=0)
    ax.grid(axis="y", visible=False)

    # ---- 7. action queue -------------------------------------------------
    ax = fig.add_subplot(gs[2, 1:])
    ax.axis("off")
    ax.set_title("Ranked action queue", loc="left", color=INK, pad=14)
    ax.text(0.0, 1.015, "what the loss-control officer works from, in order of kL",
            transform=ax.transAxes, fontsize=8.4, color=MUTED, style="italic",
            va="bottom")

    q = []
    for r in segs[segs.verdict != "no finding"].itertuples():
        q.append((r.total_loss_kl, r.verdict, r.segment_id,
                  f"onset day {r.onset_day}" if r.onset_day is not None else "",
                  r.evidence[:52]))
    for r in meters[meters.verdict == "meter drift"].itertuples():
        note = "exceeds legal MPE — prove now" if r.exceeds_mpe else "within MPE — monitor"
        q.append((abs(r.attributed_kl), "meter drift", r.meter_id,
                  f"{r.peak_error_pct:+.2f}% at proving", note))
    for r in res["spikes"].itertuples():
        q.append((abs(r.excess_kl), r.verdict, str(r.asset_id),
                  f"day {r.day}", r.note[:52]))
    q.sort(reverse=True, key=lambda z: z[0])

    hdr = ["kL", "cause", "location", "when", "evidence / action"]
    xs = [0.005, 0.085, 0.215, 0.345, 0.500]
    for x, h in zip(xs, hdr):
        ax.text(x, 0.93, h.upper(), transform=ax.transAxes, fontsize=7.6,
                fontweight="bold", color=MUTED)
    ax.plot([0, 1], [0.895, 0.895], transform=ax.transAxes, color="#DDE4E8", lw=1.0)
    for i, (kl, cause, loc, when, ev) in enumerate(q[:8]):
        yv = 0.815 - i * 0.108
        col = MECH_COLOUR.get(cause, MUTED)
        ax.add_patch(plt.Rectangle((0.005, yv - 0.018), 0.010, 0.058, color=col,
                                   transform=ax.transAxes, clip_on=False))
        ax.text(xs[0] + 0.020, yv, f"{kl:,.0f}", transform=ax.transAxes,
                fontsize=9.4, fontweight="bold", color=INK)
        ax.text(xs[1], yv, cause, transform=ax.transAxes, fontsize=9, color=col,
                fontweight="bold")
        ax.text(xs[2], yv, loc, transform=ax.transAxes, fontsize=9, color=INK)
        ax.text(xs[3], yv, when, transform=ax.transAxes, fontsize=8.4, color=STEEL)
        ax.text(xs[4], yv, ev, transform=ax.transAxes, fontsize=8.2, color=STEEL)

    path = os.path.join(OUT, "dashboard_diagnostics.png")
    fig.savefig(path, dpi=150, facecolor=PAPER)
    plt.close(fig)
    return path


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    res = T.run(verbose=False)
    p1, p2 = overview(res), diagnostics(res)
    print(f"wrote {p1}")
    print(f"wrote {p2}")


if __name__ == "__main__":
    main()
