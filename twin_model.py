
from __future__ import annotations

import os

import numpy as np
import pandas as pd

import config as C
import petroleum as P

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, C.PATHS["data"])
OUT = os.path.join(HERE, C.PATHS["out"])


# ===========================================================================
# Detectors -- each one is the equation from the methodology slide
# ===========================================================================
def fit_linear(y: np.ndarray) -> tuple[float, float]:
   
    n = len(y)
    if n < 4:
        return 0.0, 0.0
    t = np.arange(n, dtype=float)
    X = np.column_stack([t, np.ones(n)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - 2
    s2 = resid @ resid / dof if dof > 0 else 0.0
    var_a = s2 * np.linalg.inv(X.T @ X)[0, 0]
    se = np.sqrt(var_a) if var_a > 0 else np.inf
    return float(beta[0]), float(beta[0] / se) if np.isfinite(se) and se > 0 else 0.0


def fit_quadratic(y: np.ndarray) -> tuple[float, float]:
    
    n = len(y)
    if n < 6:
        return 0.0, 0.0
    t = np.arange(n, dtype=float)
    X = np.column_stack([t ** 2, t, np.ones(n)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - 3
    s2 = resid @ resid / dof if dof > 0 else 0.0
    var_c = s2 * np.linalg.inv(X.T @ X)[0, 0]
    se = np.sqrt(var_c) if var_c > 0 else np.inf
    return float(beta[0]), float(beta[0] / se) if np.isfinite(se) and se > 0 else 0.0


def cusum(y: np.ndarray, mu0: float, sigma: float,
          k_sigma: float = C.CUSUM_K_SIGMA,
          h_sigma: float = C.CUSUM_H_SIGMA) -> tuple[np.ndarray, int | None]:
   
    k = k_sigma * sigma
    h = h_sigma * sigma
    S = np.zeros(len(y))
    alarm = None
    for i, v in enumerate(y):
        prev = S[i - 1] if i else 0.0
        S[i] = max(0.0, prev + (v - mu0) - k)
        if alarm is None and S[i] > h:
            alarm = i
    return S, alarm


def hampel(y: np.ndarray, eta: float = C.HAMPEL_ETA) -> np.ndarray:
    
    med = np.median(y)
    mad = 1.4826 * np.median(np.abs(y - med))
    if mad <= 0:
        return np.zeros(len(y), dtype=bool)
    return np.abs(y - med) / mad > eta


# ===========================================================================
# Stage 0 -- ingest and rebuild the balance
# ===========================================================================
def load() -> dict:
    names = ["terminal_assets", "operations_log", "tank_dips",
             "bay_meter_readings", "truck_load_verification",
             "pipeline_receipts", "pipeline_segments",
             "evaporation_log", "terminal_balance"]
    return {n: pd.read_csv(os.path.join(DATA, f"{n}.csv")) for n in names}


def rebuild_balance(d: dict) -> pd.DataFrame:
    
    dips = d["tank_dips"].copy()
    # the twin recomputes VCF itself from dip temperature and density
    dips["vcf_twin"] = [P.vcf(t, r) for t, r in
                        zip(dips.temp_c, dips.density_15c)]
    dips["net15"] = dips.observed_volume_kl * dips.vcf_twin

    # A day's opening stock is the PREVIOUS day's closing dip -- not that day's
    # first dip, which already contains the first shift's movements.  Getting
    # this wrong leaves one shift of receipts and deliveries double-counted and
    # inflates the daily scatter by an order of magnitude.
    last_slot = dips.slot.max()
    closing = (dips[dips.slot == last_slot].groupby("day").net15.sum()
               .rename("closing_net_kl15"))
    close_obs = (dips[dips.slot == last_slot].groupby("day").observed_volume_kl.sum()
                 .rename("closing_obs_kl"))
    opening = closing.shift(1).rename("opening_net_kl15")
    open_obs = close_obs.shift(1).rename("opening_obs_kl")

    rec = d["pipeline_receipts"].groupby("day").dispatched_kl15.sum().rename(
        "receipts_net_kl15")
    mtr = d["bay_meter_readings"]
    dlv = mtr.groupby("day").interval_net_kl15.sum().rename("deliveries_net_kl15")
    dlv_obs = mtr.groupby("day").interval_gross_kl.sum().rename("deliveries_obs_kl")
    rec_obs = (d["pipeline_receipts"].assign(
        obs=lambda x: x.dispatched_kl15 / [P.vcf(t, r) for t, r in
                                           zip(x.temp_c, x.density_15c)])
        .groupby("day").obs.sum().rename("receipts_obs_kl"))

    b = pd.concat([opening, rec, dlv, closing,
                   open_obs, rec_obs, dlv_obs, close_obs], axis=1).reset_index()
    b = b.dropna(subset=["opening_net_kl15"]).reset_index(drop=True)  # day 0 has no prior close
    b["unaccounted_net_kl15"] = (b.opening_net_kl15 + b.receipts_net_kl15
                                 - b.deliveries_net_kl15 - b.closing_net_kl15)
    b["unaccounted_raw_kl"] = (b.opening_obs_kl + b.receipts_obs_kl
                               - b.deliveries_obs_kl - b.closing_obs_kl)
    b["throughput_kl15"] = b.deliveries_net_kl15
    b["date"] = pd.Timestamp(C.START_DATE) + pd.to_timedelta(b.day, unit="D")
    return b


def model_evaporation(d: dict) -> pd.DataFrame:
    
    ev = d["evaporation_log"]
    rows = []
    for tid, g in ev.groupby("tank_id"):
        standing = 0.0
        for _, r in g.iterrows():
            standing += P.standing_loss_kl(
                r.vapour_space_m3, r.ullage_m, r.liquid_temp_c,
                r.diurnal_swing_c, r.vent_area_m2, r.pv_setting_kpa,
                C.RHO15_NOMINAL, hours=C.HOURS_PER_SLOT,
                tvp_ref_kpa=C.TVP_REF_KPA, m_vapour=C.M_VAPOUR)
        working = g.working_loss_kl.sum()
        rows.append(dict(tank_id=tid, standing_kl=standing, working_kl=working,
                         total_kl=standing + working,
                         vent_area_m2=g.vent_area_m2.iloc[0],
                         pv_setting_kpa=g.pv_setting_kpa.iloc[0]))
    # loading-bay vapour displacement, from bay meter throughput
    mtr = d["bay_meter_readings"]
    dips = d["tank_dips"].groupby(["day", "slot"]).temp_c.mean()
    bay = 0.0
    for (day, slot), g in mtr.groupby(["day", "slot"]):
        t = dips.get((day, slot), C.AMBIENT_MEAN_C)
        bay += P.loading_loss_kl(g.interval_net_kl15.sum(), t, C.RHO15_NOMINAL,
                                 C.BAY_LOAD_SATURATION, C.VRU_EFFICIENCY,
                                 C.TVP_REF_KPA, C.M_VAPOUR)
    rows.append(dict(tank_id="loading bays", standing_kl=0.0, working_kl=bay,
                     total_kl=bay, vent_area_m2=np.nan, pv_setting_kpa=np.nan))
    return pd.DataFrame(rows)


# ===========================================================================
# Stage 1 -- pipeline segments
# ===========================================================================
def analyse_segments(d: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    seg = d["pipeline_segments"].copy()
    seg["loss"] = seg.inlet_kl15 - seg.outlet_kl15
    daily = seg.groupby(["segment_id", "day"]).loss.sum().reset_index()

    findings, series = [], []
    for sid, g in daily.groupby("segment_id"):
        y = g.sort_values("day").loss.to_numpy()
        days = g.sort_values("day").day.to_numpy()
        series.append(pd.DataFrame(dict(segment_id=sid, day=days, loss=y)))

        # baseline from the first 30 days, robustly
        base = y[:30]
        mu0 = float(np.median(base))
        sigma = float(1.4826 * np.median(np.abs(base - mu0))) or float(np.std(base))
        sigma = max(sigma, 1e-6)

        # -- 1  TREND ------------------------------------------------------
        slope, t_slope = fit_linear(y)
        curv, t_curv = fit_quadratic(y)
        # -- 2  CHANGE -----------------------------------------------------
        S, alarm = cusum(y, mu0, sigma)
       
        material, t_step = False, 0.0
        if alarm is not None and alarm < len(y) - 5:
            post = y[alarm:]
            se_diff = sigma * np.sqrt(1.0 / len(post) + 1.0 / max(len(base), 1))
            t_step = (post.mean() - mu0) / se_diff if se_diff > 0 else 0.0
            material = t_step > 3.0
        # -- 3  SPIKE ------------------------------------------------------
        spikes = hampel(y)
      
        spikes = spikes & (y > np.median(y))
        n_peak = int(spikes.sum())

        total = float(y.sum() - mu0 * len(y))
        verdict, onset, conf = "no finding", None, ""
      
        material_total = total > 4.0 * sigma * np.sqrt(len(y))

        # attribution, in the order the shapes are separable
        if n_peak >= 2 and material_total and not (curv > 0 and t_curv >= C.CURVATURE_T_STAT):
            verdict = "tapping"
            onset = int(days[np.argmax(spikes)])
            conf = f"{n_peak} isolated spikes, no sustained step"
        elif n_peak == 1 and material_total and not material:
            verdict = "spill"
            onset = int(days[np.argmax(spikes)])
            conf = "single isolated spike"
        elif curv > 0 and t_curv >= C.CURVATURE_T_STAT and material:
            verdict = "seepage"
            # for L(t) = c(t-t0)^2 the fitted parabola's vertex recovers the
            # onset: t0 = -a / 2c
            a_lin, _ = fit_linear(y)
            vertex = -np.polyfit(np.arange(len(y)), y, 2)[1] / (2 * curv)
            onset = int(np.clip(vertex, 0, len(y) - 1))
            conf = f"curvature c={curv:.2e}, t={t_curv:.1f}"
        elif material and t_slope >= C.TREND_T_STAT:
            verdict = "leakage"
            onset = int(days[alarm])
            conf = f"CUSUM alarm d{days[alarm]}, step t={t_step:.1f}"
        elif material:
            verdict = "leakage"
            onset = int(days[alarm])
            conf = f"CUSUM alarm d{days[alarm]}, step t={t_step:.1f}"
        elif alarm is not None:
            conf = f"CUSUM alarm d{days[alarm]} not material (t={t_step:.1f})"

        findings.append(dict(
            segment_id=sid, verdict=verdict, onset_day=onset,
            total_loss_kl=total, mean_kl_per_day=total / len(y),
            slope=slope, t_slope=t_slope, curvature=curv, t_curvature=t_curv,
            cusum_alarm_day=int(days[alarm]) if alarm is not None else None,
            step_t_stat=t_step, n_peaks=n_peak, evidence=conf,
            baseline_sigma=sigma))

    return (pd.DataFrame(findings).sort_values("total_loss_kl", ascending=False),
            pd.concat(series, ignore_index=True))


# ===========================================================================
# Stage 2 -- loading bay meters
# ===========================================================================
def fit_linear_xy(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Least-squares line on explicit x. Returns (slope, intercept, t of slope)."""
    n = len(y)
    if n < 4:
        return 0.0, 0.0, 0.0
    X = np.column_stack([x, np.ones(n)])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = n - 2
    s2 = resid @ resid / dof if dof > 0 else 0.0
    var = s2 * np.linalg.inv(X.T @ X)[0, 0]
    se = np.sqrt(var) if var > 0 else np.inf
    t = float(beta[0] / se) if np.isfinite(se) and se > 0 else 0.0
    return float(beta[0]), float(beta[1]), t


def analyse_meters(d: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    
    v = d["truck_load_verification"].copy()
    v["error_pct"] = 100.0 * (v.weighbridge_kl15 - v.meter_kl15) / v.meter_kl15

    mtr = d["bay_meter_readings"]
    vol = mtr.groupby(["meter_id", "day"]).interval_net_kl15.sum()

    rows, traj = [], []
    for bay, g in v.groupby("meter_id"):
        daily = g.groupby("days_since_proving").error_pct.mean()
        slope, icept, t_stat = fit_linear_xy(daily.index.to_numpy(float),
                                             daily.to_numpy())
        peak_pct = slope * C.PROVING_INTERVAL_DAYS + icept

        # kL attributable: the modelled fractional error applied to the volume
        # this meter actually passed on each day
        vb = vol.loc[bay]
        dsp = np.array([day % C.PROVING_INTERVAL_DAYS for day in vb.index], float)
        err_frac = (slope * dsp + icept) / 100.0
        attributed = float((err_frac * vb.to_numpy()).sum())

        drifting = abs(t_stat) >= C.TREND_T_STAT
        rows.append(dict(
            meter_id=bay, drift_pct_per_day=slope, intercept_pct=icept,
            peak_error_pct=peak_pct, t_stat=t_stat,
            n_verified_loads=len(g),
            attributed_kl=attributed if drifting else 0.0,
            exceeds_mpe=abs(peak_pct) > C.METER_MPE_PCT,
            verdict="meter drift" if drifting else "within noise",
            direction="under-registering" if slope > 0 else "over-registering",
            days_to_mpe=(int((C.METER_MPE_PCT - abs(icept)) / abs(slope))
                         if drifting and abs(slope) > 1e-9 else None)))
        traj.append(pd.DataFrame(dict(
            meter_id=bay, days_since_proving=daily.index.to_numpy(),
            observed_error_pct=daily.to_numpy(),
            fitted_error_pct=slope * daily.index.to_numpy() + icept)))

    meters = pd.DataFrame(rows).sort_values("attributed_kl", ascending=False)
    return meters, pd.concat(traj, ignore_index=True)


# ===========================================================================
# Stage 3 -- terminal-level spikes (bay spills) + operational verification
# ===========================================================================
def analyse_terminal_spikes(balance: pd.DataFrame, pipeline_daily: pd.DataFrame,
                            ops: pd.DataFrame) -> pd.DataFrame:
   
    pl = pipeline_daily.groupby("day").loss.sum().reindex(balance.day).fillna(0.0)
    r = balance.set_index("day").unaccounted_net_kl15 - pl.to_numpy()
    y = r.to_numpy()
    med = np.median(y)
    mad = 1.4826 * np.median(np.abs(y - med))
    spikes = hampel(y) & (y > med)

    spill_log = ops[ops.entry_type == "SPILL REPORTED"]
    logged = {r.day: (r.quantity_kl, r.asset_id) for r in spill_log.itertuples()}

    rows = []
    # 1. logged spills: quantity comes from the spill report, and the twin
    #    corroborates it against the balance rather than taking it on trust
    for day, (qty, asset) in logged.items():
        idx = balance.index[balance.day == day]
        excess = float(y[idx[0]] - med) if len(idx) else np.nan
        sigmas = excess / mad if mad > 0 else np.nan
        rows.append(dict(
            day=int(day), asset_id=asset, excess_kl=float(qty), verdict="spill",
            operations_log="SPILL REPORTED",
            note=f"logged {qty:.1f} kL; balance {excess:+.1f} kL ({sigmas:+.1f}σ)"))
    # 2. spikes with no operational explanation -- the ones worth investigating
    for idx in np.flatnonzero(spikes):
        day = int(balance.day.iloc[idx])
        if day in logged:
            continue
        rows.append(dict(
            day=day, asset_id="terminal", excess_kl=float(y[idx] - med),
            verdict="unexplained spike", operations_log="no entry",
            note="no operational explanation - investigate"))
    cols = ["day", "asset_id", "excess_kl", "verdict", "operations_log", "note"]
    return pd.DataFrame(rows, columns=cols).sort_values("day").reset_index(drop=True)


# ===========================================================================
# Attribution
# ===========================================================================
def attribute(balance: pd.DataFrame, evap: pd.DataFrame, segs: pd.DataFrame,
              meters: pd.DataFrame, spikes: pd.DataFrame) -> pd.DataFrame:
    raw = float(balance.unaccounted_raw_kl.sum())
    corrected = float(balance.unaccounted_net_kl15.sum())
    evap_kl = float(evap.total_kl.sum())

    by_verdict = segs.groupby("verdict").total_loss_kl.sum()
    leak = float(by_verdict.get("leakage", 0.0))
    seep = float(by_verdict.get("seepage", 0.0))
    tap = float(by_verdict.get("tapping", 0.0))
    seg_spill = float(by_verdict.get("spill", 0.0))
    spill = float(spikes.loc[spikes.verdict == "spill", "excess_kl"].sum()) + seg_spill

    drift = float(meters.loc[meters.verdict == "meter drift", "attributed_kl"].sum())
    explained = evap_kl + leak + seep + tap + spill + drift

    return pd.DataFrame([
        dict(step="Raw volumetric unaccounted (no VCF)", kl=raw, kind="start"),
        dict(step="Temperature correction (VCF, ASTM D1250)", kl=corrected - raw, kind="correction"),
        dict(step="Evaporation (API 19.1, measured params)", kl=-evap_kl, kind="loss"),
        dict(step="Meter drift (apparent / metrology)", kl=-drift, kind="loss"),
        dict(step="Leakage (pipeline segment)", kl=-leak, kind="loss"),
        dict(step="Seepage (pipeline segment)", kl=-seep, kind="loss"),
        dict(step="Tapping (pipeline segment)", kl=-tap, kind="loss"),
        dict(step="Operational spill (bay)", kl=-spill, kind="loss"),
        dict(step="Residual unexplained", kl=corrected - explained, kind="end"),
    ])


def score(attr: pd.DataFrame, segs: pd.DataFrame, meters: pd.DataFrame,
          spikes: pd.DataFrame | None = None,
          ops: pd.DataFrame | None = None) -> pd.DataFrame:
    """Compare the twin's attribution against the injected truth."""
    truth = pd.read_csv(os.path.join(DATA, "ground_truth_losses.csv"))
    t = truth.groupby("mechanism").kl15.sum()

    a = {r.step: -r.kl for r in attr.itertuples()}
    got = {
        "evaporation": a["Evaporation (API 19.1, measured params)"],
        "meter drift": a["Meter drift (apparent / metrology)"],
        "leakage": a["Leakage (pipeline segment)"],
        "seepage": a["Seepage (pipeline segment)"],
        "tapping": a["Tapping (pipeline segment)"],
        "spill": a["Operational spill (bay)"],
    }
    # localisation: did the twin name the right asset?
    loc_truth = {
        "leakage": C.LEAK["segment"], "seepage": C.SEEPAGE["segment"],
        "tapping": C.TAPPING["segment"] if C.INCLUDE_TAPPING else None}
    loc_found = {v: (segs.loc[segs.verdict == v, "segment_id"].tolist())
                 for v in ("leakage", "seepage", "tapping")}
    loc_found["meter drift"] = meters.loc[meters.verdict == "meter drift",
                                          "meter_id"].tolist()
    if ops is not None and spikes is not None and not spikes.empty:
        sl = ops[ops.entry_type == "SPILL REPORTED"]
        days = set(spikes.loc[spikes.verdict == "spill", "day"])
        loc_found["spill"] = sl.loc[sl.day.isin(days), "asset_id"].tolist()
        loc_truth["spill"] = ", ".join(s["bay"] for s in C.SPILLS) if C.INCLUDE_SPILL else None

    rows = []
    for mech, actual in t.items():
        est = got.get(mech, 0.0)
        rows.append(dict(
            mechanism=mech, truth_kl=actual, detected_kl=est,
            error_kl=est - actual,
            error_pct=(100.0 * (est - actual) / actual) if actual else np.nan,
            truth_location=loc_truth.get(mech, "-"),
            detected_location=", ".join(loc_found.get(mech, [])) or "-"))
    return pd.DataFrame(rows).sort_values("truth_kl", ascending=False)


# ===========================================================================
# Orchestration
# ===========================================================================
def run(verbose: bool = True) -> dict:
    os.makedirs(OUT, exist_ok=True)
    d = load()
    balance = rebuild_balance(d)
    evap = model_evaporation(d)
    segs, seg_series = analyse_segments(d)
    meters, meter_traj = analyse_meters(d)
    spikes = analyse_terminal_spikes(balance, seg_series, d["operations_log"])
    attr = attribute(balance, evap, segs, meters, spikes)
    sc = score(attr, segs, meters, spikes, d["operations_log"])

    res = dict(balance=balance, evaporation=evap, segments=segs,
               segment_series=seg_series, meters=meters,
               meter_trajectory=meter_traj, spikes=spikes,
               attribution=attr, scorecard=sc)
    for name, df in res.items():
        df.to_csv(os.path.join(OUT, f"twin_{name}.csv"), index=False)

    if verbose:
        report(res)
    return res


def report(res: dict) -> None:
    b, attr, segs, meters = (res["balance"], res["attribution"],
                             res["segments"], res["meters"])
    W = 78
    print("=" * W)
    print("LOSS-CONTROL DIGITAL TWIN  --  ATTRIBUTION REPORT")
    print("=" * W)
    print(f"  period                 {b.day.min()}-{b.day.max()} "
          f"({len(b)} days, {len(C.SLOTS)} readings/day)")
    print(f"  throughput             {b.throughput_kl15.sum():>12,.0f} kL @15 degC")
    print(f"  raw unaccounted        {b.unaccounted_raw_kl.sum():>12,.1f} kL   "
          f"(daily sd {b.unaccounted_raw_kl.std():.1f})")
    print(f"  VCF-corrected          {b.unaccounted_net_kl15.sum():>12,.1f} kL   "
          f"(daily sd {b.unaccounted_net_kl15.std():.1f})")
    print(f"                         {'':>12} "
          f"{100*b.unaccounted_net_kl15.sum()/b.throughput_kl15.sum():.3f}% of throughput")

    print("-" * W)
    print("  ATTRIBUTION")
    for r in attr.itertuples():
        print(f"    {r.step:<48} {r.kl:>12,.2f} kL")

    print("-" * W)
    print("  PIPELINE SEGMENTS WITH A FINDING")
    hit = segs[segs.verdict != "no finding"]
    if hit.empty:
        print("    none")
    for r in hit.itertuples():
        onset = f"day {r.onset_day}" if r.onset_day is not None else "-"
        print(f"    {r.segment_id:<8} {r.verdict:<10} {r.total_loss_kl:>9,.1f} kL   "
              f"onset {onset:<9} {r.evidence}")
    clean = len(segs) - len(hit)
    print(f"    ... {clean} of {len(segs)} segments clean (no false positives)"
          if clean else "")

    print("-" * W)
    print("  LOADING BAY METERS")
    for r in meters.itertuples():
        flag = "  << EXCEEDS MPE" if r.exceeds_mpe else ""
        if r.verdict == "meter drift":
            print(f"    {r.meter_id:<8} {r.verdict:<14} peak error "
                  f"{r.peak_error_pct:>+6.2f}%  t={r.t_stat:>5.1f}  "
                  f"{r.attributed_kl:>8,.1f} kL  {r.direction}{flag}")
    q = meters[meters.verdict == "within noise"]
    print(f"    ... {len(q)} of {len(meters)} meters within noise")

    if not res["spikes"].empty:
        print("-" * W)
        print("  ISOLATED EVENTS (operational verification)")
        for r in res["spikes"].itertuples():
            print(f"    day {r.day:<5} {r.excess_kl:>7,.1f} kL  {r.verdict:<18} "
                  f"{r.operations_log:<16} {r.note}")

    print("-" * W)
    print("  SCORECARD  vs injected ground truth (kL @15 degC)")
    print(f"    {'mechanism':<14}{'truth':>10}{'detected':>11}{'error':>9}"
          f"{'err %':>8}   {'truth loc':<8} {'found loc'}")
    for r in res["scorecard"].itertuples():
        ep = f"{r.error_pct:>7.1f}%" if pd.notna(r.error_pct) else "      -"
        print(f"    {r.mechanism:<14}{r.truth_kl:>10,.1f}{r.detected_kl:>11,.1f}"
              f"{r.error_kl:>9,.1f}{ep}   {str(r.truth_location):<8} "
              f"{r.detected_location}")
    print("=" * W)


if __name__ == "__main__":
    run()
