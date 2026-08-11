
from __future__ import annotations

SEED = 20260810

# ---------------------------------------------------------------------------
# Horizon and sampling
# ---------------------------------------------------------------------------
START_DATE = "2026-01-01"
N_DAYS = 180
SLOTS = [("06:00", 6), ("14:00", 14), ("22:00", 22)]   # 3 readings per day
HOURS_PER_SLOT = 8.0

PRODUCT = "HSD"
RHO15_NOMINAL = 832.0            # kg/m3 at 15 degC, BS-VI HSD
RHO15_BATCH_SD = 2.5             # batch-to-batch density variation

# ---------------------------------------------------------------------------
# 1. Tank farm -- 5 HSD tanks, fixed roof
# ---------------------------------------------------------------------------
# vent_area_m2   total open vent area (measurable with a tape)
# pv_setting_kpa pressure-vacuum valve setting; 0.0 = free/open vent, which
#                breathes on every diurnal cycle
TANKS = [
    # id     cap_kl  dia_m  height_m  opening_kl  vent_area_m2  pv_kpa  note
    dict(tank_id="TK-01", capacity_kl=4000, diameter_m=20.0, height_m=13.0,
         opening_kl=2600, vent_area_m2=0.09, pv_setting_kpa=2.50, vent_note="PV valve, set 2.5 kPa"),
    dict(tank_id="TK-02", capacity_kl=4000, diameter_m=20.0, height_m=13.0,
         opening_kl=3100, vent_area_m2=0.09, pv_setting_kpa=2.50, vent_note="PV valve, set 2.5 kPa"),
    dict(tank_id="TK-03", capacity_kl=3000, diameter_m=18.0, height_m=12.0,
         opening_kl=1800, vent_area_m2=0.28, pv_setting_kpa=0.00, vent_note="FREE VENT - PV valve missing/stuck open"),
    dict(tank_id="TK-04", capacity_kl=4000, diameter_m=20.0, height_m=13.0,
         opening_kl=2400, vent_area_m2=0.09, pv_setting_kpa=2.50, vent_note="PV valve, set 2.5 kPa"),
    dict(tank_id="TK-05", capacity_kl=5000, diameter_m=22.0, height_m=13.5,
         opening_kl=3300, vent_area_m2=0.11, pv_setting_kpa=2.50, vent_note="PV valve, set 2.5 kPa"),
]

TANK_FILL_SATURATION = 0.60      # submerged fill
BAY_LOAD_SATURATION = 0.50
VRU_EFFICIENCY = 0.0             # no vapour recovery on HSD bays
TVP_REF_KPA = 0.40               # HSD true vapour pressure at 30 degC
M_VAPOUR = 130.0                 # kg/kmol, middle-distillate vapour

# ---------------------------------------------------------------------------
# 2. Loading bays -- 10 meters, 2 of them drifting
# ---------------------------------------------------------------------------
N_BAYS = 10
BAY_IDS = [f"LB-{i:02d}" for i in range(1, N_BAYS + 1)]

DISPATCH_KL_PER_DAY = 700.0      # terminal average
DISPATCH_DAY_SD = 70.0
SLOT_SHARE = [0.34, 0.42, 0.24]  # 06-14 busiest mid-day, quieter night
TRUCK_KL = 12.0

METER_MPE_PCT = 0.50             # legal maximum permissible error, +/- %
PROVING_INTERVAL_DAYS = 90       # meter proving resets the drift to zero
METER_NOISE_PCT = 0.035          # random per-interval meter repeatability


VERIFY_FRACTION = 0.25           # share of loads weighbridge-verified
WEIGHBRIDGE_NOISE_PCT = 0.30     # weighbridge repeatability on a ~10 t load

# --- INJECTED FAULT: meter drift -------------------------------------------
# Signature: a linear ramp that resets at each proving -> sawtooth.

DRIFTING_METERS = {
    "LB-03": dict(drift_pct_per_day=+0.0100, sign="under-registering"),
    "LB-07": dict(drift_pct_per_day=-0.0060, sign="over-registering"),
}

# ---------------------------------------------------------------------------
# 3. Pipeline -- 20 segments, leakage in B, seepage in D
# ---------------------------------------------------------------------------
SEGMENT_IDS = [f"SEG-{c}" for c in "ABCDEFGHIJKLMNOPQRST"]
SEGMENT_LENGTH_KM = 12.0
RECEIPT_KL_PER_DAY = 700.0       # matched to dispatch, so stock is stationary
RECEIPT_DAY_SD = 90.0
SEGMENT_METER_NOISE_PCT = 0.045  # per-segment flow measurement repeatability

# --- INJECTED FAULT: leakage (step) ----------------------------------------
LEAK = dict(segment="SEG-B", start_day=62, rate_kl_per_day=1.20)

# --- INJECTED FAULT: seepage (accelerating) --------------------------------
# rate(t) = coeff * (t - start)^2  -- quadratic growth, positive curvature
SEEPAGE = dict(segment="SEG-D", start_day=45, coeff_kl_per_day=0.00020)

# ---------------------------------------------------------------------------
# 4. Extra faults so all five detectors are exercised
# ---------------------------------------------------------------------------

INCLUDE_TAPPING = True
INCLUDE_SPILL = True

# Tapping: repeated illegal draw-offs -> several spikes in one window.
TAPPING = dict(segment="SEG-K", start_day=118, end_day=142,
               events_per_week=3, kl_per_event=3.2, kl_per_event_sd=0.7)

# Operational spill: isolated, one-off, and logged by the operator.
SPILLS = [dict(day=77, slot=1, bay="LB-05", kl=8.5, cause="hose coupling failure during loading"),
          dict(day=134, slot=0, bay="LB-09", kl=6.5, cause="tank overfill at bay, bund recovered")]

# ---------------------------------------------------------------------------
# 5. Temperature and measurement noise
# ---------------------------------------------------------------------------
AMBIENT_MEAN_C = 28.0
AMBIENT_SEASONAL_AMP_C = 7.0     # annual swing
AMBIENT_DIURNAL_AMP_C = 5.5      # day/night swing
AMBIENT_NOISE_C = 1.1
TANK_THERMAL_LAG = 0.72          # product temperature damping vs ambient
VAPOUR_SPACE_EXTRA_SWING_C = 1.6 # vapour space swings more than the liquid

DIP_NOISE_MM = 3.0               # automatic tank gauge repeatability
THERMOMETER_NOISE_C = 0.25
DENSITY_NOISE_KGM3 = 0.6

# ---------------------------------------------------------------------------
# 6. Detector settings for the twin 
# ---------------------------------------------------------------------------
WINDOW_DAYS = 21                 # sliding window W_t
CUSUM_K_SIGMA = 0.5              # slack k, in sigma of the in-control series
CUSUM_H_SIGMA = 8.0              # alarm threshold h, in sigma (ARL0 tuned for 20 segments x 180 d)
HAMPEL_ETA = 3.5                 # Hampel cutoff on the MAD-scaled deviation
TREND_T_STAT = 3.0               # |t| on the fitted slope to call it real
CURVATURE_T_STAT = 2.5           # |t| on the quadratic coefficient

PATHS = dict(data="data", out="out")
