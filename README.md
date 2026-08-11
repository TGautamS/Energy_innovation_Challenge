# Loss-control digital twin — model simulation

A demonstration on synthetic data: a hypothetical HSD terminal fed by a
cross-country pipeline, with faults injected on purpose, and the twin recovering
them from the instrument records alone.

```bash
pip install numpy pandas matplotlib
python dashboard.py      # runs the twin on the data and renders the dashboards
```

`python twin_model.py` prints the attribution report on its own. The analysis is
deterministic, so runs reproduce.

---

## Files

- **`config.py`** — the whole scenario in one place: tanks, bay meters, pipeline
  segments, the injected faults, the temperature and noise levels, and the
  detector thresholds.
- **`petroleum.py`** — the petroleum physics: temperature volume correction
  (API MPMS 11.1 / ASTM D1250) and evaporation (API MPMS 19.1).
- **`twin_model.py`** — the digital twin: reads the instrument CSVs, rebuilds the
  temperature-corrected balance, and runs the trend / change / spike detectors to
  attribute and localise each loss.
- **`dashboard.py`** — renders the twin's results as two dashboard screens
  (overview and diagnostics).

---

## Synthetic data

The twin runs on a set of synthetic CSVs in `terminal_data(synthetic)/` that
stand in for a terminal's own instrument records over 180 days — tank dips
(level, temperature and density, three times a day), loading-bay meter readings,
pipeline-segment flows and pressures, product receipts, an operations log, and
the daily stock balance. A separate ground-truth file lists the injected losses
and is read only to score the twin, never during detection.

---

## The scenario

Declared in `config.py`. The generator simulates the *physical truth* and then
the *observations*, so the gap between them is made of exactly these
mechanisms and nothing else.

| Asset | Configuration |
|---|---|
| Tank farm | 5 fixed-roof HSD tanks, 3,000–5,000 kL, with vent area and PV-valve setting per tank |
| Loading bays | 10 bay meters, ±0.5 % legal MPE, proved every 90 days |
| Pipeline | 20 segments (SEG-A … SEG-T), 12 km each, inlet/outlet flow and pressure |
| Horizon | 180 days × 3 readings/day (06:00, 14:00, 22:00) ≈ 124,000 kL throughput |

**Injected faults**

| Mechanism | Where | Shape | Size |
|---|---|---|---|
| Meter drift | **LB-03** under-registering | linear ramp, resets at proving (sawtooth) | +0.010 %/day → +0.90 % at proving |
| Meter drift | **LB-07** over-registering | same, opposite sign | −0.006 %/day → −0.54 % |
| Leakage | **SEG-B** | step from day 62 | 1.20 kL/day |
| Seepage | **SEG-D** | quadratic growth from day 45 | 0.0002·(t−45)² kL/day |
| Tapping | **SEG-K** | repeated spikes, days 118–142 | ~3.2 kL/event |
| Spill | **LB-05**, **LB-09** | isolated, logged events | 8.5 and 6.5 kL |
| Evaporation | all tanks + bays | standing, working, loading displacement | computed, not assumed |
| Temperature | everywhere | seasonal + diurnal | drives the VCF |

The two drifting meters point in **opposite directions on purpose**. In the
terminal aggregate they partly cancel (+55.8 and −34.6 kL net to +21.2 kL),
which is exactly why loss has to be attributed per asset rather than read off
the total.

`INCLUDE_TAPPING` and `INCLUDE_SPILL` default to `True` so all five detectors
are exercised; the brief specified only drift, leakage and seepage. Set either
to `False` and the corresponding detector correctly reports nothing.
