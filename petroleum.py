from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
R_GAS = 8.314          # kPa.m3 / (kmol.K)
P_ATM = 101.325        # kPa, standard atmosphere
KPA_PER_PSI = 6.894757
M_PER_FT = 0.3048
T_BASE = 15.0          # degC, Indian/ISO base temperature for volume correction


# ---------------------------------------------------------------------------
# 1. Thermal volume correction -- API MPMS Ch. 11.1 (2004), ASTM D1250
# ---------------------------------------------------------------------------

#
#            rho_min      rho_max        K0         K1          K2       label
_BANDS = (
    (610.6,    770.3520,  346.4228,  0.4388,      0.0,          "gasolines"),
    (770.3520, 787.5195, 2680.3206,  1.2580,     -0.00336312,   "transition zone"),
    (787.5195, 838.3127,  594.5418,  0.0,         0.0,          "jet fuels / kerosenes / HSD"),
    (838.3127, 1163.5,    186.9696,  0.4862,      0.0,          "fuel oils"),
)


def commodity_band(rho15: float) -> tuple:
    """Return (K0, K1, K2, label) for a density at base temperature."""
    for lo, hi, k0, k1, k2, label in _BANDS:
        if lo <= rho15 < hi:
            return k0, k1, k2, label
    raise ValueError(
        f"density {rho15} kg/m3 outside the API MPMS 11.1 range "
        f"({_BANDS[0][0]}-{_BANDS[-1][1]} kg/m3)")


def alpha_15(rho15: float) -> float:
    """Coefficient of thermal expansion at 15 degC, in degC^-1."""
    k0, k1, k2, _ = commodity_band(rho15)
    return k0 / rho15 ** 2 + k1 / rho15 + k2


def vcf(t_obs_c: float, rho15: float) -> float:
    """
    Volume Correction Factor from observed temperature to 15 degC.

        VCF = exp[ -alpha * dT * (1 + 0.8 * alpha * (dT + 0.01374)) ]

    """
    a = alpha_15(rho15)
    dt = t_obs_c - T_BASE
    return math.exp(-a * dt * (1.0 + 0.8 * a * (dt + 0.01374)))


def to_net15(volume_obs: float, t_obs_c: float, rho15: float) -> float:
    """Observed volume at t_obs -> net (standard) volume at 15 degC."""
    return volume_obs * vcf(t_obs_c, rho15)


def to_observed(volume_net15: float, t_obs_c: float, rho15: float) -> float:
    """Net volume at 15 degC -> the volume a dip/meter would read at t_obs."""
    return volume_net15 / vcf(t_obs_c, rho15)


def density_at(t_obs_c: float, rho15: float) -> float:
    """Observed density at temperature (mass conserved, volume corrected)."""
    return rho15 * vcf(t_obs_c, rho15)


# ---------------------------------------------------------------------------
# 2. Evaporation
# ---------------------------------------------------------------------------

def true_vapour_pressure(t_c: float, tvp_ref_kpa: float = 0.40,
                         t_ref_c: float = 30.0, dh_over_r: float = 4200.0) -> float:
    """
    True vapour pressure at temperature, by Clausius-Clapeyron anchored on a
    measured reference point.
    """
    t_k, t_ref_k = t_c + 273.15, t_ref_c + 273.15
    return tvp_ref_kpa * math.exp(dh_over_r * (1.0 / t_ref_k - 1.0 / t_k))


def vapour_density(t_c: float, p_vap_kpa: float, m_vapour: float = 130.0) -> float:
    """Density of the vapour in the vapour space, kg/m3 (ideal gas)."""
    return m_vapour * p_vap_kpa / (R_GAS * (t_c + 273.15))


def vent_capacity_factor(vent_area_m2: float, required_flow_m3s: float,
                         dp_kpa: float = 0.35, cd: float = 0.62,
                         rho_air: float = 1.18) -> float:
    
    if required_flow_m3s <= 0:
        return 0.0
    q_cap = cd * vent_area_m2 * math.sqrt(2.0 * dp_kpa * 1000.0 / rho_air)
    return min(1.0, q_cap / required_flow_m3s)


def standing_loss_kl(vapour_space_m3: float, ullage_m: float, t_liq_c: float,
                     dt_vapour_k: float, vent_area_m2: float,
                     pv_setting_kpa: float, rho15: float,
                     hours: float = 8.0, tvp_ref_kpa: float = 0.40,
                     m_vapour: float = 130.0) -> float:

    if vapour_space_m3 <= 0:
        return 0.0
    p_va = true_vapour_pressure(t_liq_c, tvp_ref_kpa)
    w_v = vapour_density(t_liq_c, p_va, m_vapour)
    t_la_k = t_liq_c + 273.15

    # daily vapour-pressure swing across the diurnal temperature swing
    p_hi = true_vapour_pressure(t_liq_c + dt_vapour_k / 2.0, tvp_ref_kpa)
    p_lo = true_vapour_pressure(t_liq_c - dt_vapour_k / 2.0, tvp_ref_kpa)
    dp_v = p_hi - p_lo

    k_e = (dt_vapour_k / t_la_k) + (dp_v - pv_setting_kpa) / (P_ATM - p_va)
    k_e = max(0.0, k_e) * (hours / 24.0)          # pro-rate the diurnal cycle
    if k_e == 0.0:
        return 0.0

    h_vo_ft = ullage_m / M_PER_FT
    p_va_psia = p_va / KPA_PER_PSI
    k_s = 1.0 / (1.0 + 0.053 * p_va_psia * h_vo_ft)

    # can the vent actually pass the breathing flow?
    required_flow = vapour_space_m3 * k_e / (hours * 3600.0)
    k_vent = vent_capacity_factor(vent_area_m2, required_flow)

    mass_kg = vapour_space_m3 * w_v * k_e * k_s * k_vent
    return mass_kg / rho15                                  # kg -> m3 -> kL


def working_loss_kl(volume_in_kl: float, t_liq_c: float, rho15: float,
                    saturation: float = 0.6, tvp_ref_kpa: float = 0.40,
                    m_vapour: float = 130.0) -> float:

    if volume_in_kl <= 0:
        return 0.0
    p_va = true_vapour_pressure(t_liq_c, tvp_ref_kpa)
    w_v = vapour_density(t_liq_c, p_va, m_vapour)
    return (volume_in_kl * saturation * w_v) / rho15


def loading_loss_kl(volume_loaded_kl: float, t_liq_c: float, rho15: float,
                    saturation: float = 0.5, vru_efficiency: float = 0.0,
                    tvp_ref_kpa: float = 0.40, m_vapour: float = 130.0) -> float:
 
    if volume_loaded_kl <= 0:
        return 0.0
    p_va = true_vapour_pressure(t_liq_c, tvp_ref_kpa)
    w_v = vapour_density(t_liq_c, p_va, m_vapour)
    return (volume_loaded_kl * saturation * w_v * (1.0 - vru_efficiency)) / rho15


# ---------------------------------------------------------------------------
# Tank geometry helper
# ---------------------------------------------------------------------------
def tank_geometry(volume_kl: float, diameter_m: float, height_m: float) -> dict:
    """Liquid level, ullage and vapour space for a vertical cylindrical tank."""
    area = math.pi * diameter_m ** 2 / 4.0
    level = max(0.0, min(height_m, volume_kl / area))
    ullage = height_m - level
    return {"cross_section_m2": area, "level_m": level,
            "ullage_m": ullage, "vapour_space_m3": area * ullage}


if __name__ == "__main__":                                    # quick self-check
    rho = 832.0
    print(f"HSD rho15={rho}  band={commodity_band(rho)[3]!r}")
    print(f"alpha_15 = {alpha_15(rho):.6e} /degC   "
          f"(= {alpha_15(rho)/1.8:.6e} /degF)")
    for t in (15, 25, 30, 35, 40):
        print(f"  T={t:>3} degC   VCF={vcf(t, rho):.5f}   "
              f"20,000 kL observed -> {20000*vcf(t, rho):9.1f} kL @15 degC")
    print(f"TVP(30 degC) = {true_vapour_pressure(30):.3f} kPa")
