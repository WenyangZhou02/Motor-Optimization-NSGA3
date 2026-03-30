"""
plot_hypervolume.py — Hypervolume convergence + per-generation population statistics.

Produces two separate PNG files:
    hypervolume_{ts}.png        — HV convergence curve
    population_stats_{ts}.png   — three trend subplots (torque / ripple / efficiency)

Usage:
    python Visualize/plot_hypervolume.py [path/to/all_generations_xxx.csv]

If no path is given, automatically picks the newest all_generations_*.csv
in the Results/ directory (relative to this script's parent).

Requires:
    pip install pymoo
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ── pymoo check (fail loudly before anything else) ───────────────────────────
try:
    from pymoo.indicators.hv import HV
except ImportError:
    print("ERROR: pymoo is required for hypervolume computation.")
    print("       Install with:  pip install pymoo")
    sys.exit(1)

# ── Reuse CSV helpers and Pareto logic from the sibling module ────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_pareto_evolution import (
    _find_latest_csv,
    _read_generations,
    _ts_from_path,
    _pareto_front,
)


# ============================================================
# Helpers
# ============================================================

def _compute_hypervolumes(gen_data):
    """
    Return (gens, hv_values) aligned by generation.

    Converts to minimization form: (-torque, ripple, -efficiency).
    Reference point = (−min_torque·0.95, max_ripple·1.05, −min_eff·0.95),
    strictly worse than every solution on every dimension.
    """
    all_solutions = [s for sols in gen_data.values() for s in sols]
    min_torque = min(s[0] for s in all_solutions)
    max_ripple = max(s[1] for s in all_solutions)
    min_eff    = min(s[2] for s in all_solutions)

    ref_point    = np.array([-min_torque * 0.95,
                              max_ripple  * 1.05,
                             -min_eff     * 0.95])
    hv_indicator = HV(ref_point=ref_point)

    gens      = sorted(gen_data.keys())
    hv_values = []
    for g in gens:
        pf = _pareto_front(gen_data[g])
        if not pf:
            hv_values.append(0.0)
            continue
        F = np.array([[-s[0], s[1], -s[2]] for s in pf])
        hv_values.append(float(hv_indicator.do(F)))

    return gens, hv_values


def _trend_ylim(mean_vals, lo_vals, hi_vals, margin=0.10):
    """
    Compute tight Y limits using 5th–95th percentile of all displayed values.
    Prevents extreme outliers in the fill_between bands from collapsing the
    visible trend into a flat line.
    """
    combined = np.concatenate([mean_vals, lo_vals, hi_vals])
    p5  = float(np.percentile(combined, 5))
    p95 = float(np.percentile(combined, 95))
    span = max(p95 - p5, 1e-6)
    return (p5 - span * margin, p95 + span * margin)


# ============================================================
# Figure 1 — Hypervolume convergence
# ============================================================

def plot_hv(csv_path):
    gen_data, _ = _read_generations(csv_path)
    if not gen_data:
        print("No valid data found in CSV.")
        return

    ts         = _ts_from_path(csv_path)
    output_png = os.path.join(os.path.dirname(csv_path), f"hypervolume_{ts}.png")

    gens, hv_values = _compute_hypervolumes(gen_data)

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle("NSGA-III Optimization Convergence - SPMSM", fontsize=13)

    HV_COLOR = "#1a3a6b"
    ax.plot(gens, hv_values, color=HV_COLOR, linewidth=2,
            marker="o", markersize=6)
    for g, hv in zip(gens, hv_values):
        ax.annotate(f"{hv:.4g}",
                    xy=(g, hv), xytext=(0, 7),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=7, color=HV_COLOR)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume")
    ax.set_title("Hypervolume Convergence")
    ax.set_xticks(gens)
    ax.grid(True, linestyle="--", alpha=0.5)

    ax.set_ylim(min(hv_values) * 0.97, max(hv_values) * 1.025)
    plt.tight_layout()
    plt.savefig(output_png, dpi=180, bbox_inches="tight", pad_inches=0.3)
    plt.show()
    print(f"Plot saved -> {output_png}")


# ============================================================
# Figure 2 — Per-generation population statistics
# ============================================================

def plot_stats(csv_path):
    gen_data, _ = _read_generations(csv_path)
    if not gen_data:
        print("No valid data found in CSV.")
        return

    ts         = _ts_from_path(csv_path)
    output_png = os.path.join(os.path.dirname(csv_path), f"population_stats_{ts}.png")

    gens = sorted(gen_data.keys())

    # Per-generation statistics
    mean_torque = np.array([np.mean([s[0] for s in gen_data[g]]) for g in gens])
    max_torque  = np.array([max(s[0]  for s in gen_data[g]) for g in gens])
    min_torque  = np.array([min(s[0]  for s in gen_data[g]) for g in gens])

    mean_ripple = np.array([np.mean([s[1] for s in gen_data[g]]) for g in gens])
    max_ripple  = np.array([max(s[1]  for s in gen_data[g]) for g in gens])
    min_ripple  = np.array([min(s[1]  for s in gen_data[g]) for g in gens])

    mean_eff    = np.array([np.mean([s[2] for s in gen_data[g]]) for g in gens])
    max_eff     = np.array([max(s[2]  for s in gen_data[g]) for g in gens])
    min_eff_gen = np.array([min(s[2]  for s in gen_data[g]) for g in gens])

    fig, (ax_t, ax_r, ax_e) = plt.subplots(3, 1, figsize=(10, 9),
                                            sharex=True)
    fig.suptitle("Per-Generation Population Statistics - SPMSM", fontsize=13)

    # ── Avg Torque (red) ─────────────────────────────────────────────────────
    ax_t.plot(gens, mean_torque, color="tab:red", linewidth=2,
              marker="o", markersize=5, label="Mean")
    ax_t.fill_between(gens, min_torque, max_torque,
                      color="tab:red", alpha=0.15, label="Min–Max range")
    ax_t.set_ylim(min(min_torque)  * 0.98, max(max_torque)  * 1.01)
    ax_t.set_ylabel("Avg Torque (Nm)")
    ax_t.legend(loc="lower right", fontsize=7)
    ax_t.grid(True, alpha=0.3)

    # ── Avg Torque Ripple (blue) — lower is better ────────────────────────────
    ax_r.plot(gens, mean_ripple, color="tab:blue", linewidth=2,
              marker="s", markersize=5, label="Mean")
    ax_r.fill_between(gens, min_ripple, max_ripple,
                      color="tab:blue", alpha=0.15, label="Min–Max range")
    ax_r.set_ylim(min(min_ripple)  * 0.90, max(max_ripple)  * 1.05)
    ax_r.set_ylabel("Avg Torque Ripple (%)")
    ax_r.legend(loc="upper right", fontsize=7)
    ax_r.grid(True, alpha=0.3)

    # ── Avg Efficiency (green) ────────────────────────────────────────────────
    ax_e.plot(gens, mean_eff, color="tab:green", linewidth=2,
              marker="^", markersize=5, label="Mean")
    ax_e.fill_between(gens, min_eff_gen, max_eff,
                      color="tab:green", alpha=0.15, label="Min–Max range")
    ax_e.set_ylim(min(min_eff_gen) * 0.998, max(max_eff)    * 1.002)
    ax_e.set_ylabel("Avg Efficiency (%)")
    ax_e.set_xlabel("Generation")
    ax_e.legend(loc="lower right", fontsize=7)
    ax_e.grid(True, alpha=0.3)

    # Shared X ticks (propagates to ax_r and ax_e via sharex)
    ax_t.set_xticks(gens)

    plt.tight_layout()
    plt.savefig(output_png, dpi=180, bbox_inches="tight", pad_inches=0.3)
    plt.show()
    print(f"Plot saved -> {output_png}")


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        script_dir  = os.path.dirname(os.path.abspath(__file__))
        results_dir = os.path.normpath(os.path.join(script_dir, "..", "Results"))
        csv_path    = _find_latest_csv(results_dir)
        print(f"Using: {csv_path}")

    plot_hv(csv_path)
    plot_stats(csv_path)
