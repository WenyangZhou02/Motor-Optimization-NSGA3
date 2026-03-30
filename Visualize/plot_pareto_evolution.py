"""
plot_pareto_evolution.py — Visualize NSGA-III Pareto front evolution from CSV.

Usage:
    python Visualize/plot_pareto_evolution.py [path/to/all_generations_xxx.csv]

If no path is given, automatically picks the newest all_generations_*.csv
in the Results/ directory (relative to this script's parent).
"""

import os
import re
import csv
import sys
import glob
import matplotlib.pyplot as plt


# ============================================================
# CSV helpers
# ============================================================

def _find_latest_csv(results_dir):
    pattern = os.path.join(results_dir, "all_generations_*.csv")
    files   = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No all_generations_*.csv found in: {results_dir}")
    return max(files)   # lexicographic sort works — timestamps are yyyymmdd_hhmmss


def _read_generations(csv_path):
    """
    Read all_generations CSV and group valid rows by generation.

    Returns:
        gen_data : dict  {gen_idx: [(torque, ripple_pct, eff), ...]}
        last_gen : int
    """
    gen_data = {}
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("is_valid", "1") == "0":
                continue
            try:
                gen    = int(row["generation"])
                torque = float(row["AvgTorque"])
                ripple = float(row["TorqueRipple"])
                eff    = float(row["AvgEfficiency"])
            except (ValueError, KeyError):
                continue
            gen_data.setdefault(gen, []).append((torque, ripple, eff))
    last_gen = max(gen_data.keys()) if gen_data else 0
    return gen_data, last_gen


def _ts_from_path(csv_path):
    m = re.search(r'(\d{8}_\d{6})', os.path.basename(csv_path))
    return m.group(1) if m else "unknown"


# ============================================================
# Pareto front (non-dominated set)
# Objectives: Torque maximize, Ripple minimize, Efficiency maximize
# ============================================================

def _pareto_front(solutions):
    """Return Pareto-non-dominated solutions from a list of (torque, ripple, eff) tuples."""
    front = []
    for a in solutions:
        dominated = False
        for b in solutions:
            if b is a:
                continue
            # b dominates a if b is at least as good on all objectives and strictly better on one
            if (b[0] >= a[0] and b[1] <= a[1] and b[2] >= a[2] and
                    (b[0] > a[0] or b[1] < a[1] or b[2] > a[2])):
                dominated = True
                break
        if not dominated:
            front.append(a)
    return front


# ============================================================
# Plot
# ============================================================

def plot(csv_path):
    gen_data, last_gen = _read_generations(csv_path)
    if not gen_data:
        print("No valid data found in CSV.")
        return

    ts         = _ts_from_path(csv_path)
    output_png = os.path.join(os.path.dirname(csv_path), f"pareto_evolution_{ts}.png")

    # Build per-generation Pareto fronts
    gen_pareto_fronts = []
    for gen_idx in sorted(gen_data.keys()):
        pf = _pareto_front(gen_data[gen_idx])
        gen_pareto_fronts.append((gen_idx, pf))

    n_fronts = len(gen_pareto_fronts)
    cmap     = plt.cm.coolwarm
    norm     = plt.Normalize(vmin=0, vmax=last_gen)

    fig = plt.figure(figsize=(12, 8))
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1.25, 1.0, 0.8))

    for i, (gen_idx, pf) in enumerate(gen_pareto_fronts):
        if not pf:
            continue
        t          = gen_idx / max(last_gen, 1)   # 0.0 (Gen 0) -> 1.0 (last gen)
        color      = cmap(t)
        alpha      = 0.30 + 0.70 * t              # 0.30 -> 1.00
        size       = 18   + 42   * t              # 18   -> 60
        edge_color = "k"   if i == n_fronts - 1 else "none"
        linewidth  = 0.6   if i == n_fronts - 1 else 0.0

        torques = [s[0] for s in pf]
        ripples = [s[1] for s in pf]
        effs    = [s[2] for s in pf]

        ax.scatter(torques, ripples, effs,
                   color=color, s=size, alpha=alpha,
                   edgecolors=edge_color, linewidths=linewidth,
                   depthshade=True, label=f"Gen {gen_idx}")

    # Colorbar keyed to generation number
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, label="Generation", shrink=0.65, pad=0.10)
    tick_step = max(1, last_gen // 6)
    cbar.set_ticks(range(0, last_gen + 1, tick_step))

    final_pf_len = len(gen_pareto_fronts[-1][1]) if gen_pareto_fronts else 0
    ax.set_xlabel("Avg Torque (Nm)", labelpad=10)
    ax.set_ylabel("Torque Ripple (%)", labelpad=10)
    ax.set_zlabel("Avg Efficiency (%)", labelpad=10)
    ax.set_title(f"NSGA-III Pareto Front Evolution - SPMSM\n"
                 f"({last_gen} gen, {n_fronts} fronts shown, "
                 f"final front={final_pf_len} pts)")
    ax.invert_yaxis()   # ripple: lower is better, visually push front forward
    ax.view_init(elev=20, azim=35)
    ax.legend(loc='upper left', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"Plot saved -> {output_png}")


def plot_2d(csv_path):
    gen_data, last_gen = _read_generations(csv_path)
    if not gen_data:
        print("No valid data found in CSV.")
        return

    ts         = _ts_from_path(csv_path)
    output_png = os.path.join(os.path.dirname(csv_path), f"pareto_projections_{ts}.png")

    # Build per-generation Pareto fronts
    gen_pareto_fronts = []
    for gen_idx in sorted(gen_data.keys()):
        pf = _pareto_front(gen_data[gen_idx])
        gen_pareto_fronts.append((gen_idx, pf))

    n_fronts = len(gen_pareto_fronts)
    cmap     = plt.cm.coolwarm
    norm     = plt.Normalize(vmin=0, vmax=last_gen)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    ax1, ax2, ax3 = axes

    # Axis labels for each projection
    ax1.set_xlabel("Avg Torque (Nm)")
    ax1.set_ylabel("Torque Ripple (%)")
    ax2.set_xlabel("Avg Torque (Nm)")
    ax2.set_ylabel("Avg Efficiency (%)")
    ax3.set_xlabel("Torque Ripple (%)")
    ax3.set_ylabel("Avg Efficiency (%)")

    for i, (gen_idx, pf) in enumerate(gen_pareto_fronts):
        if not pf:
            continue
        t          = gen_idx / max(last_gen, 1)
        color      = cmap(t)
        alpha      = 0.30 + 0.70 * t
        size       = 18   + 42   * t
        edge_color = "k"   if i == n_fronts - 1 else "none"
        linewidth  = 0.6   if i == n_fronts - 1 else 0.0
        label      = f"Gen {gen_idx}"

        torques = [s[0] for s in pf]
        ripples = [s[1] for s in pf]
        effs    = [s[2] for s in pf]

        scatter_kw = dict(color=color, s=size, alpha=alpha,
                          edgecolors=edge_color, linewidths=linewidth)
        ax1.scatter(torques, ripples, label=label, **scatter_kw)
        ax2.scatter(torques, effs,    label=label, **scatter_kw)
        ax3.scatter(ripples, effs,    label=label, **scatter_kw)

    # Invert ripple axes so that "lower ripple = better" reads naturally
    ax1.invert_yaxis()   # ax1 Y-axis is ripple
    ax3.invert_xaxis()   # ax3 X-axis is ripple

    # Colorbar anchored to ax3 only to avoid squeezing the other subplots
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax3, label="Generation",
                        shrink=0.80, pad=0.15)
    tick_step = max(1, last_gen // 6)
    cbar.set_ticks(range(0, last_gen + 1, tick_step))

    fig.suptitle(f"NSGA-III Pareto Front Projections - SPMSM\n"
                 f"({last_gen} gen, {n_fronts} fronts shown)",
                 fontsize=12)
    plt.subplots_adjust(left=0.12, right=0.88)

    # Unified legend in the left margin (handles from ax1; all subplots share the same labels)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="center left", bbox_to_anchor=(0.01, 0.5),
               fontsize=7, framealpha=0.8)
    plt.savefig(output_png, dpi=180, bbox_inches="tight")
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

    plot(csv_path)
    plot_2d(csv_path)
