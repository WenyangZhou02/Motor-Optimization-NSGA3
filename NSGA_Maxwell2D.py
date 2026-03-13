"""
SPMSM Motor Multi-Objective Optimization - NSGA-III + PyAEDT
=============================================================
Decision variables:
    Gap      : Air gap width          0.3 - 1.0 mm
    Polearc  : Pole arc coefficient   0.70 - 0.82
    Mag_th   : Magnet thickness       2.8 - 4.2 mm
    Bs0      : Stator slot opening    2.8 - 4.0 mm

Objectives:
    AvgTorque       -> Maximize
    AvgTorqueRipple -> Minimize
    AvgEfficiency   -> Maximize

PRE-REQUISITE (one-time AEDT setup)
------------------------------------
Create 3 reports in AEDT Results tree (saved with the exact names below).
All three use Domain = "Average and RMS".

  Rpt_AvgTorque      Category: Torque        Qty: AvgTorque
  Rpt_TorqueRipple   Category: Torque        Qty: AvgTorqueRipple
  Rpt_AvgEfficiency  Category: Misc. Solution  Qty: AvgEfficiency

After each simulation AEDT auto-updates these reports; Python exports them
to CSV and reads the current values -> same as what you see in the AEDT UI.
"""

import os
import re
import random
import csv
import signal
import threading
import numpy as np
import matplotlib.pyplot as plt
from math import factorial

from deap import base, creator, tools, algorithms

# ============================================================
# Global abort control
# ============================================================

_stop_flag  = False
_m2d_global = None


def _signal_handler(signum, frame):
    global _stop_flag
    _stop_flag = True
    print("\n\n[Ctrl+C] Stop requested...")
    if _m2d_global is not None:
        try:
            _m2d_global.odesktop.AbortSolve()
            print("[AEDT] Abort sent.")
        except Exception as e:
            print(f"[AEDT] Abort via API failed ({e}).")
            print("       Fallback: Get-Process MAXWELL2DCOMENGINE | Stop-Process -Force")


signal.signal(signal.SIGINT, _signal_handler)

# ============================================================
# Section 1 AEDT Project Settings
# ============================================================

AEDT_PROJECT  = "SPMSM"
AEDT_DESIGN   = "Maxwell2DDesign1"
AEDT_SETUP    = "Setup1"       # verify in AEDT -> Analysis

NON_GRAPHICAL = False   # False = use open AEDT window
NEW_DESKTOP   = False   # False = attach to existing AEDT session

OUTPUT_ALL_GEN   = "all_generations.csv"     # every generation's pop (appended)
OUTPUT_FINAL_POP = "final_population.csv"    # last generation's pop
OUTPUT_PARETO    = "final_pareto_front.csv"  # non-dominated front of last pop
OUTPUT_PNG       = "pareto_front.png"

# ============================================================
# Section 2 - Decision Variable Bounds
# ============================================================

# (aedt_variable_name, lower_bound, upper_bound, unit_string)
# unit_string = "" for dimensionless variables
VARIABLES = [
    ("Gap",     0.3,  1.0,  "mm"),
    ("Polearc", 0.70, 0.82, ""),
    ("Mag_th",  2.8,  4.2,  "mm"),
    ("Bs0",     2.8,  4.0,  "mm"),
]

# Column layout shared by all three CSV files (defined after VARIABLES)
_CSV_FIELDS = (["generation", "individual_id"]
               + [v[0] for v in VARIABLES]
               + ["AvgTorque", "TorqueRipple", "AvgEfficiency", "is_valid"])

NDIM      = len(VARIABLES)
BOUND_LOW = [v[1] for v in VARIABLES]
BOUND_UP  = [v[2] for v in VARIABLES]

# ============================================================
# Section 3 - NSGA-III Parameters
# ============================================================

NOBJ      = 3     # number of objectives
P         = 6     # reference-point divisions (increase for more Pareto density)
NGEN      = 6     # generations (increase for longer optimization)
CXPB      = 0.9   # crossover probability
MUTPB     = 0.3   # mutation probability (per individual)
IND_MUTPB = 1.0 / NDIM  # mutation probability per gene

# ============================================================
# Section 4 - DEAP Setup
# ============================================================

# weights: +1 = maximize, -1 = minimize
creator.create("FitnessMotor", base.Fitness, weights=(1.0, -1.0, 1.0))
creator.create("Individual",   list,         fitness=creator.FitnessMotor)

toolbox = base.Toolbox()
toolbox.register("individual", tools.initIterate, creator.Individual,
                 lambda: [random.uniform(lo, hi)
                          for lo, hi in zip(BOUND_LOW, BOUND_UP)])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate",   tools.cxSimulatedBinaryBounded,
                 low=BOUND_LOW, up=BOUND_UP, eta=30.0)
toolbox.register("mutate", tools.mutPolynomialBounded,
                 low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=IND_MUTPB)

ref_points = tools.uniform_reference_points(nobj=NOBJ, p=P)
toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

H  = int(factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1)))
MU = H if H % 4 == 0 else H + (4 - H % 4)
print(f"[NSGA-III] Objectives={NOBJ}, p={P}, "
      f"Ref points={len(ref_points)}, Population={MU}")

# ============================================================
# Section 5 - Simulation & Post-Processing
# ============================================================

def set_variables(m2d, individual):
    """Write decision-variable values into the AEDT design."""
    for (name, _, _, unit), value in zip(VARIABLES, individual):
        m2d[name] = f"{value:.6f}{unit}" if unit else f"{value:.6f}"


_SOLVE_ERROR_KEYWORDS = (
    "error in solving",
    "execution error",
    "has failed",
    "solve failed",
    "simulation completed with execution error",
)


def _desktop_messages(m2d):
    """
    Read AEDT message-manager lines for current project/design.
    Uses the first callable signature that returns data.
    """
    desktop = getattr(m2d, "odesktop", None)
    if desktop is None or not hasattr(desktop, "GetMessages"):
        return []

    proj = getattr(m2d, "project_name", "")
    des = getattr(m2d, "design_name", "")
    signatures = [
        (proj, des, 0),
        (proj, des, 2),
        ("", "", 0),
        ("", "", 2),
    ]

    first_success = None
    for args in signatures:
        try:
            raw = desktop.GetMessages(*args)
        except Exception:
            continue

        if isinstance(raw, str):
            msgs = [raw]
        elif isinstance(raw, (list, tuple)):
            msgs = [str(x) for x in raw if str(x).strip()]
        else:
            msgs = []

        if first_success is None:
            first_success = msgs
        if msgs:
            return msgs

    return first_success or []


def _has_solver_error(messages):
    """Return True if AEDT messages indicate a solve failure."""
    for msg in messages:
        low = msg.lower()
        if any(key in low for key in _SOLVE_ERROR_KEYWORDS):
            return True
    return False


def run_simulation(m2d):
    """
    Delete any stored solutions, then run the setup in a background thread.
    Deleting first guarantees the CSV export after solving contains exactly
    one data row -> the current individual's result.
    Calls save_project() after solving to flush data to disk.
    """
    messages_before = _desktop_messages(m2d)

    # Try to clear previous solutions so the report CSV is less likely to mix runs.
    # Do not hard-fail here: first individual may have nothing to clear.
    try:
        clear_ok = m2d.odesign.DeleteFullVariation("All", False)
        if isinstance(clear_ok, bool) and not clear_ok:
            print("    [Info] DeleteFullVariation returned False (nothing to clear or API behavior).")
    except Exception as e:
        print(f"    [Warn] Could not clear old solutions: {e}")

    exc_holder = [None]
    solve_ok_holder = [None]

    def _solve():
        try:
            solve_ok_holder[0] = m2d.analyze_setup(AEDT_SETUP)
        except Exception as e:
            exc_holder[0] = e

    t = threading.Thread(target=_solve, daemon=True)
    t.start()
    while t.is_alive():
        t.join(timeout=1.0)
        if _stop_flag:
            t.join(timeout=15.0)
            raise KeyboardInterrupt("Simulation aborted by user.")
    if exc_holder[0] is not None:
        raise exc_holder[0]
    if solve_ok_holder[0] is False:
        raise RuntimeError(f"analyze_setup('{AEDT_SETUP}') returned False.")

    messages_after = _desktop_messages(m2d)
    new_messages = (messages_after[len(messages_before):]
                    if len(messages_after) >= len(messages_before)
                    else messages_after)
    if _has_solver_error(new_messages):
        err_line = next(
            (m for m in new_messages
             if any(k in m.lower() for k in _SOLVE_ERROR_KEYWORDS)),
            new_messages[0] if new_messages else "Unknown solver error"
        )
        raise RuntimeError(f"AEDT reported solve error: {err_line}")

    m2d.save_project()


def _read_report_csv(m2d, report_name):
    """
    Export an AEDT report to CSV and return (Y value, Y unit from header).

    Because run_simulation() deletes old solutions before each solve,
    the exported CSV always contains exactly one data row -> the current
    individual's result. No row-matching is needed.
    """
    proj_file = getattr(m2d, "project_file",
                        getattr(m2d, "project_path", None))
    if proj_file is None:
        raise AttributeError("Cannot determine project directory.")
    project_dir = os.path.dirname(proj_file)

    csv_path = m2d.post.export_report_to_csv(
        project_dir=project_dir,
        plot_name=report_name,
    )
    if not csv_path or not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"CSV export failed for '{report_name}'. "
            "Verify the report exists in the AEDT Results tree."
        )

    # Parse CSV - skip header rows (contain letters), read numeric rows.
    # Typical AEDT format:
    #   "X","AvgTorque [mNewtonMeter]"   -> header
    #   0.650000,237.2855               -> single data row after solution delete
    with open(csv_path, encoding="utf-8-sig") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    report_unit = _extract_report_unit(lines)

    for line in reversed(lines):          # start from last row (most recent)
        parts = [p.strip().strip('"') for p in re.split(r'[,\t]', line)]
        nums  = [float(p) for p in parts if _is_number(p)]
        if nums:
            return nums[-1], report_unit  # last column = quantity value

    raise ValueError(f"No numeric data found in {csv_path}")


def _is_number(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _extract_report_unit(lines):
    """
    Extract Y-axis unit from AEDT CSV header.
    Typical header cell: "AvgTorque [mNewtonMeter]".
    """
    for line in lines:
        parts = [p.strip().strip('"') for p in re.split(r'[,\t]', line)]
        # Prefer the last column (Y quantity) in case X column also has a unit.
        for part in reversed(parts):
            m = re.search(r"\[([^\[\]]+)\]", part)
            if m:
                return m.group(1).strip()
        # numeric rows start; headers should have appeared before this
        if any(_is_number(p) for p in parts):
            break
    return ""


def _torque_to_nm(value, unit, report_name):
    """
    Normalize torque-like quantities to Nm.
    Supported units include Nm / NewtonMeter and mNm / mNewtonMeter variants.
    """
    norm = re.sub(r"[^a-z0-9]+", "", (unit or "").lower())

    nm_units = {
        "nm", "newtonmeter", "newtonmeters", "newtonmetre", "newtonmetres"
    }
    mnm_units = {
        "mnm", "mnewtonmeter", "mnewtonmeters", "mnewtonmetre", "mnewtonmetres",
        "millinewtonmeter", "millinewtonmeters", "millinewtonmetre", "millinewtonmetres"
    }

    if norm in nm_units:
        return value
    if norm in mnm_units:
        return value / 1000.0
    if not norm:
        print(f"    [Warn] Unit missing in {report_name}; assuming Nm.")
        return value

    raise ValueError(
        f"Unsupported torque unit '{unit}' in {report_name}. "
        "Please use Nm or mNm in AEDT report settings."
    )


# Report names to create in AEDT (one-time setup):
#   Results -> New Report -> (choose the matching Category & Quantity)
#   Save with these exact names:
REPORT_AVG_TORQUE    = "Rpt_AvgTorque"      # Category: Torque,        Qty: AvgTorque
REPORT_TORQUE_RIPPLE = "Rpt_TorqueRipple"   # Category: Torque,        Qty: AvgTorqueRipple
REPORT_EFFICIENCY    = "Rpt_AvgEfficiency"  # Category: Misc. Solution, Qty: AvgEfficiency
# All three reports: Domain = Average and RMS


def get_objectives(m2d):
    """
    Export AEDT reports to CSV and read the current simulation's values.

    Old solutions are deleted before each simulation, so the CSV contains
    exactly one row -> unambiguously the current individual's result.
    """
    try:
        avg_torque_raw, avg_torque_unit = _read_report_csv(m2d, REPORT_AVG_TORQUE)
        ripple_raw, ripple_unit = _read_report_csv(m2d, REPORT_TORQUE_RIPPLE)
        avg_eff, _ = _read_report_csv(m2d, REPORT_EFFICIENCY)

        avg_torque = _torque_to_nm(avg_torque_raw, avg_torque_unit, REPORT_AVG_TORQUE)
        ripple_Nm = _torque_to_nm(ripple_raw, ripple_unit, REPORT_TORQUE_RIPPLE)

        # AvgTorqueRipple is peak-to-peak in Nm; convert to % of avg torque.
        torque_ripple = (ripple_Nm / avg_torque * 100.0
                         if avg_torque != 0 else 999.0)

        print(f"    -> Torque={avg_torque:.2f} Nm, "
              f"Ripple={ripple_Nm:.2f} Nm pk-pk -> {torque_ripple:.2f}%, "
              f"Eff={avg_eff:.2f}%")
        return avg_torque, torque_ripple, avg_eff

    except Exception as e:
        print(f"    [Error] Report CSV read failed: {e}")
        return 0.0, 999.0, 0.0


def _save_population(pop, generation, filepath, mode="a"):
    """
    Write one generation's population to a CSV file.

    mode = "w"  -> write header + rows (first call)
    mode = "a"  -> append rows only (subsequent calls)

    is_valid: 1 = real simulation result,
              0 = penalty value (TorqueRipple -> 999 signals failure).
    """
    rows = []
    for idx, ind in enumerate(pop):
        if not ind.fitness.valid:
            continue
        fv = ind.fitness.values
        row = {"generation": generation, "individual_id": idx}
        for i, (name, *_) in enumerate(VARIABLES):
            row[name] = round(ind[i], 6)
        row["AvgTorque"]     = fv[0]
        row["TorqueRipple"]  = fv[1]
        row["AvgEfficiency"] = fv[2]
        row["is_valid"]      = 0 if fv[1] >= 999.0 else 1
        rows.append(row)

    with open(filepath, mode, newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=_CSV_FIELDS)
        if mode == "w":
            w.writeheader()
        w.writerows(rows)


def evaluate_individual(individual, m2d):
    """Set variables -> simulate -> extract objectives."""
    print("    Variables: " + ", ".join(
        f"{v[0]}={x:.4f}{v[3]}" for v, x in zip(VARIABLES, individual)))

    try:
        set_variables(m2d, individual)
    except AttributeError:
        # Desktop was dropped; reconnect once.
        global _m2d_global
        from ansys.aedt.core import Maxwell2d
        m2d = Maxwell2d(project=AEDT_PROJECT, design=AEDT_DESIGN,
                        non_graphical=NON_GRAPHICAL, new_desktop=NEW_DESKTOP)
        _m2d_global = m2d
        toolbox.register("evaluate", evaluate_individual, m2d=m2d)
        set_variables(m2d, individual)

    try:
        run_simulation(m2d)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"    [Error] Simulation failed: {e}")
        return 0.0, 999.0, 0.0

    return get_objectives(m2d)

# ============================================================
# Section 6 - Main Optimization Loop
# ============================================================

def main():
    print("\n[Step 1] Connecting to AEDT...")
    from ansys.aedt.core import Maxwell2d
    m2d = Maxwell2d(project=AEDT_PROJECT, design=AEDT_DESIGN,
                    non_graphical=NON_GRAPHICAL, new_desktop=NEW_DESKTOP)
    global _m2d_global
    _m2d_global = m2d
    print(f"  Connected: {m2d.project_name} / {m2d.design_name}")

    toolbox.register("evaluate", evaluate_individual, m2d=m2d)

    print(f"\n[Step 2] Initializing population (size={MU})...")
    pop = toolbox.population(n=MU)

    print(f"\n[Step 3] Evaluating (Ctrl+C to stop and save at any time)...")
    invalid = [ind for ind in pop if not ind.fitness.valid]
    last_gen = 0  # track last completed generation for final CSV files
    try:
        for i, ind in enumerate(invalid):
            print(f"\n  [Gen 0 | {i+1}/{len(invalid)}]")
            ind.fitness.values = toolbox.evaluate(ind)

        # Save Gen 0 (initial evaluated population) -> creates the file with header
        _save_population(pop, generation=0, filepath=OUTPUT_ALL_GEN, mode="w")
        print(f"  [CSV] Gen 0 saved -> {OUTPUT_ALL_GEN}")
        last_gen = 0

        for gen in range(1, NGEN + 1):
            print(f"\n{'='*55}\n  Generation {gen}/{NGEN}\n{'='*55}")
            offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)
            invalid   = [ind for ind in offspring if not ind.fitness.valid]
            print(f"  Evaluating {len(invalid)} new individuals...")
            for i, ind in enumerate(invalid):
                print(f"\n  [Gen {gen} | {i+1}/{len(invalid)}]")
                ind.fitness.values = toolbox.evaluate(ind)

            # Selection -> pop is now the new parent population for the next gen
            pop = toolbox.select(pop + offspring, MU)

            # Append this generation's selected pop to the history file
            _save_population(pop, generation=gen, filepath=OUTPUT_ALL_GEN, mode="a")
            print(f"  [CSV] Gen {gen} saved -> {OUTPUT_ALL_GEN}")
            last_gen = gen

        print("\n[Optimization finished]")

    except KeyboardInterrupt:
        print("\n[Interrupted -> saving results so far...]")
        pop = [ind for ind in pop if ind.fitness.valid]

    # final_population.csv -> last generation's full population
    _save_population(pop, generation=last_gen, filepath=OUTPUT_FINAL_POP, mode="w")
    print(f"  Saved -> {OUTPUT_FINAL_POP}")

    # Print table (final population)
    hdr = (f"{'Gap':>7} {'Polearc':>9} {'Mag_th':>8} {'Bs0':>6} | "
           f"{'Torque':>8} {'Ripple%':>8} {'Eff%':>7}")
    print(f"\n{hdr}\n{'-'*len(hdr)}")
    for ind in pop:
        if not ind.fitness.valid:
            continue
        fv = ind.fitness.values
        print(f"{ind[0]:7.4f} {ind[1]:9.4f} {ind[2]:8.4f} "
              f"{ind[3]:6.4f} | {fv[0]:8.4f} {fv[1]:8.4f} {fv[2]:7.4f}")

    m2d.save_project()
    print("\n  Project saved. AEDT remains open.")
    return pop, last_gen

# ============================================================
# Section 7 - Run & Plot
# ============================================================

if __name__ == "__main__":
    final_pop, last_gen = main()

    if not final_pop:
        print("No results to plot.")
        exit(0)

    # Keep only the non-dominated (Pareto front) solutions.
    # sortNondominated returns a list of fronts; index 0 is the true Pareto front.
    pareto_front = tools.sortNondominated(
        final_pop, len(final_pop), first_front_only=True
    )[0]
    print(f"\n  Final Pareto front: {len(pareto_front)} / {len(final_pop)} individuals")

    # final_pareto_front.csv -> non-dominated solutions from the last generation
    _save_population(pareto_front, generation=last_gen,
                     filepath=OUTPUT_PARETO, mode="w")
    print(f"  Saved -> {OUTPUT_PARETO}")

    torque_vals = np.array([ind.fitness.values[0] for ind in pareto_front])
    ripple_vals = np.array([ind.fitness.values[1] for ind in pareto_front])
    eff_vals    = np.array([ind.fitness.values[2] for ind in pareto_front])

    fig = plt.figure(figsize=(10, 7))
    ax  = fig.add_subplot(111, projection="3d")
    # Keep axis proportions stable in static export to reduce label overlap/occlusion.
    ax.set_box_aspect((1.25, 1.0, 0.8))

    sc = ax.scatter(torque_vals, ripple_vals, eff_vals,
                    c=torque_vals, cmap="plasma", s=50,
                    edgecolors="k", linewidths=0.3)
    plt.colorbar(sc, ax=ax, label="Avg Torque (Nm)", shrink=0.72, pad=0.08)

    ax.set_xlabel("Avg Torque (Nm)", labelpad=10)
    ax.set_ylabel("Torque Ripple (%)", labelpad=10)
    ax.set_zlabel("Avg Efficiency (%)", labelpad=10)
    ax.set_title(f"NSGA-III Pareto Front - SPMSM\n"
                 f"({NGEN} gen, pop={MU}, "
                 f"Pareto pts={len(pareto_front)})")
    # A more readable default static viewpoint (similar to a manually rotated good view).
    ax.view_init(elev=20, azim=35)

    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"Plot saved -> {OUTPUT_PNG}")
