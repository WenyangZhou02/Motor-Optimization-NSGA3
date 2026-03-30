"""
SPMSM Motor Multi-Objective Optimization - NSGA-III + PyAEDT
=============================================================
Decision variables:
    Gap      : Air gap width          0.3 - 1.0 mm
    Polearc  : Pole arc coefficient   0.65 - 0.78
    Mag_th   : Magnet thickness       2.5 - 4.5 mm
    Bs0      : Stator slot opening    1.5 - 3.5 mm

Objectives:
    AvgTorque       -> Maximize
    AvgTorqueRipple -> Minimize
    AvgEfficiency   -> Maximize

PRE-REQUISITE (one-time AEDT setup)
------------------------------------
Create 3 reports in AEDT Results tree (saved with the exact names below).
All three use Domain = "Average and RMS".

  Rpt_AvgTorque      Category: Torque          Qty: AvgTorque
  Rpt_TorqueRipple   Category: Torque          Qty: AvgTorqueRipple
  Rpt_AvgEfficiency  Category: Misc. Solution  Qty: AvgEfficiency
"""

import os
import re
import csv
import signal
import threading
import numpy as np
from math import factorial
from datetime import datetime

from deap import base, creator, tools, algorithms

# ============================================================
# Global abort control
# ============================================================

_stop_flag  = False
_m2d_global = None


def _kill_solver_process():
    """Force-kill the Maxwell 2D solver engine process as a last resort."""
    import subprocess
    for exe in ("maxwell2dcomengine.exe", "Maxwell2DComEngine.exe"):
        try:
            r = subprocess.run(
                ["taskkill", "/F", "/IM", exe],
                capture_output=True, text=True, timeout=10,
            )
            if r.returncode == 0:
                print(f"[AEDT] Solver process ({exe}) killed.", flush=True)
                return True
        except Exception:
            continue
    return False


def _signal_handler(signum, frame):
    global _stop_flag
    _stop_flag = True
    print("\n\n[Ctrl+C] Stop requested...", flush=True)
    if _m2d_global is not None:
        # Try 1: AEDT API abort
        try:
            _m2d_global.odesktop.AbortSolve()
            print("[AEDT] Abort sent.", flush=True)
            return
        except Exception:
            pass
        # Try 2: kill solver process directly
        if _kill_solver_process():
            return
        # All failed
        print("[AEDT] Could not stop solver automatically.", flush=True)
        print("       Manual: taskkill /F /IM maxwell2dcomengine.exe", flush=True)


signal.signal(signal.SIGINT, _signal_handler)

# ============================================================
# Section 1 - AEDT Project Settings
# ============================================================

AEDT_PROJECT  = "SPMSM_new"
AEDT_DESIGN   = "Maxwell2DDesign"
AEDT_SETUP    = "Setup1"       # verify in AEDT -> Analysis

NON_GRAPHICAL = False   # False = use open AEDT window
NEW_DESKTOP   = False   # False = attach to existing AEDT session

_TS         = datetime.now().strftime("%Y%m%d_%H%M%S")
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Results")
os.makedirs(_RESULTS_DIR, exist_ok=True)

OUTPUT_ALL_GEN = os.path.join(_RESULTS_DIR, f"all_generations_{_TS}.csv")
OUTPUT_PARETO  = os.path.join(_RESULTS_DIR, f"final_pareto_front_{_TS}.csv")

# ============================================================
# Section 2 - Decision Variable Bounds
# ============================================================

# (aedt_variable_name, lower_bound, upper_bound, unit_string)
VARIABLES = [
    ("Gap",     0.3,  1.0,  "mm"),
    ("Polearc", 0.65, 0.78, ""),
    ("Mag_th",  2.5,  4.5,  "mm"),
    ("Bs0",     1.5,  3.5,  "mm"),
]

_CSV_FIELDS = (["generation", "individual_id"]
               + [v[0] for v in VARIABLES]
               + ["AvgTorque", "TorqueRipple", "AvgEfficiency", "is_valid"])

NDIM      = len(VARIABLES)
BOUND_LOW = [v[1] for v in VARIABLES]
BOUND_UP  = [v[2] for v in VARIABLES]

PENALTY_FITNESS = (0.0, 999.0, 0.0)

# ============================================================
# Section 3 - NSGA-III Parameters
# ============================================================

NOBJ      = 3     # number of objectives
P         = 8    # reference-point divisions
NGEN      = 10     # generations
CXPB      = 0.9   # crossover probability
MUTPB     = 0.5   # mutation probability (per individual)
IND_MUTPB = 1.0 / NDIM  # mutation probability per gene

# ============================================================
# Section 4 - DEAP Setup
# ============================================================

creator.create("FitnessMotor", base.Fitness, weights=(1.0, -1.0, 1.0))
creator.create("Individual",   list,         fitness=creator.FitnessMotor)

toolbox = base.Toolbox()
toolbox.register("mate",   tools.cxSimulatedBinaryBounded,
                 low=BOUND_LOW, up=BOUND_UP, eta=15.0)
toolbox.register("mutate", tools.mutPolynomialBounded,
                 low=BOUND_LOW, up=BOUND_UP, eta=20.0, indpb=IND_MUTPB)

ref_points = tools.uniform_reference_points(nobj=NOBJ, p=P)
toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

H  = int(factorial(NOBJ + P - 1) / (factorial(P) * factorial(NOBJ - 1)))
MU = H if H % 4 == 0 else H + (4 - H % 4)
print(f"[NSGA-III] Objectives={NOBJ}, p={P}, "
      f"Ref points={len(ref_points)}, Population={MU}", flush=True)


def _lhs_population(n):
    """Generate n individuals via Latin Hypercube Sampling over VARIABLES bounds."""
    from scipy.stats.qmc import LatinHypercube
    samples = LatinHypercube(d=NDIM).random(n=n)
    scaled  = np.array(BOUND_LOW) + samples * (np.array(BOUND_UP) - np.array(BOUND_LOW))
    return [creator.Individual(row.tolist()) for row in scaled]


# ============================================================
# Section 5 - Simulation & Post-Processing
# ============================================================

REPORT_AVG_TORQUE    = "Rpt_AvgTorque"
REPORT_TORQUE_RIPPLE = "Rpt_TorqueRipple"
REPORT_EFFICIENCY    = "Rpt_AvgEfficiency"

_SOLVE_ERROR_KEYWORDS = (
    "error in solving",
    "execution error",
    "has failed",
    "solve failed",
    "simulation completed with execution error",
)

# Additional keywords for detecting failed variations in batch sweep messages
_BATCH_FAIL_KEYWORDS = _SOLVE_ERROR_KEYWORDS + (
    "mesh generation failed",
    "mesh failed",
    "failed to recover",
    "clone mesh failed",
)


def set_variables(m2d, individual):
    """Write decision-variable values into the AEDT design."""
    for (name, _, _, unit), value in zip(VARIABLES, individual):
        m2d[name] = f"{value:.6f}{unit}" if unit else f"{value:.6f}"


def _desktop_messages(m2d):
    """Read AEDT message-manager lines, trying multiple GetMessages signatures."""
    desktop = getattr(m2d, "odesktop", None)
    if desktop is None or not hasattr(desktop, "GetMessages"):
        return []

    proj = getattr(m2d, "project_name", "")
    des  = getattr(m2d, "design_name", "")
    first_success = None
    for args in [(proj, des, 0), (proj, des, 2), ("", "", 0), ("", "", 2)]:
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
    return any(any(k in m.lower() for k in _SOLVE_ERROR_KEYWORDS) for m in messages)


def run_simulation(m2d):
    """Clear old solutions, solve in a background thread, then save project."""
    messages_before = _desktop_messages(m2d)

    try:
        m2d.odesign.DeleteFullVariation("All", False)
    except Exception:
        pass

    exc_holder      = [None]
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
    new_msgs = (messages_after[len(messages_before):]
                if len(messages_after) >= len(messages_before)
                else messages_after)
    if _has_solver_error(new_msgs):
        err_line = next(
            (m for m in new_msgs if any(k in m.lower() for k in _SOLVE_ERROR_KEYWORDS)),
            new_msgs[0] if new_msgs else "Unknown solver error"
        )
        raise RuntimeError(f"AEDT reported solve error: {err_line}")

    m2d.save_project()


def _is_number(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _extract_report_unit(lines):
    """Extract Y-axis unit string from AEDT CSV header, e.g. 'mNewtonMeter'."""
    for line in lines:
        parts = [p.strip().strip('"') for p in re.split(r'[,\t]', line)]
        for part in reversed(parts):
            m = re.search(r"\[([^\[\]]+)\]", part)
            if m:
                return m.group(1).strip()
        if any(_is_number(p) for p in parts):
            break
    return ""


def _export_report_csv(m2d, report_name):
    """
    Export an AEDT report to CSV and return ([all Y values], Y unit string).

    Uses odesign.GetModule("ReportSetup").ExportToFile() directly, which avoids
    PyAEDT re-initialising the post processor (and re-parsing the project file)
    on every call.  Falls back to m2d.post.export_report_to_csv() if needed.
    """
    proj_file = getattr(m2d, "project_file", getattr(m2d, "project_path", None))
    if proj_file is None:
        raise AttributeError("Cannot determine project directory.")

    proj_dir = os.path.dirname(proj_file)
    csv_path = os.path.join(proj_dir, f"{report_name}.csv")

    try:
        # Direct COM call - no post-processor re-init, no project file re-parse
        m2d.odesign.GetModule("ReportSetup").ExportToFile(report_name, csv_path)
    except Exception:
        # Fallback to PyAEDT high-level API
        result = m2d.post.export_report_to_csv(
            project_dir=proj_dir,
            plot_name=report_name,
        )
        if result:
            csv_path = result

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(
            f"CSV export failed for '{report_name}'. "
            "Verify the report exists in the AEDT Results tree."
        )

    with open(csv_path, encoding="utf-8-sig") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    unit   = _extract_report_unit(lines)
    values = []
    for line in lines:
        parts = [p.strip().strip('"') for p in re.split(r'[,\t]', line)]
        nums  = [float(p) for p in parts if _is_number(p)]
        if nums:
            values.append(nums[-1])

    if not values:
        raise ValueError(f"No numeric data found in {csv_path}")
    return values, unit


def _torque_to_nm(value, unit, report_name):
    """Normalize torque to Nm; supports Nm and mNm unit variants."""
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
        print(f"    [Warn] Unit missing in {report_name}; assuming Nm.", flush=True)
        return value
    raise ValueError(
        f"Unsupported torque unit '{unit}' in {report_name}. "
        "Please use Nm or mNm in AEDT report settings."
    )


def get_objectives(m2d):
    """Read the three AEDT reports and return (avg_torque, torque_ripple%, avg_eff)."""
    try:
        torque_vals, torque_unit = _export_report_csv(m2d, REPORT_AVG_TORQUE)
        ripple_vals, ripple_unit = _export_report_csv(m2d, REPORT_TORQUE_RIPPLE)
        eff_vals, _              = _export_report_csv(m2d, REPORT_EFFICIENCY)

        avg_torque = _torque_to_nm(torque_vals[-1], torque_unit, REPORT_AVG_TORQUE)
        ripple_nm  = _torque_to_nm(ripple_vals[-1], ripple_unit, REPORT_TORQUE_RIPPLE)
        avg_eff    = eff_vals[-1]
        ripple_pct = ripple_nm / avg_torque * 100.0 if avg_torque != 0 else 999.0

        print(f"    -> Torque={avg_torque:.2f} Nm, "
              f"Ripple={ripple_nm:.2f} Nm -> {ripple_pct:.2f}%, "
              f"Eff={avg_eff:.2f}%", flush=True)
        return avg_torque, ripple_pct, avg_eff

    except Exception as e:
        print(f"    [Error] Report CSV read failed: {e}", flush=True)
        return PENALTY_FITNESS


def _save_population(pop, generation, filepath, mode="a"):
    """
    Append (mode='a') or create (mode='w') a generation's population in CSV.
    is_valid=0 flags penalty rows (TorqueRipple >= 999).
    """
    rows = []
    for idx, ind in enumerate(pop):
        if not ind.fitness.valid:
            continue
        fv  = ind.fitness.values
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
    """Set variables -> simulate -> read objectives (used as serial fallback)."""
    print("    Variables: " + ", ".join(
        f"{v[0]}={x:.4f}{v[3]}" for v, x in zip(VARIABLES, individual)),
        flush=True)

    try:
        set_variables(m2d, individual)
    except AttributeError:
        global _m2d_global
        from ansys.aedt.core import Maxwell2d
        m2d = Maxwell2d(project=AEDT_PROJECT, design=AEDT_DESIGN,
                        non_graphical=NON_GRAPHICAL, new_desktop=NEW_DESKTOP)
        _m2d_global = m2d
        set_variables(m2d, individual)

    try:
        run_simulation(m2d)
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"    [Error] Simulation failed: {e}", flush=True)
        return PENALTY_FITNESS

    return get_objectives(m2d)


# ============================================================
# Section 5b - Batch Parallel Evaluation via Optimetrics
# ============================================================

_BATCH_SETUP_NAME = "NSGA_BatchSweep"


def _delete_batch_setup(m2d):
    try:
        if _BATCH_SETUP_NAME in list(m2d.ooptimetrics.GetSetupNames()):
            m2d.ooptimetrics.DeleteSetups([_BATCH_SETUP_NAME])
    except Exception:
        pass


def _aedt_var_unit(m2d, name):
    """Query the actual unit suffix of an AEDT design variable."""
    try:
        var_obj = m2d.variable_manager[name]
        # Try PyAEDT's .units attribute first (most reliable)
        for attr in ("units", "unit"):
            u = getattr(var_obj, attr, None)
            if u and isinstance(u, str) and u.lower() not in ("", "none"):
                return u
        # Fallback: extract trailing letters from expression, e.g. "0.5mm" -> "mm"
        expr = str(var_obj.expression).strip()
        m = re.search(r'([a-zA-Z]+)\s*$', expr)
        return m.group(1) if m else ""
    except Exception:
        return dict((v[0], v[3]) for v in VARIABLES).get(name, "")


def _create_batch_setup(m2d, individuals):
    """
    Register a synchronized discrete Optimetrics sweep for parallel solving.
    Format matches AEDT's native recorded-macro structure:
      Data = "val1 val2 val3" (no DIS prefix, values with unit suffix).
      Synchronize=1 pairs values row-by-row (no Cartesian product).
    No Goals registered — after the sweep completes, results are read by
    calling set_variables() + get_objectives() on each cached solution.
    """
    _delete_batch_setup(m2d)
    try:
        m2d.odesign.DeleteFullVariation("All", False)
    except Exception:
        pass

    sweep_defs = []
    for k, (name, _, _, _) in enumerate(VARIABLES):
        unit = _aedt_var_unit(m2d, name)
        vals_str = " ".join(
            (f"{ind[k]:.6f}{unit}" if unit else f"{ind[k]:.6f}")
            for ind in individuals
        )
        sweep_defs.append([
            "NAME:SweepDefinition",
            "Variable:=",    name,
            "Data:=",        vals_str,
            "OffsetF1:=",    False,
            "Synchronize:=", 1,
        ])

    m2d.ooptimetrics.InsertSetup("OptiParametric", [
        f"NAME:{_BATCH_SETUP_NAME}",
        "IsEnabled:=", True,
        ["NAME:ProdOptiSetupDataV2",
         "SaveFields:=", False, "CopyMesh:=", False,
         "SolveWithCopiedMeshOnly:=", False],
        "InterpolationPoints:=", 0,
        ["NAME:StartingPoint"],
        "Sim. Setups:=", [AEDT_SETUP],
        ["NAME:Sweeps"] + sweep_defs,
        ["NAME:Sweep Operations"],
        ["NAME:Goals"],      # empty — results read from cached solutions via set_variables()
    ])
    print("  [Batch] Setup created.", flush=True)


def _run_batch_setup(m2d):
    """Solve the batch parametric setup in a daemon thread (Ctrl+C safe)."""
    exc_holder = [None]

    def _solve():
        try:
            m2d.ooptimetrics.SolveSetup(_BATCH_SETUP_NAME)
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


def _collect_failed_variations(m2d, msg_offset=0):
    """
    After a batch sweep, parse AEDT messages to identify failed variations.

    msg_offset: number of messages present BEFORE this sweep started.
    Only messages from msgs[msg_offset:] are inspected, so errors from
    previous generations are never carried over to the current one.

    AEDT message order per variation:
      1. "A variation (Bs0=... Gap=...) has been requested"   <- start marker
      2. (geometry / mesh messages for this variation)
      3. Error messages if failed, e.g. "Surface Mesh Generation Failed"
      4. "A variation (...) has been requested"               <- next variation starts

    Uses a FORWARD-ONLY window: errors between request[i] and request[i+1]
    belong exclusively to variation i.

    Returns a list of dicts, e.g. [{"Gap": 0.82, "Bs0": 2.29, ...}, ...]
    """
    all_msgs = _desktop_messages(m2d)
    msgs = all_msgs[msg_offset:]   # only messages from this sweep onward
    if not msgs:
        return []

    # Collect all "A variation (...) has been requested" messages with index
    var_requests = []
    for i, msg in enumerate(msgs):
        m = re.search(
            r'variation\s*\(([^)]+)\)\s+has\s+been\s+requested', msg, re.IGNORECASE)
        if m:
            var_requests.append((i, m.group(1)))

    if not var_requests:
        return []

    failed = []
    seen   = set()
    for k, (req_idx, var_str) in enumerate(var_requests):
        if var_str in seen:
            continue

        # Forward window: from the line AFTER this request to the line BEFORE
        # the next request (exclusively belonging to this variation)
        next_req_idx = var_requests[k + 1][0] if k + 1 < len(var_requests) else len(msgs)
        window = msgs[req_idx + 1 : next_req_idx]

        if any(any(kw in msg.lower() for kw in _BATCH_FAIL_KEYWORDS) for msg in window):
            seen.add(var_str)
            vals = {}
            for name_str, val_str in re.findall(r"(\w+)\s*=\s*'([^']*)'", var_str):
                try:
                    vals[name_str] = float(re.sub(r'[a-zA-Z]+$', '', val_str))
                except ValueError:
                    pass
            if vals:
                failed.append(vals)

    return failed


def _ind_matches_failed(individual, failed_list):
    """Check if an individual's variable values match any failed variation."""
    for failed_vals in failed_list:
        if all(
            abs(individual[k] - failed_vals.get(name, float('inf'))) < 1e-4
            for k, (name, _, _, _) in enumerate(VARIABLES)
        ):
            return True
    return False


def _read_variation_result(m2d, individual, timeout=120):
    """
    Set design variables to navigate to the cached solution, then read objectives.

    After the batch parametric sweep all solutions are stored in the project.
    set_variables() at this point only selects the active cached variation —
    AEDT does not re-solve or recalculate geometry.

    Runs inside a daemon thread so Ctrl+C and timeouts are always handled.
    """
    result_holder = [None]
    exc_holder    = [None]

    def _worker():
        try:
            # After the batch sweep all solutions are cached.
            # set_variables() here only navigates to the stored solution;
            # it does NOT trigger a re-solve or geometry recalculation.
            set_variables(m2d, individual)
            result_holder[0] = get_objectives(m2d)
        except Exception as e:
            exc_holder[0] = e

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        raise TimeoutError(
            f"Result reading timed out after {timeout}s (AEDT unresponsive).")
    if _stop_flag:
        raise KeyboardInterrupt("Aborted by user.")
    if exc_holder[0] is not None:
        raise exc_holder[0]
    if result_holder[0] is None:
        raise ValueError("Worker completed but returned no result.")
    return result_holder[0]


def _parse_batch_results(m2d, individuals, n, msg_offset=0):
    """
    Read results for each solved parametric variation.
    Calls set_variables() + get_objectives() per individual; because all
    solutions are already cached from the batch sweep, AEDT only navigates
    to the stored result and does not re-solve.
    msg_offset: message count recorded just before this sweep was solved,
    passed through to _collect_failed_variations to ignore older messages.
    """
    failed_list = _collect_failed_variations(m2d, msg_offset)
    if failed_list:
        print(f"  [Batch] Detected {len(failed_list)} failed variation(s).", flush=True)

    results = []
    for i, ind in enumerate(individuals):
        if _stop_flag:
            print(f"\n  [Batch] Interrupted ({len(results)}/{n} read).", flush=True)
            break

        if _ind_matches_failed(ind, failed_list):
            print(f"    [{i+1}/{n}] FAILED (mesh/solver error) -> penalty", flush=True)
            results.append(PENALTY_FITNESS)
            continue

        print(f"    [{i+1}/{n}]", end=" ", flush=True)
        try:
            results.append(_read_variation_result(m2d, ind))
        except KeyboardInterrupt:
            print(f"\n  [Batch] Interrupted ({len(results)}/{n} read).", flush=True)
            break
        except Exception as e:
            print(f"[Error] {e}", flush=True)
            results.append(PENALTY_FITNESS)
    return results


def _serial_fallback(individuals, m2d):
    """Evaluate individuals one-by-one when batch sweep is unavailable."""
    results = []
    for i, ind in enumerate(individuals):
        print(f"\n  [Serial {i+1}/{len(individuals)}]", flush=True)
        try:
            results.append(evaluate_individual(ind, m2d))
        except KeyboardInterrupt:
            print(f"\n  [Serial] Interrupted after {len(results)}/{len(individuals)}.", flush=True)
            break          # return partial results, don't discard them
        except Exception as e:
            print(f"    [Error] {e}", flush=True)
            results.append(PENALTY_FITNESS)
    return results


def evaluate_population_batch(individuals, m2d):
    """
    Evaluate all individuals via one Optimetrics parametric sweep.
    Falls back to serial evaluation if the sweep setup or solve fails.
    """
    n = len(individuals)
    if n == 0:
        return []

    print(f"  [Batch] Creating sweep for {n} individuals...", flush=True)
    try:
        _create_batch_setup(m2d, individuals)
        # Snapshot message count BEFORE solving so _collect_failed_variations
        # only looks at messages produced by THIS sweep, not earlier generations.
        msg_offset = len(_desktop_messages(m2d))
        print("  [Batch] Solving...", flush=True)
        _run_batch_setup(m2d)
        m2d.save_project()
    except KeyboardInterrupt:
        raise
    except Exception as e:
        print(f"  [Batch] Sweep failed: {e}  ->  falling back to serial.", flush=True)
        return _serial_fallback(individuals, m2d)

    print("  [Batch] Reading results...", flush=True)
    try:
        return _parse_batch_results(m2d, individuals, n, msg_offset)
    except Exception as e:
        print(f"  [Batch] Result reading failed: {e}  ->  falling back to serial.", flush=True)
        return _serial_fallback(individuals, m2d)


# ============================================================
# Section 6 - Main Optimization Loop
# ============================================================

def main():
    print("\n[Step 1] Connecting to AEDT...", flush=True)
    from ansys.aedt.core import Maxwell2d
    m2d = Maxwell2d(project=AEDT_PROJECT, design=AEDT_DESIGN,
                    non_graphical=NON_GRAPHICAL, new_desktop=NEW_DESKTOP)
    global _m2d_global
    _m2d_global = m2d
    print(f"  Connected: {m2d.project_name} / {m2d.design_name}", flush=True)

    print(f"\n[Step 2] Initializing population with LHS (size={MU})...", flush=True)
    pop = _lhs_population(MU)

    print("\n[Step 3] Evaluating (Ctrl+C to stop and save at any time)...", flush=True)
    invalid  = [ind for ind in pop if not ind.fitness.valid]
    last_gen = 0

    try:
        print(f"\n  [Gen 0] Evaluating {len(invalid)} individuals...", flush=True)
        batch_results = evaluate_population_batch(invalid, m2d)
        for ind, result in zip(invalid, batch_results):
            ind.fitness.values = result
        if _stop_flag:
            raise KeyboardInterrupt("Stopped after partial evaluation.")
        _save_population(pop, generation=0, filepath=OUTPUT_ALL_GEN, mode="w")
        print(f"  [Gen 0] Done -> {OUTPUT_ALL_GEN}", flush=True)

        for gen in range(1, NGEN + 1):
            print(f"\n{'='*55}\n  Generation {gen}/{NGEN}\n{'='*55}", flush=True)
            offspring = algorithms.varAnd(pop, toolbox, CXPB, MUTPB)
            invalid   = [ind for ind in offspring if not ind.fitness.valid]
            batch_results = evaluate_population_batch(invalid, m2d)
            for ind, result in zip(invalid, batch_results):
                ind.fitness.values = result
            if _stop_flag:
                raise KeyboardInterrupt("Stopped after partial evaluation.")
            pop = toolbox.select(pop + offspring, MU)
            _save_population(pop, generation=gen, filepath=OUTPUT_ALL_GEN, mode="a")
            print(f"  [Gen {gen}] Done -> {OUTPUT_ALL_GEN}", flush=True)
            last_gen = gen

        print("\n[Optimization finished]", flush=True)

    except KeyboardInterrupt:
        print("\n[Interrupted -> saving results so far...]", flush=True)
        pop = [ind for ind in pop if ind.fitness.valid]

    hdr = (f"{'Gap':>7} {'Polearc':>9} {'Mag_th':>8} {'Bs0':>6} | "
           f"{'Torque':>8} {'Ripple%':>8} {'Eff%':>7}")
    print(f"\n{hdr}\n{'-'*len(hdr)}", flush=True)
    for ind in pop:
        if not ind.fitness.valid:
            continue
        fv = ind.fitness.values
        print(f"{ind[0]:7.4f} {ind[1]:9.4f} {ind[2]:8.4f} "
              f"{ind[3]:6.4f} | {fv[0]:8.4f} {fv[1]:8.4f} {fv[2]:7.4f}")

    m2d.save_project()
    print("\n  Project saved. AEDT remains open.", flush=True)
    return pop, last_gen


# ============================================================
# Section 7 - Run
# ============================================================

if __name__ == "__main__":
    final_pop, last_gen = main()

    if not final_pop:
        exit(0)

    # Save final Pareto front to CSV
    final_pf = tools.sortNondominated(final_pop, len(final_pop),
                                      first_front_only=True)[0]
    print(f"\n  Final Pareto front: {len(final_pf)} / {len(final_pop)} individuals",
          flush=True)
    _save_population(final_pf, generation=last_gen, filepath=OUTPUT_PARETO, mode="w")
    print(f"  Saved -> {OUTPUT_PARETO}", flush=True)
