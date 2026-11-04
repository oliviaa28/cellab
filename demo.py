"""demo.py – Phase 1 demonstration script.

Shows:
1. All configuration details (automaton parameters, world rules, initial state).
2. Generation-by-generation evolution (in console with counts/grid + Matplotlib plots).
3. Storage: the full run is saved in runs/<name>_<date>_<nr>/.
4. The 10+10=20 state recovery test.

Usage:
    python demo.py
    python demo.py experiments/brians_brain_random.json
    python demo.py experiments/rule_30_center.json
    python demo.py --resume runs/<run folder> 10     (continue a saved run for 10 more generations)
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from cellab.initial_state import create_initial_state
from cellab.loader import load_automaton, load_experiment, load_files, read_json, safe_path
from cellab.simulation import run
from cellab.storage import load_run, save_run
from cellab.validator import has_errors, validate_automaton, validate_experiment

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_EXPERIMENT = "experiments/forest_fire_statement.json"
MAX_GENERATIONS = 100_000
LINE = "=" * 70


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _print_issues(issues: list[dict[str, Any]]) -> None:
    """Print every validation issue (errors and warnings)."""
    for issue in issues:
        print(f"  [{issue['level']}] [{issue['code']}] {issue['where']}: {issue['message']}")


def _load_config(exp_path: str) -> tuple[dict, dict, dict, dict]:
    """Validate and load an experiment and its automaton with cellab.loader.load_files.

    Returns (automaton, experiment, raw_automaton, raw_experiment).
    Prints every issue found. Stops the demo if there is any error.
    """
    res = load_files(exp_path, project_root=PROJECT_ROOT)
    _print_issues(res["automaton_issues"] + res["experiment_issues"])
    if res["automaton"] is None or res["experiment"] is None:
        raise ValueError("Validation failed")
    print("✓ Schema and rule validation: OK (0 errors)")

    # load_files already checked both files; read them again, raw, for printing and saving
    raw_exp, _ = read_json(exp_path, PROJECT_ROOT)
    raw_aut, _ = read_json(raw_exp["automaton"], PROJECT_ROOT)
    return res["automaton"], res["experiment"], raw_aut, raw_exp


def _load_saved_files(raw_aut: dict, raw_exp: dict) -> tuple[dict, dict]:
    """Validate and load the automaton and experiment copies stored in a saved run."""
    issues = validate_automaton(raw_aut)
    if not has_errors(issues):
        issues = issues + validate_experiment(raw_exp, raw_aut)
    _print_issues(issues)
    if has_errors(issues):
        raise ValueError("The saved run contains invalid files")

    automaton = load_automaton(raw_aut, raw_exp)
    experiment = load_experiment(raw_exp, automaton)
    return automaton, experiment


def _code_to_symbol(automaton: dict[str, Any]) -> dict[int, str]:
    """Return a mapping from state code to its symbol character."""
    result: dict[int, str] = {}
    for symbol, code in automaton["symbols"].items():
        result[code] = symbol
    return result


def _seed_of(experiment: dict[str, Any]) -> int:
    """The experiment seed (0 for deterministic experiments without a seed)."""
    if experiment["seed"] is None:
        return 0
    return int(experiment["seed"])


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

def _print_details(raw_aut: dict[str, Any], raw_exp: dict[str, Any]) -> None:
    """Print the automaton and the experiment settings."""
    print(LINE)
    print("              EXPERIMENT AND AUTOMATON DETAILS")
    print(LINE)
    print(f"• Automaton name    : {raw_aut['name']}")
    if raw_aut.get("description"):
        print(f"• Description       : {raw_aut['description']}")
    print(f"• World kind        : {raw_aut.get('kind', 'cells')}")
    print(f"• Neighborhood      : {raw_aut['neighborhood']['type']} (radius={raw_aut['neighborhood'].get('radius', 1)})")
    st_info = [f"{s['name']} (symbol '{s['symbol']}', color {s.get('color', 'N/A')})" for s in raw_aut['states']]
    print(f"• Available states  : {', '.join(st_info)}")

    if raw_aut.get("parameters") or raw_exp.get("parameters"):
        params = dict(raw_aut.get("parameters", {}))
        params.update(raw_exp.get("parameters", {}))
        print(f"• Active parameters : {params}")

    print(f"• Rules ({len(raw_aut['rules'])} rules):")
    for i, r in enumerate(raw_aut['rules'], 1):
        when_str = f" when {r['when']}" if "when" in r else ""
        prob_str = f" (probability {r['probability']})" if "probability" in r else ""
        print(f"   [{i}] From '{r.get('from')}' -> '{r.get('to', '(same)')}'{when_str}{prob_str}")

    world_cfg = raw_exp["world"]
    print(f"• Grid dimensions   : {world_cfg['rows']} rows x {world_cfg['cols']} cols")
    boundary_info = world_cfg["boundary"]
    if boundary_info == "fixed" and "outside" in world_cfg:
        boundary_info += f" (outside state: {world_cfg['outside'].get('state')})"
    print(f"• Boundary type     : {boundary_info}")

    init_cfg = raw_exp["initial_state"]["cells"]
    print(f"• Initial state     : mode '{init_cfg['mode']}'")
    print(f"• Total generations : {raw_exp['generations']}")
    print(f"• Storage mode      : {raw_exp.get('storage', {'mode': 'all'})}")
    if "seed" in raw_exp:
        print(f"• Random seed       : {raw_exp['seed']}")
    print(LINE)
    print()


def _print_states(states: list[dict[str, Any]], automaton: dict[str, Any]) -> None:
    """Print the cell counts of every stored generation, and the grid if it is small."""
    print("\n" + LINE)
    print(f"         GENERATION EVOLUTION (Total stored: {len(states)})")
    print(LINE)

    code_to_sym = _code_to_symbol(automaton)
    for st in states:
        cells = st["cells"]
        counts = []
        for sname, scode in automaton["state_codes"].items():
            cnt = int(np.sum(cells == scode))
            if cnt > 0:
                counts.append(f"{sname}={cnt}")
        print(f"\n▶ Generation {st['generation']:2d} | Cell counts: {', '.join(counts)}")

        cell_rows, cell_cols = cells.shape
        if (cell_rows <= 10 and cell_cols <= 30) or cell_rows == 1:
            for r in range(cell_rows):
                line = "".join(code_to_sym.get(int(cells[r, c]), "?") for c in range(cell_cols))
                print(f"   {line}")


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def show_evolution(
    states: list[dict[str, Any]],
    automaton: dict[str, Any],
    max_plots: int = 12,
    save_path: Path | None = None,
) -> None:
    """Display multiple intermediate generations side by side in Matplotlib."""
    n_states = len(automaton["state_names"])
    hex_colors = [automaton["colors"][i] or "#888888" for i in range(n_states)]
    cmap = mcolors.ListedColormap(hex_colors)

    # Pick a representative sample of generations to display
    total_stored = len(states)
    if total_stored <= max_plots:
        selected = states
    else:
        indices = np.linspace(0, total_stored - 1, max_plots, dtype=int)
        selected = [states[i] for i in indices]

    num_to_show = len(selected)
    cols = min(4, num_to_show)
    rows = (num_to_show + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(3.5 * cols, 3.5 * rows))
    if num_to_show == 1:
        axes = np.array([axes])
    axes = np.ravel(axes)

    for idx, st in enumerate(selected):
        ax = axes[idx]
        cells = np.asarray(st["cells"])
        ax.imshow(cells, cmap=cmap, vmin=0, vmax=n_states - 1, interpolation="nearest")
        gen_num = st.get("generation", idx)
        ax.set_title(f"Generation {gen_num}", fontsize=11, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    for idx in range(num_to_show, len(axes)):
        axes[idx].axis("off")

    plt.suptitle(f"Automaton Evolution: {automaton['name']}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
        print(f"• Plot image saved to: {save_path}")

    if plt.get_backend().lower() not in ("agg", "template", "ps", "pdf", "svg"):
        plt.show()  # Keeps window open until closed by user
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Recovery test
# ---------------------------------------------------------------------------

def _recovery_test(automaton: dict, experiment: dict, raw_aut: dict, raw_exp: dict) -> None:
    """Run N steps, save, reload, run M more steps and compare with N+M steps in one go."""
    print("\n" + LINE)
    print("             RECOVERY TEST (SAVE, RELOAD, CONTINUE)")
    print(LINE)
    generations = int(experiment["generations"])
    storage = experiment["storage"]
    half = min(10, generations)
    rest = min(generations - half, 10)

    rng_split = np.random.default_rng(_seed_of(experiment))
    s0_split = create_initial_state(automaton, experiment, rng_split)
    res_half = run(automaton, s0_split, half, storage, rng=rng_split)
    half_dir = save_run(name=raw_aut["name"] + "_half", automaton_raw=raw_aut,
                        experiment_raw=raw_exp, run_result=res_half)

    loaded = load_run(half_dir)
    print(f"• {half} steps saved and reloaded (last saved generation = {loaded['meta']['last_generation']})")
    res_continued = run(automaton, loaded["last_state"], rest, storage, rng=loaded["rng"])
    shutil.rmtree(half_dir, ignore_errors=True)

    rng_cont = np.random.default_rng(_seed_of(experiment))
    s0_cont = create_initial_state(automaton, experiment, rng_cont)
    res_direct = run(automaton, s0_cont, half + rest, storage, rng=rng_cont)

    match = np.array_equal(res_continued["last_state"]["cells"], res_direct["last_state"]["cells"])
    print(f"• Recovery test result ({half} + {rest} == {half + rest}): {'PASS ✓' if match else 'FAIL ✗'}")


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def run_experiment(exp_path: str) -> None:
    """Load an experiment, simulate it, save the full run and run the recovery test."""
    automaton, experiment, raw_aut, raw_exp = _load_config(exp_path)
    _print_details(raw_aut, raw_exp)

    generations = int(experiment["generations"])
    if not 0 <= generations <= MAX_GENERATIONS:
        raise ValueError(f"generations out of bounds [0, {MAX_GENERATIONS}]: {generations}")

    rng = np.random.default_rng(_seed_of(experiment))
    state0 = create_initial_state(automaton, experiment, rng)
    print(f"✓ Initial state created (Generation 0, shape {state0['cells'].shape}).")
    result = run(automaton, state0, generations, experiment["storage"], rng=rng)
    _print_states(result["states"], automaton)

    run_dir = save_run(name=raw_aut["name"], automaton_raw=raw_aut,
                       experiment_raw=raw_exp, run_result=result)
    print(f"\n• Full run ({generations} generations) saved to:\n  {run_dir}")
    print(f"  Continue it with: python demo.py --resume {run_dir.relative_to(PROJECT_ROOT).as_posix()} 10")

    _recovery_test(automaton, experiment, raw_aut, raw_exp)

    print("\nOpening panel displaying intermediate generations evolution...")
    show_evolution(result["states"], automaton, max_plots=12, save_path=run_dir / "evolution.png")
    print("Demo completed successfully!\n")


def resume_experiment(run_dir: str, extra_generations: int) -> None:
    """Reload a saved run and simulate more generations from its last state."""
    run_path = safe_path(run_dir, PROJECT_ROOT)
    if run_path is None:
        raise ValueError(f"The run folder '{run_dir}' must be inside the project folder")
    loaded = load_run(run_path)
    automaton, experiment = _load_saved_files(loaded["automaton_raw"], loaded["experiment_raw"])
    _print_details(loaded["automaton_raw"], loaded["experiment_raw"])

    start = loaded["last_state"]["generation"]
    print(f"✓ Reloaded {run_dir} at generation {start}; simulating {extra_generations} more.")
    result = run(automaton, loaded["last_state"], extra_generations, experiment["storage"], rng=loaded["rng"])
    _print_states(result["states"], automaton)

    new_dir = save_run(name=loaded["automaton_raw"]["name"], automaton_raw=loaded["automaton_raw"],
                       experiment_raw=loaded["experiment_raw"], run_result=result)
    print(f"\n• Continued run (generations {start}..{start + extra_generations}) saved to:\n  {new_dir}")
    show_evolution(result["states"], automaton, max_plots=12, save_path=new_dir / "evolution.png")


def main(argv: list[str]) -> int:
    """Command line entry point. Returns the exit code."""
    try:
        if len(argv) >= 1 and argv[0] == "--resume":
            if len(argv) != 3:
                print("Usage: python demo.py --resume runs/<run folder> <generations>")
                return 2
            resume_experiment(argv[1], int(argv[2]))
        elif len(argv) <= 1:
            run_experiment(argv[0] if argv else DEFAULT_EXPERIMENT)
        else:
            print("Usage: python demo.py [experiments/<file>.json]")
            return 2
    except NotImplementedError as error:
        print(f"\nNot supported yet: {error}")
        return 1
    except (ValueError, FileNotFoundError) as error:
        print(f"\nStopped: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
