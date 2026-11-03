"""demo.py – Phase 1 demonstration script for Person 3.

Shows:
1. All configuration details (automaton parameters, world rules, initial state).
2. Generation-by-generation evolution (in console with counts/grid + Matplotlib plots).
3. The 10+10=20 state recovery test.
4. Storage verification (saved runs directory).

Usage:
    python demo.py
    python demo.py automata/brians_brain.json experiments/brians_brain_random.json
    python demo.py automata/rule_30.json experiments/rule_30_center.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _load_config(exp_path: Path, project_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Validate and load the experiment (and its automaton) with cellab.loader.load_files.

    Prints every issue found. Stops the demo if there is any error.
    """
    from cellab.loader import load_files

    rel_exp = str(exp_path.relative_to(project_root)).replace("\\", "/")
    res = load_files(rel_exp, project_root=project_root)

    issues = res["automaton_issues"] + res["experiment_issues"]
    for issue in issues:
        print(f"  [{issue['level']}] [{issue['code']}] {issue['where']}: {issue['message']}")

    if res["automaton"] is None or res["experiment"] is None:
        raise ValueError("Validation failed")

    print("✓ Schema and rule validation: OK (0 errors)")
    return res["automaton"], res["experiment"]


def _code_to_symbol(automaton: dict[str, Any]) -> dict[int, str]:
    """Return a mapping from state code to its symbol character."""
    result: dict[int, str] = {}
    for symbol, code in automaton["symbols"].items():
        result[code] = symbol
    return result


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
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    from cellab.initial_state import create_initial_state
    from cellab.simulation import run
    from cellab.storage import save_run, load_run

    project_root = Path(__file__).resolve().parent

    # Pick files
    if len(sys.argv) == 3:
        aut_path = project_root / sys.argv[1]
        exp_path = project_root / sys.argv[2]
    else:
        aut_path = project_root / "automata" / "forest_fire.json"
        exp_path = project_root / "experiments" / "forest_fire_manual.json"

    with open(aut_path, encoding="utf-8") as f:
        raw_aut = json.load(f)
    with open(exp_path, encoding="utf-8") as f:
        raw_exp = json.load(f)

    # -----------------------------------------------------------------------
    # 1. Configuration details
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("              EXPERIMENT AND AUTOMATON DETAILS")
    print("=" * 70)
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
    print("=" * 70)
    print()

    # Step 1 + 2: Validate and load automaton + experiment
    automaton, experiment = _load_config(exp_path, project_root)

    # Step 3: Build initial state
    raw_seed = raw_exp.get("seed")
    effective_seed = 0 if raw_seed is None else int(raw_seed)
    rng = np.random.default_rng(effective_seed)
    state0 = create_initial_state(automaton, experiment, rng)
    print(f"\n✓ Initial state created (Generation 0, shape {state0['cells'].shape}).")

    # Step 4: Run full simulation
    max_generations = 100_000
    generations = int(experiment["generations"])
    if not 0 <= generations <= max_generations:
        raise ValueError(f"generations out of bounds [0, {max_generations}]: {generations}")

    storage = experiment.get("storage", {"mode": "all"})
    result = run(automaton, state0, generations, storage, rng=rng)

    # -----------------------------------------------------------------------
    # 2. Generation-by-generation evolution
    # -----------------------------------------------------------------------
    all_states = result["states"]
    print("\n" + "=" * 70)
    print(f"         GENERATION EVOLUTION (Total stored: {len(all_states)})")
    print("=" * 70)

    rows, cols = world_cfg["rows"], world_cfg["cols"]
    is_small_grid = (rows <= 10 and cols <= 30) or (rows == 1)

    for st in all_states:
        gen_i = st.get("generation", 0)
        cells = st["cells"]

        counts = []
        for sname, scode in automaton["state_codes"].items():
            cnt = int(np.sum(cells == scode))
            if cnt > 0:
                counts.append(f"{sname}={cnt}")

        counts_str = ", ".join(counts)
        print(f"\n▶ Generation {gen_i:2d} | Cell counts: {counts_str}")

        if is_small_grid:
            code_to_sym = _code_to_symbol(automaton)
            cell_rows, cell_cols = cells.shape
            for r in range(cell_rows):
                line = "".join(code_to_sym.get(int(cells[r, c]), "?") for c in range(cell_cols))
                print(f"   {line}")

    # -----------------------------------------------------------------------
    # 3. 10+10=20 state recovery test (Save, Reload, Resume)
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("      STORAGE VERIFICATION, RECOVERY AND 10+10=20 TEST")
    print("=" * 70)
    half = min(10, generations)
    rest = min(generations - half, 10)

    # Run first half with dedicated RNG
    rng_split = np.random.default_rng(effective_seed)
    s0_split = create_initial_state(automaton, experiment, rng_split)
    res_half = run(automaton, s0_split, half, storage, rng=rng_split)

    name_slug = raw_aut["name"].lower().replace(" ", "_").replace("'", "")
    run_dir = save_run(
        name=name_slug,
        automaton_raw=raw_aut,
        experiment_raw=raw_exp,
        run_result=res_half,
    )
    print(f"• Intermediate run ({half} steps) saved to:\n  {run_dir}")

    # Reload
    loaded = load_run(run_dir)
    print(f"• Reloaded successfully! Last saved generation = {loaded['meta']['last_generation']}")
    print("• RNG state restored identically.")

    # Continue remaining steps
    res_continued = run(automaton, loaded["last_state"], rest, storage, rng=loaded["rng"])
    print(f"• Simulated {rest} more steps from reloaded state (total: gen {res_continued['last_state']['generation']}).")

    # Compare with uninterrupted run
    rng_cont = np.random.default_rng(effective_seed)
    s0_cont = create_initial_state(automaton, experiment, rng_cont)
    res_direct = run(automaton, s0_cont, half + rest, storage, rng=rng_cont)

    match = np.array_equal(res_continued["last_state"]["cells"], res_direct["last_state"]["cells"])
    print(f"• Recovery test result ({half} + {rest} == {half + rest}): {'PASS ✓' if match else 'FAIL ✗'}")

    # -----------------------------------------------------------------------
    # 4. Visual display with Matplotlib
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("                  VISUAL DISPLAY (MATPLOTLIB)")
    print("=" * 70)
    print("Opening panel displaying intermediate generations evolution...")
    show_evolution(all_states, automaton, max_plots=12, save_path=run_dir / "evolution.png")
    print("Demo completed successfully!\n")


if __name__ == "__main__":
    main()
