"""demo_cli.py – Command-line interface with parameter overrides.

Wraps the existing demo logic, adding argparse-based CLI flags that let the
user override experiment parameters without editing JSON files.

Usage examples:
    python demo_cli.py experiments/brians_brain_random.json
    python demo_cli.py experiments/brians_brain_random.json --generations 50
    python demo_cli.py experiments/brians_brain_random.json --rows 80 --cols 80 --seed 123
    python demo_cli.py experiments/forest_fire_stochastic_random.json --param p=0.1 --generations 300
    python demo_cli.py experiments/cyclic_random.json --boundary toroidal --storage last
    python demo_cli.py --resume runs/<run_folder> 15
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

from cellab.initial_state import create_initial_state
from cellab.loader import load_automaton, load_experiment, load_files, read_json, safe_path
from cellab.simulation import run
from cellab.storage import load_run, save_run
from cellab.validator import has_errors, validate_automaton, validate_experiment

from demo import (
    DEFAULT_EXPERIMENT,
    LINE,
    MAX_GENERATIONS,
    PROJECT_ROOT,
    _print_details,
    _print_issues,
    _print_states,
    _recovery_test,
    _seed_of,
    show_evolution,
)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """Build the argparse parser with all supported overrides."""
    parser = argparse.ArgumentParser(
        prog="demo_cli",
        description="Run a CelLab experiment with optional parameter overrides.",
        epilog=(
            "Examples:\n"
            "  python demo_cli.py experiments/brians_brain_random.json\n"
            "  python demo_cli.py experiments/brians_brain_random.json --generations 50 --seed 7\n"
            "  python demo_cli.py experiments/forest_fire_stochastic_random.json --rows 100 --cols 100\n"
            "  python demo_cli.py experiments/forest_fire_stochastic_random.json --param p=0.1\n"
            "  python demo_cli.py --resume runs/forest_fire_20261105_001 15\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Positional: experiment file (optional, has default)
    parser.add_argument(
        "experiment",
        nargs="?",
        default=DEFAULT_EXPERIMENT,
        help=f"Path to the experiment JSON file (default: {DEFAULT_EXPERIMENT})",
    )

    # World overrides
    world = parser.add_argument_group("world overrides")
    world.add_argument("--rows", type=int, help="Number of grid rows")
    world.add_argument("--cols", type=int, help="Number of grid columns")
    world.add_argument(
        "--boundary",
        choices=["fixed", "toroidal"],
        help="Boundary type (fixed or toroidal)",
    )

    # Simulation overrides
    sim = parser.add_argument_group("simulation overrides")
    sim.add_argument("-g", "--generations", type=int, help="Number of generations to simulate")
    sim.add_argument("-s", "--seed", type=int, help="Random seed for reproducibility")
    sim.add_argument(
        "--storage",
        choices=["all", "last", "every"],
        help="Storage mode",
    )
    sim.add_argument(
        "--every",
        type=int,
        help="Store every N generations (only with --storage every)",
    )

    # Automaton parameter overrides
    sim.add_argument(
        "--param",
        action="append",
        metavar="NAME=VALUE",
        help="Override an automaton parameter (e.g. --param p=0.08). Can be repeated.",
    )

    # Resume mode
    resume = parser.add_argument_group("resume a saved run")
    resume.add_argument(
        "--resume",
        metavar="RUN_DIR",
        help="Path to a saved run folder to resume",
    )
    resume.add_argument(
        "--extra",
        type=int,
        default=10,
        metavar="N",
        help="Extra generations to simulate when resuming (default: 10)",
    )

    # Flags
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip the Matplotlib visualization window",
    )
    parser.add_argument(
        "--no-recovery-test",
        action="store_true",
        help="Skip the save-reload-continue recovery test",
    )

    return parser


# ---------------------------------------------------------------------------
# Override helpers
# ---------------------------------------------------------------------------

def _parse_params(param_list: list[str] | None) -> dict[str, float]:
    """Parse --param NAME=VALUE arguments into a dict."""
    if not param_list:
        return {}
    result: dict[str, float] = {}
    for item in param_list:
        if "=" not in item:
            raise ValueError(f"Invalid --param format: '{item}' (expected NAME=VALUE)")
        name, value_str = item.split("=", 1)
        try:
            result[name.strip()] = float(value_str.strip())
        except ValueError:
            raise ValueError(f"Invalid --param value: '{item}' (value must be a number)")
    return result


def _apply_overrides(raw_exp: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """Apply CLI overrides to the raw experiment dict. Returns a list of applied changes."""
    changes: list[str] = []

    if args.rows is not None:
        raw_exp["world"]["rows"] = args.rows
        changes.append(f"rows={args.rows}")

    if args.cols is not None:
        raw_exp["world"]["cols"] = args.cols
        changes.append(f"cols={args.cols}")

    if args.boundary is not None:
        raw_exp["world"]["boundary"] = args.boundary
        if args.boundary == "toroidal" and "outside" in raw_exp["world"]:
            del raw_exp["world"]["outside"]
        changes.append(f"boundary={args.boundary}")

    if args.generations is not None:
        raw_exp["generations"] = args.generations
        changes.append(f"generations={args.generations}")

    if args.seed is not None:
        raw_exp["seed"] = args.seed
        changes.append(f"seed={args.seed}")

    if args.storage is not None:
        raw_exp["storage"]["mode"] = args.storage
        if args.storage != "every" and "every" in raw_exp["storage"]:
            del raw_exp["storage"]["every"]
        changes.append(f"storage={args.storage}")

    if args.every is not None:
        raw_exp["storage"]["mode"] = "every"
        raw_exp["storage"]["every"] = args.every
        changes.append(f"every={args.every}")

    param_overrides = _parse_params(args.param)
    if param_overrides:
        if "parameters" not in raw_exp:
            raw_exp["parameters"] = {}
        raw_exp["parameters"].update(param_overrides)
        for k, v in param_overrides.items():
            changes.append(f"param {k}={v}")

    # Handle conflict: manual grid mode + changed dimensions
    if (args.rows is not None or args.cols is not None):
        cell_mode = raw_exp.get("initial_state", {}).get("cells", {}).get("mode")
        if cell_mode == "manual":
            print(f"⚠ Warning: changing grid dimensions with mode 'manual' — "
                  f"switching initial state to mode 'fill' with the first state.")
            raw_exp["initial_state"]["cells"] = {"mode": "fill", "state": "_FIRST_"}
            changes.append("initial_state→fill (auto)")

    return changes


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

def _load_and_validate(exp_path: str) -> tuple[dict, dict, dict, dict]:
    """Load and validate experiment + automaton via load_files."""
    res = load_files(exp_path, project_root=PROJECT_ROOT)
    _print_issues(res["automaton_issues"] + res["experiment_issues"])
    if res["automaton"] is None or res["experiment"] is None:
        raise ValueError("Validation failed")
    print("✓ Schema and rule validation: OK (0 errors)")

    raw_exp, _ = read_json(exp_path, PROJECT_ROOT)
    raw_aut, _ = read_json(raw_exp["automaton"], PROJECT_ROOT)
    return res["automaton"], res["experiment"], raw_aut, raw_exp


def _revalidate(raw_aut: dict, raw_exp: dict) -> tuple[dict, dict]:
    """Re-validate and re-load after overrides have been applied."""
    issues = validate_automaton(raw_aut)
    if not has_errors(issues):
        issues = issues + validate_experiment(raw_exp, raw_aut)
    _print_issues(issues)
    if has_errors(issues):
        raise ValueError("Validation failed after applying CLI overrides")
    automaton = load_automaton(raw_aut, raw_exp)
    experiment = load_experiment(raw_exp, automaton)
    return automaton, experiment


def run_experiment_cli(args: argparse.Namespace) -> None:
    """Load an experiment, apply CLI overrides, simulate, save and optionally test recovery."""
    exp_path = args.experiment

    # First load to get raw dicts and validate the base file
    raw_exp, problem = read_json(exp_path, PROJECT_ROOT)
    if problem is not None:
        _print_issues([problem])
        raise ValueError("Cannot read experiment file")
    raw_aut, problem = read_json(raw_exp["automaton"], PROJECT_ROOT)
    if problem is not None:
        _print_issues([problem])
        raise ValueError("Cannot read automaton file")

    # Apply overrides before validation
    changes = _apply_overrides(raw_exp, args)

    # Fix the fill placeholder: use the first state from the automaton
    cells_cfg = raw_exp.get("initial_state", {}).get("cells", {})
    if cells_cfg.get("state") == "_FIRST_" and cells_cfg.get("mode") == "fill":
        first_state = raw_aut["states"][0]["name"]
        cells_cfg["state"] = first_state

    if changes:
        print(f"• CLI overrides applied: {', '.join(changes)}")

    # Validate the (possibly modified) experiment
    automaton, experiment = _revalidate(raw_aut, raw_exp)
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
    print(f"  Continue it with: python demo_cli.py --resume {run_dir.relative_to(PROJECT_ROOT).as_posix()} --extra 10")

    if not args.no_recovery_test:
        _recovery_test(automaton, experiment, raw_aut, raw_exp)

    if not args.no_plot:
        print("\nOpening panel displaying intermediate generations evolution...")
        show_evolution(result["states"], automaton, max_plots=12, save_path=run_dir / "evolution.png")

    print("Demo completed successfully!\n")


def resume_experiment_cli(args: argparse.Namespace) -> None:
    """Reload a saved run and simulate more generations."""
    run_path = safe_path(args.resume, PROJECT_ROOT)
    if run_path is None:
        raise ValueError(f"The run folder '{args.resume}' must be inside the project folder")
    loaded = load_run(run_path)

    raw_aut = loaded["automaton_raw"]
    raw_exp = loaded["experiment_raw"]
    automaton, experiment = _revalidate(raw_aut, raw_exp)
    _print_details(raw_aut, raw_exp)

    extra = args.extra
    start = loaded["last_state"]["generation"]
    print(f"✓ Reloaded {args.resume} at generation {start}; simulating {extra} more.")
    result = run(automaton, loaded["last_state"], extra, experiment["storage"], rng=loaded["rng"])
    _print_states(result["states"], automaton)

    new_dir = save_run(name=raw_aut["name"], automaton_raw=raw_aut,
                       experiment_raw=raw_exp, run_result=result)
    print(f"\n• Continued run (generations {start}..{start + extra}) saved to:\n  {new_dir}")

    if not args.no_plot:
        show_evolution(result["states"], automaton, max_plots=12, save_path=new_dir / "evolution.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    """Parse arguments and dispatch to the right command."""
    parser = _build_parser()
    args = parser.parse_args()

    try:
        if args.resume:
            resume_experiment_cli(args)
        else:
            run_experiment_cli(args)
    except NotImplementedError as error:
        print(f"\nNot supported yet: {error}")
        return 1
    except (ValueError, FileNotFoundError) as error:
        print(f"\nStopped: {error}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
