"""storage.py – Save, load, and resume simulation runs.

Phase 1 storage layout (per section 6.7 of the contract):

    runs/<name>_<date>_<nr>/
        automaton.json     – copy of the automaton (for reproducibility)
        experiment.json    – the experiment parameters
        generations.npz    – stored generations: cells + each variable
        last_state.npz     – last state, saved ALWAYS
        meta.json          – format_version, created, last_generation,
                             stored_generations, rng_state

Agents (agents.json) are Phase 3.
Fingerprints are Phase 2.
"""

from __future__ import annotations

import datetime
import json
import re
from pathlib import Path
from typing import Any

import numpy as np

# Use shared project root from conventions.py if available (contract 5.2)
try:
    from cellab.conventions import PROJECT_ROOT
except (ImportError, ModuleNotFoundError):
    PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """Return the project root folder."""
    return PROJECT_ROOT


def _sanitize_name(name: str) -> str:
    """Sanitize name to keep run directory names clean and safe."""
    clean = re.sub(r"[^a-zA-Z0-9_\-]", "_", name.strip())
    return clean or "run"


def _next_run_dir(name: str) -> Path:
    """Create a unique run directory under runs/."""
    runs = _project_root() / "runs"
    runs.mkdir(exist_ok=True)
    date_str = datetime.datetime.now().strftime("%Y%m%d")
    clean_name = _sanitize_name(name)
    nr = 1
    while True:
        dirname = f"{clean_name}_{date_str}_{nr:03d}"
        path = runs / dirname
        if not path.exists():
            path.mkdir(parents=True)
            return path
        nr += 1


def _serialize_rng_state(rng: np.random.Generator) -> dict[str, Any]:
    """Extract the full RNG state as JSON-serializable data."""
    bg_state = rng.bit_generator.state
    result: dict[str, Any] = {"bit_generator": bg_state["bit_generator"]}
    s = bg_state["state"]
    serialised_state: dict[str, Any] = {}
    for k, v in s.items():
        if isinstance(v, np.ndarray):
            serialised_state[k] = v.tolist()
        elif hasattr(v, "item"):
            serialised_state[k] = v.item()
        else:
            serialised_state[k] = v
    result["state"] = serialised_state
    if "has_uint32" in bg_state:
        result["has_uint32"] = int(bg_state["has_uint32"])
    if "uinteger" in bg_state:
        result["uinteger"] = int(bg_state["uinteger"])
    return result


def _restore_rng_state(state_dict: dict[str, Any]) -> np.random.Generator:
    """Reconstruct a Generator from a saved state dict."""
    rng = np.random.default_rng(seed=0)
    bg_state: dict[str, Any] = {
        "bit_generator": state_dict["bit_generator"],
    }
    s: dict[str, Any] = {}
    for k, v in state_dict["state"].items():
        if isinstance(v, list):
            s[k] = np.array(v, dtype=np.uint64)
        else:
            s[k] = v
    bg_state["state"] = s
    if "has_uint32" in state_dict:
        bg_state["has_uint32"] = state_dict["has_uint32"]
    if "uinteger" in state_dict:
        bg_state["uinteger"] = state_dict["uinteger"]
    rng.bit_generator.state = bg_state
    return rng


def _save_state_npz(path: Path, state: dict[str, Any]) -> None:
    """Save cells (and values) of a single state to an .npz file."""
    data: dict[str, np.ndarray] = {"cells": np.asarray(state["cells"])}
    for var_name, var_arr in state.get("values", {}).items():
        data[f"val_{var_name}"] = np.asarray(var_arr)
    np.savez_compressed(str(path), **data)


def _load_state_npz(path: Path, generation: int) -> dict[str, Any]:
    """Load a state from an .npz file."""
    values: dict[str, np.ndarray] = {}
    with np.load(str(path)) as loaded:
        cells = loaded["cells"]
        for key in loaded.files:
            if key.startswith("val_"):
                var_name = key[4:]
                values[var_name] = loaded[key]
    return {
        "cells": cells,
        "values": values,
        "agents": [],
        "generation": generation,
    }


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------

def save_run(
    name: str,
    automaton_raw: dict[str, Any],
    experiment_raw: dict[str, Any],
    run_result: dict[str, Any],
    stored_generations_indices: list[int] | None = None,
) -> Path:
    """Save a completed (or partial) run to disk.

    Parameters
    ----------
    name : human-readable run name
    automaton_raw : the raw automaton JSON dict (for reproducibility)
    experiment_raw : the raw experiment JSON dict
    run_result : the dict returned by ``simulation.run``
    stored_generations_indices : list of generation numbers stored;
        if None, computed from the generation field of each state.

    Returns
    -------
    Path to the created run directory.
    """
    run_dir = _next_run_dir(name)

    # automaton.json
    with open(run_dir / "automaton.json", "w", encoding="utf-8") as f:
        json.dump(automaton_raw, f, indent=2, ensure_ascii=False)

    # experiment.json
    with open(run_dir / "experiment.json", "w", encoding="utf-8") as f:
        json.dump(experiment_raw, f, indent=2, ensure_ascii=False)

    # generations.npz – all stored states packed together
    states = run_result["states"]
    if stored_generations_indices is None:
        stored_generations_indices = [s.get("generation", i) for i, s in enumerate(states)]

    gen_data: dict[str, np.ndarray] = {}
    for idx, st in zip(stored_generations_indices, states):
        gen_data[f"gen_{idx}_cells"] = np.asarray(st["cells"])
        for var_name, var_arr in st.get("values", {}).items():
            gen_data[f"gen_{idx}_val_{var_name}"] = np.asarray(var_arr)

    np.savez_compressed(str(run_dir / "generations.npz"), **gen_data)

    # last_state.npz – always saved
    last = run_result["last_state"]
    _save_state_npz(run_dir / "last_state.npz", last)

    # meta.json
    rng = run_result["rng"]
    meta = {
        "format_version": "2.0",
        "created": datetime.datetime.now().isoformat(),
        "last_generation": last.get("generation", 0),
        "stored_generations": stored_generations_indices,
        "rng_state": _serialize_rng_state(rng),
    }
    with open(run_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return run_dir


def load_run(run_dir: str | Path) -> dict[str, Any]:
    """Load a saved run from disk.

    Returns
    -------
    dict with keys:
        ``automaton_raw`` – the raw automaton JSON
        ``experiment_raw`` – the raw experiment JSON
        ``last_state`` – the last state dict
        ``meta`` – the meta.json contents
        ``rng`` – a restored numpy Generator
        ``states`` – list of stored state dicts (cells + values + generation)
    """
    resolved_dir = Path(run_dir).resolve()
    if not resolved_dir.is_dir():
        raise FileNotFoundError(f"Run directory not found: {resolved_dir}")

    with open(resolved_dir / "automaton.json", "r", encoding="utf-8") as f:
        automaton_raw = json.load(f)
    with open(resolved_dir / "experiment.json", "r", encoding="utf-8") as f:
        experiment_raw = json.load(f)
    with open(resolved_dir / "meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    rng = _restore_rng_state(meta["rng_state"])

    last_state = _load_state_npz(
        resolved_dir / "last_state.npz", meta["last_generation"]
    )

    # Reload stored generations
    stored_gens = meta["stored_generations"]
    states: list[dict[str, Any]] = []
    gen_npz_path = resolved_dir / "generations.npz"
    if gen_npz_path.exists():
        with np.load(str(gen_npz_path)) as loaded:
            for gen_idx in stored_gens:
                cells_key = f"gen_{gen_idx}_cells"
                if cells_key in loaded.files:
                    cells = loaded[cells_key]
                    values: dict[str, np.ndarray] = {}
                    for key in loaded.files:
                        prefix = f"gen_{gen_idx}_val_"
                        if key.startswith(prefix):
                            var_name = key[len(prefix):]
                            values[var_name] = loaded[key]
                    states.append({
                        "cells": cells,
                        "values": values,
                        "agents": [],
                        "generation": gen_idx,
                    })

    return {
        "automaton_raw": automaton_raw,
        "experiment_raw": experiment_raw,
        "last_state": last_state,
        "meta": meta,
        "rng": rng,
        "states": states,
    }
