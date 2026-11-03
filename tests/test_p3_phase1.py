"""Unit tests for Person 3 - Phase 1.

Covers:
- cellab.initial_state: all cells modes (fill, random, manual, pattern)
  and values modes (constant with points, random, manual, defaults)
- cellab.simulation: step dispatch, run with all/last/every storage modes
- cellab.storage: save_run, load_run, and the mandatory 10+10=20 recovery test
"""

import shutil
import numpy as np

from cellab.initial_state import create_initial_state
from cellab.simulation import step, run
from cellab.storage import save_run, load_run

MOORE = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def make_test_automaton():
    return {
        "format_version": "2.0",
        "kind": "cells",
        "name": "Test CA",
        "state_codes": {"DEAD": 0, "ALIVE": 1},
        "state_names": ["DEAD", "ALIVE"],
        "symbols": {".": 0, "A": 1},
        "colors": ["#000000", "#FFFFFF"],
        "variable_types": {"heat": "float", "count": "int"},
        "variable_defaults": {"heat": 0.0, "count": 0},
        "parameters": {"p": 0.5},
        "neighbor_offsets": MOORE,
        "world": {"boundary": "fixed", "outside_code": 0, "outside_values": {"heat": 0.0}},
        "compiled_rules": [
            {
                "from_codes": [0],
                "to_code": 1,
                "set": {},
                "when": {"count": 1, "min": 1},
                "probability": 1.0,
            }
        ],
        "rules": [],
    }


# ---------------------------------------------------------------------------
# Initial State Tests
# ---------------------------------------------------------------------------

def test_initial_state_fill():
    aut = make_test_automaton()
    exp = {
        "world": {"rows": 4, "cols": 4, "boundary": "fixed"},
        "initial_state": {"cells": {"mode": "fill", "state": "ALIVE"}},
    }
    rng = np.random.default_rng(0)
    st = create_initial_state(aut, exp, rng)
    assert st["cells"].shape == (4, 4)
    assert np.all(st["cells"] == 1)
    assert st["generation"] == 0


def test_initial_state_manual():
    aut = make_test_automaton()
    exp = {
        "world": {"rows": 2, "cols": 3, "boundary": "fixed"},
        "initial_state": {
            "cells": {"mode": "manual", "grid": [".A.", "A.A"]}
        },
    }
    rng = np.random.default_rng(0)
    st = create_initial_state(aut, exp, rng)
    expected = np.array([[0, 1, 0], [1, 0, 1]], dtype=np.int64)
    np.testing.assert_array_equal(st["cells"], expected)


def test_initial_state_pattern():
    aut = make_test_automaton()
    exp = {
        "world": {"rows": 3, "cols": 3, "boundary": "fixed"},
        "initial_state": {
            "cells": {
                "mode": "pattern",
                "background": "DEAD",
                "pattern": ["A"],
                "position": [1, 1],
            }
        },
    }
    rng = np.random.default_rng(0)
    st = create_initial_state(aut, exp, rng)
    expected = np.zeros((3, 3), dtype=np.int64)
    expected[1, 1] = 1
    np.testing.assert_array_equal(st["cells"], expected)


def test_initial_state_random_proportions():
    aut = make_test_automaton()
    exp = {
        "world": {"rows": 100, "cols": 100, "boundary": "fixed"},
        "initial_state": {
            "cells": {"mode": "random", "proportions": {"DEAD": 0.8, "ALIVE": 0.2}}
        },
    }
    rng = np.random.default_rng(42)
    st = create_initial_state(aut, exp, rng)
    alive_ratio = np.mean(st["cells"] == 1)
    assert 0.15 < alive_ratio < 0.25


def test_initial_state_values_constant_and_points():
    aut = make_test_automaton()
    exp = {
        "world": {"rows": 3, "cols": 3, "boundary": "fixed"},
        "initial_state": {
            "cells": {"mode": "fill", "state": "DEAD"},
            "values": {
                "heat": {
                    "mode": "constant",
                    "value": 10.0,
                    "points": [{"position": [0, 0], "value": 99.0}],
                }
            },
        },
    }
    rng = np.random.default_rng(0)
    st = create_initial_state(aut, exp, rng)
    assert st["values"]["heat"][0, 0] == 99.0
    assert st["values"]["heat"][1, 1] == 10.0
    # count was not specified in initial_state, must have default value 0
    assert np.all(st["values"]["count"] == 0)


# ---------------------------------------------------------------------------
# Simulation Tests
# ---------------------------------------------------------------------------

def test_simulation_step_and_run_storage_modes():
    aut = make_test_automaton()
    exp = {
        "world": {"rows": 3, "cols": 3, "boundary": "fixed"},
        "initial_state": {
            "cells": {
                "mode": "pattern",
                "background": "DEAD",
                "pattern": ["A"],
                "position": [1, 1],
            }
        },
    }
    rng = np.random.default_rng(0)
    st0 = create_initial_state(aut, exp, rng)

    # step advances generation
    st1 = step(aut, st0, rng)
    assert st1["generation"] == 1

    # mode: all
    res_all = run(aut, st0, 4, {"mode": "all"}, rng=np.random.default_rng(0))
    assert len(res_all["states"]) == 5  # gen 0, 1, 2, 3, 4
    assert res_all["last_state"]["generation"] == 4

    # mode: last
    res_last = run(aut, st0, 4, {"mode": "last"}, rng=np.random.default_rng(0))
    assert len(res_last["states"]) == 1
    assert res_last["states"][0]["generation"] == 4

    # mode: every
    res_every = run(aut, st0, 4, {"mode": "every", "every": 2}, rng=np.random.default_rng(0))
    gens = [s["generation"] for s in res_every["states"]]
    assert gens == [0, 2, 4]


# ---------------------------------------------------------------------------
# Storage & 10+10=20 Recovery Test
# ---------------------------------------------------------------------------

def test_storage_recovery_10_plus_10_equals_20():
    # Stochastic automaton with probability
    aut = {
        "format_version": "2.0",
        "kind": "cells",
        "name": "Stochastic CA",
        "state_codes": {"DEAD": 0, "ALIVE": 1},
        "state_names": ["DEAD", "ALIVE"],
        "symbols": {".": 0, "A": 1},
        "colors": ["#000000", "#FFFFFF"],
        "variable_types": {},
        "variable_defaults": {},
        "parameters": {"p": 0.3},
        "neighbor_offsets": MOORE,
        "world": {"boundary": "toroidal", "outside_code": None, "outside_values": {}},
        "compiled_rules": [
            {
                "from_codes": [0],
                "to_code": 1,
                "set": {},
                "when": None,
                "probability": 0.3,
            }
        ],
        "rules": [],
    }
    exp = {
        "format_version": "2.0",
        "automaton": "dummy.json",
        "world": {"rows": 10, "cols": 10, "boundary": "toroidal"},
        "initial_state": {"cells": {"mode": "fill", "state": "DEAD"}},
        "generations": 20,
        "storage": {"mode": "all"},
        "seed": 999,
    }

    # 1. 20 steps continuously
    rng_full = np.random.default_rng(999)
    st0_full = create_initial_state(aut, exp, rng_full)
    res_full = run(aut, st0_full, 20, {"mode": "all"}, rng=rng_full)

    # 2. 10 steps, save, reload, 10 more steps
    rng_split = np.random.default_rng(999)
    st0_split = create_initial_state(aut, exp, rng_split)
    res_first = run(aut, st0_split, 10, {"mode": "all"}, rng=rng_split)

    run_dir = save_run("stochastic_ca", aut, exp, res_first)
    try:
        loaded = load_run(run_dir)
        assert loaded["meta"]["last_generation"] == 10
        rng_resumed = loaded["rng"]

        res_second = run(aut, loaded["last_state"], 10, {"mode": "all"}, rng=rng_resumed)

        # Final cells must match exactly
        np.testing.assert_array_equal(
            res_full["last_state"]["cells"],
            res_second["last_state"]["cells"],
        )
        assert res_second["last_state"]["generation"] == 20
    finally:
        shutil.rmtree(run_dir, ignore_errors=True)
