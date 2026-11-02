import sys
from pathlib import Path

import numpy as np

# The repository has a project directory named ``cellab`` containing the
# actual Python package, so make the test work from either repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cellab.engine_cells import step_cells


def _state(cells):
    return {
        "cells": np.asarray(cells, dtype=np.int64),
        "values": {},
        "agents": [],
        "generation": 0,
    }


def test_step_cells_applies_1d_patterns_synchronously():
    automaton = {
        "neighborhood": {"type": "1d", "radius": 1},
        "boundary": "toroidal",
        "rules": [
            {"from": "*", "to": 1, "when": {"pattern": [0, 0, 1]}},
        ],
    }
    initial = _state([[0, 1, 0]])

    result = step_cells(automaton, initial, np.random.default_rng(0))

    np.testing.assert_array_equal(result["cells"], [[1, 1, 0]])
    np.testing.assert_array_equal(initial["cells"], [[0, 1, 0]])
    assert result["generation"] == 1


def test_step_cells_uses_fixed_outside_state_for_neighbor_counts():
    automaton = {
        "neighborhood": {"type": "1d", "radius": 1},
        "world": {"boundary": "fixed", "outside_state": 1},
        "rules": [
            {"from": 0, "to": 1, "when": {"count": 1, "min": 1}},
        ],
    }

    result = step_cells(automaton, _state([[0]]), np.random.default_rng(0))

    # Both out-of-grid neighbors have state 1, so the count is two.
    np.testing.assert_array_equal(result["cells"], [[1]])
