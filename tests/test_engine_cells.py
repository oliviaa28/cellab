"""Tests for engine_cells.step_cells.

The automata below are written by hand in the *loaded* form from
docs/contract.md (compiled_rules, neighbor_offsets, world). When the loader
is ready, these helpers should be replaced by loading the real JSON files.
"""

import numpy as np
import pytest

from cellab.engine_cells import step_cells

MOORE = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
LINE = [(0, -1), (0, 1)]


def make_state(cells):
    return {"cells": np.array(cells, dtype=np.int64), "values": {}, "agents": [], "generation": 0}


def make_rule(from_codes, to_code, when=None, probability=1.0):
    return {"from_codes": from_codes, "to_code": to_code, "set": {}, "when": when, "probability": probability}


def make_automaton(rules, offsets, boundary="fixed", outside_code=0):
    return {
        "compiled_rules": rules,
        "neighbor_offsets": offsets,
        "world": {"boundary": boundary, "outside_code": outside_code, "outside_values": {}},
    }


def rows_to_cells(rows):
    return [[int(ch) for ch in row] for row in rows]


# ---------- Forest Fire: EMPTY=0, TREE=1, FIRE=2 ----------

FOREST_FIRE = make_automaton(
    [
        make_rule([1], 2, {"count": 2, "min": 1}),
        make_rule([2], 0),
    ],
    MOORE,
    boundary="fixed",
    outside_code=0,
)


def test_forest_fire_5x5_matches_hand_calculation():
    state = make_state(rows_to_cells(["11111", "11111", "11211", "11111", "11111"]))
    rng = np.random.default_rng(0)

    state = step_cells(FOREST_FIRE, state, rng)
    np.testing.assert_array_equal(state["cells"], rows_to_cells(["11111", "12221", "12021", "12221", "11111"]))

    state = step_cells(FOREST_FIRE, state, rng)
    np.testing.assert_array_equal(state["cells"], rows_to_cells(["22222", "20002", "20002", "20002", "22222"]))

    state = step_cells(FOREST_FIRE, state, rng)
    np.testing.assert_array_equal(state["cells"], np.zeros((5, 5)))

    stable = step_cells(FOREST_FIRE, state, rng)
    np.testing.assert_array_equal(stable["cells"], state["cells"])
    assert stable["generation"] == 4


# ---------- Brian's Brain: OFF=0, FIRING=1, DYING=2 ----------

BRIANS_BRAIN = make_automaton(
    [
        make_rule([1], 2),
        make_rule([2], 0),
        make_rule([0], 1, {"count": 1, "min": 2, "max": 2}),
    ],
    MOORE,
    boundary="fixed",
    outside_code=0,
)


def test_brians_brain_fires_with_exactly_two_firing_neighbors():
    two = make_state([[1, 0, 1], [0, 0, 0], [0, 0, 0]])
    three = make_state([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
    rng = np.random.default_rng(0)

    assert step_cells(BRIANS_BRAIN, two, rng)["cells"][1, 1] == 1
    assert step_cells(BRIANS_BRAIN, three, rng)["cells"][1, 1] == 0


def test_brians_brain_firing_becomes_dying_then_off():
    state = make_state([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    rng = np.random.default_rng(0)

    state = step_cells(BRIANS_BRAIN, state, rng)
    assert state["cells"][1, 1] == 2
    state = step_cells(BRIANS_BRAIN, state, rng)
    assert state["cells"][1, 1] == 0


# ---------- Rule 30: OFF=0, ON=1 ----------

def rule_30():
    rules = []
    for index in range(8):
        left = (index >> 2) & 1
        center = (index >> 1) & 1
        right = index & 1
        rules.append(make_rule([0, 1], (30 >> index) & 1, {"pattern": [left, center, right]}))
    return make_automaton(rules, LINE, boundary="fixed", outside_code=0)


def test_rule_30_first_generations_match_known_triangle():
    expected = ["00000100000", "00001110000", "00011001000", "00110111100", "01100100010"]
    automaton = rule_30()
    state = make_state([[int(ch) for ch in expected[0]]])
    rng = np.random.default_rng(0)

    for row in expected[1:]:
        state = step_cells(automaton, state, rng)
        np.testing.assert_array_equal(state["cells"], [[int(ch) for ch in row]])


# ---------- boundaries ----------

def test_fixed_boundary_uses_outside_code():
    automaton = make_automaton([make_rule([0], 1, {"count": 1, "min": 2})], LINE, boundary="fixed", outside_code=1)
    result = step_cells(automaton, make_state([[0]]), np.random.default_rng(0))
    np.testing.assert_array_equal(result["cells"], [[1]])


def test_toroidal_boundary_wraps_around():
    # The cell at the right edge sees the left edge as its east neighbor.
    automaton = make_automaton([make_rule([0], 1, {"pattern": [0, 0, 1]})], LINE, boundary="toroidal")
    result = step_cells(automaton, make_state([[1, 0, 0]]), np.random.default_rng(0))
    np.testing.assert_array_equal(result["cells"], [[1, 0, 1]])


def test_missing_world_is_an_error():
    automaton = {"compiled_rules": [], "neighbor_offsets": LINE}
    with pytest.raises(KeyError):
        step_cells(automaton, make_state([[0]]), np.random.default_rng(0))


# ---------- rule semantics ----------

def test_first_matching_rule_wins_and_unmatched_cells_stay():
    automaton = make_automaton([make_rule([0], 1), make_rule([0], 2)], LINE)
    result = step_cells(automaton, make_state([[0, 3]]), np.random.default_rng(0))
    np.testing.assert_array_equal(result["cells"], [[1, 3]])


def test_update_is_synchronous_and_input_is_not_modified():
    # Every ON cell turns OFF and every OFF cell next to an ON cell turns ON.
    automaton = make_automaton(
        [make_rule([1], 0), make_rule([0], 1, {"count": 1, "min": 1})], LINE
    )
    initial = make_state([[0, 1, 0, 0]])
    result = step_cells(automaton, initial, np.random.default_rng(0))

    np.testing.assert_array_equal(result["cells"], [[1, 0, 1, 0]])
    np.testing.assert_array_equal(initial["cells"], [[0, 1, 0, 0]])


# ---------- randomness ----------

def test_same_seed_gives_same_result():
    automaton = make_automaton([make_rule([0], 1, probability=0.5)], MOORE, boundary="toroidal")
    state = make_state(np.zeros((20, 20), dtype=np.int64))

    first = step_cells(automaton, state, np.random.default_rng(42))
    second = step_cells(automaton, state, np.random.default_rng(42))
    np.testing.assert_array_equal(first["cells"], second["cells"])


def test_one_full_matrix_is_drawn_per_probabilistic_rule():
    # Two rules with probability < 1 and one with probability 1:
    # exactly two matrices must be drawn, even though the second rule matches no cell.
    automaton = make_automaton(
        [
            make_rule([0], 1, probability=0.3),
            make_rule([5], 1, probability=0.3),
            make_rule([1], 0, probability=1.0),
        ],
        MOORE,
        boundary="toroidal",
    )
    shape = (4, 6)
    rng_engine = np.random.default_rng(7)
    rng_manual = np.random.default_rng(7)

    step_cells(automaton, make_state(np.zeros(shape, dtype=np.int64)), rng_engine)
    rng_manual.random(shape)
    rng_manual.random(shape)

    assert rng_engine.random() == rng_manual.random()


# ---------- Phase 2: formulas and numeric values (real JSON files) ----------

from cellab.expressions import ExpressionRuntimeError  # noqa: E402
from cellab.initial_state import create_initial_state  # noqa: E402
from cellab.loader import load_files  # noqa: E402


def load(experiment_path):
    loaded = load_files(experiment_path)
    assert loaded["automaton"] is not None, loaded
    rng = np.random.default_rng(loaded["experiment"]["seed"] or 0)
    state = create_initial_state(loaded["automaton"], loaded["experiment"], rng)
    return loaded["automaton"], state, rng


def test_sandpile_3x3_by_hand():
    # experiments/sandpile_manual.json: the 8 in the middle topples twice, then two cells at the edge.
    automaton, state, rng = load("experiments/sandpile_manual.json")

    state = step_cells(automaton, state, rng)
    np.testing.assert_array_equal(state["values"]["grains"], [[0, 2, 2], [3, 4, 2], [1, 3, 0]])
    state = step_cells(automaton, state, rng)
    np.testing.assert_array_equal(state["values"]["grains"], [[0, 3, 2], [4, 0, 3], [1, 4, 0]])
    # (1, 0) and (2, 1) topple; the grains that fall outside the grid are lost
    state = step_cells(automaton, state, rng)
    np.testing.assert_array_equal(state["values"]["grains"], [[1, 3, 2], [0, 2, 3], [3, 0, 1]])
    assert state["values"]["grains"].dtype == np.int64


def test_sandpile_keeps_every_grain_when_nothing_reaches_the_edge():
    automaton, state, rng = load("experiments/sandpile_center.json")
    for _ in range(400):
        state = step_cells(automaton, state, rng)
    grains = state["values"]["grains"]
    assert grains.sum() == 300
    assert grains.max() <= 3
    # the pile is symmetric
    np.testing.assert_array_equal(grains, grains.T)
    np.testing.assert_array_equal(grains, grains[::-1, :])


def test_diffusion_point_by_hand():
    automaton, state, rng = load("experiments/diffusion_point.json")
    state = step_cells(automaton, state, rng)
    expected = np.array([[0, 20, 0], [20, 20, 20], [0, 20, 0]], dtype=float)
    np.testing.assert_allclose(state["values"]["heat"], expected)
    for _ in range(50):
        state = step_cells(automaton, state, rng)
    # heat is kept (toroidal) and spreads out evenly
    assert state["values"]["heat"].sum() == pytest.approx(100.0)
    np.testing.assert_allclose(state["values"]["heat"], np.full((3, 3), 100 / 9), atol=1e-6)


def test_voter_takes_the_majority_and_keeps_ties():
    automaton, state, rng = load("experiments/voter_random.json")
    a, b = automaton["state_codes"]["A"], automaton["state_codes"]["B"]
    automaton = dict(automaton, world={"boundary": "fixed", "outside_code": a, "outside_values": {}})
    state = make_state([[b, b, b], [b, a, b], [b, b, b]])

    result = step_cells(automaton, state, rng)["cells"]
    assert result[1, 1] == b    # 8 B neighbors
    assert result[0, 0] == a    # 2 B against 6 A (5 of them outside)
    assert result[0, 1] == b    # 4 B against 4 A: a tie, so it stays


def make_formula_automaton(rules, variables=None, parameters=None):
    """A small loaded automaton with states A=0, B=1 and the given variables (name -> type)."""
    variables = variables or {}
    automaton = make_automaton(rules, MOORE, boundary="toroidal")
    automaton.update({
        "state_names": ["A", "B"], "state_codes": {"A": 0, "B": 1},
        "variable_types": variables, "parameters": parameters or {},
    })
    return automaton


def test_int_variable_is_rounded_down_and_float_is_kept():
    rule = make_rule([0], None)
    rule["set"] = {"n": "n / 2", "x": "x / 2"}
    automaton = make_formula_automaton([rule], {"n": "int", "x": "float"})
    state = make_state(np.zeros((2, 2), dtype=np.int64))
    state["values"] = {"n": np.full((2, 2), 5, dtype=np.int64), "x": np.full((2, 2), 5.0)}

    result = step_cells(automaton, state, np.random.default_rng(0))
    assert (result["values"]["n"] == 2).all() and result["values"]["n"].dtype == np.int64
    assert (result["values"]["x"] == 2.5).all()
    assert (state["values"]["n"] == 5).all()      # the input is not modified


def test_set_only_where_the_rule_applies_and_from_the_previous_generation():
    # Every A cell takes 1 + the sum of its neighbors' n; B cells keep their n.
    rule = make_rule([0], None)
    rule["set"] = {"n": "1 + sum(n)"}
    automaton = make_formula_automaton([rule], {"n": "int"})
    state = make_state([[0, 1, 0]])
    state["values"] = {"n": np.array([[1, 10, 100]], dtype=np.int64)}
    automaton["neighbor_offsets"] = LINE

    result = step_cells(automaton, state, np.random.default_rng(0))
    # (0, 0): 1 + 100 + 10 (toroidal); (0, 2): 1 + 10 + 1 -- both from the old values
    np.testing.assert_array_equal(result["values"]["n"], [[111, 10, 12]])


def test_division_by_zero_stops_with_rule_and_generation():
    rule = make_rule([0], None)
    rule["set"] = {"n": "10 // n"}
    automaton = make_formula_automaton([rule], {"n": "int"})
    state = make_state([[0, 1]])
    state["values"] = {"n": np.array([[0, 0]], dtype=np.int64)}
    state["generation"] = 4

    with pytest.raises(ExpressionRuntimeError) as error:
        step_cells(automaton, state, np.random.default_rng(0))
    assert "rule 0" in str(error.value) and "generation 5" in str(error.value)
    assert "10 // n" in str(error.value)

    # no error when the dangerous cells do not use the rule
    state["cells"] = np.array([[1, 1]])
    step_cells(automaton, state, np.random.default_rng(0))


def test_random_draw_order_condition_then_probability_then_set():
    rule = make_rule([0], 1, when={"expr": "random() < 0.5"}, probability=0.5)
    rule["set"] = {"x": "random()"}
    automaton = make_formula_automaton([rule], {"x": "float"})
    shape = (3, 5)
    state = make_state(np.zeros(shape, dtype=np.int64))
    state["values"] = {"x": np.zeros(shape)}

    result = step_cells(automaton, state, np.random.default_rng(9))

    rng = np.random.default_rng(9)
    condition, probability, set_draw = rng.random(shape), rng.random(shape), rng.random(shape)
    applies = (condition < 0.5) & (probability < 0.5)
    np.testing.assert_array_equal(result["cells"], applies.astype(np.int64))
    np.testing.assert_array_equal(result["values"]["x"], np.where(applies, set_draw, 0.0))
