"""Tests for cellab.expressions: reading formulas safely and computing them on a grid.

The expected values are worked out by hand on small grids.
"""

import numpy as np
import pytest

from cellab.conventions import NEIGHBOR_OFFSETS
from cellab.expressions import (
    ExpressionError,
    ExpressionRuntimeError,
    evaluate_expression,
    parse_expression,
    rule_formulas,
    uses_random,
)

STATES = ["A", "B"]
VARIABLES = ["grains", "heat"]
PARAMETERS = {"p": 0.5, "k": 4}


def parse(text, expected=None):
    return parse_expression(text, STATES, VARIABLES, PARAMETERS, expected)


def shifted(matrix, offset, outside):
    """The neighbor in direction offset of every cell, with a fixed boundary (written cell by cell)."""
    rows, cols = matrix.shape
    result = np.full(matrix.shape, outside, dtype=matrix.dtype)
    for r in range(rows):
        for c in range(cols):
            nr, nc = r + offset[0], c + offset[1]
            if 0 <= nr < rows and 0 <= nc < cols:
                result[r, c] = matrix[nr, nc]
    return result


def make_context(cells, values, neighborhood="von_neumann", seed=0):
    neighbors = []
    for offset in NEIGHBOR_OFFSETS[neighborhood]:
        neighbor = {"cells": shifted(cells, offset, 0)}
        for name, matrix in values.items():
            neighbor[name] = shifted(matrix, offset, 0)
        neighbors.append(neighbor)
    return {
        "self": values,
        "neighbors": neighbors,
        "state_codes": {"A": 0, "B": 1},
        "parameters": PARAMETERS,
        "rng": np.random.default_rng(seed),
        "shape": cells.shape,
    }


# ---------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------

def test_a_condition_is_bool_and_a_calculation_is_a_number():
    assert parse("count(B) > count(A)")["type"] == "bool"
    assert parse("(heat + sum(heat)) / 5")["type"] == "number"


def test_the_names_really_used_are_listed():
    formula = parse("random() < p * count(B) and grains >= k")
    assert formula["states"] == ["B"]
    assert formula["variables"] == ["grains"]
    assert formula["parameters"] == ["k", "p"]
    assert formula["random_count"] == 1


@pytest.mark.parametrize("text", [
    "__import__('os').system('dir')",
    "().__class__.__bases__",
    "grains.real",
    "[x for x in (1, 2)]",
    "lambda: 1",
    "heat ** 2",
    "grains if heat else 1",
    "open('secret.txt')",
    "eval('1')",
    "'text'",
    "grains[0]",
    "(heat := 3)",
])
def test_anything_outside_the_vocabulary_is_refused(text):
    with pytest.raises(ExpressionError) as error:
        parse(text)
    assert error.value.code == "BAD_EXPRESSION"


@pytest.mark.parametrize("text, code", [
    ("count(C) > 1", "UNKNOWN_STATE"),
    ("temperature + 1", "UNKNOWN_VARIABLE"),
    ("A + 1", "UNKNOWN_VARIABLE"),
    ("count(grains)", "UNKNOWN_STATE"),
    ("count(1)", "BAD_EXPRESSION"),
    ("sum(heat, grains)", "BAD_EXPRESSION"),
    ("random(3)", "BAD_EXPRESSION"),
    ("sum(count(A))", "BAD_EXPRESSION"),
    ("count_where(random() < p)", "BAD_EXPRESSION"),
    ("count_where(grains)", "EXPRESSION_TYPE"),
    ("not grains", "EXPRESSION_TYPE"),
    ("grains and heat", "EXPRESSION_TYPE"),
    ("count(B) >", "BAD_EXPRESSION"),
    ("", "BAD_EXPRESSION"),
])
def test_wrong_formulas_give_the_right_code(text, code):
    with pytest.raises(ExpressionError) as error:
        parse(text)
    assert error.value.code == code


def test_the_expected_kind_of_result_is_checked():
    with pytest.raises(ExpressionError) as error:
        parse("count(B)", expected="bool")
    assert error.value.code == "EXPRESSION_TYPE"
    with pytest.raises(ExpressionError) as error:
        parse("grains >= 4", expected="number")
    assert error.value.code == "EXPRESSION_TYPE"
    assert parse("1 * (grains >= 4)", expected="number")["type"] == "number"


def test_uses_random_and_rule_formulas():
    rule = {"from": "A", "when": {"all": [{"count": 1, "min": 1}, {"not": {"expr": "random() < p"}}]},
            "set": {"grains": "grains + 1"}}
    assert rule_formulas(rule) == ["random() < p", "grains + 1"]
    assert uses_random("random() < p")
    assert not uses_random("count(B) > 2")
    assert not uses_random("broken (")


# ---------------------------------------------------------------------
# Computing
# ---------------------------------------------------------------------

def test_sandpile_3x3_by_hand():
    # The 8 in the middle topples: it keeps 4 and each of its 4 neighbors gets 1.
    grains = np.array([[0, 1, 2],
                       [2, 8, 1],
                       [1, 2, 0]])
    cells = np.zeros((3, 3), dtype=np.int64)
    context = make_context(cells, {"grains": grains})
    formula = parse("grains - 4 * (grains >= 4) + count_where(grains >= 4)")
    expected = np.array([[0, 2, 2],
                         [3, 4, 2],
                         [1, 3, 0]])
    assert np.array_equal(evaluate_expression(formula, context), expected)


def test_diffusion_point_by_hand():
    heat = np.zeros((3, 3))
    heat[1, 1] = 100.0
    context = make_context(np.zeros((3, 3), dtype=np.int64), {"heat": heat})
    result = evaluate_expression(parse("(heat + sum(heat)) / 5"), context)
    expected = np.array([[0, 20, 0],
                         [20, 20, 20],
                         [0, 20, 0]], dtype=float)
    assert np.allclose(result, expected)


def test_voter_majority_by_hand():
    # Moore neighborhood, the outside counts as A (code 0).
    cells = np.array([[1, 1, 1],
                      [1, 0, 1],
                      [1, 1, 0]])
    context = make_context(cells, {}, neighborhood="moore")
    counts = evaluate_expression(parse("count(B)"), context)
    # the center sees 7 B and 1 A; the corner (0, 0) sees 2 B, 1 A and 5 outside cells (A)
    assert counts[1, 1] == 7 and counts[0, 0] == 2
    result = evaluate_expression(parse("count(B) > count(A)"), context)
    assert result[1, 1] and not result[0, 0]


def test_avg_min_max_over_neighbors():
    heat = np.array([[1.0, 2.0, 3.0],
                     [4.0, 5.0, 6.0],
                     [7.0, 8.0, 9.0]])
    context = make_context(np.zeros((3, 3), dtype=np.int64), {"heat": heat})
    # the center has neighbors N=2, W=4, E=6, S=8
    assert evaluate_expression(parse("avg(heat)"), context)[1, 1] == 5.0
    assert evaluate_expression(parse("min(heat)"), context)[1, 1] == 2.0
    assert evaluate_expression(parse("max(heat)"), context)[1, 1] == 8.0
    # a corner has two outside neighbors with value 0
    assert evaluate_expression(parse("min(heat)"), context)[0, 0] == 0.0


def test_true_and_false_count_as_1_and_0():
    context = make_context(np.zeros((2, 2), dtype=np.int64), {"grains": np.full((2, 2), 5)})
    result = evaluate_expression(parse("(grains > 1) + (grains > 2)"), context)
    assert np.array_equal(result, np.full((2, 2), 2))


def test_chained_comparison():
    grains = np.array([[0, 2], [5, 9]])
    context = make_context(np.zeros((2, 2), dtype=np.int64), {"grains": grains})
    result = evaluate_expression(parse("1 <= grains < 6"), context)
    assert result.tolist() == [[False, True], [True, False]]


def test_floor_division_and_modulo_like_python():
    grains = np.array([[-7, 7]])
    context = make_context(np.zeros((1, 2), dtype=np.int64), {"grains": grains})
    assert evaluate_expression(parse("grains // 2"), context).tolist() == [[-4, 3]]
    assert evaluate_expression(parse("grains % 3"), context).tolist() == [[2, 1]]
    assert evaluate_expression(parse("floor(grains / 2)"), context).tolist() == [[-4.0, 3.0]]
    assert evaluate_expression(parse("abs(grains)"), context).tolist() == [[7, 7]]


def test_division_by_zero_only_matters_where_the_result_is_used():
    grains = np.array([[0, 2]])
    context = make_context(np.zeros((1, 2), dtype=np.int64), {"grains": grains})
    formula = parse("4 / grains")
    with pytest.raises(ExpressionRuntimeError):
        evaluate_expression(formula, context)
    result = evaluate_expression(formula, context, mask=np.array([[False, True]]))
    assert result[0, 1] == 2.0


def test_and_protects_the_right_side_like_python():
    grains = np.array([[0, 2]])
    context = make_context(np.zeros((1, 2), dtype=np.int64), {"grains": grains})
    result = evaluate_expression(parse("grains > 0 and 4 / grains > 1"), context)
    assert result.tolist() == [[False, True]]
    result = evaluate_expression(parse("grains == 0 or 4 / grains > 1"), context)
    assert result.tolist() == [[True, True]]


def test_random_draws_one_full_matrix_per_call_in_reading_order():
    cells = np.zeros((3, 4), dtype=np.int64)
    context = make_context(cells, {}, seed=11)
    formula = parse("random() < 0.3 or random() > 0.9")
    result = evaluate_expression(formula, context)

    rng = np.random.default_rng(11)
    first, second = rng.random((3, 4)), rng.random((3, 4))
    assert np.array_equal(result, (first < 0.3) | (second > 0.9))
    # nothing else was drawn
    assert context["rng"].random() == rng.random()


def test_parameters_and_constants_fill_the_whole_grid():
    context = make_context(np.zeros((2, 3), dtype=np.int64), {})
    result = evaluate_expression(parse("k * 2"), context)
    assert result.shape == (2, 3) and (result == 8).all()
