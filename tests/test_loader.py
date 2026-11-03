"""Tests for cellab.loader, including the first real chain:
JSON files -> loader -> engine of Person 2.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from cellab.conventions import PROJECT_ROOT
from cellab.engine_cells import step_cells
from cellab.loader import load_files

EXPERIMENT_FILES = sorted((PROJECT_ROOT / "experiments").glob("*.json"))


def grid_to_cells(automaton, rows):
    """Temporary stand-in for Person 3's initial_state (manual mode only)."""
    cells = []
    for row in rows:
        line = []
        for symbol in row:
            line.append(automaton["symbols"][symbol])
        cells.append(line)
    return np.array(cells, dtype=np.int64)


def make_state(cells):
    return {"cells": cells, "values": {}, "agents": [], "generation": 0}


@pytest.mark.parametrize("path", EXPERIMENT_FILES, ids=[path.stem for path in EXPERIMENT_FILES])
def test_every_example_experiment_loads(path):
    result = load_files(path)
    assert result["automaton_issues"] == []
    assert result["experiment_issues"] == []
    assert result["automaton"] is not None
    assert result["experiment"] is not None


def test_forest_fire_is_translated_like_the_contract_says():
    automaton = load_files("experiments/forest_fire_manual.json")["automaton"]

    assert automaton["state_codes"] == {"EMPTY": 0, "TREE": 1, "FIRE": 2}
    assert automaton["state_names"] == ["EMPTY", "TREE", "FIRE"]
    assert automaton["symbols"] == {".": 0, "T": 1, "F": 2}
    assert automaton["neighbor_offsets"] == [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    assert automaton["world"] == {"boundary": "fixed", "outside_code": 0, "outside_values": {}}
    assert automaton["compiled_rules"] == [
        {"from_codes": [1], "to_code": 2, "set": {}, "when": {"count": 2, "min": 1}, "probability": 1.0},
        {"from_codes": [2], "to_code": 0, "set": {}, "when": None, "probability": 1.0},
    ]
    # missing optional fields appear empty, never missing
    assert automaton["variables"] == []
    assert automaton["parameters"] == {}


def test_experiment_overrides_parameters():
    automaton = load_files("experiments/forest_fire_stochastic_random.json")["automaton"]
    assert automaton["parameters"] == {"p": 0.05, "q": 0.0001}
    assert automaton["compiled_rules"][3]["probability"] == 0.05     # p, overridden by the experiment
    assert automaton["compiled_rules"][2]["probability"] == 0.0001   # q, from the automaton
    assert automaton["world"]["outside_code"] is None                # toroidal


def test_star_means_every_state_and_pattern_uses_codes():
    automaton = load_files("experiments/rule_30_center.json")["automaton"]
    first_rule = automaton["compiled_rules"][0]
    assert first_rule["from_codes"] == [0, 1]
    assert first_rule["when"] == {"pattern": [1, 1, 1]}
    assert automaton["neighbor_offsets"] == [(0, -1), (0, 1)]


def test_loaded_forest_fire_runs_on_the_engine():
    # The 5x5 example from the hand calculation, now read from the JSON files.
    result = load_files("experiments/forest_fire_manual.json")
    automaton = result["automaton"]
    state = make_state(grid_to_cells(automaton, result["experiment"]["initial_state"]["cells"]["grid"]))
    rng = np.random.default_rng(0)

    state = step_cells(automaton, state, rng)
    expected = grid_to_cells(automaton, ["TTTTT", "TFFFT", "TF.FT", "TFFFT", "TTTTT"])
    np.testing.assert_array_equal(state["cells"], expected)

    state = step_cells(automaton, state, rng)
    expected = grid_to_cells(automaton, ["FFFFF", "F...F", "F...F", "F...F", "FFFFF"])
    np.testing.assert_array_equal(state["cells"], expected)


def test_loaded_rule_30_runs_on_the_engine():
    # The automaton comes from the JSON file; the 11-cell row is written here.
    automaton = load_files("experiments/rule_30_center.json")["automaton"]
    rows = [".....X.....", "....XXX....", "...XX..X...", "..XX.XXXX..", ".XX..X...X."]
    state = make_state(grid_to_cells(automaton, [rows[0]]))
    rng = np.random.default_rng(0)
    for row in rows[1:]:
        state = step_cells(automaton, state, rng)
        np.testing.assert_array_equal(state["cells"], grid_to_cells(automaton, [row]))


def test_missing_file_is_reported_not_raised():
    result = load_files("experiments/does_not_exist.json")
    assert result["automaton"] is None
    assert result["experiment_issues"][0]["code"] == "SCHEMA"


def test_broken_json_is_reported_not_raised(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text('{"format_version": "2.0",', encoding="utf-8")
    result = load_files(path)
    assert result["automaton"] is None
    assert "not valid JSON" in result["experiment_issues"][0]["message"]


def test_invalid_automaton_stops_loading(tmp_path):
    raw_aut = json.loads((PROJECT_ROOT / "automata" / "forest_fire.json").read_text(encoding="utf-8"))
    raw_aut["rules"][0]["to"] = "ASH"
    aut_path = tmp_path / "bad_automaton.json"
    aut_path.write_text(json.dumps(raw_aut), encoding="utf-8")

    raw_exp = json.loads((PROJECT_ROOT / "experiments" / "forest_fire_manual.json").read_text(encoding="utf-8"))
    raw_exp["automaton"] = str(aut_path)
    exp_path = tmp_path / "experiment.json"
    exp_path.write_text(json.dumps(raw_exp), encoding="utf-8")

    result = load_files(exp_path)
    assert result["automaton"] is None
    assert result["automaton_issues"][0]["code"] == "UNKNOWN_STATE"
    assert result["automaton_issues"][0]["where"] == "/rules/0/to"


def test_loading_does_not_change_the_raw_data():
    path = PROJECT_ROOT / "automata" / "brians_brain.json"
    before = json.loads(path.read_text(encoding="utf-8"))
    automaton = load_files("experiments/brians_brain_random.json")["automaton"]
    automaton["rules"][0]["to"] = "CHANGED"
    assert json.loads(path.read_text(encoding="utf-8")) == before


def test_agent_automaton_loads_with_empty_cell_rules():
    automaton = load_files("experiments/langtons_ant_center.json")["automaton"]
    assert automaton["kind"] == "agents"
    assert automaton["compiled_rules"] == []
    assert automaton["agent_types"][0]["name"] == "ANT"
