"""Tests for cellab.validator.

Every file in tests/invalid/ is one broken case:
    {"description": ..., "expect": {"file": "automaton" | "experiment", "code": ..., "where": ...},
     "automaton": {...} or a path to an example automaton,
     "experiment": {...}}            (only for experiment cases)
"""

import json
from pathlib import Path

import pytest

from cellab.conventions import PROJECT_ROOT, json_pointer
from cellab.validator import has_errors, validate_automaton, validate_experiment

INVALID_DIR = Path(__file__).parent / "invalid"
INVALID_FILES = sorted(INVALID_DIR.glob("*.json"))
AUTOMATON_FILES = sorted((PROJECT_ROOT / "automata").glob("*.json"))


def read_json(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def codes_and_places(issues):
    pairs = []
    for issue in issues:
        pairs.append((issue["code"], issue["where"]))
    return pairs


def test_there_are_at_least_ten_invalid_files():
    assert len(INVALID_FILES) >= 10


@pytest.mark.parametrize("path", INVALID_FILES, ids=[path.stem for path in INVALID_FILES])
def test_invalid_file_reports_expected_issue(path):
    case = read_json(path)
    expect = case["expect"]

    raw_aut = case["automaton"]
    if isinstance(raw_aut, str):
        raw_aut = read_json(PROJECT_ROOT / raw_aut)

    automaton_issues = validate_automaton(raw_aut)
    if expect["file"] == "automaton":
        issues = automaton_issues
    else:
        assert not has_errors(automaton_issues), automaton_issues
        issues = validate_experiment(case["experiment"], raw_aut)

    assert (expect["code"], expect["where"]) in codes_and_places(issues), issues


@pytest.mark.parametrize("path", AUTOMATON_FILES, ids=[path.stem for path in AUTOMATON_FILES])
def test_example_automata_have_no_issues(path):
    assert validate_automaton(read_json(path)) == []


def test_all_problems_are_reported_together():
    raw_aut = read_json(PROJECT_ROOT / "automata" / "forest_fire.json")
    raw_aut["rules"] = [
        {"from": "TREE", "to": "ASH"},
        {"from": "FIRE", "to": "EMPTY", "probability": 3},
    ]
    codes = []
    for issue in validate_automaton(raw_aut):
        codes.append(issue["code"])
    assert "UNKNOWN_STATE" in codes
    assert "BAD_PROBABILITY" in codes


def test_sense_checks_do_not_run_when_the_form_is_wrong():
    # "states" is missing, so only SCHEMA issues can appear.
    raw_aut = read_json(PROJECT_ROOT / "automata" / "forest_fire.json")
    del raw_aut["states"]
    for issue in validate_automaton(raw_aut):
        assert issue["code"] == "SCHEMA"


def test_every_issue_has_the_four_fields():
    case = read_json(INVALID_DIR / "unknown_state_in_count.json")
    issue = validate_automaton(case["automaton"])[0]
    assert sorted(issue.keys()) == ["code", "level", "message", "where"]
    assert "BURNING" in issue["message"]


def test_json_pointer():
    assert json_pointer([]) == ""
    assert json_pointer(["rules", 2, "when", "count"]) == "/rules/2/when/count"
    assert json_pointer(["parameters", "a/b~c"]) == "/parameters/a~1b~0c"
