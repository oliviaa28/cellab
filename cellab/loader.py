"""Turns valid JSON files into the dictionaries everyone else uses (contract 5.4 and 6).

The translation happens once, here:
  - state names become codes:     "FIRE"      -> 2
  - parameter names become values: "p"         -> 0.05
  - the neighborhood becomes offsets: "moore"  -> [(-1, -1), (-1, 0), ...]
  - the boundary from the experiment is attached to the automaton (the engine needs it).

load_automaton and load_experiment expect data that already passed the validator.
load_files does everything (read, validate, load) and never crashes on a bad file:
it returns the problems instead.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

from cellab.conventions import NEIGHBOR_OFFSETS, PROJECT_ROOT, merge_parameters
from cellab.validator import has_errors, make_issue, validate_automaton, validate_experiment


# =====================================================================
# Everything at once, from files
# =====================================================================

def load_files(experiment_path, project_root=PROJECT_ROOT) -> dict:
    """Read, validate and load an experiment and its automaton.

    Returns {"automaton": ..., "experiment": ..., "automaton_issues": [...], "experiment_issues": [...]}.
    If there is any error, "automaton" and "experiment" are None. Warnings do not stop loading.

    The automaton path written inside the experiment must stay inside project_root
    (normally the project folder; tests can pass a temporary folder).
    """
    result = {"automaton": None, "experiment": None, "automaton_issues": [], "experiment_issues": []}

    # 1. read the experiment; its path can come from the command line, so it is checked too
    experiment_file = safe_path(experiment_path, project_root)
    if experiment_file is None:
        message = f"The experiment path '{experiment_path}' must stay inside the project folder."
        result["experiment_issues"].append(make_issue("SCHEMA", [], message))
        return result
    raw_exp, problem = read_json(experiment_file, experiment_path)
    if problem is not None:
        result["experiment_issues"].append(problem)
        return result
    if not isinstance(raw_exp, dict) or not isinstance(raw_exp.get("automaton"), str):
        message = "The experiment must be an object with an 'automaton' path."
        result["experiment_issues"].append(make_issue("SCHEMA", [], message))
        return result

    # 2. read the automaton it points to; that path comes from the file, so it is checked first
    automaton_file = safe_path(raw_exp["automaton"], project_root)
    if automaton_file is None:
        message = f"The automaton path '{raw_exp['automaton']}' must stay inside the project folder."
        result["experiment_issues"].append(make_issue("SCHEMA", ["automaton"], message))
        return result

    raw_aut, problem = read_json(automaton_file, raw_exp["automaton"])
    if problem is not None:
        problem["message"] = "Cannot use the automaton file: " + problem["message"]
        problem["where"] = "/automaton"
        result["experiment_issues"].append(problem)
        return result

    # 3. validate the automaton, then the experiment
    result["automaton_issues"] = validate_automaton(raw_aut)
    if has_errors(result["automaton_issues"]):
        return result

    result["experiment_issues"] = validate_experiment(raw_exp, raw_aut)
    if has_errors(result["experiment_issues"]):
        return result

    # 4. everything is correct: translate
    result["automaton"] = load_automaton(raw_aut, raw_exp)
    result["experiment"] = load_experiment(raw_exp, result["automaton"])
    return result


def safe_path(path: str, project_root) -> Path | None:
    """The full path, or None if it leaves the project folder.

    A path written inside a file is never trusted: "../../secret.json" or
    "C:/Windows/..." could make us read any file on the computer.
    """
    root = os.path.realpath(project_root)
    full_path = os.path.realpath(os.path.join(root, path))
    try:
        if os.path.commonpath([root, full_path]) != root:
            return None
    except ValueError:
        # on Windows: the two paths are on different drives
        return None
    return Path(full_path)


def read_json(full_path: Path, shown_name):
    """Return (data, None) if the file can be read, or (None, issue) if not.
    shown_name is the path as the user wrote it, used in messages."""
    try:
        with open(full_path, encoding="utf-8") as file:
            return json.load(file), None
    except FileNotFoundError:
        return None, make_issue("SCHEMA", [], f"File '{shown_name}' does not exist.")
    except json.JSONDecodeError as error:
        message = (f"File '{shown_name}' is not valid JSON: {error.msg} "
                   f"(line {error.lineno}, column {error.colno}).")
        return None, make_issue("SCHEMA", [], message)


# =====================================================================
# The automaton
# =====================================================================

def load_automaton(raw_aut: dict, raw_exp: dict) -> dict:
    """The loaded automaton (contract 5.4), with the final parameters and the world."""

    # 1. fields copied from the file (missing optional fields become empty)
    automaton = {
        "format_version": raw_aut["format_version"],
        "kind": raw_aut["kind"],
        "name": raw_aut["name"],
        "description": raw_aut.get("description", ""),
        "states": copy.deepcopy(raw_aut["states"]),
        "variables": copy.deepcopy(raw_aut.get("variables", [])),
        "neighborhood": copy.deepcopy(raw_aut["neighborhood"]),
        "rules": copy.deepcopy(raw_aut["rules"]),
        # agent fields (Phase 3), copied as they are; Person 4 finalizes their details
        "agent_types": copy.deepcopy(raw_aut.get("agent_types", [])),
        "agent_order": raw_aut.get("agent_order", "fixed"),
        "agent_rules": copy.deepcopy(raw_aut.get("agent_rules", [])),
    }

    # 2. states: codes 0, 1, 2... in the order of the list
    automaton["state_codes"] = {}
    automaton["state_names"] = []
    automaton["symbols"] = {}
    automaton["colors"] = []
    for code, state in enumerate(raw_aut["states"]):
        automaton["state_codes"][state["name"]] = code
        automaton["state_names"].append(state["name"])
        automaton["symbols"][state["symbol"]] = code
        automaton["colors"].append(state.get("color"))   # None if missing

    # 3. variables
    automaton["variable_types"] = {}
    automaton["variable_defaults"] = {}
    for variable in raw_aut.get("variables", []):
        automaton["variable_types"][variable["name"]] = variable["type"]
        automaton["variable_defaults"][variable["name"]] = variable["default"]

    # 4. parameters, neighbors, world
    automaton["parameters"] = merge_parameters(raw_aut, raw_exp)
    automaton["neighbor_offsets"] = list(NEIGHBOR_OFFSETS[raw_aut["neighborhood"]["type"]])
    automaton["world"] = _load_world(raw_exp["world"], automaton)

    # 5. rules, translated
    automaton["compiled_rules"] = []
    for rule in raw_aut["rules"]:
        automaton["compiled_rules"].append(_compile_rule(rule, automaton))
    return automaton


def _load_world(raw_world: dict, automaton: dict) -> dict:
    """The boundary, as the engine needs it."""
    if raw_world["boundary"] == "toroidal":
        return {"boundary": "toroidal", "outside_code": None, "outside_values": {}}

    outside = raw_world["outside"]

    # Every variable gets an outside value: the one given, or its default.
    outside_values = dict(automaton["variable_defaults"])
    for name, value in outside.get("values", {}).items():
        outside_values[name] = value

    return {
        "boundary": "fixed",
        "outside_code": automaton["state_codes"][outside["state"]],
        "outside_values": outside_values,
    }


def _compile_rule(rule: dict, automaton: dict) -> dict:
    """One rule with names replaced by codes and parameters by numbers.

    {"from": "OFF", "to": "FIRING", "when": {"count": "FIRING", "min": 2, "max": 2}}
    becomes
    {"from_codes": [0], "to_code": 1, "set": {}, "when": {"count": 1, "min": 2, "max": 2}, "probability": 1.0}
    """
    codes = automaton["state_codes"]
    parameters = automaton["parameters"]

    # from: one name or a list; "*" means every state
    from_names = rule["from"]
    if isinstance(from_names, str):
        from_names = [from_names]
    from_codes = []
    if "*" in from_names:
        from_codes = list(codes.values())
    else:
        for name in from_names:
            from_codes.append(codes[name])

    to_code = None
    if "to" in rule:
        to_code = codes[rule["to"]]

    when = None
    if "when" in rule:
        when = _compile_condition(rule["when"], codes, parameters)

    return {
        "from_codes": from_codes,
        "to_code": to_code,
        "set": dict(rule.get("set", {})),
        "when": when,
        "probability": float(_number(rule.get("probability", 1), parameters)),
    }


def _compile_condition(condition: dict, codes: dict, parameters: dict) -> dict:
    """A condition with names replaced by codes and parameters by numbers."""
    if "count" in condition:
        return _compile_count(condition, codes, parameters)

    if "pattern" in condition:
        pattern_codes = []
        for name in condition["pattern"]:
            pattern_codes.append(codes[name])
        return {"pattern": pattern_codes}

    if "all" in condition or "any" in condition:
        key = "all" if "all" in condition else "any"
        parts = []
        for part in condition[key]:
            parts.append(_compile_condition(part, codes, parameters))
        return {key: parts}

    if "not" in condition:
        return {"not": _compile_condition(condition["not"], codes, parameters)}

    # "expr": kept as text; the formula interpreter comes in Phase 2
    return {"expr": condition["expr"]}


def _compile_count(condition: dict, codes: dict, parameters: dict) -> dict:
    """{"count": "FIRE", "min": "k"} becomes {"count": 2, "min": 3}.

    A list of states stays a list: {"count": ["I1", "I2"]} becomes {"count": [1, 2]}.
    """
    if isinstance(condition["count"], list):
        compiled = {"count": []}
        for name in condition["count"]:
            compiled["count"].append(codes[name])
    else:
        compiled = {"count": codes[condition["count"]]}
    if "min" in condition:
        compiled["min"] = int(_number(condition["min"], parameters))
    if "max" in condition:
        compiled["max"] = int(_number(condition["max"], parameters))
    if "in" in condition:
        compiled["in"] = []
        for value in condition["in"]:
            compiled["in"].append(int(_number(value, parameters)))
    return compiled


def _number(value, parameters: dict):
    """A number stays a number; a text is a parameter name and becomes its value."""
    if isinstance(value, str):
        return parameters[value]
    return value


# =====================================================================
# The experiment
# =====================================================================

def load_experiment(raw_exp: dict, automaton: dict) -> dict:
    """The experiment's fields, with missing optional ones filled in.

    The final parameter values are in automaton["parameters"];
    here "parameters" keeps only the experiment's own overrides.
    """
    initial_state = copy.deepcopy(raw_exp["initial_state"])
    if "values" not in initial_state:
        initial_state["values"] = {}

    return {
        "format_version": raw_exp["format_version"],
        "automaton": raw_exp["automaton"],
        "parameters": dict(raw_exp.get("parameters", {})),
        "world": copy.deepcopy(raw_exp["world"]),
        "initial_state": initial_state,
        "generations": raw_exp["generations"],
        "storage": dict(raw_exp["storage"]),
        "seed": raw_exp.get("seed"),
    }