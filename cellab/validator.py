"""Checks whether automaton and experiment files are correct (contract 5.1 and 5.5).

Two kinds of checks, in this order:
  1. FORM  - is every field there, with the right type?  (the JSON Schemas in schema/)
  2. SENSE - does the content make sense?                 (the functions in this file)
If the form is wrong, the sense checks do not run.

Every problem becomes an "issue" with 4 fields:
  {"level": "error", "code": "UNKNOWN_STATE", "where": "/rules/2/when/count", "message": "..."}
All issues found are returned together, so the user sees everything to fix.

Every sense check below has the same shape:
    for <each thing in the file>:
        if <something is wrong>:
            issues.append(make_issue(CODE, where, message))
"""

from __future__ import annotations

import json

from jsonschema import Draft202012Validator

from cellab.conventions import NEIGHBOR_OFFSETS, PROJECT_ROOT, json_pointer, merge_parameters
from cellab.expressions import ExpressionError, parse_expression, rule_formulas, uses_random

SCHEMA_DIR = PROJECT_ROOT / "schema"


# =====================================================================
# The two functions everybody calls
# =====================================================================

def validate_automaton(raw_aut) -> list:
    """All problems of an automaton file."""
    issues = check_form(raw_aut, "automaton")
    if has_errors(issues):
        return issues

    issues += _check_states(raw_aut)
    issues += _check_variables(raw_aut)
    issues += _check_rules(raw_aut)
    issues += _check_values(raw_aut, raw_aut.get("parameters", {}), None)
    issues += _check_unreachable_rules(raw_aut)
    issues += _check_unused(raw_aut)
    return issues


def validate_experiment(raw_exp, raw_aut: dict) -> list:
    """All problems of an experiment file. raw_aut must already be valid."""
    issues = check_form(raw_exp, "experiment")
    if has_errors(issues):
        return issues

    issues += _check_parameter_overrides(raw_exp, raw_aut)
    issues += _check_world(raw_exp, raw_aut)
    issues += _check_initial_cells(raw_exp, raw_aut)
    issues += _check_initial_values(raw_exp, raw_aut)
    issues += _check_seed(raw_exp, raw_aut)
    return issues


# =====================================================================
# Issues
# =====================================================================

def make_issue(code: str, path: list, message: str, level: str = "error") -> dict:
    """One issue. path is a list like ["rules", 2, "when"]."""
    return {"level": level, "code": code, "where": json_pointer(path), "message": message}


def has_errors(issues: list) -> bool:
    """True if at least one issue is an error. Warnings do not stop anything."""
    for issue in issues:
        if issue["level"] == "error":
            return True
    return False


# =====================================================================
# 1. FORM
# =====================================================================

def check_form(raw, kind: str) -> list:
    """Compare a file with its schema. kind is "automaton" or "experiment"."""
    with open(SCHEMA_DIR / (kind + ".schema.json"), encoding="utf-8") as file:
        schema = json.load(file)

    issues = []
    for error in Draft202012Validator(schema).iter_errors(raw):
        issues.append(make_issue("SCHEMA", list(error.absolute_path), error.message))

    issues.sort(key=_where)   # in the order of the file
    return issues


def _where(issue: dict) -> str:
    return issue["where"]


# =====================================================================
# Small helpers
# =====================================================================

def _names(items: list) -> list:
    """["OFF", "FIRING"] from [{"name": "OFF", ...}, {"name": "FIRING", ...}]."""
    names = []
    for item in items:
        names.append(item["name"])
    return names


def _from_names(rule: dict, all_states: list) -> list:
    """The states a rule starts from, always as a list. "*" means every state."""
    value = rule["from"]
    if isinstance(value, str):
        value = [value]
    if "*" in value:
        return list(all_states)
    return list(value)


def _rule_conditions(rule: dict, rule_path: list) -> list:
    """The simple conditions of a rule (count, pattern, expr), each with its path.

    all / any / not are opened up, so the callers never need recursion.
    Example: {"all": [A, B]} gives [(A, [..., "when", "all", 0]), (B, [..., "when", "all", 1])].
    """
    if "when" not in rule:
        return []
    return _simple_conditions(rule["when"], rule_path + ["when"])


def _simple_conditions(condition: dict, path: list) -> list:
    if "all" in condition or "any" in condition:
        key = "all" if "all" in condition else "any"
        result = []
        for position, part in enumerate(condition[key]):
            result += _simple_conditions(part, path + [key, position])
        return result
    if "not" in condition:
        return _simple_conditions(condition["not"], path + ["not"])
    return [(condition, path)]


def _resolve(value, parameters: dict):
    """A number stays a number; a text is a parameter name and becomes its value.
    Returns None for an unknown parameter (reported by another check)."""
    if isinstance(value, str):
        return parameters.get(value)
    return value


def _parameter_names(values: list) -> list:
    """Keep only the texts: those are parameter names."""
    names = []
    for value in values:
        if isinstance(value, str):
            names.append(value)
    return names


def _defined_list(names: list) -> str:
    """Text added to messages, e.g. " (defined: OFF, FIRING)"."""
    if len(names) == 0:
        return " (none are defined)"
    return " (defined: " + ", ".join(names) + ")"


def _unknown_state(name: str, path: list, states: list) -> list:
    if name in states:
        return []
    message = f"State '{name}' is not defined{_defined_list(states)}."
    return [make_issue("UNKNOWN_STATE", path, message)]


def _unknown_variable(name: str, path: list, variables: list) -> list:
    if name in variables:
        return []
    message = f"Variable '{name}' is not defined{_defined_list(variables)}."
    return [make_issue("UNKNOWN_VARIABLE", path, message)]


def _unknown_parameter(value, path: list, parameters: dict) -> list:
    if not isinstance(value, str) or value in parameters:
        return []
    message = f"Parameter '{value}' is not defined{_defined_list(list(parameters))}."
    return [make_issue("UNKNOWN_PARAMETER", path, message)]


# =====================================================================
# 2. SENSE: the automaton
# =====================================================================

def _check_states(raw_aut: dict) -> list:
    """Names and symbols must be unique; a symbol is exactly one character."""
    issues = []
    seen_names = []
    seen_symbols = []

    for index, state in enumerate(raw_aut["states"]):
        name = state["name"]
        symbol = state["symbol"]

        if name in seen_names:
            message = f"State name '{name}' is used more than once."
            issues.append(make_issue("DUPLICATE_NAME", ["states", index, "name"], message))

        if len(symbol) != 1:
            message = f"Symbol '{symbol}' of state '{name}' must be exactly one character."
            issues.append(make_issue("BAD_SYMBOL", ["states", index, "symbol"], message))
        elif symbol in seen_symbols:
            message = f"Symbol '{symbol}' is used by more than one state."
            issues.append(make_issue("DUPLICATE_SYMBOL", ["states", index, "symbol"], message))

        seen_names.append(name)
        seen_symbols.append(symbol)
    return issues


def _check_variables(raw_aut: dict) -> list:
    """Variable names must be unique and different from parameter names."""
    issues = []
    parameters = raw_aut.get("parameters", {})
    seen = []

    for index, variable in enumerate(raw_aut.get("variables", [])):
        name = variable["name"]
        if name in seen:
            message = f"Variable name '{name}' is used more than once."
            issues.append(make_issue("DUPLICATE_NAME", ["variables", index, "name"], message))
        if name in parameters:
            message = f"'{name}' is both a variable and a parameter."
            issues.append(make_issue("NAME_CONFLICT", ["variables", index, "name"], message))
        seen.append(name)
    return issues


def _check_rules(raw_aut: dict) -> list:
    """Every name used in a rule must be declared; a rule must change something."""
    issues = []
    names = {
        "states": _names(raw_aut["states"]),
        "variables": _names(raw_aut.get("variables", [])),
        "parameters": raw_aut.get("parameters", {}),
        "is_1d": raw_aut["neighborhood"]["type"] == "1d",
    }

    for index, rule in enumerate(raw_aut["rules"]):
        path = ["rules", index]
        issues += _check_rule_parts(rule, path, names)
        for condition, condition_path in _rule_conditions(rule, path):
            issues += _check_condition_names(condition, condition_path, names)
    return issues


def _check_rule_parts(rule: dict, path: list, names: dict) -> list:
    """from, to, set and probability of one rule."""
    issues = []

    # from: one name or a list of names; "*" means any state
    if isinstance(rule["from"], str):
        if rule["from"] != "*":
            issues += _unknown_state(rule["from"], path + ["from"], names["states"])
    else:
        for position, name in enumerate(rule["from"]):
            if name != "*":
                issues += _unknown_state(name, path + ["from", position], names["states"])

    if "to" in rule:
        issues += _unknown_state(rule["to"], path + ["to"], names["states"])

    for variable, formula in rule.get("set", {}).items():
        issues += _unknown_variable(variable, path + ["set", variable], names["variables"])
        issues += _check_formula(formula, "number", path + ["set", variable], names)

    if "to" not in rule and len(rule.get("set", {})) == 0:
        message = "Rule has neither 'to' nor 'set', so it changes nothing."
        issues.append(make_issue("EMPTY_RULE", path, message))

    issues += _unknown_parameter(rule.get("probability", 1), path + ["probability"], names["parameters"])
    return issues


def _check_condition_names(condition: dict, path: list, names: dict) -> list:
    """One simple condition: the states and parameters it uses must exist."""
    if "count" in condition:
        return _check_count_names(condition, path, names)
    if "pattern" in condition:
        return _check_pattern_names(condition["pattern"], path + ["pattern"], names)
    if "expr" in condition:
        return _check_formula(condition["expr"], "bool", path + ["expr"], names)
    return []


def _check_formula(text: str, expected: str, path: list, names: dict) -> list:
    """A formula must use only allowed elements and known names, and give the right kind of result:
    true/false for a condition, a number for a set formula."""
    try:
        parse_expression(text, names["states"], names["variables"], names["parameters"], expected)
    except ExpressionError as error:
        return [make_issue(error.code, path, error.message)]
    return []


def _check_count_names(condition: dict, path: list, names: dict) -> list:
    """A count condition: the counted states and the parameters in min/max/in must exist."""
    issues = []
    if isinstance(condition["count"], list):
        for position, name in enumerate(condition["count"]):
            issues += _unknown_state(name, path + ["count", position], names["states"])
    else:
        issues += _unknown_state(condition["count"], path + ["count"], names["states"])

    for key in ["min", "max"]:
        if key in condition:
            issues += _unknown_parameter(condition[key], path + [key], names["parameters"])
    for position, value in enumerate(condition.get("in", [])):
        issues += _unknown_parameter(value, path + ["in", position], names["parameters"])
    return issues


def _check_pattern_names(pattern: list, path: list, names: dict) -> list:
    """A pattern condition: only for 1d, exactly 3 elements, known states."""
    issues = []
    if not names["is_1d"]:
        message = "'pattern' can only be used with a 1d neighborhood."
        issues.append(make_issue("PATTERN_NOT_1D", path, message))
    elif len(pattern) != 3:
        message = f"'pattern' must have exactly 3 elements [left, center, right], not {len(pattern)}."
        issues.append(make_issue("PATTERN_NOT_1D", path, message))
    for position, name in enumerate(pattern):
        issues += _unknown_state(name, path + [position], names["states"])
    return issues


def _check_values(raw_aut: dict, parameters: dict, overrides) -> list:
    """Probabilities must be in [0, 1]; neighbor counts must be possible.

    Checked with the given parameter values, so it works for both files:
      overrides = None -> checking the automaton: report every problem at the rule.
      overrides = {...} -> checking the experiment: report only the problems caused
                           by a parameter the experiment overrides, at /parameters/<name>.
    """
    issues = []
    neighbor_count = len(NEIGHBOR_OFFSETS[raw_aut["neighborhood"]["type"]])

    for index, rule in enumerate(raw_aut["rules"]):
        path = ["rules", index]

        raw_probability = rule.get("probability", 1)
        probability = _resolve(raw_probability, parameters)
        if probability is not None and not 0 <= probability <= 1:
            message = f"Probability must be between 0 and 1, but it is {probability}."
            _add_value_issue(issues, "BAD_PROBABILITY", path + ["probability"], message,
                             [raw_probability], overrides)

        for condition, condition_path in _rule_conditions(rule, path):
            if "count" in condition:
                _check_count_values(issues, condition, condition_path, parameters, neighbor_count, overrides)
    return issues


def _check_count_values(issues: list, condition: dict, path: list, parameters: dict,
                        neighbor_count: int, overrides) -> None:
    """min, max and in of one count condition must be possible numbers of neighbors."""
    for key in ["min", "max"]:
        if key in condition:
            problem = _count_problem(_resolve(condition[key], parameters), neighbor_count)
            if problem != "":
                _add_value_issue(issues, "IMPOSSIBLE_COUNT", path + [key],
                                 f"'{key}' {problem}", [condition[key]], overrides)

    for position, raw_value in enumerate(condition.get("in", [])):
        problem = _count_problem(_resolve(raw_value, parameters), neighbor_count)
        if problem != "":
            _add_value_issue(issues, "IMPOSSIBLE_COUNT", path + ["in", position],
                             f"Value {problem}", [raw_value], overrides)

    low = _resolve(condition.get("min"), parameters)
    high = _resolve(condition.get("max"), parameters)
    if low is not None and high is not None and low > high:
        message = f"'min' ({low}) is greater than 'max' ({high}), so the condition is never true."
        _add_value_issue(issues, "IMPOSSIBLE_COUNT", path, message,
                         [condition.get("min"), condition.get("max")], overrides)


def _count_problem(value, neighbor_count: int) -> str:
    """"" if value can be a number of neighbors, otherwise what is wrong with it."""
    if value is None:
        return ""   # unknown parameter, reported by _check_rules
    if isinstance(value, bool) or not (isinstance(value, int) or float(value).is_integer()):
        return f"must be a whole number, but it is {value}."
    if value < 0 or value > neighbor_count:
        return f"is {value}, but a cell has only {neighbor_count} neighbors."
    return ""


def _add_value_issue(issues: list, code: str, path: list, message: str, raw_values: list, overrides) -> None:
    """Add a problem found by _check_values, at the right place (see its docstring)."""
    if overrides is None:
        issues.append(make_issue(code, path, message))
        return

    for name in _parameter_names(raw_values):
        if name in overrides:
            message = message.rstrip(".") + f" (value set here; used by the automaton at {json_pointer(path)})."
            issues.append(make_issue(code, ["parameters", name], message))
            return


def _check_unreachable_rules(raw_aut: dict) -> list:
    """Warnings for two simple cases of a rule that can never apply:
      - its probability is 0;
      - an earlier rule covers all its states, has no condition and probability 1,
        so the earlier rule always wins first.
    """
    issues = []
    states = _names(raw_aut["states"])
    parameters = raw_aut.get("parameters", {})
    rules = raw_aut["rules"]

    for index, rule in enumerate(rules):
        if _resolve(rule.get("probability", 1), parameters) == 0:
            message = "Rule has probability 0, so it never applies."
            issues.append(make_issue("UNREACHABLE_RULE", ["rules", index], message, level="warning"))
            continue

        for earlier_index in range(index):
            if _always_wins_before(rules[earlier_index], rule, states, parameters):
                message = (f"Rule {earlier_index} always applies first to the same states, "
                           f"so this rule never applies.")
                issues.append(make_issue("UNREACHABLE_RULE", ["rules", index], message, level="warning"))
                break
    return issues


def _always_wins_before(earlier: dict, rule: dict, states: list, parameters: dict) -> bool:
    """True if `earlier` has no condition, probability 1 and covers all the states of `rule`."""
    if "when" in earlier:
        return False
    if _resolve(earlier.get("probability", 1), parameters) != 1:
        return False
    earlier_states = _from_names(earlier, states)
    for state in _from_names(rule, states):
        if state not in earlier_states:
            return False
    return True


def _check_unused(raw_aut: dict) -> list:
    """Warnings for states and parameters that no rule mentions."""
    issues = []
    used = _collect_used_names(raw_aut)

    # Agent automata also use states in agent_rules; those are checked in Phase 3.
    if raw_aut["kind"] == "cells":
        for index, name in enumerate(_names(raw_aut["states"])):
            if name not in used["states"]:
                message = f"State '{name}' is not used by any rule."
                issues.append(make_issue("UNUSED_STATE", ["states", index], message, level="warning"))

    for name in raw_aut.get("parameters", {}):
        if name not in used["parameters"]:
            message = f"Parameter '{name}' is not used by any rule."
            issues.append(make_issue("UNUSED_PARAMETER", ["parameters", name], message, level="warning"))
    return issues


def _collect_used_names(raw_aut: dict) -> dict:
    """Everything the rules mention: {"states": [...], "parameters": [...]}."""
    used = {"states": [], "parameters": []}
    states = _names(raw_aut["states"])

    for index, rule in enumerate(raw_aut["rules"]):
        used["states"].extend(_from_names(rule, states))
        if "to" in rule:
            used["states"].append(rule["to"])
        used["parameters"].extend(_parameter_names([rule.get("probability")]))
        for formula in rule.get("set", {}).values():
            _add_formula_names(used, formula, raw_aut)

        for condition, _path in _rule_conditions(rule, ["rules", index]):
            if "count" in condition:
                if isinstance(condition["count"], list):
                    used["states"].extend(condition["count"])
                else:
                    used["states"].append(condition["count"])
                used["parameters"].extend(_parameter_names([condition.get("min"), condition.get("max")]))
                used["parameters"].extend(_parameter_names(condition.get("in", [])))
            elif "pattern" in condition:
                used["states"].extend(condition["pattern"])
            elif "expr" in condition:
                _add_formula_names(used, condition["expr"], raw_aut)
    return used


def _add_formula_names(used: dict, text: str, raw_aut: dict) -> None:
    """The states and parameters a formula really uses (a broken formula is reported elsewhere)."""
    try:
        formula = parse_expression(text, _names(raw_aut["states"]), _names(raw_aut.get("variables", [])),
                                   raw_aut.get("parameters", {}))
    except ExpressionError:
        return
    used["states"].extend(formula["states"])
    used["parameters"].extend(formula["parameters"])


# =====================================================================
# 2. SENSE: the experiment
# =====================================================================

def _check_parameter_overrides(raw_exp: dict, raw_aut: dict) -> list:
    """The experiment may only override parameters that exist, with sensible values."""
    issues = []
    known = raw_aut.get("parameters", {})
    overrides = raw_exp.get("parameters", {})

    for name in overrides:
        if name not in known:
            message = f"The automaton has no parameter '{name}'{_defined_list(list(known))}."
            issues.append(make_issue("UNKNOWN_PARAMETER", ["parameters", name], message))

    issues += _check_values(raw_aut, merge_parameters(raw_aut, raw_exp), overrides)
    return issues


def _check_world(raw_exp: dict, raw_aut: dict) -> list:
    """A fixed boundary needs to say what is outside, using known names."""
    issues = []
    world = raw_exp["world"]

    if world["boundary"] == "fixed" and "outside" not in world:
        message = "A fixed boundary needs 'outside': the state of the cells outside the grid."
        issues.append(make_issue("MISSING_OUTSIDE", ["world"], message))

    if "outside" in world:
        states = _names(raw_aut["states"])
        variables = _names(raw_aut.get("variables", []))
        issues += _unknown_state(world["outside"]["state"], ["world", "outside", "state"], states)
        for name in world["outside"].get("values", {}):
            issues += _unknown_variable(name, ["world", "outside", "values", name], variables)
    return issues


def _check_initial_cells(raw_exp: dict, raw_aut: dict) -> list:
    """The initial cells, for each of the 4 modes."""
    cells = raw_exp["initial_state"]["cells"]
    path = ["initial_state", "cells"]
    states = _names(raw_aut["states"])
    rows = raw_exp["world"]["rows"]
    cols = raw_exp["world"]["cols"]

    if cells["mode"] == "fill":
        return _unknown_state(cells["state"], path + ["state"], states)
    if cells["mode"] == "random":
        return _check_proportions(cells["proportions"], path + ["proportions"], states)
    if cells["mode"] == "manual":
        issues = _check_grid_size(cells["grid"], rows, cols, path + ["grid"], "cells")
        issues += _check_symbols(cells["grid"], path + ["grid"], raw_aut)
        return issues
    return _check_pattern(cells, path, raw_aut, rows, cols)


def _check_proportions(proportions: dict, path: list, states: list) -> list:
    """Known states, and the shares add up to 1."""
    issues = []
    total = 0
    for name, share in proportions.items():
        issues += _unknown_state(name, path + [name], states)
        total = total + share
    if abs(total - 1) > 1e-9:
        message = f"Proportions must add up to 1, but they add up to {total}."
        issues.append(make_issue("PROPORTIONS_SUM", path, message))
    return issues


def _check_pattern(cells: dict, path: list, raw_aut: dict, rows: int, cols: int) -> list:
    """Pattern mode: known background, a rectangular pattern of known symbols, inside the world."""
    issues = []
    pattern = cells["pattern"]
    issues += _unknown_state(cells["background"], path + ["background"], _names(raw_aut["states"]))

    width = len(pattern[0])
    for index, line in enumerate(pattern):
        if len(line) != width:
            message = f"All rows of the pattern must have the same length ({width})."
            issues.append(make_issue("GRID_SIZE", path + ["pattern", index], message))
    issues += _check_symbols(pattern, path + ["pattern"], raw_aut)

    top = cells["position"][0]
    left = cells["position"][1]
    fits = top >= 0 and left >= 0 and top + len(pattern) <= rows and left + width <= cols
    if not fits:
        message = (f"A {len(pattern)}x{width} pattern at row {top}, column {left} "
                   f"does not fit in a {rows}x{cols} world.")
        issues.append(make_issue("OUT_OF_BOUNDS", path, message))
    return issues


def _check_grid_size(grid: list, rows: int, cols: int, path: list, what: str) -> list:
    """A manual grid must have exactly the size of the world. what is "cells" or "values"."""
    issues = []
    if len(grid) != rows:
        message = f"Grid has {len(grid)} rows, but the world has {rows}."
        issues.append(make_issue("GRID_SIZE", path, message))
    for index, line in enumerate(grid):
        if len(line) != cols:
            message = f"Row {index} has {len(line)} {what}, but the world has {cols} columns."
            issues.append(make_issue("GRID_SIZE", path + [index], message))
    return issues


def _check_symbols(grid: list, path: list, raw_aut: dict) -> list:
    """Every character of a grid must be the symbol of a state."""
    issues = []
    symbols = []
    for state in raw_aut["states"]:
        symbols.append(state["symbol"])

    for index, line in enumerate(grid):
        unknown = []
        for character in line:
            if character not in symbols and character not in unknown:
                unknown.append(character)
        if len(unknown) > 0:
            message = f"Row {index} uses unknown symbols {unknown} (known: {symbols})."
            issues.append(make_issue("UNKNOWN_SYMBOL", path + [index], message))
    return issues


def _check_initial_values(raw_exp: dict, raw_aut: dict) -> list:
    """Initial values of the variables: known names, grids and points inside the world."""
    issues = []
    variables = _names(raw_aut.get("variables", []))
    rows = raw_exp["world"]["rows"]
    cols = raw_exp["world"]["cols"]

    for name, spec in raw_exp["initial_state"].get("values", {}).items():
        path = ["initial_state", "values", name]
        if name not in variables:
            issues += _unknown_variable(name, path, variables)
            continue

        if spec["mode"] == "manual":
            issues += _check_grid_size(spec["grid"], rows, cols, path + ["grid"], "values")

        if spec["mode"] == "constant":
            issues += _check_points(spec.get("points", []), path + ["points"], rows, cols)
    return issues


def _check_points(points: list, path: list, rows: int, cols: int) -> list:
    """Every point of a constant value must be inside the world."""
    issues = []
    for index, point in enumerate(points):
        row = point["position"][0]
        col = point["position"][1]
        if not (0 <= row < rows and 0 <= col < cols):
            message = f"Point [{row}, {col}] is outside the {rows}x{cols} world."
            issues.append(make_issue("OUT_OF_BOUNDS", path + [index], message))
    return issues


def _check_seed(raw_exp: dict, raw_aut: dict) -> list:
    """Anything random needs a seed, so the experiment can be repeated exactly."""
    if "seed" in raw_exp:
        return []

    reasons = []
    if raw_exp["initial_state"]["cells"]["mode"] == "random":
        reasons.append("the initial cells are random")

    for name, spec in raw_exp["initial_state"].get("values", {}).items():
        if spec["mode"] == "random":
            reasons.append(f"the initial values of '{name}' are random")

    parameters = merge_parameters(raw_aut, raw_exp)
    for index, rule in enumerate(raw_aut["rules"]):
        reasons += _random_parts_of_rule(index, rule, parameters)

    if raw_aut.get("agent_order") == "random":
        reasons.append("agents act in random order")

    if len(reasons) == 0:
        return []
    message = "The experiment needs a 'seed' because " + "; ".join(reasons) + "."
    return [make_issue("MISSING_SEED", [], message)]


def _random_parts_of_rule(index: int, rule: dict, parameters: dict) -> list:
    """Why a rule is random: a probability below 1, or random() in one of its formulas."""
    reasons = []
    probability = _resolve(rule.get("probability", 1), parameters)
    if probability is not None and probability < 1:
        reasons.append(f"rule {index} has probability {probability}")
    for text in rule_formulas(rule):
        if uses_random(text):
            reasons.append(f"rule {index} uses random() in '{text}'")
    return reasons
