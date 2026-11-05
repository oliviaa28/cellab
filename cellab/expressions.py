"""Formulas: reading them safely and computing them on the whole grid at once.

A formula is a short text written in the automaton, for example
    "grains - 4 * (grains >= 4) + count_where(grains >= 4)"      (a "set" formula)
    "count(B) > count(A)"                                         (an "expr" condition)

Only these elements are accepted (section 5.6 of the plan):
    numbers, variables, parameters        4   grains   p
    arithmetic                            +  -  *  /  //  %   and parentheses
    comparisons and logic                 ==  !=  <  <=  >  >=   and  or  not
    count(S)                              how many neighbors are in state S
    count_where(c)                        how many neighbors make c true (names mean the neighbor's values)
    sum(x)  avg(x)  min(x)  max(x)        over the neighbors' values of x
    random()  abs(x)  floor(x)

The text is read once with the ast module and turned into small tuples, for example
    "grains >= 4"  ->  ("compare", [">="], [("var", "grains"), ("num", 4)])
Anything not in the list above is refused. The text is never run with eval or exec.

parse_expression    -> checks a formula, returns it ready to compute (or raises ExpressionError)
evaluate_expression -> computes it for every cell at once, with NumPy matrices
"""

from __future__ import annotations

import ast
from functools import lru_cache

import numpy as np

# Functions that look at the neighbors; their argument is computed once for each neighbor.
NEIGHBOR_FUNCTIONS = ["count_where", "sum", "avg", "min", "max"]
# Functions on the current cell.
CELL_FUNCTIONS = ["abs", "floor"]
ALL_FUNCTIONS = ["count"] + NEIGHBOR_FUNCTIONS + ["random"] + CELL_FUNCTIONS

_ARITHMETIC = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/", ast.FloorDiv: "//", ast.Mod: "%"}
_COMPARISONS = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">", ast.GtE: ">="}

# How a refused element is named in the error message.
_REFUSED = {
    ast.Pow: "'**'", ast.BitAnd: "'&'", ast.BitOr: "'|'", ast.BitXor: "'^'",
    ast.LShift: "'<<'", ast.RShift: "'>>'", ast.MatMult: "'@'", ast.Invert: "'~'",
    ast.Attribute: "attribute access ('.')", ast.Subscript: "indexing ('[...]')",
    ast.Lambda: "lambda", ast.IfExp: "'... if ... else ...'", ast.Is: "'is'", ast.IsNot: "'is not'",
    ast.In: "'in'", ast.NotIn: "'not in'", ast.List: "lists", ast.Tuple: "tuples",
    ast.Dict: "dictionaries", ast.Set: "sets", ast.ListComp: "comprehensions",
    ast.GeneratorExp: "comprehensions", ast.NamedExpr: "':='", ast.JoinedStr: "text",
}


class ExpressionError(ValueError):
    """A formula that cannot be used. code is a validator code (BAD_EXPRESSION, ...)."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


class ExpressionRuntimeError(ArithmeticError):
    """A formula that failed while computing (for example a division by 0)."""


# =====================================================================
# Reading a formula
# =====================================================================

def parse_expression(text: str, states, variables, parameters, expected: str | None = None) -> dict:
    """Check a formula and return it ready to compute.

    states, variables, parameters: the names the formula may use.
    expected: "bool" for a condition (when/expr), "number" for a set formula, None for no check.

    Returns {"text", "tree", "type", "random_count", "states", "variables", "parameters"}:
    the last three are the names the formula really uses.
    Raises ExpressionError if the formula is not allowed.
    """
    return _parse_cached(text, tuple(states), tuple(variables), tuple(parameters), expected)


@lru_cache(maxsize=1024)
def _parse_cached(text, states, variables, parameters, expected) -> dict:
    if not isinstance(text, str) or text.strip() == "":
        raise ExpressionError("BAD_EXPRESSION", "The formula is empty.")
    try:
        body = ast.parse(text.strip(), mode="eval").body
    except SyntaxError as error:
        raise ExpressionError("BAD_EXPRESSION", f"'{text}' is not a valid formula ({error.msg}).") from None
    except (RecursionError, MemoryError, ValueError):
        raise ExpressionError("BAD_EXPRESSION", f"'{text}' is too long or too deeply nested.") from None

    reader = _Reader(states, variables, parameters)
    tree, kind = reader.read(body, inside_neighbors=False)

    if expected == "bool" and kind != "bool":
        raise ExpressionError("EXPRESSION_TYPE",
                              f"The condition '{text}' gives a number, not true/false "
                              f"(for example, write '... > 0').")
    if expected == "number" and kind != "number":
        raise ExpressionError("EXPRESSION_TYPE",
                              f"The formula '{text}' gives true/false, not a number "
                              f"(to get 1 or 0, write '1 * (...)').")

    return {
        "text": text,
        "tree": tree,
        "type": kind,
        "random_count": reader.random_count,
        "states": sorted(reader.used["states"]),
        "variables": sorted(reader.used["variables"]),
        "parameters": sorted(reader.used["parameters"]),
    }


class _Reader:
    """Turns an ast tree into tuples, checking every element on the way.

    read() returns (tuple, kind), where kind is "bool" (true/false) or "number".
    """

    def __init__(self, states, variables, parameters):
        self.states = states
        self.variables = variables
        self.parameters = parameters
        self.random_count = 0     # random() calls get the numbers 0, 1, 2... in reading order
        self.used = {"states": set(), "variables": set(), "parameters": set()}

    def read(self, node, inside_neighbors: bool):
        if isinstance(node, ast.Constant):
            return self._constant(node)
        if isinstance(node, ast.Name):
            return self._name(node.id)
        if isinstance(node, ast.BinOp):
            return self._arithmetic(node, inside_neighbors)
        if isinstance(node, ast.UnaryOp):
            return self._unary(node, inside_neighbors)
        if isinstance(node, ast.BoolOp):
            return self._logic(node, inside_neighbors)
        if isinstance(node, ast.Compare):
            return self._compare(node, inside_neighbors)
        if isinstance(node, ast.Call):
            return self._call(node, inside_neighbors)
        raise _refused(node)

    def _constant(self, node):
        value = node.value
        if isinstance(value, bool):
            return ("num", value), "bool"
        if isinstance(value, (int, float)):
            return ("num", value), "number"
        raise ExpressionError("BAD_EXPRESSION", f"Only numbers are allowed in a formula, not {value!r}.")

    def _name(self, name: str):
        # a variable first, then a parameter
        if name in self.variables:
            self.used["variables"].add(name)
            return ("var", name), "number"
        if name in self.parameters:
            self.used["parameters"].add(name)
            return ("param", name), "number"
        if name in ALL_FUNCTIONS:
            raise ExpressionError("BAD_EXPRESSION", f"'{name}' is a function: write {name}(...).")
        known = list(self.variables) + list(self.parameters)
        defined = ", ".join(known) if known else "none are defined"
        message = f"Name '{name}' is neither a variable nor a parameter (defined: {defined})."
        if name in self.states:
            message = f"State '{name}' can only be used inside count({name})."
        raise ExpressionError("UNKNOWN_VARIABLE", message)

    def _arithmetic(self, node, inside):
        if type(node.op) not in _ARITHMETIC:
            raise _refused(node.op)
        left, _ = self.read(node.left, inside)
        right, _ = self.read(node.right, inside)
        return ("arith", _ARITHMETIC[type(node.op)], left, right), "number"

    def _unary(self, node, inside):
        operand, kind = self.read(node.operand, inside)
        if isinstance(node.op, ast.Not):
            _need_bool(kind, "'not'")
            return ("not", operand), "bool"
        if isinstance(node.op, ast.USub):
            return ("neg", operand), "number"
        if isinstance(node.op, ast.UAdd):
            return ("pos", operand), "number"
        raise _refused(node.op)

    def _logic(self, node, inside):
        word = "and" if isinstance(node.op, ast.And) else "or"
        parts = []
        for value in node.values:
            part, kind = self.read(value, inside)
            _need_bool(kind, f"'{word}'")
            parts.append(part)
        return (word, parts), "bool"

    def _compare(self, node, inside):
        operators = []
        for operator in node.ops:
            if type(operator) not in _COMPARISONS:
                raise _refused(operator)
            operators.append(_COMPARISONS[type(operator)])
        operands = [self.read(node.left, inside)[0]]
        for value in node.comparators:
            operands.append(self.read(value, inside)[0])
        return ("compare", operators, operands), "bool"

    def _call(self, node, inside):
        if not isinstance(node.func, ast.Name) or node.func.id not in ALL_FUNCTIONS:
            name = node.func.id if isinstance(node.func, ast.Name) else ast.unparse(node.func)
            raise ExpressionError("BAD_EXPRESSION",
                                  f"Function '{name}' does not exist (available: {', '.join(ALL_FUNCTIONS)}).")
        name = node.func.id
        if node.keywords:
            raise ExpressionError("BAD_EXPRESSION", f"{name}() does not take named arguments.")
        wanted = 0 if name == "random" else 1
        if len(node.args) != wanted:
            raise ExpressionError("BAD_EXPRESSION",
                                  f"{name}() takes {wanted} argument{'s' if wanted != 1 else ''}, "
                                  f"not {len(node.args)}.")

        if name == "random":
            index = self.random_count
            self.random_count += 1
            if inside:
                raise ExpressionError("BAD_EXPRESSION", "random() cannot be used inside a neighbor function.")
            return ("random", index), "number"
        if name == "count":
            return self._count(node.args[0], inside)
        if name in CELL_FUNCTIONS:
            argument, _ = self.read(node.args[0], inside)
            return (name, argument), "number"
        return self._neighbor_function(name, node.args[0], inside)

    def _count(self, argument, inside):
        if inside:
            raise ExpressionError("BAD_EXPRESSION", "count() cannot be used inside a neighbor function.")
        if not isinstance(argument, ast.Name):
            raise ExpressionError("BAD_EXPRESSION", "count() takes a state name, for example count(FIRE).")
        if argument.id not in self.states:
            raise ExpressionError("UNKNOWN_STATE",
                                  f"State '{argument.id}' is not defined (defined: {', '.join(self.states)}).")
        self.used["states"].add(argument.id)
        return ("count", argument.id), "number"

    def _neighbor_function(self, name, argument, inside):
        if inside:
            raise ExpressionError("BAD_EXPRESSION",
                                  f"{name}() cannot be used inside another neighbor function.")
        inner, kind = self.read(argument, inside_neighbors=True)
        if name == "count_where":
            _need_bool(kind, "count_where()")
        return ("neighbors", name, inner), "number"


def uses_random(text: str) -> bool:
    """True if the formula calls random(). A formula that cannot be read gives False."""
    try:
        tree = ast.parse(text.strip(), mode="eval")
    except (SyntaxError, RecursionError, MemoryError, ValueError, AttributeError):
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "random":
            return True
    return False


def rule_formulas(rule: dict) -> list:
    """The formula texts of one rule (raw or loaded), in the order the engines compute them:
    the expr conditions as they appear in "when", then the set formulas."""
    texts = _condition_formulas(rule.get("when"))
    texts += list((rule.get("set") or {}).values())
    return texts


def _condition_formulas(condition) -> list:
    if not condition:
        return []
    if "expr" in condition:
        return [condition["expr"]]
    texts = []
    for part in condition.get("all", []) + condition.get("any", []):
        texts += _condition_formulas(part)
    return texts + _condition_formulas(condition.get("not"))


def _need_bool(kind: str, where: str) -> None:
    if kind != "bool":
        raise ExpressionError("EXPRESSION_TYPE", f"{where} needs a condition (true/false), not a number.")


def _refused(node) -> ExpressionError:
    what = _REFUSED.get(type(node), f"'{type(node).__name__}'")
    return ExpressionError("BAD_EXPRESSION", f"{what} is not allowed in a formula.")


# =====================================================================
# Computing a formula
# =====================================================================

def evaluate_expression(formula: dict, context: dict, mask=None):
    """The value of a formula for every cell, as a matrix of the grid's shape.

    context (built once per step by the engine):
        "self":        {variable: matrix}                 the values of each cell
        "neighbors":   [{"cells": matrix, variable: matrix}, ...]   one per direction,
                       already shifted, with the boundary applied
        "state_codes", "parameters", "rng", "shape"
    mask: the cells where the result is really used. A division by 0 is an error only there.

    Random numbers: one full matrix per random() of the formula, drawn first, in reading order,
    even if no cell uses it.
    """
    shape = tuple(context["shape"])
    draws = []
    for _ in range(formula["random_count"]):
        draws.append(context["rng"].random(shape))

    where = np.ones(shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    computer = _Computer(context, draws, formula["text"])
    value = computer.compute(formula["tree"], where, context["self"])

    result = np.broadcast_to(value, shape)
    if formula["type"] == "bool":
        return result.astype(bool)
    return result.copy()


class _Computer:
    """Computes the tuples made by _Reader. `values` are the current names: the cell's or a neighbor's."""

    def __init__(self, context, draws, text):
        self.context = context
        self.draws = draws
        self.text = text

    def compute(self, node, where, values):
        kind = node[0]
        if kind == "num":
            return node[1]
        if kind == "var":
            return values[node[1]]
        if kind == "param":
            return self.context["parameters"][node[1]]
        if kind == "arith":
            return self._arithmetic(node, where, values)
        if kind == "neg":
            return -_number(self.compute(node[1], where, values))
        if kind == "pos":
            return _number(self.compute(node[1], where, values))
        if kind == "not":
            return np.logical_not(self.compute(node[1], where, values))
        if kind in ("and", "or"):
            return self._logic(node, where, values)
        if kind == "compare":
            return self._compare(node, where, values)
        if kind == "count":
            return self._count(node[1])
        if kind == "neighbors":
            return self._neighbors(node[1], node[2], where)
        if kind == "random":
            return self.draws[node[1]]
        if kind == "abs":
            return np.abs(_number(self.compute(node[1], where, values)))
        # "floor"
        return np.floor(_number(self.compute(node[1], where, values)))

    def _arithmetic(self, node, where, values):
        _, operator, left_node, right_node = node
        left = _number(self.compute(left_node, where, values))
        right = _number(self.compute(right_node, where, values))
        if operator == "+":
            return left + right
        if operator == "-":
            return left - right
        if operator == "*":
            return left * right

        if np.any((np.asarray(right) == 0) & where):
            raise ExpressionRuntimeError(f"division by 0 in '{self.text}'")
        with np.errstate(divide="ignore", invalid="ignore"):
            if operator == "/":
                return np.true_divide(left, right)
            if operator == "//":
                return np.floor_divide(left, right)
            return np.mod(left, right)

    def _logic(self, node, where, values):
        # Like Python: a later part matters only where the earlier parts did not decide.
        # Every part is still computed on the whole grid, so random() always draws.
        word, parts = node
        result = np.asarray(self.compute(parts[0], where, values), dtype=bool)
        for part in parts[1:]:
            undecided = where & (result if word == "and" else ~result)
            value = np.asarray(self.compute(part, undecided, values), dtype=bool)
            result = (result & value) if word == "and" else (result | value)
        return result

    def _compare(self, node, where, values):
        # a < b < c means a < b and b < c
        _, operators, operands = node
        left = self.compute(operands[0], where, values)
        result = np.ones(np.shape(where), dtype=bool)
        for operator, right_node in zip(operators, operands[1:]):
            right = self.compute(right_node, where & result, values)
            result = result & _COMPARE_FUNCTIONS[operator](left, right)
            left = right
        return result

    def _count(self, state_name):
        code = self.context["state_codes"][state_name]
        total = np.zeros(tuple(self.context["shape"]), dtype=np.int64)
        for neighbor in self.context["neighbors"]:
            total += neighbor["cells"] == code
        return total

    def _neighbors(self, function, inner, where):
        # the argument is computed once for each neighbor, with the neighbor's values
        results = []
        for neighbor in self.context["neighbors"]:
            results.append(self.compute(inner, where, neighbor))

        if function == "count_where":
            total = np.zeros(tuple(self.context["shape"]), dtype=np.int64)
            for value in results:
                total += np.asarray(value, dtype=bool)
            return total
        if function == "min":
            return _reduce(np.minimum, results)
        if function == "max":
            return _reduce(np.maximum, results)

        total = _number(results[0])
        for value in results[1:]:
            total = total + _number(value)
        if function == "sum":
            return total
        return total / len(results)    # "avg"


_COMPARE_FUNCTIONS = {
    "==": np.equal, "!=": np.not_equal, "<": np.less,
    "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal,
}


def _number(value):
    """true/false becomes 1/0 in a calculation (NumPy would keep it true/false: True + True = True)."""
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, np.ndarray) and value.dtype == bool:
        return value.astype(np.int64)
    return value


def _reduce(function, values):
    result = _number(values[0])
    for value in values[1:]:
        result = function(result, _number(value))
    return result
