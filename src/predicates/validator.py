"""Parsing and validation for structured denial-constraint rules."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import json
import re
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence, Union


class RuleValidationError(ValueError):
    """Raised when a denial-constraint rule does not match the public grammar."""


@dataclass(frozen=True)
class Column:
    alias: str
    name: str


@dataclass(frozen=True)
class Constant:
    value: Union[str, Decimal]
    is_text: bool = False


@dataclass(frozen=True)
class Arithmetic:
    left: "Expression"
    operator: str
    right: "Expression"


@dataclass(frozen=True)
class Group:
    expression: "Expression"


Expression = Union[Column, Constant, Arithmetic, Group]


@dataclass(frozen=True)
class PredicateNode:
    left: Expression
    operator: str
    right: Optional[Union[Expression, tuple[Constant, ...]]] = None


@dataclass(frozen=True)
class LogicalNode:
    operator: str
    items: tuple["ParsedRule", ...]


ParsedRule = Union[PredicateNode, LogicalNode]
Rule = Union[str, Mapping[str, Any]]
Rules = Optional[Sequence[Rule]]


@dataclass(frozen=True)
class _Token:
    kind: str
    value: str
    position: int


_COLUMN_RE = re.compile(r"t\d+\.[A-Za-z_][A-Za-z0-9_]*")
_NUMBER_RE = re.compile(r"(?:\d+(?:\.\d*)?|\.\d+)")
_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_ARITHMETIC_PRECEDENCE = {"+": 1, "-": 1, "*": 2, "/": 2, "%": 2, "^": 3}
_COMPARISON_OPERATORS = {"=", "!=", "<", "<=", ">", ">="}


def _tokenize(text: str) -> list[_Token]:
    tokens: list[_Token] = []
    position = 0
    while position < len(text):
        char = text[position]
        if char.isspace():
            position += 1
            continue

        if char == "'":
            start = position
            position += 1
            value: list[str] = []
            while position < len(text):
                if text[position] == "'":
                    if position + 1 < len(text) and text[position + 1] == "'":
                        value.append("'")
                        position += 2
                        continue
                    position += 1
                    break
                value.append(text[position])
                position += 1
            else:
                raise RuleValidationError(
                    f"Unterminated text constant at position {start}"
                )
            tokens.append(_Token("STRING", "".join(value), start))
            continue

        column_match = _COLUMN_RE.match(text, position)
        if column_match:
            tokens.append(_Token("COLUMN", column_match.group(0), position))
            position = column_match.end()
            continue

        number_match = _NUMBER_RE.match(text, position)
        if number_match:
            tokens.append(_Token("NUMBER", number_match.group(0), position))
            position = number_match.end()
            continue

        if text.startswith(("!=", "<=", ">="), position):
            tokens.append(
                _Token("PREDICATE_OP", text[position : position + 2], position)
            )
            position += 2
            continue
        if char in "=<>":
            tokens.append(_Token("PREDICATE_OP", char, position))
            position += 1
            continue
        if char in "+-*/%^":
            tokens.append(_Token("ARITHMETIC_OP", char, position))
            position += 1
            continue
        if char == "(":
            tokens.append(_Token("LPAREN", char, position))
            position += 1
            continue
        if char == ")":
            tokens.append(_Token("RPAREN", char, position))
            position += 1
            continue
        if char == ",":
            tokens.append(_Token("COMMA", char, position))
            position += 1
            continue

        word_match = _WORD_RE.match(text, position)
        if word_match:
            tokens.append(_Token("WORD", word_match.group(0).upper(), position))
            position = word_match.end()
            continue

        raise RuleValidationError(
            f"Unexpected character {char!r} at position {position}"
        )

    tokens.append(_Token("EOF", "", len(text)))
    return tokens


class _PredicateParser:
    def __init__(self, text: str, columns: Optional[set[str]]) -> None:
        self.text = text
        self.tokens = _tokenize(text)
        self.index = 0
        self.columns = columns

    @property
    def current(self) -> _Token:
        return self.tokens[self.index]

    def advance(self) -> _Token:
        token = self.current
        self.index += 1
        return token

    def parse(self) -> PredicateNode:
        if not self.text.strip():
            raise RuleValidationError("Predicate cannot be empty")

        left = self.parse_expression()
        operator = self.parse_predicate_operator()

        if operator in {"IS NULL", "IS NOT NULL"}:
            right: Optional[Union[Expression, tuple[Constant, ...]]] = None
        elif operator in {"IN", "NOT IN"}:
            right = self.parse_constant_list()
        elif operator in {"LIKE", "NOT LIKE"}:
            right = self.parse_pattern()
        else:
            right = self.parse_expression()

        if self.current.kind != "EOF":
            if self._is_predicate_operator_start():
                raise RuleValidationError(
                    "Predicate must contain exactly one predicate operator"
                )
            if self.current.kind in {"LPAREN", "RPAREN"}:
                raise RuleValidationError("Unbalanced parentheses in predicate")
            raise RuleValidationError(
                f"Unexpected token {self.current.value!r} at position {self.current.position}"
            )

        return PredicateNode(left=left, operator=operator, right=right)

    def parse_predicate_operator(self) -> str:
        token = self.current
        if token.kind == "PREDICATE_OP" and token.value in _COMPARISON_OPERATORS:
            self.advance()
            return token.value
        if token.kind == "WORD" and token.value in {"IN", "LIKE"}:
            self.advance()
            return token.value
        if token.kind == "WORD" and token.value == "NOT":
            self.advance()
            if self.current.kind == "WORD" and self.current.value in {"IN", "LIKE"}:
                return f"NOT {self.advance().value}"
        if token.kind == "WORD" and token.value == "IS":
            self.advance()
            if self.current.kind == "WORD" and self.current.value == "NOT":
                self.advance()
                if self.current.kind == "WORD" and self.current.value == "NULL":
                    self.advance()
                    return "IS NOT NULL"
            elif self.current.kind == "WORD" and self.current.value == "NULL":
                self.advance()
                return "IS NULL"
        raise RuleValidationError(
            "Predicate must contain exactly one predicate operator"
        )

    def parse_constant_list(self) -> tuple[Constant, ...]:
        if self.current.kind != "LPAREN":
            raise RuleValidationError(
                "IN and NOT IN require a parenthesized constant list"
            )
        self.advance()
        values = [self.parse_constant()]
        while self.current.kind == "COMMA":
            self.advance()
            values.append(self.parse_constant())
        if self.current.kind != "RPAREN":
            raise RuleValidationError("Unbalanced parentheses in constant list")
        self.advance()
        return tuple(values)

    def parse_pattern(self) -> Constant:
        """LIKE / NOT LIKE take a single SQL pattern string (paper §2, pattern predicates)."""
        if self.current.kind != "STRING":
            raise RuleValidationError("LIKE and NOT LIKE require a quoted pattern string")
        return Constant(self.advance().value, is_text=True)

    def parse_constant(self) -> Constant:
        negative = False
        if self.current.kind == "ARITHMETIC_OP" and self.current.value == "-":
            negative = True
            self.advance()
        token = self.current
        if token.kind == "STRING" and not negative:
            self.advance()
            return Constant(token.value, is_text=True)
        if token.kind == "NUMBER":
            self.advance()
            value = _decimal(token.value)
            return Constant(-value if negative else value)
        raise RuleValidationError("IN and NOT IN lists may contain only constants")

    def parse_expression(self, minimum_precedence: int = 0) -> Expression:
        left = self.parse_term()
        while self.current.kind == "ARITHMETIC_OP":
            operator = self.current.value
            precedence = _ARITHMETIC_PRECEDENCE[operator]
            if precedence < minimum_precedence:
                break
            self.advance()
            next_precedence = precedence if operator == "^" else precedence + 1
            right = self.parse_expression(next_precedence)
            if _contains_text(left) or _contains_text(right):
                raise RuleValidationError(
                    "Text constants cannot be used in arithmetic expressions"
                )
            left = Arithmetic(left, operator, right)
        return left

    def parse_term(self) -> Expression:
        token = self.current
        if token.kind == "COLUMN":
            self.advance()
            alias, name = token.value.split(".", 1)
            if self.columns is not None and name not in self.columns:
                raise RuleValidationError(
                    f"Column does not exist in dataset: {token.value}"
                )
            return Column(alias, name)
        if token.kind == "STRING":
            self.advance()
            return Constant(token.value, is_text=True)
        if token.kind == "NUMBER":
            self.advance()
            return Constant(_decimal(token.value))
        if token.kind == "ARITHMETIC_OP" and token.value == "-":
            self.advance()
            if self.current.kind != "NUMBER":
                raise RuleValidationError(
                    "Unary minus is allowed only for numeric constants"
                )
            number = self.advance()
            return Constant(-_decimal(number.value))
        if token.kind == "LPAREN":
            self.advance()
            expression = self.parse_expression()
            if self.current.kind != "RPAREN":
                raise RuleValidationError("Unbalanced parentheses in predicate")
            self.advance()
            return Group(expression)
        if token.kind == "WORD":
            raise RuleValidationError(
                f"Column must use TABELA.NOME_COLUNA format near {token.value!r}"
            )
        if token.kind == "EOF":
            raise RuleValidationError("Missing predicate operand")
        if token.kind == "RPAREN":
            raise RuleValidationError("Unbalanced parentheses in predicate")
        raise RuleValidationError(
            f"Expected column, constant, or expression at position {token.position}"
        )

    def _is_predicate_operator_start(self) -> bool:
        return self.current.kind == "PREDICATE_OP" or (
            self.current.kind == "WORD" and self.current.value in {"IN", "NOT", "IS", "LIKE"}
        )


def _decimal(value: str) -> Decimal:
    try:
        return Decimal(value)
    except InvalidOperation as exc:
        raise RuleValidationError(f"Invalid numeric constant: {value}") from exc


def _contains_text(expression: Expression) -> bool:
    if isinstance(expression, Constant):
        return expression.is_text
    if isinstance(expression, Arithmetic):
        return _contains_text(expression.left) or _contains_text(expression.right)
    if isinstance(expression, Group):
        return _contains_text(expression.expression)
    return False


def _schema_columns(schema: Any) -> Optional[set[str]]:
    if schema is None:
        return None
    if isinstance(schema, Mapping):
        tables = schema.get("tables")
        if isinstance(tables, Sequence):
            return {
                column["name"]
                for table in tables
                if isinstance(table, Mapping)
                for column in table.get("columns", [])
                if isinstance(column, Mapping) and isinstance(column.get("name"), str)
            }
        return {str(column) for column in schema}
    return {str(column) for column in schema}


def parse_predicate(predicate: str, schema: Any = None) -> PredicateNode:
    """Parse and validate one predicate string."""
    if not isinstance(predicate, str):
        raise RuleValidationError("Predicate must be a string")
    return _PredicateParser(predicate, _schema_columns(schema)).parse()


def parse_rule(rule: Rule, schema: Any = None) -> ParsedRule:
    """Parse one predicate or recursive AND/OR node."""
    if isinstance(rule, str):
        return parse_predicate(rule, schema)
    if not isinstance(rule, Mapping):
        raise RuleValidationError("Rule must be a predicate string or logical node")
    if len(rule) != 1:
        raise RuleValidationError("Logical node must contain exactly one key")
    operator, raw_items = next(iter(rule.items()))
    if operator not in {"AND", "OR"}:
        raise RuleValidationError("Logical node key must be AND or OR")
    if not isinstance(raw_items, list) or len(raw_items) < 2:
        raise RuleValidationError(
            f"Logical node {operator} must contain at least 2 items"
        )
    return LogicalNode(operator, tuple(parse_rule(item, schema) for item in raw_items))


def parse_rules(rules: Rules, schema: Any = None) -> Optional[tuple[ParsedRule, ...]]:
    """Parse the public sentence output: a list of independent rules or null."""
    if rules is None:
        return None
    if not isinstance(rules, list):
        raise RuleValidationError("Rules must be a list of independent rules or null")
    if not rules:
        raise RuleValidationError(
            "Rules list must contain at least one rule or be null"
        )
    return tuple(parse_rule(rule, schema) for rule in rules)


def normalize_rules(rules: Any, schema: Any = None) -> Optional[list[Rule]]:
    """Validate rules and convert legacy ``[[predicate, ...]]`` rules at the boundary."""
    if rules is None:
        return None
    if not isinstance(rules, list):
        raise RuleValidationError("Rules must be a list of independent rules or null")
    if not rules:
        raise RuleValidationError(
            "Rules list must contain at least one rule or be null"
        )

    normalized: list[Rule] = []
    for rule in rules:
        if isinstance(rule, list):
            if not rule or not all(isinstance(predicate, str) for predicate in rule):
                raise RuleValidationError(
                    "Legacy rules must be non-empty lists of predicate strings"
                )
            normalized.append(rule[0] if len(rule) == 1 else {"AND": list(rule)})
        else:
            normalized.append(rule)

    parsed = parse_rules(normalized, schema)
    assert parsed is not None
    return [rule_to_data(rule) for rule in parsed]


def predicate_strings(rules: Rules, schema: Any = None) -> list[str]:
    """Return canonical predicate leaves from an independent rule list."""
    parsed = parse_rules(normalize_rules(rules, schema), schema)
    if parsed is None:
        return []
    return [
        _canonical_predicate(predicate)
        for rule in parsed
        for predicate in iter_predicates(rule)
    ]


def transform_predicates(
    rules: Rules, transform: Callable[[str], str], schema: Any = None
) -> Optional[list[Rule]]:
    """Apply a string transformation to every predicate leaf, preserving logic."""
    normalized = normalize_rules(rules, schema)
    if normalized is None:
        return None

    def transform_rule(rule: Rule) -> Rule:
        if isinstance(rule, str):
            return transform(rule)
        operator, items = next(iter(rule.items()))
        return {operator: [transform_rule(item) for item in items]}

    return normalize_rules([transform_rule(rule) for rule in normalized], schema)


def rule_to_data(rule: ParsedRule) -> Rule:
    """Serialize a parsed rule to the public JSON-compatible representation."""
    if isinstance(rule, PredicateNode):
        return _canonical_predicate(rule)
    return {rule.operator: [rule_to_data(item) for item in rule.items]}


def iter_predicates(rule: ParsedRule) -> Iterable[PredicateNode]:
    """Yield predicate leaves in depth-first order."""
    if isinstance(rule, PredicateNode):
        yield rule
        return
    for item in rule.items:
        yield from iter_predicates(item)


def collect_aliases(rule: ParsedRule) -> set[str]:
    """Return all tuple aliases referenced by a parsed rule."""
    aliases: set[str] = set()
    for predicate in iter_predicates(rule):
        for expression in (predicate.left, predicate.right):
            if isinstance(expression, tuple) or expression is None:
                continue
            aliases.update(_expression_aliases(expression))
    return aliases


def _expression_aliases(expression: Expression) -> set[str]:
    if isinstance(expression, Column):
        return {expression.alias}
    if isinstance(expression, Arithmetic):
        return _expression_aliases(expression.left) | _expression_aliases(
            expression.right
        )
    if isinstance(expression, Group):
        return _expression_aliases(expression.expression)
    return set()


def canonical_rule(rule: Union[Rule, ParsedRule], schema: Any = None) -> str:
    """Return a stable representation, ignoring whitespace and logical item order."""
    parsed = (
        rule
        if isinstance(rule, (PredicateNode, LogicalNode))
        else parse_rule(rule, schema)
    )
    if isinstance(parsed, PredicateNode):
        return _canonical_predicate(parsed)
    children = sorted(canonical_rule(item) for item in parsed.items)
    return json.dumps(
        {parsed.operator: children}, ensure_ascii=True, separators=(",", ":")
    )


def canonical_rules(rules: Rules, schema: Any = None) -> Optional[tuple[str, ...]]:
    """Canonicalize an unordered list of independent rules."""
    parsed = parse_rules(rules, schema)
    if parsed is None:
        return None
    return tuple(sorted(canonical_rule(rule) for rule in parsed))


def _canonical_predicate(predicate: PredicateNode) -> str:
    left = _canonical_expression(predicate.left)
    if predicate.operator in {"IS NULL", "IS NOT NULL"}:
        return f"{left} {predicate.operator}"
    if isinstance(predicate.right, tuple):
        values = ",".join(_canonical_constant(value) for value in predicate.right)
        return f"{left} {predicate.operator} ({values})"
    assert predicate.right is not None
    return f"{left} {predicate.operator} {_canonical_expression(predicate.right)}"


def _canonical_expression(expression: Expression) -> str:
    if isinstance(expression, Column):
        return f"{expression.alias}.{expression.name}"
    if isinstance(expression, Constant):
        return _canonical_constant(expression)
    if isinstance(expression, Group):
        return f"({_canonical_expression(expression.expression)})"
    return (
        f"{_canonical_expression(expression.left)} {expression.operator} "
        f"{_canonical_expression(expression.right)}"
    )


def _canonical_constant(constant: Constant) -> str:
    if constant.is_text:
        return "'" + str(constant.value).replace("'", "''") + "'"
    value = constant.value
    assert isinstance(value, Decimal)
    normalized = format(value.normalize(), "f")
    return "0" if normalized in {"-0", ""} else normalized
