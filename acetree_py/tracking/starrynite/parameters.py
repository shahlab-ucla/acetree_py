"""Safe, lossless parsing for legacy StarryNite parameter files.

This module intentionally implements a small data grammar rather than running
MATLAB syntax.  Recognized assignments become typed Python values; every other
statement remains available as an opaque record and the original source can be
rendered byte-for-byte (after text decoding) when no changes are requested.
"""

from __future__ import annotations

import math
import os
import re
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence, TypeAlias


ScalarValue: TypeAlias = bool | int | float | str
ParameterValue: TypeAlias = ScalarValue | tuple["ParameterValue", ...]

_NAME_RE = re.compile(r"^[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*$", re.ASCII)
_ASSIGNMENT_RE = re.compile(
    r"^(?P<name>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*=\s*(?P<value>.+)$",
    re.ASCII | re.DOTALL,
)
_CLASSIC_RE = re.compile(
    r"^(?P<name>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)[ \t]+(?P<value>\S(?:.*\S)?)$",
    re.ASCII | re.DOTALL,
)
_MATLAB_KEYWORDS = frozenset(
    {
        "break",
        "case",
        "catch",
        "classdef",
        "continue",
        "else",
        "elseif",
        "end",
        "for",
        "function",
        "global",
        "if",
        "otherwise",
        "parfor",
        "persistent",
        "return",
        "spmd",
        "switch",
        "try",
        "while",
    }
)
_UNQUOTED_STRING_RE = re.compile(r"^[^\s;=()\[\]{}]+$", re.ASCII)
_MODEL_NAME_MARKERS = ("model", "classifier", "training", "weights")


class ParameterParseError(ValueError):
    """Raised when a standalone value is outside the supported safe grammar."""


@dataclass(frozen=True, slots=True)
class ParameterRecord:
    """One source fragment and, when recognized, its semantic interpretation."""

    kind: str
    source: str
    name: str | None = None
    value: ParameterValue | None = None
    model_path: str | None = None
    syntax: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"assignment", "load", "opaque", "comment", "trivia"}:
            raise ValueError(f"Unsupported parameter record kind: {self.kind!r}")


@dataclass(frozen=True, slots=True)
class ModelReference:
    """A model path found in a ``load`` statement or model-like setting."""

    raw_path: str
    record_index: int
    parameter_name: str | None = None
    source_kind: str = "load"

    def resolve(
        self,
        base_directory: str | Path,
        *,
        must_exist: bool = False,
    ) -> Path:
        """Resolve this reference without expanding shell or environment syntax."""

        candidate = Path(self.raw_path)
        if not candidate.is_absolute():
            candidate = Path(base_directory) / candidate
        candidate = candidate.resolve(strict=False)
        if must_exist and not candidate.is_file():
            raise FileNotFoundError(
                f"StarryNite model referenced by record {self.record_index} was not "
                f"found: {candidate}"
            )
        return candidate


@dataclass(frozen=True, slots=True)
class StarryNiteParameters:
    """Parsed StarryNite parameters with lossless source preservation."""

    source: str
    records: tuple[ParameterRecord, ...]
    source_path: Path | None = None
    syntax: str = "mixed"
    settings: Mapping[str, ParameterValue] = field(init=False, repr=False)
    normalized_settings: Mapping[str, ParameterValue] = field(init=False, repr=False)
    model_references: tuple[ModelReference, ...] = field(init=False)

    def __post_init__(self) -> None:
        effective: dict[str, ParameterValue] = {}
        normalized: dict[str, ParameterValue] = {}
        references: list[ModelReference] = []
        last_assignment_index: dict[str, int] = {}
        seen_references: set[tuple[str, str | None, str]] = set()

        for index, record in enumerate(self.records):
            if record.kind == "assignment" and record.name is not None:
                effective[record.name] = record.value  # type: ignore[assignment]
                normalized[normalize_parameter_name(record.name)] = (
                    record.value  # type: ignore[assignment]
                )
                last_assignment_index[record.name] = index
            elif record.kind == "load" and record.model_path is not None:
                key = (record.model_path, None, "load")
                if key not in seen_references:
                    references.append(
                        ModelReference(
                            raw_path=record.model_path,
                            record_index=index,
                        )
                    )
                    seen_references.add(key)

        # A model-like setting follows the same last-assignment-wins rule as all
        # other settings. Explicit ``load`` statements remain cumulative.
        for name, value in effective.items():
            if isinstance(value, str) and _looks_like_model_setting(name, value):
                key = (value, name, "setting")
                if key not in seen_references:
                    references.append(
                        ModelReference(
                            raw_path=value,
                            record_index=last_assignment_index[name],
                            parameter_name=name,
                            source_kind="setting",
                        )
                    )
                    seen_references.add(key)

        object.__setattr__(self, "settings", MappingProxyType(effective))
        object.__setattr__(self, "normalized_settings", MappingProxyType(normalized))
        object.__setattr__(self, "model_references", tuple(references))
        if self.source_path is not None:
            object.__setattr__(self, "source_path", Path(self.source_path))

    @property
    def opaque_records(self) -> tuple[ParameterRecord, ...]:
        """Unsupported statements retained without execution or interpretation."""

        return tuple(record for record in self.records if record.kind == "opaque")

    def resolve_model_references(
        self,
        base_directory: str | Path | None = None,
        *,
        must_exist: bool = False,
    ) -> tuple[Path, ...]:
        """Resolve every discovered model reference relative to the parameter file."""

        if base_directory is None:
            base = self.source_path.parent if self.source_path is not None else Path.cwd()
        else:
            base = Path(base_directory)
        return tuple(
            reference.resolve(base, must_exist=must_exist)
            for reference in self.model_references
        )

    def render(
        self,
        overrides: Mapping[str, ParameterValue] | None = None,
        *,
        append: Mapping[str, ParameterValue] | None = None,
        syntax: str | None = None,
    ) -> str:
        """Render source exactly, optionally appending effective assignments.

        Overrides are deliberately appended instead of rewriting historical
        statements.  MATLAB and classic StarryNite files both use last-assignment
        wins semantics, so this preserves every original comment and unknown line.
        """

        overrides = dict(overrides or {})
        append = dict(append or {})
        overlap = set(overrides) & set(append)
        if overlap:
            names = ", ".join(sorted(overlap))
            raise ValueError(f"Settings cannot be both overridden and appended: {names}")
        if not overrides and not append:
            return self.source

        selected_syntax = syntax or ("classic" if self.syntax == "classic" else "matlab")
        if selected_syntax not in {"classic", "matlab"}:
            raise ValueError("Render syntax must be 'classic' or 'matlab'")
        additions = {**overrides, **append}
        rendered = self.source
        newline = "\r\n" if "\r\n" in self.source else "\n"
        if rendered and not rendered.endswith(("\n", "\r")):
            rendered += newline
        rendered += "".join(
            _format_assignment(name, value, selected_syntax, newline=newline)
            for name, value in additions.items()
        )
        return rendered


def parse_parameter_text(
    source: str,
    *,
    source_path: str | Path | None = None,
) -> StarryNiteParameters:
    """Parse MATLAB-command or whitespace key-value StarryNite parameters."""

    if not isinstance(source, str):
        raise TypeError("Parameter source must be text")
    records: list[ParameterRecord] = []
    syntaxes: list[str] = []
    # Parameter files are evaluated as scripts, and some distributed examples
    # reuse an earlier inert scalar as a later value (for example,
    # ``distribution_file2=distribution_file``).  Keeping only successfully
    # parsed direct workspace variables reproduces that source-order behavior
    # without evaluating MATLAB or resolving dotted workspace/object access.
    constants: dict[str, ParameterValue] = {}
    for chunk in _scan_source(source):
        if chunk.category in {"comment", "trivia"}:
            records.append(ParameterRecord(chunk.category, chunk.source))
            continue
        matlab_target = _direct_matlab_assignment_target(chunk.source)
        record = _parse_statement(chunk.source, constants=constants)
        records.append(record)
        if record.syntax is not None:
            syntaxes.append(record.syntax)
        if matlab_target is not None:
            # A failed/dynamic reassignment must invalidate an older inert
            # value. Otherwise ``x=1; x=rand(); y=x`` would be misread as
            # ``y=1`` instead of failing closed at the dynamic boundary.
            constants.pop(matlab_target, None)
            if (
                record.kind == "assignment"
                and record.name == matlab_target
                and record.value is not None
            ):
                constants[matlab_target] = record.value

    if syntaxes and all(item == "classic" for item in syntaxes):
        syntax = "classic"
    elif syntaxes and all(item == "matlab" for item in syntaxes):
        syntax = "matlab"
    else:
        syntax = "mixed"
    return StarryNiteParameters(
        source=source,
        records=tuple(records),
        source_path=None if source_path is None else Path(source_path),
        syntax=syntax,
    )


def read_parameter_file(path: str | Path, *, encoding: str = "utf-8-sig") -> StarryNiteParameters:
    """Read and parse a StarryNite parameter file without executing it."""

    source_path = Path(path)
    try:
        with source_path.open("r", encoding=encoding, newline="") as stream:
            source = stream.read()
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"Could not decode StarryNite parameter file {source_path} as {encoding}; "
            "pass its actual text encoding explicitly"
        ) from exc
    return parse_parameter_text(source, source_path=source_path)


def write_parameter_file(
    parameters: StarryNiteParameters,
    path: str | Path,
    *,
    overrides: Mapping[str, ParameterValue] | None = None,
    append: Mapping[str, ParameterValue] | None = None,
    syntax: str | None = None,
    encoding: str = "utf-8",
) -> Path:
    """Atomically save a lossless parameter source with optional overrides."""

    if not isinstance(parameters, StarryNiteParameters):
        raise TypeError("parameters must be StarryNiteParameters")
    destination = Path(path).expanduser()
    if not destination.name:
        raise ValueError("Parameter destination must name a file")
    if not destination.parent.is_dir():
        raise FileNotFoundError(
            f"Parameter destination directory does not exist: {destination.parent}"
        )
    rendered = parameters.render(
        overrides,
        append=append,
        syntax=syntax,
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding=encoding,
            newline="",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
            temporary_path = Path(stream.name)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination.resolve(strict=False)


def parse_parameter_value(source: str) -> ParameterValue:
    """Parse one value using the deliberately small, non-executable grammar."""

    parser = _ValueParser(source)
    value = parser.parse()
    return _freeze_value(value)


def parse_parameter_expression(
    source: str,
    *,
    constants: Mapping[str, ParameterValue],
) -> ParameterValue:
    """Parse safe arithmetic with an explicit, inert constant environment.

    Legacy regional boxes commonly use expressions such as
    ``450 * downsample``.  Supporting those expressions does not require (and
    must never grow into) MATLAB workspace evaluation: names are resolved only
    from the caller-provided mapping and function calls, indexing, dotted
    access, and assignment remain outside the grammar.
    """

    normalized: dict[str, ParameterValue] = {}
    for name, value in constants.items():
        if not re.fullmatch(r"[A-Za-z_]\w*", name, re.ASCII):
            raise ValueError(f"Invalid parameter-expression constant: {name!r}")
        frozen = _freeze_value(value)
        if isinstance(frozen, str):
            raise ValueError(
                f"Parameter-expression constant {name!r} must be numeric or boolean"
            )
        normalized[name] = frozen
    parser = _ValueParser(source, constants=normalized)
    value = parser.parse()
    return _freeze_value(value)


def normalize_parameter_name(name: str) -> str:
    """Normalize a dotted MATLAB name to stable lowercase snake case segments."""

    if not _NAME_RE.fullmatch(name):
        raise ValueError(f"Invalid dotted parameter name: {name!r}")
    normalized_segments = []
    for segment in name.split("."):
        segment = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", segment)
        segment = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", segment)
        normalized_segments.append(segment.lower())
    return ".".join(normalized_segments)


# Compatibility-friendly aliases for callers that use load/parse terminology.
parse_parameters = parse_parameter_text
load_parameter_file = read_parameter_file


@dataclass(frozen=True, slots=True)
class _SourceChunk:
    source: str
    category: str


def _scan_source(source: str) -> tuple[_SourceChunk, ...]:
    """Split source at top-level newlines/semicolons while retaining every byte."""

    chunks: list[_SourceChunk] = []
    start = 0
    index = 0
    bracket_depth = 0
    parenthesis_depth = 0
    quote: str | None = None
    length = len(source)

    def emit(end: int, category: str | None = None) -> None:
        nonlocal start
        if end <= start:
            return
        text = source[start:end]
        if category is None:
            category = "statement" if _statement_code(text) else "trivia"
        chunks.append(_SourceChunk(text, category))
        start = end

    while index < length:
        char = source[index]
        if quote is not None:
            if char == quote:
                # MATLAB escapes a quote by doubling it.
                if index + 1 < length and source[index + 1] == quote:
                    index += 2
                    continue
                quote = None
            index += 1
            continue

        if char in {"'", '"'}:
            quote = char
            index += 1
            continue
        if char == "[":
            bracket_depth += 1
        elif char == "]" and bracket_depth:
            bracket_depth -= 1
        elif char == "(":
            parenthesis_depth += 1
        elif char == ")" and parenthesis_depth:
            parenthesis_depth -= 1
        elif char in {"%", "#"}:
            emit(index)
            end = source.find("\n", index)
            end = length if end < 0 else end + 1
            start = index
            emit(end, "comment")
            index = end
            continue
        elif char == ";" and bracket_depth == 0 and parenthesis_depth == 0:
            emit(index + 1)
        elif char == "\n" and bracket_depth == 0 and parenthesis_depth == 0:
            emit(index + 1)
        index += 1
    emit(length)
    return tuple(chunks)


def _statement_code(source: str) -> str:
    code = source.strip()
    if code.endswith(";"):
        code = code[:-1].rstrip()
    return code


def _parse_statement(
    source: str,
    *,
    constants: Mapping[str, ParameterValue] | None = None,
) -> ParameterRecord:
    code = _statement_code(source)
    if not code:
        return ParameterRecord("trivia", source)

    load_path = _parse_load_statement(code)
    if load_path is not None:
        return ParameterRecord(
            "load",
            source,
            model_path=load_path,
            syntax="matlab",
        )

    assignment = _ASSIGNMENT_RE.fullmatch(code)
    syntax = "matlab"
    if assignment is None:
        assignment = _CLASSIC_RE.fullmatch(code)
        syntax = "classic"
    if assignment is None:
        return ParameterRecord("opaque", source, reason="unrecognized statement")

    name = assignment.group("name")
    raw_value = assignment.group("value").strip()
    if syntax == "classic" and name.lower() in _MATLAB_KEYWORDS:
        return ParameterRecord("opaque", source, reason="MATLAB control statement")
    try:
        value = _parse_parameter_value_with_constants(raw_value, constants or {})
    except ParameterParseError as exc:
        if syntax == "classic" and _UNQUOTED_STRING_RE.fullmatch(raw_value):
            value = raw_value
        else:
            return ParameterRecord("opaque", source, reason=str(exc))
    return ParameterRecord(
        "assignment",
        source,
        name=name,
        value=value,
        syntax=syntax,
    )


def _direct_matlab_assignment_target(source: str) -> str | None:
    """Return a direct script-workspace target, excluding dotted fields."""

    assignment = _ASSIGNMENT_RE.fullmatch(_statement_code(source))
    if assignment is None:
        return None
    name = assignment.group("name")
    return name if "." not in name else None


def _parse_parameter_value_with_constants(
    source: str,
    constants: Mapping[str, ParameterValue],
) -> ParameterValue:
    """Parse a value using only previously proven inert workspace values."""

    parser = _ValueParser(source, constants=constants)
    return _freeze_value(parser.parse())


def _parse_load_statement(code: str) -> str | None:
    match = re.fullmatch(r"load\s*\(\s*(.+?)\s*\)", code, re.IGNORECASE | re.DOTALL)
    if match is None:
        match = re.fullmatch(r"load[ \t]+(.+)", code, re.IGNORECASE | re.DOTALL)
    if match is None:
        return None
    argument = match.group(1).strip()
    try:
        value = parse_parameter_value(argument)
    except ParameterParseError:
        if _UNQUOTED_STRING_RE.fullmatch(argument):
            value = argument
        else:
            return None
    return value if isinstance(value, str) and value else None


def _looks_like_model_setting(name: str, value: str) -> bool:
    normalized = normalize_parameter_name(name).replace(".", "_")
    lower_value = value.lower()
    return lower_value.endswith(".mat") or any(
        marker in normalized for marker in _MODEL_NAME_MARKERS
    )


def _format_assignment(
    name: str,
    value: ParameterValue,
    syntax: str,
    *,
    newline: str,
) -> str:
    if not _NAME_RE.fullmatch(name):
        raise ValueError(f"Invalid dotted parameter name: {name!r}")
    rendered = _format_value(_freeze_value(value))
    if syntax == "classic":
        return f"{name} {rendered}{newline}"
    return f"{name} = {rendered};{newline}"


def _format_value(value: ParameterValue) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return "'" + value.replace("'", "''") + "'"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if math.isinf(value):
            return "Inf" if value > 0 else "-Inf"
        return repr(value)
    if isinstance(value, tuple):
        if value and all(isinstance(row, tuple) for row in value):
            rows = tuple(row for row in value if isinstance(row, tuple))
            widths = {len(row) for row in rows}
            if (
                len(widths) == 1
                and next(iter(widths)) > 0
                and all(
                    not isinstance(item, tuple)
                    for row in rows
                    for item in row
                )
            ):
                return "[" + "; ".join(
                    ", ".join(_format_value(item) for item in row) for row in rows
                ) + "]"
        return "[" + ", ".join(_format_value(item) for item in value) + "]"
    raise TypeError(f"Unsupported parameter value: {type(value).__name__}")


def _freeze_value(value: Any) -> ParameterValue:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_value(item) for item in value)
    raise TypeError(
        "Parameter values must be bools, finite numeric scalars, strings, or vectors"
    )


@dataclass(frozen=True, slots=True)
class _Token:
    kind: str
    text: str
    position: int


class _ValueParser:
    """Recursive-descent parser for literals and simple numeric arithmetic."""

    _NUMBER_RE = re.compile(
        r"(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?",
        re.ASCII,
    )
    _OPERATORS = (".*", "./", ".^", "+", "-", "*", "/", "^")

    def __init__(
        self,
        source: str,
        *,
        constants: Mapping[str, ParameterValue] | None = None,
    ) -> None:
        self.source = source
        self.tokens = self._tokenize(source)
        self.index = 0
        self.constants = dict(constants or {})

    def parse(self) -> Any:
        if not self.tokens:
            raise ParameterParseError("empty value")
        value = self._parse_additive()
        if self._peek() is not None:
            token = self._peek()
            raise ParameterParseError(
                f"unsupported token {token.text!r} at column {token.position + 1}"
            )
        return value

    def _parse_additive(self) -> Any:
        value = self._parse_multiplicative()
        while self._peek_text() in {"+", "-"}:
            operator = self._take().text
            value = _apply_numeric_operator(value, self._parse_multiplicative(), operator)
        return value

    def _parse_multiplicative(self) -> Any:
        value = self._parse_power()
        while self._peek_text() in {"*", "/", ".*", "./"}:
            operator = self._take().text
            value = _apply_numeric_operator(value, self._parse_power(), operator)
        return value

    def _parse_power(self) -> Any:
        value = self._parse_unary()
        if self._peek_text() in {"^", ".^"}:
            operator = self._take().text
            value = _apply_numeric_operator(value, self._parse_power(), operator)
        return value

    def _parse_unary(self) -> Any:
        if self._peek_text() in {"+", "-"}:
            operator = self._take().text
            value = self._parse_unary()
            if not _is_numeric_value(value):
                raise ParameterParseError("unary signs require numeric values")
            return value if operator == "+" else _map_numeric(value, lambda item: -item)
        return self._parse_primary()

    def _parse_primary(self) -> Any:
        token = self._take()
        if token.kind == "number":
            return _number_value(token.text)
        if token.kind == "string":
            return _unquote_matlab(token.text)
        if token.kind == "identifier":
            lower = token.text.lower()
            if lower == "true":
                return True
            if lower == "false":
                return False
            if lower in {"inf", "infinity"}:
                return math.inf
            if lower == "nan":
                return math.nan
            if token.text in self.constants:
                return self.constants[token.text]
            raise ParameterParseError(f"unsupported identifier {token.text!r}")
        if token.text == "(":
            value = self._parse_additive()
            self._expect(")")
            return value
        if token.text == "[":
            return self._parse_vector()
        raise ParameterParseError(f"unexpected token {token.text!r}")

    def _parse_vector(self) -> tuple[Any, ...]:
        rows: list[list[Any]] = [[]]
        saw_row_separator = False

        def finish() -> tuple[Any, ...]:
            if not saw_row_separator:
                return tuple(rows[0])
            widths = {len(row) for row in rows}
            if 0 in widths or len(widths) != 1:
                raise ParameterParseError(
                    "matrix rows must be non-empty and have equal lengths"
                )
            if any(
                not _is_number(item)
                for row in rows
                for item in row
            ):
                raise ParameterParseError(
                    "matrix rows must contain numeric scalar expressions"
                )
            # A MATLAB column vector remains a parameter vector; matrices
            # with two or more columns preserve their row structure.
            if next(iter(widths)) == 1:
                return tuple(row[0] for row in rows)
            return tuple(tuple(row) for row in rows)

        if self._peek_text() == "]":
            self._take()
            return ()
        while True:
            rows[-1].append(self._parse_additive())
            token = self._peek()
            if token is None:
                raise ParameterParseError("unterminated vector")
            if token.text == "]":
                self._take()
                return finish()
            if token.text == ",":
                self._take()
                continue
            if token.text == ";":
                self._take()
                saw_row_separator = True
                if not rows[-1]:
                    raise ParameterParseError("matrix rows cannot be empty")
                if self._peek_text() == "]":
                    # MATLAB accepts a trailing row separator.
                    self._take()
                    return finish()
                rows.append([])
                continue
            # Whitespace-separated literals are common in MATLAB vectors. They
            # are safe when the next token clearly begins another primary.
            if token.kind in {"number", "string", "identifier"} or token.text in {"[", "("}:
                continue
            raise ParameterParseError("vectors must contain comma/space-separated values")

    def _peek(self) -> _Token | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def _peek_text(self) -> str | None:
        token = self._peek()
        return None if token is None else token.text

    def _take(self) -> _Token:
        token = self._peek()
        if token is None:
            raise ParameterParseError("unexpected end of value")
        self.index += 1
        return token

    def _expect(self, text: str) -> None:
        token = self._take()
        if token.text != text:
            raise ParameterParseError(f"expected {text!r}, received {token.text!r}")

    @classmethod
    def _tokenize(cls, source: str) -> tuple[_Token, ...]:
        tokens: list[_Token] = []
        index = 0
        while index < len(source):
            char = source[index]
            if char.isspace():
                index += 1
                continue
            if char in {"'", '"'}:
                end = _quoted_end(source, index, char)
                tokens.append(_Token("string", source[index:end], index))
                index = end
                continue
            number = cls._NUMBER_RE.match(source, index)
            if number is not None:
                tokens.append(_Token("number", number.group(0), index))
                index = number.end()
                continue
            if char.isalpha() or char == "_":
                end = index + 1
                while end < len(source) and (source[end].isalnum() or source[end] == "_"):
                    end += 1
                tokens.append(_Token("identifier", source[index:end], index))
                index = end
                continue
            operator = next(
                (item for item in cls._OPERATORS if source.startswith(item, index)),
                None,
            )
            if operator is not None:
                tokens.append(_Token("operator", operator, index))
                index += len(operator)
                continue
            if char in "[](),;":
                tokens.append(_Token("punctuation", char, index))
                index += 1
                continue
            raise ParameterParseError(
                f"unsupported character {char!r} at column {index + 1}"
            )
        return tuple(tokens)


def _quoted_end(source: str, start: int, quote: str) -> int:
    index = start + 1
    while index < len(source):
        if source[index] == quote:
            if index + 1 < len(source) and source[index + 1] == quote:
                index += 2
                continue
            return index + 1
        index += 1
    raise ParameterParseError(f"unterminated {quote} string")


def _unquote_matlab(source: str) -> str:
    quote = source[0]
    return source[1:-1].replace(quote * 2, quote)


def _number_value(source: str) -> int | float:
    if not any(char in source for char in ".eE"):
        return int(source)
    return float(source)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_numeric_value(value: Any) -> bool:
    return _is_number(value) or (
        isinstance(value, tuple) and all(_is_numeric_value(item) for item in value)
    )


def _map_numeric(value: Any, function) -> Any:
    if _is_number(value):
        return function(value)
    if isinstance(value, tuple):
        return tuple(_map_numeric(item, function) for item in value)
    raise ParameterParseError("arithmetic requires numeric scalars or vectors")


def _apply_numeric_operator(left: Any, right: Any, operator: str) -> Any:
    if not _is_numeric_value(left) or not _is_numeric_value(right):
        raise ParameterParseError("arithmetic requires numeric scalars or vectors")

    if _is_number(left) and _is_number(right):
        return _numeric_operation(left, right, operator)
    if isinstance(left, tuple) and _is_number(right):
        return tuple(_apply_numeric_operator(item, right, operator) for item in left)
    if _is_number(left) and isinstance(right, tuple):
        return tuple(_apply_numeric_operator(left, item, operator) for item in right)
    if isinstance(left, tuple) and isinstance(right, tuple):
        if len(left) != len(right):
            raise ParameterParseError("element-wise vectors must have equal lengths")
        return tuple(
            _apply_numeric_operator(first, second, operator)
            for first, second in zip(left, right, strict=True)
        )
    raise ParameterParseError("unsupported arithmetic operands")


def _numeric_operation(left: int | float, right: int | float, operator: str) -> Any:
    try:
        if operator == "+":
            return left + right
        if operator == "-":
            return left - right
        if operator in {"*", ".*"}:
            return left * right
        if operator in {"/", "./"}:
            return left / right
        if operator in {"^", ".^"}:
            return left**right
    except (ArithmeticError, OverflowError) as exc:
        raise ParameterParseError(f"invalid numeric operation: {exc}") from exc
    raise ParameterParseError(f"unsupported numeric operator {operator!r}")
