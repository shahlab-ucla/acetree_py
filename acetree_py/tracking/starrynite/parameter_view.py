"""Exact, non-executable views over staged and regional legacy parameters.

The upstream ``getParameter`` helper has a small but consequential contract:
stage boundaries are strict, regional boxes are lower-exclusive and
upper-inclusive, and the regional value replaces the staged default only for
the matching stage.  This module implements that contract without evaluating
the surrounding MATLAB parameter file.
"""

from __future__ import annotations

import bisect
import math
import re
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Sequence

from .parameters import (
    ParameterParseError,
    ParameterValue,
    StarryNiteParameters,
    parse_parameter_expression,
)


_REGION_INITIALIZER_RE = re.compile(
    r"^parameters\.regions\s*=\s*cell\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)$",
    re.ASCII,
)
_REGION_ASSIGNMENT_RE = re.compile(
    r"^parameters\.regions\s*\{\s*(\d+)\s*\}\s*\.\s*"
    r"([A-Za-z_]\w*)\s*=\s*(.+)$",
    re.ASCII | re.DOTALL,
)
_CONTROL_RE = re.compile(
    r"^(?:if|elseif|else|end|switch|case|otherwise|for|while|try|catch)\b",
    re.ASCII | re.IGNORECASE,
)


class LegacyParameterResolutionError(ValueError):
    """Raised when exact ``getParameter`` behavior cannot be established."""


@dataclass(frozen=True, slots=True)
class LegacyRegionalParseIssue:
    """One regional statement that could not be represented exactly."""

    record_index: int
    message: str


@dataclass(frozen=True, slots=True)
class LegacyRegionDefinition:
    """The final MATLAB struct stored in one ``parameters.regions`` cell."""

    matlab_stage_index: int
    area: tuple[float, float, float, float, float, float] | None
    values: Mapping[str, ParameterValue]
    source_record_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.matlab_stage_index < 1:
            raise ValueError("matlab_stage_index must be one-based")
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))


@dataclass(frozen=True, slots=True)
class LegacyRegionalParameterTable:
    """A safely reconstructed ``parameters.regions`` cell array."""

    capacity: int
    regions: Mapping[int, LegacyRegionDefinition]
    consumed_record_indices: tuple[int, ...] = ()
    issues: tuple[LegacyRegionalParseIssue, ...] = ()

    def __post_init__(self) -> None:
        if self.capacity < 0:
            raise ValueError("capacity must be non-negative")
        object.__setattr__(self, "regions", MappingProxyType(dict(self.regions)))

    def for_stage(self, matlab_stage_index: int) -> LegacyRegionDefinition | None:
        """Return a region only when MATLAB's cell-length guard would allow it."""

        if matlab_stage_index < 1 or matlab_stage_index > self.capacity:
            return None
        return self.regions.get(matlab_stage_index)


@dataclass(frozen=True, slots=True)
class LegacyParameterResolution:
    """An auditable result from the exact staged/regional lookup."""

    name: str
    cell_count: int
    stage_index: int
    matlab_stage_index: int
    base_value: ParameterValue
    value: ParameterValue
    location: tuple[float, float, float] | None
    regional_override_applied: bool = False
    regional_area: tuple[float, float, float, float, float, float] | None = None
    regional_source_record_indices: tuple[int, ...] = ()


@dataclass(slots=True)
class _MutableRegion:
    values: dict[str, ParameterValue] = field(default_factory=dict)
    record_indices: list[int] = field(default_factory=list)


def build_legacy_region_table(
    parameters: StarryNiteParameters,
    *,
    strict: bool = True,
) -> LegacyRegionalParameterTable:
    """Reconstruct safe regional assignments in source order.

    Only literal ``cell(rows, columns)`` initializers and direct
    ``parameters.regions{N}.field = expression`` assignments are accepted.
    Expressions may reference previously assigned inert numeric constants, but
    never functions or the MATLAB workspace.  Reinitializing the cell array
    resets prior values exactly as MATLAB would.
    """

    if not isinstance(parameters, StarryNiteParameters):
        raise TypeError("parameters must be StarryNiteParameters")

    constants: dict[str, ParameterValue] = {}
    mutable_regions: dict[int, _MutableRegion] = {}
    capacity = 0
    saw_initializer = False
    consumed: list[int] = []
    issues: list[LegacyRegionalParseIssue] = []
    control_records: list[int] = []

    for record_index, record in enumerate(parameters.records):
        code = _statement_code(record.source)
        if record.kind == "assignment" and record.name is not None:
            if "." not in record.name and not isinstance(record.value, str):
                constants[record.name] = record.value  # type: ignore[assignment]

        initializer = _REGION_INITIALIZER_RE.fullmatch(code)
        if initializer is not None:
            rows, columns = (int(item) for item in initializer.groups())
            capacity = max(rows, columns)
            mutable_regions.clear()
            saw_initializer = True
            consumed.append(record_index)
            continue

        assignment = _REGION_ASSIGNMENT_RE.fullmatch(code)
        if assignment is not None:
            matlab_stage_index = int(assignment.group(1))
            field_name = assignment.group(2)
            expression = assignment.group(3).strip()
            consumed.append(record_index)
            if matlab_stage_index < 1:
                issues.append(
                    LegacyRegionalParseIssue(
                        record_index,
                        "regional cell indices must be one-based positive integers",
                    )
                )
                continue
            if not saw_initializer:
                issues.append(
                    LegacyRegionalParseIssue(
                        record_index,
                        "regional assignment appears before parameters.regions is initialized",
                    )
                )
            try:
                value = parse_parameter_expression(expression, constants=constants)
            except (ParameterParseError, ValueError) as exc:
                issues.append(
                    LegacyRegionalParseIssue(
                        record_index,
                        f"could not safely resolve regional field {field_name}: {exc}",
                    )
                )
                continue
            capacity = max(capacity, matlab_stage_index)
            region = mutable_regions.setdefault(matlab_stage_index, _MutableRegion())
            region.values[field_name] = value
            region.record_indices.append(record_index)
            continue

        if record.kind == "opaque":
            if re.search(r"\bparameters\.regions\b", code, re.ASCII):
                issues.append(
                    LegacyRegionalParseIssue(
                        record_index,
                        "unsupported parameters.regions statement",
                    )
                )
            if _CONTROL_RE.match(code):
                control_records.append(record_index)

    if consumed and control_records:
        first = control_records[0]
        issues.append(
            LegacyRegionalParseIssue(
                first,
                "regional definitions coexist with unsupported MATLAB control flow",
            )
        )

    frozen_regions: dict[int, LegacyRegionDefinition] = {}
    for stage, mutable in mutable_regions.items():
        raw_area = mutable.values.get("area")
        area: tuple[float, float, float, float, float, float] | None = None
        if raw_area is not None:
            try:
                area = _coerce_area(raw_area)
            except LegacyParameterResolutionError as exc:
                issues.append(
                    LegacyRegionalParseIssue(
                        mutable.record_indices[-1],
                        f"invalid stage-{stage} regional area: {exc}",
                    )
                )
        values = {name: value for name, value in mutable.values.items() if name != "area"}
        frozen_regions[stage] = LegacyRegionDefinition(
            matlab_stage_index=stage,
            area=area,
            values=values,
            source_record_indices=tuple(mutable.record_indices),
        )

    table = LegacyRegionalParameterTable(
        capacity=capacity,
        regions=frozen_regions,
        consumed_record_indices=tuple(consumed),
        issues=tuple(issues),
    )
    if strict and table.issues:
        details = "; ".join(
            f"record {issue.record_index}: {issue.message}" for issue in table.issues
        )
        raise LegacyParameterResolutionError(
            f"Legacy regional parameters cannot be resolved exactly: {details}"
        )
    return table


def resolve_legacy_parameter(
    parameters: StarryNiteParameters,
    name: str,
    *,
    cell_count: int,
    location: Sequence[float] | None = None,
    region_table: LegacyRegionalParameterTable | None = None,
) -> LegacyParameterResolution:
    """Match upstream ``getParameter(name, numcells, location)`` semantics."""

    if not isinstance(parameters, StarryNiteParameters):
        raise TypeError("parameters must be StarryNiteParameters")
    field_name = _parameter_field_name(name)
    if isinstance(cell_count, bool) or int(cell_count) != cell_count:
        raise LegacyParameterResolutionError("cell_count must be an integer")
    count = int(cell_count)

    boundaries = _numeric_vector(
        "parameters.staging",
        parameters.settings.get("parameters.staging"),
    )
    if not boundaries:
        raise LegacyParameterResolutionError("parameters.staging cannot be empty")
    if any(left > right for left, right in zip(boundaries, boundaries[1:])):
        raise LegacyParameterResolutionError(
            "parameters.staging must be sorted in nondecreasing order"
        )
    stage_index = bisect.bisect_left(boundaries, float(count))

    source_name = f"parameters.{field_name}"
    if source_name not in parameters.settings:
        raise LegacyParameterResolutionError(
            f"Legacy parameter {source_name!r} is missing"
        )
    base_value = _select_stage_value(
        source_name,
        parameters.settings[source_name],
        stage_index,
    )
    value = base_value
    matlab_stage_index = stage_index + 1
    table = region_table or build_legacy_region_table(parameters, strict=True)
    region = table.for_stage(matlab_stage_index)
    applied = False
    point: tuple[float, float, float] | None = None
    if region is not None and field_name in region.values:
        if region.area is None:
            raise LegacyParameterResolutionError(
                f"Stage {matlab_stage_index} defines regional {field_name} without "
                "a valid six-value area"
            )
        if location is None:
            raise LegacyParameterResolutionError(
                f"A three-coordinate location is required to resolve regional {field_name}"
            )
        point = _coerce_location(location)
        if _inside_legacy_region(point, region.area):
            value = region.values[field_name]
            applied = True
    elif location is not None:
        point = _coerce_location(location)

    return LegacyParameterResolution(
        name=field_name,
        cell_count=count,
        stage_index=stage_index,
        matlab_stage_index=matlab_stage_index,
        base_value=base_value,
        value=value,
        location=point,
        regional_override_applied=applied,
        regional_area=region.area if region is not None else None,
        regional_source_record_indices=(
            region.source_record_indices if region is not None else ()
        ),
    )


def _statement_code(source: str) -> str:
    code = source.strip()
    return code[:-1].rstrip() if code.endswith(";") else code


def _parameter_field_name(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise LegacyParameterResolutionError("parameter name must be non-empty text")
    field_name = name[len("parameters.") :] if name.startswith("parameters.") else name
    if not re.fullmatch(r"[A-Za-z_]\w*", field_name, re.ASCII):
        raise LegacyParameterResolutionError(
            "parameter name must identify one direct parameters struct field"
        )
    return field_name


def _numeric_vector(name: str, value: ParameterValue | None) -> tuple[float, ...]:
    if value is None:
        raise LegacyParameterResolutionError(f"Legacy parameter {name!r} is missing")
    items = value if isinstance(value, tuple) else (value,)
    result: list[float] = []
    for item in items:
        if isinstance(item, (bool, str, tuple)):
            raise LegacyParameterResolutionError(f"{name} must be a numeric vector")
        number = float(item)
        if not math.isfinite(number):
            raise LegacyParameterResolutionError(f"{name} must contain finite values")
        result.append(number)
    return tuple(result)


def _select_stage_value(
    name: str,
    value: ParameterValue,
    stage_index: int,
) -> ParameterValue:
    if not isinstance(value, tuple):
        return value
    if len(value) == 1:
        return value[0]
    if stage_index >= len(value):
        raise LegacyParameterResolutionError(
            f"{name} has {len(value)} stage values but MATLAB stage "
            f"{stage_index + 1} was requested"
        )
    return value[stage_index]


def _coerce_area(
    value: ParameterValue,
) -> tuple[float, float, float, float, float, float]:
    if not isinstance(value, tuple) or len(value) != 6:
        raise LegacyParameterResolutionError("area must contain exactly six values")
    numbers = _finite_coordinates(value, label="area")
    area = tuple(numbers)
    if any(area[index] >= area[index + 1] for index in (0, 2, 4)):
        raise LegacyParameterResolutionError(
            "each area lower bound must be smaller than its upper bound"
        )
    return area  # type: ignore[return-value]


def _coerce_location(location: Sequence[float]) -> tuple[float, float, float]:
    if isinstance(location, (str, bytes, bytearray)) or len(location) != 3:
        raise LegacyParameterResolutionError("location must contain three coordinates")
    values = _finite_coordinates(location, label="location")
    return values[0], values[1], values[2]


def _finite_coordinates(
    values: Sequence[object],
    *,
    label: str,
) -> tuple[float, ...]:
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise LegacyParameterResolutionError(f"{label} values must be numeric")
        number = float(value)
        if not math.isfinite(number):
            raise LegacyParameterResolutionError(f"{label} values must be finite")
        result.append(number)
    return tuple(result)


def _inside_legacy_region(
    location: tuple[float, float, float],
    area: tuple[float, float, float, float, float, float],
) -> bool:
    return all(
        location[axis] > area[axis * 2]
        and location[axis] <= area[axis * 2 + 1]
        for axis in range(3)
    )


__all__ = [
    "LegacyParameterResolution",
    "LegacyParameterResolutionError",
    "LegacyRegionDefinition",
    "LegacyRegionalParameterTable",
    "LegacyRegionalParseIssue",
    "build_legacy_region_table",
    "resolve_legacy_parameter",
]
