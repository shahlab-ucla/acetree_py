"""Clean-room inspection of legacy StarryNite MATLAB model artifacts.

The loader extracts data with :mod:`scipy.io` when available.  It never
instantiates MATLAB code or attempts to reproduce proprietary/GPL classifier
implementations; unsupported MATLAB objects are represented by metadata-only
values so callers can make an explicit compatibility decision.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


_HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
_MAX_RECURSION_DEPTH = 64


class StarryNiteModelError(RuntimeError):
    """Base class for actionable StarryNite model-loading failures."""


class MissingScipyError(StarryNiteModelError):
    """Raised when classic MAT-file support is unavailable."""


class UnsupportedMatlabVersionError(StarryNiteModelError):
    """Raised for MATLAB v7.3/HDF5 artifacts unsupported by scipy.io."""


class MatlabModelLoadError(StarryNiteModelError):
    """Raised when a classic MAT-file is corrupt or otherwise unreadable."""


@dataclass(frozen=True, slots=True)
class OpaqueMatlabValue:
    """Metadata retained for a MATLAB value that cannot be safely normalized."""

    python_type: str
    reason: str
    matlab_class: str | None = None
    shape: tuple[int, ...] | None = None
    dtype: str | None = None
    field_names: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ClassifierIdentification:
    """Conservative old/new classifier identification and its evidence."""

    flavor: str
    evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.flavor not in {"legacy_classifier", "new_classifier", "unknown"}:
            raise ValueError(f"Unsupported classifier flavor: {self.flavor!r}")


@dataclass(frozen=True, slots=True)
class StarryNiteModel:
    """Normalized, provenance-rich view of one MATLAB model file."""

    path: Path
    sha256: str
    fields: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    classifier_flavor: str = "unknown"
    classifier_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if not re.fullmatch(r"[0-9a-f]{64}", self.sha256):
            raise ValueError("sha256 must be a lowercase hexadecimal SHA-256 digest")
        object.__setattr__(self, "fields", _freeze_mapping(self.fields))
        object.__setattr__(self, "metadata", _freeze_mapping(self.metadata))
        identification = ClassifierIdentification(
            self.classifier_flavor,
            tuple(self.classifier_evidence),
        )
        object.__setattr__(self, "classifier_flavor", identification.flavor)
        object.__setattr__(self, "classifier_evidence", identification.evidence)


def load_matlab_model(
    path: str | Path,
    *,
    max_array_elements: int = 1_000_000,
) -> StarryNiteModel:
    """Load a classic MATLAB MAT-file into typed, recursively exposed fields.

    MATLAB v7.3 uses HDF5 and is intentionally rejected with conversion advice;
    silently interpreting it through a second backend would make compatibility
    behavior dependent on optional packages and MATLAB object conventions.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"StarryNite model file was not found: {source}")
    if isinstance(max_array_elements, bool) or max_array_elements < 1:
        raise ValueError("max_array_elements must be a positive integer")

    digest = sha256_file(source)
    if _is_hdf5_mat_file(source):
        raise UnsupportedMatlabVersionError(
            f"{source} is a MATLAB v7.3/HDF5 MAT-file. scipy.io cannot read this "
            "format; export a compatibility copy with MATLAB '-v7' (or provide an "
            "explicit, separately tested HDF5 adapter) and keep the original model "
            "for provenance."
        )

    try:
        scipy_io, scipy_version = _import_scipy_io()
    except (ImportError, ModuleNotFoundError) as exc:
        raise MissingScipyError(
            "Loading classic StarryNite .mat models requires scipy.io. Install a "
            "compatible SciPy build, then retry; the model file was not modified."
        ) from exc

    try:
        loaded = scipy_io.loadmat(
            source,
            struct_as_record=False,
            squeeze_me=True,
            chars_as_strings=True,
        )
    except NotImplementedError as exc:
        raise UnsupportedMatlabVersionError(
            f"SciPy reports that {source} requires an HDF5/v7.3 reader. Export a "
            "MATLAB '-v7' compatibility copy before loading it."
        ) from exc
    except Exception as exc:
        # SciPy exposes additional reader-specific exception classes (for
        # example ``MatReadError``) that are not stable members of our public
        # boundary.  Normalize every classic-MAT parser failure while allowing
        # process-control exceptions such as KeyboardInterrupt to propagate.
        raise MatlabModelLoadError(
            f"Could not read StarryNite model {source} as a classic MATLAB MAT-file: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    metadata = {
        "matlab_header": _decode_metadata(loaded.get("__header__")),
        "matlab_version": _decode_metadata(loaded.get("__version__")),
        "matlab_globals": tuple(
            str(item) for item in loaded.get("__globals__", ())
        ),
        "scipy_version": str(scipy_version),
        "source_size_bytes": source.stat().st_size,
    }
    fields = {
        str(name): _normalize_matlab_value(
            value,
            max_array_elements=max_array_elements,
            path=str(name),
        )
        for name, value in loaded.items()
        if not str(name).startswith("__")
    }
    identification = identify_classifier_flavor(fields, path=source)
    return StarryNiteModel(
        path=source.resolve(strict=False),
        sha256=digest,
        fields=fields,
        metadata=metadata,
        classifier_flavor=identification.flavor,
        classifier_evidence=identification.evidence,
    )


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for provenance and model identity."""

    if isinstance(chunk_size, bool) or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def identify_classifier_flavor(
    fields: Mapping[str, Any],
    *,
    path: str | Path | None = None,
) -> ClassifierIdentification:
    """Identify StarryNite's legacy/new Bayesian classifier conservatively.

    Explicit ``old/new`` metadata and MATLAB class names are treated as strong
    evidence.  Generic numeric layouts are reported as unknown rather than
    guessed, which keeps behavioral compatibility claims honest.
    """

    new_evidence: list[str] = []
    legacy_evidence: list[str] = []

    if path is not None:
        lower_path = str(path).replace("\\", "/").lower()
        if "newmatlab" in lower_path or re.search(r"(?:^|[/_.-])new(?:[/_.-]|$)", lower_path):
            new_evidence.append("path identifies the newmatlab classifier family")
        if "oldmatlab" in lower_path or re.search(r"(?:^|[/_.-])old(?:[/_.-]|$)", lower_path):
            legacy_evidence.append("path identifies the oldmatlab classifier family")

    for field_path, value in _walk_fields(fields):
        key = field_path.lower().replace("-", "_")
        if isinstance(value, OpaqueMatlabValue):
            matlab_class = (value.matlab_class or "").lower()
            if (
                "classificationnaivebayes" in matlab_class
                or "compactclassification" in matlab_class
            ):
                new_evidence.append(
                    f"{field_path} has MATLAB class {value.matlab_class}"
                )
            elif matlab_class == "naivebayes" or matlab_class.endswith(".naivebayes"):
                legacy_evidence.append(
                    f"{field_path} has MATLAB class {value.matlab_class}"
                )
            continue

        if not isinstance(value, str):
            continue
        marker = value.strip().lower().replace("-", "_").replace(" ", "_")
        explicit_key = any(
            item in key
            for item in (
                "classifier_flavor",
                "classifier_family",
                "classifier_type",
                "matlab_family",
            )
        )
        if explicit_key and marker in {"new", "new_classifier", "newmatlab"}:
            new_evidence.append(f"{field_path} explicitly declares {value!r}")
        elif explicit_key and marker in {
            "old",
            "legacy",
            "legacy_classifier",
            "oldmatlab",
        }:
            legacy_evidence.append(f"{field_path} explicitly declares {value!r}")
        elif "classificationnaivebayes" in marker or "compactclassification" in marker:
            new_evidence.append(f"{field_path} names classifier {value!r}")
        elif marker == "naivebayes" or marker.endswith(".naivebayes"):
            legacy_evidence.append(f"{field_path} names classifier {value!r}")

    if new_evidence and not legacy_evidence:
        return ClassifierIdentification("new_classifier", tuple(dict.fromkeys(new_evidence)))
    if legacy_evidence and not new_evidence:
        return ClassifierIdentification(
            "legacy_classifier",
            tuple(dict.fromkeys(legacy_evidence)),
        )
    if new_evidence and legacy_evidence:
        evidence = tuple(
            dict.fromkeys(
                ["conflicting legacy and new classifier markers", *new_evidence, *legacy_evidence]
            )
        )
        return ClassifierIdentification("unknown", evidence)
    return ClassifierIdentification("unknown")


# Compatibility-friendly alias.
load_model = load_matlab_model


def _import_scipy_io():
    import scipy
    from scipy import io as scipy_io

    return scipy_io, scipy.__version__


def _is_hdf5_mat_file(path: Path) -> bool:
    with path.open("rb") as stream:
        header = stream.read(1024)
    return header.startswith(b"MATLAB 7.3 MAT-file") or any(
        header.startswith(_HDF5_SIGNATURE, offset)
        for offset in (0, 512)
        if len(header) >= offset + len(_HDF5_SIGNATURE)
    )


def _decode_metadata(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _normalize_matlab_value(
    value: Any,
    *,
    max_array_elements: int,
    path: str,
    depth: int = 0,
    seen: set[int] | None = None,
) -> Any:
    if depth > _MAX_RECURSION_DEPTH:
        return _opaque(value, "maximum structure depth exceeded")
    if seen is None:
        seen = set()

    if value is None or isinstance(value, (bool, int, float, complex, str)):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return _opaque(value, "non-UTF-8 MATLAB byte string")
    if isinstance(value, np.generic):
        return _normalize_matlab_value(
            value.item(),
            max_array_elements=max_array_elements,
            path=path,
            depth=depth + 1,
            seen=seen,
        )

    identity = id(value)
    if identity in seen:
        return _opaque(value, f"circular reference encountered at {path}")

    matlab_class = getattr(value, "classname", None)
    if matlab_class:
        return _opaque(
            value,
            "MATLAB class object retained as metadata; execution is unsupported",
            matlab_class=str(matlab_class),
        )
    if type(value).__name__ in {"MatlabOpaque", "MatlabFunction"}:
        return _opaque(
            value,
            "opaque MATLAB object retained as metadata; execution is unsupported",
            matlab_class=_opaque_matlab_class(value),
        )

    field_names = getattr(value, "_fieldnames", None)
    if field_names is not None:
        seen.add(identity)
        try:
            return MappingProxyType(
                {
                    str(name): _normalize_matlab_value(
                        getattr(value, name),
                        max_array_elements=max_array_elements,
                        path=f"{path}.{name}",
                        depth=depth + 1,
                        seen=seen,
                    )
                    for name in field_names
                }
            )
        finally:
            seen.remove(identity)

    if isinstance(value, Mapping):
        seen.add(identity)
        try:
            return MappingProxyType(
                {
                    str(key): _normalize_matlab_value(
                        item,
                        max_array_elements=max_array_elements,
                        path=f"{path}.{key}",
                        depth=depth + 1,
                        seen=seen,
                    )
                    for key, item in value.items()
                }
            )
        finally:
            seen.remove(identity)

    if isinstance(value, np.ndarray):
        if value.size > max_array_elements:
            return _opaque(
                value,
                f"array exceeds extraction limit of {max_array_elements} elements",
            )
        seen.add(identity)
        try:
            if value.dtype.names:
                normalized_items = [
                    MappingProxyType(
                        {
                            str(name): _normalize_matlab_value(
                                item[name],
                                max_array_elements=max_array_elements,
                                path=f"{path}.{name}",
                                depth=depth + 1,
                                seen=seen,
                            )
                            for name in value.dtype.names
                        }
                    )
                    for item in value.reshape(-1)
                ]
                return _reshape_flat(normalized_items, value.shape)
            return _normalize_python_container(
                value.tolist(),
                max_array_elements=max_array_elements,
                path=path,
                depth=depth + 1,
                seen=seen,
            )
        finally:
            seen.remove(identity)

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        seen.add(identity)
        try:
            return tuple(
                _normalize_matlab_value(
                    item,
                    max_array_elements=max_array_elements,
                    path=f"{path}[{index}]",
                    depth=depth + 1,
                    seen=seen,
                )
                for index, item in enumerate(value)
            )
        finally:
            seen.remove(identity)

    return _opaque(value, "unsupported MATLAB/Python object representation")


def _normalize_python_container(
    value: Any,
    *,
    max_array_elements: int,
    path: str,
    depth: int,
    seen: set[int],
) -> Any:
    if isinstance(value, list):
        return tuple(
            _normalize_python_container(
                item,
                max_array_elements=max_array_elements,
                path=f"{path}[{index}]",
                depth=depth + 1,
                seen=seen,
            )
            for index, item in enumerate(value)
        )
    return _normalize_matlab_value(
        value,
        max_array_elements=max_array_elements,
        path=path,
        depth=depth,
        seen=seen,
    )


def _reshape_flat(items: list[Any], shape: tuple[int, ...]) -> Any:
    if not shape:
        return items[0] if items else ()
    if len(shape) == 1:
        return tuple(items)
    stride = int(np.prod(shape[1:], dtype=int))
    return tuple(
        _reshape_flat(items[index : index + stride], shape[1:])
        for index in range(0, len(items), stride)
    )


def _opaque(
    value: Any,
    reason: str,
    *,
    matlab_class: str | None = None,
) -> OpaqueMatlabValue:
    shape_value = getattr(value, "shape", None)
    shape = None
    if shape_value is not None:
        try:
            shape = tuple(int(item) for item in shape_value)
        except (TypeError, ValueError):
            shape = None
    dtype_value = getattr(value, "dtype", None)
    field_names = getattr(value, "_fieldnames", ()) or ()
    if not field_names and dtype_value is not None:
        field_names = getattr(dtype_value, "names", ()) or ()
    return OpaqueMatlabValue(
        python_type=f"{type(value).__module__}.{type(value).__qualname__}",
        reason=reason,
        matlab_class=matlab_class,
        shape=shape,
        dtype=None if dtype_value is None else str(dtype_value),
        field_names=tuple(str(name) for name in field_names),
    )


def _opaque_matlab_class(value: Any) -> str | None:
    """Extract the inert MCOS class label carried by SciPy ``MatlabOpaque``."""

    dtype = getattr(value, "dtype", None)
    names = getattr(dtype, "names", ()) or ()
    if "_Class" not in names:
        return None
    try:
        raw = value["_Class"]
        while isinstance(raw, np.ndarray) and raw.size == 1:
            raw = raw.reshape(-1)[0]
        if isinstance(raw, bytes):
            text = raw.decode("utf-8", errors="replace")
        else:
            text = str(raw)
    except (IndexError, KeyError, TypeError, ValueError):
        return None
    text = text.strip()
    if not text or len(text) > 256 or not re.fullmatch(r"[A-Za-z0-9_.]+", text):
        return None
    return text


def _freeze_mapping(value: Mapping[str, Any]) -> Mapping[str, Any]:
    return MappingProxyType({str(key): _freeze_nested(item) for key, item in value.items()})


def _freeze_nested(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, list):
        return tuple(_freeze_nested(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_nested(item) for item in value)
    return value


def _walk_fields(value: Any, path: str = ""):
    if isinstance(value, Mapping):
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            yield from _walk_fields(item, child_path)
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            yield from _walk_fields(item, f"{path}[{index}]")
        return
    yield path, value
