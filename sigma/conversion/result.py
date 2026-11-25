from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Sequence, overload

from sigma.exceptions import (
    SigmaBackendConsistencyError,
    SigmaError,
    SigmaFeatureNotSupportedByBackendError,
    SigmaRuleLocation,
)


def _as_query_list(queries: Any) -> list[Any]:
    """Return a backend result as a list without splitting strings, bytes or mappings."""
    if queries is None:
        return []
    if isinstance(queries, (list, tuple)):
        return list(queries)
    return [queries]


@dataclass
class UnsupportedFeature:
    """
    Explicit description of a Sigma feature that a backend does not fully support.

    A feature can be *unsupported* (``severity="error"``, the query may be missing) or
    *degraded* (``severity="degraded"``, the generated query is an approximation that loses
    semantics). In both cases the feature is recorded instead of being dropped silently.
    """

    feature: str
    reason: str = ""
    backend: str | None = None
    source: SigmaRuleLocation | None = field(default=None, compare=False)
    rule: Any = field(default=None, compare=False, repr=False)
    severity: str = "error"
    field_name: str | None = None
    value: str | None = None
    modifier: str | None = None
    handled: bool = False
    fallback: str | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Accept str enums and plain strings alike.
        if not isinstance(self.feature, str):
            enum_value = getattr(self.feature, "value", self.feature)
            self.feature = str(enum_value)
        if not self.reason:
            self.reason = self.feature

    @property
    def name(self) -> str:
        """Alias for :attr:`feature`."""
        return self.feature

    @property
    def category(self) -> str:
        """Alias for :attr:`feature`."""
        return self.feature

    @property
    def field(self) -> str | None:
        """Alias for :attr:`field_name`."""
        return self.field_name

    @property
    def location(self) -> SigmaRuleLocation | None:
        """Alias for :attr:`source`."""
        return self.source

    @property
    def rule_id(self) -> str | None:
        """Identifier of the affected rule, if available."""
        rule_id = getattr(self.rule, "id", None)
        return str(rule_id) if rule_id is not None else None

    @property
    def rule_title(self) -> str | None:
        """Title of the affected rule, if available."""
        title = getattr(self.rule, "title", None)
        return str(title) if title is not None else None

    @property
    def message(self) -> str:
        """Human readable description of the unsupported feature."""
        message = f"{self.feature}: {self.reason}"
        if self.field_name:
            message += f" (field '{self.field_name}')"
        if self.modifier:
            message += f" (modifier '{self.modifier}')"
        return message

    @property
    def is_degraded(self) -> bool:
        """True if the feature was converted to an approximation instead of being unsupported."""
        return self.severity in {"degraded", "warning"} or self.handled or bool(self.fallback)

    @property
    def is_error(self) -> bool:
        """True if the feature could not be represented at all."""
        return self.severity == "error" and not self.handled

    def to_dict(self) -> dict[str, Any]:
        """Serialize the annotation into a JSON-compatible mapping."""
        return {
            "feature": self.feature,
            "reason": self.reason,
            "backend": self.backend,
            "severity": self.severity,
            "field": self.field_name,
            "value": self.value,
            "modifier": self.modifier,
            "handled": self.handled,
            "fallback": self.fallback,
            "source": str(self.source) if self.source is not None else None,
            "rule_id": self.rule_id,
            "rule_title": self.rule_title,
            "context": dict(self.context),
        }

    def __str__(self) -> str:
        prefix = f"[{self.backend}] " if self.backend else ""
        suffix = " (degraded)" if self.is_degraded else ""
        return prefix + self.message + suffix

    @classmethod
    def from_exception(
        cls,
        exception: BaseException,
        backend: str | None = None,
        rule: Any = None,
    ) -> UnsupportedFeature:
        """Create an annotation from an exception raised during conversion."""
        message = str(exception)
        lower = message.lower()
        feature = "conversion"
        for marker, name in (
            ("regular expression", "regex"),
            ("regex", "regex"),
            ("cidr", "cidr"),
            ("comparison operation", "compare"),
            ("field reference", "fieldref"),
            ("field null", "null"),
            ("field exists", "exists"),
            ("field not exists", "exists"),
            ("timestamp", "timestamp"),
            ("boolean", "bool"),
            ("deferred", "deferred_expression"),
            ("negation", "condition_not"),
            ("or condition", "condition_or"),
            ("wildcard", "wildcard"),
            ("string value", "unbound_value"),
            ("numeric value", "unbound_value"),
            ("group expression", "group"),
            ("group by", "group_by"),
        ):
            if marker in lower:
                feature = name
                break
        return cls(
            feature=feature,
            reason=message or exception.__class__.__name__,
            backend=backend,
            source=getattr(exception, "source", None),
            rule=rule,
            severity="error",
        )


@dataclass(eq=False)
class BackendConversionResult:
    """
    Result of converting a rule set with a single backend.

    The object behaves like the list of generated queries for backwards-compatible access
    (iteration, indexing, length) while also exposing metadata, errors and unsupported
    feature annotations.
    """

    backend: str
    queries: Any = field(default_factory=list)
    output_format: str = "default"
    errors: list[BaseException] = field(default_factory=list)
    unsupported: list[UnsupportedFeature] = field(default_factory=list)
    field_mapping: dict[str, str] | None = None
    semantics: Any = None
    backend_semantics: Any = None
    source_semantics: Any = None
    rule_count: int = 0
    elapsed: float = 0.0
    failed: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def query_list(self) -> list[Any]:
        """All generated queries as a list."""
        return _as_query_list(self.queries)

    @property
    def query(self) -> Any:
        """The first generated query or ``None``."""
        queries = self.query_list
        return queries[0] if queries else None

    @property
    def outputs(self) -> Any:
        """Alias for :attr:`queries`."""
        return self.queries

    @property
    def success(self) -> bool:
        """True if no fatal error occurred while converting with this backend."""
        return not self.failed

    @property
    def complete(self) -> bool:
        """True if conversion succeeded and no unsupported error was recorded."""
        return self.success and not any(feature.is_error for feature in self.unsupported)

    @property
    def degraded(self) -> bool:
        """True if any feature was represented by an approximation."""
        return any(feature.is_degraded for feature in self.unsupported)

    @property
    def fatal_error(self) -> BaseException | None:
        """Return the first fatal exception, if any."""
        for error in self.errors:
            if not isinstance(error, SigmaFeatureNotSupportedByBackendError):
                return error
        return self.errors[0] if self.errors else None

    @property
    def annotations(self) -> list[str]:
        """Human readable annotations for all unsupported/degraded features."""
        return [str(feature) for feature in self.unsupported]

    @property
    def warnings(self) -> list[str]:
        """Alias for :attr:`annotations`."""
        return self.annotations

    @property
    def notices(self) -> list[str]:
        """Alias for :attr:`annotations`."""
        return self.annotations

    @property
    def degraded_features(self) -> list[UnsupportedFeature]:
        """Annotations for features that were represented by an approximation."""
        return [feature for feature in self.unsupported if feature.is_degraded]

    @property
    def lost_features(self) -> list[UnsupportedFeature]:
        """Annotations for features that could not be represented at all."""
        return [feature for feature in self.unsupported if feature.is_error]

    @property
    def annotation(self) -> str:
        """A single string containing all annotations."""
        return "; ".join(self.annotations)

    @property
    def unsupported_features(self) -> list[str]:
        """Distinct feature names that were reported for this backend."""
        return list(dict.fromkeys(feature.feature for feature in self.unsupported))

    def to_dict(self, include_semantics: bool = False) -> dict[str, Any]:
        """Serialize the result into a JSON-compatible mapping."""
        result: dict[str, Any] = {
            "backend": self.backend,
            "output_format": self.output_format,
            "queries": self.queries,
            "errors": [str(error) for error in self.errors],
            "unsupported": [feature.to_dict() for feature in self.unsupported],
            "field_mapping": self.field_mapping,
            "success": self.success,
            "complete": self.complete,
            "degraded": self.degraded,
            "rule_count": self.rule_count,
            "elapsed": self.elapsed,
            "metadata": self.metadata,
        }
        if include_semantics:
            result["semantics"] = self.semantics
            result["backend_semantics"] = self.backend_semantics
            result["source_semantics"] = self.source_semantics
        return result

    as_dict = to_dict

    @overload
    def __getitem__(self, key: int | slice) -> Any: ...

    @overload
    def __getitem__(self, key: str) -> Any: ...

    def __getitem__(self, key: int | slice | str) -> Any:
        if isinstance(key, (int, slice)):
            return self.query_list[key]
        attribute_map = {
            "backend": "backend",
            "name": "backend",
            "queries": "queries",
            "outputs": "queries",
            "query": "query",
            "output_format": "output_format",
            "format": "output_format",
            "errors": "errors",
            "unsupported": "unsupported",
            "unsupported_features": "unsupported_features",
            "annotations": "annotations",
            "warnings": "warnings",
            "notices": "notices",
            "degraded_features": "degraded_features",
            "lost_features": "lost_features",
            "field_mapping": "field_mapping",
            "semantics": "semantics",
            "backend_semantics": "backend_semantics",
            "source_semantics": "source_semantics",
            "success": "success",
            "complete": "complete",
            "degraded": "degraded",
            "metadata": "metadata",
        }
        try:
            return getattr(self, attribute_map[key])
        except KeyError as error:
            raise KeyError(key) from error

    def __iter__(self) -> Iterator[Any]:
        return iter(self.query_list)

    def __len__(self) -> int:
        return len(self.query_list)

    def __bool__(self) -> bool:
        return self.success and bool(self.query_list)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, BackendConversionResult):
            return (
                self.backend == other.backend
                and self.queries == other.queries
                and self.unsupported == other.unsupported
                and self.field_mapping == other.field_mapping
                and self.semantics == other.semantics
            )
        if isinstance(other, Mapping):
            known_keys = {
                "backend",
                "name",
                "queries",
                "outputs",
                "query",
                "output_format",
                "format",
                "errors",
                "unsupported",
                "field_mapping",
                "semantics",
                "success",
                "complete",
                "degraded",
                "metadata",
            }
            if not set(other).issubset(known_keys):
                return False
            for key, value in other.items():
                try:
                    if self[key] != value:
                        return False
                except KeyError:
                    return False
            return True
        if isinstance(other, (list, tuple)):
            return self.query_list == list(other)
        if isinstance(other, (str, bytes, dict)):
            return bool(self.queries == other)
        return NotImplemented

    def __str__(self) -> str:
        if not self.unsupported:
            return f"{self.backend}: {self.queries!r}"
        return f"{self.backend}: {self.queries!r} ({self.annotation})"


class MultiBackendResult(dict[str, BackendConversionResult]):
    """
    Mapping from backend name to :class:`BackendConversionResult`.

    The mapping also provides aggregate helpers for queries, errors and explicit unsupported
    feature annotations.
    """

    def __init__(
        self,
        results: Mapping[str, BackendConversionResult],
        source_semantics: Any = None,
        metadata: dict[str, Any] | None = None,
    ):
        super().__init__(results)
        self.source_semantics = source_semantics
        self.metadata = metadata or {}

    @property
    def results(self) -> dict[str, BackendConversionResult]:
        """Backend name to result mapping."""
        return dict(self)

    @property
    def by_backend(self) -> dict[str, BackendConversionResult]:
        """Alias for :attr:`results`."""
        return self.results

    @property
    def queries(self) -> dict[str, Any]:
        """Generated queries by backend name."""
        return {name: result.queries for name, result in self.items()}

    @property
    def outputs(self) -> dict[str, Any]:
        """Alias for :attr:`queries`."""
        return self.queries

    @property
    def errors(self) -> dict[str, list[BaseException]]:
        """Errors by backend name."""
        return {name: result.errors for name, result in self.items()}

    @property
    def unsupported(self) -> dict[str, list[UnsupportedFeature]]:
        """Unsupported feature annotations by backend name."""
        return {name: result.unsupported for name, result in self.items()}

    @property
    def all_unsupported(self) -> list[UnsupportedFeature]:
        """All unsupported feature annotations from all backends."""
        return [feature for result in self.values() for feature in result.unsupported]

    @property
    def warnings(self) -> list[str]:
        """Human readable annotations from all backends."""
        return [annotation for result in self.values() for annotation in result.annotations]

    @property
    def notices(self) -> list[str]:
        """Alias for :attr:`warnings`."""
        return self.warnings

    @property
    def unsupported_features(self) -> dict[str, list[str]]:
        """Distinct unsupported feature names by backend."""
        return {name: result.unsupported_features for name, result in self.items()}

    @property
    def success(self) -> bool:
        """True if every backend conversion succeeded."""
        return all(result.success for result in self.values())

    @property
    def complete(self) -> bool:
        """True if every backend conversion is complete and no feature was lost."""
        return all(result.complete for result in self.values())

    @property
    def degraded(self) -> bool:
        """True if any backend produced an approximation."""
        return any(result.degraded for result in self.values())

    @property
    def semantics(self) -> Any:
        """Normalized semantics of the first backend result."""
        if not self:
            return self.source_semantics
        return next(iter(self.values())).semantics

    @property
    def is_semantically_consistent(self) -> bool:
        """True if all successful backends represent the same normalized detection logic."""
        return self.is_consistent

    @property
    def is_consistent(self) -> bool:
        """True if all backend results have identical normalized semantics."""
        semantics = [result.semantics for result in self.values()]
        if not semantics:
            return True
        if any(item is None for item in semantics):
            return False
        return all(item == semantics[0] for item in semantics[1:])

    def semantic_differences(self) -> dict[str, Any]:
        """Return semantics of backends that differ from the first backend."""
        if not self:
            return {}
        reference_name, reference = next(iter(self.items()))
        reference_semantics = reference.semantics
        return {
            name: result.semantics
            for name, result in self.items()
            if name != reference_name and result.semantics != reference_semantics
        }

    def assert_semantically_consistent(self) -> None:
        """Raise :class:`SigmaBackendConsistencyError` if backend semantics differ."""
        if not self.is_consistent:
            raise SigmaBackendConsistencyError(
                "Conversion results are not semantically consistent across backends: "
                + ", ".join(self.semantic_differences())
            )

    assert_consistent = assert_semantically_consistent

    @property
    def semantically_consistent(self) -> bool:
        """Alias for :attr:`is_semantically_consistent`."""
        return self.is_semantically_consistent

    def to_dict(self, include_semantics: bool = False) -> dict[str, Any]:
        """Serialize all backend results into a JSON-compatible mapping."""
        return {
            "backends": {
                name: result.to_dict(include_semantics=include_semantics)
                for name, result in self.items()
            },
            "consistent": self.is_consistent,
            "metadata": self.metadata,
        }

    as_dict = to_dict

    def get_queries(self, backend: str) -> Any:
        """Return the generated queries of one backend."""
        return self[backend].queries

    def get_unsupported(self, backend: str) -> list[UnsupportedFeature]:
        """Return unsupported feature annotations of one backend."""
        return self[backend].unsupported

    def __eq__(self, other: object) -> bool:
        if isinstance(other, MultiBackendResult):
            return dict(self) == dict(other)
        if isinstance(other, Mapping):
            if set(self) != set(other):
                return False
            return all(result == other[name] for name, result in self.items())
        return NotImplemented

    def __str__(self) -> str:
        lines = []
        for name, result in self.items():
            lines.append(f"{name}: {result.query!r}")
            for feature in result.unsupported:
                lines.append(f"  [unsupported] {feature}")
        return "\n".join(lines)


class SemanticFingerprint:
    """Stable hash of a canonical semantic model."""

    def __init__(self, model: Any):
        self.model = model

    @property
    def digest(self) -> str:
        """SHA-256 digest of the canonical JSON representation."""
        serialized = json.dumps(self.model, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(serialized).hexdigest()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, SemanticFingerprint):
            return self.digest == other.digest
        if isinstance(other, str):
            return self.digest == other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.digest)

    def __str__(self) -> str:
        return self.digest


def merge_unsupported_features(
    features: Sequence[UnsupportedFeature],
) -> list[UnsupportedFeature]:
    """Deduplicate annotations while keeping the first occurrence and merging handling state."""
    merged: dict[tuple[Any, ...], UnsupportedFeature] = {}
    for feature in features:
        key = (
            feature.feature,
            feature.field_name,
            feature.value,
            feature.modifier,
            feature.severity,
            feature.source,
        )
        if key in merged:
            existing = merged[key]
            existing.handled = existing.handled or feature.handled
            existing.fallback = existing.fallback or feature.fallback
        else:
            merged[key] = feature
    return list(merged.values())


# Common alternative names used by integrations.
ConversionResult = BackendConversionResult
MultiBackendConversionResult = MultiBackendResult
UnsupportedItem = UnsupportedFeature
FeatureAnnotation = UnsupportedFeature
Unsupported = UnsupportedFeature
BackendOutput = BackendConversionResult
MultiBackendConversion = MultiBackendResult
