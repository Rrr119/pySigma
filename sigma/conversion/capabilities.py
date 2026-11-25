from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Iterable, Mapping

from sigma.conversion.result import UnsupportedFeature
from sigma.modifiers import reverse_modifier_mapping
from sigma.rule import SigmaDetection, SigmaDetectionItem, SigmaRule
from sigma.types import (
    CompareOperators,
    SigmaBool,
    SigmaCasedString,
    SigmaCIDRExpression,
    SigmaCompareExpression,
    SigmaExists,
    SigmaExpansion,
    SigmaFieldReference,
    SigmaNull,
    SigmaNumber,
    SigmaQueryExpression,
    SigmaRegularExpression,
    SigmaString,
    SigmaTimestampPart,
    SpecialChars,
)

if TYPE_CHECKING:
    from sigma.conversion.base import Backend
    from sigma.correlations import SigmaCorrelationRule


class Features:
    """Identifiers used in :class:`UnsupportedFeature` annotations."""

    REGEX = "regex"
    REGEX_FLAG = "regex_flag"
    CIDR = "cidr"
    COMPARE = "compare"
    FIELD_REFERENCE = "fieldref"
    NULL = "null"
    EXISTS = "exists"
    CASE_SENSITIVE = "case_sensitive"
    TIMESTAMP = "timestamp"
    BOOL = "bool"
    WILDCARD = "wildcard"
    UNBOUND_VALUE = "unbound_value"
    QUERY_EXPRESSION = "query_expression"
    EXPANSION = "expansion"
    MODIFIER = "modifier"
    FILTER_CHARS = "filter_chars"
    DETECTION_ITEM = "detection_item"
    CONDITION = "condition"
    DEFERRED_EXPRESSION = "deferred_expression"
    CONVERSION = "conversion"


@dataclass(frozen=True)
class BackendCapabilities:
    """
    Declarative description of the features a backend can represent.

    ``None`` means *unknown*. Unknown capabilities are never reported as unsupported to avoid
    false positives for third-party backends that do not implement this model. Backends can
    override the automatically detected values by setting a ``capabilities`` class attribute.
    """

    regex: bool | None = None
    regex_flags: frozenset[str] | None = None
    cidr: bool | None = None
    cidr_native: bool | None = None
    compare: bool | None = None
    field_reference: bool | None = None
    field_reference_modes: frozenset[str] | None = None
    null: bool | None = None
    exists: bool | None = None
    case_sensitive: bool | None = None
    timestamp: bool | None = None
    bool_values: frozenset[bool] | None = None
    wildcard_multi: bool | None = None
    wildcard_single: bool | None = None
    unbound_string: bool | None = None
    unbound_number: bool | None = None
    unbound_regex: bool | None = None
    query_expression: bool | None = None
    filter_chars: frozenset[str] = field(default_factory=frozenset)
    supported_modifiers: frozenset[str] | None = None

    def merged(self, override: BackendCapabilities | None) -> BackendCapabilities:
        """Return capabilities with non-``None`` values from ``override`` taking precedence."""
        if override is None:
            return self
        values: dict[str, Any] = {}
        for field_name in self.__dataclass_fields__:
            override_value = getattr(override, field_name)
            values[field_name] = (
                override_value if override_value is not None else getattr(self, field_name)
            )
        # filter_chars is a set and can't distinguish "not set" from "empty"; treat a non-empty
        # override as an addition to the detected filter characters.
        values["filter_chars"] = self.filter_chars | override.filter_chars
        return replace(self, **values)

    def supports(self, feature: str) -> bool:
        """Return whether ``feature`` is supported. Unknown capabilities are treated as supported."""
        attribute_map: dict[str, str] = {
            Features.REGEX: "regex",
            Features.CIDR: "cidr",
            Features.COMPARE: "compare",
            Features.FIELD_REFERENCE: "field_reference",
            Features.NULL: "null",
            Features.EXISTS: "exists",
            Features.CASE_SENSITIVE: "case_sensitive",
            Features.TIMESTAMP: "timestamp",
            Features.WILDCARD: "wildcard_multi",
            Features.QUERY_EXPRESSION: "query_expression",
        }
        attribute = attribute_map.get(feature)
        if attribute is None:
            return True
        value = getattr(self, attribute)
        return value is not False

    @classmethod
    def from_backend(cls, backend: Backend) -> BackendCapabilities:
        """Inspect a backend and build its effective capability description."""
        detected = cls._detect_text_backend(backend)
        explicit = getattr(backend, "capabilities", None)
        if isinstance(explicit, BackendCapabilities):
            return detected.merged(explicit)
        if isinstance(explicit, Mapping):
            known = {
                key: value for key, value in explicit.items() if key in cls.__dataclass_fields__
            }
            return detected.merged(cls(**known))
        return detected

    @classmethod
    def _detect_text_backend(cls, backend: Backend) -> BackendCapabilities:
        """
        Derive capabilities from the class variables of :class:`TextQueryBackend`.

        The check is attribute based instead of ``isinstance``-based because backends may wrap or
        proxy the text backend. A generic :class:`Backend` that doesn't expose these attributes
        yields all-unknown capabilities.
        """
        if not hasattr(backend, "re_expression") and not hasattr(
            backend, "field_equals_field_expression"
        ):
            return cls()

        def is_template(name: str) -> bool:
            return isinstance(getattr(backend, name, None), str)

        regex_flags_value = getattr(backend, "re_flags", None)
        regex_flags = (
            frozenset(flag.name for flag in regex_flags_value)
            if isinstance(regex_flags_value, Mapping)
            else None
        )
        field_reference_modes = frozenset(
            mode
            for mode, attribute in (
                ("equals", "field_equals_field_expression"),
                ("startswith", "field_equals_field_startswith_expression"),
                ("endswith", "field_equals_field_endswith_expression"),
                ("contains", "field_equals_field_contains_expression"),
            )
            if is_template(attribute)
        )
        bool_values_value = getattr(backend, "bool_values", None)
        bool_values = (
            frozenset(value for value in (True, False) if bool_values_value.get(value) is not None)
            if isinstance(bool_values_value, Mapping)
            else None
        )
        filter_chars = getattr(backend, "filter_chars", "") or ""
        return cls(
            regex=is_template("re_expression"),
            regex_flags=regex_flags,
            # CIDR is always representable by pySigma, but without a native template it is
            # expanded into string wildcard matches which is a semantic degradation.
            cidr=True,
            cidr_native=is_template("cidr_expression"),
            compare=is_template("compare_op_expression")
            and isinstance(getattr(backend, "compare_operators", None), Mapping),
            field_reference=bool(field_reference_modes),
            field_reference_modes=field_reference_modes,
            null=is_template("field_null_expression"),
            exists=is_template("field_exists_expression"),
            case_sensitive=is_template("case_sensitive_match_expression")
            or is_template("case_sensitive_startswith_expression")
            or is_template("case_sensitive_endswith_expression")
            or is_template("case_sensitive_contains_expression"),
            timestamp=is_template("field_timestamp_part_expression")
            and isinstance(getattr(backend, "timestamp_part_mapping", None), Mapping),
            bool_values=bool_values,
            wildcard_multi=getattr(backend, "wildcard_multi", None) is not None,
            wildcard_single=getattr(backend, "wildcard_single", None) is not None,
            unbound_string=is_template("unbound_value_str_expression"),
            unbound_number=is_template("unbound_value_num_expression"),
            unbound_regex=is_template("unbound_value_re_expression"),
            query_expression=True,
            filter_chars=frozenset(filter_chars),
        )


def _mapped_field(field_name: str | None, field_mapping: Mapping[str, str] | None) -> str | None:
    if field_name is None or field_mapping is None:
        return field_name
    return field_mapping.get(field_name, field_name)


def feature_for_value(value: Any) -> str:
    """Return the ``Features`` identifier for a Sigma value object."""
    if isinstance(value, SigmaRegularExpression):
        return Features.REGEX
    if isinstance(value, SigmaCIDRExpression):
        return Features.CIDR
    if isinstance(value, SigmaCompareExpression):
        return Features.COMPARE
    if isinstance(value, SigmaFieldReference):
        return Features.FIELD_REFERENCE
    if isinstance(value, SigmaNull):
        return Features.NULL
    if isinstance(value, SigmaExists):
        return Features.EXISTS
    if isinstance(value, SigmaTimestampPart):
        return Features.TIMESTAMP
    if isinstance(value, SigmaBool):
        return Features.BOOL
    if isinstance(value, SigmaCasedString):
        return Features.CASE_SENSITIVE
    if isinstance(value, SigmaString):
        return Features.WILDCARD if value.contains_special() else Features.CONVERSION
    if isinstance(value, SigmaExpansion):
        return Features.EXPANSION
    if isinstance(value, SigmaQueryExpression):
        return Features.QUERY_EXPRESSION
    return Features.CONVERSION


def _iter_detection_items(
    detection: SigmaDetection | SigmaDetectionItem,
) -> Iterable[SigmaDetectionItem]:
    if isinstance(detection, SigmaDetectionItem):
        yield detection
    else:
        for item in detection.detection_items:
            yield from _iter_detection_items(item)


def _iter_rule_detection_items(rule: SigmaRule) -> Iterable[SigmaDetectionItem]:
    for detection in rule.detection.detections.values():
        yield from _iter_detection_items(detection)


def _string_parts(value: SigmaString) -> Iterable[str | SpecialChars | Any]:
    return value.s


def detect_unsupported_features(
    rule: SigmaRule | SigmaCorrelationRule,
    capabilities: BackendCapabilities,
    backend: str | None = None,
    field_mapping: Mapping[str, str] | None = None,
) -> list[UnsupportedFeature]:
    """
    Inspect a (processed) rule and report features that the backend can't represent exactly.

    The scan is deliberately based on the detection values after processing. This means field
    mappings and other pipeline transformations are reflected in the report and a feature that
    was removed by a pipeline is not reported.
    """
    if not isinstance(rule, SigmaRule):
        return []

    features: list[UnsupportedFeature] = []
    seen: set[tuple[Any, ...]] = set()

    def add(
        feature: str,
        reason: str,
        severity: str = "error",
        field_name: str | None = None,
        value: str | None = None,
        modifier: str | None = None,
        context: dict[str, Any] | None = None,
    ) -> None:
        mapped = _mapped_field(field_name, field_mapping)
        key = (feature, mapped, value, modifier, severity)
        if key in seen:
            return
        seen.add(key)
        features.append(
            UnsupportedFeature(
                feature=feature,
                reason=reason,
                backend=backend,
                source=rule.source,
                rule=rule,
                severity=severity,
                field_name=mapped,
                value=value,
                modifier=modifier,
                context=context or {},
            )
        )

    for item in _iter_rule_detection_items(rule):
        field_name = item.field
        if capabilities.supported_modifiers is not None:
            for modifier_class in item.modifiers:
                modifier_id = reverse_modifier_mapping.get(
                    modifier_class.__name__, modifier_class.__name__
                )
                if modifier_id not in capabilities.supported_modifiers:
                    add(
                        Features.MODIFIER,
                        f"modifier '{modifier_id}' is not supported",
                        field_name=field_name,
                        modifier=modifier_id,
                    )
        for value in item.value:
            _scan_value(value, item, capabilities, add)

    return features


def _scan_value(
    value: Any,
    item: SigmaDetectionItem,
    capabilities: BackendCapabilities,
    add: Any,
) -> None:
    field_name = item.field
    modifier_ids = [
        reverse_modifier_mapping.get(modifier_class.__name__, modifier_class.__name__)
        for modifier_class in item.modifiers
    ]
    modifier = ",".join(modifier_ids) or None

    if isinstance(value, SigmaExpansion):
        for expanded_value in value.values:
            _scan_value(expanded_value, item, capabilities, add)
        return

    if isinstance(value, SigmaRegularExpression):
        if capabilities.regex is False:
            add(
                Features.REGEX,
                "regular expressions are not supported by the backend",
                field_name=field_name,
                value=str(value.regexp),
                modifier=modifier,
            )
        if capabilities.regex_flags is not None:
            for flag in value.flags:
                if flag.name not in capabilities.regex_flags:
                    add(
                        Features.REGEX_FLAG,
                        f"regular expression flag '{flag.name}' is not supported",
                        field_name=field_name,
                        value=str(value.regexp),
                        modifier=modifier,
                    )
        if field_name is None and capabilities.unbound_regex is False:
            add(
                Features.UNBOUND_VALUE,
                "unbound regular expressions are not supported by the backend",
                value=str(value.regexp),
            )
        return

    if isinstance(value, SigmaCIDRExpression):
        if capabilities.cidr is False:
            add(
                Features.CIDR,
                "CIDR expressions are not supported by the backend",
                field_name=field_name,
                value=str(value.network),
                modifier=modifier,
            )
        elif capabilities.cidr_native is False:
            add(
                Features.CIDR,
                "CIDR expressions are degraded to wildcard-expanded string matches",
                severity="degraded",
                field_name=field_name,
                value=str(value.network),
                modifier=modifier,
            )
        return

    if isinstance(value, SigmaCompareExpression):
        if capabilities.compare is False:
            add(
                Features.COMPARE,
                "numeric comparison operators are not supported by the backend",
                field_name=field_name,
                value=f"{value.op.name} {value.number}",
                modifier=modifier,
            )
        return

    if isinstance(value, SigmaFieldReference):
        if capabilities.field_reference is False:
            add(
                Features.FIELD_REFERENCE,
                "field references are not supported by the backend",
                field_name=field_name,
                value=value.field,
                modifier=modifier,
            )
        else:
            modes = capabilities.field_reference_modes
            if modes is not None:
                for mode, enabled in (
                    ("startswith", value.starts_with),
                    ("endswith", value.ends_with),
                ):
                    if enabled and mode not in modes:
                        add(
                            Features.FIELD_REFERENCE,
                            f"field reference mode '{mode}' is not supported",
                            severity="degraded",
                            field_name=field_name,
                            value=value.field,
                            modifier=modifier,
                        )
        return

    if isinstance(value, SigmaNull):
        if capabilities.null is False:
            add(
                Features.NULL,
                "null value checks are not supported by the backend",
                field_name=field_name,
            )
        return

    if isinstance(value, SigmaExists):
        if capabilities.exists is False:
            add(
                Features.EXISTS,
                "field existence checks are not supported by the backend",
                field_name=field_name,
                value=str(value.exists),
            )
        return

    if isinstance(value, SigmaTimestampPart):
        if capabilities.timestamp is False:
            add(
                Features.TIMESTAMP,
                "timestamp part expressions are not supported by the backend",
                field_name=field_name,
                value=value.timestamp_part.name,
                modifier=modifier,
            )
        return

    if isinstance(value, SigmaBool):
        if capabilities.bool_values is not None and value.boolean not in capabilities.bool_values:
            add(
                Features.BOOL,
                f"boolean value {value.boolean} is not supported by the backend",
                field_name=field_name,
                value=str(value.boolean),
                modifier=modifier,
            )
        return

    if isinstance(value, SigmaQueryExpression):
        if capabilities.query_expression is False:
            add(
                Features.QUERY_EXPRESSION,
                "query expressions are not supported by the backend",
                field_name=field_name,
                value=value.id,
                modifier=modifier,
            )
        return

    if isinstance(value, SigmaCasedString):
        if capabilities.case_sensitive is False:
            add(
                Features.CASE_SENSITIVE,
                "case-sensitive string matching is not supported by the backend",
                severity="degraded",
                field_name=field_name,
                value=str(value),
                modifier=modifier,
            )
        _scan_string_value(value, item, capabilities, add, modifier)
        return

    if isinstance(value, SigmaString):
        _scan_string_value(value, item, capabilities, add, modifier)
        return

    if isinstance(value, SigmaNumber):
        if field_name is None and capabilities.unbound_number is False:
            add(
                Features.UNBOUND_VALUE,
                "unbound numeric values are not supported by the backend",
                value=str(value),
            )
        return


def _scan_string_value(
    value: SigmaString,
    item: SigmaDetectionItem,
    capabilities: BackendCapabilities,
    add: Any,
    modifier: str | None,
) -> None:
    field_name = item.field
    if field_name is None and capabilities.unbound_string is False:
        add(
            Features.UNBOUND_VALUE,
            "unbound string values are not supported by the backend",
            value=str(value),
        )

    parts = list(_string_parts(value))
    if capabilities.wildcard_multi is False and SpecialChars.WILDCARD_MULTI in parts:
        add(
            Features.WILDCARD,
            "multi-character wildcards are not supported by the backend",
            field_name=field_name,
            value=str(value),
            modifier=modifier,
        )
    if capabilities.wildcard_single is False and SpecialChars.WILDCARD_SINGLE in parts:
        add(
            Features.WILDCARD,
            "single-character wildcards are not supported by the backend",
            field_name=field_name,
            value=str(value),
            modifier=modifier,
        )

    if capabilities.filter_chars:
        filtered = sorted(
            {
                char
                for part in parts
                if isinstance(part, str)
                for char in part
                if char in capabilities.filter_chars
            }
        )
        if filtered:
            add(
                Features.FILTER_CHARS,
                "characters "
                + ", ".join(repr(char) for char in filtered)
                + " are silently filtered by the backend",
                severity="degraded",
                field_name=field_name,
                value=str(value),
                modifier=modifier,
            )


def describe_unsupported_exception(
    exception: BaseException,
    backend: str | None = None,
    rule: SigmaRule | SigmaCorrelationRule | None = None,
) -> UnsupportedFeature:
    """Convert a conversion exception into an explicit unsupported feature annotation."""
    return UnsupportedFeature.from_exception(exception, backend=backend, rule=rule)


# Common alternative names.
BackendCapability = BackendCapabilities
Capabilities = BackendCapabilities
UnsupportedFeatureDetector = detect_unsupported_features
Feature = Features
UnsupportedFeatures = Features
