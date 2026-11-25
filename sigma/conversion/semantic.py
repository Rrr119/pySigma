from __future__ import annotations

import json
from typing import Any, Callable, Iterable, Mapping

from sigma.conditions import (
    ConditionAND,
    ConditionFieldEqualsValueExpression,
    ConditionItem,
    ConditionNOT,
    ConditionOR,
    ConditionValueExpression,
)
from sigma.correlations import SigmaCorrelationRule
from sigma.rule import SigmaRule
from sigma.types import (
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
)

FieldMapper = Callable[[str], str]


def inverse_field_mapping(field_mapping: Mapping[str, str] | None) -> dict[str, str]:
    """
    Build an inverse mapping for normalizing backend semantics.

    If multiple source fields map to the same target field, the inverse is ambiguous and the
    target field is kept as-is.
    """
    if not field_mapping:
        return {}
    inverse: dict[str, str] = {}
    for source, target in field_mapping.items():
        # Keep the first source for a target. Processing pipeline tracking contains both
        # intermediate (fieldA -> mappedA) and final (mappedA -> target) mappings; the first
        # entry is the original Sigma field in that case. For genuinely conflicting independent
        # mappings the first one is a deterministic choice.
        inverse.setdefault(target, source)
    return inverse


def _canonical_number(value: SigmaNumber) -> Any:
    return value.number


def canonical_value(value: Any, field_mapper: FieldMapper | None = None) -> Any:
    """Convert a Sigma value into a backend-independent, JSON-compatible representation."""
    if isinstance(value, SigmaExpansion):
        return {
            "type": "expansion",
            "values": [canonical_value(item, field_mapper) for item in value.values],
        }
    if isinstance(value, SigmaCasedString):
        return {
            "type": "string",
            "value": value.to_plain(),
            "cased": True,
        }
    if isinstance(value, SigmaString):
        return {
            "type": "string",
            "value": value.to_plain(),
            "cased": False,
        }
    if isinstance(value, SigmaTimestampPart):
        return {
            "type": "timestamp",
            "part": value.timestamp_part.name,
            "value": _canonical_number(value),
        }
    if isinstance(value, SigmaBool):
        return {"type": "bool", "value": value.boolean}
    if isinstance(value, SigmaNumber):
        return {"type": "number", "value": _canonical_number(value)}
    if isinstance(value, SigmaNull):
        return {"type": "null"}
    if isinstance(value, SigmaExists):
        return {"type": "exists", "value": value.exists}
    if isinstance(value, SigmaRegularExpression):
        return {
            "type": "regex",
            "value": str(value.regexp),
            "flags": sorted(flag.name for flag in value.flags),
        }
    if isinstance(value, SigmaCIDRExpression):
        return {"type": "cidr", "value": str(value.network)}
    if isinstance(value, SigmaCompareExpression):
        return {
            "type": "compare",
            "op": value.op.name,
            "value": _canonical_number(value.number),
        }
    if isinstance(value, SigmaFieldReference):
        field_name = value.field
        if field_mapper is not None:
            field_name = field_mapper(field_name)
        return {
            "type": "fieldref",
            "field": field_name,
            "startswith": value.starts_with,
            "endswith": value.ends_with,
        }
    if isinstance(value, SigmaQueryExpression):
        return {"type": "query", "id": value.id, "expression": value.expr}
    try:
        plain = value.to_plain()
    except Exception:
        plain = None
    return {"type": type(value).__name__, "value": plain}


def canonical_condition(
    condition: (
        ConditionItem | ConditionFieldEqualsValueExpression | ConditionValueExpression | None
    ),
    field_mapper: FieldMapper | None = None,
) -> Any:
    """Convert a parsed condition tree into a canonical nested mapping."""
    if condition is None:
        return None
    if isinstance(condition, ConditionNOT):
        return {
            "op": "not",
            "args": [canonical_condition(condition.args[0], field_mapper)],
        }
    if isinstance(condition, ConditionAND):
        return {
            "op": "and",
            "args": [canonical_condition(arg, field_mapper) for arg in condition.args],
        }
    if isinstance(condition, ConditionOR):
        return {
            "op": "or",
            "args": [canonical_condition(arg, field_mapper) for arg in condition.args],
        }
    if isinstance(condition, ConditionFieldEqualsValueExpression):
        field_name = condition.field
        if field_mapper is not None:
            field_name = field_mapper(field_name)
        return {
            "field": field_name,
            "value": canonical_value(condition.value, field_mapper),
        }
    if isinstance(condition, ConditionValueExpression):
        return {"value": canonical_value(condition.value, field_mapper)}
    return {"type": type(condition).__name__, "repr": repr(condition)}


def semantic_model(
    rule: SigmaRule | SigmaCorrelationRule,
    field_mapper: FieldMapper | None = None,
) -> dict[str, Any]:
    """
    Build a canonical semantic model of a rule.

    The model only contains detection semantics, not backend-specific syntax. It can therefore be
    compared across backends; field names are normalized with ``field_mapper`` if provided.
    """
    if isinstance(rule, SigmaRule):
        return {
            "title": rule.title,
            "id": str(rule.id) if rule.id is not None else None,
            "logsource": {
                "category": rule.logsource.category,
                "product": rule.logsource.product,
                "service": rule.logsource.service,
            },
            "conditions": [
                canonical_condition(condition.parsed, field_mapper)
                for condition in rule.detection.parsed_condition
            ],
        }
    if isinstance(rule, SigmaCorrelationRule):
        return {
            "title": rule.title,
            "id": str(rule.id) if rule.id is not None else None,
            "correlation": {
                "type": str(rule.type),
                "generate": rule.generate,
                "group_by": list(rule.group_by or []),
                "timespan": rule.timespan.spec,
                "rules": sorted(reference.reference for reference in (rule.rules or [])),
            },
        }
    return {"type": type(rule).__name__}


def semantic_models(
    rules: Iterable[SigmaRule | SigmaCorrelationRule],
    field_mapper: FieldMapper | None = None,
) -> list[dict[str, Any]]:
    """Build canonical semantic models for multiple rules."""
    return [semantic_model(rule, field_mapper) for rule in rules]


def semantic_digest(model: Any) -> str:
    """Return a stable SHA-256 digest for a semantic model."""
    import hashlib

    serialized = json.dumps(model, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def semantically_equivalent(first: Any, second: Any) -> bool:
    """Compare two semantic models."""
    return semantic_digest(first) == semantic_digest(second)
