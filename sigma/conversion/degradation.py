from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from sigma.conversion.capabilities import BackendCapabilities, Features
from sigma.conversion.result import UnsupportedFeature
from sigma.processing.pipeline import ProcessingItem, ProcessingPipeline
from sigma.processing.transformations.base import DetectionItemTransformation
from sigma.rule import SigmaDetectionItem
from sigma.types import (
    SigmaCasedString,
    SigmaExpansion,
    SigmaNumber,
    SigmaRegularExpression,
    SigmaString,
    SigmaTimestampPart,
    SigmaType,
)

_SIMPLE_REGEX = re.compile(r"^[A-Za-z0-9 _/@:.,\-*?\\]*$")


def regex_to_sigma_string(regex: SigmaRegularExpression) -> SigmaString | None:
    """
    Best-effort conversion of a simple regular expression into a Sigma wildcard string.

    Only expressions without groups, alternations, character classes or anchors in the middle
    are converted. Complex expressions return ``None`` instead of producing a wrong query.
    """
    expression = regex.regexp.to_plain(regex=True)
    if expression.startswith("^"):
        expression = expression[1:]
    if expression.endswith("$") and not expression.endswith("\\$"):
        expression = expression[:-1]
    if not _SIMPLE_REGEX.fullmatch(expression):
        return None
    expression = expression.replace("\\.", ".")
    if "\\" in expression:
        return None
    expression = expression.replace(".*", "*").replace(".", "?")
    try:
        return SigmaString(expression)
    except Exception:
        return None


@dataclass
class SimpleDegradationTransformation(DetectionItemTransformation):
    """
    Replace features that a backend can't represent exactly with documented approximations.

    Every replacement is recorded in :attr:`degraded`; nothing is dropped silently. This
    transformation is only added when a caller explicitly asks for degradation.
    """

    backend_name: str = "backend"
    capabilities: BackendCapabilities = field(default_factory=BackendCapabilities)
    degraded: list[UnsupportedFeature] = field(default_factory=list)

    def apply_detection_item(self, detection_item: SigmaDetectionItem) -> SigmaDetectionItem | None:
        changed = False
        values: list[SigmaType] = []
        for value in detection_item.value:
            replacement = self._degrade_value(value, detection_item)
            if replacement is not None:
                values.append(replacement)
                changed = True
            else:
                values.append(value)
        if changed:
            detection_item.value = values
            return detection_item
        return None

    def _degrade_value(
        self, value: SigmaType, detection_item: SigmaDetectionItem
    ) -> SigmaType | None:
        if isinstance(value, SigmaExpansion):
            changed = False
            values: list[SigmaType] = []
            for expanded in value.values:
                replacement = self._degrade_value(expanded, detection_item)
                if replacement is not None:
                    values.append(replacement)
                    changed = True
                else:
                    values.append(expanded)
            if changed:
                return SigmaExpansion(values)
            return None

        if isinstance(value, SigmaRegularExpression) and self.capabilities.regex is False:
            replacement = regex_to_sigma_string(value)
            if replacement is not None:
                self._record(
                    Features.REGEX,
                    "regular expression approximated as wildcard string match",
                    detection_item,
                    str(value.regexp),
                    "regex_to_wildcard",
                )
                return replacement
            return None

        if isinstance(value, SigmaCasedString) and self.capabilities.case_sensitive is False:
            self._record(
                Features.CASE_SENSITIVE,
                "case-sensitive string matching approximated as case-insensitive match",
                detection_item,
                str(value),
                "case_insensitive",
            )
            return SigmaString(value.original)

        if isinstance(value, SigmaTimestampPart) and self.capabilities.timestamp is False:
            self._record(
                Features.TIMESTAMP,
                "timestamp part approximated as plain number",
                detection_item,
                value.timestamp_part.name,
                "numeric_timestamp_part",
            )
            return SigmaNumber(value.number)

        return None

    def _record(
        self,
        feature: str,
        reason: str,
        detection_item: SigmaDetectionItem,
        value: str,
        fallback: str,
    ) -> None:
        self.degraded.append(
            UnsupportedFeature(
                feature=feature,
                reason=reason,
                backend=self.backend_name,
                source=detection_item.source,
                severity="degraded",
                field_name=detection_item.field,
                value=value,
                handled=True,
                fallback=fallback,
            )
        )


def degradation_pipeline(
    backend_name: str,
    capabilities: BackendCapabilities,
) -> tuple[ProcessingPipeline, SimpleDegradationTransformation]:
    """Build a processing pipeline and transformation recording all degradations."""
    transformation = SimpleDegradationTransformation(
        backend_name=backend_name,
        capabilities=capabilities,
    )
    pipeline = ProcessingPipeline(
        name=f"{backend_name}_degradation",
        items=[
            ProcessingItem(
                transformation=transformation,
                identifier=f"{backend_name}_degradation",
            )
        ],
    )
    return pipeline, transformation
