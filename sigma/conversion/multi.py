from __future__ import annotations

import copy
import re
import time
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, Callable, Sequence, cast

from sigma.collection import SigmaCollection
from sigma.conversion.base import Backend
from sigma.conversion.capabilities import BackendCapabilities, detect_unsupported_features
from sigma.conversion.degradation import SimpleDegradationTransformation, degradation_pipeline
from sigma.conversion.result import (
    BackendConversionResult,
    MultiBackendResult,
    UnsupportedFeature,
    merge_unsupported_features,
)
from sigma.conversion.semantic import (
    inverse_field_mapping,
    semantic_models,
)
from sigma.correlations import SigmaCorrelationRule
from sigma.exceptions import (
    SigmaBackendError,
    SigmaError,
    SigmaUnsupportedFeatureError,
)
from sigma.processing.pipeline import ProcessingItem, ProcessingPipeline
from sigma.processing.transformations import FieldMappingTransformation
from sigma.rule import SigmaRule

BackendFactory = type[Backend] | Callable[[], Backend]
BackendReference = Backend | BackendFactory
UnsupportedHandler = Callable[[UnsupportedFeature, Any, Backend], Any]


class UnsupportedPolicy(str, Enum):
    """Policies for handling features a backend can't represent exactly."""

    ANNOTATE = "annotate"
    DEGRADE = "degrade"
    FALLBACK = "fallback"
    WARN = "warn"
    IGNORE = "ignore"
    DROP = "drop"
    RAISE = "raise"
    ERROR = "error"


UnsupportedPolicyType = (
    str | UnsupportedPolicy | UnsupportedHandler | Mapping[str, UnsupportedHandler]
)


def _snake_case(value: str) -> str:
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    value = re.sub(r"[^A-Za-z0-9]+", "_", value)
    return value.strip("_").lower()


def _default_backend_name(backend: BackendReference | BackendSpec) -> str:
    if isinstance(backend, BackendSpec):
        if backend.name:
            return backend.name
        backend = backend.backend
    identifier = getattr(backend, "identifier", None)
    if not identifier:
        identifier = getattr(backend, "name", None)
    if not identifier:
        identifier = backend.__name__ if isinstance(backend, type) else type(backend).__name__
    identifier = str(identifier)
    for suffix in ("Backend", "backend"):
        if identifier.endswith(suffix):
            identifier = identifier[: -len(suffix)]
            break
    return _snake_case(identifier) or "backend"


def field_mapping_pipeline(
    mapping: Mapping[str, str],
    name: str | None = None,
) -> ProcessingPipeline:
    """Build a regular processing pipeline performing a field name mapping."""
    processing_mapping: dict[str | None, str | list[str]] = {
        source: target for source, target in mapping.items()
    }
    return ProcessingPipeline(
        name=name or "field_mapping",
        items=[
            ProcessingItem(
                transformation=FieldMappingTransformation(processing_mapping),
                identifier=name or "field_mapping",
            )
        ],
    )


@dataclass
class FieldMapper:
    """Small helper for backend-specific field name mappings."""

    mapping: dict[str, str] = field(default_factory=dict)
    name: str | None = None

    def map(self, field: str) -> str:
        """Map a field name, leaving unmapped names unchanged."""
        return self.mapping.get(field, field)

    def pipeline(self) -> ProcessingPipeline:
        """Build a processing pipeline performing this mapping."""
        return field_mapping_pipeline(self.mapping, name=self.name)


FieldMapping = FieldMapper


def _first_mapping(
    mappings: Sequence[Mapping[str, str] | None],
) -> dict[str, str] | None:
    result: dict[str, str] = {}
    for mapping in mappings:
        if mapping:
            result.update(mapping)
    return result or None


@dataclass
class BackendSpec:
    """
    Configuration of one backend inside a :class:`UnifiedBackend`.

    Besides the backend itself it can carry a backend-specific processing pipeline, an additional
    field name mapping and the output format to use. ``backend`` may be an instance, a backend
    class or a factory callable.
    """

    name: str
    backend: BackendReference
    field_mapping: dict[str, str] | None = None
    pipeline: ProcessingPipeline | None = None
    output_format: str | None = None
    correlation_method: str | None = None
    enabled: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    def create_backend(self) -> Backend:
        """Return the backend instance, instantiating classes and factories on demand."""
        if isinstance(self.backend, Backend):
            return self.backend
        if isinstance(self.backend, type) and issubclass(self.backend, Backend):
            return self.backend()
        if callable(self.backend):
            instance = self.backend()
            if not isinstance(instance, Backend):
                raise SigmaBackendError(
                    f"Backend factory for '{self.name}' did not return a Backend instance"
                )
            return instance
        raise SigmaBackendError(f"Invalid backend reference for '{self.name}'")

    def effective_field_mapping(
        self, global_mapping: Mapping[str, str] | None = None
    ) -> dict[str, str] | None:
        """Combine the global field mapping with the backend-specific mapping."""
        return _first_mapping((global_mapping, self.field_mapping))


class UnifiedBackend:
    """
    Convert Sigma rules with multiple backends through one interface.

    Features:

    * a single ``convert()`` call returns per-backend queries,
    * per-backend processing pipelines and field mappings,
    * explicit reporting of unsupported or degraded features instead of silent drops,
    * a normalized semantic model that can be compared across backends.

    The class is intentionally independent from the rule model. Rules are deep-copied for every
    backend so that pipelines and field mappings of one backend can't influence another backend.
    """

    def __init__(
        self,
        backends: (
            Mapping[str, BackendReference]
            | Iterable[BackendReference | BackendSpec | tuple[str, BackendReference]]
            | BackendReference
            | BackendSpec
            | None
        ) = None,
        *additional_backends: BackendReference | BackendSpec | tuple[str, BackendReference],
        field_mapping: Mapping[str, str] | None = None,
        field_mappings: Mapping[str, Mapping[str, str]] | None = None,
        pipeline: ProcessingPipeline | None = None,
        pipelines: Mapping[str, ProcessingPipeline] | None = None,
        output_formats: Mapping[str, str] | None = None,
        on_unsupported: UnsupportedPolicyType = UnsupportedPolicy.ANNOTATE,
        unsupported_policy: UnsupportedPolicyType | None = None,
        fallback: UnsupportedPolicyType | None = None,
        collect_errors: bool = True,
        clone_rules: bool = True,
        degrade: bool = True,
        raise_on_error: bool = False,
        **options: Any,
    ):
        self.global_field_mapping = dict(field_mapping or {})
        self.field_mappings = {
            name: dict(mapping) for name, mapping in (field_mappings or {}).items()
        }
        self.global_pipeline = pipeline
        self.pipelines = dict(pipelines or {})
        self.output_formats = dict(output_formats or {})
        self.on_unsupported: UnsupportedPolicyType = (
            unsupported_policy
            if unsupported_policy is not None
            else fallback if fallback is not None else on_unsupported
        )
        self.collect_errors = collect_errors
        self.clone_rules = clone_rules
        self.degrade = degrade
        self.raise_on_error = raise_on_error
        self.options = options
        self.specs: list[BackendSpec] = []

        if backends is not None:
            self._register(backends)
        for backend in additional_backends:
            self._register(backend)

    def _unique_name(self, name: str) -> str:
        if not any(spec.name == name for spec in self.specs):
            return name
        index = 2
        while any(spec.name == f"{name}_{index}" for spec in self.specs):
            index += 1
        return f"{name}_{index}"

    def _make_spec(
        self,
        backend: BackendReference | BackendSpec,
        name: str | None = None,
    ) -> BackendSpec:
        if isinstance(backend, BackendSpec):
            spec = replace(backend)
        else:
            spec = BackendSpec(name=name or "", backend=backend)
        if name:
            spec.name = name
        if not spec.name:
            spec.name = _default_backend_name(spec)
            spec.name = self._unique_name(spec.name)
        elif any(existing.name == spec.name for existing in self.specs):
            spec.name = self._unique_name(spec.name)

        def lookup(mapping: Mapping[str, Any]) -> Any:
            for key in (
                spec.name,
                getattr(spec.backend, "identifier", None),
                getattr(spec.backend, "name", None),
            ):
                if key is not None and key in mapping:
                    return mapping[key]
            return None

        if spec.field_mapping is None:
            spec.field_mapping = lookup(self.field_mappings)
        if spec.pipeline is None:
            spec.pipeline = lookup(self.pipelines)
        if spec.output_format is None:
            spec.output_format = lookup(self.output_formats)
        return spec

    def _register(
        self,
        entry: (
            Mapping[str, BackendReference]
            | Iterable[BackendReference | BackendSpec | tuple[str, BackendReference]]
            | BackendReference
            | BackendSpec
            | tuple[str, BackendReference]
        ),
    ) -> None:
        if isinstance(entry, BackendSpec):
            self._add_spec(entry)
            return
        if isinstance(entry, Backend):
            self._add_spec(self._make_spec(entry))
            return
        if isinstance(entry, type) and issubclass(entry, Backend):
            self._add_spec(self._make_spec(entry))
            return
        if callable(entry) and not isinstance(entry, (str, bytes, Mapping)):
            self._add_spec(self._make_spec(cast(BackendFactory, entry)))
            return
        if isinstance(entry, Mapping):
            for name, backend in entry.items():
                self._add_spec(self._make_spec(backend, name=str(name)))
            return
        if isinstance(entry, tuple) and len(entry) == 2 and isinstance(entry[0], str):
            self._add_spec(self._make_spec(entry[1], name=entry[0]))
            return
        if isinstance(entry, Iterable):
            for item in entry:
                self._register(cast(Any, item))
            return
        raise SigmaBackendError(f"Unsupported backend specification: {entry!r}")

    def _add_spec(self, spec: BackendSpec) -> BackendSpec:
        if any(existing.name == spec.name for existing in self.specs):
            raise SigmaBackendError(f"Backend '{spec.name}' is already registered")
        self.specs.append(spec)
        return spec

    def add_backend(
        self,
        backend: BackendReference | BackendSpec | tuple[str, BackendReference],
        name: str | None = None,
        *,
        field_mapping: Mapping[str, str] | None = None,
        pipeline: ProcessingPipeline | None = None,
        output_format: str | None = None,
    ) -> BackendSpec:
        """Register another backend and return its specification."""
        if isinstance(backend, tuple):
            if name is None:
                name = backend[0]
            backend = backend[1]
        spec = self._make_spec(backend, name=name)
        if field_mapping is not None:
            spec.field_mapping = dict(field_mapping)
        if pipeline is not None:
            spec.pipeline = pipeline
        if output_format is not None:
            spec.output_format = output_format
        return self._add_spec(spec)

    register_backend = add_backend

    def remove_backend(self, name: str) -> None:
        """Remove a registered backend by name."""
        self.specs = [spec for spec in self.specs if spec.name != name]

    def clear(self) -> None:
        """Remove all registered backends."""
        self.specs.clear()

    def get_backend(self, name: str) -> Backend:
        """Return the backend instance registered under ``name``."""
        for spec in self.specs:
            if spec.name == name:
                return spec.create_backend()
        raise KeyError(name)

    @property
    def backends(self) -> dict[str, Backend]:
        """Registered backend instances by name."""
        return {spec.name: spec.create_backend() for spec in self.specs}

    @property
    def backend_names(self) -> list[str]:
        """Names of all registered backends."""
        return [spec.name for spec in self.specs]

    def __getitem__(self, name: str) -> Backend:
        return self.get_backend(name)

    def __iter__(self) -> Iterator[str]:
        return (spec.name for spec in self.specs)

    def __len__(self) -> int:
        return len(self.specs)

    def __contains__(self, name: object) -> bool:
        return any(spec.name == name for spec in self.specs)

    def __call__(self, rules: Any, **kwargs: Any) -> MultiBackendResult:
        """Alias for :meth:`convert`."""
        return self.convert(rules, **kwargs)

    def _coerce_collection(
        self,
        rules: (
            SigmaCollection
            | SigmaRule
            | SigmaCorrelationRule
            | Mapping[str, Any]
            | str
            | Sequence[SigmaRule | SigmaCorrelationRule | Mapping[str, Any]]
        ),
    ) -> SigmaCollection:
        if isinstance(rules, SigmaCollection):
            return rules
        if isinstance(rules, (SigmaRule, SigmaCorrelationRule)):
            return SigmaCollection([rules])
        if isinstance(rules, str):
            return SigmaCollection.from_yaml(rules)
        if isinstance(rules, Mapping):
            return SigmaCollection.from_dicts([dict(rules)])
        if isinstance(rules, Sequence):
            sequence_rules = rules
            if not sequence_rules:
                return SigmaCollection([])
            if all(isinstance(rule, Mapping) for rule in sequence_rules):
                mapping_rules = cast(Sequence[Mapping[str, Any]], sequence_rules)
                return SigmaCollection.from_dicts([dict(rule) for rule in mapping_rules])
            return SigmaCollection(cast(Any, list(sequence_rules)))
        return SigmaCollection(cast(Any, list(cast(Iterable[Any], rules))))

    def _clone_collection(self, collection: SigmaCollection) -> SigmaCollection:
        if not self.clone_rules:
            return collection
        try:
            return copy.deepcopy(collection)
        except Exception:
            # Some custom policy or source objects are not deep-copyable. Copy the rules
            # individually as a best effort so that pipelines still don't leak between backends.
            return SigmaCollection(
                [copy.deepcopy(rule) for rule in collection.rules],
                collect_filters=True,
            )

    def _compose_pipeline(self, spec: BackendSpec, backend: Backend) -> ProcessingPipeline | None:
        pipelines: list[ProcessingPipeline | None] = [
            backend.processing_pipeline,
            self.global_pipeline,
            spec.pipeline,
        ]
        mapping = spec.effective_field_mapping(self.global_field_mapping)
        if mapping:
            pipelines.append(field_mapping_pipeline(mapping, name=f"{spec.name}_field_mapping"))
        combined: ProcessingPipeline | None = None
        for pipeline in pipelines:
            if pipeline is None:
                continue
            combined = pipeline if combined is None else combined + pipeline
        if combined is not None and combined.name is None:
            combined.name = f"{spec.name}_pipeline"
        return combined

    def _source_semantics(self, collection: SigmaCollection) -> Any:
        try:
            return semantic_models(collection.rules)
        except Exception:
            return None

    def _pipeline_field_mapping(self, backend: Backend) -> dict[str, str]:
        """Build a source-to-target field mapping from the applied pipeline tracking."""
        try:
            tracking = backend.last_processing_pipeline.field_mappings
        except Exception:
            return {}
        mapping: dict[str, str] = {}
        for source, targets in tracking.items():
            if source is None or len(targets) != 1:
                continue
            mapping[source] = next(iter(targets))
        return mapping

    def _processed_semantics(
        self,
        collection: SigmaCollection,
        field_mapping: Mapping[str, str] | None,
        backend: Backend,
    ) -> tuple[Any, Any]:
        backend_semantics: Any
        try:
            backend_semantics = semantic_models(collection.rules)
        except Exception:
            backend_semantics = None
        normalized: Any
        try:
            effective_mapping = _first_mapping(
                (self._pipeline_field_mapping(backend), field_mapping)
            )
            inverse = inverse_field_mapping(effective_mapping)
            normalized = semantic_models(
                collection.rules,
                field_mapper=lambda field: inverse.get(field, field),
            )
        except Exception:
            normalized = backend_semantics
        return backend_semantics, normalized

    def _collect_state_unsupported(
        self, collection: SigmaCollection, backend: Backend
    ) -> list[UnsupportedFeature]:
        features: list[UnsupportedFeature] = []
        for rule in collection.rules:
            try:
                states = rule.get_conversion_states()
            except SigmaError:
                continue
            for state in states:
                for feature in state.unsupported:
                    if isinstance(feature, UnsupportedFeature):
                        if feature.backend is None:
                            feature.backend = backend.backend_identifier
                        features.append(feature)
        return features

    def _collect_capability_unsupported(
        self,
        collection: SigmaCollection,
        backend: Backend,
        field_mapping: Mapping[str, str] | None,
    ) -> list[UnsupportedFeature]:
        features: list[UnsupportedFeature] = []
        capabilities = backend.get_capabilities()
        for rule in collection.rules:
            try:
                features.extend(
                    detect_unsupported_features(
                        rule,
                        capabilities,
                        backend=backend.backend_identifier,
                        field_mapping=field_mapping,
                    )
                )
            except Exception:
                continue
        return features

    def _apply_unsupported_handlers(
        self,
        policy: UnsupportedPolicyType,
        features: list[UnsupportedFeature],
        collection: SigmaCollection,
        backend: Backend,
    ) -> None:
        if isinstance(policy, str):
            return
        for index, feature in enumerate(features):
            handler: UnsupportedHandler | None
            if isinstance(policy, Mapping):
                handler = policy.get(feature.feature) or policy.get("*")
            else:
                handler = policy
            if handler is None:
                continue
            result = handler(feature, collection, backend)
            if result is not None and result is not False:
                feature.handled = True
                if isinstance(result, str):
                    feature.fallback = result
                elif isinstance(result, UnsupportedFeature):
                    features[index] = result

    def _convert_backend(
        self,
        spec: BackendSpec,
        collection: SigmaCollection,
        output_format: str | None,
        correlation_method: str | None,
        source_semantics: Any,
        policy: UnsupportedPolicyType,
        callback: (
            Callable[[SigmaRule | SigmaCorrelationRule, str | None, int, Any, Any], Any] | None
        ) = None,
    ) -> BackendConversionResult:
        backend = spec.create_backend()
        cloned = self._clone_collection(collection)
        field_mapping = spec.effective_field_mapping(self.global_field_mapping)
        pipeline = self._compose_pipeline(spec, backend)
        policy_name = policy.value if isinstance(policy, UnsupportedPolicy) else policy
        degradation_transform: SimpleDegradationTransformation | None = None
        if self.degrade and isinstance(policy_name, str) and policy_name in {"degrade", "fallback"}:
            degradation, degradation_transform = degradation_pipeline(
                backend.backend_identifier,
                backend.get_capabilities(),
            )
            pipeline = degradation if pipeline is None else pipeline + degradation
        format_name = (
            spec.output_format or output_format or getattr(backend, "default_format", "default")
        )
        method = spec.correlation_method or correlation_method

        previous_pipeline = backend.processing_pipeline
        previous_collect_errors = backend.collect_errors
        previous_error_count = len(backend.errors)
        backend.processing_pipeline = pipeline
        backend.collect_errors = self.collect_errors
        start = time.perf_counter()
        fatal: BaseException | None = None
        queries: Any = []
        try:
            queries = backend.convert(
                cloned,
                output_format=format_name,
                correlation_method=method,
                callback=callback,
            )
        except BaseException as fatal_error:
            if self.raise_on_error:
                raise
            fatal = fatal_error
        finally:
            elapsed = time.perf_counter() - start
            backend.processing_pipeline = previous_pipeline
            backend.collect_errors = previous_collect_errors
            backend_errors = [error for _, error in backend.errors[previous_error_count:]]

        errors: list[BaseException] = []
        if fatal is not None:
            errors.append(fatal)
        errors.extend(backend_errors)

        unsupported: list[UnsupportedFeature] = []
        if degradation_transform is not None:
            unsupported.extend(degradation_transform.degraded)
        unsupported.extend(self._collect_capability_unsupported(cloned, backend, field_mapping))
        unsupported.extend(self._collect_state_unsupported(cloned, backend))
        for backend_error in backend_errors:
            if isinstance(backend_error, SigmaError) and (
                "not supported" in str(backend_error).lower()
                or backend_error.__class__.__name__ == "SigmaFeatureNotSupportedByBackendError"
            ):
                feature = UnsupportedFeature.from_exception(
                    backend_error, backend=backend.backend_identifier
                )
                if not any(existing.feature == feature.feature for existing in unsupported):
                    unsupported.append(feature)
        if fatal is not None:
            feature = UnsupportedFeature.from_exception(fatal, backend=backend.backend_identifier)
            if not any(existing.feature == feature.feature for existing in unsupported):
                unsupported.append(feature)
        unsupported = merge_unsupported_features(unsupported)
        self._apply_unsupported_handlers(policy, unsupported, cloned, backend)

        if isinstance(policy, str) and policy in {"raise", "error"}:
            for feature in unsupported:
                if feature.is_error:
                    raise SigmaUnsupportedFeatureError(
                        feature=feature.feature,
                        backend=backend.backend_identifier,
                        reason=feature.reason,
                        source=feature.source,
                    )
            if fatal is not None:
                raise fatal

        backend_semantics, normalized_semantics = self._processed_semantics(
            cloned, field_mapping, backend
        )
        failed = (
            fatal is not None
            or bool(backend_errors)
            or any(feature.is_error for feature in unsupported)
        )
        return BackendConversionResult(
            backend=spec.name,
            queries=queries,
            output_format=str(format_name),
            errors=errors,
            unsupported=unsupported,
            field_mapping=field_mapping,
            semantics=(
                None
                if fatal is not None
                else (
                    normalized_semantics if normalized_semantics is not None else source_semantics
                )
            ),
            backend_semantics=None if fatal is not None else backend_semantics,
            source_semantics=source_semantics,
            rule_count=len(collection.rules),
            elapsed=elapsed,
            failed=failed,
            metadata={
                "backend_identifier": backend.backend_identifier,
                "pipeline": pipeline.name if pipeline is not None else None,
                "on_unsupported": policy if isinstance(policy, str) else "handler",
                "degrade": self.degrade,
                **spec.metadata,
            },
        )

    def convert(
        self,
        rules: (
            SigmaCollection
            | SigmaRule
            | SigmaCorrelationRule
            | Mapping[str, Any]
            | str
            | Sequence[SigmaRule | SigmaCorrelationRule | Mapping[str, Any]]
        ),
        output_format: str | None = None,
        correlation_method: str | None = None,
        on_unsupported: UnsupportedPolicyType | None = None,
        callback: (
            Callable[[SigmaRule | SigmaCorrelationRule, str | None, int, Any, Any], Any] | None
        ) = None,
    ) -> MultiBackendResult:
        """
        Convert rules with every registered backend.

        Args:
            rules: Sigma collection, single rule, YAML string, rule mapping or sequence thereof.
            output_format: Output format used for backends without a backend-specific format.
            correlation_method: Correlation method used for correlation rules.
            on_unsupported: Override the configured unsupported feature policy for this call.
            callback: Optional per-condition conversion callback. It is passed to backends that
                support it.

        Returns:
            A mapping from backend name to :class:`BackendConversionResult`.
        """
        collection = self._coerce_collection(rules)
        source_semantics = self._source_semantics(collection)
        policy = self.on_unsupported if on_unsupported is None else on_unsupported
        results: dict[str, BackendConversionResult] = {}
        for spec in self.specs:
            if not spec.enabled:
                continue
            results[spec.name] = self._convert_backend(
                spec,
                collection,
                output_format,
                correlation_method,
                source_semantics,
                policy,
                callback,
            )
        return MultiBackendResult(
            results,
            source_semantics=source_semantics,
            metadata={
                "backend_count": len(results),
                "field_mapping": dict(self.global_field_mapping),
            },
        )

    def convert_rule(
        self,
        rule: SigmaRule | SigmaCorrelationRule,
        output_format: str | None = None,
        correlation_method: str | None = None,
        on_unsupported: UnsupportedPolicyType | None = None,
    ) -> MultiBackendResult:
        """Convert a single rule with every registered backend."""
        return self.convert(
            rule,
            output_format=output_format,
            correlation_method=correlation_method,
            on_unsupported=on_unsupported,
        )

    def convert_collection(
        self,
        collection: SigmaCollection,
        output_format: str | None = None,
        correlation_method: str | None = None,
        on_unsupported: UnsupportedPolicyType | None = None,
    ) -> MultiBackendResult:
        """Convert a collection with every registered backend."""
        return self.convert(
            collection,
            output_format=output_format,
            correlation_method=correlation_method,
            on_unsupported=on_unsupported,
        )

    convert_all = convert
    convert_rules = convert


def convert_backend_with_report(
    backend: Backend,
    rules: (
        SigmaCollection
        | SigmaRule
        | SigmaCorrelationRule
        | Mapping[str, Any]
        | str
        | Sequence[SigmaRule | SigmaCorrelationRule | Mapping[str, Any]]
    ),
    output_format: str | None = None,
    correlation_method: str | None = None,
    callback: (
        Callable[[SigmaRule | SigmaCorrelationRule, str | None, int, Any, Any], Any] | None
    ) = None,
) -> BackendConversionResult:
    """Convert rules with a single backend and return a structured report."""
    unified = UnifiedBackend(
        {backend.backend_identifier: backend},
        collect_errors=True,
        clone_rules=True,
    )
    collection = unified._coerce_collection(rules)
    source_semantics = unified._source_semantics(collection)
    return unified._convert_backend(
        unified.specs[0],
        collection,
        output_format,
        correlation_method,
        source_semantics,
        unified.on_unsupported,
        callback,
    )


def convert_to_multiple_backends(
    rules: (
        SigmaCollection
        | SigmaRule
        | SigmaCorrelationRule
        | Mapping[str, Any]
        | str
        | Sequence[SigmaRule | SigmaCorrelationRule | Mapping[str, Any]]
    ),
    backends: (
        Mapping[str, BackendReference]
        | Iterable[BackendReference | BackendSpec | tuple[str, BackendReference]]
    ),
    *,
    field_mapping: Mapping[str, str] | None = None,
    field_mappings: Mapping[str, Mapping[str, str]] | None = None,
    pipeline: ProcessingPipeline | None = None,
    on_unsupported: UnsupportedPolicyType = UnsupportedPolicy.ANNOTATE,
    output_format: str | None = None,
) -> MultiBackendResult:
    """Convenience function converting rules with multiple backends."""
    return UnifiedBackend(
        backends,
        field_mapping=field_mapping,
        field_mappings=field_mappings,
        pipeline=pipeline,
        on_unsupported=on_unsupported,
    ).convert(rules, output_format=output_format)


def convert_multi(
    rules: Any,
    backends: Any,
    **kwargs: Any,
) -> MultiBackendResult:
    """Alias of :func:`convert_to_multiple_backends`."""
    return convert_to_multiple_backends(rules, backends, **kwargs)


def convert_to_backends(
    rules: Any,
    backends: Any,
    **kwargs: Any,
) -> MultiBackendResult:
    """Alias of :func:`convert_to_multiple_backends`."""
    return convert_to_multiple_backends(rules, backends, **kwargs)


def convert(
    rules: Any,
    backends: Any,
    **kwargs: Any,
) -> MultiBackendResult:
    """Alias of :func:`convert_to_multiple_backends`."""
    return convert_to_multiple_backends(rules, backends, **kwargs)


def convert_rule_to_backends(
    rule: SigmaRule | SigmaCorrelationRule,
    backends: Any,
    **kwargs: Any,
) -> MultiBackendResult:
    """Convert a single rule with multiple backends."""
    return convert_to_multiple_backends(rule, backends, **kwargs)


# Common names used by integrations.
MultiBackend = UnifiedBackend
SigmaMultiBackend = UnifiedBackend
CompositeBackend = UnifiedBackend
BackendGroup = UnifiedBackend
MultiBackendConverter = UnifiedBackend
BackendConverter = UnifiedBackend
