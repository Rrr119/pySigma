from __future__ import annotations

import re

import pytest

from sigma.backends.test import TextQueryTestBackend
from sigma.collection import SigmaCollection
from sigma.conversion.base import TextQueryBackend
from sigma.conversion.capabilities import BackendCapabilities, Features
from sigma.conversion.multi import (
    BackendSpec,
    UnifiedBackend,
    convert_to_multiple_backends,
    field_mapping_pipeline,
)
from sigma.conversion.result import BackendConversionResult, MultiBackendResult
from sigma.exceptions import SigmaUnsupportedFeatureError
from sigma.processing.pipeline import ProcessingItem, ProcessingPipeline
from sigma.processing.transformations import FieldMappingTransformation
from sigma.rule import SigmaRule
from sigma.types import CompareOperators


class VendorQueryBackend(TextQueryBackend):
    """Minimal text backend used to simulate a vendor query language."""

    name = "Vendor query backend"
    group_expression = "({expr})"
    or_token = "OR"
    and_token = "AND"
    not_token = "NOT"
    eq_token = "="
    str_quote = '"'
    escape_char = "\\"
    wildcard_multi = "*"
    wildcard_single = "?"
    field_quote = '"'
    field_quote_pattern = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
    field_quote_pattern_negation = True
    re_expression = "{field}=/{regex}/"
    re_escape_char = "\\"
    re_escape = ["/"]
    cidr_expression = "cidrmatch('{field}', '{value}')"
    compare_op_expression = "{field}{operator}{value}"
    compare_operators = {
        CompareOperators.LT: "<",
        CompareOperators.LTE: "<=",
        CompareOperators.GT: ">",
        CompareOperators.GTE: ">=",
        CompareOperators.NEQ: "!=",
    }
    field_equals_field_expression = "{field1}=fieldref({field2})"
    field_null_expression = "{field} IS NULL"
    field_exists_expression = "exists({field})"
    field_not_exists_expression = "notexists({field})"
    unbound_value_str_expression = '"{value}"'
    unbound_value_num_expression = "{value}"
    unbound_value_re_expression = "/{value}/"


class VendorOneBackend(VendorQueryBackend):
    identifier = "vendor_one"
    and_token = "AND"
    or_token = "OR"


class VendorTwoBackend(VendorQueryBackend):
    identifier = "vendor_two"
    and_token = "&&"
    or_token = "||"


class VendorThreeBackend(VendorQueryBackend):
    identifier = "vendor_three"
    and_token = " and "
    or_token = " or "


class NoRegexBackend(VendorQueryBackend):
    identifier = "no_regex"
    re_expression = None


class NoNativeCidrBackend(VendorQueryBackend):
    identifier = "no_native_cidr"
    cidr_expression = None


def rule_yaml(detection: str) -> str:
    return f"""
title: Multi backend test
id: 11111111-1111-1111-1111-111111111111
status: test
logsource:
    category: process_creation
    product: windows
detection:
{detection}
"""


@pytest.fixture
def rule() -> SigmaRule:
    return SigmaRule.from_yaml(rule_yaml("""
    selection:
        CommandLine|contains: mimikatz
        Image|endswith: '\\powershell.exe'
    condition: selection
"""))


def test_three_backends_are_semantically_consistent(rule: SigmaRule):
    unified = UnifiedBackend(
        {
            "one": VendorOneBackend(),
            "two": VendorTwoBackend(),
            "three": VendorThreeBackend(),
        },
        field_mappings={
            "one": {"CommandLine": "process.command_line", "Image": "process.executable"},
            "two": {"CommandLine": "proc_cmdline", "Image": "proc_image"},
            "three": {"CommandLine": "cmdline", "Image": "image"},
        },
    )
    result = unified.convert(rule)

    assert isinstance(result, MultiBackendResult)
    assert isinstance(result, dict)
    assert set(result) == {"one", "two", "three"}
    assert result.is_semantically_consistent
    assert result.is_consistent
    result.assert_semantically_consistent()
    assert result["one"].queries != result["two"].queries
    assert all(not backend.unsupported for backend in result.values())
    assert all(backend.complete for backend in result.values())

    # Semantic models are normalized back to the original Sigma field names.
    assert "process.command_line" in result["one"].queries[0]
    assert "proc_cmdline" in result["two"].queries[0]
    semantic_json = str(result.semantics)
    assert "CommandLine" in semantic_json
    assert "Image" in semantic_json
    assert "process.command_line" not in semantic_json


def test_backends_do_not_mutate_source_or_each_other(rule: SigmaRule):
    unified = UnifiedBackend(
        [("a", VendorOneBackend()), ("b", VendorTwoBackend())],
        field_mappings={"a": {"CommandLine": "a_cmd"}, "b": {"CommandLine": "b_cmd"}},
    )
    result = unified.convert(rule)

    assert "a_cmd" in result["a"].queries[0]
    assert "a_cmd" not in result["b"].queries[0]
    assert "b_cmd" in result["b"].queries[0]

    # The original rule must not be modified by any backend pipeline.
    selection = rule.detection.detections["selection"]
    fields = {item.field for item in selection.detection_items if hasattr(item, "field")}
    assert fields == {"CommandLine", "Image"}


def test_result_is_list_and_mapping_compatible(rule: SigmaRule):
    result = UnifiedBackend({"vendor": VendorOneBackend()}).convert(rule)
    backend_result: BackendConversionResult = result["vendor"]

    assert backend_result == backend_result.queries
    assert list(backend_result) == backend_result.query_list
    assert backend_result["queries"] == backend_result.queries
    assert result == {"vendor": backend_result.queries}
    assert result.queries == {"vendor": backend_result.queries}


def test_unsupported_regex_is_explicitly_annotated():
    rule = SigmaRule.from_yaml(rule_yaml("""
    selection:
        CommandLine|re: 'mimi.*'
    condition: selection
"""))
    result = UnifiedBackend({"vendor": NoRegexBackend()}).convert(rule)
    backend_result = result["vendor"]

    assert backend_result.queries == []
    assert backend_result.failed
    assert not backend_result.success
    assert backend_result.unsupported
    assert backend_result.unsupported_features == [Features.REGEX]
    assert "regex" in backend_result.annotation
    assert any(feature.is_error for feature in backend_result.unsupported)

    # The error is still available for callers that need the original exception.
    assert backend_result.errors


def test_unsupported_regex_can_raise_structured_error():
    rule = SigmaRule.from_yaml(rule_yaml("""
    selection:
        CommandLine|re: 'mimi.*'
    condition: selection
"""))
    unified = UnifiedBackend({"vendor": NoRegexBackend()}, on_unsupported="raise")

    with pytest.raises(SigmaUnsupportedFeatureError, match="regex"):
        unified.convert(rule)


def test_unsupported_callback_marks_feature_handled():
    rule = SigmaRule.from_yaml(rule_yaml("""
    selection:
        CommandLine|re: 'mimi.*'
    condition: selection
"""))
    handled: list[str] = []

    def fallback(feature, collection, backend):
        handled.append(feature.feature)
        return "custom fallback"

    result = UnifiedBackend(
        {"vendor": NoRegexBackend()},
        on_unsupported=fallback,
    ).convert(rule)
    feature = result["vendor"].unsupported[0]

    assert handled == [Features.REGEX]
    assert feature.handled
    assert feature.fallback == "custom fallback"


def test_cidr_degradation_is_annotated():
    rule = SigmaRule.from_yaml(rule_yaml("""
    selection:
        DestinationIp|cidr: 192.168.0.0/16
    condition: selection
"""))
    result = UnifiedBackend({"vendor": NoNativeCidrBackend()}).convert(rule)
    backend_result = result["vendor"]

    assert backend_result.queries
    assert backend_result.degraded
    assert backend_result.unsupported_features == [Features.CIDR]
    assert backend_result.unsupported[0].is_degraded
    assert "degraded" in str(backend_result.unsupported[0])


def test_regex_can_be_degraded_to_wildcard_match():
    rule = SigmaRule.from_yaml(rule_yaml("""
    selection:
        CommandLine|re: 'mimi.*'
    condition: selection
"""))
    result = UnifiedBackend(
        {"vendor": NoRegexBackend()},
        on_unsupported="degrade",
    ).convert(rule)
    backend_result = result["vendor"]

    assert backend_result.queries
    assert "mimi*" in backend_result.queries[0]
    assert backend_result.degraded
    assert backend_result.unsupported_features == [Features.REGEX]
    assert backend_result.unsupported[0].handled
    assert backend_result.unsupported[0].fallback == "regex_to_wildcard"


def test_existing_and_additional_pipelines_are_compatible(rule: SigmaRule):
    backend_pipeline = field_mapping_pipeline(
        {"CommandLine": "backend_command_line"}, name="backend_mapping"
    )
    additional_pipeline = ProcessingPipeline(
        name="additional",
        items=[
            ProcessingItem(
                transformation=FieldMappingTransformation({"backend_command_line": "final_cmd"})
            )
        ],
    )
    backend = VendorOneBackend(processing_pipeline=backend_pipeline)
    result = UnifiedBackend([BackendSpec("vendor", backend, pipeline=additional_pipeline)]).convert(
        rule
    )

    assert "final_cmd" in result["vendor"].queries[0]
    assert result["vendor"].metadata["pipeline"] is not None


def test_capability_override():
    class OverriddenBackend(VendorQueryBackend):
        identifier = "overridden"
        capabilities = BackendCapabilities(regex=False)

    capabilities = OverriddenBackend().get_capabilities()
    assert not capabilities.regex
    assert not OverriddenBackend().supports(Features.REGEX)


def test_backend_convert_with_report():
    rule = SigmaRule.from_yaml(rule_yaml("""
    selection:
        CommandLine|re: 'mimi.*'
    condition: selection
"""))
    result = NoRegexBackend().convert_with_report(SigmaCollection([rule]))

    assert result.backend == "no_regex"
    assert result.failed
    assert result.unsupported_features == [Features.REGEX]


def test_convenience_function_and_input_types(rule: SigmaRule):
    result = convert_to_multiple_backends(
        rule,
        [VendorOneBackend(), VendorTwoBackend()],
        field_mapping={"CommandLine": "cmd"},
    )
    assert set(result) == {"vendor_one", "vendor_two"}
    assert all("cmd" in queries[0] for queries in result.queries.values())

    yaml_result = UnifiedBackend({"vendor": VendorOneBackend()}).convert(rule_yaml("""
    selection:
        CommandLine: test
    condition: selection
"""))
    assert yaml_result["vendor"].queries
