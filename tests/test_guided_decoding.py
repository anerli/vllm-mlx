# SPDX-License-Identifier: Apache-2.0
"""Tests for guided decoding utilities."""

import pytest

from vllm_mlx.guided_decoding import (
    GuidedDecodingParams,
    OutlinesLogitsProcessor,
    build_logits_processor,
)


class TestGuidedDecodingParams:
    """Tests for GuidedDecodingParams."""

    def test_imports(self):
        assert GuidedDecodingParams is not None
        assert OutlinesLogitsProcessor is not None
        assert build_logits_processor is not None

    def test_json_schema_params(self):
        params = GuidedDecodingParams(
            kind="json_schema",
            json_schema={"type": "object"},
        )
        assert params.kind == "json_schema"
        assert params.json_schema == {"type": "object"}
        assert params.regex is None
        assert params.enable_thinking is False

    def test_regex_params(self):
        params = GuidedDecodingParams(kind="regex", regex=r"\d+")
        assert params.kind == "regex"
        assert params.regex == r"\d+"

    def test_params_with_thinking(self):
        params = GuidedDecodingParams(
            kind="json_schema",
            json_schema={"type": "object"},
            enable_thinking=True,
        )
        assert params.enable_thinking is True


class TestBuildLogitsProcessorValidation:
    """Tests for build_logits_processor validation."""

    def test_missing_json_schema_raises(self):
        with pytest.raises(ValueError, match="json_schema is required"):
            build_logits_processor(
                None,
                GuidedDecodingParams(kind="json_schema", json_schema=None),
            )

    def test_missing_regex_raises(self):
        with pytest.raises(ValueError, match="regex is required"):
            build_logits_processor(
                None,
                GuidedDecodingParams(kind="regex", regex=None),
            )

    def test_unsupported_kind_raises(self):
        with pytest.raises(ValueError, match="Unsupported"):
            build_logits_processor(None, GuidedDecodingParams(kind="unknown"))

    def test_missing_grammar_raises(self):
        with pytest.raises(ValueError, match="grammar is required"):
            build_logits_processor(None, GuidedDecodingParams(kind="cfg", grammar=None))

    def test_missing_choice_raises(self):
        with pytest.raises(ValueError, match="choice requires"):
            build_logits_processor(None, GuidedDecodingParams(kind="choice", choice=None))

    def test_empty_choice_raises(self):
        with pytest.raises(ValueError, match="choice requires"):
            build_logits_processor(None, GuidedDecodingParams(kind="choice", choice=[]))

    def test_cfg_with_thinking_raises(self):
        with pytest.raises(ValueError, match="not compatible with CFG"):
            build_logits_processor(
                None,
                GuidedDecodingParams(kind="cfg", grammar="start: 'x'", enable_thinking=True),
            )