# SPDX-License-Identifier: Apache-2.0
"""Slow integration tests for structured output and guided decoding."""

import asyncio
import json

import pytest


def _parse_json_output(text):
    """Parse JSON from output, stripping thinking tags if present."""
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return json.loads(text)


@pytest.mark.slow
class TestModelLoading:
    @pytest.fixture(scope="class")
    def model_and_tokenizer(self):
        from vllm_mlx.utils.tokenizer import load_model_with_fallback

        return load_model_with_fallback("mlx-community/Qwen3.5-9B-4bit")

    def test_model_loads(self, model_and_tokenizer):
        model, tokenizer = model_and_tokenizer
        assert model is not None
        assert tokenizer is not None

    def test_hybrid_cache_structure(self, model_and_tokenizer):
        model, _ = model_and_tokenizer
        from mlx_lm.models.cache import KVCache, make_prompt_cache

        cache = make_prompt_cache(model)
        kv_count = sum(1 for c in cache if isinstance(c, KVCache))
        non_kv_count = len(cache) - kv_count

        assert len(cache) > 0
        assert non_kv_count > 0

    def test_build_json_processor(self, model_and_tokenizer):
        _, tokenizer = model_and_tokenizer
        from vllm_mlx.guided_decoding import (
            GuidedDecodingParams,
            OutlinesLogitsProcessor,
            build_logits_processor,
        )

        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        proc = build_logits_processor(
            tokenizer,
            GuidedDecodingParams(kind="json_schema", json_schema=schema),
        )
        assert isinstance(proc, OutlinesLogitsProcessor)

    def test_build_thinking_processor(self, model_and_tokenizer):
        _, tokenizer = model_and_tokenizer
        from vllm_mlx.guided_decoding import GuidedDecodingParams, build_logits_processor

        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        proc = build_logits_processor(
            tokenizer,
            GuidedDecodingParams(
                kind="json_schema",
                json_schema=schema,
                enable_thinking=True,
            ),
        )
        assert proc is not None

    def test_build_regex_processor(self, model_and_tokenizer):
        _, tokenizer = model_and_tokenizer
        from vllm_mlx.guided_decoding import GuidedDecodingParams, build_logits_processor

        proc = build_logits_processor(
            tokenizer,
            GuidedDecodingParams(kind="regex", regex=r"\d{4}"),
        )
        assert proc is not None


@pytest.mark.slow
class TestSimpleEngine:
    @pytest.fixture(scope="class")
    def engine(self):
        from vllm_mlx.engine.simple import SimpleEngine

        engine = SimpleEngine("mlx-community/Qwen3.5-9B-4bit")
        asyncio.get_event_loop().run_until_complete(engine.start())
        return engine

    @pytest.mark.asyncio
    async def test_basic_generation(self, engine):
        out = await engine.generate(prompt="Say hello:", max_tokens=10, temperature=0.0)
        assert out.text

    @pytest.mark.asyncio
    async def test_chat_no_constraints(self, engine):
        out = await engine.chat(
            messages=[{"role": "user", "content": "Say hi"}],
            max_tokens=15,
            temperature=0.0,
        )
        assert out.text

    @pytest.mark.asyncio
    async def test_constrained_simple_object(self, engine):
        from vllm_mlx.guided_decoding import GuidedDecodingParams

        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name"],
        }
        out = await engine.chat(
            messages=[{"role": "user", "content": "Give me a name as JSON"}],
            max_tokens=50,
            temperature=0.0,
            guided_decoding=GuidedDecodingParams(
                kind="json_schema",
                json_schema=schema,
                enable_thinking=True,
            ),
        )
        parsed = _parse_json_output(out.text)
        assert "name" in parsed
        assert isinstance(parsed["name"], str)

    @pytest.mark.asyncio
    async def test_constrained_nested_object(self, engine):
        from vllm_mlx.guided_decoding import GuidedDecodingParams

        schema = {
            "type": "object",
            "properties": {
                "person": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                    "required": ["name", "age"],
                }
            },
            "required": ["person"],
        }
        out = await engine.chat(
            messages=[
                {"role": "user", "content": "Give me a fictional person as nested JSON"}
            ],
            max_tokens=80,
            temperature=0.0,
            guided_decoding=GuidedDecodingParams(
                kind="json_schema",
                json_schema=schema,
                enable_thinking=True,
            ),
        )
        parsed = _parse_json_output(out.text)
        assert "person" in parsed
        assert isinstance(parsed["person"]["age"], int)

    @pytest.mark.asyncio
    async def test_constrained_array(self, engine):
        from vllm_mlx.guided_decoding import GuidedDecodingParams

        schema = {
            "type": "object",
            "properties": {"items": {"type": "array", "items": {"type": "string"}}},
            "required": ["items"],
        }
        out = await engine.chat(
            messages=[{"role": "user", "content": "List 3 fruits as JSON"}],
            max_tokens=80,
            temperature=0.0,
            guided_decoding=GuidedDecodingParams(
                kind="json_schema",
                json_schema=schema,
                enable_thinking=True,
            ),
        )
        parsed = _parse_json_output(out.text)
        assert isinstance(parsed["items"], list)
        assert len(parsed["items"]) > 0

    @pytest.mark.asyncio
    async def test_constrained_enum(self, engine):
        from vllm_mlx.guided_decoding import GuidedDecodingParams

        schema = {
            "type": "object",
            "properties": {
                "color": {"type": "string", "enum": ["red", "green", "blue"]}
            },
            "required": ["color"],
        }
        out = await engine.chat(
            messages=[{"role": "user", "content": "Pick a color"}],
            max_tokens=50,
            temperature=0.0,
            guided_decoding=GuidedDecodingParams(
                kind="json_schema",
                json_schema=schema,
                enable_thinking=True,
            ),
        )
        parsed = _parse_json_output(out.text)
        assert parsed["color"] in ["red", "green", "blue"]

    @pytest.mark.asyncio
    async def test_constrained_boolean_number(self, engine):
        from vllm_mlx.guided_decoding import GuidedDecodingParams

        schema = {
            "type": "object",
            "properties": {"flag": {"type": "boolean"}, "score": {"type": "number"}},
            "required": ["flag", "score"],
        }
        out = await engine.chat(
            messages=[{"role": "user", "content": "Is sky blue? Rate 1-10. JSON."}],
            max_tokens=50,
            temperature=0.0,
            guided_decoding=GuidedDecodingParams(
                kind="json_schema",
                json_schema=schema,
                enable_thinking=True,
            ),
        )
        parsed = _parse_json_output(out.text)
        assert isinstance(parsed["flag"], bool)
        assert isinstance(parsed["score"], (int, float))

    @pytest.mark.asyncio
    async def test_constrained_complex_schema(self, engine):
        from vllm_mlx.guided_decoding import GuidedDecodingParams

        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "tags": {"type": "array", "items": {"type": "string"}},
                "metadata": {
                    "type": "object",
                    "properties": {"version": {"type": "integer"}},
                    "required": ["version"],
                },
            },
            "required": ["name", "tags", "metadata"],
        }
        out = await engine.chat(
            messages=[{"role": "user", "content": "Describe a software project as JSON"}],
            max_tokens=120,
            temperature=0.0,
            guided_decoding=GuidedDecodingParams(
                kind="json_schema",
                json_schema=schema,
                enable_thinking=True,
            ),
        )
        parsed = _parse_json_output(out.text)
        assert isinstance(parsed["name"], str)
        assert isinstance(parsed["tags"], list)
        assert isinstance(parsed["metadata"]["version"], int)

    @pytest.mark.asyncio
    async def test_no_thinking_mode(self, engine):
        from vllm_mlx.guided_decoding import GuidedDecodingParams

        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }
        out = await engine.chat(
            messages=[{"role": "user", "content": "Give me 42 as JSON"}],
            max_tokens=30,
            temperature=0.0,
            guided_decoding=GuidedDecodingParams(
                kind="json_schema",
                json_schema=schema,
                enable_thinking=False,
            ),
        )
        parsed = json.loads(out.text)
        assert isinstance(parsed["x"], int)

    @pytest.mark.asyncio
    async def test_backwards_compat_no_guided(self, engine):
        out = await engine.chat(
            messages=[{"role": "user", "content": "Hi"}],
            max_tokens=10,
            temperature=0.0,
        )
        assert out.text is not None


@pytest.mark.slow
class TestBatchedEngine:
    @pytest.fixture(scope="class")
    def engine(self):
        from vllm_mlx.engine.batched import BatchedEngine

        engine = BatchedEngine("mlx-community/Qwen3.5-9B-4bit")
        asyncio.get_event_loop().run_until_complete(engine.start())
        return engine

    @pytest.mark.asyncio
    async def test_basic_generation(self, engine):
        out = await engine.generate(prompt="Hello:", max_tokens=10, temperature=0.0)
        assert out.text

    @pytest.mark.asyncio
    async def test_chat_no_constraints(self, engine):
        out = await engine.chat(
            messages=[{"role": "user", "content": "Say hi"}],
            max_tokens=20,
            temperature=0.0,
        )
        assert out.text

    @pytest.mark.asyncio
    async def test_constrained_json(self, engine):
        from vllm_mlx.guided_decoding import GuidedDecodingParams

        schema = {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
        }
        out = await engine.chat(
            messages=[{"role": "user", "content": "What is 5*5?"}],
            max_tokens=50,
            temperature=0.0,
            guided_decoding=GuidedDecodingParams(
                kind="json_schema",
                json_schema=schema,
                enable_thinking=True,
            ),
        )
        parsed = _parse_json_output(out.text)
        assert "answer" in parsed
        assert isinstance(parsed["answer"], int)

    @pytest.mark.asyncio
    async def test_concurrent_constrained(self, engine):
        from vllm_mlx.guided_decoding import GuidedDecodingParams

        schema = {
            "type": "object",
            "properties": {"result": {"type": "integer"}},
            "required": ["result"],
        }

        async def req(question):
            return await engine.chat(
                messages=[{"role": "user", "content": f"{question} Answer as JSON."}],
                max_tokens=50,
                temperature=0.0,
                guided_decoding=GuidedDecodingParams(
                    kind="json_schema",
                    json_schema=schema,
                    enable_thinking=True,
                ),
            )

        results = await asyncio.gather(req("What is 2+2?"), req("What is 10-3?"))
        for result in results:
            parsed = _parse_json_output(result.text)
            assert isinstance(parsed["result"], int)

    @pytest.mark.asyncio
    async def test_mixed_concurrent(self, engine):
        from vllm_mlx.guided_decoding import GuidedDecodingParams

        async def unconstrained():
            return await engine.chat(
                messages=[{"role": "user", "content": "Say hello briefly"}],
                max_tokens=20,
                temperature=0.0,
            )

        schema = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
            "required": ["x"],
        }

        async def constrained():
            return await engine.chat(
                messages=[{"role": "user", "content": "What is 7+7?"}],
                max_tokens=50,
                temperature=0.0,
                guided_decoding=GuidedDecodingParams(
                    kind="json_schema",
                    json_schema=schema,
                    enable_thinking=True,
                ),
            )

        r1, r2, r3 = await asyncio.gather(
            constrained(),
            unconstrained(),
            constrained(),
        )
        for result in [r1, r3]:
            parsed = _parse_json_output(result.text)
            assert "x" in parsed
        assert r2.text

    @pytest.mark.asyncio
    async def test_multi_turn(self, engine):
        messages = [
            {"role": "system", "content": "You are a math tutor."},
            {"role": "user", "content": "What is 3+3?"},
        ]
        out1 = await engine.chat(messages=messages, max_tokens=30, temperature=0.0)
        assert out1.text

        out2 = await engine.chat(
            messages=messages
            + [
                {"role": "assistant", "content": out1.text},
                {"role": "user", "content": "Now what is 4+4?"},
            ],
            max_tokens=30,
            temperature=0.0,
        )
        assert out2.text

    @pytest.mark.asyncio
    async def test_backwards_compat(self, engine):
        out = await engine.generate(prompt="Test:", max_tokens=5, temperature=0.0)
        assert out.text is not None