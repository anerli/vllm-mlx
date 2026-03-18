"""Run integration tests for fork changes. Requires mlx-community/Qwen3.5-9B-4bit."""

import asyncio
import json
import sys
import traceback

passed = []
failed = []

def report(name, success, detail=""):
    if success:
        passed.append(name)
        print(f"  ✅ {name}")
    else:
        failed.append(name)
        print(f"  ❌ {name}: {detail}")

def parse_json(text):
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return json.loads(text)

async def run_all():
    from vllm_mlx.guided_decoding import GuidedDecodingParams, build_logits_processor

    # --- MODEL LOADING ---
    print("\n--- MODEL LOADING ---")
    from vllm_mlx.utils.tokenizer import load_model_with_fallback
    model, tokenizer = load_model_with_fallback("mlx-community/Qwen3.5-9B-4bit")
    report("Model loads", True)

    from mlx_lm.models.cache import make_prompt_cache, KVCache
    cache = make_prompt_cache(model)
    kv = sum(1 for c in cache if isinstance(c, KVCache))
    non_kv = len(cache) - kv
    report(f"Hybrid cache ({kv} KV, {non_kv} non-KV)", non_kv > 0)

    # Processor builds
    schema = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
    proc = build_logits_processor(tokenizer, GuidedDecodingParams(kind="json_schema", json_schema=schema))
    report("JSON processor builds", proc is not None)
    proc = build_logits_processor(tokenizer, GuidedDecodingParams(kind="json_schema", json_schema=schema, enable_thinking=True))
    report("Thinking processor builds", proc is not None)
    proc = build_logits_processor(tokenizer, GuidedDecodingParams(kind="regex", regex=r"\d{4}"))
    report("Regex processor builds", proc is not None)

    # --- SIMPLE ENGINE ---
    print("\n--- SIMPLE ENGINE ---")
    from vllm_mlx.engine.simple import SimpleEngine
    engine = SimpleEngine("mlx-community/Qwen3.5-9B-4bit")
    await engine.start()

    # Basic generation
    try:
        out = await engine.generate(prompt="Say hello:", max_tokens=10, temperature=0.0)
        report("Basic generation", len(out.text) > 0)
    except Exception as e:
        report("Basic generation", False, str(e))

    # Chat no constraints
    try:
        out = await engine.chat(messages=[{"role": "user", "content": "Say hi"}], max_tokens=15, temperature=0.0)
        report("Chat no constraints", len(out.text) > 0)
    except Exception as e:
        report("Chat no constraints", False, str(e))

    # Constrained: simple object
    try:
        s = {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]}
        out = await engine.chat(
            messages=[{"role": "user", "content": "Give a name as JSON"}],
            max_tokens=50, temperature=0.0,
            guided_decoding=GuidedDecodingParams(kind="json_schema", json_schema=s, enable_thinking=True),
        )
        p = parse_json(out.text)
        report("Constrained simple object", "name" in p and isinstance(p["name"], str))
    except Exception as e:
        report("Constrained simple object", False, str(e))

    # Constrained: nested object
    try:
        s = {"type": "object", "properties": {"person": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}, "required": ["name", "age"]}}, "required": ["person"]}
        out = await engine.chat(
            messages=[{"role": "user", "content": "Fictional person as nested JSON"}],
            max_tokens=80, temperature=0.0,
            guided_decoding=GuidedDecodingParams(kind="json_schema", json_schema=s, enable_thinking=True),
        )
        p = parse_json(out.text)
        report("Constrained nested object", "person" in p and isinstance(p["person"]["age"], int))
    except Exception as e:
        report("Constrained nested object", False, str(e))

    # Constrained: array
    try:
        s = {"type": "object", "properties": {"items": {"type": "array", "items": {"type": "string"}}}, "required": ["items"]}
        out = await engine.chat(
            messages=[{"role": "user", "content": "List 3 fruits as JSON"}],
            max_tokens=80, temperature=0.0,
            guided_decoding=GuidedDecodingParams(kind="json_schema", json_schema=s, enable_thinking=True),
        )
        p = parse_json(out.text)
        report("Constrained array", isinstance(p["items"], list) and len(p["items"]) > 0)
    except Exception as e:
        report("Constrained array", False, str(e))

    # Constrained: enum
    try:
        s = {"type": "object", "properties": {"color": {"type": "string", "enum": ["red", "green", "blue"]}}, "required": ["color"]}
        out = await engine.chat(
            messages=[{"role": "user", "content": "Pick a color"}],
            max_tokens=50, temperature=0.0,
            guided_decoding=GuidedDecodingParams(kind="json_schema", json_schema=s, enable_thinking=True),
        )
        p = parse_json(out.text)
        report("Constrained enum", p["color"] in ["red", "green", "blue"])
    except Exception as e:
        report("Constrained enum", False, str(e))

    # Constrained: boolean + number
    try:
        s = {"type": "object", "properties": {"flag": {"type": "boolean"}, "score": {"type": "number"}}, "required": ["flag", "score"]}
        out = await engine.chat(
            messages=[{"role": "user", "content": "Is sky blue? Rate 1-10. JSON."}],
            max_tokens=50, temperature=0.0,
            guided_decoding=GuidedDecodingParams(kind="json_schema", json_schema=s, enable_thinking=True),
        )
        p = parse_json(out.text)
        report("Constrained boolean+number", isinstance(p["flag"], bool) and isinstance(p["score"], (int, float)))
    except Exception as e:
        report("Constrained boolean+number", False, str(e))

    # Constrained: complex nested
    try:
        s = {"type": "object", "properties": {"name": {"type": "string"}, "tags": {"type": "array", "items": {"type": "string"}}, "metadata": {"type": "object", "properties": {"version": {"type": "integer"}}, "required": ["version"]}}, "required": ["name", "tags", "metadata"]}
        out = await engine.chat(
            messages=[{"role": "user", "content": "Describe a software project as JSON"}],
            max_tokens=120, temperature=0.0,
            guided_decoding=GuidedDecodingParams(kind="json_schema", json_schema=s, enable_thinking=True),
        )
        p = parse_json(out.text)
        report("Constrained complex nested", isinstance(p["tags"], list) and isinstance(p["metadata"]["version"], int))
    except Exception as e:
        report("Constrained complex nested", False, str(e))

    # No thinking mode
    try:
        s = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
        out = await engine.chat(
            messages=[{"role": "user", "content": "Give me 42 as JSON"}],
            max_tokens=30, temperature=0.0,
            guided_decoding=GuidedDecodingParams(kind="json_schema", json_schema=s, enable_thinking=False),
        )
        p = json.loads(out.text)  # Should be pure JSON, no thinking tags
        report("No thinking mode", isinstance(p["x"], int))
    except Exception as e:
        report("No thinking mode", False, str(e))

    # Backwards compat
    try:
        out = await engine.chat(messages=[{"role": "user", "content": "Hi"}], max_tokens=10, temperature=0.0)
        report("Backwards compat (no guided)", out.text is not None)
    except Exception as e:
        report("Backwards compat (no guided)", False, str(e))

    # --- BATCHED ENGINE ---
    print("\n--- BATCHED ENGINE ---")
    from vllm_mlx.engine.batched import BatchedEngine
    bengine = BatchedEngine("mlx-community/Qwen3.5-9B-4bit")
    await bengine.start()

    # Basic
    try:
        out = await bengine.generate(prompt="Hello:", max_tokens=10, temperature=0.0)
        report("Batched basic generation", len(out.text) > 0)
    except Exception as e:
        report("Batched basic generation", False, str(e))

    # Chat no constraints
    try:
        out = await bengine.chat(messages=[{"role": "user", "content": "Say hi"}], max_tokens=20, temperature=0.0)
        report("Batched chat no constraints", len(out.text) > 0)
    except Exception as e:
        report("Batched chat no constraints", False, str(e))

    # Constrained JSON
    try:
        s = {"type": "object", "properties": {"answer": {"type": "integer"}}, "required": ["answer"]}
        out = await bengine.chat(
            messages=[{"role": "user", "content": "What is 5*5?"}],
            max_tokens=50, temperature=0.0,
            guided_decoding=GuidedDecodingParams(kind="json_schema", json_schema=s, enable_thinking=True),
        )
        p = parse_json(out.text)
        report("Batched constrained JSON", isinstance(p["answer"], int))
    except Exception as e:
        report("Batched constrained JSON", False, str(e))

    # Concurrent constrained
    try:
        s = {"type": "object", "properties": {"result": {"type": "integer"}}, "required": ["result"]}
        async def creq(q):
            return await bengine.chat(
                messages=[{"role": "user", "content": f"{q} Answer as JSON."}],
                max_tokens=50, temperature=0.0,
                guided_decoding=GuidedDecodingParams(kind="json_schema", json_schema=s, enable_thinking=True),
            )
        r1, r2 = await asyncio.gather(creq("What is 2+2?"), creq("What is 10-3?"))
        ok = all(isinstance(parse_json(r.text)["result"], int) for r in [r1, r2])
        report("Concurrent constrained", ok)
    except Exception as e:
        report("Concurrent constrained", False, str(e))

    # Mixed concurrent
    try:
        async def unc():
            return await bengine.chat(messages=[{"role": "user", "content": "Say hello"}], max_tokens=20, temperature=0.0)
        s = {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}
        async def con():
            return await bengine.chat(
                messages=[{"role": "user", "content": "What is 7+7?"}],
                max_tokens=50, temperature=0.0,
                guided_decoding=GuidedDecodingParams(kind="json_schema", json_schema=s, enable_thinking=True),
            )
        r1, r2, r3 = await asyncio.gather(con(), unc(), con())
        ok = all("x" in parse_json(r.text) for r in [r1, r3]) and len(r2.text) > 0
        report("Mixed concurrent", ok)
    except Exception as e:
        report("Mixed concurrent", False, str(e))

    # Multi-turn
    try:
        m1 = [{"role": "system", "content": "You are a math tutor."}, {"role": "user", "content": "What is 3+3?"}]
        o1 = await bengine.chat(messages=m1, max_tokens=30, temperature=0.0)
        m2 = m1 + [{"role": "assistant", "content": o1.text}, {"role": "user", "content": "Now what is 4+4?"}]
        o2 = await bengine.chat(messages=m2, max_tokens=30, temperature=0.0)
        report("Multi-turn conversation", len(o2.text) > 0)
    except Exception as e:
        report("Multi-turn conversation", False, str(e))

    # Backwards compat
    try:
        out = await bengine.generate(prompt="Test:", max_tokens=5, temperature=0.0)
        report("Batched backwards compat", out.text is not None)
    except Exception as e:
        report("Batched backwards compat", False, str(e))

    # --- ERROR HANDLING ---
    print("\n--- ERROR HANDLING ---")

    # Malformed schema should raise (not silently degrade)
    try:
        out = await engine.chat(
            messages=[{"role": "user", "content": "test"}],
            max_tokens=10, temperature=0.0,
            guided_decoding=GuidedDecodingParams(kind="json_schema", json_schema=None),
        )
        report("Malformed schema raises error", False, "Should have raised")
    except (ValueError, Exception) as e:
        report("Malformed schema raises error", "json_schema is required" in str(e))

    # Unsupported kind should raise
    try:
        out = await engine.chat(
            messages=[{"role": "user", "content": "test"}],
            max_tokens=10, temperature=0.0,
            guided_decoding=GuidedDecodingParams(kind="cfg"),
        )
        report("Unsupported kind raises error", False, "Should have raised")
    except (ValueError, Exception) as e:
        report("Unsupported kind raises error", "Unsupported" in str(e))

    # --- SUMMARY ---
    print(f"\n{'='*60}")
    print(f"RESULTS: {len(passed)} passed, {len(failed)} failed")
    print(f"{'='*60}")
    if failed:
        print("FAILED:")
        for f in failed:
            print(f"  ❌ {f}")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED ✅")

asyncio.run(run_all())