"""Guided decoding via Outlines logits processors.

Reference implementation: mlx-omni-server's OutlinesLogitsProcessor.
Requires outlines==1.0.4 and outlines-core.
"""

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger(__name__)


def _patch_cfg_guide() -> None:
    """Monkey-patch Outlines' CFGGuide to handle terminal parser states.

    Outlines uses None as a sentinel for terminated parser state in CFGGuide,
    but _get_parser_state_token_applied() doesn't guard against it — it does
    copy.copy(None) then tries to access .lexer on the result, crashing with:
        AttributeError: 'NoneType' object has no attribute 'lexer'

    The fix adds a guard that raises EOFError for terminal states, which
    iter_valid_token_ids() already catches, so terminal states simply yield
    no valid non-EOS tokens.

    See: https://github.com/dottxt-ai/outlines/issues/959
    """
    try:
        from outlines.processors.guide import CFGGuide
    except ImportError:
        return

    original = CFGGuide._get_parser_state_token_applied

    if getattr(original, "_patched_by_vllm_mlx", False):
        return  # already patched

    def _patched_get_parser_state_token_applied(self, state, token_id):
        if state.parser_state is None:
            raise EOFError("Cannot apply token to terminated CFG state")
        return original(self, state, token_id)

    _patched_get_parser_state_token_applied._patched_by_vllm_mlx = True
    CFGGuide._get_parser_state_token_applied = _patched_get_parser_state_token_applied
    logger.debug("Patched Outlines CFGGuide._get_parser_state_token_applied")


_patch_cfg_guide()

# Thinking pattern for reasoning models (e.g., Qwen3.5)
# Matches optional <think>...</think>\n prefix
_THINKING_PATTERN = (
    r"<think>([^<]|<[^\/]|<\/[^t]|<\/t[^h]|<\/th[^i]"
    r"|<\/thi[^n]|<\/thin[^k]|<\/think[^>])*<\/think>\n"
)


@dataclass
class GuidedDecodingParams:
    """Parameters for constrained/guided decoding."""

    kind: Literal["json_schema", "regex", "cfg", "choice"] = "json_schema"
    json_schema: dict[str, Any] | str | None = None
    regex: str | None = None
    grammar: str | None = None  # Lark/EBNF grammar string for CFG
    choice: list[str] | None = field(default=None)  # Allowed output strings
    enable_thinking: bool = False  # Allow <think>...</think> prefix


class OutlinesLogitsProcessor:
    """Wraps an Outlines logits processor for use with mlx-lm.

    Handles tokenizer wrapping, tensor reshaping, and optional
    thinking-tag support for reasoning models like Qwen3.5.
    """

    def __init__(self, tokenizer: Any, params: GuidedDecodingParams):
        try:
            from outlines.models.transformers import TransformerTokenizer
            from outlines.processors import JSONLogitsProcessor
            from outlines.processors.structured import RegexLogitsProcessor
        except ImportError as e:
            raise ImportError(
                "Outlines is required for structured output. "
                "Install with: pip install 'vllm-mlx[structured]'"
            ) from e

        # Validate params early, before touching tokenizer
        if params.kind not in ("json_schema", "regex", "cfg", "choice"):
            raise ValueError(f"Unsupported guided decoding kind: {params.kind}")
        if params.kind == "json_schema" and params.json_schema is None:
            raise ValueError("json_schema is required for json_schema guided decoding")
        if params.kind == "regex" and params.regex is None:
            raise ValueError("regex is required for regex guided decoding")
        if params.kind == "cfg" and params.grammar is None:
            raise ValueError("grammar is required for cfg guided decoding")
        if params.kind == "choice" and (not params.choice or len(params.choice) == 0):
            raise ValueError("choice requires a non-empty list of strings")
        if params.kind == "cfg" and params.enable_thinking:
            raise ValueError(
                "enable_thinking is not compatible with CFG grammars. "
                "CFG grammars define their own structure and cannot be "
                "composed with the thinking pattern."
            )

        try:
            import mlx.core as mx

            self._mx = mx
        except ImportError as e:
            raise ImportError("mlx is required for guided decoding") from e

        # Unwrap to get the raw tokenizer that TransformerTokenizer expects
        raw_tokenizer = tokenizer
        if hasattr(tokenizer, "_tokenizer"):
            raw_tokenizer = tokenizer._tokenizer

        outlines_tokenizer = TransformerTokenizer(raw_tokenizer)

        if params.kind == "json_schema":
            schema = params.json_schema

            if params.enable_thinking:
                from outlines.types import JsonSchema
                from outlines_core.fsm.json_schema import build_regex_from_schema

                schema_str = (
                    JsonSchema(schema).schema if not isinstance(schema, str) else schema
                )
                json_regex = build_regex_from_schema(schema_str)
                combined_regex = f"({_THINKING_PATTERN})?{json_regex}"

                self._processor = RegexLogitsProcessor(
                    combined_regex,
                    outlines_tokenizer,
                    tensor_library_name="mlx",
                )
            else:
                self._processor = JSONLogitsProcessor(
                    schema,
                    outlines_tokenizer,
                    tensor_library_name="mlx",
                )

        elif params.kind == "regex":
            pattern = params.regex
            if params.enable_thinking:
                pattern = f"({_THINKING_PATTERN})?{pattern}"

            self._processor = RegexLogitsProcessor(
                pattern,
                outlines_tokenizer,
                tensor_library_name="mlx",
            )

        elif params.kind == "cfg":
            from outlines.processors.structured import CFGLogitsProcessor

            self._processor = CFGLogitsProcessor(
                params.grammar,
                outlines_tokenizer,
                tensor_library_name="mlx",
            )

        elif params.kind == "choice":
            escaped = [re.escape(c) for c in params.choice]
            choice_regex = f"({'|'.join(escaped)})"

            if params.enable_thinking:
                choice_regex = f"({_THINKING_PATTERN})?{choice_regex}"

            self._processor = RegexLogitsProcessor(
                choice_regex,
                outlines_tokenizer,
                tensor_library_name="mlx",
            )

    def __call__(self, tokens: Any, logits: Any) -> Any:
        mx = self._mx

        # Outlines processor expects 1D logits
        logits_1d = logits.reshape(-1)

        # Ensure float32
        processed = self._processor(tokens, logits_1d.astype(mx.float32))

        # Convert back to mx.array and reshape
        return mx.array(processed).reshape(logits.shape)


def build_logits_processor(
    tokenizer: Any, params: GuidedDecodingParams
) -> OutlinesLogitsProcessor:
    """Create an Outlines logits processor for one request.

    Each request needs its own processor instance (they are stateful).
    """
    return OutlinesLogitsProcessor(tokenizer, params)
