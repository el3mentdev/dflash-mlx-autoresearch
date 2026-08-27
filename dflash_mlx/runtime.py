# Copyright 2026 bstnxbt
# MIT License — see LICENSE file
# Based on DFlash (arXiv:2602.06036)

import os
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import cache as cache_mod
from mlx_lm.models import gated_delta as gated_delta_mod
from mlx_lm.models.base import (
    create_attention_mask,
    create_ssm_mask,
    scaled_dot_product_attention,
)
from mlx_lm.utils import load, load_model

from dflash_mlx.model import (
    ContextOnlyDraftKVCache,
    DFlashDraftModel,
    DFlashDraftModelArgs,
    extract_context_feature,
)
from dflash_mlx.recurrent_rollback_cache import RecurrentRollbackCache

# Optional DDTree integration — lazy imports to avoid circular dependency
# (ddtree_mlx imports from dflash_mlx, so module-level import would deadlock)
_DDTREE_AVAILABLE: Optional[bool] = None  # resolved on first use
_ddtree_imports: dict[str, Any] = {}


def _ensure_ddtree_imports() -> bool:
    global _DDTREE_AVAILABLE, _ddtree_imports
    if _DDTREE_AVAILABLE is not None:
        return _DDTREE_AVAILABLE
    try:
        from ddtree_mlx.runtime import _build_tree_from_mlx_logits
        from ddtree_mlx.compile import compile_tree
        from ddtree_mlx.verify import tree_verify_forward
        from ddtree_mlx.tree import follow_verified_tree
        from ddtree_mlx.cache import fast_path_commit as ddtree_fast_path_commit
        from ddtree_mlx.cache import tree_aware_path_commit
        from ddtree_mlx.runtime import _tree_token_ids

        _ddtree_imports.update({
            "build_tree": _build_tree_from_mlx_logits,
            "compile_tree": compile_tree,
            "tree_verify_forward": tree_verify_forward,
            "follow_verified_tree": follow_verified_tree,
            "fast_path_commit": ddtree_fast_path_commit,
            "tree_aware_path_commit": tree_aware_path_commit,
            "tree_token_ids": _tree_token_ids,
        })
        _DDTREE_AVAILABLE = True
    except ImportError:
        _DDTREE_AVAILABLE = False
    return _DDTREE_AVAILABLE


def _resolve_ddtree_acceptance_threshold() -> float:
    raw = os.environ.get("DDTREE_ACCEPTANCE_THRESHOLD", "").strip()
    if raw:
        return float(raw)
    return 0.65


def resolve_model_ref(model_ref: str | Path | None, *, kind: str) -> str:
    if model_ref:
        candidate = Path(model_ref).expanduser()
        return str(candidate if candidate.exists() else model_ref)
    raise ValueError(f"{kind} model reference is required")


def _get_dflash_model_classes(config: dict[str, Any]):
    return DFlashDraftModel, DFlashDraftModelArgs


def _resolve_local_model_path(model_ref: str | Path) -> Path:
    candidate = Path(model_ref).expanduser()
    if candidate.exists():
        return candidate
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise FileNotFoundError(f"Model path does not exist and huggingface_hub is unavailable: {model_ref}") from exc

    snapshot_path = snapshot_download(
        repo_id=str(model_ref),
        allow_patterns=["*.json", "*.safetensors", "*.py", "*.txt", "tokenizer*"],
    )
    return Path(snapshot_path)


def _prepare_prompt_tokens(tokenizer: Any, prompt: str, *, use_chat_template: bool) -> list[int]:
    if use_chat_template and hasattr(tokenizer, "apply_chat_template"):
        return list(
            tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=True,
                add_generation_prompt=True,
            )
        )
    return list(tokenizer.encode(prompt))


def sample_tokens(logits: mx.array) -> mx.array:
    return mx.argmax(logits, axis=-1)


def build_suppress_token_mask(
    vocab_size: int,
    suppress_token_ids: Optional[list[int]],
) -> Optional[mx.array]:
    token_ids = sorted(
        {
            int(token_id)
            for token_id in (suppress_token_ids or [])
            if 0 <= int(token_id) < vocab_size
        }
    )
    if not token_ids:
        return None
    vocab_indices = mx.arange(vocab_size, dtype=mx.int32)
    token_array = mx.array(token_ids, dtype=mx.int32)
    return mx.any(mx.equal(vocab_indices[:, None], token_array[None, :]), axis=1)


def sample_tokens_with_mask(
    logits: mx.array,
    suppress_token_mask: Optional[mx.array] = None,
) -> mx.array:
    if suppress_token_mask is None:
        return sample_tokens(logits)
    floor = mx.array(-1e9, dtype=logits.dtype)
    return mx.argmax(mx.where(suppress_token_mask, floor, logits), axis=-1)


def greedy_tokens_with_mask(
    logits: mx.array,
    suppress_token_mask: Optional[mx.array] = None,
) -> mx.array:
    if suppress_token_mask is None:
        return mx.argmax(logits, axis=-1).astype(mx.uint32)
    floor = mx.array(-1e9, dtype=logits.dtype)
    masked_logits = mx.where(suppress_token_mask, floor, logits)
    return mx.argmax(masked_logits, axis=-1).astype(mx.uint32)


def _dspark_draft_block(
    draft_model: Any,
    base_logits: mx.array,
    anchor: mx.array,
    suppress_token_mask: Optional[mx.array] = None,
) -> mx.array:
    """DSpark greedy block proposal (shifted next-token semantics).

    Every block hidden — anchor position included — predicts the token one
    position ahead, so ``base_logits`` of shape (block_len, vocab) yields
    block_len drafts. The Markov bigram bias is chained left-to-right, seeded
    by the anchor (bonus) token, matching SGLang's ``run_markov_block``.
    """
    markov_head = getattr(draft_model, "markov_head", None)
    prev = anchor.reshape(1)
    drafted: list[mx.array] = []
    for step in range(int(base_logits.shape[0])):
        step_logits = base_logits[step][None]
        if markov_head is not None:
            step_logits = step_logits + markov_head(prev)
        token = greedy_tokens_with_mask(step_logits, suppress_token_mask).reshape(1)
        drafted.append(token)
        prev = token
    return mx.concatenate(drafted, axis=0).astype(mx.uint32)


def _match_acceptance_length(
    drafted_tokens: mx.array,
    posterior_tokens: mx.array,
) -> mx.array:
    if int(drafted_tokens.shape[0]) == 0:
        return mx.array(0, dtype=mx.int32)
    matches = mx.equal(drafted_tokens, posterior_tokens).astype(mx.int32)
    return mx.sum(mx.cumprod(matches, axis=0))


def _concat_hidden_state_chunks(
    hidden_state_chunks: list[list[mx.array]],
) -> list[mx.array]:
    if not hidden_state_chunks:
        raise ValueError("expected at least one hidden-state chunk")
    if len(hidden_state_chunks) == 1:
        return hidden_state_chunks[0]
    return [
        mx.concatenate([chunk[index] for chunk in hidden_state_chunks], axis=1)
        for index in range(len(hidden_state_chunks[0]))
    ]


def _concat_hidden_state_chunk_dicts(
    hidden_state_chunks: list[dict[int, mx.array]],
    capture_layer_ids: set[int],
) -> dict[int, mx.array]:
    if not hidden_state_chunks:
        raise ValueError("expected at least one hidden-state chunk")
    if len(hidden_state_chunks) == 1:
        return hidden_state_chunks[0]
    return {
        layer_id: mx.concatenate([chunk[layer_id] for chunk in hidden_state_chunks], axis=1)
        for layer_id in sorted(capture_layer_ids)
    }


def _eval_logits_and_captured(
    logits: mx.array,
    captured: list[mx.array] | dict[int, mx.array],
) -> None:
    if isinstance(captured, dict):
        mx.eval(logits, *captured.values())
    else:
        mx.eval(logits, *captured)


def _target_text_wrapper(target_model: Any) -> Any:
    if hasattr(target_model, "model"):
        return target_model
    if hasattr(target_model, "language_model"):
        return target_model.language_model
    raise AttributeError(f"Unsupported target model wrapper: {type(target_model)!r}")


def _target_text_model(target_model: Any) -> Any:
    wrapper = _target_text_wrapper(target_model)
    if hasattr(wrapper, "model"):
        return wrapper.model
    raise AttributeError(f"Unsupported target text model: {type(wrapper)!r}")


def detect_target_family(target_model: Any) -> str:
    inner = _target_text_model(target_model)
    has_linear = any(
        hasattr(layer, "linear_attn") or hasattr(layer, "is_linear")
        for layer in inner.layers
    )
    return "hybrid_gdn" if has_linear else "pure_attention"


def _target_embed_tokens(target_model: Any) -> Any:
    return _target_text_model(target_model).embed_tokens


def _lm_head_logits(target_model: Any, hidden_states: mx.array) -> mx.array:
    wrapper = _target_text_wrapper(target_model)
    if getattr(getattr(wrapper, "args", None), "tie_word_embeddings", True):
        return wrapper.model.embed_tokens.as_linear(hidden_states)
    return wrapper.lm_head(hidden_states)


def extract_context_feature_from_dict(
    captured_dict: dict[int, mx.array],
    target_layer_ids: list[int],
) -> mx.array:
    selected = [captured_dict[layer_id + 1] for layer_id in target_layer_ids]
    return mx.concatenate(selected, axis=-1)


def _resolve_verify_len_cap(target_model: Any, block_tokens: int) -> int:
    override_raw = os.environ.get("DFLASH_VERIFY_LEN", "").strip()
    if override_raw:
        try:
            override = int(override_raw)
        except ValueError:
            override = 0
        if override > 0:
            return max(1, min(int(block_tokens), override))
    return int(block_tokens)


def _resolve_dflash_max_ctx() -> int:
    raw = os.environ.get("DFLASH_MAX_CTX", "0").strip()
    try:
        max_ctx = int(raw)
    except ValueError:
        max_ctx = 0
    if max_ctx <= 0:
        return sys.maxsize
    return max_ctx


def _resolve_think_budget() -> int:
    raw = os.environ.get("DFLASH_THINK_BUDGET", "").strip()
    if not raw:
        return 0
    try:
        return max(0, int(raw))
    except ValueError:
        return 0


def _resolve_think_end_id(tokenizer: Any) -> Optional[int]:
    # Only usable when </think> is a single token (true for Qwen3 vocabs).
    try:
        ids = tokenizer.encode("</think>")
    except Exception:
        return None
    if isinstance(ids, list) and len(ids) == 1:
        return int(ids[0])
    return None


def _resolve_draft_window() -> tuple[int, int]:
    sink = int(os.environ.get("DFLASH_DRAFT_SINK", "64").strip())
    window = int(os.environ.get("DFLASH_DRAFT_WINDOW", "1024").strip())
    return max(0, sink), max(1, window)


def _profile_dflash_cycles_enabled() -> bool:
    raw = os.environ.get("DFLASH_PROFILE", "").strip().lower()
    return raw not in {"", "0", "false", "no"}


def _ns_to_us(ns: int | float) -> float:
    return float(ns) / 1_000.0


def _should_quantize_draft(quantize_draft: bool = False) -> bool:
    if quantize_draft:
        return True
    raw = os.environ.get("DFLASH_QUANTIZE_DRAFT", "").strip().lower()
    return raw not in {"", "0", "false", "no"}




def _linear_forward(x: mx.array, weight: mx.array, bias: Optional[mx.array]) -> mx.array:
    out = x @ weight.T
    return out if bias is None else out + bias


_EXACT_SMALL_PROJ_PAD_M = 16


class _ExactSmallProjPad(nn.Module):
    def __init__(self, linear: nn.Module, *, pad_m: int = _EXACT_SMALL_PROJ_PAD_M):
        super().__init__()
        self.linear = linear
        self.pad_m = int(pad_m)
        self._dflash_exact_small_proj_wrapped = True

    @property
    def weight(self) -> mx.array:
        return self.linear.weight

    @weight.setter
    def weight(self, value: mx.array) -> None:
        self.linear.weight = value

    @property
    def bias(self) -> Optional[mx.array]:
        return getattr(self.linear, "bias", None)

    @bias.setter
    def bias(self, value: Optional[mx.array]) -> None:
        self.linear.bias = value

    def __call__(self, x: mx.array) -> mx.array:
        if x.ndim == 3 and x.shape[1] < self.pad_m:
            batch_size, seq_len, hidden_dim = x.shape
            pad = mx.zeros((batch_size, self.pad_m - seq_len, hidden_dim), dtype=x.dtype)
            out = self.linear(mx.concatenate([x, pad], axis=1))
            return out[:, :seq_len, :]
        return self.linear(x)


def _install_exact_small_proj_hooks(
    linear_attn: Any,
    *,
    pad_m: int = _EXACT_SMALL_PROJ_PAD_M,
) -> None:
    for attr_name in ("in_proj_b", "in_proj_a"):
        proj = getattr(linear_attn, attr_name, None)
        if proj is None or getattr(proj, "_dflash_exact_small_proj_wrapped", False):
            continue
        setattr(linear_attn, attr_name, _ExactSmallProjPad(proj, pad_m=pad_m))


def _attention_num_heads(attn: Any) -> int:
    for attr in ("num_attention_heads", "n_heads"):
        value = getattr(attn, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError(f"{type(attn).__name__} missing attention head count attribute")


def _attention_num_kv_heads(attn: Any) -> int:
    for attr in ("num_key_value_heads", "n_kv_heads"):
        value = getattr(attn, attr, None)
        if value is not None:
            return int(value)
    raise AttributeError(f"{type(attn).__name__} missing KV head count attribute")


def _attention_has_gated_q_proj(attn: Any) -> bool:
    q_proj = getattr(attn, "q_proj", None)
    q_norm = getattr(attn, "q_norm", None)
    q_proj_weight = getattr(q_proj, "weight", None)
    q_norm_weight = getattr(q_norm, "weight", None)
    if q_proj_weight is None or q_norm_weight is None:
        return False
    try:
        num_attention_heads = _attention_num_heads(attn)
    except AttributeError:
        return False
    expected_out_dim = 2 * num_attention_heads * int(q_norm_weight.shape[0])
    return int(q_proj_weight.shape[0]) == expected_out_dim


def pack_target_model_weights(target_model: Any, *, validate: bool = True) -> dict[str, Any]:
    return pack_target_model_weights_selective(
        target_model,
        validate=validate,
        pack_mlp=True,
        pack_attention=False,
    )


def pack_target_model_weights_selective(
    target_model: Any,
    *,
    validate: bool = True,
    pack_mlp: bool = True,
    pack_attention: bool = False,
) -> dict[str, Any]:
    text_model = _target_text_model(target_model)
    if getattr(text_model, "_dflash_pack_info", None) is not None:
        return text_model._dflash_pack_info

    pack_info = {
        "enabled": True,
        "validated": validate,
        "pack_mlp": pack_mlp,
        "pack_attention": pack_attention,
        "packed_mlp_layers": [],
        "packed_attention_layers": [],
    }
    text_model._dflash_pack_info = pack_info
    return pack_info


def _install_speculative_linear_cache_hook(linear_attn: Any) -> None:
    cls = type(linear_attn)
    if getattr(cls, "_dflash_speculative_call_installed", False):
        return

    original_call = cls.__call__

    def speculative_call(
        self,
        inputs: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if not isinstance(cache, RecurrentRollbackCache) or not getattr(cache, "_armed", False):
            return original_call(self, inputs, mask=mask, cache=cache)

        from mlx.nn.layers.distributed import sum_gradients

        B, S, _ = inputs.shape
        sharding_group = getattr(self, "sharding_group", None)

        if sharding_group is not None:
            inputs = sum_gradients(sharding_group)(inputs)

        qkv = self.in_proj_qkv(inputs)
        z_proj = self.in_proj_z(inputs)
        z = z_proj.reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)

        if cache[0] is not None:
            conv_state = cache[0]
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim),
                dtype=inputs.dtype,
            )

        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        cache[0] = conv_input[:, -(self.conv_kernel_size - 1) :]
        conv_out = nn.silu(self.conv1d(conv_input))

        q, k, v = [
            tensor.reshape(B, S, heads, dim)
            for tensor, heads, dim in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
                strict=True,
            )
        ]

        state = cache[1]
        inv_scale = k.shape[-1] ** -0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)
        g = gated_delta_mod.compute_g(self.A_log, a, self.dt_bias)
        beta = mx.sigmoid(b)

        if state is None:
            _, _, h_k, d_k = q.shape
            h_v, d_v = v.shape[-2:]
            state = mx.zeros((B, h_v, d_v, d_k), dtype=q.dtype)
        state_in = state

        if (
            mx.default_device() == mx.gpu
            and mx.metal.is_available()
            and not self.training
        ):
            if getattr(cache, "_armed", False):
                from dflash_mlx.kernels import gated_delta_kernel_with_tape

                out, state, innovation_tape = gated_delta_kernel_with_tape(
                    q, k, v, g, beta, state, mask
                )
                cache.record_tape(
                    tape=innovation_tape,
                    k=k,
                    g=g,
                    qkv=qkv,
                )
            else:
                out, state = gated_delta_mod.gated_delta_kernel(q, k, v, g, beta, state, mask)
        else:
            out, state = gated_delta_mod.gated_delta_ops(q, k, v, g, beta, state, mask)
            if getattr(cache, "_armed", False):
                decay = g[..., None, :] if g.ndim == 4 else g[..., None, None]
                decayed_state = state_in[:, None, ...] * decay
                kv_mem = (decayed_state * k[..., None, :]).sum(axis=-1)
                innovation_tape = (v - kv_mem) * beta[..., None]
                cache.record_tape(
                    tape=innovation_tape,
                    k=k,
                    g=g,
                    qkv=qkv,
                )

        cache[1] = state
        out = self.norm(out, z)
        out_flat = out.reshape(B, S, -1)
        out = self.out_proj(out_flat)

        if sharding_group is not None:
            out = mx.distributed.all_sum(out, group=sharding_group)

        return out

    cls.__call__ = speculative_call
    cls._dflash_speculative_call_installed = True


def _split_sdpa_mask(
    mask: Optional[Any],
    *,
    query_start: int,
    query_end: int,
    key_end: int,
) -> Optional[Any]:
    if mask is None or mask == "causal":
        return mask
    return mask[..., query_start:query_end, :key_end]


def _split_sdpa_output(
    *,
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    mask: Optional[Any],
    cache: Optional[Any],
    chunk_size: int,
    cached_prefix_len: int,
) -> mx.array:
    q_len = int(queries.shape[2])
    if q_len <= chunk_size:
        return scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=scale, mask=mask
        )

    outputs: list[mx.array] = []
    for start in range(0, q_len, chunk_size):
        end = min(start + chunk_size, q_len)
        key_end = cached_prefix_len + end
        chunk_mask = _split_sdpa_mask(mask, query_start=start, query_end=end, key_end=key_end)
        outputs.append(
            scaled_dot_product_attention(
                queries[:, :, start:end, :],
                keys[:, :, :key_end, :],
                values[:, :, :key_end, :],
                cache=cache,
                scale=scale,
                mask=chunk_mask,
            )
        )
    return mx.concatenate(outputs, axis=2)


_HYBRID_SDPA_EXACT_KV_THRESHOLD = 1024


def _install_split_full_attention_hook(attn: Any) -> None:
    cls = type(attn)
    if getattr(cls, "_dflash_split_full_attention_installed", False):
        return

    original_call = cls.__call__

    def split_call(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if not getattr(self, "_dflash_split_sdpa_enabled", False):
            return original_call(self, x, mask=mask, cache=cache)
        if not _attention_has_gated_q_proj(self):
            return original_call(self, x, mask=mask, cache=cache)

        B, L, _ = x.shape
        q_proj_output = self.q_proj(x)
        num_attention_heads = _attention_num_heads(self)
        num_key_value_heads = _attention_num_kv_heads(self)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, num_attention_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)

        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, num_key_value_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )

        cached_prefix_len = int(getattr(cache, "offset", 0) or 0) if cache is not None else 0
        if cache is not None:
            queries = self.rope(queries, offset=cached_prefix_len)
            keys = self.rope(keys, offset=cached_prefix_len)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        total_kv_len = int(keys.shape[2])
        exact_prefix_threshold = int(
            getattr(
                self,
                "_dflash_split_sdpa_exact_kv_threshold",
                _HYBRID_SDPA_EXACT_KV_THRESHOLD,
            )
        )
        should_split = (
            cache is not None
            and cached_prefix_len >= exact_prefix_threshold
            and (mask is None or mask == "causal" or isinstance(mask, mx.array))
        )
        should_use_batched_2pass = (
            should_split
            and int(queries.shape[2]) == 16
            and queries.dtype in (mx.bfloat16, mx.float16)
            and int(queries.shape[-1]) in (128, 256)
            and int(values.shape[-1]) in (128, 256)
        )
        if should_use_batched_2pass:
            from dflash_mlx.kernels import batched_sdpa_2pass_exact

            output = batched_sdpa_2pass_exact(
                queries=queries,
                keys=keys,
                values=values,
                scale=self.scale,
                mask=mask if isinstance(mask, mx.array) else None,
            )
            if output is None:
                output = _split_sdpa_output(
                    queries=queries,
                    keys=keys,
                    values=values,
                    scale=self.scale,
                    mask=mask,
                    cache=cache,
                    chunk_size=1,
                    cached_prefix_len=cached_prefix_len,
                )
        elif should_split:
            output = _split_sdpa_output(
                queries=queries,
                keys=keys,
                values=values,
                scale=self.scale,
                mask=mask,
                cache=cache,
                chunk_size=1,
                cached_prefix_len=cached_prefix_len,
            )
        else:
            output = scaled_dot_product_attention(
                queries, keys, values, cache=cache, scale=self.scale, mask=mask
            )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        gated_output = output * mx.sigmoid(gate)
        return self.o_proj(gated_output)

    cls.__call__ = split_call
    cls._dflash_split_full_attention_installed = True


def _install_target_speculative_hooks(target_model: Any) -> None:
    text_model = _target_text_model(target_model)
    if getattr(text_model, "_dflash_speculative_hooks_installed", False):
        return
    if detect_target_family(target_model) == "pure_attention":
        text_model._dflash_speculative_hooks_installed = True
        return
    for layer in text_model.layers:
        if getattr(layer, "is_linear", False) and hasattr(layer, "linear_attn"):
            _install_exact_small_proj_hooks(layer.linear_attn)
            _install_speculative_linear_cache_hook(layer.linear_attn)
        elif not getattr(layer, "is_linear", False) and hasattr(layer, "self_attn"):
            _install_split_full_attention_hook(layer.self_attn)
    text_model._dflash_speculative_hooks_installed = True


def configure_full_attention_split(
    target_model: Any,
    *,
    enabled: bool,
    chunk_size: int = 8,
) -> None:
    text_model = _target_text_model(target_model)
    if detect_target_family(target_model) == "pure_attention":
        return
    _install_target_speculative_hooks(target_model)
    for layer in text_model.layers:
        if not getattr(layer, "is_linear", False) and hasattr(layer, "self_attn"):
            layer.self_attn._dflash_split_sdpa_enabled = enabled
            layer.self_attn._dflash_split_sdpa_chunk_size = int(chunk_size)
            layer.self_attn._dflash_split_sdpa_exact_kv_threshold = (
                _HYBRID_SDPA_EXACT_KV_THRESHOLD
            )


def make_target_cache(
    target_model: Any,
    *,
    enable_speculative_linear_cache: bool,
    quantize_kv_cache: bool = False,
) -> list[Any]:
    text_model = _target_text_model(target_model)
    caches: list[Any] = []
    for layer_index, layer in enumerate(text_model.layers):
        if getattr(layer, "is_linear", False) and hasattr(layer, "linear_attn"):
            if enable_speculative_linear_cache:
                _install_target_speculative_hooks(target_model)
                conv_kernel_size = int(getattr(layer.linear_attn, "conv_kernel_size", 4))
                caches.append(
                    RecurrentRollbackCache(size=2, conv_kernel_size=conv_kernel_size)
                )
            else:
                caches.append(cache_mod.ArraysCache(size=2))
        else:
            if quantize_kv_cache:
                caches.append(cache_mod.QuantizedKVCache(group_size=64, bits=8))
            else:
                caches.append(cache_mod.KVCache())
    return caches


def load_target_bundle(
    model_ref: str | Path | None = None,
    *,
    lazy: bool = True,
    pack_target_weights: bool = False,
    pack_attention_weights: bool = False,
    validate_packing: bool = True,
    split_full_attention_sdpa: bool = True,
    split_full_attention_chunk_size: int = 8,
    quantize_kv_cache: bool = False,
):
    resolved_ref = resolve_model_ref(model_ref, kind="target")
    model, tokenizer, config = load(resolved_ref, lazy=lazy, return_config=True)
    target_family = detect_target_family(model)
    if target_family == "hybrid_gdn":
        _install_target_speculative_hooks(model)
        configure_full_attention_split(
            model,
            enabled=split_full_attention_sdpa and not quantize_kv_cache,
            chunk_size=split_full_attention_chunk_size,
        )
    meta = {
        "resolved_model_ref": resolved_ref,
        "config": config,
        "quantize_kv_cache": bool(quantize_kv_cache),
        "target_family": target_family,
    }
    if pack_target_weights:
        meta["packing"] = pack_target_model_weights_selective(
            model,
            validate=validate_packing,
            pack_mlp=True,
            pack_attention=pack_attention_weights,
        )
    return model, tokenizer, meta


def load_draft_bundle(
    model_ref: str | Path | None = None,
    *,
    lazy: bool = True,
    quantize_draft: bool = False,
):
    resolved_ref = resolve_model_ref(model_ref, kind="draft")
    model_path = _resolve_local_model_path(resolved_ref)
    model, config = load_model(
        model_path,
        lazy=lazy,
        get_model_classes=_get_dflash_model_classes,
    )
    quantized = _should_quantize_draft(quantize_draft)
    if quantized:
        nn.quantize(model, bits=4, group_size=64)
    return model, {
        "resolved_model_ref": str(model_ref) if model_ref is not None else str(resolved_ref),
        "config": config,
        "quantize_draft": bool(quantized),
    }


def target_forward_with_hidden_states(
    target_model: Any,
    *,
    input_ids: Optional[mx.array] = None,
    cache: Optional[list[Any]] = None,
    input_embeddings: Optional[mx.array] = None,
    capture_layer_ids: Optional[set[int]] = None,
) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
    inner = _target_text_model(target_model)
    hidden_states = input_embeddings if input_embeddings is not None else inner.embed_tokens(input_ids)
    if cache is None:
        cache = [None] * len(inner.layers)
    capture_all = capture_layer_ids is None
    if capture_all:
        captured: list[mx.array] | dict[int, mx.array] = [hidden_states]
    else:
        capture_layer_ids = set(capture_layer_ids)
        captured = {0: hidden_states} if 0 in capture_layer_ids else {}
    h = hidden_states

    if hasattr(inner, "fa_idx") and hasattr(inner, "ssm_idx"):
        fa_mask = create_attention_mask(hidden_states, cache[inner.fa_idx])
        ssm_mask = create_ssm_mask(hidden_states, cache[inner.ssm_idx])
        for layer_index, (layer, layer_cache) in enumerate(zip(inner.layers, cache, strict=True)):
            mask = ssm_mask if getattr(layer, "is_linear", False) else fa_mask
            h = layer(h, mask=mask, cache=layer_cache)
            capture_key = layer_index + 1
            if capture_all:
                captured.append(h)
            elif capture_layer_ids is not None and capture_key in capture_layer_ids:
                captured[capture_key] = h
    else:
        mask = create_attention_mask(hidden_states, cache[0])
        for layer_index, (layer, layer_cache) in enumerate(zip(inner.layers, cache, strict=True)):
            h = layer(h, mask, layer_cache)
            capture_key = layer_index + 1
            if capture_all:
                captured.append(h)
            elif capture_layer_ids is not None and capture_key in capture_layer_ids:
                captured[capture_key] = h
    normalized = inner.norm(h)
    logits = _lm_head_logits(target_model, normalized)
    return logits, captured


def trim_cache_to(cache_entries: list[Any], size: int) -> int:
    if not cache_entries:
        return 0
    current_size = int(getattr(cache_entries[0], "offset", 0) or 0)
    if current_size <= size:
        return 0
    return int(cache_mod.trim_prompt_cache(cache_entries, current_size - size) or 0)


def _arm_target_rollback(cache_entries: list[Any]) -> None:
    for cache_entry in cache_entries:
        if hasattr(cache_entry, "arm_rollback"):
            cache_entry.arm_rollback()


def _arm_target_rollback_with_prefix(
    cache_entries: list[Any],
    *,
    prefix_len: int,
) -> None:
    for cache_entry in cache_entries:
        if hasattr(cache_entry, "arm_rollback"):
            cache_entry.arm_rollback(prefix_len=int(prefix_len))


def _clear_rollback_state(cache_entry: Any) -> None:
    if hasattr(cache_entry, "clear_transients"):
        cache_entry.clear_transients()
        return
    if hasattr(cache_entry, "_armed"):
        cache_entry._armed = False
    if hasattr(cache_entry, "_tape"):
        cache_entry._tape = None
    if hasattr(cache_entry, "_tape_k"):
        cache_entry._tape_k = None
    if hasattr(cache_entry, "_tape_g"):
        cache_entry._tape_g = None
    if hasattr(cache_entry, "_tape_qkv"):
        cache_entry._tape_qkv = None
    if hasattr(cache_entry, "_snapshot"):
        cache_entry._snapshot = None


def _cleanup_generation_caches(
    target_cache: list[Any],
    draft_cache: list[Any],
) -> None:
    for cache_entry in target_cache:
        if hasattr(cache_entry, "clear_transients"):
            cache_entry.clear_transients()
    draft_cache.clear()
    target_cache.clear()


def _restore_target_cache_after_acceptance(
    cache_entries: list[Any],
    *,
    target_len: int,
    acceptance_length: int,
    drafted_tokens: int = 0,
) -> int:
    replay_ns_total = 0
    fully_accepted = drafted_tokens > 0 and acceptance_length == drafted_tokens
    for cache_entry in cache_entries:
        if hasattr(cache_entry, "rollback"):
            if fully_accepted:
                _clear_rollback_state(cache_entry)
                continue
            replay_start_ns = time.perf_counter_ns()
            cache_entry.rollback(acceptance_length)
            replay_ns_total += time.perf_counter_ns() - replay_start_ns
        elif hasattr(cache_entry, "trim"):
            offset = int(getattr(cache_entry, "offset", 0) or 0)
            if offset > target_len:
                replay_start_ns = time.perf_counter_ns()
                cache_entry.trim(offset - target_len)
                replay_ns_total += time.perf_counter_ns() - replay_start_ns
        elif hasattr(cache_entry, "offset"):
            offset = int(getattr(cache_entry, "offset", 0) or 0)
            if offset > target_len:
                cache_entry.offset = target_len
        elif hasattr(cache_entry, "crop"):
            cache_entry.crop(target_len)
    return replay_ns_total


def _verify_target_block(
    *,
    target_model: Any,
    verify_ids: mx.array,
    target_cache: list[Any],
    verify_chunk_tokens: Optional[int],
    capture_layer_ids: Optional[set[int]] = None,
) -> tuple[mx.array, list[mx.array] | dict[int, mx.array]]:
    total_tokens = int(verify_ids.shape[1])
    if total_tokens <= 0:
        raise ValueError("verify block must contain at least one token")

    chunk_size = max(1, int(verify_chunk_tokens or total_tokens))
    if chunk_size >= total_tokens:
        verify_logits, verify_hidden_states = target_forward_with_hidden_states(
            target_model,
            input_ids=verify_ids,
            cache=target_cache,
            capture_layer_ids=capture_layer_ids,
        )
        return verify_logits, verify_hidden_states

    logits_chunks: list[mx.array] = []
    hidden_state_chunks: list[list[mx.array]] | list[dict[int, mx.array]]
    hidden_state_chunks = []
    for offset in range(0, total_tokens, chunk_size):
        verify_chunk = verify_ids[:, offset : offset + chunk_size]
        chunk_logits, chunk_hidden_states = target_forward_with_hidden_states(
            target_model,
            input_ids=verify_chunk,
            cache=target_cache,
            capture_layer_ids=capture_layer_ids,
        )
        logits_chunks.append(chunk_logits)
        hidden_state_chunks.append(chunk_hidden_states)

    if capture_layer_ids is None:
        return mx.concatenate(logits_chunks, axis=1), _concat_hidden_state_chunks(hidden_state_chunks)
    return (
        mx.concatenate(logits_chunks, axis=1),
        _concat_hidden_state_chunk_dicts(hidden_state_chunks, capture_layer_ids),
    )


def generate_baseline_once(
    *,
    target_model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    use_chat_template: bool = False,
    stop_token_ids: Optional[list[int]] = None,
    suppress_token_ids: Optional[list[int]] = None,
    prompt_tokens_override: Optional[list[int]] = None,
    quantize_kv_cache: bool = False,
) -> dict[str, Any]:
    if hasattr(mx, "reset_peak_memory"):
        try:
            mx.reset_peak_memory()
        except Exception:
            pass
    prompt_tokens = (
        list(prompt_tokens_override)
        if prompt_tokens_override is not None
        else _prepare_prompt_tokens(tokenizer, prompt, use_chat_template=use_chat_template)
    )
    stop_token_ids = list(stop_token_ids or [])

    if max_new_tokens <= 0:
        return {
            "elapsed_us": 0.0,
            "prompt_token_count": len(prompt_tokens),
            "generated_token_ids": [],
            "generation_tokens": 0,
        }

    prompt_array = mx.array(prompt_tokens, dtype=mx.uint32)[None]
    cache = make_target_cache(
        target_model,
        enable_speculative_linear_cache=False,
        quantize_kv_cache=quantize_kv_cache,
    )
    start_ns = time.perf_counter_ns()

    prefill_start_ns = time.perf_counter_ns()
    logits = target_model(prompt_array, cache=cache)
    mx.eval(logits)
    prefill_ns = time.perf_counter_ns() - prefill_start_ns
    suppress_token_mask = build_suppress_token_mask(int(logits.shape[-1]), suppress_token_ids)
    next_token = int(greedy_tokens_with_mask(logits[:, -1, :], suppress_token_mask).item())
    generated_tokens = [next_token]

    while len(generated_tokens) < max_new_tokens:
        if next_token in stop_token_ids:
            break
        token_array = mx.array([[next_token]], dtype=mx.uint32)
        logits = target_model(token_array, cache=cache)
        next_token = int(greedy_tokens_with_mask(logits[:, -1, :], suppress_token_mask).item())
        generated_tokens.append(next_token)

    elapsed_us = (time.perf_counter_ns() - start_ns) / 1_000.0
    return {
        "elapsed_us": elapsed_us,
        "prefill_us": prefill_ns / 1_000.0,
        "prompt_token_count": len(prompt_tokens),
        "generated_token_ids": generated_tokens,
        "generation_tokens": len(generated_tokens),
        "peak_memory_gb": float(mx.get_peak_memory()) / 1e9 if hasattr(mx, "get_peak_memory") else None,
    }


def stream_baseline_generate(
    *,
    target_model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int,
    use_chat_template: bool = False,
    stop_token_ids: Optional[list[int]] = None,
    suppress_token_ids: Optional[list[int]] = None,
    prompt_tokens_override: Optional[list[int]] = None,
    quantize_kv_cache: bool = False,
    fallback_reason: Optional[str] = None,
) -> Iterator[dict[str, Any]]:
    prompt_tokens = (
        list(prompt_tokens_override)
        if prompt_tokens_override is not None
        else _prepare_prompt_tokens(tokenizer, prompt, use_chat_template=use_chat_template)
    )
    prompt_len = len(prompt_tokens)
    stop_token_ids = list(stop_token_ids or [])
    prompt_array = mx.array(prompt_tokens, dtype=mx.uint32)[None]
    cache = make_target_cache(
        target_model,
        enable_speculative_linear_cache=False,
        quantize_kv_cache=quantize_kv_cache,
    )
    start_ns = time.perf_counter_ns()

    prefill_start_ns = time.perf_counter_ns()
    logits = target_model(prompt_array, cache=cache)
    mx.eval(logits)
    prefill_ns = time.perf_counter_ns() - prefill_start_ns
    suppress_token_mask = build_suppress_token_mask(int(logits.shape[-1]), suppress_token_ids)
    next_token = int(greedy_tokens_with_mask(logits[:, -1, :], suppress_token_mask).item())
    generated_tokens = [next_token]

    yield {
        "event": "prefill",
        "prefill_us": prefill_ns / 1_000.0,
        "prompt_token_count": prompt_len,
        "fallback_ar": True,
        "fallback_reason": fallback_reason,
    }

    yield {
        "event": "token",
        "token_id": next_token,
        "generated_tokens": 1,
        "acceptance_ratio": 0.0,
        "cycles_completed": 0,
        "fallback_ar": True,
        "fallback_reason": fallback_reason,
    }

    while len(generated_tokens) < max_new_tokens:
        if next_token in stop_token_ids:
            break
        token_array = mx.array([[next_token]], dtype=mx.uint32)
        logits = target_model(token_array, cache=cache)
        next_token = int(greedy_tokens_with_mask(logits[:, -1, :], suppress_token_mask).item())
        generated_tokens.append(next_token)
        yield {
            "event": "token",
            "token_id": next_token,
            "generated_tokens": len(generated_tokens),
            "acceptance_ratio": 0.0,
            "cycles_completed": 0,
            "fallback_ar": True,
            "fallback_reason": fallback_reason,
        }

    elapsed_us = (time.perf_counter_ns() - start_ns) / 1_000.0
    yield {
        "event": "summary",
        "elapsed_us": elapsed_us,
        "prompt_token_count": prompt_len,
        "generated_token_ids": generated_tokens,
        "generation_tokens": len(generated_tokens),
        "accepted_from_draft": 0,
        "acceptance_ratio": 0.0,
        "cycles_completed": 0,
        "phase_timings_us": {
            "prefill": prefill_ns / 1_000.0,
            "draft": 0.0,
            "draft_prefill": 0.0,
            "draft_incremental": 0.0,
            "verify": 0.0,
            "replay": 0.0,
            "commit": 0.0,
        },
        "verify_len_cap": None,
        "fallback_ar": True,
        "fallback_reason": fallback_reason,
    }


def generate_dflash_once(
    *,
    target_model: Any,
    tokenizer: Any,
    draft_model: DFlashDraftModel,
    prompt: str,
    max_new_tokens: int,
    use_chat_template: bool = False,
    block_tokens: Optional[int] = None,
    verify_chunk_tokens: Optional[int] = None,
    stop_token_ids: Optional[list[int]] = None,
    suppress_token_ids: Optional[list[int]] = None,
    prompt_tokens_override: Optional[list[int]] = None,
    quantize_kv_cache: bool = False,
) -> dict[str, Any]:
    if hasattr(mx, "reset_peak_memory"):
        try:
            mx.reset_peak_memory()
        except Exception:
            pass
    if quantize_kv_cache:
        configure_full_attention_split(target_model, enabled=False)
    draft_sink_size, draft_window_size = _resolve_draft_window()

    prompt_tokens = (
        list(prompt_tokens_override)
        if prompt_tokens_override is not None
        else _prepare_prompt_tokens(tokenizer, prompt, use_chat_template=use_chat_template)
    )
    prompt_len = len(prompt_tokens)
    dflash_max_ctx = _resolve_dflash_max_ctx()
    if prompt_len >= dflash_max_ctx:
        fallback_reason = f"prompt_len={prompt_len} >= DFLASH_MAX_CTX={dflash_max_ctx}"
        baseline = generate_baseline_once(
            target_model=target_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            use_chat_template=use_chat_template,
            stop_token_ids=stop_token_ids,
            suppress_token_ids=suppress_token_ids,
            prompt_tokens_override=prompt_tokens,
            quantize_kv_cache=quantize_kv_cache,
        )
        baseline.update(
            {
                "accepted_from_draft": 0,
                "acceptance_ratio": 0.0,
                "cycles_completed": 0,
                "phase_timings_us": {
                    "prefill": baseline["elapsed_us"],
                    "draft": 0.0,
                    "draft_prefill": 0.0,
                    "draft_incremental": 0.0,
                    "verify": 0.0,
                    "replay": 0.0,
                    "commit": 0.0,
                },
                "speculative_linear_cache": False,
                "verify_chunk_tokens": None,
                "verify_len_cap": None,
                "quantize_kv_cache": bool(quantize_kv_cache),
                "fallback_ar": True,
                "fallback_reason": fallback_reason,
            }
        )
        return baseline
    prompt_array = mx.array(prompt_tokens, dtype=mx.uint32)[None]
    stop_token_ids = list(stop_token_ids or [])
    stop_token_array = (
        mx.array(stop_token_ids, dtype=mx.uint32) if stop_token_ids else None
    )

    # See stream_dflash_generate: force-close <think> after DFLASH_THINK_BUDGET tokens.
    think_budget = _resolve_think_budget()
    think_end_id = _resolve_think_end_id(tokenizer) if think_budget > 0 else None
    think_closed = False
    think_forced = False

    use_speculative_linear_cache = verify_chunk_tokens is None
    target_cache = make_target_cache(
        target_model,
        enable_speculative_linear_cache=use_speculative_linear_cache,
        quantize_kv_cache=quantize_kv_cache,
    )

    draft_cache = [
        ContextOnlyDraftKVCache(
            sink_size=draft_sink_size,
            window_size=draft_window_size,
        )
        for _ in range(len(draft_model.layers))
    ]
    capture_layer_ids = {int(layer_id) + 1 for layer_id in draft_model.target_layer_ids}

    try:
        start_ns = time.perf_counter_ns()
        prefill_start_ns = time.perf_counter_ns()
        prefill_step_size = 2048
        prefill_logits = None
        target_hidden_chunks: list[mx.array] = []
        for chunk_start in range(0, prompt_len, prefill_step_size):
            chunk_end = min(chunk_start + prefill_step_size, prompt_len)
            chunk_ids = prompt_array[:, chunk_start:chunk_end]
            prefill_logits, prefill_hidden_states = target_forward_with_hidden_states(
                target_model,
                input_ids=chunk_ids,
                cache=target_cache,
                capture_layer_ids=capture_layer_ids,
            )
            _eval_logits_and_captured(prefill_logits, prefill_hidden_states)
            target_hidden_chunks.append(
                extract_context_feature_from_dict(
                    prefill_hidden_states,
                    list(draft_model.target_layer_ids),
                )
            )
        prefill_ns = time.perf_counter_ns() - prefill_start_ns

        suppress_token_mask = build_suppress_token_mask(int(prefill_logits.shape[-1]), suppress_token_ids)
        staged_first = greedy_tokens_with_mask(prefill_logits[:, -1, :], suppress_token_mask).reshape(-1)
        target_hidden = (
            target_hidden_chunks[0]
            if len(target_hidden_chunks) == 1
            else mx.concatenate(target_hidden_chunks, axis=1)
        )

        draft_block_size = int(draft_model.block_size)
        requested_block_tokens = draft_block_size if block_tokens is None else int(block_tokens)
        effective_block_tokens = max(1, min(requested_block_tokens, draft_block_size))
        is_dspark = bool(getattr(draft_model, "is_dspark", False))
        # DSpark verifies [anchor] + block_len drafts; z-lab verifies the block.
        verify_width = effective_block_tokens + 1 if is_dspark else effective_block_tokens
        generated_token_buffer = mx.full((max_new_tokens,), draft_model.mask_token_id, dtype=mx.uint32)
        block_token_buffer = mx.full((effective_block_tokens,), draft_model.mask_token_id, dtype=mx.uint32)
        generated_token_count = 0
        accepted_from_draft = 0
        cycles_completed = 0
        verify_len_cap = _resolve_verify_len_cap(target_model, verify_width)
        start = prompt_len

        draft_ns_total = 0
        draft_prefill_ns = 0
        draft_incremental_ns = 0
        verify_ns_total = 0
        replay_ns_total = 0
        commit_ns_total = 0
        seen_draft_cycle = False
        acceptance_history: list[int] = []
        profile_cycles = _profile_dflash_cycles_enabled()
        cycle_profiles: list[dict[str, Any]] = []
        profile_totals_ns = {
            "draft": 0,
            "verify": 0,
            "acceptance": 0,
            "hidden_extraction": 0,
            "rollback": 0,
            "other": 0,
            "cycle_total": 0,
        }

        while generated_token_count < max_new_tokens:
            cycle_start_ns = time.perf_counter_ns()
            draft_cycle_ns = 0
            verify_cycle_ns = 0
            replay_cycle_ns = 0
            commit_cycle_ns = 0
            acceptance_cycle_ns = 0
            hidden_extract_cycle_ns = 0
            remaining = max_new_tokens - generated_token_count
            # DSpark commits up to block_len + 1 tokens (anchor + drafts).
            block_budget = remaining - 1 if is_dspark else remaining
            block_len = max(1, min(effective_block_tokens, block_budget))
            block_token_buffer[:block_len] = draft_model.mask_token_id
            block_token_buffer[:1] = staged_first
            block_token_ids = block_token_buffer[:block_len]

            if block_len > 1:
                draft_start_ns = time.perf_counter_ns()
                noise_embedding = _target_embed_tokens(target_model)(block_token_ids[None])
                draft_hidden = draft_model(
                    noise_embedding=noise_embedding,
                    target_hidden=target_hidden,
                    cache=draft_cache,
                )
                if is_dspark:
                    draft_logits = _lm_head_logits(target_model, draft_hidden)
                    dspark_drafted = _dspark_draft_block(
                        draft_model,
                        draft_logits[0],
                        staged_first,
                        suppress_token_mask,
                    )
                    mx.eval(dspark_drafted)
                else:
                    draft_logits = _lm_head_logits(target_model, draft_hidden[:, 1:, :])
                    mx.async_eval(draft_logits)
                    mx.eval(draft_logits)
                    drafted = greedy_tokens_with_mask(draft_logits, suppress_token_mask).squeeze(0)
                    block_token_ids[1:block_len] = drafted
                draft_cycle_ns = time.perf_counter_ns() - draft_start_ns
                draft_ns_total += draft_cycle_ns
                if not seen_draft_cycle:
                    draft_prefill_ns += draft_cycle_ns
                    seen_draft_cycle = True
                else:
                    draft_incremental_ns += draft_cycle_ns

            if is_dspark and block_len > 1:
                verify_token_ids = mx.concatenate(
                    [block_token_ids[:1], dspark_drafted], axis=0
                )[: min(block_len + 1, verify_len_cap)]
            else:
                verify_token_ids = block_token_ids[: min(block_len, verify_len_cap)]
            verify_ids = verify_token_ids[None]
            if use_speculative_linear_cache:
                _arm_target_rollback_with_prefix(target_cache, prefix_len=start)
            verify_start_ns = time.perf_counter_ns()
            verify_logits, verify_hidden_states = _verify_target_block(
                target_model=target_model,
                verify_ids=verify_ids,
                target_cache=target_cache,
                verify_chunk_tokens=verify_chunk_tokens,
                capture_layer_ids=capture_layer_ids,
            )
            if profile_cycles:
                _eval_logits_and_captured(verify_logits, verify_hidden_states)
            verify_cycle_ns = time.perf_counter_ns() - verify_start_ns
            verify_ns_total += verify_cycle_ns

            acceptance_start_ns = time.perf_counter_ns()
            posterior = greedy_tokens_with_mask(verify_logits[0], suppress_token_mask)
            acceptance_len = int(
                _match_acceptance_length(verify_token_ids[1:], posterior[:-1]).item()
            )
            acceptance_cycle_ns = time.perf_counter_ns() - acceptance_start_ns
            acceptance_history.append(acceptance_len)

            hidden_extract_start_ns = time.perf_counter_ns()
            committed_hidden = extract_context_feature_from_dict(
                verify_hidden_states,
                list(draft_model.target_layer_ids),
            )[:, : (1 + acceptance_len), :]
            mx.eval(committed_hidden, posterior)
            hidden_extract_cycle_ns = time.perf_counter_ns() - hidden_extract_start_ns

            commit_count = 1 + acceptance_len
            committed_segment = verify_token_ids[:commit_count]
            generated_token_buffer[generated_token_count : generated_token_count + commit_count] = committed_segment
            generated_token_count += commit_count
            accepted_from_draft += acceptance_len

            commit_start_ns = time.perf_counter_ns()
            start += commit_count
            target_hidden = committed_hidden
            replay_cycle_ns = _restore_target_cache_after_acceptance(
                target_cache,
                target_len=start,
                acceptance_length=acceptance_len,
                drafted_tokens=block_len - 1,
            )
            replay_ns_total += replay_cycle_ns
            cycles_completed += 1
            commit_wall_ns = time.perf_counter_ns() - commit_start_ns
            commit_ns_total += commit_wall_ns
            commit_cycle_ns = max(0, commit_wall_ns - replay_cycle_ns)

            stop_hit = False
            if stop_token_array is not None:
                stop_hit = bool(
                    mx.any(
                        mx.equal(
                            committed_segment[:, None],
                            stop_token_array[None, :],
                        )
                    ).item()
                )
            if stop_hit:
                break

            staged_first = posterior[acceptance_len : acceptance_len + 1]

            if think_end_id is not None and not think_closed:
                if bool(
                    mx.any(
                        mx.equal(
                            committed_segment,
                            mx.array(think_end_id, dtype=mx.uint32),
                        )
                    ).item()
                ):
                    think_closed = True
                elif not think_forced and generated_token_count >= think_budget:
                    staged_first = mx.array([think_end_id], dtype=mx.uint32)
                    think_forced = True

            if profile_cycles:
                cycle_total_ns = time.perf_counter_ns() - cycle_start_ns
                named_ns = (
                    draft_cycle_ns
                    + verify_cycle_ns
                    + acceptance_cycle_ns
                    + hidden_extract_cycle_ns
                    + replay_cycle_ns
                )
                other_cycle_ns = max(0, cycle_total_ns - named_ns)
                cycle_profiles.append(
                    {
                        "cycle": cycles_completed,
                        "block_len": int(block_len),
                        "commit_count": int(commit_count),
                        "acceptance_len": int(acceptance_len),
                        "draft_us": _ns_to_us(draft_cycle_ns),
                        "verify_us": _ns_to_us(verify_cycle_ns),
                        "acceptance_us": _ns_to_us(acceptance_cycle_ns),
                        "hidden_extraction_us": _ns_to_us(hidden_extract_cycle_ns),
                        "rollback_us": _ns_to_us(replay_cycle_ns),
                        "other_us": _ns_to_us(other_cycle_ns),
                        "cycle_total_us": _ns_to_us(cycle_total_ns),
                    }
                )
                profile_totals_ns["draft"] += draft_cycle_ns
                profile_totals_ns["verify"] += verify_cycle_ns
                profile_totals_ns["acceptance"] += acceptance_cycle_ns
                profile_totals_ns["hidden_extraction"] += hidden_extract_cycle_ns
                profile_totals_ns["rollback"] += replay_cycle_ns
                profile_totals_ns["other"] += other_cycle_ns
                profile_totals_ns["cycle_total"] += cycle_total_ns

        elapsed_us = (time.perf_counter_ns() - start_ns) / 1_000.0
        generated_token_ids = (
            generated_token_buffer[:generated_token_count].tolist()
            if generated_token_count > 0
            else []
        )
        first_20 = acceptance_history[:20]
        last_20 = acceptance_history[-20:]
        result = {
            "elapsed_us": elapsed_us,
            "prompt_token_count": prompt_len,
            "generated_token_ids": generated_token_ids,
            "generation_tokens": len(generated_token_ids),
            "accepted_from_draft": accepted_from_draft,
            "acceptance_ratio": (
                accepted_from_draft / len(generated_token_ids) if generated_token_ids else 0.0
            ),
            "block_tokens": effective_block_tokens,
            "cycles_completed": cycles_completed,
            "phase_timings_us": {
                "prefill": prefill_ns / 1_000.0,
                "draft": draft_ns_total / 1_000.0,
                "draft_prefill": draft_prefill_ns / 1_000.0,
                "draft_incremental": draft_incremental_ns / 1_000.0,
                "verify": verify_ns_total / 1_000.0,
                "replay": replay_ns_total / 1_000.0,
                "commit": commit_ns_total / 1_000.0,
            },
            "speculative_linear_cache": use_speculative_linear_cache,
            "verify_chunk_tokens": int(verify_chunk_tokens) if verify_chunk_tokens else None,
            "verify_len_cap": int(verify_len_cap),
            "quantize_kv_cache": bool(quantize_kv_cache),
            "tokens_per_cycle": (len(generated_token_ids) / cycles_completed) if cycles_completed > 0 else 0.0,
            "acceptance_first_20_avg": (sum(first_20) / len(first_20)) if first_20 else 0.0,
            "acceptance_last_20_avg": (sum(last_20) / len(last_20)) if last_20 else 0.0,
            "peak_memory_gb": float(mx.get_peak_memory()) / 1e9 if hasattr(mx, "get_peak_memory") else None,
        }
        if profile_cycles:
            result["cycle_profile_us"] = cycle_profiles
            result["cycle_profile_totals_us"] = {key: _ns_to_us(value) for key, value in profile_totals_ns.items()}
        return result
    finally:
        _cleanup_generation_caches(target_cache, draft_cache)
        del draft_cache
        del target_cache


def stream_dflash_generate(
    *,
    target_model: Any,
    tokenizer: Any,
    draft_model: DFlashDraftModel,
    prompt: str,
    max_new_tokens: int,
    use_chat_template: bool = False,
    block_tokens: Optional[int] = None,
    stop_token_ids: Optional[list[int]] = None,
    suppress_token_ids: Optional[list[int]] = None,
    prompt_tokens_override: Optional[list[int]] = None,
    quantize_kv_cache: bool = False,
    cache_snapshot: Optional[dict[str, Any]] = None,
    emit_cache_snapshot: bool = False,
    commit_boundary: Optional[int] = None,
    ddtree_mode: str = "off",
    tree_budget: int = 4,
) -> Iterator[dict[str, Any]]:
    if quantize_kv_cache:
        configure_full_attention_split(target_model, enabled=False)
    draft_sink_size, draft_window_size = _resolve_draft_window()

    prompt_tokens = (
        list(prompt_tokens_override)
        if prompt_tokens_override is not None
        else _prepare_prompt_tokens(tokenizer, prompt, use_chat_template=use_chat_template)
    )
    fallback_ar = False
    fallback_reason: Optional[str] = None

    prompt_len = len(prompt_tokens)
    dflash_max_ctx = _resolve_dflash_max_ctx()
    if prompt_len >= dflash_max_ctx:
        fallback_reason = f"prompt_len={prompt_len} >= DFLASH_MAX_CTX={dflash_max_ctx}"
        yield from stream_baseline_generate(
            target_model=target_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            use_chat_template=use_chat_template,
            stop_token_ids=stop_token_ids,
            suppress_token_ids=suppress_token_ids,
            prompt_tokens_override=prompt_tokens,
            quantize_kv_cache=quantize_kv_cache,
            fallback_reason=fallback_reason,
        )
        return
    prompt_array = mx.array(prompt_tokens, dtype=mx.uint32)[None]
    stop_token_ids = list(stop_token_ids or [])
    stop_token_array = (
        mx.array(stop_token_ids, dtype=mx.uint32) if stop_token_ids else None
    )

    # DFLASH_THINK_BUDGET: force-close an open <think> block after N generated
    # tokens by staging </think> as the next committed anchor. The forced token
    # flows through verify like a sampled one, so caches stay coherent.
    think_budget = _resolve_think_budget()
    think_end_id = _resolve_think_end_id(tokenizer) if think_budget > 0 else None
    think_closed = False
    think_forced = False

    capture_layer_ids = {int(layer_id) + 1 for layer_id in draft_model.target_layer_ids}

    try:
        start_ns = time.perf_counter_ns()
        prefill_start_ns = time.perf_counter_ns()

        # commit_boundary = end of the stable conversation content (no generation-prompt
        # suffix tokens like Qwen's trailing `<think>\n`). Snapshot is taken HERE, not at
        # prompt_len, so future turns whose tokens match the committed prefix can hit.
        if commit_boundary is None or commit_boundary < 0 or commit_boundary > prompt_len:
            commit_boundary = prompt_len

        cached_len = (
            int(cache_snapshot.get("prompt_len", 0))
            if cache_snapshot is not None
            else 0
        )
        if cached_len > prompt_len:
            # Stale snapshot — treat as cold.
            cache_snapshot = None
            cached_len = 0

        # Fast path: snapshot matches the full request exactly (only possible when the
        # stored snapshot key equals prompt_len, i.e., raw-completion round-trip).
        if cache_snapshot is not None and cached_len == prompt_len:
            target_cache = cache_snapshot["target_cache"]
            target_hidden = cache_snapshot["target_hidden"]
            staged_first = cache_snapshot["staged_first"]
            suppress_token_mask = cache_snapshot["suppress_token_mask"]
            draft_cache = cache_snapshot["draft_cache"]
            prefill_ns = int(cache_snapshot.get("prefill_ns", 0))
            yield {
                "event": "prefill",
                "prefill_us": prefill_ns / 1_000.0,
                "prompt_token_count": prompt_len,
                "cached_prefill": True,
                "cached_len": prompt_len,
            }
        else:
            if cache_snapshot is not None:
                target_cache = cache_snapshot["target_cache"]
                cached_target_hidden = cache_snapshot["target_hidden"]
                draft_cache = cache_snapshot["draft_cache"]
                cached_suppress_mask = cache_snapshot.get("suppress_token_mask")
                cached_staged_first = cache_snapshot.get("staged_first")
            else:
                target_cache = make_target_cache(
                    target_model,
                    enable_speculative_linear_cache=True,
                    quantize_kv_cache=quantize_kv_cache,
                )
                draft_cache = [
                    ContextOnlyDraftKVCache(
                        sink_size=draft_sink_size,
                        window_size=draft_window_size,
                    )
                    for _ in range(len(draft_model.layers))
                ]
                cached_target_hidden = None
                cached_suppress_mask = None
                cached_staged_first = None

            prefill_step_size = 2048
            phase1_end = max(cached_len, min(commit_boundary, prompt_len))

            # Phase 1: forward on [cached_len : phase1_end]. Extends target_cache to
            # phase1_end (== commit_boundary when prompt has a generation-prompt suffix).
            phase1_logits = None
            phase1_hidden_chunks: list[mx.array] = []
            for chunk_start in range(cached_len, phase1_end, prefill_step_size):
                chunk_end = min(chunk_start + prefill_step_size, phase1_end)
                chunk_ids = prompt_array[:, chunk_start:chunk_end]
                phase1_logits, phase1_hidden_states = target_forward_with_hidden_states(
                    target_model,
                    input_ids=chunk_ids,
                    cache=target_cache,
                    capture_layer_ids=capture_layer_ids,
                )
                _eval_logits_and_captured(phase1_logits, phase1_hidden_states)
                phase1_hidden_chunks.append(
                    extract_context_feature_from_dict(
                        phase1_hidden_states,
                        list(draft_model.target_layer_ids),
                    )
                )
                yield {
                    "event": "prefill_progress",
                    "tokens_processed": chunk_end,
                    "tokens_total": prompt_len,
                }

            # committed_hidden represents target-hidden features through phase1_end.
            if phase1_hidden_chunks:
                phase1_hidden = (
                    phase1_hidden_chunks[0]
                    if len(phase1_hidden_chunks) == 1
                    else mx.concatenate(phase1_hidden_chunks, axis=1)
                )
                committed_hidden = (
                    mx.concatenate([cached_target_hidden, phase1_hidden], axis=1)
                    if cached_target_hidden is not None
                    else phase1_hidden
                )
            else:
                committed_hidden = cached_target_hidden  # may be None

            # Resolve suppress mask: cache > fresh phase1 logits > deferred.
            if cached_suppress_mask is not None:
                commit_suppress_mask = cached_suppress_mask
            elif phase1_logits is not None:
                commit_suppress_mask = build_suppress_token_mask(
                    int(phase1_logits.shape[-1]), suppress_token_ids
                )
            else:
                commit_suppress_mask = None

            if phase1_logits is not None:
                staged_first_at_commit = greedy_tokens_with_mask(
                    phase1_logits[:, -1, :], commit_suppress_mask
                ).reshape(-1)
            else:
                staged_first_at_commit = cached_staged_first

            # Emit snapshot at the commit boundary — the stable, generation-prompt-free
            # state that future turns of the same conversation can prefix-match against.
            # Server deep-copies on receipt (before phase 2 mutates target_cache).
            did_phase1 = phase1_end > cached_len
            # Snapshot iff phase 1 did new work — that guarantees phase1_logits is
            # populated, so committed_hidden and staged_first_at_commit are non-None.
            # commit_suppress_mask may legitimately be None (no suppression requested);
            # downstream greedy_tokens_with_mask handles None mask fine.
            if emit_cache_snapshot and did_phase1:
                yield {
                    "event": "cache_snapshot",
                    "target_cache": target_cache,
                    "target_hidden": committed_hidden,
                    "staged_first": staged_first_at_commit,
                    "suppress_token_mask": commit_suppress_mask,
                    "draft_cache": draft_cache,
                    "prefill_ns": int(time.perf_counter_ns() - prefill_start_ns),
                    "prompt_len": phase1_end,
                }

            # Phase 2: forward on [phase1_end : prompt_len]. Ephemeral — extends cache
            # through the generation-prompt suffix but never gets snapshotted (those
            # tokens diverge from what later turns see in the middle of a conversation).
            phase2_logits = None
            phase2_hidden_chunks: list[mx.array] = []
            for chunk_start in range(phase1_end, prompt_len, prefill_step_size):
                chunk_end = min(chunk_start + prefill_step_size, prompt_len)
                chunk_ids = prompt_array[:, chunk_start:chunk_end]
                phase2_logits, phase2_hidden_states = target_forward_with_hidden_states(
                    target_model,
                    input_ids=chunk_ids,
                    cache=target_cache,
                    capture_layer_ids=capture_layer_ids,
                )
                _eval_logits_and_captured(phase2_logits, phase2_hidden_states)
                phase2_hidden_chunks.append(
                    extract_context_feature_from_dict(
                        phase2_hidden_states,
                        list(draft_model.target_layer_ids),
                    )
                )
                yield {
                    "event": "prefill_progress",
                    "tokens_processed": chunk_end,
                    "tokens_total": prompt_len,
                }

            prefill_ns = time.perf_counter_ns() - prefill_start_ns

            if phase2_hidden_chunks:
                phase2_hidden = (
                    phase2_hidden_chunks[0]
                    if len(phase2_hidden_chunks) == 1
                    else mx.concatenate(phase2_hidden_chunks, axis=1)
                )
                target_hidden = (
                    mx.concatenate([committed_hidden, phase2_hidden], axis=1)
                    if committed_hidden is not None
                    else phase2_hidden
                )
                final_logits = phase2_logits
            else:
                target_hidden = committed_hidden
                final_logits = phase1_logits

            if commit_suppress_mask is not None:
                suppress_token_mask = commit_suppress_mask
            elif final_logits is not None:
                suppress_token_mask = build_suppress_token_mask(
                    int(final_logits.shape[-1]), suppress_token_ids
                )
            else:
                suppress_token_mask = build_suppress_token_mask(0, suppress_token_ids)

            if final_logits is not None:
                staged_first = greedy_tokens_with_mask(
                    final_logits[:, -1, :], suppress_token_mask
                ).reshape(-1)
            else:
                staged_first = staged_first_at_commit

            yield {
                "event": "prefill",
                "prefill_us": prefill_ns / 1_000.0,
                "prompt_token_count": prompt_len,
                "cached_prefill": cache_snapshot is not None,
                "cached_len": int(cached_len),
            }

        draft_block_size = int(draft_model.block_size)
        requested_block_tokens = draft_block_size if block_tokens is None else int(block_tokens)
        effective_block_tokens = max(1, min(requested_block_tokens, draft_block_size))
        is_dspark = bool(getattr(draft_model, "is_dspark", False))
        # DSpark verifies [anchor] + block_len drafts; z-lab verifies the block.
        verify_width = effective_block_tokens + 1 if is_dspark else effective_block_tokens
        block_token_buffer = mx.full((effective_block_tokens,), draft_model.mask_token_id, dtype=mx.uint32)
        generated_token_ids: list[int] = []
        accepted_from_draft = 0
        cycles_completed = 0
        verify_len_cap = _resolve_verify_len_cap(target_model, verify_width)
        start = prompt_len

        draft_ns_total = 0
        draft_prefill_ns = 0
        draft_incremental_ns = 0
        verify_ns_total = 0
        replay_ns_total = 0
        commit_ns_total = 0
        seen_draft_cycle = False
        profile_cycles = _profile_dflash_cycles_enabled()
        cycle_profiles: list[dict[str, Any]] = []
        profile_totals_ns = {
            "draft": 0,
            "verify": 0,
            "acceptance": 0,
            "hidden_extraction": 0,
            "rollback": 0,
            "other": 0,
            "cycle_total": 0,
        }

        # DDTree adaptive routing
        _ddtree_available = _ensure_ddtree_imports() if ddtree_mode in ("adaptive", "always") else False
        # DDTree builds its tree from z-lab-shaped draft logits; DSpark drafters
        # use shifted semantics, so keep them on the single-chain path.
        _ddtree_active = (
            ddtree_mode in ("adaptive", "always") and _ddtree_available and not is_dspark
        )
        if ddtree_mode in ("adaptive", "always") and not _ddtree_available:
            sys.stderr.write(
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} [dflash] WARNING: "
                f"ddtree_mode={ddtree_mode!r} but ddtree-mlx is not installed. "
                f"Falling back to DFlash-only.\n"
            )
            sys.stderr.flush()
        _ddtree_threshold = _resolve_ddtree_acceptance_threshold()
        _ddtree_cycles = 0
        _dflash_cycles = 0
        _acceptance_window: list[int] = []

        while len(generated_token_ids) < max_new_tokens:
            cycle_start_ns = time.perf_counter_ns()
            draft_cycle_ns = 0
            verify_cycle_ns = 0
            replay_cycle_ns = 0
            commit_cycle_ns = 0
            acceptance_cycle_ns = 0
            hidden_extract_cycle_ns = 0
            remaining = max_new_tokens - len(generated_token_ids)
            # DSpark commits up to block_len + 1 tokens (anchor + drafts).
            block_budget = remaining - 1 if is_dspark else remaining
            block_len = max(1, min(effective_block_tokens, block_budget))
            block_token_buffer[:block_len] = draft_model.mask_token_id
            block_token_buffer[:1] = staged_first
            block_token_ids = block_token_buffer[:block_len]

            if block_len > 1:
                draft_start_ns = time.perf_counter_ns()
                noise_embedding = _target_embed_tokens(target_model)(block_token_ids[None])
                draft_hidden = draft_model(
                    noise_embedding=noise_embedding,
                    target_hidden=target_hidden,
                    cache=draft_cache,
                )
                if is_dspark:
                    draft_logits = _lm_head_logits(target_model, draft_hidden)
                    dspark_drafted = _dspark_draft_block(
                        draft_model,
                        draft_logits[0],
                        staged_first,
                        suppress_token_mask,
                    )
                    mx.eval(dspark_drafted)
                else:
                    draft_logits = _lm_head_logits(target_model, draft_hidden[:, 1:, :])
                    mx.async_eval(draft_logits)
                    mx.eval(draft_logits)
                    drafted = greedy_tokens_with_mask(draft_logits, suppress_token_mask).squeeze(0)
                    block_token_ids[1:block_len] = drafted
                draft_cycle_ns = time.perf_counter_ns() - draft_start_ns
                draft_ns_total += draft_cycle_ns
                if not seen_draft_cycle:
                    draft_prefill_ns += draft_cycle_ns
                    seen_draft_cycle = True
                else:
                    draft_incremental_ns += draft_cycle_ns

            # --- Per-cycle DDTree routing decision ---
            _use_ddtree_this_cycle = False
            if _ddtree_active:
                if ddtree_mode == "always":
                    _use_ddtree_this_cycle = True
                elif ddtree_mode == "adaptive" and len(_acceptance_window) >= 2:
                    _recent_avg = sum(_acceptance_window[-5:]) / len(_acceptance_window[-5:])
                    _use_ddtree_this_cycle = _recent_avg < _ddtree_threshold * effective_block_tokens

            if _use_ddtree_this_cycle and block_len > 1:
                # ── DDTree path: tree_build → tree_verify → tree_walk → commit ──
                # Key: use tree_aware_linear=True so recurrent (GatedDeltaNet)
                # layers fork state per branch. Without this, recurrent state
                # is corrupted and output degenerates.
                _ddtree_cycles += 1
                root_token = int(staged_first.item())

                verify_start_ns = time.perf_counter_ns()
                # Do NOT arm rollback when tree-aware — tree_verify_forward
                # manages recurrent state internally via per-node forking.
                tree = _ddtree_imports["build_tree"](draft_logits[0], budget=tree_budget)
                compiled = _ddtree_imports["compile_tree"](tree, root_token, prefix_len=start)

                _tree_cache_state: dict[str, Any] = {}
                tree_logits, tree_hidden_states = _ddtree_imports["tree_verify_forward"](
                    target_model,
                    compiled_tree=compiled,
                    cache=target_cache,
                    capture_layer_ids=capture_layer_ids,
                    tree_aware_linear=True,
                    tree_cache_state=_tree_cache_state,
                )
                mx.eval(tree_logits)
                verify_cycle_ns = time.perf_counter_ns() - verify_start_ns
                verify_ns_total += verify_cycle_ns

                acceptance_start_ns = time.perf_counter_ns()
                posterior = greedy_tokens_with_mask(tree_logits[0], suppress_token_mask)
                posterior_list = posterior.tolist()
                accepted_indices, bonus_token = _ddtree_imports["follow_verified_tree"](
                    tree.child_maps, posterior_list
                )
                acceptance_len = len(accepted_indices) - 1  # exclude root (it's the staged_first)
                acceptance_cycle_ns = time.perf_counter_ns() - acceptance_start_ns

                hidden_extract_start_ns = time.perf_counter_ns()
                all_hidden = extract_context_feature_from_dict(
                    tree_hidden_states,
                    list(draft_model.target_layer_ids),
                )
                accepted_idx_array = mx.array(accepted_indices, dtype=mx.int32)
                committed_hidden = all_hidden[:, accepted_idx_array, :]
                mx.eval(committed_hidden)
                hidden_extract_cycle_ns = time.perf_counter_ns() - hidden_extract_start_ns

                commit_count = len(accepted_indices)
                accepted_token_ids = _ddtree_imports["tree_token_ids"](tree, root_token, accepted_indices)
                committed_segment = mx.array(accepted_token_ids, dtype=mx.uint32)

                commit_start_ns = time.perf_counter_ns()
                # tree_aware_path_commit installs correct recurrent state for
                # the accepted path and packs attention KV entries.
                _ddtree_imports["tree_aware_path_commit"](
                    target_cache,
                    prefix_len=start,
                    accepted_indices=accepted_indices,
                    tree_cache_state=_tree_cache_state,
                )
                start += commit_count
                target_hidden = committed_hidden
                replay_cycle_ns = 0
                replay_ns_total += replay_cycle_ns
                cycles_completed += 1
                commit_wall_ns = time.perf_counter_ns() - commit_start_ns
                commit_ns_total += commit_wall_ns
                commit_cycle_ns = max(0, commit_wall_ns - replay_cycle_ns)

                accepted_from_draft += acceptance_len
                _acceptance_window.append(acceptance_len)
                committed_ids = accepted_token_ids
                staged_first = mx.array([bonus_token], dtype=mx.uint32)

            else:
                # ── DFlash path: single-chain verify (existing, unchanged) ──
                _dflash_cycles += 1
                if is_dspark and block_len > 1:
                    verify_token_ids = mx.concatenate(
                        [block_token_ids[:1], dspark_drafted], axis=0
                    )[: min(block_len + 1, verify_len_cap)]
                else:
                    verify_token_ids = block_token_ids[: min(block_len, verify_len_cap)]
                verify_ids = verify_token_ids[None]
                _arm_target_rollback_with_prefix(target_cache, prefix_len=start)
                verify_start_ns = time.perf_counter_ns()
                verify_logits, verify_hidden_states = _verify_target_block(
                    target_model=target_model,
                    verify_ids=verify_ids,
                    target_cache=target_cache,
                    verify_chunk_tokens=None,
                    capture_layer_ids=capture_layer_ids,
                )
                if profile_cycles:
                    _eval_logits_and_captured(verify_logits, verify_hidden_states)
                verify_cycle_ns = time.perf_counter_ns() - verify_start_ns
                verify_ns_total += verify_cycle_ns

                acceptance_start_ns = time.perf_counter_ns()
                posterior = greedy_tokens_with_mask(verify_logits[0], suppress_token_mask)
                acceptance_len = int(
                    _match_acceptance_length(verify_token_ids[1:], posterior[:-1]).item()
                )
                acceptance_cycle_ns = time.perf_counter_ns() - acceptance_start_ns
                hidden_extract_start_ns = time.perf_counter_ns()
                committed_hidden = extract_context_feature_from_dict(
                    verify_hidden_states,
                    list(draft_model.target_layer_ids),
                )[:, : (1 + acceptance_len), :]
                mx.eval(committed_hidden, posterior)
                hidden_extract_cycle_ns = time.perf_counter_ns() - hidden_extract_start_ns

                commit_count = 1 + acceptance_len
                committed_segment = verify_token_ids[:commit_count]
                commit_start_ns = time.perf_counter_ns()
                start += commit_count
                target_hidden = committed_hidden
                replay_cycle_ns = _restore_target_cache_after_acceptance(
                    target_cache,
                    target_len=start,
                    acceptance_length=acceptance_len,
                    drafted_tokens=int(verify_token_ids.shape[0]) - 1,
                )
                replay_ns_total += replay_cycle_ns
                cycles_completed += 1
                commit_wall_ns = time.perf_counter_ns() - commit_start_ns
                commit_ns_total += commit_wall_ns
                commit_cycle_ns = max(0, commit_wall_ns - replay_cycle_ns)

                accepted_from_draft += acceptance_len
                _acceptance_window.append(acceptance_len)
                committed_ids = [int(token_id) for token_id in committed_segment.tolist()]
                staged_first = posterior[acceptance_len : acceptance_len + 1]

            # ── Shared: yield tokens, stop detection, profiling ──
            for token_id in committed_ids:
                if token_id is None:
                    continue
                if len(generated_token_ids) >= max_new_tokens:
                    break
                generated_token_ids.append(token_id)
                yield {
                    "event": "token",
                    "token_id": token_id,
                    "generated_tokens": len(generated_token_ids),
                    "acceptance_ratio": (
                        accepted_from_draft / len(generated_token_ids) if generated_token_ids else 0.0
                    ),
                    "cycles_completed": cycles_completed,
                }

            stop_hit = False
            if stop_token_array is not None:
                stop_hit = bool(
                    mx.any(
                        mx.equal(
                            committed_segment[:, None],
                            stop_token_array[None, :],
                        )
                    ).item()
                )
            if stop_hit:
                break

            if think_end_id is not None and not think_closed:
                if think_end_id in committed_ids:
                    think_closed = True
                elif not think_forced and len(generated_token_ids) >= think_budget:
                    staged_first = mx.array([think_end_id], dtype=mx.uint32)
                    think_forced = True

            if profile_cycles:
                cycle_total_ns = time.perf_counter_ns() - cycle_start_ns
                named_ns = (
                    draft_cycle_ns
                    + verify_cycle_ns
                    + acceptance_cycle_ns
                    + hidden_extract_cycle_ns
                    + replay_cycle_ns
                )
                other_cycle_ns = max(0, cycle_total_ns - named_ns)
                cycle_profiles.append(
                    {
                        "cycle": cycles_completed,
                        "block_len": int(block_len),
                        "commit_count": int(commit_count),
                        "acceptance_len": int(acceptance_len),
                        "draft_us": _ns_to_us(draft_cycle_ns),
                        "verify_us": _ns_to_us(verify_cycle_ns),
                        "acceptance_us": _ns_to_us(acceptance_cycle_ns),
                        "hidden_extraction_us": _ns_to_us(hidden_extract_cycle_ns),
                        "rollback_us": _ns_to_us(replay_cycle_ns),
                        "other_us": _ns_to_us(other_cycle_ns),
                        "cycle_total_us": _ns_to_us(cycle_total_ns),
                    }
                )
                profile_totals_ns["draft"] += draft_cycle_ns
                profile_totals_ns["verify"] += verify_cycle_ns
                profile_totals_ns["acceptance"] += acceptance_cycle_ns
                profile_totals_ns["hidden_extraction"] += hidden_extract_cycle_ns
                profile_totals_ns["rollback"] += replay_cycle_ns
                profile_totals_ns["other"] += other_cycle_ns
                profile_totals_ns["cycle_total"] += cycle_total_ns

        elapsed_us = (time.perf_counter_ns() - start_ns) / 1_000.0
        summary = {
            "event": "summary",
            "elapsed_us": elapsed_us,
            "prompt_token_count": prompt_len,
            "generated_token_ids": generated_token_ids,
            "generation_tokens": len(generated_token_ids),
            "accepted_from_draft": accepted_from_draft,
            "acceptance_ratio": (
                accepted_from_draft / len(generated_token_ids) if generated_token_ids else 0.0
            ),
            "block_tokens": effective_block_tokens,
            "cycles_completed": cycles_completed,
            "phase_timings_us": {
                "prefill": prefill_ns / 1_000.0,
                "draft": draft_ns_total / 1_000.0,
                "draft_prefill": draft_prefill_ns / 1_000.0,
                "draft_incremental": draft_incremental_ns / 1_000.0,
                "verify": verify_ns_total / 1_000.0,
                "replay": replay_ns_total / 1_000.0,
                "commit": commit_ns_total / 1_000.0,
            },
            "verify_len_cap": int(verify_len_cap),
        }
        if _ddtree_active:
            summary["ddtree_mode"] = ddtree_mode
            summary["ddtree_cycles"] = _ddtree_cycles
            summary["dflash_cycles"] = _dflash_cycles
        if profile_cycles:
            summary["cycle_profile_us"] = cycle_profiles
            summary["cycle_profile_totals_us"] = {
                key: _ns_to_us(value) for key, value in profile_totals_ns.items()
            }
        yield summary
    finally:
        _cleanup_generation_caches(target_cache, draft_cache)
        del draft_cache
        del target_cache
