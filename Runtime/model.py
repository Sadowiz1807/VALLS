"""
VSAD Local Model & Multi-turn Context Inference Runtime.
Hỗ trợ đưa lịch sử hội thoại (context) và trạng thái hệ thống (state) vào mô hình.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_model
from tokenizers import Tokenizer
from torch import Tensor

# Constants
NONE = "NONE"
ABSENT = "ABSENT"
INPUT_SPAN = "INPUT_SPAN"
STATE_REFERENCE = "STATE_REFERENCE"


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.embedding(token_ids) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int, dropout: float) -> None:
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-(math.log(10_000.0) / d_model))
        )
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        self.max_length = max_length
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=True)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs):
        key = prefix + "encoding"
        if key in state_dict:
            ckpt_encoding = state_dict[key]
            if ckpt_encoding.shape != self.encoding.shape:
                state_dict[key] = self.encoding.clone()
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.size(1) > self.max_length:
            raise ValueError("Sequence exceeds configured maximum length")
        return self.dropout(
            hidden_states + self.encoding[:, : hidden_states.size(1)].to(dtype=hidden_states.dtype)
        )


class LayerNormalization(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        mean = hidden_states.mean(dim=-1, keepdim=True)
        variance = hidden_states.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * (hidden_states - mean) / torch.sqrt(variance + self.eps) + self.bias


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.heads = heads
        self.d_k = d_model // heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        batch_size = q.size(0)
        q_heads = self.query(q).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        k_heads = self.key(k).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        v_heads = self.value(v).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))
        attention = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(attention, v_heads).transpose(1, 2).contiguous()
        return self.output(context.view(batch_size, -1, self.heads * self.d_k))


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.layers(hidden_states)


class ResidualBlock(nn.Module):
    def __init__(self, sublayer: nn.Module, d_model: int) -> None:
        super().__init__()
        self.sublayer = sublayer
        self.norm = LayerNormalization(d_model)

    def forward(self, hidden_states: Tensor, *args: Any) -> Tensor:
        return hidden_states + self.sublayer(self.norm(hidden_states), *args)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attention = MultiHeadAttention(d_model, heads, dropout)
        self.attention_residual = ResidualBlock(self.attention, d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.feed_forward_residual = ResidualBlock(self.feed_forward, d_model)

    def forward(self, hidden_states: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        hidden_states = self.attention_residual(hidden_states, hidden_states, hidden_states, mask)
        return self.feed_forward_residual(hidden_states)


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, d_model: int, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(self, hidden_states: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        if mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(1).unsqueeze(2)
        for layer in self.layers:
            hidden_states = layer(hidden_states, mask)
        return self.norm(hidden_states)


class DecoderBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, heads, dropout)
        self.self_residual = ResidualBlock(self.self_attention, d_model)
        self.cross_attention = MultiHeadAttention(d_model, heads, dropout)
        self.cross_residual = ResidualBlock(self.cross_attention, d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.feed_forward_residual = ResidualBlock(self.feed_forward, d_model)

    def forward(self, hidden_states: Tensor, memory: Tensor, self_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None) -> Tensor:
        hidden_states = self.self_residual(hidden_states, hidden_states, hidden_states, self_mask)
        hidden_states = self.cross_residual(hidden_states, memory, memory, memory_mask)
        return self.feed_forward_residual(hidden_states)


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, d_model: int, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization(d_model)

    def forward(self, hidden_states: Tensor, memory: Tensor, self_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None) -> Tensor:
        if memory_mask is not None and memory_mask.dim() == 2:
            memory_mask = memory_mask.unsqueeze(1).unsqueeze(2)
        for layer in self.layers:
            hidden_states = layer(hidden_states, memory, self_mask, memory_mask)
        return self.norm(hidden_states)


class MultiTaskTransformer(nn.Module):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        model = config["model"]
        d_model = model["d_model"]
        self.config = config
        self.embedding = TokenEmbedding(config["tokenizer"]["vocab_size"], d_model)
        self.position = PositionalEncoding(d_model, model["max_input_length"], model["dropout"])
        self.encoder = Encoder(
            nn.ModuleList(EncoderBlock(d_model, model["attention_heads"], model["d_ff"], model["dropout"]) for _ in range(model["encoder_layers"])),
            d_model, model.get("gradient_checkpointing", False)
        )
        self.response_position = PositionalEncoding(d_model, model["max_response_length"], model["dropout"])
        self.decoder = Decoder(
            nn.ModuleList(DecoderBlock(d_model, model["attention_heads"], model["d_ff"], model["dropout"]) for _ in range(model["decoder_layers"])),
            d_model, model.get("gradient_checkpointing", False)
        )
        self.response_head = nn.Linear(d_model, config["tokenizer"]["vocab_size"], bias=False)
        if model.get("tie_response_embeddings", True):
            self.response_head.weight = self.embedding.embedding.weight

        self.act_head = nn.Linear(d_model, len(config["labels"]["acts"]))
        self.goal_head = nn.Linear(d_model, len(config["labels"]["goals"]))
        self.categorical_heads = nn.ModuleDict({k: nn.Linear(d_model, len(v)) for k, v in config["heads"]["categorical"].items()})
        self.span_start_heads = nn.ModuleDict({k: nn.Linear(d_model, 1) for k in config["heads"]["spans"]})
        self.span_end_heads = nn.ModuleDict({k: nn.Linear(d_model, 1) for k in config["heads"]["spans"]})

    def encode(self, input_ids: Tensor, input_mask: Tensor) -> Tensor:
        return self.encoder(self.position(self.embedding(input_ids)), input_mask)

    def forward(self, input_ids: Tensor, input_mask: Tensor, response_input_ids: Optional[Tensor] = None) -> Dict[str, Any]:
        hidden = self.encode(input_ids, input_mask)
        pooled = hidden[:, 0]
        outputs: Dict[str, Any] = {
            "act_logits": self.act_head(pooled),
            "goal_logits": self.goal_head(pooled),
            "categorical_logits": {k: head(pooled) for k, head in self.categorical_heads.items()},
            "span_start_logits": {k: head(hidden).squeeze(-1) for k, head in self.span_start_heads.items()},
            "span_end_logits": {k: head(hidden).squeeze(-1) for k, head in self.span_end_heads.items()},
        }
        if response_input_ids is not None:
            seq_len = response_input_ids.size(1)
            causal_mask = torch.tril(torch.ones((seq_len, seq_len), device=input_ids.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(1)
            dec_hidden = self.decoder(self.response_position(self.embedding(response_input_ids)), hidden, causal_mask, input_mask)
            outputs["response_logits"] = self.response_head(dec_hidden)
        return outputs


def build_model(config: dict[str, Any]) -> MultiTaskTransformer:
    return MultiTaskTransformer(config)


def serialize_model_turn(
    text: str,
    context: Optional[Sequence[Dict[str, Any]]],
    state: Optional[Dict[str, Any]],
    special_tokens: Dict[str, str]
) -> str:
    ctx_json = json.dumps(context or [], ensure_ascii=False, separators=(",", ":"))
    state_json = json.dumps(state or {}, ensure_ascii=False, separators=(",", ":"))
    meta_json = json.dumps({"asr_noise": "CLEAN", "language_mode": "MIXED", "locale": "vi-VN"}, ensure_ascii=False, separators=(",", ":"))
    return "".join([
        special_tokens["context_open"], ctx_json, special_tokens["context_close"],
        special_tokens["state_open"], state_json, special_tokens["state_close"],
        special_tokens["metadata_open"], meta_json, special_tokens["metadata_close"],
        special_tokens["input_open"], text, special_tokens["input_close"],
    ])


class VSADModel:
    def __init__(self, model_dir: Path | str, device: Optional[torch.device] = None):
        self.model_dir = Path(model_dir)
        self.config = json.loads((self.model_dir / "config.json").read_text(encoding="utf-8"))
        self.config["model"]["max_response_length"] = 1024
        self.tokenizer = Tokenizer.from_file(str(self.model_dir / "tokenizer.json"))
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = build_model(self.config).to(self.device)
        weight_path = self.model_dir / "VASD.safetensors" if (self.model_dir / "VASD.safetensors").exists() else self.model_dir / "model.safetensors"
        load_model(self.model, str(weight_path), strict=False, device=str(self.device))
        self.model.eval()

    def infer(self, text: str, context: Optional[Sequence[Dict[str, Any]]] = None, state: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        special_tokens = self.config["tokenizer"]["special_tokens"]
        serialized = serialize_model_turn(text, context, state, special_tokens)
        encoding = self.tokenizer.encode(serialized)
        
        input_ids = torch.tensor([encoding.ids], dtype=torch.long, device=self.device)
        input_mask = torch.ones_like(input_ids, dtype=torch.float32, device=self.device)
        offsets = encoding.offsets
        
        with torch.no_grad():
            outputs = self.model(input_ids, input_mask)
            act_idx = torch.argmax(outputs["act_logits"][0]).item()
            goal_idx = torch.argmax(outputs["goal_logits"][0]).item()
            
            act = self.config["labels"]["acts"][act_idx]
            goal = self.config["labels"]["goals"][goal_idx]
            
            params: Dict[str, Any] = {}
            if goal != NONE and goal in self.config["ontology"]["goal_parameters"]:
                spec = self.config["ontology"]["goal_parameters"][goal]["properties"]
                for name, p_spec in spec.items():
                    key = f"{goal}__{name}"
                    if key in outputs["categorical_logits"]:
                        cat_idx = torch.argmax(outputs["categorical_logits"][key][0]).item()
                        cat_val = self.config["heads"]["categorical"][key][cat_idx]
                        if cat_val != ABSENT:
                            params[name] = cat_val
                    if key in outputs["span_start_logits"]:
                        s_idx = torch.argmax(outputs["span_start_logits"][key][0]).item()
                        e_idx = torch.argmax(outputs["span_end_logits"][key][0]).item()
                        if s_idx <= e_idx < len(offsets):
                            c_start = offsets[s_idx][0]
                            c_end = offsets[e_idx][1]
                            val = serialized[c_start:c_end]
                            if val and c_start >= serialized.find(special_tokens["input_open"]):
                                params[name] = {"source": "input_span", "start": c_start, "end": c_end, "value": val}

            # Autoregressive Response Generation
            bos_id = self.tokenizer.token_to_id(special_tokens["bos"])
            eos_id = self.tokenizer.token_to_id(special_tokens["eos"])
            resp_ids = [bos_id]
            
            hidden = self.model.encode(input_ids, input_mask)
            for _ in range(self.config["model"]["max_response_length"]):
                curr_t = torch.tensor([resp_ids], dtype=torch.long, device=self.device)
                seq_l = curr_t.size(1)
                c_mask = torch.tril(torch.ones((seq_l, seq_l), device=self.device, dtype=torch.bool)).unsqueeze(0).unsqueeze(1)
                dec_h = self.model.decoder(self.model.response_position(self.model.embedding(curr_t)), hidden, c_mask, input_mask)
                logits = self.model.response_head(dec_h)[:, -1, :]
                next_id = torch.argmax(logits, dim=-1).item()
                if next_id == eos_id:
                    break
                resp_ids.append(next_id)

            resp_text = self.tokenizer.decode(resp_ids[1:]).strip()

        return {
            "act": act,
            "goal": goal,
            "parameters": params,
            "response": resp_text
        }
