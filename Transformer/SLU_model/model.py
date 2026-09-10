"""VALLS SLU V1 neural model.

The model understands the current audio turn and emits an untrusted semantic
contract. Runtime context, graph/memory, policy, tools, and execution belong to
the Harness and are intentionally absent here.
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .config import (
    CONTEXT_REFERENCE_TYPES,
    OPERATION_RELATIONS,
    ROOT_ACTION_DOMAINS,
    TURN_RELATIONS,
    capability_schemas,
    canonical_action_id,
    validate_config,
)


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, values: Tensor) -> Tensor:
        return F.layer_norm(values, (values.size(-1),), self.weight, self.bias, self.eps)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 4096) -> None:
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model))
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(positions * frequencies[: encoding[:, 1::2].shape[1]])
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=True)

    def forward(self, values: Tensor) -> Tensor:
        if values.size(1) > self.encoding.size(1):
            raise ValueError("sequence exceeds positional encoding capacity")
        return values + self.encoding[:, : values.size(1)].to(values)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % heads:
            raise ValueError("d_model must be divisible by attention heads")
        self.d_model = d_model
        self.heads = heads
        self.head_size = d_model // heads
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split(self, values: Tensor) -> Tensor:
        batch, length, _ = values.shape
        return values.view(batch, length, self.heads, self.head_size).transpose(1, 2)

    def forward(self, queries: Tensor, keys: Tensor, values: Tensor, mask: Tensor | None = None) -> Tensor:
        query = self._split(self.query(queries))
        key = self._split(self.key(keys))
        value = self._split(self.value(values))
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_size)
        if mask is not None:
            if mask.dim() == 2:
                mask = mask[:, None, None, :]
            elif mask.dim() != 4:
                raise ValueError("attention mask must have rank 2 or 4")
            scores = scores.masked_fill(~mask.bool(), torch.finfo(scores.dtype).min)
        attended = self.dropout(scores.softmax(dim=-1)) @ value
        attended = attended.transpose(1, 2).contiguous().view(queries.size(0), queries.size(1), self.d_model)
        return self.output(attended)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, heads, dropout)
        self.ffn_norm = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, mask: Tensor | None) -> Tensor:
        normalised = self.attention_norm(values)
        values = values + self.dropout(self.attention(normalised, normalised, normalised, mask))
        return values + self.dropout(self.ffn(self.ffn_norm(values)))


class ConformerConvModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        if kernel_size <= 1 or kernel_size % 2 == 0:
            raise ValueError("Conformer kernel must be odd and greater than one")
        self.norm = LayerNorm(d_model)
        self.pointwise_in = nn.Conv1d(d_model, 2 * d_model, 1)
        self.depthwise = nn.Conv1d(d_model, d_model, kernel_size, padding=kernel_size // 2, groups=d_model)
        self.channel_norm = LayerNorm(d_model)
        self.pointwise_out = nn.Conv1d(d_model, d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, mask: Tensor | None) -> Tensor:
        values = self.pointwise_in(self.norm(values).transpose(1, 2))
        values = F.glu(values, dim=1)
        values = self.depthwise(values).transpose(1, 2)
        values = F.silu(self.channel_norm(values)).transpose(1, 2)
        values = self.pointwise_out(values).transpose(1, 2)
        if mask is not None:
            values = values * mask.unsqueeze(-1).to(values.dtype)
        return self.dropout(values)


class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, heads: int, d_ff: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        self.ffn1_norm = LayerNorm(d_model)
        self.ffn1 = FeedForward(d_model, d_ff, dropout)
        self.attention_norm = LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, heads, dropout)
        self.attention_dropout = nn.Dropout(dropout)
        self.conv = ConformerConvModule(d_model, kernel_size, dropout)
        self.ffn2_norm = LayerNorm(d_model)
        self.ffn2 = FeedForward(d_model, d_ff, dropout)
        self.final_norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, mask: Tensor | None) -> Tensor:
        def masked(item: Tensor) -> Tensor:
            return item * mask.unsqueeze(-1).to(item.dtype) if mask is not None else item

        values = masked(values + 0.5 * self.dropout(self.ffn1(self.ffn1_norm(values))))
        normalised = self.attention_norm(values)
        values = masked(values + self.attention_dropout(self.attention(normalised, normalised, normalised, mask)))
        values = masked(values + self.conv(values, mask))
        values = masked(values + 0.5 * self.dropout(self.ffn2(self.ffn2_norm(values))))
        return self.final_norm(masked(values))


class TransformerEncoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, d_model: int, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(d_model)
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, values: Tensor, mask: Tensor | None) -> Tensor:
        for layer in self.layers:
            if mask is not None:
                values = values * mask.unsqueeze(-1).to(values.dtype)
            if self.training and self.gradient_checkpointing:
                values = checkpoint(lambda states, current=layer: current(states, mask), values, use_reentrant=False)
            else:
                values = layer(values, mask)
            if mask is not None:
                values = values * mask.unsqueeze(-1).to(values.dtype)
        return self.norm(values)


class LogMelFrontend(nn.Module):
    def __init__(self, audio: Mapping[str, Any]) -> None:
        super().__init__()
        self.n_fft = int(audio["n_fft"])
        self.hop_length = int(audio["hop_length"])
        self.win_length = int(audio["win_length"])
        self.feature_dim = int(audio["feature_dim"])
        self.sample_rate = int(audio["sample_rate"])
        self.f_min = float(audio.get("f_min", 0.0))
        self.f_max = float(audio.get("f_max", self.sample_rate / 2))
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)
        self.register_buffer("mel_filterbank", self._build_filterbank(), persistent=True)

    def _build_filterbank(self) -> Tensor:
        def hz_to_mel(value: Tensor) -> Tensor:
            return 2595 * torch.log10(1 + value / 700)
        def mel_to_hz(value: Tensor) -> Tensor:
            return 700 * (10 ** (value / 2595) - 1)
        low, high = hz_to_mel(torch.tensor(self.f_min)), hz_to_mel(torch.tensor(min(self.f_max, self.sample_rate / 2)))
        points = mel_to_hz(torch.linspace(low, high, self.feature_dim + 2))
        bins = torch.floor((self.n_fft + 1) * points / self.sample_rate).long()
        filters = torch.zeros(self.feature_dim, self.n_fft // 2 + 1)
        for index in range(self.feature_dim):
            left, centre, right = bins[index:index + 3].tolist()
            left = max(0, min(left, filters.size(1) - 1))
            centre = max(left + 1, min(centre, filters.size(1) - 1))
            right = max(centre + 1, min(right, filters.size(1)))
            filters[index, left:centre] = torch.linspace(0, 1, centre - left)
            filters[index, centre:right] = torch.linspace(1, 0, right - centre)
        return filters

    def _extract_one(self, waveform: Tensor) -> tuple[Tensor, int]:
        original_length = int(waveform.numel())
        if original_length < self.win_length:
            waveform = F.pad(waveform, (0, self.win_length - original_length))
        spectrum = torch.stft(
            waveform, n_fft=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
            window=self.window.to(waveform), center=True, return_complex=True,
        )
        power = spectrum.abs().square()
        features = torch.log(torch.einsum("mf,ft->tm", self.mel_filterbank.to(power), power).clamp_min(1e-5))
        features = (features - features.mean()).div(features.std().clamp_min(1e-5))
        valid_frames = min(features.size(0), max(1, original_length // self.hop_length + 1))
        return features, valid_frames

    def forward(self, waveform: Tensor, audio_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if waveform.dim() != 2 or waveform.size(1) < 2:
            raise ValueError("waveform must have shape [batch, samples]")
        if audio_mask is None:
            lengths = torch.full((waveform.size(0),), waveform.size(1), dtype=torch.long, device=waveform.device)
        else:
            if audio_mask.shape != waveform.shape:
                raise ValueError("audio_mask must have shape [batch, samples]")
            lengths = audio_mask.long().sum(1).clamp_min(2)
        extracted = [self._extract_one(waveform[row, : int(lengths[row])].float()) for row in range(waveform.size(0))]
        maximum = max(values.size(0) for values, _ in extracted)
        features = waveform.new_zeros(waveform.size(0), maximum, self.feature_dim)
        mask = torch.zeros(waveform.size(0), maximum, dtype=torch.bool, device=waveform.device)
        for row, (values, valid_frames) in enumerate(extracted):
            features[row, : values.size(0)] = values
            mask[row, :valid_frames] = True
        return features, mask


class ConvSubsampling(nn.Module):
    def __init__(self, feature_dim: int, d_model: int, factor: int) -> None:
        super().__init__()
        if factor != 4:
            raise ValueError("V1 currently requires a 4x convolutional subsampler")
        channels = max(8, d_model // 2)
        frequency = math.ceil(math.ceil(feature_dim / 2) / 2)
        self.network = nn.Sequential(
            nn.Conv2d(1, channels, 3, stride=2, padding=1), nn.GELU(),
            nn.Conv2d(channels, channels, 3, stride=2, padding=1), nn.GELU(),
        )
        self.projection = nn.Linear(channels * frequency, d_model)

    def forward(self, features: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        values = self.network(features.unsqueeze(1))
        batch, channels, frames, frequency = values.shape
        values = self.projection(values.transpose(1, 2).contiguous().view(batch, frames, channels * frequency))
        reduced_mask = F.interpolate(mask.float().unsqueeze(1), size=frames, mode="nearest").squeeze(1).bool()
        return values * reduced_mask.unsqueeze(-1).to(values.dtype), reduced_mask


class SpeechEncoder(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        audio, speech = config["audio"], config["speech_encoder"]
        d_model = int(speech["d_model"])
        self.frontend = LogMelFrontend(audio)
        self.subsampling = ConvSubsampling(int(audio["feature_dim"]), d_model, int(speech["subsampling_factor"]))
        self.position = PositionalEncoding(d_model)
        block_type = speech["architecture"]
        block = ConformerBlock if block_type == "conformer" else TransformerBlock
        self.encoder = TransformerEncoder(
            nn.ModuleList(
                block(d_model, int(speech["attention_heads"]), int(speech["d_ff"]), int(speech["conv_kernel_size"]), float(speech["dropout"]))
                if block_type == "conformer" else block(d_model, int(speech["attention_heads"]), int(speech["d_ff"]), float(speech["dropout"]))
                for _ in range(int(speech["layers"]))
            ), d_model, bool(config["model"].get("gradient_checkpointing", False)),
        )

    def forward(self, waveform: Tensor, audio_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        features, mask = self.frontend(waveform, audio_mask)
        values, mask = self.subsampling(features, mask)
        return self.encoder(self.position(values), mask), mask


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: Tensor, memory: Tensor, mask: Tensor | None) -> Tensor:
        queries = self.norm(queries)
        return queries + self.dropout(self.attention(queries, memory, memory, mask))


class SemanticResampler(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        section = config["semantic_resampler"]
        d_model = int(section["d_model"])
        self.latent_tokens = int(section["latent_tokens"])
        self.queries = nn.Parameter(torch.empty(1, self.latent_tokens, d_model))
        nn.init.normal_(self.queries, std=d_model ** -0.5)
        self.cross_attention = CrossAttention(d_model, int(section["attention_heads"]), float(section["dropout"]))
        self.norm = LayerNorm(d_model)

    def forward(self, speech_states: Tensor, speech_mask: Tensor) -> tuple[Tensor, Tensor]:
        queries = self.queries.expand(speech_states.size(0), -1, -1)
        values = self.norm(self.cross_attention(queries, speech_states, speech_mask))
        return values, torch.ones(values.size(0), values.size(1), dtype=torch.bool, device=values.device)


class SemanticCore(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        section = config["semantic_core"]
        d_model = int(section["d_model"])
        self.encoder = TransformerEncoder(
            nn.ModuleList(TransformerBlock(d_model, int(section["attention_heads"]), int(section["d_ff"]), float(section["dropout"])) for _ in range(int(section["layers"]))),
            d_model, bool(config["model"].get("gradient_checkpointing", False)),
        )

    def forward(self, values: Tensor, mask: Tensor) -> Tensor:
        return self.encoder(values, mask)


class OperationDecoder(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        section = config["operation_decoder"]
        d_model = int(section["d_model"])
        self.max_operations = int(section["max_operations"])
        self.queries = nn.Parameter(torch.empty(1, self.max_operations, d_model))
        nn.init.normal_(self.queries, std=d_model ** -0.5)
        self.cross_attention = CrossAttention(d_model, int(section["attention_heads"]), float(section["dropout"]))
        self.norm = LayerNorm(d_model)
        self.presence_head = nn.Linear(d_model, 1)

    def forward(self, values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        queries = self.queries.expand(values.size(0), -1, -1)
        queries = self.norm(self.cross_attention(queries, values, mask))
        return queries, self.presence_head(queries).squeeze(-1)


class SchemaTextEncoder(nn.Module):
    """Small trainable text encoder with fixed hash buckets and dynamic length."""

    def __init__(self, d_model: int, hash_buckets: int = 8192, max_tokens: int = 256) -> None:
        super().__init__()
        self.hash_buckets = hash_buckets
        self.max_tokens = max_tokens
        self.embedding = nn.Embedding(hash_buckets, d_model)
        self.projection = nn.Linear(d_model, d_model)
        self.norm = LayerNorm(d_model)

    def _ids(self, text: str) -> list[int]:
        words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        tokens = words + [f"{word[:n]}" for word in words for n in (2, 3) if len(word) >= n]
        tokens = tokens[: self.max_tokens] or ["<empty>"]
        return [int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "little") % self.hash_buckets for token in tokens]

    def encode_texts(self, texts: Sequence[str], device: torch.device | None = None) -> Tensor:
        device = device or self.embedding.weight.device
        if not texts:
            return torch.empty(0, self.embedding.embedding_dim, device=device)
        rows = [self._ids(text) for text in texts]
        width = max(len(row) for row in rows)
        ids = torch.zeros(len(rows), width, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        for row, values in enumerate(rows):
            ids[row, : len(values)] = torch.tensor(values, dtype=torch.long, device=device)
            mask[row, : len(values)] = True
        values = self.embedding(ids)
        values = (values * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1).to(values.dtype)
        return F.normalize(self.norm(self.projection(values)), dim=-1)

    @staticmethod
    def action_text(domain: str, capability: Mapping[str, Any], action_path: str, action: Mapping[str, Any]) -> str:
        parameters = " ".join(f"parameter {name} type {spec.get('type', '')} {spec.get('description', '')}" for name, spec in action.get("parameters", {}).items())
        return " ".join([domain, str(capability.get("description", "")), canonical_action_id(domain, action_path), str(action.get("description", "")), parameters])

    def schema_text(self, schema: Mapping[str, Any]) -> str:
        actions_value = schema.get("actions", {})
        if isinstance(actions_value, Mapping):
            actions = " ".join(self.action_text(str(schema["name"]), schema, path, action) for path, action in actions_value.items())
        else:
            actions = " ".join(f"{schema['name']} action {action}" for action in actions_value)
        return f"{schema['name']} {schema.get('description', '')} {actions}"

    def encode_schemas(self, schemas: Sequence[Mapping[str, Any]], device: torch.device | None = None) -> Tensor:
        return self.encode_texts([self.schema_text(schema) for schema in schemas], device)


class CapabilitySchemaRetriever(nn.Module):
    def __init__(self, schema_encoder: SchemaTextEncoder, d_model: int) -> None:
        super().__init__()
        self.schema_encoder = schema_encoder
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, queries: Tensor, schemas: Sequence[Mapping[str, Any]], embeddings: Tensor | None = None) -> Tensor:
        if not schemas:
            raise ValueError("runtime capability schema set cannot be empty")
        embeddings = self.schema_encoder.encode_schemas(schemas, queries.device) if embeddings is None else embeddings.to(queries)
        if embeddings.dim() != 2 or embeddings.size(0) != len(schemas) or embeddings.size(1) != queries.size(-1):
            raise ValueError("schema embeddings must have shape [dynamic_schema_count, d_model]")
        return F.normalize(queries, dim=-1) @ F.normalize(embeddings, dim=-1).transpose(0, 1) / self.temperature.abs().clamp_min(1e-3)


class ActionResolver(nn.Module):
    def __init__(self, schema_encoder: SchemaTextEncoder, d_model: int) -> None:
        super().__init__()
        self.schema_encoder = schema_encoder
        self.query = nn.Linear(d_model, d_model)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, queries: Tensor, schemas: Sequence[Mapping[str, Any]]) -> list[Tensor]:
        queries = F.normalize(self.query(queries), dim=-1)
        scores: list[Tensor] = []
        for schema in schemas:
            action_value = schema.get("actions", {})
            if not isinstance(action_value, Mapping):
                raise ValueError(f"{schema.get('name')}.actions must be an action mapping")
            actions = list(action_value.items())
            if not actions:
                raise ValueError(f"schema {schema.get('name')} has no actions")
            embeddings = self.schema_encoder.encode_texts(
                [self.schema_encoder.action_text(str(schema["name"]), schema, path, action) for path, action in actions],
                queries.device,
            )
            scores.append(queries @ embeddings.transpose(0, 1) / self.temperature.abs().clamp_min(1e-3))
        return scores


class EnumExtractor(nn.Module):
    def __init__(self, schema_encoder: SchemaTextEncoder, d_model: int) -> None:
        super().__init__()
        self.schema_encoder = schema_encoder
        self.query = nn.Linear(d_model, d_model)

    def forward(self, values: Tensor, choices: Sequence[str]) -> Tensor:
        embeddings = self.schema_encoder.encode_texts([str(choice) for choice in choices], values.device)
        return F.normalize(self.query(values), dim=-1) @ embeddings.transpose(0, 1)


class NumberExtractor(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values).squeeze(-1)


class BooleanExtractor(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.head = nn.Linear(d_model, 2)

    def forward(self, values: Tensor) -> Tensor:
        return self.head(values)


class SpanExtractor(nn.Module):
    SOURCES = ["INPUT_SPAN", "STATE_REFERENCE"]

    def __init__(self, schema_encoder: SchemaTextEncoder, d_model: int) -> None:
        super().__init__()
        self.schema_encoder = schema_encoder
        self.source_head = nn.Linear(d_model, len(self.SOURCES))
        self.span_query = nn.Linear(d_model, d_model)
        self.span_key = nn.Linear(d_model, d_model)
        self.reference_query = nn.Linear(d_model, d_model)

    def forward(self, values: Tensor, span_states: Tensor, reference_paths: Sequence[str]) -> dict[str, Tensor]:
        query = self.span_query(values)
        result = {
            "source_logits": self.source_head(values),
            "span_start_logits": torch.einsum("bod,btd->bot", query, self.span_key(span_states)),
            "span_end_logits": torch.einsum("bod,btd->bot", query, self.span_key(span_states)),
        }
        if reference_paths:
            refs = self.schema_encoder.encode_texts(list(reference_paths), values.device)
            result["reference_logits"] = F.normalize(self.reference_query(values), dim=-1) @ refs.transpose(0, 1)
        else:
            result["source_logits"][..., self.SOURCES.index("STATE_REFERENCE")] = torch.finfo(values.dtype).min
        return result


class ContextReferenceExtractor(nn.Module):
    def __init__(self, schema_encoder: SchemaTextEncoder, d_model: int) -> None:
        super().__init__()
        self.schema_encoder = schema_encoder
        self.query = nn.Linear(d_model, d_model)

    def forward(self, values: Tensor, reference_types: Sequence[str]) -> Tensor:
        if not reference_types:
            raise ValueError("CONTEXT_REFERENCE requires reference types")
        refs = self.schema_encoder.encode_texts(list(reference_types), values.device)
        return F.normalize(self.query(values), dim=-1) @ refs.transpose(0, 1)


class TypedParameterExtractor(nn.Module):
    """Shared type extractors conditioned by action and parameter schemas."""

    def __init__(self, schema_encoder: SchemaTextEncoder, d_model: int) -> None:
        super().__init__()
        self.schema_encoder = schema_encoder
        self.conditioner = nn.Sequential(nn.Linear(3 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.presence_head = nn.Linear(d_model, 2)
        self.extractors = nn.ModuleDict({
            "ENUM": EnumExtractor(schema_encoder, d_model),
            "NUMBER": NumberExtractor(d_model),
            "BOOLEAN": BooleanExtractor(d_model),
            "ENTITY": SpanExtractor(schema_encoder, d_model),
            "FREE_TEXT": SpanExtractor(schema_encoder, d_model),
            "STATE_REFERENCE": SpanExtractor(schema_encoder, d_model),
            "CONTEXT_REFERENCE": ContextReferenceExtractor(schema_encoder, d_model),
        })

    def forward(self, operation_queries: Tensor, speech_states: Tensor, schemas: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, dict[str, dict[str, Tensor]]]]:
        result: dict[str, dict[str, dict[str, dict[str, Tensor]]]] = {}
        for schema in schemas:
            domain = str(schema["name"])
            result[domain] = {}
            actions_value = schema.get("actions", {})
            if not isinstance(actions_value, Mapping):
                raise ValueError(f"{schema.get('name')}.actions must be an action mapping")
            actions = actions_value.items()
            for action_path, action in actions:
                action_embedding = self.schema_encoder.encode_texts(
                    [self.schema_encoder.action_text(domain, schema, action_path, action)], operation_queries.device
                )[0]
                action_conditioned = torch.cat([operation_queries, action_embedding.view(1, 1, -1).expand_as(operation_queries)], dim=-1)
                result[domain][action_path] = {}
                action_parameters = action.get("parameters", {})
                for name, parameter in action_parameters.items():
                    parameter_embedding = self.schema_encoder.encode_texts(
                        [f"{domain} {action_path} parameter {name} type {parameter['type']} {parameter.get('description', '')}"], operation_queries.device
                    )[0]
                    conditioned = self.conditioner(torch.cat([action_conditioned, parameter_embedding.view(1, 1, -1).expand_as(operation_queries)], dim=-1))
                    output: dict[str, Tensor] = {"presence_logits": self.presence_head(conditioned)}
                    extractor = self.extractors[str(parameter["type"])]
                    if parameter["type"] == "ENUM":
                        output["enum_logits"] = extractor(conditioned, parameter["values"])
                    elif parameter["type"] == "NUMBER":
                        output["number_value"] = extractor(conditioned)
                    elif parameter["type"] == "BOOLEAN":
                        output["boolean_logits"] = extractor(conditioned)
                    elif parameter["type"] == "CONTEXT_REFERENCE":
                        output["context_reference_logits"] = extractor(conditioned, parameter.get("context_reference_types", CONTEXT_REFERENCE_TYPES))
                    else:
                        output.update(extractor(conditioned, speech_states, parameter.get("state_reference_paths", [])))
                    result[domain][action_path][name] = output
        return result


class BridgeAlignmentHead(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))

    def forward(self, values: Tensor) -> Tensor:
        return F.normalize(self.projection(values), dim=-1)


class LexicalSpanResolver:
    def __init__(self, vocabulary: Sequence[str], blank_id: int = 0) -> None:
        if len(vocabulary) < 257 or vocabulary[blank_id] != "<BLANK>":
            raise ValueError("lexical vocabulary must contain blank plus byte tokens")
        self.vocabulary = list(vocabulary)
        self.blank_id = int(blank_id)

    def resolve(self, logits: Tensor, start_frame: int, end_frame: int) -> str:
        if logits.dim() != 2 or not logits.size(0):
            raise ValueError("lexical logits must have shape [frames, vocabulary]")
        start = max(0, min(int(start_frame), logits.size(0) - 1))
        end = max(start, min(int(end_frame), logits.size(0) - 1))
        values: list[int] = []
        previous: int | None = None
        for token in logits[start:end + 1].argmax(-1).tolist():
            token = int(token)
            if token == self.blank_id or token == previous:
                previous = token
                continue
            if not 1 <= token <= 256:
                raise ValueError(f"invalid lexical token id: {token}")
            values.append(token - 1)
            previous = token
        return bytes(values).decode("utf-8", errors="replace").strip()


class VoiceNativeSLU(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        validate_config(config)
        self.config = dict(config)
        d_model = int(config["semantic_core"]["d_model"])
        self.schema_encoder = SchemaTextEncoder(d_model)
        self.speech_encoder = SpeechEncoder(config)
        self.semantic_resampler = SemanticResampler(config)
        self.semantic_core = SemanticCore(config)
        self.operation_decoder = OperationDecoder(config)
        self.schema_retriever = CapabilitySchemaRetriever(self.schema_encoder, d_model)
        self.action_resolver = ActionResolver(self.schema_encoder, d_model)
        self.parameter_extractor = TypedParameterExtractor(self.schema_encoder, d_model)
        self.act_head = nn.Linear(d_model, len(config["ontology"]["acts"]))
        self.turn_relation_head = nn.Linear(d_model, len(config["relations"]["turn"]))
        self.context_required_head = nn.Linear(d_model, 1)
        self.context_reference_head = nn.Linear(d_model, len(config["relations"]["context_reference_types"]))
        self.operation_relation_head = nn.Linear(2 * d_model, len(config["relations"]["operation"]))
        self.confidence_head = nn.Linear(d_model, 3)
        self.ood_head = nn.Linear(d_model, 1)
        self.bridge_alignment_head = BridgeAlignmentHead(d_model)
        lexical = config["lexical_branch"]
        self.lexical_span_resolver = LexicalSpanResolver(lexical["vocab"], int(lexical["blank_id"]))
        self.lexical_head = nn.Linear(d_model, int(lexical["vocab_size"])) if lexical.get("enabled", True) else None
        self.apply(self._reset_parameters)

    @staticmethod
    def _reset_parameters(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
        weights = mask.to(values.dtype).unsqueeze(-1)
        return (values * weights).sum(1) / weights.sum(1).clamp_min(1.0)

    def default_schemas(self) -> list[dict[str, Any]]:
        return capability_schemas(self.config)

    def encode_audio(self, waveform: Tensor, audio_mask: Tensor | None = None) -> dict[str, Tensor]:
        speech_states, speech_mask = self.speech_encoder(waveform, audio_mask)
        semantic_latents, semantic_mask = self.semantic_resampler(speech_states, speech_mask)
        semantic_states = self.semantic_core(semantic_latents, semantic_mask)
        return {"speech_states": speech_states, "speech_mask": speech_mask, "semantic_latents": semantic_latents, "semantic_mask": semantic_mask, "semantic_states": semantic_states}

    def forward(
        self,
        waveform: Tensor,
        audio_mask: Tensor | None = None,
        capability_schemas: Sequence[Mapping[str, Any]] | None = None,
        capability_embeddings: Tensor | None = None,
        bridge_only: bool = False,
    ) -> dict[str, Any]:
        schemas = self.default_schemas() if capability_schemas is None else [dict(schema) for schema in capability_schemas]
        if not schemas:
            raise ValueError("capability_schemas cannot be empty")
        if bridge_only:
            speech_states, speech_mask = self.speech_encoder(waveform, audio_mask)
            semantic_latents, semantic_mask = self.semantic_resampler(speech_states, speech_mask)
            return {"speech_states": speech_states, "speech_mask": speech_mask, "semantic_latents": semantic_latents, "semantic_mask": semantic_mask, "bridge_states": self.bridge_alignment_head(semantic_latents)}
        encoded = self.encode_audio(waveform, audio_mask)
        semantic_states, semantic_mask = encoded["semantic_states"], encoded["semantic_mask"]
        pooled = self._masked_mean(semantic_states, semantic_mask)
        operation_queries, presence = self.operation_decoder(semantic_states, semantic_mask)
        pairwise = torch.cat([
            operation_queries[:, :, None, :].expand(-1, -1, operation_queries.size(1), -1),
            operation_queries[:, None, :, :].expand(-1, operation_queries.size(1), -1, -1),
        ], dim=-1)
        outputs: dict[str, Any] = {
            **encoded,
            "pooled_semantic": pooled,
            "act_logits": self.act_head(pooled),
            "turn_relation_logits": self.turn_relation_head(pooled),
            "context_required_logit": self.context_required_head(pooled).squeeze(-1),
            "context_reference_logits": self.context_reference_head(pooled),
            "operation_relation_logits": self.operation_relation_head(pairwise),
            "confidence_logits": self.confidence_head(pooled),
            "ood_logit": self.ood_head(pooled).squeeze(-1),
            "operation_queries": operation_queries,
            "operation_presence_logits": presence,
            "schema_names": [str(schema["name"]) for schema in schemas],
            "capability_schemas": schemas,
            "goal_scores": self.schema_retriever(operation_queries, schemas, capability_embeddings),
            "action_scores": self.action_resolver(operation_queries, schemas),
            "parameter_outputs": self.parameter_extractor(operation_queries, encoded["speech_states"], schemas),
            "bridge_states": self.bridge_alignment_head(encoded["semantic_latents"]),
        }
        if self.lexical_head is not None:
            outputs["lexical_ctc_logits"] = self.lexical_head(encoded["speech_states"])
            outputs["bridge_lexical_logits"] = self.lexical_head(encoded["semantic_latents"])
        return outputs


SpeechSemanticAdapter = SemanticResampler
GoalQueryGenerator = OperationDecoder
MultiTaskTransformer = VoiceNativeSLU


def build_model(config: Mapping[str, Any]) -> VoiceNativeSLU:
    validate_config(config)
    return VoiceNativeSLU(config)
