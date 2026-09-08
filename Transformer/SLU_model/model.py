"""Neural modules for the voice-native VALLS SLU Model V0.

The model owns audio-to-semantic understanding only.  Capability schemas and
parameter descriptions are encoded at runtime, so adding a capability does not
add a classifier row or a goal-specific parameter module to the checkpoint.
Execution and response generation remain outside this module.
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

from .config import capability_schemas, validate_config


class LayerNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, hidden_states: Tensor) -> Tensor:
        return F.layer_norm(
            hidden_states,
            (hidden_states.size(-1),),
            self.weight,
            self.bias,
            self.eps,
        )


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_length: int = 4096) -> None:
        super().__init__()
        positions = torch.arange(max_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        encoding = torch.zeros(max_length, d_model)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        encoding[:, 1::2] = torch.cos(
            positions * frequencies[: encoding[:, 1::2].shape[1]]
        )
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=True)

    def forward(self, hidden_states: Tensor) -> Tensor:
        length = hidden_states.size(1)
        if length > self.encoding.size(1):
            raise ValueError("sequence exceeds positional encoding capacity")
        return hidden_states + self.encoding[:, :length].to(hidden_states)


class MultiHeadAttention(nn.Module):
    """Self/cross attention with a boolean valid-key mask."""

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

    @staticmethod
    def _normalise_mask(mask: Tensor | None, batch: int, keys: int) -> Tensor | None:
        if mask is None:
            return None
        if mask.dim() == 2:
            if mask.shape != (batch, keys):
                raise ValueError("attention mask must have shape [batch, key_length]")
            return mask[:, None, None, :].bool()
        if mask.dim() == 4:
            return mask.bool()
        raise ValueError("attention mask must have rank 2 or 4")

    def forward(
        self,
        query_states: Tensor,
        key_states: Tensor,
        value_states: Tensor,
        key_mask: Tensor | None = None,
    ) -> Tensor:
        query = self._split(self.query(query_states))
        key = self._split(self.key(key_states))
        value = self._split(self.value(value_states))
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_size)
        valid = self._normalise_mask(key_mask, query_states.size(0), key_states.size(1))
        if valid is not None:
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
        attended = self.dropout(scores.softmax(dim=-1)) @ value
        attended = attended.transpose(1, 2).contiguous().view(
            query_states.size(0), query_states.size(1), self.d_model
        )
        return self.output(attended)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        return self.network(hidden_states)


class TransformerBlock(nn.Module):
    """Pre-norm transformer block used by the semantic core."""

    def __init__(self, d_model: int, heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm_attention = LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, heads, dropout)
        self.norm_feed_forward = LayerNorm(d_model)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor, mask: Tensor | None) -> Tensor:
        normalised = self.norm_attention(hidden_states)
        hidden_states = hidden_states + self.dropout(
            self.attention(normalised, normalised, normalised, mask)
        )
        hidden_states = hidden_states + self.dropout(
            self.feed_forward(self.norm_feed_forward(hidden_states))
        )
        return hidden_states


class ConformerConvModule(nn.Module):
    """Depthwise convolution module from a Conformer block."""

    def __init__(self, d_model: int, kernel_size: int, dropout: float) -> None:
        super().__init__()
        if kernel_size <= 1 or kernel_size % 2 == 0:
            raise ValueError("Conformer convolution kernel must be odd and greater than one")
        self.norm = LayerNorm(d_model)
        self.pointwise_in = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)
        self.depthwise = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=d_model,
        )
        # LayerNorm is batch-size independent and remains valid for one-frame
        # utterances during smoke tests and small-batch acoustic warm-up.
        self.batch_norm = LayerNorm(d_model)
        self.pointwise_out = nn.Conv1d(d_model, d_model, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: Tensor, mask: Tensor | None) -> Tensor:
        values = self.norm(hidden_states)
        values = self.pointwise_in(values.transpose(1, 2))
        values = F.glu(values, dim=1)
        values = self.depthwise(values).transpose(1, 2)
        values = self.batch_norm(values).transpose(1, 2)
        values = F.silu(values)
        values = self.pointwise_out(values).transpose(1, 2)
        if mask is not None:
            values = values * mask.unsqueeze(-1).to(values.dtype)
        return self.dropout(values)


class ConformerBlock(nn.Module):
    """Macaron-style FFN -> MHSA -> convolution -> FFN speech block."""

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

    def forward(self, hidden_states: Tensor, mask: Tensor | None) -> Tensor:
        def apply_mask(values: Tensor) -> Tensor:
            if mask is not None:
                return values * mask.unsqueeze(-1).to(values.dtype)
            return values

        hidden_states = apply_mask(
            hidden_states + 0.5 * self.dropout(self.ffn1(self.ffn1_norm(hidden_states)))
        )
        normalised = self.attention_norm(hidden_states)
        hidden_states = apply_mask(
            hidden_states + self.attention_dropout(
                self.attention(normalised, normalised, normalised, mask)
            )
        )
        hidden_states = apply_mask(hidden_states + self.conv(hidden_states, mask))
        hidden_states = apply_mask(
            hidden_states + 0.5 * self.dropout(self.ffn2(self.ffn2_norm(hidden_states)))
        )
        return self.final_norm(apply_mask(hidden_states))


class TransformerEncoder(nn.Module):
    def __init__(self, layers: nn.ModuleList, d_model: int, gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(d_model)
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, hidden_states: Tensor, mask: Tensor | None) -> Tensor:
        for layer in self.layers:
            if mask is not None:
                hidden_states = hidden_states * mask.unsqueeze(-1).to(hidden_states.dtype)
            if self.training and self.gradient_checkpointing:
                hidden_states = checkpoint(
                    lambda states, current=layer: current(states, mask),
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(hidden_states, mask)
            if mask is not None:
                hidden_states = hidden_states * mask.unsqueeze(-1).to(hidden_states.dtype)
        if mask is not None:
            hidden_states = hidden_states * mask.unsqueeze(-1).to(hidden_states.dtype)
        return self.norm(hidden_states)


class LogMelFrontend(nn.Module):
    """Convert each unpadded waveform independently to normalized log-mel frames."""

    def __init__(self, audio_config: Mapping[str, Any]) -> None:
        super().__init__()
        self.sample_rate = int(audio_config["sample_rate"])
        self.n_fft = int(audio_config["n_fft"])
        self.hop_length = int(audio_config["hop_length"])
        self.win_length = int(audio_config["win_length"])
        self.feature_dim = int(audio_config["feature_dim"])
        self.f_min = float(audio_config.get("f_min", 0.0))
        self.f_max = float(audio_config.get("f_max", self.sample_rate / 2))
        self.register_buffer("window", torch.hann_window(self.win_length), persistent=False)
        self.register_buffer("mel_filterbank", self._build_mel_filterbank(), persistent=True)

    def _build_mel_filterbank(self) -> Tensor:
        def hz_to_mel(value: Tensor) -> Tensor:
            return 2595.0 * torch.log10(1.0 + value / 700.0)

        def mel_to_hz(value: Tensor) -> Tensor:
            return 700.0 * (10.0 ** (value / 2595.0) - 1.0)

        low = hz_to_mel(torch.tensor(self.f_min))
        high = hz_to_mel(torch.tensor(min(self.f_max, self.sample_rate / 2)))
        points = torch.linspace(low, high, self.feature_dim + 2)
        frequencies = mel_to_hz(points)
        bins = torch.floor((self.n_fft + 1) * frequencies / self.sample_rate).long()
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
        if waveform.numel() < self.win_length:
            waveform = F.pad(waveform, (0, self.win_length - waveform.numel()))
        window = self.window.to(device=waveform.device, dtype=waveform.dtype)
        spectrum = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=window,
            center=True,
            return_complex=True,
        )
        power = spectrum.abs().square()
        mel = torch.einsum(
            "mf,ft->tm",
            self.mel_filterbank.to(device=power.device, dtype=power.dtype),
            power,
        )
        features = torch.log(mel.clamp_min(1e-5))
        normalized = (features - features.mean()).div(features.std().clamp_min(1e-5))
        valid_frames = min(normalized.size(0), max(1, original_length // self.hop_length + 1))
        return normalized, valid_frames

    def forward(self, waveform: Tensor, audio_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        if waveform.dim() != 2 or waveform.size(1) < 2:
            raise ValueError("waveform must have shape [batch, samples]")
        waveform = waveform.float()
        if audio_mask is None:
            lengths = torch.full(
                (waveform.size(0),), waveform.size(1), device=waveform.device, dtype=torch.long
            )
        else:
            if audio_mask.shape != waveform.shape:
                raise ValueError("audio_mask must have shape [batch, samples]")
            lengths = audio_mask.long().sum(dim=1).clamp_min(2)
        individual = [self._extract_one(waveform[row, : int(lengths[row])]) for row in range(waveform.size(0))]
        maximum = max(values.size(0) for values, _ in individual)
        features = waveform.new_zeros(waveform.size(0), maximum, self.feature_dim)
        frame_mask = torch.zeros(waveform.size(0), maximum, device=waveform.device, dtype=torch.bool)
        for row, (values, valid_frames) in enumerate(individual):
            features[row, : values.size(0)] = values
            frame_mask[row, :valid_frames] = True
        return features, frame_mask


class ConvSubsampling(nn.Module):
    def __init__(self, feature_dim: int, d_model: int, factor: int = 4) -> None:
        super().__init__()
        if factor != 4:
            raise ValueError("V0 ConvSubsampling currently requires factor=4")
        channels = max(8, d_model // 2)
        frequency = math.ceil(math.ceil(feature_dim / 2) / 2)
        self.network = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1),
            nn.GELU(),
        )
        self.projection = nn.Linear(channels * frequency, d_model)

    def forward(self, features: Tensor, frame_mask: Tensor) -> tuple[Tensor, Tensor]:
        hidden_states = self.network(features.unsqueeze(1))
        batch, channels, frames, frequency = hidden_states.shape
        hidden_states = hidden_states.transpose(1, 2).contiguous().view(batch, frames, channels * frequency)
        hidden_states = self.projection(hidden_states)
        mask = F.interpolate(frame_mask.float().unsqueeze(1), size=frames, mode="nearest").squeeze(1).bool()
        return hidden_states * mask.unsqueeze(-1).to(hidden_states.dtype), mask


class SpeechEncoder(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        audio = config["audio"]
        speech = config["speech_encoder"]
        d_model = int(speech["d_model"])
        self.frontend = LogMelFrontend(audio)
        self.subsampling = ConvSubsampling(int(audio["feature_dim"]), d_model, int(speech["subsampling_factor"]))
        self.position = PositionalEncoding(d_model)
        block_type = speech["architecture"]
        block_class = ConformerBlock if block_type == "conformer" else TransformerBlock
        self.encoder = TransformerEncoder(
            nn.ModuleList(
                (
                    block_class(
                        d_model,
                        int(speech["attention_heads"]),
                        int(speech["d_ff"]),
                        int(speech["conv_kernel_size"]),
                        float(speech["dropout"]),
                    )
                    if block_type == "conformer"
                    else block_class(
                        d_model,
                        int(speech["attention_heads"]),
                        int(speech["d_ff"]),
                        float(speech["dropout"]),
                    )
                )
                for _ in range(int(speech["layers"]))
            ),
            d_model,
            bool(config["model"].get("gradient_checkpointing", False)),
        )

    def forward(self, waveform: Tensor, audio_mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        features, frame_mask = self.frontend(waveform, audio_mask)
        hidden_states, mask = self.subsampling(features, frame_mask)
        return self.encoder(self.position(hidden_states), mask), mask


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries: Tensor, memory: Tensor, memory_mask: Tensor | None) -> Tensor:
        normalised = self.norm(queries)
        return queries + self.dropout(self.attention(normalised, memory, memory, memory_mask))


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
        latents = self.norm(self.cross_attention(queries, speech_states, speech_mask))
        return latents, torch.ones(latents.size(0), latents.size(1), device=latents.device, dtype=torch.bool)


class SemanticCore(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        section = config["semantic_core"]
        d_model = int(section["d_model"])
        self.encoder = TransformerEncoder(
            nn.ModuleList(
                TransformerBlock(d_model, int(section["attention_heads"]), int(section["d_ff"]), float(section["dropout"]))
                for _ in range(int(section["layers"]))
            ),
            d_model,
            bool(config["model"].get("gradient_checkpointing", False)),
        )

    def forward(self, semantic_latents: Tensor, semantic_mask: Tensor) -> Tensor:
        return self.encoder(semantic_latents, semantic_mask)


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

    def forward(self, semantic_states: Tensor, semantic_mask: Tensor) -> tuple[Tensor, Tensor]:
        queries = self.queries.expand(semantic_states.size(0), -1, -1)
        queries = self.norm(self.cross_attention(queries, semantic_states, semantic_mask))
        return queries, self.presence_head(queries).squeeze(-1)


class SchemaTextEncoder(nn.Module):
    """Encode arbitrary capability text with a fixed hash-bucket vocabulary."""

    def __init__(self, d_model: int, hash_buckets: int = 8192, max_tokens: int = 256) -> None:
        super().__init__()
        self.hash_buckets = hash_buckets
        self.max_tokens = max_tokens
        self.embedding = nn.Embedding(hash_buckets, d_model)
        self.projection = nn.Linear(d_model, d_model)
        self.norm = LayerNorm(d_model)

    def _token_ids(self, text: str) -> list[int]:
        words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
        tokens = words + [f"{word[:n]}" for word in words for n in (2, 3) if len(word) >= n]
        if not tokens:
            tokens = ["<empty>"]
        return [
            int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "little") % self.hash_buckets
            for token in tokens[: self.max_tokens]
        ]

    def encode_texts(self, texts: Sequence[str], device: torch.device | None = None) -> Tensor:
        if not texts:
            return torch.empty(0, self.embedding.embedding_dim, device=device or self.embedding.weight.device)
        ids = [self._token_ids(text) for text in texts]
        maximum = max(len(row) for row in ids)
        token_ids = torch.zeros(len(ids), maximum, dtype=torch.long, device=device or self.embedding.weight.device)
        mask = torch.zeros_like(token_ids, dtype=torch.bool)
        for row, values in enumerate(ids):
            token_ids[row, : len(values)] = torch.tensor(values, device=token_ids.device)
            mask[row, : len(values)] = True
        values = self.embedding(token_ids)
        values = (values * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp_min(1).to(values.dtype)
        return F.normalize(self.norm(self.projection(values)), dim=-1)

    @staticmethod
    def schema_text(schema: Mapping[str, Any]) -> str:
        parameters = schema.get("parameters", {})
        parameter_text = " ".join(
            f"parameter {name} type {value.get('type', '')} {value.get('description', '')}"
            for name, value in parameters.items()
        )
        return " ".join(
            [
                str(schema.get("name", "")),
                str(schema.get("description", "")),
                "actions",
                " ".join(str(action) for action in schema.get("actions", [])),
                parameter_text,
            ]
        )

    def encode_schemas(self, schemas: Sequence[Mapping[str, Any]], device: torch.device | None = None) -> Tensor:
        return self.encode_texts([self.schema_text(schema) for schema in schemas], device)


class CapabilitySchemaRetriever(nn.Module):
    """Dynamic schema retrieval; no parameter is allocated per capability."""

    def __init__(self, config: Mapping[str, Any], schema_encoder: SchemaTextEncoder) -> None:
        super().__init__()
        self.schema_encoder = schema_encoder
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(
        self,
        operation_queries: Tensor,
        schemas: Sequence[Mapping[str, Any]],
        schema_embeddings: Tensor | None = None,
    ) -> Tensor:
        if not schemas:
            raise ValueError("at least one capability schema is required")
        embeddings = schema_embeddings
        if embeddings is None:
            embeddings = self.schema_encoder.encode_schemas(schemas, operation_queries.device)
        if embeddings.dim() != 2 or embeddings.size(1) != operation_queries.size(-1):
            raise ValueError("schema embeddings must have shape [dynamic_schema_count, d_model]")
        if embeddings.size(0) != len(schemas):
            raise ValueError("schema embedding count must match the runtime schema set")
        temperature = self.temperature.abs().clamp_min(1e-3)
        return F.normalize(operation_queries, dim=-1) @ F.normalize(embeddings, dim=-1).transpose(0, 1) / temperature


class ActionResolver(nn.Module):
    """Resolve action from the selected schema's runtime action list."""

    def __init__(self, schema_encoder: SchemaTextEncoder, d_model: int) -> None:
        super().__init__()
        self.schema_encoder = schema_encoder
        self.query = nn.Linear(d_model, d_model)
        self.temperature = nn.Parameter(torch.tensor(0.07))

    def forward(self, operation_queries: Tensor, schemas: Sequence[Mapping[str, Any]]) -> list[Tensor]:
        result: list[Tensor] = []
        queries = F.normalize(self.query(operation_queries), dim=-1)
        for schema in schemas:
            actions = [str(action) for action in schema.get("actions", [])]
            if not actions:
                raise ValueError(f"schema {schema.get('name')} has no actions")
            action_embeddings = self.schema_encoder.encode_texts(
                [f"{schema.get('name', '')} action {action}" for action in actions],
                operation_queries.device,
            )
            result.append(queries @ action_embeddings.transpose(0, 1) / self.temperature.abs().clamp_min(1e-3))
        return result


class EnumExtractor(nn.Module):
    def __init__(self, schema_encoder: SchemaTextEncoder, d_model: int) -> None:
        super().__init__()
        self.schema_encoder = schema_encoder
        self.query = nn.Linear(d_model, d_model)

    def forward(self, conditioned: Tensor, values: Sequence[str]) -> Tensor:
        embeddings = self.schema_encoder.encode_texts([str(value) for value in values], conditioned.device)
        return F.normalize(self.query(conditioned), dim=-1) @ embeddings.transpose(0, 1)


class NumberExtractor(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.network = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.Linear(d_model, 1))

    def forward(self, conditioned: Tensor) -> Tensor:
        return self.network(conditioned).squeeze(-1)


class BooleanExtractor(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.head = nn.Linear(d_model, 2)

    def forward(self, conditioned: Tensor) -> Tensor:
        return self.head(conditioned)


class SpanExtractor(nn.Module):
    SOURCES = ["INPUT_SPAN", "STATE_REFERENCE"]

    def __init__(self, schema_encoder: SchemaTextEncoder, d_model: int) -> None:
        super().__init__()
        self.schema_encoder = schema_encoder
        self.source_head = nn.Linear(d_model, len(self.SOURCES))
        self.span_query = nn.Linear(d_model, d_model)
        self.span_key = nn.Linear(d_model, d_model)
        self.reference_query = nn.Linear(d_model, d_model)

    def forward(
        self,
        conditioned: Tensor,
        span_states: Tensor,
        reference_paths: Sequence[str],
    ) -> dict[str, Tensor]:
        query = self.span_query(conditioned)
        keys = self.span_key(span_states)
        source_logits = self.source_head(conditioned)
        if not reference_paths:
            source_logits[..., self.SOURCES.index("STATE_REFERENCE")] = torch.finfo(source_logits.dtype).min
        result: dict[str, Tensor] = {
            "source_logits": source_logits,
            "span_start_logits": torch.einsum("bod,btd->bot", query, keys),
            "span_end_logits": torch.einsum("bod,btd->bot", query, keys),
        }
        if reference_paths:
            references = self.schema_encoder.encode_texts(list(reference_paths), conditioned.device)
            result["reference_logits"] = F.normalize(self.reference_query(conditioned), dim=-1) @ references.transpose(0, 1)
        return result


class StateReferenceExtractor(nn.Module):
    def __init__(self, schema_encoder: SchemaTextEncoder, d_model: int) -> None:
        super().__init__()
        self.schema_encoder = schema_encoder
        self.query = nn.Linear(d_model, d_model)

    def forward(self, conditioned: Tensor, reference_paths: Sequence[str]) -> dict[str, Tensor]:
        if not reference_paths:
            raise ValueError("STATE_REFERENCE requires an allowlisted path")
        references = self.schema_encoder.encode_texts(list(reference_paths), conditioned.device)
        return {"reference_logits": F.normalize(self.query(conditioned), dim=-1) @ references.transpose(0, 1)}


class TypedParameterExtractor(nn.Module):
    """Shared type extractors conditioned by runtime parameter schemas."""

    def __init__(self, schema_encoder: SchemaTextEncoder, d_model: int) -> None:
        super().__init__()
        self.schema_encoder = schema_encoder
        self.conditioner = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.presence_head = nn.Linear(d_model, 2)
        self.extractors = nn.ModuleDict(
            {
                "ENUM": EnumExtractor(schema_encoder, d_model),
                "NUMBER": NumberExtractor(d_model),
                "BOOLEAN": BooleanExtractor(d_model),
                "ENTITY": SpanExtractor(schema_encoder, d_model),
                "FREE_TEXT": SpanExtractor(schema_encoder, d_model),
                "STATE_REFERENCE": StateReferenceExtractor(schema_encoder, d_model),
            }
        )

    @staticmethod
    def parameter_text(schema: Mapping[str, Any], name: str, parameter: Mapping[str, Any]) -> str:
        return " ".join(
            [
                str(schema.get("name", "")),
                "parameter",
                name,
                "type",
                str(parameter.get("type", "")),
                str(parameter.get("description", "")),
            ]
        )

    def forward(
        self,
        operation_queries: Tensor,
        span_states: Tensor,
        schemas: Sequence[Mapping[str, Any]],
    ) -> dict[str, dict[str, dict[str, Tensor]]]:
        result: dict[str, dict[str, dict[str, Tensor]]] = {}
        for schema in schemas:
            schema_name = str(schema["name"])
            parameters = schema.get("parameters", {})
            result[schema_name] = {}
            for name, parameter in parameters.items():
                parameter_embedding = self.schema_encoder.encode_texts(
                    [self.parameter_text(schema, name, parameter)], operation_queries.device
                )[0]
                conditioned = self.conditioner(
                    torch.cat(
                        [operation_queries, parameter_embedding.view(1, 1, -1).expand_as(operation_queries)],
                        dim=-1,
                    )
                )
                output: dict[str, Tensor] = {
                    "presence_logits": self.presence_head(conditioned),
                }
                parameter_type = str(parameter["type"])
                extractor = self.extractors[parameter_type]
                if parameter_type == "ENUM":
                    output["enum_logits"] = extractor(conditioned, parameter["values"])
                elif parameter_type == "NUMBER":
                    output["number_value"] = extractor(conditioned)
                elif parameter_type == "BOOLEAN":
                    output["boolean_logits"] = extractor(conditioned)
                elif parameter_type == "STATE_REFERENCE":
                    output.update(extractor(conditioned, parameter.get("state_reference_paths", [])))
                else:
                    output.update(extractor(conditioned, span_states, parameter.get("state_reference_paths", [])))
                result[schema_name][name] = output
        return result


class BridgeAlignmentHead(nn.Module):
    """Project semantic latents into the speech representation space."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, semantic_states: Tensor) -> Tensor:
        return F.normalize(self.projection(semantic_states), dim=-1)


class LexicalSpanResolver:
    """Resolve CTC frame spans into UTF-8 text values at inference time."""

    def __init__(self, vocabulary: Sequence[str], blank_id: int = 0) -> None:
        if len(vocabulary) < 257 or vocabulary[blank_id] != "<BLANK>":
            raise ValueError("lexical vocabulary must contain a blank and 256 byte tokens")
        self.vocabulary = list(vocabulary)
        self.blank_id = int(blank_id)

    def resolve(self, logits: Tensor, start_frame: int, end_frame: int) -> str:
        if logits.dim() != 2:
            raise ValueError("lexical logits must have shape [frames, vocabulary]")
        start = max(0, min(int(start_frame), logits.size(0) - 1))
        end = max(start, min(int(end_frame), logits.size(0) - 1))
        token_ids = logits[start : end + 1].argmax(-1).tolist()
        values: list[int] = []
        previous: int | None = None
        for token_id in token_ids:
            token_id = int(token_id)
            if token_id == self.blank_id or token_id == previous:
                previous = token_id
                continue
            if not 1 <= token_id <= 256:
                raise ValueError(f"invalid lexical token id: {token_id}")
            values.append(token_id - 1)
            previous = token_id
        return bytes(values).decode("utf-8", errors="replace").strip()


class VoiceNativeSLU(nn.Module):
    """VALLS SLU V0: waveform -> semantic execution frame ingredients."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        validate_config(config)
        self.config = dict(config)
        d_model = int(config["semantic_core"]["d_model"])
        schema_encoder = SchemaTextEncoder(d_model)
        self.schema_encoder = schema_encoder
        self.speech_encoder = SpeechEncoder(config)
        self.semantic_resampler = SemanticResampler(config)
        self.semantic_core = SemanticCore(config)
        self.operation_decoder = OperationDecoder(config)
        self.schema_retriever = CapabilitySchemaRetriever(config, schema_encoder)
        self.action_resolver = ActionResolver(schema_encoder, d_model)
        self.parameter_extractor = TypedParameterExtractor(schema_encoder, d_model)
        self.act_head = nn.Linear(d_model, len(config["ontology"]["acts"]))
        self.confidence_head = nn.Linear(d_model, 3)
        self.ood_head = nn.Linear(d_model, 1)
        self.bridge_alignment_head = BridgeAlignmentHead(d_model)
        lexical = config["lexical_branch"]
        self.lexical_span_resolver = LexicalSpanResolver(
            lexical["vocab"], int(lexical["blank_id"])
        )
        self.lexical_head = nn.Linear(d_model, int(lexical["vocab_size"])) if lexical.get("enabled", True) else None
        self.apply(self._reset_parameters)

    @staticmethod
    def _reset_parameters(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _masked_mean(hidden_states: Tensor, mask: Tensor) -> Tensor:
        weights = mask.to(hidden_states.dtype).unsqueeze(-1)
        return (hidden_states * weights).sum(1) / weights.sum(1).clamp_min(1.0)

    def default_schemas(self) -> list[dict[str, Any]]:
        return capability_schemas(self.config)

    def encode_audio(self, waveform: Tensor, audio_mask: Tensor | None = None) -> dict[str, Tensor]:
        speech_states, speech_mask = self.speech_encoder(waveform, audio_mask)
        semantic_latents, semantic_mask = self.semantic_resampler(speech_states, speech_mask)
        semantic_states = self.semantic_core(semantic_latents, semantic_mask)
        return {
            "speech_states": speech_states,
            "speech_mask": speech_mask,
            "semantic_latents": semantic_latents,
            "semantic_mask": semantic_mask,
            "semantic_states": semantic_states,
        }

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
        for schema in schemas:
            if "name" not in schema:
                raise ValueError("every capability schema requires a name")
        if bridge_only:
            # BRIDGE stops at the resampler. Do not even execute SemanticCore:
            # it is an untrained/frozen semantic transformation at this stage.
            speech_states, speech_mask = self.speech_encoder(waveform, audio_mask)
            semantic_latents, semantic_mask = self.semantic_resampler(speech_states, speech_mask)
            return {
                "speech_states": speech_states,
                "speech_mask": speech_mask,
                "semantic_latents": semantic_latents,
                "semantic_mask": semantic_mask,
                "bridge_states": self.bridge_alignment_head(semantic_latents),
            }
        encoded = self.encode_audio(waveform, audio_mask)
        semantic_states = encoded["semantic_states"]
        semantic_mask = encoded["semantic_mask"]
        pooled = self._masked_mean(semantic_states, semantic_mask)
        operation_queries, operation_presence_logits = self.operation_decoder(semantic_states, semantic_mask)
        outputs: dict[str, Any] = {
            **encoded,
            "pooled_semantic": pooled,
            "act_logits": self.act_head(pooled),
            "confidence_logits": self.confidence_head(pooled),
            "ood_logit": self.ood_head(pooled).squeeze(-1),
            "operation_queries": operation_queries,
            "operation_presence_logits": operation_presence_logits,
            "schema_names": [str(schema["name"]) for schema in schemas],
            "capability_schemas": schemas,
            "goal_scores": self.schema_retriever(operation_queries, schemas, capability_embeddings),
            "action_scores": self.action_resolver(operation_queries, schemas),
            "parameter_outputs": self.parameter_extractor(operation_queries, encoded["speech_states"], schemas),
            "bridge_states": self.bridge_alignment_head(encoded["semantic_latents"]),
        }
        if self.lexical_head is not None:
            # CTC stays on acoustic states for ACOUSTIC.  BRIDGE uses the
            # bridge_states output and an explicit representation-alignment
            # objective so the resampler receives a linguistic preservation signal.
            outputs["lexical_ctc_logits"] = self.lexical_head(encoded["speech_states"])
            outputs["bridge_lexical_logits"] = self.lexical_head(encoded["semantic_latents"])
        return outputs


SpeechSemanticAdapter = SemanticResampler
GoalQueryGenerator = OperationDecoder
MultiTaskTransformer = VoiceNativeSLU


def build_model(config: Mapping[str, Any]) -> VoiceNativeSLU:
    validate_config(config)
    return VoiceNativeSLU(config)
