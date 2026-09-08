"""Neural modules for the voice-native VALLS SLU Model V0.

The model owns the audio-to-semantic path only.  It emits an untrusted
semantic prediction; validation, registry resolution, policy and execution
remain outside this module.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.checkpoint import checkpoint

from .config import validate_config


class LayerNorm(nn.Module):
    """Small explicit LayerNorm kept local so the model has one dependency."""

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
    """Sinusoidal position encoding for variable-length frame sequences."""

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
    """Self/cross attention using a boolean valid-key mask."""

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
    """Pre-norm transformer block for sequence self-attention."""

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


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        layers: nn.ModuleList,
        d_model: int,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNorm(d_model)
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, hidden_states: Tensor, mask: Tensor | None) -> Tensor:
        for layer in self.layers:
            if self.training and self.gradient_checkpointing:
                hidden_states = checkpoint(
                    lambda states, current=layer: current(states, mask),
                    hidden_states,
                    use_reentrant=False,
                )
            else:
                hidden_states = layer(hidden_states, mask)
        return self.norm(hidden_states)


class LogMelFrontend(nn.Module):
    """Convert mono waveforms to log-mel features without torchaudio."""

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
        self.register_buffer(
            "mel_filterbank",
            self._build_mel_filterbank(),
            persistent=True,
        )

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
            centre = max(centre, left + 1)
            right = max(right, centre + 1)
            left = max(0, min(left, filters.size(1) - 1))
            centre = max(0, min(centre, filters.size(1)))
            right = max(centre + 1, min(right, filters.size(1)))
            if centre > left:
                filters[index, left:centre] = torch.linspace(0, 1, centre - left)
            if right > centre:
                filters[index, centre:right] = torch.linspace(1, 0, right - centre)
        return filters

    def forward(
        self,
        waveform: Tensor,
        audio_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if waveform.dim() != 2:
            raise ValueError("waveform must have shape [batch, samples]")
        if waveform.size(1) < 2:
            raise ValueError("waveform must contain at least two samples")
        waveform = waveform.float()
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
            "mf,bft->btm",
            self.mel_filterbank.to(device=power.device, dtype=power.dtype),
            power,
        )
        features = torch.log(mel.clamp_min(1e-5))
        features = (features - features.mean(dim=(1, 2), keepdim=True)) / (
            features.std(dim=(1, 2), keepdim=True).clamp_min(1e-5)
        )
        frames = features.size(1)
        if audio_mask is None:
            frame_mask = torch.ones(
                waveform.size(0), frames, device=waveform.device, dtype=torch.bool
            )
        else:
            if audio_mask.shape != waveform.shape:
                raise ValueError("audio_mask must have shape [batch, samples]")
            lengths = audio_mask.long().sum(dim=1)
            # center=True contributes one valid edge frame; this is conservative
            # and keeps padded audio from becoming semantic evidence.
            valid_frames = torch.div(lengths, self.hop_length, rounding_mode="floor") + 1
            frame_indices = torch.arange(frames, device=waveform.device)[None, :]
            frame_mask = frame_indices < valid_frames[:, None]
        return features, frame_mask


class ConvSubsampling(nn.Module):
    """Reduce acoustic frame rate before the speech transformer."""

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
        hidden_states = hidden_states.transpose(1, 2).contiguous().view(
            batch, frames, channels * frequency
        )
        hidden_states = self.projection(hidden_states)
        mask = F.interpolate(
            frame_mask.float().unsqueeze(1), size=frames, mode="nearest"
        ).squeeze(1).bool()
        return hidden_states, mask


class SpeechEncoder(nn.Module):
    """Acoustic/linguistic representation encoder."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        audio = config["audio"]
        speech = config["speech_encoder"]
        d_model = int(speech["d_model"])
        self.frontend = LogMelFrontend(audio)
        self.subsampling = ConvSubsampling(
            int(audio["feature_dim"]),
            d_model,
            int(speech["subsampling_factor"]),
        )
        self.position = PositionalEncoding(d_model)
        self.encoder = TransformerEncoder(
            nn.ModuleList(
                TransformerBlock(
                    d_model,
                    int(speech["attention_heads"]),
                    int(speech["d_ff"]),
                    float(speech["dropout"]),
                )
                for _ in range(int(speech["layers"]))
            ),
            d_model,
            bool(config["model"].get("gradient_checkpointing", False)),
        )

    def forward(
        self,
        waveform: Tensor,
        audio_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        features, frame_mask = self.frontend(waveform, audio_mask)
        hidden_states, mask = self.subsampling(features, frame_mask)
        return self.encoder(self.position(hidden_states), mask), mask


class CrossAttention(nn.Module):
    def __init__(self, d_model: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.norm = LayerNorm(d_model)
        self.attention = MultiHeadAttention(d_model, heads, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        queries: Tensor,
        memory: Tensor,
        memory_mask: Tensor | None,
    ) -> Tensor:
        normalised = self.norm(queries)
        return queries + self.dropout(
            self.attention(normalised, memory, memory, memory_mask)
        )


class SemanticResampler(nn.Module):
    """Compress long speech sequences into fixed semantic latent tokens."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        section = config["semantic_resampler"]
        d_model = int(section["d_model"])
        self.latent_tokens = int(section["latent_tokens"])
        self.queries = nn.Parameter(torch.empty(1, self.latent_tokens, d_model))
        nn.init.normal_(self.queries, std=d_model ** -0.5)
        self.cross_attention = CrossAttention(
            d_model,
            int(section["attention_heads"]),
            float(section["dropout"]),
        )
        self.norm = LayerNorm(d_model)

    def forward(self, speech_states: Tensor, speech_mask: Tensor) -> tuple[Tensor, Tensor]:
        queries = self.queries.expand(speech_states.size(0), -1, -1)
        latents = self.cross_attention(queries, speech_states, speech_mask)
        latents = self.norm(latents)
        return latents, torch.ones(
            latents.size(0), latents.size(1), device=latents.device, dtype=torch.bool
        )


class SemanticCore(nn.Module):
    """Contextual semantic representation over the resampled speech latents."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        section = config["semantic_core"]
        d_model = int(section["d_model"])
        self.encoder = TransformerEncoder(
            nn.ModuleList(
                TransformerBlock(
                    d_model,
                    int(section["attention_heads"]),
                    int(section["d_ff"]),
                    float(section["dropout"]),
                )
                for _ in range(int(section["layers"]))
            ),
            d_model,
            bool(config["model"].get("gradient_checkpointing", False)),
        )

    def forward(self, semantic_latents: Tensor, semantic_mask: Tensor) -> Tensor:
        return self.encoder(semantic_latents, semantic_mask)


class OperationDecoder(nn.Module):
    """Decode a bounded sequence of operation queries from semantic memory."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        section = config["operation_decoder"]
        d_model = int(section["d_model"])
        self.max_operations = int(section["max_operations"])
        self.queries = nn.Parameter(torch.empty(1, self.max_operations, d_model))
        nn.init.normal_(self.queries, std=d_model ** -0.5)
        self.cross_attention = CrossAttention(
            d_model,
            int(section["attention_heads"]),
            float(section["dropout"]),
        )
        self.norm = LayerNorm(d_model)
        self.presence_head = nn.Linear(d_model, 1)

    def forward(self, semantic_states: Tensor, semantic_mask: Tensor) -> tuple[Tensor, Tensor]:
        queries = self.queries.expand(semantic_states.size(0), -1, -1)
        queries = self.norm(self.cross_attention(queries, semantic_states, semantic_mask))
        return queries, self.presence_head(queries).squeeze(-1)


class CapabilitySchemaRetriever(nn.Module):
    """Resolve goals by similarity to capability schema embeddings.

    This is intentionally not an ``N_goals`` classifier head.  A caller may
    provide externally encoded schema vectors at inference time; the learned
    vectors are only the local fallback when no registry encoder is available.
    """

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        section = config["semantic_core"]
        d_model = int(section["d_model"])
        goals = list(config["ontology"]["schema_order"])
        self.goals = goals
        self.temperature = nn.Parameter(torch.tensor(0.07))
        self.schema_embeddings = nn.Parameter(torch.empty(len(goals), d_model))
        nn.init.normal_(self.schema_embeddings, std=d_model ** -0.5)

    def forward(
        self,
        operation_queries: Tensor,
        capability_embeddings: Tensor | None = None,
    ) -> Tensor:
        schemas = self.schema_embeddings if capability_embeddings is None else capability_embeddings
        if schemas.dim() != 2 or schemas.size(0) != len(self.goals) or schemas.size(1) != operation_queries.size(-1):
            raise ValueError("capability_embeddings must have shape [capabilities, d_model]")
        queries = F.normalize(operation_queries, dim=-1)
        schemas = F.normalize(schemas.to(operation_queries), dim=-1)
        temperature = self.temperature.abs().clamp_min(1e-3)
        return queries @ schemas.transpose(0, 1) / temperature


class ParameterHead(nn.Module):
    """Typed parameter extractor head for one schema parameter."""

    SOURCES = ["ABSENT", "INPUT_SPAN", "STATE_REFERENCE"]

    def __init__(self, parameter: Mapping[str, Any], d_model: int) -> None:
        super().__init__()
        self.parameter_type = str(parameter["type"])
        self.values = list(parameter.get("values", []))
        self.reference_paths = list(parameter.get("state_reference_paths", []))
        self.value_projection = nn.Linear(d_model, d_model)
        self.presence_head = nn.Linear(d_model, 2)
        if self.parameter_type == "ENUM":
            self.enum_head = nn.Linear(d_model, len(self.values))
        elif self.parameter_type == "NUMBER":
            self.number_head = nn.Linear(d_model, 1)
        elif self.parameter_type in {"ENTITY", "FREE_TEXT", "STATE_REFERENCE"}:
            self.source_head = nn.Linear(d_model, len(self.SOURCES))
            self.span_query = nn.Linear(d_model, d_model)
            self.token_projection = nn.Linear(d_model, d_model)
            if self.parameter_type == "STATE_REFERENCE":
                self.reference_head = nn.Linear(d_model, len(self.reference_paths))
        elif self.parameter_type == "BOOLEAN":
            self.boolean_head = nn.Linear(d_model, 2)
        else:
            raise ValueError(f"unsupported parameter type: {self.parameter_type}")

    def forward(self, operation_queries: Tensor, semantic_states: Tensor) -> dict[str, Tensor]:
        value_states = self.value_projection(operation_queries)
        result: dict[str, Tensor] = {
            "presence_logits": self.presence_head(value_states),
        }
        if self.parameter_type == "ENUM":
            result["enum_logits"] = self.enum_head(value_states)
        elif self.parameter_type == "NUMBER":
            result["number_value"] = self.number_head(value_states).squeeze(-1)
        elif self.parameter_type in {"ENTITY", "FREE_TEXT", "STATE_REFERENCE"}:
            result["source_logits"] = self.source_head(value_states)
            if self.parameter_type in {"ENTITY", "FREE_TEXT"}:
                query = self.span_query(value_states)
                tokens = self.token_projection(semantic_states)
                result["span_start_logits"] = torch.einsum("bnd,bsd->bns", query, tokens)
                result["span_end_logits"] = torch.einsum("bnd,bsd->bns", query, tokens)
                result["entity_query"] = F.normalize(query, dim=-1)
            if self.parameter_type == "STATE_REFERENCE":
                result["reference_logits"] = self.reference_head(value_states)
        elif self.parameter_type == "BOOLEAN":
            result["boolean_logits"] = self.boolean_head(value_states)
        return result


class TypedParameterExtractor(nn.Module):
    """Apply schema-specific typed heads to every decoded operation query."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        d_model = int(config["semantic_core"]["d_model"])
        self.heads = nn.ModuleDict()
        self.metadata: dict[str, dict[str, Any]] = {}
        for goal in config["ontology"]["schema_order"]:
            for name, parameter in config["ontology"]["capabilities"][goal]["parameters"].items():
                key = f"{goal}__{name}"
                self.heads[key] = ParameterHead(parameter, d_model)
                self.metadata[key] = dict(parameter)

    def forward(self, operation_queries: Tensor, semantic_states: Tensor) -> dict[str, dict[str, dict[str, Tensor]]]:
        result: dict[str, dict[str, dict[str, Tensor]]] = {}
        for key, head in self.heads.items():
            goal, name = key.split("__", 1)
            result.setdefault(goal, {})[name] = head(operation_queries, semantic_states)
        return result


class VoiceNativeSLU(nn.Module):
    """VALLS SLU V0: audio -> semantic execution frame ingredients."""

    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        validate_config(config)
        self.config = dict(config)
        d_model = int(config["semantic_core"]["d_model"])
        self.speech_encoder = SpeechEncoder(config)
        self.semantic_resampler = SemanticResampler(config)
        self.semantic_core = SemanticCore(config)
        self.operation_decoder = OperationDecoder(config)
        self.schema_retriever = CapabilitySchemaRetriever(config)
        self.parameter_extractor = TypedParameterExtractor(config)
        self.act_head = nn.Linear(d_model, len(config["ontology"]["acts"]))
        self.confidence_head = nn.Linear(d_model, 3)
        self.ood_head = nn.Linear(d_model, 1)
        lexical = config["lexical_branch"]
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

    def encode_audio(
        self,
        waveform: Tensor,
        audio_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
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
        capability_embeddings: Tensor | None = None,
    ) -> dict[str, Any]:
        encoded = self.encode_audio(waveform, audio_mask)
        semantic_states = encoded["semantic_states"]
        semantic_mask = encoded["semantic_mask"]
        pooled = self._masked_mean(semantic_states, semantic_mask)
        operation_queries, operation_presence_logits = self.operation_decoder(
            semantic_states, semantic_mask
        )
        outputs: dict[str, Any] = {
            **encoded,
            "pooled_semantic": pooled,
            "act_logits": self.act_head(pooled),
            "confidence_logits": self.confidence_head(pooled),
            "ood_logit": self.ood_head(pooled).squeeze(-1),
            "operation_queries": operation_queries,
            "operation_presence_logits": operation_presence_logits,
            "goal_scores": self.schema_retriever(operation_queries, capability_embeddings),
            "parameter_outputs": self.parameter_extractor(operation_queries, semantic_states),
        }
        if self.lexical_head is not None:
            outputs["lexical_ctc_logits"] = self.lexical_head(encoded["speech_states"])
        return outputs


# Descriptive aliases make the architecture names available to callers without
# duplicating implementation classes.
SpeechSemanticAdapter = SemanticResampler
GoalQueryGenerator = OperationDecoder


def build_model(config: Mapping[str, Any]) -> VoiceNativeSLU:
    """Validate configuration and construct the VALLS SLU V0 model."""

    validate_config(config)
    return VoiceNativeSLU(config)


# Backward-compatible name is intentionally an alias, not the old text model.
MultiTaskTransformer = VoiceNativeSLU
