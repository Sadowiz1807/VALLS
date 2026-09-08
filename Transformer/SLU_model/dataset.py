"""Dataset boundaries for the voice-native VALLS SLU V0 model.

The V0 corpus is waveform-first.  Transcript text is not the semantic input,
but it is required for ENTITY/FREE_TEXT lexical supervision.  Span targets use
normalized speech-frame ratios produced by a forced aligner, so they remain
valid after batching and convolutional subsampling.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .config import default_lexical_vocab, validate_config


class DatasetContractError(ValueError):
    """Raised when a record cannot be trusted as a VALLS training example."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise DatasetContractError(f"invalid JSON at {source}:{line_number}") from error
        if not isinstance(record, dict):
            raise DatasetContractError(f"record at {source}:{line_number} must be an object")
        records.append(record)
    return records


def encode_lexical_text(text: str, vocab: Sequence[str] | None = None) -> list[int]:
    """Encode UTF-8 bytes as ids 1..256; id 0 is the CTC blank."""

    active_vocab = list(vocab or default_lexical_vocab())
    if len(active_vocab) < 257 or active_vocab[0] != "<BLANK>":
        raise DatasetContractError("lexical vocabulary must contain blank plus 256 byte tokens")
    return [byte + 1 for byte in text.encode("utf-8")]


def ctc_target_length_is_valid(input_length: int, target_length: int) -> bool:
    """Return whether a CTC target can fit the available timesteps."""

    return int(target_length) <= int(input_length)


def decode_lexical_ids(ids: Sequence[int], vocab: Sequence[str] | None = None) -> str:
    """Decode greedy lexical ids, ignoring CTC blank and repeated ids."""

    active_vocab = list(vocab or default_lexical_vocab())
    if len(active_vocab) < 257:
        raise DatasetContractError("lexical vocabulary is incomplete")
    bytes_out: list[int] = []
    previous = None
    for index in ids:
        index = int(index)
        if index == 0 or index == previous:
            previous = index
            continue
        if not 1 <= index <= 256:
            raise DatasetContractError(f"invalid lexical token id: {index}")
        bytes_out.append(index - 1)
        previous = index
    return bytes(bytes_out).decode("utf-8", errors="replace")


def _validate_alignment(value: dict[str, Any]) -> None:
    alignment = value.get("alignment")
    if not isinstance(alignment, dict):
        raise DatasetContractError("input_span requires alignment={start,end} ratios")
    start, end = alignment.get("start"), alignment.get("end")
    if (
        isinstance(start, bool)
        or not isinstance(start, (int, float))
        or isinstance(end, bool)
        or not isinstance(end, (int, float))
        or not 0 <= float(start) < float(end) <= 1
    ):
        raise DatasetContractError("alignment ratios must satisfy 0 <= start < end <= 1")


def _validate_input_span(value: dict[str, Any], transcript: str | None) -> None:
    # Inference frames use speech-frame ratios; supervised records additionally
    # carry transcript character offsets for lexical/span alignment.
    if transcript is None and "start_frame" in value:
        _validate_alignment({"alignment": {"start": value.get("start_ratio"), "end": value.get("end_ratio")}})
        if not isinstance(value.get("start_frame"), int) or not isinstance(value.get("end_frame"), int):
            raise DatasetContractError("predicted input_span requires integer frame bounds")
        if not isinstance(value.get("value"), str) or not value["value"]:
            raise DatasetContractError("predicted input_span requires a resolved lexical value")
        return
    start, end, text = value.get("start"), value.get("end"), value.get("value")
    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(end, bool)
        or not isinstance(end, int)
        or start < 0
        or end <= start
        or not isinstance(text, str)
        or not text
    ):
        raise DatasetContractError("input_span requires start, end and non-empty value")
    if transcript is None:
        raise DatasetContractError("input_span requires transcript")
    if text != transcript[start:end]:
        raise DatasetContractError("input_span value must equal transcript[start:end]")
    _validate_alignment(value)


def _validate_parameter(goal: str, name: str, value: Any, schema: dict[str, Any], transcript: str | None) -> None:
    parameter_type = schema["type"]
    if parameter_type == "ENUM":
        if value not in schema["values"]:
            raise DatasetContractError(f"invalid enum value for {goal}.{name}: {value!r}")
    elif parameter_type == "NUMBER":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DatasetContractError(f"{goal}.{name} must be numeric")
        if "minimum" in schema and value < schema["minimum"]:
            raise DatasetContractError(f"{goal}.{name} is below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            raise DatasetContractError(f"{goal}.{name} is above maximum")
    elif parameter_type == "BOOLEAN":
        if not isinstance(value, bool):
            raise DatasetContractError(f"{goal}.{name} must be boolean")
    elif parameter_type in {"ENTITY", "FREE_TEXT"}:
        if not isinstance(value, dict):
            raise DatasetContractError(f"{goal}.{name} must be a typed value")
        source = value.get("source")
        if source == "input_span":
            _validate_input_span(value, transcript)
        elif source == "state_reference":
            if value.get("path") not in schema.get("state_reference_paths", []):
                raise DatasetContractError(f"state reference is not allowlisted for {goal}.{name}")
        else:
            raise DatasetContractError(f"unsupported source for {goal}.{name}: {source!r}")
    elif parameter_type == "STATE_REFERENCE":
        if not isinstance(value, dict) or value.get("source") != "state_reference":
            raise DatasetContractError(f"{goal}.{name} must use a state_reference value")
        if value.get("path") not in schema.get("state_reference_paths", []):
            raise DatasetContractError(f"state reference is not allowlisted for {goal}.{name}")


def _requires_transcript(target: dict[str, Any], config: dict[str, Any]) -> bool:
    for operation in target.get("operations", []):
        goal = operation.get("goal")
        if goal not in config["ontology"]["capabilities"]:
            continue
        schemas = config["ontology"]["capabilities"][goal]["parameters"]
        for name, value in operation.get("parameters", {}).items():
            if schemas.get(name, {}).get("type") in {"ENTITY", "FREE_TEXT"} and isinstance(value, dict) and value.get("source") == "input_span":
                return True
    return False


def validate_target(
    target: dict[str, Any],
    config: dict[str, Any],
    transcript: str | None = None,
    capability_schemas: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    acts = config["ontology"]["acts"]
    schema_map = (
        {str(schema["name"]): schema for schema in capability_schemas}
        if capability_schemas is not None
        else config["ontology"]["capabilities"]
    )
    if target.get("act") not in acts:
        raise DatasetContractError(f"unknown ACT: {target.get('act')!r}")
    operations = target.get("operations", [])
    if not isinstance(operations, list):
        raise DatasetContractError("target.operations must be a list")
    maximum = int(config["operation_decoder"]["max_operations"])
    if len(operations) > maximum:
        raise DatasetContractError(f"target contains more than {maximum} operations")

    for order, operation in enumerate(operations, 1):
        if not isinstance(operation, dict):
            raise DatasetContractError("each operation must be an object")
        if operation.get("order", order) != order:
            raise DatasetContractError("operation order must be contiguous and 1-based")
        goal = operation.get("goal")
        if goal not in schema_map:
            raise DatasetContractError(f"unknown capability schema: {goal!r}")
        schema = schema_map[goal]
        action = operation.get("action")
        if action not in schema.get("actions", []):
            raise DatasetContractError(f"invalid action for {goal}: {action!r}")
        parameters = operation.get("parameters", {})
        if not isinstance(parameters, dict):
            raise DatasetContractError("operation.parameters must be an object")
        parameter_schemas = schema["parameters"]
        extra = set(parameters) - set(parameter_schemas)
        if extra:
            raise DatasetContractError(f"unknown parameters for {goal}: {sorted(extra)}")
        for name, parameter_schema in parameter_schemas.items():
            if name not in parameters:
                if parameter_schema.get("required"):
                    raise DatasetContractError(f"missing required parameter {goal}.{name}")
                continue
            _validate_parameter(goal, name, parameters[name], parameter_schema, transcript)

    if target["act"] == "EXECUTE" and not operations:
        raise DatasetContractError("EXECUTE requires at least one operation")
    if target["act"] != "EXECUTE" and operations:
        raise DatasetContractError("non-EXECUTE targets cannot contain operations")


def validate_record(record: dict[str, Any], config: dict[str, Any]) -> None:
    required = {"sample_id", "audio", "target"}
    missing = required - set(record)
    if missing:
        raise DatasetContractError(f"record is missing fields: {sorted(missing)}")
    if not isinstance(record["sample_id"], str) or not record["sample_id"]:
        raise DatasetContractError("sample_id must be a non-empty string")
    audio = record["audio"]
    if not isinstance(audio, (str, list, tuple, Tensor)):
        raise DatasetContractError("audio must be a path, waveform sequence or Tensor")
    if isinstance(audio, Tensor) and (audio.dim() != 1 or audio.numel() < 2):
        raise DatasetContractError("audio Tensor must have shape [samples]")
    if isinstance(audio, (list, tuple)) and len(audio) < 2:
        raise DatasetContractError("audio sequence must contain at least two samples")
    if not isinstance(record["target"], dict):
        raise DatasetContractError("target must be an object")
    transcript = record.get("transcript")
    if transcript is not None and not isinstance(transcript, str):
        raise DatasetContractError("transcript must be a string when provided")
    if _requires_transcript(record["target"], config) and not transcript:
        raise DatasetContractError("transcript is required for ENTITY/FREE_TEXT input spans")
    validate_target(record["target"], config, transcript)


class VoiceSLUDataset(Dataset[dict[str, Any]]):
    """Waveform dataset adapter with transcript and lexical supervision."""

    def __init__(self, records: Sequence[dict[str, Any]], config: dict[str, Any], audio_loader: Callable[[Any], Tensor] | None = None) -> None:
        validate_config(config)
        for record in records:
            validate_record(record, config)
        self.records = list(records)
        self.config = config
        self.audio_loader = audio_loader or self._default_audio_loader
        self.lexical_vocab = config["lexical_branch"]["vocab"]
        self.sample_rate = int(config["audio"]["sample_rate"])
        self.hop_length = int(config["audio"]["hop_length"])
        self.subsampling_factor = int(config["speech_encoder"]["subsampling_factor"])

    @staticmethod
    def _default_audio_loader(audio: Any) -> Tensor:
        if isinstance(audio, Tensor):
            waveform = audio.detach().clone().float()
        elif isinstance(audio, (list, tuple)):
            waveform = torch.tensor(audio, dtype=torch.float32)
        else:
            raise DatasetContractError("an audio_loader is required for path-based audio records")
        if waveform.dim() != 1 or waveform.numel() < 2:
            raise DatasetContractError("loaded waveform must have shape [samples]")
        return waveform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        waveform = self.audio_loader(record["audio"])
        transcript = record.get("transcript")
        lexical_ids = encode_lexical_text(transcript or "", self.lexical_vocab)
        estimated_input_frames = max(1, int(waveform.numel()) // self.hop_length + 1)
        estimated_ctc_frames = max(1, (estimated_input_frames + self.subsampling_factor - 1) // self.subsampling_factor)
        repeated_tokens = sum(left == right for left, right in zip(lexical_ids, lexical_ids[1:]))
        required_ctc_frames = len(lexical_ids) + repeated_tokens
        if not ctc_target_length_is_valid(estimated_ctc_frames, required_ctc_frames):
            raise DatasetContractError(
                f"lexical target is too long for CTC: target={len(lexical_ids)}, required={required_ctc_frames}, estimated_input={estimated_ctc_frames}"
            )
        return {
            "sample_id": record["sample_id"],
            "waveform": waveform,
            "target": record["target"],
            "transcript": transcript,
            "lexical_labels": torch.tensor(lexical_ids, dtype=torch.long),
            "estimated_ctc_input_length": estimated_ctc_frames,
            "required_ctc_length": required_ctc_frames,
            "metadata": record.get("metadata", {}),
        }


def collate_audio_batch(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty batch")
    lengths = [int(item["waveform"].numel()) for item in items]
    maximum = max(lengths)
    waveform = torch.zeros(len(items), maximum, dtype=torch.float32)
    audio_mask = torch.zeros(len(items), maximum, dtype=torch.bool)
    for row, item in enumerate(items):
        values = item["waveform"].float()
        waveform[row, : values.numel()] = values
        audio_mask[row, : values.numel()] = True
    labels = [item["lexical_labels"] for item in items]
    return {
        "waveform": waveform,
        "audio_mask": audio_mask,
        "sample_id": [item["sample_id"] for item in items],
        "target": [item["target"] for item in items],
        "transcript": [item["transcript"] for item in items],
        "lexical_labels": torch.cat(labels) if any(label.numel() for label in labels) else torch.empty(0, dtype=torch.long),
        "lexical_target_lengths": torch.tensor([label.numel() for label in labels], dtype=torch.long),
        "metadata": [item["metadata"] for item in items],
        "ctc_invalid_count": sum(
            int(item["required_ctc_length"] > item["estimated_ctc_input_length"])
            for item in items
        ),
        "ctc_sample_count": len(items),
    }


MultiTaskDataset = VoiceSLUDataset
collate_batch = collate_audio_batch
