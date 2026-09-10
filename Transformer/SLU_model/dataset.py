"""VALLS SLU V1 dataset contracts and waveform batching."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .config import (
    CONTEXT_REFERENCE_TYPES,
    OPERATION_RELATIONS,
    TURN_RELATIONS,
    default_lexical_vocab,
)


class DatasetContractError(ValueError):
    """Raised when an input record is not a trusted VALLS training record."""


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
    active_vocab = list(vocab or default_lexical_vocab())
    if len(active_vocab) < 257 or active_vocab[0] != "<BLANK>":
        raise DatasetContractError("lexical vocabulary must contain blank plus byte tokens")
    return [byte + 1 for byte in text.encode("utf-8")]


def decode_lexical_ids(ids: Sequence[int], vocab: Sequence[str] | None = None) -> str:
    active_vocab = list(vocab or default_lexical_vocab())
    if len(active_vocab) < 257 or active_vocab[0] != "<BLANK>":
        raise DatasetContractError("lexical vocabulary must contain blank plus byte tokens")
    values: list[int] = []
    previous: int | None = None
    for token in ids:
        token = int(token)
        if token == 0 or token == previous:
            previous = token
            continue
        if not 1 <= token <= 256:
            raise DatasetContractError(f"invalid lexical token id: {token}")
        values.append(token - 1)
        previous = token
    return bytes(values).decode("utf-8", errors="replace").strip()


def ctc_required_length(labels: Sequence[int]) -> int:
    return len(labels) + sum(left == right for left, right in zip(labels, labels[1:]))


def ctc_target_length_is_valid(input_length: int, target_length: int) -> bool:
    return int(target_length) <= int(input_length)


def _validate_alignment(value: Mapping[str, Any]) -> None:
    alignment = value.get("alignment")
    if not isinstance(alignment, Mapping):
        raise DatasetContractError("input_span requires alignment={start,end}")
    start, end = alignment.get("start"), alignment.get("end")
    if (
        isinstance(start, bool) or not isinstance(start, (int, float))
        or isinstance(end, bool) or not isinstance(end, (int, float))
        or not 0 <= float(start) < float(end) <= 1
    ):
        raise DatasetContractError("alignment ratios must satisfy 0 <= start < end <= 1")


def _validate_input_span(value: Mapping[str, Any], transcript: str | None) -> None:
    if transcript is None and "start_frame" in value:
        if not isinstance(value.get("value"), str) or not value["value"]:
            raise DatasetContractError("predicted input_span requires a resolved value")
        _validate_alignment({"alignment": {"start": value.get("start_ratio"), "end": value.get("end_ratio")}})
        if not isinstance(value.get("start_frame"), int) or not isinstance(value.get("end_frame"), int):
            raise DatasetContractError("predicted input_span requires integer frame bounds")
        return
    start, end, text = value.get("start"), value.get("end"), value.get("value")
    if (
        isinstance(start, bool) or not isinstance(start, int)
        or isinstance(end, bool) or not isinstance(end, int)
        or start < 0 or end <= start or not isinstance(text, str) or not text
    ):
        raise DatasetContractError("input_span requires start, end and non-empty value")
    if transcript is None:
        raise DatasetContractError("input_span requires transcript")
    if text != transcript[start:end]:
        raise DatasetContractError("input_span value must equal transcript[start:end]")
    _validate_alignment(value)


def _validate_parameter(
    domain: str,
    action_path: str,
    name: str,
    value: Any,
    schema: Mapping[str, Any],
    transcript: str | None,
) -> None:
    parameter_type = schema["type"]
    if parameter_type == "ENUM":
        if value not in schema["values"]:
            raise DatasetContractError(f"invalid enum value for {domain}.{action_path}.{name}")
    elif parameter_type == "NUMBER":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise DatasetContractError(f"{domain}.{action_path}.{name} must be numeric")
        if schema.get("minimum") is not None and value < schema["minimum"]:
            raise DatasetContractError(f"{domain}.{action_path}.{name} is below minimum")
        if schema.get("maximum") is not None and value > schema["maximum"]:
            raise DatasetContractError(f"{domain}.{action_path}.{name} is above maximum")
    elif parameter_type == "BOOLEAN":
        if not isinstance(value, bool):
            raise DatasetContractError(f"{domain}.{action_path}.{name} must be boolean")
    elif parameter_type in {"ENTITY", "FREE_TEXT"}:
        if not isinstance(value, Mapping):
            raise DatasetContractError(f"{domain}.{action_path}.{name} must be typed")
        source = value.get("source")
        if source == "input_span":
            _validate_input_span(value, transcript)
        elif source == "state_reference":
            allowed_paths = schema.get("state_reference_paths", ["state.current_target_application", "state.active_browser", "state.active_url", "state.active_media"])
            if value.get("path") not in allowed_paths:
                raise DatasetContractError(f"state reference is not allowlisted for {domain}.{action_path}.{name}")
        else:
            raise DatasetContractError(f"unsupported source for {domain}.{action_path}.{name}: {source!r}")
    elif parameter_type == "STATE_REFERENCE":
        if not isinstance(value, Mapping) or value.get("source") != "state_reference":
            raise DatasetContractError(f"{domain}.{action_path}.{name} requires state_reference")
        allowed_paths = schema.get("state_reference_paths", ["state.current_target_application", "state.active_browser", "state.active_url", "state.active_media"])
        if value.get("path") not in allowed_paths:
            raise DatasetContractError(f"state reference is not allowlisted for {domain}.{action_path}.{name}")
    elif parameter_type == "CONTEXT_REFERENCE":
        if not isinstance(value, Mapping) or value.get("source") != "context_reference":
            raise DatasetContractError(f"{domain}.{action_path}.{name} requires context_reference")
        if value.get("reference_type") not in schema.get("context_reference_types", CONTEXT_REFERENCE_TYPES):
            raise DatasetContractError(f"invalid context reference for {domain}.{action_path}.{name}")


def validate_target(
    target: dict[str, Any],
    config: dict[str, Any],
    transcript: str | None = None,
    capability_schemas: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    acts = config["ontology"]["acts"]
    if target.get("act") not in acts:
        raise DatasetContractError(f"unknown ACT: {target.get('act')!r}")
    schemas = {
        str(schema["name"]): schema for schema in capability_schemas
    } if capability_schemas is not None else {
        str(name): schema for name, schema in config["ontology"]["capabilities"].items()
    }
    context = target.get("context", {})
    requires_context = False
    if context:
        if not isinstance(context, Mapping):
            raise DatasetContractError("target.context must be an object")
        if context.get("relation") not in TURN_RELATIONS:
            raise DatasetContractError("unknown turn relation")
        requires_context = context.get("requires_context", False)
        if not isinstance(requires_context, bool):
            raise DatasetContractError("context.requires_context must be boolean")
        reference_type = context.get("reference_type")
        if reference_type is not None and reference_type not in CONTEXT_REFERENCE_TYPES:
            raise DatasetContractError("unknown context reference type")

    operations = target.get("operations", [])
    if not isinstance(operations, list):
        raise DatasetContractError("target.operations must be a list")
    maximum = int(config["operation_decoder"]["max_operations"])
    if len(operations) > maximum:
        raise DatasetContractError(f"target contains more than {maximum} operations")
    for order, operation in enumerate(operations, 1):
        if not isinstance(operation, Mapping):
            raise DatasetContractError("each operation must be an object")
        if operation.get("order", order) != order:
            raise DatasetContractError("operation order must be contiguous and 1-based")
        domain = operation.get("domain", operation.get("goal"))
        action_id = operation.get("action")
        if domain and action_id and "." not in str(action_id):
            action_id = f"{domain}.{action_id}"
        if not isinstance(domain, str) or domain not in schemas:
            raise DatasetContractError(f"unknown root action domain: {domain!r}")
        prefix = f"{domain}."
        if not isinstance(action_id, str) or not action_id.startswith(prefix):
            raise DatasetContractError(f"action is not canonical for domain {domain}: {action_id!r}")
        action_path = action_id[len(prefix):]
        action_collection = schemas[domain].get("actions", {})
        if isinstance(action_collection, Mapping):
            action_schema = action_collection.get(action_path)
        else:
            action_schema = {"parameters": schemas[domain].get("parameters", {})} if action_path in action_collection else None
        if action_schema is None:
            raise DatasetContractError(f"unknown action: {action_id}")
        parameters = operation.get("parameters", {})
        if not isinstance(parameters, Mapping):
            raise DatasetContractError("operation.parameters must be an object")
        parameter_schemas = action_schema.get("parameters", {})
        if not parameter_schemas and isinstance(schemas[domain].get("parameters"), Mapping):
            parameter_schemas = schemas[domain]["parameters"]
        extra = set(parameters) - set(parameter_schemas)
        if extra:
            raise DatasetContractError(f"unknown parameters for {action_id}: {sorted(extra)}")
        for name, parameter_schema in parameter_schemas.items():
            if name not in parameters:
                if parameter_schema.get("required") and not requires_context:
                    raise DatasetContractError(f"missing required parameter {action_id}.{name}")
                continue
            _validate_parameter(domain, action_path, name, parameters[name], parameter_schema, transcript)

    if target["act"] == "EXECUTE" and not operations:
        raise DatasetContractError("EXECUTE requires at least one operation")
    if target["act"] != "EXECUTE" and operations:
        raise DatasetContractError("non-EXECUTE targets cannot contain operations")

    relations = target.get("relations", [])
    if not isinstance(relations, list):
        raise DatasetContractError("target.relations must be a list")
    for relation in relations:
        if not isinstance(relation, Mapping):
            raise DatasetContractError("operation relation must be an object")
        source, destination = relation.get("source"), relation.get("target")
        if not isinstance(source, int) or not isinstance(destination, int) or not 0 <= source < len(operations) or not 0 <= destination < len(operations):
            raise DatasetContractError("operation relation endpoints are invalid")
        if relation.get("type") not in OPERATION_RELATIONS:
            raise DatasetContractError("unknown operation relation")


def validate_record(record: dict[str, Any], config: dict[str, Any], *, require_semantic: bool = False) -> None:
    required = {"sample_id", "audio"}
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
    transcript = record.get("transcript")
    if transcript is not None and not isinstance(transcript, str):
        raise DatasetContractError("transcript must be a string when provided")
    if require_semantic and not isinstance(record.get("target"), dict):
        raise DatasetContractError("semantic record requires target")
    if isinstance(record.get("target"), dict):
        validate_target(record["target"], config, transcript)


class VoiceSLUDataset(Dataset[dict[str, Any]]):
    """Waveform dataset supporting acoustic-only and semantic V1 records."""

    def __init__(self, records: Sequence[dict[str, Any]], config: dict[str, Any], audio_loader: Callable[[Any], Tensor] | None = None, require_semantic: bool = False) -> None:
        for record in records:
            validate_record(record, config, require_semantic=require_semantic)
        self.records = list(records)
        self.config = config
        self.audio_loader = audio_loader or self._default_audio_loader
        self.lexical_vocab = config["lexical_branch"]["vocab"]
        self.hop_length = int(config["audio"]["hop_length"])
        self.subsampling_factor = int(config["speech_encoder"]["subsampling_factor"])

    @staticmethod
    def _default_audio_loader(audio: Any) -> Tensor:
        if isinstance(audio, Tensor):
            values = audio.detach().clone().float()
        elif isinstance(audio, (list, tuple)):
            values = torch.tensor(audio, dtype=torch.float32)
        else:
            raise DatasetContractError("an audio_loader is required for path-based audio")
        if values.dim() != 1 or values.numel() < 2:
            raise DatasetContractError("loaded waveform must have shape [samples]")
        return values

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        waveform = self.audio_loader(record["audio"])
        transcript = record.get("transcript")
        labels = encode_lexical_text(transcript or "", self.lexical_vocab)
        estimated_input = max(1, int(waveform.numel()) // self.hop_length + 1)
        estimated_ctc = max(1, (estimated_input + self.subsampling_factor - 1) // self.subsampling_factor)
        required_ctc = ctc_required_length(labels)
        if labels and not ctc_target_length_is_valid(estimated_ctc, required_ctc):
            raise DatasetContractError(f"CTC target is impossible: required={required_ctc}, input={estimated_ctc}")
        return {
            "sample_id": record["sample_id"],
            "waveform": waveform,
            "target": record.get("target"),
            "transcript": transcript,
            "lexical_labels": torch.tensor(labels, dtype=torch.long),
            "estimated_ctc_input_length": estimated_ctc,
            "required_ctc_length": required_ctc,
            "metadata": record.get("metadata", {}),
        }


def collate_audio_batch(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not items:
        raise ValueError("cannot collate an empty batch")
    maximum = max(int(item["waveform"].numel()) for item in items)
    waveform = torch.zeros(len(items), maximum, dtype=torch.float32)
    audio_mask = torch.zeros(len(items), maximum, dtype=torch.bool)
    for row, item in enumerate(items):
        values = item["waveform"].float()
        waveform[row, : values.numel()] = values
        audio_mask[row, : values.numel()] = True
    labels = [item["lexical_labels"] for item in items]
    invalid = sum(item["required_ctc_length"] > item["estimated_ctc_input_length"] for item in items)
    return {
        "waveform": waveform,
        "audio_mask": audio_mask,
        "sample_id": [item["sample_id"] for item in items],
        "target": [item["target"] for item in items],
        "transcript": [item["transcript"] for item in items],
        "lexical_labels": torch.cat(labels) if any(label.numel() for label in labels) else torch.empty(0, dtype=torch.long),
        "lexical_target_lengths": torch.tensor([label.numel() for label in labels], dtype=torch.long),
        "ctc_invalid_count": invalid,
        "ctc_sample_count": len(items),
        "metadata": [item["metadata"] for item in items],
    }


MultiTaskDataset = VoiceSLUDataset
collate_batch = collate_audio_batch
