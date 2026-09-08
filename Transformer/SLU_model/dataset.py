"""Dataset boundaries for the voice-native VALLS SLU V0 model.

The repository does not contain the V0 dataset yet.  This module therefore
provides strict record validation and a small audio dataset adapter, without
inventing a missing corpus format or response-generation targets.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .config import PARAMETER_TYPES, validate_config


class DatasetContractError(ValueError):
    """Raised when a record cannot be trusted as a VALLS training example."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load non-empty JSONL records without silently skipping malformed rows."""

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


def _validate_parameter(
    goal: str,
    name: str,
    value: Any,
    schema: dict[str, Any],
) -> None:
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
        if not isinstance(value, dict) or value.get("source") != "input_span":
            raise DatasetContractError(f"{goal}.{name} must use an input_span value")
        _validate_input_span(value)
    elif parameter_type == "STATE_REFERENCE":
        if not isinstance(value, dict) or value.get("source") != "state_reference":
            raise DatasetContractError(f"{goal}.{name} must use a state_reference value")
        if value.get("path") not in schema.get("state_reference_paths", []):
            raise DatasetContractError(f"state reference is not allowlisted for {goal}.{name}")


def _validate_input_span(value: dict[str, Any]) -> None:
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


def validate_target(target: dict[str, Any], config: dict[str, Any]) -> None:
    """Validate one untrusted semantic target against capability schemas."""

    acts = config["ontology"]["acts"]
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
        if goal not in config["ontology"]["capabilities"]:
            raise DatasetContractError(f"unknown capability schema: {goal!r}")
        schema = config["ontology"]["capabilities"][goal]
        parameters = operation.get("parameters", {})
        if not isinstance(parameters, dict):
            raise DatasetContractError("operation.parameters must be an object")
        parameter_schemas = schema["parameters"]
        extra = set(parameters) - set(parameter_schemas)
        if extra:
            raise DatasetContractError(f"unknown parameters for {goal}: {sorted(extra)}")
        for name, parameter_schema in parameter_schemas.items():
            required = bool(parameter_schema.get("required", False))
            if name not in parameters:
                if required:
                    raise DatasetContractError(f"missing required parameter {goal}.{name}")
                continue
            _validate_parameter(goal, name, parameters[name], parameter_schema)

    if target["act"] == "EXECUTE" and not operations:
        raise DatasetContractError("EXECUTE requires at least one operation")
    if target["act"] != "EXECUTE" and operations:
        raise DatasetContractError("non-EXECUTE targets cannot contain operations")


def validate_record(record: dict[str, Any], config: dict[str, Any]) -> None:
    """Validate the V0 record envelope without assuming a physical audio format."""

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
    validate_target(record["target"], config)


class VoiceSLUDataset(Dataset[dict[str, Any]]):
    """Minimal dataset adapter for future waveform-backed V0 records.

    ``audio_loader`` is injected because the source dataset and storage format
    are intentionally not part of the current architecture contract.
    """

    def __init__(
        self,
        records: Sequence[dict[str, Any]],
        config: dict[str, Any],
        audio_loader: Callable[[Any], Tensor] | None = None,
    ) -> None:
        validate_config(config)
        for record in records:
            validate_record(record, config)
        self.records = list(records)
        self.config = config
        self.audio_loader = audio_loader or self._default_audio_loader

    @staticmethod
    def _default_audio_loader(audio: Any) -> Tensor:
        if isinstance(audio, Tensor):
            waveform = audio.detach().clone().float()
        elif isinstance(audio, (list, tuple)):
            waveform = torch.tensor(audio, dtype=torch.float32)
        else:
            raise DatasetContractError(
                "an audio_loader is required for path-based audio records"
            )
        if waveform.dim() != 1 or waveform.numel() < 2:
            raise DatasetContractError("loaded waveform must have shape [samples]")
        return waveform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        waveform = self.audio_loader(record["audio"])
        if waveform.dim() != 1 or waveform.numel() < 2:
            raise DatasetContractError("audio_loader returned an invalid waveform")
        return {
            "sample_id": record["sample_id"],
            "waveform": waveform,
            "target": record["target"],
            "metadata": record.get("metadata", {}),
        }


def collate_audio_batch(items: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Pad waveform records and retain targets for training-time encoding."""

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
    return {
        "waveform": waveform,
        "audio_mask": audio_mask,
        "sample_id": [item["sample_id"] for item in items],
        "target": [item["target"] for item in items],
        "metadata": [item["metadata"] for item in items],
    }


# Compatibility names are intentionally descriptive aliases for the new
# waveform dataset; no legacy text/response targets are reintroduced.
MultiTaskDataset = VoiceSLUDataset
collate_batch = collate_audio_batch
