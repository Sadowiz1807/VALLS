"""Canonical records to masked multi-task labels."""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .config import ABSENT, INPUT_SPAN, NONE, STATE_REFERENCE

IGNORE_INDEX = -100
PREMATURE_SUCCESS_PATTERNS = (
    r"\bđã\s+(được\s+)?(mở|đóng|phát|dừng|chuyển|tìm|truy cập|thực hiện|chạy|cuộn|tải)\b",
    r"\b(mở|đóng|phát|dừng|chuyển|tìm|truy cập|thực hiện|chạy|cuộn|tải)\b.*\b(rồi|xong|hoàn tất)\b",
    r"\bđã\b.*\b(thành công|xong|hoàn tất)\b",
    r"\b(opened|closed|played|stopped|searched|navigated|executed|launched|downloaded|completed|done|successfully)\b",
)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def serialize_model_input(
    record: dict[str, Any], special_tokens: dict[str, str],
    context: Sequence[dict[str, Any]] | None = None,
) -> str:
    state = {
        key: record["state"][key]
        for key in ("current_target_application", "active_media", "active_url", "active_browser")
        if key in record["state"]
    }
    if isinstance(record["state"].get("source_active_frames"), list):
        state["source_active_frames"] = record["state"]["source_active_frames"][-1:]
    metadata = {
        key: record["metadata"][key]
        for key in ("language_mode", "locale", "asr_noise")
        if key in record["metadata"]
    }
    return "".join([
        special_tokens["context_open"], canonical_json(record["context"] if context is None else context), special_tokens["context_close"],
        special_tokens["state_open"], canonical_json(state), special_tokens["state_close"],
        special_tokens["metadata_open"], canonical_json(metadata), special_tokens["metadata_close"],
        special_tokens["input_open"], record["current_text"], special_tokens["input_close"],
    ])


def _token_span(offsets: list[tuple[int, int]], start: int, end: int) -> tuple[int, int]:
    indices = [index for index, (left, right) in enumerate(offsets) if right > start and left < end]
    if not indices:
        raise ValueError(f"Character span [{start}, {end}) does not align to any input token")
    if offsets[indices[0]][0] != start or offsets[indices[-1]][1] != end:
        raise ValueError(f"Character span [{start}, {end}) does not match exact token boundaries")
    return indices[0], indices[-1]


def _validate_source_span(value: dict[str, Any], text: str) -> None:
    start, end = value.get("start"), value.get("end")
    if not isinstance(start, int) or not isinstance(end, int) or not 0 <= start < end <= len(text):
        raise ValueError("input_span requires valid character start/end")
    if value.get("value") != text[start:end]:
        raise ValueError("input_span value must equal the source text slice")


def validate_response_target(record: dict[str, Any]) -> None:
    text = record.get("gold_response_text")
    metadata = record.get("response_metadata")
    if not isinstance(text, str) or not text.strip():
        raise ValueError("gold_response_text must be a non-empty string")
    if not isinstance(metadata, dict):
        raise ValueError("response_metadata must be an object")
    if metadata.get("owner") != "dataset_gold" or metadata.get("model_generated") is not False:
        raise ValueError("Gold responses require owner=dataset_gold and model_generated=false")
    target = record.get("target", {})
    if target.get("mapping_status") == "CANDIDATE":
        target = target.get("canonical_semantic", {})
    act = target.get("act")
    expected_phase = "pre_execution" if act == "EXECUTE" else "direct_response"
    if metadata.get("phase") != expected_phase:
        raise ValueError(f"{act} response phase must be {expected_phase}")
    if not isinstance(metadata.get("provenance"), str) or not metadata["provenance"]:
        raise ValueError("response_metadata.provenance is required")
    if act == "EXECUTE" and any(
        re.search(pattern, text, re.IGNORECASE) for pattern in PREMATURE_SUCCESS_PATTERNS
    ):
        raise ValueError("EXECUTE pre-execution response must not claim success")


def validate_generated_response(response_text: str, act: str) -> str:
    """Reject unsafe model text before it reaches the user or executor."""
    response_text = response_text.strip()
    if not response_text:
        raise ValueError("Model generated an empty response")
    if act == "EXECUTE" and any(
        re.search(pattern, response_text, re.IGNORECASE)
        for pattern in PREMATURE_SUCCESS_PATTERNS
    ):
        raise ValueError("EXECUTE response contains a premature success claim")
    return response_text


def _parameter_labels(
    target: dict[str, Any], text: str, serialized_offsets: list[tuple[int, int]],
    input_char_start: int, config: dict[str, Any],
) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
    goal = target.get("goal")
    categorical = {
        key: torch.tensor(IGNORE_INDEX, dtype=torch.long)
        for key in config["heads"]["categorical"]
    }
    spans = {
        key: torch.tensor([IGNORE_INDEX, IGNORE_INDEX], dtype=torch.long)
        for key in config["heads"]["spans"]
    }
    if goal not in config["ontology"]["goal_parameters"]:
        return categorical, spans

    parameters = target.get("parameters", {})
    specification = config["ontology"]["goal_parameters"][goal]
    for name, parameter_spec in specification["properties"].items():
        key = f"{goal}__{name}"
        value = parameters.get(name, ABSENT)
        if parameter_spec.get("type") == "dynamic":
            source_key = f"{key}__source"
            if value == ABSENT:
                label = ABSENT
            elif not isinstance(value, dict):
                raise ValueError(f"{key} must be a typed dynamic value")
            elif value.get("source") == "input_span":
                _validate_source_span(value, text)
                label = INPUT_SPAN
                local = _token_span(
                    serialized_offsets,
                    input_char_start + value["start"],
                    input_char_start + value["end"],
                )
                spans[key] = torch.tensor([local[0] + 1, local[1] + 1], dtype=torch.long)
            elif value.get("source") == "state_reference":
                paths = config["heads"]["references"].get(key, [])
                if value.get("path") not in paths:
                    raise ValueError(f"Non-allowlisted state reference for {key}")
                label = STATE_REFERENCE
                categorical[f"{key}__reference"] = torch.tensor(
                    paths.index(value["path"]), dtype=torch.long
                )
            else:
                raise ValueError(f"Unknown dynamic source for {key}")
            categorical[source_key] = torch.tensor(
                config["heads"]["categorical"][source_key].index(label), dtype=torch.long
            )
        elif name == "tab_index":
            source_key = f"{key}__source"
            if value == ABSENT:
                categorical[source_key] = torch.tensor(0, dtype=torch.long)
            else:
                matches = list(re.finditer(rf"(?<!\d){int(value)}(?!\d)", text))
                if len(matches) != 1:
                    raise ValueError("tab_index must appear exactly once in current_text")
                match = matches[0]
                local = _token_span(
                    serialized_offsets,
                    input_char_start + match.start(),
                    input_char_start + match.end(),
                )
                spans[key] = torch.tensor([local[0] + 1, local[1] + 1])
                categorical[source_key] = torch.tensor(
                    config["heads"]["categorical"][source_key].index(INPUT_SPAN)
                )
        elif name == "arguments":
            if value != ABSENT and value is not None and value != {}:
                raise ValueError("RUN_COMMAND.arguments requires a command-specific schema")
        elif key in categorical:
            label = str(value) if value != ABSENT else ABSENT
            try:
                categorical[key] = torch.tensor(
                    config["heads"]["categorical"][key].index(label), dtype=torch.long
                )
            except ValueError as exc:
                raise ValueError(f"Invalid categorical value {label!r} for {key}") from exc
    return categorical, spans


class MultiTaskDataset(Dataset):
    def __init__(self, records: Sequence[dict[str, Any]], tokenizer: Any, config: dict[str, Any]) -> None:
        for record in records:
            missing_fields = set(config["data"]["required_fields"]) - set(record)
            if missing_fields:
                raise ValueError(f"Record is missing required fields: {sorted(missing_fields)}")
            if not isinstance(record["current_text"], str) or not record["current_text"]:
                raise ValueError("current_text must be a non-empty string")
            if not isinstance(record["context"], list) or not isinstance(record["state"], dict) or not isinstance(record["metadata"], dict):
                raise ValueError("context/state/metadata have invalid types")
            target = record.get("target", {})
            if "mapping_status" in target and target.get("mapping_status") != "CANDIDATE":
                raise ValueError("MultiTaskDataset accepts only CANDIDATE records")
            validate_response_target(record)
        self.records = records
        self.tokenizer = tokenizer
        self.config = config
        special = config["tokenizer"]["special_tokens"]
        self.bos_id = tokenizer.token_to_id(special["bos"])
        self.eos_id = tokenizer.token_to_id(special["eos"])
        self.pad_id = tokenizer.token_to_id(special["pad"])
        if None in {self.bos_id, self.eos_id, self.pad_id}:
            raise ValueError("Tokenizer is missing BOS/EOS/PAD")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        target_container = record["target"]
        if target_container.get("mapping_status") == "CANDIDATE":
            target = target_container.get("canonical_semantic")
            if not isinstance(target, dict):
                raise ValueError("CANDIDATE requires target.canonical_semantic")
        else:
            target = target_container
        act = target["act"]
        goal = target.get("goal") or NONE
        if act not in self.config["labels"]["acts"]:
            raise ValueError(f"Unknown ACT: {act}")
        if goal not in self.config["labels"]["goals"]:
            raise ValueError(f"Unknown GOAL: {goal}")
        if goal not in self.config["labels"]["goal_masks_by_act"][act]:
            raise ValueError(f"GOAL {goal} is incompatible with ACT {act}")
        parameters = target.get("parameters", {})
        if not isinstance(parameters, dict):
            raise ValueError("target.parameters must be an object")
        specification = self.config["ontology"]["goal_parameters"].get(target.get("goal"), {})
        properties = specification.get("properties", {})
        extra = set(parameters) - set(properties)
        missing = set(specification.get("required", [])) - set(parameters)
        if extra or missing:
            raise ValueError(f"Invalid parameters: extra={sorted(extra)}, missing={sorted(missing)}")

        context = record["context"]
        serialized = serialize_model_input(
            record, self.config["tokenizer"]["special_tokens"], context
        )
        full_encoding = self.tokenizer.encode(serialized)
        maximum = int(self.config["model"]["max_input_length"])
        while context and len(full_encoding.ids) + 2 > maximum:
            context = context[1:]
            serialized = serialize_model_input(
                record, self.config["tokenizer"]["special_tokens"], context
            )
            full_encoding = self.tokenizer.encode(serialized)
        input_char_start = serialized.rindex(record["current_text"])
        ids = [self.bos_id, *full_encoding.ids, self.eos_id]
        if len(ids) > maximum:
            raise ValueError(f"Input requires {len(ids)} tokens, maximum is {maximum}")
        ids.extend([self.pad_id] * (maximum - len(ids)))
        labels, spans = _parameter_labels(
            target,
            record["current_text"],
            full_encoding.offsets,
            input_char_start,
            self.config,
        )
        text_end = input_char_start + len(record["current_text"])
        token_offsets = [(-1, -1)]
        token_offsets.extend(
            (left - input_char_start, right - input_char_start)
            if left >= input_char_start and right <= text_end else (-1, -1)
            for left, right in full_encoding.offsets
        )
        token_offsets.append((-1, -1))
        token_offsets.extend([(-1, -1)] * (maximum - len(token_offsets)))
        input_ids = torch.tensor(ids, dtype=torch.long)
        response_text = record["gold_response_text"].strip()
        response_ids = self.tokenizer.encode(response_text).ids
        response_maximum = int(self.config["model"]["max_response_length"])
        if len(response_ids) + 1 > response_maximum:
            raise ValueError(
                f"Response requires {len(response_ids) + 1} tokens, maximum is {response_maximum}"
            )
        # Dynamic response tokens (unpadded at item level, padded dynamically in collate_batch)
        decoder_input_ids = [self.bos_id, *response_ids]
        response_labels = [*response_ids, self.eos_id]
        return {
            "input_ids": input_ids,
            "token_offsets": torch.tensor(token_offsets, dtype=torch.long),
            "input_mask": (input_ids != self.pad_id).unsqueeze(0).unsqueeze(0),
            "act_label": torch.tensor(self.config["labels"]["acts"].index(act)),
            "goal_label": torch.tensor(self.config["labels"]["goals"].index(goal)),
            "categorical_labels": labels,
            "span_labels": spans,
            "decoder_input_ids": torch.tensor(decoder_input_ids, dtype=torch.long),
            "response_labels": torch.tensor(response_labels, dtype=torch.long),
            "gold_response_text": response_text,
            "response_metadata": record["response_metadata"],
            "text": record["current_text"],
            "target": target,
        }


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    # ponytail: dynamic padding for decoder sequence (min 64, max 1024, batch-aligned)
    max_dec_len = max(len(item["decoder_input_ids"]) for item in items)
    batch_dec_len = min(1024, max(64, max_dec_len))
    pad_id = 0  # unigram bytelevel pad
    ignore_index = IGNORE_INDEX

    padded_dec_inputs = []
    padded_resp_labels = []
    resp_masks = []

    for item in items:
        dec_ids = item["decoder_input_ids"].tolist()
        resp_lbls = item["response_labels"].tolist()
        cur_len = len(dec_ids)
        pad_len = max(0, batch_dec_len - cur_len)
        padded_dec_inputs.append(torch.tensor(dec_ids[:batch_dec_len] + [pad_id] * pad_len, dtype=torch.long))
        padded_resp_labels.append(torch.tensor(resp_lbls[:batch_dec_len] + [ignore_index] * pad_len, dtype=torch.long))
        resp_masks.append(torch.tensor([True] * min(cur_len, batch_dec_len) + [False] * pad_len, dtype=torch.bool))

    return {
        "input_ids": torch.stack([item["input_ids"] for item in items]),
        "token_offsets": torch.stack([item["token_offsets"] for item in items]),
        "input_mask": torch.stack([item["input_mask"] for item in items]),
        "act_label": torch.stack([item["act_label"] for item in items]),
        "goal_label": torch.stack([item["goal_label"] for item in items]),
        "categorical_labels": {
            key: torch.stack([item["categorical_labels"][key] for item in items])
            for key in items[0]["categorical_labels"]
        },
        "span_labels": {
            key: torch.stack([item["span_labels"][key] for item in items])
            for key in items[0]["span_labels"]
        },
        "decoder_input_ids": torch.stack(padded_dec_inputs),
        "response_labels": torch.stack(padded_resp_labels),
        "response_mask": torch.stack(resp_masks),
        "gold_response_text": [item["gold_response_text"] for item in items],
        "response_metadata": [item["response_metadata"] for item in items],
        "text": [item["text"] for item in items],
        "target": [item["target"] for item in items],
    }
