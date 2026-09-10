"""Decode VALLS SLU V1 logits into a TurnUnderstanding contract.

This module performs deterministic semantic decoding only. It does not resolve
runtime context, graph/memory references, policy, providers, or execution.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from .config import (
    CONTEXT_REFERENCE_TYPES,
    OPERATION_RELATIONS,
    TURN_RELATIONS,
    canonical_action_id,
)
from .dataset import DatasetContractError
from .model import SpanExtractor, VoiceNativeSLU


def _confidence(values: Tensor, index: int) -> float:
    return float(values.softmax(-1)[index])


def decode_turn_understanding(
    outputs: Mapping[str, Any],
    model: VoiceNativeSLU,
    *,
    config: Mapping[str, Any] | None = None,
    request_id: str | None = None,
) -> dict[str, Any]:
    """Decode one-item model output without applying Harness resolution."""

    config = config or model.config
    if outputs["act_logits"].size(0) != 1:
        raise ValueError("decode_turn_understanding accepts one sample")
    act_logits = outputs["act_logits"][0]
    act_id = int(act_logits.argmax())
    act = config["ontology"]["acts"][act_id]
    context_logits = outputs["turn_relation_logits"][0]
    relation_id = int(context_logits.argmax())
    relation = TURN_RELATIONS[relation_id]
    requires_context = bool(outputs["context_required_logit"][0].sigmoid() >= 0.5)
    reference_id = int(outputs["context_reference_logits"][0].argmax())
    reference_type = CONTEXT_REFERENCE_TYPES[reference_id] if requires_context else None

    schemas = outputs["capability_schemas"]
    schema_names = outputs["schema_names"]
    presence = outputs["operation_presence_logits"][0].sigmoid() >= float(config["inference"]["operation_presence_threshold"])
    operations: list[dict[str, Any]] = []
    for index in range(outputs["operation_queries"].size(1)):
        if not bool(presence[index]):
            continue
        goal_scores = outputs["goal_scores"][0, index]
        schema_index = int(goal_scores.argmax())
        domain = schema_names[schema_index]
        schema = schemas[schema_index]
        action_logits = outputs["action_scores"][schema_index][0, index]
        action_index = int(action_logits.argmax())
        action_path = list(schema["actions"])[action_index]
        action_id = canonical_action_id(domain, action_path)
        parameters: dict[str, Any] = {}
        for name, parameter in schema["actions"][action_path].get("parameters", {}).items():
            head = outputs["parameter_outputs"][domain][action_path][name]
            if int(head["presence_logits"][0, index].argmax()) != 1:
                continue
            kind = parameter["type"]
            if kind == "ENUM":
                parameters[name] = parameter["values"][int(head["enum_logits"][0, index].argmax())]
            elif kind == "NUMBER":
                value = float(head["number_value"][0, index])
                if parameter.get("minimum") is not None:
                    value = max(value, float(parameter["minimum"]))
                if parameter.get("maximum") is not None:
                    value = min(value, float(parameter["maximum"]))
                parameters[name] = int(round(value)) if value.is_integer() else value
            elif kind == "BOOLEAN":
                parameters[name] = bool(head["boolean_logits"][0, index].argmax())
            elif kind == "CONTEXT_REFERENCE":
                choices = parameter.get("context_reference_types", CONTEXT_REFERENCE_TYPES)
                parameters[name] = {"source": "context_reference", "reference_type": choices[int(head["context_reference_logits"][0, index].argmax())]}
            elif kind == "STATE_REFERENCE":
                choices = parameter.get("state_reference_paths", [])
                if "reference_logits" not in head or not choices:
                    raise DatasetContractError(f"no state reference contract for {action_id}.{name}")
                parameters[name] = {"source": "state_reference", "path": choices[int(head["reference_logits"][0, index].argmax())]}
            elif kind in {"ENTITY", "FREE_TEXT"}:
                source_id = int(head["source_logits"][0, index].argmax())
                if source_id == SpanExtractor.SOURCES.index("STATE_REFERENCE"):
                    choices = parameter.get("state_reference_paths", [])
                    if "reference_logits" not in head or not choices:
                        raise DatasetContractError(f"no state reference contract for {action_id}.{name}")
                    parameters[name] = {"source": "state_reference", "path": choices[int(head["reference_logits"][0, index].argmax())]}
                elif source_id == SpanExtractor.SOURCES.index("INPUT_SPAN"):
                    start = int(head["span_start_logits"][0, index].argmax())
                    end = max(start, int(head["span_end_logits"][0, index].argmax()))
                    value = model.lexical_span_resolver.resolve(outputs["lexical_ctc_logits"][0], start, end)
                    if not value:
                        raise DatasetContractError(f"lexical span is empty for {action_id}.{name}")
                    frame_count = head["span_start_logits"].size(-1)
                    parameters[name] = {"source": "input_span", "value": value, "start_frame": start, "end_frame": end, "start_ratio": start / max(frame_count - 1, 1), "end_ratio": (end + 1) / max(frame_count, 1)}
                else:
                    raise DatasetContractError(f"invalid source prediction for {action_id}.{name}")
            else:
                raise DatasetContractError(f"unsupported parameter type: {kind}")
        operations.append({"order": index + 1, "domain": domain, "action": action_id, "parameters": parameters})

    relations: list[dict[str, Any]] = []
    relation_logits = outputs["operation_relation_logits"][0]
    for source in range(len(operations)):
        for destination in range(len(operations)):
            if source == destination:
                continue
            relation_id = int(relation_logits[source, destination].argmax())
            if relation_id:
                relations.append({"source": source, "target": destination, "type": OPERATION_RELATIONS[relation_id]})
    return {
        "request_id": request_id,
        "act": act,
        "context": {"relation": relation, "requires_context": requires_context, "reference_type": reference_type},
        "operations": operations,
        "relations": relations,
        "confidence": {"act": _confidence(act_logits, act_id), "goal": 0.0 if not operations else 1.0, "parameters": 1.0, "ood": float(outputs["ood_logit"][0].sigmoid()), "overall": _confidence(act_logits, act_id)},
    }


# Short name for callers that already own the model/output boundary.
decode = decode_turn_understanding
