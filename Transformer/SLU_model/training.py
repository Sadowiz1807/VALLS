"""Training and evaluation utilities for VALLS SLU V1.

This module trains semantic understanding heads only. Runtime context, policy,
graph/memory resolution, tool selection, and execution remain Harness duties.
"""

from __future__ import annotations

import copy
import json
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import load_model as load_safetensors
from safetensors.torch import save_model as save_safetensors
from torch import Tensor
from tqdm.auto import tqdm

from .config import OPERATION_RELATIONS, TURN_RELATIONS, validate_config
from .dataset import DatasetContractError
from .model import SpanExtractor, VoiceNativeSLU, build_model

IGNORE_INDEX = -100


class TrainingStage(str, Enum):
    ACOUSTIC = "ACOUSTIC"
    BRIDGE = "BRIDGE"
    SEMANTIC = "SEMANTIC"
    PARAMETER = "PARAMETER"
    SAFETY = "SAFETY"
    JOINT = "JOINT"


_STAGE_MODULES: dict[TrainingStage, set[str]] = {
    TrainingStage.ACOUSTIC: {"speech_encoder", "lexical_head"},
    TrainingStage.BRIDGE: {"semantic_resampler", "bridge_alignment_head"},
    TrainingStage.SEMANTIC: {
        "semantic_resampler", "semantic_core", "operation_decoder", "schema_encoder",
        "schema_retriever", "action_resolver", "act_head", "turn_relation_head",
        "context_required_head", "context_reference_head", "operation_relation_head",
    },
    TrainingStage.PARAMETER: {
        "semantic_resampler", "semantic_core", "operation_decoder", "schema_encoder",
        "schema_retriever", "action_resolver", "parameter_extractor",
    },
    TrainingStage.SAFETY: {"semantic_core", "confidence_head", "ood_head"},
    TrainingStage.JOINT: {
        "speech_encoder", "semantic_resampler", "semantic_core", "operation_decoder",
        "schema_encoder", "schema_retriever", "action_resolver", "parameter_extractor",
        "act_head", "turn_relation_head", "context_required_head", "context_reference_head",
        "operation_relation_head", "confidence_head", "ood_head", "lexical_head",
    },
}


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def _schema_map(schemas: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(schema["name"]): schema for schema in schemas}


def _target_act_ids(batch: Mapping[str, Any], config: Mapping[str, Any], device: torch.device) -> Tensor:
    acts = config["ontology"]["acts"]
    return torch.tensor([acts.index(target["act"]) for target in batch["target"]], dtype=torch.long, device=device)


def _target_operation_presence(batch: Mapping[str, Any], maximum: int, device: torch.device) -> Tensor:
    return torch.tensor(
        [[index < len(target.get("operations", [])) for index in range(maximum)] for target in batch["target"]],
        dtype=torch.float32,
        device=device,
    )


def _target_ood(batch: Mapping[str, Any], device: torch.device) -> Tensor:
    return torch.tensor(
        [float(bool(target.get("ood", target.get("act") == "UNSUPPORTED"))) for target in batch["target"]],
        dtype=torch.float32,
        device=device,
    )


def _target_confidence(batch: Mapping[str, Any], device: torch.device) -> tuple[Tensor, Tensor]:
    values: list[list[float]] = []
    masks: list[list[float]] = []
    for target in batch["target"]:
        confidence = target.get("confidence")
        if not isinstance(confidence, Mapping):
            values.append([0.0, 0.0, 0.0])
            masks.append([0.0, 0.0, 0.0])
            continue
        row = [confidence.get(key) for key in ("act", "goal", "parameters")]
        values.append([float(value) if value is not None else 0.0 for value in row])
        masks.append([1.0 if value is not None else 0.0 for value in row])
    return torch.tensor(values, device=device), torch.tensor(masks, device=device)


def _zero(outputs: Mapping[str, Any]) -> Tensor:
    anchor = outputs.get("act_logits", outputs["bridge_states"])
    return anchor.sum() * 0.0


def _parameter_loss(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    schemas: Sequence[Mapping[str, Any]],
) -> list[Tensor]:
    """Train only the selected action's parameters; no unrelated-action negatives."""

    device = outputs["operation_queries"].device
    schema_map = _schema_map(schemas)
    losses: list[Tensor] = []
    for operation_index in range(outputs["operation_queries"].size(1)):
        for row, target in enumerate(batch["target"]):
            operations = target.get("operations", [])
            if operation_index >= len(operations):
                continue
            operation = operations[operation_index]
            domain = str(operation.get("domain", ""))
            canonical_action = str(operation.get("action", ""))
            prefix = f"{domain}."
            action_path = canonical_action[len(prefix):] if canonical_action.startswith(prefix) else ""
            schema = schema_map.get(domain)
            if schema is None or action_path not in schema.get("actions", {}):
                raise DatasetContractError(f"unknown V1 action: {canonical_action}")
            action_schema = schema["actions"][action_path]
            outputs_for_action = outputs["parameter_outputs"][domain][action_path]
            gold_parameters = operation.get("parameters", {})
            for name, parameter in action_schema.get("parameters", {}).items():
                head = outputs_for_action[name]
                present = name in gold_parameters
                label = torch.tensor([int(present)], dtype=torch.long, device=device)
                losses.append(F.cross_entropy(
                    head["presence_logits"][row:row + 1, operation_index],
                    label,
                    weight=torch.tensor([0.25, 1.0], dtype=torch.float32, device=device),
                ))
                if not present:
                    continue
                value = gold_parameters[name]
                kind = parameter["type"]
                if kind == "ENUM":
                    label = torch.tensor([parameter["values"].index(value)], dtype=torch.long, device=device)
                    losses.append(F.cross_entropy(head["enum_logits"][row:row + 1, operation_index], label))
                elif kind == "NUMBER":
                    losses.append(F.smooth_l1_loss(
                        head["number_value"][row:row + 1, operation_index],
                        torch.tensor([float(value)], dtype=torch.float32, device=device),
                    ))
                elif kind == "BOOLEAN":
                    label = torch.tensor([int(value)], dtype=torch.long, device=device)
                    losses.append(F.cross_entropy(head["boolean_logits"][row:row + 1, operation_index], label))
                elif kind in {"ENTITY", "FREE_TEXT", "STATE_REFERENCE"}:
                    source = value.get("source") if isinstance(value, Mapping) else None
                    if kind == "STATE_REFERENCE" or source == "state_reference":
                        paths = parameter.get("state_reference_paths", [])
                        if value.get("path") not in paths or "reference_logits" not in head:
                            raise DatasetContractError(f"invalid state reference for {domain}.{action_path}.{name}")
                        label = torch.tensor([paths.index(value["path"])], dtype=torch.long, device=device)
                        losses.append(F.cross_entropy(head["reference_logits"][row:row + 1, operation_index], label))
                        if kind in {"ENTITY", "FREE_TEXT"}:
                            source_label = torch.tensor([SpanExtractor.SOURCES.index("STATE_REFERENCE")], dtype=torch.long, device=device)
                            losses.append(F.cross_entropy(head["source_logits"][row:row + 1, operation_index], source_label))
                    elif source == "input_span":
                        if kind == "STATE_REFERENCE":
                            raise DatasetContractError(f"STATE_REFERENCE cannot use input_span: {domain}.{action_path}.{name}")
                        alignment = value.get("alignment")
                        if not isinstance(alignment, Mapping):
                            raise DatasetContractError(f"missing alignment for {domain}.{action_path}.{name}")
                        source_label = torch.tensor([SpanExtractor.SOURCES.index("INPUT_SPAN")], dtype=torch.long, device=device)
                        losses.append(F.cross_entropy(head["source_logits"][row:row + 1, operation_index], source_label))
                        frame_count = head["span_start_logits"].size(-1)
                        start = torch.tensor([round(float(alignment["start"]) * (frame_count - 1))], dtype=torch.long, device=device)
                        end = torch.tensor([round(float(alignment["end"]) * (frame_count - 1))], dtype=torch.long, device=device)
                        losses.extend([
                            F.cross_entropy(head["span_start_logits"][row:row + 1, operation_index], start),
                            F.cross_entropy(head["span_end_logits"][row:row + 1, operation_index], end),
                        ])
                    else:
                        raise DatasetContractError(f"unsupported parameter source for {domain}.{action_path}.{name}")
                elif kind == "CONTEXT_REFERENCE":
                    if value.get("source") != "context_reference":
                        raise DatasetContractError(f"invalid context reference for {domain}.{action_path}.{name}")
                    choices = parameter.get("context_reference_types", [])
                    label = torch.tensor([choices.index(value["reference_type"])], dtype=torch.long, device=device)
                    losses.append(F.cross_entropy(head["context_reference_logits"][row:row + 1, operation_index], label))
    return losses


def compute_loss(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    config: Mapping[str, Any],
    enabled_losses: set[str] | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute only the losses enabled by the selected curriculum stage."""

    enabled = set(config["training"]["loss_weights"]) if enabled_losses is None else set(enabled_losses)
    losses: dict[str, Tensor] = {}
    device = outputs.get("act_logits", outputs["bridge_states"]).device
    zero = _zero(outputs)
    schemas = outputs.get("capability_schemas", [])

    if "act" in enabled and "act_logits" in outputs:
        losses["act"] = F.cross_entropy(outputs["act_logits"], _target_act_ids(batch, config, device))
    if "turn_relation" in enabled and "turn_relation_logits" in outputs:
        labels = torch.tensor([TURN_RELATIONS.index(target.get("context", {}).get("relation", "NEW")) for target in batch["target"]], dtype=torch.long, device=device)
        losses["turn_relation"] = F.cross_entropy(outputs["turn_relation_logits"], labels)
    if "context_required" in enabled and "context_required_logit" in outputs:
        labels = torch.tensor([float(target.get("context", {}).get("requires_context", False)) for target in batch["target"]], device=device)
        losses["context_required"] = F.binary_cross_entropy_with_logits(outputs["context_required_logit"], labels)
    if "context_reference" in enabled and "context_reference_logits" in outputs:
        active_rows, labels = [], []
        choices = config["relations"]["context_reference_types"]
        for row, target in enumerate(batch["target"]):
            value = target.get("context", {}).get("reference_type")
            if value is not None:
                active_rows.append(row); labels.append(choices.index(value))
        losses["context_reference"] = F.cross_entropy(outputs["context_reference_logits"][active_rows], torch.tensor(labels, dtype=torch.long, device=device)) if labels else zero
    if "operation_relations" in enabled and "operation_relation_logits" in outputs:
        labels = torch.full(outputs["operation_relation_logits"].shape[:3], IGNORE_INDEX, dtype=torch.long, device=device)
        for row, target in enumerate(batch["target"]):
            operation_count = len(target.get("operations", []))
            if operation_count < 2:
                continue
            for source in range(operation_count):
                for destination in range(operation_count):
                    if source != destination:
                        labels[row, source, destination] = 0
            for relation in target.get("relations", []):
                labels[row, relation["source"], relation["target"]] = OPERATION_RELATIONS.index(relation["type"])
        active = labels != IGNORE_INDEX
        losses["operation_relations"] = F.cross_entropy(outputs["operation_relation_logits"][active], labels[active]) if active.any() else zero
    if "operation_presence" in enabled and "operation_presence_logits" in outputs:
        losses["operation_presence"] = F.binary_cross_entropy_with_logits(outputs["operation_presence_logits"], _target_operation_presence(batch, outputs["operation_presence_logits"].size(1), device))
    if "goal_retrieval" in enabled and "goal_scores" in outputs:
        labels, logits = [], []
        for row, target in enumerate(batch["target"]):
            for index, operation in enumerate(target.get("operations", [])):
                domain = operation["domain"]
                labels.append(next(i for i, schema in enumerate(schemas) if schema["name"] == domain))
                logits.append(outputs["goal_scores"][row, index])
        losses["goal_retrieval"] = F.cross_entropy(torch.stack(logits), torch.tensor(labels, dtype=torch.long, device=device)) if logits else zero
    if "action" in enabled and "action_scores" in outputs:
        labels, logits = [], []
        for row, target in enumerate(batch["target"]):
            for index, operation in enumerate(target.get("operations", [])):
                domain = str(operation.get("domain", "")); action = str(operation["action"])
                path = action.removeprefix(f"{domain}.")
                schema_index = next(i for i, schema in enumerate(schemas) if schema["name"] == domain)
                actions = list(schemas[schema_index]["actions"].keys())
                labels.append(actions.index(path)); logits.append(outputs["action_scores"][schema_index][row, index])
        losses["action"] = F.cross_entropy(torch.stack(logits), torch.tensor(labels, dtype=torch.long, device=device)) if logits else zero
    if "parameters" in enabled:
        values = _parameter_loss(outputs, batch, schemas)
        losses["parameters"] = torch.stack(values).mean() if values else zero
    if "confidence" in enabled and "confidence_logits" in outputs:
        values, mask = _target_confidence(batch, device)
        losses["confidence"] = ((outputs["confidence_logits"].sigmoid() - values).square() * mask).sum() / mask.sum().clamp_min(1.0)
    if "ood" in enabled and "ood_logit" in outputs:
        losses["ood"] = F.binary_cross_entropy_with_logits(outputs["ood_logit"], _target_ood(batch, device))
    if "bridge_alignment" in enabled:
        if "bridge_states" not in outputs:
            raise DatasetContractError("bridge_alignment requires bridge-only output")
        speech = F.normalize(outputs["speech_states"], dim=-1)
        mask = outputs["speech_mask"].to(speech.dtype).unsqueeze(-1)
        speech_summary = F.normalize(((speech * mask).sum(1) / mask.sum(1).clamp_min(1.0)).detach(), dim=-1)
        bridge_summary = F.normalize(outputs["bridge_states"].mean(1), dim=-1)
        losses["bridge_alignment"] = 1.0 - F.cosine_similarity(bridge_summary, speech_summary, dim=-1).mean()
    if "lexical" in enabled:
        if int(batch.get("ctc_invalid_count", 0)) > 0:
            raise DatasetContractError("ctc_invalid_ratio is non-zero")
        logits_key = "bridge_lexical_logits" if config["training"].get("lexical_source") == "bridge" else "lexical_ctc_logits"
        if logits_key not in outputs or "lexical_labels" not in batch:
            losses["lexical"] = zero
        else:
            log_probs = outputs[logits_key].log_softmax(-1).transpose(0, 1)
            input_lengths = (outputs["semantic_mask"] if logits_key == "bridge_lexical_logits" else outputs["speech_mask"]).sum(1).long()
            target_lengths = batch["lexical_target_lengths"].to(device)
            labels = batch["lexical_labels"].to(device)
            required = []
            offset = 0
            for length in target_lengths.tolist():
                target = labels[offset:offset + int(length)]
                required.append(int(length) + int((target[1:] == target[:-1]).sum()) if length else 0)
                offset += int(length)
            required = torch.tensor(required, dtype=torch.long, device=device)
            if (required > input_lengths).any():
                raise DatasetContractError(f"ctc_invalid_ratio={float((required > input_lengths).float().mean()):.6f}")
            losses["lexical"] = F.ctc_loss(log_probs, labels, input_lengths, target_lengths, blank=int(config["lexical_branch"]["blank_id"]), zero_infinity=False)

    if not losses:
        raise ValueError("no enabled training losses")
    weights = config["training"]["loss_weights"]
    total = sum(loss * float(weights.get(name, 1.0)) for name, loss in losses.items())
    return total, losses


def get_trainable_parameter_names(model: VoiceNativeSLU) -> list[str]:
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def configure_stage(model: VoiceNativeSLU, stage: TrainingStage | str) -> None:
    selected = TrainingStage(stage)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for module_name in _STAGE_MODULES[selected]:
        module = getattr(model, module_name, None)
        if module is None:
            continue
        for parameter in module.parameters():
            parameter.requires_grad = True


def build_stage_optimizer(model: VoiceNativeSLU, config: Mapping[str, Any], stage: TrainingStage | str) -> torch.optim.Optimizer:
    selected = TrainingStage(stage)
    configure_stage(model, selected)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError(f"stage {selected.value} has no trainable parameters")
    return torch.optim.AdamW(parameters, lr=float(config["training"]["learning_rate"]), weight_decay=float(config["training"]["weight_decay"]))


def train_epoch(
    model: VoiceNativeSLU,
    batches: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    epoch: int | None = None,
    stage: TrainingStage | str = TrainingStage.JOINT,
) -> float:
    selected = TrainingStage(stage)
    configure_stage(model, selected)
    model.train()
    for module in model.children():
        module.train(any(parameter.requires_grad for parameter in module.parameters()))
    enabled = set(config["training"]["stages"][selected.value]["enabled_losses"])
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    use_amp = device.type == "cuda" and config["training"]["precision"] == "fp16_mixed"
    scaler = scaler or torch.amp.GradScaler(device.type, enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)
    total = count = pending = 0

    def step(micro_batches: int) -> None:
        scaler.unscale_(optimizer)
        if micro_batches != accumulation:
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(accumulation / micro_batches)
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["training"]["gradient_clip_norm"]))
        scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)

    for batch in tqdm(batches, desc=f"{selected.value} {epoch}" if epoch is not None else selected.value, unit="batch"):
        tensors = _move_batch(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(tensors["waveform"], tensors.get("audio_mask"), bridge_only=selected == TrainingStage.BRIDGE)
            stage_config = dict(config); stage_config["training"] = dict(config["training"])
            stage_config["training"]["lexical_source"] = config["training"]["stages"][selected.value].get("lexical_source", "speech")
            loss, _ = compute_loss(outputs, tensors, stage_config, enabled)
        scaler.scale(loss / accumulation).backward()
        total += float(loss.detach()); count += 1; pending += 1
        if pending == accumulation: step(pending); pending = 0
    if pending: step(pending)
    if not count: raise ValueError("no training batches")
    return total / count


def _parameter_exact_match(outputs: Mapping[str, Any], target: Mapping[str, Any], schemas: Sequence[Mapping[str, Any]], output_row: int = 0) -> tuple[int, int, bool, int, int]:
    """Evaluate all parameter slots, including absent optional slots."""

    correct = total = presence_correct = presence_total = 0
    frame_correct = True
    schema_map = _schema_map(schemas)
    for operation_index, operation in enumerate(target.get("operations", [])):
        domain = str(operation["domain"]); path = str(operation["action"]).removeprefix(f"{domain}.")
        action_schema = schema_map[domain]["actions"][path]
        gold = operation.get("parameters", {})
        for name, parameter in action_schema.get("parameters", {}).items():
            head = outputs["parameter_outputs"][domain][path][name]
            gold_present = name in gold
            predicted_present = int(head["presence_logits"][output_row, operation_index].argmax()) == 1
            slot_correct = gold_present == predicted_present
            presence_correct += int(slot_correct); presence_total += 1
            if gold_present and predicted_present:
                value = gold[name]; kind = parameter["type"]
                if kind == "ENUM": slot_correct = parameter["values"][int(head["enum_logits"][output_row, operation_index].argmax())] == value
                elif kind == "NUMBER": slot_correct = abs(float(head["number_value"][output_row, operation_index]) - float(value)) <= 1
                elif kind == "BOOLEAN": slot_correct = bool(head["boolean_logits"][output_row, operation_index].argmax()) == value
                elif kind in {"STATE_REFERENCE", "CONTEXT_REFERENCE"}:
                    key = "reference_logits" if kind == "STATE_REFERENCE" else "context_reference_logits"
                    choices = parameter.get("state_reference_paths", parameter.get("context_reference_types", []))
                    slot_correct = key in head and choices[int(head[key][output_row, operation_index].argmax())] == value.get("path", value.get("reference_type"))
                elif kind in {"ENTITY", "FREE_TEXT"}:
                    source = value.get("source") if isinstance(value, Mapping) else None
                    source_id = int(head["source_logits"][output_row, operation_index].argmax())
                    if source == "state_reference":
                        slot_correct = source_id == SpanExtractor.SOURCES.index("STATE_REFERENCE") and "reference_logits" in head and parameter["state_reference_paths"][int(head["reference_logits"][output_row, operation_index].argmax())] == value["path"]
                    elif source == "input_span":
                        frame_count = head["span_start_logits"].size(-1)
                        expected_start = round(float(value["alignment"]["start"]) * (frame_count - 1))
                        expected_end = round(float(value["alignment"]["end"]) * (frame_count - 1))
                        slot_correct = source_id == SpanExtractor.SOURCES.index("INPUT_SPAN") and abs(int(head["span_start_logits"][output_row, operation_index].argmax()) - expected_start) <= 1 and abs(int(head["span_end_logits"][output_row, operation_index].argmax()) - expected_end) <= 1
                    else: slot_correct = False
                else: slot_correct = False
            correct += int(slot_correct); total += 1; frame_correct = frame_correct and slot_correct
    return correct, total, frame_correct, presence_correct, presence_total


def validate_model(model: VoiceNativeSLU, batches: Iterable[Mapping[str, Any]], config: Mapping[str, Any], device: torch.device, capability_schemas: Sequence[Mapping[str, Any]] | None = None) -> dict[str, float]:
    model.eval()
    samples = act_correct = unsafe = operation_correct = operation_total = 0
    goal_correct = goal_total = action_correct = action_total = 0
    parameter_correct = parameter_total = presence_correct = presence_total = 0
    frame_correct = count_correct = 0
    turn_relation_correct = context_required_correct = context_reference_correct = operation_graph_exact = 0
    turn_relation_total = context_reference_total = 0
    turn_understanding_exact_count = 0
    ctc_invalid = ctc_samples = 0
    total_loss = 0.0
    with torch.no_grad():
        for raw in batches:
            ctc_invalid += int(raw.get("ctc_invalid_count", 0)); ctc_samples += int(raw.get("ctc_sample_count", len(raw.get("target", []))))
            batch = _move_batch(raw, device)
            outputs = model(batch["waveform"], batch.get("audio_mask"), capability_schemas=capability_schemas)
            enabled = set(config["training"]["loss_weights"])
            loss, _ = compute_loss(outputs, batch, config, enabled)
            total_loss += float(loss) * len(batch["target"])
            labels = _target_act_ids(batch, config, device); predictions = outputs["act_logits"].argmax(-1)
            samples += len(labels); act_correct += int((predictions == labels).sum()); execute = config["ontology"]["acts"].index("EXECUTE"); unsafe += int(((predictions == execute) & (labels != execute)).sum())
            gold_presence = _target_operation_presence(batch, outputs["operation_presence_logits"].size(1), device).bool(); predicted_presence = outputs["operation_presence_logits"].sigmoid() >= float(config["inference"]["operation_presence_threshold"])
            operation_total += int(gold_presence.numel()); operation_correct += int((gold_presence == predicted_presence).sum())
            schemas = outputs["capability_schemas"]
            for row, target in enumerate(batch["target"]):
                turn_relation = target.get("context", {}).get("relation", "NEW")
                turn_relation_correct += int(TURN_RELATIONS[int(outputs["turn_relation_logits"][row].argmax())] == turn_relation)
                turn_relation_total += 1
                expected_required = bool(target.get("context", {}).get("requires_context", False))
                context_required_correct += int(bool(outputs["context_required_logit"][row].sigmoid() >= 0.5) == expected_required)
                expected_reference = target.get("context", {}).get("reference_type")
                if expected_reference is not None:
                    context_reference_total += 1
                    context_reference_correct += int(config["relations"]["context_reference_types"][int(outputs["context_reference_logits"][row].argmax())] == expected_reference)
                gold_count = len(target.get("operations", [])); pred_count = int(predicted_presence[row].sum()); count_correct += int(gold_count == pred_count)
                exact = predictions[row].item() == labels[row].item() and torch.equal(predicted_presence[row], gold_presence[row]) and gold_count == pred_count
                for index, operation in enumerate(target.get("operations", [])):
                    domain = operation["domain"]; path = operation["action"].removeprefix(f"{domain}."); schema_index = next(i for i,schema in enumerate(schemas) if schema["name"] == domain)
                    predicted_domain = schemas[int(outputs["goal_scores"][row,index].argmax())]["name"]; predicted_path = list(schemas[schema_index]["actions"].keys())[int(outputs["action_scores"][schema_index][row,index].argmax())]
                    goal_total += 1; goal_correct += int(predicted_domain == domain); action_total += 1; action_correct += int(predicted_path == path); exact = exact and predicted_domain == domain and predicted_path == path
                correct, total, params_exact, pcorrect, ptotal = _parameter_exact_match(outputs, target, schemas, output_row=row)
                parameter_correct += correct; parameter_total += total; presence_correct += pcorrect; presence_total += ptotal; exact = exact and params_exact
                gold_relations = {(r["source"], r["target"], r["type"]) for r in target.get("relations", [])}
                predicted_relations = set()
                for source in range(gold_count):
                    for destination in range(gold_count):
                        if source != destination:
                            relation_id = int(outputs["operation_relation_logits"][row, source, destination].argmax())
                            if relation_id:
                                predicted_relations.add((source, destination, OPERATION_RELATIONS[relation_id]))
                graph_exact = gold_relations == predicted_relations
                operation_graph_exact += int(graph_exact)
                turn_relation_match = TURN_RELATIONS[int(outputs["turn_relation_logits"][row].argmax())] == target.get("context", {}).get("relation", "NEW")
                context_required_match = bool(outputs["context_required_logit"][row].sigmoid() >= 0.5) == bool(target.get("context", {}).get("requires_context", False))
                expected_reference = target.get("context", {}).get("reference_type")
                context_reference_match = expected_reference is None or config["relations"]["context_reference_types"][int(outputs["context_reference_logits"][row].argmax())] == expected_reference
                turn_understanding_exact_count += int(exact and graph_exact and turn_relation_match and context_required_match and context_reference_match)
                frame_correct += int(exact and graph_exact and turn_relation_match and context_required_match and context_reference_match)
    if not samples: raise ValueError("no validation batches")
    return {"validation_loss": total_loss / samples, "act_accuracy": act_correct / samples, "goal_retrieval_accuracy": goal_correct / max(goal_total,1), "action_accuracy": action_correct / max(action_total,1), "parameter_presence_accuracy": presence_correct / max(presence_total,1), "parameter_accuracy": parameter_correct / max(parameter_total,1), "operation_presence_accuracy": operation_correct / max(operation_total,1), "operation_count_accuracy": count_correct / samples, "turn_relation_accuracy": turn_relation_correct / max(turn_relation_total,1), "context_required_accuracy": context_required_correct / samples, "context_reference_accuracy": context_reference_correct / max(context_reference_total,1), "operation_graph_exact_accuracy": operation_graph_exact / samples, "turn_understanding_exact_accuracy": turn_understanding_exact_count / samples, "semantic_frame_exact_accuracy": frame_correct / samples, "unsafe_false_execute_rate": unsafe / samples, "ctc_invalid_ratio": ctc_invalid / max(ctc_samples,1)}


def assemble_frame(prediction: Mapping[str, Any], config: Mapping[str, Any], *, request_id: str | None = None, model_version: str | None = None, capability_schemas: Sequence[Mapping[str, Any]] | None = None) -> dict[str, Any]:
    from .dataset import validate_target
    validate_config(config)
    frame = copy.deepcopy(dict(prediction)); confidence = frame.get("confidence")
    if not isinstance(confidence, Mapping): raise DatasetContractError("confidence is required")
    if any(not isinstance(value,(int,float)) or not 0 <= float(value) <= 1 for value in confidence.values()): raise DatasetContractError("confidence values must be in [0,1]")
    frame_context = frame.get("context") or {"relation": "NEW", "requires_context": False, "reference_type": None}
    validate_target({"act": frame.get("act"), "context": frame_context, "operations": frame.get("operations", []), "relations": frame.get("relations", [])}, config, capability_schemas=capability_schemas)
    if float(confidence.get("ood",1.0)) > float(config["inference"]["max_ood_score"]): raise DatasetContractError("prediction is out of distribution")
    if float(confidence.get("overall",0.0)) < float(config["inference"]["act_min_confidence"]): raise DatasetContractError("prediction confidence is below threshold")
    operations = copy.deepcopy(frame.get("operations", []))
    for operation in operations:
        operation.setdefault("goal", operation.get("domain"))
    return {"model_version": model_version or config["project"]["architecture"], "request_id": request_id, "act": frame["act"], "context": frame.get("context", {}), "operations": operations, "relations": frame.get("relations", []), "confidence": dict(confidence)}


def save_checkpoint(path: str | Path, state: Mapping[str, Any]) -> None:
    destination = Path(path); destination.parent.mkdir(parents=True, exist_ok=True); temporary = destination.with_suffix(destination.suffix + ".tmp")
    try: torch.save(dict(state), temporary); temporary.replace(destination)
    finally: temporary.unlink(missing_ok=True)


def export_package(path: str | Path, model: VoiceNativeSLU, config: Mapping[str, Any]) -> None:
    validate_config(config); destination = Path(path); destination.mkdir(parents=True, exist_ok=True)
    expected = {"model.safetensors", "config.json", "lexical_vocab.json", "requirements.txt"}
    for child in destination.iterdir():
        if child.is_dir(): raise ValueError(f"package contains directory: {child.name}")
        if child.name not in expected: child.unlink()
    save_safetensors(model, str(destination / "model.safetensors"))
    (destination / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (destination / "lexical_vocab.json").write_text(json.dumps(config["lexical_branch"]["vocab"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (destination / "requirements.txt").write_text("torch\nsafetensors\ntqdm\n", encoding="utf-8")


def load_package(path: str | Path, device: torch.device) -> tuple[VoiceNativeSLU, dict[str, Any]]:
    source = Path(path); expected = {"model.safetensors", "config.json", "lexical_vocab.json", "requirements.txt"}; entries = list(source.iterdir())
    if {child.name for child in entries} != expected or not all(child.is_file() for child in entries): raise ValueError("V1 package must contain exactly four regular files")
    config = json.loads((source / "config.json").read_text(encoding="utf-8")); vocabulary = json.loads((source / "lexical_vocab.json").read_text(encoding="utf-8"))
    if vocabulary != config["lexical_branch"]["vocab"]: raise ValueError("lexical vocabulary does not match config")
    validate_config(config); model = build_model(config).to(device); load_safetensors(model, str(source / "model.safetensors"), device=str(device)); model.eval(); return model, config


def generate_response(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("VALLS SLU V1 has no response decoder; use Response Module after Grounded Result")


def evaluate_generation(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("VALLS SLU V1 evaluates TurnUnderstanding, not neural response generation")


MultiTaskTransformer = VoiceNativeSLU
