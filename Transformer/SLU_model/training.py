"""Training, staged optimization, evaluation and package I/O for VALLS SLU V0.

The training code produces semantic execution-frame ingredients only.  It does
not train or generate response text; Response Module runs after a grounded
Harness result.
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

from .config import validate_config
from .dataset import DatasetContractError, validate_target
from .model import VoiceNativeSLU, build_model

IGNORE_INDEX = -100


class TrainingStage(str, Enum):
    """Supported V0 curriculum stages."""

    ACOUSTIC = "ACOUSTIC"
    BRIDGE = "BRIDGE"
    SEMANTIC = "SEMANTIC"
    PARAMETER = "PARAMETER"
    SAFETY = "SAFETY"
    JOINT = "JOINT"


_STAGE_MODULES: dict[TrainingStage, set[str]] = {
    TrainingStage.ACOUSTIC: {"speech_encoder", "lexical_head"},
    TrainingStage.BRIDGE: {"semantic_resampler", "bridge_alignment_head"},
    TrainingStage.SEMANTIC: {"semantic_resampler", "semantic_core", "operation_decoder", "schema_encoder", "schema_retriever", "action_resolver", "act_head"},
    TrainingStage.PARAMETER: {"semantic_resampler", "semantic_core", "operation_decoder", "schema_encoder", "schema_retriever", "action_resolver", "parameter_extractor"},
    TrainingStage.SAFETY: {"semantic_core", "confidence_head", "ood_head"},
    TrainingStage.JOINT: {"speech_encoder", "semantic_resampler", "semantic_core", "operation_decoder", "schema_encoder", "schema_retriever", "action_resolver", "parameter_extractor", "act_head", "confidence_head", "ood_head", "lexical_head"},
}


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, Tensor) else value for key, value in batch.items()}


def _cross_entropy(logits: Tensor, labels: Tensor) -> Tensor | None:
    active = labels != IGNORE_INDEX
    return F.cross_entropy(logits[active], labels[active]) if active.any() else None


def _runtime_schemas(outputs: Mapping[str, Any]) -> list[dict[str, Any]]:
    schemas = outputs.get("capability_schemas")
    if not isinstance(schemas, list) or not schemas:
        raise DatasetContractError("model output has no runtime capability schemas")
    return schemas


def _target_act_ids(batch: Mapping[str, Any], config: Mapping[str, Any], device: torch.device) -> Tensor:
    acts = config["ontology"]["acts"]
    return torch.tensor([acts.index(target["act"]) for target in batch["target"]], dtype=torch.long, device=device)


def _target_operation_presence(batch: Mapping[str, Any], max_operations: int, device: torch.device) -> Tensor:
    return torch.tensor(
        [[index < len(target.get("operations", [])) for index in range(max_operations)] for target in batch["target"]],
        dtype=torch.float32,
        device=device,
    )


def _target_ood(batch: Mapping[str, Any], device: torch.device) -> Tensor:
    # OOD is an explicit binary safety label, never a guessed confidence value.
    return torch.tensor(
        [float(bool(target.get("ood", target.get("act") == "UNSUPPORTED"))) for target in batch["target"]],
        dtype=torch.float32,
        device=device,
    )


def _target_confidence(batch: Mapping[str, Any], device: torch.device) -> tuple[Tensor, Tensor]:
    """Return confidence targets and a mask; missing calibration labels are ignored."""

    values: list[list[float]] = []
    mask: list[list[float]] = []
    for target in batch["target"]:
        confidence = target.get("confidence")
        if not isinstance(confidence, Mapping):
            values.append([0.0, 0.0, 0.0])
            mask.append([0.0, 0.0, 0.0])
            continue
        row = [confidence.get(key) for key in ("act", "goal", "parameters")]
        values.append([float(item) if item is not None else 0.0 for item in row])
        mask.append([1.0 if item is not None else 0.0 for item in row])
    return (
        torch.tensor(values, dtype=torch.float32, device=device),
        torch.tensor(mask, dtype=torch.float32, device=device),
    )


def _schema_index(schemas: Sequence[Mapping[str, Any]], name: str) -> int:
    for index, schema in enumerate(schemas):
        if schema.get("name") == name:
            return index
    raise DatasetContractError(f"target capability is not in the runtime schema set: {name}")


def _parameter_loss(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    schemas: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[Tensor]:
    """Train shared typed extractors from runtime schemas, not goal modules."""

    device = outputs["operation_queries"].device
    max_operations = int(config["operation_decoder"]["max_operations"])
    parameter_losses: list[Tensor] = []
    target_goals = {
        str(operation.get("goal"))
        for target in batch["target"]
        for operation in target.get("operations", [])
    }
    hard_negative_limit = int(config["training"]["loss_weights"].get("parameter_hard_negative_schemas", 4))
    selected_schema_names = sorted(target_goals)
    for schema in schemas:
        if schema["name"] not in target_goals and len(selected_schema_names) >= len(target_goals) + hard_negative_limit:
            continue
        if schema["name"] not in target_goals:
            selected_schema_names.append(str(schema["name"]))
    selected_schema_names = set(selected_schema_names)
    for schema in schemas:
        if str(schema["name"]) not in selected_schema_names:
            continue
        goal = str(schema["name"])
        outputs_for_goal = outputs["parameter_outputs"].get(goal, {})
        for name, parameter in schema.get("parameters", {}).items():
            head = outputs_for_goal.get(name)
            if head is None:
                raise DatasetContractError(f"missing shared parameter output for {goal}.{name}")
            for operation_index in range(max_operations):
                presence_labels: list[float] = []
                active_values: list[Any] = []
                active_rows: list[int] = []
                for row, target in enumerate(batch["target"]):
                    operations = target.get("operations", [])
                    value = None
                    if operation_index < len(operations) and operations[operation_index].get("goal") == goal:
                        value = operations[operation_index].get("parameters", {}).get(name)
                    presence_labels.append(float(value is not None))
                    if value is not None:
                        active_rows.append(row)
                        active_values.append(value)
                negative_weight = float(config["training"]["loss_weights"].get("parameter_presence_negative_weight", 0.25))
                parameter_losses.append(
                    F.cross_entropy(
                        head["presence_logits"][:, operation_index],
                        torch.tensor(presence_labels, dtype=torch.long, device=device),
                        weight=torch.tensor([negative_weight, 1.0], dtype=torch.float32, device=device),
                    )
                )
                if not active_rows:
                    continue
                rows = torch.tensor(active_rows, dtype=torch.long, device=device)
                parameter_type = str(parameter["type"])
                if parameter_type == "ENUM":
                    labels = torch.tensor(
                        [parameter["values"].index(value) for value in active_values],
                        dtype=torch.long,
                        device=device,
                    )
                    parameter_losses.append(F.cross_entropy(head["enum_logits"][rows, operation_index], labels))
                elif parameter_type == "NUMBER":
                    labels = torch.tensor([float(value) for value in active_values], dtype=torch.float32, device=device)
                    parameter_losses.append(F.smooth_l1_loss(head["number_value"][rows, operation_index], labels))
                elif parameter_type == "BOOLEAN":
                    labels = torch.tensor([int(value) for value in active_values], dtype=torch.long, device=device)
                    parameter_losses.append(F.cross_entropy(head["boolean_logits"][rows, operation_index], labels))
                elif parameter_type == "STATE_REFERENCE":
                    labels = torch.tensor(
                        [parameter["state_reference_paths"].index(value["path"]) for value in active_values],
                        dtype=torch.long,
                        device=device,
                    )
                    parameter_losses.append(F.cross_entropy(head["reference_logits"][rows, operation_index], labels))
                elif parameter_type in {"ENTITY", "FREE_TEXT"}:
                    start_ratios: list[float] = []
                    end_ratios: list[float] = []
                    for value in active_values:
                        if not isinstance(value, dict) or value.get("source") != "input_span":
                            raise DatasetContractError(f"{goal}.{name} requires an input_span target")
                        alignment = value.get("alignment")
                        if not isinstance(alignment, Mapping):
                            raise DatasetContractError(f"{goal}.{name} requires frame alignment")
                        start_ratios.append(float(alignment["start"]))
                        end_ratios.append(float(alignment["end"]))
                    start_logits = head["span_start_logits"][rows, operation_index]
                    end_logits = head["span_end_logits"][rows, operation_index]
                    frame_count = start_logits.size(-1)
                    start_labels = torch.tensor(start_ratios, device=device).mul(frame_count - 1).round().long().clamp(0, frame_count - 1)
                    end_labels = torch.tensor(end_ratios, device=device).mul(frame_count - 1).round().long().clamp(0, frame_count - 1)
                    parameter_losses.extend([F.cross_entropy(start_logits, start_labels), F.cross_entropy(end_logits, end_labels)])
    return parameter_losses


def _ctc_required_lengths(labels: Tensor, target_lengths: Tensor) -> Tensor:
    """Account for repeated adjacent labels that need an intervening blank."""

    required: list[int] = []
    offset = 0
    for length in target_lengths.tolist():
        length = int(length)
        target = labels[offset : offset + length]
        repeats = int((target[1:] == target[:-1]).sum()) if length > 1 else 0
        required.append(length + repeats)
        offset += length
    return torch.tensor(required, dtype=torch.long, device=target_lengths.device)


def compute_loss(
    outputs: Mapping[str, Any],
    batch: Mapping[str, Any],
    config: Mapping[str, Any],
    enabled_losses: set[str] | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute staged V0 semantic losses."""

    enabled = set(config["training"]["loss_weights"]) if enabled_losses is None else set(enabled_losses)
    device = outputs["act_logits"].device
    schemas = _runtime_schemas(outputs)
    losses: dict[str, Tensor] = {}
    zero = outputs["act_logits"].sum() * 0.0

    if "act" in enabled:
        losses["act"] = F.cross_entropy(outputs["act_logits"], _target_act_ids(batch, config, device))
    if "operation_presence" in enabled:
        labels = _target_operation_presence(batch, int(config["operation_decoder"]["max_operations"]), device)
        losses["operation_presence"] = F.binary_cross_entropy_with_logits(outputs["operation_presence_logits"], labels)
    if "goal_retrieval" in enabled:
        labels: list[int] = []
        logits: list[Tensor] = []
        for row, target in enumerate(batch["target"]):
            for operation_index, operation in enumerate(target.get("operations", [])):
                labels.append(_schema_index(schemas, str(operation["goal"])))
                logits.append(outputs["goal_scores"][row, operation_index])
        losses["goal_retrieval"] = F.cross_entropy(torch.stack(logits), torch.tensor(labels, dtype=torch.long, device=device)) if logits else zero
    if "action" in enabled:
        labels = []
        logits = []
        for row, target in enumerate(batch["target"]):
            for operation_index, operation in enumerate(target.get("operations", [])):
                schema_index = _schema_index(schemas, str(operation["goal"]))
                schema = schemas[schema_index]
                actions = list(schema.get("actions", []))
                if operation.get("action") not in actions:
                    raise DatasetContractError(f"invalid action for {schema['name']}: {operation.get('action')}")
                labels.append(actions.index(operation["action"]))
                logits.append(outputs["action_scores"][schema_index][row, operation_index])
        losses["action"] = F.cross_entropy(torch.stack(logits), torch.tensor(labels, dtype=torch.long, device=device)) if logits else zero
    if "parameters" in enabled:
        values = _parameter_loss(outputs, batch, schemas, config)
        losses["parameters"] = torch.stack(values).mean() if values else zero
    if "confidence" in enabled:
        values, mask = _target_confidence(batch, device)
        error = (outputs["confidence_logits"].sigmoid() - values).square()
        losses["confidence"] = (error * mask).sum() / mask.sum().clamp_min(1.0)
    if "ood" in enabled:
        losses["ood"] = F.binary_cross_entropy_with_logits(outputs["ood_logit"], _target_ood(batch, device))
    if "bridge_alignment" in enabled:
        speech = F.normalize(outputs["speech_states"], dim=-1)
        speech_mask = outputs["speech_mask"].to(speech.dtype).unsqueeze(-1)
        speech_summary = (speech * speech_mask).sum(1) / speech_mask.sum(1).clamp_min(1.0)
        bridge = F.normalize(outputs["bridge_states"], dim=-1)
        speech_summary = speech_summary.unsqueeze(1).expand_as(bridge)
        losses["bridge_alignment"] = 1.0 - (bridge * speech_summary).sum(-1).mean()
    if "lexical" in enabled:
        if int(batch.get("ctc_invalid_count", 0)) > 0:
            raise DatasetContractError(
                f"ctc_invalid_ratio={int(batch['ctc_invalid_count']) / max(int(batch.get('ctc_sample_count', 1)), 1):.6f}"
            )
        if "lexical_ctc_logits" not in outputs or "lexical_labels" not in batch:
            losses["lexical"] = zero
        else:
            lexical_key = "bridge_lexical_logits" if config["training"].get("lexical_source") == "bridge" else "lexical_ctc_logits"
            log_probs = outputs[lexical_key].log_softmax(-1).transpose(0, 1)
            input_lengths = outputs["semantic_mask"].sum(1).long() if lexical_key == "bridge_lexical_logits" else outputs["speech_mask"].sum(1).long()
            lexical_labels = batch["lexical_labels"].to(device)
            target_lengths = batch["lexical_target_lengths"].to(device)
            required_lengths = _ctc_required_lengths(lexical_labels, target_lengths)
            invalid = required_lengths > input_lengths
            if invalid.any():
                invalid_ratio = float(invalid.float().mean())
                raise DatasetContractError(
                    f"ctc_invalid_ratio={invalid_ratio:.6f}; target requires more timesteps than the selected lexical branch"
                )
            losses["lexical"] = F.ctc_loss(
                log_probs,
                lexical_labels,
                input_lengths.to(device),
                target_lengths,
                blank=int(config["lexical_branch"]["blank_id"]),
                zero_infinity=False,
            )

    if not losses:
        raise ValueError("no enabled training losses")
    weights = config["training"]["loss_weights"]
    total = sum(loss * float(weights.get(name, 1.0)) for name, loss in losses.items())
    return total, losses


def configure_stage(model: VoiceNativeSLU, stage: TrainingStage | str) -> None:
    """Freeze all modules not owned by a curriculum stage."""

    selected = TrainingStage(stage)
    trainable = _STAGE_MODULES[selected]
    for name, module in model.named_children():
        enabled = name in trainable
        for parameter in module.parameters():
            parameter.requires_grad = enabled


def build_stage_optimizer(
    model: VoiceNativeSLU,
    config: Mapping[str, Any],
    stage: TrainingStage | str,
) -> torch.optim.Optimizer:
    selected = TrainingStage(stage)
    multiplier = float(config["training"]["stages"][selected.value]["learning_rate_multiplier"])
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError(f"training stage {selected.value} has no trainable parameters")
    return torch.optim.AdamW(
        parameters,
        lr=float(config["training"]["learning_rate"]) * multiplier,
        weight_decay=float(config["training"]["weight_decay"]),
    )


def _set_stage_modes(model: VoiceNativeSLU) -> None:
    """Keep frozen modules deterministic while trainable modules stay in train mode."""

    for module in model.children():
        if any(parameter.requires_grad for parameter in module.parameters()):
            module.train()
        else:
            module.eval()


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
    """Train one epoch with the selected curriculum-stage loss set."""

    model.train()
    _set_stage_modes(model)
    selected = TrainingStage(stage)
    enabled = set(config["training"]["stages"][selected.value]["enabled_losses"])
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    clip_norm = float(config["training"]["gradient_clip_norm"])
    use_amp = device.type == "cuda" and config["training"]["precision"] == "fp16_mixed"
    scaler = scaler or torch.amp.GradScaler(device.type, enabled=use_amp)
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    count = 0
    pending = 0

    def optimizer_step(micro_batches: int) -> None:
        scaler.unscale_(optimizer)
        if micro_batches != accumulation:
            correction = accumulation / micro_batches
            for parameter in model.parameters():
                if parameter.grad is not None:
                    parameter.grad.mul_(correction)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    progress = tqdm(batches, desc=f"{selected.value} {epoch}" if epoch is not None else selected.value, unit="batch")
    for raw_batch in progress:
        batch = _move_batch(raw_batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(batch["waveform"], batch.get("audio_mask"))
            stage_config = dict(config)
            stage_config["training"] = dict(config["training"])
            stage_config["training"]["lexical_source"] = config["training"]["stages"][selected.value].get("lexical_source", "speech")
            loss, _ = compute_loss(outputs, batch, stage_config, enabled)
        scaler.scale(loss / accumulation).backward()
        total += float(loss.detach())
        count += 1
        pending += 1
        progress.set_postfix(loss=f"{total / count:.4f}", refresh=False)
        if pending == accumulation:
            optimizer_step(pending)
            pending = 0
    if pending:
        optimizer_step(pending)
    if not count:
        raise ValueError("no training batches")
    return total / count


def _parameter_exact_match(
    outputs: Mapping[str, Any], batch: Mapping[str, Any], schemas: Sequence[Mapping[str, Any]], config: Mapping[str, Any], row_index: int = 0
) -> tuple[int, int]:
    correct = total = 0
    for row, target in enumerate(batch["target"]):
        for operation_index, operation in enumerate(target.get("operations", [])):
            goal = str(operation["goal"])
            schema_index = _schema_index(schemas, goal)
            for name, value in operation.get("parameters", {}).items():
                head = outputs["parameter_outputs"][goal][name]
                parameter = schemas[schema_index]["parameters"][name]
                parameter_type = parameter["type"]
                if parameter_type == "ENUM":
                    predicted = parameter["values"][int(head["enum_logits"][row_index, operation_index].argmax())]
                    is_correct = predicted == value
                elif parameter_type == "NUMBER":
                    predicted = float(head["number_value"][row_index, operation_index])
                    is_correct = abs(predicted - float(value)) <= 1.0
                elif parameter_type == "BOOLEAN":
                    is_correct = bool(head["boolean_logits"][row_index, operation_index].argmax()) == value
                elif parameter_type == "STATE_REFERENCE":
                    reference_id = int(head["reference_logits"][row_index, operation_index].argmax())
                    is_correct = parameter["state_reference_paths"][reference_id] == value["path"]
                else:
                    start = int(head["span_start_logits"][row_index, operation_index].argmax())
                    end = int(head["span_end_logits"][row_index, operation_index].argmax())
                    alignment = value.get("alignment", {}) if isinstance(value, Mapping) else {}
                    frame_count = head["span_start_logits"].size(-1)
                    expected_start = round(float(alignment.get("start", -1)) * (frame_count - 1))
                    expected_end = round(float(alignment.get("end", -1)) * (frame_count - 1))
                    is_correct = abs(start - expected_start) <= 1 and abs(end - expected_end) <= 1
                total += 1
                correct += int(is_correct)
    return correct, total


def validate_model(
    model: VoiceNativeSLU,
    batches: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    device: torch.device,
    capability_schemas: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, float]:
    """Evaluate goal/action/parameter/frame and safety metrics."""

    model.eval()
    samples = act_correct = unsafe = operation_total = operation_correct = 0
    goal_total = goal_correct = action_total = action_correct = frame_exact = 0
    parameter_correct = parameter_total = 0
    total_loss = 0.0
    with torch.no_grad():
        for raw_batch in batches:
            batch = _move_batch(raw_batch, device)
            outputs = model(batch["waveform"], batch.get("audio_mask"), capability_schemas=capability_schemas)
            loss, _ = compute_loss(outputs, batch, config, set(config["training"]["loss_weights"]))
            total_loss += float(loss) * len(batch["target"])
            labels = _target_act_ids(batch, config, device)
            predictions = outputs["act_logits"].argmax(-1)
            samples += len(labels)
            act_correct += int((predictions == labels).sum())
            execute_id = config["ontology"]["acts"].index("EXECUTE")
            unsafe += int(((predictions == execute_id) & (labels != execute_id)).sum())
            gold_presence = _target_operation_presence(batch, int(config["operation_decoder"]["max_operations"]), device).bool()
            predicted_presence = outputs["operation_presence_logits"].sigmoid() >= float(config["inference"]["operation_presence_threshold"])
            operation_total += int(gold_presence.numel())
            operation_correct += int((predicted_presence == gold_presence).sum())
            schemas = _runtime_schemas(outputs)
            # Per-row frame metrics use the runtime schema set and preserve
            # operation ordering; no fixed ontology index is assumed.
            # full frame validation is performed by assemble_frame below.
            for row, target in enumerate(batch["target"]):
                all_correct = predictions[row].item() == labels[row].item()
                if not target.get("operations"):
                    frame_exact += int(all_correct and not bool(predicted_presence[row].any().item()))
                    continue
                for operation_index, operation in enumerate(target["operations"]):
                    schema_index = _schema_index(schemas, operation["goal"])
                    predicted_goal = schemas[int(outputs["goal_scores"][row, operation_index].argmax())]["name"]
                    predicted_action = schemas[schema_index]["actions"][int(outputs["action_scores"][schema_index][row, operation_index].argmax())]
                    goal_correct += int(predicted_goal == operation["goal"])
                    action_correct += int(predicted_action == operation["action"])
                    goal_total += 1
                    action_total += 1
                    all_correct = all_correct and predicted_goal == operation["goal"] and predicted_action == operation["action"]
                correct, total = _parameter_exact_match(outputs, {"target": [target]}, schemas, config, row_index=row)
                parameter_correct += correct
                parameter_total += total
                frame_exact += int(all_correct and correct == total)
    if not samples:
        raise ValueError("no validation batches")
    return {
        "ctc_invalid_ratio": 0.0,
        "validation_loss": total_loss / samples,
        "act_accuracy": act_correct / samples,
        "goal_retrieval_accuracy": goal_correct / max(goal_total, 1),
        "action_accuracy": action_correct / max(action_total, 1),
        "parameter_accuracy": parameter_correct / max(parameter_total, 1),
        "semantic_frame_exact_accuracy": frame_exact / max(samples, 1),
        "operation_presence_accuracy": operation_correct / max(operation_total, 1),
        "unsafe_false_execute_rate": unsafe / max(samples, 1),
    }


def predict(
    model: VoiceNativeSLU,
    waveform: Tensor,
    config: Mapping[str, Any],
    audio_mask: Tensor | None = None,
    capability_schemas: Sequence[Mapping[str, Any]] | None = None,
    capability_embeddings: Tensor | None = None,
) -> dict[str, Any]:
    """Produce an untrusted semantic prediction from one waveform."""

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() != 2 or waveform.size(0) != 1:
        raise ValueError("predict accepts one waveform with shape [samples] or [1, samples]")
    if audio_mask is not None and audio_mask.dim() == 1:
        audio_mask = audio_mask.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        outputs = model(waveform, audio_mask, capability_schemas, capability_embeddings)
    acts = config["ontology"]["acts"]
    act_probabilities = outputs["act_logits"].softmax(-1)[0]
    act_id = int(act_probabilities.argmax())
    act = acts[act_id]
    act_confidence = float(act_probabilities[act_id])
    ood_score = float(outputs["ood_logit"].sigmoid()[0])
    base_confidence = {"act": act_confidence, "goal": 0.0, "parameters": 0.0, "ood": ood_score, "overall": 0.0}
    if act_confidence < float(config["inference"]["act_min_confidence"]) or ood_score > float(config["inference"]["max_ood_score"]):
        return {"act": "UNSUPPORTED", "operations": [], "confidence": base_confidence}

    schemas = outputs["capability_schemas"]
    schema_names = outputs["schema_names"]
    lexical_resolver = model.lexical_span_resolver
    presence = outputs["operation_presence_logits"].sigmoid()[0]
    confidence_values = outputs["confidence_logits"].sigmoid()[0]
    operations: list[dict[str, Any]] = []
    for operation_index in range(outputs["operation_queries"].size(1)):
        if float(presence[operation_index]) < float(config["inference"]["operation_presence_threshold"]):
            continue
        goal_scores = outputs["goal_scores"][0, operation_index].softmax(-1)
        goal_id = int(goal_scores.argmax())
        goal_confidence = float(goal_scores[goal_id])
        if goal_confidence < float(config["inference"]["goal_min_confidence"]):
            return {"act": "ASK_CLARIFICATION", "operations": [], "confidence": {**base_confidence, "goal": goal_confidence}}
        goal = schema_names[goal_id]
        schema = schemas[goal_id]
        action_scores = outputs["action_scores"][goal_id][0, operation_index].softmax(-1)
        action_id = int(action_scores.argmax())
        action_confidence = float(action_scores[action_id])
        if action_confidence < float(config["inference"]["action_min_confidence"]):
            return {"act": "ASK_CLARIFICATION", "operations": [], "confidence": {**base_confidence, "goal": goal_confidence}}
        action = schema["actions"][action_id]
        parameters: dict[str, Any] = {}
        parameter_confidence = 1.0
        for name, parameter_schema in schema.get("parameters", {}).items():
            head = outputs["parameter_outputs"][goal][name]
            present_probability = float(head["presence_logits"][0, operation_index].softmax(-1)[1])
            if present_probability < float(config["inference"]["parameter_min_confidence"]):
                if parameter_schema.get("required"):
                    return {"act": "ASK_CLARIFICATION", "operations": [], "confidence": {**base_confidence, "goal": goal_confidence, "parameters": present_probability}}
                continue
            parameter_confidence = min(parameter_confidence, present_probability)
            parameter_type = parameter_schema["type"]
            if parameter_type == "ENUM":
                parameters[name] = parameter_schema["values"][int(head["enum_logits"][0, operation_index].argmax())]
            elif parameter_type == "NUMBER":
                value = float(head["number_value"][0, operation_index])
                value = max(float(parameter_schema.get("minimum", value)), min(float(parameter_schema.get("maximum", value)), value))
                parameters[name] = int(round(value)) if value.is_integer() else value
            elif parameter_type == "BOOLEAN":
                parameters[name] = bool(head["boolean_logits"][0, operation_index].argmax())
            elif parameter_type == "STATE_REFERENCE":
                reference_id = int(head["reference_logits"][0, operation_index].argmax())
                parameters[name] = {"source": "state_reference", "path": parameter_schema["state_reference_paths"][reference_id]}
            else:
                start = int(head["span_start_logits"][0, operation_index].argmax())
                end = max(start, int(head["span_end_logits"][0, operation_index].argmax()))
                frame_count = head["span_start_logits"].size(-1)
                lexical_logits = outputs.get("lexical_ctc_logits")
                if lexical_logits is None:
                    return {"act": "ASK_CLARIFICATION", "operations": [], "confidence": {**base_confidence, "goal": goal_confidence, "parameters": parameter_confidence}}
                lexical_value = lexical_resolver.resolve(lexical_logits[0], start, end)
                if not lexical_value:
                    return {"act": "ASK_CLARIFICATION", "operations": [], "confidence": {**base_confidence, "goal": goal_confidence, "parameters": parameter_confidence}}
                parameters[name] = {
                    "source": "input_span",
                    "value": lexical_value,
                    "start_frame": start,
                    "end_frame": end,
                    "start_ratio": start / max(frame_count - 1, 1),
                    "end_ratio": (end + 1) / max(frame_count, 1),
                }
        operations.append({"order": operation_index + 1, "goal": goal, "action": action, "parameters": parameters, "confidence": min(goal_confidence, action_confidence)})

    if act == "EXECUTE" and not operations:
        return {"act": "ASK_CLARIFICATION", "operations": [], "confidence": {**base_confidence}}
    overall = min(act_confidence, float(confidence_values.min()), 1.0 - ood_score)
    return {
        "act": act,
        "operations": operations,
        "confidence": {
            "act": act_confidence,
            "goal": min((operation["confidence"] for operation in operations), default=1.0),
            "parameters": float(confidence_values[2]),
            "ood": ood_score,
            "overall": overall,
        },
    }


def assemble_frame(
    prediction: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    request_id: str | None = None,
    model_version: str | None = None,
    capability_schemas: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate and serialize the V0 Semantic Execution Frame."""

    validate_config(config)
    frame = copy.deepcopy(dict(prediction))
    confidence = frame.get("confidence")
    if not isinstance(confidence, Mapping):
        raise DatasetContractError("confidence is required")
    for key, value in confidence.items():
        if not isinstance(value, (int, float)) or not 0 <= float(value) <= 1:
            raise DatasetContractError(f"confidence.{key} must be in [0,1]")
    validate_target(
        {"act": frame.get("act"), "operations": frame.get("operations", [])},
        config,
        capability_schemas=capability_schemas,
    )
    if float(confidence.get("ood", 1.0)) > float(config["inference"]["max_ood_score"]):
        raise DatasetContractError("prediction is out of distribution")
    if float(confidence.get("overall", 0.0)) < float(config["inference"]["act_min_confidence"]):
        raise DatasetContractError("prediction confidence is below execution threshold")
    return {
        "model_version": model_version or config["project"]["architecture"],
        "request_id": request_id,
        "act": frame["act"],
        "operations": frame.get("operations", []),
        "confidence": dict(confidence),
    }


def save_checkpoint(path: str | Path, state: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        torch.save(dict(state), temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def export_package(path: str | Path, model: VoiceNativeSLU, config: Mapping[str, Any]) -> None:
    """Export model, validated JSON config and the lexical vocabulary artifact."""

    validate_config(config)
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    expected = {"model.safetensors", "config.json", "lexical_vocab.json", "requirements.txt"}
    for child in destination.iterdir():
        if child.is_dir():
            raise ValueError(f"package destination contains directory: {child.name}")
        if child.name not in expected:
            child.unlink()
    save_safetensors(model, str(destination / "model.safetensors"))
    (destination / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (destination / "lexical_vocab.json").write_text(json.dumps(config["lexical_branch"]["vocab"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (destination / "requirements.txt").write_text("torch\nsafetensors\ntqdm\n", encoding="utf-8")


def load_package(path: str | Path, device: torch.device) -> tuple[VoiceNativeSLU, dict[str, Any]]:
    source = Path(path)
    expected = {"model.safetensors", "config.json", "lexical_vocab.json", "requirements.txt"}
    entries = list(source.iterdir())
    if {child.name for child in entries} != expected or not all(child.is_file() for child in entries):
        raise ValueError("V0 package must contain exactly four regular files")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    lexical_vocab = json.loads((source / "lexical_vocab.json").read_text(encoding="utf-8"))
    if lexical_vocab != config["lexical_branch"]["vocab"]:
        raise ValueError("lexical vocabulary artifact does not match config")
    validate_config(config)
    model = build_model(config).to(device)
    load_safetensors(model, str(source / "model.safetensors"), device=str(device))
    model.eval()
    return model, config


def generate_response(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("VALLS SLU V0 has no neural response decoder; use the template Response Module after Grounded Result")


def evaluate_generation(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("VALLS SLU V0 evaluates semantic frames, not neural response generation")


MultiTaskTransformer = VoiceNativeSLU
