"""Training, evaluation, frame validation and package I/O for VALLS SLU V0.

No response-text decoder is trained here.  Response generation belongs to the
post-execution template Response Module, after Harness returns Grounded Result.
"""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

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


def _move_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor fields while preserving untrusted target metadata."""

    return {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(1) / weights.sum(1).clamp_min(1.0)


def _cross_entropy(logits: Tensor, labels: Tensor) -> Tensor | None:
    active = labels != IGNORE_INDEX
    if not active.any():
        return None
    return F.cross_entropy(logits[active], labels[active])


def _target_act_ids(batch: Mapping[str, Any], config: Mapping[str, Any], device: torch.device) -> Tensor:
    acts = config["ontology"]["acts"]
    values = []
    for target in batch["target"]:
        values.append(acts.index(target["act"]))
    return torch.tensor(values, dtype=torch.long, device=device)


def _target_operation_presence(
    batch: Mapping[str, Any], max_operations: int, device: torch.device
) -> Tensor:
    return torch.tensor(
        [
            [index < len(target.get("operations", [])) for index in range(max_operations)]
            for target in batch["target"]
        ],
        dtype=torch.float32,
        device=device,
    )


def _target_goal_indices(
    batch: Mapping[str, Any], config: Mapping[str, Any], device: torch.device
) -> Tensor:
    goals = config["ontology"]["schema_order"]
    maximum = int(config["operation_decoder"]["max_operations"])
    values = []
    for target in batch["target"]:
        operations = target.get("operations", [])
        values.append(
            [goals.index(operation["goal"]) if index < len(operations) else IGNORE_INDEX for index in range(maximum)]
        )
    return torch.tensor(values, dtype=torch.long, device=device)


def _target_ood(batch: Mapping[str, Any], device: torch.device) -> Tensor:
    return torch.tensor(
        [float(target.get("ood_score", 0.0)) for target in batch["target"]],
        dtype=torch.float32,
        device=device,
    )


def compute_loss(
    outputs: Mapping[str, Any], batch: Mapping[str, Any], config: Mapping[str, Any]
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute V0 semantic losses; response generation is intentionally absent."""

    act_labels = _target_act_ids(batch, config, outputs["act_logits"].device)
    losses: dict[str, Tensor] = {
        "act": F.cross_entropy(outputs["act_logits"], act_labels),
    }

    presence_labels = _target_operation_presence(
        batch,
        int(config["operation_decoder"]["max_operations"]),
        outputs["operation_presence_logits"].device,
    )
    losses["operation_presence"] = F.binary_cross_entropy_with_logits(
        outputs["operation_presence_logits"], presence_labels
    )

    goal_labels = _target_goal_indices(batch, config, outputs["goal_scores"].device)
    goal_loss = _cross_entropy(
        outputs["goal_scores"].reshape(-1, outputs["goal_scores"].size(-1)),
        goal_labels.reshape(-1),
    )
    losses["goal_retrieval"] = goal_loss or outputs["act_logits"].sum() * 0.0

    parameter_losses: list[Tensor] = []
    # Parameter targets are schema-shaped dictionaries.  The target encoder is
    # deliberately explicit so invalid/missing data fails closed rather than
    # being silently converted into a guessed label.
    for operation_index in range(int(config["operation_decoder"]["max_operations"])):
        for goal in config["ontology"]["schema_order"]:
            for name, parameter in config["ontology"]["capabilities"][goal]["parameters"].items():
                head = outputs["parameter_outputs"][goal][name]
                values: list[Any] = []
                active: list[bool] = []
                for target in batch["target"]:
                    operations = target.get("operations", [])
                    if operation_index >= len(operations) or operations[operation_index]["goal"] != goal:
                        values.append(None)
                        active.append(False)
                        continue
                    operation = operations[operation_index]
                    if name not in operation.get("parameters", {}):
                        values.append(None)
                        active.append(False)
                        continue
                    values.append(operation["parameters"][name])
                    active.append(True)
                if not any(active):
                    continue
                indices = torch.tensor([i for i, enabled in enumerate(active) if enabled], device=outputs["act_logits"].device)
                typed_values = [value for value, enabled in zip(values, active) if enabled]
                parameter_type = parameter["type"]
                if parameter_type == "ENUM":
                    labels = torch.tensor(
                        [parameter["values"].index(value) for value in typed_values],
                        dtype=torch.long,
                        device=indices.device,
                    )
                    parameter_losses.append(F.cross_entropy(head["enum_logits"][indices, operation_index], labels))
                elif parameter_type == "NUMBER":
                    labels = torch.tensor([float(value) for value in typed_values], device=indices.device)
                    parameter_losses.append(F.smooth_l1_loss(head["number_value"][indices, operation_index], labels))
                elif parameter_type == "BOOLEAN":
                    labels = torch.tensor([int(value) for value in typed_values], dtype=torch.long, device=indices.device)
                    parameter_losses.append(F.cross_entropy(head["boolean_logits"][indices, operation_index], labels))
                elif parameter_type in {"ENTITY", "FREE_TEXT"}:
                    # Span labels require token/frame alignment supplied by a
                    # future dataset.  The head remains available for training
                    # but is not assigned a fabricated loss here.
                    continue
                elif parameter_type == "STATE_REFERENCE":
                    labels = torch.tensor(
                        [parameter["state_reference_paths"].index(value["path"]) for value in typed_values],
                        dtype=torch.long,
                        device=indices.device,
                    )
                    parameter_losses.append(F.cross_entropy(head["reference_logits"][indices, operation_index], labels))
    losses["parameters"] = (
        torch.stack(parameter_losses).mean()
        if parameter_losses
        else outputs["act_logits"].sum() * 0.0
    )

    confidence_target = torch.stack(
        [
            torch.tensor(
                [
                    float(target.get("confidence", {}).get("act", 1.0)),
                    float(target.get("confidence", {}).get("goal", 1.0)),
                    float(target.get("confidence", {}).get("parameters", 1.0)),
                ],
                device=outputs["act_logits"].device,
            )
            for target in batch["target"]
        ]
    )
    losses["confidence"] = F.mse_loss(outputs["confidence_logits"].sigmoid(), confidence_target)
    losses["ood"] = F.binary_cross_entropy_with_logits(
        outputs["ood_logit"], _target_ood(batch, outputs["ood_logit"].device)
    )

    if "lexical_ctc_logits" in outputs and "lexical_labels" in batch:
        lexical = outputs["lexical_ctc_logits"].log_softmax(-1).transpose(0, 1)
        losses["lexical"] = F.ctc_loss(
            lexical,
            batch["lexical_labels"],
            batch["lexical_input_lengths"],
            batch["lexical_target_lengths"],
            blank=int(config["lexical_branch"]["blank_id"]),
            zero_infinity=True,
        )
    else:
        losses["lexical"] = outputs["act_logits"].sum() * 0.0

    weights = config["training"]["loss_weights"]
    total = sum(losses[name] * float(weights.get(name, 0.0)) for name in losses)
    return total, losses


def train_epoch(
    model: VoiceNativeSLU,
    batches: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    config: Mapping[str, Any],
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    epoch: int | None = None,
) -> float:
    """Train one epoch with gradient accumulation and fail-closed batches."""

    model.train()
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

    progress = tqdm(batches, desc=f"Epoch {epoch}" if epoch is not None else "Train", unit="batch")
    for raw_batch in progress:
        batch = _move_batch(raw_batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(batch["waveform"], batch.get("audio_mask"))
            loss, _ = compute_loss(outputs, batch, config)
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


def _goal_predictions(outputs: Mapping[str, Any], config: Mapping[str, Any]) -> Tensor:
    return outputs["goal_scores"].argmax(-1)


def validate_model(
    model: VoiceNativeSLU,
    batches: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    device: torch.device,
) -> dict[str, float]:
    """Evaluate semantic heads and safety metrics, never response text."""

    model.eval()
    samples = 0
    act_correct = 0
    unsafe = 0
    operations = 0
    operation_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for raw_batch in batches:
            batch = _move_batch(raw_batch, device)
            outputs = model(batch["waveform"], batch.get("audio_mask"))
            loss, _ = compute_loss(outputs, batch, config)
            total_loss += float(loss) * len(batch["target"])
            labels = _target_act_ids(batch, config, device)
            predictions = outputs["act_logits"].argmax(-1)
            samples += len(labels)
            act_correct += int((predictions == labels).sum())
            execute_id = config["ontology"]["acts"].index("EXECUTE")
            unsafe += int(((predictions == execute_id) & (labels != execute_id)).sum())
            gold_presence = _target_operation_presence(
                batch, int(config["operation_decoder"]["max_operations"]), device
            ).bool()
            predicted_presence = outputs["operation_presence_logits"].sigmoid() >= float(config["inference"]["operation_presence_threshold"])
            operations += int(gold_presence.numel())
            operation_correct += int((predicted_presence == gold_presence).sum())
    if not samples:
        raise ValueError("no validation batches")
    return {
        "validation_loss": total_loss / samples,
        "act_accuracy": act_correct / samples,
        "operation_presence_accuracy": operation_correct / max(operations, 1),
        "unsafe_false_execute_rate": unsafe / max(samples - (unsafe == 0 and 0), 1),
    }


def predict(
    model: VoiceNativeSLU,
    waveform: Tensor,
    config: Mapping[str, Any],
    audio_mask: Tensor | None = None,
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
        outputs = model(waveform, audio_mask, capability_embeddings)
    acts = config["ontology"]["acts"]
    act_probabilities = outputs["act_logits"].softmax(-1)[0]
    act_id = int(act_probabilities.argmax())
    act_confidence = float(act_probabilities[act_id])
    ood_score = float(outputs["ood_logit"].sigmoid()[0])
    if (
        act_confidence < float(config["inference"]["act_min_confidence"])
        or ood_score > float(config["inference"]["max_ood_score"])
    ):
        return {
            "act": "UNSUPPORTED",
            "operations": [],
            "confidence": {"act": act_confidence, "goal": 0.0, "parameters": 0.0, "ood": ood_score, "overall": 0.0},
        }

    goals = config["ontology"]["schema_order"]
    operation_queries = outputs["operation_queries"]
    presence = outputs["operation_presence_logits"].sigmoid()[0]
    confidence_values = outputs["confidence_logits"].sigmoid()[0]
    operations: list[dict[str, Any]] = []
    for index in range(operation_queries.size(1)):
        if float(presence[index]) < float(config["inference"]["operation_presence_threshold"]):
            continue
        goal_scores = outputs["goal_scores"][0, index].softmax(-1)
        goal_id = int(goal_scores.argmax())
        goal_confidence = float(goal_scores[goal_id])
        if goal_confidence < float(config["inference"]["goal_min_confidence"]):
            return {
                "act": "ASK_CLARIFICATION",
                "operations": [],
                "confidence": {"act": act_confidence, "goal": goal_confidence, "parameters": 0.0, "ood": ood_score, "overall": 0.0},
            }
        goal = goals[goal_id]
        parameters: dict[str, Any] = {}
        parameter_confidence = 1.0
        for name, schema in config["ontology"]["capabilities"][goal]["parameters"].items():
            head = outputs["parameter_outputs"][goal][name]
            present_probability = head["presence_logits"][0, index].softmax(-1)[1]
            if float(present_probability) < float(config["inference"]["parameter_min_confidence"]):
                if schema.get("required"):
                    return {
                        "act": "ASK_CLARIFICATION",
                        "operations": [],
                        "confidence": {"act": act_confidence, "goal": goal_confidence, "parameters": float(present_probability), "ood": ood_score, "overall": 0.0},
                    }
                continue
            parameter_confidence = min(parameter_confidence, float(present_probability))
            parameter_type = schema["type"]
            if parameter_type == "ENUM":
                value_id = int(head["enum_logits"][0, index].argmax())
                parameters[name] = schema["values"][value_id]
            elif parameter_type == "NUMBER":
                value = float(head["number_value"][0, index])
                value = max(float(schema.get("minimum", value)), min(float(schema.get("maximum", value)), value))
                parameters[name] = int(round(value)) if value.is_integer() else value
            elif parameter_type == "BOOLEAN":
                parameters[name] = bool(head["boolean_logits"][0, index].argmax())
            elif parameter_type == "STATE_REFERENCE":
                source_id = int(head["source_logits"][0, index].argmax())
                if source_id != 2 or not schema.get("state_reference_paths"):
                    if schema.get("required"):
                        return {"act": "ASK_CLARIFICATION", "operations": [], "confidence": {"act": act_confidence, "goal": goal_confidence, "parameters": parameter_confidence, "ood": ood_score, "overall": 0.0}}
                    continue
                reference_id = int(head["reference_logits"][0, index].argmax())
                parameters[name] = {"source": "state_reference", "path": schema["state_reference_paths"][reference_id]}
            else:
                source_id = int(head["source_logits"][0, index].argmax())
                if source_id != 1:
                    if schema.get("required"):
                        return {"act": "ASK_CLARIFICATION", "operations": [], "confidence": {"act": act_confidence, "goal": goal_confidence, "parameters": parameter_confidence, "ood": ood_score, "overall": 0.0}}
                    continue
                # Speech-frame spans require a lexical/frame alignment from the
                # dataset.  Do not fabricate character offsets at inference.
                if schema.get("required"):
                    return {"act": "ASK_CLARIFICATION", "operations": [], "confidence": {"act": act_confidence, "goal": goal_confidence, "parameters": parameter_confidence, "ood": ood_score, "overall": 0.0}}
        operations.append({"order": index + 1, "goal": goal, "parameters": parameters, "confidence": goal_confidence})

    if act == "EXECUTE" and not operations:
        return {"act": "ASK_CLARIFICATION", "operations": [], "confidence": {"act": act_confidence, "goal": 0.0, "parameters": 0.0, "ood": ood_score, "overall": 0.0}}
    return {
        "act": act,
        "operations": operations,
        "confidence": {
            "act": act_confidence,
            "goal": min((operation["confidence"] for operation in operations), default=1.0),
            "parameters": float(confidence_values[2]),
            "ood": ood_score,
            "overall": min(act_confidence, float(confidence_values.min()), 1.0 - ood_score),
        },
    }


def assemble_frame(
    prediction: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    request_id: str | None = None,
    model_version: str | None = None,
) -> dict[str, Any]:
    """Validate prediction and return the V0 Semantic Execution Frame."""

    frame = copy.deepcopy(dict(prediction))
    validate_config(config)
    confidence = frame.get("confidence")
    if not isinstance(confidence, dict):
        raise DatasetContractError("confidence is required")
    for key in ("act", "goal", "parameters", "ood", "overall"):
        if key in confidence and not isinstance(confidence[key], (int, float)):
            raise DatasetContractError(f"confidence.{key} must be numeric")
    validate_target({"act": frame.get("act"), "operations": frame.get("operations", [])}, config)
    if float(confidence.get("ood", 1.0)) > float(config["inference"]["max_ood_score"]):
        raise DatasetContractError("prediction is out of distribution")
    if float(confidence.get("overall", 0.0)) < float(config["inference"]["act_min_confidence"]):
        raise DatasetContractError("prediction confidence is below execution threshold")
    return {
        "model_version": model_version or config["project"]["architecture"],
        "request_id": request_id,
        "act": frame["act"],
        "operations": frame.get("operations", []),
        "confidence": confidence,
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


def export_package(
    path: str | Path,
    model: VoiceNativeSLU,
    config: Mapping[str, Any],
) -> None:
    """Export only the V0 model and config; tokenizer/response decoder are absent."""

    validate_config(config)
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    expected = {"model.safetensors", "config.json", "requirements.txt"}
    for child in destination.iterdir():
        if child.is_dir():
            raise ValueError(f"package destination contains directory: {child.name}")
        if child.name not in expected:
            child.unlink()
    save_safetensors(model, str(destination / "model.safetensors"))
    (destination / "config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (destination / "requirements.txt").write_text("torch\nsafetensors\ntqdm\n", encoding="utf-8")


def load_package(
    path: str | Path,
    device: torch.device,
) -> tuple[VoiceNativeSLU, dict[str, Any]]:
    source = Path(path)
    expected = {"model.safetensors", "config.json", "requirements.txt"}
    entries = list(source.iterdir())
    if {child.name for child in entries} != expected or not all(child.is_file() for child in entries):
        raise ValueError("V0 package must contain exactly three regular files")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    validate_config(config)
    model = build_model(config).to(device)
    load_safetensors(model, str(source / "model.safetensors"), device=str(device))
    model.eval()
    return model, config


# Historical entry point retained only as an explicit rejection.  A neural
# response generator would violate the V0 grounded-response boundary.
def generate_response(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("VALLS SLU V0 has no neural response decoder; use the template Response Module after Grounded Result")


def evaluate_generation(*_args: Any, **_kwargs: Any) -> None:
    raise RuntimeError("VALLS SLU V0 evaluates semantic frames, not neural response generation")


# This name remains importable for callers that only need the model type.
MultiTaskTransformer = VoiceNativeSLU
