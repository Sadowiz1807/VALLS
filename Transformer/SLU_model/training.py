"""Loss, inference assembly, training loop, checkpointing and four-file packages."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from safetensors.torch import load_model as load_safetensors
from safetensors.torch import save_model as save_safetensors
from tokenizers import Tokenizer
from torch import Tensor
from tqdm.auto import tqdm

from .config import ABSENT, INPUT_SPAN, NONE, STATE_REFERENCE, validate_config
from .dataset import IGNORE_INDEX, validate_generated_response
from .model import MultiTaskTransformer, build_model


def _active_cross_entropy(logits: Tensor, labels: Tensor) -> Tensor | None:
    active = labels != IGNORE_INDEX
    return F.cross_entropy(logits[active], labels[active]) if active.any() else None


def compute_loss(
    outputs: dict[str, Any], batch: dict[str, Any], config: dict[str, Any]
) -> tuple[Tensor, dict[str, Tensor]]:
    act_logits = outputs["act_logits"]
    goal_logits = outputs["goal_logits"].clone()
    acts = config["labels"]["acts"]
    goals = config["labels"]["goals"]
    for row, act_label in enumerate(batch["act_label"].tolist()):
        allowed = {goals.index(goal) for goal in config["labels"]["goal_masks_by_act"][acts[act_label]]}
        blocked = [index for index in range(len(goals)) if index not in allowed]
        goal_logits[row, blocked] = torch.finfo(goal_logits.dtype).min
    losses: dict[str, Tensor] = {
        "act": F.cross_entropy(act_logits, batch["act_label"]),
        "goal": F.cross_entropy(goal_logits, batch["goal_label"]),
    }
    parameter_losses: list[Tensor] = []
    for key, logits in outputs["categorical_logits"].items():
        loss = _active_cross_entropy(logits, batch["categorical_labels"][key])
        if loss is not None:
            parameter_losses.append(loss)
    for key, start_logits in outputs["span_start_logits"].items():
        labels = batch["span_labels"][key]
        start_loss = _active_cross_entropy(start_logits, labels[:, 0])
        end_loss = _active_cross_entropy(outputs["span_end_logits"][key], labels[:, 1])
        if start_loss is not None and end_loss is not None:
            parameter_losses.append((start_loss + end_loss) / 2)
    losses["parameters"] = (
        torch.stack(parameter_losses).mean()
        if parameter_losses
        else outputs["act_logits"].sum() * 0
    )
    if "response_logits" not in outputs:
        raise ValueError("Training requires response_logits")
    losses["response"] = F.cross_entropy(
        outputs["response_logits"].transpose(1, 2),
        batch["response_labels"],
        ignore_index=IGNORE_INDEX,
        label_smoothing=float(config["training"].get("label_smoothing", 0.0)),
    )
    weights = config["training"]["loss_weights"]
    total = sum(losses[name] * float(weights[name]) for name in losses)
    return total, losses


def _move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved = {
        key: value.to(device) if isinstance(value, Tensor) else value
        for key, value in batch.items()
    }
    moved["categorical_labels"] = {
        key: value.to(device) for key, value in batch["categorical_labels"].items()
    }
    moved["span_labels"] = {
        key: value.to(device) for key, value in batch["span_labels"].items()
    }
    return moved


def macro_f1(gold: list[int], predicted: list[int], labels: Iterable[int]) -> float:
    scores = []
    for label in labels:
        tp = sum(g == label and p == label for g, p in zip(gold, predicted))
        fp = sum(g != label and p == label for g, p in zip(gold, predicted))
        fn = sum(g == label and p != label for g, p in zip(gold, predicted))
        if tp + fp + fn:
            scores.append(2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores) if scores else 0.0


def validate_model(
    model: MultiTaskTransformer,
    batches: Iterable[dict[str, Any]],
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    counts = {
        "samples": 0, "act_correct": 0, "unsafe": 0, "non_execute_total": 0,
        "parameter_total": 0, "parameter_correct": 0,
        "span_total": 0, "span_correct": 0,
        "reference_total": 0, "reference_correct": 0,
        "response_total": 0, "response_correct": 0,
    }
    losses: list[float] = []
    response_losses: list[float] = []
    goal_gold: list[int] = []
    goal_predicted: list[int] = []
    acts, goals = config["labels"]["acts"], config["labels"]["goals"]
    execute_id = acts.index("EXECUTE")

    use_amp = device.type == "cuda" and config["training"]["precision"] == "fp16_mixed"
    with torch.no_grad():
        for raw_batch in batches:
            batch = _move_batch(raw_batch, device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                outputs = model(
                    batch["input_ids"], batch["input_mask"],
                    batch["decoder_input_ids"], batch["response_mask"],
                )
                loss, parts = compute_loss(outputs, batch, config)
            batch_size = len(batch["act_label"])
            losses.append(float(loss) * batch_size)
            response_losses.append(float(parts["response"]) * batch_size)

            active_response = batch["response_labels"] != IGNORE_INDEX
            response_predictions = outputs["response_logits"].argmax(-1)
            counts["response_total"] += int(active_response.sum())
            counts["response_correct"] += int(
                ((response_predictions == batch["response_labels"]) & active_response).sum()
            )

            act_predictions = outputs["act_logits"].argmax(-1)
            goal_logits = outputs["goal_logits"].clone()
            for row, act_id in enumerate(act_predictions.tolist()):
                allowed = {
                    goals.index(goal)
                    for goal in config["labels"]["goal_masks_by_act"][acts[act_id]]
                }
                blocked = [index for index in range(len(goals)) if index not in allowed]
                goal_logits[row, blocked] = torch.finfo(goal_logits.dtype).min
            goal_predictions = goal_logits.argmax(-1)

            counts["samples"] += len(act_predictions)
            counts["act_correct"] += int((act_predictions == batch["act_label"]).sum())
            counts["unsafe"] += int(
                ((act_predictions == execute_id) & (batch["act_label"] != execute_id)).sum()
            )
            counts["non_execute_total"] += int((batch["act_label"] != execute_id).sum())
            goal_gold.extend(batch["goal_label"].tolist())
            goal_predicted.extend(goal_predictions.tolist())

            for key, logits in outputs["categorical_logits"].items():
                labels_for_head = batch["categorical_labels"][key]
                active = labels_for_head != IGNORE_INDEX
                correct = int((logits.argmax(-1)[active] == labels_for_head[active]).sum())
                counts["parameter_total"] += int(active.sum())
                counts["parameter_correct"] += correct
                if key.endswith("__reference"):
                    counts["reference_total"] += int(active.sum())
                    counts["reference_correct"] += correct

            for key, start_logits in outputs["span_start_logits"].items():
                labels_for_head = batch["span_labels"][key]
                active = labels_for_head[:, 0] != IGNORE_INDEX
                exact = (
                    (start_logits.argmax(-1) == labels_for_head[:, 0])
                    & (outputs["span_end_logits"][key].argmax(-1) == labels_for_head[:, 1])
                    & active
                )
                counts["span_total"] += int(active.sum())
                counts["span_correct"] += int(exact.sum())

    if not losses:
        raise ValueError("No validation batches")

    def ratio(correct: str, total: str) -> float:
        return counts[correct] / counts[total] if counts[total] else 0.0

    return {
        "validation_loss": sum(losses) / counts["samples"],
        "act_accuracy": ratio("act_correct", "samples"),
        "goal_macro_f1": macro_f1(goal_gold, goal_predicted, range(len(goals))),
        "parameter_accuracy": ratio("parameter_correct", "parameter_total"),
        "span_exact_match": ratio("span_correct", "span_total"),
        "state_reference_accuracy": ratio("reference_correct", "reference_total"),
        "unsafe_false_execute_rate": ratio("unsafe", "non_execute_total"),
        "response_token_accuracy": ratio("response_correct", "response_total"),
        "response_perplexity": math.exp(min(sum(response_losses) / counts["samples"], 20.0)),
    }


def targets_reached(metrics: dict[str, float], targets: dict[str, float]) -> bool:
    missing = set(targets) - set(metrics)
    if missing:
        raise ValueError(f"Missing target metrics: {sorted(missing)}")
    maximum_targets = {
        "unsafe_false_execute_rate",
        "response_empty_rate",
        "response_truncation_rate",
        "premature_success_claim_rate",
        "invalid_frame_rate",
    }
    return all(
        metrics[name] <= target if name in maximum_targets else metrics[name] >= target
        for name, target in targets.items()
    )


def _masked_argmax(logits: Tensor, allowed: list[int]) -> tuple[int, float]:
    probabilities = logits.softmax(-1)
    mask = torch.zeros_like(probabilities, dtype=torch.bool)
    mask[..., allowed] = True
    probabilities = probabilities.masked_fill(~mask, 0)
    index = int(probabilities.argmax(-1).item())
    return index, float(probabilities[..., index].item())


def predict(
    model: MultiTaskTransformer,
    batch: dict[str, Any],
    config: dict[str, Any],
) -> dict[str, Any]:
    if batch["input_ids"].shape[0] != 1:
        raise ValueError("predict accepts exactly one sample")
    model.eval()
    with torch.no_grad():
        outputs = model(batch["input_ids"], batch["input_mask"])
    act_probabilities = outputs["act_logits"].softmax(-1)
    act_id = int(act_probabilities.argmax(-1).item())
    act = config["labels"]["acts"][act_id]
    act_confidence = float(act_probabilities[0, act_id].item())
    if act_confidence < float(config["inference"]["act_threshold"]):
        return {
            "act": "UNSUPPORTED",
            "goal": None,
            "parameters": {},
            "confidence": {"act": act_confidence},
        }
    allowed_goals = [
        config["labels"]["goals"].index(goal)
        for goal in config["labels"]["goal_masks_by_act"][act]
    ]
    goal_id, goal_confidence = _masked_argmax(outputs["goal_logits"][0], allowed_goals)
    goal = config["labels"]["goals"][goal_id]
    if goal_confidence < float(config["inference"]["goal_threshold"]):
        return {
            "act": "UNSUPPORTED",
            "goal": None,
            "parameters": {},
            "confidence": {"act": act_confidence, "goal": goal_confidence},
        }
    result: dict[str, Any] = {
        "act": act,
        "goal": None if goal == NONE else goal,
        "parameters": {},
        "confidence": {
            "act": float(outputs["act_logits"].softmax(-1)[0, act_id].item()),
            "goal": goal_confidence,
        },
    }
    if goal == NONE:
        return result
    specification = config["ontology"]["goal_parameters"].get(goal, {})
    for name, parameter in specification.get("properties", {}).items():
        key = f"{goal}__{name}"
        head_key = f"{key}__source" if parameter.get("type") == "dynamic" else key
        if head_key not in outputs["categorical_logits"]:
            continue
        labels = config["heads"]["categorical"][head_key]
        label_id = int(outputs["categorical_logits"][head_key].argmax(-1).item())
        value = labels[label_id]
        parameter_confidence = float(
            outputs["categorical_logits"][head_key].softmax(-1)[0, label_id].item()
        )
        result["confidence"][key] = parameter_confidence
        if parameter_confidence < float(config["inference"]["parameter_threshold"]):
            result["act"] = "ASK_CLARIFICATION"
            result["parameters"] = {}
            return result
        if value == ABSENT:
            continue
        if value == INPUT_SPAN:
            start_logits = outputs["span_start_logits"][key][0]
            offsets = batch["token_offsets"][0].to(start_logits.device)
            valid = offsets[:, 0] >= 0
            start_scores = start_logits.masked_fill(
                ~valid, torch.finfo(start_logits.dtype).min
            )
            start_token = int(start_scores.argmax().item())
            end_scores = outputs["span_end_logits"][key][0].clone()
            end_scores[~valid] = torch.finfo(end_scores.dtype).min
            end_scores[:start_token] = torch.finfo(end_scores.dtype).min
            end_token = int(end_scores.argmax().item())
            start = int(offsets[start_token, 0].item())
            end = int(offsets[end_token, 1].item())
            text = batch["text"][0]
            span_value = text[start:end]
            result["parameters"][name] = (
                int(span_value)
                if name == "tab_index"
                else {
                    "source": "input_span",
                    "start": start,
                    "end": end,
                    "value": span_value,
                }
            )
        elif value == STATE_REFERENCE:
            reference_key = f"{key}__reference"
            reference_id = int(outputs["categorical_logits"][reference_key].argmax(-1).item())
            path = config["heads"]["categorical"][reference_key][reference_id]
            result["parameters"][name] = {"source": "state_reference", "path": path}
        elif parameter.get("type") == "integer":
            result["parameters"][name] = int(value)
        else:
            result["parameters"][name] = value
    return result


def generate_response(
    model: MultiTaskTransformer,
    batch: dict[str, Any],
    tokenizer: Tokenizer,
    config: dict[str, Any],
    semantic_frame: dict[str, Any],
) -> dict[str, Any]:
    """Greedy autoregressive response generation for one encoded sample."""
    if batch["input_ids"].shape[0] != 1:
        raise ValueError("generate_response accepts exactly one sample")
    special = config["tokenizer"]["special_tokens"]
    bos_id = tokenizer.token_to_id(special["bos"])
    eos_id = tokenizer.token_to_id(special["eos"])
    if bos_id is None or eos_id is None:
        raise ValueError("Tokenizer is missing BOS/EOS")
    maximum = int(config["model"]["max_response_length"])
    blocked_ids = {
        tokenizer.token_to_id(token)
        for name, token in special.items()
        if name != "eos"
    }
    blocked_ids.discard(None)
    model.eval()
    use_amp = batch["input_ids"].device.type == "cuda" and config["training"]["precision"] == "fp16_mixed"
    with torch.no_grad():
        with torch.amp.autocast(device_type=batch["input_ids"].device.type, enabled=use_amp):
            memory = model.encode(batch["input_ids"], batch["input_mask"])
        token_ids = [bos_id]
        for _ in range(maximum):
            decoder_ids = torch.tensor([token_ids], device=memory.device, dtype=torch.long)
            response_mask = torch.ones_like(decoder_ids, dtype=torch.bool)
            with torch.amp.autocast(device_type=memory.device.type, enabled=use_amp):
                decoded = model.decode(decoder_ids, memory, response_mask, batch["input_mask"])
                logits = model.response_head(decoded[:, -1])
            if blocked_ids:
                logits[:, list(blocked_ids)] = torch.finfo(logits.dtype).min
            next_id = int(logits.argmax(-1).item())
            token_ids.append(next_id)
            if next_id == eos_id:
                break
    content_ids = token_ids[1:-1] if token_ids[-1] == eos_id else token_ids[1:]
    response_text = validate_generated_response(
        tokenizer.decode(content_ids), semantic_frame["act"]
    )
    return {
        "response_text": response_text,
        "token_ids": token_ids,
        "response_metadata": {
            "owner": "multi_task_transformer_v2",
            "model_generated": True,
            "terminated_by_eos": token_ids[-1] == eos_id,
        },
    }


def evaluate_generation(
    model: MultiTaskTransformer,
    batches: Iterable[dict[str, Any]],
    tokenizer: Tokenizer,
    config: dict[str, Any],
    device: torch.device,
) -> dict[str, float]:
    """Run locked-test autoregressive response gates one sample at a time."""
    counts = {"samples": 0, "eos": 0, "empty": 0, "truncated": 0, "consistent": 0, "premature": 0, "invalid_frame": 0}
    maximum = int(config["model"]["max_response_length"])
    for raw_batch in batches:
        batch = _move_batch(raw_batch, device)
        if batch["input_ids"].shape[0] != 1:
            raise ValueError("Generation evaluation requires batch_size=1")
        counts["samples"] += 1
        try:
            use_amp = device.type == "cuda" and config["training"]["precision"] == "fp16_mixed"
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                prediction = predict(model, batch, config)
            frame = assemble_frame(prediction, config)
            generated = generate_response(model, batch, tokenizer, config, frame)
        except ValueError as error:
            counts["empty"] += int("empty" in str(error).lower())
            counts["premature"] += int("premature" in str(error).lower())
            counts["invalid_frame"] += int(
                "empty" not in str(error).lower() and "premature" not in str(error).lower()
            )
            continue
        text = generated["response_text"]
        terminated = generated["response_metadata"]["terminated_by_eos"]
        counts["eos"] += int(terminated)
        counts["truncated"] += int(not terminated and len(generated["token_ids"]) == maximum + 1)
        dynamic_values = [
            str(value["value"])
            for value in frame["parameters"].values()
            if isinstance(value, dict) and value.get("source") == "input_span"
        ]
        counts["consistent"] += int(
            all(value.casefold() in text.casefold() for value in dynamic_values)
        )
    if not counts["samples"]:
        raise ValueError("No generation evaluation batches")
    total = counts["samples"]
    return {
        "response_eos_rate": counts["eos"] / total,
        "response_empty_rate": counts["empty"] / total,
        "response_truncation_rate": counts["truncated"] / total,
        "response_semantic_consistency": counts["consistent"] / total,
        "premature_success_claim_rate": counts["premature"] / total,
        "invalid_frame_rate": counts["invalid_frame"] / total,
    }


def assemble_frame(prediction: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    act, goal = prediction["act"], prediction.get("goal")
    if act not in config["labels"]["acts"]:
        raise ValueError(f"Unknown ACT: {act}")
    normalized_goal = goal or NONE
    if normalized_goal not in config["labels"]["goal_masks_by_act"][act]:
        raise ValueError(f"GOAL {normalized_goal} is incompatible with ACT {act}")
    parameters = copy.deepcopy(prediction.get("parameters", {}))
    specification = config["ontology"]["goal_parameters"].get(goal, {})
    properties = specification.get("properties", {})
    if not set(parameters) <= set(properties):
        raise ValueError("Prediction contains parameters outside the GOAL schema")
    missing = set(specification.get("required", [])) - set(parameters)
    if missing:
        raise ValueError(f"Prediction is missing required parameters: {sorted(missing)}")
    for name, value in parameters.items():
        parameter = properties[name]
        if "enum" in parameter and value not in parameter["enum"]:
            raise ValueError(f"Invalid value for {goal}.{name}: {value!r}")
        if parameter.get("type") == "integer":
            if not isinstance(value, int) or not parameter.get("minimum", value) <= value <= parameter.get("maximum", value):
                raise ValueError(f"Invalid integer for {goal}.{name}: {value!r}")
        if parameter.get("type") == "dynamic":
            if not isinstance(value, dict) or value.get("source") not in {"input_span", "state_reference"}:
                raise ValueError(f"Invalid dynamic value for {goal}.{name}")
            if value["source"] == "input_span":
                start, end, text = value.get("start"), value.get("end"), value.get("value")
                if not isinstance(start, int) or not isinstance(end, int) or not isinstance(text, str) or start < 0 or end <= start:
                    raise ValueError(f"Invalid input span for {goal}.{name}")
            elif value.get("path") not in config["heads"]["references"].get(f"{goal}__{name}", []):
                raise ValueError(f"Non-allowlisted state reference for {goal}.{name}")
    return {
        "schema_version": config["project"]["contract_version"],
        "act": act,
        "goal": goal,
        "parameters": parameters,
    }


def train_epoch(
    model: MultiTaskTransformer,
    batches: Iterable[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    config: dict[str, Any],
    device: torch.device,
    scaler: torch.amp.GradScaler | None = None,
    epoch: int | None = None,
) -> float:
    model.train()
    accumulation = int(config["training"]["gradient_accumulation_steps"])
    clip_norm = float(config["training"]["gradient_clip_norm"])
    use_amp = device.type == "cuda" and config["training"]["precision"] == "fp16_mixed"
    scaler = scaler or torch.amp.GradScaler(device.type, enabled=use_amp)
    total = count = pending = 0
    optimizer.zero_grad(set_to_none=True)

    def step(micro_batches: int) -> None:
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

    progress = tqdm(
        batches,
        desc=f"Epoch {epoch}" if epoch is not None else "Train",
        unit="batch",
        dynamic_ncols=True,
        leave=True,
    )
    for batch in progress:
        tensors = _move_batch(batch, device)
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            outputs = model(
                tensors["input_ids"], tensors["input_mask"],
                tensors["decoder_input_ids"], tensors["response_mask"],
            )
            loss, _ = compute_loss(outputs, tensors, config)
        scaler.scale(loss / accumulation).backward()
        total += float(loss.detach())
        count += 1
        progress.set_postfix(loss=f"{total / count:.4f}", refresh=False)
        pending += 1
        if pending == accumulation:
            step(pending)
            pending = 0
    if pending:
        step(pending)
    if not count:
        raise ValueError("No training batches")
    return total / count


def save_checkpoint(path: str | Path, state: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    try:
        torch.save(state, temporary)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def export_package(
    path: str | Path,
    model: MultiTaskTransformer,
    tokenizer: Tokenizer,
    config: dict[str, Any],
) -> None:
    validate_config(config)
    destination = Path(path)
    destination.mkdir(parents=True, exist_ok=True)
    expected = {"model.safetensors", "config.json", "tokenizer.json", "requirements.txt"}
    extra_directories = [child.name for child in destination.iterdir() if child.is_dir()]
    if extra_directories:
        raise ValueError(f"Package destination contains directories: {extra_directories}")
    for child in destination.iterdir():
        if child.name not in expected:
            child.unlink()
    save_safetensors(model, str(destination / "model.safetensors"))
    (destination / "config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tokenizer.save(str(destination / "tokenizer.json"))
    (destination / "requirements.txt").write_text(
        "torch\ntokenizers\nsafetensors\ntqdm\n", encoding="utf-8"
    )


def load_package(
    path: str | Path, device: torch.device
) -> tuple[MultiTaskTransformer, Tokenizer, dict[str, Any]]:
    source = Path(path)
    expected = {"model.safetensors", "config.json", "tokenizer.json", "requirements.txt"}
    entries = list(source.iterdir())
    if {child.name for child in entries} != expected or not all(child.is_file() for child in entries):
        raise ValueError("Package must contain exactly four regular files")
    config = json.loads((source / "config.json").read_text(encoding="utf-8"))
    validate_config(config)
    tokenizer = Tokenizer.from_file(str(source / "tokenizer.json"))
    if tokenizer.get_vocab_size() != config["tokenizer"]["vocab_size"]:
        raise ValueError("Tokenizer vocabulary does not match config")
    model = build_model(config).to(device)
    load_safetensors(model, str(source / "model.safetensors"), device=str(device))
    model.eval()
    return model, tokenizer, config
