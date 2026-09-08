"""Configuration and closed label spaces for Multi-task Transformer V2."""

from __future__ import annotations

from typing import Any

NONE = "NONE"
ABSENT = "ABSENT"
INPUT_SPAN = "INPUT_SPAN"
STATE_REFERENCE = "STATE_REFERENCE"

SPECIAL_TOKENS = {
    "unk": "<UNK>", "pad": "<PAD>", "bos": "<BOS>", "eos": "<EOS>",
    "context_open": "<CONTEXT>", "context_close": "</CONTEXT>",
    "state_open": "<STATE>", "state_close": "</STATE>",
    "metadata_open": "<METADATA>", "metadata_close": "</METADATA>",
    "input_open": "<INPUT>", "input_close": "</INPUT>",
    "contract": "<CONTRACT:2.0>",
}
STATE_REFERENCE_PATHS = {
    "application": ["state.current_target_application"],
    "query": ["state.active_media"],
    "target": ["state.active_url"],
    "browser": ["state.active_browser"],
    "tab_reference": ["state.active_browser"],
}
COMMANDS = {
    "LOCK_SCREEN": {}, "SHUTDOWN_SYSTEM": {}, "RESTART_SYSTEM": {},
    "SLEEP_SYSTEM": {}, "TAKE_SCREENSHOT": {},
    "SET_VOLUME": {"level": {"type": "integer", "minimum": 0, "maximum": 100}},
}
GOAL_PARAMETERS = {
    "APPLICATION_CONTROL": {"required": ["action", "application"], "properties": {"action": {"enum": ["OPEN", "CLOSE", "FOCUS"]}, "application": {"type": "dynamic"}}},
    "MEDIA_CONTROL": {"required": ["action"], "properties": {"action": {"enum": ["PLAY", "PAUSE", "RESUME", "STOP", "NEXT", "PREVIOUS", "VOLUME_UP", "VOLUME_DOWN", "SET_VOLUME"]}, "query": {"type": "dynamic"}, "platform": {"enum": ["YOUTUBE", "SPOTIFY", "LOCAL", "DEFAULT"]}, "volume": {"type": "integer", "minimum": 0, "maximum": 100}}},
    "WEB_OPEN": {"required": ["target"], "properties": {"target": {"type": "dynamic"}, "browser": {"type": "dynamic"}}},
    "WEB_SEARCH": {"required": ["query"], "properties": {"query": {"type": "dynamic"}, "engine": {"type": "dynamic"}}},
    "WEB_NAVIGATE": {"required": ["action"], "properties": {"action": {"enum": ["BACK", "FORWARD", "REFRESH", "SCROLL_UP", "SCROLL_DOWN", "GO_HOME"]}, "amount": {"type": "integer", "minimum": 1, "maximum": 10}}},
    "TAB_CONTROL": {"required": ["action"], "properties": {"action": {"enum": ["NEW", "CLOSE", "SWITCH", "REOPEN"]}, "tab_index": {"type": "integer", "minimum": 1}, "tab_reference": {"type": "dynamic"}}},
    "RUN_COMMAND": {"required": ["command_id"], "properties": {"command_id": {"enum": list(COMMANDS)}, "arguments": {"type": "object"}}},
    "TASK_STATUS": {"required": [], "properties": {"scope": {"enum": ["LAST_TASK", "ACTIVE_TASK"]}}},
    "SOCIAL_RESPONSE": {"required": ["intent"], "properties": {"intent": {"enum": ["GREETING", "THANKS", "GOODBYE", "ACKNOWLEDGEMENT"]}}},
}
ACTS = ["EXECUTE", "ASK_CLARIFICATION", "CONFIRM", "CANCEL", "RESPOND", "UNSUPPORTED"]
GOALS = ["APPLICATION_CONTROL", "MEDIA_CONTROL", "WEB_OPEN", "WEB_SEARCH", "WEB_NAVIGATE", "TAB_CONTROL", "RUN_COMMAND", "TASK_STATUS", "SOCIAL_RESPONSE"]


def _head_definitions(ontology: dict[str, Any]) -> dict[str, Any]:
    categorical: dict[str, list[str]] = {}
    spans: list[str] = []
    references: dict[str, list[str]] = {}
    for goal, specification in ontology["goal_parameters"].items():
        required = set(specification["required"])
        for name, parameter in specification["properties"].items():
            key = f"{goal}__{name}"
            optional = [] if name in required else [ABSENT]
            if "enum" in parameter:
                categorical[key] = [*optional, *parameter["enum"]]
            elif parameter.get("type") == "dynamic":
                references[key] = ontology["state_reference_paths"].get(name, [])
                sources = [*optional, INPUT_SPAN]
                if references[key]:
                    sources.append(STATE_REFERENCE)
                categorical[f"{key}__source"] = sources
                spans.append(key)
                if references[key]:
                    categorical[f"{key}__reference"] = references[key]
            elif name == "volume":
                categorical[key] = [*optional, *(str(value) for value in range(101))]
            elif name == "amount":
                categorical[key] = [*optional, *(str(value) for value in range(1, 11))]
            elif name == "tab_index":
                categorical[f"{key}__source"] = [*optional, INPUT_SPAN]
                spans.append(key)
            elif name == "arguments":
                # No head until command-specific typed argument schemas exist.
                continue
        if goal == "RUN_COMMAND":
            categorical[f"{goal}__command_id"] = list(ontology["commands"])
    return {"categorical": categorical, "spans": spans, "references": references}


def get_config() -> dict[str, Any]:
    ontology = {
        "acts": ACTS.copy(), "goals": GOALS.copy(),
        "goal_parameters": GOAL_PARAMETERS,
        "state_reference_paths": STATE_REFERENCE_PATHS,
        "commands": COMMANDS,
    }
    config = {
        "project": {"name": "AI_voice_assistant", "task": "multi_task_semantics_and_response", "initialization": "random_weights", "contract_version": "2.0", "baseline": "multi_task_transformer_v2"},
        "tokenizer": {"library": "tokenizers", "model": "unigram_bytelevel", "vocab_size": 8_000, "path": "artifacts/multitask_v2/tokenizer.json", "normalization": "NFC", "special_tokens": SPECIAL_TOKENS.copy()},
        "model": {"architecture": "multi_task_encoder_response_decoder", "d_model": 512, "encoder_layers": 6, "decoder_layers": 4, "attention_heads": 8, "d_ff": 2048, "dropout": 0.1, "max_input_length": 256, "max_response_length": 1024, "gradient_checkpointing": True, "tie_response_embeddings": True},
        "ontology": ontology,
        "data": {"train_path": "Data/multitask_v2/train_fit.jsonl", "validation_path": "Data/multitask_v2/validation.jsonl", "locked_test_path": "Data/multitask_v2/test_20.jsonl", "required_fields": ["sample_id", "dialogue_id", "turn_id", "context", "state", "metadata", "current_text", "target", "gold_response_text", "response_metadata"]},
        "training": {"seed": 42, "epochs": 20, "early_stopping_patience": 3, "early_stopping_min_delta": 0.001, "batch_size": 3, "validation_batch_size": 3, "gradient_accumulation_steps": 6, "learning_rate": 1e-4, "weight_decay": 0.01, "label_smoothing": 0.1, "gradient_clip_norm": 1.0, "precision": "fp16_mixed", "num_workers": 0, "checkpoint_dir": "artifacts/multitask_v2/checkpoints", "checkpoint_prefix": "multi_task_transformer_v2", "auto_resume": True, "resume_from": None, "log_dir": "artifacts/multitask_v2/runs", "loss_weights": {"act": 1.0, "goal": 1.0, "parameters": 1.0, "response": 1.0}},
        "evaluation": {"act_accuracy": 0.95, "goal_macro_f1": 0.92, "parameter_accuracy": 0.90, "unsafe_false_execute_rate": 0.0, "response_token_accuracy": 0.80, "response_eos_rate": 0.99, "response_empty_rate": 0.0, "response_truncation_rate": 0.01, "premature_success_claim_rate": 0.0, "invalid_frame_rate": 0.0},
        "inference": {"act_threshold": 0.0, "goal_threshold": 0.0, "parameter_threshold": 0.0, "response_decoding": "greedy", "response_temperature": 1.0},
    }
    acts = list(config["ontology"]["acts"])
    goals = [*config["ontology"]["goals"], NONE]
    config["labels"] = {
        "acts": acts,
        "goals": goals,
        "goal_masks_by_act": {
            "EXECUTE": [
                goal for goal in goals
                if goal not in {NONE, "TASK_STATUS", "SOCIAL_RESPONSE"}
            ],
            "ASK_CLARIFICATION": [goal for goal in goals if goal != NONE],
            "RESPOND": ["TASK_STATUS", "SOCIAL_RESPONSE"],
            "CONFIRM": [NONE],
            "CANCEL": [NONE],
            "UNSUPPORTED": [NONE],
        },
    }
    config["heads"] = _head_definitions(config["ontology"])
    return config


def validate_config(config: dict[str, Any]) -> None:
    model = config["model"]
    if model.get("architecture") != "multi_task_encoder_response_decoder":
        raise ValueError("model.architecture must be multi_task_encoder_response_decoder")
    for name in ("d_model", "encoder_layers", "decoder_layers", "attention_heads", "d_ff", "max_input_length", "max_response_length"):
        if not isinstance(model.get(name), int) or model[name] <= 0:
            raise ValueError(f"model.{name} must be a positive integer")
    if model["d_model"] % model["attention_heads"]:
        raise ValueError("model.d_model must be divisible by model.attention_heads")
    if not 0 <= float(model["dropout"]) < 1:
        raise ValueError("model.dropout must be in [0, 1)")
    if config["project"]["contract_version"] != "2.0":
        raise ValueError("Multi-task Transformer V2 requires contract_version 2.0")
    if model["max_response_length"] < 3:
        raise ValueError("model.max_response_length must allow BOS, text and EOS")
    if float(config["training"]["loss_weights"].get("response", 0)) <= 0:
        raise ValueError("training.loss_weights.response must be positive")
    if int(config["training"]["early_stopping_patience"]) <= 0:
        raise ValueError("training.early_stopping_patience must be positive")
    if float(config["training"]["early_stopping_min_delta"]) < 0:
        raise ValueError("training.early_stopping_min_delta must be non-negative")
    acts, goals = config["labels"]["acts"], config["labels"]["goals"]
    if len(acts) != len(set(acts)) or len(goals) != len(set(goals)):
        raise ValueError("ACT and GOAL labels must be unique")
    if NONE not in goals:
        raise ValueError("GOAL labels must include NONE")
    known_goals = set(goals)
    for act, allowed in config["labels"]["goal_masks_by_act"].items():
        if act not in acts or not set(allowed) <= known_goals or not allowed:
            raise ValueError(f"Invalid GOAL mask for ACT {act}")
    for key, values in config["heads"]["categorical"].items():
        if not values or len(values) != len(set(values)):
            raise ValueError(f"Categorical head {key} must have unique labels")


def label_id(config: dict[str, Any], group: str, value: str) -> int:
    try:
        return config["labels"][group].index(value)
    except ValueError as exc:
        raise ValueError(f"Unknown {group} label: {value}") from exc


def masked_goal_ids(config: dict[str, Any], act: str) -> list[int]:
    return [label_id(config, "goals", goal) for goal in config["labels"]["goal_masks_by_act"][act]]


validate_config(get_config())
