"""VALLS SLU V1 configuration and action-specific semantic schemas."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

CONFIG_VERSION = "VALLS-SLU-V1"
MODEL_ARCHITECTURE = "voice_native_slu_v1"
CONTRACT_VERSION = "2.0"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")

ACTS = ["EXECUTE", "ASK_CLARIFICATION", "CONFIRM", "CANCEL", "RESPOND", "UNSUPPORTED"]
ROOT_ACTION_DOMAINS = ["APPLICATION_CONTROL", "WEB_CONTROL", "SYSTEM_CONTROL", "RUN_COMMAND"]
PARAMETER_TYPES = {"ENUM", "NUMBER", "ENTITY", "FREE_TEXT", "BOOLEAN", "STATE_REFERENCE", "CONTEXT_REFERENCE"}
TURN_RELATIONS = ["NEW", "APPEND_AFTER", "MODIFY", "SUPERSEDE", "CONTINUE", "REPEAT", "REFERENCE"]
OPERATION_RELATIONS = ["NONE", "AFTER_SUCCESS", "AFTER_FAILURE", "AFTER_TERMINAL", "AFTER_START"]
CONTEXT_REFERENCE_TYPES = ["FOCUSED_ACTION", "ACTIVE_ACTION", "LAST_ACTION", "FOCUSED_TASK_GROUP", "LAST_TASK_GROUP", "RECENT_ACTION"]
INPUT_SPAN = "INPUT_SPAN"
STATE_REFERENCE = "STATE_REFERENCE"
CONTEXT_REFERENCE = "CONTEXT_REFERENCE"
IGNORE_INDEX = -100


def default_lexical_vocab() -> list[str]:
    return ["<BLANK>"] + [f"<BYTE:{value:02x}>" for value in range(256)]


def _parameter(
    parameter_type: str,
    description: str,
    *,
    required: bool = False,
    values: list[str] | None = None,
    minimum: float | None = None,
    maximum: float | None = None,
    state_reference_paths: list[str] | None = None,
    context_reference_types: list[str] | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {"type": parameter_type, "description": description, "required": required}
    if values is not None:
        result["values"] = list(values)
    if minimum is not None:
        result["minimum"] = minimum
    if maximum is not None:
        result["maximum"] = maximum
    if state_reference_paths:
        result["state_reference_paths"] = list(state_reference_paths)
    if context_reference_types:
        result["context_reference_types"] = list(context_reference_types)
    return result


def _action(description: str, parameters: dict[str, dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"description": description, "parameters": parameters or {}}


def _capability(description: str, actions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {"description": description, "actions": actions}


def _entity(description: str, required: bool = False, state_reference_paths: list[str] | None = None) -> dict[str, Any]:
    return _parameter("ENTITY", description, required=required, state_reference_paths=state_reference_paths)


def _free_text(description: str, required: bool = False) -> dict[str, Any]:
    return _parameter("FREE_TEXT", description, required=required)


def _browser() -> dict[str, Any]:
    return _entity("Browser name or alias.", state_reference_paths=["state.active_browser"])


def _default_capabilities() -> dict[str, dict[str, Any]]:
    app = _entity("Native/local application name or alias.", required=True, state_reference_paths=["state.current_target_application"])
    optional_app = _entity("Native/local application name or alias.", state_reference_paths=["state.current_target_application"])
    browser = _browser()
    return {
        "APPLICATION_CONTROL": _capability(
            "Control native/local applications.",
            {
                "OPEN": _action("Open an application.", {"application": app}),
                "CLOSE": _action("Close an application.", {"application": app}),
                "FOCUS": _action("Focus an application.", {"application": app}),
                "PLAY": _action("Play media through a native application.", {"application": app, "query": _free_text("Media query.", True)}),
                "PAUSE": _action("Pause native application media.", {"application": optional_app}),
                "RESUME": _action("Resume native application media.", {"application": optional_app}),
                "STOP": _action("Stop native application media.", {"application": optional_app}),
                "NEXT": _action("Skip to next native application media.", {"application": optional_app}),
                "PREVIOUS": _action("Skip to previous native application media.", {"application": optional_app}),
            },
        ),
        "WEB_CONTROL": _capability(
            "Control browsers, websites, web media, tabs, and navigation.",
            {
                "OPEN": _action("Open a web target.", {"target": _parameter("FREE_TEXT", "URL or web target.", required=True, state_reference_paths=["state.active_url"]), "browser": browser}),
                "SEARCH": _action("Search the web.", {"query": _free_text("Search query.", True), "browser": browser, "engine": _entity("Search engine name or alias.")}),
                "BACK": _action("Navigate back in a browser.", {"browser": browser}),
                "FORWARD": _action("Navigate forward in a browser.", {"browser": browser}),
                "REFRESH": _action("Refresh a browser page.", {"browser": browser}),
                "SCROLL_UP": _action("Scroll up in a browser.", {"amount": _parameter("NUMBER", "Scroll amount.", minimum=1, maximum=10), "browser": browser}),
                "SCROLL_DOWN": _action("Scroll down in a browser.", {"amount": _parameter("NUMBER", "Scroll amount.", minimum=1, maximum=10), "browser": browser}),
                "TAB.NEW": _action("Open a new browser tab.", {"browser": browser}),
                "TAB.CLOSE": _action("Close a browser tab.", {"tab_reference": _parameter("CONTEXT_REFERENCE", "Semantic tab reference.", context_reference_types=["FOCUSED_ACTION", "ACTIVE_ACTION", "LAST_ACTION"])}),
                "TAB.SWITCH": _action("Switch to a browser tab.", {"tab_reference": _parameter("CONTEXT_REFERENCE", "Semantic tab reference.", required=True, context_reference_types=["FOCUSED_ACTION", "ACTIVE_ACTION", "LAST_ACTION"])}),
                "TAB.REOPEN": _action("Reopen a browser tab."),
                "PLAY": _action("Play media through a website or browser.", {"query": _free_text("Media query.", True), "site": _entity("Website or media site."), "browser": browser}),
                "PAUSE": _action("Pause web media.", {"browser": browser}),
                "RESUME": _action("Resume web media.", {"browser": browser}),
                "STOP": _action("Stop web media.", {"browser": browser}),
                "NEXT": _action("Skip to next web media.", {"browser": browser}),
                "PREVIOUS": _action("Skip to previous web media.", {"browser": browser}),
            },
        ),
        "SYSTEM_CONTROL": _capability(
            "Control operating-system and device state.",
            {
                "MEDIA.SET_VOLUME": _action("Set device output volume.", {"volume": _parameter("NUMBER", "Device volume percentage.", required=True, minimum=0, maximum=100)}),
                "MEDIA.VOLUME_UP": _action("Increase device output volume.", {"amount": _parameter("NUMBER", "Volume step.", minimum=1, maximum=100)}),
                "MEDIA.VOLUME_DOWN": _action("Decrease device output volume.", {"amount": _parameter("NUMBER", "Volume step.", minimum=1, maximum=100)}),
                "MEDIA.MUTE": _action("Mute device output."),
                "MEDIA.UNMUTE": _action("Unmute device output."),
                "POWER.LOCK": _action("Lock the device."),
                "POWER.SHUTDOWN": _action("Shut down the device."),
                "POWER.RESTART": _action("Restart the device."),
                "POWER.SLEEP": _action("Put the device to sleep."),
                "SCREENSHOT.CAPTURE": _action("Capture a screenshot."),
            },
        ),
        "RUN_COMMAND": _capability(
            "Run a registered safe command through the Harness.",
            {"EXECUTE": _action("Execute a registered command.", {"command_id": _entity("Registered command identifier or alias.", True), "arguments": _free_text("Registered command arguments.")})},
        ),
    }


def canonical_action_id(domain: str, action_path: str) -> str:
    return f"{domain}.{action_path}"


def capability_schemas(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [{"name": name, **copy.deepcopy(config["ontology"]["capabilities"][name])} for name in config["ontology"]["schema_order"]]


def count_action_slots(config: Mapping[str, Any]) -> tuple[int, int]:
    capabilities = config["ontology"]["capabilities"]
    actions = sum(len(schema["actions"]) for schema in capabilities.values())
    slots = sum(len(action["parameters"]) for schema in capabilities.values() for action in schema["actions"].values())
    return actions, slots


def get_config() -> dict[str, Any]:
    lexical_vocab = default_lexical_vocab()
    capabilities = _default_capabilities()
    return {
        "project": {"name": "VALLS", "task": "voice_native_spoken_language_understanding", "architecture": CONFIG_VERSION, "contract_version": CONTRACT_VERSION},
        "audio": {"sample_rate": 16000, "channels": 1, "frontend": "log_mel", "feature_dim": 80, "n_fft": 400, "hop_length": 160, "win_length": 400, "f_min": 0.0, "f_max": 8000.0},
        "speech_encoder": {"architecture": "conformer", "d_model": 512, "layers": 8, "attention_heads": 8, "d_ff": 2048, "dropout": 0.1, "subsampling_factor": 4, "conv_kernel_size": 31},
        "semantic_resampler": {"type": "learnable_query_cross_attention", "latent_tokens": 32, "d_model": 512, "attention_heads": 8, "dropout": 0.1},
        "semantic_core": {"type": "transformer_encoder", "layers": 4, "d_model": 512, "attention_heads": 8, "d_ff": 2048, "dropout": 0.1},
        "operation_decoder": {"type": "learned_query_cross_attention", "max_operations": 4, "d_model": 512, "attention_heads": 8, "dropout": 0.1},
        "lexical_branch": {"enabled": True, "type": "ctc_utf8_bytes", "vocab_size": len(lexical_vocab), "blank_id": 0, "vocab_artifact": "lexical_vocab.json", "vocab": lexical_vocab},
        "model": {"architecture": MODEL_ARCHITECTURE, "d_model": 512, "gradient_checkpointing": True},
        "ontology": {"acts": list(ACTS), "root_action_domains": list(ROOT_ACTION_DOMAINS), "capabilities": capabilities, "schema_order": list(capabilities)},
        "relations": {"turn": list(TURN_RELATIONS), "operation": list(OPERATION_RELATIONS), "context_reference_types": list(CONTEXT_REFERENCE_TYPES)},
        "data": {"required_fields": ["sample_id", "audio", "target"], "transcript_required_for": ["ENTITY", "FREE_TEXT"]},
        "training": {"seed": 42, "epochs": 20, "batch_size": 4, "validation_batch_size": 4, "gradient_accumulation_steps": 4, "learning_rate": 1e-4, "weight_decay": 0.01, "gradient_clip_norm": 1.0, "precision": "fp16_mixed", "num_workers": 0, "loss_weights": {"act": 1.0, "operation_presence": 1.0, "goal_retrieval": 1.0, "action": 1.0, "parameters": 1.0, "turn_relation": 0.5, "context_required": 0.5, "context_reference": 0.5, "operation_relations": 0.5, "confidence": 0.25, "ood": 0.5, "lexical": 0.25, "bridge_alignment": 0.5, "parameter_presence_negative_weight": 0.25, "parameter_hard_negative_schemas": 4}, "stages": {"ACOUSTIC": {"enabled_losses": ["lexical"]}, "BRIDGE": {"enabled_losses": ["bridge_alignment"]}, "SEMANTIC": {"enabled_losses": ["act", "operation_presence", "goal_retrieval", "action", "turn_relation", "context_required", "context_reference", "operation_relations"]}, "PARAMETER": {"enabled_losses": ["parameters"]}, "SAFETY": {"enabled_losses": ["confidence", "ood"]}, "JOINT": {"enabled_losses": ["act", "operation_presence", "goal_retrieval", "action", "parameters", "turn_relation", "context_required", "context_reference", "operation_relations", "confidence", "ood", "lexical"]}}},
        "inference": {"act_min_confidence": 0.70, "goal_min_confidence": 0.70, "action_min_confidence": 0.60, "parameter_min_confidence": 0.60, "operation_presence_threshold": 0.50, "max_ood_score": 0.50},
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_config(config: Mapping[str, Any]) -> None:
    _require(config["project"]["architecture"] == CONFIG_VERSION, "unsupported V1 architecture")
    _require(config["project"]["contract_version"] == CONTRACT_VERSION, "unsupported V1 contract")
    _require(config["ontology"]["acts"] == ACTS, "invalid ACT ontology")
    domains = config["ontology"]["root_action_domains"]
    _require(domains == ROOT_ACTION_DOMAINS, "V1 requires exactly four root action domains")
    capabilities = config["ontology"]["capabilities"]
    _require(set(capabilities) == set(ROOT_ACTION_DOMAINS), "old or unknown root domain exists")
    seen: set[str] = set()
    action_count = slot_count = 0
    for domain in ROOT_ACTION_DOMAINS:
        schema = capabilities[domain]
        _require(isinstance(schema.get("actions"), Mapping), f"{domain}.actions must be a dict")
        for path, action in schema["actions"].items():
            _require(path and not path.startswith(".") and not path.endswith(".") and ".." not in path, f"invalid action path: {domain}.{path}")
            canonical = canonical_action_id(domain, path)
            _require(canonical not in seen, f"duplicate canonical action: {canonical}")
            seen.add(canonical); action_count += 1
            parameters = action.get("parameters", {})
            _require(isinstance(parameters, Mapping), f"{canonical}.parameters must be a dict")
            slot_count += len(parameters)
            for name, parameter in parameters.items():
                _require(parameter.get("type") in PARAMETER_TYPES, f"invalid parameter type: {canonical}.{name}")
                _require(isinstance(parameter.get("required"), bool), f"required must be bool: {canonical}.{name}")
                if parameter["type"] == "ENUM":
                    _require(isinstance(parameter.get("values"), list) and parameter["values"], f"ENUM requires values: {canonical}.{name}")
    _require(action_count == 37, f"V1 requires 37 actions, got {action_count}")
    _require(slot_count == 38, f"V1 requires 38 parameter slots, got {slot_count}")
    _require(config["relations"]["turn"] == TURN_RELATIONS, "invalid turn relations")
    _require(config["relations"]["operation"] == OPERATION_RELATIONS, "invalid operation relations")
    _require(config["lexical_branch"]["vocab_size"] == len(config["lexical_branch"]["vocab"]), "lexical vocab mismatch")
    for section in ("speech_encoder", "semantic_resampler", "semantic_core", "operation_decoder"):
        _require(config[section]["d_model"] % config[section]["attention_heads"] == 0, f"{section}.d_model must divide heads")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(Path(path).read_text(encoding="utf-8")); validate_config(config); return config


def save_config(config: Mapping[str, Any], path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    validate_config(config); destination=Path(path); destination.parent.mkdir(parents=True,exist_ok=True); tmp=destination.with_suffix('.tmp'); tmp.write_text(json.dumps(config,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); tmp.replace(destination); return destination

read_config = load_config
write_config = save_config
validate_config(get_config())
