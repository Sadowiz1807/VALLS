"""Configuration and capability schemas for the voice-native VALLS SLU V0 model."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

CONFIG_VERSION = "VALLS-SLU-V0"
MODEL_ARCHITECTURE = "voice_native_slu_v0"
CONTRACT_VERSION = "1.0"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.json")

ACTS = [
    "EXECUTE",
    "ASK_CLARIFICATION",
    "CONFIRM",
    "CANCEL",
    "RESPOND",
    "UNSUPPORTED",
]
PARAMETER_TYPES = {"ENUM", "NUMBER", "ENTITY", "STATE_REFERENCE", "FREE_TEXT", "BOOLEAN"}
INPUT_SPAN = "INPUT_SPAN"
STATE_REFERENCE = "STATE_REFERENCE"
ABSENT = "ABSENT"
IGNORE_INDEX = -100


def default_lexical_vocab() -> list[str]:
    """Return a vocabulary that can represent every UTF-8 transcript byte."""

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
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "type": parameter_type,
        "description": description,
        "required": required,
    }
    if values is not None:
        result["values"] = list(values)
    if minimum is not None:
        result["minimum"] = minimum
    if maximum is not None:
        result["maximum"] = maximum
    if state_reference_paths:
        result["state_reference_paths"] = list(state_reference_paths)
    return result


def _capability(
    description: str,
    actions: list[str],
    parameters: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "description": description,
        "actions": list(actions),
        "parameters": parameters,
    }


def _default_capabilities() -> dict[str, dict[str, Any]]:
    """The initial registry; the model can score any runtime schema set."""

    return {
        "APPLICATION_CONTROL": _capability(
            "Open, close, focus, or otherwise control a supported application.",
            ["OPEN", "CLOSE", "FOCUS"],
            {
                "application": _parameter(
                    "ENTITY",
                    "Name or alias of the application.",
                    required=True,
                    state_reference_paths=["state.current_target_application"],
                ),
            },
        ),
        "MEDIA_CONTROL": _capability(
            "Control playback or volume for a supported media provider.",
            ["PLAY", "PAUSE", "RESUME", "STOP", "NEXT", "PREVIOUS", "VOLUME_UP", "VOLUME_DOWN", "SET_VOLUME"],
            {
                "query": _parameter("FREE_TEXT", "Optional media or track query.", state_reference_paths=["state.active_media"]),
                "platform": _parameter("ENUM", "Media platform.", values=["YOUTUBE", "SPOTIFY", "LOCAL", "DEFAULT"]),
                "volume": _parameter("NUMBER", "Target volume percentage.", minimum=0, maximum=100),
            },
        ),
        "WEB_OPEN": _capability(
            "Open a URL or web target in a supported browser.",
            ["OPEN"],
            {
                "target": _parameter("FREE_TEXT", "URL or web target.", required=True, state_reference_paths=["state.active_url"]),
                "browser": _parameter("ENTITY", "Browser to use.", state_reference_paths=["state.active_browser"]),
            },
        ),
        "WEB_SEARCH": _capability(
            "Search the web using a supported search provider.",
            ["SEARCH"],
            {
                "query": _parameter("FREE_TEXT", "The natural-language search query.", required=True),
                "engine": _parameter("ENTITY", "Optional search engine."),
            },
        ),
        "WEB_NAVIGATE": _capability(
            "Navigate within the active browser page.",
            ["BACK", "FORWARD", "REFRESH", "SCROLL_UP", "SCROLL_DOWN", "GO_HOME"],
            {
                "amount": _parameter("NUMBER", "Scroll amount.", minimum=1, maximum=10),
            },
        ),
        "TAB_CONTROL": _capability(
            "Create, close, switch, or reopen a browser tab.",
            ["NEW", "CLOSE", "SWITCH", "REOPEN"],
            {
                "tab_index": _parameter("NUMBER", "One-based tab index.", minimum=1),
                "tab_reference": _parameter("STATE_REFERENCE", "Reference to the active browser.", state_reference_paths=["state.active_browser"]),
            },
        ),
        "RUN_COMMAND": _capability(
            "Run a registered safe system command through the harness.",
            ["RUN"],
            {
                "command_id": _parameter(
                    "ENUM",
                    "Registered command identifier.",
                    required=True,
                    values=["LOCK_SCREEN", "SHUTDOWN_SYSTEM", "RESTART_SYSTEM", "SLEEP_SYSTEM", "TAKE_SCREENSHOT", "SET_VOLUME"],
                ),
            },
        ),
        "TASK_STATUS": _capability(
            "Report the status of a previous or active task.",
            ["STATUS"],
            {"scope": _parameter("ENUM", "Task scope.", values=["LAST_TASK", "ACTIVE_TASK"])},
        ),
        "SOCIAL_RESPONSE": _capability(
            "Respond to a social utterance without executing a system operation.",
            ["RESPOND"],
            {"intent": _parameter("ENUM", "Social intent.", required=True, values=["GREETING", "THANKS", "GOODBYE", "ACKNOWLEDGEMENT"])},
        ),
    }


def capability_schemas(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Convert the config registry to the runtime schema representation."""

    ontology = config["ontology"]
    return [
        {"name": name, **copy.deepcopy(ontology["capabilities"][name])}
        for name in ontology["schema_order"]
    ]


def get_config() -> dict[str, Any]:
    lexical_vocab = default_lexical_vocab()
    capabilities = _default_capabilities()
    return {
        "project": {
            "name": "VALLS",
            "task": "voice_native_spoken_language_understanding",
            "architecture": CONFIG_VERSION,
            "contract_version": CONTRACT_VERSION,
        },
        "audio": {
            "sample_rate": 16_000,
            "channels": 1,
            "frontend": "log_mel",
            "feature_dim": 80,
            "n_fft": 400,
            "hop_length": 160,
            "win_length": 400,
            "f_min": 0.0,
            "f_max": 8_000.0,
        },
        "speech_encoder": {
            "architecture": "conformer",
            "d_model": 512,
            "layers": 8,
            "attention_heads": 8,
            "d_ff": 2_048,
            "dropout": 0.1,
            "subsampling_factor": 4,
            "conv_kernel_size": 31,
        },
        "semantic_resampler": {
            "type": "learnable_query_cross_attention",
            "latent_tokens": 32,
            "d_model": 512,
            "attention_heads": 8,
            "dropout": 0.1,
        },
        "semantic_core": {
            "type": "transformer_encoder",
            "layers": 4,
            "d_model": 512,
            "attention_heads": 8,
            "d_ff": 2_048,
            "dropout": 0.1,
        },
        "operation_decoder": {
            "type": "learned_query_cross_attention",
            "max_operations": 2,
            "d_model": 512,
            "attention_heads": 8,
            "dropout": 0.1,
        },
        "lexical_branch": {
            "enabled": True,
            "type": "ctc_utf8_bytes",
            "vocab_size": len(lexical_vocab),
            "blank_id": 0,
            "vocab_artifact": "lexical_vocab.json",
            "vocab": lexical_vocab,
        },
        "model": {
            "architecture": MODEL_ARCHITECTURE,
            "d_model": 512,
            "gradient_checkpointing": True,
        },
        "ontology": {
            "acts": list(ACTS),
            "capabilities": capabilities,
            "schema_order": list(capabilities),
        },
        "data": {
            "train_path": "Data/valls_slu_v0/train.jsonl",
            "validation_path": "Data/valls_slu_v0/validation.jsonl",
            "test_path": "Data/valls_slu_v0/test.jsonl",
            "required_fields": ["sample_id", "audio", "target"],
            "transcript_required_for": ["ENTITY", "FREE_TEXT"],
        },
        "training": {
            "seed": 42,
            "epochs": 20,
            "batch_size": 4,
            "validation_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 1e-4,
            "weight_decay": 0.01,
            "gradient_clip_norm": 1.0,
            "precision": "fp16_mixed",
            "num_workers": 0,
            "checkpoint_dir": "artifacts/valls_slu_v0/checkpoints",
            "checkpoint_prefix": "valls_slu_v0",
            "auto_resume": True,
            "resume_from": None,
            "loss_weights": {
                "act": 1.0,
                "operation_presence": 1.0,
                "goal_retrieval": 1.0,
                "action": 1.0,
                "parameters": 1.0,
                "confidence": 0.25,
                "ood": 0.5,
                "lexical": 0.25,
            },
            "stages": {
                "ACOUSTIC": {"enabled_losses": ["lexical"], "learning_rate_multiplier": 1.0},
                "BRIDGE": {"enabled_losses": ["lexical", "operation_presence"], "learning_rate_multiplier": 0.5},
                "SEMANTIC": {"enabled_losses": ["act", "operation_presence", "goal_retrieval", "action"], "learning_rate_multiplier": 1.0},
                "PARAMETER": {"enabled_losses": ["parameters"], "learning_rate_multiplier": 0.5},
                "SAFETY": {"enabled_losses": ["confidence", "ood"], "learning_rate_multiplier": 0.5},
                "JOINT": {"enabled_losses": ["act", "operation_presence", "goal_retrieval", "action", "parameters", "confidence", "ood", "lexical"], "learning_rate_multiplier": 1.0},
            },
        },
        "evaluation": {
            "act_accuracy": 0.95,
            "goal_retrieval_accuracy": 0.92,
            "action_accuracy": 0.92,
            "parameter_accuracy": 0.90,
            "unsafe_false_execute_rate": 0.0,
            "unsupported_recall": 0.90,
            "invalid_frame_rate": 0.0,
        },
        "inference": {
            "act_min_confidence": 0.70,
            "goal_min_confidence": 0.70,
            "action_min_confidence": 0.60,
            "parameter_min_confidence": 0.60,
            "operation_presence_threshold": 0.50,
            "max_ood_score": 0.50,
        },
    }


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_config(config: Mapping[str, Any]) -> None:
    """Validate the persisted V0 architecture and dynamic capability contract."""

    _require(isinstance(config, Mapping), "config must be an object")
    project = config.get("project")
    _require(isinstance(project, Mapping), "project must be an object")
    _require(project.get("architecture") == CONFIG_VERSION, "unsupported model architecture")
    _require(project.get("contract_version") == CONTRACT_VERSION, "unsupported contract version")

    audio = config.get("audio")
    _require(isinstance(audio, Mapping), "audio must be an object")
    for key in ("sample_rate", "channels", "feature_dim", "n_fft", "hop_length", "win_length"):
        _require(isinstance(audio.get(key), int) and audio[key] > 0, f"audio.{key} must be positive")
    _require(audio["channels"] == 1, "V0 requires mono audio")
    _require(audio["win_length"] <= audio["n_fft"], "audio.win_length cannot exceed n_fft")

    dimensions: list[int] = []
    for section_name in ("speech_encoder", "semantic_resampler", "semantic_core", "operation_decoder"):
        section = config.get(section_name)
        _require(isinstance(section, Mapping), f"{section_name} must be an object")
        for key in ("d_model", "attention_heads"):
            _require(isinstance(section.get(key), int) and section[key] > 0, f"{section_name}.{key} must be positive")
        _require(section["d_model"] % section["attention_heads"] == 0, f"{section_name}.d_model must divide attention_heads")
        dimensions.append(section["d_model"])
        if "layers" in section:
            _require(isinstance(section["layers"], int) and section["layers"] > 0, f"{section_name}.layers must be positive")
        _require(0 <= float(section.get("dropout", 0.0)) < 1, f"{section_name}.dropout must be in [0, 1)")
    _require(len(set(dimensions)) == 1, "all V0 representation dimensions must match")
    model = config.get("model")
    _require(isinstance(model, Mapping), "model must be an object")
    _require(model.get("architecture") == MODEL_ARCHITECTURE, "model architecture mismatch")
    _require(model.get("d_model") == dimensions[0], "model.d_model must match representation dimensions")

    speech = config["speech_encoder"]
    _require(speech.get("architecture") in {"conformer", "transformer_speech_encoder"}, "unsupported speech encoder")
    if speech.get("architecture") == "conformer":
        _require(int(speech.get("conv_kernel_size", 0)) > 1 and int(speech["conv_kernel_size"]) % 2 == 1, "Conformer kernel must be odd and > 1")

    lexical = config.get("lexical_branch")
    _require(isinstance(lexical, Mapping), "lexical_branch must be an object")
    vocab = lexical.get("vocab")
    _require(isinstance(vocab, list) and len(vocab) == lexical.get("vocab_size"), "lexical vocab must match vocab_size")
    _require(0 <= int(lexical.get("blank_id", -1)) < len(vocab), "invalid lexical blank_id")

    ontology = config.get("ontology")
    _require(isinstance(ontology, Mapping), "ontology must be an object")
    _require(ontology.get("acts") == list(ACTS), "ontology.acts must use the V0 ACT contract")
    capabilities = ontology.get("capabilities")
    schema_order = ontology.get("schema_order")
    _require(isinstance(capabilities, Mapping) and capabilities, "ontology.capabilities must be non-empty")
    _require(isinstance(schema_order, list) and set(schema_order) == set(capabilities), "schema_order must cover capabilities exactly")
    for goal in schema_order:
        schema = capabilities[goal]
        _require(isinstance(schema, Mapping) and isinstance(schema.get("description"), str) and schema["description"], f"invalid capability: {goal}")
        actions = schema.get("actions")
        _require(isinstance(actions, list) and actions and len(actions) == len(set(actions)), f"{goal}.actions must be unique")
        parameters = schema.get("parameters")
        _require(isinstance(parameters, Mapping), f"{goal}.parameters must be an object")
        for name, parameter in parameters.items():
            _require(isinstance(name, str) and name, f"invalid parameter name in {goal}")
            _require(isinstance(parameter, Mapping), f"invalid parameter schema: {goal}.{name}")
            parameter_type = parameter.get("type")
            _require(parameter_type in PARAMETER_TYPES, f"unsupported parameter type: {goal}.{name}")
            _require(isinstance(parameter.get("description"), str) and parameter["description"], f"missing description: {goal}.{name}")
            if parameter_type == "ENUM":
                values = parameter.get("values")
                _require(isinstance(values, list) and values and len(values) == len(set(values)), f"{goal}.{name} enum must be unique")
            if parameter_type == "NUMBER":
                if parameter.get("minimum") is not None and parameter.get("maximum") is not None:
                    _require(float(parameter["minimum"]) <= float(parameter["maximum"]), f"invalid numeric range: {goal}.{name}")
            references = parameter.get("state_reference_paths", [])
            _require(isinstance(references, list) and len(references) == len(set(references)), f"invalid state references: {goal}.{name}")

    inference = config.get("inference")
    _require(isinstance(inference, Mapping), "inference must be an object")
    for key in ("act_min_confidence", "goal_min_confidence", "action_min_confidence", "parameter_min_confidence", "operation_presence_threshold", "max_ood_score"):
        value = float(inference.get(key, -1))
        _require(0 <= value <= 1, f"inference.{key} must be in [0, 1]")


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    source = Path(path)
    config = json.loads(source.read_text(encoding="utf-8"))
    validate_config(config)
    return config


def save_config(config: Mapping[str, Any], path: str | Path = DEFAULT_CONFIG_PATH) -> Path:
    validate_config(config)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


read_config = load_config
write_config = save_config
validate_config(get_config())
