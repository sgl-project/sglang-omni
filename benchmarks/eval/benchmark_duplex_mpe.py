# SPDX-License-Identifier: Apache-2.0
"""Evaluate full-duplex and turn-based /v1/realtime servers on Duplex-MPE."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import re
import shlex
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal
from urllib.parse import urlsplit

import httpx
import websockets
from pydantic import JsonValue

from benchmarks.dataset.duplex_mpe import (
    DUPLEX_MPE_DATASET_ID,
    DUPLEX_MPE_DATASET_REVISION,
    DUPLEX_MPE_DUTY_PREAMBLE_RELATIVE_PATH,
    AddressingSelection,
    DuplexMpeScenarioRecord,
    compute_file_sha256,
    inspect_duplex_mpe_dataset,
    load_duplex_mpe_scenarios,
    read_json_file,
    resolve_dataset_audio_path,
    resolve_retained_scenario_audio_path,
    validate_safe_identifier,
    write_json_atomic,
)
from benchmarks.duplex.mpe.metrics import (
    REFERENCE_SILERO_VAD_VERSION,
    format_markdown_score_table,
    score_scenario_reports,
    validate_scenario_reports,
)
from benchmarks.duplex.mpe.protocol import (
    compute_protocol_fingerprint,
    load_reference_seeds,
    resolve_scenario_seed,
)
from benchmarks.duplex.mpe.realtime_session import (
    DEFAULT_SESSION_TIMEOUT_SECONDS,
    SEED_POLICY_METADATA_ONLY,
    ProtocolSessionError,
    RealtimeProtocolSelection,
    negotiate_realtime_endpoint,
    run_realtime_scenario,
    summarize_transport_exception,
    validate_realtime_base_url,
)
from benchmarks.duplex.mpe.semantic import (
    DEFAULT_ASR_MODEL_ID,
    JudgeEndpointConfig,
    build_semantic_tasks,
    judge_tasks_via_http,
    load_judge_endpoint_config,
    load_validated_existing_verdicts,
    make_deterministic_empty_asr_verdict,
    public_judge_metadata,
    resolve_chat_completions_url,
    resolve_transcriptions_url,
    sanitize_endpoint_origin,
    semantic_task_key,
    transcribe_tasks_via_http,
    validate_http_endpoint_url,
    validate_transcribed_against_tasks,
)
from benchmarks.duplex.mpe.vad import silero_package_version
from benchmarks.runtime_metrics import collect_benchmark_provenance
from sglang_omni.utils.logging import configure_dependency_loggers

logger = logging.getLogger(__name__)

EvaluationPhase = Literal["all", "generate", "asr", "judge", "score"]


@dataclass(frozen=True, kw_only=True)
class DuplexMpeEvalConfig:
    dataset_root: Path
    output_dir: Path
    phase: EvaluationPhase
    addressing: AddressingSelection = "all"
    mini: bool = False
    max_scenarios: int | None = None
    model: str | None = None
    model_revision: str | None = None
    launch_command: str | None = None
    base_url: str | None = None
    protocol: RealtimeProtocolSelection = "auto"
    input_sample_rate_hz: int | None = None
    output_sample_rate_hz: int | None = None
    max_concurrency: int = 1
    warmup_sessions: int = 1
    session_timeout_seconds: float = DEFAULT_SESSION_TIMEOUT_SECONDS
    asr_base_url: str | None = None
    asr_model: str | None = None
    asr_api_key_env: str | None = None
    asr_max_concurrency: int = 8
    judge_config_path: Path | None = None
    strict_reference_vad_version: bool = False
    verify_retained_audio: bool = True
    trust_env: bool = False

    def __post_init__(self) -> None:
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be >= 1")
        elif self.warmup_sessions < 0:
            raise ValueError("warmup_sessions must be >= 0")
        elif self.session_timeout_seconds <= 0.0:
            raise ValueError("session_timeout_seconds must be positive")
        elif self.asr_max_concurrency < 1:
            raise ValueError("asr_max_concurrency must be >= 1")
        elif self.asr_model is not None and not self.asr_model.strip():
            raise ValueError("asr_model must be non-empty when provided")
        else:
            pass
        resolved_dataset = self.dataset_root.expanduser().resolve()
        resolved_output = self.output_dir.expanduser().resolve()
        if resolved_output == resolved_dataset or resolved_output.is_relative_to(
            resolved_dataset
        ):
            raise ValueError("output_dir must be outside dataset_root")
        else:
            pass
        if self.phase in ("all", "generate"):
            if not self.base_url:
                raise ValueError(f"--base-url is required for phase={self.phase}")
            else:
                validate_realtime_base_url(self.base_url)
        else:
            pass
        if self.phase == "asr" and not self.asr_base_url:
            raise ValueError("--asr-base-url is required for phase=asr")
        elif self.asr_base_url:
            validate_http_endpoint_url(self.asr_base_url)
        else:
            pass
        if self.phase == "judge" and self.judge_config_path is None:
            raise ValueError("--judge-config is required for phase=judge")
        elif (
            self.phase == "all"
            and self.judge_config_path is not None
            and not self.asr_base_url
            and not (resolved_output / "transcribed.json").is_file()
        ):
            raise ValueError(
                "--asr-base-url is required when --judge-config is set for phase=all "
                "without an existing transcribed.json"
            )
        else:
            pass
        if self.launch_command is not None:
            sanitize_launch_command(self.launch_command)
        else:
            pass


SENSITIVE_ENV_SEGMENTS: frozenset[str] = frozenset(
    {
        "KEY",
        "KEYS",
        "APIKEY",
        "TOKEN",
        "SECRET",
        "SECRETS",
        "PASS",
        "PASSWORD",
        "PASSWD",
        "PASSPHRASE",
        "CREDENTIAL",
        "CREDENTIALS",
        "AUTH",
        "AUTHORIZATION",
        "BEARER",
        "COOKIE",
        "COOKIES",
        "SESSIONID",
        "HEADER",
        "HEADERS",
        "JWT",
        "NETRC",
        "USERINFO",
        "CERT",
        "OAUTH",
        "OAUTH2",
        "SASL",
    }
)
SENSITIVE_FLAG_SEGMENTS: frozenset[str] = frozenset(
    {
        "key",
        "keys",
        "apikey",
        "token",
        "secret",
        "secrets",
        "pass",
        "password",
        "passwd",
        "pwd",
        "passphrase",
        "credential",
        "credentials",
        "auth",
        "authorization",
        "bearer",
        "cookie",
        "cookies",
        "sessionid",
        "header",
        "headers",
        "jwt",
        "user",
        "username",
        "login",
        "netrc",
        "cert",
        "oauth",
        "oauth2",
        "sasl",
    }
)
SAFE_KEY_FLAGS: frozenset[str] = frozenset(
    {"sort-key", "sort-keys", "cache-key", "partition-key", "extra-keys"}
)
SAFE_ENV_NAMES: frozenset[str] = frozenset(
    {
        "CUDA_VISIBLE_DEVICES",
        "ASCEND_RT_VISIBLE_DEVICES",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NCCL_P2P_DISABLE",
        "NCCL_IB_DISABLE",
        "NCCL_DEBUG",
        "TORCH_CUDA_ARCH_LIST",
        "SGLANG_ENABLE_TORCH_COMPILE",
        "PYTHONUNBUFFERED",
        "VLLM_ATTENTION_BACKEND",
    }
)
SHELL_WRAPPER_NAMES: frozenset[str] = frozenset(
    {"sh", "bash", "zsh", "dash", "ksh", "fish", "csh", "tcsh"}
)
SAFE_SERVER_EXECUTABLES: frozenset[str] = frozenset(
    {
        "env",
        "uv",
        "python",
        "python3",
        "sgl-omni",
        "sglang_omni",
        "sglang",
        "vllm",
        "torchrun",
    }
)
SAFE_SUBCOMMANDS: frozenset[str] = frozenset({"run", "serve", "launch"})
SAFE_BOOLEAN_FLAGS: frozenset[str] = frozenset(
    {
        "--enable-realtime",
        "--no-enable-realtime",
        "--text-only",
        "--trust-remote-code",
        "--disable-cuda-graph",
        "--disable-radix-cache",
        "--enable-metrics",
        "--enable-torch-compile",
        "--skip-server-warmup",
    }
)
SAFE_VALUE_FLAGS: frozenset[str] = frozenset(
    {
        "--model-path",
        "--model",
        "--model-name",
        "--served-model-name",
        "--variant",
        "--port",
        "--host",
        "--tp",
        "--tp-size",
        "--dp",
        "--dp-size",
        "--pp-size",
        "--ep-size",
        "--dtype",
        "--quantization",
        "--mem-fraction-static",
        "--max-running-requests",
        "--max-total-tokens",
        "--max-seq-len",
        "--context-length",
        "--attention-backend",
        "--sampling-backend",
        "--reasoning-parser",
        "--device",
        "--seed",
        "--log-level",
        "--chat-template",
        "--config",
        "--config-path",
        "--profile",
        "--max-sessions",
        "--max_sessions",
        "--voice",
        "--timeout-s",
        "--timeout_s",
        "--realtime-idle-timeout-s",
        "--realtime_idle_timeout_s",
        "--realtime-session-update-timeout-s",
        "--realtime_session_update_timeout_s",
    }
)
SAFE_DOTTED_CONFIG_PREFIXES: frozenset[str] = frozenset(
    {
        "sampling",
        "vision",
        "speech",
        "window",
        "preprocessing",
        "perception",
        "thinker",
        "talker",
        "talker_ar",
        "code2wav",
        "mimi_encode",
        "lm",
        "asr",
        "tts",
        "vad",
        "turn_detection",
        "session_nodes",
        "services",
        "stages",
        "engine",
    }
)
SAFE_URL_OPTION_SUFFIXES: frozenset[str] = frozenset(
    {
        ".base_url",
        ".alignment_url",
        ".heard_asr_url",
    }
)
SAFE_EMPTY_VALUE_OPTION_SUFFIXES: frozenset[str] = frozenset(
    {
        ".alignment_url",
        ".heard_asr_url",
    }
)
SAFE_DOTTED_CONFIG_FLAG_RE = re.compile(
    r"^--[A-Za-z_][A-Za-z0-9_-]*(?:\.[A-Za-z_][A-Za-z0-9_-]*)+$"
)
SAFE_PLAIN_LONG_FLAG_RE = re.compile(r"^--[A-Za-z_][A-Za-z0-9_-]*$")
BLOCKED_LONG_CODE_FLAGS: frozenset[str] = frozenset(
    {"c", "e", "eval", "command", "exec", "code", "script"}
)
SAFE_EXECUTABLE_TOKEN_RE = re.compile(r"^(?:/[A-Za-z0-9_./-]+|[A-Za-z0-9_.-]+)$")
SAFE_SCALAR_VALUE_RE = re.compile(r"^[A-Za-z0-9_./-]+$")
SAFE_HOST_VALUE_RE = re.compile(r"^\[?[A-Za-z0-9_.:-]+\]?$")
SAFE_ENV_VALUE_RE = re.compile(r"^[A-Za-z0-9_.,-]+$")
SAFE_MODULE_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
EMBEDDED_URL_USERINFO_RE = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s'\"/\\<>]+@")


def is_allowed_server_executable(executable_name: str) -> bool:
    return (
        executable_name in SAFE_SERVER_EXECUTABLES
        or executable_name in SHELL_WRAPPER_NAMES
        or bool(re.fullmatch(r"python3\.\d+", executable_name))
    )


def is_safe_executable_token(raw_token: str) -> bool:
    if (
        ".." in raw_token
        or SAFE_EXECUTABLE_TOKEN_RE.fullmatch(raw_token) is None
        or has_url_userinfo(raw_token)
        or contains_embedded_credential_syntax(raw_token)
    ):
        return False
    else:
        return is_allowed_server_executable(Path(raw_token).name.lower())


def is_safe_dotted_config_flag(flag_token: str) -> bool:
    if SAFE_DOTTED_CONFIG_FLAG_RE.fullmatch(flag_token) is None:
        return False
    elif KNOWN_TOKEN_PREFIX_RE.search(flag_token) is not None:
        return False
    elif is_sensitive_cli_flag(flag_token):
        return False
    else:
        dotted_parts = flag_token.lstrip("-").split(".")
        if len(dotted_parts) > 5:
            return False
        elif not all(
            SAFE_DOTTED_PATH_COMPONENT_RE.fullmatch(part) is not None
            for part in dotted_parts
        ):
            return False
        else:
            top_namespace = dotted_parts[0].lower()
            segments = [
                part.lower()
                for part in split_identifier_segments(flag_token.lstrip("-"))
            ]
            return top_namespace in SAFE_DOTTED_CONFIG_PREFIXES and not any(
                part in BLOCKED_LONG_CODE_FLAGS for part in segments
            )


def is_safe_value_flag(flag_token: str) -> bool:
    return flag_token.lower() in SAFE_VALUE_FLAGS or is_safe_dotted_config_flag(
        flag_token
    )


def is_safe_http_endpoint_value(value_token: str) -> bool:
    if has_url_userinfo(value_token) or contains_embedded_credential_syntax(
        value_token
    ):
        return False
    else:
        try:
            parsed = urlsplit(value_token)
            return (
                parsed.scheme in ("http", "https")
                and bool(parsed.hostname)
                and (parsed.port is None or 1 <= parsed.port <= 65535)
                and parsed.username is None
                and parsed.password is None
                and not parsed.query
                and not parsed.fragment
                and bool(re.fullmatch(r"^[A-Za-z0-9_./-]*$", parsed.path))
            )
        except ValueError:
            return False


SAFE_DOTTED_PATH_COMPONENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]{0,31}$")
KNOWN_TOKEN_PREFIX_RE = re.compile(
    r"(?:sk-[A-Za-z0-9_-]{8,}|hf_[A-Za-z0-9]{8,}|ghp_[A-Za-z0-9]{8,}|"
    r"github_pat_[A-Za-z0-9_]{8,}|xox[bp]-[A-Za-z0-9-]{8,}|"
    r"AKIA[0-9A-Z]{12,}|ASIA[0-9A-Z]{12,}|eyJ[A-Za-z0-9_-]{10,}\.)"
)
CANONICAL_SENSITIVE_LEAF_NAMES: frozenset[str] = frozenset(
    {
        "api_key",
        "apikey",
        "openai_api_key",
        "anthropic_api_key",
        "hf_token",
        "token",
        "access_token",
        "auth_token",
        "bearer_token",
        "secret",
        "client_secret",
        "password",
        "passwd",
        "pass",
        "passphrase",
        "credential",
        "credentials",
        "authorization",
        "header",
        "headers",
        "cookie",
        "cookies",
        "user",
        "username",
        "proxy_user",
        "cert",
    }
)


def is_well_formed_sensitive_flag_name(flag_token: str) -> bool:
    if (
        has_url_userinfo(flag_token)
        or KNOWN_TOKEN_PREFIX_RE.search(flag_token) is not None
    ):
        return False
    elif SAFE_PLAIN_LONG_FLAG_RE.fullmatch(flag_token) is not None:
        normalized_plain = flag_token.lstrip("-").lower().replace("-", "_")
        return normalized_plain in CANONICAL_SENSITIVE_LEAF_NAMES
    elif SAFE_DOTTED_CONFIG_FLAG_RE.fullmatch(flag_token) is not None:
        dotted_parts = flag_token.lstrip("-").split(".")
        if len(dotted_parts) > 5:
            return False
        else:
            leaf_normalized = dotted_parts[-1].lower().replace("-", "_")
            non_leaf_parts = dotted_parts[:-1]
            return (
                leaf_normalized in CANONICAL_SENSITIVE_LEAF_NAMES
                and all(
                    SAFE_DOTTED_PATH_COMPONENT_RE.fullmatch(part) is not None
                    for part in non_leaf_parts
                )
                and not any(
                    is_sensitive_cli_flag(f"--{part}") for part in non_leaf_parts
                )
            )
    else:
        return False


def is_safe_flag_scalar_value(flag_token: str, value_token: str) -> bool:
    lowered_flag = flag_token.lower()
    if value_token == "" and any(
        lowered_flag.endswith(suffix) for suffix in SAFE_EMPTY_VALUE_OPTION_SUFFIXES
    ):
        return True
    elif has_url_userinfo(value_token) or contains_embedded_credential_syntax(
        value_token
    ):
        return False
    elif lowered_flag == "--host":
        return "://" not in value_token and bool(
            SAFE_HOST_VALUE_RE.fullmatch(value_token)
        )
    elif lowered_flag == "--device":
        return bool(
            re.fullmatch(
                r"^(?:cpu|cuda(?::\d+)?|xpu(?::\d+)?|npu(?::\d+)?)$", value_token
            )
        )
    elif any(lowered_flag.endswith(suffix) for suffix in SAFE_URL_OPTION_SUFFIXES):
        return is_safe_http_endpoint_value(value_token)
    else:
        return bool(SAFE_SCALAR_VALUE_RE.fullmatch(value_token))


def split_identifier_segments(identifier: str) -> list[str]:
    decamelized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", identifier)
    return [part for part in re.split(r"[^A-Za-z0-9]+", decamelized) if part]


def is_sensitive_env_name(variable_name: str) -> bool:
    segments = [part.upper() for part in split_identifier_segments(variable_name)]
    return any(part in SENSITIVE_ENV_SEGMENTS for part in segments)


def is_sensitive_cli_flag(flag_token: str) -> bool:
    if flag_token == "-H":
        return True
    elif not flag_token.startswith("--"):
        return False
    else:
        segments = [
            part.lower() for part in split_identifier_segments(flag_token.lstrip("-"))
        ]
        hyphenated = "-".join(segments)
        if hyphenated in SAFE_KEY_FLAGS:
            return False
        else:
            return any(part in SENSITIVE_FLAG_SEGMENTS for part in segments)


def has_url_userinfo(token_value: str) -> bool:
    if EMBEDDED_URL_USERINFO_RE.search(token_value) is not None:
        return True
    elif "://" not in token_value:
        return False
    else:
        try:
            parsed = urlsplit(token_value)
            return parsed.username is not None or parsed.password is not None
        except ValueError:
            return True


def contains_embedded_credential_syntax(argument_value: str) -> bool:
    if has_url_userinfo(argument_value):
        return True
    elif re.search(r"\bBearer\s+\S+", argument_value, flags=re.IGNORECASE):
        return True
    elif re.search(
        r"\b(?:sk-[A-Za-z0-9_-]{8,}|hf_[A-Za-z0-9]{8,}|ghp_[A-Za-z0-9]{8,}|"
        r"github_pat_[A-Za-z0-9_]{8,}|xox[bp]-[A-Za-z0-9-]{8,}|"
        r"AKIA[0-9A-Z]{12,}|ASIA[0-9A-Z]{12,}|eyJ[A-Za-z0-9_-]{10,}\.)\b",
        argument_value,
    ):
        return True
    else:
        for key_match in re.finditer(
            r'([A-Za-z][A-Za-z0-9_.-]*)["\']*\s*(?:[:=]|->)', argument_value
        ):
            key_identifier = key_match.group(1)
            if is_sensitive_cli_flag(f"--{key_identifier}") or is_sensitive_env_name(
                key_identifier
            ):
                return True
            else:
                pass
        for xml_match in re.finditer(r"<([A-Za-z][A-Za-z0-9_.-]*)>", argument_value):
            tag_identifier = xml_match.group(1)
            if is_sensitive_cli_flag(f"--{tag_identifier}") or is_sensitive_env_name(
                tag_identifier
            ):
                return True
            else:
                pass
        if not any(
            ch.isspace() for ch in argument_value
        ) and not argument_value.startswith("-"):
            return False
        else:
            try:
                inner_tokens = shlex.split(argument_value, posix=True)
            except ValueError:
                return True
            for subtoken in inner_tokens:
                if (
                    subtoken.startswith("-")
                    and not subtoken.startswith("--")
                    and len(subtoken) >= 2
                    and not subtoken[1:].isdigit()
                ):
                    return True
                elif subtoken.startswith("-") and "=" in subtoken:
                    subflag, subval = subtoken.split("=", 1)
                    if is_sensitive_cli_flag(subflag) or has_url_userinfo(subval):
                        return True
                    else:
                        pass
                elif is_sensitive_cli_flag(subtoken):
                    return True
                elif has_url_userinfo(subtoken):
                    return True
                else:
                    pass
            return False


def sanitize_launch_command(
    launch_command: str | None, *, nesting_depth: int = 0
) -> str | None:
    if launch_command is None:
        return None
    elif nesting_depth > 3:
        raise ValueError(
            "Nested shell wrappers exceed maximum depth in --launch-command"
        )
    elif any(char in launch_command for char in ("\n", "\r", "\x00")):
        raise ValueError("Control characters are not permitted in --launch-command")
    elif "`" in launch_command or "$(" in launch_command:
        raise ValueError("Command substitution is not permitted in --launch-command")
    else:
        try:
            lexer = shlex.shlex(launch_command, posix=True, punctuation_chars=True)
            lexer.whitespace_split = True
            punct_tokens = list(lexer)
            raw_tokens = shlex.split(launch_command, posix=True)
        except ValueError:
            raise ValueError("Invalid shell quoting in --launch-command") from None
        if any(
            punct_tok in {";", "|", "||", "&&", "&", "(", ")"}
            for punct_tok in punct_tokens
        ):
            raise ValueError(
                "Shell compound operators are not permitted in --launch-command"
            )
        else:
            pass
        sanitized_tokens: list[str] = []
        executable_seen = False
        current_executable = ""
        index = 0
        while index < len(raw_tokens):
            token = raw_tokens[index]
            if not executable_seen:
                if (
                    not token.startswith("-")
                    and "=" in token
                    and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token.split("=", 1)[0])
                ):
                    env_name, env_val = token.split("=", 1)
                    if has_url_userinfo(env_val) or contains_embedded_credential_syntax(
                        env_val
                    ):
                        if is_sensitive_env_name(env_name):
                            sanitized_tokens.append(f"{env_name}=<REDACTED>")
                        else:
                            raise ValueError(
                                "Embedded credentials in environment assignment are not permitted in --launch-command"
                            )
                    elif is_sensitive_env_name(env_name):
                        sanitized_tokens.append(f"{env_name}=<REDACTED>")
                    elif env_name in SAFE_ENV_NAMES and bool(
                        SAFE_ENV_VALUE_RE.fullmatch(env_val)
                    ):
                        sanitized_tokens.append(token)
                    else:
                        raise ValueError(
                            "Unsupported environment assignment in --launch-command; "
                            "only allowlisted runtime variables or redacted credential variables are permitted"
                        )
                    index += 1
                elif token.startswith("-"):
                    raise ValueError(
                        "Unexpected option before executable in --launch-command"
                    )
                else:
                    if not is_safe_executable_token(token):
                        raise ValueError(
                            "Unsupported or unsafe executable token in --launch-command"
                        )
                    else:
                        candidate_name = Path(token).name.lower()
                    if candidate_name == "env":
                        sanitized_tokens.append(token)
                        index += 1
                    elif candidate_name in SHELL_WRAPPER_NAMES:
                        if index + 3 != len(raw_tokens):
                            raise ValueError(
                                f"Shell wrapper {candidate_name} in --launch-command must have exact form "
                                f"'{candidate_name} -c|-lc <quoted-command>'"
                            )
                        else:
                            wrapper_flag = raw_tokens[index + 1]
                            script_body = raw_tokens[index + 2]
                            if wrapper_flag not in (
                                "-c",
                                "-lc",
                            ) or script_body.startswith("-"):
                                raise ValueError(
                                    f"Unsupported {candidate_name} wrapper flags in --launch-command"
                                )
                            else:
                                nested_sanitized = sanitize_launch_command(
                                    script_body, nesting_depth=nesting_depth + 1
                                )
                                assert nested_sanitized is not None
                                sanitized_tokens.extend(
                                    [token, wrapper_flag, nested_sanitized]
                                )
                                index += 3
                                executable_seen = True
                                current_executable = candidate_name
                    else:
                        executable_seen = True
                        current_executable = candidate_name
                        sanitized_tokens.append(token)
                        index += 1
            elif token.startswith("-") and not token.startswith("--"):
                if token == "-H":
                    if index + 1 >= len(raw_tokens) or raw_tokens[index + 1].startswith(
                        "-"
                    ):
                        raise ValueError(
                            "Option -H requires a header value in --launch-command"
                        )
                    else:
                        sanitized_tokens.extend(["-H", "<REDACTED>"])
                        index += 2
                elif token == "-m" and current_executable.startswith("python"):
                    if index + 1 >= len(raw_tokens) or raw_tokens[index + 1].startswith(
                        "-"
                    ):
                        raise ValueError(
                            "Option -m requires a Python module name in --launch-command"
                        )
                    else:
                        module_token = raw_tokens[index + 1]
                        if not SAFE_MODULE_NAME_RE.fullmatch(
                            module_token
                        ) or contains_embedded_credential_syntax(module_token):
                            raise ValueError(
                                "Invalid Python module name in --launch-command"
                            )
                        else:
                            sanitized_tokens.extend(["-m", module_token])
                            index += 2
                else:
                    raise ValueError(
                        "Unsupported or ambiguous short option in --launch-command; "
                        "use explicit long --flag options"
                    )
            elif token.startswith("--") and "=" in token:
                flag_part, value_part = token.split("=", 1)
                normalized_flag = flag_part.lstrip("-").lower()
                if normalized_flag in BLOCKED_LONG_CODE_FLAGS:
                    raise ValueError(
                        "Inline code option is not permitted in --launch-command"
                    )
                elif "." in normalized_flag and not is_safe_value_flag(flag_part):
                    if (
                        is_sensitive_cli_flag(flag_part)
                        and is_well_formed_sensitive_flag_name(flag_part)
                        and (
                            normalized_flag.split(".", 1)[0]
                            in SAFE_DOTTED_CONFIG_PREFIXES
                        )
                    ):
                        sanitized_tokens.append(f"{flag_part}=<REDACTED>")
                        index += 1
                        continue
                    else:
                        raise ValueError(
                            "Unsupported dotted option in --launch-command"
                        )
                elif is_sensitive_cli_flag(flag_part):
                    if not is_well_formed_sensitive_flag_name(flag_part):
                        raise ValueError(
                            "Malformed sensitive option name in --launch-command"
                        )
                    else:
                        sanitized_tokens.append(f"{flag_part}=<REDACTED>")
                elif is_safe_value_flag(flag_part) and is_safe_flag_scalar_value(
                    flag_part, value_part
                ):
                    sanitized_tokens.append(token)
                else:
                    raise ValueError(
                        "Unsupported option or non-scalar value in --launch-command"
                    )
                index += 1
            elif token.startswith("--"):
                normalized_flag = token.lstrip("-").lower()
                if normalized_flag in BLOCKED_LONG_CODE_FLAGS:
                    raise ValueError(
                        "Inline code option is not permitted in --launch-command"
                    )
                elif "." in normalized_flag and not is_safe_value_flag(token):
                    if (
                        is_sensitive_cli_flag(token)
                        and is_well_formed_sensitive_flag_name(token)
                        and (
                            normalized_flag.split(".", 1)[0]
                            in SAFE_DOTTED_CONFIG_PREFIXES
                        )
                    ):
                        if index + 1 >= len(raw_tokens) or raw_tokens[
                            index + 1
                        ].startswith("-"):
                            raise ValueError(
                                "Sensitive option requires a non-flag value in --launch-command"
                            )
                        else:
                            sanitized_tokens.extend([token, "<REDACTED>"])
                            index += 2
                            continue
                    else:
                        raise ValueError(
                            "Unsupported dotted option in --launch-command"
                        )
                elif is_sensitive_cli_flag(token):
                    if not is_well_formed_sensitive_flag_name(token):
                        raise ValueError(
                            "Malformed sensitive option name in --launch-command"
                        )
                    elif index + 1 >= len(raw_tokens) or raw_tokens[
                        index + 1
                    ].startswith("-"):
                        raise ValueError(
                            "Sensitive option requires a non-flag value in --launch-command"
                        )
                    else:
                        sanitized_tokens.extend([token, "<REDACTED>"])
                        index += 2
                elif normalized_flag in {
                    flag.lstrip("-") for flag in SAFE_BOOLEAN_FLAGS
                }:
                    sanitized_tokens.append(token)
                    index += 1
                elif is_safe_value_flag(token):
                    if index + 1 >= len(raw_tokens) or raw_tokens[index + 1].startswith(
                        "-"
                    ):
                        raise ValueError(
                            "Allowlisted option requires a scalar value in --launch-command"
                        )
                    else:
                        val_tok = raw_tokens[index + 1]
                        if not is_safe_flag_scalar_value(token, val_tok):
                            raise ValueError(
                                "Unsafe or non-scalar option value in --launch-command"
                            )
                        else:
                            sanitized_tokens.extend([token, val_tok])
                            index += 2
                else:
                    raise ValueError("Unsupported CLI option in --launch-command")
            elif has_url_userinfo(token) or contains_embedded_credential_syntax(token):
                raise ValueError(
                    "Embedded credentials are not permitted in --launch-command"
                )
            else:
                if token in SAFE_SUBCOMMANDS:
                    sanitized_tokens.append(token)
                    index += 1
                elif is_safe_executable_token(token):
                    candidate_name = Path(token).name.lower()
                    if candidate_name in SHELL_WRAPPER_NAMES:
                        if index + 3 != len(raw_tokens):
                            raise ValueError(
                                f"Shell wrapper {candidate_name} in --launch-command must have exact form "
                                f"'{candidate_name} -c|-lc <quoted-command>'"
                            )
                        else:
                            wrapper_flag = raw_tokens[index + 1]
                            script_body = raw_tokens[index + 2]
                            if wrapper_flag not in (
                                "-c",
                                "-lc",
                            ) or script_body.startswith("-"):
                                raise ValueError(
                                    f"Unsupported {candidate_name} wrapper flags in --launch-command"
                                )
                            else:
                                nested_sanitized = sanitize_launch_command(
                                    script_body, nesting_depth=nesting_depth + 1
                                )
                                assert nested_sanitized is not None
                                sanitized_tokens.extend(
                                    [token, wrapper_flag, nested_sanitized]
                                )
                                index += 3
                                current_executable = candidate_name
                    else:
                        if candidate_name.startswith("python"):
                            current_executable = candidate_name
                        else:
                            pass
                        sanitized_tokens.append(token)
                        index += 1
                else:
                    raise ValueError(
                        "Unsupported positional argument in --launch-command"
                    )
        if not executable_seen:
            raise ValueError("Missing command executable in --launch-command")
        else:
            return shlex.join(sanitized_tokens)


def compute_selected_scenarios_sha256(
    scenarios: list[DuplexMpeScenarioRecord],
) -> str:
    canonical_rows = [
        {
            "scenario_id": scenario.scenario_id,
            "pair_id": scenario.pair_id,
            "addressing": scenario.addressing,
            "trigger_index": scenario.trigger_index,
            "supporting_indices": scenario.supporting_indices,
            "gold_answer": scenario.gold_answer,
            "turns": [
                {
                    "turn_index": turn.turn_index,
                    "event_type": turn.event_type,
                    "speaker": turn.speaker,
                    "text": turn.text,
                    "audio_path": turn.audio_path,
                }
                for turn in scenario.turns
            ],
        }
        for scenario in scenarios
    ]
    return hashlib.sha256(
        json.dumps(canonical_rows, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()


def compute_selected_turn_audio_sha256(
    dataset_root: Path,
    scenarios: list[DuplexMpeScenarioRecord],
) -> str:
    unique_relative_paths = sorted(
        {turn.audio_path for scenario in scenarios for turn in scenario.turns}
    )
    asset_digests = {
        relative_path: compute_file_sha256(
            resolve_dataset_audio_path(dataset_root, relative_path)
        )
        for relative_path in unique_relative_paths
    }
    return hashlib.sha256(
        json.dumps(asset_digests, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def build_scenario_bindings(
    dataset_root: Path,
    scenarios: list[DuplexMpeScenarioRecord],
) -> dict[str, JsonValue]:
    return {
        scenario.scenario_id: {
            "scenario_sha256": compute_selected_scenarios_sha256([scenario]),
            "turn_audio_sha256": compute_selected_turn_audio_sha256(
                dataset_root, [scenario]
            ),
        }
        for scenario in scenarios
    }


def verify_saved_dataset_bindings(
    *,
    dataset_root: Path,
    scenarios_by_id: dict[str, DuplexMpeScenarioRecord],
    reports_by_id: dict[str, dict[str, JsonValue]],
    saved_run_manifest: dict[str, JsonValue],
) -> None:
    saved_dataset_sha256 = saved_run_manifest.get("dataset_sha256")
    if isinstance(saved_dataset_sha256, str) and saved_dataset_sha256:
        dataset_json_path = resolve_dataset_audio_path(dataset_root, "dataset.json")
        current_dataset_sha256 = (
            compute_file_sha256(dataset_json_path)
            if dataset_json_path.is_file()
            else None
        )
        if current_dataset_sha256 != saved_dataset_sha256:
            raise ValueError(
                "Current dataset.json SHA-256 differs from recorded run.json dataset_sha256"
            )
        else:
            pass
    else:
        pass

    saved_preamble_sha256 = saved_run_manifest.get("preamble_sha256")
    if isinstance(saved_preamble_sha256, str) and saved_preamble_sha256:
        preamble_path = resolve_dataset_audio_path(
            dataset_root, DUPLEX_MPE_DUTY_PREAMBLE_RELATIVE_PATH
        )
        current_preamble_sha256 = (
            compute_file_sha256(preamble_path) if preamble_path.is_file() else None
        )
        if current_preamble_sha256 != saved_preamble_sha256:
            raise ValueError(
                "Current duty preamble WAV SHA-256 differs from recorded run.json preamble_sha256"
            )
        else:
            pass
    else:
        pass

    saved_manifest_digests = saved_run_manifest.get("manifest_sha256")
    if isinstance(saved_manifest_digests, dict):
        selected_conditions = {
            scenarios_by_id[scenario_id].addressing
            for scenario_id in reports_by_id
            if scenario_id in scenarios_by_id
        }
        for condition_name in sorted(selected_conditions):
            expected_manifest_digest = saved_manifest_digests.get(condition_name)
            if isinstance(expected_manifest_digest, str) and expected_manifest_digest:
                manifest_path = resolve_dataset_audio_path(
                    dataset_root, f"manifests/{condition_name}.json"
                )
                actual_manifest_digest = (
                    compute_file_sha256(manifest_path)
                    if manifest_path.is_file()
                    else None
                )
                if actual_manifest_digest != expected_manifest_digest:
                    raise ValueError(
                        f"Current manifests/{condition_name}.json SHA-256 differs from recorded run.json"
                    )
                else:
                    pass
            else:
                pass
    else:
        pass

    saved_scenario_bindings = saved_run_manifest.get("scenario_bindings")
    if isinstance(saved_scenario_bindings, dict):
        for scenario_id in sorted(reports_by_id):
            scenario_record = scenarios_by_id[scenario_id]
            recorded_binding = saved_scenario_bindings.get(scenario_id)
            if not isinstance(recorded_binding, dict):
                raise ValueError(
                    f"Scenario {scenario_id} is missing from recorded run.json scenario_bindings"
                )
            else:
                expected_scenario_sha = recorded_binding.get("scenario_sha256")
                expected_audio_sha = recorded_binding.get("turn_audio_sha256")
                if expected_scenario_sha != compute_selected_scenarios_sha256(
                    [scenario_record]
                ):
                    raise ValueError(
                        f"Scenario manifest fields for {scenario_id} differ from recorded run.json"
                    )
                elif expected_audio_sha != compute_selected_turn_audio_sha256(
                    dataset_root, [scenario_record]
                ):
                    raise ValueError(
                        f"Referenced human-turn audio for {scenario_id} differs from recorded run.json"
                    )
                else:
                    pass
    else:
        pass


def verify_retained_assistant_audio(
    reports_by_id: dict[str, dict[str, JsonValue]],
    audio_directory: Path,
) -> None:
    for scenario_id, report in reports_by_id.items():
        validate_safe_identifier(scenario_id, "scenario_id")
        expected_sha256 = report.get("assistant_audio_sha256")
        if not isinstance(expected_sha256, str) or not expected_sha256:
            continue
        else:
            audio_path = resolve_retained_scenario_audio_path(
                audio_directory, scenario_id, must_exist=True
            )
            if compute_file_sha256(audio_path) != expected_sha256:
                raise ValueError(
                    f"Retained assistant audio modified on disk for {scenario_id}"
                )
            else:
                pass


@dataclass(frozen=True, kw_only=True)
class GeneratePhaseResult:
    reports_by_id: dict[str, dict[str, JsonValue]]
    failed_scenarios: dict[str, str] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.reports_by_id)

    def __contains__(self, scenario_id: str) -> bool:
        return scenario_id in self.reports_by_id

    def __getitem__(self, scenario_id: str) -> dict[str, JsonValue]:
        return self.reports_by_id[scenario_id]


def load_validated_failure_ledger(ledger_path: Path) -> dict[str, str]:
    if not ledger_path.is_file():
        return {}
    else:
        raw_ledger = read_json_file(ledger_path)
        if not isinstance(raw_ledger, dict):
            raise ValueError(f"{ledger_path} must contain a JSON object")
        else:
            validated_ledger: dict[str, str] = {}
            for entry_key, entry_value in raw_ledger.items():
                if not isinstance(entry_key, str) or not isinstance(entry_value, str):
                    raise ValueError(
                        f"{ledger_path} entries must map string keys to string reasons"
                    )
                else:
                    validated_ledger[entry_key] = entry_value
            return validated_ledger


def is_compatible_saved_run_manifest(
    saved_manifest: dict[str, JsonValue],
    current_manifest: dict[str, JsonValue],
) -> bool:
    if set(saved_manifest) != set(current_manifest):
        return False
    else:
        for field_name, current_value in current_manifest.items():
            saved_value = saved_manifest.get(field_name)
            if field_name == "session_timeout_seconds":
                if (
                    not isinstance(saved_value, (int, float))
                    or type(saved_value) is bool
                    or not isinstance(current_value, (int, float))
                    or type(current_value) is bool
                    or float(current_value) < float(saved_value)
                ):
                    return False
                else:
                    pass
            elif saved_value != current_value:
                return False
            else:
                pass
        return True


def load_saved_reports(report_path: Path) -> dict[str, dict[str, JsonValue]]:
    if not report_path.is_file():
        return {}
    else:
        raw_reports = read_json_file(report_path)
        if not isinstance(raw_reports, dict):
            raise ValueError(f"{report_path} must contain a JSON object")
        else:
            typed_reports: dict[str, dict[str, JsonValue]] = {}
            for scenario_id, entry in raw_reports.items():
                validate_safe_identifier(scenario_id, "scenario_id")
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"Malformed scenario report for {scenario_id} in {report_path}"
                    )
                else:
                    typed_reports[scenario_id] = entry
            return typed_reports


async def run_generate_phase(
    config: DuplexMpeEvalConfig,
    scenarios: list[DuplexMpeScenarioRecord],
) -> GeneratePhaseResult:
    assert config.base_url is not None
    current_vad_version = silero_package_version()
    if (
        config.strict_reference_vad_version
        and current_vad_version != REFERENCE_SILERO_VAD_VERSION
    ):
        raise ValueError(
            f"--strict-reference-vad-version requires silero-vad=={REFERENCE_SILERO_VAD_VERSION}, "
            f"found {current_vad_version}"
        )
    else:
        pass
    dataset_root = config.dataset_root.expanduser().resolve()
    endpoint_spec = await negotiate_realtime_endpoint(
        config.base_url,
        protocol_selection=config.protocol,
        input_sample_rate_hz=config.input_sample_rate_hz,
        output_sample_rate_hz=config.output_sample_rate_hz,
        trust_env=config.trust_env,
    )
    effective_model = (
        config.model
        if config.model is not None
        else (endpoint_spec.reported_model or "unknown")
    )
    seeds = load_reference_seeds()
    for scenario in scenarios:
        resolve_scenario_seed(scenario.scenario_id, seeds)

    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    run_manifest_path = output_dir / "run.json"
    report_path = output_dir / "report.json"
    failures_path = output_dir / "failures.json"
    audio_directory = output_dir / "audio"

    if report_path.is_file() and not run_manifest_path.is_file():
        raise ValueError(
            f"Existing {report_path} has no {run_manifest_path} manifest; "
            "choose a new --output-dir"
        )
    else:
        pass

    manifest_digests: dict[str, JsonValue] = {}
    for condition_name in ("explicit", "implicit"):
        condition_path = resolve_dataset_audio_path(
            dataset_root, f"manifests/{condition_name}.json"
        )
        if condition_path.is_file():
            manifest_digests[condition_name] = compute_file_sha256(condition_path)
        else:
            pass
    dataset_json_path = resolve_dataset_audio_path(dataset_root, "dataset.json")
    dataset_sha256 = (
        compute_file_sha256(dataset_json_path) if dataset_json_path.is_file() else None
    )
    preamble_path = resolve_dataset_audio_path(
        dataset_root, DUPLEX_MPE_DUTY_PREAMBLE_RELATIVE_PATH
    )
    preamble_sha256 = (
        compute_file_sha256(preamble_path) if preamble_path.is_file() else None
    )
    endpoint_websocket_sha256 = hashlib.sha256(
        endpoint_spec.websocket_url.encode("utf-8")
    ).hexdigest()

    run_manifest: dict[str, JsonValue] = {
        "model": effective_model,
        "reported_model": endpoint_spec.reported_model,
        "model_revision": config.model_revision,
        "launch_command": sanitize_launch_command(config.launch_command),
        "base_url": sanitize_endpoint_origin(config.base_url),
        "endpoint_websocket_sha256": endpoint_websocket_sha256,
        "protocol": endpoint_spec.protocol,
        "input_sample_rate_hz": endpoint_spec.input_sample_rate_hz,
        "output_sample_rate_hz": endpoint_spec.output_sample_rate_hz,
        "turn_detection": endpoint_spec.turn_detection,
        "max_concurrency": config.max_concurrency,
        "warmup_sessions": config.warmup_sessions,
        "session_timeout_seconds": config.session_timeout_seconds,
        "addressing": config.addressing,
        "mini": config.mini,
        "max_scenarios": config.max_scenarios,
        "evaluation_clock": "realtime_wall_clock",
        "seed_policy": SEED_POLICY_METADATA_ONLY,
        "silero_vad_version": current_vad_version,
        "code_fingerprint": compute_protocol_fingerprint(),
        "dataset_sha256": dataset_sha256,
        "preamble_sha256": preamble_sha256,
        "manifest_sha256": manifest_digests,
        "selected_scenarios_sha256": compute_selected_scenarios_sha256(scenarios),
        "selected_turn_audio_sha256": compute_selected_turn_audio_sha256(
            dataset_root, scenarios
        ),
        "scenario_bindings": build_scenario_bindings(dataset_root, scenarios),
        "seed_map_sha256": hashlib.sha256(
            json.dumps(seeds, sort_keys=True).encode("utf-8")
        ).hexdigest(),
    }
    if run_manifest_path.is_file():
        existing_manifest = read_json_file(run_manifest_path)
        if not isinstance(
            existing_manifest, dict
        ) or not is_compatible_saved_run_manifest(existing_manifest, run_manifest):
            raise ValueError(
                f"Run configuration in {run_manifest_path} changed; choose a new --output-dir"
            )
        elif existing_manifest != run_manifest:
            write_json_atomic(run_manifest_path, run_manifest)
        else:
            pass
    else:
        write_json_atomic(run_manifest_path, run_manifest)

    saved_reports = load_saved_reports(report_path)
    pending_scenarios: list[DuplexMpeScenarioRecord] = []
    for scenario in scenarios:
        if scenario.scenario_id in saved_reports:
            recorded_sha256 = saved_reports[scenario.scenario_id].get(
                "assistant_audio_sha256"
            )
            try:
                wav_path = resolve_retained_scenario_audio_path(
                    audio_directory, scenario.scenario_id, must_exist=True
                )
            except FileNotFoundError as exc:
                raise ValueError(
                    f"Saved audio missing or modified for {scenario.scenario_id}"
                ) from exc
            if compute_file_sha256(wav_path) != recorded_sha256:
                raise ValueError(
                    f"Saved audio missing or modified for {scenario.scenario_id}"
                )
            else:
                continue
        else:
            pending_scenarios.append(scenario)

    if not pending_scenarios:
        if failures_path.is_file():
            failures_path.unlink()
        else:
            pass
        logger.info(
            f"All {len(scenarios)} scenarios already recorded in {report_path}."
        )
        return GeneratePhaseResult(reports_by_id=saved_reports, failed_scenarios={})
    else:
        pass

    if config.warmup_sessions > 0:
        warmup_directory = output_dir / "warmup"
        warmup_scenario = pending_scenarios[0]
        logger.info(
            f"Running {config.warmup_sessions} unscored warmup session(s) on "
            f"{warmup_scenario.scenario_id} ..."
        )
        warmup_semaphore = asyncio.Semaphore(config.max_concurrency)

        async def run_warmup_one(warmup_index: int) -> None:
            async with warmup_semaphore:
                await run_realtime_scenario(
                    warmup_scenario,
                    dataset_root=dataset_root,
                    endpoint_spec=endpoint_spec,
                    output_audio_path=warmup_directory / f"warmup_{warmup_index}.wav",
                    session_timeout_seconds=config.session_timeout_seconds,
                    seed_mapping=seeds,
                )

        await asyncio.gather(
            *(run_warmup_one(index) for index in range(config.warmup_sessions))
        )
    else:
        pass

    semaphore = asyncio.Semaphore(config.max_concurrency)
    save_lock = asyncio.Lock()
    existing_failure_ledger = load_validated_failure_ledger(failures_path)
    failed_scenarios: dict[str, str] = {
        scenario_key: reason
        for scenario_key, reason in existing_failure_ledger.items()
        if scenario_key not in saved_reports
    }

    async def record_scenario(scenario: DuplexMpeScenarioRecord) -> None:
        async with semaphore:
            wav_path = resolve_retained_scenario_audio_path(
                audio_directory, scenario.scenario_id, must_exist=False
            )
            try:
                scenario_report = await run_realtime_scenario(
                    scenario,
                    dataset_root=dataset_root,
                    endpoint_spec=endpoint_spec,
                    output_audio_path=wav_path,
                    session_timeout_seconds=config.session_timeout_seconds,
                    seed_mapping=seeds,
                )
            except (
                OSError,
                ProtocolSessionError,
                RuntimeError,
                TimeoutError,
                ValueError,
                httpx.HTTPError,
                websockets.WebSocketException,
            ) as exc:
                failure_reason = (
                    str(exc)
                    if isinstance(
                        exc,
                        (ProtocolSessionError, RuntimeError, TimeoutError, ValueError),
                    )
                    else summarize_transport_exception(exc)
                )
                async with save_lock:
                    failed_scenarios[scenario.scenario_id] = failure_reason
                    await asyncio.to_thread(
                        write_json_atomic, failures_path, dict(failed_scenarios)
                    )
                    logger.error(
                        f"Scenario {scenario.scenario_id} failed ({failure_reason}); "
                        "remaining pending for resume"
                    )
                return

            async with save_lock:
                saved_reports[scenario.scenario_id] = scenario_report
                failed_scenarios.pop(scenario.scenario_id, None)
                await asyncio.to_thread(
                    write_json_atomic, report_path, dict(saved_reports)
                )
                if failed_scenarios:
                    await asyncio.to_thread(
                        write_json_atomic, failures_path, dict(failed_scenarios)
                    )
                elif failures_path.is_file():
                    await asyncio.to_thread(failures_path.unlink)
                else:
                    pass
                target_block = scenario_report.get("T")
                outcome_label = (
                    target_block.get("outcome")
                    if isinstance(target_block, dict)
                    else None
                )
                logger.info(
                    f"Recorded {scenario.scenario_id}: T={outcome_label} "
                    f"({len(saved_reports)}/{len(scenarios)})"
                )

    await asyncio.gather(*(record_scenario(scenario) for scenario in pending_scenarios))
    if not failed_scenarios and failures_path.is_file():
        failures_path.unlink()
    else:
        pass
    return GeneratePhaseResult(
        reports_by_id=saved_reports, failed_scenarios=failed_scenarios
    )


async def run_duplex_mpe_evaluation(
    config: DuplexMpeEvalConfig,
) -> dict[str, JsonValue]:
    """Execute the selected Duplex-MPE phase(s) and write summary artifacts."""
    dataset_root = config.dataset_root.expanduser().resolve()
    output_dir = config.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_identity = inspect_duplex_mpe_dataset(dataset_root)
    scenarios = load_duplex_mpe_scenarios(
        dataset_root,
        addressing=config.addressing,
        mini=config.mini,
        max_scenarios=config.max_scenarios,
    )
    if not scenarios:
        raise ValueError("No Duplex-MPE scenarios selected")
    else:
        pass
    scenarios_by_id = {scenario.scenario_id: scenario for scenario in scenarios}

    judge_config: JudgeEndpointConfig | None = (
        load_judge_endpoint_config(config.judge_config_path)
        if config.judge_config_path is not None
        else None
    )

    run_manifest_path = output_dir / "run.json"
    report_path = output_dir / "report.json"
    failures_path = output_dir / "failures.json"
    judge_failures_path = output_dir / "judge_failures.json"
    tasks_path = output_dir / "tasks.json"
    transcribed_path = output_dir / "transcribed.json"
    labels_path = output_dir / "labels.json"
    audio_directory = output_dir / "audio"

    failed_scenarios: dict[str, str] = {}
    failed_semantic_task_keys: tuple[str, ...] = ()
    allowed_unselected_task_keys: frozenset[str] = frozenset()
    if config.phase in ("all", "generate"):
        generate_result = await run_generate_phase(config, scenarios)
        reports_by_id = generate_result.reports_by_id
        failed_scenarios = dict(generate_result.failed_scenarios)
    else:
        raw_failures = load_validated_failure_ledger(failures_path)
        failed_scenarios = {
            scenario_key: reason
            for scenario_key, reason in raw_failures.items()
            if scenario_key in scenarios_by_id
        }
        all_saved = load_saved_reports(report_path)
        reports_by_id = {
            scenario_id: all_saved[scenario_id]
            for scenario_id in scenarios_by_id
            if scenario_id in all_saved
        }
        if not reports_by_id:
            raise FileNotFoundError(
                f"No matching scenario reports found in {report_path}"
            )
        else:
            pass
        unselected_ids = set(all_saved) - set(reports_by_id)
        if unselected_ids:
            all_dataset_scenarios = load_duplex_mpe_scenarios(
                dataset_root, addressing="all"
            )
            all_scenarios_by_id = {
                row.scenario_id: row for row in all_dataset_scenarios
            }
            unknown_saved_ids = sorted(unselected_ids - set(all_scenarios_by_id))
            if unknown_saved_ids:
                raise ValueError(
                    f"Unknown scenario_id in {report_path}: {unknown_saved_ids[0]}"
                )
            else:
                unselected_reports = {
                    scenario_id: all_saved[scenario_id]
                    for scenario_id in unselected_ids
                }
                unselected_tasks = build_semantic_tasks(
                    unselected_reports, all_scenarios_by_id
                )
                allowed_unselected_task_keys = frozenset(
                    semantic_task_key(task) for task in unselected_tasks
                )
        else:
            pass

    saved_run_manifest: dict[str, JsonValue] = {}
    if run_manifest_path.is_file():
        raw_run_manifest = read_json_file(run_manifest_path)
        if isinstance(raw_run_manifest, dict):
            saved_run_manifest = raw_run_manifest
        else:
            raise ValueError(f"{run_manifest_path} must contain a JSON object")
    else:
        pass
    saved_model = saved_run_manifest.get("model")
    if (
        config.model is not None
        and isinstance(saved_model, str)
        and saved_model
        and config.model != saved_model
    ):
        raise ValueError(
            f"Requested --model={config.model!r} conflicts with saved run.json model={saved_model!r}"
        )
    else:
        pass
    saved_revision = saved_run_manifest.get("model_revision")
    if (
        config.model_revision is not None
        and saved_revision is not None
        and config.model_revision != saved_revision
    ):
        raise ValueError(
            f"Requested --model-revision={config.model_revision!r} conflicts with saved run.json"
        )
    else:
        pass
    effective_model = (
        config.model
        if config.model is not None
        else (
            saved_model if isinstance(saved_model, str) and saved_model else "unknown"
        )
    )
    effective_model_revision = (
        config.model_revision
        if config.model_revision is not None
        else (saved_revision if isinstance(saved_revision, str) else None)
    )
    sanitized_cli_launch = sanitize_launch_command(config.launch_command)
    saved_launch_command = saved_run_manifest.get("launch_command")
    if (
        sanitized_cli_launch is not None
        and saved_launch_command is not None
        and sanitized_cli_launch != saved_launch_command
    ):
        raise ValueError("Requested --launch-command conflicts with saved run.json")
    else:
        effective_launch_command = (
            sanitized_cli_launch
            if sanitized_cli_launch is not None
            else (
                saved_launch_command if isinstance(saved_launch_command, str) else None
            )
        )

    missing_selected_scenarios = sorted(set(scenarios_by_id) - set(reports_by_id))

    if config.phase == "generate":
        is_generate_complete = (not missing_selected_scenarios) and (
            not failed_scenarios
        )
        return {
            "status": "complete" if is_generate_complete else "incomplete",
            "phase": "generate",
            "scenarios_selected": len(scenarios_by_id),
            "scenarios_recorded": len(reports_by_id),
            "missing_scenarios": missing_selected_scenarios,
            "failed_scenarios": dict(failed_scenarios),
            "report_path": str(report_path),
        }
    elif not reports_by_id:
        raise RuntimeError(
            f"All {len(scenarios_by_id)} selected scenarios failed during generation; "
            f"inspect {failures_path}"
        )
    elif config.phase not in ("all", "generate") and saved_run_manifest:
        verify_saved_dataset_bindings(
            dataset_root=dataset_root,
            scenarios_by_id=scenarios_by_id,
            reports_by_id=reports_by_id,
            saved_run_manifest=saved_run_manifest,
        )
    else:
        pass

    semantic_tasks = build_semantic_tasks(reports_by_id, scenarios_by_id)
    if (not allowed_unselected_task_keys and config.phase != "score") or (
        not tasks_path.is_file()
    ):
        write_json_atomic(tasks_path, semantic_tasks)
    else:
        pass

    resolved_asr_model = (
        config.asr_model if config.asr_model is not None else DEFAULT_ASR_MODEL_ID
    )
    expected_offline_asr_endpoint_sha256 = (
        hashlib.sha256(
            resolve_transcriptions_url(config.asr_base_url).encode("utf-8")
        ).hexdigest()
        if config.asr_base_url is not None
        else None
    )
    expected_offline_asr_model = (
        config.asr_model
        if config.asr_model is not None
        else (DEFAULT_ASR_MODEL_ID if config.asr_base_url is not None else None)
    )
    has_transcribed_cache = False
    should_run_asr = config.phase == "asr" or (
        config.phase == "all" and config.asr_base_url is not None
    )
    if should_run_asr:
        assert config.asr_base_url is not None
        transcribed_tasks = await transcribe_tasks_via_http(
            semantic_tasks,
            audio_directory=audio_directory,
            output_path=transcribed_path,
            asr_base_url=config.asr_base_url,
            asr_model_id=resolved_asr_model,
            api_key_env=config.asr_api_key_env,
            max_concurrency=config.asr_max_concurrency,
            trust_env=config.trust_env,
            allowed_unselected_task_keys=allowed_unselected_task_keys,
        )
        has_transcribed_cache = True
        if config.phase == "asr":
            is_asr_complete = (not missing_selected_scenarios) and (
                not failed_scenarios
            )
            return {
                "status": "complete" if is_asr_complete else "incomplete",
                "phase": "asr",
                "tasks_transcribed": len(transcribed_tasks),
                "missing_scenarios": missing_selected_scenarios,
                "failed_scenarios": dict(failed_scenarios),
                "transcribed_path": str(transcribed_path),
            }
        else:
            pass
    elif transcribed_path.is_file():
        raw_transcribed = read_json_file(transcribed_path)
        if not isinstance(raw_transcribed, list):
            raise ValueError(f"{transcribed_path} must contain a JSON array")
        else:
            has_transcribed_cache = True
            all_transcribed_rows = [
                row for row in raw_transcribed if isinstance(row, dict)
            ]
            selected_transcribed_map = validate_transcribed_against_tasks(
                semantic_tasks,
                all_transcribed_rows,
                expected_asr_model=expected_offline_asr_model,
                expected_asr_endpoint_sha256=expected_offline_asr_endpoint_sha256,
                allowed_unselected_task_keys=allowed_unselected_task_keys,
            )
            transcribed_tasks = list(selected_transcribed_map.values())
    else:
        transcribed_tasks = []

    should_run_judge = config.phase == "judge" or (
        config.phase == "all" and judge_config is not None
    )
    expected_selected_task_keys: frozenset[str] = frozenset(
        semantic_task_key(task) for task in semantic_tasks
    )
    if should_run_judge:
        if not has_transcribed_cache and semantic_tasks:
            raise RuntimeError(
                f"Cannot run semantic judge without ASR transcripts in {transcribed_path}"
            )
        else:
            assert judge_config is not None
            existing_judge_failures = load_validated_failure_ledger(judge_failures_path)
            judge_result = await judge_tasks_via_http(
                transcribed_tasks,
                judge_config=judge_config,
                output_path=labels_path,
                trust_env=config.trust_env,
                allowed_unselected_task_keys=allowed_unselected_task_keys,
            )
            verdicts = list(judge_result.verdicts)
            resolved_verdict_keys = {semantic_task_key(row) for row in verdicts}
            preserved_unselected_judge_failures = {
                task_key: reason
                for task_key, reason in existing_judge_failures.items()
                if task_key not in expected_selected_task_keys
                and task_key not in resolved_verdict_keys
            }
            merged_judge_failures = {
                **preserved_unselected_judge_failures,
                **judge_result.failure_reasons,
            }
            if merged_judge_failures:
                write_json_atomic(judge_failures_path, merged_judge_failures)
            elif judge_failures_path.is_file():
                judge_failures_path.unlink()
            else:
                pass
            failed_semantic_task_keys = judge_result.failed_task_keys
        if config.phase == "judge":
            unresolved_selected_task_keys = sorted(
                expected_selected_task_keys
                - {semantic_task_key(row) for row in verdicts}
            )
            is_judge_complete = (
                (not missing_selected_scenarios)
                and (not failed_scenarios)
                and (not failed_semantic_task_keys)
                and (not unresolved_selected_task_keys)
            )
            return {
                "status": "complete" if is_judge_complete else "incomplete",
                "phase": "judge",
                "verdicts_written": len(verdicts),
                "missing_scenarios": missing_selected_scenarios,
                "failed_scenarios": dict(failed_scenarios),
                "failed_semantic_tasks": len(failed_semantic_task_keys),
                "failed_task_keys": list(failed_semantic_task_keys),
                "labels_path": str(labels_path),
            }
        else:
            pass
    elif labels_path.is_file():
        raw_labels = read_json_file(labels_path)
        if not isinstance(raw_labels, list):
            raise ValueError(f"{labels_path} must contain a JSON array")
        else:
            raw_verdict_rows = [row for row in raw_labels if isinstance(row, dict)]
            if raw_verdict_rows:
                if not has_transcribed_cache:
                    raise ValueError(
                        f"Cannot verify {labels_path} without {transcribed_path}"
                    )
                else:
                    indexed_transcribed = validate_transcribed_against_tasks(
                        semantic_tasks, transcribed_tasks
                    )
                    expected_judge_endpoint_sha256 = (
                        hashlib.sha256(
                            resolve_chat_completions_url(judge_config.base_url).encode(
                                "utf-8"
                            )
                        ).hexdigest()
                        if judge_config is not None
                        else None
                    )
                    validated_verdicts = load_validated_existing_verdicts(
                        indexed_transcribed,
                        raw_verdict_rows,
                        judge_config=judge_config,
                        expected_endpoint_sha256=expected_judge_endpoint_sha256,
                        allowed_unselected_task_keys=allowed_unselected_task_keys,
                    )
                    verdicts = list(validated_verdicts.values())
            else:
                verdicts = []
        persisted_judge_failures = load_validated_failure_ledger(judge_failures_path)
        resolved_verdict_keys = {semantic_task_key(row) for row in verdicts}
        failed_semantic_task_keys = tuple(
            sorted(
                task_key
                for task_key in persisted_judge_failures
                if task_key in expected_selected_task_keys
                and task_key not in resolved_verdict_keys
            )
        )
    else:
        verdicts = []
        persisted_judge_failures = load_validated_failure_ledger(judge_failures_path)
        failed_semantic_task_keys = tuple(
            sorted(
                task_key
                for task_key in persisted_judge_failures
                if task_key in expected_selected_task_keys
            )
        )

    if transcribed_tasks:
        validated_transcribed = validate_transcribed_against_tasks(
            semantic_tasks, transcribed_tasks
        )
        verdict_keys_present: set[tuple[str, int, str]] = {
            (str(row["scenario_id"]), int(row["turn_index"]), str(row["kind"]))
            for row in verdicts
            if isinstance(row.get("scenario_id"), str)
            and type(row.get("turn_index")) is int
            and isinstance(row.get("kind"), str)
        }
        for task in validated_transcribed.values():
            task_tuple = (
                str(task["scenario_id"]),
                int(task["turn_index"]),
                str(task["kind"]),
            )
            if (
                not str(task.get("hyp", "")).strip()
                and task_tuple not in verdict_keys_present
            ):
                verdicts.append(make_deterministic_empty_asr_verdict(task))
                verdict_keys_present.add(task_tuple)
            else:
                pass
    else:
        pass

    if config.verify_retained_audio:
        verify_retained_assistant_audio(reports_by_id, audio_directory)
    else:
        pass

    validate_scenario_reports(
        reports_by_id,
        scenarios_by_id,
        strict_reference_vad_version=config.strict_reference_vad_version,
    )
    first_report = next(iter(reports_by_id.values()))
    first_aed = first_report.get("aed_config")
    recorded_vad_version = (
        first_aed.get("silero_vad_version") if isinstance(first_aed, dict) else None
    )
    observed_report_protocols = {
        str(report.get("realtime_protocol"))
        for report in reports_by_id.values()
        if isinstance(report.get("realtime_protocol"), str)
        and report.get("realtime_protocol")
    }
    valid_protocols = {"native", "turn_based", "legacy"}
    unknown_protocols = observed_report_protocols - valid_protocols
    if unknown_protocols:
        raise ValueError(
            f"Unsupported realtime_protocol values in {report_path}: {sorted(unknown_protocols)}"
        )
    elif len(observed_report_protocols) > 1:
        raise ValueError(
            f"Mixed realtime_protocol values in {report_path}: {sorted(observed_report_protocols)}"
        )
    else:
        report_protocol = (
            next(iter(observed_report_protocols))
            if len(observed_report_protocols) == 1
            else None
        )
    raw_manifest_protocol = saved_run_manifest.get("protocol")
    manifest_protocol: str | None = None
    if raw_manifest_protocol is not None:
        if (
            not isinstance(raw_manifest_protocol, str)
            or raw_manifest_protocol not in valid_protocols
        ):
            raise ValueError(
                f"Unsupported protocol={raw_manifest_protocol!r} in {run_manifest_path}"
            )
        else:
            manifest_protocol = raw_manifest_protocol
    else:
        pass
    if (
        manifest_protocol is not None
        and report_protocol is not None
        and manifest_protocol != report_protocol
    ):
        raise ValueError(
            f"Protocol mismatch between {run_manifest_path} ({manifest_protocol!r}) "
            f"and {report_path} ({report_protocol!r})"
        )
    elif (
        config.protocol != "auto"
        and manifest_protocol is not None
        and config.protocol != manifest_protocol
    ):
        raise ValueError(
            f"Requested --protocol={config.protocol!r} conflicts with {run_manifest_path} "
            f"({manifest_protocol!r})"
        )
    elif (
        config.protocol != "auto"
        and report_protocol is not None
        and config.protocol != report_protocol
    ):
        raise ValueError(
            f"Requested --protocol={config.protocol!r} conflicts with {report_path} "
            f"({report_protocol!r})"
        )
    elif manifest_protocol is not None:
        effective_protocol = manifest_protocol
    elif report_protocol is not None:
        effective_protocol = report_protocol
    else:
        effective_protocol = config.protocol if config.protocol != "auto" else "native"
    effective_turn_detection = saved_run_manifest.get("turn_detection")
    is_probe_only = effective_protocol != "native"
    evaluation_mode = (
        "unscored_turn_based_probe" if is_probe_only else "scored_full_duplex"
    )
    scored_metrics = score_scenario_reports(
        reports_by_id,
        scenarios_by_id,
        verdicts,
        probe_only=is_probe_only,
    )
    missing_scenarios = sorted(set(scenarios_by_id) - set(reports_by_id))
    pooled_row = scored_metrics.get("overall_pooled", {})
    pending_semantic_count = 0
    for metric_name in ("conditional_answer_accuracy", "silence_preservation"):
        metric_cell = pooled_row.get(metric_name)
        if isinstance(metric_cell, dict):
            raw_pending = metric_cell.get("pending")
            if type(raw_pending) is int and raw_pending > 0:
                pending_semantic_count += raw_pending
            else:
                pass
        else:
            pass
    has_pending_semantics = pending_semantic_count > 0
    resolved_selected_verdict_count = sum(
        1 for row in verdicts if semantic_task_key(row) in expected_selected_task_keys
    )

    is_run_complete = (
        (not missing_scenarios)
        and (not failed_scenarios)
        and (not has_pending_semantics)
        and (not failed_semantic_task_keys)
    )
    evaluation_input_sha256 = compute_selected_scenarios_sha256(scenarios)
    effective_base_url = (
        sanitize_endpoint_origin(config.base_url)
        if config.base_url
        else saved_run_manifest.get("base_url")
    )
    effective_concurrency = (
        saved_run_manifest.get("max_concurrency")
        if type(saved_run_manifest.get("max_concurrency")) is int
        else config.max_concurrency
    )
    effective_warmup_sessions = (
        saved_run_manifest.get("warmup_sessions")
        if type(saved_run_manifest.get("warmup_sessions")) is int
        else config.warmup_sessions
    )
    raw_saved_timeout = saved_run_manifest.get("session_timeout_seconds")
    effective_session_timeout_seconds = (
        float(raw_saved_timeout)
        if isinstance(raw_saved_timeout, (int, float))
        and type(raw_saved_timeout) is not bool
        else config.session_timeout_seconds
    )
    observed_asr_models: set[str] = {
        str(row["asr_model"])
        for row in transcribed_tasks
        if isinstance(row.get("asr_model"), str) and row["asr_model"]
    }
    if len(observed_asr_models) > 1:
        raise ValueError(
            f"Mixed asr_model values in {transcribed_path}: {sorted(observed_asr_models)}"
        )
    elif len(observed_asr_models) == 1:
        effective_asr_model: str | None = next(iter(observed_asr_models))
    elif config.asr_base_url is not None:
        effective_asr_model = resolved_asr_model
    else:
        effective_asr_model = None
    observed_pacing_lags_ms = [
        float(report["max_pacing_lag_ms"])
        for report in reports_by_id.values()
        if isinstance(report.get("max_pacing_lag_ms"), (int, float))
        and type(report.get("max_pacing_lag_ms")) is not bool
    ]
    max_observed_pacing_lag_ms = (
        round(max(observed_pacing_lags_ms), 3) if observed_pacing_lags_ms else None
    )
    inferred_judge_metadata: JsonValue = None
    if judge_config is not None:
        inferred_judge_metadata = public_judge_metadata(judge_config)
    else:
        non_empty_verdicts = [
            row for row in verdicts if row.get("error_kind") != "asr_empty"
        ]
        if non_empty_verdicts:
            sample_verdict = non_empty_verdicts[0]
            answer_decoding = next(
                (
                    row.get("judge_decoding")
                    for row in non_empty_verdicts
                    if row.get("kind") == "answer" and "judge_decoding" in row
                ),
                None,
            )
            silence_decoding = next(
                (
                    row.get("judge_decoding")
                    for row in non_empty_verdicts
                    if row.get("kind") == "silence" and "judge_decoding" in row
                ),
                None,
            )
            inferred_judge_metadata = {
                "model": sample_verdict.get("judge_model"),
                "model_revision": sample_verdict.get("judge_model_revision"),
                "enable_thinking": sample_verdict.get("judge_enable_thinking"),
                "answer_decoding": answer_decoding,
                "silence_decoding": silence_decoding,
            }
        else:
            pass
    server_declaration: dict[str, JsonValue] = {
        "base_url": effective_base_url,
        "protocol": effective_protocol,
        "evaluation_mode": evaluation_mode,
        "turn_detection": effective_turn_detection,
        "max_concurrency": effective_concurrency,
        "warmup_sessions": effective_warmup_sessions,
        "session_timeout_seconds": effective_session_timeout_seconds,
        "max_pacing_lag_ms": max_observed_pacing_lag_ms,
        "seed_policy": SEED_POLICY_METADATA_ONLY,
        "silero_vad_version": recorded_vad_version,
        "reference_vad_version_matched": (
            recorded_vad_version == REFERENCE_SILERO_VAD_VERSION
        ),
        "strict_reference_vad_version": config.strict_reference_vad_version,
        "retained_audio_verified": config.verify_retained_audio,
        "asr_base_url": (
            sanitize_endpoint_origin(config.asr_base_url)
            if config.asr_base_url
            else None
        ),
        "asr_model": effective_asr_model,
        "judge": inferred_judge_metadata,
        "trust_env": config.trust_env,
    }
    provenance = collect_benchmark_provenance(
        model_id=effective_model,
        model_revision=effective_model_revision,
        dataset_id=DUPLEX_MPE_DATASET_ID,
        dataset_revision=(
            DUPLEX_MPE_DATASET_REVISION
            if dataset_identity.metadata_matches_expected_revision
            else None
        ),
        launch_command=effective_launch_command,
        server_config=server_declaration,
        evaluation_input_sha256=evaluation_input_sha256,
    )

    markdown_table = format_markdown_score_table(
        scored_metrics,
        interaction_protocol=effective_protocol,
        turn_detection=effective_turn_detection,
        selected_scenarios=len(scenarios_by_id),
        completed_scenarios=len(reports_by_id),
        missing_scenarios=len(missing_scenarios),
        failed_scenarios=len(failed_scenarios),
        required_semantic_tasks=len(expected_selected_task_keys),
        resolved_semantic_tasks=resolved_selected_verdict_count,
        pending_semantic_tasks=pending_semantic_count,
        failed_semantic_tasks=len(failed_semantic_task_keys),
        is_run_complete=is_run_complete,
    )
    heading_suffix = "" if is_run_complete else " — PARTIAL / INCOMPLETE"
    (output_dir / "RESULTS.md").write_text(
        f"# Duplex-MPE Evaluation ({effective_model}){heading_suffix}\n\n{markdown_table}",
        encoding="utf-8",
    )
    summary_payload: dict[str, JsonValue] = {
        "status": "complete" if is_run_complete else "incomplete",
        "evaluation_mode": evaluation_mode,
        "interaction": effective_protocol,
        "turn_detection": effective_turn_detection,
        "model": effective_model,
        "dataset_identity": asdict(dataset_identity),
        "selected_scenarios": len(scenarios_by_id),
        "completed_scenarios": len(reports_by_id),
        "missing_scenarios": list(missing_scenarios),
        "failed_scenarios": dict(failed_scenarios),
        "semantic_tasks": len(semantic_tasks),
        "semantic_verdicts": len(verdicts),
        "failed_semantic_tasks": len(failed_semantic_task_keys),
        "results": scored_metrics,
        "provenance": provenance,
    }
    summary_path = output_dir / "duplex_mpe_results.json"
    write_json_atomic(summary_path, summary_payload)
    print(markdown_table, end="")
    print(f"Saved summary to {summary_path}")
    return summary_payload


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate /v1/realtime speech models on Duplex-MPE."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("benchmarks/cache/duplex_mpe"),
        help="Path to materialized Duplex-MPE dataset directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/duplex_mpe"),
        help="Output directory outside dataset-root for audio, reports, and scores.",
    )
    parser.add_argument(
        "--phase",
        choices=("all", "generate", "asr", "judge", "score"),
        default="all",
        help="Pipeline phase to run (default: all).",
    )
    parser.add_argument(
        "--addressing",
        choices=("all", "explicit", "implicit"),
        default="all",
        help="Addressing condition subset to evaluate.",
    )
    parser.add_argument(
        "--mini",
        action="store_true",
        help="Select deterministic 20-pair (40-scenario) smoke-test subset.",
    )
    parser.add_argument(
        "--max-scenarios",
        type=int,
        default=None,
        help="Optional cap on the number of scenarios evaluated.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Served model identifier (inherited from run.json during offline phases).",
    )
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Optional model checkpoint commit SHA recorded in provenance.",
    )
    parser.add_argument(
        "--launch-command",
        default=None,
        help="Optional server launch command recorded in provenance.",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Realtime server root or /v1/realtime URL (e.g., http://127.0.0.1:8000).",
    )
    parser.add_argument(
        "--protocol",
        choices=("auto", "native", "turn_based", "legacy"),
        default="auto",
        help="Realtime session protocol (auto probes capabilities or session.created).",
    )
    parser.add_argument(
        "--input-sample-rate-hz",
        type=int,
        default=None,
        help="Override input PCM16 sample rate in Hz.",
    )
    parser.add_argument(
        "--output-sample-rate-hz",
        type=int,
        default=None,
        help="Override output PCM16 sample rate in Hz.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum concurrent /v1/realtime scenario sessions.",
    )
    parser.add_argument(
        "--warmup",
        dest="warmup_sessions",
        type=int,
        default=1,
        help="Unscored warmup sessions before measured recording (0 disables).",
    )
    parser.add_argument(
        "--session-timeout-s",
        dest="session_timeout_seconds",
        type=float,
        default=DEFAULT_SESSION_TIMEOUT_SECONDS,
        help="Per-scenario WebSocket session timeout in seconds.",
    )
    parser.add_argument(
        "--asr-base-url",
        default=None,
        help="OpenAI-compatible ASR server URL serving Qwen/Qwen3-ASR-1.7B.",
    )
    parser.add_argument(
        "--asr-model",
        default=None,
        help=(
            f"Model name sent to /v1/audio/transcriptions (default when running ASR: "
            f"{DEFAULT_ASR_MODEL_ID})."
        ),
    )
    parser.add_argument(
        "--asr-api-key-env",
        default=None,
        help="Optional environment variable holding the ASR bearer token.",
    )
    parser.add_argument(
        "--asr-max-concurrency",
        type=int,
        default=8,
        help="Concurrent /v1/audio/transcriptions requests.",
    )
    parser.add_argument(
        "--judge-config",
        dest="judge_config_path",
        type=Path,
        default=None,
        help="Path to JSON judge endpoint configuration.",
    )
    parser.add_argument(
        "--strict-reference-vad-version",
        action="store_true",
        help="Require exact reference silero-vad==6.2.1 when validating scenario reports.",
    )
    parser.add_argument(
        "--verify-retained-audio",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify retained assistant WAV SHA-256 digests before scoring (default: true).",
    )
    parser.add_argument(
        "--trust-env",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Honor HTTP_PROXY/HTTPS_PROXY/NO_PROXY environment variables.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_argument_parser()
    parsed_args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    configure_dependency_loggers()
    eval_config = DuplexMpeEvalConfig(
        dataset_root=parsed_args.dataset_root,
        output_dir=parsed_args.output_dir,
        phase=parsed_args.phase,
        addressing=parsed_args.addressing,
        mini=parsed_args.mini,
        max_scenarios=parsed_args.max_scenarios,
        model=parsed_args.model,
        model_revision=parsed_args.model_revision,
        launch_command=parsed_args.launch_command,
        base_url=parsed_args.base_url,
        protocol=parsed_args.protocol,
        input_sample_rate_hz=parsed_args.input_sample_rate_hz,
        output_sample_rate_hz=parsed_args.output_sample_rate_hz,
        max_concurrency=parsed_args.max_concurrency,
        warmup_sessions=parsed_args.warmup_sessions,
        session_timeout_seconds=parsed_args.session_timeout_seconds,
        asr_base_url=parsed_args.asr_base_url,
        asr_model=parsed_args.asr_model,
        asr_api_key_env=parsed_args.asr_api_key_env,
        asr_max_concurrency=parsed_args.asr_max_concurrency,
        judge_config_path=parsed_args.judge_config_path,
        strict_reference_vad_version=parsed_args.strict_reference_vad_version,
        verify_retained_audio=parsed_args.verify_retained_audio,
        trust_env=parsed_args.trust_env,
    )
    outcome = asyncio.run(run_duplex_mpe_evaluation(eval_config))
    return 0 if outcome.get("status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
else:
    pass
