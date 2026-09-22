"""Strict validators for the hm-loop protocol and report contracts.

The schema bundles are immutable release artifacts vendored beside this module,
then pinned by SHA-256 so validation cannot silently use another generation.
"""
from __future__ import annotations

from collections.abc import Mapping
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path
from typing import Any


_SCHEMA_SHA256 = {
    "protocol-schemas-v5.json": "fafae503e2f1a201c5afc3bdc04018911b930a0636fe7c4bb0a64f64a8b44b60",
    "report-schemas-v4.json": "880b97de7cece836143cea834d77d67016b3cf7c9d72cbf7a1d7b6fad3d369d8",
}


class ContractValidationError(ValueError):
    """A deterministic validation failure carrying its stable negative ID."""

    def __init__(self, error_id: str, message: str) -> None:
        self.error_id = error_id
        super().__init__(f"{error_id}: {message}")


def _reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ContractValidationError(
                "DUPLICATE_JSON_KEY", f"duplicate JSON object key {key!r}"
            )
        result[key] = value
    return result


def _reject_non_json_number(value: str) -> None:
    raise ContractValidationError("WRONG_TYPE", f"invalid JSON numeric literal {value!r}")


def parse_contract_json(value: str | bytes | bytearray) -> Any:
    """Parse strict JSON before any schema validation occurs."""
    if isinstance(value, bytearray):
        value = bytes(value)
    try:
        return json.loads(
            value,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_non_json_number,
        )
    except ContractValidationError:
        raise
    except (TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ContractValidationError("WRONG_TYPE", f"invalid JSON: {exc}") from exc


def canonical_json_bytes(value: Any) -> bytes:
    """Canonical compact sorted UTF-8 JSON with no terminal newline."""
    try:
        encoded = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ContractValidationError("WRONG_TYPE", f"not canonical JSON: {exc}") from exc
    if encoded.endswith(b"\n"):
        raise AssertionError("canonical JSON must not have a terminal LF")
    return encoded


def canonical_json_sha256(value: Any) -> str:
    return sha256(canonical_json_bytes(value)).hexdigest()


def _schema_path(filename: str) -> Path:
    path = Path(__file__).with_name("hm_loop_schemas") / filename
    if not path.is_file():
        raise RuntimeError(f"exact hm-loop schema artifact is missing: {path}")
    return path


@lru_cache(maxsize=2)
def _load_schema(filename: str) -> dict[str, Any]:
    raw = _schema_path(filename).read_bytes()
    expected_sha256 = _SCHEMA_SHA256[filename]
    actual_sha256 = sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"{filename} hash mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    parsed = parse_contract_json(raw)
    if not isinstance(parsed, dict):
        raise RuntimeError(f"{filename} must contain a JSON object")
    try:
        from jsonschema import Draft202012Validator
        Draft202012Validator.check_schema(parsed)
    except ImportError as exc:
        raise RuntimeError("jsonschema is required for strict hm-loop validation") from exc
    return parsed


def protocol_schema() -> dict[str, Any]:
    return _load_schema("protocol-schemas-v5.json")


def report_schema() -> dict[str, Any]:
    return _load_schema("report-schemas-v4.json")


def _error_id(error: Any) -> str:
    if error.validator == "additionalProperties":
        return "UNEXPECTED_PROPERTY"
    if error.validator == "required":
        return "MISSING_REQUIRED"
    return "WRONG_TYPE"


def _validate_definition(schema: Mapping[str, Any], definition: str, payload: Any) -> Any:
    definitions = schema.get("$defs")
    if not isinstance(definitions, Mapping) or definition not in definitions:
        raise ContractValidationError("UNKNOWN_CONTRACT", f"unknown contract {definition!r}")
    try:
        from jsonschema import Draft202012Validator
    except ImportError as exc:
        raise RuntimeError("jsonschema is required for strict hm-loop validation") from exc
    validator = Draft202012Validator(
        {"$schema": schema.get("$schema"), "$defs": definitions, "$ref": f"#/$defs/{definition}"}
    )
    errors = sorted(validator.iter_errors(payload), key=lambda error: (list(error.absolute_path), error.message))
    if errors:
        first = errors[0]
        location = "$" + "".join(
            f"[{part}]" if isinstance(part, int) else f".{part}"
            for part in first.absolute_path
        )
        raise ContractValidationError(_error_id(first), f"{location}: {first.message}")
    return payload


def validate_protocol(contract_name: str, payload: Mapping[str, Any] | str | bytes | bytearray) -> Any:
    """Validate one exact protocol-v5 definition and return the payload."""
    if not isinstance(contract_name, str) or not contract_name:
        raise ContractValidationError("UNKNOWN_CONTRACT", "contract name must be non-empty")
    if isinstance(payload, (str, bytes, bytearray)):
        parsed = parse_contract_json(payload)
    elif isinstance(payload, Mapping):
        parsed = dict(payload)
    else:
        raise ContractValidationError("WRONG_TYPE", "payload must be a mapping or JSON object")
    return _validate_definition(protocol_schema(), contract_name, parsed)


def validate_report_definition(report_name: str, payload: Mapping[str, Any] | str | bytes | bytearray) -> Any:
    """Validate a report payload against the exact report-v4 index mapping."""
    schema = report_schema()
    report_index = schema.get("x-report-index", [])
    definition_ref = next(
        (
            entry.get("schema_ref")
            for entry in report_index
            if entry.get("report_name") == report_name
        ),
        None,
    )
    if not isinstance(definition_ref, str) or not definition_ref.startswith("#/$defs/"):
        raise ContractValidationError("UNKNOWN_REPORT", f"unknown report {report_name!r}")
    if isinstance(payload, (str, bytes, bytearray)):
        parsed = parse_contract_json(payload)
    elif isinstance(payload, Mapping):
        parsed = dict(payload)
    else:
        raise ContractValidationError("WRONG_TYPE", "payload must be a mapping or JSON object")
    return _validate_definition(schema, definition_ref.removeprefix("#/$defs/"), parsed)
