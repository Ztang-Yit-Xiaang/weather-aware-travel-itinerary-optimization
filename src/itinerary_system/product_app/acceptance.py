"""Crash-recoverable W5 acceptance ledger and accepted-plan pointer."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from ..plans.repository import PlanRepository, PlanRepositoryConflict, _plan_from_record
from .models import utc_now
from .persistence import LocalStateLayout, StateFileLock, StateLockTimeoutError

POINTER_SCHEMA_VERSION = "workspace-pointer-v1"
DECISION_SCHEMA_VERSION = "acceptance-decision-v1"
SNAPSHOT_SCHEMA_VERSION = "acceptance-snapshot-v1"
TRANSACTION_SCHEMA_VERSION = "acceptance-transaction-v1"
RESULT_SCHEMA_VERSION = "acceptance-result-v1"

_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{1,128}\Z")
_IDEMPOTENCY_KEY = re.compile(r"[A-Za-z0-9_.:-]{16,128}\Z")
_DECISION_ID = re.compile(r"decision_[0-9a-f]{32}\Z")


class AcceptanceError(ValueError):
    """Stable, path-free W5 transaction error."""

    def __init__(self, code: str, status_code: int = 409) -> None:
        super().__init__(code)
        self.code = code
        self.status_code = status_code


@dataclass(frozen=True)
class AcceptanceResult:
    decision: dict[str, Any]
    pointer: dict[str, Any]
    recovery_performed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "decision": deepcopy(self.decision),
            "pointer": deepcopy(self.pointer),
            "recovery_performed": self.recovery_performed,
        }


class AcceptanceStore:
    """Trip-scoped plan, decision, pointer, and recovery transaction owner."""

    MAX_DECISIONS_PER_TRIP = 1000
    MAX_LEDGER_BYTES = 4 * 1024 * 1024

    def __init__(
        self,
        state_root: Path,
        *,
        lock_timeout_seconds: float = 2.0,
        failure_injector: Callable[[str], None] | None = None,
    ) -> None:
        self.root = Path(state_root)
        self.plan_repository = PlanRepository(self.root / "plan_repository")
        self.decisions_dir = self.root / "decisions"
        self.workspaces_dir = self.root / "workspaces"
        self.transactions_dir = self.root / "transactions"
        self.snapshots_dir = self.root / "acceptance_snapshots"
        self.locks_dir = self.root / "locks"
        self.lock_timeout_seconds = float(lock_timeout_seconds)
        self.failure_injector = failure_injector
        self._ensure_directories()

    def ensure_pointer(
        self,
        *,
        trip_id: str,
        run_id: str,
        original_plan: dict[str, Any],
    ) -> dict[str, Any]:
        self._validate_identifier(trip_id, "acceptance_trip_invalid")
        self._validate_identifier(run_id, "acceptance_run_invalid")
        plan = self._validated_plan(original_plan)
        try:
            with self._lock(trip_id):
                self._recover_locked(trip_id)
                pointer_path = self._pointer_path(trip_id)
                if pointer_path.is_file():
                    pointer = self._read_pointer_locked(trip_id)
                    if (
                        pointer["trip_id"] != trip_id
                        or pointer["run_id"] != run_id
                        or pointer["original_plan_id"] != plan.plan_id
                        or pointer["original_plan_content_hash"] != plan.content_hash
                    ):
                        raise AcceptanceError("acceptance_pointer_context_mismatch")
                    return pointer
                with self._plan_lock():
                    self.plan_repository.save(plan)
                pointer = self._pointer_record(
                    trip_id=trip_id,
                    run_id=run_id,
                    original_plan_id=plan.plan_id,
                    original_plan_content_hash=plan.content_hash,
                    accepted_plan_id=plan.plan_id,
                    accepted_plan_content_hash=plan.content_hash,
                    decision_id=None,
                    revision=0,
                )
                LocalStateLayout._atomic_json_write(pointer_path, pointer)
                return pointer
        except StateLockTimeoutError:
            raise AcceptanceError("acceptance_lock_timeout", 503) from None
        except OSError:
            raise AcceptanceError("acceptance_store_unavailable", 503) from None

    def load_pointer(self, trip_id: str) -> dict[str, Any]:
        self._validate_identifier(trip_id, "acceptance_trip_invalid")
        try:
            with self._lock(trip_id):
                self._recover_locked(trip_id)
                return self._read_pointer_locked(trip_id)
        except StateLockTimeoutError:
            raise AcceptanceError("acceptance_lock_timeout", 503) from None
        except OSError:
            raise AcceptanceError("acceptance_store_unavailable", 503) from None

    def find_idempotent(self, trip_id: str, idempotency_key: str) -> AcceptanceResult | None:
        self._validate_idempotency_key(idempotency_key)
        try:
            with self._lock(trip_id):
                recovered = self._recover_locked(trip_id)
                matches = [
                    row
                    for row in self._history_locked(trip_id)
                    if row["idempotency_key"] == idempotency_key
                ]
                if not matches:
                    return None
                if len(matches) != 1:
                    raise AcceptanceError("acceptance_ledger_invalid")
                return AcceptanceResult(matches[0], self._read_pointer_locked(trip_id), recovered)
        except StateLockTimeoutError:
            raise AcceptanceError("acceptance_lock_timeout", 503) from None
        except OSError:
            raise AcceptanceError("acceptance_store_unavailable", 503) from None

    def commit(
        self,
        *,
        trip_id: str,
        run_id: str,
        session_id: str,
        session_revision_before: int,
        expected_pointer_revision: int,
        expected_plan_id: str,
        expected_plan_content_hash: str,
        decision_type: str,
        idempotency_key: str,
        selected_plan: dict[str, Any],
        snapshot: dict[str, Any] | None,
        proposal_identity: dict[str, Any] | None,
    ) -> AcceptanceResult:
        self._validate_commit_inputs(
            trip_id=trip_id,
            run_id=run_id,
            session_id=session_id,
            session_revision_before=session_revision_before,
            expected_pointer_revision=expected_pointer_revision,
            decision_type=decision_type,
            idempotency_key=idempotency_key,
        )
        plan = self._validated_plan(selected_plan)
        try:
            with self._lock(trip_id):
                recovered = self._recover_locked(trip_id)
                existing = [
                    row
                    for row in self._history_locked(trip_id)
                    if row["idempotency_key"] == idempotency_key
                ]
                if existing:
                    if len(existing) != 1:
                        raise AcceptanceError("acceptance_ledger_invalid")
                    row = existing[0]
                    identity = proposal_identity or {}
                    if (
                        row["decision_type"] != decision_type
                        or row["session_id"] != session_id
                        or row["session_revision_before"] != session_revision_before
                        or row["pointer_revision_before"] != expected_pointer_revision
                        or row["previous_plan_id"] != expected_plan_id
                        or row["previous_plan_content_hash"] != expected_plan_content_hash
                        or row["selected_plan_id"] != plan.plan_id
                        or row["selected_plan_content_hash"] != plan.content_hash
                        or any(
                            row[name] != identity.get(name)
                            for name in (
                                "proposal_id",
                                "draft_content_hash",
                                "diff_id",
                                "diff_content_hash",
                                "certificate_id",
                                "certificate_content_hash",
                            )
                        )
                    ):
                        raise AcceptanceError("acceptance_idempotency_conflict")
                    return AcceptanceResult(row, self._read_pointer_locked(trip_id), recovered)
                pointer = self._read_pointer_locked(trip_id)
                if (
                    pointer["trip_id"] != trip_id
                    or pointer["run_id"] != run_id
                    or pointer["revision"] != expected_pointer_revision
                    or pointer["accepted_plan_id"] != expected_plan_id
                    or pointer["accepted_plan_content_hash"] != expected_plan_content_hash
                ):
                    raise AcceptanceError("acceptance_pointer_conflict")
                with self._plan_lock():
                    self.plan_repository.save(plan)
                self._inject("after_plan_save")
                decision_id = f"decision_{uuid4().hex}"
                snapshot_hash = None
                if snapshot is not None:
                    snapshot_record = self._snapshot_record(
                        decision_id=decision_id,
                        trip_id=trip_id,
                        run_id=run_id,
                        plan=selected_plan,
                        snapshot=snapshot,
                    )
                    snapshot_hash = snapshot_record["content_hash"]
                    self._append_only_json(self._snapshot_path(decision_id), snapshot_record)
                next_pointer = self._pointer_record(
                    trip_id=trip_id,
                    run_id=run_id,
                    original_plan_id=pointer["original_plan_id"],
                    original_plan_content_hash=pointer["original_plan_content_hash"],
                    accepted_plan_id=plan.plan_id,
                    accepted_plan_content_hash=plan.content_hash,
                    decision_id=decision_id,
                    revision=pointer["revision"] + 1,
                )
                decision = self._decision_record(
                    decision_id=decision_id,
                    idempotency_key=idempotency_key,
                    decision_type=decision_type,
                    trip_id=trip_id,
                    run_id=run_id,
                    session_id=session_id,
                    session_revision_before=session_revision_before,
                    pointer=pointer,
                    next_pointer=next_pointer,
                    proposal_identity=proposal_identity,
                    snapshot_content_hash=snapshot_hash,
                )
                transaction = self._transaction_record(
                    trip_id=trip_id,
                    decision=decision,
                    previous_pointer=pointer,
                    next_pointer=next_pointer,
                )
                LocalStateLayout._atomic_json_write(self._transaction_path(trip_id), transaction)
                self._inject("after_prepare")
                self._append_decision_locked(trip_id, decision)
                self._inject("after_ledger_append")
                LocalStateLayout._atomic_json_write(self._pointer_path(trip_id), next_pointer)
                self._inject("after_pointer_replace")
                self._transaction_path(trip_id).unlink(missing_ok=True)
                LocalStateLayout._fsync_directory(self.transactions_dir)
                return AcceptanceResult(decision, next_pointer, recovered)
        except AcceptanceError:
            raise
        except PlanRepositoryConflict:
            raise AcceptanceError("acceptance_plan_repository_conflict") from None
        except StateLockTimeoutError:
            raise AcceptanceError("acceptance_lock_timeout", 503) from None
        except (OSError, TypeError, ValueError):
            raise AcceptanceError("acceptance_transaction_failed", 503) from None

    def history(self, trip_id: str) -> tuple[dict[str, Any], ...]:
        self._validate_identifier(trip_id, "acceptance_trip_invalid")
        try:
            with self._lock(trip_id):
                self._recover_locked(trip_id)
                return tuple(deepcopy(self._history_locked(trip_id)))
        except StateLockTimeoutError:
            raise AcceptanceError("acceptance_lock_timeout", 503) from None
        except OSError:
            raise AcceptanceError("acceptance_store_unavailable", 503) from None

    def load_snapshot(self, decision_id: str) -> dict[str, Any]:
        if _DECISION_ID.fullmatch(decision_id) is None:
            raise AcceptanceError("acceptance_decision_invalid", 422)
        path = self._snapshot_path(decision_id)
        if not path.is_file():
            raise AcceptanceError("acceptance_snapshot_not_found", 404)
        try:
            payload = LocalStateLayout._read_json_file(path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            raise AcceptanceError("acceptance_snapshot_invalid") from None
        self._validate_snapshot(payload)
        return payload

    def _recover_locked(self, trip_id: str) -> bool:
        path = self._transaction_path(trip_id)
        if not path.is_file():
            return False
        try:
            transaction = LocalStateLayout._read_json_file(path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            raise AcceptanceError("acceptance_transaction_invalid") from None
        self._validate_transaction(transaction, trip_id)
        decision = transaction["decision"]
        previous = transaction["previous_pointer"]
        next_pointer = transaction["next_pointer"]
        history = self._history_locked(trip_id)
        matching = [row for row in history if row["decision_id"] == decision["decision_id"]]
        if not matching:
            self._append_decision_locked(trip_id, decision)
        elif len(matching) != 1 or matching[0] != decision:
            raise AcceptanceError("acceptance_recovery_conflict")
        current = self._read_pointer_locked(trip_id)
        if current == previous:
            LocalStateLayout._atomic_json_write(self._pointer_path(trip_id), next_pointer)
        elif current != next_pointer:
            raise AcceptanceError("acceptance_recovery_conflict")
        path.unlink(missing_ok=True)
        LocalStateLayout._fsync_directory(self.transactions_dir)
        return True

    def _history_locked(self, trip_id: str) -> list[dict[str, Any]]:
        path = self._ledger_path(trip_id)
        if not path.exists():
            return []
        if path.stat().st_size > self.MAX_LEDGER_BYTES:
            raise AcceptanceError("acceptance_ledger_limit_reached")
        rows: list[dict[str, Any]] = []
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.endswith("\n"):
                        raise AcceptanceError("acceptance_ledger_invalid")
                    row = json.loads(line)
                    try:
                        self._validate_decision(row)
                    except AcceptanceError:
                        raise AcceptanceError("acceptance_ledger_invalid") from None
                    rows.append(row)
        except AcceptanceError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise AcceptanceError("acceptance_ledger_invalid") from None
        if len(rows) > self.MAX_DECISIONS_PER_TRIP:
            raise AcceptanceError("acceptance_ledger_limit_reached")
        if len({row["decision_id"] for row in rows}) != len(rows) or len(
            {row["idempotency_key"] for row in rows}
        ) != len(rows):
            raise AcceptanceError("acceptance_ledger_invalid")
        if any(
            row["pointer_revision_after"] != row["pointer_revision_before"] + 1
            for row in rows
        ):
            raise AcceptanceError("acceptance_ledger_invalid")
        return rows

    def _append_decision_locked(self, trip_id: str, decision: dict[str, Any]) -> None:
        history = self._history_locked(trip_id)
        if any(row["decision_id"] == decision["decision_id"] for row in history):
            return
        if any(row["idempotency_key"] == decision["idempotency_key"] for row in history):
            raise AcceptanceError("acceptance_idempotency_conflict")
        if len(history) >= self.MAX_DECISIONS_PER_TRIP:
            raise AcceptanceError("acceptance_ledger_limit_reached")
        path = self._ledger_path(trip_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(_canonical_json(decision) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        LocalStateLayout._fsync_directory(path.parent)

    def _read_pointer_locked(self, trip_id: str) -> dict[str, Any]:
        path = self._pointer_path(trip_id)
        if not path.is_file():
            raise AcceptanceError("acceptance_pointer_not_found", 404)
        try:
            pointer = LocalStateLayout._read_json_file(path)
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            raise AcceptanceError("acceptance_pointer_invalid") from None
        self._validate_pointer(pointer)
        if pointer["trip_id"] != trip_id:
            raise AcceptanceError("acceptance_pointer_invalid")
        return pointer

    def _validated_plan(self, record: dict[str, Any]):
        if not _finite_json(record):
            raise AcceptanceError("acceptance_plan_invalid", 422)
        try:
            plan = _plan_from_record(record)
        except (KeyError, TypeError, ValueError):
            raise AcceptanceError("acceptance_plan_invalid", 422) from None
        if record.get("content_hash") != plan.content_hash:
            raise AcceptanceError("acceptance_plan_hash_mismatch")
        return plan

    def _pointer_record(self, **values: Any) -> dict[str, Any]:
        payload = {"schema_version": POINTER_SCHEMA_VERSION, **values, "updated_at": utc_now()}
        payload["content_hash"] = _content_hash(payload)
        self._validate_pointer(payload)
        return payload

    def _decision_record(
        self,
        *,
        decision_id: str,
        idempotency_key: str,
        decision_type: str,
        trip_id: str,
        run_id: str,
        session_id: str,
        session_revision_before: int,
        pointer: dict[str, Any],
        next_pointer: dict[str, Any],
        proposal_identity: dict[str, Any] | None,
        snapshot_content_hash: str | None,
    ) -> dict[str, Any]:
        identity = proposal_identity or {}
        payload = {
            "schema_version": DECISION_SCHEMA_VERSION,
            "decision_id": decision_id,
            "idempotency_key": idempotency_key,
            "decision_type": decision_type,
            "trip_id": trip_id,
            "run_id": run_id,
            "session_id": session_id,
            "session_revision_before": session_revision_before,
            "session_revision_after": session_revision_before + 1,
            "pointer_revision_before": pointer["revision"],
            "pointer_revision_after": next_pointer["revision"],
            "previous_plan_id": pointer["accepted_plan_id"],
            "previous_plan_content_hash": pointer["accepted_plan_content_hash"],
            "selected_plan_id": next_pointer["accepted_plan_id"],
            "selected_plan_content_hash": next_pointer["accepted_plan_content_hash"],
            "proposal_id": identity.get("proposal_id"),
            "draft_content_hash": identity.get("draft_content_hash"),
            "diff_id": identity.get("diff_id"),
            "diff_content_hash": identity.get("diff_content_hash"),
            "certificate_id": identity.get("certificate_id"),
            "certificate_content_hash": identity.get("certificate_content_hash"),
            "snapshot_content_hash": snapshot_content_hash,
            "created_at": utc_now(),
        }
        payload["request_fingerprint"] = _content_hash(
            {
                key: payload[key]
                for key in (
                    "decision_type",
                    "trip_id",
                    "run_id",
                    "session_id",
                    "session_revision_before",
                    "pointer_revision_before",
                    "previous_plan_id",
                    "previous_plan_content_hash",
                    "selected_plan_id",
                    "selected_plan_content_hash",
                    "proposal_id",
                    "draft_content_hash",
                    "diff_id",
                    "diff_content_hash",
                    "certificate_id",
                    "certificate_content_hash",
                    "snapshot_content_hash",
                )
            }
        )
        payload["content_hash"] = _content_hash(payload)
        self._validate_decision(payload)
        return payload

    def _snapshot_record(
        self,
        *,
        decision_id: str,
        trip_id: str,
        run_id: str,
        plan: dict[str, Any],
        snapshot: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "schema_version": SNAPSHOT_SCHEMA_VERSION,
            "decision_id": decision_id,
            "trip_id": trip_id,
            "run_id": run_id,
            "plan": deepcopy(plan),
            "diff": deepcopy(snapshot.get("diff")),
            "certificate": deepcopy(snapshot.get("certificate")),
            "route_legs": deepcopy(snapshot.get("route_legs")),
            "geography_plan": deepcopy(snapshot.get("geography_plan")),
            "proposal_identity": deepcopy(snapshot.get("proposal_identity")),
        }
        if "planning_bundle" in snapshot:
            payload["planning_bundle"] = deepcopy(snapshot["planning_bundle"])
        if not _finite_json(payload):
            raise AcceptanceError("acceptance_snapshot_invalid", 422)
        payload["content_hash"] = _content_hash(payload)
        self._validate_snapshot(payload)
        return payload

    def _transaction_record(
        self,
        *,
        trip_id: str,
        decision: dict[str, Any],
        previous_pointer: dict[str, Any],
        next_pointer: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            "schema_version": TRANSACTION_SCHEMA_VERSION,
            "trip_id": trip_id,
            "decision": deepcopy(decision),
            "previous_pointer": deepcopy(previous_pointer),
            "next_pointer": deepcopy(next_pointer),
            "prepared_at": utc_now(),
        }
        payload["content_hash"] = _content_hash(payload)
        return payload

    @staticmethod
    def _validate_pointer(pointer: Any) -> None:
        fields = {
            "schema_version",
            "trip_id",
            "run_id",
            "original_plan_id",
            "original_plan_content_hash",
            "accepted_plan_id",
            "accepted_plan_content_hash",
            "decision_id",
            "revision",
            "updated_at",
            "content_hash",
        }
        if not isinstance(pointer, dict) or set(pointer) != fields:
            raise AcceptanceError("acceptance_pointer_invalid")
        payload = dict(pointer)
        content_hash = payload.pop("content_hash", None)
        if content_hash != _content_hash(payload):
            raise AcceptanceError("acceptance_pointer_invalid")
        if pointer["schema_version"] != POINTER_SCHEMA_VERSION:
            raise AcceptanceError("acceptance_pointer_schema_unsupported")
        if not isinstance(pointer["revision"], int) or isinstance(pointer["revision"], bool) or pointer["revision"] < 0:
            raise AcceptanceError("acceptance_pointer_invalid")
        for name in (
            "trip_id",
            "run_id",
            "original_plan_id",
            "original_plan_content_hash",
            "accepted_plan_id",
            "accepted_plan_content_hash",
            "updated_at",
        ):
            if not isinstance(pointer[name], str) or not pointer[name]:
                raise AcceptanceError("acceptance_pointer_invalid")
        if pointer["decision_id"] is not None and _DECISION_ID.fullmatch(pointer["decision_id"]) is None:
            raise AcceptanceError("acceptance_pointer_invalid")

    @staticmethod
    def _validate_decision(decision: Any) -> None:
        fields = {
            "schema_version",
            "decision_id",
            "idempotency_key",
            "decision_type",
            "trip_id",
            "run_id",
            "session_id",
            "session_revision_before",
            "session_revision_after",
            "pointer_revision_before",
            "pointer_revision_after",
            "previous_plan_id",
            "previous_plan_content_hash",
            "selected_plan_id",
            "selected_plan_content_hash",
            "proposal_id",
            "draft_content_hash",
            "diff_id",
            "diff_content_hash",
            "certificate_id",
            "certificate_content_hash",
            "snapshot_content_hash",
            "created_at",
            "request_fingerprint",
            "content_hash",
        }
        if not isinstance(decision, dict) or set(decision) != fields:
            raise AcceptanceError("acceptance_decision_invalid")
        payload = dict(decision)
        content_hash = payload.pop("content_hash", None)
        if content_hash != _content_hash(payload):
            raise AcceptanceError("acceptance_decision_invalid")
        if decision["schema_version"] != DECISION_SCHEMA_VERSION or _DECISION_ID.fullmatch(decision["decision_id"]) is None:
            raise AcceptanceError("acceptance_decision_invalid")
        if _IDEMPOTENCY_KEY.fullmatch(decision["idempotency_key"]) is None:
            raise AcceptanceError("acceptance_decision_invalid")
        if decision["decision_type"] not in {"accept_repair", "keep_original", "rollback_to_eligible_plan"}:
            raise AcceptanceError("acceptance_decision_invalid")
        for name in (
            "session_revision_before",
            "session_revision_after",
            "pointer_revision_before",
            "pointer_revision_after",
        ):
            if not isinstance(decision[name], int) or isinstance(decision[name], bool) or decision[name] < 0:
                raise AcceptanceError("acceptance_decision_invalid")
        if decision["session_revision_after"] != decision["session_revision_before"] + 1:
            raise AcceptanceError("acceptance_decision_invalid")
        for name in (
            "proposal_id",
            "draft_content_hash",
            "diff_id",
            "diff_content_hash",
            "certificate_id",
            "certificate_content_hash",
            "snapshot_content_hash",
        ):
            if decision[name] is not None and (not isinstance(decision[name], str) or not decision[name]):
                raise AcceptanceError("acceptance_decision_invalid")

    def _validate_snapshot(self, snapshot: Any) -> None:
        fields = {
            "schema_version",
            "decision_id",
            "trip_id",
            "run_id",
            "plan",
            "diff",
            "certificate",
            "route_legs",
            "geography_plan",
            "proposal_identity",
            "content_hash",
        }
        if isinstance(snapshot, dict) and "planning_bundle" in snapshot:
            fields.add("planning_bundle")
        if not isinstance(snapshot, dict) or set(snapshot) != fields:
            raise AcceptanceError("acceptance_snapshot_invalid")
        payload = dict(snapshot)
        content_hash = payload.pop("content_hash", None)
        if content_hash != _content_hash(payload) or snapshot["schema_version"] != SNAPSHOT_SCHEMA_VERSION:
            raise AcceptanceError("acceptance_snapshot_invalid")
        self._validated_plan(snapshot["plan"])

    def _validate_transaction(self, transaction: Any, trip_id: str) -> None:
        fields = {
            "schema_version",
            "trip_id",
            "decision",
            "previous_pointer",
            "next_pointer",
            "prepared_at",
            "content_hash",
        }
        if not isinstance(transaction, dict) or set(transaction) != fields:
            raise AcceptanceError("acceptance_transaction_invalid")
        payload = dict(transaction)
        content_hash = payload.pop("content_hash", None)
        if content_hash != _content_hash(payload) or transaction["schema_version"] != TRANSACTION_SCHEMA_VERSION:
            raise AcceptanceError("acceptance_transaction_invalid")
        if transaction["trip_id"] != trip_id:
            raise AcceptanceError("acceptance_transaction_invalid")
        self._validate_decision(transaction["decision"])
        self._validate_pointer(transaction["previous_pointer"])
        self._validate_pointer(transaction["next_pointer"])

    def _validate_commit_inputs(self, **values: Any) -> None:
        for name in ("trip_id", "run_id", "session_id"):
            self._validate_identifier(values[name], f"{name}_invalid")
        for name in ("session_revision_before", "expected_pointer_revision"):
            value = values[name]
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise AcceptanceError("acceptance_revision_invalid", 422)
        if values["decision_type"] not in {"accept_repair", "keep_original", "rollback_to_eligible_plan"}:
            raise AcceptanceError("acceptance_decision_type_invalid", 422)
        self._validate_idempotency_key(values["idempotency_key"])

    @staticmethod
    def _validate_identifier(value: Any, code: str) -> None:
        if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
            raise AcceptanceError(code, 422)

    @staticmethod
    def _validate_idempotency_key(value: Any) -> None:
        if not isinstance(value, str) or _IDEMPOTENCY_KEY.fullmatch(value) is None:
            raise AcceptanceError("acceptance_idempotency_key_invalid", 422)

    def _append_only_json(self, path: Path, payload: dict[str, Any]) -> None:
        if path.exists():
            existing = LocalStateLayout._read_json_file(path)
            if existing != payload:
                raise AcceptanceError("acceptance_snapshot_conflict")
            return
        temporary = path.parent / f".{uuid4().hex}.tmp"
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = temporary.open("x", encoding="utf-8", newline="\n")
        try:
            with handle:
                handle.write(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, path)
            LocalStateLayout._fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)

    def _inject(self, stage: str) -> None:
        if self.failure_injector is not None:
            self.failure_injector(stage)

    def _ensure_directories(self) -> None:
        for path in (
            self.plan_repository.root,
            self.decisions_dir,
            self.workspaces_dir,
            self.transactions_dir,
            self.snapshots_dir,
            self.locks_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def _lock(self, trip_id: str) -> StateFileLock:
        return StateFileLock(self.locks_dir / f"acceptance-{_trip_key(trip_id)}.lock", self.lock_timeout_seconds)

    def _plan_lock(self) -> StateFileLock:
        return StateFileLock(self.locks_dir / "plan-repository.lock", self.lock_timeout_seconds)

    def _pointer_path(self, trip_id: str) -> Path:
        return self.workspaces_dir / f"{_trip_key(trip_id)}.json"

    def _ledger_path(self, trip_id: str) -> Path:
        return self.decisions_dir / _trip_key(trip_id) / "ledger.jsonl"

    def _transaction_path(self, trip_id: str) -> Path:
        return self.transactions_dir / f"{_trip_key(trip_id)}.json"

    def _snapshot_path(self, decision_id: str) -> Path:
        return self.snapshots_dir / f"{decision_id}.json"


def _trip_key(trip_id: str) -> str:
    return hashlib.sha256(trip_id.encode("utf-8")).hexdigest()[:32]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _content_hash(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _finite_json(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(isinstance(key, str) and _finite_json(item) for key, item in value.items())
    if isinstance(value, (list, tuple)):
        return all(_finite_json(item) for item in value)
    return value is None or isinstance(value, (str, int, bool))
