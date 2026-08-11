"""Durable, local-only Copilot conversation persistence."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from .models import (
    ConversationV1,
    CopilotHighlightsV1,
    CopilotIntentV1,
    CopilotInterpretationV1,
    CopilotTurnV1,
    ProductSession,
)
from .persistence import ConversationFileStore, StateLockTimeoutError

CONVERSATION_SCHEMA_VERSION = "product-conversation-v1"
TURN_SCHEMA_VERSION = "copilot-turn-v1"
INTERPRETATION_SCHEMA_VERSION = "copilot-interpretation-v1"
RETENTION = timedelta(days=30)
MAX_TURNS = 200
MAX_FILE_BYTES = 1024 * 1024
MAX_FILES = 128
CLEANUP_INTERVAL_SECONDS = 60 * 60

_CONVERSATION_ID = re.compile(r"conversation_[0-9a-f]{32}\Z")
_SESSION_ID = re.compile(r"session_[0-9a-f]{32}\Z")
_TURN_ID = re.compile(r"copilot_turn_[0-9a-f]{32}\Z")
_CLIENT_MESSAGE_ID = re.compile(r"client_message_[0-9a-f]{32}\Z")
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}\Z")

_INTENT_TYPES = {
    "review_registered_repair",
    "keep_original",
    "keep_stop",
    "lock_stop",
    "mark_flexible",
    "move_day",
    "route_feedback",
    "replace_nearby",
    "add_candidate",
    "request_clarification",
    "unsupported_request",
    "external_action_not_supported",
}
_PREFERENCES = {
    "reduce_contextual_risk",
    "reduce_driving",
    "preserve_experience",
}
_INTERPRETATION_STATES = {
    "clarification_required",
    "permission_required",
    "proposal_ready",
    "unsupported",
    "refused",
    "failed",
}


class ConversationError(ValueError):
    """Stable, path-free conversation persistence failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class ConversationRepository:
    """Strict conversation files serialized under the shared layout lock."""

    def __init__(
        self,
        state_root: Path,
        *,
        lock_timeout_seconds: float = 2.0,
        clock: Callable[[], datetime] | None = None,
        monotonic: Callable[[], float] | None = None,
    ) -> None:
        self._files = ConversationFileStore(
            state_root,
            lock_timeout_seconds=lock_timeout_seconds,
            max_file_bytes=MAX_FILE_BYTES,
        )
        self.root = self._files.root
        self.lock_path = self._files.lock_path
        self._clock = clock or (lambda: datetime.now(UTC))
        self._monotonic = monotonic or time.monotonic
        self._last_cleanup: float | None = None
        try:
            self._files.ensure_directory()
            with self._files.locked():
                self._purge_expired_locked(self._now())
                self._last_cleanup = self._monotonic()
        except StateLockTimeoutError:
            raise ConversationError("state_lock_timeout") from None
        except ConversationError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            raise ConversationError("conversation_store_invalid") from None

    def get_or_create(self, session: ProductSession) -> ConversationV1:
        self._validate_session(session)
        try:
            with self._files.locked():
                now = self._now()
                self._maybe_purge_locked(now)
                conversation_id = session.conversation_id
                if conversation_id is not None:
                    self._require_id(conversation_id)
                    path = self._path(conversation_id)
                    if self._entry_exists(path):
                        conversation = self._load(path, expected_id=conversation_id)
                        self._require_owner(conversation, session.session_id)
                        if conversation.trip_id != session.trip_id or conversation.run_id != session.run_id:
                            raise ConversationError("conversation_session_mismatch")
                        if self._is_expired(conversation, now):
                            self._files.delete(path)
                        else:
                            return conversation
                else:
                    conversation_id = self._new_id_locked()

                self._require_capacity_locked()
                created_at = self._format_timestamp(now)
                conversation = ConversationV1(
                    conversation_id=conversation_id,
                    session_id=session.session_id,
                    trip_id=session.trip_id,
                    run_id=session.run_id,
                    revision=0,
                    turns=(),
                    created_at=created_at,
                    updated_at=created_at,
                    expires_at=self._format_timestamp(now + RETENTION),
                )
                self._save(conversation)
                session.conversation_id = conversation_id
                return conversation
        except StateLockTimeoutError:
            raise ConversationError("state_lock_timeout") from None
        except ConversationError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            raise ConversationError("conversation_write_failed") from None

    def get(self, conversation_id: str, session_id: str) -> ConversationV1:
        self._require_id(conversation_id)
        self._require_session_id(session_id)
        try:
            with self._files.locked():
                now = self._now()
                self._maybe_purge_locked(now)
                path = self._path(conversation_id)
                if not self._entry_exists(path):
                    raise ConversationError("unknown_conversation")
                conversation = self._load(path, expected_id=conversation_id)
                self._require_owner(conversation, session_id)
                if self._is_expired(conversation, now):
                    self._files.delete(path)
                    raise ConversationError("unknown_conversation")
                return conversation
        except StateLockTimeoutError:
            raise ConversationError("state_lock_timeout") from None
        except ConversationError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            raise ConversationError("conversation_invalid") from None

    def append_turn(
        self,
        conversation_id: str,
        session_id: str,
        turn: CopilotTurnV1,
    ) -> ConversationV1:
        self._require_id(conversation_id)
        self._require_session_id(session_id)
        self._validate_turn(turn.as_dict())
        try:
            with self._files.locked():
                now = self._now()
                self._maybe_purge_locked(now)
                path = self._path(conversation_id)
                if not self._entry_exists(path):
                    raise ConversationError("unknown_conversation")
                conversation = self._load(path, expected_id=conversation_id)
                self._require_owner(conversation, session_id)
                if self._is_expired(conversation, now):
                    self._files.delete(path)
                    raise ConversationError("unknown_conversation")

                for existing in conversation.turns:
                    if existing.client_message_id != turn.client_message_id:
                        continue
                    if self._message_hash(existing.user_message) == self._message_hash(turn.user_message):
                        return conversation
                    raise ConversationError("message_id_conflict")
                if len(conversation.turns) >= MAX_TURNS:
                    raise ConversationError("conversation_turn_limit_reached")

                updated_at = self._format_timestamp(now)
                updated = ConversationV1(
                    conversation_id=conversation.conversation_id,
                    session_id=conversation.session_id,
                    trip_id=conversation.trip_id,
                    run_id=conversation.run_id,
                    revision=conversation.revision + 1,
                    turns=(*conversation.turns, turn),
                    created_at=conversation.created_at,
                    updated_at=updated_at,
                    expires_at=self._format_timestamp(now + RETENTION),
                )
                self._save(updated)
                return updated
        except StateLockTimeoutError:
            raise ConversationError("state_lock_timeout") from None
        except ConversationError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            raise ConversationError("conversation_write_failed") from None

    def delete(self, conversation_id: str, session_id: str) -> bool:
        self._require_id(conversation_id)
        self._require_session_id(session_id)
        try:
            with self._files.locked():
                self._maybe_purge_locked(self._now())
                path = self._path(conversation_id)
                if not self._entry_exists(path):
                    return False
                conversation = self._load(path, expected_id=conversation_id)
                self._require_owner(conversation, session_id)
                self._files.delete(path)
                return True
        except StateLockTimeoutError:
            raise ConversationError("state_lock_timeout") from None
        except ConversationError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            raise ConversationError("conversation_delete_failed") from None

    def delete_all(self) -> int:
        try:
            with self._files.locked():
                paths = self._conversation_paths_locked()
                for path in paths:
                    self._load(path, expected_id=path.stem)
                for path in paths:
                    self._files.delete(path, fsync=False)
                if paths:
                    self._files.fsync_directory()
                return len(paths)
        except StateLockTimeoutError:
            raise ConversationError("state_lock_timeout") from None
        except ConversationError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            raise ConversationError("conversation_delete_failed") from None

    def purge_expired(self, now: datetime | None = None) -> int:
        try:
            with self._files.locked():
                count = self._purge_expired_locked(self._coerce_now(now) if now is not None else self._now())
                self._last_cleanup = self._monotonic()
                return count
        except StateLockTimeoutError:
            raise ConversationError("state_lock_timeout") from None
        except ConversationError:
            raise
        except (OSError, UnicodeError, json.JSONDecodeError, TypeError, ValueError):
            raise ConversationError("conversation_cleanup_failed") from None

    def _maybe_purge_locked(self, now: datetime) -> None:
        current = self._monotonic()
        if self._last_cleanup is None or current - self._last_cleanup >= CLEANUP_INTERVAL_SECONDS:
            self._purge_expired_locked(now)
            self._last_cleanup = current

    def _purge_expired_locked(self, now: datetime) -> int:
        expired: list[Path] = []
        for path in self._conversation_paths_locked():
            conversation = self._load(path, expected_id=path.stem)
            if self._is_expired(conversation, now):
                expired.append(path)
        for path in expired:
            self._files.delete(path, fsync=False)
        if expired:
            self._files.fsync_directory()
        return len(expired)

    def _require_capacity_locked(self) -> None:
        if len(self._conversation_paths_locked()) >= MAX_FILES:
            raise ConversationError("conversation_capacity_reached")

    def _new_id_locked(self) -> str:
        for _ in range(8):
            value = f"conversation_{uuid4().hex}"
            if not self._entry_exists(self._path(value)):
                return value
        raise ConversationError("conversation_id_generation_failed")

    def _conversation_paths_locked(self) -> tuple[Path, ...]:
        paths = self._files.paths()
        for path in paths:
            if not _CONVERSATION_ID.fullmatch(path.stem) or path.suffix != ".json":
                raise ConversationError("conversation_store_invalid")
        return paths

    def _load(self, path: Path, *, expected_id: str) -> ConversationV1:
        return self._decode(self._files.read(path), expected_id=expected_id)

    def _save(self, conversation: ConversationV1) -> None:
        payload = conversation.as_dict()
        self._decode(payload, expected_id=conversation.conversation_id)
        encoded_size = len(
            (json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
                "utf-8"
            )
        )
        if encoded_size > MAX_FILE_BYTES:
            raise ConversationError("conversation_size_limit_reached")
        self._files.write(
            self._path(conversation.conversation_id),
            payload,
            validate=lambda value: self._decode(value, expected_id=conversation.conversation_id),
        )

    @classmethod
    def _decode(cls, payload: Any, *, expected_id: str) -> ConversationV1:
        fields = {
            "schema_version",
            "conversation_id",
            "session_id",
            "trip_id",
            "run_id",
            "revision",
            "turns",
            "created_at",
            "updated_at",
            "expires_at",
        }
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ConversationError("conversation_invalid")
        if payload.get("schema_version") != CONVERSATION_SCHEMA_VERSION:
            raise ConversationError("conversation_schema_unsupported")
        conversation_id = payload.get("conversation_id")
        if conversation_id != expected_id or not isinstance(conversation_id, str):
            raise ConversationError("conversation_invalid")
        cls._require_id(conversation_id)
        session_id = payload.get("session_id")
        if not isinstance(session_id, str):
            raise ConversationError("conversation_invalid")
        cls._require_session_id(session_id)
        trip_id = cls._text(payload.get("trip_id"), 256)
        run_id = cls._text(payload.get("run_id"), 256)
        revision = payload.get("revision")
        if not isinstance(revision, int) or isinstance(revision, bool) or revision < 0:
            raise ConversationError("conversation_invalid")
        turns_payload = payload.get("turns")
        if not isinstance(turns_payload, (list, tuple)) or len(turns_payload) > MAX_TURNS:
            raise ConversationError("conversation_invalid")
        turns = tuple(cls._validate_turn(item) for item in turns_payload)
        client_ids = [turn.client_message_id for turn in turns]
        turn_ids = [turn.turn_id for turn in turns]
        if len(client_ids) != len(set(client_ids)) or len(turn_ids) != len(set(turn_ids)):
            raise ConversationError("conversation_invalid")
        if revision != len(turns):
            raise ConversationError("conversation_invalid")
        created_at = cls._parse_timestamp(payload.get("created_at"))
        updated_at = cls._parse_timestamp(payload.get("updated_at"))
        expires_at = cls._parse_timestamp(payload.get("expires_at"))
        if created_at > updated_at or expires_at != updated_at + RETENTION:
            raise ConversationError("conversation_invalid")
        return ConversationV1(
            conversation_id=conversation_id,
            session_id=session_id,
            trip_id=trip_id,
            run_id=run_id,
            revision=revision,
            turns=turns,
            created_at=payload["created_at"],
            updated_at=payload["updated_at"],
            expires_at=payload["expires_at"],
        )

    @classmethod
    def _validate_turn(cls, payload: Any) -> CopilotTurnV1:
        fields = {
            "schema_version",
            "turn_id",
            "client_message_id",
            "context_revision",
            "provider",
            "model",
            "prompt_version",
            "prompt_sha256",
            "state",
            "user_message",
            "assistant_message",
            "interpretation",
            "error_code",
            "created_at",
            "completed_at",
        }
        if not isinstance(payload, dict) or set(payload) != fields or payload.get("schema_version") != TURN_SCHEMA_VERSION:
            raise ConversationError("conversation_invalid")
        turn_id = payload.get("turn_id")
        client_message_id = payload.get("client_message_id")
        if not isinstance(turn_id, str) or not _TURN_ID.fullmatch(turn_id):
            raise ConversationError("conversation_invalid")
        if not isinstance(client_message_id, str) or not _CLIENT_MESSAGE_ID.fullmatch(client_message_id):
            raise ConversationError("conversation_invalid")
        context_revision = payload.get("context_revision")
        if not isinstance(context_revision, int) or isinstance(context_revision, bool) or context_revision < 0:
            raise ConversationError("conversation_invalid")
        provider = payload.get("provider")
        if provider not in {"deterministic", "openai"}:
            raise ConversationError("conversation_invalid")
        model = payload.get("model")
        if model is not None:
            model = cls._text(model, 128)
        prompt_version = cls._text(payload.get("prompt_version"), 128)
        prompt_sha256 = payload.get("prompt_sha256")
        if not isinstance(prompt_sha256, str) or not _HEX_SHA256.fullmatch(prompt_sha256):
            raise ConversationError("conversation_invalid")
        state = payload.get("state")
        if state not in {"completed", "refused", "failed"}:
            raise ConversationError("conversation_invalid")
        user_message = cls._text(payload.get("user_message"), 2000)
        assistant_message = payload.get("assistant_message")
        if not isinstance(assistant_message, str) or len(assistant_message) > 4000:
            raise ConversationError("conversation_invalid")
        interpretation_payload = payload.get("interpretation")
        interpretation = None if interpretation_payload is None else cls._validate_interpretation(interpretation_payload)
        error_code = payload.get("error_code")
        if error_code is not None:
            error_code = cls._text(error_code, 128)
        created_at = cls._parse_timestamp(payload.get("created_at"))
        completed_at = cls._parse_timestamp(payload.get("completed_at"))
        if completed_at < created_at:
            raise ConversationError("conversation_invalid")
        return CopilotTurnV1(
            turn_id=turn_id,
            client_message_id=client_message_id,
            context_revision=context_revision,
            provider=provider,
            model=model,
            prompt_version=prompt_version,
            prompt_sha256=prompt_sha256,
            state=state,
            user_message=user_message,
            assistant_message=assistant_message,
            interpretation=interpretation,
            error_code=error_code,
            created_at=payload["created_at"],
            completed_at=payload["completed_at"],
        )

    @classmethod
    def _validate_interpretation(cls, payload: Any) -> CopilotInterpretationV1:
        fields = {"schema_version", "state", "assistant_message", "intents", "permission", "highlights"}
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ConversationError("conversation_invalid")
        if payload.get("schema_version") != INTERPRETATION_SCHEMA_VERSION:
            raise ConversationError("conversation_invalid")
        state = payload.get("state")
        if state not in _INTERPRETATION_STATES:
            raise ConversationError("conversation_invalid")
        assistant_message = cls._text(payload.get("assistant_message"), 4000)
        intents_payload = payload.get("intents")
        if not isinstance(intents_payload, (list, tuple)) or len(intents_payload) > 4:
            raise ConversationError("conversation_invalid")
        intents = tuple(cls._validate_intent(item) for item in intents_payload)
        permission = payload.get("permission")
        if permission not in {None, "external_booking_change"}:
            raise ConversationError("conversation_invalid")
        highlights = cls._validate_highlights(payload.get("highlights"))
        return CopilotInterpretationV1(
            state=state,
            assistant_message=assistant_message,
            intents=intents,
            permission=permission,
            highlights=highlights,
        )

    @classmethod
    def _validate_intent(cls, payload: Any) -> CopilotIntentV1:
        fields = {
            "type",
            "target_stop_id",
            "destination_day",
            "candidate_id",
            "preference",
            "clarification_question",
        }
        if not isinstance(payload, dict) or set(payload) != fields or payload.get("type") not in _INTENT_TYPES:
            raise ConversationError("conversation_invalid")
        optional_text: dict[str, str | None] = {}
        for name in ("target_stop_id", "candidate_id", "clarification_question"):
            value = payload.get(name)
            optional_text[name] = None if value is None else cls._text(value, 512)
        day = payload.get("destination_day")
        if day is not None and (not isinstance(day, int) or isinstance(day, bool) or day < 1):
            raise ConversationError("conversation_invalid")
        preference = payload.get("preference")
        if preference is not None and preference not in _PREFERENCES:
            raise ConversationError("conversation_invalid")
        return CopilotIntentV1(
            type=payload["type"],
            target_stop_id=optional_text["target_stop_id"],
            destination_day=day,
            candidate_id=optional_text["candidate_id"],
            preference=preference,
            clarification_question=optional_text["clarification_question"],
        )

    @classmethod
    def _validate_highlights(cls, payload: Any) -> CopilotHighlightsV1:
        fields = {"day_ids", "stop_ids", "segment_ids", "candidate_ids"}
        if not isinstance(payload, dict) or set(payload) != fields:
            raise ConversationError("conversation_invalid")
        day_ids = payload.get("day_ids")
        if not isinstance(day_ids, (list, tuple)) or len(day_ids) > 128 or any(
            not isinstance(day, int) or isinstance(day, bool) or day < 1 for day in day_ids
        ):
            raise ConversationError("conversation_invalid")
        values: dict[str, tuple[str, ...]] = {}
        for name in ("stop_ids", "segment_ids", "candidate_ids"):
            raw = payload.get(name)
            if not isinstance(raw, (list, tuple)) or len(raw) > 128:
                raise ConversationError("conversation_invalid")
            values[name] = tuple(cls._text(item, 256) for item in raw)
        return CopilotHighlightsV1(
            day_ids=tuple(day_ids),
            stop_ids=values["stop_ids"],
            segment_ids=values["segment_ids"],
            candidate_ids=values["candidate_ids"],
        )

    @staticmethod
    def _validate_session(session: ProductSession) -> None:
        if not isinstance(session, ProductSession):
            raise ConversationError("invalid_session")
        ConversationRepository._require_session_id(session.session_id)
        ConversationRepository._text(session.trip_id, 256)
        ConversationRepository._text(session.run_id, 256)

    @staticmethod
    def _require_owner(conversation: ConversationV1, session_id: str) -> None:
        if conversation.session_id != session_id:
            raise ConversationError("conversation_session_mismatch")

    @staticmethod
    def _require_id(value: str) -> None:
        if not isinstance(value, str) or not _CONVERSATION_ID.fullmatch(value):
            raise ConversationError("invalid_conversation_id")

    @staticmethod
    def _require_session_id(value: str) -> None:
        if not isinstance(value, str) or not _SESSION_ID.fullmatch(value):
            raise ConversationError("invalid_session_id")

    @staticmethod
    def _text(value: Any, maximum: int) -> str:
        if not isinstance(value, str) or not value or len(value) > maximum:
            raise ConversationError("conversation_invalid")
        return value

    @staticmethod
    def _parse_timestamp(value: Any) -> datetime:
        if not isinstance(value, str) or not value:
            raise ConversationError("conversation_invalid")
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            raise ConversationError("conversation_invalid") from None
        if parsed.tzinfo is None or parsed.utcoffset() != timedelta(0):
            raise ConversationError("conversation_invalid")
        return parsed.astimezone(UTC)

    @staticmethod
    def _format_timestamp(value: datetime) -> str:
        return value.astimezone(UTC).isoformat()

    @staticmethod
    def _coerce_now(value: datetime) -> datetime:
        if not isinstance(value, datetime) or value.tzinfo is None:
            raise ConversationError("invalid_cleanup_time")
        return value.astimezone(UTC)

    def _now(self) -> datetime:
        return self._coerce_now(self._clock())

    @classmethod
    def _is_expired(cls, conversation: ConversationV1, now: datetime) -> bool:
        return cls._parse_timestamp(conversation.expires_at) <= now

    def _path(self, conversation_id: str) -> Path:
        self._require_id(conversation_id)
        return self.root / f"{conversation_id}.json"

    @staticmethod
    def _entry_exists(path: Path) -> bool:
        return path.exists() or path.is_symlink()

    @staticmethod
    def _message_hash(message: str) -> str:
        return hashlib.sha256(message.encode("utf-8")).hexdigest()


ConversationRepositoryError = ConversationError
