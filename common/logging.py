"""Structured logging for the medical data platform.

Every pipeline stage emits JSON lines so that runs can be shipped to a log
aggregator and replayed for audit purposes. Medical data pipelines need an
audit trail: who ran what, over which dataset version, and what was rejected.

Example:
    >>> from common.logging import get_logger, configure_logging
    >>> configure_logging(level="INFO", json_logs=True)
    >>> log = get_logger(__name__)
    >>> log.info("ingest.start", extra={"context": {"n_files": 42}})
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_CONFIGURED = False

# Attributes the stdlib puts on every LogRecord. Anything not in this set was
# added by the caller and is therefore worth emitting as structured context.
_RESERVED: frozenset[str] = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "message",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)


class JsonFormatter(logging.Formatter):
    """Render a LogRecord as a single JSON line.

    Any keyword passed via ``extra=`` is merged into the payload, so callers can
    attach arbitrary structured context without a bespoke formatter per module.
    """

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class HumanFormatter(logging.Formatter):
    """Compact human-readable format for interactive terminal runs."""

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.fromtimestamp(record.created).strftime("%H:%M:%S")
        base = f"{ts} {record.levelname:<7} {record.name:<38} {record.getMessage()}"
        context = {
            k: v for k, v in record.__dict__.items() if k not in _RESERVED and not k.startswith("_")
        }
        if context:
            base += "  " + " ".join(f"{k}={v}" for k, v in context.items())
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str = "INFO", json_logs: bool = False) -> None:
    """Install the platform log handler on the root logger.

    Idempotent: safe to call from every entrypoint. Later calls only adjust the
    level, so a library import can never clobber an application's handler setup.

    Args:
        level: Standard logging level name, e.g. ``"DEBUG"``.
        json_logs: Emit JSON lines instead of the human-readable format.
    """
    global _CONFIGURED
    root = logging.getLogger()
    if _CONFIGURED:
        root.setLevel(level.upper())
        return

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_logs else HumanFormatter())
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
    # nibabel chatters about every header it reads; that noise buries real QC events.
    logging.getLogger("nibabel").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    _CONFIGURED = True


class SafeExtraAdapter(logging.LoggerAdapter):
    """Logger adapter that renames ``extra`` keys colliding with LogRecord fields.

    ``logging.Logger.makeRecord`` raises ``KeyError`` if ``extra`` contains a
    reserved name such as ``message``, ``module`` or ``name``. That turns a
    logging statement -- the one line that should never be able to fail -- into
    an exception that takes down the pipeline around it.

    Since structured context here is arbitrary caller data (a QC finding legitimately
    *has* a "message" field), collisions are a matter of when, not if. This adapter
    renames them to ``<key>_`` instead of raising.
    """

    def process(self, msg: Any, kwargs: Any) -> tuple[Any, Any]:
        extra = kwargs.get("extra")
        if extra:
            kwargs["extra"] = {(f"{k}_" if k in _RESERVED else k): v for k, v in extra.items()}
        return msg, kwargs


def get_logger(name: str) -> logging.LoggerAdapter:
    """Return a logger, configuring the platform defaults on first use.

    Returns a :class:`SafeExtraAdapter` so that structured context can carry any
    key without risking a ``KeyError`` from the stdlib.
    """
    if not _CONFIGURED:
        configure_logging()
    return SafeExtraAdapter(logging.getLogger(name), {})
