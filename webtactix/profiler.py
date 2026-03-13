"""webtactix/profiler.py

Central profiler helper.

``Profiler(mode)`` routes all emit calls to the real ``agent_ipc``
implementation when *mode* is ``"child"``, or to lightweight no-ops that
return ``None`` immediately when *mode* is ``"parent"``.

Usage in every agent / workflow::

    from webtactix.profiler import Profiler

    class MyAgent:
        def __init__(self, ..., mode: str = "child") -> None:
            ...
            self.profiler = Profiler(mode)

        def some_method(self):
            cid = self.profiler.emit_llm_start(...)
            ...
            self.profiler.emit_llm_end(cid, ...)
"""
from __future__ import annotations

from typing import Any


# ── No-op stubs (used when mode == "parent") ──────────────────────────────

def _noop(*_a: Any, **_kw: Any) -> None:
    return None


# Explicit named no-ops so the class attributes are self-documenting.
def emit_llm_start(*_a: Any, **_kw: Any) -> None:   # noqa: F811
    return None

def emit_llm_end(*_a: Any, **_kw: Any) -> None:     # noqa: F811
    return None

def emit_exec_start(*_a: Any, **_kw: Any) -> None:  # noqa: F811
    return None

def emit_exec_end(*_a: Any, **_kw: Any) -> None:    # noqa: F811
    return None

def emit_step_start(*_a: Any, **_kw: Any) -> None:  # noqa: F811
    return None

def emit_step_end(*_a: Any, **_kw: Any) -> None:    # noqa: F811
    return None


# ── Profiler class ─────────────────────────────────────────────────────────

class Profiler:
    """Routes profiler emit calls based on run *mode*.

    Parameters
    ----------
    mode:
        ``"child"``  — forward calls to the real ``agent_ipc`` functions.
        ``"parent"`` — replace all emit calls with no-ops (return ``None``).
    """

    def __init__(self, mode: str = "child") -> None:
        if mode == "child":
            import sys
            # agent-benchmark is the canonical path; agent-branchmark is a
            # legacy directory that may also contain agent_ipc.
            for _p in (
                "/home/ivohra6/agent-profiling/agent-benchmark",
                "/home/ivohra6/agent-profiling/agent-branchmark",
            ):
                if _p not in sys.path:
                    sys.path.append(_p)
            from agent_ipc import (  # type: ignore[import]
                emit_llm_start as _els,
                emit_llm_end as _ele,
                emit_exec_start as _ees,
                emit_exec_end as _eee,
            )
            self.emit_llm_start = _els
            self.emit_llm_end = _ele
            self.emit_exec_start = _ees
            self.emit_exec_end = _eee
        else:
            # parent mode — all calls are silent no-ops
            self.emit_llm_start = _noop
            self.emit_llm_end = _noop
            self.emit_exec_start = _noop
            self.emit_exec_end = _noop

        # These are no-ops for now in both modes (reserved for future use).
        self.emit_step_start = _noop
        self.emit_step_end = _noop
