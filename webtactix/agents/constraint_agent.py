from __future__ import annotations

from dataclasses import dataclass
from typing import List
from webtactix.core.schemas import TaskSpec
from webtactix.llm.openai_compat import OpenAICompatClient

# ── profiler ──────────────────────────────────────────────────────────────
from webtactix.profiler import Profiler
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Constraint:
    text: str
    kind: str = "general"


class ConstraintAgent:
    def __init__(self, llm: OpenAICompatClient, task: TaskSpec, mode: str = "child") -> None:
        self.llm = llm
        self.task = task
        self.profiler = Profiler(mode)

    async def run(self) -> List[Constraint]:
        # ── PROFILER: preprocessing step ──────────────────────────────────
        _sid_pre = self.profiler.emit_step_start(
            stage     = "pre",
            step_name = "pre:constraint",
            agent     = "constraint",
        )
        # ─────────────────────────────────────────────────────────────────

        system = (
            "You need to extract explicit, checkable constraints from a user request as a web agent's input. "
            "Return JSON only."
        )
        user = (
            "Return JSON with key constraints.\n"
            "Each item: {kind, text}.\n"
            f"Request:\n{self.task.intent}"
        )

        # ── PROFILER: end preprocessing step ──────────────────────────────
        self.profiler.emit_step_end(
            _sid_pre,
            output_summary = {"system_chars": len(system), "user_chars": len(user)},
        )
        # ─────────────────────────────────────────────────────────────────

        # ── PROFILER: record call start ───────────────────────────────────
        _cid = self.profiler.emit_llm_start(
            step_name     = "constraint",
            model_name    = getattr(self.llm, "model", ""),
            system_prompt = system,
            user_prompt   = user,
        )
        # ─────────────────────────────────────────────────────────────────

        obj, usage = await self.llm.chat_json(system=system, user=user)

        # ── PROFILER: record call end with exact usage from LLM response ──
        self.profiler.emit_llm_end(_cid, step_name = "constraint", usage=usage, output_obj=obj)
        # ─────────────────────────────────────────────────────────────────

        # ── PROFILER: postprocessing step ─────────────────────────────────
        _sid_post = self.profiler.emit_step_start(
            stage         = "post",
            step_name     = "post:constraint",
            agent         = "constraint",
            input_summary = {"obj_type": type(obj).__name__},
        )
        # ─────────────────────────────────────────────────────────────────

        items = obj if isinstance(obj, list) else (obj.get("constraints") or obj.get("items") or [])

        out: List[Constraint] = []
        if isinstance(items, list):
            for it in items:
                if not isinstance(it, dict):
                    continue
                text = str(it.get("text") or "").strip()
                if not text:
                    continue
                kind = str(it.get("kind") or "general").strip() or "general"
                out.append(Constraint(text=text, kind=kind))

        # ── PROFILER: end postprocessing step ─────────────────────────────
        self.profiler.emit_step_end(
            _sid_post,
            output_summary = {"n_constraints": len(out)},
        )
        # ─────────────────────────────────────────────────────────────────

        return out