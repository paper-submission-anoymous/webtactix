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
        system = (
            "You need to extract explicit, checkable constraints from a user request as a web agent's input. "
            "Return JSON only."
        )
        user = (
            "Return JSON with key constraints.\n"
            "Each item: {kind, text}.\n"
            f"Request:\n{self.task.intent}"
        )

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
        return out