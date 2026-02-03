# webtactix/agents/constraints_agent.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List
from webtactix.core.schemas import TaskSpec
from webtactix.llm.openai_compat import OpenAICompatClient


@dataclass(frozen=True)
class Constraint:
    text: str
    kind: str = "general"


class ConstraintAgent:
    def __init__(self, llm: OpenAICompatClient, task: TaskSpec) -> None:
        self.llm = llm
        self.task = task

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

        obj, usage = await self.llm.chat_json(system=system, user=user)
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
