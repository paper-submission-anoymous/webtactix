from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional

from webtactix.core.schemas import EvalSpec, TaskSpec


@dataclass(frozen=True)
class WebArenaAdapter:
    """
    Reads tasks from WebArena official repo without modifying it.

    We require that somewhere under `webarena_root` there exists:
      config_files/
        0.json, 1.json, ...
    """
    webarena_root: Path
    limit: Optional[int] = None

    def iter_tasks(self) -> Iterator[TaskSpec]:
        for fp in self._task_files():
            obj = json.loads(fp.read_text(encoding="utf-8"))
            yield self._parse(obj, source_path=str(fp))

    def _find_config_dir(self) -> Path:
        direct = self.webarena_root / "config_files"
        if direct.exists() and direct.is_dir():
            return direct

        # Find config_files within a small search space under webarena_root
        hits = [p for p in self.webarena_root.rglob("config_files") if p.is_dir()]
        if not hits:
            raise FileNotFoundError(f"config_files not found under: {self.webarena_root}")

        # Prefer the shallowest match
        hits.sort(key=lambda p: len(p.parts))
        return hits[0]

    def _task_files(self) -> List[Path]:
        base = self._find_config_dir()

        files = sorted(
            base.glob("*.json"),
            key=lambda p: (p.stem.isdigit(), int(p.stem) if p.stem.isdigit() else p.stem),
        )
        if not files:
            raise FileNotFoundError(f"No task json found under: {base}")

        if self.limit is not None:
            files = files[: self.limit]
        return files

    def _parse(self, obj: dict, source_path: str) -> TaskSpec:
        eval_obj = obj.get("eval") or {}
        eval_spec = EvalSpec(
            eval_types=list(eval_obj.get("eval_types") or []),
            reference_answers=dict(eval_obj.get("reference_answers") or {}),
            reference_url=str(eval_obj.get("reference_url") or ""),
            string_note=str(eval_obj.get("string_note") or ""),
            program_html=list(eval_obj.get("program_html") or []),
            reference_answer_raw_annotation=str(eval_obj.get("reference_answer_raw_annotation") or ""),
            raw=dict(eval_obj),
        )

        return TaskSpec(
            dataset="webarena",
            task_id=int(obj.get("task_id", -1)),
            intent=str(obj.get("intent") or ""),
            start_url=str(obj.get("start_url") or ""),
            sites=list(obj.get("sites") or []),
            require_login=bool(obj.get("require_login", False)),
            storage_state_path=str(obj.get("storage_state") or "") or None,
            geolocation=obj.get("geolocation", None),
            require_reset=bool(obj.get("require_reset", False)),
            intent_template=str(obj.get("intent_template") or ""),
            instantiation_dict=dict(obj.get("instantiation_dict") or {}),
            eval_spec=eval_spec,
            source_path=source_path,
        )
