from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional
from urllib.parse import urlparse

from webtactix.core.schemas import EvalSpec, TaskSpec


@dataclass(frozen=True)
class OnlineMind2WebAdapter:
    """
    Reads Online-Mind2Web tasks from a local JSON file (a list of dicts).

    Example item:
      {
        "task_id": "...",
        "confirmed_task": "...",
        "website": "https://www.traderjoes.com/",
        "reference_length": 6,
        "level": "medium"
      }

    This adapter converts each item into TaskSpec.
    """
    data_path: Path
    limit: Optional[int] = None

    # How to map Online task_id (str) into TaskSpec.task_id (int)
    # - "index": use item index as int id, store original string id in instantiation_dict["task_id_raw"]
    # - "hash32": stable 32-bit hash from task_id string
    id_mode: str = "index"

    def iter_tasks(self) -> Iterator[TaskSpec]:
        items = self._load_items()
        if self.limit is not None:
            items = items[: self.limit]
        for idx, obj in enumerate(items):
            yield self._parse(obj, idx=idx, source_path=str(self.data_path))

    def _load_items(self) -> List[Dict[str, Any]]:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Online-Mind2Web json not found: {self.data_path}")

        try:
            data = json.loads(self.data_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # some files might be jsonl; try line-by-line
            data = []
            with self.data_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data.append(json.loads(line))

        if not isinstance(data, list):
            raise ValueError(f"Expected a list in {self.data_path}, got: {type(data)}")

        return data

    def _parse(self, obj: Dict[str, Any], idx: int, source_path: str) -> TaskSpec:
        task_id_raw = str(obj.get("task_id") or "").strip()
        website = str(obj.get("website") or "").strip()

        # Prefer confirmed_task, fallback to task_description if your file uses that field
        intent = str(
            obj.get("confirmed_task")
            or obj.get("task_description")
            or obj.get("intent")
            or ""
        ).strip()

        reference_length = int(obj.get("reference_length") or 0)
        level = str(obj.get("level") or "").strip()

        task_id_int = self._map_task_id(task_id_raw=task_id_raw, row_index=idx)
        site = self._infer_site(website)

        eval_spec = EvalSpec(
            eval_types=[],
            reference_answers={},
            reference_url="",
            string_note="",
            program_html=[],
            reference_answer_raw_annotation="",
            raw={
                "reference_length": reference_length,
                "level": level,
            },
        )

        return TaskSpec(
            dataset="online_mind2web",
            task_id=task_id_int,
            intent=intent,
            start_url=website,
            sites=[site] if site else [],
            require_login=False,
            storage_state_path=None,
            geolocation=None,
            require_reset=False,
            intent_template="",
            instantiation_dict={
                "task_id_raw": task_id_raw,
                "website": website,
                "reference_length": reference_length,
                "level": level,
            },
            eval_spec=eval_spec,
            source_path=source_path,
        )

    def _infer_site(self, website_url: str) -> str:
        try:
            u = urlparse(website_url)
            return (u.netloc or "").lower()
        except Exception:
            return ""

    def _map_task_id(self, task_id_raw: str, row_index: int) -> int:
        mode = (self.id_mode or "index").lower()

        if mode == "index":
            return int(row_index)

        if mode == "hash32":
            # FNV-1a 32-bit
            h = 2166136261
            for b in task_id_raw.encode("utf-8"):
                h ^= b
                h = (h * 16777619) & 0xFFFFFFFF
            return int(h)

        raise ValueError(f"Unknown id_mode: {self.id_mode!r}")


if __name__ == "__main__":
    # Assuming Online_Mind2Web.json is in the same directory as this script
    here = Path(__file__).resolve().parent
    adapter = OnlineMind2WebAdapter(data_path=here / "Online_Mind2Web.json", limit=3)

    for t in adapter.iter_tasks():
        print(
            {
                "dataset": t.dataset,
                "task_id": t.task_id,
                "task_id_raw": t.instantiation_dict.get("task_id_raw"),
                "site": t.sites,
                "start_url": t.start_url,
                "intent": t.intent[:80],
                "reference_length": t.instantiation_dict.get("reference_length"),
                "level": t.instantiation_dict.get("level"),
            }
        )
