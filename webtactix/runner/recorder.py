# webtactix/runner/recorder.py
from __future__ import annotations

import json
import time
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional
from webtactix.core.schemas import TaskSpec

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _to_jsonable(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, Enum):
        return x.value
    if is_dataclass(x):
        return _to_jsonable(asdict(x))
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return str(x)


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(obj), ensure_ascii=False, indent=2), encoding="utf-8")


def _append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(_to_jsonable(obj), ensure_ascii=False) + "\n")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


class Recorder:
    """
    Minimal structured recorder.

    Layout:

      record/<dataset>/task_<task_id>/
        meta.json
          - runtime config for this task run, for example key_num, headless, max_rounds
        task.json
          - task spec and extracted constraints
          - { "task": <TaskSpec>, "constraints": <List[Constraint]> }

        round_<000>/
          round.json
            - inputs of this round and round start time
            - { "round": int, "frontier": [NodeId], "f_parent": NodeId, "t_round_start": float }

          plan_<node_id>.json
            - planner output for this node, with usage and time
            - { "node_id": str, "t_end": float, "usage": {...}, "result": <PlanningResult> }

          decision.json
            - decision agent output with usage and time
            - { "t_end": float, "usage": {...}, "result": <DecisionResult> }

          actions.json
            - ordered execution logs, one JSON per line (jsonl)
            - contains from_node, to_node, action_sig, error, etc

          pages/
            <node_id>_actree.txt
              - EncodedObservation.actree_yaml as plain text
            <node_id>_snapshot.png
              - screenshot of the node's page
    """

    def __init__(
            self,
            *,
            base_dir: Path = Path("record"),
            task: TaskSpec = None,
            model_name: str = "unknown_model",
    ) -> None:
        if task is None:
            raise ValueError("Recorder requires a TaskSpec: task cannot be None")

        self.task = task
        self.dataset = task.dataset or "unknown_dataset"
        self.task_id = str(task.task_id)
        self.model_name = (model_name or "").strip() or "unknown_model"

        self.root = base_dir / self.dataset / self.model_name
        self.task_dir = self.root / f"task_{self.task_id}"
        self._data_dirs: Dict[str, Path] = {}

        shutil.rmtree(self.task_dir, ignore_errors=True)

        self.task_dir.mkdir(parents=True, exist_ok=True)

        self._t0_perf = time.perf_counter()
        self._t0_iso = _now_iso()

        self._cur_round: Optional[int] = None
        self._cur_round_dir: Optional[Path] = None

        # usage totals (auto accumulated)
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_tokens: int = 0
        self.estimated_calls: int = 0
        self.llm_calls: int = 0

        # counts (auto accumulated)
        self.planner_calls: int = 0
        self.decision_calls: int = 0
        self.data_calls: int = 0

        # used to compute per-call durations if you want
        self._plan_start_t: Dict[str, float] = {}
        self._decision_start_t: Optional[float] = None
        self._data_start_t: Dict[str, float] = {}

        # round stats
        self.round_count: int = 0
        self.planner_frontier_sizes: Dict[int, int] = {}

        # user may set later
        self.total_nodes: Optional[int] = None

    # ---------------- time helpers ----------------

    def t(self) -> float:
        """Seconds since task start."""
        return float(time.perf_counter() - self._t0_perf)

    # ---------------- meta / task info ----------------

    def write_meta(self, meta: Dict[str, Any]) -> None:
        payload = dict(meta)
        payload.setdefault("dataset", self.dataset)
        payload.setdefault("task_id", self.task_id)
        payload.setdefault("t0_iso", self._t0_iso)
        _write_json(self.task_dir / "meta.json", payload)

    def write_task_info(self, task: Any, *, constraints: Any = None) -> None:
        payload: Dict[str, Any] = {"task": task}
        if constraints is not None:
            payload["constraints"] = constraints
        _write_json(self.task_dir / "task.json", payload)

    # ---------------- summary ----------------

    def set_total_nodes(self, n: int) -> None:
        self.total_nodes = int(n)

    def _add_usage(self, usage: Dict[str, Any]) -> None:
        pt = int(usage.get("prompt_tokens", 0) or 0)
        ct = int(usage.get("completion_tokens", 0) or 0)
        tt = int(usage.get("total_tokens", 0) or (pt + ct))
        est = bool(usage.get("estimated", False))

        self.total_prompt_tokens += pt
        self.total_completion_tokens += ct
        self.total_tokens += tt
        self.llm_calls += 1
        if est:
            self.estimated_calls += 1

    # ---------------- final result ----------------

    def write_final(
        self,
        *,
        status: str,
        answer: str = "",
        final_node_id: Optional[str] = None,
        reason: str = "",
        eval_result: Any = None,
    ) -> None:
        """
        Write the final outcome of this task run.
        Saved at record/<dataset>/task_<task_id>/final.json
        """
        payload: Dict[str, Any] = {
            "time_utc": _now_iso(),
            "t_end": self.t(),
            "dataset": self.dataset,
            "task_id": self.task_id,
            "status": str(status),
            "answer": str(answer or ""),
            "final_node_id": (None if final_node_id is None else str(final_node_id)),
            "reason": str(reason or ""),
            "real_answer": self.task.eval_spec,
            "eval_result": eval_result,
            "summary": {
                "total_tokens": self.total_tokens,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "llm_calls": self.llm_calls,
                "estimated_calls": self.estimated_calls,
                "planner_calls": self.planner_calls,
                "decision_calls": self.decision_calls,
                "total_rounds": self.round_count,
                "total_nodes": self.total_nodes,
                "planner_frontier_sizes": self.planner_frontier_sizes,
            },
        }
        _write_json(self.task_dir / "final.json", payload)


    # ---------------- round lifecycle ----------------

    def start_round(self, round_idx: int, *, frontier: Any, f_parent: Any) -> None:
        self._cur_round = int(round_idx)
        self._cur_round_dir = self.task_dir / f"round_{round_idx:03d}"
        self.round_count = max(self.round_count, int(round_idx) + 1)

        frontier_list = list(frontier) if frontier is not None else []
        self.planner_frontier_sizes[int(round_idx)] = int(len(frontier_list))

        _write_json(self._cur_round_dir / "round.json", {
            "round": int(round_idx),
            "frontier": frontier_list,
            "f_parent": str(f_parent or ""),
            "t_round_start": self.t(),
            "time_utc": _now_iso(),
        })

    # ---------------- planner logging ----------------

    def plan_begin(self, node_id: str) -> None:
        self._plan_start_t[str(node_id)] = self.t()

    def save_plan(self, *, node_id: str, result: Any, usage: Dict[str, Any]) -> None:
        if self._cur_round_dir is None:
            return

        nid = str(node_id)
        t_end = self.t()
        t_start = self._plan_start_t.pop(nid, None)

        payload = {
            "node_id": nid,
            "t_end": t_end,
            "t_start": t_start,
            "duration": (t_end - t_start) if t_start is not None else None,
            "usage": usage,
            "result": result,
        }

        _write_json(self._cur_round_dir / f"plan_{nid}.json", payload)

        self.planner_calls += 1
        self._add_usage(usage)

    # ---------------- decision logging ----------------

    def decision_begin(self) -> None:
        self._decision_start_t = self.t()

    def save_decision(self, *, result: Any, usage: Dict[str, Any]) -> None:
        if self._cur_round_dir is None:
            return

        t_end = self.t()
        t_start = self._decision_start_t
        self._decision_start_t = None

        payload = {
            "t_end": t_end,
            "t_start": t_start,
            "duration": (t_end - t_start) if t_start is not None else None,
            "usage": usage,
            "result": result,
        }

        _write_json(self._cur_round_dir / "decision.json", payload)

        self.decision_calls += 1
        self._add_usage(usage)

    # ---------------- actions (jsonl) ----------------

    def log_action(self, item: Dict[str, Any]) -> None:
        if self._cur_round_dir is None:
            return

        out = dict(item)
        out.setdefault("time_utc", _now_iso())
        out.setdefault("t", self.t())
        out.setdefault("round", self._cur_round)
        _append_jsonl(self._cur_round_dir / "actions.json", out)

    # ---------------- pages artifacts ----------------

    def save_actree(self, *, node_id: str, actree_text: str) -> None:
        if self._cur_round_dir is None:
            return
        _write_text(self._cur_round_dir / "pages" / f"{node_id}_actree.txt", actree_text or "")

    def snapshot_path(self, *, node_id: str) -> Path:
        if self._cur_round_dir is None:
            p = self.task_dir / "pages" / f"{node_id}_snapshot.png"
        else:
            p = self._cur_round_dir / "pages" / f"{node_id}_snapshot.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ---------------- data_extraction logging ----------------

    def _data_dir(self, node_id: str) -> Path:
        if self._cur_round_dir is None:
            # 没有 round 时也能落盘
            d = self.task_dir / "data_extraction" / f"node_{str(node_id)}"
        else:
            d = self._cur_round_dir / "data_extraction" / f"node_{str(node_id)}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def data_begin(self, *, node_id: str, goal: str, url: str = "") -> None:
        nid = str(node_id)
        d = self._data_dir(nid)
        self._data_dirs[nid] = d
        self._data_start_t[nid] = self.t()

        _write_json(d / "begin.json", {
            "node_id": nid,
            "goal": str(goal or ""),
            "url": str(url or ""),
            "time_utc": _now_iso(),
            "t_start": self._data_start_t[nid],
            "round": self._cur_round,
        })

    def data_log_turn(
        self,
        *,
        node_id: str,
        turn_idx: int,
        stage: str,
        obj: Any = None,
        usage: Optional[Dict[str, Any]] = None,
        note: str = "",
        step: Any = None,
        action_sig: str = "",
        url: str = "",
        error: str = "",
    ) -> None:
        nid = str(node_id)
        d = self._data_dirs.get(nid) or self._data_dir(nid)

        record: Dict[str, Any] = {
            "time_utc": _now_iso(),
            "t": self.t(),
            "round": self._cur_round,
            "node_id": nid,
            "turn": int(turn_idx),
            "stage": str(stage),
            "url": str(url or ""),
            "note": str(note or ""),
            "step": step,
            "action_sig": str(action_sig or ""),
            "error": str(error or ""),
            "obj": obj,
        }
        if usage is not None:
            record["usage"] = usage

        _append_jsonl(d / "turns.jsonl", record)

        if usage is not None:
            self.data_calls += 1
            self._add_usage(usage)

    def data_save_actree(self, *, node_id: str, turn_idx: int, actree_text: str) -> None:
        nid = str(node_id)
        d = self._data_dirs.get(nid) or self._data_dir(nid)
        _write_text(d / f"turn_{int(turn_idx):03d}_actree.txt", actree_text or "")

    def data_snapshot_path(self, *, node_id: str, turn_idx: int) -> Path:
        nid = str(node_id)
        d = self._data_dirs.get(nid) or self._data_dir(nid)
        p = d / f"turn_{int(turn_idx):03d}.png"
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def data_end(
        self,
        *,
        node_id: str,
        extracted: str,
        done: bool,
        history: Any = None,
        notes: Any = None,
        reason: str = "",
    ) -> None:
        nid = str(node_id)
        d = self._data_dirs.get(nid) or self._data_dir(nid)

        t_end = self.t()
        t_start = self._data_start_t.pop(nid, None)

        _write_json(d / "final.json", {
            "time_utc": _now_iso(),
            "round": self._cur_round,
            "node_id": nid,
            "t_start": t_start,
            "t_end": t_end,
            "duration": (t_end - t_start) if t_start is not None else None,
            "done": bool(done),
            "reason": str(reason or ""),
            "extracted": str(extracted or ""),
            "notes": notes,
            "history": history,
        })
