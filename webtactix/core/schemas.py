from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class EvalSpec:
    eval_types: List[str] = field(default_factory=list)
    reference_answers: Dict[str, Any] = field(default_factory=dict)
    reference_url: str = ""
    string_note: str = ""

    # WebArena specific but very common
    program_html: List[Dict[str, Any]] = field(default_factory=list)
    reference_answer_raw_annotation: str = ""

    # keep everything else for forward compatibility
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskSpec:
    """
    Unified task representation. Core runtime should ONLY depend on this schema,
    never on dataset-specific JSON fields.
    """
    dataset: str
    task_id: int
    intent: str
    start_url: str

    sites: List[str] = field(default_factory=list)
    require_login: bool = False
    storage_state_path: Optional[str] = None
    geolocation: Any = None
    require_reset: bool = False

    intent_template: str = ""
    instantiation_dict: Dict[str, Any] = field(default_factory=dict)

    eval_spec: EvalSpec = field(default_factory=EvalSpec)

    # for traceability and offline analysis
    source_path: str = ""

    def storage_state_abs(self, webarena_root: Path) -> Optional[Path]:
        if not self.storage_state_path:
            return None
        p = Path(self.storage_state_path)
        return p if p.is_absolute() else (webarena_root / p).resolve()


# =========================
# Planning and semantic tree schemas
# Append only below this line
# =========================

NodeId = str

class ActionType(str, Enum):
    CLICK = "click"
    INPUT = "input"
    SELECT = "select"
    PRESS_ENTER = "press_enter"
    GOTO = "goto"


@dataclass(frozen=False)
class ActionStep:
    action: ActionType
    index: int = 0
    role: str = ""
    name: str = ""
    nth: int = 0
    role_nth: int = 0
    text: Optional[str] = ""
    node_id: Optional[str] = ""


@dataclass
class Plan:
    type: str
    name: str
    goal: str
    answer: str = "" # especially for data extraction
    partially_done: str = "" # especially for partially done
    go_back: bool = 0  # for go_back and reselect
    steps: Tuple[ActionStep, ...] = ()


@dataclass(frozen=False)
class SemanticNode:
    node_id: NodeId
    page_summary: str = ""
    url: str = ""


@dataclass(frozen=True)
class Edge:
    src: NodeId
    dst: NodeId
    plan: Plan
