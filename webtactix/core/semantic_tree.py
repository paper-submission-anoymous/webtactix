from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple
from playwright.async_api import Page
from urllib.parse import urlparse, urlunparse

from webtactix.core.schemas import ActionStep, NodeId, Plan, SemanticNode, Edge, ActionType
from webtactix.preprocess.observation_encoder import EncodedObservation

@dataclass
class NodeState:
    """
    Runtime-only state for a node.
    url: current page url for this node
    replay_steps: in-page actions on this url that reached the node state
                 and do NOT change url. Used for fast replay.
    next_plans: plans produced by planner for this node
    failures: failed incoming action signatures recorded on parent node
    """
    url: str = ""
    replay_steps: Tuple[ActionStep, ...] = ()
    progress_analysis: str = ""
    next_plans: Tuple[Plan, ...] = ()
    page: Page = None
    reflection: List[str] = field(default_factory=list)
    extra_inforamtion: str = ""
    enc: EncodedObservation = field(default_factory=EncodedObservation)

class SemanticTree:
    def __init__(self) -> None:
        self.nodes: Dict[NodeId, SemanticNode] = {}
        self.edges: List[Edge] = []
        self._root_id: Optional[NodeId] = None
        self._next_id: int = 1

        # runtime state
        self.state: Dict[NodeId, NodeState] = {}

        # structural bookkeeping for history reconstruction
        self.parent: Dict[NodeId, Optional[NodeId]] = {}
        self.incoming_action_sig: Dict[NodeId, str] = {}
        # parent_id -> [child_id, ...]
        self.children_map: Dict[NodeId, List[NodeId]] = {}

    def new_node_id(self) -> NodeId:
        nid = f"v{self._next_id}"
        self._next_id += 1
        return nid

    def add_root(self, url: str = "", enc: EncodedObservation = None) -> NodeId:
        if self._root_id is not None:
            raise RuntimeError("Root already exists.")
        self.nodes["virtual"] = SemanticNode(node_id="virtual", url="")

        rid = self.new_node_id()
        self.nodes[rid] = SemanticNode(node_id=rid, url=url)
        self.state[rid] = NodeState(url=url,enc=enc)
        self._root_id = rid

        self.children_map.setdefault("virtual", []).append(rid)
        self.parent[rid] = "virtual"
        self.incoming_action_sig[rid] = ""
        self.children_map[rid] = []
        return rid

    def set_next_plans(self, node_id: NodeId, progress_analysis: str, plans: Sequence[Plan]) -> None:
        st = self.state.get(node_id)
        st.progress_analysis = progress_analysis
        if st is None:
            self.state[node_id] = NodeState(next_plans=tuple(plans))
        else:
            st.next_plans = tuple(plans)

    def get_next_plans(self, node_id: NodeId) -> Tuple[Plan, ...]:
        st = self.state.get(node_id)
        return tuple(st.next_plans) if st is not None else ()

    def get_url(self, node_id: NodeId) -> str:
        st = self.state.get(node_id)
        if st is not None and st.url:
            return st.url
        n = self.nodes.get(node_id)
        return n.url if n is not None else ""

    def get_replay_steps(self, node_id: NodeId) -> Tuple[ActionStep, ...]:
        st = self.state.get(node_id)
        return tuple(st.replay_steps) if st is not None else ()

    def get_incoming_action_sig(self, node_id: NodeId) -> str:
        return self.incoming_action_sig.get(node_id, "")

    @staticmethod
    def _sig_from_plan(plan: Plan) -> str:
        if not getattr(plan, "steps", None):
            if plan.name == "data_extraction":
                return f"data_extraction: {plan.goal} -> extracted_result(ground truth): {plan.answer}"
            elif plan.name == "partially_done":
                return f"partially_done: {plan.goal}\n"
        if plan.name != "web_operation":
            return " -> ".join([plan.name, plan.goal])
        parts = []
        for s in plan.steps:
            if s.text is None or s.text == "":
                parts.append(f"{s.action.value}({s.role} {s.name})")
            elif s.action == ActionType.GOTO or s.action == ActionType.PRESS_ENTER:
                parts.append(f"{s.action.value} {s.text}")
            else:
                parts.append(f"{s.action.value}({s.role} {s.name}, {s.text})")
        return " -> ".join(parts)

    def add_child(
        self,
        parent: NodeId,
        plan: Plan,
        child_enc: EncodedObservation = None,
        url_after: str = "",
        page: Page = None
    ) -> NodeId:
        """
        child_url is stored into SemanticNode for traceability.
        url_after is the actual page.url after executing plan.
        We use url change to decide replay_steps behavior.

        Rule:
          - If url changes (url_after != parent_url_before), child replay_steps resets to empty
          - If url same, child replay_steps = parent.replay_steps + plan.steps
        """
        parsed_url = urlparse(url_after)

        url_after = parsed_url._replace(fragment="").geturl()

        if parent not in self.nodes:
            raise KeyError(f"Unknown parent node: {parent}")

        child = self.new_node_id()
        self.nodes[child] = SemanticNode(node_id=child, url=url_after)

        self.edges.append(Edge(src=parent, dst=child, plan=plan))

        # structural links
        self.parent[child] = parent
        self.incoming_action_sig[child] = self._sig_from_plan(plan).strip()
        self.children_map.setdefault(parent, []).append(child)
        self.children_map.setdefault(child, [])

        # compute replay_steps
        parent_state = self.state.get(parent)
        same_url = bool(parent_state.url) and bool(url_after) and (url_after == parent_state.url)

        if same_url:
            replay_steps = tuple(parent_state.replay_steps) + tuple(plan.steps)
        else:
            replay_steps = ()

        self.state[child] = NodeState(
            url=url_after,
            replay_steps=replay_steps,
            enc=child_enc,
            page=page
        )

        return child

    def path_to_root(self, node_id: NodeId) -> List[NodeId]:
        """
        Returns [root, ..., node_id]. If node_id unknown, returns [].
        """
        if node_id not in self.nodes:
            return []
        path: List[NodeId] = []
        cur: Optional[NodeId] = node_id
        while cur is not None:
            path.append(cur)
            cur = self.parent.get(cur)
        path.reverse()
        return path

    def history_for_planner(self, node_id: NodeId, *, max_turns: int = 20) -> str | tuple[str, int]:
        """
        A compact history text that includes both actions and resulting page summaries.
        This is what your planner should see, instead of only shared_history_actions.
        """
        path = self.path_to_root(node_id)
        if not path:
            return "", 0
        items: List[str] = []
        for nid in path[1:]:
            sig = self.get_incoming_action_sig(nid)
            if "partially_done" in sig:
                items = []
            summ = (self.nodes.get(nid).page_summary if self.nodes.get(nid) else "") or ""
            summ = summ.strip().replace("\n", " ")
            if sig:
                if summ and "partially_done" not in sig:
                    items.append(f"{sig} | {summ}")
                else:
                    items.append(f"Last action: {sig}")
            else:
                items.append(f"Initial Page (no actions)|{summ}")
        if len(items) > max_turns:
            items = items[-max_turns:]
        return "\n".join(f"- {x}" for x in items), len(items)
