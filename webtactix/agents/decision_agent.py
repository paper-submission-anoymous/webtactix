# webtactix/agents/decision_agent.py
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from webtactix.core.schemas import NodeId, Plan
from webtactix.core.priority_queue import PriorityQueue
from webtactix.llm.openai_compat import OpenAICompatClient
from webtactix.agents.constraint_agent import Constraint
from webtactix.core.semantic_tree import SemanticTree, NodeState, SemanticNode
from webtactix.browser.playwright_session import PlaywrightSession, wait_for_page_stable
from webtactix.runner.recorder import Recorder
from webtactix.workflows.execute import Executor
from webtactix.datasets.webarena_evaluator import EvalResult
from webtactix.preprocess.observation_encoder import ObservationEncoder, EncodedObservation

@dataclass(frozen=True)
class DecisionResult:
    """
    kind:
      - select_and_execute: pick a viable node v* (not all go_back) and execute its next plans later
      - reflect_and_replan: all go_back, and decide to replan on parent u with a reflection
      - reselect_and_execute: all go_back, and decide to abandon u, pop from global queue Q
    """
    kind: str
    selected_node_id: Optional[NodeId] = None
    reflection: str = ""
    reason: str = ""
    new_child: List[NodeId] = field(default_factory=list)
    eval_result: EvalResult = None


@dataclass(frozen=True)
class DecisionAgentConfig:
    max_candidates: int = 8
    max_next_plans_per_candidate: int = 6
    max_history_actions: int = 30
    max_history_summaries: int = 10



class DecisionAgent:
    def __init__(
            self,
            llm: OpenAICompatClient,
            q: str,
            constraints: Optional[Sequence[Constraint]] = None,
            executor: Executor = None,
            tree: SemanticTree = None,
            sess: PlaywrightSession = None,
            rec: Recorder = None,
            cfg: Optional[DecisionAgentConfig] = None,
    ) -> None:
        self.llm = llm
        self.q = q
        self.constraints = list(constraints or [])
        self.executor = executor
        self.tree = tree
        self.sess = sess
        self.rec = rec
        self.cfg = cfg or DecisionAgentConfig()
        self.queue = PriorityQueue()
        self.encoder = ObservationEncoder()

    @staticmethod
    def _is_all_go_back(candidates: Sequence[NodeState]) -> bool:
        if not candidates:
            return True
        for c in candidates:
            for p in c.next_plans:
                if p.name != "go_back":
                    return False
        return True

    def _candidate_payload(self, n: NodeState, c: SemanticNode) -> Dict[str, Any]:
        return {
            "node_id": c.node_id,
            "incoming_action": self.tree.incoming_action_sig[c.node_id],
            "page_summary": c.page_summary,
            "progress_analysis": n.progress_analysis,
            "next_plans": [{"name": p.name, "goal": p.goal} for p in n.next_plans],
        }

    async def reflect_or_reselect(
        self,
        *,
        parent_node_id: NodeId,
        candidates: Sequence[NodeState],
        next_nodes: Sequence[SemanticNode],
    ) -> DecisionResult:
        """
        Workflow 2 or 3.
        Precondition: all candidates are go_back.
        Ask the model whether the parent u is still worth exploring.
        If yes: return reflect_and_replan with reflection text.
        If no: pop from Q and return reselect_and_execute.
        """

        system = (
            "You are a decision agent for a long-horizon web task.\n"
            "Decide whether to re-plan at the parent node or reselect another branch from a global queue.\n"
            "Return JSON only."
        )

        payload = [self._candidate_payload(n, c) for n, c in zip(candidates, next_nodes)]
        payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
        hist, _ = self.tree.history_for_planner(parent_node_id)
        user = "".join([
            "All candidate nodes have only go_back as their next plan.\n"
            "You must decide whether the common parent node is still worth exploring. If not worth exploring, you can set explore_parent to 0 "
            "and jump to another node to continue.\n\n"
            "If explore_parent is true, you can write a reflection for the PLANNER if go back represent irrelevant page.\n"
            "If explore_parent is false, you can write reflection to record current history and current detailed information that is relevant to the task.\n"
            "Never mention element index [xx] in reflection, but can mention relevant element and URL.\n\n"
            "Output JSON schema:\n"
            '{ "explore_parent": boolean, "reflection": string, "reason": string}\n\n'
            f"user's task:\n{self.q}\n\n"
            "Constraints:\n"
            + "\n".join(f"- [{c.kind}] {c.text}" for c in self.constraints) + "\n\n"
            "parent_node_id:\n"
            f"{parent_node_id}\n\n"
            "parent_page_summary:\n"
            f"{self.tree.nodes[parent_node_id].page_summary}\n\n"
            "shared_history_actions:\n"
            f"{hist}\n\n"
            "candidates_json:\n"
            f"{payload_json}\n\n"
        ])

        if len(self.queue) == 0:
            user += "Notice: Since the backtrace nodes are empty you can only explore parent."

        obj, usage = await self.llm.chat_json(system=system, user=user)
        explore_parent = True
        reflection = ""
        reason = ""
        if isinstance(obj, dict):
            explore_parent = bool(obj.get("explore_parent", True))
            reflection = str(obj.get("reflection") or "").strip()
            reason = str(obj.get("reason") or "").strip()
            print(f'[DECISION AGENT REFLEX] {reflection}')

        exp_flag = False
        if not explore_parent and len(self.queue) == 0:
            exp_flag = True

        if explore_parent or exp_flag:
            selected = next_nodes[0].node_id
            state = self.tree.state[selected]
            page = state.page
            print("[DECISION AGENT] explore parent")
            await page.go_back()
            await wait_for_page_stable(page)
            cur = await self.sess.get_snapshot(page)
            cur_enc = self.encoder.encode(cur_snapshot=cur)
            plan = state.next_plans[0]

            new_node = self.tree.add_child(
                parent=next_nodes[0].node_id,
                plan=plan,
                child_enc=cur_enc,
                url_after=page.url,
                page=page
            )

            self.tree.state[new_node].reflection.append(reflection)
            self.rec.save_decision(result={"kind": "reflect_and_replan", "reflection": reflection, "reason": reason},
                                   usage=usage)
            return DecisionResult(kind="reflect_and_replan", selected_node_id=selected, reflection=reflection, reason=reason, new_child=[new_node])

        chosen = self.queue.pop()

        if chosen is None:
            raise EOFError
            # self.rec.save_decision(result={"kind": "reflect_and_replan", "reflection": "", "reason": "Queue empty."},
            #                        usage=usage)
            # return DecisionResult(
            #     kind="reflect_and_replan",
            #     selected_node_id="v1",
            #     new_child=['v1'],
            #     reason=reason or "Queue empty.",
            # )

        print("[DECISION AGENT] explore queue")
        exec_outcom = await self.executor.execute_next_plans(selected_node_id=chosen)
        new_child = [out.new_node_id for out in exec_outcom]

        for out in exec_outcom:
            print("[DECISION AGENT] ", out.new_node_id, out.kind)
            self.tree.state[out.new_node_id].reflection.append(reflection)

        self.rec.save_decision(result={"kind": "reselect_and_execute", "reason": reason}, usage=usage)
        return DecisionResult(kind="reselect_and_execute", selected_node_id=str(chosen), reason=reason, new_child=new_child)

    async def select_and_execute(
        self,
        *,
        parent_node_id: NodeId,
        candidates: Sequence[NodeState],
        next_nodes: Sequence[SemanticNode]
    ) -> DecisionResult:
        viable = [
            (n, c)
            for n, c in zip(candidates, next_nodes)
        ]

        if not viable:
            return DecisionResult(kind="reflect_and_replan", reflection="All candidates appear to be go_back.", reason="No viable candidates.")

        reason = ""  # do not delete
        extra_info = ""
        if len(next_nodes) > 1:
            system = (
                "You are a decision agent for a long-horizon web task.\n"
                "You must select which candidate page branch to continue.\n"
                "Return JSON only."
            )

            payload = [self._candidate_payload(n, c) for n, c in viable]
            payload_json = json.dumps(payload, ensure_ascii=False, indent=2)
            hist, _ = self.tree.history_for_planner(parent_node_id)
            user = "".join([
                "You are given candidate leaf nodes that share the same history up to a common parent.\n"
                "You must select ONE node to continue.\n\n"
                "You can see:\n"
                "- parent_node_id\n"
                "- parent_page_summary\n"
                "- shared_history_actions\n"
                "- for each candidate node v: incoming_action, page_summary, next_plans\n\n"
                "Selection criteria:\n"
                "Choose the single best candidate to continue.\n"
                "Then prioritize efficiency: among feasible branches, prefer partially done and prefer the one that can directly reach the goal with fewer steps, less detouring, and simpler interactions. Never choose go back. \n"
                "Prefer branches whose next_plans are concrete and directly move toward the answer, and with smaller workload when available.\n"
                "Other rules:\n"
                "Your 'reason' must brief and short to explain (1) why this branch can progress the task, and (2) why it can most satisfy the constraints.\n"
                "If the unselected action also contain useful information, summarize the actions and its outcome's task related part in detail; otherwise set extra_info to an empty string.\n\n"
                "Output JSON schema:\n"
                '{ "selected_node_id": string, "reason": string, "extra_info": string, "replay" }\n\n'
                f"user's task:\n{self.q}\n\n"
                "parent_node_id:\n"
                f"{parent_node_id}\n\n"
                "parent_page_summary:\n"
                f"{self.tree.nodes[parent_node_id].page_summary}\n\n"
                "shared_history_actions:\n"
                f"{hist}\n\n"
                "candidates_json:\n"
                f"{payload_json}\n"
            ])

            obj, usage = await self.llm.chat_json(system=system, user=user)

            selected: str = ""

            if isinstance(obj, dict):
                selected = obj.get("selected_node_id", None)
                reason = str(obj.get("reason") or "").strip()
                extra_info = str(obj.get("extra_info") or "").strip()
                state = self.tree.state[selected]
                state.extra_inforamtion += extra_info + "\n"

            selected = str(selected or "").strip()

            viable_ids = {c.node_id for _, c in viable}
            if selected not in viable_ids:
                raise Exception("Wrong choice for selected node")

            for _, c in viable:
                if c.node_id != selected:
                    self.queue.push(c.node_id)
        else:
            selected = next_nodes[0].node_id
            reason = "This is the only option."
            usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "estimated": False,
                "model": "Pass",
            }
        print(f'[DECISION] node {selected}')
        self.rec.save_decision(result={"kind": "select_and_execute", "selected": selected, "reason": reason, "extra_info": extra_info}, usage=usage)
        print(f'[DECISION] extra info: {extra_info}')
        exec_outcom = await self.executor.execute_next_plans(selected_node_id=selected)
        new_child = [out.new_node_id for out in exec_outcom if out.new_node_id]
        all_error = bool(exec_outcom) and all(o.kind == "error" for o in exec_outcom)
        if exec_outcom[0].kind == "finish":
            return DecisionResult(kind="finish", reason=exec_outcom[0].executed_plan.goal, eval_result=exec_outcom[0].eval_result)
        elif all_error:
            state = self.tree.state[selected]
            print("[EXEC] ALL ERROR")
            reflection = "The following actions have failed to execute on this page, here are the reasons:\n\n"
            for out in exec_outcom:
                reflection += f"{out.action_sig}:\n {out.error}\n\n"

            state.reflection.append(reflection)
            return DecisionResult(kind="select_and_execute", selected_node_id=parent_node_id, new_child=[selected],
                                  reason="All plans fail to execute.")
        if len(next_nodes) > 1:
            return DecisionResult(kind="select_and_execute", selected_node_id=selected, new_child=new_child, reason=reason)
        else:
            return DecisionResult(kind="select_and_execute", selected_node_id=selected, new_child=new_child,
                                  reason="Only one node, no need to decide.")

    async def run(
        self,
        parent_node_id: NodeId,
    ) -> DecisionResult:
        children_node_id = list(self.tree.children_map.get(parent_node_id, []))
        candidates = [self.tree.state[nid] for nid in children_node_id]
        next_nodes = [self.tree.nodes[nid] for nid in children_node_id]
        self.rec.decision_begin()
        if self._is_all_go_back(candidates):
            return await self.reflect_or_reselect(
                parent_node_id=parent_node_id,
                candidates=candidates,
                next_nodes=next_nodes
            )

        return await self.select_and_execute(
            parent_node_id=parent_node_id,
            candidates=candidates,
            next_nodes=next_nodes
        )