from __future__ import annotations

import asyncio
import copy
import random
import traceback
from dataclasses import dataclass
from typing import List, Optional, Any

from webtactix.core.schemas import NodeId, Plan, ActionType
from webtactix.browser.playwright_session import PlaywrightSession, wait_for_page_stable
from webtactix.preprocess.observation_encoder import ObservationEncoder, EncodedObservation
from webtactix.core.semantic_tree import SemanticTree
from webtactix.core.schemas import ActionStep
from webtactix.runner.recorder import Recorder
from webtactix.agents.data_agent import DataExtractionAgent
from webtactix.datasets.webarena_evaluator import WebArenaEvaluator, EvalResult
from playwright.async_api import Error as PWError
# ── profiler IPC ──────────────────────────────────────────────────────────
from webtactix.profiler import Profiler



def _action_sig_from_plan(plan: Plan) -> str:
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


@dataclass(frozen=True)
class ExecuteOutcome:
    executed_plan: Plan
    action_sig: str
    ok: bool
    kind: str  # new_node | error | finish | skipped | data_extraction
    new_node_id: Optional[NodeId] = None
    new_url: str = ""
    error: str = ""
    extracted: str = ""
    partially_done: str = ""
    eval_result: Optional[EvalResult] = None


class Executor:
    def __init__(
        self,
        *,
        sess: PlaywrightSession,
        tree: SemanticTree,
        rec: Recorder = None,
        data_agent: DataExtractionAgent = None,
        evaluator: WebArenaEvaluator = None,
        max_parallel: int = 4,
        mode: str = "child",
    ) -> None:
        self.sess = sess
        self.tree = tree
        self.encoder = ObservationEncoder()
        self.rec = rec
        self.data_agent = data_agent
        self.evaluator = evaluator
        self.max_parallel = int(max_parallel)
        self.profiler = Profiler(mode)

    async def replay_to_node(self, *, page: Any, node_id: NodeId) -> ActionStep:
        # replay_to_node is always called as a sub-operation inside
        # run_one_web_operation_plan_in_fresh_tab which already has an exec span.
        # Tracking it separately would create double-counted time windows.
        # It is therefore NOT individually instrumented.
        url = self.tree.get_url(node_id)
        state = self.tree.state[node_id]
        if not url:
            raise RuntimeError(f"Missing url for node {node_id}")

        await self.sess.goto(page, url)

        cur = await self.sess.get_snapshot(page)
        cur_enc = await asyncio.to_thread(self.encoder.encode, cur_snapshot=cur)

        if cur_enc.actree_yaml == state.enc.actree_yaml:
            print(f"[EXEC][replay] skipped...")
            return None

        replay_steps = self.tree.get_replay_steps(node_id)

        print(f"[EXEC][replay] node={node_id} url={url} replay_steps={len(replay_steps)}")
        current_step = None
        for s in replay_steps:
            cur = await self.sess.get_snapshot(page)
            cur_enc = self.encoder.encode(cur_snapshot=cur)
            if cur_enc.actree_yaml == state.enc.actree_yaml:
                print(f"[EXEC][replay] skipped...")
                return current_step
            print(f"[EXEC][replay] replay: {s.action, s.role, s.name}")
            try:
                await self.sess.apply_step(page, s, True)
                current_step = copy.deepcopy(s)
            except Exception as e:
                print("[EXEC][replay_err]", e)
                continue
        return current_step

    async def _save_page_artifacts(self, *, page: Any, node_id: NodeId, enc: EncodedObservation) -> None:
        self.rec.save_actree(node_id=str(node_id), actree_text=enc.actree_yaml)
        path = self.rec.snapshot_path(node_id=str(node_id))
        try:
            full_page = True
            try:
                h = await page.evaluate(
                    "() => Math.max(document.body?.scrollHeight || 0, document.documentElement?.scrollHeight || 0)"
                )
                if isinstance(h, (int, float)) and h > 20000:
                    full_page = False
            except Exception:
                full_page = False

            await page.screenshot(
                path=path,
                full_page=full_page,
                timeout=8000,
                animations="disabled",
                caret="hide",
            )

        except asyncio.CancelledError:
            raise
        except PWError as e:
            print("screenshot error:", repr(e))
            try:
                if hasattr(page, "is_closed") and page.is_closed():
                    return
                await page.screenshot(
                    path=path,
                    full_page=False,
                    timeout=8000,
                    animations="disabled",
                    caret="hide",
                )
            except Exception as e2:
                print("screenshot fallback error:", repr(e2))
        except Exception as e:
            print("screenshot error:", repr(e))


    async def run_one_web_operation_plan_in_fresh_tab(
        self,
        *,
        selected_node_id: NodeId,
        plan: Plan,
        only: bool,
        _plan_idx: int = 0,   # index within the parallel batch (0-based), used for step_name
    ) -> ExecuteOutcome:
        action_sig = _action_sig_from_plan(plan)
        print(f"[DBG] task start action_sig={action_sig}")

        st = self.tree.state[selected_node_id]
        last_step = None

        # ── PROFILER: span start ──────────────────────────────────────────
        # We need url_before but the page doesn't exist yet if only=False.
        # Use the tree's recorded URL for this node as the starting point.
        url_before_known = self.tree.get_url(selected_node_id) or ""
        exec_id = self.profiler.emit_exec_start(
            exec_type  = "web_op",
            step_name  = f"exec:web_op|node={selected_node_id}|plan={_plan_idx}",
            node_id    = str(selected_node_id),
            action_sig = action_sig,
            url_before = url_before_known,
            plan_goal  = plan.goal,
        )
        # ─────────────────────────────────────────────────────────────────

        try:
            if not only or st.page is None:
                page = await self.sess.new_page()
                print(f"[DBG] got new_page action_sig={action_sig}")
                last_step = await self.replay_to_node(page=page, node_id=selected_node_id)
            else:
                page = st.page
                replay_steps = self.tree.get_replay_steps(selected_node_id)
                if len(replay_steps):
                    last_step = replay_steps[-1]

            url_before = page.url

            print(f"[EXEC] run web_operation from node={selected_node_id} url_before={url_before}")
            print(f"[EXEC] plan goal={plan.goal}")
            print(f"[EXEC] plan action_sig={action_sig}")

            if self.rec:
                self.rec.log_action({
                    "type": "exec_begin",
                    "src": str(selected_node_id),
                    "action_sig": action_sig,
                    "plan_name": plan.name,
                    "goal": plan.goal,
                })
            init = await self.sess.get_snapshot(page)
            enc = self.encoder.encode(cur_snapshot=init)
            steps = []
            for i in range(len(plan.steps)):
                s = plan.steps[i]
                print(f"[DBG] step={s}")
                try:
                    await self.sess.apply_step(page, s, False)
                    steps.append(s)
                except Exception as e:
                    steps.append(s)
                    plan.steps = steps
                    action_sig = _action_sig_from_plan(plan)
                    print("\n[EXEC][ERROR] plan failed")
                    print(f"[EXEC][ERROR] selected_node_id={selected_node_id}")
                    print(f"[EXEC][ERROR] plan_name={plan.name} goal={plan.goal}")
                    print(f"[EXEC][ERROR] action_sig={action_sig}")
                    print(f"[EXEC][ERROR] error={type(e).__name__}: {e}")

                    if self.rec:
                        self.rec.log_action({
                            "type": "exec_error",
                            "src": str(selected_node_id),
                            "action_sig": action_sig,
                            "error": f"{type(e).__name__}: {e}",
                        })

                    # ── PROFILER: span end (step failure) ─────────────────
                    self.profiler.emit_exec_end(
                        exec_id,
                        url_after   = page.url,
                        steps_count = len(steps),
                        success     = False,
                        error       = f"{type(e).__name__}: {e}",
                        kind        = "error",
                    )
                    # ─────────────────────────────────────────────────────

                    return ExecuteOutcome(
                        executed_plan=plan,
                        action_sig=action_sig,
                        ok=False,
                        kind="error",
                        error=f"{type(e).__name__}: {e}",
                    )

            plan.steps = steps

            url_after = page.url
            print(f"[EXEC] url_after={url_after}")

            cur_final = await self.sess.get_snapshot(page)
            enc_final = self.encoder.encode(cur_snapshot=cur_final)

            new_node = self.tree.add_child(
                parent=selected_node_id,
                plan=plan,
                child_enc=enc_final,
                url_after=url_after,
                page=page
            )
            await self._save_page_artifacts(page=page, node_id=new_node, enc=enc_final)
            if self.rec:
                self.rec.log_action({
                    "type": "exec_ok",
                    "src": str(selected_node_id),
                    "dst": str(new_node),
                    "action_sig": action_sig,
                    "url_after": url_after,
                })

            # ── PROFILER: span end (success) ──────────────────────────────
            self.profiler.emit_exec_end(
                exec_id,
                url_after   = url_after,
                steps_count = len(steps),
                new_node_id = str(new_node),
                success     = True,
                kind        = "new_node",
            )
            # ─────────────────────────────────────────────────────────────

            return ExecuteOutcome(
                executed_plan=plan,
                action_sig=action_sig,
                ok=True,
                kind="new_node",
                new_node_id=new_node,
                new_url=url_after,
            )

        except Exception as e:
            print('other Exception', e)
            # ── PROFILER: span end (outer exception) ──────────────────────
            self.profiler.emit_exec_end(
                exec_id,
                success = False,
                error   = f"{type(e).__name__}: {e}",
                kind    = "error",
            )
            # ─────────────────────────────────────────────────────────────
            return ExecuteOutcome(
                executed_plan=plan,
                action_sig=action_sig,
                ok=True,
                kind="new_node"
            )

    async def run_data_extraction(
        self,
        *,
        selected_node_id: NodeId,
        plan: Plan,
    ) -> ExecuteOutcome:

        node_state = self.tree.state[selected_node_id]
        node_id = self.tree.nodes[selected_node_id].node_id

        # ── PROFILER: exec span for new_page + stabilize ──────────────────
        _exec_setup = self.profiler.emit_exec_start(
            exec_type  = "data_extraction_setup",
            step_name  = f"exec:data_extraction_setup|node={node_id}",
            node_id    = str(node_id),
            action_sig = "new_page",
            url_before = "",
            plan_goal  = (plan.goal or "").strip(),
        )
        # ─────────────────────────────────────────────────────────────────

        page = await self.sess.new_page()
        await wait_for_page_stable(page)

        # ── PROFILER: end setup exec span ─────────────────────────────────
        self.profiler.emit_exec_end(
            _exec_setup,
            url_after   = page.url,
            steps_count = 1,
            success     = True,
            kind        = "data_extraction_setup",
        )
        # ─────────────────────────────────────────────────────────────────

        # data_agent.run() is internally instrumented with granular spans:
        #   exec:data_setup, pre:data|turn=N, llm_start/end,
        #   post:data|turn=N, exec:data_action
        result = await self.data_agent.run(node_state=node_state, node_id=node_id)
        plan.answer = result.answer

        new_node = self.tree.add_child(
            parent=selected_node_id,
            plan=plan,
            child_enc=result.enc,
            url_after=self.tree.get_url(selected_node_id),
        )

        if self.rec:
            self.rec.log_action({
                "type": "data_extraction_ok",
                "src": str(selected_node_id),
                "dst": str(new_node),
                "extracted_len": len(result.answer or ""),
            })

        return ExecuteOutcome(
            executed_plan=plan,
            action_sig=_action_sig_from_plan(plan),
            ok=True,
            kind="data_extraction",
            new_node_id=new_node,
            new_url=result.url,
            extracted=result.answer or "",
        )

    async def run_partially_done(
            self,
            *,
            selected_node_id: NodeId,
            plan: Plan,
    ) -> ExecuteOutcome:
        # Pure in-memory tree update — no browser I/O, no network, ~0ms.
        # Not instrumented: there is nothing to measure here.
        plan.partially_done = plan.goal
        new_node = self.tree.add_child(
            parent=selected_node_id,
            plan=plan,
            child_enc=self.tree.state[selected_node_id].enc,
            url_after=self.tree.get_url(selected_node_id),
        )

        if self.rec:
            self.rec.log_action({
                "type": "partially_done_ok",
                "src": str(selected_node_id),
                "dst": str(new_node),
                "compressed_content": plan.partially_done,
            })

        return ExecuteOutcome(
            executed_plan=plan,
            action_sig=_action_sig_from_plan(plan),
            ok=True,
            kind="",
            new_node_id=new_node,
            partially_done=plan.partially_done,
        )

    async def execute_next_plans(self, *, selected_node_id: NodeId) -> List[ExecuteOutcome]:
        next_plans = list(self.tree.get_next_plans(selected_node_id))
        if not next_plans:
            print(f"[EXEC] selected_node={selected_node_id} has no next_plans")
            return []

        for p in next_plans:
            if p.name == "finish":
                sig = _action_sig_from_plan(p)
                print(f"[EXEC] finish plan goal={p.goal}")
                last_url = self.tree.state[selected_node_id].url

                # ── PROFILER: evaluator call span ─────────────────────────
                # evaluator.evaluate() calls an external scoring endpoint —
                # it is a real external I/O call with measurable latency
                # and potential network traffic. Track it as its own span.
                exec_id = self.profiler.emit_exec_start(
                    exec_type  = "finish_eval",
                    step_name  = f"exec:finish_eval|node={selected_node_id}",
                    node_id    = str(selected_node_id),
                    action_sig = f"finish -> {p.goal}",
                    url_before = last_url or "",
                    plan_goal  = p.goal,
                )
                # ─────────────────────────────────────────────────────────

                if self.evaluator:
                    eval_result = await self.evaluator.evaluate(final_answer=p.goal, last_url=last_url)
                else:
                    eval_result = None

                # ── PROFILER: evaluator span end ──────────────────────────
                self.profiler.emit_exec_end(
                    exec_id,
                    url_after = last_url or "",
                    success   = True,
                    kind      = "finish",
                )
                # ─────────────────────────────────────────────────────────

                return [ExecuteOutcome(
                    executed_plan=p,
                    action_sig=sig,
                    ok=True,
                    kind="finish",
                    new_url=self.tree.get_url(selected_node_id),
                    eval_result=eval_result
                )]

        for p in next_plans:
            if p.name == "data_extraction":
                return [await self.run_data_extraction(selected_node_id=selected_node_id, plan=p)]
            elif p.name == "partially_done":
                return [await self.run_partially_done(selected_node_id=selected_node_id, plan=p)]

        web_ops: List[Plan] = [p for p in next_plans if p.name == "web_operation"]
        if not web_ops:
            return [ExecuteOutcome(
                executed_plan=p,
                action_sig=_action_sig_from_plan(p),
                ok=True,
                kind="skipped",
                new_url=self.tree.get_url(selected_node_id),
            ) for p in next_plans]

        # Pass plan index so each parallel web_op gets a unique step_name.
        # When len==1: only=True (reuse existing tab if possible).
        # When len>1:  only=False (each plan gets its own fresh tab).
        async def _run(p: Plan, only: bool, idx: int) -> ExecuteOutcome:
            return await self.run_one_web_operation_plan_in_fresh_tab(
                selected_node_id=selected_node_id,
                plan=p,
                only=only,
                _plan_idx=idx,
            )

        if len(web_ops) == 1:
            return await asyncio.gather(*[_run(p, True,  i) for i, p in enumerate(web_ops)])
        else:
            return await asyncio.gather(*[_run(p, False, i) for i, p in enumerate(web_ops)])