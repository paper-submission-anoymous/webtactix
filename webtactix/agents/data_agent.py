# webtactix/agents/data_agent.py
from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from webtactix.core.schemas import TaskSpec, ActionStep, ActionType
from webtactix.core.semantic_tree import SemanticTree
from webtactix.preprocess.observation_encoder import ObservationEncoder, EncodedObservation, ObservationEncoderConfig
from webtactix.llm.openai_compat import OpenAICompatClient
from webtactix.runner.recorder import Recorder
from webtactix.browser.playwright_session import PlaywrightSession, wait_for_page_stable
import sys
import subprocess

@dataclass
class DataExtractionAgentConfig:
    max_rounds: int = 20
    max_history_items: int = 20
    max_steps_per_round: int = 8

external_knowledge = '''
- Date format: **Month/Day/YYYY** (e.g., "1/31/2024").
- Brand and product type can be infer from product name.
- 0 or not found can also be a result.
- The descending order for dates means that the earlier dates are located at the top, you must combined with the table to verify.\n
- All task can ONLY operate under the website as follow. Following URL shows the homepage of these websites.
  1 REDDIT: http://127.0.0.1:9999
  2 GITLAB: http://127.0.0.1:8023
  3 SHOPPING: http://127.0.0.1:7770
  4 SHOPPING_ADMIN: http://127.0.0.1:7780
  5 OPENSTREETMAP: https://www.openstreetmap.org/ (For map task, you can use your external knowledge.)
'''

@dataclass(frozen=True)
class DataExtractionResult:
    done: bool
    answer: str
    notes: Tuple[str, ...]
    history: Tuple[str, ...]
    enc: EncodedObservation
    url: str = ""


class DataExtractionAgent:
    """
    Runs data_extraction on a single page state.

    Pattern:
      NOTE -> ACTION -> NOTE -> ACTION ...

    It does NOT build pages by itself when you pass page and already_replayed=True.
    """

    def __init__(
        self,
        task: TaskSpec,
        llm: OpenAICompatClient,
        tree: SemanticTree,
        sess: PlaywrightSession,
        rec: Recorder,
        cfg: Optional[DataExtractionAgentConfig] = None,
    ) -> None:
        self.task = task
        self.llm = llm
        self.tree = tree
        self.sess = sess
        self.rec = rec
        self.encoder = ObservationEncoder(ObservationEncoderConfig(table_max_rows=1e10))
        self.cfg = cfg or DataExtractionAgentConfig()

    async def run(
        self,
        *,
        node_state,  # NodeState
        node_id
    ) -> DataExtractionResult:

        goal = ""
        if getattr(node_state, "next_plans", None):
            p0 = node_state.next_plans[0] if len(node_state.next_plans) > 0 else None
            goal = (getattr(p0, "goal", "") or "").strip()

        page = await self.sess.new_page()
        await self.sess.goto(page, node_state.url)
        await wait_for_page_stable(page)

        self.rec.data_begin(node_id=node_id, goal=goal, url=node_state.url)
        # replay to reach the node state
        cur = await self.sess.get_snapshot(page)
        cur_enc = self.encoder.encode(cur_snapshot=cur)

        if cur_enc.roles == node_state.enc.roles and cur_enc.role_nums == node_state.enc.role_nums:
            print("[DATA][replay_no_need]")
        else:
            replay_steps = self.tree.get_replay_steps(node_id)
            for s in replay_steps:
                try:
                    await self.sess.apply_step(page, s, False)
                except Exception as e:
                    print("[DATA][replay_err]", e)
                    break

        notes: List[str] = []
        hist: List[str] = []  # alternating entries: NOTE: ... / ACTION: ...
        answer = ""
        done = False
        page_initial = await self.sess.get_snapshot(page)
        enc_initial = self.encoder.encode(cur_snapshot=page_initial)
        common_hist, _ = self.tree.history_for_planner(node_id)
        for turn_idx in range(self.cfg.max_rounds):
            print('[DATA EXTRACTION] ', turn_idx)
            await wait_for_page_stable(page)
            cur = await self.sess.get_snapshot(page)
            enc = self.encoder.encode(cur_snapshot=cur)

            # 1) LLM decides: record key info + next actions OR done with final extraction
            obj, usage = await self.llm.chat_json(
                system=self._system_prompt(),
                user=self._user_prompt(goal=goal, enc=enc, common_history=common_hist, history=hist, url=page.url),
            )

            parsed = self._parse_llm(obj)
            note = parsed.get("note", "").strip()
            if note:
                notes.append(note)
                hist.append(f"NOTE{turn_idx}: {note}")

            self.rec.data_log_turn(
                node_id=node_id,
                turn_idx=turn_idx,
                stage="llm",
                obj=obj,
                usage=usage,
                note=note,
                step=parsed.get("step", {}),
                url=page.url,
            )
            self.rec.data_save_actree(node_id=node_id, turn_idx=turn_idx, actree_text=enc.actree_yaml)
            try:
                await page.screenshot(path=self.rec.data_snapshot_path(node_id=node_id, turn_idx=turn_idx), full_page=True)
            except:
                pass

            if parsed.get("done", False):
                answer = (parsed.get("answer", "") or "").strip()
                done = True
                break

            step = parsed.get("step", {})
            if not step:
                # no steps and not done: stop to avoid infinite loop
                break

            if step['action'] == "go_back":
                await page.go_back()
                action_sig = "go back"
                hist.append(f"ACTION: GO BACK")

            elif step['action'] == "wait":
                await asyncio.sleep(30)
                hist.append("ACTION: wait 30s for page loading")

            elif step['action'] == "click":
                idx = step['index']
                role = enc.roles[idx]
                name_ = enc.names[idx]
                nth = enc.nums[idx]
                role_nth = enc.role_nums[idx]
                action = ActionStep(index=idx, action=ActionType.CLICK, role=role, name=name_, nth=nth, role_nth=role_nth)
                action_sig = self._sig_from_steps(action)
                try:
                    await self.sess.apply_step(page, action, False)

                    hist.append(f"ACTION: {action_sig}")
                    print(f'[DATA AGENT] Click successful: {action_sig}')
                except Exception as e:
                    error_type = type(e).__name__
                    error_msg = str(e)
                    error_sig = f"{action_sig} failed: {error_type}: {error_msg}"
                    hist.append(f"ERROR: {error_sig}")
                    print(f'[DATA AGENT ERROR] {error_sig}')
                    action_sig = f"{action_sig} failed: {error_type}: {error_msg}"

            elif step['action'] == "goto":
                try:
                    URL = step.get("URL", "about:blank")
                    await page.goto(URL, timeout=60000)
                    await wait_for_page_stable(page)
                    action_sig = f"goto {URL}"
                    hist.append(f"ACTION: GOTO {URL}")
                    print(f'[DATA AGENT] goto {URL}')
                except Exception as e:
                    error_type = type(e).__name__
                    error_msg = str(e)
                    error_sig = f"goto {URL} failed: {error_type}: {error_msg}"
                    hist.append(f"ERROR: {error_sig}")
                    print(f'[DATA AGENT ERROR] {error_sig}')
                    action_sig = f"ERROR: {error_sig}"

            elif step['action'] == "code":
                # analyze = parsed.get("analyze", "").strip()
                raw = step.get("executable_code", "") or ""
                code = self._strip_code_fence(raw)

                stdout, stderr = self._run_code_capture(code, timeout_s=5.0)
                out_text = "\n".join([t for t in [stdout, stderr] if t]).strip() or "(no output)"

                hist.append(f"ACTION: Write code and execute.\n CODE:\n{code}\n Code's output:\n {out_text}\n")

                action_sig = f"code's output:\n{out_text}"
            else:
                print(f"[DATA AGENT] {step['action']}")
                action_sig = ""

            self.rec.data_log_turn(
                node_id=node_id,
                turn_idx=turn_idx,
                stage="action",
                action_sig=action_sig,
                url=page.url,
            )

            # trim history
            if len(hist) > self.cfg.max_history_items:
                hist = hist[-self.cfg.max_history_items :]

        if not answer:
            answer = "\n".join(notes).strip()
        print(f'[DATA EXTRACT] {answer}')

        return DataExtractionResult(
            done=done,
            answer=answer,
            notes=tuple(notes),
            history=tuple(hist),
            enc=enc_initial,
            url=page.url
        )

    @staticmethod
    def _system_prompt() -> str:
        return (
            "You are a data extraction agent.\n"
            "You must record key information and, if needed, take actions to collect more evidence.\n"
            "Return JSON only."
        )

    def _user_prompt(self, *, goal: str, enc: EncodedObservation, common_history:str, history: List[str], url: str) -> str:

        h = "\n".join([common_history] + history).strip()

        return (
            "User's Task(groundtruth):"
            f"{self.task.intent}"
            "current goal:\n"
            f"{goal}\n\n"

            "Output JSON schema:\n"
            "{\n"
            '  "analyze": string,\n'
            '  "note": string,\n'
            '  "done": boolean,\n'
            '  "answer": string,\n'
            '  "step": \n'
            "    {\n"
            '      "action": "click|goto|code|go_back|wait",\n'
            '      "index": int,\n'
            '      "URL": string,\n'
            '      "executable_code": string,\n'
            "    }\n"
            "}\n\n"
            "analyze's Rules:\n"
            "- Choose exactly one action.\n"
            "- Analyze the history(especially last code's result) and actree to provide an action to advance the tasks on this page\n"
            "- Do not repeat actions(especially code) that have already acted in history unless necessary, If you do, you must give reason to why perform repeated action.\n"
            "- You should treat the code's result as ground truth and do not need further verification.\n"
            "- 'analyze' Must be brief and short.\n\n"
            "note's Rules:\n"
            "Give the differences between this page and the previous one of what is visible on the current page that is relevant to the task in detail.\n"
            "code's Rules:\n"
            "- ONLY use 'code' to handle tasks related to complex MATH calculation, not text tasks. But you need to first copy the useful information from actree into your code.\n"
            "- The code must be executable python code with no input and no variables(including the actree) other than those defined by yourself and code's output should also provide explanation. All the information needs to be defined by yourself.\n"
            "- Treat the code's output as ground truth that do not need further coding.\n"
            "goto's Rules:\n"
            "- Try using <goto> action if you can infer the destination page's URL(http(s)://...) rather than click.\n"
            "wait's Rules:\n"
            "- If you think the page is loading and you need to wait for it to finish loading, use 'wait' to wait for 30 seconds."
            "answer's Rules:\n"
            "- Set done=true to give the final output and put the final answer and its explanation in 'answer'.\n"
            "- If the task cannot be completed correctly within current page(The details or editing page, which is a derivative of the current page, is also considered part of the current page.) because extra information or other navigation need to be acted, set done=true and explain(must mention cannot be done just in current page, not a negation of the entire task ). \n"
            "- If completing the task would require an unreasonable amount of repetitive clicking or navigation, for example more than 10 items to open one by one, set done=true and explain in answer. \n"
            "- If the task requires obtaining the answer from an extremely long table(>100 rows) and it is clearly possible to set up a filter to optimize the extraction process, set done=true and explain in answer. You can't set filters yourself\n"
            "- If you want to apply filters, set done=true and ask PLANNER to do this in 'answer'.\n\n"
            "- If the output result is more than 10, only the URL(display on the browser) where the result is located and a description are required.\n\n"
            # f"TIPS:\n {external_knowledge}\n\n"
            f"History (older to newer):\n{h}\n\n"
            "actree:\n"
            f"{enc.actree_yaml}\n\n"
            f"current URL: {url}\n\n"
        )

    def _parse_llm(self, obj: Any) -> Dict[str, Any]:
        out: Dict[str, Any] = {"note": "", "done": False, "answer": "", "steps": []}

        if not isinstance(obj, dict):
            return out

        out["note"] = str(obj.get("note") or "").strip()
        out["done"] = bool(obj.get("done", False))
        out["answer"] = str(obj.get("answer") or "").strip()

        step = obj.get("step", {})

        # enforce schema consistency
        if out["done"]:
            out["step"] = {}
        else:
            out["answer"] = ""
            out["step"] = step

        return out

    @staticmethod
    def _sig_from_steps(step: ActionStep) -> str:
        parts: List[str] = []
        act = getattr(step.action, "value", None) or str(step.action)
        if step.name:
            tgt = f'{step.role}("{step.name}")'
        else:
            tgt = f"{step.role}"
        txt = (step.text or "").strip()
        if txt and act in {"input", "select"}:
            parts.append(f"{act}({tgt},{txt})")
        else:
            parts.append(f"{act} {step.index} ({tgt})")
        return " -> ".join(parts)

    @staticmethod
    def _strip_code_fence(code: str) -> str:
        s = (code or "").strip()
        if s.startswith("```"):
            lines = s.splitlines()
            if lines and lines[0].strip().startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            s = "\n".join(lines).strip()
        return s

    @staticmethod
    def _run_code_capture(code: str, *, timeout_s: float = 5.0) -> Tuple[str, str]:
        try:
            p = subprocess.run(
                [sys.executable, "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout_s,
            )
            return (p.stdout or "").strip(), (p.stderr or "").strip()
        except subprocess.TimeoutExpired:
            return "", f"TIMEOUT after {timeout_s}s"
        except Exception as e:
            return "", f"RUN_ERROR: {e}"