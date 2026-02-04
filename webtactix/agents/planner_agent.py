from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from webtactix.core.schemas import ActionStep, ActionType, Plan
from webtactix.llm.openai_compat import OpenAICompatClient
from webtactix.core.semantic_tree import SemanticTree
from webtactix.core.schemas import NodeId
from webtactix.agents.constraint_agent import Constraint
from webtactix.runner.recorder import Recorder

_ALLOWED_PLAN_NAMES = {"web_operation", "data_extraction", "partially_done", "go_back", "finish"}
_ALLOWED_ACTIONS = {"click", "input", "select", "press_enter", "goto"}

TIPS = '''
- Date format in shopping-admin: **Month/Day/YYYY** (e.g., "1/31/2024").\n
- Date format in gitlab: **Year-Month-Day** (e.g., "2024-01-01").\n
- All the data on shopping website falls within the time span from January 1, 2022 to December 31, 2023.\n
- All task can ONLY operate under the website as follow. Following URL shows the homepage of these websites.\n
  1 REDDIT: http://127.0.0.1:9999\n
  2 GITLAB: http://127.0.0.1:8023\n
  3 SHOPPING: http://127.0.0.1:7770\n
  SHOPPING ACCOUNT: \n
        "username": "emma.lopez@gmail.com",\n
        "password": "Password.123",\n
  4 SHOPPING_ADMIN: http://127.0.0.1:7780/admin ("username": "admin", "password": "admin1234")\n
  5 OPENSTREETMAP: https://127.0.0.1:3000 (For map task, you can use your external knowledge.)\n
  6 wikipedia: http://127.0.0.1:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing\n
- Never click link or elements like <Download> or <Export> or <log out>, which will download sth on local that is forbid.\n
- Brand and product type can be infer from product name. All reviews can be found under Marketing section.\n
- The descending order(↓) for 'purchase dates' means that the earlier dates(oldest) are located at the top, newest at the bottom.\n
- Ask for product recommendations, should posts new comments or submissions for the product\n
- In GitLab, the edit page(including license) can be accessed via the URL pattern `http://127.0.0.1:8023/-/edit/master/<file>`)\n
'''

def _infer_type_from_name(name: str) -> str:
    if name in {"finish", "go_back"}:
        return "A"
    if name in {"data_extraction"}:
        return "B"
    return "C"


def _parse_action_type(s: str) -> Optional[ActionType]:
    s = (s or "").strip().lower()
    if s == "click":
        return ActionType.CLICK
    if s == "input":
        return ActionType.INPUT
    if s == "select":
        return ActionType.SELECT
    if s in {"press_enter", "pressenter", "enter"}:
        return ActionType.PRESS_ENTER
    if s in {"goto"}:
        return ActionType.GOTO
    return None


@dataclass(frozen=False)
class PlanningResult:
    page_summary: str
    progress_analysis: str
    plans: Tuple[Plan, ...]


@dataclass(frozen=True)
class PlannerAgentConfig:
    max_plans: int = 4
    max_history_actions: int = 10
    max_history_summaries: int = 15


class PlannerAgent:
    """
    Planner outputs plan "name" and optional action steps only.
    Engineering infers Plan.type (A/B/C) from plan name.
    """

    def __init__(
            self,
            llm: OpenAICompatClient,
            q: str,
            constraints: Optional[Sequence[Constraint]] = None,
            tree: SemanticTree = None,
            rec: Recorder = None,
            cfg: Optional[PlannerAgentConfig] = None,
    ) -> None:
        self.llm = llm
        self.q = q
        self.constraints = list(constraints or [])
        self.tree = tree
        self.rec = rec
        self.cfg = cfg or PlannerAgentConfig()

    async def run(self, node_id: NodeId, _round: int) -> PlanningResult:
        st = self.tree.state.get(node_id)
        parent_id = self.tree.parent.get(node_id, None)
        parent_st = self.tree.state.get(parent_id, None)

        actree_yaml = (st.enc.actree_yaml if st is not None else "") or ""
        history_text, len_hist = self.tree.history_for_planner(node_id, max_turns=self.cfg.max_history_actions)

        system = (
            "You are a web automation planner that is brave in trying and good at exploring..\n"
            "Follow the output schema and rules.\n"
            "Return JSON only."
        )

        user = (
            "You are given a user request, shared history context, and the current page accessibility tree actree_yaml.\n"
            "You must choose EXACTLY ONE plan category as plan type for this turn.\n\n"
            "Plan categories:\n"
            "1) web_operation\n"
            "2) data_extraction\n"
            "3) partially_done\n"
            "4) go_back\n"
            "5) finish\n\n"
            "Mutual exclusivity:\n"
            "You MUST output plans from only ONE category above.\n"
            "Do NOT mix categories in the same response.\n"
            "Only web_operation may include multiple alternative plans.\n"
            "Output JSON schema:\n"
            "{\n"
            '  "page_summary": string,\n'
            '  "progress_analysis": string,\n'
            '  "plans": [\n'
            "    {\n"
            '      "name": "web_operation" | "data_extraction" | "partially_done" | "go_back" | "finish",\n'
            '      "goal": string,\n'
            '      "steps": [ {"action":"click|select|input|goto|press_enter","index":int|null,"text":string|null} ]\n'
            "    },\n"
            "  ]\n"
            "}\n\n"
            "page_summary requirement:\n"
            "Give the differences between this page and the previous summary of what is new on the current page that is relevant to the task, summarize them in detail(including provided <extra information> and relevant URL and exact numbers). The summary must be facts and mustn't contain any subjective information.\n "
            "Begin with 'The page'. Never mention the elements's [index] because this will misleading due to page change, only mention name and role. The content in page summary must relevant to the task.\n\n"
            "progress_analysis requirement:\n"
            "- Analyze the what have done in the history(especially last data_extraction's result). Analyze the current actree to provide ALL possible ways(parallel plans, not sequential) to advance the tasks on this page using valid action space or summarize important information using partially done.\n"
            "- Choose exactly one category for this turn. The category must match every 'plans.name' you output .\n"
            "- Do not repeat repetitive actions(especially extract same content using data_extraction) that have already acted in history unless necessary, If you do, you must give reason to why perform repeated action.\n"
            "- You should treat the data_extraction and partially_done result as ground truth and do not need further verification.\n"
            "- progress_analysis must be short and brief."
            "Rules for plans:\n"
            "1) web_operation\n"
            "   - Use when you need take actions on THIS page now.\n"
            "   - If there are multiple plausible next actions on this page, you must output all alternative web_operation plans. However, when it comes to status changes, filters setting or irreversible actions such as submissions, only one plan can be output.\n"
            "   - Each web_operation plan is an action or a sequence of actions executed on the current page.\n"
            "   - For <select>, use only for elements with role 'listbox' and 'combobox', 'text' must be a visible option. For 'click|input', 'index' must be a index visible in the current 'actree'. For 'goto', leave index empty.\n"
            "   - All actions in a plan must be directly executable on the current page state (current actree); do not include steps that require expanding/revealing UI first.\n"
            "   - <goto> is an effective operation since it does not need to interact. Try using <goto> action if you can infer the destination page's URL(http(s)://...) as an alternative way. If click a link with an url, use <goto> to reach that page.\n"
            "   - steps MUST be non-empty.\n"
            "2) data_extraction\n"
            "   - Use ONLY after you have finished ALL navigation and filtering before extract some unknown information. The number of click actions that need to be performed to extract information should not exceed 15.\n"
            "   - Use this when the remaining work is to extract information through multi-pages tables, multi-pages web page, long table, or click some items one by one to view their details.\n"
            "   - If the information is obvious and easily to get, do not use this type"
            "   - Output exactly ONE detailed plan and the plan's goal can only contain the final information you want and mustn't contain any subjective instruction.\n"
            "3) partially_done\n"
            "   - Use this when you found important information or partially completed the task. With this type, you can compress the complex historical records into a concise completed description, along with the tasks that still need to be accomplished below.\n"
            "   - A typical usage scenario is a multi-stage task. Once a stage is completed, the completed part can be summarized. \n"
            "   - When use this plan, leave step empty and write in goal, the completed part must be ground truth and cannot be changed.\n"
            "   - Example: (1) We have already obtained the xxx information, next we will xxx based on this information. (2) We have already finish book a hotel(time), next we will buy ticket before the time."
            "4) go_back\n"
            "   - Use when the current page does not support useful actions for progress.\n"
            "   - If you want to go to previous page, please use category web operation's goto action.\n"
            "   - Output exactly ONE plan. Leave steps empty.\n"
            "5) finish\n"
            "   - Use when you can obtained the final answer from history and current observation or task completed.\n"
            "   - Output exactly ONE plan with the final answer in goal, only contain <the direct answer> without explanation or other text.\n"
            "TIPS: \n"
            "- Stop as soon as the user's request is satisfied (“good-enough” is correct).\n"
            "- If the user asks for ONE item/example, return the first valid match and DO NOT continue searching or comparing.\n"
            "- Do not apply extra filters or open extra details unless needed to produce the requested answer.\n"
            "- Do not verify across multiple candidates unless the user explicitly asks for “best / all / compare / exhaustive”.\n"
            "- For user task of <Viewing/showing/display/browse/get report/find out> or other similar task, just present the content the user needs on the page and describe what needs to do on this page is fine, never make unnecessary actions(extraction, page by page examine...).\n"
            "- If the relevant entries that meet the criteria have already been displayed on the webpage, there is no need to perform the filtering process.\n"
            "- Sometimes exact filters are not exist, you can make some deduction to identify the constraints instead of evidence.\n"
            "- 0, N/A, not found or unavaliable can also be consider as answer or result. \n"
            "- For user task, you can also use the external knowledge that you already knew.\n"
            f"{TIPS}\n\n"
            "User's task:\n"
            f"{self.q}\n\n"
        )
        if self.constraints:
            user += "Constraints:\n" + "\n".join(f"- [{c.kind}] {c.text}" for c in self.constraints) + "\n\n"

        if history_text.strip():
            user += "History Actions you have done (older to newer):\n" + history_text.strip() + "\n\n"
        else:
            user += "History (older to newer):\n This is the start website, no actions have been done before. You should make progress on the current page.\n\n"

        reflection_text = "\n".join(st.reflection)
        if len(st.reflection):
            user += (
                "Explanation of Reflection (from branches under the same current node):\n"
                "- This is a brief summary of what was tried and why it suggest go back.\n"
                "- In progress_analysis, you should explicitly consider this reflection.\n"
                "- When proposing plans, do NOT repeat actions that the reflection says acted, unless you also explain what is different now.\n\n"
                "Reflection:\n"
                f"{reflection_text}\n\n"
            )

        if parent_st and parent_st.extra_inforamtion.strip():
            print("[PLANNER EXTRA INFORMATION]", parent_st.extra_inforamtion)
            user += (
                "Extra information that can be treated as ground truth:\n"
                f"{parent_st.extra_inforamtion.strip()}\n\n"
            )
        print(f"[PLAN] LEN {_round}")

        if _round >= 13:
            user += "Since you exceed the maximum number of actions that can be executed, the task needs to be immediately terminated. Please use finish directly provide the answer.\n\n"
        elif 14 > _round >= 10:
            user += f"You can still execute {14 - _round} action(s) to finish the task. Use them carefully.\n\n"
        user += "current page's actree:\n" + actree_yaml + "\n"
        user += "current_url: " + st.url

        self.rec.plan_begin(node_id)

        obj, usage = await self.llm.chat_json(system=system, user=user)

        page_summary = ""
        progress_analysis = ""
        raw_plans: List[Dict[str, Any]] = []

        if isinstance(obj, dict):
            page_summary = str(obj.get("page_summary") or "").strip()
            progress_analysis = str(obj.get("progress_analysis") or "").strip()
            rp = obj.get("plans", [])
            if isinstance(rp, list):
                raw_plans = [x for x in rp if isinstance(x, dict)]
        elif isinstance(obj, list):
            raw_plans = [x for x in obj if isinstance(x, dict)]


        plans: List[Plan] = []
        for it in raw_plans:
            name = str(it.get("name") or "").strip()
            if name not in _ALLOWED_PLAN_NAMES:
                continue

            goal = str(it.get("goal") or "").strip()
            if not goal:
                continue

            steps_obj = it.get("steps", [])
            steps: List[ActionStep] = []

            if name == "web_operation":
                if not isinstance(steps_obj, list) or len(steps_obj) == 0:
                    continue

                need_goto = False

                for st2 in steps_obj:
                    if not isinstance(st2, dict):
                        continue

                    act_s = str(st2.get("action") or "").strip().lower()
                    if act_s not in _ALLOWED_ACTIONS:
                        if act_s == "wait":
                            print("[PLANNER] WAIT 30s")
                            await asyncio.sleep(30)
                        continue

                    a = _parse_action_type(act_s)
                    if a is None:
                        continue

                    idx_raw = st2.get("index", None)

                    def _idx_invalid(x) -> bool:
                        if x is None:
                            return True
                        if x == "":
                            return True
                        try:
                            ix = int(x)
                        except Exception:
                            return True
                        return ix < 0

                    if a == ActionType.CLICK and _idx_invalid(idx_raw):
                        need_goto = True
                        continue

                    idx = None
                    if idx_raw is not None and idx_raw != "":
                        try:
                            idx = int(idx_raw)
                        except Exception:
                            idx = None

                    if idx is not None and len(st.enc.roles) > 0:
                        if idx >= len(st.enc.roles):
                            return None

                        text = st2.get("text", "")
                        role = st.enc.roles[idx]
                        name_ = st.enc.names[idx]
                        nth = st.enc.nums[idx]
                        role_nth = st.enc.role_nums[idx]
                        steps.append(ActionStep(
                            index=idx, action=a, role=role, name=name_,
                            nth=nth, role_nth=role_nth, text=text, node_id=node_id
                        ))
                    elif len(st.enc.roles) == 0:
                        print("[PLAN IDX] EMPTY Page")
                    else:
                        text = st2.get("text", "")
                        steps.append(ActionStep(action=a, text=text, node_id=node_id))

                    if a == ActionType.CLICK and not _idx_invalid(idx_raw):
                        break

                if len(raw_plans) == 0:
                    need_goto = True

                if need_goto:
                    print(f'[PLAN IDX ERR] {rp}')
                    cur_url = st.url or self.tree.get_url(node_id)
                    if cur_url:
                        steps = [ActionStep(action=ActionType.GOTO, text=cur_url, node_id=node_id)]
                    else:
                        steps = []

                if len(steps) == 0:
                    continue

            ptype = _infer_type_from_name(name)
            plans.append(Plan(type=ptype, name=name, goal=goal, steps=tuple(steps)))

            if len(plans) >= self.cfg.max_plans:
                break

        n = self.tree.nodes.get(node_id)
        if n is not None:
            n.page_summary = page_summary
        self.tree.set_next_plans(node_id, progress_analysis, plans)

        planning_result = PlanningResult(
            page_summary=page_summary,
            progress_analysis=progress_analysis,
            plans=tuple(plans)
        )

        self.rec.save_plan(node_id=node_id, result=planning_result, usage=usage)

        return planning_result

