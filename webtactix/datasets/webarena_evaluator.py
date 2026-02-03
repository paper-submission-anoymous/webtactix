# webtactix/datasets/webarena_evaluator.py
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, is_dataclass
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit, urlparse

from jsonref import requests

from webtactix.core.schemas import TaskSpec
from webtactix.browser.playwright_session import PlaywrightSession, wait_for_page_stable

REDDIT = "http://127.0.0.1:9999"
SHOPPING = "http://127.0.0.1:7770"
SHOPPING_ADMIN = "http://127.0.0.1:7780/admin"
GITLAB = "http://127.0.0.1:8023"
WIKIPEDIA = "http://127.0.0.1:8888/wikipedia_en_all_maxi_2022-05/A/User:The_other_Kiwix_guy/Landing"
MAP = "https://www.openstreetmap.org/"
HOMEPAGE = "http://127.0.0.1:4399"

ACCOUNTS = {
    "reddit": {"username": "MarvelsGrantMan136", "password": "test1234"},
    "gitlab": {"username": "byteblaze", "password": "hello1234"},
    "shopping": {
        "username": "emma.lopez@gmail.com",
        "password": "Password.123",
    },
    "shopping_admin": {"username": "admin", "password": "admin1234"},
    "shopping_site_admin": {"username": "admin", "password": "admin1234"},
}


@dataclass(frozen=True)
class EvalResult:
    ok: bool
    score: float
    reason: str
    details: Dict[str, Any]

def shopping_get_sku_latest_review_rating(sku: str) -> str:
    """Get the latest review for shopping admin."""
    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }
    response = requests.get(
        f"{SHOPPING}/rest/V1/products/{sku}/reviews", headers=header
    )
    assert response.status_code == 200
    response_obj = response.json()
    if len(response_obj) == 0:
        return ""
    assert response_obj[0]["ratings"][0]["rating_name"] == "Rating"
    rating: str = str(response_obj[-1]["ratings"][0]["percent"])
    return rating

def reddit_get_post_url(url: str) -> str:
    """Get the post url"""
    # Url is http://domain/f/subreddit/post_id/...
    # get domain, subreddit, post_id
    domain = urlparse(url).netloc
    tok_url = urlparse(url).path.split("/")
    # not a valid post/comment url, return the url as is
    if len(tok_url) < 4:
        return url
    if tok_url[1] != "f":
        return url
    subreddit = urlparse(url).path.split("/")[2]
    post_id = urlparse(url).path.split("/")[3]
    scheme = urlparse(url).scheme
    post_url = f"{scheme}://{domain}/f/{subreddit}/{post_id}/"
    return post_url


def gitlab_get_project_memeber_role(page, account_name: str) -> str:
    # get the account index
    try:
        account_idx = page.evaluate(
            f"""(() => {{
                const elements = document.querySelectorAll("td[data-label='Account'] span.gl-avatar-labeled-sublabel");
                let index = -1;  // Default value if not found

                for(let i = 0; i < elements.length; i++) {{
                    if(elements[i].outerText === '@{account_name}') {{
                        index = i;
                        break;
                    }}
                }}

                return index;
            }})()"""
        )

        # get the role
        role: str = page.evaluate(
            f"""(() => {{
                return document.querySelectorAll("td.col-max-role span")[{account_idx}].outerText;
            }})()"""
        )
    except Exception:
        role = ""

    return role

def shopping_get_auth_token() -> str:
    response = requests.post(
        url=f"{SHOPPING}/rest/default/V1/integration/admin/token",
        headers={"content-type": "application/json"},
        data=json.dumps(
            {
                "username": ACCOUNTS["shopping_site_admin"]["username"],
                "password": ACCOUNTS["shopping_site_admin"]["password"],
            }
        ),
    )
    token: str = response.json()
    return token


def shopping_get_latest_order_url() -> str:
    """Get the latest order url from the shopping website."""

    header = {
        "Authorization": f"Bearer {shopping_get_auth_token()}",
        "Content-Type": "application/json",
    }

    params = {
        "searchCriteria[sortOrders][0][field]": "created_at",
        "searchCriteria[sortOrders][0][direction]": "DESC",
        "searchCriteria[pageSize]": "1",
    }

    response = requests.get(
        f"{SHOPPING}/rest/V1/orders", params=params, headers=header
    )
    assert response.status_code == 200
    response_obj = response.json()["items"][0]
    order_id = int(response_obj["increment_id"])
    order_url = f"{SHOPPING}/sales/order/view/order_id/{order_id}/"
    return order_url

def _to_dict(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, dict):
        return {str(k): _to_dict(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_dict(v) for v in x]
    if is_dataclass(x):
        return _to_dict(asdict(x))
    return x


def _norm_text(s: Any) -> str:
    if s is None:
        return ""
    t = str(s)
    t = t.replace("\u00a0", " ")
    t = " ".join(t.split())
    return t.strip()


def _norm_text_lower(s: Any) -> str:
    return _norm_text(s).lower()


def exact_match(pred: Any, ref: Any) -> bool:
    return _norm_text_lower(pred) == _norm_text_lower(ref)


def must_include(pred: Any, items: Sequence[str]) -> bool:
    p = _norm_text_lower(pred)
    for it in items:
        if _norm_text_lower(it) not in p:
            return False
    return True


def _norm_url(u: str) -> str:
    u = _norm_text(u)
    if not u:
        return ""
    sp = urlsplit(u)
    scheme = (sp.scheme or "").lower()
    netloc = (sp.netloc or "").lower()
    path = sp.path or ""
    if path != "/" and path.endswith("/"):
        path = path[:-1]
    query = sp.query or ""
    fragment = ""  # ignore fragment
    return urlunsplit((scheme, netloc, path, query, fragment))


def _is_awaitable(x: Any) -> bool:
    return hasattr(x, "__await__")


def _coerce_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return []


def _coerce_dict(x: Any) -> Dict[str, Any]:
    if x is None:
        return {}
    if isinstance(x, dict):
        return x
    return {}


def _llm_chat_json(llm: Any, *, system: str, user: str) -> Tuple[Any, Dict[str, Any]]:
    """
    Compatible wrapper:
    - if llm.chat_json returns (obj, usage) -> use it
    - if returns obj only -> usage={}
    """
    if llm is None:
        return None, {}
    out = llm.chat_json(system=system, user=user)
    if isinstance(out, tuple) and len(out) == 2:
        obj, usage = out
        return obj, (_coerce_dict(usage))
    return out, {}


def _fallback_fuzzy(pred: str, ref: str, *, threshold: float = 0.80) -> bool:
    p = _norm_text_lower(pred)
    r = _norm_text_lower(ref)
    if not p or not r:
        return False
    if r in p:
        return True
    ratio = SequenceMatcher(None, p, r).ratio()
    return ratio >= threshold


class WebArenaEvaluator:
    """
    Evaluate a TaskSpec according to WebArena-style eval.

    Supported eval_types:
      - string_match: compare final_answer with reference_answers
      - url_match: compare last_url with reference_url
      - program_html: run JS locator on page and compare with required_contents

    Notes:
      - program_html.url may be "last" or an explicit url
      - required_contents may contain: exact_match, must_include, fuzzy_match
      - fuzzy_match is judged by LLM if provided, otherwise fallback to difflib (and mark fallback in details)
    """

    def __init__(self, task: TaskSpec, *, llm: Any = None, sess: Optional[PlaywrightSession] = None) -> None:
        self.task = task
        self.llm = llm
        self.sess = sess

        ev = getattr(task, "eval_spec", None)
        self.eval_spec: Dict[str, Any] = _to_dict(ev) or {}

        raw = self.eval_spec.get("raw", None)
        if isinstance(raw, dict):
            merged = dict(raw)
            merged.update(self.eval_spec)
            self.eval_spec = merged

    async def evaluate(
        self,
        *,
        final_answer: str = "",
        last_url: str = "",
    ) -> EvalResult:
        eval_types = self._get_list(self.eval_spec, "eval_types")
        if not eval_types:
            return EvalResult(ok=True, score=1.0, reason="no_eval_types", details={})

        results: List[EvalResult] = []
        for t in eval_types:
            t = str(t)
            if t == "string_match":
                r = await self._eval_string_match(final_answer)
                results.append(r)
            elif t == "url_match":
                results.append(self._eval_url_match(last_url))
            elif t == "program_html":
                r = await self._eval_program_html(last_url)
                results.append(r)
            else:
                results.append(EvalResult(ok=False, score=0.0, reason=f"unknown_eval_type:{t}", details={}))

        ok = all(r.ok for r in results)
        score = 1.0 if ok else 0.0
        reason = "ok" if ok else "failed"
        details = {"per_type": [asdict(r) for r in results]}
        return EvalResult(ok=ok, score=score, reason=reason, details=details)

    # ---------------- string_match ----------------

    async def _eval_string_match(self, final_answer: str) -> EvalResult:
        ref = self._get_dict(self.eval_spec, "reference_answers")
        if not ref:
            return EvalResult(ok=False, score=0.0, reason="missing_reference_answers", details={})

        pred = _norm_text(final_answer)
        per_kind: Dict[str, Any] = {}
        ok_any = False

        for kind, refs_any in ref.items():
            kind = str(kind)
            refs_list = refs_any if isinstance(refs_any, list) else [refs_any]
            refs_list = [str(x) for x in refs_list]

            if kind == "exact_match":
                ok_k = any(exact_match(pred, r) for r in refs_list)
                per_kind[kind] = {"ok": ok_k, "refs": refs_list, "pred": pred}
            elif kind == "must_include":
                ok_k = must_include(pred, refs_list)
                per_kind[kind] = {"ok": ok_k, "refs": refs_list, "pred": pred}
            elif kind == "fuzzy_match":
                ok_k, detail = await self._fuzzy_match_any(pred=pred, refs=refs_list)
                per_kind[kind] = detail
            else:
                ok_k = False
                per_kind[kind] = {"ok": False, "reason": "unknown_string_match_kind", "refs": refs_list, "pred": pred}

            ok_any = ok_any or bool(ok_k)

        return EvalResult(
            ok=ok_any,
            score=1.0 if ok_any else 0.0,
            reason="string_match_ok" if ok_any else "string_match_failed",
            details={"pred": pred, "checks": per_kind},
        )

    # ---------------- url_match ----------------

    def _eval_url_match(self, last_url: str) -> EvalResult:
        ref_url = _norm_text(self.eval_spec.get("reference_url", ""))
        if not ref_url:
            return EvalResult(ok=False, score=0.0, reason="missing_reference_url", details={})

        pred = _norm_url(last_url)
        refn = _norm_url(ref_url)

        ok = pred == refn
        return EvalResult(
            ok=ok,
            score=1.0 if ok else 0.0,
            reason="url_match_ok" if ok else "url_match_failed",
            details={"pred": pred, "ref": refn},
        )

    # ---------------- program_html ----------------

    async def _eval_program_html(self, last_url) -> EvalResult:
        items = self._get_list(self.eval_spec, "program_html")
        if not items:
            return EvalResult(ok=False, score=0.0, reason="missing_program_html", details={})

        if self.sess is None:
            return EvalResult(ok=False, score=0.0, reason="missing_session_for_program_html", details={})

        page = await self.sess.new_page()
        import html as _html

        # 方案 A，把 func 注入 eval 环境
        # 这里用 globals() 从当前模块拿函数，避免你忘了在字典里手动写一遍
        allowed = {
            "shopping_get_sku_latest_review_rating": shopping_get_sku_latest_review_rating,
            "shopping_get_latest_order_url": shopping_get_latest_order_url,
            "shopping_get_auth_token": shopping_get_auth_token,
            "reddit_get_post_url": reddit_get_post_url,
            "gitlab_get_project_memeber_role": gitlab_get_project_memeber_role,
        }
        allowed = {k: v for k, v in allowed.items() if callable(v)}

        per_item: List[Dict[str, Any]] = []
        all_ok = True

        for idx, it_any in enumerate(items):
            it = it_any if isinstance(it_any, dict) else {}
            target_url = _norm_text(it.get("url", ""))
            if target_url == "last":
                target_url = last_url
            if target_url.startswith("func"):
                func = target_url.split("func:")[1]
                func = func.replace("__last_url__", last_url)
                target_url = eval(func)

            locator = _norm_text(it.get("locator", ""))
            required = it.get("required_contents", {})
            required = required if isinstance(required, dict) else {}

            item_detail: Dict[str, Any] = {
                "idx": idx,
                "target_url": target_url,
                "locator": locator,
                "required_contents": required,
            }

            try:
                nav_url = target_url
                item_detail["resolved_url"] = nav_url

                await page.goto(nav_url, wait_until="domcontentloaded")
                await wait_for_page_stable(page)

                if not locator.strip():
                    selected_element = await page.content()

                elif locator.startswith("document.") or locator.startswith("[...document."):
                    prep_actions = it.get("prep_actions", None)
                    if isinstance(prep_actions, list):
                        for pa in prep_actions:
                            try:
                                pa_s = str(pa).strip()
                                if pa_s:
                                    await page.evaluate(f"() => {pa_s}")
                            except Exception:
                                pass

                    try:
                        val = await page.evaluate(f"() => {locator}")
                        selected_element = "" if val is None else str(val)
                        if not selected_element:
                            selected_element = ""
                    except Exception:
                        selected_element = ""

                elif locator.startswith("func:"):
                    func_expr = locator.split("func:", 1)[1].strip()
                    func_expr = func_expr.replace("__page__", "page")

                    # 如果函数没在 allowed 里，提前给出清晰错误
                    # 这样你一眼就知道是忘了实现还是忘了注入
                    import re
                    m = re.match(r"^\s*([A-Za-z_]\w*)\s*\(", func_expr)
                    if m:
                        fname = m.group(1)
                        if fname not in allowed:
                            raise NameError(
                                f"{fname} is not available. Please define it in this module, "
                                f"or add it into the allowed function map."
                            )

                    selected_element = str(eval(func_expr, {"page": page, **allowed}))

                else:
                    raise ValueError(f"Unknown locator: {locator}")

                selected_element = _html.unescape(selected_element or "")
                item_detail["selected_element_len"] = len(selected_element)
                item_detail["selected_element_preview"] = selected_element[:300]

                ok_item, chk_detail = await self._check_required_async(
                    value=selected_element, required=required
                )
                item_detail["check"] = chk_detail

            except Exception as e:
                ok_item = False
                err = f"{type(e).__name__}: {e}"
                item_detail["error"] = err

            item_detail["ok"] = ok_item
            per_item.append(item_detail)
            all_ok = all_ok and ok_item

        return EvalResult(
            ok=all_ok,
            score=1.0 if all_ok else 0.0,
            reason="program_html_ok" if all_ok else "program_html_failed",
            details={"items": per_item},
        )

    async def _check_required_async(self, *, value: Any, required: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        pred = _norm_text(value)

        if "exact_match" in required:
            # ref = required.get("exact_match", "")
            # ok = exact_match(pred, ref)
            # return ok, {"type": "exact_match", "ref": ref, "pred": pred, "ok": ok}
            lst_any = required.get("exact_match", [])
            lst = lst_any if isinstance(lst_any, list) else [lst_any]
            lst = [str(x) for x in lst]
            ok, detail = await self._fuzzy_match_any(pred=pred, refs=lst)
            detail["type"] = "fuzzy_match"
            return ok, detail

        if "must_include" in required:
            lst_any = required.get("must_include", [])
            lst = lst_any if isinstance(lst_any, list) else [lst_any]
            lst = [str(x) for x in lst]
            ok = must_include(pred, lst)
            return ok, {"type": "must_include", "refs": lst, "pred": pred, "ok": ok}

        if "fuzzy_match" in required:
            lst_any = required.get("fuzzy_match", [])
            lst = lst_any if isinstance(lst_any, list) else [lst_any]
            lst = [str(x) for x in lst]
            ok, detail = await self._fuzzy_match_any(pred=pred, refs=lst)
            detail["type"] = "fuzzy_match"
            return ok, detail

        return False, {"type": "none", "pred": pred, "ok": False}

    # ---------------- fuzzy match (LLM) ----------------

    async def _fuzzy_match_any(self, *, pred: str, refs: Sequence[str]) -> Tuple[bool, Dict[str, Any]]:
        """
        Returns (ok, detail). Uses LLM if available; otherwise fallback to difflib.
        """
        refs_list = [str(x) for x in refs if _norm_text(x)]
        if not refs_list:
            return False, {"ok": False, "reason": "empty_refs", "pred": pred, "refs": []}

        # Prefer LLM
        if self.llm is not None:
            for r in refs_list:
                ok, detail = await self._fuzzy_match_llm(pred=pred, ref=r)
                if ok:
                    return True, detail
            return False, {"ok": False, "pred": pred, "refs": refs_list, "method": "llm", "reason": "no_ref_matched"}

        # Fallback
        matched = False
        best_ref = ""
        best_ratio = -1.0
        for r in refs_list:
            ratio = SequenceMatcher(None, _norm_text_lower(pred), _norm_text_lower(r)).ratio() if pred and r else 0.0
            if ratio > best_ratio:
                best_ratio = ratio
                best_ref = r
            if _fallback_fuzzy(pred, r):
                matched = True
                break

        return matched, {
            "ok": matched,
            "pred": pred,
            "refs": refs_list,
            "method": "difflib_fallback",
            "best_ref": best_ref,
            "best_ratio": best_ratio,
            "note": "LLM not provided, used difflib fallback",
        }

    async def _fuzzy_match_llm(self, *, pred: str, ref: str) -> Tuple[bool, Dict[str, Any]]:
        system = (
            "You are an evaluator.\n"
            "Decide whether the predicted answer satisfies the reference answer requirement.\n"
            "Return JSON only."
        )
        user = (
            "Return JSON:\n"
            '{ "ok": boolean, "reason": string }\n\n'
            "Rules:\n"
            "- Treat formatting differences as OK.\n"
            "- If reference is a phrase, predicted answer can include extra context.\n"
            "- If reference is a date/number, allow equivalent formats.\n"
            "- Be strict about identity: wrong entity or value is NOT ok.\n\n"
            f"Reference:\n{ref}\n\n"
            f"Predicted:\n{pred}\n"
        )

        obj, usage = _llm_chat_json(self.llm, system=system, user=user)
        ok = False
        reason = ""

        if isinstance(obj, dict):
            ok = bool(obj.get("ok", False))
            reason = _norm_text(obj.get("reason", ""))

        return ok, {
            "ok": ok,
            "pred": pred,
            "ref": ref,
            "method": "llm",
            "reason": reason,
            "usage": usage,
        }

    # ---------------- helpers ----------------

    @staticmethod
    def _get_list(d: Dict[str, Any], k: str) -> List[Any]:
        v = d.get(k, [])
        return v if isinstance(v, list) else []

    @staticmethod
    def _get_dict(d: Dict[str, Any], k: str) -> Dict[str, Any]:
        v = d.get(k, {})
        return v if isinstance(v, dict) else {}
