# webtactix/tools/build_record_site.py
from __future__ import annotations

import argparse
import html
import json
import os
from dataclasses import is_dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------- basic io ----------------

def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def _read_json(p: Path) -> Optional[Dict[str, Any]]:
    if not p.exists():
        return None
    try:
        return json.loads(_read_text(p))
    except Exception:
        return None


def _read_jsonl(p: Path) -> List[Dict[str, Any]]:
    if not p.exists():
        return []
    out: List[Dict[str, Any]] = []
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    out.append(obj)
            except Exception:
                continue
    return out


def _safe(x: Any) -> str:
    return html.escape("" if x is None else str(x))


def _fmt_float(x: Any, nd: int = 3) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return ""


def _to_jsonable(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (str, int, float, bool)):
        return x
    if isinstance(x, Path):
        return str(x)
    if is_dataclass(x):
        return _to_jsonable(asdict(x))
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    return str(x)


# ---------------- record parsing ----------------

def _list_round_dirs(task_dir: Path) -> List[Path]:
    rounds = []
    for p in task_dir.iterdir():
        if p.is_dir() and p.name.startswith("round_"):
            rounds.append(p)
    rounds.sort(key=lambda x: x.name)
    return rounds


def _list_plans(round_dir: Path) -> List[Path]:
    plans = []
    for p in round_dir.iterdir():
        if p.is_file() and p.name.startswith("plan_") and p.suffix == ".json":
            plans.append(p)
    plans.sort(key=lambda x: x.name)
    return plans


def _list_pages(round_dir: Path) -> Tuple[List[Path], List[Path]]:
    pages_dir = round_dir / "pages"
    if not pages_dir.exists():
        return [], []
    pngs = sorted([p for p in pages_dir.glob("*.png") if p.is_file()], key=lambda x: x.name)
    actrees = sorted([p for p in pages_dir.glob("*_actree.txt") if p.is_file()], key=lambda x: x.name)
    return pngs, actrees


def _list_data_extraction_nodes(round_dir: Path) -> List[Path]:
    de = round_dir / "data_extraction"
    if not de.exists():
        return []
    nodes = []
    for p in de.iterdir():
        if p.is_dir() and p.name.startswith("node_"):
            nodes.append(p)
    nodes.sort(key=lambda x: x.name)
    return nodes


def _derive_exec_durations(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Derive durations from exec_begin -> exec_ok/exec_error pairs.
    We pair by (src, action_sig) with a stack (last begin wins).
    """
    stack: Dict[Tuple[str, str], Dict[str, Any]] = {}
    out: List[Dict[str, Any]] = []

    for a in actions:
        typ = str(a.get("type", ""))
        src = str(a.get("src", "") or "")
        sig = str(a.get("action_sig", "") or "")
        t = a.get("t", None)

        if typ == "exec_begin":
            stack[(src, sig)] = a
            continue

        if typ in {"exec_ok", "exec_error"}:
            key = (src, sig)
            b = stack.pop(key, None)
            if b is None:
                continue
            try:
                dt = float(a.get("t", 0.0)) - float(b.get("t", 0.0))
            except Exception:
                dt = None

            out.append({
                "src": src,
                "dst": str(a.get("dst", "") or ""),
                "action_sig": sig,
                "ok": typ == "exec_ok",
                "t_begin": b.get("t", None),
                "t_end": a.get("t", None),
                "duration_s": dt,
            })
    return out


def _round_type(round_dir: Path, plans: List[Dict[str, Any]], actions: List[Dict[str, Any]]) -> str:
    # data_extraction
    if (round_dir / "data_extraction").exists():
        return "data_extraction"

    # web_operation by exec logs
    for a in actions:
        if str(a.get("type", "")) in {"exec_begin", "exec_ok", "exec_error"}:
            return "web_operation"

    # terminal by plan name
    for p in plans:
        result = p.get("result", {}) if isinstance(p.get("result", {}), dict) else {}
        plans_list = result.get("plans", []) if isinstance(result.get("plans", []), list) else []
        if plans_list:
            name = str(plans_list[0].get("name", "") or "")
            if name in {"finish", "go_back"}:
                return "terminal"

    # init or unknown
    return "plan_only"


def _extract_plan_name(pl: Dict[str, Any]) -> str:
    result = pl.get("result", {}) if isinstance(pl.get("result", {}), dict) else {}
    plans_list = result.get("plans", []) if isinstance(result.get("plans", []), list) else []
    if not plans_list:
        return ""
    return str(plans_list[0].get("name", "") or "")


def _sum_usage(items: List[Dict[str, Any]]) -> Dict[str, int]:
    pt = 0
    ct = 0
    tt = 0
    for it in items:
        u = it.get("usage", {}) if isinstance(it.get("usage", {}), dict) else {}
        try:
            pt += int(u.get("prompt_tokens", 0) or 0)
            ct += int(u.get("completion_tokens", 0) or 0)
            tt += int(u.get("total_tokens", 0) or 0)
        except Exception:
            continue
    return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt}


def _sum_usage_from_turns(turns: List[Dict[str, Any]]) -> Dict[str, int]:
    pt = 0
    ct = 0
    tt = 0
    for t in turns:
        u = t.get("usage", {}) if isinstance(t.get("usage", {}), dict) else {}
        try:
            pt += int(u.get("prompt_tokens", 0) or 0)
            ct += int(u.get("completion_tokens", 0) or 0)
            tt += int(u.get("total_tokens", 0) or 0)
        except Exception:
            continue
    return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": tt}


def parse_task(task_dir: Path) -> Dict[str, Any]:
    meta = _read_json(task_dir / "meta.json") or {}
    task = _read_json(task_dir / "task.json") or {}
    final = _read_json(task_dir / "final.json") or {}

    rounds_out: List[Dict[str, Any]] = []
    for rd in _list_round_dirs(task_dir):
        rj = _read_json(rd / "round.json") or {}

        plan_files = _list_plans(rd)
        plans = []
        for pf in plan_files:
            obj = _read_json(pf)
            if isinstance(obj, dict):
                obj["_file"] = pf.name
                plans.append(obj)

        decision = _read_json(rd / "decision.json") or None
        actions = _read_jsonl(rd / "actions.json")

        pngs, actrees = _list_pages(rd)

        data_nodes = _list_data_extraction_nodes(rd)
        data_runs = []
        for dn in data_nodes:
            begin = _read_json(dn / "begin.json") or {}
            final_de = _read_json(dn / "final.json")  # can be None
            turns = _read_jsonl(dn / "turns.jsonl")

            artifacts_png = sorted([p for p in dn.glob("turn_*.png") if p.is_file()], key=lambda x: x.name)
            artifacts_actree = sorted([p for p in dn.glob("turn_*_actree.txt") if p.is_file()], key=lambda x: x.name)

            data_runs.append({
                "node_dir": dn.name,
                "node_id": str(begin.get("node_id", "") or dn.name.replace("node_", "")),
                "begin": begin,
                "final": final_de,
                "turns": turns,
                "artifacts_png": [p.name for p in artifacts_png],
                "artifacts_actree": [p.name for p in artifacts_actree],
            })

        exec_durations = _derive_exec_durations(actions)
        rt = _round_type(rd, plans, actions)
        rounds_out.append({
            "dir": rd.name,
            "path": rd,
            "round_json": rj,
            "plans": plans,
            "decision": decision,
            "actions": actions,
            "exec_durations": exec_durations,
            "pages_png": [p.name for p in pngs],
            "pages_actree": [p.name for p in actrees],
            "data_extraction": data_runs,
            "round_type": rt,
        })

    return {
        "task_dir": task_dir,
        "meta": meta,
        "task": task,
        "final": final,
        "rounds": rounds_out,
    }


# ---------------- rendering helpers ----------------

def _rel(from_dir: Path, to_path: Path) -> str:
    try:
        return os.path.relpath(str(to_path), str(from_dir))
    except Exception:
        return str(to_path)


def _card(title: str, body_html: str, *, cls: str = "") -> str:
    c = f"card {cls}".strip()
    return f"""
    <section class="{c}">
      <div class="card-title">{_safe(title)}</div>
      <div class="card-body">{body_html}</div>
    </section>
    """


def _kv_table(d: Dict[str, Any], keys: List[str], *, title: str = "", max_val_len: int = 160) -> str:
    rows = []
    for k in keys:
        v = d.get(k, None)
        s = "" if v is None else str(v)
        if len(s) > max_val_len:
            s = s[:max_val_len] + "…"
        rows.append(f"<tr><td class='k'>{_safe(k)}</td><td class='v'>{_safe(s)}</td></tr>")
    t = "<table class='kv'>" + "".join(rows) + "</table>"
    if title:
        return f"<div class='subttl'>{_safe(title)}</div>{t}"
    return t


def _pre_snippet(text: str, *, max_chars: int = 1200) -> str:
    s = text
    if len(s) > max_chars:
        s = s[:max_chars] + "\n…"
    return f"<pre class='snippet'>{_safe(s)}</pre>"


def _badge(text: str, kind: str) -> str:
    return f"<span class='badge {kind}'>{_safe(text)}</span>"


def _fmt_tokens(u: Dict[str, Any]) -> str:
    pt = u.get("prompt_tokens", 0)
    ct = u.get("completion_tokens", 0)
    tt = u.get("total_tokens", 0)
    model = u.get("model", "")
    est = u.get("estimated", False)
    parts = [f"prompt {pt}", f"completion {ct}", f"total {tt}"]
    if model:
        parts.append(f"model {model}")
    if est:
        parts.append("estimated")
    return " | ".join(parts)


def _plan_block(plan_obj: Dict[str, Any]) -> str:
    nid = str(plan_obj.get("node_id", "") or "")
    usage = plan_obj.get("usage", {}) if isinstance(plan_obj.get("usage", {}), dict) else {}
    dur = plan_obj.get("duration", None)
    result = plan_obj.get("result", {}) if isinstance(plan_obj.get("result", {}), dict) else {}

    page_summary = str(result.get("page_summary", "") or "")
    progress = str(result.get("progress_analysis", "") or "")
    plans_list = result.get("plans", []) if isinstance(result.get("plans", []), list) else []

    plans_html = ""
    if plans_list:
        rows = []
        for i, p in enumerate(plans_list):
            if not isinstance(p, dict):
                continue
            name = str(p.get("name", "") or "")
            goal = str(p.get("goal", "") or "")
            steps = p.get("steps", [])
            steps_n = len(steps) if isinstance(steps, list) else 0
            rows.append(
                f"<tr><td>#{i}</td><td>{_safe(name)}</td><td>{_safe(goal)}</td><td>{steps_n}</td></tr>"
            )
        plans_html = "<table class='tbl'><thead><tr><th>#</th><th>name</th><th>goal</th><th>steps</th></tr></thead><tbody>" + "".join(rows) + "</tbody></table>"

    hdr = f"""
    <div class="row">
      <div class="col">{_badge(f"node {nid}", "gray")}</div>
      <div class="col right">{_badge(_fmt_tokens(usage), "blue")}</div>
    </div>
    <div class="meta">duration {(_fmt_float(dur, 3) + "s") if dur is not None else "n/a"}</div>
    """

    body = hdr
    if page_summary:
        body += f"<div class='subttl'>page_summary</div>{_pre_snippet(page_summary, max_chars=900)}"
    if progress:
        body += f"<div class='subttl'>progress_analysis</div>{_pre_snippet(progress, max_chars=1200)}"
    if plans_html:
        body += f"<div class='subttl'>plans</div>{plans_html}"

    return body


def _decision_block(dec: Dict[str, Any]) -> str:
    usage = dec.get("usage", {}) if isinstance(dec.get("usage", {}), dict) else {}
    dur = dec.get("duration", None)
    result = dec.get("result", {}) if isinstance(dec.get("result", {}), dict) else {}

    kind = str(result.get("kind", "") or "")
    selected = str(result.get("selected", "") or "")
    reason = str(result.get("reason", "") or "")

    body = f"""
    <div class="row">
      <div class="col">{_badge(kind or "decision", "purple")}</div>
      <div class="col right">{_badge(_fmt_tokens(usage), "blue")}</div>
    </div>
    <div class="meta">duration {(_fmt_float(dur, 6) + "s") if dur is not None else "n/a"}</div>
    """
    if selected:
        body += f"<div class='meta'>selected: <b>{_safe(selected)}</b></div>"
    if reason:
        body += f"<div class='subttl'>reason</div>{_pre_snippet(reason, max_chars=800)}"
    if not (kind or selected or reason):
        body += _pre_snippet(json.dumps(_to_jsonable(result), ensure_ascii=False, indent=2), max_chars=1200)

    return body


def _actions_table(actions: List[Dict[str, Any]], exec_durs: List[Dict[str, Any]]) -> str:
    if not actions:
        return "<div class='muted'>actions.json missing or empty</div>"

    counts: Dict[str, int] = {}
    for a in actions:
        typ = str(a.get("type", "") or "")
        counts[typ] = counts.get(typ, 0) + 1

    dur_stats = ""
    dvals = [d.get("duration_s") for d in exec_durs if d.get("duration_s") is not None]
    if dvals:
        avg = sum(float(x) for x in dvals) / max(1, len(dvals))
        dur_stats = f"<div class='meta'>derived exec durations: count {len(dvals)} | avg {_fmt_float(avg, 3)}s | min {_fmt_float(min(dvals), 3)}s | max {_fmt_float(max(dvals), 3)}s</div>"

    counts_html = " ".join([_badge(f"{k}:{v}", "gray") for k, v in sorted(counts.items(), key=lambda x: x[0])])
    rows = []
    for a in actions:
        typ = str(a.get("type", "") or "")
        src = str(a.get("src", "") or "")
        dst = str(a.get("dst", "") or "")
        sig = str(a.get("action_sig", "") or "")
        url_after = str(a.get("url_after", "") or "")
        err = str(a.get("error", "") or "")
        t = a.get("t", None)
        rows.append(
            "<tr>"
            f"<td>{_safe(typ)}</td>"
            f"<td>{_safe(src)}</td>"
            f"<td>{_safe(dst)}</td>"
            f"<td class='mono'>{_safe(sig)}</td>"
            f"<td class='mono'>{_safe(url_after)}</td>"
            f"<td class='mono'>{_safe(err)}</td>"
            f"<td class='mono'>{_safe(t)}</td>"
            "</tr>"
        )

    table = (
        "<div class='meta'>type counts: " + counts_html + "</div>"
        + dur_stats
        + "<div class='tblwrap'>"
        + "<table class='tbl'><thead><tr>"
          "<th>type</th><th>src</th><th>dst</th><th>action_sig</th><th>url_after</th><th>error</th><th>t</th>"
          "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )
    return table


def _pages_panel(round_dir: Path, pages_png: List[str], pages_actree: List[str], out_dir: Path, *, actree_lines: int = 60) -> str:
    pages_path = round_dir / "pages"
    if not pages_png and not pages_actree:
        return "<div class='muted'>pages/ missing</div>"

    imgs_html = ""
    if pages_png:
        cards = []
        for fn in pages_png:
            src = _rel(out_dir, pages_path / fn)
            cards.append(
                f"<a class='imgcard' href='{_safe(src)}' target='_blank' rel='noopener'>"
                f"<img src='{_safe(src)}' loading='lazy'/>"
                f"<div class='imgcap'>{_safe(fn)}</div>"
                f"</a>"
            )
        imgs_html = "<div class='imggrid'>" + "".join(cards) + "</div>"

    actree_html = ""
    if pages_actree:
        blocks = []
        for fn in pages_actree:
            p = pages_path / fn
            txt = ""
            if p.exists():
                raw = _read_text(p)
                lines = raw.splitlines()
                txt = "\n".join(lines[:actree_lines])
                if len(lines) > actree_lines:
                    txt += "\n…"
            src = _rel(out_dir, p)
            blocks.append(
                f"<details class='actree'><summary>{_safe(fn)} <span class='muted'>(click to expand)</span></summary>"
                f"<div class='meta'><a href='{_safe(src)}' target='_blank' rel='noopener'>open file</a></div>"
                f"{_pre_snippet(txt, max_chars=5000)}"
                f"</details>"
            )
        actree_html = "<div class='actrees'>" + "".join(blocks) + "</div>"

    return imgs_html + actree_html


def _data_extraction_panel(round_dir: Path, data_runs: List[Dict[str, Any]], out_dir: Path, *, actree_lines: int = 50) -> str:
    if not data_runs:
        return "<div class='muted'>data_extraction none</div>"

    de_root = round_dir / "data_extraction"
    blocks = []
    for run in data_runs:
        node_dir = str(run.get("node_dir", ""))
        node_id = str(run.get("node_id", ""))
        begin = run.get("begin", {}) if isinstance(run.get("begin", {}), dict) else {}
        turns = run.get("turns", [])
        turns = turns if isinstance(turns, list) else []

        node_path = de_root / node_dir
        begin_html = _kv_table(begin, ["node_id", "goal", "url", "round", "t_start", "time_utc"], title="begin.json")

        usage_sum = _sum_usage_from_turns(turns)
        head = f"<div class='row'><div class='col'>{_badge(f'node {node_id}', 'gray')}</div><div class='col right'>{_badge(_fmt_tokens(usage_sum), 'blue')}</div></div>"

        # turns table
        turn_rows = []
        for t in turns:
            stage = str(t.get("stage", "") or "")
            turn_idx = t.get("turn", None)
            note = str(t.get("note", "") or "")
            step = t.get("step", None)
            sig = str(t.get("action_sig", "") or "")
            err = str(t.get("error", "") or "")
            u = t.get("usage", {}) if isinstance(t.get("usage", {}), dict) else {}
            turn_rows.append(
                "<tr>"
                f"<td class='mono'>{_safe(turn_idx)}</td>"
                f"<td>{_safe(stage)}</td>"
                f"<td class='mono'>{_safe(_fmt_tokens(u))}</td>"
                f"<td>{_safe(note[:120] + ('…' if len(note) > 120 else ''))}</td>"
                f"<td class='mono'>{_safe(json.dumps(_to_jsonable(step), ensure_ascii=False)[:160])}</td>"
                f"<td class='mono'>{_safe(sig[:160])}</td>"
                f"<td class='mono'>{_safe(err[:160])}</td>"
                "</tr>"
            )
        turns_html = (
            "<div class='subttl'>turns.jsonl</div>"
            "<div class='tblwrap'><table class='tbl'>"
            "<thead><tr><th>turn</th><th>stage</th><th>usage</th><th>note</th><th>step</th><th>action_sig</th><th>error</th></tr></thead>"
            "<tbody>" + "".join(turn_rows) + "</tbody></table></div>"
        ) if turns else "<div class='muted'>turns.jsonl empty</div>"

        # artifacts
        artifacts_png = run.get("artifacts_png", [])
        artifacts_actree = run.get("artifacts_actree", [])

        artifacts_html = ""
        if artifacts_png:
            cards = []
            for fn in artifacts_png:
                src = _rel(out_dir, node_path / fn)
                cards.append(
                    f"<a class='imgcard' href='{_safe(src)}' target='_blank' rel='noopener'>"
                    f"<img src='{_safe(src)}' loading='lazy'/>"
                    f"<div class='imgcap'>{_safe(fn)}</div>"
                    f"</a>"
                )
            artifacts_html += "<div class='subttl'>turn screenshots</div><div class='imggrid'>" + "".join(cards) + "</div>"

        if artifacts_actree:
            blocks2 = []
            for fn in artifacts_actree:
                p = node_path / fn
                txt = ""
                if p.exists():
                    raw = _read_text(p)
                    lines = raw.splitlines()
                    txt = "\n".join(lines[:actree_lines])
                    if len(lines) > actree_lines:
                        txt += "\n…"
                src = _rel(out_dir, p)
                blocks2.append(
                    f"<details class='actree'><summary>{_safe(fn)} <span class='muted'>(click to expand)</span></summary>"
                    f"<div class='meta'><a href='{_safe(src)}' target='_blank' rel='noopener'>open file</a></div>"
                    f"{_pre_snippet(txt, max_chars=5000)}"
                    f"</details>"
                )
            artifacts_html += "<div class='subttl'>turn actrees</div><div class='actrees'>" + "".join(blocks2) + "</div>"

        body = head + begin_html + turns_html + artifacts_html
        blocks.append(_card(f"data_extraction {node_dir}", body, cls="de"))

    return "".join(blocks)


def _build_graph_data(rounds: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    nodes: Dict[str, Dict[str, Any]] = {}
    edges: List[Dict[str, Any]] = []

    def _ensure_node(nid: str) -> None:
        if not nid:
            return
        if nid in nodes:
            return
        nodes[nid] = {"id": nid, "label": nid}

    for r in rounds:
        actions = r.get("actions", [])
        if not isinstance(actions, list):
            continue
        for a in actions:
            typ = str(a.get("type", "") or "")
            if typ not in {"exec_ok", "data_extraction_ok"}:
                continue
            src = str(a.get("src", "") or "")
            dst = str(a.get("dst", "") or "")
            sig = str(a.get("action_sig", "") or "")
            if typ == "data_extraction_ok":
                sig = "data_extraction_ok"
            _ensure_node(src)
            _ensure_node(dst)
            if src and dst:
                edges.append({
                    "from": src,
                    "to": dst,
                    "label": sig[:80],
                    "type": typ,
                })

    # stable ordering
    nodes_list = [nodes[k] for k in sorted(nodes.keys())]
    return nodes_list, edges


# ---------------- html templates ----------------

def _base_css() -> str:
    return """
:root{
  --bg:#0b1220;
  --panel:#0f1b33;
  --card:#0c1730;
  --muted:#94a3b8;
  --text:#e5e7eb;
  --line:#233055;
  --accent:#60a5fa;
  --good:#34d399;
  --warn:#fbbf24;
  --bad:#fb7185;
  --purple:#c4b5fd;
}
*{box-sizing:border-box}
body{
  margin:0;
  font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji","Segoe UI Emoji";
  background: radial-gradient(1000px 600px at 20% 10%, rgba(96,165,250,.15), transparent 50%),
              radial-gradient(800px 500px at 80% 20%, rgba(196,181,253,.12), transparent 45%),
              var(--bg);
  color:var(--text);
}
a{color:var(--accent); text-decoration:none}
a:hover{text-decoration:underline}
header{
  padding:18px 22px;
  border-bottom:1px solid var(--line);
  background: rgba(15, 27, 51, 0.65);
  backdrop-filter: blur(10px);
  position: sticky;
  top:0;
  z-index:10;
}
.wrap{max-width:1200px; margin:0 auto; padding:18px 22px}
h1{font-size:20px; margin:0}
h2{font-size:16px; margin:0 0 10px 0; color:#dbeafe}
.grid{display:grid; gap:12px}
.grid.cols2{grid-template-columns: repeat(2, minmax(0,1fr))}
.grid.cols3{grid-template-columns: repeat(3, minmax(0,1fr))}
@media (max-width: 980px){
  .grid.cols2,.grid.cols3{grid-template-columns: 1fr}
}
.card{
  background: linear-gradient(180deg, rgba(12, 23, 48, .96), rgba(12, 23, 48, .86));
  border: 1px solid rgba(35,48,85,.75);
  border-radius: 14px;
  padding: 12px 12px 14px;
  box-shadow: 0 12px 28px rgba(0,0,0,.25);
}
.card-title{
  font-weight: 650;
  color:#e0f2fe;
  margin-bottom:10px;
  display:flex;
  align-items:center;
  gap:10px;
}
.card-body{color:var(--text)}
.subttl{margin:10px 0 6px; font-weight:650; color:#bfdbfe}
.meta{color:var(--muted); font-size:12px; margin:6px 0}
.muted{color:var(--muted)}
.row{display:flex; gap:10px; align-items:center; justify-content:space-between}
.col{flex:1}
.right{text-align:right}
.badge{
  display:inline-flex;
  align-items:center;
  padding:3px 8px;
  border-radius:999px;
  font-size:12px;
  border:1px solid rgba(148,163,184,.25);
  background: rgba(148,163,184,.08);
  color: #e5e7eb;
  white-space:nowrap;
}
.badge.blue{border-color: rgba(96,165,250,.35); background: rgba(96,165,250,.12)}
.badge.green{border-color: rgba(52,211,153,.35); background: rgba(52,211,153,.12)}
.badge.yellow{border-color: rgba(251,191,36,.35); background: rgba(251,191,36,.12)}
.badge.red{border-color: rgba(251,113,133,.35); background: rgba(251,113,133,.12)}
.badge.purple{border-color: rgba(196,181,253,.35); background: rgba(196,181,253,.12)}
.badge.gray{border-color: rgba(148,163,184,.25); background: rgba(148,163,184,.08)}
.kv{width:100%; border-collapse: collapse; font-size:13px}
.kv td{padding:6px 8px; border-top: 1px solid rgba(35,48,85,.6)}
.kv td.k{width:180px; color:#cbd5e1}
.kv td.v{color:#e5e7eb}
.tblwrap{overflow:auto; border: 1px solid rgba(35,48,85,.65); border-radius: 12px}
.tbl{width:100%; border-collapse: collapse; font-size:13px; min-width: 760px}
.tbl th,.tbl td{padding:8px 10px; border-bottom: 1px solid rgba(35,48,85,.55); vertical-align: top}
.tbl th{color:#c7d2fe; text-align:left; background: rgba(15,27,51,.55); position: sticky; top:0}
.tbl td.mono, .mono{font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace}
.snippet{
  background: rgba(15,27,51,.55);
  border:1px solid rgba(35,48,85,.6);
  border-radius: 12px;
  padding: 10px 12px;
  overflow:auto;
  max-height: 320px;
  font-size:12px;
  line-height: 1.45;
}
details{border:1px solid rgba(35,48,85,.6); border-radius:12px; padding:8px 10px; background: rgba(15,27,51,.35); margin:10px 0}
summary{cursor:pointer; color:#e0f2fe; font-weight:650}
.imggrid{display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:10px; margin-top:10px}
@media (max-width: 980px){ .imggrid{grid-template-columns: 1fr} }
.imgcard{
  display:block;
  border:1px solid rgba(35,48,85,.65);
  border-radius: 12px;
  overflow:hidden;
  background: rgba(15,27,51,.4);
}
.imgcard img{width:100%; height:auto; display:block}
.imgcap{padding:8px 10px; font-size:12px; color:var(--muted)}
.timeline{
  display:flex;
  flex-direction: column;
  gap:10px;
}
.tline-item{
  padding:10px 12px;
  border:1px solid rgba(35,48,85,.65);
  border-radius: 14px;
  background: rgba(15,27,51,.35);
  display:flex;
  justify-content:space-between;
  align-items:flex-start;
  gap:10px;
}
.tline-left{min-width: 240px}
.tline-right{flex:1}
.tline-rt{margin-top:6px}
.pill{
  display:inline-flex;
  align-items:center;
  gap:8px;
}
.hr{height:1px; background: rgba(35,48,85,.75); margin:14px 0}
.footer{color:var(--muted); font-size:12px; padding:20px 0}
.small{font-size:12px}
.de .card-title{color:#ddd6fe}
.graph{
  height: 360px;
  border:1px solid rgba(35,48,85,.65);
  border-radius: 14px;
  background: rgba(15,27,51,.35);
}
.star{
  margin-left:6px;
  color:var(--warn);
  font-size:14px;
  vertical-align:middle;
  cursor:help;
}
"""

def _base_js_cdn() -> str:
    # Prefer local assets (./assets/...), fallback to CDN, and show visible errors.
    return r"""
<script>
(function(){
  function loadScript(src){
    return new Promise(function(resolve,reject){
      var s=document.createElement('script');
      s.src=src;
      s.async=true;
      s.onload=function(){resolve(true)};
      s.onerror=function(){reject(new Error('load failed: '+src))};
      document.head.appendChild(s);
    });
  }

  function ensureBanner(){
    var el = document.getElementById('__lib_status');
    if(!el){
      el = document.createElement('div');
      el.id='__lib_status';
      el.style.cssText='position:fixed;bottom:12px;right:12px;max-width:520px;z-index:99999;background:#111;color:#fff;padding:10px 12px;border-radius:10px;font:12px/1.4 system-ui,Segoe UI,Arial;opacity:.92';
      el.innerHTML='<b>Lib status</b><div id="__lib_lines" style="margin-top:6px"></div>';
      document.body.appendChild(el);
    }
    return document.getElementById('__lib_lines');
  }

  function logLine(msg){
    try{
      var lines = ensureBanner();
      var d=document.createElement('div');
      d.textContent = msg;
      lines.appendChild(d);
    }catch(e){}
  }

  async function loadEither(name, localSrc, cdnSrc){
    try{
      await loadScript(localSrc);
      logLine(name+': local OK ('+localSrc+')');
      return true;
    }catch(e1){
      logLine(name+': local FAIL ('+localSrc+')');
    }
    try{
      await loadScript(cdnSrc);
      logLine(name+': cdn OK');
      return true;
    }catch(e2){
      logLine(name+': cdn FAIL ('+cdnSrc+')');
      return false;
    }
  }

  window.__loadLibs = async function(){
    // Use relative paths so python -m http.server works
    var okChart = await loadEither(
      'Chart.js',
      './assets/chart.umd.min.js',
      'https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js'
    );
    var okVis = await loadEither(
      'vis-network',
      './assets/vis-network.min.js',
      'https://cdn.jsdelivr.net/npm/vis-network@9.1.9/standalone/umd/vis-network.min.js'
    );

    // Expose flags for your renderer
    window.__libs_ok__ = { chart: okChart, vis: okVis };

    if(!okChart || !okVis){
      logLine('Hint: put libs at ./assets/ and reload with Ctrl+F5');
    }
  };
})();
</script>
"""


def render_index(tasks: List[Dict[str, Any]], *, dataset: str, model: str) -> str:
    rows = []
    for t in tasks:
        tid = t.get("task_id", "")
        intent = t.get("intent", "")
        status = t.get("status", "")
        ok = t.get("eval_ok", None)
        score = t.get("eval_score", None)
        eval_reason = str(t.get("eval_reason", "") or "")
        total_tokens = t.get("total_tokens", None)
        t_end = t.get("t_end", None)
        wf = t.get("wf_counts", {})
        de_n = wf.get("data_extraction", 0)
        wo_n = wf.get("web_operation", 0)
        link = f"task_{tid}.html"

        ok_badge = _badge("ok", "green") if ok is True else (_badge("fail", "red") if ok is False else _badge("n/a", "gray"))
        star = "<span class='star' title='human correct'>★</span>" if eval_reason == "human correct" else ""


        score_s = "" if score is None else str(score)

        rows.append(
            "<tr>"
            f"<td class='mono'><a href='{_safe(link)}'>{_safe(tid)}</a></td>"
            f"<td>{_safe(intent[:120] + ('…' if len(intent) > 120 else ''))}</td>"
            f"<td>{_safe(status)}</td>"
            f"<td>{ok_badge}{star} <span class='mono'>{_safe(score_s)}</span></td>"
            f"<td class='mono'>{_safe(total_tokens)}</td>"
            f"<td class='mono'>{_safe(_fmt_float(t_end, 3))}</td>"
            f"<td class='mono'>wo {wo_n} / de {de_n}</td>"
            "</tr>"
        )

    body = f"""
    <div class="wrap">
      <div class="grid cols2">
        {_card("Dataset / Model", _kv_table({"dataset":dataset,"model":model,"generated_at":datetime.utcnow().isoformat()+"Z"}, ["dataset","model","generated_at"]))}
        {_card("Tasks", f"<div class='meta'>click task_id to open detail page</div><div class='tblwrap'><table class='tbl'><thead><tr><th>task_id</th><th>intent</th><th>status</th><th>eval</th><th>tokens</th><th>t_end</th><th>workflow</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>")}
      </div>
      <div class="footer">site generated from record</div>
    </div>
    """
    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Record Site - {html.escape(dataset)} / {html.escape(model)}</title>
  <style>{_base_css()}</style>
</head>
<body>
  <header>
    <div class="wrap">
      <h1>Record Site <span class="muted">| {html.escape(dataset)} / {html.escape(model)}</span></h1>
    </div>
  </header>
  {body}
</body>
</html>
"""


def render_task(task_run: Dict[str, Any], *, dataset: str, model: str, out_dir: Path) -> str:
    meta = task_run.get("meta", {}) if isinstance(task_run.get("meta", {}), dict) else {}
    task = task_run.get("task", {}) if isinstance(task_run.get("task", {}), dict) else {}
    final = task_run.get("final", {}) if isinstance(task_run.get("final", {}), dict) else {}
    rounds = task_run.get("rounds", []) if isinstance(task_run.get("rounds", []), list) else []
    task_id = str(task.get("task", {}).get("task_id", "")) if isinstance(task.get("task", {}), dict) else ""
    if not task_id:
        # recorder's task.json is {"task": {TaskSpec...}, "constraints": [...]}
        task_id = str((task.get("task_id", None) or task.get("task", {}).get("task_id", "") or ""))

    task_spec = task.get("task", {}) if isinstance(task.get("task", {}), dict) else task
    constraints = task.get("constraints", [])

    intent = str(task_spec.get("intent", "") or "")
    start_url = str(task_spec.get("start_url", "") or "")
    sites = task_spec.get("sites", [])
    sites_s = ", ".join([str(x) for x in sites]) if isinstance(sites, list) else str(sites)
    storage_state = str(task_spec.get("storage_state_path", "") or "")

    final_status = str(final.get("status", "") or "")
    final_answer = str(final.get("answer", "") or "")
    final_reason = str(final.get("reason", "") or "")
    eval_result = final.get("eval_result", {})
    eval_result = final.get("eval_result", {})
    if eval_result and eval_result is True:
        eval_ok = True
    elif eval_result and eval_result is False:
        eval_ok = False
    else:
        eval_ok = eval_result.get("ok", None) if isinstance(eval_result, dict) else None
    eval_score = eval_result.get("score", None) if isinstance(eval_result, dict) else None
    eval_reason = eval_result.get("reason", "") if isinstance(eval_result, dict) else ""

    summary = final.get("summary", {}) if isinstance(final.get("summary", {}), dict) else {}
    total_tokens = summary.get("total_tokens", None)
    total_prompt = summary.get("total_prompt_tokens", None)
    total_comp = summary.get("total_completion_tokens", None)
    llm_calls = summary.get("llm_calls", None)
    planner_calls = summary.get("planner_calls", None)
    decision_calls = summary.get("decision_calls", None)
    total_rounds = summary.get("total_rounds", len(rounds))
    t_end = final.get("t_end", None)

    # workflow counts
    wf_counts: Dict[str, int] = {}
    for r in rounds:
        rt = str(r.get("round_type", "unknown") or "unknown")
        wf_counts[rt] = wf_counts.get(rt, 0) + 1

    # per-round series for charts
    round_ids: List[int] = []
    planner_pt: List[int] = []
    planner_ct: List[int] = []
    de_pt: List[int] = []
    de_ct: List[int] = []
    exec_durs_avg: List[float] = []

    for r in rounds:
        rj = r.get("round_json", {}) if isinstance(r.get("round_json", {}), dict) else {}
        rid = int(rj.get("round", 0) or 0)
        round_ids.append(rid)

        plans = r.get("plans", [])
        plans = plans if isinstance(plans, list) else []
        u_plan = _sum_usage(plans)
        planner_pt.append(int(u_plan["prompt_tokens"]))
        planner_ct.append(int(u_plan["completion_tokens"]))

        de_runs = r.get("data_extraction", [])
        de_runs = de_runs if isinstance(de_runs, list) else []
        turns_all: List[Dict[str, Any]] = []
        for dr in de_runs:
            turns = dr.get("turns", [])
            if isinstance(turns, list):
                turns_all.extend([t for t in turns if isinstance(t, dict) and isinstance(t.get("usage", {}), dict)])
        u_de = _sum_usage_from_turns(turns_all)
        de_pt.append(int(u_de["prompt_tokens"]))
        de_ct.append(int(u_de["completion_tokens"]))

        durs = r.get("exec_durations", [])
        durs = durs if isinstance(durs, list) else []
        ds = [d.get("duration_s") for d in durs if d.get("duration_s") is not None]
        if ds:
            exec_durs_avg.append(sum(float(x) for x in ds) / len(ds))
        else:
            exec_durs_avg.append(0.0)

    nodes, edges = _build_graph_data(rounds)

    # timeline html
    titems = []
    for r in rounds:
        rj = r.get("round_json", {}) if isinstance(r.get("round_json", {}), dict) else {}
        rid = int(rj.get("round", 0) or 0)
        rt = str(r.get("round_type", "unknown") or "unknown")
        frontier = rj.get("frontier", [])
        fsz = len(frontier) if isinstance(frontier, list) else 0
        f_parent = str(rj.get("f_parent", "") or "")
        t_start = rj.get("t_round_start", None)

        plans = r.get("plans", [])
        plans = plans if isinstance(plans, list) else []
        u_plan = _sum_usage(plans)

        de_runs = r.get("data_extraction", [])
        de_runs = de_runs if isinstance(de_runs, list) else []
        turns_all = []
        for dr in de_runs:
            turns = dr.get("turns", [])
            if isinstance(turns, list):
                turns_all.extend(turns)
        u_de = _sum_usage_from_turns(turns_all)

        badge_kind = "gray"
        if rt == "web_operation":
            badge_kind = "blue"
        elif rt == "data_extraction":
            badge_kind = "purple"
        elif rt == "terminal":
            badge_kind = "green"

        titems.append(f"""
        <div class="tline-item" id="round-{rid}">
          <div class="tline-left">
            <div class="pill">{_badge(f"round {rid:03d}", "gray")} {_badge(rt, badge_kind)}</div>
            <div class="meta">t_round_start {_fmt_float(t_start, 3)}s</div>
            <div class="meta">frontier {fsz} | f_parent {_safe(f_parent)}</div>
          </div>
          <div class="tline-right">
            <div class="tline-rt">
              {_badge("planner " + _fmt_tokens(u_plan), "blue")}
              {_badge("data " + _fmt_tokens(u_de), "purple")}
            </div>
          </div>
        </div>
        """)

    # constraints snippet
    c_html = ""
    if isinstance(constraints, list) and constraints:
        head = constraints[:6]
        c_rows = []
        for c in head:
            if not isinstance(c, dict):
                continue
            c_rows.append(f"<tr><td>{_safe(c.get('kind',''))}</td><td>{_safe(c.get('text',''))}</td></tr>")
        c_html = "<div class='tblwrap'><table class='tbl'><thead><tr><th>kind</th><th>text</th></tr></thead><tbody>" + "".join(c_rows) + "</tbody></table></div>"
    else:
        c_html = "<div class='muted'>none</div>"

    # eval badge
    eval_badge = _badge("ok", "green") if eval_ok is True else (_badge("fail", "red") if eval_ok is False else _badge("n/a", "gray"))
    eval_star = "<span class='star' title='human correct'>★</span>" if eval_reason == "human correct" else ""


    overview_left = _kv_table(
        {
            "intent": intent,
            "start_url": start_url,
            "sites": sites_s,
            "storage_state_path": storage_state,
        },
        ["intent", "start_url", "sites", "storage_state_path"],
    )
    overview_right = _kv_table(
        {
            "status": final_status,
            "answer": final_answer,
            "reason": final_reason,
            "eval.ok": eval_ok,
            "eval.score": eval_score,
            "eval.reason": eval_reason,
            "total_tokens": total_tokens,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_comp,
            "llm_calls": llm_calls,
            "planner_calls": planner_calls,
            "decision_calls": decision_calls,
            "total_rounds": total_rounds,
            "t_end": t_end,
        },
        ["status", "answer", "reason", "eval.ok", "eval.score", "eval.reason", "total_tokens", "prompt_tokens", "completion_tokens", "llm_calls", "planner_calls", "decision_calls", "total_rounds", "t_end"],
    )

    charts = f"""
    <div class="grid cols2">
      {_card("Tokens by round", "<canvas id='tokChart' height='140'></canvas><div class='meta'>planner and data_extraction tokens split into prompt/completion</div>")}
      {_card("Exec duration by round", "<canvas id='durChart' height='140'></canvas><div class='meta'>derived from actions: exec_begin -> exec_ok</div>")}
    </div>
    <div class="grid cols2" style="margin-top:12px">
      {_card("Workflow distribution", "<canvas id='wfChart' height='140'></canvas>")}
      {_card("Semantic graph", "<div id='graph' class='graph'></div><div id='graphNote' class='meta'>graph edges from actions exec_ok and data_extraction_ok</div>")}
    </div>
    """

    # per-round detail
    round_details = []
    for r in rounds:
        rd: Path = r.get("path")
        rj = r.get("round_json", {}) if isinstance(r.get("round_json", {}), dict) else {}
        rid = int(rj.get("round", 0) or 0)
        rt = str(r.get("round_type", "unknown") or "unknown")

        # round header
        header = f"<div class='row'><div class='col'>{_badge(f'round {rid:03d}', 'gray')} {_badge(rt, 'purple' if rt=='data_extraction' else ('blue' if rt=='web_operation' else 'gray'))}</div><div class='col right'><a href='#round-{rid}'>jump to timeline</a></div></div>"
        header += _kv_table(rj, ["round", "f_parent", "t_round_start", "time_utc"], title="round.json", max_val_len=300)

        plans = r.get("plans", [])
        plans = plans if isinstance(plans, list) else []
        plans_html = "<div class='muted'>no plan_*.json</div>"
        if plans:
            plans_html = "".join([_card(pl.get("_file", "plan"), _plan_block(pl), cls="") for pl in plans])

        decision = r.get("decision", None)
        decision_html = "<div class='muted'>decision.json missing</div>"
        if isinstance(decision, dict) and decision:
            decision_html = _card("decision.json", _decision_block(decision), cls="")

        actions = r.get("actions", [])
        actions = actions if isinstance(actions, list) else []
        exec_durs = r.get("exec_durations", [])
        exec_durs = exec_durs if isinstance(exec_durs, list) else []
        actions_html = _card("actions.json", _actions_table(actions, exec_durs), cls="")

        pages_html = _card("pages", _pages_panel(rd, r.get("pages_png", []), r.get("pages_actree", []), out_dir), cls="")

        de_html = _card("data_extraction", _data_extraction_panel(rd, r.get("data_extraction", []), out_dir), cls="de")

        body = header + "<div class='hr'></div>" + "<div class='grid cols2'>" + _card("planner outputs", plans_html) + _card("decision", decision_html) + "</div>" + actions_html + "<div class='grid cols2'>" + pages_html + de_html + "</div>"
        round_details.append(f"<details open><summary>round {rid:03d} detail</summary>{body}</details>")

    # embed data for JS
    js_data = {
        "round_ids": round_ids,
        "planner_pt": planner_pt,
        "planner_ct": planner_ct,
        "de_pt": de_pt,
        "de_ct": de_ct,
        "exec_durs_avg": exec_durs_avg,
        "wf_counts": wf_counts,
        "graph_nodes": nodes,
        "graph_edges": edges,
    }

    back_link = "index.html"
    head = f"""
<header>
  <div class="wrap">
    <div class="row">
      <div class="col">
        <h1>Task {html.escape(str(task_spec.get('task_id','')))} <span class="muted">| {html.escape(dataset)} / {html.escape(model)}</span></h1>
        <div class="meta"><a href="{_safe(back_link)}">← back to index</a></div>
      </div>
      <div class="col right">
        {eval_badge}{eval_star}
        {_badge(f"tokens {total_tokens}", "blue")}
        {_badge(f"rounds {len(rounds)}", "gray")}
      </div>
    </div>
  </div>
</header>
"""

    body = f"""
<div class="wrap">
  <div class="grid cols2">
    {_card("Task", overview_left)}
    {_card("Final & Summary", overview_right)}
  </div>

  <div class="grid cols2" style="margin-top:12px">
    {_card("Constraints", c_html)}
    {_card("Meta", _kv_table(meta, list(meta.keys())[:10], title="meta.json", max_val_len=180))}
  </div>

  <div style="margin-top:14px">
    {_card("Timeline", "<div class='timeline'>" + "".join(titems) + "</div>")}
  </div>

  <div style="margin-top:14px">
    {charts}
  </div>

  <div style="margin-top:14px">
    {_card("Round details", "".join(round_details))}
  </div>

  <div class="footer">site generated from record</div>
</div>
"""

    libs = _base_js_cdn()
    js = f"""
<script>
const DATA = {json.dumps(js_data, ensure_ascii=False)};
function byId(id){{ return document.getElementById(id); }}

function safeChart() {{
  if (typeof Chart === 'undefined') {{
    byId('tokChart').outerHTML = '<div class="muted">Chart.js not loaded. If you are offline, charts are disabled.</div>';
    byId('durChart').outerHTML = '<div class="muted">Chart.js not loaded. If you are offline, charts are disabled.</div>';
    byId('wfChart').outerHTML = '<div class="muted">Chart.js not loaded. If you are offline, charts are disabled.</div>';
    return;
  }}

  // tokens chart
  const labels = DATA.round_ids.map(x => String(x));
  new Chart(byId('tokChart'), {{
    type: 'line',
    data: {{
      labels,
      datasets: [
        {{ label: 'planner prompt', data: DATA.planner_pt }},
        {{ label: 'planner completion', data: DATA.planner_ct }},
        {{ label: 'data prompt', data: DATA.de_pt }},
        {{ label: 'data completion', data: DATA.de_ct }},
      ]
    }},
    options: {{
      responsive: true,
      scales: {{
        x: {{ title: {{ display:true, text:'round' }} }},
        y: {{ title: {{ display:true, text:'tokens' }} }}
      }}
    }}
  }});

  // duration chart
  new Chart(byId('durChart'), {{
    type: 'bar',
    data: {{
      labels,
      datasets: [
        {{ label: 'avg exec duration (s)', data: DATA.exec_durs_avg }},
      ]
    }},
    options: {{
      responsive: true,
      scales: {{
        x: {{ title: {{ display:true, text:'round' }} }},
        y: {{ title: {{ display:true, text:'seconds' }} }}
      }}
    }}
  }});

  // workflow pie
  const wfLabels = Object.keys(DATA.wf_counts);
  const wfVals = wfLabels.map(k => DATA.wf_counts[k] || 0);
  new Chart(byId('wfChart'), {{
    type: 'doughnut',
    data: {{
      labels: wfLabels,
      datasets: [{{ label: 'workflow', data: wfVals }}]
    }},
    options: {{ responsive: true }}
  }});
}}

function safeGraph() {{
  const el = byId('graph');
  if (typeof vis === 'undefined' || !vis.Network) {{
    el.innerHTML = '<div class="muted" style="padding:12px">vis-network not loaded. If you are offline, graph is disabled.</div>';
    return;
  }}
  const nodes = new vis.DataSet(DATA.graph_nodes.map(n => {{
    id: n.id, label: n.label, shape: 'dot'
  }}));
  const edges = new vis.DataSet(DATA.graph_edges.map(e => {{
    from: e.from, to: e.to, label: e.label, arrows: 'to'
  }}));
  const net = new vis.Network(el, {{ nodes, edges }}, {{
    physics: {{ stabilization: true }},
    edges: {{ font: {{ align: 'top' }}, smooth: {{ type: 'dynamic' }} }},
    nodes: {{ font: {{ color: '#e5e7eb' }} }}
  }});
}}

(async function init() {{
  if (window.__loadLibs) {{
    await window.__loadLibs();
  }}
  safeChart();
  safeGraph();
}})();
</script>
"""

    return f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Task {html.escape(str(task_spec.get('task_id','')))} - {html.escape(dataset)} / {html.escape(model)}</title>
  <style>{_base_css()}</style>
</head>
<body>
  {head}
  {body}
  {libs}
  {js}
</body>
</html>
"""


# ---------------- scanning ----------------

def scan(base: Path, dataset: str, model: str, tasks: Optional[List[int]] = None) -> List[Path]:
    root = base / dataset / model
    if not root.exists():
        return []
    task_dirs = []
    for p in root.iterdir():
        if p.is_dir() and p.name.startswith("task_"):
            if tasks is not None:
                try:
                    tid = int(p.name.replace("task_", ""))
                except Exception:
                    continue
                if tid not in set(tasks):
                    continue
            task_dirs.append(p)
    task_dirs.sort(key=lambda x: int(x.name.replace("task_", "")) if x.name.replace("task_", "").isdigit() else x.name)
    return task_dirs


def summarize_for_index(task_run: Dict[str, Any]) -> Dict[str, Any]:
    task = task_run.get("task", {}) if isinstance(task_run.get("task", {}), dict) else {}
    final = task_run.get("final", {}) if isinstance(task_run.get("final", {}), dict) else {}
    rounds = task_run.get("rounds", []) if isinstance(task_run.get("rounds", []), list) else []

    task_spec = task.get("task", {}) if isinstance(task.get("task", {}), dict) else task
    tid = task_spec.get("task_id", None)
    intent = str(task_spec.get("intent", "") or "")

    eval_result = final.get("eval_result", {})
    if eval_result and eval_result is True:
        eval_ok = True
    elif eval_result and eval_result is False:
        eval_ok = False
    else:
        eval_ok = eval_result.get("ok", None) if isinstance(eval_result, dict) else None
    eval_score = eval_result.get("score", None) if isinstance(eval_result, dict) else None
    eval_reason = eval_result.get("reason", "") if isinstance(eval_result, dict) else ""

    summary = final.get("summary", {}) if isinstance(final.get("summary", {}), dict) else {}
    total_tokens = summary.get("total_tokens", None)
    t_end = final.get("t_end", None)

    wf_counts: Dict[str, int] = {}
    for r in rounds:
        rt = str(r.get("round_type", "unknown") or "unknown")
        wf_counts[rt] = wf_counts.get(rt, 0) + 1

    return {
        "task_id": tid,
        "intent": intent,
        "status": str(final.get("status", "") or ""),
        "eval_ok": eval_ok,
        "eval_score": eval_score,
        "eval_reason": eval_reason,  # 新增
        "total_tokens": total_tokens,
        "t_end": t_end,
        "wf_counts": wf_counts,
    }


# ---------------- main ----------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="record")
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--task", type=int, nargs="*", default=None, help="optional list of task ids")
    ap.add_argument("--out", type=str, default="", help="output dir, default: <base>/site/<dataset>/<model>")
    args = ap.parse_args()

    base = Path(args.base).resolve()
    dataset = args.dataset
    model = args.model
    tasks_filter = args.task if args.task else None

    task_dirs = scan(base, dataset, model, tasks_filter)
    if not task_dirs:
        print(f"[ERR] no tasks found under {base}/{dataset}/{model}")
        return

    out_root = Path(args.out).resolve() if args.out else (base / "site" / dataset / model).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    parsed: List[Dict[str, Any]] = []
    for td in task_dirs:
        run = parse_task(td)
        parsed.append(run)

    # write task pages
    for run in parsed:
        t = run.get("task", {}) if isinstance(run.get("task", {}), dict) else {}
        task_spec = t.get("task", {}) if isinstance(t.get("task", {}), dict) else t
        tid = task_spec.get("task_id", None)
        if tid is None:
            # fallback from dir name
            tid = int(run["task_dir"].name.replace("task_", ""))

        html_text = render_task(run, dataset=dataset, model=model, out_dir=out_root)
        (out_root / f"task_{tid}.html").write_text(html_text, encoding="utf-8")
        print(f"[OK] wrote task_{tid}.html")

    # write index
    idx_items = [summarize_for_index(r) for r in parsed]
    index_html = render_index(idx_items, dataset=dataset, model=model)
    (out_root / "index.html").write_text(index_html, encoding="utf-8")
    print(f"[OK] wrote index.html at {out_root}")


if __name__ == "__main__":
    main()
