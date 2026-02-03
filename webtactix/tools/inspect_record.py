# webtactix/tools/inspect_record.py
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------- utils ----------------

def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")


def _load_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(_read_text(path))
    except Exception as e:
        return {"__error__": f"{type(e).__name__}: {e}", "__path__": str(path)}


def _iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                yield {"__raw__": line}


def _keys(obj: Any) -> List[str]:
    return list(obj.keys()) if isinstance(obj, dict) else []


def _preview_str(s: Any, n: int = 180) -> str:
    t = "" if s is None else str(s)
    t = t.replace("\n", "\\n")
    if len(t) <= n:
        return t
    return t[: n] + "..."


def _preview_json(obj: Any, n: int = 320) -> str:
    try:
        s = json.dumps(obj, ensure_ascii=False)
    except Exception:
        s = str(obj)
    return _preview_str(s, n=n)


def _list_dirs(p: Path) -> List[str]:
    if not p.exists():
        return []
    return sorted([x.name for x in p.iterdir() if x.is_dir()])


def _list_files(p: Path, suffix: Optional[str] = None) -> List[str]:
    if not p.exists():
        return []
    out = []
    for x in p.iterdir():
        if not x.is_file():
            continue
        if suffix and x.suffix != suffix:
            continue
        out.append(x.name)
    return sorted(out)


def _extract_node_id_from_page_filename(name: str) -> Optional[str]:
    # v2_snapshot.png / v2_actree.txt
    if name.endswith("_snapshot.png"):
        return name[: -len("_snapshot.png")]
    if name.endswith("_actree.txt"):
        return name[: -len("_actree.txt")]
    return None


# ---------------- inspectors ----------------

def scan_base(base: Path) -> Tuple[List[str], Dict[str, List[str]]]:
    datasets = _list_dirs(base)
    models_by_dataset: Dict[str, List[str]] = {}
    for ds in datasets:
        models_by_dataset[ds] = _list_dirs(base / ds)
    return datasets, models_by_dataset


def inspect_task(task_dir: Path) -> None:
    print(f"\n######## Inspect: {task_dir.as_posix()} ########\n")
    if not task_dir.exists():
        print("Task dir not found.")
        return

    meta_path = task_dir / "meta.json"
    task_path = task_dir / "task.json"
    final_path = task_dir / "final.json"

    meta = _load_json(meta_path) or {}
    task = _load_json(task_path) or {}
    final = _load_json(final_path) or {}

    # meta.json
    if meta_path.exists():
        print(f"[meta.json] keys: {_keys(meta)}")
        print("[meta.json] preview:")
        if isinstance(meta, dict):
            for k in _keys(meta):
                print(f"  - {k}: {meta.get(k)}")
    else:
        print("[meta.json] MISSING")

    # task.json
    if task_path.exists():
        if isinstance(task, dict):
            task_obj = task.get("task", {})
            cons = task.get("constraints", [])
            print(f"\n[task.json] task keys: {_keys(task_obj)}")
            print(f"[task.json] intent: {task_obj.get('intent')}")
            print(f"[task.json] start_url: {task_obj.get('start_url')}")
            print(f"[task.json] constraints count: {len(cons) if isinstance(cons, list) else 'NA'}")
            if isinstance(cons, list) and cons:
                print(f"[task.json] constraints head: {cons[0]}")
        else:
            print("\n[task.json] invalid json structure")
    else:
        print("\n[task.json] MISSING")

    # final.json
    if final_path.exists():
        if isinstance(final, dict):
            print(f"\n[final.json] status: {final.get('status')}")
            print(f"[final.json] answer: {final.get('answer')}")
            print(f"[final.json] reason: {final.get('reason')}")
            ev = final.get("eval_result", None)
            print(f"[final.json] eval_result type: {type(ev)}")
            if isinstance(ev, dict):
                print(f"[final.json] eval_result keys: {_keys(ev)}")
                # print a few useful fields if exist
                for k in ["ok", "score", "reason", "details"]:
                    if k in ev:
                        if k == "details":
                            print(f"[final.json] eval_result.details preview: {_preview_json(ev.get('details'), n=420)}")
                        else:
                            print(f"[final.json] eval_result.{k}: {ev.get(k)}")
            else:
                print(f"[final.json] eval_result preview: {_preview_json(ev)}")

            summ = final.get("summary", None)
            if isinstance(summ, dict):
                print("\n== [final.json] summary ==")
                for k in [
                    "total_tokens",
                    "total_prompt_tokens",
                    "total_completion_tokens",
                    "llm_calls",
                    "estimated_calls",
                    "planner_calls",
                    "decision_calls",
                    "data_calls",
                    "total_rounds",
                    "total_nodes",
                ]:
                    if k in summ:
                        print(f"- {k}: {summ.get(k)}")
            else:
                print("\n[final.json] summary MISSING or invalid")
        else:
            print("\n[final.json] invalid json structure")
    else:
        print("\n[final.json] MISSING")

    # rounds
    round_dirs = sorted([p for p in task_dir.iterdir() if p.is_dir() and p.name.startswith("round_")])
    print(f"\n[rounds] count: {len(round_dirs)}")
    print(f"[rounds] names: {[p.name for p in round_dirs]}")

    # Build a global node->(png,actree) index for convenience
    global_pages = task_dir / "pages"
    global_node_assets: Dict[str, Dict[str, str]] = {}
    if global_pages.exists():
        for fn in _list_files(global_pages):
            nid = _extract_node_id_from_page_filename(fn)
            if not nid:
                continue
            global_node_assets.setdefault(nid, {})
            if fn.endswith("_snapshot.png"):
                global_node_assets[nid]["png"] = str((global_pages / fn).as_posix())
            elif fn.endswith("_actree.txt"):
                global_node_assets[nid]["actree"] = str((global_pages / fn).as_posix())

    for rd in round_dirs:
        inspect_round(rd, global_node_assets)


def inspect_round(round_dir: Path, global_node_assets: Dict[str, Dict[str, str]]) -> None:
    print(f"\n---------------- {round_dir.name} ----------------")

    round_json = _load_json(round_dir / "round.json")
    if isinstance(round_json, dict):
        show = {k: round_json.get(k) for k in ["round", "f_parent", "t_round_start", "time_utc"] if k in round_json}
        print(f"[round.json] {show}")
        frontier = round_json.get("frontier", [])
        if isinstance(frontier, list):
            print(f"[round.json] frontier size: {len(frontier)}")
            if frontier:
                print(f"[round.json] frontier head: {frontier[:5]}")
    else:
        print("[round.json] MISSING")

    # plans
    plan_files = sorted([p.name for p in round_dir.iterdir() if p.is_file() and p.name.startswith("plan_") and p.suffix == ".json"])
    print(f"[plans] files: {plan_files}")
    for pf in plan_files:
        p = _load_json(round_dir / pf)
        if not isinstance(p, dict):
            continue
        nid = p.get("node_id", pf.replace("plan_", "").replace(".json", ""))
        usage = p.get("usage", {})
        duration = p.get("duration", None)
        result = p.get("result", {})
        page_summary = None
        progress_analysis = None
        plans = None
        if isinstance(result, dict):
            page_summary = result.get("page_summary")
            progress_analysis = result.get("progress_analysis")
            plans = result.get("plans")

        print(f"\n  [plan] {pf} node={nid}")
        print(f"    usage: {usage}")
        print(f"    duration: {duration}")
        if page_summary is not None:
            print(f"    page_summary: {_preview_str(page_summary, 160)}")
        if progress_analysis is not None:
            print(f"    progress_analysis: {_preview_str(progress_analysis, 160)}")
        if isinstance(plans, list):
            print(f"    plans count: {len(plans)}")
            for i, pl in enumerate(plans[:5]):
                if isinstance(pl, dict):
                    print(f"    plan#{i} name: {pl.get('name')} goal: {_preview_str(pl.get('goal'), 120)}")
                else:
                    print(f"    plan#{i}: {_preview_json(pl, 160)}")

    # decision
    decision_path = round_dir / "decision.json"
    dec = _load_json(decision_path)
    if isinstance(dec, dict):
        print(f"\n[decision.json] keys: {_keys(dec)}")
        usage = dec.get("usage", {})
        print(f"[decision.json] usage: {usage}")
        print(f"[decision.json] duration: {dec.get('duration')}")
        res = dec.get("result", {})
        if isinstance(res, dict):
            print(f"[decision.json] result keys: {_keys(res)}")
            # explicitly print selected related fields if present
            for k in ["kind", "reason", "selected", "selected_node_id", "explore_parent", "reflection"]:
                if k in res:
                    print(f"[decision.json] result.{k}: {_preview_json(res.get(k), 420) if k in ['reflection'] else res.get(k)}")
            # fallback full preview
            print(f"[decision.json] result preview: {_preview_json(res, 520)}")
        else:
            print(f"[decision.json] result preview: {_preview_json(res, 520)}")
    else:
        print("\n[decision.json] MISSING")

    # actions
    actions_path = round_dir / "actions.json"
    if actions_path.exists():
        lines = list(_iter_jsonl(actions_path))
        print(f"\n[actions.json] lines: {len(lines)}")
        # type counts + show errors
        type_counts: Dict[str, int] = {}
        err_lines: List[Dict[str, Any]] = []
        for it in lines:
            t = str(it.get("type", "unknown"))
            type_counts[t] = type_counts.get(t, 0) + 1
            if "error" in it and it.get("error"):
                err_lines.append(it)
            if it.get("type") in {"exec_error", "exec_fail"}:
                err_lines.append(it)

        print(f"[actions.json] type_counts: {type_counts}")
        if err_lines:
            print(f"[actions.json] error-like lines: {len(err_lines)}")
            for it in err_lines[:3]:
                print(f"  err: {_preview_json(it, 520)}")

        # head lines
        for i, it in enumerate(lines[:4]):
            print(f"  line#{i}: {_preview_json(it, 520)}")

        # try derive simple exec durations if possible
        # match exec_begin and exec_ok by (src, action_sig)
        begins: Dict[Tuple[str, str], float] = {}
        durs: List[float] = []
        for it in lines:
            tp = it.get("type")
            if tp == "exec_begin":
                key = (str(it.get("src", "")), str(it.get("action_sig", "")))
                begins[key] = float(it.get("t", 0.0) or 0.0)
            if tp == "exec_ok":
                key = (str(it.get("src", "")), str(it.get("action_sig", "")))
                if key in begins:
                    d = float(it.get("t", 0.0) or 0.0) - begins[key]
                    if d >= 0:
                        durs.append(d)
        if durs:
            avg = sum(durs) / max(1, len(durs))
            print(f"[actions.json] derived exec durations count={len(durs)} avg={avg:.3f}s min={min(durs):.3f}s max={max(durs):.3f}s")
    else:
        print("\n[actions.json] MISSING")

    # pages
    pages_dir = round_dir / "pages"
    if pages_dir.exists():
        pngs = [x for x in _list_files(pages_dir) if x.endswith(".png")]
        actrees = [x for x in _list_files(pages_dir) if x.endswith(".txt")]
        print(f"\n[pages] png: {pngs}")
        print(f"[pages] actree: {actrees}")
        if actrees:
            a0 = pages_dir / actrees[0]
            snippet = _read_text(a0)[:360]
            snippet = snippet.replace("\n", "\\n")
            print(f"[pages] actree snippet: {snippet}")
    else:
        print("\n[pages] MISSING")
        # show any global node assets if round pages missing
        if global_node_assets:
            # do nothing noisy, just a hint
            pass

    # data_extraction
    data_dir = round_dir / "data_extraction"
    if data_dir.exists():
        print("\n[data_extraction] FOUND")
        node_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])
        print(f"[data_extraction] nodes: {[p.name for p in node_dirs]}")
        for nd in node_dirs[:3]:
            inspect_data_extraction_node(nd)
    else:
        print("\n[data_extraction] none")


def inspect_data_extraction_node(node_dir: Path) -> None:
    # expected:
    # begin.json, final.json, turns.jsonl, turn_XXX_actree.txt, turn_XXX.png
    begin = _load_json(node_dir / "begin.json")
    final = _load_json(node_dir / "final.json")
    turns_path = node_dir / "turns.jsonl"

    print(f"  [data.node] {node_dir.name}")
    if isinstance(begin, dict):
        print(f"    begin keys: {_keys(begin)}")
        for k in ["node_id", "goal", "url", "round", "t_start", "time_utc"]:
            if k in begin:
                print(f"    begin.{k}: {_preview_str(begin.get(k), 200)}")
    else:
        print("    begin.json MISSING")

    if isinstance(final, dict):
        print(f"    final keys: {_keys(final)}")
        for k in ["done", "reason", "duration", "t_start", "t_end", "time_utc"]:
            if k in final:
                print(f"    final.{k}: {_preview_str(final.get(k), 220)}")
        if "extracted" in final:
            print(f"    final.extracted preview: {_preview_str(final.get('extracted'), 260)}")
    else:
        print("    final.json MISSING")

    if turns_path.exists():
        turns = list(_iter_jsonl(turns_path))
        print(f"    turns.jsonl lines: {len(turns)}")
        # show first few with emphasis on usage and action
        for it in turns[:5]:
            stage = it.get("stage")
            usage = it.get("usage")
            step = it.get("step")
            action_sig = it.get("action_sig")
            err = it.get("error")
            print(f"    turn: stage={stage} usage={usage} step={step} action_sig={_preview_str(action_sig, 120)} error={_preview_str(err, 120)}")
    else:
        print("    turns.jsonl MISSING")

    # list a few artifacts
    pngs = sorted([p.name for p in node_dir.iterdir() if p.is_file() and p.suffix == ".png"])[:5]
    actrees = sorted([p.name for p in node_dir.iterdir() if p.is_file() and p.name.endswith("_actree.txt")])[:5]
    if pngs or actrees:
        print(f"    artifacts png(head): {pngs}")
        print(f"    artifacts actree(head): {actrees}")


# ---------------- main ----------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", type=str, default="record")
    ap.add_argument("--dataset", type=str, default="")
    ap.add_argument("--model", type=str, default="")
    ap.add_argument("--task", type=str, default="")
    args = ap.parse_args()

    base = Path(args.base).expanduser().resolve()
    ds = (args.dataset or "").strip()
    model = (args.model or "").strip()
    task = (args.task or "").strip()

    datasets, models_by_dataset = scan_base(base)
    print("\n=== SCAN RESULT ===")
    print(f"base: {base.as_posix()}")
    print(f"datasets: {datasets}")
    print(f"models_by_dataset: {models_by_dataset}")

    if not ds or not model or task == "":
        print("\nUsage example:")
        print("  python webtactix/tools/inspect_record.py --base record --dataset webarena --model deepseek --task 0")
        return

    task_dir = base / ds / model / f"task_{task}"
    inspect_task(task_dir)


if __name__ == "__main__":
    main()
