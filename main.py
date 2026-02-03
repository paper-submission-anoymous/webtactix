# webtactix/main.py
from __future__ import annotations

import argparse
import asyncio
import multiprocessing as mp
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Sequence, List, Dict, Any, Optional

from webtactix.browser.playwright_session import PlaywrightConfig, PlaywrightSession
from webtactix.core.semantic_tree import SemanticTree
from webtactix.llm.openai_compat import OpenAICompatClient
from webtactix.llm.presets import preset_deepseek_chat, preset_qwen32b, preset_chatgpt
from webtactix.agents.planner_agent import PlannerAgent
from webtactix.agents.decision_agent import DecisionAgent
from webtactix.agents.constraint_agent import ConstraintAgent
from webtactix.agents.data_agent import DataExtractionAgent
from webtactix.runner.experiment_runner import ExperimentRunner, RunnerConfig
from webtactix.runner.recorder import Recorder
from webtactix.datasets.webarena_adapter import WebArenaAdapter
from webtactix.datasets.webarena_evaluator import WebArenaEvaluator
from webtactix.datasets.online_min2web_adapter import OnlineMind2WebAdapter
from webtactix.core.schemas import TaskSpec
from webtactix.workflows.execute import Executor

SHOPPING_ADMIN = [0, 1, 2, 3, 4, 5, 6, 11, 12, 13, 14, 15, 41, 42, 43, 62, 63, 64, 65, 77, 78, 79, 94, 95, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 119, 120, 121, 122, 123, 127, 128, 129, 130, 131, 157, 183, 184, 185, 186, 187, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 243, 244, 245, 246, 247, 288, 289, 290, 291, 292, 344, 345, 346, 347, 348, 374, 375, 423, 453, 454, 455, 456, 457, 458, 459, 460, 461, 462, 463, 464, 470, 471, 472, 473, 474, 486, 487, 488, 489, 490, 491, 492, 493, 494, 495, 496, 497, 498, 499, 500, 501, 502, 503, 504, 505, 538, 539, 540, 541, 542, 543, 544, 545, 546, 547, 548, 549, 550, 551, 676, 677, 678, 679, 680, 694, 695, 696, 697, 698, 699, 700, 701, 702, 703, 704, 705, 706, 707, 708, 709, 710, 711, 712, 713, 768, 769, 770, 771, 772, 773, 774, 775, 776, 777, 778, 779, 780, 781, 782, 790],
MAP = [7, 8, 9, 10, 16, 17, 18, 19, 20, 32, 33, 34, 35, 36, 37, 38, 39, 40, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 70, 71, 72, 73, 74, 75, 76, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 98, 99, 100, 101, 137, 138, 139, 140, 151, 152, 153, 154, 155, 218, 219, 220, 221, 222, 223, 224, 236, 237, 248, 249, 250, 251, 252, 253, 254, 255, 256, 257, 287, 356, 363, 364, 365, 366, 367, 369, 370, 371, 372, 373, 377, 378, 379, 380, 381, 382, 383, 757, 758, 761, 762, 763, 764, 765, 766, 767],
SHOPPING = [21, 22, 23, 24, 25, 26, 47, 48, 49, 50, 51, 96, 117, 118, 124, 125, 126, 141, 142, 143, 144, 145, 146, 147, 148, 149, 150, 158, 159, 160, 161, 162, 163, 164, 165, 166, 167, 188, 189, 190, 191, 192, 225, 226, 227, 228, 229, 230, 231, 232, 233, 234, 235, 238, 239, 240, 241, 242, 260, 261, 262, 263, 264, 269, 270, 271, 272, 273, 274, 275, 276, 277, 278, 279, 280, 281, 282, 283, 284, 285, 286, 298, 299, 300, 301, 302, 313, 319, 320, 321, 322, 323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 337, 338, 351, 352, 353, 354, 355, 358, 359, 360, 361, 362, 368, 376, 384, 385, 386, 387, 388, 431, 432, 433, 434, 435, 436, 437, 438, 439, 440, 465, 466, 467, 468, 469, 506, 507, 508, 509, 510, 511, 512, 513, 514, 515, 516, 517, 518, 519, 520, 521, 528, 529, 530, 531, 532, 571, 572, 573, 574, 575, 585, 586, 587, 588, 589, 653, 654, 655, 656, 657, 689, 690, 691, 692, 693, 792, 793, 794, 795, 796, 797, 798],
REDDIT = [27, 28, 29, 30, 31, 66, 67, 68, 69, 399, 400, 401, 402, 403, 404, 405, 406, 407, 408, 409, 410, 580, 581, 582, 583, 584, 595, 596, 597, 598, 599, 600, 601, 602, 603, 604, 605, 606, 607, 608, 609, 610, 611, 612, 613, 614, 615, 616, 617, 618, 619, 620, 621, 622, 623, 624, 625, 626, 627, 628, 629, 630, 631, 632, 633, 634, 635, 636, 637, 638, 639, 640, 641, 642, 643, 644, 645, 646, 647, 648, 649, 650, 651, 652, 714, 715, 716, 717, 718, 719, 720, 721, 722, 723, 724, 725, 726, 727, 728, 729, 730, 731, 732, 733, 734, 735],
GITLAB = [44, 45, 46, 102, 103, 104, 105, 106, 132, 133, 134, 135, 136, 156, 168, 169, 170, 171, 172, 173, 174, 175, 176, 177, 178, 179, 180, 181, 182, 205, 206, 207, 258, 259, 293, 294, 295, 296, 297, 303, 304, 305, 306, 307, 308, 309, 310, 311, 312, 314, 315, 316, 317, 318, 339, 340, 341, 342, 343, 349, 350, 357, 389, 390, 391, 392, 393, 394, 395, 396, 397, 398, 411, 412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 441, 442, 443, 444, 445, 446, 447, 448, 449, 450, 451, 452, 475, 476, 477, 478, 479, 480, 481, 482, 483, 484, 485, 522, 523, 524, 525, 526, 527, 533, 534, 535, 536, 537, 567, 568, 569, 570, 576, 577, 578, 579, 590, 591, 592, 593, 594, 658, 659, 660, 661, 662, 663, 664, 665, 666, 667, 668, 669, 670, 736, 742, 743, 744, 745, 746, 747, 748, 749, 750, 751, 752, 753, 754, 755, 756, 783, 784, 785, 786, 787, 788, 789, 799, 800, 801, 802, 803, 804, 805, 806, 807, 808, 809, 810, 811],
MULTISITE = [97, 265, 266, 267, 268, 424, 425, 426, 427, 428, 429, 430, 552, 553, 554, 555, 556, 557, 558, 559, 560, 561, 562, 563, 564, 565, 566, 671, 672, 673, 674, 675, 681, 682, 683, 684, 685, 686, 687, 688, 737, 738, 739, 740, 741, 759, 760, 791]

@dataclass(frozen=True)
class TaskRunOutput:
    task_id: str
    status: str
    reason: str


def _make_llm(runner_cfg: RunnerConfig, *, key_num: int) -> OpenAICompatClient:
    if runner_cfg.llm_type == "deepseek":
        llm_cfg = preset_deepseek_chat(key_num=key_num)
    elif runner_cfg.llm_type == "qwen32b":
        llm_cfg = preset_qwen32b(key_num=key_num)
    elif runner_cfg.llm_type == "gpt-4o":
        llm_cfg = preset_chatgpt(key_num=key_num)
    else:
        raise ValueError(f"Unknown llm_type: {runner_cfg.llm_type}")
    return OpenAICompatClient(llm_cfg)


def _load_tasks_for_dataset(
    *,
    dataset: str,
    webarena_root: Path,
    dataset_path: Optional[Path],
) -> Dict[int, TaskSpec]:
    dataset = dataset.lower().strip()

    if dataset == "webarena":
        adapter = WebArenaAdapter(webarena_root=webarena_root)
        return {t.task_id: t for t in adapter.iter_tasks()}

    if dataset in {"online_mind2web", "online-mind2web", "mind2web_online"}:
        if dataset_path is None:
            raise ValueError("online_mind2web requires dataset_path, e.g. Path('./Online_Mind2Web.json')")

        adapter = OnlineMind2WebAdapter(data_path=dataset_path)
        return {t.task_id: t for t in adapter.iter_tasks()}

    raise ValueError(f"Unknown dataset: {dataset}")


async def _run_one_task(
    *,
    task: TaskSpec,
    webarena_root: Path,
    headless: bool,
    max_rounds: int,
    max_parallel: int,
    table_max_rows: int,
    key_num: int,
) -> TaskRunOutput:
    sess = PlaywrightSession(PlaywrightConfig(headless=headless))
    tree = SemanticTree()
    runner_cfg = RunnerConfig()
    llm = _make_llm(runner_cfg, key_num=key_num)

    rec = Recorder(
        base_dir=Path("record"),
        task=task,
        model_name=runner_cfg.llm_type,
    )

    meta = asdict(runner_cfg)
    meta["max_rounds"] = int(max_rounds)
    meta["max_parallel"] = int(max_parallel)
    meta["table_max_rows"] = int(table_max_rows)
    meta["llm_type"] = runner_cfg.llm_type
    meta["key_num"] = int(key_num)
    meta["dataset"] = str(task.dataset)
    rec.write_meta(meta=meta)

    print("\n================= [RUNNER INPUT] =================")
    print(f"dataset        {task.dataset}")
    print(f"task_id        {task.task_id}")
    print(f"start_url      {task.start_url}")
    print(f"require_login  {task.require_login}")
    print(f"intent         {task.intent}")
    print(f"key_num(lane)  {key_num}")
    print("==================================================\n")

    cons_agent = ConstraintAgent(llm=llm, task=task)
    constraints = await cons_agent.run()
    print(f"[MAIN] {constraints}")
    rec.write_task_info(task, constraints=constraints)

    extractor = DataExtractionAgent(task=task, llm=llm, tree=tree, sess=sess, rec=rec)

    evaluator = None
    if task.dataset == "webarena":
        evaluator = WebArenaEvaluator(task=task, llm=llm, sess=sess)

    # Mind2Web does not need evaluator. We pass None here.
    # If your Executor does not accept evaluator=None, modify Executor to make it optional.
    executor = Executor(sess=sess, tree=tree, rec=rec, data_agent=extractor, evaluator=evaluator)

    planner = PlannerAgent(llm=llm, q=task.intent, constraints=constraints, tree=tree, rec=rec)
    decision = DecisionAgent(
        llm=llm,
        q=task.intent,
        constraints=constraints,
        executor=executor,
        tree=tree,
        sess=sess,
        rec=rec,
    )

    runner = ExperimentRunner(sess=sess, tree=tree, planner=planner, decision=decision, task=task, rec=rec)

    storage_state = task.storage_state_abs(webarena_root)

    res = await runner.run(
        start_url=task.start_url,
        storage_state=storage_state if task.require_login else None,
        geolocation=task.geolocation,
    )

    print("\n================= [RUNNER RESULT] =================")
    print(f"dataset        {task.dataset}")
    print(f"task_id        {task.task_id}")
    print(f"task           {task.intent}")
    print(f"status         {res.status}")
    print(f"answer         {res.answer}")
    if res.status == "finish" and evaluator is not None:
        print(f"eval           {res.eval_result}")
    print("===================================================\n")

    return TaskRunOutput(
        task_id=str(task.task_id),
        status=str(res.status),
        reason=str(res.answer),
    )


async def _run_lane_async(
    *,
    lane_id: int,
    task_ids: Sequence[int],
    dataset: str,
    dataset_path: Optional[Path],
    webarena_root: Path,
    headless: bool,
    max_rounds: int,
    max_parallel: int,
    table_max_rows: int,
) -> List[TaskRunOutput]:
    all_tasks = _load_tasks_for_dataset(
        dataset=dataset,
        dataset_path=dataset_path,
        webarena_root=webarena_root,
    )

    outs: List[TaskRunOutput] = []
    for tid in task_ids:
        if tid not in all_tasks:
            raise ValueError(f"Unknown task_id: {tid}")

        out = await _run_one_task(
            task=all_tasks[tid],
            webarena_root=webarena_root,
            headless=headless,
            max_rounds=max_rounds,
            max_parallel=max_parallel,
            table_max_rows=table_max_rows,
            key_num=lane_id,
        )
        outs.append(out)

    return outs


def _lane_entrypoint(
    lane_id: int,
    task_ids: List[int],
    cfg: Dict[str, Any],
    out_q: mp.Queue,
) -> None:
    outs = asyncio.run(
        _run_lane_async(
            lane_id=lane_id,
            task_ids=task_ids,
            dataset=str(cfg["dataset"]),
            dataset_path=Path(cfg["dataset_path"]) if cfg.get("dataset_path") else None,
            webarena_root=Path(cfg["webarena_root"]),
            headless=bool(cfg["headless"]),
            max_rounds=int(cfg["max_rounds"]),
            max_parallel=int(cfg["max_parallel"]),
            table_max_rows=int(cfg["table_max_rows"]),
        )
    )
    out_q.put(("ok", lane_id, [o.__dict__ for o in outs]))


def amain_process(
    *,
    dataset: str = "webarena",
    dataset_path: Optional[Path] = None,
    webarena_root: Path,
    lane_task_ids: Sequence[Sequence[int]],
    headless: bool = True,
    max_rounds: int = 8,
    max_parallel: int = 4,
    table_max_rows: int = 6,
) -> int:
    ctx = mp.get_context("spawn")
    out_q: mp.Queue = ctx.Queue()

    cfg = {
        "dataset": dataset,
        "dataset_path": str(dataset_path) if dataset_path else "",
        "webarena_root": str(webarena_root),
        "headless": headless,
        "max_rounds": max_rounds,
        "max_parallel": max_parallel,
        "table_max_rows": table_max_rows,
    }

    procs: List[mp.Process] = []
    for lane_id, ids in enumerate(lane_task_ids):
        ids_list = list(ids)
        if not ids_list:
            continue
        p = ctx.Process(
            target=_lane_entrypoint,
            args=(lane_id, ids_list, cfg, out_q),
            daemon=False,
        )
        p.start()
        procs.append(p)

    results: List[Dict[str, Any]] = []
    for _ in range(len(procs)):
        kind, lane_id, payload = out_q.get()
        results.append({"kind": kind, "lane_id": lane_id, "payload": payload})

    for p in procs:
        p.join()

    print("\n================= [BATCH SUMMARY] =================")
    for r in sorted(results, key=lambda x: x["lane_id"]):
        if r["kind"] == "ok":
            for o in r["payload"]:
                print(f"lane={r['lane_id']} task_id={o['task_id']} status={o['status']} answer={o['reason']}")
        else:
            print(f"lane={r['lane_id']} ERROR {r['payload']}")
    print("===================================================\n")

    if any(r["kind"] == "error" for r in results):
        return 2
    return 0


def main() -> None:
    #
    # WebArena
    dataset = "webarena"
    dataset_path = None
    #
    # Online Mind2Web
    # dataset = "online_mind2web"
    # dataset_path = Path("./webtactix/datasets/Online_Mind2Web.json")

    exit_code = amain_process(
        dataset=dataset,
        dataset_path=dataset_path,
        webarena_root=Path("./webarena"),
        lane_task_ids=[SHOPPING, SHOPPING_ADMIN, MAP, REDDIT, MULTISITE],
        headless=False,
        max_rounds=8,
        max_parallel=40,
        table_max_rows=10,
    )
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()