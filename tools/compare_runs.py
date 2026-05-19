#!/usr/bin/env python3
"""
Compare all runs for all task IDs in a record directory.

Usage:
    python tools/compare_runs.py <record_dir>
    python tools/compare_runs.py record/webtactix/250326
    python tools/compare_runs.py record/webtactix/250326 --task task_597
    python tools/compare_runs.py record/webtactix/250326 --json > report.json
"""

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path


SKIP_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}

# Fields that differ purely due to timing — not semantically meaningful
TIMING_KEYS = {"t_start", "t_end", "duration", "t_round_start", "time_utc", "timestamp", "t"}


def md5(path: Path) -> str:
    h = hashlib.md5()
    h.update(path.read_bytes())
    return h.hexdigest()


def semantic_md5(path: Path) -> str:
    """MD5 after stripping timing-only keys from JSON, so timing noise doesn't show as a diff."""
    if path.suffix != ".json":
        return md5(path)
    try:
        data = json.loads(path.read_text())
        stripped = strip_timing(data)
        return hashlib.md5(json.dumps(stripped, sort_keys=True).encode()).hexdigest()
    except Exception:
        return md5(path)


def strip_timing(obj):
    if isinstance(obj, dict):
        return {k: strip_timing(v) for k, v in obj.items() if k not in TIMING_KEYS}
    if isinstance(obj, list):
        return [strip_timing(i) for i in obj]
    return obj


def collect_files(task_dir: Path) -> dict[str, Path]:
    """Return {relative_path: absolute_path} for all non-image files under task_dir."""
    files = {}
    for p in sorted(task_dir.rglob("*")):
        if p.is_file() and p.suffix not in SKIP_EXTENSIONS:
            rel = str(p.relative_to(task_dir))
            files[rel] = p
    return files


def compare_task(task_id: str, run_dirs: dict[str, Path]) -> dict:
    """
    Compare all files for a single task_id across multiple runs.
    Returns a structured report.
    """
    # Gather all relative file paths that exist in any run
    all_files: set[str] = set()
    run_files: dict[str, dict[str, Path]] = {}
    for run, task_dir in run_dirs.items():
        rf = collect_files(task_dir)
        run_files[run] = rf
        all_files.update(rf.keys())

    runs = sorted(run_dirs.keys())
    file_report = {}

    for rel in sorted(all_files):
        present_in = [r for r in runs if rel in run_files[r]]
        missing_in = [r for r in runs if rel not in run_files[r]]

        hashes = {r: md5(run_files[r][rel]) for r in present_in}
        sem_hashes = {r: semantic_md5(run_files[r][rel]) for r in present_in}

        unique_hashes = set(hashes.values())
        unique_sem = set(sem_hashes.values())

        identical = len(unique_hashes) == 1
        sem_identical = len(unique_sem) == 1

        # Group runs by hash
        hash_groups: dict[str, list[str]] = defaultdict(list)
        for r, h in sem_hashes.items():
            hash_groups[h].append(r)

        file_report[rel] = {
            "present_in": present_in,
            "missing_in": missing_in,
            "identical": identical,
            "sem_identical": sem_identical,  # identical after stripping timing fields
            "groups": [sorted(v) for v in hash_groups.values()],
        }

    # Find the first round where semantic content diverges
    first_divergence = None
    for rel in sorted(all_files):
        info = file_report[rel]
        if not info["sem_identical"] and len(info["present_in"]) > 1:
            first_divergence = rel
            break

    return {
        "task_id": task_id,
        "runs": runs,
        "first_semantic_divergence": first_divergence,
        "files": file_report,
    }


def print_report(report: dict, verbose: bool = False):
    task_id = report["task_id"]
    runs = report["runs"]
    files = report["files"]
    diverg = report["first_semantic_divergence"]

    total = len(files)
    identical_count = sum(1 for f in files.values() if f["sem_identical"] and not f["missing_in"])
    diverged_count = sum(1 for f in files.values() if not f["sem_identical"])
    missing_count = sum(1 for f in files.values() if f["missing_in"])

    print(f"\n{'='*70}")
    print(f"  TASK: {task_id}   RUNS: {', '.join(runs)}")
    print(f"{'='*70}")
    print(f"  Files compared : {total}")
    print(f"  Identical      : {identical_count}")
    print(f"  Diverged       : {diverged_count}")
    print(f"  Missing in ≥1  : {missing_count}")
    print(f"  First divergence: {diverg or 'none'}")

    # Always print diverged files
    diverged = {k: v for k, v in files.items() if not v["sem_identical"]}
    if diverged:
        print(f"\n  {'─'*60}")
        print(f"  DIVERGED FILES ({len(diverged)}):")
        print(f"  {'─'*60}")
        for rel, info in diverged.items():
            groups = info["groups"]
            group_str = "  |  ".join("+".join(g) for g in groups)
            print(f"  {rel}")
            print(f"    groups: [{group_str}]")
            if info["missing_in"]:
                print(f"    missing in: {info['missing_in']}")

    # Only print identical files if verbose
    if verbose:
        identical = {k: v for k, v in files.items() if v["sem_identical"] and not v["missing_in"]}
        if identical:
            print(f"\n  {'─'*60}")
            print(f"  IDENTICAL FILES ({len(identical)}):")
            print(f"  {'─'*60}")
            for rel in identical:
                print(f"  ✓ {rel}")

    # Files only in some runs
    missing = {k: v for k, v in files.items() if v["missing_in"]}
    if missing:
        print(f"\n  {'─'*60}")
        print(f"  MISSING IN SOME RUNS ({len(missing)}):")
        print(f"  {'─'*60}")
        for rel, info in missing.items():
            print(f"  {rel}")
            print(f"    present: {info['present_in']}  missing: {info['missing_in']}")


def main():
    parser = argparse.ArgumentParser(description="Compare runs across tasks in a record directory.")
    parser.add_argument("record_dir", help="Directory containing run_0, run_1, ... subdirectories")
    parser.add_argument("--task", help="Only compare a specific task ID (e.g. task_597)")
    parser.add_argument("--json", action="store_true", help="Output full report as JSON")
    parser.add_argument("--verbose", "-v", action="store_true", help="Also list identical files")
    args = parser.parse_args()

    base = Path(args.record_dir)
    if not base.is_dir():
        print(f"ERROR: {base} is not a directory", file=sys.stderr)
        sys.exit(1)

    # Find all run_* directories
    run_dirs = sorted([d for d in base.iterdir() if d.is_dir() and d.name.startswith("run_")])
    if not run_dirs:
        print(f"ERROR: no run_* directories found in {base}", file=sys.stderr)
        sys.exit(1)

    # Collect all task IDs across all runs
    task_to_runs: dict[str, dict[str, Path]] = defaultdict(dict)
    for run_dir in run_dirs:
        for task_dir in sorted(run_dir.iterdir()):
            if task_dir.is_dir() and task_dir.name.startswith("task_"):
                task_to_runs[task_dir.name][run_dir.name] = task_dir

    if args.task:
        if args.task not in task_to_runs:
            print(f"ERROR: {args.task} not found. Available: {sorted(task_to_runs)}", file=sys.stderr)
            sys.exit(1)
        task_to_runs = {args.task: task_to_runs[args.task]}

    # Only compare tasks that appear in ≥2 runs
    comparable = {t: r for t, r in task_to_runs.items() if len(r) >= 2}
    only_one = [t for t, r in task_to_runs.items() if len(r) < 2]

    if not comparable:
        print("No tasks appear in more than one run — nothing to compare.")
        sys.exit(0)

    all_reports = []
    for task_id in sorted(comparable):
        report = compare_task(task_id, comparable[task_id])
        all_reports.append(report)

    if args.json:
        print(json.dumps(all_reports, indent=2))
        return

    print(f"\nRecord dir : {base}")
    print(f"Runs found : {[d.name for d in run_dirs]}")
    print(f"Tasks in ≥2 runs: {len(comparable)}  |  Tasks in only 1 run: {len(only_one)}")
    if only_one:
        print(f"  (skipped single-run tasks: {only_one})")

    for report in all_reports:
        print_report(report, verbose=args.verbose)

    # Summary table
    print(f"\n{'='*70}")
    print("  SUMMARY ACROSS ALL TASKS")
    print(f"{'='*70}")
    print(f"  {'TASK':<15} {'RUNS':<20} {'IDENTICAL':>10} {'DIVERGED':>10} {'FIRST DIVERGENCE'}")
    print(f"  {'─'*65}")
    for r in all_reports:
        files = r["files"]
        runs_str = "+".join(r["runs"])
        ident = sum(1 for f in files.values() if f["sem_identical"] and not f["missing_in"])
        div = sum(1 for f in files.values() if not f["sem_identical"])
        first = r["first_semantic_divergence"] or "—"
        print(f"  {r['task_id']:<15} {runs_str:<20} {ident:>10} {div:>10}  {first}")


if __name__ == "__main__":
    main()
