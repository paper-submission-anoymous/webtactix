"""
Standalone script to run ConstraintAgent on a single WebArena config file.

Usage:
    python run_constraint_agent.py
    python run_constraint_agent.py --config /path/to/407.json
"""
import asyncio
import json
import argparse
from pathlib import Path

from webtactix import set_seeds
from webtactix.core.schemas import TaskSpec
from webtactix.llm.vllm_client import VLLMClient, VLLMConfig
from webtactix.agents.constraint_agent import ConstraintAgent


def load_task_from_config(config_path: Path) -> TaskSpec:
    raw = json.loads(config_path.read_text())
    return TaskSpec(
        dataset="webarena",
        task_id=int(raw["task_id"]),
        intent=str(raw["intent"]),
        start_url=str(raw["start_url"]),
        sites=raw.get("sites", []),
        require_login=bool(raw.get("require_login", False)),
        storage_state_path=raw.get("storage_state"),
        geolocation=raw.get("geolocation"),
        require_reset=bool(raw.get("require_reset", False)),
        intent_template=raw.get("intent_template", ""),
        instantiation_dict=raw.get("instantiation_dict", {}),
        source_path=str(config_path),
    )


async def main(config_path: Path) -> None:
    task = load_task_from_config(config_path)
    print(f"task_id : {task.task_id}")
    print(f"intent  : {task.intent}")
    print()

    llm = VLLMClient(VLLMConfig())
    agent = ConstraintAgent(llm=llm, task=task)
    constraints = await agent.run()

    print(f"Constraints ({len(constraints)}):")
    for c in constraints:
        print(f"  [{c.kind}] {c.text}")


if __name__ == "__main__":
    set_seeds()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("/home/ivohra6/benchmarks/webarena/config_files/581.json"),
    )
    args = parser.parse_args()
    asyncio.run(main(args.config))
