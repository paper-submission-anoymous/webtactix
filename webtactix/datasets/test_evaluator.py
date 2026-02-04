
from __future__ import annotations

import asyncio
import json

from webtactix.browser.playwright_session import PlaywrightConfig, PlaywrightSession
from webtactix.core.schemas import TaskSpec, EvalSpec
from webtactix.datasets.webarena_evaluator import WebArenaEvaluator


async def amain() -> int:
    sess = PlaywrightSession(PlaywrightConfig(headless=False))
    await sess.start()
    try:
        ev = EvalSpec(
            eval_types=["program_html"],
            program_html=[
                {
                    "url": "https://example.com",
                    "locator": "document.querySelector('h1').innerText",
                    "required_contents": {"exact_match": "Example Domain"},
                },
                {
                    "url": "https://example.com",
                    "locator": "document.querySelector('#not-exist').value",
                    "required_contents": {"must_include": ["anything"]},
                },
            ],
        )

        task = TaskSpec(
            dataset="debug",
            task_id=0,
            intent="debug program_html evaluator",
            start_url="https://example.com",
            eval_spec=ev,
        )

        evaluator = WebArenaEvaluator(task=task, llm=None)

        evaluator.sess = sess

        res = await evaluator._eval_program_html()

        print("\n================= [PROGRAM_HTML TEST RESULT] =================")
        print(f"ok     : {res.ok}")
        print(f"score  : {res.score}")
        print(f"reason : {res.reason}")
        print("details:")
        print(json.dumps(res.details, ensure_ascii=False, indent=2))
        print("==============================================================\n")

        return 0

    finally:
        try:
            await sess.close()
        except Exception:
            pass


def main() -> None:
    raise SystemExit(asyncio.run(amain()))


if __name__ == "__main__":
    main()
