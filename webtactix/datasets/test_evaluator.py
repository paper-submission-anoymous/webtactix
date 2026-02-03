
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
        # 1) 构造一个只跑 program_html 的 TaskSpec
        ev = EvalSpec(
            eval_types=["program_html"],
            program_html=[
                {
                    "url": "https://example.com",
                    # 你的实现里会执行: page.evaluate("() => " + locator)
                    "locator": "document.querySelector('h1').innerText",
                    "required_contents": {"exact_match": "Example Domain"},
                },
                {
                    # 反例: 故意写一个会报错的 locator，验证异常路径是否会 selected_element=""
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

        # 2) 关键: 你的 _eval_program_html 用 self.sess
        #    如果你 __init__ 里没加 sess，这里就手动挂上去
        evaluator.sess = sess

        # 3) 直接调用你改过的 _eval_program_html
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
