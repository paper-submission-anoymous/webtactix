from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re
from playwright.async_api import async_playwright, Browser, BrowserContext
from webtactix.core.schemas import ActionStep, ActionType
import yaml, time
from typing import Callable, Iterable, Optional, Set
import asyncio
from playwright.async_api import Page, Request, Dialog

CLOSE_NAME_RE = re.compile(r"I Accept|Continue|Allow All Cookies|Accept all|accept|save|no thanks|不要", re.I)

def start_modal_watcher(page: Page, interval: float = 0.25) -> asyncio.Task:
    async def watcher():
        while True:
            try:
                if page.is_closed():
                    return

                # 兼容 iframe：很多订阅弹窗在 frame 里
                for frame in page.frames:
                    # 找所有 aria dialog（DOM modal）
                    dialogs = frame.get_by_role("dialog")
                    count = await dialogs.count()
                    if count == 0:
                        continue

                    # 遍历可见的 dialog，尝试点关闭
                    for i in range(min(count, 5)):  # 上限防炸
                        dlg = dialogs.nth(i)
                        if not await dlg.is_visible():
                            continue

                        # 优先找语义化的 close button
                        close_btn = dlg.get_by_role("button", name=CLOSE_NAME_RE).first
                        if await close_btn.count() > 0 and await close_btn.is_visible():
                            await close_btn.click(timeout=500)
                            continue

                        # 再兜底：aria-label 类 close
                        # close_btn2 = dlg.locator(
                        #     "[aria-label*='Close' i], [aria-label*='关闭' i]"
                        # ).first
                        # if await close_btn2.count() > 0 and await close_btn2.is_visible():
                        #     await close_btn2.click(timeout=500)
                        #     continue

                        # 最后兜底：ESC（不少 modal 支持）
                        # try:
                        #     await page.keyboard.press("Escape")
                        # except Exception:
                        #     pass

            except Exception:
                # 任何异常都吞掉，避免 watcher 把主流程搞崩
                pass

            await asyncio.sleep(interval)

    return asyncio.create_task(watcher())

async def _capture_new_page(page: Page, click_coro, timeout_ms: int = 10000) -> Page | None:
    """
    在执行 click_coro() 的同时，捕获是否出现新 tab（new Page）。
    成功返回 new_page，否则返回 None。
    """
    ctx = page.context
    before = set(ctx.pages)

    try:
        async with ctx.expect_page(timeout=timeout_ms) as pinfo:
            await click_coro()
        new_page = await pinfo.value
        # 等到至少 DOMReady，后面你的 wait_for_page_stable 还能再稳一遍
        await new_page.wait_for_load_state("domcontentloaded")
        return new_page
    except Exception:
        # 没捕获到新页：给一点点时间让 page 注册进 context
        await asyncio.sleep(1)
        after = [p for p in ctx.pages if p not in before]
        return after[-1] if after else None

@dataclass
class PlaywrightConfig:
    headless: bool = True
    slow_mo_ms: int = 0
    viewport: Optional[dict] = None


class PlaywrightSession:
    """
    Minimal Playwright wrapper:
    - create context with storage_state (cookies + localStorage)
    - open a page and navigate
    """

    def __init__(self, cfg: PlaywrightConfig) -> None:
        self.cfg = cfg
        self._pw = None
        self._browser: Optional[Browser] = None
        self._context: Optional[BrowserContext] = None

    async def start(self, storage_state: Optional[Path] = None, geolocation: Any = None) -> None:
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=self.cfg.headless,
            slow_mo=self.cfg.slow_mo_ms,
            args=["--disable-http2"]
        )

        context_kwargs = {}
        if storage_state is not None:
            context_kwargs["storage_state"] = str(storage_state)
        if geolocation is not None:
            context_kwargs["geolocation"] = geolocation
        if self.cfg.viewport is not None:
            context_kwargs["viewport"] = self.cfg.viewport

        self._context = await self._browser.new_context(**context_kwargs)

    async def accept_cookies_if_present(self, page: Page) -> None:
        candidates = [
            "button:has-text('Accept')",
            "button:has-text('Accept all')",
            "button:has-text('Agree')",
            "button:has-text('I agree')",
            "button:has-text('同意')",
            "button:has-text('全部接受')",
            "button:has-text('接受全部')",
            "text=Accept all",
        ]

        for sel in candidates:
            try:
                btn = page.locator(sel).first
                if await btn.is_visible(timeout=800):
                    await btn.click(timeout=800)
                    return
            except Exception:
                pass

    async def new_page(self) -> Page:
        if self._context is None:
            raise RuntimeError("Session not started")

        page = await self._context.new_page()

        # ✅ 原生 JS dialog：真的能事件监听到
        async def handle_js_dialog(d: Dialog):
            print("js dialog:", d.type, d.message)
            await d.accept()

        page.on("dialog", handle_js_dialog)

        # ✅ DOM modal：用后台 watcher 自动关（不需要你手动在 goto 后点）
        page._modal_watcher_task = start_modal_watcher(page)

        return page

    async def goto(self, page: Page, url: str, timeout_ms: int = 30000) -> None:
        await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        await wait_for_page_stable(page)
        await self.accept_cookies_if_present(page)


    async def close(self) -> None:
        if self._context is not None:
            await self._context.close()
            self._context = None
        if self._browser is not None:
            await self._browser.close()
            self._browser = None
        if self._pw is not None:
            await self._pw.stop()
            self._pw = None

    @staticmethod
    async def get_snapshot(page) -> object:
        try:
            frame = page.main_frame
            raw_yaml = await frame.locator("body").aria_snapshot(timeout=60000)
            return yaml.safe_load(raw_yaml)
        except Exception as e:
            print(f"[SNPASHOT ERROR] {e}")
            return {}

    @staticmethod
    async def apply_step(page: Page, step: ActionStep, replay: bool) -> Page:
        await page.mouse.wheel(0, 100)
        frame = page.main_frame

        if step.action == ActionType.GOTO:
            if page.url != step.text:
                await page.goto(url=step.text, timeout=30000)
                await wait_for_page_stable(page, replay=replay)
            return page

        if step.action == ActionType.PRESS_ENTER:
            await page.keyboard.press("Enter")
            await wait_for_page_stable(page, replay=replay)
            return page

        loc_with_name = None
        if step.name:
            loc_with_name = frame.get_by_role(step.role, name=step.name, exact=True).nth(step.nth)

        loc_role_only = frame.get_by_role(step.role).nth(step.role_nth)

        async def _do(loc, page: Page, replay: bool, name: bool = True) -> Page:
            if step.action == ActionType.CLICK:
                if name:
                    try:
                        async def _click():
                            await loc.dispatch_event('click', timeout=3000)
                            # await loc.click()

                        newp = await _capture_new_page(page, _click, timeout_ms=10000)
                        if newp is not None:
                            page = newp
                    except Exception as e:
                        print("[CLICK ERR] try1: ", e)
                        raise Exception("[CLICK ERR] ", e)
                else:
                    try:
                        async def _click():
                            await loc.scroll_into_view_if_needed(timeout=3000)
                            box = await loc.bounding_box()
                            if not box:
                                raise Exception("bounding_box is None")
                            await page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)

                        newp = await _capture_new_page(page, _click, timeout_ms=10000)
                        if newp is not None:
                            page = newp
                    except Exception as e:
                        print("[CLICK ERR] try2: ", e)
                        raise Exception("[CLICK ERR] ", e)

            elif step.action == ActionType.INPUT:
                # Focus on the element
                await loc.focus()

                # Wait for the page to stabilize (if necessary)
                await wait_for_page_stable(page, replay=replay)

                # Clear the input field by pressing Backspace repeatedly or use fill("")
                await page.keyboard.press('Control+A')  # Select all
                await page.keyboard.press('Backspace')  # Clear the input

                # Wait for the page to stabilize again before typing
                await wait_for_page_stable(page, replay=replay)

                # Type the text sequentially
                for char in step.text:
                    await page.keyboard.press(char)
                    await page.wait_for_timeout(50)  # Add a small delay between key presses


            elif step.action == ActionType.SELECT:
                try:
                    await page.get_by_label(step.name).select_option(step.text, timeout=3000)
                except Exception as e:
                    raise Exception(
                        f"{loc} cannot be select as {step.text}, if the option visible, you should use click instead. Detailed failure reason: {e}.")

            else:
                raise ValueError(f"Unknown action: {step.action}")

            return page

        # 1) 先用 name 尝试
        if loc_with_name is not None:
            try:
                page = await _do(loc_with_name, page, replay)
                if step.action and step.action != ActionType.INPUT:
                    await wait_for_page_stable(page, replay=replay)
                return page
            except Exception as e:
                print("[EXEC LOC FALLBACK] with_name failed, fallback to role_only. ", e)
                if replay:
                    return page

        # 2) fallback：role_only
        page = await _do(loc_role_only, page, replay, name=False)
        await wait_for_page_stable(page, replay=replay)
        return page


def _ms_since(t0: float) -> int:
    return int((time.monotonic() - t0) * 1000)


async def wait_for_layout_stable(
    page: Page,
    *,
    stable_frames: int = 2,
    timeout_ms: int = 2_000,
) -> None:
    """
    Optional. Uses rAF to ensure the page geometry stays stable for N frames.
    This mirrors Playwright's own notion of "stable" in actionability checks
    (stable bounding box across consecutive animation frames). :contentReference[oaicite:5]{index=5}
    """
    await page.main_frame.wait_for_function(
        """
        (needFrames) => new Promise(resolve => {
          let last = null;
          let ok = 0;

          const snapshot = () => {
            const de = document.documentElement;
            const r = de.getBoundingClientRect();
            return [
              Math.round(r.width), Math.round(r.height),
              de.scrollWidth, de.scrollHeight,
              window.innerWidth, window.innerHeight
            ].join(",");
          };

          const step = () => {
            const cur = snapshot();
            ok = (cur === last) ? (ok + 1) : 0;
            last = cur;
            if (ok >= needFrames) return resolve(true);
            requestAnimationFrame(step);
          };

          requestAnimationFrame(step);
        })
        """,
        arg=stable_frames,
        timeout=timeout_ms,
        polling="raf",
    )


async def wait_for_page_stable(
    page: Page,
    *,
    timeout_ms: int = 10_000,
    domcontentloaded_budget_ms: int = 10000,
    replay=False,
    network_idle_ms: int = 10000,
    layout_stable: bool = True,
) -> None:
    # 1) 很短的 DOMContentLoaded 等待，避免卡死
    start_t = time.time()
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=60000)
        await page.wait_for_load_state("networkidle", timeout=min(domcontentloaded_budget_ms, timeout_ms))
    except Exception:
        print('[PlayWright Err] wait_for_load_state')

    # if not replay:
    #     await asyncio.sleep(10)
    # else:
    #     await asyncio.sleep(1)
    mid_1 = time.time()
    print('[PlayWright 1] ', mid_1-start_t)
    # 2) 网络空闲（事件驱动，比注入补丁稳）
    # try:
    #     await wait_for_network_idle(
    #         page,
    #         timeout_ms=timeout_ms,
    #         idle_ms=network_idle_ms,
    #         ignore_url=lambda u: (
    #             "analytics" in u
    #             or "sockjs" in u
    #             or "hot-update" in u
    #         ),
    #     )
    # except Exception:
    #     print('[PlayWright Err] wait_for_network_idle')
    #
    # mid_2 = time.time()
    # print('[PlayWright 2] ', mid_2-start_t)

    # 3) 可选的布局稳定
    # try:
    #     if not replay:
    #         await wait_for_layout_stable(page, stable_frames=300, timeout_ms=10000)
    #     else:
    #         await wait_for_layout_stable(page, stable_frames=60, timeout_ms=5000)
    # except:
    #     print('[PlayWright Err] wait_for_layout_stable')
    # await asyncio.sleep(1)
    # end_t = time.time()
    # print('[PlayWright END] ', end_t-start_t)

