from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import re
from playwright.async_api import async_playwright, Browser, BrowserContext
from playwright.async_api import Error as PWError
from webtactix.core.schemas import ActionStep, ActionType
import yaml, time
from typing import Callable, Iterable, Optional, Set
import asyncio
from playwright.async_api import Page, Request, Dialog

CLOSE_NAME_RE = re.compile(r"I Accept|Continue|Allow All Cookies|Accept all|accept|save|no thanks|不要", re.I)

def start_modal_watcher(page: Page, interval: float = 0.5) -> asyncio.Task:
    async def watcher():
        while True:
            try:
                if page.is_closed():
                    return
                for frame in page.frames:
                    dialogs = frame.get_by_role("dialog")
                    count = await dialogs.count()
                    if count == 0:
                        continue

                    for i in range(min(count, 5)):
                        dlg = dialogs.nth(i)
                        if not await dlg.is_visible():
                            continue

                        close_btn = dlg.get_by_role("button", name=CLOSE_NAME_RE).first
                        if await close_btn.count() > 0 and await close_btn.is_visible():
                            await close_btn.click(timeout=500)
                            continue

                        # close_btn2 = dlg.locator(
                        #     "[aria-label*='Close' i], [aria-label*='关闭' i]"
                        # ).first
                        # if await close_btn2.count() > 0 and await close_btn2.is_visible():
                        #     await close_btn2.click(timeout=500)
                        #     continue

                        # try:
                        #     await page.keyboard.press("Escape")
                        # except Exception:
                        #     pass

            except Exception:
                pass

            await asyncio.sleep(interval)

    return asyncio.create_task(watcher())

RETRY_ERRORS = ("ERR_NETWORK_CHANGED", "ERR_CONNECTION_CLOSED", "ERR_TIMED_OUT")
async def safe_goto(page, url, timeout_ms=60000, retries=3):
    for attempt in range(retries + 1):
        try:
            return await page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
        except PWError as e:
            msg = str(e)
            if any(k in msg for k in RETRY_ERRORS) and attempt < retries:
                await asyncio.sleep(0.8 * (attempt + 1))
                continue
            raise
    return None


async def _capture_new_page(page: Page, click_coro, timeout_ms: int = 10000) -> Page | None:
    ctx = page.context
    before = set(ctx.pages)

    try:
        async with ctx.expect_page(timeout=timeout_ms) as pinfo:
            await click_coro()
        new_page = await pinfo.value
        await new_page.wait_for_load_state("domcontentloaded")
        return new_page
    except Exception:
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
            slow_mo=self.cfg.slow_mo_ms
        )

        context_kwargs = {}
        if storage_state is not None:
            context_kwargs["storage_state"] = str(storage_state)
        if geolocation is not None:
            context_kwargs["geolocation"] = geolocation
        if self.cfg.viewport is not None:
            context_kwargs["viewport"] = self.cfg.viewport

        self._context = await self._browser.new_context(**context_kwargs)

    async def new_page(self) -> Page:
        if self._context is None:
            raise RuntimeError("Session not started")

        page = await self._context.new_page()

        async def handle_js_dialog(d: Dialog):
            print("js dialog:", d.type, d.message)
            await d.accept()

        page.on("dialog", handle_js_dialog)
        page._modal_watcher_task = start_modal_watcher(page)

        return page

    async def goto(self, page: Page, url: str, timeout_ms: int = 60000) -> None:
        await safe_goto(page, url, timeout_ms=timeout_ms)
        await wait_for_page_stable(page)

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
        frame = page.main_frame
        if step.action == ActionType.GOTO:
            if page.url != step.text:
                await safe_goto(page, step.text)
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
                            # await loc.dispatch_event('click', timeout=3000)
                            # await loc.click()
                            await loc.scroll_into_view_if_needed(timeout=5000)
                            box = await loc.bounding_box()
                            if not box:
                                raise Exception("[CLICK ERR] element's bounding_box is None: ")
                            await page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)

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
                if step.role == "textbox" and step.name == "URL" and step.index==11:
                    # If filling, this will cause 500 Server Error
                    print("no filling...")
                    return page
                # try:
                #     await loc.fill(step.text)
                # except Exception as e:
                # Focus on the element
                await loc.focus()
                await wait_for_page_stable(page, replay=replay)

                # Clear the input field by pressing Backspace repeatedly or use fill("")
                await page.keyboard.press('Control+A')  # Select all
                await page.keyboard.press('Backspace')  # Clear the input

                # Wait for the page to stabilize again before typing
                await wait_for_page_stable(page, replay=replay)

                # Type the text sequentially
                for char in step.text:
                    try:
                        await page.keyboard.press(char)
                        await page.wait_for_timeout(50)  # Add a small delay between key presses
                    except Exception as e:
                        print("[PLAYWRIGHT ERR] TYPE: ", char)

            elif step.action == ActionType.SELECT:
                try:
                    await loc.select_option(step.text, timeout=3000)
                except Exception as e:
                    raise Exception(
                        f"{loc} cannot be select as {step.text}, if the option visible, you should use click instead. Detailed failure reason: {e}.")

            else:
                raise ValueError(f"Unknown action: {step.action}")

            return page

        # 1) with name first
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
        timeout_ms: int = 10_000,
        quiet_ms: int = 600,
        stable_frames: int = 2,
        check_spinners: bool = True,
        spinner_selectors: Optional[Iterable[str]] = None,
        allow_websockets: bool = True,
) -> None:
    """
    Best-effort "page is ready":
    - domcontentloaded (+ try load)
    - pending fetch/xhr == 0 and no net activity for quiet_ms
    - no DOM mutations for quiet_ms
    - layout snapshot stable for stable_frames rAF frames
    - optionally: no visible spinner/busy indicator

    Notes:
    - Can't guarantee *everything* is loaded (pages can stream data forever).
    - Designed to be robust and not get stuck on long-lived connections.
    """

    DEFAULT_SPINNER_SELECTORS = [
        # aria / role
        "[aria-busy='true']",
        "[role='progressbar']",
        "[role='status']",

        # common class / id patterns
        ".spinner", ".loading", ".loader", ".progress",
        "[class*='spinner']", "[class*='loading']", "[class*='loader']",
        "[id*='spinner']", "[id*='loading']", "[id*='loader']",

        # frameworks
        ".ant-spin", ".ant-spin-spinning",
        ".MuiCircularProgress-root",
    ]

    selectors = list(spinner_selectors) if spinner_selectors is not None else list(DEFAULT_SPINNER_SELECTORS)

    # 2) Install in-page trackers (fetch/xhr + mutations + layout snapshot)
    await page.evaluate(
        """
        (cfg) => {
          if (window.__pw_ready_installed) return;
          window.__pw_ready_installed = true;

          const state = window.__pw_ready_state = {
            pending: 0,
            lastNet: Date.now(),
            lastMut: Date.now(),
            lastSnap: null,
            stableOk: 0,
          };

          // Track DOM mutations (layout/content churn)
          try {
            const mo = new MutationObserver(() => { state.lastMut = Date.now(); });
            mo.observe(document, { subtree: true, childList: true, attributes: true, characterData: true });
          } catch (e) {}

          // Track fetch
          try {
            const origFetch = window.fetch;
            if (typeof origFetch === "function") {
              window.fetch = function(...args) {
                state.pending += 1;
                state.lastNet = Date.now();
                return origFetch.apply(this, args)
                  .catch((e) => { throw e; })
                  .finally(() => {
                    state.pending = Math.max(0, state.pending - 1);
                    state.lastNet = Date.now();
                  });
              };
            }
          } catch (e) {}

          // Track XHR
          try {
            const XHR = window.XMLHttpRequest;
            if (XHR && XHR.prototype && XHR.prototype.open && XHR.prototype.send) {
              const origSend = XHR.prototype.send;
              XHR.prototype.send = function(...args) {
                state.pending += 1;
                state.lastNet = Date.now();
                this.addEventListener("loadend", () => {
                  state.pending = Math.max(0, state.pending - 1);
                  state.lastNet = Date.now();
                }, { once: true });
                return origSend.apply(this, args);
              };
            }
          } catch (e) {}

          // Layout snapshot per rAF
          const snapshot = () => {
            const de = document.documentElement;
            if (!de) return "no-de";
            const r = de.getBoundingClientRect();
            return [
              Math.round(r.width), Math.round(r.height),
              de.scrollWidth, de.scrollHeight,
              window.innerWidth, window.innerHeight
            ].join(",");
          };

          const step = () => {
            const cur = snapshot();
            if (cur === state.lastSnap) state.stableOk += 1;
            else state.stableOk = 0;
            state.lastSnap = cur;
            requestAnimationFrame(step);
          };
          requestAnimationFrame(step);
        }
        """,
        {"quiet_ms": quiet_ms, "stable_frames": stable_frames, "selectors": selectors},
    )

    # 3) Wait until all conditions satisfied
    await page.wait_for_function(
        """
        (cfg) => {
          const st = window.__pw_ready_state;
          if (!st) return false;

          const now = Date.now();
          const quiet = cfg.quiet_ms ?? 600;
          const needStable = cfg.stable_frames ?? 2;

          // Network quiet: no pending and no recent activity
          const netQuiet = (st.pending === 0) && ((now - st.lastNet) >= quiet);

          // DOM quiet: no recent mutations
          const domQuiet = (now - st.lastMut) >= quiet;

          // Layout stable: stable for N frames
          const layoutStable = st.stableOk >= needStable;

          // Spinner/busy detection (best-effort)
          const isVisible = (el) => {
            if (!el) return false;
            const cs = window.getComputedStyle(el);
            if (!cs) return false;
            if (cs.display === "none" || cs.visibility === "hidden") return false;
            if (parseFloat(cs.opacity || "1") <= 0.01) return false;
            const r = el.getBoundingClientRect();
            return (r.width > 1 && r.height > 1);
          };

          let spinnerGone = true;
          if (cfg.check_spinners) {
            const sels = cfg.selectors || [];
            for (const sel of sels) {
              try {
                const nodes = document.querySelectorAll(sel);
                for (const n of nodes) {
                  if (isVisible(n)) { spinnerGone = false; break; }
                }
              } catch (e) {}
              if (!spinnerGone) break;
            }
          }

          // If page streams forever, you might want to relax netQuiet/domQuiet.
          return netQuiet && domQuiet && layoutStable && spinnerGone;
        }
        """,
        arg={
            "quiet_ms": quiet_ms,
            "stable_frames": stable_frames,
            "check_spinners": check_spinners,
            "selectors": selectors,
            "allow_websockets": allow_websockets,
        },
        timeout=timeout_ms,
        polling=100,
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
    start_t = time.time()
    try:
        await page.wait_for_load_state("domcontentloaded", timeout=60000)
        await page.wait_for_load_state("networkidle", timeout=network_idle_ms)
        await wait_for_layout_stable(page, timeout_ms=6000)
    except Exception:
        print('[PlayWright Err] wait_for_load_state')
    # if not replay:
    #     await asyncio.sleep(10)
    # else:
    #     await asyncio.sleep(1)
    mid_1 = time.time()
    print('[PlayWright 1] ', mid_1-start_t)
