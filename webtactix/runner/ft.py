import asyncio
import time

from playwright.async_api import async_playwright

async def main():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)  # 显式设置为非 headless
        page = await browser.new_page()
        await page.goto("http://127.0.0.1:7780/admin")
        print(time.time())
        input()
        # await asyncio.sleep(180)  # 保持10秒，方便你观察

asyncio.run(main())