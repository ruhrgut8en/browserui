# -*- coding: utf-8 -*-
# @Time    : 2025/1/2
# @Author  : wenshao
# @ProjectName: browser-use-webui
# @FileName: test_browser_use.py
import pdb

from dotenv import load_dotenv

load_dotenv()
import sys

sys.path.append(".")
import os
import sys
from pprint import pprint

import asyncio
from browser_use import Agent
from browser_use.agent.views import AgentHistoryList

from src.utils import utils


async def test_browser_use_org():
    from browser_use.browser.browser import Browser, BrowserConfig
    from browser_use.browser.context import (
        BrowserContext,
        BrowserContextConfig,
        BrowserContextWindowSize,
    )
    llm = utils.get_llm_model(
        provider="azure_openai",
        model_name="gpt-4o",
        temperature=0.8,
        base_url=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", "")
    )

    window_w, window_h = 1920, 1080

    browser = Browser(
        config=BrowserConfig(
            headless=False,
            disable_security=True,
            extra_chromium_args=[f'--window-size={window_w},{window_h}'],
        )
    )
    async with await browser.new_context(
            config=BrowserContextConfig(
                trace_path='./tmp/traces',
                save_recording_path="./tmp/record_videos",
                no_viewport=False,
                browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h),
            )
    ) as browser_context:
        agent = Agent(
            task="go to google.com and type 'OpenAI' click search and give me the first url",
            llm=llm,
            browser_context=browser_context,
        )
        history: AgentHistoryList = await agent.run(max_steps=10)

        print('Final Result:')
        pprint(history.final_result(), indent=4)

        print('\nErrors:')
        pprint(history.errors(), indent=4)

        # e.g. xPaths the model clicked on
        print('\nModel Outputs:')
        pprint(history.model_actions(), indent=4)

        print('\nThoughts:')
        pprint(history.model_thoughts(), indent=4)
    # close browser
    await browser.close()


async def test_browser_use_custom():
    from playwright.async_api import async_playwright
    from browser_use.browser.context import BrowserContextWindowSize

    from src.browser.custom_browser import CustomBrowser, BrowserConfig
    from src.browser.custom_context import BrowserContext, BrowserContextConfig
    from src.controller.custom_controller import CustomController
    from src.agent.custom_agent import CustomAgent
    from src.agent.custom_prompts import CustomSystemPrompt
    from src.browser.custom_context import CustomBrowserContext

    window_w, window_h = 1920, 1080

    # llm = utils.get_llm_model(
    #     provider="azure_openai",
    #     model_name="gpt-4o",
    #     temperature=0.8,
    #     base_url=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
    #     api_key=os.getenv("AZURE_OPENAI_API_KEY", "")
    # )

    # llm = utils.get_llm_model(
    #     provider="gemini",
    #     model_name="gemini-2.0-flash-exp",
    #     temperature=1.0,
    #     api_key=os.getenv("GOOGLE_API_KEY", "")
    # )

    # llm = utils.get_llm_model(
    #     provider="deepseek",
    #     model_name="deepseek-chat",
    #     temperature=0.8
    # )

    llm = utils.get_llm_model(
        provider="ollama",
        model_name="qwen2.5:7b",
        temperature=0.8
    )

    controller = CustomController()
    use_own_browser = False
    disable_security = True
    use_vision = False
    playwright = None
    browser_context_ = None
    try:
        if use_own_browser:
            playwright = await async_playwright().start()
            chrome_exe = os.getenv("CHROME_PATH", "")
            chrome_use_data = os.getenv("CHROME_USER_DATA", "")
            browser_context_ = await playwright.chromium.launch_persistent_context(
                user_data_dir=chrome_use_data,
                executable_path=chrome_exe,
                no_viewport=False,
                headless=False,  # 保持浏览器窗口可见
                user_agent=(
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                    '(KHTML, like Gecko) Chrome/85.0.4183.102 Safari/537.36'
                ),
                java_script_enabled=True,
                bypass_csp=disable_security,
                ignore_https_errors=disable_security,
                record_video_dir="./tmp/record_videos",
                record_video_size={'width': window_w, 'height': window_h}
            )
        else:
            browser_context_ = None

        browser = CustomBrowser(
            config=BrowserConfig(
                headless=False,
                disable_security=True,
                extra_chromium_args=[f'--window-size={window_w},{window_h}'],
            )
        )

        async with await browser.new_context(
                config=BrowserContextConfig(
                    trace_path='./tmp/result_processing',
                    save_recording_path="./tmp/record_videos",
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(width=window_w, height=window_h),
                ),
                context=browser_context_
        ) as browser_context:
            agent = CustomAgent(
                task="go to google.com and type 'OpenAI' click search and give me the first url",
                add_infos="",  # some hints for llm to complete the task
                llm=llm,
                browser_context=browser_context,
                controller=controller,
                system_prompt_class=CustomSystemPrompt,
                use_vision=use_vision
            )
            history: AgentHistoryList = await agent.run(max_steps=10)

            print('Final Result:')
            pprint(history.final_result(), indent=4)

            print('\nErrors:')
            pprint(history.errors(), indent=4)

            # e.g. xPaths the model clicked on
            print('\nModel Outputs:')
            pprint(history.model_actions(), indent=4)

            print('\nThoughts:')
            pprint(history.model_thoughts(), indent=4)
            # close browser
    except Exception as e:
        import traceback
        traceback.print_exc()
    finally:
        # 显式关闭持久化上下文
        if browser_context_:
            await browser_context_.close()

        # 关闭 Playwright 对象
        if playwright:
            await playwright.stop()

        await browser.close()


if __name__ == '__main__':
    # asyncio.run(test_browser_use_org())
    asyncio.run(test_browser_use_custom())
