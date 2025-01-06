import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

from browser_use.agent.views import AgentHistoryList
from src.agent.custom_agent import CustomAgent
from src.browser.custom_browser import CustomBrowser, BrowserConfig
from src.browser.custom_context import BrowserContext, BrowserContextConfig
from src.controller.custom_controller import CustomController
from src.agent.custom_prompts import CustomSystemPrompt
from browser_use.browser.context import BrowserContextWindowSize

@dataclass
class ChatMessage:
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

class ChatService:
    def __init__(self, llm, window_size=(1920, 1080)):
        self.llm = llm
        self.window_w, self.window_h = window_size
        self.messages: List[ChatMessage] = []
        self.browser = None
        self.browser_context = None
        self.controller = CustomController()

    async def initialize_browser(self, headless=False, disable_security=True):
        """Initialize browser if not already running"""
        if not self.browser:
            self.browser = CustomBrowser(
                config=BrowserConfig(
                    headless=headless,
                    disable_security=disable_security,
                    extra_chromium_args=[f'--window-size={self.window_w},{self.window_h}'],
                )
            )
            self.browser_context = await self.browser.new_context(
                config=BrowserContextConfig(
                    trace_path='./tmp/traces',
                    save_recording_path="./tmp/chat_recordings",
                    no_viewport=False,
                    browser_window_size=BrowserContextWindowSize(
                        width=self.window_w, 
                        height=self.window_h
                    ),
                )
            )

    async def close_browser(self):
        """Close browser and cleanup"""
        if self.browser_context:
            await self.browser_context.close()
        if self.browser:
            await self.browser.close()
            self.browser = None
            self.browser_context = None

    def add_message(self, role: str, content: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a message to the chat history"""
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now(),
            metadata=metadata
        )
        self.messages.append(message)
        return message

    async def process_user_message(self, message: str) -> ChatMessage:
        """Process a user message and return agent response"""
        try:
            # Initialize browser if needed
            await self.initialize_browser()

            # Add user message to history
            self.add_message("user", message)

            # Create and run agent
            agent = CustomAgent(
                task=message,
                add_infos="",
                llm=self.llm,
                browser_context=self.browser_context,
                controller=self.controller,
                system_prompt_class=CustomSystemPrompt,
                use_vision=True
            )

            history: AgentHistoryList = await agent.run(max_steps=10)

            # Extract results
            result = history.final_result()
            errors = history.errors()
            actions = history.model_actions()
            thoughts = history.model_thoughts()

            # Create response message
            response_content = f"Result: {result}\n"
            if errors:
                response_content += f"\nErrors: {errors}"

            # Add response with metadata
            return self.add_message(
                "assistant", 
                response_content,
                metadata={
                    "actions": actions,
                    "thoughts": thoughts,
                    "errors": errors
                }
            )

        except Exception as e:
            error_msg = f"Error processing request: {str(e)}"
            return self.add_message("assistant", error_msg, metadata={"error": str(e)})

    def get_chat_history(self) -> List[ChatMessage]:
        """Get the full chat history"""
        return self.messages

    def clear_chat_history(self):
        """Clear the chat history"""
        self.messages.clear()
