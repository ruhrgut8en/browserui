import gradio as gr
from datetime import datetime
from typing import Optional
from src.utils.utils import get_llm_model
from .chat_service import ChatService

class ChatUI:
    def __init__(self):
        self.chat_service: Optional[ChatService] = None
        
    def initialize_chat_service(self, llm_config):
        """Initialize chat service with LLM configuration"""
        try:
            llm = get_llm_model(**llm_config)
            self.chat_service = ChatService(llm)
            return "✅ Chat service initialized successfully!"
        except Exception as e:
            return f"❌ Error initializing chat service: {str(e)}"

    async def handle_message(self, message: str, history):
        """Handle incoming chat messages"""
        if not self.chat_service:
            return "⚠️ Please initialize the chat service first!", history

        try:
            # Process message through chat service
            response = await self.chat_service.process_user_message(message)
            
            # Format response for display
            formatted_response = response.content
            if response.metadata:
                if "errors" in response.metadata and response.metadata["errors"]:
                    formatted_response += f"\n\n❌ Errors occurred: {response.metadata['errors']}"
                if "thoughts" in response.metadata and response.metadata["thoughts"]:
                    formatted_response += f"\n\n💭 Thoughts: {response.metadata['thoughts']}"

            history.append((message, formatted_response))
            return "", history
        except Exception as e:
            error_message = f"❌ Error processing message: {str(e)}"
            history.append((message, error_message))
            return "", history

    async def cleanup(self):
        """Cleanup resources"""
        if self.chat_service:
            await self.chat_service.close_browser()

    def create_interface(self):
        """Create the chat interface"""
        with gr.Group():
            # Chat Configuration Section
            with gr.Row():
                with gr.Column(scale=1):
                    # LLM Configuration
                    with gr.Group():
                        gr.Markdown("### 🔧 Chat Configuration")
                        llm_provider = gr.Dropdown(
                            ["anthropic", "openai", "gemini", "azure_openai", "deepseek", "ollama"],
                            label="LLM Provider",
                            value="gemini"
                        )
                        llm_model = gr.Textbox(
                            label="Model Name",
                            value="gemini-2.0-flash-exp"
                        )
                        llm_temp = gr.Slider(
                            minimum=0.0,
                            maximum=2.0,
                            value=1.0,
                            label="Temperature"
                        )
                        llm_base_url = gr.Textbox(label="Base URL (optional)")
                        llm_api_key = gr.Textbox(
                            label="API Key",
                            type="password"
                        )
                        
                        init_btn = gr.Button("🚀 Initialize Chat", variant="primary")
                        init_status = gr.Textbox(
                            label="Status",
                            interactive=False
                        )

                # Chat Interface Section
                with gr.Column(scale=2):
                    gr.Markdown("### 💬 Chat")
                    chatbot = gr.Chatbot(
                        height=500,
                        show_label=False
                    )
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Enter your browser automation command",
                            placeholder="e.g., Go to google.com and search for 'OpenAI'",
                            lines=3,
                            scale=4
                        )
                        with gr.Column(scale=1):
                            submit_btn = gr.Button("📤 Send", variant="primary")
                            clear_btn = gr.Button("🗑️ Clear Chat")

            # Event handlers
            init_btn.click(
                fn=self.initialize_chat_service,
                inputs=[{
                    "provider": llm_provider,
                    "model_name": llm_model,
                    "temperature": llm_temp,
                    "base_url": llm_base_url,
                    "api_key": llm_api_key
                }],
                outputs=init_status
            )

            submit_btn.click(
                fn=self.handle_message,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )

            msg.submit(
                fn=self.handle_message,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )

            clear_btn.click(
                fn=lambda: ([], []),
                outputs=[msg, chatbot]
            )

def create_chat_ui():
    chat_ui = ChatUI()
    return chat_ui.create_interface()
