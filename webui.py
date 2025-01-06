import argparse
import gradio as gr
from gradio.themes import Base, Default, Soft, Monochrome, Glass, Origin, Citrus, Ocean
import os, glob
from src.chat.chat_ui import create_chat_ui

# Define the theme map globally
theme_map = {
    "Default": Default(),
    "Soft": Soft(),
    "Monochrome": Monochrome(),
    "Glass": Glass(),
    "Origin": Origin(),
    "Citrus": Citrus(),
    "Ocean": Ocean()
}

def create_ui(theme_name="Ocean"):
    css = """
    .gradio-container {
        max-width: 1200px !important;
        margin: auto !important;
        padding-top: 20px !important;
    }
    .header-text {
        text-align: center;
        margin-bottom: 30px;
    }
    .theme-section {
        margin-bottom: 20px;
        padding: 15px;
        border-radius: 10px;
    }
    """

    js = """
    function refresh() {
        const url = new URL(window.location);
        if (url.searchParams.get('__theme') !== 'dark') {
            url.searchParams.set('__theme', 'dark');
            window.location.href = url.href;
        }
    }
    """
    
    with gr.Blocks(title="Browser Use WebUI", theme=theme_map[theme_name], css=css, js=js) as demo:
        with gr.Row():
            gr.Markdown(
                """
                # 🌐 Browser Use WebUI
                ### Control your browser with AI assistance
                """,
                elem_classes=["header-text"]
            )
        
        with gr.Tabs() as tabs:
            # Add Chat Interface as the first tab
            with gr.TabItem("💬 Chat Interface", id=0):
                chat_interface = create_chat_ui()

            # Original tabs remain unchanged
            with gr.TabItem("🤖 Agent Settings", id=1):
                with gr.Group():
                    agent_type = gr.Radio(
                        ["org", "custom"],
                        label="Agent Type",
                        value="custom",
                        info="Select the type of agent to use"
                    )
                    max_steps = gr.Slider(
                        minimum=1,
                        maximum=200,
                        value=100,
                        step=1,
                        label="Max Run Steps",
                        info="Maximum number of steps the agent will take"
                    )
                    use_vision = gr.Checkbox(
                        label="Use Vision",
                        value=True,
                        info="Enable visual processing capabilities"
                    )

            with gr.TabItem("🔧 LLM Configuration", id=2):
                with gr.Group():
                    llm_provider = gr.Dropdown(
                        ["anthropic", "openai", "gemini", "azure_openai", "deepseek", "ollama"],
                        label="LLM Provider",
                        value="gemini",
                        info="Select your preferred language model provider"
                    )
                    llm_model_name = gr.Textbox(
                        label="Model Name",
                        value="gemini-2.0-flash-exp",
                        info="Specify the model to use"
                    )
                    llm_temperature = gr.Slider(
                        minimum=0.0,
                        maximum=2.0,
                        value=1.0,
                        step=0.1,
                        label="Temperature",
                        info="Controls randomness in model outputs"
                    )
                    with gr.Row():
                        llm_base_url = gr.Textbox(
                            label="Base URL",
                            info="API endpoint URL (if required)"
                        )
                        llm_api_key = gr.Textbox(
                            label="API Key",
                            type="password",
                            info="Your API key"
                        )

            with gr.TabItem("🌐 Browser Settings", id=3):
                with gr.Group():
                    with gr.Row():
                        use_own_browser = gr.Checkbox(
                            label="Use Own Browser",
                            value=False,
                            info="Use your existing browser instance"
                        )
                        headless = gr.Checkbox(
                            label="Headless Mode",
                            value=False,
                            info="Run browser without GUI"
                        )
                        disable_security = gr.Checkbox(
                            label="Disable Security",
                            value=True,
                            info="Disable browser security features"
                        )
                    
                    with gr.Row():
                        window_w = gr.Number(
                            label="Window Width",
                            value=1920,
                            info="Browser window width"
                        )
                        window_h = gr.Number(
                            label="Window Height",
                            value=1080,
                            info="Browser window height"
                        )
                    
                    save_recording_path = gr.Textbox(
                        label="Recording Path",
                        placeholder="e.g. ./tmp/record_videos",
                        value="./tmp/record_videos",
                        info="Path to save browser recordings"
                    )

            with gr.TabItem("📝 Task Settings", id=4):
                task = gr.Textbox(
                    label="Task Description",
                    lines=4,
                    placeholder="Enter your task here...",
                    value="go to google.com and type 'OpenAI' click search and give me the first url",
                    info="Describe what you want the agent to do"
                )
                add_infos = gr.Textbox(
                    label="Additional Information",
                    lines=3,
                    placeholder="Add any helpful context or instructions...",
                    info="Optional hints to help the LLM complete the task"
                )

                with gr.Row():
                    run_button = gr.Button("▶️ Run Agent", variant="primary", scale=2)
                    stop_button = gr.Button("⏹️ Stop", variant="stop", scale=1)

            with gr.TabItem("🎬 Recordings", id=5):
                recording_display = gr.Video(label="Latest Recording")

                with gr.Group():
                    gr.Markdown("### Results")
                    with gr.Row():
                        with gr.Column():
                            final_result_output = gr.Textbox(
                                label="Final Result",
                                lines=3,
                                show_label=True
                            )
                        with gr.Column():
                            errors_output = gr.Textbox(
                                label="Errors",
                                lines=3,
                                show_label=True
                            )
                    with gr.Row():
                        with gr.Column():
                            model_actions_output = gr.Textbox(
                                label="Model Actions",
                                lines=3,
                                show_label=True
                            )
                        with gr.Column():
                            model_thoughts_output = gr.Textbox(
                                label="Model Thoughts",
                                lines=3,
                                show_label=True
                            )

        # Run button click handler
        run_button.click(
            fn=run_browser_agent,
            inputs=[
                agent_type, llm_provider, llm_model_name, llm_temperature,
                llm_base_url, llm_api_key, use_own_browser, headless,
                disable_security, window_w, window_h, save_recording_path,
                task, add_infos, max_steps, use_vision
            ],
            outputs=[final_result_output, errors_output, model_actions_output, model_thoughts_output, recording_display]
        )

    return demo

def main():
    parser = argparse.ArgumentParser(description="Gradio UI for Browser Agent")
    parser.add_argument("--ip", type=str, default="127.0.0.1", help="IP address to bind to")
    parser.add_argument("--port", type=int, default=7788, help="Port to listen on")
    parser.add_argument("--theme", type=str, default="Ocean", choices=theme_map.keys(), help="Theme to use for the UI")
    parser.add_argument("--dark-mode", action="store_true", help="Enable dark mode")
    args = parser.parse_args()

    demo = create_ui(theme_name=args.theme)
    demo.launch(server_name=args.ip, server_port=args.port)

if __name__ == '__main__':
    main()
