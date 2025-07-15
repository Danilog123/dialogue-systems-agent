import os
import gradio as gr
import requests
import inspect
import pandas as pd
import asyncio
from datetime import datetime

from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI
from tools import (
    duckduckgo_tool,
    summarize_webpage_tool,
    weather_tool,
    date_tool,
    browse_rausgegangen_de_categories_tool,
    classify_query_tool,
)

# --- Constants ---
DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

# --- Setup Agent ---
today = datetime.now().strftime("%Y-%m-%d")
api_key = os.getenv("OPENAI_API_KEY")
llm = OpenAI(model="gpt-4o", api_key=api_key)
tools = [
    duckduckgo_tool(),
    summarize_webpage_tool(),
    weather_tool(),
    date_tool(),
    browse_rausgegangen_de_categories_tool(),
    classify_query_tool(),
]
memory = ChatMemoryBuffer.from_defaults(token_limit=40000)

agent = ReActAgent(
    tools=tools,
    llm=llm,
    system_prompt="You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: FINAL ANSWER: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."
)

ctx = Context(agent)


class BasicAgent:
    def __init__(self):
        self.agent = agent
        self.ctx = ctx
        self.memory = memory

    async def aanswer(self, message: str) -> str:
        print(f"Agent startet für Nachricht: {message[:30]}...")
        handler = self.agent.run(message, return_stream=True, ctx=self.ctx, memory=self.memory)
        async for event in handler.stream_events():
            print(f"Event empfangen: {event}")  # Oder ggf. nur z.B. `print(".", end="")`
        print("Stream Events komplett eingelesen, warte auf Ergebnis...")
        final_result = await handler
        print(f"Agent Ergebnis erhalten: {final_result}")
        return str(final_result)


# --- Async helper for Gradio ---
def run_async(coro):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    else:
        return asyncio.create_task(coro)


# --- Main Function ---
def run_and_submit_all(profile: gr.OAuthProfile | None):
    space_id = os.getenv("SPACE_ID")
    username = profile.username if profile else None
    if not username:
        return "Please Login to Hugging Face with the button.", None

    agent_instance = BasicAgent()
    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"
    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"

    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
    except Exception as e:
        return f"Error fetching questions: {e}", None

    if not questions_data:
        return "Fetched questions list is empty or invalid format.", None

    answers_payload = []
    results_log = []

    async def process_all():
        for item in questions_data:
            task_id = item.get("task_id")
            question_text = item.get("question")
            if not task_id or not question_text:
                continue
            try:
                answer = await agent_instance.aanswer(question_text)
                answers_payload.append({"task_id": task_id, "submitted_answer": answer})
                results_log.append({"Task ID": task_id, "Question": question_text, "Submitted Answer": answer})
            except Exception as e:
                results_log.append({
                    "Task ID": task_id,
                    "Question": question_text,
                    "Submitted Answer": f"AGENT ERROR: {e}"
                })

        if not answers_payload:
            return "Agent did not return any answers.", pd.DataFrame(results_log)

        submission_data = {
            "username": username.strip(),
            "agent_code": agent_code,
            "answers": answers_payload
        }

        try:
            response = requests.post(submit_url, json=submission_data, timeout=60)
            response.raise_for_status()
            result_data = response.json()
            final_status = (
                f"Submission Successful!\n"
                f"User: {result_data.get('username')}\n"
                f"Overall Score: {result_data.get('score', 'N/A')}% "
                f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
                f"Message: {result_data.get('message', 'No message received.')}"
            )
            return final_status, pd.DataFrame(results_log)
        except Exception as e:
            return f"Submission failed: {e}", pd.DataFrame(results_log)

    return run_async(process_all())


# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown("**Instructions:** ...")
    gr.LoginButton()
    run_button = gr.Button("Run Evaluation & Submit All Answers")
    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(fn=run_and_submit_all, outputs=[status_output, results_table])

if __name__ == "__main__":
    print("-" * 30 + " App Starting " + "-" * 30)
    space_host = os.getenv("SPACE_HOST")
    space_id = os.getenv("SPACE_ID")
    if space_host:
        print(f"✅ SPACE_HOST found: {space_host}")
        print(f"   Runtime URL should be: https://{space_host}.hf.space")
    if space_id:
        print(f"✅ SPACE_ID found: {space_id}")
        print(f"   Repo URL: https://huggingface.co/spaces/{space_id}")
        print(f"   Tree URL: https://huggingface.co/spaces/{space_id}/tree/main")
    print("-" * 60)
    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)
