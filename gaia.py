from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.agent.workflow import ReActAgent, AgentWorkflow, AgentStream, ToolCallResult
from llama_index.core import PromptTemplate
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI
from tools import search_tool, duckduckgo_tool, weather_tool, date_tool, summarize_webpage_tool, browse_rausgegangen_de_categories_tool, classify_query_tool, load_facts, store_fact_tool, create_ics_tool, more_information_tool
from dotenv import load_dotenv
import gradio as gr
import os
from datetime import datetime
import json
import os
today = datetime.now().strftime("%Y-%m-%d")

def append_string_to_protocol(string: str):
  with open('gaia_protocol.txt', 'a') as f:
    f.write(string)
    
def append_message_to_protocol(file_path: str, new_input: str, new_output: str):
  # Step 1: Load existing messages or initialize empty list
  if os.path.exists(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
      try:
        messages = json.load(f)
      except json.JSONDecodeError:
        messages = []
  else:
    messages = []

  # Step 2: Append new message
  new_message = {"input": new_input, "output": new_output}
  messages.append(new_message)

  # Step 3: Save updated list back to file
  with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(messages, f, indent=2, ensure_ascii=False)

load_dotenv()

llm = OpenAI(model="gpt-4o")

# Import tools
tools = [duckduckgo_tool(), summarize_webpage_tool(), weather_tool(), date_tool(), browse_rausgegangen_de_categories_tool(), classify_query_tool(), store_fact_tool(), create_ics_tool(), more_information_tool()]

#Init Memory
memory = ChatMemoryBuffer.from_defaults(token_limit=40000)

# System_prompt
gaia_system_prompt = "You are a general AI assistant. I will ask you a question. Report your thoughts, and finish your answer with the following template: Answer: [YOUR FINAL ANSWER]. YOUR FINAL ANSWER should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string."

react_header_prompt = """
## Tools

You have access to a wide variety of tools. You are responsible for using the tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools to complete each subtask.

You have access to the following tools:
{tool_desc}


## Output Format

Please answer in the same language as the question and use the following format:

```
Thought: The current language of the user is: (user's language). I need to use a tool to help me answer the question.
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
```

Please ALWAYS start with a Thought.

NEVER surround your response with markdown code markers. You may use code markers within your response if you need to.

Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}. If you include the "Action:" line, then you MUST include the "Action Input:" line too, even if the tool does not need kwargs, in that case you MUST use "Action Input: {{}}".

If this format is used, the tool will respond in the following format:

```
Observation: tool response
```

You should keep repeating the above format till you have enough information to answer the question without using any more tools. At that point, you MUST respond in one of the following two formats:

```
Thought: I can answer without using any more tools. I'll use the user's language to answer
Answer: [your answer here (In the same language as the user's question)]
```

```
Thought: I cannot answer the question with the provided tools.
Answer: [your answer here (In the same language as the user's question)]

## Current Conversation

Below is the current conversation consisting of interleaving human and assistant messages.

```
"""


# Define Agent
agent = ReActAgent(
    tools=tools,
    llm=llm
)

react_system_prompt = PromptTemplate(gaia_system_prompt + react_header_prompt)

agent.update_prompts({'react_header': react_system_prompt})

ctx = Context(agent)

async def run_agent(message):
  handler = agent.run(message, return_stream=True, ctx=ctx, memory=memory)
  toughts, tool_calls, final, protocol = "", "", "", ""

  async for ev in handler.stream_events():
    if isinstance(ev, ToolCallResult):
      tool_calls += f"🔧 {ev.tool_name}({ev.tool_kwargs}) => {ev.tool_output}\n\n"
      protocol += f"🔧 {ev.tool_name}({ev.tool_kwargs}) => {ev.tool_output}\n\n"
    elif isinstance(ev, AgentStream):
      toughts += ev.delta
      protocol += ev.delta

  final_result = await handler
  final += str(final_result)
  return toughts.strip(), tool_calls.strip(), final.strip(), protocol.strip()


with gr.Blocks(fill_height=True) as gradio_ui:
  gr.Markdown("# Gaia Benchmark Agent")
  with gr.Row():
    chatbot = gr.Chatbot(type="messages", show_copy_button=True)
    with gr.Column(visible=False) as right_column:
      thoughts_box = gr.Textbox(label="🧠 Agent Thoughts", lines=8)
      tools_box = gr.Textbox(label="🔧 Tool Calls", lines=8)
      file_download = gr.File(label="📅 ICS-file", visible=False)

  msg = gr.Textbox(label="Your message")
  send_btn = gr.Button("Send")
  toggle_btn = gr.Button("Toggle Debug View")


  async def respond(user_input, chat_history):
    thoughts, tools, final, protocol = await run_agent(user_input)
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": final})
    download_path = None
    file_output = gr.update(value=None, visible=False)
    if ".ics" in tools:
      for line in tools.splitlines():
        if ".ics" in line:
          potential = line.split("=>")[-1].strip()
          if os.path.exists(potential):
            file_output = gr.update(value=potential, visible=True)
            break
    
    append_message_to_protocol("gaia_protocol.json", user_input, protocol)

    return chat_history, thoughts, tools, "", file_output


  send_btn.click(fn=respond, inputs=[msg, chatbot], outputs=[chatbot, thoughts_box, tools_box, msg, file_download])
  msg.submit(fn=respond, inputs=[msg, chatbot], outputs=[chatbot, thoughts_box, tools_box, msg, file_download])

  # Keep track of visibility state
  show_debug = gr.State(value=True)

  def toggle_debug_view(show):
    return gr.update(visible=not show), not show

  toggle_btn.click(fn=toggle_debug_view, inputs=[show_debug], outputs=[right_column, show_debug])

if __name__ == "__main__":
  gradio_ui.launch(inbrowser=True)