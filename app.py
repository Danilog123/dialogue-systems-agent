import asyncio
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.agent.workflow import AgentWorkflow, AgentStream, ToolCallResult
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI
from tools import search_tool, duckduckgo_tool, weather_tool, date_tool, summarize_webpage_tool, browse_rausgegangen_de_categories_tool, classify_query_tool
from dotenv import load_dotenv
import gradio as gr

load_dotenv()

llm = OpenAI(model="gpt-4o-mini")

# Import tools
tools = [duckduckgo_tool(), summarize_webpage_tool(), weather_tool(), date_tool(), browse_rausgegangen_de_categories_tool(), classify_query_tool()]

#Init Memory
memory = ChatMemoryBuffer.from_defaults(token_limit=40000)

# Define Agent
agent = AgentWorkflow.from_tools_or_functions(
    tools_or_functions=tools,
    llm=llm,
    system_prompt=(
        "You are a helpful assistant that must use tools to answer questions whenever possible. "
        "If you call a tool and receive information, always base your final answer on that result. "
        "Use the date_tool to get the current date and time. "
        "If you call duckduckgo_websearch, extract and reason over the URLs' content before answering. "
        "If the user asks for events in a german city you can (but do not mandatory) use the browse_rausgegangen_de_categories tool. For that classify the request into one of the following categories: party, konzerte-und-musik, markt, theater, shows-und-performances, ausstellung, gesprochenes, food-und-drinks, aktiv-und-kreativ, feste-und-festival, sport, film, or kinder-und-familien. "
        "You can use the classify_query_tool() to get examples for each category. "
        "Then call the BrowseRausgegangenDeCategories tool to get the appropriate category page. After that, use the SummarizeWebpage tool to read and summarize the page content for the user and return the url.If you don't find a promising event, try another possible category. After two categories try , use Websearch."
        "Then, pass the most relevant URL to the WebPageQA tool to extract the content. "
        "Use the weather tool to get the current weather for a given city. Take weather into account while reasoning"
        "Only return an answer after reading and understanding the page."
        "Give precise information of the event, including time and data, price, exact location and a link you used for the suggested event. Always make sure the event is at the desired date. "
    )
)

ctx = Context(agent)

async def run_agent(message):
  handler = agent.run(message, return_stream=True, ctx=ctx, memory=memory)
  toughts, tool_calls, final = "", "", ""
  
  async for ev in handler.stream_events():
    if isinstance(ev, ToolCallResult):
      tool_calls += f"🔧 {ev.tool_name}({ev.tool_kwargs}) => {ev.tool_output}\n\n"
    elif isinstance(ev, AgentStream):
      toughts += ev.delta
  
  final_result = await handler
  final += str(final_result)
  return toughts.strip(), tool_calls.strip(), final.strip()

with gr.Blocks() as gradio_ui:
  gr.Markdown("# 🧠 LlamaIndex Tool-Using Chatbot")

  with gr.Row():
    chatbot = gr.Chatbot(type="messages")
    with gr.Column():
      thoughts_box = gr.Textbox(label="🧠 Agent Thoughts", lines=8)
      tools_box = gr.Textbox(label="🔧 Tool Calls", lines=8)

  msg = gr.Textbox(label="Your message")
  send_btn = gr.Button("Send")

  async def respond(user_input, chat_history):
    thoughts, tools, final = await run_agent(user_input)
    chat_history.append({"role": "user", "content": user_input})
    chat_history.append({"role": "assistant", "content": final})
    return chat_history, thoughts, tools

  send_btn.click(fn=respond, inputs=[msg, chatbot], outputs=[chatbot, thoughts_box, tools_box])

if __name__ == "__main__":
  gradio_ui.launch()
