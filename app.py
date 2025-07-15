from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.agent.workflow import ReActAgent, AgentWorkflow, AgentStream, ToolCallResult
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI
from tools import search_tool, duckduckgo_tool, weather_tool, date_tool, summarize_webpage_tool, browse_rausgegangen_de_categories_tool, classify_query_tool
from dotenv import load_dotenv
import gradio as gr
from datetime import datetime
today = datetime.now().strftime("%Y-%m-%d")

load_dotenv()

llm = OpenAI(model="gpt-4o-mini")

# Import tools
tools = [duckduckgo_tool(), summarize_webpage_tool(), weather_tool(), date_tool(), browse_rausgegangen_de_categories_tool(), classify_query_tool()]

#Init Memory
memory = ChatMemoryBuffer.from_defaults(token_limit=40000)

system_prompt = """You are a helpful assistant that supports users in finding real-world events using tools.
        Always answer in the query language.

        GENERAL BEHAVIOR:
        - Always use tools to answer questions whenever possible.
        - If the user’s request is vague (e.g., no date or city), ask clarifying follow-up questions. 
        - Prefix clarifying questions with "Follow-up:" and ask **only one thing at a time**.

        CURRENT DATE:
        - Today’s date is: {today}.
        - Always use the GetDate tool to confirm current date when uncertain.
        - Never assume today's date implicitly — reason only based on explicit values.

        TOOL USAGE RULES:
        - If you use duckduckgo_websearch or BrowseRausgegangenDeCategories, you MUST follow up with SummarizeWebPage to extract page content.
        - Use BrowseRausgegangenDeCategories only for events in Germany. If you do not find an event with BrowseRausgegangenDeCategories then try on web search.
        - NEVER answer based only on search result titles or URLs.
        - Use classify_query_tool to choose a suitable category.
        - If no events are found in one category, try another. After two unsuccessful attempts, use websearch instead.
        - Always use the weather tool if the request involves outdoor activities.

        PAGE CONTENT RULES:
        - Only suggest events if the name, date, time, and location were extracted from the actual page content (via SummarizeWebPage).
        - Do not invent or assume events based on hints, vibes, or similar past results.
        - If the page content is empty, vague, or outdated, say so honestly.
        - Do NOT suggest events labeled with “tomorrow”, “this weekend”, or “soon” — only those clearly happening TODAY.

        FINAL ANSWER CHECKLIST:
        - Did you confirm the event info from the page itself?
        - Is the date explicitly today?
        - Are the time, place, and price mentioned?
        - Did you include the source link?

        If any of these are missing, say: “I could not find a confirmed event for today based on the available pages.”

        Your goal is to be **factual, cautious, and honest**. It's better to admit uncertainty than to make something up.
        """

# Define Agent
agent = AgentWorkflow.from_tools_or_functions(
    tools_or_functions=tools,
    llm=llm,
    system_prompt=system_prompt
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
  gradio_ui.launch(inbrowser=True)
