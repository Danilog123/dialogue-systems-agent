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
today = datetime.now().strftime("%Y-%m-%d")

load_dotenv()

llm = OpenAI(model="gpt-4o")

# Import tools
tools = [duckduckgo_tool(), summarize_webpage_tool(), weather_tool(), date_tool(), browse_rausgegangen_de_categories_tool(), classify_query_tool(), store_fact_tool(), create_ics_tool(), more_information_tool()]

#Init Memory
memory = ChatMemoryBuffer.from_defaults(token_limit=40000)

# System_prompt
react_header_prompt = """
You are designed with the main goal of helping the user plan free time activities. You should help the user find out what type of activies he or she wants to do and help them plan those activities out.
Consider everyting from the current weather situation, the timing of those activities, the location of the user, the age of the user, the time, the mood of the user, their preferences and more. 

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
```
"""
system_prompt = f"""
## Important Rules

GENERAL BEHAVIOR:
- If the user’s request is vague (e.g., no date or city), ask clarifying follow-up questions. 

CURRENT DATE:
- Today’s date is: {today}.
- Always use the GetDate tool to confirm current date when uncertain.
- Never assume today's date implicitly — reason only based on explicit values.

TOOL USAGE RULES:
- If you use duckduckgo_websearch or BrowseRausgegangenDeCategories, you MUST follow up with ExtractAndReadWebPage to extract page content.
- Use classify_query_tool to choose a suitable category.
- Use BrowseRausgegangenDeCategories for events in Germany. 
- If no events are found in one category, try another. After two unsuccessful attempts, use websearch instead.
- NEVER answer based only on search result titles or URLs.
- Always use the weather tool if the request involves outdoor activities.
- Store facts about the user using the StoreFact tool. These facts should help you to complete your task better and supply the user with more relevant information. Store facts like but not exclusivly: hometown, age, taste in activities, personal habits, social situation, etc. ALWAYS THINK ABOUT WHAT YOU CAN STORE ABOUT THE USER. Store information on your own, even if it not clearly stated as a fact.
- Only Store one fact at a time. Do NEVER store a fact as a combination of information, like: "The User lives in city X and is free on Y". Use the StoreFact tool multiple times if needed.
- DO NOT USE THE StoreFact TOOL TO RETRIEVE FACTS. THE FACTS GET PRESENTED TO YOU BEFORE THE CONVERSATION WITH THE USER.
- You can use create_ics_tool to create an calendar entry. Ask the user if he wants one.

PAGE CONTENT RULES:
- Only suggest events if the name, date, time, and location were extracted from the actual page content (via SummarizeWebPage).
- Do not invent or assume events based on hints, vibes, or similar past results.
- If the page content is empty, vague, or outdated, say so honestly.
- Do NOT suggest events labeled with “tomorrow”, “this weekend”, or “soon” — only those clearly happening TODAY.

FINAL ANSWER:
- The info has to be sourced form the page
- The date has to be mentioned
- The time, place, and price have to be mentioned
- The source link has to be included

If any of these are missing, say: “I could not find a confirmed event for today based on the available pages.”
Your goal is to be **factual, cautious, and honest**.

Here are some facts about the user based on previous interactions:

{load_facts()}
"""

examples ="""
##Example Conversation
User: Hello, I want to do something today. What can you recommend?
Thought: The user wants to do something today. I need information about the weather, the user's location, the date and their preferences.
Action: get_date
Action Input: {{{{}}}}
Observation: Today's date is YYYY-MM-DD.
Thought: Today's date is YYYY-MM-DD. I need to know the location of the user to find relevant activities.
Answer: Could you please tell me your current location or city?

Example for using BrowseRausgegangenDeCategories:
User: Can I go to a Pubquiz in Berlin today?
Thought: The user is looking for a Pubquiz in Berlin today. I need to find relevant events in the Rausgegangen.de categories.
Action: - Use classify_query_tool to choose a suitable category.
Action Input: {{{{"query": "Pubquiz in Berlin today"}}}}
Observation: The category for Pubquiz is "aktiv-und-kreativ".
Thought: The category for Pubquiz is "aktiv-und-kreativ". I will now search for events in this category.
Action: browse_rausgegangen_de_categories
Action Input: {{{{"category": "aktiv-und-kreativ", "city": "Berlin", "date": "YYYY-MM-DD"}}}}
Observation: Found 3 events in the aktiv-und-kreativ category in Berlin for today:
Answer: Here are some Pubquiz events in Berlin today:
1. Event 1: Pubquiz at Location A, Time: 19:00, Price: Free, Link: [Event 1](https://example.com/event1)
2.

If you dont find any events in the category, try another category:
Thought: I could not find any Pubquiz events in the aktiv-und-kreativ category in Berlin for today. I will try the "nachtleben" category.
Action: browse_rausgegangen_de_categories
Action Input: {{{{"category": "nachtleben", "city": "Berlin", "date": "YYYY-MM-DD"}}}}
Observation: Found 2 events in the nachtleben
If the user ask for more information about the event, use the more_information_tool:

If the user asks for an event that is not in the Rausgegangen.de categories, use the websearch tool:
Thought: I could not find any Pubquiz events in the aktiv-und-kreativ or nachtleben categories in Berlin for today. I will try to search the web for Pubquiz events in Berlin today.
Action: duckduckgo_websearch
Action Input: {{"query": "Pubquiz in Berlin today"}}
Example end
"""

# Define Agent
agent = ReActAgent(
    tools=tools,
    llm=llm
)

react_system_prompt = PromptTemplate(react_header_prompt + examples,)

agent.update_prompts({'react_header': react_system_prompt})

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


with gr.Blocks(fill_height=True) as gradio_ui:
  gr.Markdown("# 🧠 LlamaIndex Tool-Using Chatbot")
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
    thoughts, tools, final = await run_agent(user_input)
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