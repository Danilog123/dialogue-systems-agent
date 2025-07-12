import asyncio
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.agent.workflow import AgentWorkflow, AgentStream, ToolCallResult
from llama_index.core.workflow import Context
from llama_index.llms.openai import OpenAI
from tools import search_tool, duckduckgo_tool, weather_tool, date_tool, summarize_webpage_tool, browse_rausgegangen_de_categories_tool, classify_query_tool

#Use the GPT-4o mini as llm
import os
#add OpenAIKey
os.environ["OPENAI_API_KEY"] = ""
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

# Run chat
async def chat():

    while True:
        user_input = input("User: ")
        if user_input == "exit":
            break
        try:
            handler = agent.run(user_input, return_stream=True, ctx=ctx, memory=memory)
            # Show Tool calls
            async for ev in handler.stream_events():
                if isinstance(ev, ToolCallResult):
                    print("")
                    print("Called tool: ", ev.tool_name, ev.tool_kwargs, "=>", ev.tool_output)
                elif isinstance(ev, AgentStream):  # showing the thought process
                    print(ev.delta, end="", flush=True)

            final = await handler
            print("\n\nAgent final RESPONSE:\n", final)

        except Exception as e:
            print(e)

if __name__ == "__main__":
    asyncio.run(chat())

