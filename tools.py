from typing import Optional
from llama_index.core.tools import FunctionTool
from llama_index.core.schema import Document
from playwright.sync_api import sync_playwright
from ddgs import DDGS
import requests
from datetime import datetime
import json
import os
from ics import Calendar, Event
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec

FACTS_FILE = 'facts.json'

def search_tool():
    '''
    Use DuckDuckGoSearchTool for web search.
    '''
    tool_spec = DuckDuckGoSearchToolSpec()
    return FunctionTool.from_defaults(
        fn=tool_spec.duckduckgo_full_search,
        name="WebSearch",
        description="Search for relevant web pages based on a query. Returns a list of search results with title, body and URL."
    )

# Search tool
def duckduckgo_search(query: str, max_results: int = 5) -> list[Document]:
    documents = []
    with DDGS() as ddgs:
        results = ddgs.text(query, max_results=max_results)
        for result in results:
            content = result.get("body", "")
            title = result.get("title", "")
            url = result.get("href", "")
            metadata = {"title": title, "url": url}
            documents.append(Document(text=content, metadata=metadata))
    return documents

def duckduckgo_tool():
    """
    Search the Web with DuckDuckGo
    """
    return FunctionTool.from_defaults(
        fn=duckduckgo_search,
        name="duckduckgo_websearch",
        description="Use this to answer factual questions about public figures, dates, countries, laws, or historical facts. Do not guess. Return a short fact and source URL."
                    "Search for relevant web pages based on a query. Returns a list of search results with title, body and URL.",
    )
# Date
def get_date():
    now = datetime.now()
    return now.strftime("%Y-%m-%d %H:%M:%S")

def date_tool():
    '''
    Get current date and time.
    '''
    return FunctionTool.from_defaults(
        fn=get_date,
        name="GetDateandTime",
        description="Get current date and time for an answer in YYYY-MM-DD H:M:S format."
    )
# Weather Tool
def get_weather(city: str) -> str:
    # Use wttr.in, simple web page for the weather forecast of next 3 days
    try:
        response = requests.get(f"https://wttr.in/{city}", timeout=10)
        response.raise_for_status()
        return "Observation: " + response.text.strip() + "\n"
    except Exception as e:
        return "Observation: " + f"An error occurred: {e}" + "\n"

def weather_tool():
    '''
    Get the weather forcast for the next 3 days for a city.
    '''
    return FunctionTool.from_defaults(
        fn=get_weather,
        name="GetWeather",
        description="Use this tool for outdoor activities to get the weather forcast for the next 3 days for a given city. "
                    "Input is a city name string."
    )

def summarize_webpage(url: str) -> str:
    """
    Loads a webpage using Playwright and returns its inner text content.
    """
    with sync_playwright() as playwright:
        chromium = playwright.chromium
        browser = chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(url)
        text = page.inner_text("body")
        browser.close()
        return "Observation: " + text + "\n"

def summarize_webpage_tool():
    '''
    Extract the content of a webpage using Playwright and returns its inner text content.
    '''
    return FunctionTool.from_defaults(
        fn=summarize_webpage,
        name="ExtractAndReadWebPage",
        description=(
            "Use this tool to extract and summarize the full content of a webpage. "
            "Provide a URL, and it will return the full text content from the page's body."
        )
    )

def get_category_examples() -> str:
    # Read json-file with examples of events taken from rausgegangen.de
    with open("example_categories.json", "r") as f:
        category_examples = json.load(f)
    return "Observation: " + str(category_examples) + "\n"


def classify_query_tool():
    '''
    Classify an event in one of the rausgegangen.de categories.
    '''
    return FunctionTool.from_defaults(
        fn=get_category_examples,
        name="ClassifyQuery",
        description="Use this tool to get a dictionary with examples for each category"
                    "Classify the users query as one of the following categories: party, konzerte-und-musik, markt, theater, shows-und-performances, ausstellung, gesprochenes, food-und-drinks, aktiv-und-kreativ, feste-und-festival, sport, film or kinder-und-familien."
        ,
    )

def browse_rausgegangen_de_categories(city:str, category: str,) -> str:
    url= f"https://rausgegangen.de/{city}/kategorie/{category}"
    print(url)
    return "Observation: " + url + "\n"

def browse_rausgegangen_de_categories_tool():
    '''
    Browse the rausgegangen.de, the urls have a defined structure.
    '''
    return FunctionTool.from_defaults(
        fn=browse_rausgegangen_de_categories,
        name="BrowseRausgegangenDeCategories",
        description=(
            "Return the url link of the website."
            "The input parameter are: city name in english in small letters, and one of the given categories."
            "Use this tool only for german cities!"
        )
    )

def more_information_rausgegangen_event(url:str, event_name:str) -> str:
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(headless=False)
        context = browser.new_context()
        # create a new page inside context.
        page = context.new_page()
        page.goto(url)
        page.get_by_role("link", name=event_name).click()
        context.close()
        return "Observation: " + page.url + "\n"

def more_information_tool():
    return FunctionTool.from_defaults(
        fn=more_information_rausgegangen_event,
        name="Extract_Event_URL",
        description=("Use this tool to get an url about one event. As input it gets the url of the category website and the name of the event."
                     "It only returns the link of the website.")
    )

def load_facts():
    if not os.path.exists(FACTS_FILE):
        return []
    with open(FACTS_FILE, "r") as f:
        return json.load(f)


def store_fact(new_fact: str) -> str:
    facts = load_facts()
    if new_fact in facts:
        return "Observation: " + f"Fact '{new_fact}' was already stored." + "\n"
    else:
        facts.append(new_fact)
        with open(FACTS_FILE, "w") as f:
            json.dump(facts, f)
        return "Observation: " + f"Fact stored: {new_fact}" + "\n"


def store_fact_tool():
    '''
    Store facts about the user in a json file.
    '''
    return FunctionTool.from_defaults(
        fn=store_fact,
        name="StoreFact",
        description="""
      Use this tool to store a fact about the user.
      The fact is supplied as a string and stored for future use.
      This tool returns the fact that was stored.
    """
    )

#CalendarTool
# Create a .ics file of the event to export it to a calendar
def create_ics_event(name:str, date:str, time:str, location:Optional[str] = None, url:Optional[str] = None) -> str:
    c = Calendar()
    e = Event()
    e.name = name
    e.begin = f"{date} {time}"
    e.location = location if location else ""
    e.url = url if url else ""
    c.events.add(e)
    filename = f"{name}.ics"
    path = os.path.join("calendar", filename)
    # Create a folder, if necessary
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.writelines(c.serialize_iter())
    return "Observation: " + f"{path}" + "\n"

def create_ics_tool():
    '''
    Create a calendar entry of an event. Name, date and time are mandatory. location, url are optional.
    '''
    return FunctionTool.from_defaults(
        fn=create_ics_event,
        name="CreateICSEvent",
        description="Create an .ics file. with a calendar entry. "
                    "It takes event name, date and starting time of the event as input. The location of the event and the url of the event are optional inputs. "
                    "It returns the path of the file."
    )


