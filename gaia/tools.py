from llama_index.core.tools import FunctionTool
from llama_index.core.schema import Document
from playwright.sync_api import sync_playwright
from ddgs import DDGS
import requests
from datetime import datetime
import json
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec

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
    return tool
# Search tool
def duckduckgo_search(query: str, max_results: int = 5) -> list[Document]:
    """
    Search the Web with DuckDuckGo
    """
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
    return FunctionTool.from_defaults(
        fn=duckduckgo_search,
        name="duckduckgo_websearch",
        description="Suche nach aktuellen Webinhalten über DuckDuckGo.",
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
    # Use wttr.in, simply web page for the weather forecast of next 3 days
    try:
        response = requests.get(f"https://wttr.in/{city}", timeout=10)
        response.raise_for_status()
        return response.text.strip()
    except Exception as e:
        return f"An error occurred: {e}"

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
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)
        page = browser.new_page()
        page.goto(url, timeout=60000)
        page.wait_for_timeout(5000)  # wait for lazy content to load
        text = page.inner_text("body")
        browser.close()
        return text

def summarize_webpage_tool():
    return FunctionTool.from_defaults(
        fn=summarize_webpage,
        name="ExtractAndReadWebPage",
        description=(
            "Use this tool to extract and summarize the full content of a webpage. "
            "Provide a URL, and it will return the full text content from the page's body."
        )
    )
def get_category_examples() -> str:
    with open("example_categories.json", "r") as f:
        category_examples = json.load(f)
    return category_examples

def classify_query_tool():
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
    return url

def browse_rausgegangen_de_categories_tool():
    return FunctionTool.from_defaults(
        fn=browse_rausgegangen_de_categories,
        name="BrowseRausgegangenDeCategories",
        description=(
            "Return the url link of the website."
            "The input parameter are: city name in english in small letters, and one of the given categories."
            "Use this tool only for german cities!"
        )
    )
