#from llama_index.core.tools import FunctionTool
#from llama_index.core.tools.ondemand_loader_tool import OnDemandLoaderTool
from llama_index.tools.duckduckgo import DuckDuckGoSearchToolSpec
import requests
from datetime import datetime

# Search tool
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
        description="Use this tool to get the weather forcast for the next 3 days for a given city. Input is a city name string."
    )

from llama_index.core.tools import FunctionTool
from playwright.sync_api import sync_playwright

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
        name="SummarizeWebPage",
        description=(
            "Use this tool to extract and summarize the full content of a webpage. "
            "Provide a URL, and it will return the full text content from the page's body."
        )
    )
#NEW Tool
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
            "Classify the users query as one of the following categories: party, konzerte-und-musik, markt, theater, shows-und-performances, ausstellung, gesprochenes, food-und-drinks, aktiv-und-kreativ, feste-und-festival, sport, film or kinder-und-familien."
            "The input parameter are: city name in english in small letters, and one of the given categories."
        )
    )
