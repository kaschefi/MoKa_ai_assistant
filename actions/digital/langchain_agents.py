from datetime import datetime
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage
from langgraph.prebuilt import create_react_agent
from langchain_ollama import ChatOllama


qwen25 = ChatOllama(model="qwen2.5:1.5b", temperature=0, base_url="http://localhost:11434")


# 1. Prompt callable — MUST return a list of BaseMessage objects, not a plain string.
#    Returning a bare str breaks the ReAct message structure and causes the LLM to echo
#    the raw tool-call JSON instead of actually executing the tool.
def get_weather_prompt(state) -> list:
    current_time = datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

    return [SystemMessage(content=f"""You are Cozmo's specialized Weather Agent.
Your ONLY job is to provide accurate weather updates by using your tools.

CURRENT REAL-WORLD CONTEXT:
- Today's Date and Time: {current_time}
- Default Location: Vienna

STRICT DEFAULT RULES:
If the user's request is missing specific details, you MUST apply these defaults before using your tool:
1. City: If no city is specified, use the default location.
2. Date/Time: Use the Current Real-World Context to figure out what "today", "tomorrow", or "right now" means.

INSTRUCTIONS:
1. Extract the location.
2. ALWAYS use the `get_weather` tool to fetch the data. NEVER guess or hallucinate the weather.
3. Read the raw data returned by the tool.
4. Respond with a short, natural, conversational sentence that Cozmo can speak out loud.

Example Output: "Right now in Vienna, it's 14 degrees and partly cloudy."
""")]


# 2. Weather Tool
@tool
def get_weather(city: str) -> str:
    """Fetches the current weather for a specific city."""
    import requests
    print(f"   [Tool Executing: Fetching weather for {city}...]")
    try:
        response = requests.get(f'https://wttr.in/{city}?format=3', timeout=5)
        response.raise_for_status()
        # Sanitize: remove characters that Windows cp1252 console can't display (e.g. weather emoji)
        raw = response.text
        sanitized = raw.encode('ascii', errors='ignore').decode('ascii').strip()
        # Fallback: if sanitization stripped everything, return with replacement chars
        return sanitized if sanitized else raw.encode('cp1252', errors='replace').decode('cp1252').strip()
    except Exception as e:
        return f"Weather service unavailable: {e}"


weather_worker = create_react_agent(
    model=qwen25,
    tools=[get_weather],
    prompt=get_weather_prompt
)