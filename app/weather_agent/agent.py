"""Weather Agent definition for ADK Bidi-streaming demo."""

from google.adk.agents import Agent
from google.adk.tools import ToolContext, FunctionTool
from ..novasonic_llm import NovaSonic
from typing import Dict
from google.adk.models.lite_llm import LiteLlm

# Replacing the model with Nova Sonic model
# TBD need to add other params for model authentication
model = NovaSonic(model="amazon.nova-2-sonic-v1:0")

# 1. Define the tool logic
def get_weather(tool_context: ToolContext, city: str) -> Dict:
    """Retrieves the current weather information for a specific city."""
    print(f"--- Tool: get_weather called for city: {city} ---")
    if city.lower() == "bay area":
        return {"city": city, "temperature": "20°C", "conditions": "Sunny"}
    if city.lower() == "nyc":
        return {"city": city, "temperature": "-20°C", "conditions": "Freezing Snow"}
    return {"error": "Data not available"}

# 2. subagents
generic_agent = Agent(
    name="generic_agent",
    description="Use this for greetings, general small talk, or questions unrelated to weather/travel.",
    model=model,
    instruction="You are a polite assistant. Keep responses brief and friendly."
)

packer_agent = Agent(
    name="packer_agent",
    description="Use this when the user asks for packing advice, clothing suggestions, or travel prep.",
    model=model,
    instruction="""Suggest 3 specific items to pack based on the weather data in the chat history. 
    If no weather data exists yet, tell the user you need the weather for their city first."""
)

# 3. root agent
agent = Agent(
    name="weather_concierge",
    model=model,
    tools=[FunctionTool(func=get_weather)],
    sub_agents=[generic_agent, packer_agent],
    instruction="""You are the lead Weather Concierge. Follow these steps:
    1. If the user asks for weather, IMMEDIATELY call 'get_weather' and summarize the result.
    2. If the user asks what to wear or pack, first call 'get_weather' if you don't have data yet, 
       then DELEGATE to 'packer_agent' to provide the advice.
    3. For general chat, delegate to 'generic_agent'.
    Always use your tools before delegating if the sub-agent requires weather context."""
)


