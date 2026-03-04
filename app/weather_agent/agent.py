"""Weather Agent definition for ADK Bidi-streaming demo."""

from google.adk.agents import Agent
from google.adk.tools import ToolContext, FunctionTool
from ..novasonic_llm import NovaSonic
from typing import Dict

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

agent = Agent(
    name="weather_agent",
    model=model,
    tools=[FunctionTool(func=get_weather)],
    instruction="You are a helpful assistant that can provide weather information.",
)
