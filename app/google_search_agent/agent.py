"""Google Search Agent definition for ADK Bidi-streaming demo."""

from google.adk.agents import Agent
from google.adk.tools import google_search
from ..novasonic_llm import NovaSonic

# Replacing the model with Nova Sonic model
# TBD need to add other params for model authentication
model = NovaSonic(model="amazon.nova-2-sonic-v1:0")

agent = Agent(
    name="google_search_agent",
    model=model,
    tools=[google_search],
    instruction="You are a helpful assistant that can search the web.",
)
