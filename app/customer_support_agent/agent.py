"""Customer Support demo agent for ADK Bidi-streaming.

Minimal, deterministic stub tools and sub-agents for demo purposes.
Replace stubs with real integrations when wiring to backend services.
"""

from typing import Dict

from google.adk.agents import Agent
from google.adk.tools import ToolContext, FunctionTool
from ..novasonic_llm import NovaSonic

# Use NovaSonic model as in other demos
model = NovaSonic(model="amazon.nova-2-sonic-v1:0")


def product_info(tool_context: ToolContext, product_name: str) -> Dict:
    """Return a short marketing blurb and key benefits for a financial product (demo stub)."""
    p = product_name.strip().title()
    demo = {
        "Wealth Plus": {
            "name": "Wealth Plus",
            "summary": "A tailored wealth management product combining robo-advice with human oversight.",
            "benefits": ["Personalized portfolio", "Low fees", "Quarterly reviews"],
            "eligibility": "Available to residents with minimum deposit $5,000",
        },
        "Savings Boost": {
            "name": "Savings Boost",
            "summary": "High-yield savings account with flexible withdrawals and bonus APR for new customers.",
            "benefits": ["Competitive APR", "No monthly fees", "Mobile deposit"],
            "eligibility": "Open to new customers",
        },
    }
    print(f"--- Tool: product_info called for {p} ---")
    return demo.get(p, {"name": p, "summary": "A flexible financial product.", "benefits": [], "eligibility": "Contact support for details"})


def faq_search(tool_context: ToolContext, query: str) -> Dict:
    """Return a short FAQ-style answer (demo stub)."""
    q = query.lower()
    print(f"--- Tool: faq_search called for query: {q[:40]} ---")
    if "fees" in q or "price" in q or "fee" in q:
        return {"answer": "Our fees are competitive; Wealth Plus charges 0.25% AUM annually.", "source": "Product FAQ"}
    if "signup" in q or "open" in q:
        return {"answer": "Sign up is online and takes about 10 minutes. You'll need ID and a funding source.", "source": "Onboarding FAQ"}
    return {"answer": "I can route to a human for this here.", "source": "General FAQ"}


def create_lead(tool_context: ToolContext, customer_name: str, contact: str, interest: str) -> Dict:
    """Create a demo lead and return its id (deterministic stub)."""
    print(f"--- Tool: create_lead called for {customer_name} ---")
    lead_id = "LEAD" + str(abs(hash(customer_name + contact)) % 10000)
    return {"lead_id": lead_id, "customer_name": customer_name, "contact": contact, "interest": interest}


def schedule_demo(tool_context: ToolContext, customer_name: str, preferred_times: str) -> Dict:
    """Schedule a demo call (demo stub) and return a scheduled time slot."""
    print(f"--- Tool: schedule_demo called for {customer_name} ---")
    # deterministic simple schedule logic
    slot = "2026-03-15T10:00:00Z"
    return {"scheduled_for": customer_name, "slot": slot, "timezone": "UTC"}


def marketing_tool(tool_context: ToolContext, product_name: str) -> Dict:
    """Produce a short spoken-friendly pitch for the requested product."""
    p = product_info(tool_context, product_name)
    pitch = f"{p['name']}: {p['summary']} Benefits: {', '.join(p['benefits'])}."
    return {"pitch": pitch}

marketing_agent = Agent(
    name="marketing_agent",
    description="Specialist for product pitches, features, and marketing information.",
    model=model,
    tools=[FunctionTool(func=marketing_tool), FunctionTool(func=product_info)],
    instruction="""
You are a Marketing Specialist. 
Your ONLY goal is to describe products and features using `marketing_tool`.
- If the user asks about fees or opening an account, do not answer; transfer them to 'user_support_agent'.
- ALWAYS append: 'All financial products carry risk. Consultation recommended.'
""",
)


# User support subagent: handles FAQs and common operational questions
user_support_agent = Agent(
    name="user_support_agent",
    description="Specialist for FAQs, account setup questions, and fees.",
    model=model,
    tools=[FunctionTool(func=faq_search)],
    instruction="""
You are a Support Specialist. 
Your ONLY goal is to handle operational questions via `faq_search`.
- If the user wants to buy a product or schedule a demo, do not answer; transfer them to 'sales_agent'.
- Keep responses strictly factual and helpful.
""",
)


# Sales subagent: captures leads and schedules demos
sales_agent = Agent(
    name="sales_agent",
    description="Specialist for new customer onboarding, lead capture, and demo scheduling.",
    model=model,
    tools=[FunctionTool(func=create_lead), FunctionTool(func=schedule_demo)],
    instruction="""
You are a Sales Specialist. 
Your ONLY goal is to capture leads using `create_lead` and book slots with `schedule_demo`.
- Do not handle general support or detailed marketing pitches; transfer to 'marketing_agent' if needed.
- Be closing-oriented and professional.
""",
)


# Root customer support agent (simplified)
agent = Agent(
    name="product_concierge",
    model=model,
    sub_agents=[marketing_agent, user_support_agent, sales_agent],
    instruction="""
You are a pure Router. You MUST NOT answer any financial or support questions yourself.
MANDATORY PROTOCOL:
1. Analyze the user utterance.
2. IMMEDIATELY transfer control to the correct specialist:
   - For signup/onboarding/joining/new accounts -> transfer to `sales_agent`
   - For product info/features/explanations -> transfer to `marketing_agent`
   - For fees/FAQs/technical support/how-to -> transfer to `user_support_agent`
If you answer a question directly, you have failed your instruction. You are a redirector.
""",
)

