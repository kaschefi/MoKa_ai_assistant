# core/router.py
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START, END
from schemas.request_models import AgentState, RouteDecision
from actions.digital.n8n_agents import call_n8n_calendar, call_web_search
from actions.digital.langchain_agents import weather_worker
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from core.tool_vector_db import tool_rag_registry

GRAY = "\033[90m"
RESET = "\033[0m"

router_llm = ChatOllama(model="qwen2.5:1.5b", temperature=0, base_url="http://localhost:11434")
structured_router = router_llm.with_structured_output(RouteDecision)
chat_llm = ChatOllama(model="qwen2.5:1.5b", temperature=0.6, base_url="http://localhost:11434")


# --- GRAPH NODES ---

def tool_retrieval_node(state: AgentState):
    """
    RAG LAYER STEP 1: Programmatically query our tool database vector space
    to pull only the top 2-3 matching candidates.
    """
    last_message = state["messages"][-1].content
    # Pull top 2 most matching tools to keep the prompt absolutely razor sharp
    matched_tools = tool_rag_registry.search_relevant_tools(last_message, k=2)
    return {"active_tools": matched_tools}


def route_query(state: AgentState):
    """
    RAG LAYER STEP 2: Dynamically construct a prompt with ONLY the tools
    selected by step 1 and let Ollama pick the target path.
    """
    last_message = state["messages"][-1].content
    active_tools = state.get("active_tools", [])

    # Format the retrieved tools into a string snippet for the system prompt
    tool_menu_string = ""
    for tool in active_tools:
        tool_menu_string += f'- "{tool["name"]}": {tool["description"]}\n'

    dynamic_prompt = f"""You are Cozmo's routing supervisor.
Your ONLY job is to classify the user's request into exactly ONE of the current active routes.

You MUST choose one of these options provided dynamically by the retrieval network:
{tool_menu_string}- "none": Use this if the query does not match any tool option above and is just casual chat or general knowledge.

STRICT RULES:
1. Output a structured decision containing the exact string name of the chosen route.
2. If it does not belong to any retrieved tool options, output "none".
3. Never answer the query yourself.
"""

    # We bypass a static chain and compile it dynamically via an on-the-fly invocation
    decision = structured_router.invoke([
        SystemMessage(content=dynamic_prompt),
        HumanMessage(content=last_message)
    ])

    print(f"\n{GRAY} LangGraph Dynamic Decision: {decision.route}{RESET}")
    return {"next_route": decision.route}


# --- WORKER NODES  ---

def calendar_node(state: AgentState):
    last_message = state["messages"][-1].content
    n8n_reply = call_n8n_calendar(last_message)
    return {"messages": [AIMessage(content=n8n_reply)]}


def web_search_node(state: AgentState):
    last_message = state["messages"][-1].content
    reply = call_web_search(last_message)
    if not reply:
        reply = "I tried searching, but couldn't reach the search service."
    return {"messages": [AIMessage(content=reply)]}


def weather_node(state: AgentState):
    result = weather_worker.invoke({"messages": state["messages"]})
    return {"messages": [result["messages"][-1]]}


def chat_node(state: AgentState):
    print(f"{GRAY}Routing to local chat...{RESET}")
    response = chat_llm.invoke(state["messages"])
    return {"messages": [response]}


def decide_next_step(state: AgentState) -> str:
    """Evaluates router output and targets a node execution branch."""
    route = state.get("next_route", "none")
    # If the LLM returned an active valid tool node, go there. Otherwise fallback to chat.
    if route in ["calendar_node", "web_search_node", "weather_node"]:
        return route
    return "chat_node"


# --- BUILD THE GRADIENT COMPILER GRAPH ---
builder = StateGraph(AgentState)

# Add nodes
builder.add_node("tool_retrieval_node", tool_retrieval_node)
builder.add_node("route_query", route_query)
builder.add_node("calendar_node", calendar_node)
builder.add_node("web_search_node", web_search_node)
builder.add_node("weather_node", weather_node)
builder.add_node("chat_node", chat_node)

# Wire the transitions
builder.add_edge(START, "tool_retrieval_node")
builder.add_edge("tool_retrieval_node", "route_query")
builder.add_conditional_edges("route_query", decide_next_step)

builder.add_edge("calendar_node", END)
builder.add_edge("web_search_node", END)
builder.add_edge("weather_node", END)
builder.add_edge("chat_node", END)

cozmo_graph = builder.compile()


def run_cozmo_agent(user_input: str) -> str:
    initial_state = {"messages": [HumanMessage(content=user_input)]}
    result = cozmo_graph.invoke(initial_state)
    return result["messages"][-1].content