"""LangGraph Deployment - Module-level agent for LangGraph Cloud/Studio.

This module provides the compiled LangGraph agent for deployment:
- Lazy initialization of sandbox and MCP registry
- Single node wrapper around the PTCAgent deepagent
- Compatible with LangGraph Cloud deployment

"""

import asyncio

from langgraph.graph import StateGraph, MessagesState, START, END

from src.ptc_core.session import SessionManager
from src.agent.agent import PTCAgent
from src.agent.config import AgentConfig


# Global session state for sandbox persistence (lazy initialization)
_session = None
_ptc_agent = None
_config = None


async def _ensure_initialized():
    """Ensure PTC session is initialized with sandbox and MCP registry.

    This function lazily initializes:
    - AgentConfig from config.yaml
    - SessionManager with Daytona sandbox
    - PTCAgent instance

    Returns:
        Tuple of (session, ptc_agent)
    """
    global _session, _ptc_agent, _config

    if _session is None:
        # Use async config loading to avoid blocking I/O
        _config = await AgentConfig.load()
        _config.validate_api_keys()

        core_config = _config.to_core_config()
        _session = SessionManager.get_session("langgraph-deployment", core_config)
        await _session.initialize()

        # PTCAgent.__init__ calls get_llm_client() which has blocking I/O
        # (dynamic imports) - wrap in thread to be safe
        _ptc_agent = await asyncio.to_thread(PTCAgent, _config)

    return _session, _ptc_agent


async def ptc_node(state: MessagesState):
    """Main PTC agent node - initializes sandbox on first call and runs agent.

    This node:
    1. Ensures the sandbox and MCP registry are initialized
    2. Creates a deepagent with full PTC capabilities
    3. Runs the agent and returns the result

    Args:
        state: MessagesState containing the conversation history

    Returns:
        Updated state with agent response
    """
    session, ptc_agent = await _ensure_initialized()

    # Create the deepagent with full PTC capabilities
    inner_agent = ptc_agent.create_agent(
        sandbox=session.sandbox,
        mcp_registry=session.mcp_registry,
        subagent_names=["general-purpose"],
    )

    # Run the full agent (deepagent handles its own tool loop)
    result = await inner_agent.ainvoke(state)
    return result


# Build a simple wrapper graph for LangGraph deployment
# The actual agent logic is in the deepagent created by PTCAgent
workflow = StateGraph(MessagesState)
workflow.add_node("ptc", ptc_node)
workflow.add_edge(START, "ptc")
workflow.add_edge("ptc", END)

# Compile the graph for LangGraph deployment
agent = workflow.compile()
