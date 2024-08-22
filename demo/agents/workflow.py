import json

from langchain_core.agents import AgentFinish
from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agents.agent_state import AgentState
from agents.agent_nodes import run_tool_agent, execute_tools

def should_continue(data):
    # If the agent outcome is an AgentFinish, then we return `exit` string
    # This will be used when setting up the graph to define the flow
    if isinstance(data['agent_outcome'], AgentFinish):
        return "END"
    # Otherwise, an AgentAction is returned
    # Here we return `continue` string
    # This will be used when setting up the graph to define the flow
    else:
        return "CONTINUE"

def build_workflow() -> CompiledStateGraph:
    # Define a new graph
    workflow = StateGraph(AgentState)

    # When nodes are called, the functions for to the tools will be called.
    workflow.add_node("agent", run_tool_agent)


    # Add tool invocation node to the graph
    workflow.add_node("action", execute_tools)

    # Define which node the graph will invoke at start.
    workflow.set_entry_point("agent")

    # Add flow logic with static edge.
    # Each time a tool is invoked and completed we want to
    # return the result to the agent to assess if task is complete or to take further actions

    #each action invocation has an edge leading to the agent node.
    workflow.add_edge('action', 'agent')


    # Add flow logic with conditional edge.
    workflow.add_conditional_edges(
        # first parameter is the starting node for the edge
        "agent",
        # the second parameter specifies the logic function to be run
        # to determine which node the edge will point to given the state.
        should_continue,

        #third parameter defines the mapping between the logic function
        #output and the nodes on the graph
        # For each possible output of the logic function there must be a valid node.
        {
            # If 'continue' we proceed to the action node.
            "CONTINUE": "action",
            # Otherwise we end invocations with the END node.
            "END": END
        }
    )

    # Finally, compile the graph!
    # This compiles it into a LangChain Runnable,
    return workflow.compile()

def execute_agent():
    log_file_path = 'git-change.log'
    with open(log_file_path, 'r',encoding='utf-8') as file:
        content = file.read()

    inputs = {"input": content, "chat_history": []}
    config = {"configurable": {"thread_id": "1"}}
    app = build_workflow()
    for s in app.stream(inputs, config = config):
        print(list(s.values())[0])
        print("----")