import functools
import operator
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated, Sequence, Dict, Any


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    next: str


def create_workflow():
    return StateGraph(AgentState)


def create_supervisor_agent(llm, members):
    system_prompt = (
        "You are a controller tasked with managing a conversation between the"
        " following workers:  {members}. Given the following user request,"
        " respond with the worker to act next. Each worker will perform a"
        " task and respond with their results and status. When finished,"
        " respond with FINISH."
    )
    options = ["FINISH"] + members
    function_def = {
        "name": "route",
        "description": "Select the next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                }
            },
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            # (
            #     "system",
            #     "Given the conversation above, who should act next?"
            #     " Or should we FINISH? Select one of: {options}",
            # ),
        ]
    ).partial(options=str(options), members=", ".join(members))

    supervisor_chain = (
            prompt
            # | llm.bind_functions(functions=[function_def], function_call="route")
            | llm.bind_tools(tools=[function_def])
            | JsonOutputFunctionsParser()
    )
    return supervisor_chain


def create_agent(llm, tools: list, system_prompt: str):
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                system_prompt,
            ),
            MessagesPlaceholder(variable_name="messages"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )
    agent = create_tool_calling_agent(llm, tools, prompt)
    executor = AgentExecutor(agent=agent, tools=tools)
    return executor


def agent_node(state, agent, name):
    result = agent.invoke(state)
    return {"messages": [HumanMessage(content=result["output"], name=name)]}


def build_supervisor(llm, members_mapping: Dict[str, Any]):
    members_names = list(members_mapping.keys())
    # print(f"members_names: {members_names}")
    supervisor_agent = create_supervisor_agent(llm, members_names)

    workflow = create_workflow()

    workflow.set_entry_point("controller")

    workflow.add_node("controller", supervisor_agent)

    for key, value in members_mapping.items():
        '''
            {
                "Researcher": {
                    "llm": llm,
                    "tools": [TavilySearchResults(max_results=5)],
                    "prompt": "You are a web researcher."
                }
            }
        '''
        research_agent = create_agent(value["llm"], value["tools"], value["prompt"])
        research_node = functools.partial(agent_node, agent=research_agent, name=key)
        workflow.add_node(key, research_node)

    for member in members_names:
        workflow.add_edge(member, "controller")

    conditional_map = {k: k for k in members_names}
    conditional_map["FINISH"] = END
    workflow.add_conditional_edges("controller", lambda x: x["next"], conditional_map)
    return workflow.compile()
