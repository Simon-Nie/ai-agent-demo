from typing import TypedDict, Annotated,Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain.agents.output_parsers.tools import ToolAgentAction
from langchain_core.messages import BaseMessage
import operator


class AgentState(TypedDict):
   # The input string from human
   input: str
   # The list of previous messages in the conversation
   chat_history: list[BaseMessage]
   # The outcome of a given call to the agent
   # Needs 'list' as a valid type as the tool agent returns a list.
   # Needs `None` as a valid type, since this is what this will start as
   # this state will be overwritten with the latest everytime the agent is run
   agent_outcome: Union[AgentAction, list, ToolAgentAction, AgentFinish, None]

   # List of actions and corresponding observations
   # These actions should be added onto the existing so we use `operator.add`
   # to append to the list of past intermediate steps
   intermediate_steps: Annotated[list[Union[tuple[AgentAction, str], tuple[ToolAgentAction, str]]], operator.add]