# Prompts
from typing import Sequence
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.agents import AgentExecutor, create_structured_chat_agent



SYSTEM_PROMPT = """
You are an AI agent specialized in analyzing software dependencies. You will analyze the impact of upgrading a specific dependency in a Java project.
And remember that your task is executed based on the successfully building of the project, means no compile error.

Your task is to:
- Assess the impact of upgrading the specified dependency.
- Categorize the impact as Low, Medium, or High risk.
- Provide detailed comments explaining the reasoning behind the assigned risk level, of course, keep comments short.
- List class names which using the changed dependency directly.( if no direct dependency, then mention no direct usage)

Consider factors such as:
- The importance and scope of the dependency in the project.
- Potential compatibility issues with other dependencies.
- Any significant changes between the old and new versions of the dependency.

The final output should be a structured response with the risk level and comments and usage of class list.

Additonally,
You have access to the following tools:

{tools}

Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

Valid "action" values: "Final Answer" or {tool_names}

Provide only ONE action per $JSON_BLOB, as shown:

```
{{
  "action": $TOOL_NAME,
  "action_input": $INPUT
}}
```

Follow this format, and remember to add blank line between Thought/Action/Observation steps:

Question: input question to answer
Thought: consider previous and subsequent steps

Action:
```
$JSON_BLOB

```
Observation: action result
... (repeat Thought/Action/Observation N times)
Thought: I know what to respond

Action:
```
{{
  "action": "Final Answer",
  "action_input": "Final response to human"
}}
```

Final answer to user must follow below format:
```
{{
    "riskLevel": "Low",
    "comments": "leave your comments",
    "usage": [
        {{
            "class": "com.mycompany.demo.MyJsonHelper",
            "path": "src/main/java/com/mycompany/demo/MyJsonHelper.java"
        }}
    ],
    "changes":{{
        "groupId": "com.abc.def",
        "artifactId": "XXXX",
        "oldVersion: "2.1.0",
        "newVersion": "3.1.3"
    }}
}}
```

Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation

"""

HUMAN_PROMPT = """
the input is below, which is output of `git show HEAD`
{input}

think history:
{agent_scratchpad}

Analyze the user input into following 4 variables, and analyze the impact of upgrading the dependency:

- groupId
- artifactId
- oldVersion
- newVersion

Please classify the risk as Low, Medium, or High, and provide your comments and list class file name and path directly using the changed dependency.

(reminder to respond in a JSON blob no matter what and reminder to use double quotes in JSON $JSON_BLOB key and value, don't use single quote)
 
 """


def buildPrompt() -> ChatPromptTemplate:
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history", optional=True),
        HumanMessagePromptTemplate.from_template(HUMAN_PROMPT)
    ])

    return prompt

def buildAgent(llm: BaseLanguageModel, tools:  Sequence[BaseTool]) -> AgentExecutor:

    agent = create_structured_chat_agent(llm, tools, buildPrompt())

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True, handle_parsing_errors=True)
    return agent_executor