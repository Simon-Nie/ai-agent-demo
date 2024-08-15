# Prompts
from typing import Sequence
from langchain_core.tools import BaseTool
from langchain_core.language_models import BaseLanguageModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.agents import AgentExecutor, create_structured_chat_agent

SYSTEM_PROMPT = """
You are an AI agent analyzing the impact of dependency upgrades in a Java project in java file level. 
Given a JSON input that includes a risk assessment, comments, changes of jar dependency and a list of Java files with their paths, your task is to perform the following steps:

1. **Identify impact lines/functions**:
    - For each Java file listed under the "usage" node, fetch the content of the file from the specified path.
    - Identify which lines/functions use the artifact function in "changes" node.
    
2. **Identify last commit info for each files in "usage" node**:
    - Retrieve the last commit id for the output of step #1.
    - Retrieve the commit message and author of the commit id.
    - Analyze the commit messages to extract JIRA IDs that follow common patterns (e.g., C123456G-124, C12355-121).
    - Now, clearly know the key three value: "commitId", "jiraId", "author" of the commit for each files

# 3. **Fetch Test Cases JIRA id based on feature JIRA IDs**:
#     - leverage tool to fetch matched test case ids which linked by "is tested by" in JIRA system.

3. **Update the Output**:
    - Enhance the original JSON by adding a `lastCommitInfo` node for each entry in "usage" node. If "usage" node is empty from input, then ignore the this step.

Here is the JSON input structure and example:
```json
{
    "riskLevel": "Low",
    "comments": "The joda-time library is not shown to have significant compatibility issues in recent updates. The transition from version 2.10.13 to 2.12.7 primarily includes minor improvements and bug fixes without breaking changes. Additionally, the classes using joda-time are mainly focused on date and time utilities, which are expected to function correctly as no core fundamental changes were introduced in this version upgrade.",
    "usage": [
        {
            "class": "io.spring.JacksonCustomizations$DateTimeSerializer",
            "path": "src/main/java/io/spring/JacksonCustomizations.java"
        },
        {
            "class": "io.spring.application.DateTimeCursor",
            "path": "src/main/java/io/spring/application/DateTimeCursor.java"
        }
    ],
    "changes": {
        "groupId": "joda-time",
        "artifactId": "joda-time",
        "oldVersion": "2.10.13",
        "newVersion": "2.12.7"
    }
}
```

Here is the JSON output structure and example, just enhance on input json:
```json
{
    "riskLevel": "Low",
    "comments": "The update from Joda-Time 2.10.13 to 2.12.7 is a minor version upgrade. Such upgrades typically include bug fixes and minor improvements, with minimal risk of breaking changes. Compatibility issues are unlikely unless specific deprecated methods have been removed. Usage is limited to serialization functions.",
    "usage": [
        {
            "class": "io.spring.JacksonCustomizations$DateTimeSerializer",
            "path": "src/main/java/io/spring/JacksonCustomizations.java",
            "lastCommitInfo": {
                "commitId": "7dd7cba",
                "jiraId": ""C345678G-890"",
                "author": "A Name"
            }
        },
        {
            "class": "io.spring.application.DateTimeCursor",
            "path": "src/main/java/io/spring/application/DateTimeCursor.java",
            "lastCommitInfo": {
                "commitId": "7dd7cba",
                "jiraId": ""C345678G-890"",
                "author": "A Name"
            }
        }
    ],
    "changes": {
        "groupId": "joda-time",
        "artifactId": "joda-time",
        "oldVersion": "2.10.13",
        "newVersion": "2.12.7"
    }
}
```


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

Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate. Format is Action:```$JSON_BLOB```then Observation

"""

HUMAN_PROMPT = """
the input is below, a JSON input that includes a risk assessment, comments, and a list of Java files with their paths
{input}

think history:
{agent_scratchpad}

Please enhance last commit info for each file under "usage" node.

(reminder to respond in a JSON blob no matter what)
 
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