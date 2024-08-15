import os
import json
from langchain.agents import AgentExecutor, create_react_agent, create_structured_chat_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import MessagesPlaceholder, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_core.tools import StructuredTool
from langchain_openai import ChatOpenAI
import chromadb
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool

os.environ["OPENAI_API_KEY"] = "sk-WHRCoUdd39GWq4RW084f16CeBaAc4f21BdD178F6Ba55Fd80"
os.environ["OPENAI_API_BASE"] = "https://api.xty.app/v1"

#default connect to localhost 8000
new_client = chromadb.HttpClient()
# new_client = chromadb.EphemeralClient()
embeddings = OpenAIEmbeddings()
vector_db_dependency_tree = Chroma(
    client=new_client,
    collection_name="risk-evaluation-dependency-tree",
    embedding_function=embeddings,
)
vector_db_class_level_dependency = Chroma(
    client=new_client,
    collection_name="risk-evaluation-class-level-dependency",
    embedding_function=embeddings,
)

dependency_tree_retriever = vector_db_dependency_tree.as_retriever(search_type="similarity", search_kwargs={"k": 5})
class_level_dependency_retriever = vector_db_class_level_dependency.as_retriever(search_type="similarity", search_kwargs={"k": 5})

def prepare():
    dependency_tree = TextLoader('dependency-tree.log',encoding="utf-8").load_and_split(CharacterTextSplitter(
        separator="\n\n",
        chunk_size=20000,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False
    ))
    jdeps_output = TextLoader('jdeps-output.log',encoding="utf-8").load_and_split(CharacterTextSplitter(
        separator="\n",
        chunk_size=400,
        chunk_overlap=50,
        length_function=len,
        is_separator_regex=False
    ))
    vector_db_dependency_tree.add_documents(dependency_tree)
    vector_db_class_level_dependency.add_documents(jdeps_output)



def search_dependency_tree(text: str) -> str:
    docs = dependency_tree_retriever.invoke(text)
    return "\n\n".join(doc.page_content for doc in docs)

def search_class_level_dependency(text: str) -> str:
    docs = class_level_dependency_retriever.invoke(text)
    return "\n\n".join(doc.page_content for doc in docs)


# Prompts

SYSTEM_PROMPT = """
You are an AI agent specialized in analyzing software dependencies. You will analyze the impact of upgrading a specific dependency in a Java project.

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

Final answer to user must follow below format, and reminder to use double quotes in JSON $JSON_BLOB key and value, don't use single quote:
```
{{
    "riskLevel": "Low",
    "comments": "leave your comments",
    "usage": [
        {{
            "class": "com.mycompany.demo.MyJsonHelper",
            "path": "src/main/java/com/mycompany/demo/MyJsonHelper.java"
        }}
    ]
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

prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    HumanMessagePromptTemplate.from_template(HUMAN_PROMPT)
])

llm = ChatOpenAI(model='gpt-4o', temperature=0.0)

# tool = create_retriever_tool(
#     retriever,
#     "name",
#     "description",
# )

tools = [
    StructuredTool.from_function(search_class_level_dependency, name="search_class_level_dependency", description="Search output from `jdeps -v example.jar`, which provides detailed information about the dependencies used by the project."),
    StructuredTool.from_function(search_dependency_tree, name="search_dependency_tree", 
                                 description="Search the output from the `gradle dependencyTree` or `mvn dependency:tree` command, which outlines the hierarchical structure of all dependencies in the project.")
    # tool
]



if __name__ == '__main__':
    prepare()
    log_file_path = 'git-change.log'
    with open(log_file_path, 'r',encoding='utf-8') as file:
        content = file.read()

    agent = create_structured_chat_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True, handle_parsing_errors=True)

    result = agent_executor.invoke({"input": content})
    print(result['output'])


# https://python.langchain.com/v0.1/docs/use_cases/question_answering/conversational_retrieval_agents/