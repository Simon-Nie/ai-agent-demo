import chromadb
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_core.tools import StructuredTool
from langchain_google_vertexai import ChatVertexAI, VertexAIEmbeddings
import vertexai
from langchain.tools import tool
from langchain.agents import create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import CharacterTextSplitter
from langgraph.prebuilt.tool_executor import ToolExecutor

load_dotenv()

vertexai.init(project="ai-demo-432913")
llm = ChatVertexAI(model="gemini-1.5-flash-001")
embeddings = VertexAIEmbeddings(model="textembedding-gecko")
new_client = chromadb.EphemeralClient()

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


@tool
def search_dependency_tree(text: str) -> str:
    """Search the output from the `gradle dependencyTree` or `mvn dependency:tree` command, which outlines the hierarchical structure of all dependencies in the project."""
    docs = dependency_tree_retriever.invoke(text)
    return "\n\n".join(doc.page_content for doc in docs)

@tool
def search_class_level_dependency(text: str) -> str:
    """Search output from `jdeps -v example.jar`, which provides detailed information about the dependencies used by the project."""
    docs = class_level_dependency_retriever.invoke(text)
    return "\n\n".join(doc.page_content for doc in docs)


# toolkit = [
#     StructuredTool.from_function(search_class_level_dependency, name="search_class_level_dependency", description="Search output from `jdeps -v example.jar`, which provides detailed information about the dependencies used by the project."),
#     StructuredTool.from_function(search_dependency_tree, name="search_dependency_tree",
#                                  description="Search the output from the `gradle dependencyTree` or `mvn dependency:tree` command, which outlines the hierarchical structure of all dependencies in the project.")
#     # tool
# ]
toolkit = [search_dependency_tree, search_class_level_dependency]
# setup the toolkit

#define system prompt for tool calling agent
system_prompt = """
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

Your Original input is below, which is output of `git show HEAD`
Analyze the user input into following 4 variables, and analyze the impact of upgrading the dependency:
- groupId
- artifactId
- oldVersion
- newVersion

You must call {{tools}} firstly for necessary dependency info collection and then provide final output.

The final output should be a structured response with the risk level and comments and usage of class list.
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
"""

tool_calling_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

tool_runnable = create_tool_calling_agent(llm, toolkit, prompt  = tool_calling_prompt)

def run_tool_agent(state):
    agent_outcome = tool_runnable.invoke(state)

    #this agent will overwrite the agent outcome state variable
    return {"agent_outcome": agent_outcome}


# tool executor invokes the tool action specified from the agent runnable
# they will become the nodes that will be called when the agent decides on a tool action.

tool_executor = ToolExecutor(toolkit)

# Define the function to execute tools
# This node will run a different tool as specified in the state variable agent_outcome
def execute_tools(state):
    # Get the most recent agent_outcome - this is the key added in the `agent` above
    agent_action = state['agent_outcome']
    if type(agent_action) is not list:
        agent_action = [agent_action]
    steps = []
    # sca only returns an action while tool calling returns a list
    # convert single actions to a list

    for action in agent_action:
        # Execute the tool
        output = tool_executor.invoke(action)
        print(f"The agent action >>> {action}")
        print(f"The tool result <<< {output}")
        steps.append((action, str(output)))
    # Return the output
    return {"intermediate_steps": steps}

