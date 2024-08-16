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

import risk_evaluation_agent
import test_scope_evaluation_agent
import summary_agent
import tiktoken
import subprocess

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
llm = ChatOpenAI(model='gpt-4o', temperature=0.0)

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

def read_file_content(path: str) -> str:
    print("in read_file_content: " + path)
    with open(path, 'r',encoding='utf-8') as file:
        return file.read()
    
def run_git_command(command: str) -> str:
    print("in run_git_command: " + command)
    result = subprocess.run(command, shell=True, capture_output=True, text=True, encoding="utf-8")

    if result.returncode == 0:
        return result.stdout
    else:
        return f"Error: {result.stderr}"
    
    
# tool = create_retriever_tool(
#     retriever,
#     "name",
#     "description",
# )

risk_tools = [
    StructuredTool.from_function(search_class_level_dependency, name="search_class_level_dependency", description="Search output from `jdeps -v example.jar`, which provides detailed information about the dependencies used by the project."),
    StructuredTool.from_function(search_dependency_tree, name="search_dependency_tree", 
                                 description="Search the output from the `gradle dependencyTree` or `mvn dependency:tree` command, which outlines the hierarchical structure of all dependencies in the project.")
    # tool
]

test_tools = [
    StructuredTool.from_function(read_file_content, name="read_file_content", description="Read file content based input file path"),
    StructuredTool.from_function(run_git_command, name="run_git_command", 
                                 description="Run git command as shell mode, and return the standand output of git command")
    # tool
]




if __name__ == '__main__':
    prepare()
    log_file_path = 'git-change.log'
    with open(log_file_path, 'r',encoding='utf-8') as file:
        content = file.read()

    riskAgent = risk_evaluation_agent.buildAgent(llm, risk_tools)
    testScopeAgent = test_scope_evaluation_agent.buildAgent(llm, test_tools)
    summaryAgent = summary_agent.buildAgent(llm,risk_tools)

    result = riskAgent.invoke({"input": content})
    print("---------output of riskAgent-------------")
    riskEvaluationResult = json.dumps(result['output'])
    print(riskEvaluationResult)

    # riskEvaluationResult = """
    # {
    #     "riskLevel": "Medium",
    #     "comments": "Upgrading `joda-time` from version `2.10.13` to `2.12.7` may introduce breaking changes in the API. Classes that directly use `joda-time` functionalities such as `DateTime` and `DateTimeZone` might be affected, thus impacting multiple parts of the codebase that handle date and time operations.",
    #     "usage": [
    #         {
    #             "class": "io.spring.application.DateTimeCursor",
    #             "path": "src/main/java/io/spring/application/DateTimeCursor.java"
    #         },
    #         {
    #             "class": "io.spring.JacksonCustomizations$DateTimeSerializer",
    #             "path": "src/main/java/io/spring/JacksonCustomizations.java"
    #         }
    #     ],
    #     "changes": {
    #         "groupId": "joda-time",
    #         "artifactId": "joda-time",
    #         "oldVersion": "2.10.13",
    #         "newVersion": "2.12.7"
    #     }
    # }
    
    # """

    result = testScopeAgent.invoke({"input": riskEvaluationResult})
    print("---------output of testScopeAgent-------------")
    testEvaluationResult = json.dumps(result['output'])
    print(testEvaluationResult)

# https://python.langchain.com/v0.1/docs/use_cases/question_answering/conversational_retrieval_agents/