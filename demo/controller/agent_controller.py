from langchain_core.messages import HumanMessage
from langchain_google_vertexai import ChatVertexAI
import vertexai
from controller.agent_builder import build_supervisor


# llm = ChatOpenAI(model="gpt-4-turbo")
vertexai.init(project="ai-demo-432913")
llm = ChatVertexAI(model="gemini-1.5-flash-001")

members = {
    "SqlGenerator": {
        "llm": llm,
        "tools": [],
        "prompt": "You are a SQL generator."
    },
    "TextRecognitionAssistant": {
        "llm": llm,
        "tools": [],
        "prompt": "You are an image text recognition assistant, you can get the raw text from given immage. The recognized text needs to be exactly the same as the text in the image."
    }
}

supervisor = build_supervisor(llm, members)

def output_welcome():
    welcomeText = '''
    ****************************************
        Welcome to use the Agent Gallery!  
    ****************************************
    '''
    print(welcomeText)

def execute_agent():
    output_welcome()
    
    user_input = input("First please tell me what you want to do: ")

    inputContent = {"messages": [HumanMessage(content=user_input)]}
    for s in supervisor.stream(inputContent):
        if "__end__" not in s:
            print(s)
            print("----")