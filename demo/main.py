from dotenv import load_dotenv

from agents.agent_nodes import prepare
from agents.workflow import execute_agent

if __name__ == "__main__":

    prepare()
    execute_agent()
