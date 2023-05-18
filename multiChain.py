from langchain.chat_models import ChatOpenAI
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.llms import OpenAI
from langchain import SerpAPIWrapper
from langchain.agents.tools import Tool
from langchain import LLMMathChain
import os
from langchain.tools.file_management.write import WriteFileTool
from langchain.tools.file_management.read import ReadFileTool
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field, validator
from typing import List

os.environ["OPENAI_API_KEY"]='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'
os.environ["SERPAPI_API_KEY"] = "d29c92b3ac2a36b4b26a22457a031c2dd8383295b4170c2bc1fe60d02ecf1a8b"


search = SerpAPIWrapper()
llm = OpenAI(temperature=0)
llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
tools = [
    Tool(
        name = "Search",
        func=search.run,
        description="useful for when you need to answer questions about current events. Ask specific questions."
    ),
    Tool(
        name="Calculator",
        func=llm_math_chain.run,
        description="useful for when you need to answer questions about math"
    ),
    WriteFileTool(),
    ReadFileTool(),
]
model = ChatOpenAI(temperature=0)
planner = load_chat_planner(model)
executor = load_agent_executor(model, tools, verbose=True)
agent = PlanAndExecute(planner=planner, executer=executor, verbose=True)

agent.run("write a report on the top 3 best venues for a $5000 budget, for about 60 guests in SF for a corporate dinner? Do they offer in house catering?")