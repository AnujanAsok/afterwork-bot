from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
import os
from langchain.chains import LLMChain
from langchain.agents import load_tools
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain import OpenAI, ConversationChain
os.environ["OPENAI_API_KEY"]='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'
os.environ["SERPAPI_API_KEY"] = "d29c92b3ac2a36b4b26a22457a031c2dd8383295b4170c2bc1fe60d02ecf1a8b"

# llm = OpenAI(temperature='0.9')
# # text="What would be a good company name for a company that makes colorful socks?"
# # print(llm(text))

# prompt = PromptTemplate(
#     input_variables=["product"],
#     template="What is a good name for a company that makes {product}?",
# )
# print(prompt.format(product="colorful socks"))
# # print(llm(prompt.format(product='colorful socks')))

# chain = LLMChain(llm=llm, prompt=prompt)
# chain.run('colorful socks')
# # chain = LLMChain(llm=llm, prompt=prompt)
# # chain.run("colorful socks")
# # -> '\n\nSocktastic!'

# First, let's load the language model we're going to use to control the agent.
llm = OpenAI(temperature=0)

# Next, let's load some tools to use. Note that the `llm-math` tool uses an LLM, so we need to pass that in.
# tools = load_tools(["serpapi", "llm-math"], llm=llm)
conversation = ConversationChain(llm=llm, verbose=True)
output = conversation.predict(input="Hi there!")
print(output)

output = conversation.predict(input='I am doing well! Just having a conversation with an AI.')
print(output)
# Finally, let's initialize an agent with the tools, the language model, and the type of agent we want to use.
# agent = initialize_agent(tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)

# Now let's test it out!
# agent.run("Give me the top rated caterer in sf. What is their rating?")