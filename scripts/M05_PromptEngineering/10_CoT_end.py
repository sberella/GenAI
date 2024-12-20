#%% packages
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pprint import pprint
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
# %% Function Definition
MODEL_NAME = "gpt-4o-mini"
model = ChatOpenAI(model=MODEL_NAME, temperature=0)

class ChainOfThoughtResponse(BaseModel):
    equation: str = Field(description="The answer to the user's question")

#%% Function Definition
def chain_of_thought(user_query:str):
    system_role_definition = "You are a mathematic genius and can solve the game of 24. The goal of the Game of 24 is to use the four arithmetic operations (addition, subtraction, multiplication, and division) to combine four numbers and get a result of 24. Please check the final equation for correctness. Hints: Identify the basic operations, Prioritize multiplication and division, Look for combinations that make numbers divisible by 24, Consider order of operations, Use parentheses strategically, Practice with different number combinations. It is mandatory to use all four numbers. Only return the equation as JSON object with the following fields: equation. The user provides 4 numbers and you need to find the equation that solves the game of 24, think step-by-step. Please check the final equation if it equals 24."
    user_query_complete = user_query
    messages = [
        ("system", system_role_definition),
        ("user", user_query_complete)
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages)
    chain = prompt_template | model | JsonOutputParser(pydantic_object=ChainOfThoughtResponse)
    response = chain.invoke({})
    return response

#%% Test
user_prompt = "3, 4, 6, and 8"
res = chain_of_thought(user_query=user_prompt)
pprint(res)
# %%
