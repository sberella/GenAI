#%% packages
from langchain_core.prompts import ChatPromptTemplate
from pprint import pprint
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
from langchain.schema.output_parser import StrOutputParser


#%% prompt template
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an AI assistant that translates English into another language. No babbling, just answer the question."),
    ("user", "Translate this sentence: '{input}' into '{target_language}'")
])
# %% TEST
# pprint(prompt_template.invoke({"input": "I love programming", "target_language": "German"}))
# %% 
model = ChatGroq(model="llama-3.1-8b-instant", temperature=0.5)

# %%
chain = prompt_template | model | StrOutputParser()
# %%
res = chain.invoke({"input": "I love programming", "target_language": "German"})
# %%
pprint(res)
# %%
chain2 = prompt_template | model | StrOutputParser()
chain2.invoke({"input": res, "target_language": "Japanese"})
# %%
