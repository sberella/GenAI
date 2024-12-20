#%% packages
import os
from dotenv import load_dotenv, find_dotenv
from langchain_groq import ChatGroq

#%% load environment variables
load_dotenv(find_dotenv())

# %% access environment variables
# os.getenv('OPENAI_API_KEY')

# %% model instance
# Groq models: https://console.groq.com/docs/models
MODEL_NAME = 'gemma2-9b-it'
model = ChatGroq(model_name=MODEL_NAME)
res = model.invoke("What is my name?")
# %% invoke the model

# %%
res.content

#%% shortcuts
from load_my_env import load_env
load_env()



# %%
from pprint import pprint
pprint(res.model_dump_json())
