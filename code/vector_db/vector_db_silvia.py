#%% packages
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
from pprint import pprint
from langchain.vectorstores import Chroma

load_dotenv(find_dotenv())
#%%
file_name = "data/sherlock_holmes.txt"
file_name
 
#%%
text_loader = TextLoader(file_path=file_name, encoding="utf-8")
docs = text_loader.load()
# %%
pprint(docs[0].page_content[1000:1200])
 
#%% Data Chunking
splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 100)
# %%
doc_chunks = splitter.split_documents(docs)
# %%
len(doc_chunks)
# %%
from pprint import pprint
pprint(doc_chunks[100].page_content, width=50)
# %%
pprint(doc_chunks[101].page_content, width=50)
 
# %%
chunk_lengths = [len(chunk.page_content) for chunk in doc_chunks]
chunk_lengths
# %% alternative: klassische for schleife
chunk_lengths = []
for chunk in doc_chunks:
    chunk_lengths.append(len(chunk.page_content))
# %%
import seaborn as sns
# %%
sns.histplot(chunk_lengths, bins=50, binrange=(950, 1050))
# %%
len(doc_chunks)
 

#%% OpenAI - Embeddings
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
#%% retrieve data
texts = [chunk.page_content for chunk in doc_chunks]
texts
#%% retrieve data
doc_embeddings = embeddings_model.embed_documents(texts=texts)
#%%

#db= Chroma()
persistent_db_path1 ="db_silvia"
db = Chroma(persist_directory = persistent_db_path1, embedding_function=embeddings_model)
db.add_documents(doc_chunks)
#%% retrieve data
retriever=db.as_retriever()
user_query="who works with sherlock holmes?"
most_similar_docs= retriever.invoke(user_query)
most_similar_docs
#%%
from pprint import pprint
pprint(most_similar_docs[0].page_content)
#%%
doc_chunks
# %%
 
