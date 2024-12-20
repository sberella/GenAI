#%% packages
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())
#%% 
file_name = "data/sherlock_holmes.txt"
file_name

#%% 
#%% 
text_loader = TextLoader(file_path=file_name, encoding="utf-8")
docs = text_loader.load()
# %%
pprint(docs[0].page_content[1000:1500])

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

texts = [chunk.page_content for chunk in doc_chunks]

doc_embeddings = embeddings_model.embed_documents(texts=texts)
#%%
len(doc_embeddings)

#%%
len(doc_embeddings[0])

#%% OpenSource  Modell lokal
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(texts)
# %%
print(len(embeddings))
#%%
print(len(embeddings[0]))

# # %%

#%% Open Source Modell in der Cloud
from langchain_huggingface import HuggingFaceEndpointEmbeddings
embedding_function = HuggingFaceEndpointEmbeddings(model="sentence-transformers/all-MiniLM-L6-v2")
# von hier an wie oben:
# texts = [chunk.page_content for chunk in doc_chunks]

# doc_embeddings = embeddings_model.embed_documents(texts=texts)

#%% OPENAI_API_KEY