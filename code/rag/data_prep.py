#%% packages
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from datasets import load_dataset
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
#%% alternative vector DB für Stephanie, Silvia
import faiss
# %% get the dataset
dataset = load_dataset("MongoDB/embedded_movies", split="train")

# %% Anzahl der Filme in dem Datensatz
len(dataset)
# %%
dataset[0].keys()

# %% used keys:
# fullplot
# title

#%% export data for Sebastian as json
# export fullplot and title
# import json
# with open('data/movies.json', 'w') as f:
#     for doc in dataset:
#         if doc['fullplot'] is not None:
#             f.write(json.dumps({'fullplot': doc['fullplot'], 'title': doc['title']}) + '\n')

# %% extract data from our dataset
# als Ergebnis wollen wir docs = [Document(page_content=full_plot, metadata = {'title': title}), ...]
docs = []
for doc in dataset:
    title = doc['title'] if doc['title'] is not None else ""
    metadata = {'title': title}
    if doc['fullplot'] is not None:
        docs.append(Document(page_content=doc['fullplot'],
                             metadata=metadata))
        print(title)
    
    
# %%
len(docs) # 1452 documents

# %%
docs
# %% TODO: Chunk of data (chunk size 1000)
# text_splitter = RecursiveCharacterTextSplitter(chunk_size =1000, chunk_overlap = 200)
# doc_chunks = text_splitter.split_documents(docs)

#%% TODO: check the number of chunks after chunking
# len(doc_chunks)
#%% TODO: load Embedding model
page_contents = [doc.page_content for doc in docs]
# [str, str]
# %% 
# Chroma + OpenAI
embeddings_function = OpenAIEmbeddings()
db = Chroma(persist_directory="movies", embedding_function=embeddings_function)

#%%
db.add_documents(docs)

#%%
retriever = db.as_retriever()
user_query = "a detective story"
result = retriever.invoke(user_query)
result
#%%
docs[0].page_content

#%%
# Faiss + local embedding (sentence-transformers)
# %% erstelle Embeddings
from sentence_transformers import SentenceTransformer
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(page_contents)
print(len(embeddings))


# %% FAISS + OpenAI (Silvia)
embeddings_function = OpenAIEmbeddings()
from langchain.vectorstores import FAISS
faiss_db = FAISS.from_documents(docs, embeddings_function)
#%%
faiss_db.save_local("faiss_movies")

#%%
retriever = faiss_db.as_retriever()
user_query = "a detective story"
result = retriever.invoke(user_query)
result



# %% FAISS + local embedding (Stephanie)
from langchain.vectorstores import FAISS
import numpy as np
import os
dimensions = embeddings.shape[1]
class LocalEmbeddings:
    def __init__(self, model):
        self.model = model
    
    def embed_documents(self, texts):
        embeddings = self.model.encode(texts)
        return [np.array(embedding).astype(np.float32) for embedding in embeddings]
    
    def embed_query(self, text):
        embedding = self.model.encode([text])[0]
        return np.array(embedding).astype(np.float32)

embeddings_function = LocalEmbeddings(model)

#%%
faiss_db = FAISS.from_documents(docs, embeddings_function)

# %%
# %% store db
faiss_db.save_local("faiss_movies_local")

#%% invoke faiss_db
user_query = "a detective story"
user_embedding = embeddings_function.embed_query(user_query)
result = faiss_db.similarity_search_by_vector(user_embedding)
print(result)


# %%
