#%% packages
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
import os
from pprint import pprint
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
doc_chunks[-1].metadata