#%%
from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import WebBaseLoader
import os
#%% 
file_name = "sherlock_holmes.txt"
file_name

#%% 
#%% 
text_loader = TextLoader(file_path=file_name, encoding="utf-8")
docs = text_loader.load()
# %%
docs
# %% lade direkt von Project Gutenberg
url_sherlock_holmes_book = "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"
web_loader = WebBaseLoader(web_path=url_sherlock_holmes_book)
docs = web_loader.load()
# %%
docs[0].page_content
# %% load multiple URLs
url_huckleberry = "https://www.gutenberg.org/cache/epub/32325/pg32325.txt"
urls = [url_sherlock_holmes_book, url_huckleberry]
web_loader = WebBaseLoader(web_path=urls)
docs = web_loader.load()
# %%
len(docs[0].page_content)
# %%
docs[0].metadata
# %%
