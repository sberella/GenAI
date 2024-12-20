#%% packages
from langchain.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader

# %%
directory = "data/"
loader = DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
docs = loader.load()

# %%
docs
# %%
