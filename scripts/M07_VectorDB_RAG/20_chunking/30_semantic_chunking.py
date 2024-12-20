#%% Packages (1)
from langchain_experimental.text_splitter import SemanticChunker
from langchain.document_loaders import WikipediaLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from pprint import pprint
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=True))
# %% Load the article (2) 
ai_article_title = "Artificial_intelligence"
loader = WikipediaLoader(query=ai_article_title, 
                             load_all_available_meta=True, 
                             doc_content_chars_max=1000, 
                             load_max_docs=1)
doc = loader.load()

# %% check the content (3)
pprint(doc[0].page_content)
# %% Create splitter instance (4)
splitter = SemanticChunker(embeddings=OpenAIEmbeddings())

# %% Apply semantic chunking (5)
chunks = splitter.split_documents(doc)

# %% check the results (6)
text = "Cars drive on streets. The cat chased the mouse. The mouse was afraid. "

# %%
chunks = splitter.split_text(text)
# %%
chunks
# %%
