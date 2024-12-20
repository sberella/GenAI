#%% packages
# from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

def get_movie_recommendation(user_query: str) -> str:
    """
    Get movie recommendations based on user query.
    
    Args:
        user_query: String containing the user's movie query
        
    Returns:
        String containing the AI's response about relevant movies
    """
    # Retrieval
    #-------------------------
    embedding_function = OpenAIEmbeddings()
    db = FAISS.load_local("faiss_movies", embedding_function, 
                          allow_dangerous_deserialization=True)
    
    retriever = db.as_retriever()
    most_relevant_documents = retriever.invoke(user_query)
    
    # Augmentation
    #-------------------------
    context_info = "; ".join([f"{doc.metadata['title']}: {doc.page_content}" for doc in most_relevant_documents])
    messages = [
        ("system", "Du bist ein Filmexperte. Nutzer stellen dir Fragen zu Filmen und du beantwortest sie auf Basis der Contextinfos, die du mit <context></context> erhältst. Wenn die Kontextinfos nicht ausreichen, um die Frage zu beantworten, sag 'Ich weiß es nicht'. Antworte auf Deutsch."),
        ("user", "Question: {query}. <context>{context_infos}</context>")
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages)
    model = ChatOpenAI()
    
    # Generation
    #-------------------------
    chain = prompt_template | model | StrOutputParser()
    response = chain.invoke({"query": user_query, "context_infos": context_info})
    
    return response

# %%
get_movie_recommendation("Ein Detektiv sucht einen Mörder")