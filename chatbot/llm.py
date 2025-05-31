import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# create the llm
llm = ChatOpenAI(
    openai_api_key=st.secrets["OPENAI_API_KEY"],
    model_name=st.secrets["OPENAI_QA_MODEL"],
    temperature=0,
)

cypher_llm = ChatOpenAI(
    openai_api_key=st.secrets["OPENAI_API_KEY"],
    model_name=st.secrets["OPENAI_CYPHER_MODEL"],
    temperature=0,
)

# create the embedding model
embedding_provider = OpenAIEmbeddings(
    openai_api_key=st.secrets["OPENAI_API_KEY"],
    model=st.secrets["EMBEDDING_MODEL"]
)


