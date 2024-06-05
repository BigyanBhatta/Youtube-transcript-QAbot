from dotenv import load_dotenv
from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

from langchain.chains.llm import LLMChain

import os
load_dotenv()

os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

# create database of transcript from youtube URL
def create_db_youtube (url:str):
    """
    argument(str): url
    returns:
        database extracted from transcript of the youtube video 
    """

    loader = YoutubeLoader.from_youtube_url(youtube_url=url)
    transcript = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size = 250, chunk_overlap = 50)
    docs = splitter.split_documents(transcript)

    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(docs, embeddings)
    return db


def get_response_from_query(db, query, k  =4):
    """
    arguments:
        db: database from create_db_youtube function
        query: user question
        k: top k similarity

    returns:
        response(str): output from the LLM
        docs: similarity search chunks from document
    """
    chat_LLM = ChatOpenAI(model= "gpt-3.5-turbo-0125", temperature = 0.5)

    docs = db.similarity_search(query, k = k)
    # to extract the page content from each document and concatenate them into a single string seperated by a space
    docs_page_content = " ".join([d.page_content for d in docs])

    #system prompt template
    system_template = """
    You are a helpful assistant that can answer question related to youtube videos from the video's transcript.{docs}

    Only answer if you find the information, answer it factually. If answer not found, just say 'I couldn't find any answer.

    Make the answer detailed and informative.
    """
    system_message_prompt_template = SystemMessagePromptTemplate.from_template(system_template)

    #human prompt template
    human_template = """
    Answer the question: {query}
    """
    human_message_prompt_template = HumanMessagePromptTemplate.from_template(human_template)

    promt_template = ChatPromptTemplate.from_messages([human_message_prompt_template, system_message_prompt_template])


    chain = promt_template | chat_LLM
    
    message = {"query": query, "docs": docs_page_content}

    response = chain.invoke(message)
    return response, docs


#Example Usage
url = 'https://www.youtube.com/watch?v=_au3yw46lcg'
db = create_db_youtube(url)

query = 'Tell me about what Andrej has to say about his personal story.'
response, docs = get_response_from_query(db, query=query)
print(response)











    












