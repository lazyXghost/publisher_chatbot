# https://medium.com/@shaktikanungo2019/conversational-ai-unveiling-the-first-rag-chatbot-with-langchain-8b9b04ee4b63

# import PyPDF2
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import os
import pandas as pd

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain_core.messages import HumanMessage
from langchain.chains.combine_documents import create_stuff_documents_chain

import gradio as gr


DB_FAISS_PATH = "../vectorstore/db_faiss"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 256
SIMILARITY_THRESHOLD = 0.5
embedding_model = "text-embedding-3-small"
model_name = "gpt-3.5-turbo"


def setupDbAndChain(data_path):
    embeddings = OpenAIEmbeddings(model=embedding_model)
    df = pd.read_csv(data_path, sep='\t')
    # relevant_content = df['url'].apply(lambda x: ' '.join(x.split('/')[3:])).values
    relevant_content = df['url'].values
    if not os.path.exists(DB_FAISS_PATH):
        # docs = []
        # metadata = []

        # # Read PDF documents from the given path
        # pdf_docs = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.endswith('.pdf')]
        # for pdf_path in pdf_docs:
        #     with open(pdf_path, "rb") as pdf_file:
        #         pdf_reader = PyPDF2.PdfReader(pdf_file)
        #         for index, page in enumerate(pdf_reader.pages):
        #             doc_page = {
        #                 "title": os.path.basename(pdf_path) + " page " + str(index + 1),
        #                 "content": page.extract_text(),
        #             }
        #             docs.append(doc_page)

        # content = [doc["content"] for doc in docs]
        # metadata = [{"title": doc["title"]} for doc in docs]
        # print("Content and metadata are extracted from the documents")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        split_docs = text_splitter.create_documents(
            df['url_content'].tolist(), 
            metadatas=[{"title": row["url_title"], "url": row["url"]} for _, row in df.iterrows()]
        )
        print(f"Documents are split into {len(split_docs)} passages")

        db = FAISS.from_documents(split_docs, embeddings)
        print(f"Document saved in db")
        db.save_local(DB_FAISS_PATH)
    else:
        print(f"Db already exists")
        db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    
    history_aware_retriever = create_history_aware_retriever(
        ChatOpenAI(model=model_name), db.as_retriever(), ChatPromptTemplate.from_messages(
        [
            ("system", "Given a chat history and the latest user question which might reference context in the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is."),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    ))
    question_answer_chain = create_stuff_documents_chain(ChatOpenAI(model=model_name), ChatPromptTemplate.from_messages(
        [
            ("system", "You are an assistant for question-answering tasks. Use the provided context to answer the questions. If the question isn't related to the context, respond that you don't know the answer. Answer based only on the provided context, not from outside knowledge. The context is as follows: {context}"),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    ))
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    return rag_chain, relevant_content


load_dotenv(override=True)
# chain = setupDbAndChain('../../documents/')
chain, relevant_content = setupDbAndChain('../../131_webmd_vogon_sample1000_urlsContent_cleaned.tsv')
chat_history = []
def echo(question, history):
    response = chain.invoke({"input": question, "chat_history": chat_history})
    chat_history.extend([HumanMessage(content=question), response["answer"]])
    docs_info = "\n\n".join([f"Title: {doc.metadata['title']}\nUrl: {doc.metadata['url']}\nContent: {doc.page_content}" for doc in response["context"]])
    full_response = f"Answer: {response['answer']}\n\nRetrieved Documents:\n{docs_info}"
    return full_response

demo = gr.ChatInterface(fn=echo, examples=list(relevant_content), title="RAG on webmd",theme=gr.themes.Soft(), fill_height=True)
gr.close_all()
demo.launch(share=True)