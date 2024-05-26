# https://medium.com/@vikrambhat2/building-a-rag-system-and-conversational-chatbot-with-custom-data-793e9617a865

import PyPDF2
import streamlit as st
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

# from langchain_community.llms import llamacpp
# from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import os

llmtemplate = """[INST]
As an AI, provide accurate and relevant information based on the provided document. Your responses should adhere to the following guidelines:
- Answer the question based on the provided documents.
- Be direct and factual, limited to 50 words and 2-3 sentences. Begin your response without using introductory phrases like yes, no etc.
- Maintain an ethical and unbiased tone, avoiding harmful or offensive content.
- Avoid using confirmatory phrases like "Yes, you are correct" or any similar validation in your responses.
- Do not fabricate information or include questions in your responses.
- do not prompt to select answers. do not ask me questions
{question}

[/INST]
"""
prompt_template = """Use the following pieces of context and previous questions and answers to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Previous Q&A: {previous_qa}

Question: {question}
Helpful Answer:"""

# PDF_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
# LLAMA_MODEL_PATH = "llama-2-7b-chat.Q4_K_M.gguf"
DB_FAISS_PATH = "../vectorstore/db_faiss"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 256
SIMILARITY_THRESHOLD = 0.5


def prepare_db(pdf_docs):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if not os.path.exists(DB_FAISS_PATH):
        docs = []
        metadata = []
        content = []

        for pdf in pdf_docs:
            pdf_reader = PyPDF2.PdfReader(pdf)
            for index, page in enumerate(pdf_reader.pages):
                doc_page = {
                    "title": pdf.name + " page " + str(index + 1),
                    "content": page.extract_text(),
                }
                docs.append(doc_page)
        for doc in docs:
            content.append(doc["content"])
            metadata.append({"title": doc["title"]})
        print("Content and metadata are extracted from the documents")
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
        )
        split_docs = text_splitter.create_documents(content, metadatas=metadata)
        print(f"Documents are split into {len(split_docs)} passages")
        # embeddings = HuggingFaceEmbeddings(
        #     model_name=PDF_MODEL_NAME,
        #     model_kwargs={"device": "cpu"},
        # )
        db = FAISS.from_documents(split_docs, embeddings)
        print(f"Document saved in db")
        db.save_local(DB_FAISS_PATH)
    else:
        print(f"Db already exists")
        db = FAISS.load_local("./vectorstore/db_faiss", embeddings, allow_dangerous_deserialization=True)
    return db


def get_conversation_chain(vectordb):
    conversation_chain = ConversationalRetrievalChain.from_llm(
        # llm=llamacpp.LlamaCpp(
        #     model_path=LLAMA_MODEL_PATH,
        #     temperature=0.75,
        #     max_tokens=200,
        #     top_p=1,
        #     n_ctx=3000,
        #     verbose=False,
        # ),
        llm = ChatOpenAI(model="gpt-3.5-turbo"),
        retriever=vectordb.as_retriever(),
        condense_question_prompt= PromptTemplate.from_template(llmtemplate),
        
        memory=ConversationBufferMemory(
            memory_key="chat_history", return_messages=True, output_key="answer"
        ),
        return_source_documents=True,
    )
    print("Conversation chain created")
    return conversation_chain


# def validate_answer_against_sources(response_answer, source_documents):
#     model = SentenceTransformer(PDF_MODEL_NAME)
#     source_texts = [doc.page_content for doc in source_documents]
#     answer_embedding = model.encode(response_answer, convert_to_tensor=True)
#     source_embeddings = model.encode(source_texts, convert_to_tensor=True)
#     cosine_scores = util.pytorch_cos_sim(answer_embedding, source_embeddings)
#     if any(score.item() > SIMILARITY_THRESHOLD for score in cosine_scores[0]):
#         return True
#     return False


def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        template = "<div style='color: blue;'>{{MSG}}</div>"
        if i%2 != 0:
            template = "<div style='color: green;'>{{MSG}}</div>"
        st.write(
            template.replace("{{MSG}}", str(i) + ': ' + message.content),
            unsafe_allow_html=True,
        )


def main():
    load_dotenv(override=True)
    st.set_page_config(page_title="Chat with your PDFs", page_icon=":books:")
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)
    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                vectorstore = prepare_db(pdf_docs)
                # print(vectorstore.similarity_search("Tell me about add and adhd"))
                st.session_state.conversation = get_conversation_chain(vectorstore)

if __name__ == "__main__":
    main()
