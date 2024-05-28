# https://medium.com/@shaktikanungo2019/conversational-ai-unveiling-the-first-rag-chatbot-with-langchain-8b9b04ee4b63

# import PyPDF2
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS

from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI

# from langchain_community.llms import llamacpp
# from langchain.embeddings import LlamaCppEmbeddings
# from langchain_community.embeddings import HuggingFaceEmbeddings
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
SIMILARITY_THRESHOLD = 0.25
embedding_model = "text-embedding-3-small"
model_name = "gpt-3.5-turbo"
# data_file_path = "../../131_webmd_vogon_sample1000_urlsContent_cleaned.tsv"
data_file_path = "../../132_webmd_vogon_urlsContent_cleaned.tsv"
# llama_model_path = "../../models/llama-2-7b-chat.Q4_K_M.gguf"


def setupDbAndChainRetreiver(data_path):
    embeddings = OpenAIEmbeddings(model=embedding_model)
    # embeddings = LlamaCppEmbeddings(model_path=llama_model_path)
    # embeddings = HuggingFaceEmbeddings(
    #     model_name="sentence-transformers/all-MiniLM-L6-v2",
    #     model_kwargs={"device": "cpu"},
    # )
    df = pd.read_csv(data_path, sep="\t")
    relevant_content = df["url"].values
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    if not os.path.exists(DB_FAISS_PATH):
        split_docs = text_splitter.create_documents(
            df["url_content"].tolist(),
            metadatas=[
                {"title": row["url_title"], "url": row["url"]}
                for _, row in df.iterrows()
            ],
        )
        print(f"Documents are split into {len(split_docs)} passages")

        db = FAISS.from_documents(split_docs, embeddings)
        print(f"Document saved in db")
        db.save_local(DB_FAISS_PATH)
    else:
        print(f"Db already exists")
        db = FAISS.load_local(
            DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True
        )

    # llm = llamacpp.LlamaCpp(
    #     model_path=llama_model_path,
    #     temperature=0.75,
    #     max_tokens=200,
    #     top_p=1,
    #     n_ctx=3000,
    #     verbose=False,
    # )
    history_aware_retriever = create_history_aware_retriever(
        # llm,
        ChatOpenAI(model=model_name, temperature=0),
        db.as_retriever(),
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """Given a chat history and the latest user question, which may reference context from the chat history, you must formulate a standalone question that can be understood without the chat history. You are strictly forbidden from using any outside knowledge. Do not, under any circumstances, answer the question. Reformulate it if necessary; otherwise, return it as is.""",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        ),
    )
    return history_aware_retriever, relevant_content

def getChatQAChain(prompt):
    question_answer_chain = create_stuff_documents_chain(
        ChatOpenAI(model=model_name, temperature=0),
        # llm,
        ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    prompt + """ {context}.""",
                ),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        ),
    )
    return question_answer_chain


load_dotenv(override=True)
# chain = setupDbAndChainRetreiver('../../documents/')
history_aware_retriever, relevant_content = setupDbAndChainRetreiver(data_file_path)
question_answer_chain = getChatQAChain("As an assistant for question-answering tasks, your approach must be systematic and meticulous. First, identify CLUES such as keywords, phrases, contextual information, semantic relations, tones, and references that aid in determining the context of the input. Second, construct a concise diagnostic REASONING process (limiting to 130 words) based on premises supporting the INPUT relevance within the provided context. Third, utilizing the identified clues, reasoning, and input, furnish the pertinent answer for the question. Remember, you are required to use ONLY the provided context to answer the questions. If the question does not align with the context or if the context is absent, indicate that you don't know the answer. External knowledge is strictly prohibited. Failure to adhere will lead to incorrect answers. The context is as follows:")
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
chat_history = []
curr_question_no = 1


def chatWithRag(prompt, question):
    global curr_question_no, chat_history, history_aware_retriever, question_answer_chain

    if prompt != None or len(prompt):
        question_answer_chain = getChatQAChain(prompt)
        rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    response = rag_chain.invoke({"input": question, "chat_history": chat_history})
    if (
        len(response["context"]) == 0
        or "don't know" in response["answer"]
        or "cannot provide an answer" in response["answer"]
        or "I'm sorry" in response["answer"]
    ):
        print(f"[ INVALID QUESTION ] {question} ----------> ", response["answer"])
        return f"Answer: Question isn't relevant to provided context."
    print(f"[ {curr_question_no} ] {question} ----------> ", response["answer"])
    curr_question_no += 1
    chat_history.extend([HumanMessage(content=question), response["answer"]])
    # if len(chat_history) > 3:
    #     chat_history = chat_history[1:]
    # print(chat_history)
    docs_info = "\n\n".join(
        [
            f"Title: {doc.metadata['title']}\nUrl: {doc.metadata['url']}\nContent: {doc.page_content}"
            for doc in response["context"]
        ]
    )
    full_response = f"Answer: {response['answer']}\n\nRetrieved Documents:\n{docs_info}"
    return full_response


# demo = gr.ChatInterface(
#     fn=echo,
#     examples=list(relevant_content),
#     title="RAG on webmd",
#     inputs=[
#         gr.inputs.Textbox(lines=1, placeholder="Enter the system prompt"),
#         gr.inputs.Textbox(lines=1, placeholder="Enter the question")
#     ],
#     theme=gr.themes.Soft(),
#     fill_height=True,
# )

with gr.Blocks() as demo:
    gr.Markdown("# RAG on webmd")
    with gr.Row():
        prompt = gr.Textbox(lines=1, placeholder="Enter the system prompt.", label="System prompt")
        question = gr.Textbox(lines=1, placeholder="Enter the question asked", label="Question")
    output = gr.Textbox(label="Output")
    submit_btn = gr.Button("Submit")
    submit_btn.click(chatWithRag, inputs=[prompt, question], outputs=output)

gr.close_all()
demo.launch(share=True)
