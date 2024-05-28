from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import os
import pandas as pd

import gradio as gr
from openai import OpenAI

load_dotenv(override=True)
client = OpenAI()
DB_FAISS_PATH = "./vectorstore/db_faiss"
CHUNK_SIZE = 512
CHUNK_OVERLAP = 256
embedding_model_name = "text-embedding-3-small"
model_name = "gpt-3.5-turbo"
data_file_path = "./data/131_webmd_vogon_sample1000_urlsContent_cleaned.tsv"
bestReformulationPrompt = "Given a chat history and the latest user question, which may reference context from the chat history, you must formulate a standalone question that can be understood without the chat history. You are strictly forbidden from using any outside knowledge. Do not, under any circumstances, answer the question. Reformulate it if necessary; otherwise, return it as is."
bestSystemPrompt = "As an assistant for question-answering tasks, your approach must be systematic and meticulous. First, identify CLUES such as keywords, phrases, contextual information, semantic relations, tones, and references that aid in determining the context of the input. Second, construct a concise diagnostic REASONING process (limiting to 130 words) based on premises supporting the INPUT relevance within the provided context. Third, utilizing the identified clues, reasoning, and input, furnish the pertinent answer for the question. Remember, you are required to use ONLY the provided context to answer the questions. If the question does not align with the context or if the context is absent, indicate that you don't know the answer. External knowledge is strictly prohibited. Failure to adhere will lead to incorrect answers. The context is as follows:"

def setupDb(data_path):
    embeddings = OpenAIEmbeddings(model=embedding_model_name)
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
    return db, relevant_content

def reformulate_question(chat_history, latest_question, reformulationPrompt):
    system_message = {
        "role": "system",
        "content": reformulationPrompt
    }

    formatted_history = [{"role": "user", "content": q} if i % 2 == 0 else {"role": "assistant", "content": a} for i, (q, a) in enumerate(chat_history)]
    print(formatted_history)
    formatted_history.append({"role": "user", "content": latest_question})
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[system_message] + formatted_history,
        temperature=0
    )

    reformulated_question = response.choices[0].message.content
    return reformulated_question

def getQuestionAnswerOnTheBasisOfContext(question, context, systemPrompt):
    system_message = {
        "role": "system",
        "content": systemPrompt + context
    }

    response = client.chat.completions.create(
        model=model_name,
        messages=[system_message] + [{"role": "user", "content": question}],
        temperature=0
    )
    answer = response.choices[0].message.content
    return answer

db, relevant_content = setupDb(data_file_path)
chat_history = []
curr_question_no = 1

def chatWithRag(reformulationPrompt, QAPrompt, question):
    global curr_question_no, chat_history
    curr_question_prompt = bestSystemPrompt
    if QAPrompt != None or len(QAPrompt):
        curr_question_prompt = QAPrompt
    # reformulated_query = reformulate_question(chat_history, question, reformulationPrompt)

    reformulated_query = question
    db.as_retriever()
    retreived_documents = db.similarity_search_with_score(reformulated_query)
    context = [doc for doc in retreived_documents if doc[1] < 1.3]
    retreived_documents = context
    context = '. '.join([doc[0].page_content for doc in context])
    answer = getQuestionAnswerOnTheBasisOfContext(reformulated_query, context, curr_question_prompt)
    chat_history.append((question, answer))
    curr_question_no += 1
    docs_info = "\n\n".join([
        f"Title: {doc[0].metadata['title']}\nUrl: {doc[0].metadata['url']}\nContent: {doc[0].page_content}\nValue: {doc[1]}" for doc in retreived_documents
    ])
    full_response = f"Answer: {answer}\n\nReformulated question: {reformulated_query}\nRetrieved Documents:\n{docs_info}"
    return full_response

with gr.Blocks() as demo:
    gr.Markdown("# RAG on webmd")
    with gr.Row():
        reformulationPrompt = gr.Textbox(bestReformulationPrompt, lines=1, placeholder="Enter the system prompt for reformulation of query", label="Reformulation System prompt")
        QAPrompt = gr.Textbox(bestSystemPrompt, lines=1, placeholder="Enter the system prompt for QA.", label="QA System prompt")
        question = gr.Textbox(lines=1, placeholder="Enter the question asked", label="Question")
    output = gr.Textbox(label="Output")
    submit_btn = gr.Button("Submit")
    submit_btn.click(chatWithRag, inputs=[reformulationPrompt, QAPrompt, question], outputs=output)
    with gr.Accordion("Urls", open=False):
        gr.Markdown(', '.join(relevant_content))

gr.close_all()
demo.launch(share=True)
