from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)
from llama_index.llms.openai import OpenAI
from llama_index.core.settings import Settings
import os
from dotenv import load_dotenv

load_dotenv(override=True)
Settings.llm = OpenAI(model="gpt-3.5-turbo", temperature=0.1)

storage_path = "../vectorstore"
documents_path = "../../documents"


def initialize():
    if not os.path.exists(storage_path):
        documents = SimpleDirectoryReader(documents_path).load_data()
        index = VectorStoreIndex.from_documents(documents)
        index.storage_context.persist(persist_dir=storage_path)
    else:
        storage_context = StorageContext.from_defaults(persist_dir=storage_path)
        index = load_index_from_storage(storage_context)
    return index
index = initialize()

chat_engine = index.as_chat_engine(chat_mode="condense_question", verbose=True)
response = chat_engine.chat("hi tell me what i can ask you")
print(response.response)