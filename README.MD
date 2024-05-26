Neo4j RAG course - https://www.deeplearning.ai/short-courses/knowledge-graphs-rag/

## Steps for running rag -:
1. There are 3 working rag codes in this:
```
    langchain/history_aware(best)
    langchain/streamlit_LLAMA_notGoodOutput
    llama_index/chat_engine_single_query
```
2. Go into specific directory(for streamlit_LLAMA you can use local llama model too by downloading llama-2-7b-chat.Q4_K_M.gguf into models directory and uncommenting the required changes in rag.py file)
3. Create .env file in root folder and add the following environment variables
```
OPENAI_API_KEY=<YOUR OPENAI KEY>
```
4. Run the following commands:
```
    pip3 install -r requirements.txt
    python3 rag.py
``` 