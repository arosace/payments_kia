# K.I.A.

know it all chat bot.

This is a P.O.C. to showcase the ability of AI with respect to our knowledgebase.

As of now the repo offers 4 applications:

data_vectorizer.py -> vectorizes the MD documents that are in the ./data folder and saves them on PineCone

query_example_no_llm.py -> make any queries you want to the knowledge base. The application vectorizes your query, performs a similarity search on PineCone and return the most relevant lines from the documents in the knowledgebase

query_example_with_llm.py -> same as bove, but know the result from the similarity search is passed through an LLM. The LLM will embellish and enrich the answer to make it human readable. In ourder to use this you need to download the open source ```llama-2-7b-chat.Q4_0.gguf``` model and put it in the models folder

llamaChat.py -> a full fledged locally hosted chat-history-aware conversational RAG application that runs in the terminal. Upon each query the chatBot will scan PineCone, find relevant information and out put it via the LLM into the Terminal. You can then ask further questions and the application will leverage the queried context and the chat history to answer.

## Prerequisites

In order to use the application you need to:
- install python3 and the application requirements as per the requirements.txt file ```pip3 install -r requirements.txt```
- rename the config.env file into ```env``` and populate the env variables.
- download the ```llama-2-7b-chat.Q4_0.gguf``` model and put it in the models folder

If you want to use the already vectorized data ask me, otherwise open a free account on PineCone and use the data_vectorizer to upload data to it.
As of now the only data format supported is MD