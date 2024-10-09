# Medical-Chatbot-using-LLMs

### Description:

This Medical Chatbot is an advanced conversational agent designed to provide users with accurate and relevant medical information. Utilizing a combination of state-of-the-art technologies, the chatbot integrates **Llama-2-7b** as its language model, along with Hugging Face embeddings and a FAISS vector database for efficient information retrieval.

### Setup Instructions:

**Create the FAISS Vector Store:**

- Run the ingest.py file to load and preprocess your PDF documents located in the specified DATA_PATH.
- Ensure the FAISS vector store is created and saved in the designated DB_FAISS_PATH.

**Update File Paths:**

- Modify the paths in model.py, and trr.py to match your local environment:

**DATA_PATH:** The directory containing your PDF files. </br>
**DB_FAISS_PATH:** The directory where the FAISS vector store will be saved. </br>
**Model path for Llama-2-7b:** Update the path in model.py to point to your local model file.

**Run the Chatbot:**

- Once the vector store is created, you can execute the trr.py file to launch the chatbot interface.
- Users can ask questions related to medical topics, and the chatbot will analyze the query, retrieve relevant context, and generate coherent responses.
- With its user-friendly interface powered by Chainlit and the capability to provide sources for its responses, this chatbot serves as a reliable resource for anyone seeking medical knowledge.

