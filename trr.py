import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from transformers import AutoTokenizer, AutoModelForCausalLM

# Paths to your FAISS index
DB_FAISS_PATH = r'C:\Users\asus\Desktop\pro1\8th project\Llama2-Medical-Chatbot\vectorstore'

# Load Sentence Transformer for embeddings
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={'device': 'cpu'})

# Load the pre-saved FAISS index
db = FAISS.load_local(DB_FAISS_PATH, embedder, allow_dangerous_deserialization=True)

# Load Hugging Face tokenizer and model for generating responses
tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

def retrieve_context(query, k=2):
    """Retrieve context from FAISS index based on the query."""
    query_vector = embedder.embed_documents([query])
    query_vector = np.array(query_vector)  # Ensure the query vector is a NumPy array
    # Use FAISS to search for the nearest neighbors
    distances, indices = db.index.search(query_vector, k)
    # Fetch the corresponding documents
    return [db.docstore._dict[str(i)].page_content for i in indices[0] if str(i) in db.docstore._dict]

def generate_response(context, query):
    """Generate a response using the Hugging Face model."""
    prompt = f"Context: {' '.join(context)}\nQuestion: {query}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(inputs["input_ids"], max_length=150, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def chatbot(query):
    context = retrieve_context(query)
    if not context:
        context = ["No relevant information found in the database."]
    response = generate_response(context, query)
    return response

if __name__ == "__main__":
    print("Hi, Welcome to the Medical Bot. What is your query?")
    while True:
        query = input("\nEnter your query: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        response = chatbot(query)
        print("\nResponse:", response)
