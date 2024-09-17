import os
import chromadb
import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings
import faiss
import numpy as np

class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Custom embedding function using the Gemini AI API for document retrieval.

    This class extends the EmbeddingFunction class and implements the __call__ method
    to generate embeddings for a given set of documents using the Gemini AI API.

    Parameters:
    - input (Documents): A collection of documents to be embedded.

    Returns:
    - Embeddings: Embeddings generated for the input documents.
    """
    def __call__(self, input: Documents) -> Embeddings:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not gemini_api_key:
            raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"
        title = "Custom query"
        return genai.embed_content(model=model,
                                   content=input,
                                   task_type="retrieval_document",
                                   title=title)["embedding"]


def get_gemini_embeddings(texts):
    embeddings = []
    for text in texts:
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document",
            title="Embedding of IWM"
        )
        embeddings.append(result['embedding'])
    return np.array(embeddings)   

def load_chroma_collection(path, name):

    
    """
    Loads an existing Chroma collection from the specified path with the given name.

    Parameters:
    - path (str): The path where the Chroma database is stored.
    - name (str): The name of the collection within the Chroma database.

    Returns:
    - chromadb.Collection: The loaded Chroma Collection.
    """
    chroma_client = chromadb.PersistentClient(path=path)
    db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())

    return db

db=load_chroma_collection(path="./chromaGemini1", name="chunk_collection")

# Retrieve all documents and metadata from the collection
all_documents = db.get()['documents']
all_metadatas = db.get()['metadatas']

#     # Print each document and its metadata
# for doc, metadata in zip(all_documents, all_metadatas):
#     print("Document:", doc)
#     print("Metadata:", metadata)
#     print("-----------------------")



# def get_relevant_passage(query, db, n_results):
#   passage = db.query(query_texts=[query], n_results=n_results)['documents'][0]
#   return passage

# #Example usage
# relevant_texts = get_relevant_passage(query="tasks",db=db,n_results=3)
# for relevant_text in relevant_texts:
#     print(relevant_text)

query = "how much tasks are assigned to anupam"
query_embedding = get_gemini_embeddings([query])
faiss_index = faiss.read_index("./chunk_embeddings.index")
# Search the FAISS index
D, I = faiss_index.search(query_embedding, k=6)  # k is the number of nearest neighbors you want

# I contains the indices of the closest embeddings, and D their corresponding distances
for idx in I[0]:
    result = db.get(ids=[str(idx)])
    print(f"Similar Chunk: {result['documents'][0]}")