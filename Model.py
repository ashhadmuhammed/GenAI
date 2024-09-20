import os
from dotenv import load_dotenv
import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings
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
    

def generate_sql_with_gemini(prompt):
    try:
        load_dotenv()


        genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
        
        generation_config = {
            "temperature": 1,
            "top_p": 0.95,
            "top_k": 64,
            "max_output_tokens": 8192,
            "response_mime_type": "text/plain",
        }
        
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config=generation_config,
        )
        
        chat_session = model.start_chat(history=[])
        
        response = chat_session.send_message(prompt)
        
        return response.text.strip().strip("```sql").strip("```").strip()
    except Exception as e:
        print(f"Error generating SQL with Gemini: {str(e)}")
        return None
    
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