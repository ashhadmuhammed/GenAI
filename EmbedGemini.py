import json
from dotenv import load_dotenv
from sqlalchemy import create_engine, inspect
from chromadb.config import Settings
import chromadb
import google.generativeai as genai
# Define connection parameters
DATABASE_URI = 'mysql+mysqlconnector://root:root@localhost/emp'


# Define connetion parameters
username = 'CW-IWM-DEV'
password = 'Ek3mUMJkPiEgL7npPgmDnnaEPW3i-dN2f8oFPO6aFAhcyFXYrlvcXQ5UvoCvPBUJBY_F5iOv6uyHkDFwOiEufkrixqZfuNrcN7lTqM.o3MDReXkRqC5AMM7wIVMSP1zM'
host = 'c6217cff-ccbd-4e82-8ad6-27cdc04d02a0.hna0.prod-eu10.hanacloud.ondemand.com'
port =443   # Default MySQL port
database = 'H00'

# Construct the DATABASE_URI for SAP HANA using hdbcli

DATABASE_URI = f'hana+hdbcli://{username}:{password}@{host}:{port}/?databaseName={database}'


client = chromadb.PersistentClient(path='./chromaGemini1')
import os
print(os.getcwd()) 
try:
    # Try to retrieve the collection
    collection = client.get_collection("chunk_collection")
    print("Collection loaded.")
except Exception as e:
    # If the collection doesn't exist, create it
    collection = client.create_collection("chunk_collection")
    print("New collection created.")


print("connected.............")
engine = create_engine(DATABASE_URI)
inspector = inspect(engine)

schema_info = []

# Extract schema information
tables = inspector.get_table_names()
tables=[table for table in tables if table.startswith('cw_itm_wn')]
for table_name in tables:
    print(table_name)
    table_info = {
        "table_name": table_name,
        "columns": inspector.get_columns(table_name),
        "primary_keys": inspector.get_pk_constraint(table_name)["constrained_columns"],
        "indexes": inspector.get_indexes(table_name),
        "foreign_keys": inspector.get_foreign_keys(table_name),
    }
    schema_info.append(table_info)


chunks = []
for table in schema_info:
    print(table)
    chunk_text = f"Table: {table['table_name']}\n"
    chunk_text += "Columns:\n"
    for col in table['columns']:
        chunk_text += f"  - {col['name']} ({col['type']})\n"
    chunk_text += f"Primary Keys: {', '.join(table['primary_keys'])}\n"
    if table['foreign_keys']:
        chunk_text += "Foreign Keys:\n"
        for fk in table['foreign_keys']:
            chunk_text += f"  - {fk['constrained_columns']} references {fk['referred_table']}.{fk['referred_columns']}\n"
    chunks.append(chunk_text)


relationship_chunks = []
for table in schema_info:   
    for fk in table['foreign_keys']:
        chunk_text = f"Foreign Key Relationship:\n"
        chunk_text += f"  - {fk['constrained_columns']} in {table['table_name']} references {fk['referred_table']}.{fk['referred_columns']}\n"
        relationship_chunks.append(chunk_text)
print(relationship_chunks)

metadata = [{"chunk": chunk, "table": schema_info[i]['table_name']} for i, chunk in enumerate(chunks)]


print("Meta Data:::::::::",metadata)
# # Store the schema info in a JSON file
# with open("schema_info.json", "w") as f:
#     json.dump(schema_info, f, indent=4)




import faiss
from sentence_transformers import SentenceTransformer
import numpy as np
load_dotenv()
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))

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
    
def embed(): 
   
    chunk_embeddings =get_gemini_embeddings(chunks)

    # Ensure embeddings are a numpy array
    assert isinstance(chunk_embeddings, np.ndarray)
    
    # Assuming 'collection' is an instance of a vector store or similar object
    collection.add(
        documents=chunks,              # Chunks of text
        embeddings=chunk_embeddings,   # Corresponding embeddings
        metadatas=metadata,            # Metadata associated with each chunk
        ids=[str(i) for i in range(len(chunks))]  # Unique IDs for each chunk
    )


    # Create FAISS index
    d = chunk_embeddings.shape[1]  # Dimension of the embeddings
    faiss_index = faiss.IndexFlatL2(d)
    #  Add embeddings to the FAISS index
    faiss_index.add(np.array(chunk_embeddings))



    # Store index (optional)
    faiss.write_index(faiss_index, "chunk_embeddings.index")
    # If separate index for relationships:
    # faiss.write_index(relationship_faiss_index, "relationship_embeddings.index")
    print("Done..................")
    return faiss_index





faiss_index=embed()
# Example query
query = "TASKS and owners"
query_embedding = get_gemini_embeddings([query])

# Search the FAISS index
D, I = faiss_index.search(query_embedding, k=3)  # k is the number of nearest neighbors you want

# I contains the indices of the closest embeddings, and D their corresponding distances
for idx in I[0]:
    result = collection.get(ids=[str(idx)])
    print(f"Similar Chunk: {result['documents'][0]}")