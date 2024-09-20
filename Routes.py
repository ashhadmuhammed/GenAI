from dotenv import load_dotenv
from flask import jsonify, request
import Configs
import os
import chromadb
import google.generativeai as genai

import faiss
import numpy as np
import Model

# Define connetion parameters
username = Configs.username
password = Configs.password
host = Configs.host
port =Configs.port   # Default MySQL port
database = Configs.database

# Construct the DATABASE_URI for SAP HANA using hdbcli

DATABASE_URI = f'hana+hdbcli://{username}:{password}@{host}:{port}/?databaseName={database}'




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
    db = chroma_client.get_collection(name=name, embedding_function=Model.GeminiEmbeddingFunction())

    return db

db=load_chroma_collection(path="./chromaGemini1", name="chunk_collection")
faiss_index = faiss.read_index("./chunk_embeddings.index")
# Retrieve all documents and metadata from the collection
all_documents = db.get()['documents']
all_metadatas = db.get()['metadatas']



def setup_routes(app):
    @app.route('/GenQuery',methods=['Post'])   
    def GenQuery():
        
        data = request.json
        query = data.get('query')

        if not query:
            return jsonify({"error": "Query not provided"}), 400
        
        # query_embedding = Model.get_gemini_embeddings([query])

        # # Search the FAISS index
        # D, I = faiss_index.search(query_embedding, k=6)  # k is the number of nearest neighbors you want

        # # I contains the indices of the closest embeddings, and D their corresponding distances
        # schema=''
        # for idx in I[0]:
        #     result = db.get(ids=[str(idx)])
        #     # print(result)
        #     print(f"Similar Chunk: {result['documents'][0]}")
        #     schema+= result['documents'][0]



        prompt=f"You are Sql expert which converts user query which is in english to sql query with the following schema :\n{Configs.schema} \n here is the query:{query}  "

        sql_query=Model.generate_sql_with_gemini(prompt)

        print(Model.generate_sql_with_gemini(prompt))
        return jsonify({"sql_query": sql_query})