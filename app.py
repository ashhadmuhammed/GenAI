import pandas as pd
import streamlit as st
import PyPDF2
import re
import os
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from sqlalchemy import create_engine, MetaData, Table, text
from sqlalchemy.orm import sessionmaker
import mysql.connector

# Database credentials
# mydb = mysql.connector.connect(
#   host="localhost",
#   user="root",
#   password="root",
#   database="emp"
# )


# # Define connection parameters
# DATABASE_URI = 'mysql+mysqlconnector://root:root@localhost/emp'

# # Create an engine to connect to the MySQL database
# engine = create_engine(DATABASE_URI)

# Create a session
# Session = sessionmaker(bind=engine)
# session = Session()

# # Create a metadata object to reflect tables
# metadata = MetaData()
# metadata.reflect(bind=engine)

# Session = sessionmaker(bind=engine)




def extract_text_from_pdf(file):
    pdf_text = ""
    pdf_reader = PyPDF2.PdfReader(file)
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        pdf_text += page.extract_text()
    return pdf_text

def extract_db_details(pdf_text):
    db_details = {
        "tables": [],
        "columns": {},
        "relationships": []
    }
    
    lines = pdf_text.split("\n")
    current_section = None
    for line in lines:
        if "Table:" in line:
            table_name = line.split(":")[1].strip()
            db_details["tables"].append(table_name)
            current_section = table_name
            db_details["columns"][current_section] = []
        elif "Column:" in line and current_section:
            column_name = line.split(":")[1].strip()
            db_details["columns"][current_section].append(column_name)
        elif "Relationship:" in line:
            db_details["relationships"].append(line.split(":")[1].strip())
    
    return db_details

def generate_prompt_template(db_details, user_query):
    prompt = "You are an SQL expert. Here is the database schema:\n\n"
    
    for table, columns in db_details["columns"].items():
        prompt += f"Table: {table}\nColumns: {', '.join(columns)}\n\n"
    
    if db_details["relationships"]:
        prompt += "Relationships:\n"
        for relationship in db_details["relationships"]:
            prompt += f"- {relationship}\n"
    
    prompt += f"\nUsing the above schema, generate an SQL query for the following request:\n{user_query}\n"
    
    return prompt

def generate_sql_with_gemini(prompt):
    try:
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        
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
        st.error(f"Error generating SQL with Gemini: {str(e)}")
        return None

# Function to split text into chunks
def split_text(text, chunk_size=512):
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + ". "
        else:
            chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

# Function to create a FAISS index
def create_faiss_index(chunks, embedder):
    embeddings = embedder.encode(chunks)
    d = embeddings.shape[1]
    index = faiss.IndexFlatL2(d)
    index.add(embeddings)
    return index, embeddings

# Function to retrieve relevant chunks
def retrieve_relevant_chunks(query, index, chunks, embedder, top_k=5):
    query_embedding = embedder.encode([query])
    distances, indices = index.search(query_embedding, top_k)
    retrieved_chunks = [chunks[idx] for idx in indices[0]]
    return retrieved_chunks


# def execute_raw_query(query):
#     """Execute a raw SQL query and return the results"""
#     DATABASE_URI = 'mysql+mysqlconnector://root:root@localhost/emp'
#     engine = create_engine(DATABASE_URI)

#     try:
#         with engine.connect() as connection:
#             sql = text(query)
#             result = connection.execute(sql)
#             rows = result.fetchall()
#             df = pd.DataFrame(rows, columns=result.keys())
#             return df
#     except Exception as e:  # Use a generic exception to catch potential SQLAlchemy errors
#         print(f"Error executing SQL query: {e}")
#         return pd.DataFrame()
       

def execute_raw_query(query):
   """Execute a raw SQL query and return the results"""
    
   con = mysql.connector.connect(
      host="localhost",
      user="root",
      password="root",
      database="emp")
   
   try:
        mycursor = con.cursor()  
        mycursor.execute(query)
        rows = mycursor.fetchall()
        df = pd.DataFrame(rows, columns=[column[0] for column in mycursor.description])
        return df
   except mysql.connector.Error as e:
        print(f"Error executing SQL query: {e}")
        return pd.DataFrame()
   finally:
        con.close()


st.title("DB Bot")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    pdf_text = extract_text_from_pdf(uploaded_file)
    
    # Split the PDF text into chunks
    chunks = split_text(pdf_text)
    
    # Load the embedding model
    embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    
    # Create FAISS index
    index, embeddings = create_faiss_index(chunks, embedder)
    
    db_details = extract_db_details(pdf_text)
    st.write("Extracted Database Details:", db_details)

query = st.text_input("Enter your query in English:")
if st.button("Generate SQL"):
    if uploaded_file:
        # Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(query, index, chunks, embedder)
        
        # Generate the prompt with relevant information
        relevant_text = " ".join(relevant_chunks)
        db_details = extract_db_details(relevant_text)
        prompt = generate_prompt_template(db_details, query)
        
        st.write("Generated Prompt:", prompt)
        
        # Make API call to Google Gemini with the generated prompt
        generated_sql = generate_sql_with_gemini(prompt)
        print("hi")
        print(generated_sql)
        print(type(generated_sql))
        results=execute_raw_query(generated_sql)
        if not results.empty:
           st.dataframe(results)
        else:
           st.write("No results found.")

    else:
        st.write("Please upload a PDF document.")
