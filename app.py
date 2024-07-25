import streamlit as st
import PyPDF2
import re
import os
import google.generativeai as genai

def extract_text_from_pdf(file):
    pdf_text = ""
    pdf_reader = PyPDF2.PdfReader(file)
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num] 
        pdf_text += page.extract_text()
    return pdf_text

def extract_db_details(pdf_text):
    # Basic example of parsing the PDF text for database details
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
    # Start with the context
    prompt = "You are an SQL expert. Here is the database schema:\n\n"
    
    # Add tables and columns
    for table, columns in db_details["columns"].items():
        prompt += f"Table: {table}\nColumns: {', '.join(columns)}\n\n"
    
    # Add relationships if any
    if db_details["relationships"]:
        prompt += "Relationships:\n"
        for relationship in db_details["relationships"]:
            prompt += f"- {relationship}\n"
    
    # Add the user query
    prompt += f"\nUsing the above schema, generate an SQL query for the following request:\n{user_query}\n"
    
    return prompt

# Placeholder function for interacting with Google Gemini API
def generate_sql_with_gemini(prompt):
     

    
    # Configure the API key
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    
    # Create the model
    # See https://ai.google.dev/api/python/google/generativeai/GenerativeModel
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
    # safety_settings = Adjust safety settings
    # See https://ai.google.dev/gemini-api/docs/safety-settings
    )
    
    # Add initial context to the chat session
   
    chat_session = model.start_chat(
    history=[
        
    ]
    )
    
    # Send a message within the chat session
    response = chat_session.send_message(prompt)
    
    # Print the response type and text
    print(type(response))
    print(response.text)
    return response.text
        








st.title("DB Bot")

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
if uploaded_file is not None:
    pdf_text = extract_text_from_pdf(uploaded_file)
    db_details = extract_db_details(pdf_text)
    st.write("Extracted Database Details:", db_details)

query = st.text_input("Enter your query in English:")
if st.button("Generate SQL"):
    if uploaded_file:
        prompt = generate_prompt_template(db_details, query)
        st.write("Generated Prompt:", prompt)
        print(prompt)
        
        # Make API call to Google Gemini with the generated prompt
        generated_sql = generate_sql_with_gemini(prompt)
       
        st.text_area("Generated SQL Query:", value=generated_sql, height=100)
    else:
        st.write("Please upload a PDF document.")
